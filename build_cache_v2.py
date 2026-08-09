"""
build_cache_v2.py
=================

Builds SmolLM2 cached embeddings for dataset v2 (100 facts, 100 unique classes).

Requirements (H3, D3):
- reads agnis_scaling_dataset_v2.json
- sets label = f["fact_id"], NEVER derived from any string
- embeds the 3 train_prompts and 3 test_prompts per fact with mean pooling (max_len=32, L2-normalized)
- asserts before saving:
  - len(set(train_y)) == 100
  - len(set(test_y)) == 100
  - every class has exactly 3 train and 3 test rows
  - train_x shape is (300, 960) and test_x shape is (300, 960)
  - torch.equal(train_y, torch.arange(100).repeat_interleave(3))
  - torch.equal(test_y, torch.arange(100).repeat_interleave(3))
- logs model path branch and explicit MODEL_REVISION status
- saves to smollm2_embeddings_v2_100facts.pt (does NOT overwrite v1 cache)
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_V2_PATH = "smollm2_embeddings_v2_100facts.pt"
DATASET_V2_PATH = "agnis_scaling_dataset_v2.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_offline_model_path():
    for path in ["../local_smollm2", "local_smollm2"]:
        if os.path.exists(os.path.join(path, "model.safetensors")):
            print(f"[Model Loader] Branch 1: Found local offline SmolLM2 path at '{path}'. MODEL_REVISION is IGNORED for local files.")
            return path, False  # path, is_hf_hub
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files):
                if "smollm" in root.lower():
                    print(f"[Model Loader] Branch 2: Found Kaggle input SmolLM2 path at '{root}'. MODEL_REVISION is IGNORED for local files.")
                    return root, False
    print(f"[Model Loader] Branch 3: Falling back to HF Hub 'HuggingFaceTB/SmolLM2-360M'. MODEL_REVISION is ENFORCED.")
    return "HuggingFaceTB/SmolLM2-360M", True


MODEL_ID, IS_HF_HUB = find_offline_model_path()
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"


def load_model_and_tokenizer(model_id, is_hf_hub, revision):
    if not is_hf_hub:
        print(f"[Model Loader] Loading local directory '{model_id}' (MODEL_REVISION is IGNORED)")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        return tokenizer, model

    # Try local cache first to avoid network delays
    try:
        print(f"[Model Loader] Attempting load from local HF cache for '{model_id}' (revision={revision})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True).to(DEVICE)
        print("[Model Loader] Successfully loaded model from local HF cache! (MODEL_REVISION ENFORCED)")
        return tokenizer, model
    except Exception as e:
        print(f"[Model Loader] Local cache lookup skipped ({e}). Fetching from HF Hub (revision={revision})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to(DEVICE)
        print("[Model Loader] Successfully loaded model from HF Hub! (MODEL_REVISION ENFORCED)")
        return tokenizer, model


def main():
    if not os.path.exists(DATASET_V2_PATH):
        raise RuntimeError(f"Missing required dataset file: {DATASET_V2_PATH}. Run generate_dataset_v2.py first.")

    with open(DATASET_V2_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    # Sort facts by fact_id to ensure ascending labels
    facts = sorted(facts, key=lambda x: x["fact_id"])

    tokenizer, model = load_model_and_tokenizer(MODEL_ID, IS_HF_HUB, MODEL_REVISION)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    model.eval()

    def extract_pooled(prompt):
        enc = tokenizer(prompt, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            hidden = outputs.hidden_states[-1]
            mask = enc.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return F.normalize(pooled.float(), dim=-1)[0].cpu()

    train_queries, train_labels = [], []
    test_queries, test_labels = [], []

    print("[Cache Builder V2] Generating embeddings for 100 facts (3 train & 3 test prompts each)...")
    for f in facts:
        label = int(f["fact_id"])

        for prompt in f["train_prompts"]:
            train_queries.append(extract_pooled(prompt))
            train_labels.append(label)

        for prompt in f["test_prompts"]:
            test_queries.append(extract_pooled(prompt))
            test_labels.append(label)

    train_x = torch.stack(train_queries)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = torch.stack(test_queries)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    # H3 & D3 ASSERTIONS BEFORE SAVING
    print("\n--- ASSERTION AUDIT BEFORE SAVING CACHE V2 ---", flush=True)
    assert len(set(train_y.tolist())) == 100, f"train_y does not have 100 unique classes: {len(set(train_y.tolist()))}"
    assert len(set(test_y.tolist())) == 100, f"test_y does not have 100 unique classes: {len(set(test_y.tolist()))}"

    # Check exactly 3 train and 3 test rows per class
    for c in range(100):
        c_tr_count = (train_y == c).sum().item()
        c_te_count = (test_y == c).sum().item()
        assert c_tr_count == 3, f"Class {c} has {c_tr_count} train rows (expected 3)"
        assert c_te_count == 3, f"Class {c} has {c_te_count} test rows (expected 3)"

    assert train_x.shape == (300, 960), f"train_x shape mismatch: {train_x.shape}"
    assert test_x.shape == (300, 960), f"test_x shape mismatch: {test_x.shape}"

    # H3: Repeat interleave label check
    expected_labels = torch.arange(100).repeat_interleave(3)
    assert torch.equal(train_y, expected_labels), "train_y labels mismatch repeat_interleave(3)"
    assert torch.equal(test_y, expected_labels), "test_y labels mismatch repeat_interleave(3)"

    print("  [ASSERT D3.1] 100 unique classes in train_y and test_y.", flush=True)
    print("  [ASSERT D3.2] Exactly 3 train rows and 3 test rows per class.", flush=True)
    print("  [ASSERT D3.3] train_x shape (300, 960) and test_x shape (300, 960).", flush=True)
    print("  [ASSERT H3 PASSED] train_y & test_y strictly match torch.arange(100).repeat_interleave(3).", flush=True)

    cache_data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "metadata": {
            "num_facts": 100,
            "num_classes": 100,
            "prompts_per_fact": {"train": 3, "test": 3},
            "embedding_dim": 960,
            "model_id": MODEL_ID,
            "is_hf_hub": IS_HF_HUB
        }
    }

    torch.save(cache_data, CACHE_V2_PATH)
    print(f"\n[Saved] Cached embeddings successfully written to {CACHE_V2_PATH}.", flush=True)


if __name__ == "__main__":
    main()
