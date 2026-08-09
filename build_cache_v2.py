"""
build_cache_v2.py
=================

Builds SmolLM2 cached embeddings for dataset v2 (100 facts, 100 unique classes).

Requirements (D3):
- reads agnis_scaling_dataset_v2.json
- sets label = f["fact_id"], NEVER derived from any string
- embeds the 3 train_prompts and 3 test_prompts per fact with mean pooling (max_len=32, L2-normalized)
- asserts before saving:
  - len(set(train_y)) == 100
  - len(set(test_y)) == 100
  - every class has exactly 3 train and 3 test rows
  - train_x shape is (300, 960) and test_x shape is (300, 960)
  - labels are ascending by fact_id
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
            return path
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files):
                if "smollm" in root.lower():
                    return root
    return "HuggingFaceTB/SmolLM2-360M"


MODEL_ID = find_offline_model_path()
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"


def main():
    if not os.path.exists(DATASET_V2_PATH):
        raise RuntimeError(f"Missing required dataset file: {DATASET_V2_PATH}. Run generate_dataset_v2.py first.")

    with open(DATASET_V2_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    # Sort facts by fact_id to ensure ascending labels
    facts = sorted(facts, key=lambda x: x["fact_id"])

    print(f"[Cache Builder V2] Loading SmolLM2 model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(DEVICE)
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

    # D3 ASSERTIONS BEFORE SAVING
    print("\n--- ASSERTION AUDIT BEFORE SAVING CACHE V2 ---")
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

    # Check ascending label ordering
    assert torch.equal(train_y[::3], torch.arange(100)), "train_y labels are not in ascending order 0..99"
    assert torch.equal(test_y[::3], torch.arange(100)), "test_y labels are not in ascending order 0..99"

    print("  [ASSERT D3.1] 100 unique classes in train_y and test_y.")
    print("  [ASSERT D3.2] Exactly 3 train rows and 3 test rows per class.")
    print("  [ASSERT D3.3] train_x shape (300, 960) and test_x shape (300, 960).")
    print("  [ASSERT D3.4] Labels ascending strictly 0..99.")

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
            "model_id": MODEL_ID
        }
    }

    torch.save(cache_data, CACHE_V2_PATH)
    print(f"\n[Saved] Cached embeddings successfully written to {CACHE_V2_PATH}.")


if __name__ == "__main__":
    main()
