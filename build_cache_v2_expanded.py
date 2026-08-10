"""
build_cache_v2_expanded.py
===========================

Builds SmolLM2 mean-pooling embeddings cache for expanded dataset (10 train & 5 test prompts per fact).

Requirements (J7):
- Input: agnis_scaling_dataset_v2_expanded.json (100 facts, 10 train prompts & 5 test prompts each).
- Labels: train_y has shape (1000,) with torch.arange(100).repeat_interleave(10).
          test_y has shape (500,) with torch.arange(100).repeat_interleave(5).
- Model: SmolLM2-360M (revision f8027fd0eaeea54caa13c31d31b9fdc459c38b49).
- Saves to smollm2_embeddings_v2_100facts_expanded_10_5.pt (does NOT overwrite 3/3 cache).
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_EXPANDED_PATH = "agnis_scaling_dataset_v2_expanded.json"
CACHE_EXPANDED_PATH = "smollm2_embeddings_v2_100facts_expanded_10_5.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_offline_model_path():
    for path in ["../local_smollm2", "local_smollm2"]:
        if os.path.exists(os.path.join(path, "model.safetensors")):
            return path, False
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files):
                if "smollm" in root.lower():
                    return root, False
    return "HuggingFaceTB/SmolLM2-360M", True


def main():
    if not os.path.exists(DATASET_EXPANDED_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required dataset file: '{DATASET_EXPANDED_PATH}'.")

    model_id, is_hf_hub = find_offline_model_path()
    revision = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

    print(f"[Expanded Cache Builder] Loading SmolLM2 from '{model_id}'...", flush=True)
    if not is_hf_hub:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True).to(DEVICE)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    model.eval()

    with open(DATASET_EXPANDED_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    facts = sorted(facts, key=lambda x: x["fact_id"])

    all_prompts = []
    all_labels = []
    is_train_flags = []

    for f in facts:
        label = int(f["fact_id"])
        for prompt in f["train_prompts"]:
            all_prompts.append(prompt)
            all_labels.append(label)
            is_train_flags.append(True)
        for prompt in f["test_prompts"]:
            all_prompts.append(prompt)
            all_labels.append(label)
            is_train_flags.append(False)

    print(f"[Expanded Cache Builder] Extracting mean-pooling vectors for {len(all_prompts)} prompts (1000 train & 500 test)...", flush=True)

    train_queries, train_labels = [], []
    test_queries, test_labels = [], []

    batch_size = 32
    for b_i in range(0, len(all_prompts), batch_size):
        b_prompts = all_prompts[b_i : b_i + batch_size]
        b_flags = is_train_flags[b_i : b_i + batch_size]
        b_labs = all_labels[b_i : b_i + batch_size]

        enc = tokenizer(b_prompts, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            last_hidden = outputs.hidden_states[-1]  # shape (b, seq_len, 960)

            for idx in range(len(b_prompts)):
                seq_len = enc.attention_mask[idx].sum().item()
                vec = last_hidden[idx, :seq_len].mean(dim=0)
                vec_norm = F.normalize(vec.float(), dim=-1).cpu()

                if b_flags[idx]:
                    train_queries.append(vec_norm)
                    train_labels.append(b_labs[idx])
                else:
                    test_queries.append(vec_norm)
                    test_labels.append(b_labs[idx])

        if (b_i // batch_size + 1) % 10 == 0 or (b_i + batch_size) >= len(all_prompts):
            print(f"  Batch {b_i // batch_size + 1} / {len(all_prompts) // batch_size + 1} completed...", flush=True)

    train_x = torch.stack(train_queries)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = torch.stack(test_queries)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    # Assertions
    assert train_x.shape == (1000, 960), f"Expected train_x shape (1000, 960), got {train_x.shape}"
    assert test_x.shape == (500, 960), f"Expected test_x shape (500, 960), got {test_x.shape}"
    assert torch.equal(train_y, torch.arange(100).repeat_interleave(10)), "Expanded train_y mismatch!"
    assert torch.equal(test_y, torch.arange(100).repeat_interleave(5)), "Expanded test_y mismatch!"

    print("[Expanded Cache Builder] All label layout assertions passed!", flush=True)

    cache_data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "metadata": {
            "model_id": model_id,
            "revision": revision,
            "fact_count": 100,
            "train_prompts_per_fact": 10,
            "test_prompts_per_fact": 5
        }
    }
    torch.save(cache_data, CACHE_EXPANDED_PATH)
    print(f"[Expanded Cache Builder] Saved expanded cache to '{CACHE_EXPANDED_PATH}'.", flush=True)


if __name__ == "__main__":
    main()
