"""
build_lasttok_nonpunct_cache.py
================================

Rebuilds last-token cache using the last NON-PUNCTUATION token (J2).

Requirements (J2):
- Scans backwards from last non-pad token to find the first content token
  (not punctuation e.g. not '?', ':', '.', '!', ',', ';').
- Real assertion: verifies that for all sampled rows, the decoded token at non_punct_idx
  is not a pad token and not pure punctuation.
- Saves to smollm2_embeddings_v2_100facts_lasttok_nonpunct.pt (does NOT overwrite old cache).
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import json
import string
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_V2_PATH = "agnis_scaling_dataset_v2.json"
LASTTOK_NONPUNCT_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok_nonpunct.pt"
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
    if not os.path.exists(DATASET_V2_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required dataset file: '{DATASET_V2_PATH}'.")

    model_id, is_hf_hub = find_offline_model_path()
    revision = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

    print(f"[Non-Punct Last-Token] Loading SmolLM2 from '{model_id}'...", flush=True)
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

    with open(DATASET_V2_PATH, "r", encoding="utf-8") as f:
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

    train_queries, train_labels = [], []
    test_queries, test_labels = [], []

    punct_set = set(string.punctuation + " \t\nĠ")

    print(f"[Non-Punct Last-Token] Extracting non-punctuation last-token vectors for {len(all_prompts)} prompts in batches of 32...", flush=True)
    batch_size = 32
    sample_checks = 0

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
                last_idx = seq_len - 1

                tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[idx])
                
                # Fast backward scan for non-punctuation token
                non_punct_idx = last_idx
                for t_i in range(last_idx, -1, -1):
                    tok_str = tokens[t_i].replace("Ġ", "").strip()
                    if tok_str and not all(c in punct_set for c in tok_str):
                        non_punct_idx = t_i
                        break

                # J2 REAL ASSERTION
                target_tok_str = tokens[non_punct_idx].replace("Ġ", "").strip()
                assert target_tok_str != tokenizer.pad_token, f"Decoded token is pad token: {target_tok_str}"
                assert not all(c in punct_set for c in target_tok_str), f"Decoded token is pure punctuation: {target_tok_str}"
                sample_checks += 1

                vec = last_hidden[idx, non_punct_idx]
                vec_norm = F.normalize(vec.float(), dim=-1).cpu()

                if b_flags[idx]:
                    train_queries.append(vec_norm)
                    train_labels.append(b_labs[idx])
                else:
                    test_queries.append(vec_norm)
                    test_labels.append(b_labs[idx])

        print(f"  Batch {b_i//batch_size + 1}/{len(all_prompts)//batch_size + 1} processed...", flush=True)

    print(f"[J2 Assertion Passed] {sample_checks} / {len(all_prompts)} prompts verified: decoded last non-punct token is non-empty content token.", flush=True)

    train_x = torch.stack(train_queries)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = torch.stack(test_queries)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    expected_labels = torch.arange(100).repeat_interleave(3)
    assert torch.equal(train_y, expected_labels), "Non-punct lasttok train_y mismatch!"
    assert torch.equal(test_y, expected_labels), "Non-punct lasttok test_y mismatch!"

    cache_data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "metadata": {"pooling": "last_token_nonpunct"}
    }
    torch.save(cache_data, LASTTOK_NONPUNCT_CACHE_PATH)
    print(f"[Non-Punct Last-Token] Saved cache to '{LASTTOK_NONPUNCT_CACHE_PATH}'.", flush=True)


if __name__ == "__main__":
    main()
