"""
build_cache_v3_template_split.py
================================

Extracts SmolLM2-360M mean-pooling embeddings for agnis_scaling_dataset_v3_template_split.json.

Output:
- smollm2_embeddings_v3_100facts_7_3_5.pt
  - train_x: 700 x 960 (7 train prompts per fact x 100 facts)
  - train_y: 700 (labels 0..99)
  - val_x: 300 x 960 (3 val prompts per fact x 100 facts, DISJOINT TEMPLATES)
  - val_y: 300 (labels 0..99)
  - test_x: 500 x 960 (5 test prompts per fact x 100 facts, DISJOINT TEMPLATES)
  - test_y: 500 (labels 0..99)
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModel

DATASET_PATH = "agnis_scaling_dataset_v3_template_split.json"
CACHE_PATH = "smollm2_embeddings_v3_100facts_7_3_5.pt"
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


def extract_mean_embeddings(texts, model, tokenizer, batch_size=32):
    all_embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_embeds = (outputs.last_hidden_state * mask).sum(dim=1)
            mean_embeds = sum_embeds / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(mean_embeds.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f"Missing dataset file: '{DATASET_PATH}'")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = data["facts"]

    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []

    for fact in facts:
        f_id = fact["fact_id"]

        for p in fact["train_prompts"]:
            train_texts.append(p)
            train_labels.append(f_id)

        for p in fact["val_prompts"]:
            val_texts.append(p)
            val_labels.append(f_id)

        for p in fact["test_prompts"]:
            test_texts.append(p)
            test_labels.append(f_id)

    print(f"Loaded dataset v3: {len(facts)} facts. Total prompts -> Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_NAME)

    print("Extracting SmolLM2-360M mean-pooling embeddings...")
    tr_x = extract_mean_embeddings(train_texts, model, tokenizer)
    va_x = extract_mean_embeddings(val_texts, model, tokenizer)
    te_x = extract_mean_embeddings(test_texts, model, tokenizer)

    tr_y = torch.tensor(train_labels, dtype=torch.long)
    va_y = torch.tensor(val_labels, dtype=torch.long)
    te_y = torch.tensor(test_labels, dtype=torch.long)

    assert tr_x.shape == (700, 960)
    assert va_x.shape == (300, 960)
    assert te_x.shape == (500, 960)

    cache_data = {
        "train_x": tr_x,
        "train_y": tr_y,
        "val_x": va_x,
        "val_y": va_y,
        "test_x": te_x,
        "test_y": te_y,
    }

    torch.save(cache_data, CACHE_PATH)
    print(f"Successfully saved {CACHE_PATH}: train_x={tr_x.shape}, val_x={va_x.shape}, test_x={te_x.shape}.")


if __name__ == "__main__":
    main()
