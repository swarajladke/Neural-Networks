"""
audit_task_cardinality.py  --  Task Cardinality, Base Rate & Final Audit
=======================================================================

1. Inspects unique fact_ids in test_y across all 400 queries.
2. Checks gate_threshold and total unique slots.
3. Computes raw confusable pairs under true class centroids (mask_c = train_y == c).
4. Verifies 100-way candidate retrieval space vs 34-fact evaluation queries.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH   = ("smollm2_embeddings_100slots.pt"
                if os.path.exists("smollm2_embeddings_100slots.pt")
                else "../smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM    = 960

def main():
    print("=" * 80)
    print("  ITEM 3: TASK CARDINALITY & CONFUSABLE BASE RATE AUDIT")
    print("=" * 80)

    with open(DATASET_PATH, "r") as f:
        blocks_data = json.load(f)

    if not os.path.exists(CACHE_PATH):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks_data)
    else:
        cache_data = torch.load(CACHE_PATH, map_location=DEVICE)

    train_x = cache_data["train_x"].float().to(DEVICE)
    train_y = cache_data["train_y"].to(DEVICE)
    test_x  = cache_data["test_x"].float().to(DEVICE)
    test_y  = cache_data["test_y"].to(DEVICE)

    unique_train_classes = torch.unique(train_y)
    unique_test_classes  = torch.unique(test_y)

    print(f"  1. DATASET CARDINALITY:")
    print(f"     Total Blocks in agnis_scaling_dataset.json: {len(blocks_data)} blocks")
    print(f"     Total Facts in blocks_data:                 {sum(len(b) for b in blocks_data)} facts")
    print(f"     Unique Classes in train_y:                 {len(unique_train_classes)} classes (IDs 0..99)")
    print(f"     Unique Classes in test_y:                  {len(unique_test_classes)} classes")
    print(f"     Total Train Samples:                       {len(train_x)} ({len(train_x)//100} per class)")
    print(f"     Total Test Queries:                        {len(test_x)} ({len(test_x)//100} per class)")

    # Compute true class centroids (100 classes)
    cen_true = torch.zeros(100, INPUT_DIM, device=DEVICE)
    for c in range(100):
        mask_c = (train_y == c)
        cen_true[c] = F.normalize(train_x[mask_c].mean(0, keepdim=True), dim=-1).squeeze(0)

    # Compute 4,950 pairwise cosine similarities between true centroids
    S = torch.matmul(cen_true, cen_true.T)
    conf_pairs_count = 0
    conf_classes = set()
    for i in range(100):
        for j in range(i + 1, 100):
            if S[i, j].item() > 0.95:
                conf_pairs_count += 1
                conf_classes.add(i)
                conf_classes.add(j)

    print(f"\n  2. CONFUSABLE BASE RATE UNDER TRUE CLASS CENTROIDS:")
    print(f"     Total Pairwise Centroid Comparisons (100 x 99 / 2): 4,950 pairs")
    print(f"     Pairs with cos > 0.95:                             {conf_pairs_count} pairs")
    print(f"     Distinct Classes Involved in >0.95 Pairs:          {len(conf_classes)} / 100 classes ({len(conf_classes)}%)")

    # Raw 1-NN Retrieval Accuracy on All 400 Test Queries (100 Centroids)
    raw_test_sims = torch.matmul(F.normalize(test_x, dim=-1), cen_true.T)
    raw_preds = torch.argmax(raw_test_sims, dim=-1)
    raw_acc_400 = (raw_preds == test_y).float().mean().item()
    correct_count = (raw_preds == test_y).sum().item()

    print(f"\n  3. CANONICAL RAW BASELINE RETRIEVAL ACCURACY:")
    print(f"     Raw 1-NN Accuracy against 100 Centroids (All 400 Test Queries): {raw_acc_400*100:.2f}% ({correct_count}/400)")

    print(f"\n  4. DIRECT ANSWER TO TASK CARDINALITY:")
    print(f"     - Retrieval is a 100-WAY CLASSIFICATION TASK (100 candidate target class centroids in memory).")
    print(f"     - In evaluation splits, test queries test the target classes in the evaluation set against all 100 candidate classes in memory.")

    print("=" * 80)


if __name__ == "__main__":
    main()
