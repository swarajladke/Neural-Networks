"""
diagnose_cache_layout.py
========================

Diagnoses the exact structure and label layout of smollm2_embeddings_100slots.pt.
"""

import torch
import collections

CACHE_PATH = "smollm2_embeddings_100slots.pt"


def diagnose():
    print(f"==================================================")
    print(f" Diagnosing Cache Layout: {CACHE_PATH}")
    print(f"==================================================")

    data = torch.load(CACHE_PATH, map_location="cpu")

    tr_x = data["train_x"]
    tr_y = data["train_y"]
    te_x = data["test_x"]
    te_y = data["test_y"]

    print(f"Total row counts:")
    print(f"  train_x shape: {tr_x.shape}, train_y shape: {tr_y.shape}")
    print(f"  test_x shape:  {te_x.shape},  test_y shape:  {te_y.shape}")
    print(f"  train lengths match: {len(tr_x) == len(tr_y)}")
    print(f"  test  lengths match: {len(te_x) == len(te_y)}")

    # Convert y tensors to Python lists if tensor
    tr_y_list = tr_y.tolist() if isinstance(tr_y, torch.Tensor) else list(tr_y)
    te_y_list = te_y.tolist() if isinstance(te_y, torch.Tensor) else list(te_y)

    tr_counter = collections.Counter(tr_y_list)
    te_counter = collections.Counter(te_y_list)

    print(f"\nDistinct per-fact counts:")
    print(f"  train_y distinct counts: {set(tr_counter.values())}")
    print(f"  test_y  distinct counts: {set(te_counter.values())}")

    # Check ascending order
    tr_sorted = tr_y_list == sorted(tr_y_list)
    te_sorted = te_y_list == sorted(te_y_list)
    print(f"\nLabel array ordering:")
    print(f"  train_y is sorted ascending: {tr_sorted}")
    print(f"  test_y  is sorted ascending: {te_sorted}")

    # Mismatched fact IDs
    mismatched_train = {fact_id: count for fact_id, count in tr_counter.items() if count != 3}
    mismatched_test  = {fact_id: count for fact_id, count in te_counter.items() if count != 4}

    print(f"\nFacts where train_y count != 3 (Total: {len(mismatched_train)}):")
    if mismatched_train:
        for f, c in sorted(mismatched_train.items()):
            print(f"  Fact {f}: count = {c}")
    else:
        print("  None (all 100 facts have count == 3)")

    print(f"\nFacts where test_y count != 4 (Total: {len(mismatched_test)}):")
    if mismatched_test:
        for f, c in sorted(mismatched_test.items()):
            print(f"  Fact {f}: count = {c}")
    else:
        print("  None (all 100 facts have count == 4)")

    # Print first 30 entries of train_y and test_y to inspect layout pattern
    print(f"\nFirst 30 train_y labels: {tr_y_list[:30]}")
    print(f"First 30 test_y labels:  {te_y_list[:30]}")


if __name__ == "__main__":
    diagnose()
