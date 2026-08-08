"""
preregister_l4_intrinsic_dim.py  --  Pre-register L4 Intrinsic Dimension Predictions
=====================================================================================

1. Measure intrinsic dimension independently via SVD cumulative variance threshold E_90 (90% variance explained).
2. Compute E_90 for 3 Task Groupings:
   - Task 1: Base Phase (50 facts / 150 train samples)
   - Task 2: Full Dataset (100 facts / 300 train samples)
   - Task 3: Confusable Sub-Block (34 evaluated facts / 102 train samples)
3. Pre-register predicted peak k for each task into preregistered_l4_predictions.json before running the k sweep.
"""

import os
import json
import torch
import numpy as np

CACHE_PATH   = ("smollm2_embeddings_100slots.pt"
                if os.path.exists("smollm2_embeddings_100slots.pt")
                else "../smollm2_embeddings_100slots.pt")

def compute_intrinsic_dim_e90(X, threshold=0.90):
    """Computes SVD cumulative variance threshold ID_90."""
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    var = S ** 2
    cum_var = torch.cumsum(var, dim=0) / torch.sum(var)
    id_90 = int(torch.searchsorted(cum_var, threshold).item()) + 1
    return id_90, cum_var.numpy().tolist()

def main():
    print("=" * 80)
    print("  PRE-REGISTERING L4 INTRINSIC DIMENSION PREDICTIONS (E_90 THRESHOLD)")
    print("=" * 80)

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache file {CACHE_PATH} not found.")

    cache_data = torch.load(CACHE_PATH, map_location="cpu")
    train_x = cache_data["train_x"].float()
    train_y = cache_data["train_y"]

    # Task 1: Base Phase (first 50 facts, samples 0..149)
    x_task1 = train_x[:150]
    id90_task1, cumvar1 = compute_intrinsic_dim_e90(x_task1)

    # Task 2: Full Dataset (100 facts, all 300 samples)
    x_task2 = train_x
    id90_task2, cumvar2 = compute_intrinsic_dim_e90(x_task2)

    # Task 3: Confusable Sub-Block (34 evaluated facts, classes 0..33)
    valid_classes = [c.item() for c in torch.unique(train_y) if (train_y == c).sum() > 0]
    mask_task3 = torch.tensor([y.item() in valid_classes for y in train_y])
    x_task3 = train_x[mask_task3]
    id90_task3, cumvar3 = compute_intrinsic_dim_e90(x_task3)

    print(f"  Estimator: SVD Cumulative Variance Threshold E_90 (90% variance explained)")
    print(f"  Task 1 (Base Phase 50 Facts / 150 Samples):   E_90 Intrinsic Dimension = {id90_task1}")
    print(f"  Task 2 (Full Dataset 100 Facts / 300 Samples):  E_90 Intrinsic Dimension = {id90_task2}")
    print(f"  Task 3 (Confusable 34 Facts / 102 Samples):    E_90 Intrinsic Dimension = {id90_task3}")

    predictions = {
        "estimator": "SVD Cumulative Variance Threshold E_90 (90% variance explained)",
        "task1_base_phase": {
            "num_facts": 50,
            "num_samples": 150,
            "id_90": id90_task1,
            "predicted_peak_k": id90_task1
        },
        "task2_full_dataset": {
            "num_facts": 100,
            "num_samples": 300,
            "id_90": id90_task2,
            "predicted_peak_k": id90_task2
        },
        "task3_confusable_subblock": {
            "num_facts": 34,
            "num_samples": 102,
            "id_90": id90_task3,
            "predicted_peak_k": id90_task3
        }
    }

    with open("preregistered_l4_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("\nSaved pre-registered predictions to preregistered_l4_predictions.json.")
    print("Pre-registration step COMPLETE.")

if __name__ == "__main__":
    main()
