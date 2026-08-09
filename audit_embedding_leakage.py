"""
audit_embedding_leakage.py
===========================

Audits embedding space similarity, margins, and NCM upper bound (H2).

Requirements (H2):
- Reads smollm2_embeddings_v2_100facts.pt (raises RuntimeError if missing).
- OWN-CLASS vs BETWEEN-CLASS:
  - For each test vector (300 total):
    - own_class_max: max cosine to train vectors of its own class (3 train vectors).
    - other_class_max: max cosine to train vectors of ALL OTHER 99 classes (297 train vectors).
    - margin = own_class_max - other_class_max.
  - Reports distribution of margin (min, mean, std, max) and count of test vectors with negative margin.
- NCM UPPER BOUND:
  - Builds 1 centroid per class from its 3 train vectors: centroids = train_x.view(100, 3, 960).mean(dim=1).
  - L2-normalizes centroids and computes top-1 accuracy over all 300 test vectors.
- Prints all raw outputs cleanly.
"""

import os
import torch
import torch.nn.functional as F

CACHE_V2_PATH = "smollm2_embeddings_v2_100facts.pt"


def main():
    if not os.path.exists(CACHE_V2_PATH):
        raise RuntimeError(f"Missing required cache file: '{CACHE_V2_PATH}'. Cannot run audit.")

    print(f"[Embedding Audit] Loading cache from '{CACHE_V2_PATH}'...")
    data = torch.load(CACHE_V2_PATH, weights_only=False)

    train_x = data["train_x"]  # shape (300, 960)
    train_y = data["train_y"]  # shape (300,)
    test_x = data["test_x"]    # shape (300, 960)
    test_y = data["test_y"]    # shape (300,)

    # Normalize vectors just in case
    train_x = F.normalize(train_x, dim=-1)
    test_x = F.normalize(test_x, dim=-1)

    print("\n==================================================")
    print(" H2 EMBEDDING LEAKAGE & MARGIN AUDIT")
    print("==================================================")

    # 1. PER-CLASS OWN-CLASS vs OTHER-CLASS SIMILARITIES & MARGINS
    own_class_maxes = []
    other_class_maxes = []
    margins = []
    negative_margin_count = 0

    for i in range(300):
        t_vec = test_x[i]
        c_label = test_y[i].item()

        # Cosine similarities to all 300 train vectors
        all_cos = torch.mv(train_x, t_vec)  # shape (300,)

        # Mask for own class train vectors
        own_mask = (train_y == c_label)
        other_mask = ~own_mask

        own_cos = all_cos[own_mask]
        other_cos = all_cos[other_mask]

        own_max = own_cos.max().item()
        other_max = other_cos.max().item()
        margin = own_max - other_max

        own_class_maxes.append(own_max)
        other_class_maxes.append(other_max)
        margins.append(margin)

        if margin < 0:
            negative_margin_count += 1

    margins_t = torch.tensor(margins)
    own_t = torch.tensor(own_class_maxes)
    other_t = torch.tensor(other_class_maxes)

    print(f"Total Test Vectors: {len(test_x)}")
    print(f"Own-Class Max Cosine  : Mean = {own_t.mean().item():.4f}, Min = {own_t.min().item():.4f}, Max = {own_t.max().item():.4f}")
    print(f"Other-Class Max Cosine: Mean = {other_t.mean().item():.4f}, Min = {other_t.min().item():.4f}, Max = {other_t.max().item():.4f}")
    print(f"Margin (Own - Other)  : Mean = {margins_t.mean().item():.4f}, Std = {margins_t.std().item():.4f}, Min = {margins_t.min().item():.4f}, Max = {margins_t.max().item():.4f}")
    print(f"Negative Margin Count : {negative_margin_count} / {len(test_x)} ({negative_margin_count / len(test_x) * 100:.2f}%)")

    # 2. NEAREST-CENTROID CLASSIFIER (NCM) UPPER BOUND
    print("\n==================================================")
    print(" H2 NEAREST-CENTROID (NCM) UPPER BOUND")
    print("==================================================")

    # Reshape train_x to (100 classes, 3 samples, 960)
    train_x_by_class = train_x.view(100, 3, 960)
    centroids = train_x_by_class.mean(dim=1)  # shape (100, 960)
    centroids = F.normalize(centroids, dim=-1)

    # Compute cosine matrix between 300 test vectors and 100 centroids
    # test_x shape (300, 960), centroids shape (100, 960) -> sim_matrix (300, 100)
    sim_matrix = test_x @ centroids.T
    preds = sim_matrix.argmax(dim=1)

    correct_mask = (preds == test_y)
    ncm_accuracy = correct_mask.float().mean().item()
    correct_count = correct_mask.sum().item()

    print(f"Centroids Built       : 100 classes (3 train prompts per centroid)")
    print(f"Test Queries Evaluated: 300 test prompts")
    print(f"NCM Correct Predictions: {correct_count} / 300")
    print(f"NCM Top-1 Accuracy    : {ncm_accuracy * 100:.2f}% ({ncm_accuracy:.4f})")
    print("==================================================")


if __name__ == "__main__":
    main()
