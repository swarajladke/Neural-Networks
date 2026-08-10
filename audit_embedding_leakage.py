"""
audit_embedding_leakage.py
===========================

Audits embedding space similarity, margins, and NCM upper bound (PHASE I).

Standing Rules Applied:
- R4: Guard raises on missing input file.
- R6: Label-derived grouping via (y == c).nonzero(as_tuple=True)[0]. No .view(N, k, d) or fixed-stride slicing.
- R7: Statistics/centroids fit on train data only. Explicit confirmation line printed.
- I1a: Relabel biased 1-NN margin line.
- I1b: Add unbiased margin comparison (own_mean over 3 vs other_mean over 297).
- I1c: Use label-derived grouping for NCM centroids.
"""

import os
import torch
import torch.nn.functional as F

CACHE_V2_PATH = "smollm2_embeddings_v2_100facts.pt"


def main():
    # R4 & R5 Guard: Raise immediately if required cache file is missing
    if not os.path.exists(CACHE_V2_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required cache file: '{CACHE_V2_PATH}'. Cannot proceed.")

    print(f"[Phase I Audit] Loading cache from '{CACHE_V2_PATH}'...")
    data = torch.load(CACHE_V2_PATH, weights_only=False)

    train_x = data["train_x"]  # shape (300, 960)
    train_y = data["train_y"]  # shape (300,)
    test_x = data["test_x"]    # shape (300, 960)
    test_y = data["test_y"]    # shape (300,)

    # Normalize vectors
    train_x = F.normalize(train_x, dim=-1)
    test_x = F.normalize(test_x, dim=-1)

    # R7 Confirmation
    print("[R7 CONFIRMATION] No test data used to fit any statistic or centroid.")

    print("\n==================================================")
    print(" PHASE I — EMBEDDING LEAKAGE & MARGIN AUDIT")
    print("==================================================")

    own_class_maxes = []
    other_class_maxes = []
    biased_margins = []
    biased_negative_count = 0

    unbiased_diffs = []
    unbiased_negative_count = 0

    for i in range(300):
        t_vec = test_x[i]
        c_label = test_y[i].item()

        # Cosine similarities to all 300 train vectors
        all_cos = torch.mv(train_x, t_vec)  # shape (300,)

        # R6: Label-derived indexing
        own_mask = (train_y == c_label)
        other_mask = ~own_mask

        own_cos = all_cos[own_mask]
        other_cos = all_cos[other_mask]

        # 1-NN biased maxes (I1a)
        own_max = own_cos.max().item()
        other_max = other_cos.max().item()
        b_margin = own_max - other_max

        own_class_maxes.append(own_max)
        other_class_maxes.append(other_max)
        biased_margins.append(b_margin)

        if b_margin < 0:
            biased_negative_count += 1

        # Unbiased means comparison (I1b)
        own_mean = own_cos.mean().item()
        other_mean = other_cos.mean().item()
        u_diff = own_mean - other_mean
        unbiased_diffs.append(u_diff)

        if u_diff < 0:
            unbiased_negative_count += 1

    b_margins_t = torch.tensor(biased_margins)
    u_diffs_t = torch.tensor(unbiased_diffs)
    own_t = torch.tensor(own_class_maxes)
    other_t = torch.tensor(other_class_maxes)

    print(f"Total Test Vectors: {len(test_x)}")
    print(f"Own-Class Max Cosine  : Mean = {own_t.mean().item():.4f}, Min = {own_t.min().item():.4f}, Max = {own_t.max().item():.4f}")
    print(f"Other-Class Max Cosine: Mean = {other_t.mean().item():.4f}, Min = {other_t.min().item():.4f}, Max = {other_t.max().item():.4f}")

    # I1a Relabeled output line
    print(f"1-NN margin (own_max over 3 vs other_max over 297; biased by set size, only the sign is meaningful): Mean = {b_margins_t.mean().item():.4f}, Std = {b_margins_t.std().item():.4f}, Min = {b_margins_t.min().item():.4f}, Max = {b_margins_t.max().item():.4f}, Count Negative = {biased_negative_count} / 300 ({biased_negative_count / 300 * 100:.2f}%)")

    # I1b Unbiased comparison line
    print(f"Unbiased margin (own_mean over 3 vs other_mean over 297): Mean = {u_diffs_t.mean().item():.4f}, Std = {u_diffs_t.std().item():.4f}, Min = {u_diffs_t.min().item():.4f}, Max = {u_diffs_t.max().item():.4f}, Count Negative = {unbiased_negative_count} / 300 ({unbiased_negative_count / 300 * 100:.2f}%)")

    # 2. NEAREST-CENTROID CLASSIFIER (NCM) UPPER BOUND (I1c)
    print("\n==================================================")
    print(" PHASE I — NEAREST-CENTROID (NCM) UPPER BOUND (R6)")
    print("==================================================")

    # I1c & R6: Label-derived centroid construction
    unique_classes = torch.sort(torch.unique(train_y))[0]
    centroids_list = []
    for c in unique_classes:
        c_val = c.item()
        c_indices = (train_y == c_val).nonzero(as_tuple=True)[0]
        c_vecs = train_x[c_indices]
        centroids_list.append(c_vecs.mean(dim=0))

    centroids = torch.stack(centroids_list)  # shape (100, 960)
    centroids = F.normalize(centroids, dim=-1)

    # Compute cosine matrix between 300 test vectors and 100 centroids
    sim_matrix = test_x @ centroids.T
    preds = sim_matrix.argmax(dim=1)

    correct_mask = (preds == test_y)
    ncm_accuracy = correct_mask.float().mean().item()
    correct_count = correct_mask.sum().item()

    print(f"Centroids Built       : {len(unique_classes)} classes (label-derived R6 grouping)")
    print(f"Test Queries Evaluated: {len(test_x)} test prompts")
    print(f"NCM Correct Predictions: {correct_count} / 300")
    print(f"NCM Top-1 Accuracy    : {ncm_accuracy * 100:.2f}% ({ncm_accuracy:.4f})")
    print("==================================================")


if __name__ == "__main__":
    main()
