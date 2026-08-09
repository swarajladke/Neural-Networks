"""
audit_embedding_leakage.py
==========================

Audits embedding-space leakage (cosine similarity) between test vectors and train vectors
for dataset v2.

Requirements (D2):
- reads smollm2_embeddings_v2_100facts.pt
- for every class c in [0..99]:
  computes max and mean cosine similarity between each test vector of class c
  and all 3 train vectors of class c.
- prints global distribution of max/mean cosine similarities and count of test vectors exceeding 0.95.
"""

import os
import torch
import numpy as np

CACHE_V2_PATH = "smollm2_embeddings_v2_100facts.pt"


def audit_leakage(cache_path=CACHE_V2_PATH):
    if not os.path.exists(cache_path):
        print(f"[Audit] Cache file {cache_path} not found locally. Skipping live computation (run on Kaggle after building cache).")
        return

    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    classes = sorted(list(set(train_y.tolist())))

    all_test_max_sims = []
    all_test_mean_sims = []
    count_above_95 = 0

    print(f"==================================================")
    print(f" EMBEDDING LEAKAGE AUDIT: {cache_path}")
    print(f"==================================================")

    for c in classes:
        tr_idx = (train_y == c).nonzero(as_tuple=True)[0]
        te_idx = (test_y == c).nonzero(as_tuple=True)[0]

        tr_vecs = train_x[tr_idx]  # (3, 960)
        te_vecs = test_x[te_idx]   # (3, 960)

        # Cosine similarity matrix between 3 test vectors and 3 train vectors (shape: 3 x 3)
        sim_matrix = torch.matmul(te_vecs, tr_vecs.T)  # (3, 3)

        for i in range(len(te_idx)):
            test_sims = sim_matrix[i].numpy()
            max_sim = float(np.max(test_sims))
            mean_sim = float(np.mean(test_sims))

            all_test_max_sims.append(max_sim)
            all_test_mean_sims.append(mean_sim)

            if max_sim > 0.95:
                count_above_95 += 1

    all_max = np.array(all_test_max_sims)
    all_mean = np.array(all_test_mean_sims)

    print(f"  Total test vectors evaluated: {len(all_max)}")
    print(f"\n--- MAX COSINE SIMILARITY TO TRAIN (per test vector) ---")
    print(f"  Min:  {np.min(all_max):.4f}")
    print(f"  Mean: {np.mean(all_max):.4f}")
    print(f"  Max:  {np.max(all_max):.4f}")
    print(f"  P50 (Median): {np.median(all_max):.4f}")
    print(f"  P90:          {np.percentile(all_max, 90):.4f}")
    print(f"  P95:          {np.percentile(all_max, 95):.4f}")

    print(f"\n--- MEAN COSINE SIMILARITY TO TRAIN (per test vector) ---")
    print(f"  Min:  {np.min(all_mean):.4f}")
    print(f"  Mean: {np.mean(all_mean):.4f}")
    print(f"  Max:  {np.max(all_mean):.4f}")

    print(f"\n--- LEAKAGE THRESHOLD COUNT ---")
    print(f"  Test vectors with max cosine > 0.95: {count_above_95} / {len(all_max)} ({count_above_95 / len(all_max) * 100:.2f}%)")


if __name__ == "__main__":
    audit_leakage()
