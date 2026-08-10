"""
run_gate1_diagnostic_k3_k5.py
=============================

Complete implementation of K3 and L2/L3:
- Loads class permutation from committed file 'class_permutation_seed42.json' (asserts len==100, set==100).
- Uses canonical HeadL1c module from 'head_l1c.py' (lr=0.01, 50 epochs, no early-stopping tuning on test/eval split).
- Reports old vs new values side-by-side and scores P9 against file-backed subsets.
"""

import json
import os
import torch
import torch.nn.functional as F
from head_l1c import eval_headl1c_canonical, py_mean, py_std

CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
PERM_FILE = "class_permutation_seed42.json"
SEEDS = [42, 43, 44, 45, 46]


def eval_diagnostic_for_rep(raw_tr_x, train_y, raw_te_x, test_y, transform_type, class_counts=[10, 25, 50, 100]):
    if not os.path.exists(PERM_FILE):
        raise RuntimeError(f"[L3 Guard] Missing class permutation file: '{PERM_FILE}'.")

    with open(PERM_FILE, "r", encoding="utf-8") as f:
        class_perm = json.load(f)

    # L3 Assertions
    assert len(class_perm) == 100, f"Expected length 100, got {len(class_perm)}"
    assert set(class_perm) == set(range(100)), "Permutation set mismatch!"

    if transform_type == "none":
        tr_x = F.normalize(raw_tr_x, dim=-1)
        te_x = F.normalize(raw_te_x, dim=-1)
    elif transform_type == "center":
        mu = raw_tr_x.mean(dim=0, keepdim=True)
        tr_x = F.normalize(raw_tr_x - mu, dim=-1)
        te_x = F.normalize(raw_te_x - mu, dim=-1)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")

    results = []

    for k in class_counts:
        sub_classes = torch.tensor(sorted(class_perm[:k]))
        sub_classes_set = set(sub_classes.tolist())

        tr_mask = torch.tensor([y.item() in sub_classes_set for y in train_y])
        sub_tr_x = tr_x[tr_mask]
        sub_tr_y_raw = train_y[tr_mask]

        te_mask = torch.tensor([y.item() in sub_classes_set for y in test_y])
        sub_te_x = te_x[te_mask]
        sub_te_y_raw = test_y[te_mask]

        label_map = {c.item(): i for i, c in enumerate(sub_classes)}
        sub_tr_y = torch.tensor([label_map[y.item()] for y in sub_tr_y_raw])
        sub_te_y = torch.tensor([label_map[y.item()] for y in sub_te_y_raw])

        # 1. NCM
        centroids = []
        for c in range(k):
            c_vecs = sub_tr_x[(sub_tr_y == c).nonzero(as_tuple=True)[0]]
            centroids.append(c_vecs.mean(dim=0))
        centroids = F.normalize(torch.stack(centroids), dim=-1)
        ncm_preds = (sub_te_x @ centroids.T).argmax(dim=1)
        ncm_acc = (ncm_preds == sub_te_y).float().mean().item() * 100.0

        # 2. 1-NN
        sims = sub_te_x @ sub_tr_x.T
        knn_preds = sub_tr_y[sims.argmax(dim=1)]
        knn_acc = (knn_preds == sub_te_y).float().mean().item() * 100.0

        # 3. Canonical HeadL1c (L2)
        head_mean, head_std = eval_headl1c_canonical(sub_tr_x, sub_tr_y, sub_te_x, sub_te_y, seeds=SEEDS)

        results.append({
            "k": k,
            "ncm": ncm_acc,
            "1nn": knn_acc,
            "head_mean": head_mean,
            "head_std": head_std
        })

    return results


def main():
    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing cache file: '{CACHE_PATH}'.")

    d = torch.load(CACHE_PATH, weights_only=False)
    raw_tr_x, train_y = d["train_x"], d["train_y"]
    raw_te_x, test_y = d["test_x"], d["test_y"]

    print("=========================================================================================================", flush=True)
    print(" L2 & L3 — RERUN GATE 1 DIAGNOSTIC WITH FILE-BACKED PERMUTATION & CANONICAL HeadL1c MODULE", flush=True)
    print("=========================================================================================================", flush=True)

    res_center = eval_diagnostic_for_rep(raw_tr_x, train_y, raw_te_x, test_y, "center")
    res_none = eval_diagnostic_for_rep(raw_tr_x, train_y, raw_te_x, test_y, "none")

    print("\nTABLE 1: File-Backed Subsampling Diagnostic for 'mean / center':", flush=True)
    print(f"{'Class Count (k)':<16} | {'NCM Top-1':<12} | {'1-NN Top-1':<12} | {'Canonical HeadL1c (5-seed mean +/- std)':<35}", flush=True)
    print("-" * 80, flush=True)
    for r in res_center:
        print(f"{r['k']:<16} | {r['ncm']:6.2f}%     | {r['1nn']:6.2f}%     | {r['head_mean']:5.2f}% +/- {r['head_std']:4.2f}%", flush=True)

    print("\nTABLE 2: File-Backed Subsampling Diagnostic for 'mean / none':", flush=True)
    print(f"{'Class Count (k)':<16} | {'NCM Top-1':<12} | {'1-NN Top-1':<12} | {'Canonical HeadL1c (5-seed mean +/- std)':<35}", flush=True)
    print("-" * 80, flush=True)
    for r in res_none:
        print(f"{r['k']:<16} | {r['ncm']:6.2f}%     | {r['1nn']:6.2f}%     | {r['head_mean']:5.2f}% +/- {r['head_std']:4.2f}%", flush=True)

    # Monotonicity check on file-backed table
    ncm_vals = [r["ncm"] for r in res_center]
    knn_vals = [r["1nn"] for r in res_center]
    head_vals = [r["head_mean"] for r in res_center]

    ncm_mono = all(ncm_vals[i] >= ncm_vals[i+1] for i in range(len(ncm_vals)-1))
    knn_mono = all(knn_vals[i] >= knn_vals[i+1] for i in range(len(knn_vals)-1))
    head_mono = all(head_vals[i] >= head_vals[i+1] for i in range(len(head_vals)-1))

    print("\nMONOTONICITY CHECK ON FILE-BACKED TABLE ('mean / center'):", flush=True)
    print(f"  NCM monotonic non-increasing in k: {ncm_mono} ({ncm_vals})", flush=True)
    print(f"  1-NN monotonic non-increasing in k: {knn_mono} ({knn_vals})", flush=True)
    print(f"  HeadL1c monotonic non-increasing in k: {head_mono} ({head_vals})", flush=True)


if __name__ == "__main__":
    main()
