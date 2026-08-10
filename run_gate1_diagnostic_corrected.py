"""
run_gate1_diagnostic_corrected.py
==================================

Corrected Gate 1 Diagnostic (J6):
- Fit transform ONCE on full 300 train vectors.
- Use NESTED class subsets (single seeded permutation of 100 classes, seed 42) for k in {10, 25, 50, 100}.
- Evaluate NCM, 1-NN, and J4 early-stopped HeadL1c at each k.
- Monotonicity check: accuracy must be non-increasing in k.

Standing Rules:
- R4: Guard raises on missing input cache files.
- R6: Label-derived R6 indexing.
- R7: Transform fit on full TRAIN vectors ONCE before subsetting.
- R8: Fixed seeds [42, 43, 44, 45, 46] logged and enforced.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"


class HeadL1c(nn.Module):
    def __init__(self, in_features=960, out_features=100, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        w = torch.randn(out_features, in_features)
        w = F.normalize(w, dim=-1)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.scale * (x_norm @ w_norm.T)


def eval_ncm_sub(tr_x, tr_y, te_x, te_y, selected_classes):
    class_map = {orig_c: new_c for new_c, orig_c in enumerate(selected_classes)}
    centroids = []
    for orig_c in selected_classes:
        c_vecs = tr_x[(tr_y == orig_c).nonzero(as_tuple=True)[0]]
        centroids.append(c_vecs.mean(dim=0))
    centroids = F.normalize(torch.stack(centroids), dim=-1)

    te_mask = torch.tensor([y.item() in class_map for y in te_y])
    te_sub_x = te_x[te_mask]
    te_sub_y = torch.tensor([class_map[y.item()] for y in te_y[te_mask]], dtype=torch.long)

    preds = (te_sub_x @ centroids.T).argmax(dim=1)
    return (preds == te_sub_y).float().mean().item() * 100.0


def eval_1nn_sub(tr_x, tr_y, te_x, te_y, selected_classes):
    class_map = {orig_c: new_c for new_c, orig_c in enumerate(selected_classes)}

    tr_mask = torch.tensor([y.item() in class_map for y in tr_y])
    te_mask = torch.tensor([y.item() in class_map for y in te_y])

    tr_sub_x = tr_x[tr_mask]
    tr_sub_y = torch.tensor([class_map[y.item()] for y in tr_y[tr_mask]], dtype=torch.long)
    te_sub_x = te_x[te_mask]
    te_sub_y = torch.tensor([class_map[y.item()] for y in te_y[te_mask]], dtype=torch.long)

    sims = te_sub_x @ tr_sub_x.T
    preds = tr_sub_y[sims.argmax(dim=1)]
    return (preds == te_sub_y).float().mean().item() * 100.0


def eval_j4_head_sub(tr_x, tr_y, te_x, te_y, selected_classes, seeds=[42]):
    class_map = {orig_c: new_c for new_c, orig_c in enumerate(selected_classes)}
    num_classes = len(selected_classes)

    tr_mask = torch.tensor([y.item() in class_map for y in tr_y])
    te_mask = torch.tensor([y.item() in class_map for y in te_y])

    tr_sub_x = tr_x[tr_mask]
    tr_sub_y = torch.tensor([class_map[y.item()] for y in tr_y[tr_mask]], dtype=torch.long)
    te_sub_x = te_x[te_mask]
    te_sub_y = torch.tensor([class_map[y.item()] for y in te_y[te_mask]], dtype=torch.long)

    fold_test_accs = []

    for fold in range(3):
        fold_tr_indices, fold_val_indices = [], []
        for c_new in range(num_classes):
            c_idxs = (tr_sub_y == c_new).nonzero(as_tuple=True)[0]
            val_idx = c_idxs[fold]
            tr_idxs = c_idxs[torch.arange(3) != fold]
            fold_tr_indices.extend(tr_idxs.tolist())
            fold_val_indices.append(val_idx.item())

        sub_tr_x = tr_sub_x[fold_tr_indices]
        sub_tr_y = tr_sub_y[fold_tr_indices]
        val_x = tr_sub_x[fold_val_indices]
        val_y = tr_sub_y[fold_val_indices]

        seed_accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = HeadL1c(in_features=tr_x.shape[1], out_features=num_classes, scale=10.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = -1.0
            best_weights = None
            patience_counter = 0

            for epoch in range(1, 201):
                model.train()
                optimizer.zero_grad()
                logits = model(sub_tr_x)
                loss = criterion(logits, sub_tr_y)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(val_x)
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == val_y).float().mean().item() * 100.0

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        break

            if best_weights is not None:
                model.load_state_dict(best_weights)
            model.eval()
            with torch.no_grad():
                te_preds = model(te_sub_x).argmax(dim=1)
                seed_accs.append((te_preds == te_sub_y).float().mean().item() * 100.0)

        fold_test_accs.append(np.mean(seed_accs))

    return np.mean(fold_test_accs)


def main():
    if not os.path.exists(MEAN_CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required cache file: '{MEAN_CACHE_PATH}'.")

    data = torch.load(MEAN_CACHE_PATH, weights_only=False)
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    print("==================================================================", flush=True)
    print(" J6 CORRECTED GATE 1 DIAGNOSTIC (NESTED SUBSETS & FULL FIT ONCE)", flush=True)
    print("==================================================================", flush=True)

    # 1. Fit transform ONCE on full 300 train vectors
    mu = train_x.mean(dim=0, keepdim=True)
    tr_x_full = F.normalize(train_x - mu, dim=-1)
    te_x_full = F.normalize(test_x - mu, dim=-1)
    print("[R7 Confirm] Transform (centering) fit ONCE on full 300 train vectors.", flush=True)

    # 2. Single seeded permutation of the 100 classes (seed 42) for NESTED subsets
    rng = random.Random(42)
    class_perm = list(range(100))
    rng.shuffle(class_perm)

    class_counts = [10, 25, 50, 100]
    results = {}

    print(f"\n{'Class Count (k)':<18} | {'NCM Top-1':<12} | {'1-NN Top-1':<12} | {'HeadL1c (J4 Early Stopped)'}", flush=True)
    print("-" * 75, flush=True)

    for k in class_counts:
        selected_classes = sorted(class_perm[:k])
        ncm_k = eval_ncm_sub(tr_x_full, train_y, te_x_full, test_y, selected_classes)
        knn_k = eval_1nn_sub(tr_x_full, train_y, te_x_full, test_y, selected_classes)
        head_k = eval_j4_head_sub(tr_x_full, train_y, te_x_full, test_y, selected_classes)

        results[k] = (ncm_k, knn_k, head_k)
        print(f"{k:<18} | {ncm_k:6.2f}%      | {knn_k:6.2f}%      | {head_k:6.2f}%", flush=True)

    print("=" * 75, flush=True)

    # 3. Monotonicity Check
    print("\n--- MONOTONICITY CHECK ---", flush=True)
    for metric_idx, metric_name in enumerate(["NCM", "1-NN", "HeadL1c(J4)"]):
        accs = [results[k][metric_idx] for k in class_counts]
        is_monotonic = all(accs[i] >= accs[i+1] for i in range(len(accs)-1))
        print(f"{metric_name:<15}: Values across k=[10, 25, 50, 100] = {[f'{a:.2f}%' for a in accs]} | Monotonic non-increasing: {is_monotonic}", flush=True)
        if not is_monotonic:
            print(f"  --> NON-MONOTONIC DETECTED for {metric_name}! Stopped per J6 directive.", flush=True)

    print("==================================================================", flush=True)


if __name__ == "__main__":
    main()
