"""
run_gate1_diagnostic_k3_k5.py
=============================

Complete implementation of K3 (Rerun Gate 1 diagnostic on K5 selected representation 'mean / center' and 'mean / none').

- Fit transform ONCE on full train vectors.
- Nested class subsets k in {10, 25, 50, 100} using single seeded class permutation (seed 42).
- Evaluate NCM, 1-NN, and 5-seed HeadL1c (seeds 42..46).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
SEEDS = [42, 43, 44, 45, 46]


def py_mean(vals):
    return float(sum(vals)) / float(len(vals))


def py_std(vals):
    m = py_mean(vals)
    return float((sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5)


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


def eval_diagnostic_for_rep(raw_tr_x, train_y, raw_te_x, test_y, transform_type, class_counts=[10, 25, 50, 100]):
    if transform_type == "none":
        tr_x = F.normalize(raw_tr_x, dim=-1)
        te_x = F.normalize(raw_te_x, dim=-1)
    elif transform_type == "center":
        mu = raw_tr_x.mean(dim=0, keepdim=True)
        tr_x = F.normalize(raw_tr_x - mu, dim=-1)
        te_x = F.normalize(raw_te_x - mu, dim=-1)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")

    # Seeded class permutation
    torch.manual_seed(42)
    class_perm = torch.randperm(100)

    results = []

    for k in class_counts:
        sub_classes = torch.sort(class_perm[:k])[0]
        sub_classes_set = set(sub_classes.tolist())

        # Filter train
        tr_mask = torch.tensor([y.item() in sub_classes_set for y in train_y])
        sub_tr_x = tr_x[tr_mask]
        sub_tr_y_raw = train_y[tr_mask]

        # Filter test
        te_mask = torch.tensor([y.item() in sub_classes_set for y in test_y])
        sub_te_x = te_x[te_mask]
        sub_te_y_raw = test_y[te_mask]

        # Remap labels to 0..k-1
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

        # 3. HeadL1c (5 seeds)
        head_accs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = HeadL1c(in_features=960, out_features=k, scale=10.0)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

            best_val_acc = -1.0
            best_w = None
            patience = 0

            # 3-fold prompt split for validation early stopping
            val_indices = []
            tr_indices = []
            for c in range(k):
                c_idxs = (sub_tr_y == c).nonzero(as_tuple=True)[0]
                val_indices.append(c_idxs[0].item())
                tr_indices.extend(c_idxs[1:].tolist())

            fold_tr_x = sub_tr_x[tr_indices]
            fold_tr_y = sub_tr_y[tr_indices]
            fold_val_x = sub_tr_x[val_indices]
            fold_val_y = sub_tr_y[val_indices]

            for _ in range(100):
                model.train()
                opt.zero_grad()
                logits = model(fold_tr_x)
                loss = F.cross_entropy(logits, fold_tr_y)
                loss.backward()
                opt.step()

                model.eval()
                with torch.no_grad():
                    v_acc = (model(fold_val_x).argmax(dim=1) == fold_val_y).float().mean().item() * 100.0
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    best_w = {n: p.clone() for n, p in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 15:
                        break

            if best_w is not None:
                model.load_state_dict(best_w)
            model.eval()
            with torch.no_grad():
                preds = model(sub_te_x).argmax(dim=1)
                head_accs.append((preds == sub_te_y).float().mean().item() * 100.0)

        results.append({
            "k": k,
            "ncm": ncm_acc,
            "1nn": knn_acc,
            "head_mean": py_mean(head_accs),
            "head_std": py_std(head_accs)
        })

    return results


def main():
    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing cache file: '{CACHE_PATH}'.")

    d = torch.load(CACHE_PATH, weights_only=False)
    raw_tr_x, train_y = d["train_x"], d["train_y"]
    raw_te_x, test_y = d["test_x"], d["test_y"]

    print("=========================================================================================================", flush=True)
    print(" K3 — RERUN GATE 1 DIAGNOSTIC ON K5 SELECTED REPRESENTATION ('mean / center') & 'mean / none'", flush=True)
    print("=========================================================================================================", flush=True)

    res_center = eval_diagnostic_for_rep(raw_tr_x, train_y, raw_te_x, test_y, "center")
    res_none = eval_diagnostic_for_rep(raw_tr_x, train_y, raw_te_x, test_y, "none")

    print("\nTABLE 1: K5 Selected Representation ('mean / center') Subsampling Diagnostic:", flush=True)
    print(f"{'Class Count (k)':<16} | {'NCM Top-1':<12} | {'1-NN Top-1':<12} | {'HeadL1c (5-seed mean +/- std)':<30}", flush=True)
    print("-" * 75, flush=True)
    for r in res_center:
        print(f"{r['k']:<16} | {r['ncm']:6.2f}%     | {r['1nn']:6.2f}%     | {r['head_mean']:5.2f}% +/- {r['head_std']:4.2f}%", flush=True)

    print("\nTABLE 2: Previous Representation ('mean / none') Subsampling Diagnostic:", flush=True)
    print(f"{'Class Count (k)':<16} | {'NCM Top-1':<12} | {'1-NN Top-1':<12} | {'HeadL1c (5-seed mean +/- std)':<30}", flush=True)
    print("-" * 75, flush=True)
    for r in res_none:
        print(f"{r['k']:<16} | {r['ncm']:6.2f}%     | {r['1nn']:6.2f}%     | {r['head_mean']:5.2f}% +/- {r['head_std']:4.2f}%", flush=True)

    # Monotonicity check on K5 selected table
    ncm_vals = [r["ncm"] for r in res_center]
    knn_vals = [r["1nn"] for r in res_center]
    head_vals = [r["head_mean"] for r in res_center]

    # Subsets k in [10, 25, 50, 100]. Check non-increasing as k increases
    ncm_mono = all(ncm_vals[i] >= ncm_vals[i+1] for i in range(len(ncm_vals)-1))
    knn_mono = all(knn_vals[i] >= knn_vals[i+1] for i in range(len(knn_vals)-1))
    head_mono = all(head_vals[i] >= head_vals[i+1] for i in range(len(head_vals)-1))

    print("\nMONOTONICITY CHECK ON K5 SELECTED TABLE ('mean / center'):", flush=True)
    print(f"  NCM monotonic non-increasing in k: {ncm_mono} ({ncm_vals})", flush=True)
    print(f"  1-NN monotonic non-increasing in k: {knn_mono} ({knn_vals})", flush=True)
    print(f"  HeadL1c monotonic non-increasing in k: {head_mono} ({head_vals})", flush=True)


if __name__ == "__main__":
    main()
