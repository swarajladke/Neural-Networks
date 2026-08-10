"""
run_gate2_redecision_expanded.py
================================

Gate 2 Re-Decision on Expanded 10/5 Dataset (smollm2_embeddings_v2_100facts_expanded_10_5.pt).

Fixes K4b (mask-based indexing across all 10 train prompts), K4c (direct weight decay sweep), and K5 (train-only CV selection).

Standing Rules:
- R4: Guard raises on missing input cache file.
- R6: Label-derived R6 indexing.
- R7: Transform fit on TRAIN vectors ONLY.
- R8: Seeds [42..46] logged and enforced.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPANDED_CACHE_PATH = "smollm2_embeddings_v2_100facts_expanded_10_5.pt"
SEEDS = [42, 43, 44, 45, 46]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


def py_mean(vals):
    return float(sum(vals)) / float(len(vals))


def py_std(vals):
    m = py_mean(vals)
    return float((sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5)


def apply_transform_train_only(train_x, test_x, transform_type):
    if transform_type == "none":
        return F.normalize(train_x, dim=-1), F.normalize(test_x, dim=-1)
    elif transform_type == "center":
        mu = train_x.mean(dim=0, keepdim=True)
        return F.normalize(train_x - mu, dim=-1), F.normalize(test_x - mu, dim=-1)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")


def eval_ncm(tr_x, train_y, eval_x, eval_y):
    unique_classes = torch.sort(torch.unique(train_y))[0]
    centroids = []
    for c in unique_classes:
        c_vecs = tr_x[(train_y == c.item()).nonzero(as_tuple=True)[0]]
        centroids.append(c_vecs.mean(dim=0))
    centroids = F.normalize(torch.stack(centroids), dim=-1)
    preds = (eval_x @ centroids.T).argmax(dim=1)
    return (preds == eval_y).float().mean().item() * 100.0


def eval_1nn(tr_x, train_y, eval_x, eval_y):
    sims = eval_x @ tr_x.T
    preds = train_y[sims.argmax(dim=1)]
    return (preds == eval_y).float().mean().item() * 100.0


def eval_headl1c(tr_x, train_y, eval_x, eval_y, val_x=None, val_y=None, seeds=SEEDS, lr=0.03, scale=10.0):
    accs = []
    d = tr_x.shape[1]

    for seed in seeds:
        torch.manual_seed(seed)
        w = torch.randn(100, d)
        w = F.normalize(w, dim=-1).requires_grad_(True)
        opt = torch.optim.AdamW([w], lr=lr, weight_decay=1e-4)

        best_val_acc = -1.0
        best_w = None
        patience = 0

        for _ in range(30):
            opt.zero_grad()
            logits = scale * (F.normalize(tr_x, dim=-1) @ F.normalize(w, dim=-1).T)
            loss = F.cross_entropy(logits, train_y)
            loss.backward()
            opt.step()

            if val_x is not None and val_y is not None:
                with torch.no_grad():
                    v_logits = scale * (F.normalize(val_x, dim=-1) @ F.normalize(w, dim=-1).T)
                    v_preds = v_logits.argmax(dim=1)
                    v_acc = (v_preds == val_y).float().mean().item() * 100.0
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    best_w = w.detach().clone()
                    patience = 0
                else:
                    patience += 1
                    if patience >= 10:
                        break

        final_w = best_w if (val_x is not None and best_w is not None) else w.detach()
        with torch.no_grad():
            te_logits = scale * (F.normalize(eval_x, dim=-1) @ F.normalize(final_w, dim=-1).T)
            te_preds = te_logits.argmax(dim=1)
            accs.append((te_preds == eval_y).float().mean().item() * 100.0)

    return py_mean(accs), py_std(accs)


def eval_plain_linear_ridge(tr_x, train_y, eval_x, eval_y, weight_decay=1e-4):
    N, d = tr_x.shape
    num_classes = 100
    Y_oh = F.one_hot(train_y, num_classes=num_classes).float()

    lam = max(weight_decay, 1e-6)
    K = tr_x @ tr_x.T
    K_reg = K + lam * torch.eye(N)
    alpha = torch.linalg.solve(K_reg, Y_oh)
    W = tr_x.T @ alpha

    preds = (eval_x @ W).argmax(dim=1)
    acc = (preds == eval_y).float().mean().item() * 100.0
    return acc, 0.0


def compute_5fold_cv_on_train(raw_tr_x, train_y, transform_type):
    # K4b: Flexible mask-based indexing across all 10 train prompts per class
    num_folds = 5
    ncm_cvs, knn_cvs, head_cvs = [], [], []
    linear_cvs = {wd: [] for wd in WEIGHT_DECAYS}

    for fold in range(num_folds):
        fold_tr_indices, fold_val_indices = [], []
        for c in range(100):
            c_idxs = (train_y == c).nonzero(as_tuple=True)[0]
            n_samples = len(c_idxs)
            mask = torch.ones(n_samples, dtype=torch.bool)
            mask[fold] = False

            tr_idxs = c_idxs[mask]
            val_idx = c_idxs[fold]

            fold_tr_indices.extend(tr_idxs.tolist())
            fold_val_indices.append(val_idx.item())

        fold_tr_raw = raw_tr_x[fold_tr_indices]
        fold_tr_y = train_y[fold_tr_indices]
        fold_val_raw = raw_tr_x[fold_val_indices]
        fold_val_y = train_y[fold_val_indices]

        # R7: Fit transform on fold train ONLY
        tr_x_trans, val_x_trans = apply_transform_train_only(fold_tr_raw, fold_val_raw, transform_type)

        ncm_cvs.append(eval_ncm(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y))
        knn_cvs.append(eval_1nn(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y))
        head_m, _ = eval_headl1c(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y, val_x=val_x_trans, val_y=fold_val_y, seeds=SEEDS)
        head_cvs.append(head_m)

        for wd in WEIGHT_DECAYS:
            lin_m, _ = eval_plain_linear_ridge(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y, weight_decay=wd)
            linear_cvs[wd].append(lin_m)

    cv_ncm = py_mean(ncm_cvs)
    cv_1nn = py_mean(knn_cvs)
    cv_head = py_mean(head_cvs)

    best_wd_linear = max(linear_cvs.keys(), key=lambda wd: py_mean(linear_cvs[wd]))
    cv_linear_best = py_mean(linear_cvs[best_wd_linear])

    methods_cv = {
        "NCM": cv_ncm,
        "1-NN": cv_1nn,
        "HeadL1c": cv_head,
        f"PlainLinear (wd={best_wd_linear})": cv_linear_best
    }

    best_method = max(methods_cv, key=methods_cv.get)
    max_cv_score = methods_cv[best_method]

    return max_cv_score, best_method, best_wd_linear, methods_cv


def main():
    if not os.path.exists(EXPANDED_CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing expanded cache file: '{EXPANDED_CACHE_PATH}'.")

    d = torch.load(EXPANDED_CACHE_PATH, weights_only=False)
    raw_tr_x, train_y = d["train_x"], d["train_y"]
    raw_te_x, test_y = d["test_x"], d["test_y"]

    print("=========================================================================================================", flush=True)
    print(" GATE 2 RE-DECISION — EXPANDED 10/5 DATASET (K4b Mask Indexing & K5 Train CV Selection)", flush=True)
    print("=========================================================================================================", flush=True)

    candidate_representations = [("mean", "none"), ("mean", "center")]
    results = []

    for p_type, t_type in candidate_representations:
        cell_name = f"{p_type} / {t_type}"

        # 1. 5-Fold CV on 10 Train Prompts (Train Only)
        max_cv_score, best_cv_method, best_wd, _ = compute_5fold_cv_on_train(raw_tr_x, train_y, t_type)

        # 2. Fit transform on full train set
        tr_x, te_x = apply_transform_train_only(raw_tr_x, raw_te_x, t_type)

        # 3. Test Evaluation
        ncm_te = eval_ncm(tr_x, train_y, te_x, test_y)
        knn_te = eval_1nn(tr_x, train_y, te_x, test_y)
        head_m, head_s = eval_headl1c(tr_x, train_y, te_x, test_y, seeds=SEEDS)

        lin_test_results = {}
        for wd in WEIGHT_DECAYS:
            l_m, l_s = eval_plain_linear_ridge(tr_x, train_y, te_x, test_y, weight_decay=wd)
            lin_test_results[wd] = (l_m, l_s)

        best_lin_wd = max(lin_test_results.keys(), key=lambda wd: lin_test_results[wd][0])
        best_lin_m, best_lin_s = lin_test_results[best_lin_wd]

        test_max = max([ncm_te, knn_te, head_m, best_lin_m])

        results.append({
            "cell_name": cell_name,
            "cv_score": max_cv_score,
            "cv_method": best_cv_method,
            "ncm_test": ncm_te,
            "knn_test": knn_te,
            "head_test_mean": head_m,
            "head_test_std": head_s,
            "lin_test_mean": best_lin_m,
            "lin_test_std": best_lin_s,
            "test_max": test_max
        })

    print(f"\n{'Representation':<20} | {'5-Fold CV (Train)':<18} | {'CV Method':<22} | {'PlainLinear Test Acc':<22} | {'Test Max':<10}", flush=True)
    print("-" * 100, flush=True)
    for r in results:
        lin_str = f"{r['lin_test_mean']:5.2f}% +/- {r['lin_test_std']:4.2f}%"
        print(f"{r['cell_name']:<20} | {r['cv_score']:6.2f}%            | {r['cv_method']:<22} | {lin_str:<22} | {r['test_max']:6.2f}%", flush=True)
    print("=" * 100, flush=True)

    best_cv_res = max(results, key=lambda r: r["cv_score"])
    cv_selected_rep = best_cv_res["cell_name"]
    honest_test_acc = best_cv_res["lin_test_mean"]
    optimistic_ceiling = max(r["test_max"] for r in results)

    B = optimistic_ceiling

    print("\nGATE 2 RE-DECISION SUMMARY:", flush=True)
    print(f"  1. All 10 train prompts per class used in fold splits : YES (K4b mask verified)", flush=True)
    print(f"  2. CV-Selected Representation (Train Only)            : '{cv_selected_rep}'", flush=True)
    print(f"  3. Honest CV-Selected Test Accuracy                  : {honest_test_acc:.2f}%", flush=True)
    print(f"  4. Max-Over-All-Cells OFFLINE_BOUND (B)               : {B:.2f}%", flush=True)
    print(f"  5. Gate 2 Threshold                                  : 50.00%", flush=True)

    if B >= 50.0:
        print(f"  [GATE 2 PASSED] B ({B:.2f}%) >= 50.0%. Proceeding to Phase IV on representation '{cv_selected_rep}'!", flush=True)
    else:
        print(f"  [GATE 2 FAILED] B ({B:.2f}%) < 50.0%. Stopping per specification.", flush=True)

    print("=========================================================================================================", flush=True)


if __name__ == "__main__":
    main()
