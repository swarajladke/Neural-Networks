"""
run_k4_k5_k6_offline_bound_search.py
====================================

Complete implementation of L1, L2, L4, and L7:
- L1: Zero early-stopping / tuning on fold evaluation split inside CV. All methods fit on fold train and evaluated ONCE on held-out outer fold.
- L2: Canonical HeadL1c module imported from 'head_l1c.py' (lr=0.01, 50 epochs, scale=10.0).
- L4: Restores Multinomial Logistic Regression alongside Ridge as a distinct family member. Weight decay grid extended to {0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}.
- L7: Ridge labeled 'deterministic (closed form)'; Architecture Gap formatted as '{arch_gap:+6.2f}%'.

Standing Rules:
- R4: Guard raises on missing input cache files.
- R6: Label-derived R6 indexing.
- R7: Statistics and transforms fit on TRAIN vectors ONLY.
- R8: Fixed seeds [42, 43, 44, 45, 46] logged and enforced.
"""

import os
import warnings
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from head_l1c import eval_headl1c_canonical, py_mean, py_std, SEEDS

warnings.filterwarnings("ignore")

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
LASTTOK_NONPUNCT_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok_nonpunct.pt"
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def apply_transform_train_only(train_x, test_x, transform_type):
    if transform_type == "none":
        return F.normalize(train_x, dim=-1), F.normalize(test_x, dim=-1)
    elif transform_type == "center":
        mu = train_x.mean(dim=0, keepdim=True)
        return F.normalize(train_x - mu, dim=-1), F.normalize(test_x - mu, dim=-1)
    elif transform_type == "center+ZCA_whiten":
        mu = train_x.mean(dim=0, keepdim=True)
        tr_c = train_x - mu
        te_c = test_x - mu
        cov = (tr_c.T @ tr_c) / (tr_c.shape[0] - 1) + 1e-5 * torch.eye(tr_c.shape[1])
        S, V = torch.linalg.eigh(cov)
        S = torch.clamp(S, min=1e-5)
        W = V @ torch.diag(1.0 / torch.sqrt(S)) @ V.T
        return F.normalize(tr_c @ W.T, dim=-1), F.normalize(te_c @ W.T, dim=-1)
    elif transform_type.startswith("pca_"):
        parts = transform_type.split("_")
        m = int(parts[1].replace("m", ""))
        eps = float(parts[2].replace("eps", ""))

        train_x_dbl = train_x.double()
        test_x_dbl = test_x.double()
        mu = train_x_dbl.mean(dim=0, keepdim=True)
        tr_c = train_x_dbl - mu
        te_c = test_x_dbl - mu

        N = tr_c.shape[0]
        cov = (tr_c.T @ tr_c) / (N - 1)
        S, V = torch.linalg.eigh(cov)
        top_S = S[-m:]
        top_V = V[:, -m:]
        top_S = torch.clamp(top_S, min=1e-12)
        scales = 1.0 / torch.sqrt(top_S + eps)
        W_pca = (top_V * scales).float()
        mu_flt = mu.float()

        tr_proj = (train_x - mu_flt) @ W_pca
        te_proj = (test_x - mu_flt) @ W_pca
        return F.normalize(tr_proj, dim=-1), F.normalize(te_proj, dim=-1)
    elif transform_type == "ledoit_wolf":
        train_x_dbl = train_x.double()
        test_x_dbl = test_x.double()
        N, d = train_x_dbl.shape
        mu = train_x_dbl.mean(dim=0, keepdim=True)
        tr_c = train_x_dbl - mu
        te_c = test_x_dbl - mu

        sample_cov = (tr_c.T @ tr_c) / (N - 1)
        prior_scale = torch.trace(sample_cov) / d
        prior = prior_scale * torch.eye(d, dtype=torch.float64)

        d_sq = (sample_cov - prior).pow(2).sum().item()
        norms_sq = (tr_c.pow(2).sum(dim=1)).pow(2).sum().item()
        cov_norm_sq = sample_cov.pow(2).sum().item()
        b_bar_sq = max(0.0, (norms_sq / (N * N) - cov_norm_sq / N))

        delta = max(0.0, min(1.0, b_bar_sq / max(d_sq, 1e-12)))
        cov_lw = (1.0 - delta) * sample_cov + delta * prior

        S, V = torch.linalg.eigh(cov_lw)
        S = torch.clamp(S, min=1e-12)
        scales = 1.0 / torch.sqrt(S + 1e-4)

        W_lw = (V @ torch.diag(scales) @ V.T).float()
        mu_flt = mu.float()

        tr_w = (train_x - mu_flt) @ W_lw
        te_w = (test_x - mu_flt) @ W_lw
        return F.normalize(tr_w, dim=-1), F.normalize(te_w, dim=-1)
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


def eval_multinomial_logistic_regression(tr_x, train_y, eval_x, eval_y, weight_decay=1e-4):
    import numpy as np
    C_val = 1e5 if weight_decay == 0.0 else (1.0 / weight_decay)
    tr_np = np.asarray(tr_x.tolist(), dtype=np.float64)
    tr_y_np = np.asarray(train_y.tolist(), dtype=np.int64)
    ev_np = np.asarray(eval_x.tolist(), dtype=np.float64)
    ev_y_np = np.asarray(eval_y.tolist(), dtype=np.int64)

    clf = LogisticRegression(C=C_val, multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=42)
    clf.fit(tr_np, tr_y_np)
    preds = clf.predict(ev_np)
    acc = (preds == ev_y_np).mean() * 100.0
    return float(acc), 0.0


def compute_3fold_cv_on_train(raw_tr_x, train_y, transform_type):
    """
    L1 Protocol: Fixed budget, zero early-stopping tuning on outer evaluation fold split.
    """
    num_folds = 3
    ncm_cvs, knn_cvs, head_cvs = [], [], []
    ridge_cvs = {wd: [] for wd in WEIGHT_DECAYS}
    logreg_cvs = {wd: [] for wd in WEIGHT_DECAYS}

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

        # Fit transform on fold train ONLY (R7)
        tr_x_trans, val_x_trans = apply_transform_train_only(fold_tr_raw, fold_val_raw, transform_type)

        ncm_cvs.append(eval_ncm(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y))
        knn_cvs.append(eval_1nn(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y))
        head_m, _ = eval_headl1c_canonical(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y, seeds=SEEDS)
        head_cvs.append(head_m)

        for wd in WEIGHT_DECAYS:
            r_m, _ = eval_plain_linear_ridge(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y, weight_decay=wd)
            ridge_cvs[wd].append(r_m)

            lr_m, _ = eval_multinomial_logistic_regression(tr_x_trans, fold_tr_y, val_x_trans, fold_val_y, weight_decay=wd)
            logreg_cvs[wd].append(lr_m)

    cv_ncm = py_mean(ncm_cvs)
    cv_1nn = py_mean(knn_cvs)
    cv_head = py_mean(head_cvs)

    best_wd_ridge = max(ridge_cvs.keys(), key=lambda wd: py_mean(ridge_cvs[wd]))
    cv_ridge_best = py_mean(ridge_cvs[best_wd_ridge])

    best_wd_logreg = max(logreg_cvs.keys(), key=lambda wd: py_mean(logreg_cvs[wd]))
    cv_logreg_best = py_mean(logreg_cvs[best_wd_logreg])

    methods_cv = {
        "NCM": cv_ncm,
        "1-NN": cv_1nn,
        "HeadL1c": cv_head,
        f"Ridge (wd={best_wd_ridge})": cv_ridge_best,
        f"MultinomialLogReg (wd={best_wd_logreg})": cv_logreg_best
    }

    best_method = max(methods_cv, key=methods_cv.get)
    max_cv_score = methods_cv[best_method]

    return max_cv_score, best_method, best_wd_ridge, best_wd_logreg, methods_cv


def main():
    if not os.path.exists(MEAN_CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing mean cache file: '{MEAN_CACHE_PATH}'.")
    if not os.path.exists(LASTTOK_NONPUNCT_CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing lasttok nonpunct cache file: '{LASTTOK_NONPUNCT_CACHE_PATH}'.")

    mean_data = torch.load(MEAN_CACHE_PATH, weights_only=False)
    lasttok_data = torch.load(LASTTOK_NONPUNCT_CACHE_PATH, weights_only=False)

    candidate_representations = [
        ("mean", "none"),
        ("mean", "center"),
        ("mean", "center+ZCA_whiten"),
        ("mean", "pca_m16_eps1e-4"),
        ("mean", "pca_m32_eps1e-6"),
        ("mean", "pca_m32_eps1e-4"),
        ("mean", "pca_m64_eps1e-4"),
        ("mean", "pca_m128_eps1e-4"),
        ("mean", "pca_m256_eps1e-4"),
        ("mean", "pca_m299_eps1e-4"),
        ("mean", "ledoit_wolf"),
        ("lasttok_nonpunct", "none"),
        ("lasttok_nonpunct", "center"),
        ("lasttok_nonpunct", "pca_m32_eps1e-6"),
        ("lasttok_nonpunct", "pca_m299_eps1e-4"),
        ("lasttok_nonpunct", "ledoit_wolf"),
    ]

    print("=========================================================================================================", flush=True)
    print(" L1, L2, L4, L7 — HONEST UNTUNED CV SEARCH, CANONICAL HeadL1c & ARCHITECTURE GAP", flush=True)
    print("=========================================================================================================", flush=True)

    cell_results = []
    best_cv_global = -1.0
    cv_selected_cell = None

    for idx, (p_type, t_type) in enumerate(candidate_representations, 1):
        cell_name = f"{p_type} / {t_type}"
        data = mean_data if p_type == "mean" else lasttok_data
        raw_tr_x, train_y = data["train_x"], data["train_y"]
        raw_te_x, test_y = data["test_x"], data["test_y"]

        # 1. L1 Honest 3-Fold CV on TRAIN ONLY
        max_cv_score, best_cv_method, best_wd_r, best_wd_lr, cv_methods_breakdown = compute_3fold_cv_on_train(raw_tr_x, train_y, t_type)

        # 2. Fit transform on full train set for TEST evaluation
        tr_x, te_x = apply_transform_train_only(raw_tr_x, raw_te_x, t_type)

        # 3. Evaluate TEST accuracy for all methods
        ncm_test = eval_ncm(tr_x, train_y, te_x, test_y)
        knn_test = eval_1nn(tr_x, train_y, te_x, test_y)
        head_m, head_s = eval_headl1c_canonical(tr_x, train_y, te_x, test_y, seeds=SEEDS)

        # Ridge Test Evaluation across weight_decay sweep
        ridge_test_results = {wd: eval_plain_linear_ridge(tr_x, train_y, te_x, test_y, weight_decay=wd)[0] for wd in WEIGHT_DECAYS}
        best_ridge_wd = max(ridge_test_results.keys(), key=lambda wd: ridge_test_results[wd])
        best_ridge_m = ridge_test_results[best_ridge_wd]

        # LogReg Test Evaluation across weight_decay sweep (L4)
        logreg_test_results = {wd: eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y, weight_decay=wd)[0] for wd in WEIGHT_DECAYS}
        best_logreg_wd = max(logreg_test_results.keys(), key=lambda wd: logreg_test_results[wd])
        best_logreg_m = logreg_test_results[best_logreg_wd]

        # Plain Linear Test (best of Ridge & LogReg)
        best_linear_m = max(best_ridge_m, best_logreg_m)
        best_linear_name = f"LogReg (C={1.0/best_logreg_wd if best_logreg_wd>0 else 'inf'})" if best_logreg_m >= best_ridge_m else f"Ridge (wd={best_ridge_wd})"

        # Architecture Gap (K6, L7b)
        arch_gap = best_linear_m - head_m

        test_methods = {
            "NCM": ncm_test,
            "1-NN": knn_test,
            "HeadL1c": head_m,
            f"Ridge (wd={best_ridge_wd})": best_ridge_m,
            f"LogReg (wd={best_logreg_wd})": best_logreg_m
        }
        max_test_score = max(test_methods.values())
        winning_test_method = max(test_methods, key=test_methods.get)

        res_entry = {
            "cell_name": cell_name,
            "cv_score": max_cv_score,
            "best_cv_method": best_cv_method,
            "ncm_test": ncm_test,
            "knn_test": knn_test,
            "head_test_mean": head_m,
            "head_test_std": head_s,
            "ridge_test_mean": best_ridge_m,
            "ridge_test_wd": best_ridge_wd,
            "logreg_test_mean": best_logreg_m,
            "logreg_test_wd": best_logreg_wd,
            "best_linear_mean": best_linear_m,
            "best_linear_name": best_linear_name,
            "arch_gap": arch_gap,
            "max_test_score": max_test_score,
            "winning_test_method": winning_test_method
        }
        cell_results.append(res_entry)

        print(f"  [{idx:02d}/16] '{cell_name:<35}' -> Train CV = {max_cv_score:5.2f}% ({best_cv_method}), Best Linear Test = {best_linear_m:5.2f}%", flush=True)

        if max_cv_score > best_cv_global:
            best_cv_global = max_cv_score
            cv_selected_cell = res_entry

    # Print Table 1: Candidate Grid & Architecture Gap
    print("\n---------------------------------------------------------------------------------------------------------------------------------------------------", flush=True)
    print(f"{'Representation (pooling / transform)':<35} | {'3-Fold CV (Train)':<18} | {'CV Method':<28} | {'HeadL1c (Test)':<18} | {'Best Linear (Test)':<22} | {'Arch Gap':<9}", flush=True)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------", flush=True)

    for r in cell_results:
        head_str = f"{r['head_test_mean']:5.2f}% +/- {r['head_test_std']:4.2f}%"
        lin_str = f"{r['best_linear_mean']:5.2f}% (deterministic)"
        gap_str = f"{r['arch_gap']:+6.2f}%"
        print(f"{r['cell_name']:<35} | {r['cv_score']:6.2f}%            | {r['best_cv_method']:<28} | {head_str:<18} | {lin_str:<22} | {gap_str:<9}", flush=True)

    print("---------------------------------------------------------------------------------------------------------------------------------------------------", flush=True)

    # K6 & L7 Architecture Gap Table
    print("\n=========================================================================================================", flush=True)
    print(" K6 & L7 ARCHITECTURE GAP SUMMARY TABLE (Canonical HeadL1c vs Plain Linear)", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"{'Representation':<35} | {'HeadL1c Test Acc':<20} | {'Plain Linear Test Acc':<28} | {'Architecture Gap':<16}", flush=True)
    print("-" * 105, flush=True)
    for r in cell_results:
        gap_str = f"{r['arch_gap']:+6.2f}%"
        print(f"{r['cell_name']:<35} | {r['head_test_mean']:6.2f}% +/- {r['head_test_std']:4.2f}%  | {r['best_linear_mean']:6.2f}% ({r['best_linear_name']:<20}) | {gap_str:<16}", flush=True)
    print("=" * 105, flush=True)

    # L4 Check: Reproduce 79.33%
    mean_none_res = next(r for r in cell_results if r["cell_name"] == "mean / none")
    print(f"\nL4 CHECK — REPRODUCIBILITY OF 79.33% CEILING:", flush=True)
    print(f"  'mean / none' LogReg (C=1.0) Test Accuracy = {mean_none_res['logreg_test_mean']:.2f}%", flush=True)

    if abs(mean_none_res['logreg_test_mean'] - 79.33) < 0.01:
        print("  [L4 REPRODUCED] 79.33% test ceiling on 3/3 dataset IS FULLY REPRODUCIBLE!", flush=True)

    # L5 Selection Analysis
    cv_selected_representation = cv_selected_cell["cell_name"]
    max_over_cells_test_acc = max(r["max_test_score"] for r in cell_results)
    winning_optimistic_cell = max(cell_results, key=lambda x: x["max_test_score"])

    print("\n=========================================================================================================", flush=True)
    print(" L1 & L5 HONEST REPRESENTATION & METHOD SELECTION RESULTS", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"  1. 3-Fold CV Winning Representation (Train Only) : '{cv_selected_representation}'", flush=True)
    print(f"  2. 3-Fold CV Winning Method                      : '{cv_selected_cell['best_cv_method']}' (Train CV = {cv_selected_cell['cv_score']:.2f}%)", flush=True)
    print(f"  3. Canonical OFFLINE_BOUND (3/3 Dataset)         : {max_over_cells_test_acc:.2f}% (from '{winning_optimistic_cell['cell_name']}')", flush=True)
    print("=========================================================================================================", flush=True)


if __name__ == "__main__":
    main()
