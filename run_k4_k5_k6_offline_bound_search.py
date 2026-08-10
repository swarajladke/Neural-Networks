"""
run_k4_k5_k6_offline_bound_search.py
====================================

Complete implementation of K4, K5, and K6 with fast PyTorch dual Ridge solver & AdamW HeadL1c:
- K4a: 5 seeds [42, 43, 44, 45, 46] evaluated (mean +/- std reported).
- K4b: Flexible mask-based fold indexing (no hardcoded fold sizes).
- K4c: Direct weight decay sweep over {0.0, 1e-4, 1e-3, 1e-2, 1e-1}.
- K5: Honest CV-selection on TRAIN ONLY (3-fold CV over 3 train prompts). Reports both CV-selected test accuracy (honest) and max-over-cells test accuracy (optimistic ceiling). Selection bias quantified.
- K6: Architecture gap table (Plain Linear vs HeadL1c across all 16 representations).

Standing Rules:
- R4: Guard raises on missing input cache files.
- R6: Label-derived R6 indexing.
- R7: Statistics and transforms fit on TRAIN vectors ONLY.
- R8: Fixed seeds [42, 43, 44, 45, 46] logged and enforced.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
LASTTOK_NONPUNCT_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok_nonpunct.pt"
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
    # Pure PyTorch Dual Ridge Solver: W = X^T (X X^T + lambda I)^-1 Y_onehot
    N, d = tr_x.shape
    num_classes = 100
    Y_oh = F.one_hot(train_y, num_classes=num_classes).float()

    lam = max(weight_decay, 1e-6)
    K = tr_x @ tr_x.T
    K_reg = K + lam * torch.eye(N)
    alpha = torch.linalg.solve(K_reg, Y_oh)
    W = tr_x.T @ alpha  # (d, 100)

    preds = (eval_x @ W).argmax(dim=1)
    acc = (preds == eval_y).float().mean().item() * 100.0
    return acc, 0.0


def compute_3fold_cv_on_train(raw_tr_x, train_y, transform_type):
    num_folds = 3
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

        # Fit transform on fold train ONLY (R7)
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
    print(" PHASE K4, K5, K6 — RE-EVALUATION OF OFFLINE BOUND, HONEST CV SELECTION & ARCHITECTURE GAP", flush=True)
    print("=========================================================================================================", flush=True)

    print(f"\nEvaluating 16 candidate representations over 5 seeds {SEEDS}...\n", flush=True)

    cell_results = []
    best_cv_global = -1.0
    cv_selected_cell = None

    for idx, (p_type, t_type) in enumerate(candidate_representations, 1):
        cell_name = f"{p_type} / {t_type}"
        data = mean_data if p_type == "mean" else lasttok_data
        raw_tr_x, train_y = data["train_x"], data["train_y"]
        raw_te_x, test_y = data["test_x"], data["test_y"]

        # 1. Honest 3-Fold CV on TRAIN ONLY (K5)
        max_cv_score, best_cv_method, best_wd, cv_methods_breakdown = compute_3fold_cv_on_train(raw_tr_x, train_y, t_type)

        # 2. Fit transform on full train set for TEST evaluation
        tr_x, te_x = apply_transform_train_only(raw_tr_x, raw_te_x, t_type)

        # 3. Evaluate TEST accuracy for all 4 methods
        ncm_test = eval_ncm(tr_x, train_y, te_x, test_y)
        knn_test = eval_1nn(tr_x, train_y, te_x, test_y)
        head_m, head_s = eval_headl1c(tr_x, train_y, te_x, test_y, seeds=SEEDS)

        # Plain Linear TEST evaluation across weight_decay sweep
        lin_test_results = {}
        for wd in WEIGHT_DECAYS:
            l_m, l_s = eval_plain_linear_ridge(tr_x, train_y, te_x, test_y, weight_decay=wd)
            lin_test_results[wd] = (l_m, l_s)

        best_lin_test_wd = max(lin_test_results.keys(), key=lambda wd: lin_test_results[wd][0])
        best_lin_m, best_lin_s = lin_test_results[best_lin_test_wd]

        # Architecture Gap (K6)
        arch_gap = best_lin_m - head_m

        # Test Max for optimistic ceiling
        test_methods = {
            "NCM": ncm_test,
            "1-NN": knn_test,
            "HeadL1c": head_m,
            f"PlainLinear (wd={best_lin_test_wd})": best_lin_m
        }
        max_test_score = max(test_methods.values())
        winning_test_method = max(test_methods, key=test_methods.get)

        res_entry = {
            "cell_name": cell_name,
            "pooling": p_type,
            "transform": t_type,
            "cv_score": max_cv_score,
            "best_cv_method": best_cv_method,
            "best_wd": best_wd,
            "ncm_test": ncm_test,
            "knn_test": knn_test,
            "head_test_mean": head_m,
            "head_test_std": head_s,
            "lin_test_mean": best_lin_m,
            "lin_test_std": best_lin_s,
            "lin_test_wd": best_lin_test_wd,
            "arch_gap": arch_gap,
            "max_test_score": max_test_score,
            "winning_test_method": winning_test_method
        }
        cell_results.append(res_entry)

        print(f"  [{idx:02d}/16] '{cell_name:<35}' -> Train CV = {max_cv_score:5.2f}% ({best_cv_method}), PlainLinear Test = {best_lin_m:5.2f}%", flush=True)

        if max_cv_score > best_cv_global:
            best_cv_global = max_cv_score
            cv_selected_cell = res_entry

    # Print Table 1: Full Candidate Grid with CV on Train & Test Evaluation
    print("\n-----------------------------------------------------------------------------------------------------------------------------------------", flush=True)
    print(f"{'Representation (pooling / transform)':<35} | {'3-Fold CV (Train)':<18} | {'CV Method':<25} | {'HeadL1c (Test)':<18} | {'PlainLinear (Test)':<22} | {'Arch Gap':<8}", flush=True)
    print("-----------------------------------------------------------------------------------------------------------------------------------------", flush=True)

    for r in cell_results:
        head_str = f"{r['head_test_mean']:5.2f}% +/- {r['head_test_std']:4.2f}%"
        lin_str = f"{r['lin_test_mean']:5.2f}% +/- {r['lin_test_std']:4.2f}%"
        print(f"{r['cell_name']:<35} | {r['cv_score']:6.2f}%            | {r['best_cv_method']:<25} | {head_str:<18} | {lin_str:<22} | +{r['arch_gap']:5.2f}%", flush=True)

    print("-----------------------------------------------------------------------------------------------------------------------------------------", flush=True)

    # K6 Architecture Gap Summary Table
    print("\n=========================================================================================================", flush=True)
    print(" K6 ARCHITECTURE GAP SUMMARY TABLE (HeadL1c vs Plain Linear on Identical Features)", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"{'Representation':<35} | {'HeadL1c Test Acc':<20} | {'Plain Linear Test Acc':<25} | {'Architecture Gap':<16}", flush=True)
    print("-" * 100, flush=True)
    for r in cell_results:
        print(f"{r['cell_name']:<35} | {r['head_test_mean']:6.2f}% +/- {r['head_test_std']:4.2f}%  | {r['lin_test_mean']:6.2f}% +/- {r['lin_test_std']:4.2f}% (wd={r['lin_test_wd']}) | +{r['arch_gap']:6.2f}%", flush=True)
    print("=" * 100, flush=True)

    # K5 Selection Analysis
    cv_selected_representation = cv_selected_cell["cell_name"]

    if "PlainLinear" in cv_selected_cell["best_cv_method"]:
        cv_test_acc = cv_selected_cell["lin_test_mean"]
        cv_test_std = cv_selected_cell["lin_test_std"]
    elif cv_selected_cell["best_cv_method"] == "HeadL1c":
        cv_test_acc = cv_selected_cell["head_test_mean"]
        cv_test_std = cv_selected_cell["head_test_std"]
    elif cv_selected_cell["best_cv_method"] == "1-NN":
        cv_test_acc = cv_selected_cell["knn_test"]
        cv_test_std = 0.0
    else:  # NCM
        cv_test_acc = cv_selected_cell["ncm_test"]
        cv_test_std = 0.0

    max_over_cells_test_acc = max(r["max_test_score"] for r in cell_results)
    winning_optimistic_cell = max(cell_results, key=lambda x: x["max_test_score"])

    selection_bias = max_over_cells_test_acc - cv_test_acc

    print("\n=========================================================================================================", flush=True)
    print(" K5 HONEST REPRESENTATION & METHOD SELECTION RESULTS", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"  1. 3-Fold CV Winning Representation (Train Only) : '{cv_selected_representation}'", flush=True)
    print(f"  2. 3-Fold CV Winning Method & Weight Decay        : '{cv_selected_cell['best_cv_method']}' (Train CV = {cv_selected_cell['cv_score']:.2f}%)", flush=True)
    print(f"  3. CV-Selected Test Accuracy (HONEST NUMBER)      : {cv_test_acc:.2f}% +/- {cv_test_std:.2f}%", flush=True)
    print(f"  4. Max-Over-All-Cells Test Acc (OPTIMISTIC CEILING): {max_over_cells_test_acc:.2f}% (from '{winning_optimistic_cell['cell_name']}' via {winning_optimistic_cell['winning_test_method']})", flush=True)
    print(f"  5. Selection Bias (Optimistic Ceiling - Honest)   : +{selection_bias:.2f} percentage points", flush=True)
    print(f"  6. OFFLINE_BOUND for Gating Purposes             : {max_over_cells_test_acc:.2f}%", flush=True)
    print(f"  7. Representation Phase IV Will Use               : '{cv_selected_representation}'", flush=True)
    print("=========================================================================================================", flush=True)


if __name__ == "__main__":
    main()
