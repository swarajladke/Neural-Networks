"""
evaluate_m_phase_comprehensive.py
==================================

Executes all diagnostic and validation evaluations for Directives M1, M4, M5, M6:
- M1: Honest test evaluation (single validation-selected method/wd evaluation on test).
- M4: Correlation baseline (r_before vs r_after, Pearson & Spearman, all n=11 and n=10 excl ZCA).
- M5: 3-way template split disjointness checks and centroid cosine comparison (train-val vs train-test).
- M6: Eigenvalue spectrum audit and PCA dimension sweep m in {16,32,48,56,64,72,80,96,112,128,256,299}.
"""

import json
import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

from head_l1c import eval_headl1c_canonical, SEEDS
from run_k4_k5_k6_offline_bound_search import (
    apply_transform_train_only,
    eval_ncm,
    eval_1nn,
    eval_plain_linear_ridge,
    eval_multinomial_logistic_regression,
    WEIGHT_DECAYS
)

WEIGHT_DECAYS_CV = [1e-4, 1e-2, 1.0]

V3_CACHE_PATH = "smollm2_embeddings_v3_100facts_7_3_5.pt"
V2_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
DATASET_V3_PATH = "agnis_scaling_dataset_v3_template_split.json"

CANDIDATE_REPRESENTATIONS = [
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
]


def run_m1_honest_disjoint_split(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y, transform_type):
    # Fit transform on train ONLY
    tr_x, va_x = apply_transform_train_only(tr_x_raw, va_x_raw, transform_type)
    _, te_x = apply_transform_train_only(tr_x_raw, te_x_raw, transform_type)

    # 1. Validation Split Pass: Evaluate all candidate methods & hyperparameters
    ncm_val = eval_ncm(tr_x, train_y, va_x, val_y)
    knn_val = eval_1nn(tr_x, train_y, va_x, val_y)
    head_val_m = eval_headl1c_canonical(tr_x, train_y, va_x, val_y, seeds=[42])[0]

    val_candidates = {}
    val_candidates[("NCM", None)] = ncm_val
    val_candidates[("1-NN", None)] = knn_val
    val_candidates[("HeadL1c", None)] = head_val_m

    for wd in WEIGHT_DECAYS:
        ridge_acc = eval_plain_linear_ridge(tr_x, train_y, va_x, val_y, weight_decay=wd)[0]
        val_candidates[("Ridge", wd)] = ridge_acc

        logreg_acc = eval_multinomial_logistic_regression(tr_x, train_y, va_x, val_y, weight_decay=wd)[0]
        val_candidates[("MultinomialLogReg", wd)] = logreg_acc

    # Select single winning method & weight decay from validation split
    best_val_config = max(val_candidates.keys(), key=lambda k: val_candidates[k])
    best_val_score = val_candidates[best_val_config]
    val_method_name, val_wd = best_val_config

    # 2. Honest Test Evaluation: Evaluate EXACTLY THAT ONE CONFIGURATION on test, ONCE
    if val_method_name == "NCM":
        honest_test_acc = eval_ncm(tr_x, train_y, te_x, test_y)
    elif val_method_name == "1-NN":
        honest_test_acc = eval_1nn(tr_x, train_y, te_x, test_y)
    elif val_method_name == "HeadL1c":
        honest_test_acc = eval_headl1c_canonical(tr_x, train_y, te_x, test_y, seeds=[42])[0]
    elif val_method_name == "Ridge":
        honest_test_acc = eval_plain_linear_ridge(tr_x, train_y, te_x, test_y, weight_decay=val_wd)[0]
    elif val_method_name == "MultinomialLogReg":
        honest_test_acc = eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y, weight_decay=val_wd)[0]
    else:
        raise ValueError(f"Unknown method: {val_method_name}")

    # 3. Compute Optimistic Ceiling on Test (Max over N=11 test evaluations)
    ncm_test = eval_ncm(tr_x, train_y, te_x, test_y)
    knn_test = eval_1nn(tr_x, train_y, te_x, test_y)
    head_test_m = eval_headl1c_canonical(tr_x, train_y, te_x, test_y, seeds=[42])[0]

    test_evals = [ncm_test, knn_test, head_test_m]
    test_wd_candidates = {}

    for wd in WEIGHT_DECAYS:
        r_acc = eval_plain_linear_ridge(tr_x, train_y, te_x, test_y, weight_decay=wd)[0]
        l_acc = eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y, weight_decay=wd)[0]
        test_evals.extend([r_acc, l_acc])
        test_wd_candidates[("Ridge", wd)] = r_acc
        test_wd_candidates[("MultinomialLogReg", wd)] = l_acc

    optimistic_ceiling = max(test_evals)
    best_test_wd_config = max(test_wd_candidates.keys(), key=lambda k: test_wd_candidates[k])
    test_selected_wd = best_test_wd_config[1]

    N_test_evals = len(test_evals)

    return {
        "val_score": best_val_score,
        "val_method": val_method_name,
        "val_wd": val_wd,
        "honest_test_acc": honest_test_acc,
        "optimistic_ceiling": optimistic_ceiling,
        "N_test_evals": N_test_evals,
        "test_selected_wd": test_selected_wd,
        "wd_differ": (val_wd != test_selected_wd) if (val_method_name in ["Ridge", "MultinomialLogReg"]) else False
    }


def compute_3fold_cv_train_only(tr_x_raw, train_y, transform_type):
    # 3-fold CV inside train set (700 prompts, 7 prompts per fact)
    # Folds split by prompt index per fact (7 prompts -> fold0: [0,1], fold1: [2,3,4], fold2: [5,6])
    n_samples = tr_x_raw.shape[0]
    n_facts = 100
    prompts_per_fact = n_samples // n_facts

    folds = [
        [0, 1],
        [2, 3, 4],
        [5, 6]
    ]

    fold_val_accs = []

    for fold_idx, val_p_indices in enumerate(folds):
        train_p_indices = [p for p in range(prompts_per_fact) if p not in val_p_indices]

        tr_indices = []
        va_indices = []

        for f in range(n_facts):
            base = f * prompts_per_fact
            for p in train_p_indices:
                tr_indices.append(base + p)
            for p in val_p_indices:
                va_indices.append(base + p)

        tr_x_fold_raw = tr_x_raw[tr_indices]
        tr_y_fold = train_y[tr_indices]
        va_x_fold_raw = tr_x_raw[va_indices]
        va_y_fold = train_y[va_indices]

        tr_x_fold, va_x_fold = apply_transform_train_only(tr_x_fold_raw, va_x_fold_raw, transform_type)

        # Evaluate candidate methods on fold val split
        ncm_acc = eval_ncm(tr_x_fold, tr_y_fold, va_x_fold, va_y_fold)
        knn_acc = eval_1nn(tr_x_fold, tr_y_fold, va_x_fold, va_y_fold)
        head_acc = eval_headl1c_canonical(tr_x_fold, tr_y_fold, va_x_fold, va_y_fold, seeds=[42])[0]

        methods_accs = [ncm_acc, knn_acc, head_acc]
        for wd in WEIGHT_DECAYS_CV:
            methods_accs.append(eval_plain_linear_ridge(tr_x_fold, tr_y_fold, va_x_fold, va_y_fold, weight_decay=wd)[0])
            methods_accs.append(eval_multinomial_logistic_regression(tr_x_fold, tr_y_fold, va_x_fold, va_y_fold, weight_decay=wd)[0])

        best_fold_acc = max(methods_accs)
        fold_val_accs.append(best_fold_acc)

    return float(np.mean(fold_val_accs))


def run_m5_disjointness_checks():
    with open(DATASET_V3_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    facts = dataset["facts"]

    train_prompts = []
    val_prompts = []
    test_prompts = []

    for fact in facts:
        train_prompts.extend(fact["train_prompts"])
        val_prompts.extend(fact["val_prompts"])
        test_prompts.extend(fact["test_prompts"])

    def extract_5grams(texts):
        grams = set()
        for t in texts:
            tokens = t.lower().split()
            for i in range(len(tokens) - 4):
                grams.add(tuple(tokens[i : i + 5]))
        return grams

    tr_5g = extract_5grams(train_prompts)
    va_5g = extract_5grams(val_prompts)
    te_5g = extract_5grams(test_prompts)

    tr_va_overlap = len(tr_5g.intersection(va_5g))
    tr_te_overlap = len(tr_5g.intersection(te_5g))
    va_te_overlap = len(va_5g.intersection(te_5g))

    print("\n--- M5 Pairwise 5-Gram Overlap Audit ---", flush=True)
    print(f"  Train-Val 5-gram overlap: {tr_va_overlap}", flush=True)
    print(f"  Train-Test 5-gram overlap: {tr_te_overlap}", flush=True)
    print(f"  Val-Test 5-gram overlap:   {va_te_overlap}", flush=True)

    # Check answer token leakage
    leakage_count = 0
    for fact in facts:
        target = fact["answer"].lower()
        all_p = fact["train_prompts"] + fact["val_prompts"] + fact["test_prompts"]
        for p in all_p:
            if target in p.lower():
                leakage_count += 1
    print(f"  Answer token leakage in prompts: {leakage_count}", flush=True)

    # Check H1 Latin square constraints
    entity_counts_val = []
    rel_counts_val = []
    for fact in facts:
        entity_counts_val.append(fact["entity_index"])
        rel_counts_val.append(fact["relation_index"])

    print(f"  Total unique entities: {len(set(entity_counts_val))}, unique relations: {len(set(rel_counts_val))}", flush=True)

    # Compute centroid pairwise cosines using raw PyTorch tensors
    d = torch.load(V3_CACHE_PATH, weights_only=False)
    tr_x = d["train_x"] # 700 x 960
    va_x = d["val_x"]   # 300 x 960
    te_x = d["test_x"]  # 500 x 960

    # Fact centroids (100 facts)
    tr_centroids = tr_x.reshape(100, 7, 960).mean(dim=1) # 100 x 960
    va_centroids = va_x.reshape(100, 3, 960).mean(dim=1) # 100 x 960
    te_centroids = te_x.reshape(100, 5, 960).mean(dim=1) # 100 x 960

    # Normalize centroids
    tr_centroids = F.normalize(tr_centroids, dim=1)
    va_centroids = F.normalize(va_centroids, dim=1)
    te_centroids = F.normalize(te_centroids, dim=1)

    train_val_cos = float((tr_centroids * va_centroids).sum(dim=1).mean().item())
    train_test_cos = float((tr_centroids * te_centroids).sum(dim=1).mean().item())

    print(f"  Mean Train-Val Centroid Cosine:  {train_val_cos:.6f}", flush=True)
    print(f"  Mean Train-Test Centroid Cosine: {train_test_cos:.6f}", flush=True)
    print(f"  Difference (Train-Val minus Train-Test): {train_val_cos - train_test_cos:+.6f}", flush=True)

    return train_val_cos, train_test_cos


def run_m6_pca_sweep_and_eigenvalues(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y):
    # Eigenvalue spectrum audit of train covariance matrix
    # tr_x_raw is 700 x 960
    X_tr = tr_x_raw.double()
    mean_tr = X_tr.mean(dim=0, keepdim=True)
    X_c = X_tr - mean_tr
    cov = torch.matmul(X_c.T, X_c) / (X_c.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(dims=[0])

    top128_vals = eigvals[:128]
    below_1e3_count = int((top128_vals < 1e-3).sum().item())

    print("\n--- M6 Eigenvalue Spectrum Audit ---", flush=True)
    print(f"  Top 10 eigenvalues: {top128_vals[:10].tolist()}", flush=True)
    print(f"  Eigenvalue #16: {top128_vals[15].item():.6e}, #32: {top128_vals[31].item():.6e}, #64: {top128_vals[63].item():.6e}, #128: {top128_vals[127].item():.6e}", flush=True)
    print(f"  Number of top 128 eigenvalues below 1e-3: {below_1e3_count} / 128", flush=True)

    m_values = [16, 32, 48, 56, 64, 72, 80, 96, 112, 128, 256, 299]
    sweep_results = []

    print(f"\n{'m':<5} | {'Disjoint Val Acc':<18} | {'Val Method':<28} | {'Honest Test Acc':<18} | {'Optimistic Ceiling':<20}", flush=True)
    print("-" * 95, flush=True)

    for m in m_values:
        t_type = f"pca_m{m}_eps1e-4"
        res = run_m1_honest_disjoint_split(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y, t_type)
        res["m"] = m
        sweep_results.append(res)
        val_m_str = f"{res['val_method']} (wd={res['val_wd']})" if res['val_wd'] is not None else res['val_method']
        print(f"{m:<5} | {res['val_score']:6.2f}%            | {val_m_str:<28} | {res['honest_test_acc']:6.2f}%           | {res['optimistic_ceiling']:6.2f}%", flush=True)
    print("=" * 95, flush=True)

    # Check variation in m in {56..80}
    val_56_80 = [r["val_score"] for r in sweep_results if r["m"] in [56, 64, 72, 80]]
    val_range_56_80 = max(val_56_80) - min(val_56_80)
    print(f"\n  Validation Range across m in {{56, 64, 72, 80}}: {val_range_56_80:.2f} percentage points", flush=True)

    return sweep_results, below_1e3_count, val_range_56_80


def main():
    d_v3 = torch.load(V3_CACHE_PATH, weights_only=False)
    tr_x_raw, train_y = d_v3["train_x"], d_v3["train_y"]
    va_x_raw, val_y = d_v3["val_x"], d_v3["val_y"]
    te_x_raw, test_y = d_v3["test_x"], d_v3["test_y"]

    print("=========================================================================================================", flush=True)
    print(" DIRECTIVE M1 — HONEST DISJOINT SPLIT TEST EVALUATION", flush=True)
    print("=========================================================================================================", flush=True)

    m1_results = []
    for idx, (p_type, t_type) in enumerate(CANDIDATE_REPRESENTATIONS, 1):
        cell_name = f"{p_type} / {t_type}"
        res = run_m1_honest_disjoint_split(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y, t_type)
        res["cell_name"] = cell_name
        m1_results.append(res)
        val_m_str = f"{res['val_method']} (wd={res['val_wd']})" if res['val_wd'] is not None else res['val_method']
        print(f"  [{idx:02d}/11] '{cell_name:<30}' -> Val = {res['val_score']:5.2f}% ({val_m_str}), Honest Test = {res['honest_test_acc']:5.2f}%", flush=True)

    print(f"\n{'Representation':<30} | {'Disjoint Val':<12} | {'Val Method & WD':<32} | {'HONEST TEST ACC':<18} | {'Optimistic Ceiling (N=11)':<26}", flush=True)
    print("-" * 125, flush=True)
    differ_count = 0
    for r in m1_results:
        val_m_str = f"{r['val_method']} (wd={r['val_wd']})" if r['val_wd'] is not None else r['val_method']
        if r['wd_differ']:
            differ_count += 1
        diff_flag = " [WD DIFFERS]" if r['wd_differ'] else ""
        print(f"{r['cell_name']:<30} | {r['val_score']:5.2f}%     | {val_m_str:<32} | {r['honest_test_acc']:5.2f}%            | {r['optimistic_ceiling']:5.2f}%{diff_flag}", flush=True)
    print("=" * 125, flush=True)

    winning_res = max(m1_results, key=lambda r: r["val_score"])
    selected_rep = winning_res["cell_name"]
    honest_test_acc_selected = winning_res["honest_test_acc"]
    optimistic_ceiling_selected = winning_res["optimistic_ceiling"]

    print(f"\n  M1 Selected Representation                 : '{selected_rep}'", flush=True)
    print(f"  M1 Disjoint Validation Score              : {winning_res['val_score']:.2f}%", flush=True)
    print(f"  M1 HONEST_TEST_ACC (Selected Cell)         : {honest_test_acc_selected:.2f}%", flush=True)
    print(f"  M1 Optimistic Ceiling (N=11 Test Evals)   : {optimistic_ceiling_selected:.2f}%", flush=True)
    print(f"  P16 Check: WD Differed on {differ_count} of 11 cells", flush=True)

    print("\n=========================================================================================================", flush=True)
    print(" DIRECTIVE M4 — WITHIN-TRAIN CV VS DISJOINT VAL CORRELATION BASELINE", flush=True)
    print("=========================================================================================================", flush=True)

    # Compute within-train-fold CV for all 11 cells
    within_train_cv_scores = []
    for p_type, t_type in CANDIDATE_REPRESENTATIONS:
        cv_score = compute_3fold_cv_train_only(tr_x_raw, train_y, t_type)
        within_train_cv_scores.append(cv_score)

    val_scores = [r["val_score"] for r in m1_results]
    honest_test_scores = [r["honest_test_acc"] for r in m1_results]

    # Correlations for all n=11 cells
    r_before_all_p, _ = pearsonr(within_train_cv_scores, honest_test_scores)
    r_before_all_s, _ = spearmanr(within_train_cv_scores, honest_test_scores)
    r_after_all_p, _ = pearsonr(val_scores, honest_test_scores)
    r_after_all_s, _ = spearmanr(val_scores, honest_test_scores)

    # Correlations excluding center+ZCA_whiten outlier (index 2)
    excl_idx = [i for i in range(len(CANDIDATE_REPRESENTATIONS)) if CANDIDATE_REPRESENTATIONS[i][1] != "center+ZCA_whiten"]
    cv_excl = [within_train_cv_scores[i] for i in excl_idx]
    val_excl = [val_scores[i] for i in excl_idx]
    test_excl = [honest_test_scores[i] for i in excl_idx]

    r_before_excl_p, _ = pearsonr(cv_excl, test_excl)
    r_before_excl_s, _ = spearmanr(cv_excl, test_excl)
    r_after_excl_p, _ = pearsonr(val_excl, test_excl)
    r_after_excl_s, _ = spearmanr(val_excl, test_excl)

    print(f"\n{'Representation':<30} | {'Within-Train 3-Fold CV':<22} | {'Disjoint Val Acc':<18} | {'HONEST TEST ACC':<16}", flush=True)
    print("-" * 95, flush=True)
    for idx, r in enumerate(m1_results):
        print(f"{r['cell_name']:<30} | {within_train_cv_scores[idx]:6.2f}%                 | {r['val_score']:6.2f}%            | {r['honest_test_acc']:6.2f}%", flush=True)
    print("=" * 95, flush=True)

    print(f"\n  M4 Correlation Summary (n=11 All Cells):", flush=True)
    print(f"    r_before (Within-Train CV vs Honest Test)   : Pearson r = {r_before_all_p:+.4f}, Spearman rho = {r_before_all_s:+.4f}", flush=True)
    print(f"    r_after  (Disjoint Val vs Honest Test)      : Pearson r = {r_after_all_p:+.4f}, Spearman rho = {r_after_all_s:+.4f}", flush=True)
    print(f"\n  M4 Correlation Summary (n=10 Excluding ZCA Outlier):", flush=True)
    print(f"    r_before (Excl ZCA)                        : Pearson r = {r_before_excl_p:+.4f}, Spearman rho = {r_before_excl_s:+.4f}", flush=True)
    print(f"    r_after  (Excl ZCA)                        : Pearson r = {r_after_excl_p:+.4f}, Spearman rho = {r_after_excl_s:+.4f}", flush=True)

    print("\n=========================================================================================================", flush=True)
    print(" DIRECTIVE M5 — 3-WAY TEMPLATE SPLIT DISJOINTNESS & CENTROID COSINE AUDIT", flush=True)
    print("=========================================================================================================", flush=True)
    train_val_cos, train_test_cos = run_m5_disjointness_checks()

    print("\n=========================================================================================================", flush=True)
    print(" DIRECTIVE M6 — PCA DIMENSION SWEEP & EIGENVALUE SPECTRUM AUDIT", flush=True)
    print("=========================================================================================================", flush=True)
    sweep_res, below_1e3_count, val_range_56_80 = run_m6_pca_sweep_and_eigenvalues(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y)

    print("\n=========================================================================================================", flush=True)
    print(" SUMMARY OF PRE-REGISTERED PREDICTIONS (P16..P19):", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"  P16 (WD Differs on >= 5 cells)               : Differed on {differ_count}/11 cells -> Verdict: {'RIGHT' if differ_count >= 5 else 'WRONG'}", flush=True)
    print(f"  P17 (HONEST_TEST_ACC <= 85.60% - 2.0pp = 83.60%): Honest = {honest_test_acc_selected:.2f}% -> Verdict: {'RIGHT' if honest_test_acc_selected <= 83.60 else 'WRONG'}", flush=True)
    print(f"  P18 (r_before > +0.80)                        : r_before = {r_before_all_p:+.4f} -> Verdict: {'RIGHT' if r_before_all_p > 0.80 else 'WRONG'}", flush=True)
    print(f"  P19 (Train-Val Cos > Train-Test Cos)          : Train-Val = {train_val_cos:.6f}, Train-Test = {train_test_cos:.6f} -> Verdict: {'RIGHT' if train_val_cos > train_test_cos else 'WRONG'}", flush=True)
    print("=========================================================================================================", flush=True)


if __name__ == "__main__":
    main()
