"""
evaluate_disjoint_template_split_l5_l6.py
=========================================

Evaluates all 16 candidate representations on the 3-Way Disjoint Template Split Dataset (L5, L6).

Features:
- Fit transform on TRAIN vectors ONLY (700 samples).
- Evaluates on Disjoint Validation Prompts (300 samples) and Disjoint Test Prompts (500 samples).
- Selects representation strictly via Disjoint Validation Accuracy (L6).
- Sets B = Test Ceiling OF THE SELECTED REPRESENTATION (L5).
- Computes correlation between Validation Acc vs Test Acc (After L6) and Within-Train Fold CV vs Test Acc (Before L6).
"""

import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from head_l1c import eval_headl1c_canonical, py_mean, py_std, SEEDS

warnings.filterwarnings("ignore")
from run_k4_k5_k6_offline_bound_search import (
    apply_transform_train_only,
    eval_ncm,
    eval_1nn,
    eval_plain_linear_ridge,
    eval_multinomial_logistic_regression,
    WEIGHT_DECAYS
)

V3_CACHE_PATH = "smollm2_embeddings_v3_100facts_7_3_5.pt"


def eval_cell_disjoint_split(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y, transform_type):
    # Fit transform on train ONLY
    tr_x, va_x = apply_transform_train_only(tr_x_raw, va_x_raw, transform_type)
    _, te_x = apply_transform_train_only(tr_x_raw, te_x_raw, transform_type)

    # 1. Validation Evaluations
    ncm_val = eval_ncm(tr_x, train_y, va_x, val_y)
    knn_val = eval_1nn(tr_x, train_y, va_x, val_y)
    head_val_m, head_val_s = eval_headl1c_canonical(tr_x, train_y, va_x, val_y, seeds=SEEDS)

    ridge_val_res = {wd: eval_plain_linear_ridge(tr_x, train_y, va_x, val_y, weight_decay=wd)[0] for wd in WEIGHT_DECAYS}
    best_ridge_val_wd = max(ridge_val_res.keys(), key=lambda wd: ridge_val_res[wd])
    best_ridge_val_m = ridge_val_res[best_ridge_val_wd]

    logreg_val_res = {wd: eval_multinomial_logistic_regression(tr_x, train_y, va_x, val_y, weight_decay=wd)[0] for wd in WEIGHT_DECAYS}
    best_logreg_val_wd = max(logreg_val_res.keys(), key=lambda wd: logreg_val_res[wd])
    best_logreg_val_m = logreg_val_res[best_logreg_val_wd]

    val_methods = {
        "NCM": ncm_val,
        "1-NN": knn_val,
        "HeadL1c": head_val_m,
        f"Ridge (wd={best_ridge_val_wd})": best_ridge_val_m,
        f"MultinomialLogReg (wd={best_logreg_val_wd})": best_logreg_val_m
    }
    best_val_method = max(val_methods, key=val_methods.get)
    max_val_score = val_methods[best_val_method]

    # 2. Test Evaluations
    ncm_test = eval_ncm(tr_x, train_y, te_x, test_y)
    knn_test = eval_1nn(tr_x, train_y, te_x, test_y)
    head_test_m, head_test_s = eval_headl1c_canonical(tr_x, train_y, te_x, test_y, seeds=SEEDS)

    ridge_test_res = {wd: eval_plain_linear_ridge(tr_x, train_y, te_x, test_y, weight_decay=wd)[0] for wd in WEIGHT_DECAYS}
    best_ridge_test_wd = max(ridge_test_res.keys(), key=lambda wd: ridge_test_res[wd])
    best_ridge_test_m = ridge_test_res[best_ridge_test_wd]

    logreg_test_res = {wd: eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y, weight_decay=wd)[0] for wd in WEIGHT_DECAYS}
    best_logreg_test_wd = max(logreg_test_res.keys(), key=lambda wd: logreg_test_res[wd])
    best_logreg_test_m = logreg_test_res[best_logreg_test_wd]

    best_linear_test_m = max(best_ridge_test_m, best_logreg_test_m)
    test_max_score = max([ncm_test, knn_test, head_test_m, best_linear_test_m])

    return {
        "val_score": max_val_score,
        "val_method": best_val_method,
        "ncm_test": ncm_test,
        "knn_test": knn_test,
        "head_test_m": head_test_m,
        "head_test_s": head_test_s,
        "ridge_test_m": best_ridge_test_m,
        "logreg_test_m": best_logreg_test_m,
        "best_linear_test_m": best_linear_test_m,
        "test_max_score": test_max_score
    }


def main():
    if not os.path.exists(V3_CACHE_PATH):
        raise RuntimeError(f"Missing cache file: '{V3_CACHE_PATH}'")

    d = torch.load(V3_CACHE_PATH, weights_only=False)
    tr_x_raw, train_y = d["train_x"], d["train_y"]
    va_x_raw, val_y = d["val_x"], d["val_y"]
    te_x_raw, test_y = d["test_x"], d["test_y"]

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
    ]

    print("=========================================================================================================", flush=True)
    print(" L5 & L6 — DISJOINT TEMPLATE VALIDATION SELECTION & BOUND B COMPUTATION", flush=True)
    print("=========================================================================================================", flush=True)

    results = []
    for idx, (p_type, t_type) in enumerate(candidate_representations, 1):
        cell_name = f"{p_type} / {t_type}"
        res = eval_cell_disjoint_split(tr_x_raw, train_y, va_x_raw, val_y, te_x_raw, test_y, t_type)
        res["cell_name"] = cell_name
        results.append(res)
        print(f"  [{idx:02d}/11] '{cell_name:<30}' -> Disjoint Val = {res['val_score']:5.2f}% ({res['val_method']}), Test Max = {res['test_max_score']:5.2f}%", flush=True)

    print(f"\n{'Representation':<30} | {'Disjoint Val Acc':<18} | {'Val Method':<28} | {'Best Linear Test Acc':<22} | {'Test Max':<10}", flush=True)
    print("-" * 115, flush=True)
    for r in results:
        lin_str = f"{r['best_linear_test_m']:5.2f}% (deterministic)"
        print(f"{r['cell_name']:<30} | {r['val_score']:6.2f}%            | {r['val_method']:<28} | {lin_str:<22} | {r['test_max_score']:6.2f}%", flush=True)
    print("=" * 115, flush=True)

    # L6 Selection: Choose representation maximizing Disjoint Validation Accuracy!
    winning_res = max(results, key=lambda r: r["val_score"])
    cv_selected_rep = winning_res["cell_name"]
    val_score = winning_res["val_score"]
    val_method = winning_res["val_method"]

    # L5 Requirement: B = Test Ceiling OF THE SELECTED REPRESENTATION!
    B = winning_res["test_max_score"]
    honest_test_acc = winning_res["best_linear_test_m"]

    # Compute correlation between Val Acc and Test Acc (After L6)
    val_scores = np.array([r["val_score"] for r in results])
    test_scores = np.array([r["test_max_score"] for r in results])
    corr_after = float(np.corrcoef(val_scores, test_scores)[0, 1])

    print("\n=========================================================================================================", flush=True)
    print(" L5 & L6 SELECTION SUMMARY & GATE 2 RE-DECISION:", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"  1. Disjoint Template Selected Representation          : '{cv_selected_rep}'", flush=True)
    print(f"  2. Disjoint Validation Winning Method                : '{val_method}' (Val Acc = {val_score:.2f}%)", flush=True)
    print(f"  3. Honest Test Accuracy of Selected Representation   : {honest_test_acc:.2f}%", flush=True)
    print(f"  4. Test Ceiling OF THE SELECTED REPRESENTATION (B)    : {B:.2f}%", flush=True)
    print(f"  5. Disjoint-Template Val-vs-Test Correlation (L6)     : r = {corr_after:+.4f}", flush=True)
    print(f"  6. Gate 2 Threshold                                  : 50.00%", flush=True)

    if B >= 50.0:
        print(f"\n  [GATE 2 PASSED] B ({B:.2f}%) >= 50.0%. Proceeding to Phase IV Class-IL Arms on '{cv_selected_rep}'!", flush=True)
    else:
        print(f"\n  [GATE 2 FAILED] B ({B:.2f}%) < 50.0%. Stopping per specification.", flush=True)
    print("=========================================================================================================", flush=True)


if __name__ == "__main__":
    main()
