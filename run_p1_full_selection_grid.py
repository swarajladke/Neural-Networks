"""
run_p1_full_selection_grid.py
=============================

P-Phase Directive P1 & P2:
Re-run the entire selection grid under eval_core.py.
Evaluate on v3 cache (smollm2_embeddings_v3_100facts_7_3_5.pt: 700 train / 300 val / 500 test).

(a) Full 11-cell M1 table:
    - Every grid config with converged: bool and final_loss: float
    - Val-selected config among converged configs only
    - Single honest test accuracy of the val-selected config
    - Max-over-configs test accuracy (optimistic ceiling)
    - len(grid) and N = len(grid) * n_cells from variables

(b) Full 12-point m sweep (16, 32, 48, 56, 64, 72, 80, 96, 112, 128, 256, 299) at eps=1e-4

(c) Diff table (old-stack value vs new-stack value vs delta), sorted by |delta| descending

(d) SELECTED_REPRESENTATION = argmax over validation accuracy under unified stack

(e) OPTIMISTIC_CEILING computed from new stack as a variable

(f) P2: Zero-selection NCM measurement vs Selected Test Acc -> SELECTION_PENALTY
"""

import os
import math
import torch
import torch.nn.functional as F
from eval_core import (
    transform_fit_train_only, evaluate_classifier_by_name, get_candidate_grid,
    eval_ncm, CANDIDATE_REPRESENTATIONS, WEIGHT_DECAYS
)

# Reference values from old N3 / M1 / M6 stack for diff calculation
OLD_M1_VALUES = {
    "mean / none": {"val": 85.67, "honest_test": 77.80, "config": ("Ridge", 0.1)},
    "mean / center": {"val": 84.33, "honest_test": 62.20, "config": ("MultinomialLogReg", 0.01)},
    "mean / center+ZCA_whiten": {"val": 19.33, "honest_test": 8.60, "config": ("MultinomialLogReg", 1.0)},
    "mean / pca_m16_eps1e-4": {"val": 61.67, "honest_test": 40.00, "config": ("MultinomialLogReg", 0.1)},
    "mean / pca_m32_eps1e-6": {"val": 89.33, "honest_test": 70.00, "config": ("MultinomialLogReg", 0.01)},
    "mean / pca_m32_eps1e-4": {"val": 89.33, "honest_test": 70.00, "config": ("MultinomialLogReg", 0.01)},
    "mean / pca_m64_eps1e-4": {"val": 96.00, "honest_test": 82.60, "config": ("MultinomialLogReg", 0.001)},
    "mean / pca_m128_eps1e-4": {"val": 58.67, "honest_test": 49.00, "config": ("MultinomialLogReg", 0.0001)},
    "mean / pca_m256_eps1e-4": {"val": 50.33, "honest_test": 43.40, "config": ("NCM", 0.0)},
    "mean / pca_m299_eps1e-4": {"val": 48.33, "honest_test": 42.20, "config": ("MultinomialLogReg", 10.0)},
    "mean / ledoit_wolf": {"val": 51.33, "honest_test": 53.00, "config": ("MultinomialLogReg", 1.0)}
}

OLD_M6_SWEEP_VALUES = {
    16:  {"val": 61.67, "honest_test": 40.00},
    32:  {"val": 89.33, "honest_test": 70.00},
    48:  {"val": 94.00, "honest_test": 82.80},
    56:  {"val": 94.33, "honest_test": 83.20},
    64:  {"val": 96.00, "honest_test": 82.60},
    72:  {"val": 95.33, "honest_test": 80.40},
    80:  {"val": 94.67, "honest_test": 79.60},
    96:  {"val": 83.33, "honest_test": 66.00},
    112: {"val": 67.00, "honest_test": 49.80},
    128: {"val": 58.67, "honest_test": 49.00},
    256: {"val": 50.33, "honest_test": 43.40},
    299: {"val": 48.33, "honest_test": 42.20}
}


def evaluate_cell_full_grid(tr_x_raw, tr_y, va_x_raw, va_y, te_x_raw, te_y, rep_name, grid):
    """
    Evaluates every config in candidate grid for a given representation.
    Returns:
      - val_selected_config
      - val_selected_acc
      - honest_test_acc
      - max_test_acc (optimistic ceiling for this cell)
      - test_selected_config
      - config_results: dict mapping (method, wd) -> {val_acc, test_acc, converged, final_loss}
    """
    tr_x, va_x = transform_fit_train_only(tr_x_raw, va_x_raw, rep_name)
    _, te_x    = transform_fit_train_only(tr_x_raw, te_x_raw, rep_name)

    config_results = {}
    converged_val_candidates = []

    for method, wd in grid:
        va_res = evaluate_classifier_by_name(tr_x, tr_y, va_x, va_y, method, wd)
        te_res = evaluate_classifier_by_name(tr_x, tr_y, te_x, te_y, method, wd)
        
        config_results[(method, wd)] = {
            "val_acc": va_res["accuracy"],
            "test_acc": te_res["accuracy"],
            "converged": va_res["converged"],
            "final_loss": va_res["final_loss"]
        }
        
        if va_res["converged"]:
            converged_val_candidates.append((va_res["accuracy"], (method, wd)))

    if not converged_val_candidates:
        raise RuntimeError(f"Zero converged configs for {rep_name}!")

    # Stable selection: argmax by validation accuracy
    best_item = max(converged_val_candidates, key=lambda x: x[0])
    val_selected_acc = best_item[0]
    val_selected_config = best_item[1]

    # Evaluate single honest test accuracy for the val-selected config
    honest_test_acc = config_results[val_selected_config]["test_acc"]

    # Compute optimistic ceiling (max over converged configs on test)
    best_test_item = max(
        [(v["test_acc"], k) for k, v in config_results.items() if v["converged"]],
        key=lambda x: x[0]
    )
    max_test_acc = best_test_item[0]
    test_selected_config = best_test_item[1]

    # Zero-selection NCM measurement
    ncm_test_acc = config_results[("NCM", 0.0)]["test_acc"]

    return {
        "val_selected_config": val_selected_config,
        "val_selected_acc": val_selected_acc,
        "honest_test_acc": honest_test_acc,
        "max_test_acc": max_test_acc,
        "test_selected_config": test_selected_config,
        "ncm_test_acc": ncm_test_acc,
        "config_results": config_results
    }


def main():
    cache_path = "smollm2_embeddings_v3_100facts_7_3_5.pt"
    if not os.path.isfile(cache_path):
        print(f"ERROR: {cache_path} not found.")
        return

    print("=========================================================================================================")
    print(" DIRECTIVE P1 -- FULL SELECTION RE-RUN UNDER eval_core.py UNIFIED STACK (R14/R15)")
    print("=========================================================================================================")

    d = torch.load(cache_path, weights_only=False)
    tr_x_raw, tr_y = d["train_x"], d["train_y"]
    va_x_raw, va_y = d["val_x"], d["val_y"]
    te_x_raw, te_y = d["test_x"], d["test_y"]

    grid = get_candidate_grid(include_headl1c=True)
    grid_len = len(grid)
    n_m1_cells = len(CANDIDATE_REPRESENTATIONS)
    n_test_evals_m1 = grid_len * n_m1_cells

    print(f"  Candidate grid size per cell : len(grid) = {grid_len} configs")
    print(f"  M1 Table cell count          : n_cells   = {n_m1_cells} cells")
    print(f"  Total M1 test evaluations    : N_m1      = {n_test_evals_m1} test evaluations (interpolated)\n")

    # =========================================================================
    # (a) Full 11-cell M1 table
    # =========================================================================
    print("---------------------------------------------------------------------------------------------------------")
    print(" (a) FULL 11-CELL M1 TABLE (Unified Stack, Float64 Transforms, R15 Convergence Filtering)")
    print("---------------------------------------------------------------------------------------------------------")

    m1_results = {}
    for rep in CANDIDATE_REPRESENTATIONS:
        clean_rep = rep.split(" [")[0].strip()
        res = evaluate_cell_full_grid(tr_x_raw, tr_y, va_x_raw, va_y, te_x_raw, te_y, rep, grid)
        m1_results[clean_rep] = res

        cfg = res["val_selected_config"]
        cfg_str = f"{cfg[0]} (wd={cfg[1]})"
        test_cfg = res["test_selected_config"]
        test_cfg_str = f"{test_cfg[0]} (wd={test_cfg[1]})"
        wd_differ = (cfg[1] != test_cfg[1]) if cfg[0] in ["Ridge", "MultinomialLogReg"] and test_cfg[0] in ["Ridge", "MultinomialLogReg"] else "N/A"

        print(f"  {clean_rep:<35} | Val Acc: {res['val_selected_acc']:5.2f}% via {cfg_str:<28} | Honest Test: {res['honest_test_acc']:5.2f}% | Max Test: {res['max_test_acc']:5.2f}% via {test_cfg_str}")

    # =========================================================================
    # (b) Full 12-point m sweep at eps=1e-4
    # =========================================================================
    print("\n---------------------------------------------------------------------------------------------------------")
    print(" (b) FULL 12-POINT PCA DIMENSION SWEEP (eps=1e-4, Unified Stack)")
    print("---------------------------------------------------------------------------------------------------------")

    m_sweep_points = [16, 32, 48, 56, 64, 72, 80, 96, 112, 128, 256, 299]
    n_sweep_cells = len(m_sweep_points)
    n_test_evals_sweep = grid_len * n_sweep_cells
    print(f"  Dimension sweep candidate configs : N_sweep = {n_test_evals_sweep} test evaluations ({n_sweep_cells} cells * {grid_len})\n")

    sweep_results = {}
    for m in m_sweep_points:
        rep_name = f"mean / pca_m{m}_eps1e-4"
        res = evaluate_cell_full_grid(tr_x_raw, tr_y, va_x_raw, va_y, te_x_raw, te_y, rep_name, grid)
        sweep_results[m] = res
        cfg = res["val_selected_config"]
        cfg_str = f"{cfg[0]} (wd={cfg[1]})"
        print(f"  m = {m:<3} | Val Acc: {res['val_selected_acc']:5.2f}% via {cfg_str:<28} | Honest Test: {res['honest_test_acc']:5.2f}% | Max Test: {res['max_test_acc']:5.2f}%")

    # =========================================================================
    # (c) Diff table (old-stack vs new-stack) sorted by |delta| descending
    # =========================================================================
    print("\n---------------------------------------------------------------------------------------------------------")
    print(" (c) DIFF TABLE -- OLD-STACK VS NEW-STACK (Sorted by |delta_test| descending)")
    print("---------------------------------------------------------------------------------------------------------")

    diff_rows = []
    # Collect from M1 cells
    for rep, old in OLD_M1_VALUES.items():
        if rep in m1_results:
            new = m1_results[rep]
            val_delta = new["val_selected_acc"] - old["val"]
            test_delta = new["honest_test_acc"] - old["honest_test"]
            cfg_changed = (new["val_selected_config"] != old["config"])
            diff_rows.append({
                "cell": rep,
                "old_val": old["val"],
                "new_val": new["val_selected_acc"],
                "val_delta": val_delta,
                "old_test": old["honest_test"],
                "new_test": new["honest_test_acc"],
                "test_delta": test_delta,
                "old_cfg": f"{old['config'][0]}({old['config'][1]})",
                "new_cfg": f"{new['val_selected_config'][0]}({new['val_selected_config'][1]})",
                "cfg_changed": cfg_changed
            })

    diff_rows.sort(key=lambda x: abs(x["test_delta"]), reverse=True)

    print(f"  {'Cell':<32} | {'Old Val':<8} {'New Val':<8} {'dVal':<8} | {'Old Test':<8} {'New Test':<8} {'dTest':<8} | {'Config Change':<30}")
    print(f"  {'-'*32}-|-{'-'*8}-{'-'*8}-{'-'*8}-|-{'-'*8}-{'-'*8}-{'-'*8}-|-{'-'*30}")
    
    n_cfg_changed = sum(1 for r in diff_rows if r["cfg_changed"])
    n_test_moved_gt5 = sum(1 for r in diff_rows if abs(r["test_delta"]) > 5.0)

    for r in diff_rows:
        cfg_str = f"{r['old_cfg']} -> {r['new_cfg']}" if r["cfg_changed"] else f"Unchanged ({r['new_cfg']})"
        print(f"  {r['cell']:<32} | {r['old_val']:6.2f}% {r['new_val']:6.2f}% {r['val_delta']:+6.2f}pp | {r['old_test']:6.2f}% {r['new_test']:6.2f}% {r['test_delta']:+6.2f}pp | {cfg_str:<30}")

    # =========================================================================
    # (d) New SELECTED_REPRESENTATION and (e) OPTIMISTIC_CEILING
    # =========================================================================
    print("\n---------------------------------------------------------------------------------------------------------")
    print(" (d) & (e) SELECTION OUTCOME AND OPTIMISTIC CEILING")
    print("---------------------------------------------------------------------------------------------------------")

    best_m1_rep = max(m1_results.keys(), key=lambda k: m1_results[k]["val_selected_acc"])
    best_res = m1_results[best_m1_rep]
    selected_rep = best_m1_rep
    val_acc_selected = best_res["val_selected_acc"]
    honest_test_acc_selected = best_res["honest_test_acc"]
    selected_config = best_res["val_selected_config"]

    # Recompute optimistic ceiling across all M1 candidate evaluations
    recomputed_optimistic_ceiling = max(res["max_test_acc"] for res in m1_results.values())
    selected_cell_max_test = best_res["max_test_acc"]

    print(f"  SELECTED_REPRESENTATION = '{selected_rep}'")
    print(f"  Validation Accuracy     = {val_acc_selected:.2f}%")
    print(f"  Val-Selected Config     = {selected_config[0]} (wd={selected_config[1]})")
    print(f"  HONEST_TEST_ACC         = {honest_test_acc_selected:.2f}%")
    print(f"  OPTIMISTIC_CEILING      = {selected_cell_max_test:.2f}% (max over N={n_test_evals_m1} test evaluations)")

    # =========================================================================
    # P2: Zero-selection NCM measurement & SELECTION_PENALTY
    # =========================================================================
    print("\n=========================================================================================================")
    print(" DIRECTIVE P2 -- ZERO-SELECTION NCM BENCHMARK VS VAL-SELECTED HEAD")
    print("=========================================================================================================")

    ncm_test_acc = best_res["ncm_test_acc"]
    selection_penalty = honest_test_acc_selected - ncm_test_acc

    print(f"  NCM_TEST_ACC       = {ncm_test_acc:.2f}% (zero hyperparameters, single test evaluation)")
    print(f"  SELECTED_TEST_ACC  = {honest_test_acc_selected:.2f}% (val-selected {selected_config[0]} wd={selected_config[1]})")
    print(f"  SELECTION_PENALTY  = {selection_penalty:+.2f} percentage points ({honest_test_acc_selected:.2f}% - {ncm_test_acc:.2f}%)")

    if selection_penalty < 0:
        print(f"  FINDING: SELECTION_PENALTY is negative ({selection_penalty:+.2f} pp).")
        print(f"  Validation-based model selection performs worse than the parameter-free NCM baseline on this representation.")
    else:
        print(f"  FINDING: SELECTION_PENALTY is non-negative ({selection_penalty:+.2f} pp).")

    # =========================================================================
    # Pre-registered predictions verification (P28, P29, P30)
    # =========================================================================
    print("\n=========================================================================================================")
    print(" PRE-REGISTERED PREDICTIONS SCORECARD VERIFICATION (P28, P29, P30)")
    print("=========================================================================================================")

    p28_verdict = (n_cfg_changed >= 4) and (n_test_moved_gt5 >= 1)
    print(f"  P28: Config changes = {n_cfg_changed} (>=4: {n_cfg_changed >= 4}), Test deltas > 5pp = {n_test_moved_gt5} (>=1: {n_test_moved_gt5 >= 1})")
    print(f"       Verdict: {'RIGHT' if p28_verdict else 'WRONG'}")

    p29_verdict = (selected_rep == "mean / pca_m64_eps1e-4") and (abs(selected_cell_max_test - 85.80) > 0.20 or abs(recomputed_optimistic_ceiling - 85.80) > 0.20)
    print(f"  P29: Selected rep is mean/pca_m64_eps1e-4: {selected_rep == 'mean / pca_m64_eps1e-4'}, Ceiling diff: {abs(selected_cell_max_test - 85.80):.2f}pp")
    print(f"       Verdict: {'RIGHT' if p29_verdict else 'WRONG'}")

    p30_verdict = (selection_penalty < 0) and (abs(selection_penalty) > 2.0)
    print(f"  P30: SELECTION_PENALTY negative: {selection_penalty < 0}, Magnitude > 2.0pp: {abs(selection_penalty) > 2.0} (|{selection_penalty:.2f}| pp)")
    print(f"       Verdict: {'RIGHT' if p30_verdict else 'WRONG'}")

if __name__ == "__main__":
    main()
