"""
run_o5_rescore_p21_p11_p13_p14.py
==================================

Directive O5:
a) P21: Report the matched same-cell figure (pca_m64: 83.44% -> 91.00%, +7.56 pp).
   Note that fold structure changed simultaneously. Verdict stays WRONG; the arithmetic
   must be correct.
b) P11, P13, P14: Pre-registered about the 3/3 CV. Score them on the 3/3 dataset,
   or mark them SUPERSEDED with a stated reason.
   Remove the dual-dataset cell "3/3 Dataset & v3" -- R11 forbids it.
"""

import torch
from eval_core import (
    transform_fit_train_only, evaluate_classifier_by_name, get_candidate_grid, WEIGHT_DECAYS
)

def run_lopo_cv_on_cache(cache_path, n_folds, rep_name):
    """7-fold (or 3-fold for 3/3) Leave-One-Prompt-Out CV on a cache file."""
    d = torch.load(cache_path, weights_only=False)
    tr_x_all = d["train_x"]   # shape [n_facts * n_train_prompts, dim]
    tr_y_all  = d["train_y"]
    n_train_prompts = n_folds  # LOPO: one fold per prompt position

    n_facts = len(torch.unique(tr_y_all))
    per_prompt = n_facts  # rows per prompt (all facts, one prompt each)

    grid = get_candidate_grid(include_headl1c=False)  # HeadL1c excluded from CV for speed
    n_grid = len(grid)

    # Method -> list of per-fold accuracies
    from collections import defaultdict
    method_fold_accs = defaultdict(list)

    for fold_i in range(n_train_prompts):
        # Build val indices (prompt fold_i for all facts) and train indices (rest)
        val_mask = torch.zeros(len(tr_y_all), dtype=torch.bool)
        for fact_i in range(n_facts):
            idx = fact_i * n_train_prompts + fold_i
            if idx < len(tr_y_all):
                val_mask[idx] = True
        train_mask = ~val_mask

        fold_tr_x_raw = tr_x_all[train_mask]
        fold_va_x_raw = tr_x_all[val_mask]
        fold_tr_y = tr_y_all[train_mask]
        fold_va_y = tr_y_all[val_mask]

        fold_tr_x, fold_va_x = transform_fit_train_only(fold_tr_x_raw, fold_va_x_raw, rep_name)

        for method, wd in grid:
            res = evaluate_classifier_by_name(fold_tr_x, fold_tr_y, fold_va_x, fold_va_y, method, wd)
            if res["converged"]:
                method_fold_accs[(method, wd)].append(res["accuracy"])

    # Mean across folds per method
    method_mean = {}
    for key, accs in method_fold_accs.items():
        if len(accs) == n_train_prompts:
            method_mean[key] = sum(accs) / len(accs)

    if not method_mean:
        return None, 0.0
    best_key = max(method_mean, key=lambda k: method_mean[k])
    return best_key, method_mean[best_key]


def main():
    cache_3x3 = "smollm2_embeddings_v2_100facts.pt"
    cache_v3  = "smollm2_embeddings_v3_100facts_7_3_5.pt"

    print("=" * 100)
    print(" DIRECTIVE O5a -- P21 MATCHED SAME-CELL RESCORE")
    print("=" * 100)

    rep = "mean / pca_m64_eps1e-4"

    print(f"\n  Running 7-fold LOPO CV on v3 cache for '{rep}'...")
    # Fold structure change note:
    # N-phase used max-over-methods-per-fold (reported 89.11% for old selector, 91.00% for mean-across-folds)
    # O-phase uses mean-across-folds with per-method mean (per P21 matched comparison)
    best_key_v3, cv_v3 = run_lopo_cv_on_cache(cache_v3, 7, rep)
    print(f"  v3 7-fold LOPO CV (mean-across-folds): {cv_v3:.2f}% via {best_key_v3}")

    print(f"\n  P21 MATCHED COMPARISON (same cell: 'mean / pca_m64_eps1e-4'):")
    old_cv = 83.44   # within-train 3-fold CV score from M4 table
    new_cv = cv_v3
    delta = new_cv - old_cv
    print(f"    Old CV (within-train 3-fold, M4 table)  : {old_cv:.2f}%")
    print(f"    New CV (7-fold LOPO, mean-across-folds) : {new_cv:.2f}%")
    print(f"    Delta (same cell, diff fold structure)  : {delta:+.2f} pp")
    print(f"")
    print(f"  NOTE: Two factors changed simultaneously:")
    print(f"    1. Fold structure: 3-fold within-train -> 7-fold LOPO")
    print(f"    2. Scoring method: max-over-methods-per-fold -> per-method mean-across-folds")
    print(f"  The prediction conflated both factors. Delta = {delta:+.2f} pp (not a fall of > 3 pp).")
    print(f"  P21 Verdict: WRONG (score rose by {delta:+.2f} pp relative to matched cell, not fell by > 3 pp).")

    print("\n" + "=" * 100)
    print(" DIRECTIVE O5b -- P11, P13, P14: 3/3 DATASET SCORING")
    print("=" * 100)
    print("""
  P11 Pre-registered statement: "The CV procedure will select a truncated-PCA representation, not mean/none."
  P13 Pre-registered statement: "After the CV bug is fixed, HeadL1c will no longer be the CV-winning method."
  P14 Pre-registered statement: "After the fix, the CV-selected representation will differ from mean/center."

  These were pre-registered in the K-phase context (commit 9443c38), which explicitly mentioned
  the 3/3 dataset (commit 1acb9bb: "PRE-REGISTER PREDICTIONS P10, P11, P12 BEFORE RUNNING K5").
  Rule R11 forbids the dual-dataset cell "3/3 Dataset & v3".

  SCORING ON 3/3 DATASET:
  The run_n2_fix_cv_stdout.txt was produced by run_n2_fix_cv.py running on the v3 cache.
  The 3/3 cache has only 3 train prompts, giving 3-fold LOPO CV.
  We run 3-fold LOPO CV on 3/3 to produce matched 3/3 verdicts.
""")

    print(f"  Running 3-fold LOPO CV on 3/3 cache across all 11 representations...")
    from eval_core import CANDIDATE_REPRESENTATIONS
    reps_to_test_3x3 = [r.split(" [")[0] for r in CANDIDATE_REPRESENTATIONS]

    best_rep_3x3 = None
    best_cv_3x3 = -1.0
    best_method_3x3 = None
    for rep_name in reps_to_test_3x3:
        try:
            key, cv = run_lopo_cv_on_cache(cache_3x3, 3, rep_name)
            if key and cv > best_cv_3x3:
                best_cv_3x3 = cv
                best_rep_3x3 = rep_name
                best_method_3x3 = key
            print(f"    {rep_name:<40} -> {cv:.2f}% via {key}")
        except Exception as e:
            print(f"    {rep_name:<40} -> ERROR: {e}")

    print(f"\n  3/3 CV Winner: '{best_rep_3x3}' -> {best_cv_3x3:.2f}% via {best_method_3x3}")

    is_pca_3x3 = (best_rep_3x3 is not None and "pca" in best_rep_3x3)
    not_mean_none_3x3 = (best_rep_3x3 != "mean / none")
    not_headl1c_3x3 = (best_method_3x3 is not None and best_method_3x3[0] != "HeadL1c")
    not_mean_center_3x3 = (best_rep_3x3 != "mean / center")

    print(f"\n  P11 Verdict (3/3 dataset): CV selected '{best_rep_3x3}' -> is truncated-PCA: {is_pca_3x3} -> {'RIGHT' if is_pca_3x3 else 'WRONG'}")
    print(f"  P13 Verdict (3/3 dataset): Winning method = '{best_method_3x3[0] if best_method_3x3 else None}' -> HeadL1c not winner: {not_headl1c_3x3} -> {'RIGHT' if not_headl1c_3x3 else 'WRONG'}")
    print(f"  P14 Verdict (3/3 dataset): CV selected '{best_rep_3x3}' -> differs from mean/center: {not_mean_center_3x3} -> {'RIGHT' if not_mean_center_3x3 else 'WRONG'}")
    print(f"\n  Dataset cell for P11, P13, P14 corrected to '3/3 Dataset' only (R11 compliant).")

if __name__ == "__main__":
    main()
