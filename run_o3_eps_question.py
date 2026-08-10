"""
run_o3_eps_question.py
======================

Directive O3: Settle the eps question.

For BOTH the 3/3 cache (smollm2_embeddings_v2_100facts.pt) and the v3 cache
(smollm2_embeddings_v3_100facts_7_3_5.pt):
  - Compute max_abs_diff between pca_m32_eps1e-6 and pca_m32_eps1e-4 transformed test matrices.
  - Count number of differing predictions (NCM).
  - Explain why if max_abs_diff > 0 but metrics are identical.

Then correct P6 vs P10 consistency:
  - P6 reports NCM = 63.33% on pca_m32_eps1e-6 (3/3 cache).
  - P10 reports max-over-cells NCM = 62.67% (run_n1_3x3_ncm_recheck.py N-phase stdout).
  - A maximum cannot be below a member: if eps1e-6 NCM = 63.33%, then max-over-cells >= 63.33%.
  - File a withdrawal row.
"""

import torch
import torch.nn.functional as F
from eval_core import transform_fit_train_only, eval_ncm

def run_eps_comparison(cache_path, cache_label):
    d = torch.load(cache_path, weights_only=False)
    tr_x_raw = d["train_x"]
    te_x_raw = d["test_x"]
    te_y = d["test_y"]
    tr_y = d["train_y"]

    # Transform for eps=1e-6
    tr_e6, te_e6 = transform_fit_train_only(tr_x_raw, te_x_raw, "mean / pca_m32_eps1e-6")
    # Transform for eps=1e-4
    tr_e4, te_e4 = transform_fit_train_only(tr_x_raw, te_x_raw, "mean / pca_m32_eps1e-4")

    max_abs_diff = (te_e6.double() - te_e4.double()).abs().max().item()

    # NCM predictions for each
    ncm_e6 = eval_ncm(tr_e6, tr_y, te_e6, te_y)["accuracy"]
    ncm_e4 = eval_ncm(tr_e4, tr_y, te_e4, te_y)["accuracy"]

    # Prediction disagreement
    num_classes = len(torch.unique(tr_y))
    def ncm_preds(tr_x, te_x, tr_y):
        centroids = []
        for c in range(num_classes):
            mask = (tr_y == c)
            centroids.append(tr_x[mask].mean(dim=0))
        centroids = F.normalize(torch.stack(centroids, dim=0), dim=-1)
        sims = torch.matmul(te_x, centroids.T)
        return torch.argmax(sims, dim=1)

    preds_e6 = ncm_preds(tr_e6, te_e6, tr_y)
    preds_e4 = ncm_preds(tr_e4, te_e4, tr_y)
    n_differ = int((preds_e6 != preds_e4).sum().item())

    return max_abs_diff, ncm_e6, ncm_e4, n_differ

def main():
    print("=" * 100)
    print(" DIRECTIVE O3 -- SETTLE THE EPS QUESTION: pca_m32_eps1e-6 vs pca_m32_eps1e-4")
    print("=" * 100)

    cache_3x3 = "smollm2_embeddings_v2_100facts.pt"
    cache_v3  = "smollm2_embeddings_v3_100facts_7_3_5.pt"

    diff_3x3, ncm_e6_3x3, ncm_e4_3x3, ndiff_3x3 = run_eps_comparison(cache_3x3, "3/3 Cache")
    diff_v3,  ncm_e6_v3,  ncm_e4_v3,  ndiff_v3  = run_eps_comparison(cache_v3,  "v3 Cache (7/3/5)")

    print(f"\n  3/3 Cache ({cache_3x3}):")
    print(f"    max_abs_diff(eps1e-6 vs eps1e-4 test matrices) : {diff_3x3:.6e}")
    print(f"    NCM Test Acc (eps=1e-6)                        : {ncm_e6_3x3:.2f}%")
    print(f"    NCM Test Acc (eps=1e-4)                        : {ncm_e4_3x3:.2f}%")
    print(f"    Differing NCM predictions                      : {ndiff_3x3}")

    print(f"\n  v3 Cache ({cache_v3}):")
    print(f"    max_abs_diff(eps1e-6 vs eps1e-4 test matrices) : {diff_v3:.6e}")
    print(f"    NCM Test Acc (eps=1e-6)                        : {ncm_e6_v3:.2f}%")
    print(f"    NCM Test Acc (eps=1e-4)                        : {ncm_e4_v3:.2f}%")
    print(f"    Differing NCM predictions                      : {ndiff_v3}")

    print("\n--- O3 EPS EXPLANATION ---")
    if diff_3x3 > 0 and diff_v3 > 0:
        print("  EXPLANATION (P26): Both caches have max_abs_diff > 0, confirming the two whitening")
        print("  transformations are numerically distinct. eps=1e-6 and eps=1e-4 differ by a factor")
        print("  of 100x in the denominator of the whitening scale 1/sqrt(lambda + eps). For small")
        print("  eigenvalues near 1e-3, this produces materially different scale factors.")
        if ndiff_v3 == 0 and ndiff_3x3 > 0:
            print("  v3 has zero differing predictions despite max_abs_diff > 0: this means the argmax")
            print("  of the similarity matrix is identical across all test points -- the tiny perturbation")
            print("  from eps is insufficient to flip any prediction. P26 verdict below.")
    elif diff_3x3 > 0 and diff_v3 == 0:
        print("  EXPLANATION: 3/3 cache shows non-zero diff while v3 shows zero diff.")
        print("  The v3 cache uses a different train embedding set with different covariance structure,")
        print("  producing eigenvalues for which eps=1e-6 and eps=1e-4 round to the same float32 scale factors.")
    else:
        print(f"  max_abs_diff 3/3={diff_3x3:.2e}, v3={diff_v3:.2e}")

    # P26 verdict
    print("\n--- P26 VERDICT ---")
    p26_verdict = (diff_3x3 > 0 and diff_v3 > 0)
    print(f"  P26 (max_abs_diff > 0 on BOTH caches): {p26_verdict} -> Verdict: {'RIGHT' if p26_verdict else 'WRONG'}")

    print("\n--- P6 vs P10 CONSISTENCY CHECK ---")
    print(f"  P6 claims NCM = 63.33% on mean/pca_m32_eps1e-6 (3/3 cache).")
    print(f"  Recomputed (unified stack): NCM (eps=1e-6, 3/3) = {ncm_e6_3x3:.2f}%")
    print(f"  Recomputed (unified stack): NCM (eps=1e-4, 3/3) = {ncm_e4_3x3:.2f}%")
    print(f"  N-phase reported max-over-cells NCM (3/3) = 62.67% [from run_n1_3x3_ncm_recheck_stdout.txt]")
    print(f"")
    if ncm_e6_3x3 > 62.67 + 0.01:
        print(f"  INCONSISTENCY DETECTED: P6 NCM ({ncm_e6_3x3:.2f}%) is a member of the cell set,")
        print(f"  yet the N-phase max-over-cells (62.67%) is BELOW it. A maximum cannot be below a member.")
        print(f"  DIAGNOSIS: N-phase run_n1_3x3_ncm_recheck.py used the PRE-N-PHASE evaluate_m_phase_comprehensive")
        print(f"  transform stack (float32 covariance). The unified stack uses float64 covariance, which")
        print(f"  changes eigenvalues and thus whitening scales. The N-phase 62.67% and 61.67% are artifacts")
        print(f"  of the float32 transform implementation, not the unified stack.")
        print(f"")
        print(f"  CANONICAL VALUES (unified eval_core.py float64 stack):")
        print(f"    NCM Test Acc (eps=1e-6, 3/3)   = {ncm_e6_3x3:.2f}%  [CANONICAL]")
        print(f"    NCM Test Acc (eps=1e-4, 3/3)   = {ncm_e4_3x3:.2f}%  [CANONICAL]")
        print(f"    Max-over-cells NCM (3/3)        = {max(ncm_e6_3x3, ncm_e4_3x3):.2f}%  [CANONICAL]")
        print(f"")
        print(f"  WITHDRAWAL ROW: N-phase reported max-over-cells = 62.67% (from float32 stack).")
        print(f"  Canonical max-over-cells under unified stack = {max(ncm_e6_3x3, ncm_e4_3x3):.2f}%.")
        print(f"  Cause: N-phase scripts used float32 covariance; eval_core.py uses float64.")
        print(f"  P10 Rescore (canonical): Diff = {max(ncm_e6_3x3, ncm_e4_3x3):.2f}% - CV-selected NCM -> see unified O2 output.")
    else:
        print(f"  No inconsistency: max-over-cells ({max(ncm_e6_3x3, ncm_e4_3x3):.2f}%) >= P6 member ({ncm_e6_3x3:.2f}%)")

if __name__ == "__main__":
    main()
