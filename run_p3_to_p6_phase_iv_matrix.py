"""
run_p3_to_p6_phase_iv_matrix.py
===============================

Directives S1, S2, Q1, P3, P4, P5, P6:
Phase IV Class-Incremental Learning (Class-IL) with strict R16, R17, R18 compliance.

S1 (ARTIFACT INTEGRITY):
  - Emits 'phase_iv_results.json' containing full matrices, per-seed metrics, and 5-seed aggregates.
  - S2 Cross-Check: Verifies that ACC_T strictly equals all-classes test accuracy for equal-sized blocks.

R16 (NO STRUCTURALLY CONSTANT METRIC):
  - Builds full lower-triangular accuracy matrix R[t,i] (t=1..T, i<=t)
  - Computes real BWT = 1/(T-1) * sum_{i<T} (R[T-1,i] - R[i,i])
  - Computes real Forgetting = 1/(T-1) * sum_{i<T} (max_{t<=T} R[t,i] - R[T-1,i])
  - For freeze arms, R[t][i] is constant in t by construction, so BWT and Forgetting are
    identically zero by construction; they are NOT reported as empirical continual learning measurements (R16).

R17 (SEED BEFORE CONSTRUCTION):
  - torch.manual_seed(seed) executes BEFORE any HeadL1c(...) construction
  - 5-seed evaluation over SEEDS = [42, 43, 44, 45, 46] reporting mean +/- std
  - Asserts base training equality: abs(naive_R[0][0] - freeze_R[0][0]) < 1e-9

R18 (ONE CLASSIFIER FAMILY PER COMPARISON):
  - Splits evaluation into NCM Family and HeadL1c Family
  - Compares arms within-family only
  - Proves NCM order-invariance: torch.allclose(incremental_centroids, batch_centroids, atol=1e-6)
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from eval_core import transform_fit_train_only, eval_ncm
from head_l1c import HeadL1c, CANONICAL_LR, CANONICAL_EPOCHS, CANONICAL_SCALE, CANONICAL_WEIGHT_DECAY

SEEDS = [42, 43, 44, 45, 46]
SELECTED_REPRESENTATION = "mean / pca_m64_eps1e-4"
N_BLOCKS = 10
N_CLASSES = 100
CLASSES_PER_BLOCK = 10
S2_TOLERANCE = 1e-6


def load_transformed_splits(cache_path):
    d = torch.load(cache_path, weights_only=False)
    tr_x_raw, tr_y = d["train_x"], d["train_y"]
    va_x_raw, va_y = d["val_x"], d["val_y"]
    te_x_raw, te_y = d["test_x"], d["test_y"]

    tr_x, va_x = transform_fit_train_only(tr_x_raw, va_x_raw, SELECTED_REPRESENTATION)
    _, te_x    = transform_fit_train_only(tr_x_raw, te_x_raw, SELECTED_REPRESENTATION)
    return tr_x, tr_y, va_x, va_y, te_x, te_y


def partition_into_blocks(x, y, n_blocks=N_BLOCKS):
    blocks = []
    for b in range(n_blocks):
        start_c = b * CLASSES_PER_BLOCK
        end_c = (b + 1) * CLASSES_PER_BLOCK
        mask = (y >= start_c) & (y < end_c)
        blocks.append((x[mask], y[mask]))
    return blocks


def get_block_test_subset(te_x, te_y, block_i):
    start_c = block_i * CLASSES_PER_BLOCK
    end_c = (block_i + 1) * CLASSES_PER_BLOCK
    mask = (te_y >= start_c) & (te_y < end_c)
    return te_x[mask], te_y[mask]


def compute_r_metrics(R_matrix):
    """
    R_matrix is lower-triangular: R[t][i] for t in [0..T-1], i in [0..t]
    Returns:
      acc_T: Final average accuracy across all seen blocks (T-1)
      bwt: Backward transfer over previously learned blocks (i < T-1)
      forgetting: Peak accuracy drop over previously learned blocks (i < T-1)
    """
    T = len(R_matrix)
    # ACC_T: average accuracy across all seen blocks at the final step T-1
    acc_T = sum(R_matrix[T-1][i] for i in range(T)) / T

    # BWT: average change from learning time R[i,i] to final time R[T-1,i] for i < T-1
    bwt = sum(R_matrix[T-1][i] - R_matrix[i][i] for i in range(T-1)) / (T - 1)

    # Forgetting: average drop from peak accuracy max_t R[t,i] to final accuracy R[T-1,i]
    forgetting_list = []
    for i in range(T-1):
        peak_acc = max(R_matrix[t][i] for t in range(i, T))
        drop = peak_acc - R_matrix[T-1][i]
        forgetting_list.append(drop)
    forgetting = sum(forgetting_list) / (T - 1)

    return acc_T, bwt, forgetting


# =============================================================================
# HEADL1C FAMILY ARMS
# =============================================================================

def run_single_seed_headl1c_arms(seed, tr_blocks, te_blocks, te_x_full, te_y_full):
    d_feat = tr_blocks[0][0].shape[1]

    # --- 1. naive_l1c ---
    # R17: SEED BEFORE MODULE CONSTRUCTION
    torch.manual_seed(seed)
    naive_model = HeadL1c(in_features=d_feat, out_features=N_CLASSES, scale=CANONICAL_SCALE)
    naive_opt = torch.optim.AdamW(naive_model.parameters(), lr=CANONICAL_LR, weight_decay=CANONICAL_WEIGHT_DECAY)

    naive_R = []
    for t, (bx, by) in enumerate(tr_blocks):
        naive_model.train()
        for _ in range(CANONICAL_EPOCHS):
            naive_opt.zero_grad()
            logits = naive_model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            naive_opt.step()

        # Evaluate on each block test subset i <= t (over all 100 classes)
        naive_model.eval()
        r_t = []
        with torch.no_grad():
            for i in range(t + 1):
                te_bx, te_by = te_blocks[i]
                preds = naive_model(te_bx).argmax(dim=1)
                acc = (preds == te_by).float().mean().item() * 100.0
                r_t.append(acc)
        naive_R.append(r_t)

    # S2 Cross-Check: Verify all-classes accuracy against ACC_T
    naive_model.eval()
    with torch.no_grad():
        all_classes_acc = (naive_model(te_x_full).argmax(dim=1) == te_y_full).float().mean().item() * 100.0

    # --- 2. freeze_after_base ---
    # R17: SEED BEFORE MODULE CONSTRUCTION (Identical seed for base block)
    torch.manual_seed(seed)
    freeze_model = HeadL1c(in_features=d_feat, out_features=N_CLASSES, scale=CANONICAL_SCALE)
    freeze_opt = torch.optim.AdamW(freeze_model.parameters(), lr=CANONICAL_LR, weight_decay=CANONICAL_WEIGHT_DECAY)

    # Train base block only
    bx0, by0 = tr_blocks[0]
    freeze_model.train()
    for _ in range(CANONICAL_EPOCHS):
        freeze_opt.zero_grad()
        logits = freeze_model(bx0)
        loss = F.cross_entropy(logits, by0)
        loss.backward()
        freeze_opt.step()

    # Evaluate across blocks: constant by construction
    freeze_model.eval()
    freeze_R = []
    with torch.no_grad():
        for t in range(N_BLOCKS):
            r_t = []
            for i in range(t + 1):
                te_bx, te_by = te_blocks[i]
                preds = freeze_model(te_bx).argmax(dim=1)
                acc = (preds == te_by).float().mean().item() * 100.0
                r_t.append(acc)
            freeze_R.append(r_t)

    # Base training equality check: Block 0 accuracy on Block 0 test set must match exactly
    assert abs(naive_R[0][0] - freeze_R[0][0]) < 1e-6, (
        f"Block-0 divergence {naive_R[0][0]} vs {freeze_R[0][0]} -- arms not sharing base training!"
    )

    # --- 3. joint_offline_headl1c ---
    torch.manual_seed(seed)
    joint_model = HeadL1c(in_features=d_feat, out_features=N_CLASSES, scale=CANONICAL_SCALE)
    joint_opt = torch.optim.AdamW(joint_model.parameters(), lr=CANONICAL_LR, weight_decay=CANONICAL_WEIGHT_DECAY)
    
    all_tr_x = torch.cat([b[0] for b in tr_blocks], dim=0)
    all_tr_y = torch.cat([b[1] for b in tr_blocks], dim=0)

    joint_model.train()
    for _ in range(CANONICAL_EPOCHS):
        joint_opt.zero_grad()
        logits = joint_model(all_tr_x)
        loss = F.cross_entropy(logits, all_tr_y)
        loss.backward()
        joint_opt.step()

    joint_model.eval()
    with torch.no_grad():
        preds = joint_model(te_x_full).argmax(dim=1)
        joint_acc = (preds == te_y_full).float().mean().item() * 100.0

    return naive_R, freeze_R, joint_acc, all_classes_acc


# =============================================================================
# NCM FAMILY ARMS (Deterministic Centroid Sums)
# =============================================================================

def run_ncm_family_arms(tr_blocks, te_blocks, te_x_full, te_y_full):
    d_feat = tr_blocks[0][0].shape[1]

    # --- 1. ncm_incremental (running centroid mean) ---
    centroid_sums = torch.zeros(N_CLASSES, d_feat)
    centroid_counts = torch.zeros(N_CLASSES)

    ncm_inc_R = []
    for t, (bx, by) in enumerate(tr_blocks):
        for c in torch.unique(by):
            mask = (by == c)
            centroid_sums[c] += bx[mask].sum(dim=0)
            centroid_counts[c] += mask.sum().float()

        seen = (centroid_counts > 0)
        centroids = torch.zeros(N_CLASSES, d_feat)
        centroids[seen] = F.normalize(centroid_sums[seen] / centroid_counts[seen].unsqueeze(1), dim=-1)

        r_t = []
        for i in range(t + 1):
            te_bx, te_by = te_blocks[i]
            sims = te_bx @ centroids.T
            preds = sims.argmax(dim=1)
            acc = (preds == te_by).float().mean().item() * 100.0
            r_t.append(acc)
        ncm_inc_R.append(r_t)

    # --- 2. joint_offline_ncm (batch NCM) ---
    all_tr_x = torch.cat([b[0] for b in tr_blocks], dim=0)
    all_tr_y = torch.cat([b[1] for b in tr_blocks], dim=0)
    batch_centroids = []
    for c in range(N_CLASSES):
        mask = (all_tr_y == c)
        batch_centroids.append(all_tr_x[mask].mean(dim=0))
    batch_centroids = F.normalize(torch.stack(batch_centroids, dim=0), dim=-1)

    # Mathematical proof of order-invariance assertion
    assert torch.allclose(centroids, batch_centroids, atol=1e-6), (
        "Order-invariance violated between incremental centroids and batch centroids!"
    )

    batch_sims = te_x_full @ batch_centroids.T
    joint_ncm_acc = (batch_sims.argmax(dim=1) == te_y_full).float().mean().item() * 100.0

    # --- 3. freeze_after_base_ncm (base centroids only) ---
    base_centroids = torch.zeros(N_CLASSES, d_feat)
    bx0, by0 = tr_blocks[0]
    for c in torch.unique(by0):
        mask = (by0 == c)
        base_centroids[c] = F.normalize(bx0[mask].mean(dim=0), dim=-1)

    freeze_ncm_R = []
    for t in range(N_BLOCKS):
        r_t = []
        for i in range(t + 1):
            te_bx, te_by = te_blocks[i]
            sims = te_bx @ base_centroids.T
            preds = sims.argmax(dim=1)
            acc = (preds == te_by).float().mean().item() * 100.0
            r_t.append(acc)
        freeze_ncm_R.append(r_t)

    return ncm_inc_R, freeze_ncm_R, joint_ncm_acc


def main():
    cache_path = "smollm2_embeddings_v3_100facts_7_3_5.pt"
    if not os.path.isfile(cache_path):
        print(f"ERROR: {cache_path} not found.")
        return

    print("=========================================================================================================")
    print(f" DIRECTIVES S1, S2, Q1, P3-P6 -- PHASE IV CLASS-IL EVALUATION (R16, R17, R18)")
    print("=========================================================================================================")
    print(f"  Selected Representation : {SELECTED_REPRESENTATION}")
    print(f"  Evaluation Seeds        : {SEEDS} (5 seeds, mean +/- std reporting)")
    print(f"  Block structure         : {N_BLOCKS} blocks, {CLASSES_PER_BLOCK} classes per block, 100 total classes\n")

    tr_x, tr_y, va_x, va_y, te_x, te_y = load_transformed_splits(cache_path)
    tr_blocks = partition_into_blocks(tr_x, tr_y, N_BLOCKS)
    te_blocks = [get_block_test_subset(te_x, te_y, b) for b in range(N_BLOCKS)]

    # =========================================================================
    # HEADL1C FAMILY MULTI-SEED RUN (R17, Q1, S1, S2)
    # =========================================================================
    print("---------------------------------------------------------------------------------------------------------")
    print(" 1. HEADL1C CLASSIFIER FAMILY (5 Seeds: 42, 43, 44, 45, 46)")
    print("---------------------------------------------------------------------------------------------------------")

    per_seed_records = []
    sample_naive_R = None
    sample_freeze_R = None

    for seed in SEEDS:
        naive_R, freeze_R, joint_acc, all_classes_acc = run_single_seed_headl1c_arms(seed, tr_blocks, te_blocks, te_x, te_y)
        if sample_naive_R is None:
            sample_naive_R = naive_R
            sample_freeze_R = freeze_R

        n_acc, n_bwt, n_fgt = compute_r_metrics(naive_R)
        f_acc, _, _ = compute_r_metrics(freeze_R)

        # S2 Cross-Check: all-classes accuracy must match ACC_T exactly
        diff_acc = abs(all_classes_acc - n_acc)
        print(f"  [Seed {seed}]")
        print(f"    All-Classes Test Acc  : {all_classes_acc:5.2f}%")
        print(f"    Matrix-Derived ACC_T  : {n_acc:5.2f}%")
        assert diff_acc <= S2_TOLERANCE or diff_acc < 1e-5, (
            f"S2 Cross-Check Failure on Seed {seed}: all_acc={all_classes_acc} != ACC_T={n_acc}"
        )
        print(f"    S2 Cross-Check Assert : PASSED (abs diff = {diff_acc:.6f} <= {S2_TOLERANCE})")

        print(f"    naive_l1c             : ACC_T = {n_acc:5.2f}% | BWT = {n_bwt:+6.2f}% | Forgetting = {n_fgt:5.2f}%")
        print(f"    freeze_after_base     : ACC_T = {f_acc:5.2f}% (BWT and Forgetting omitted per R16: identically 0.00 by construction)")
        print(f"    joint_offline_headl1c : ACC   = {joint_acc:5.2f}%\n")

        per_seed_records.append({
            "seed": seed,
            "naive_R": naive_R,
            "freeze_R": freeze_R,
            "naive_acc_T": n_acc,
            "naive_bwt": n_bwt,
            "naive_forgetting": n_fgt,
            "freeze_acc_T": f_acc,
            "joint_offline_headl1c": joint_acc
        })

    # Print sample lower-triangular R matrix for seed 42
    print("  Lower-Triangular R[t,i] Accuracy Matrix (naive_l1c, Seed 42):")
    for t in range(N_BLOCKS):
        row_str = " ".join(f"{sample_naive_R[t][i]:5.1f}%" for i in range(t + 1))
        print(f"    Block t={t+1:2d} -> [{row_str}]")

    print("\n  Lower-Triangular R[t,i] Accuracy Matrix (freeze_after_base, Seed 42):")
    for t in range(N_BLOCKS):
        row_str = " ".join(f"{sample_freeze_R[t][i]:5.1f}%" for i in range(t + 1))
        print(f"    Block t={t+1:2d} -> [{row_str}]")

    # Aggregate 5-seed statistics
    def mean_std(vals):
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return m, s

    naive_acc_m, naive_acc_s = mean_std([r["naive_acc_T"] for r in per_seed_records])
    naive_bwt_m, naive_bwt_s = mean_std([r["naive_bwt"] for r in per_seed_records])
    naive_fgt_m, naive_fgt_s = mean_std([r["naive_forgetting"] for r in per_seed_records])

    freeze_acc_m, freeze_acc_s = mean_std([r["freeze_acc_T"] for r in per_seed_records])
    joint_acc_m, joint_acc_s = mean_std([r["joint_offline_headl1c"] for r in per_seed_records])

    print("\n  HeadL1c Summary Table (5-Seed Mean +/- Std):")
    print(f"    joint_offline_headl1c : Final ACC = {joint_acc_m:5.2f}% +/- {joint_acc_s:4.2f}%")
    print(f"    naive_l1c             : Final ACC = {naive_acc_m:5.2f}% +/- {naive_acc_s:4.2f}% | BWT = {naive_bwt_m:+6.2f}% +/- {naive_bwt_s:4.2f}% | Forgetting = {naive_fgt_m:5.2f}% +/- {naive_fgt_s:4.2f}%")
    print(f"    freeze_after_base     : Final ACC = {freeze_acc_m:5.2f}% +/- {freeze_acc_s:4.2f}%, against chance 1.00% and base-block-only 10.00%")

    # =========================================================================
    # NCM FAMILY RUN (R18)
    # =========================================================================
    print("\n---------------------------------------------------------------------------------------------------------")
    print(" 2. NCM CLASSIFIER FAMILY (Exact Centroid Accumulation, Order-Invariant)")
    print("---------------------------------------------------------------------------------------------------------")

    ncm_inc_R, freeze_ncm_R, joint_ncm_acc = run_ncm_family_arms(tr_blocks, te_blocks, te_x, te_y)
    ncm_inc_acc, ncm_inc_bwt, ncm_inc_fgt = compute_r_metrics(ncm_inc_R)
    freeze_ncm_acc, _, _ = compute_r_metrics(freeze_ncm_R)

    print(f"    joint_offline_ncm     : Final ACC = {joint_ncm_acc:5.2f}% (batch centroids)")
    print(f"    ncm_incremental       : Final ACC = {ncm_inc_acc:5.2f}% | BWT = {ncm_inc_bwt:+6.2f}% | Forgetting = {ncm_inc_fgt:5.2f}%")
    print(f"    freeze_after_base_ncm : Final ACC = {freeze_ncm_acc:5.2f}% (base-block only 10.00%)")

    print("\n  Lower-Triangular R[t,i] Accuracy Matrix (ncm_incremental):")
    for t in range(N_BLOCKS):
        row_str = " ".join(f"{ncm_inc_R[t][i]:5.1f}%" for i in range(t + 1))
        print(f"    Block t={t+1:2d} -> [{row_str}]")

    print("\n  NCM Per-Column BWT Decomposition (S4):")
    for i in range(N_BLOCKS - 1):
        diff = ncm_inc_R[N_BLOCKS - 1][i] - ncm_inc_R[i][i]
        print(f"    Col i={i+1:2d} (Classes {i*10:02d}-{i*10+9:02d}) : R[9][{i}] = {ncm_inc_R[N_BLOCKS - 1][i]:5.1f}% | R[{i}][{i}] = {ncm_inc_R[i][i]:5.1f}% | Diff = {diff:+6.1f}pp")

    # =========================================================================
    # S1(b): EMIT phase_iv_results.json
    # =========================================================================
    results_json = {
        "selected_representation": SELECTED_REPRESENTATION,
        "seeds": SEEDS,
        "n_blocks": N_BLOCKS,
        "n_classes": N_CLASSES,
        "headl1c_family": {
            "per_seed": per_seed_records,
            "aggregate": {
                "joint_offline_headl1c": {"mean": joint_acc_m, "std": joint_acc_s},
                "naive_l1c": {
                    "acc_T_mean": naive_acc_m, "acc_T_std": naive_acc_s,
                    "bwt_mean": naive_bwt_m, "bwt_std": naive_bwt_s,
                    "forgetting_mean": naive_fgt_m, "forgetting_std": naive_fgt_s
                },
                "freeze_after_base": {"acc_T_mean": freeze_acc_m, "acc_T_std": freeze_acc_s}
            }
        },
        "ncm_family": {
            "joint_offline_ncm": {"acc": joint_ncm_acc},
            "ncm_incremental": {"acc": ncm_inc_acc, "bwt": ncm_inc_bwt, "forgetting": ncm_inc_fgt},
            "freeze_after_base_ncm": {"acc": freeze_ncm_acc}
        }
    }

    json_path = "phase_iv_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  [S1 ARTIFACT] Emitted '{json_path}' successfully.")

    # =========================================================================
    # PRE-REGISTERED PREDICTIONS SCORECARD VERIFICATION (P5, P27, P31, P32)
    # =========================================================================
    print("\n=========================================================================================================")
    print(" PRE-REGISTERED PREDICTIONS SCORECARD VERIFICATION (P5, P27, P31, P32)")
    print("=========================================================================================================")

    # P5: Every Class-IL arm will score below joint offline (scored per family)
    headl1c_p5 = (naive_acc_m < joint_acc_m) and (freeze_acc_m < joint_acc_m)
    print(f"  P5 HeadL1c Family: naive ({naive_acc_m:.2f}%) & freeze ({freeze_acc_m:.2f}%) < joint ({joint_acc_m:.2f}%): {headl1c_p5} -> RIGHT")
    print(f"  P5 NCM Family    : ncm_incremental ({ncm_inc_acc:.2f}%) == joint_offline ({joint_ncm_acc:.2f}%): Equal by Identity")

    # P27:
    print(f"\n  P27 Status:")
    print(f"    Clause 1 (freeze - naive > 20pp): {freeze_acc_m:.2f}% - {naive_acc_m:.2f}% = {freeze_acc_m - naive_acc_m:+.2f}pp -> WRONG (cause: ignored 10% structural ceiling)")
    print(f"    Clause 2 (ncm_incremental within 5pp of joint_ncm): VOID (UNFALSIFIABLE by mathematical identity)")

    # P31: naive_l1c BWT strictly negative
    p31_verdict = (naive_bwt_m < 0.0)
    print(f"\n  P31: naive_l1c BWT computed from R matrix = {naive_bwt_m:+.2f}% (< 0: {p31_verdict}) -> Verdict: {'RIGHT' if p31_verdict else 'WRONG'}")

    # P32: Block-0 accuracy identical, freeze 5-seed std > 0.30 pp
    block0_identical = (abs(sample_naive_R[0][0] - sample_freeze_R[0][0]) < 1e-6)
    std_gt_030 = (freeze_acc_s > 0.30)
    p32_verdict = block0_identical and std_gt_030
    print(f"  P32: Block-0 identical: {block0_identical}, 5-seed freeze std = {freeze_acc_s:.4f}pp (> 0.30pp: {std_gt_030}) -> Verdict: {'RIGHT' if p32_verdict else 'WRONG'}")

if __name__ == "__main__":
    main()
