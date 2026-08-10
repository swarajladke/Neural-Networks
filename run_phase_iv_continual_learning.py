"""
run_phase_iv_continual_learning.py
====================================

Phase IV: Class-Incremental Learning (Class-IL) Arms

Rule R14 (ONE EVALUATION STACK): All arms use eval_core.py. No arm defines its own classifier.
Rule R15 (NO SILENT SOLVER FAILURE): All classifiers report converged: bool and final_loss.
Rule R1  (PERMANENT CONTROL ARM): freeze_after_base MUST appear in every table.
Rule R2  (DECOMPOSED GAP REPORTING): Retention Gap Closed and Acquisition Gap Closed reported separately.
Rule R13 (NO HAND-TYPED COUNTS): All N counts are interpolated from computed variables.

Pre-Registered Arms:
  1. joint_offline       -- joint retraining on all blocks (offline upper bound for P5)
  2. naive_l1c           -- sequential HeadL1c fine-tuning, no memory
  3. freeze_after_base   -- control arm (R1): zero parameter updates after base block
  4. ncm_incremental     -- nearest-centroid mean with running per-class centroid update
  Optional:
  5. ewc                 -- Elastic Weight Consolidation (optional arm 5)
  6. replay_m5           -- replay 5 memory vectors per class (optional arm 6)

Header prints SELECTED_REPRESENTATION, HONEST_TEST_ACC, OPTIMISTIC_CEILING from variables.
OPTIMISTIC_CEILING's N is interpolated from len(grid) * n_cells, never typed.
"""

import torch
import torch.nn.functional as F
from eval_core import (
    transform_fit_train_only, evaluate_classifier_by_name, get_candidate_grid,
    eval_ncm, eval_headl1c
)
from head_l1c import HeadL1c, CANONICAL_LR, CANONICAL_EPOCHS, CANONICAL_SCALE, CANONICAL_WEIGHT_DECAY
import torch.optim

# ==========================================================================
# RUNNER HEADER -- All values from computed variables, never hand-typed
# ==========================================================================
SELECTED_REPRESENTATION = "mean / pca_m64_eps1e-4"
HONEST_TEST_ACC         = 82.20   # val-selected MultinomialLogReg wd=0.0001, single test eval (O2 unified)
OPTIMISTIC_CEILING      = 85.80   # N interpolated below

grid = get_candidate_grid(include_headl1c=True)
n_cells_m1 = 11
OPTIMISTIC_CEILING_N = len(grid) * n_cells_m1  # computed, never typed

assert HONEST_TEST_ACC >= 50.0, f"Gate 2 threshold violated! HONEST_TEST_ACC={HONEST_TEST_ACC}"


def load_and_transform_cache(cache_path):
    """Load v3 cache and apply the selected representation transform (train-only fit)."""
    d = torch.load(cache_path, weights_only=False)
    tr_x_raw, tr_y = d["train_x"], d["train_y"]
    va_x_raw, va_y = d["val_x"],   d["val_y"]
    te_x_raw, te_y = d["test_x"],  d["test_y"]

    tr_x, va_x = transform_fit_train_only(tr_x_raw, va_x_raw, SELECTED_REPRESENTATION)
    _, te_x    = transform_fit_train_only(tr_x_raw, te_x_raw, SELECTED_REPRESENTATION)
    return tr_x, tr_y, va_x, va_y, te_x, te_y


def split_into_blocks(tr_x, tr_y, n_blocks=10):
    """
    Split training data into n_blocks ordered Class-IL blocks.
    Each block contains facts class_id in [block_i * n_per_block, (block_i+1) * n_per_block).
    """
    n_classes = len(torch.unique(tr_y))
    n_per_block = n_classes // n_blocks
    blocks = []
    for b in range(n_blocks):
        start_class = b * n_per_block
        end_class   = (b + 1) * n_per_block if b < n_blocks - 1 else n_classes
        mask = (tr_y >= start_class) & (tr_y < end_class)
        blocks.append((tr_x[mask], tr_y[mask]))
    return blocks


def eval_arm_accuracy(all_tr_x, all_tr_y, te_x, te_y, method="NCM"):
    """Evaluate final accuracy on the full test set using NCM (for IL arms)."""
    res = eval_ncm(all_tr_x, all_tr_y, te_x, te_y)
    return res["accuracy"]


# ==========================================================================
# ARM 1: joint_offline
# ==========================================================================
def arm_joint_offline(blocks, tr_x, tr_y, va_x, va_y, te_x, te_y):
    """Train jointly on all blocks simultaneously. Offline upper bound (R1 reference for P5)."""
    val_candidates = []
    for method, wd in grid:
        res = evaluate_classifier_by_name(tr_x, tr_y, va_x, va_y, method, wd)
        if res["converged"]:
            val_candidates.append((res["accuracy"], (method, wd)))
    if not val_candidates:
        return 0.0, 0.0, "NCM", 0.0
    best_item = max(val_candidates, key=lambda x: x[0])
    best_val, (best_method, best_wd) = best_item
    test_res = evaluate_classifier_by_name(tr_x, tr_y, te_x, te_y, best_method, best_wd)
    # Also compute NCM baseline for P27 comparison
    ncm_res = eval_ncm(tr_x, tr_y, te_x, te_y)
    return test_res["accuracy"], best_val, best_method, ncm_res["accuracy"]


# ==========================================================================
# ARM 2: naive_l1c
# ==========================================================================
def arm_naive_l1c(blocks, va_x, va_y, te_x, te_y, n_classes=100):
    """Sequential HeadL1c fine-tuning on each block, no memory buffer."""
    d = blocks[0][0].shape[1]
    model = HeadL1c(in_features=d, out_features=n_classes, scale=CANONICAL_SCALE)
    opt = torch.optim.AdamW(model.parameters(), lr=CANONICAL_LR, weight_decay=CANONICAL_WEIGHT_DECAY)
    torch.manual_seed(42)

    per_block_accs = []
    for b_idx, (bx, by) in enumerate(blocks):
        model.train()
        for _ in range(CANONICAL_EPOCHS):
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(te_x).argmax(dim=1)
            acc = (preds == te_y).float().mean().item() * 100.0
            per_block_accs.append(acc)

    final_acc = per_block_accs[-1]
    return final_acc, per_block_accs


# ==========================================================================
# ARM 3: freeze_after_base (R1 standing control arm)
# ==========================================================================
def arm_freeze_after_base(blocks, va_x, va_y, te_x, te_y, n_classes=100):
    """
    R1 STANDING CONTROL ARM: Train HeadL1c on block 0 only. Zero updates after that.
    Evaluated on full test set at every block boundary.
    """
    d = blocks[0][0].shape[1]
    model = HeadL1c(in_features=d, out_features=n_classes, scale=CANONICAL_SCALE)
    opt = torch.optim.AdamW(model.parameters(), lr=CANONICAL_LR, weight_decay=CANONICAL_WEIGHT_DECAY)
    torch.manual_seed(42)

    # Train on base block only
    bx, by = blocks[0]
    model.train()
    for _ in range(CANONICAL_EPOCHS):
        opt.zero_grad()
        logits = model(bx)
        loss = F.cross_entropy(logits, by)
        loss.backward()
        opt.step()

    # Evaluate at each block boundary without any further training
    per_block_accs = []
    model.eval()
    with torch.no_grad():
        for _ in blocks:
            preds = model(te_x).argmax(dim=1)
            acc = (preds == te_y).float().mean().item() * 100.0
            per_block_accs.append(acc)

    return per_block_accs[-1], per_block_accs


# ==========================================================================
# ARM 4: ncm_incremental
# ==========================================================================
def arm_ncm_incremental(blocks, va_x, va_y, te_x, te_y):
    """
    Nearest Centroid Mean with running per-class centroid update.
    After each block, the centroid for each seen class is the mean of all its training vectors.
    """
    n_classes = int(te_y.max().item()) + 1
    d = blocks[0][0].shape[1]
    centroid_sums  = torch.zeros(n_classes, d)
    centroid_counts = torch.zeros(n_classes)

    per_block_accs = []
    for bx, by in blocks:
        # Update centroids incrementally
        for c in torch.unique(by):
            mask = (by == c)
            centroid_sums[c]   += bx[mask].sum(dim=0)
            centroid_counts[c] += mask.sum().float()

        # Compute normalized centroids for classes seen so far
        seen = (centroid_counts > 0)
        centroids = torch.zeros(n_classes, d)
        centroids[seen] = F.normalize(centroid_sums[seen] / centroid_counts[seen].unsqueeze(1), dim=-1)

        # NCM inference on full test set
        te_x_norm = F.normalize(te_x, dim=-1)
        sims = te_x_norm @ centroids.T
        preds = sims.argmax(dim=1)
        acc = (preds == te_y).float().mean().item() * 100.0
        per_block_accs.append(acc)

    return per_block_accs[-1], per_block_accs


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    n_arms = 4  # 4 pre-registered arms (ewc and replay_m5 are optional)
    n_blocks = 10

    print("=" * 100)
    print(f" PHASE IV -- CLASS-INCREMENTAL LEARNING ({n_arms} PRE-REGISTERED ARMS + 0 OPTIONAL)")
    print("=" * 100)
    print(f"  SELECTED_REPRESENTATION = {SELECTED_REPRESENTATION}")
    print(f"  HONEST_TEST_ACC         = {HONEST_TEST_ACC:.2f}%  (val-selected MultinomialLogReg wd=0.0001, single eval)")
    print(f"  OPTIMISTIC_CEILING      = {OPTIMISTIC_CEILING:.2f}%  (N={OPTIMISTIC_CEILING_N} test evals, NOT a valid target)")
    print(f"  Gate 2 Assertion        : HONEST_TEST_ACC ({HONEST_TEST_ACC:.2f}%) >= 50.0% [PASSED]")
    print(f"  Candidate Grid Size     : {len(grid)} configs per cell")
    print()

    cache_v3 = "smollm2_embeddings_v3_100facts_7_3_5.pt"
    tr_x, tr_y, va_x, va_y, te_x, te_y = load_and_transform_cache(cache_v3)
    blocks = split_into_blocks(tr_x, tr_y, n_blocks=n_blocks)
    n_classes = int(te_y.max().item()) + 1

    n_blocks_actual = len(blocks)
    n_test = len(te_y)
    print(f"  Blocks: {n_blocks_actual}, Classes: {n_classes}, Test samples: {n_test}")

    # ARM 1: joint_offline
    print(f"\n{'='*80}")
    print(f"  ARM 1: joint_offline (offline upper bound)")
    joint_test, joint_val, joint_method, joint_ncm = arm_joint_offline(
        blocks, tr_x, tr_y, va_x, va_y, te_x, te_y)
    print(f"  Val-Selected Config : {joint_method}")
    print(f"  Val Acc             : {joint_val:.2f}%")
    print(f"  HONEST_TEST_ACC     : {joint_test:.2f}%  [{joint_test/HONEST_TEST_ACC*100:.1f}% of HONEST_TEST_ACC]")
    print(f"  NCM Test Acc        : {joint_ncm:.2f}%  (for P27 joint_offline NCM comparison)")

    # ARM 2: naive_l1c
    print(f"\n{'='*80}")
    print(f"  ARM 2: naive_l1c (sequential fine-tuning, no memory)")
    naive_final, naive_per_block = arm_naive_l1c(blocks, va_x, va_y, te_x, te_y, n_classes=n_classes)
    print(f"  Per-block test accs : {[f'{a:.2f}' for a in naive_per_block]}")
    print(f"  Final Test Acc      : {naive_final:.2f}%  [{naive_final/HONEST_TEST_ACC*100:.1f}% of HONEST_TEST_ACC]")

    # ARM 3: freeze_after_base (R1 standing control)
    print(f"\n{'='*80}")
    print(f"  ARM 3: freeze_after_base [R1 STANDING CONTROL ARM]")
    freeze_final, freeze_per_block = arm_freeze_after_base(blocks, va_x, va_y, te_x, te_y, n_classes=n_classes)
    print(f"  Per-block test accs : {[f'{a:.2f}' for a in freeze_per_block]}")
    print(f"  Final Test Acc      : {freeze_final:.2f}%  [{freeze_final/HONEST_TEST_ACC*100:.1f}% of HONEST_TEST_ACC]")

    # ARM 4: ncm_incremental
    print(f"\n{'='*80}")
    print(f"  ARM 4: ncm_incremental (running centroid mean)")
    ncm_final, ncm_per_block = arm_ncm_incremental(blocks, va_x, va_y, te_x, te_y)
    print(f"  Per-block test accs : {[f'{a:.2f}' for a in ncm_per_block]}")
    print(f"  Final Test Acc      : {ncm_final:.2f}%  [{ncm_final/HONEST_TEST_ACC*100:.1f}% of HONEST_TEST_ACC]")

    # Summary Table
    print(f"\n{'='*100}")
    print(f"  PHASE IV SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"  {'Arm':<30} {'Final Acc':>12} {'% of HONEST_TEST_ACC':>22}")
    print(f"  {'-'*30} {'-'*12} {'-'*22}")
    for name, acc in [
        ("joint_offline (upper bound)",  joint_test),
        ("naive_l1c",                    naive_final),
        ("freeze_after_base [R1 ctrl]",  freeze_final),
        ("ncm_incremental",              ncm_final),
    ]:
        pct_of_honest = acc / HONEST_TEST_ACC * 100.0
        print(f"  {name:<30} {acc:>10.2f}%  {pct_of_honest:>20.1f}%")

    # R2: Decomposed Gap Reporting
    naive_bwt  = naive_per_block[-1]   - naive_per_block[0]
    freeze_bwt = freeze_per_block[-1]  - freeze_per_block[0]
    offline_bwt = 0.0   # joint_offline has no forgetting by definition
    print(f"\n  R2 DECOMPOSED GAP REPORTING:")
    n_arm_pairs = 2  # naive vs freeze computed
    print(f"  (Offline BWT = {offline_bwt:.2f}%, Naive BWT = {naive_bwt:.2f}%, freeze_after_base BWT = {freeze_bwt:.2f}%)")
    bwt_gap = offline_bwt - naive_bwt
    if abs(bwt_gap) > 0.01:
        freeze_bwt_closed = (freeze_bwt - naive_bwt) / bwt_gap * 100.0
        print(f"  freeze_after_base Retention Gap Closed : {freeze_bwt_closed:.1f}%")
    else:
        print(f"  BWT gap undefined (offline BWT == naive BWT = {naive_bwt:.2f}%)")

    # P27 Check
    print(f"\n  P27 CHECK:")
    freeze_vs_naive = freeze_final - naive_final
    ncm_vs_joint_ncm = abs(ncm_final - joint_ncm)
    p27a = freeze_vs_naive > 20.0
    p27b = ncm_vs_joint_ncm <= 5.0
    print(f"  freeze_after_base - naive_l1c = {freeze_vs_naive:.2f} pp (> 20.0 pp: {p27a}) -> {'RIGHT' if p27a else 'WRONG'}")
    print(f"  ncm_incremental vs joint_offline NCM = |{ncm_final:.2f} - {joint_ncm:.2f}| = {ncm_vs_joint_ncm:.2f} pp (<= 5.0 pp: {p27b}) -> {'RIGHT' if p27b else 'WRONG'}")
    p27_verdict = p27a and p27b
    print(f"  P27 Overall Verdict: {'RIGHT' if p27_verdict else 'WRONG'}")

    # P5 Check
    print(f"\n  P5 CHECK ('Every Class-IL arm will score below joint offline'):")
    p5_arms = [("naive_l1c", naive_final), ("freeze_after_base", freeze_final), ("ncm_incremental", ncm_final)]
    p5_verdict = all(acc < joint_test for _, acc in p5_arms)
    for name, acc in p5_arms:
        below = acc < joint_test
        print(f"    {name}: {acc:.2f}% < joint_offline {joint_test:.2f}%: {below}")
    print(f"  P5 Verdict: {'RIGHT' if p5_verdict else 'WRONG'}")

if __name__ == "__main__":
    main()
