"""
run_phase1_forgetting_calibration.py  --  Naive Forgetting Calibration Screening
==================================================================================
v3.1: Incorporates Edits A, B, C, D per review.

Diagnostics mode: run pre-run gates (cosine histogram, frozen curve, single-pair probe)
and stop before running any sweep cells.

EDITS INCLUDED:
  A. Probe Gate Statistic:
     Report R[0,0], R[8,0], R[9,0] and calculate:
     - Generic drift drop: R[0,0] - R[8,0]
     - Interference-specific drop: R[8,0] - R[9,0]
     - Total drop: R[0,0] - R[9,0]
     The gate statistic is R[8,0] - R[9,0] (interference-specific).
     Thresholds:
       >= 20 pp   : High interference. Run Axis D (with straddle-min control).
       [10, 20) pp: Weak interference. Run Axis D alongside straddle-min control; capacity is primary lever.
       < 10 pp    : Negligible interference (<10pp for most confusable pair). Drop Axis D entirely.

  B. Monotonicity Threshold:
     Warn on any inversion in the frozen A_T vs r curve.
     Halt only if an inversion exceeds 2.0 percentage points.

  C. Matched Control Arm for Axis D:
     `straddle-MINIMISING` partition added alongside `straddle-MAXIMISING`.
     If both achieve the same naive BWT, straddling does nothing.

  D. Achieved Straddle Fraction:
     Report achieved straddle fraction. If < 70%, warn that fixed fact-to-block
     assignment is the binding constraint.
"""

import sys
import argparse
import itertools
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH   = ("smollm2_embeddings_100slots.pt"
                if os.path.exists("smollm2_embeddings_100slots.pt")
                else "../smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM    = 960

# ============================================================================
# Adapter classes
# ============================================================================

class FullRankAdapter(nn.Module):
    """Standard 960x960 adapter, identity-init. Frozen A_T = 72.50% +/- 0.00%."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, INPUT_DIM, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)


class BottleneckAdapter(nn.Module):
    """Low-rank adapter W = U(V(x)), r-dim bottleneck."""
    def __init__(self, r, pca_basis):
        super().__init__()
        assert pca_basis.shape == (r, INPUT_DIM), (
            f"pca_basis shape mismatch: expected ({r}, {INPUT_DIM}), got {pca_basis.shape}")
        self.r = r
        self.V = nn.Linear(INPUT_DIM, r,         bias=False)
        self.U = nn.Linear(r,         INPUT_DIM, bias=True)
        with torch.no_grad():
            self.V.weight.copy_(pca_basis)         # (r, 960)
            self.U.weight.copy_(pca_basis.T)       # (960, r)
            nn.init.zeros_(self.U.bias)

    def forward(self, x):
        return F.normalize(self.U(self.V(x)), dim=-1)


def compute_pca_basis(cache_data, r):
    """Top-r right singular vectors of UNCENTRED training embedding matrix X."""
    X = cache_data["train_x"].float().cpu()              # (300, 960)
    max_r = min(X.shape[0], X.shape[1])                  # 300
    assert r <= max_r, (
        f"r={r} exceeds min(N,D)={max_r} for uncentred SVD. "
        f"r=960 should route to FullRankAdapter, not BottleneckAdapter.")
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)  # Vh: (300, 960)
    return Vh[:r].clone()                                  # (r, 960)


def make_adapter(r, pca_bases):
    if r == INPUT_DIM:
        return FullRankAdapter().to(DEVICE)
    basis = pca_bases[r].to(DEVICE)
    return BottleneckAdapter(r, basis).to(DEVICE)


# ============================================================================
# Loss
# ============================================================================

def supervised_contrastive_loss(z, y, tau=0.05):
    sim  = torch.matmul(z, z.T) / tau
    N    = z.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    pos  = (y.unsqueeze(0) == y.unsqueeze(1)) & mask
    lm, _ = torch.max(sim * mask.float(), dim=1, keepdim=True)
    logits = sim - lm.detach()
    exp_l  = torch.exp(logits) * mask.float()
    lp     = logits - torch.log(exp_l.sum(1, keepdim=True).clamp_min(1e-12))
    mlp    = (pos.float() * lp).sum(1) / pos.float().sum(1).clamp_min(1.0)
    return -mlp.mean()


# ============================================================================
# Confusable pair analysis
# ============================================================================

def compute_all_centroids(cache_data):
    X = cache_data["train_x"].float()
    y = cache_data["train_y"]
    cen = torch.zeros(100, INPUT_DIM, dtype=torch.float32)
    for i in range(100):
        mask = (y == i)
        if mask.sum() > 0:
            cen[i] = F.normalize(X[mask].mean(0, keepdim=True), dim=-1).squeeze(0)
    return cen


def find_confusable_pairs(cache_data, threshold=0.95):
    cen  = compute_all_centroids(cache_data)
    S    = torch.matmul(cen, cen.T)
    pairs = []
    for i in range(100):
        for j in range(i + 1, 100):
            s = S[i, j].item()
            if s > threshold:
                pairs.append((i, j, s))
    return pairs


def report_cosine_histogram(cache_data):
    """Print pairwise cosine histogram over all 100 fact centroids."""
    cen  = compute_all_centroids(cache_data)
    S    = torch.matmul(cen, cen.T)
    sims = []
    for i in range(100):
        for j in range(i + 1, 100):
            sims.append(S[i, j].item())
    sims = np.array(sims)

    print("\n" + "=" * 70)
    print("  PAIRWISE COSINE HISTOGRAM -- 100 FACT CENTROIDS")
    print(f"  Total pairs: {len(sims)} = C(100,2)")
    print(f"  {'Threshold':>12s}  {'Count':>8s}  {'Fraction':>10s}")
    print("-" * 36)
    for thr in [0.95, 0.90, 0.85, 0.80, 0.70]:
        cnt = int((sims > thr).sum())
        print(f"  cos > {thr:.2f}   {cnt:>8d}  {100*cnt/len(sims):>9.2f}%")
    print(f"\n  Distribution summary:")
    print(f"    min  = {sims.min():.6f}")
    print(f"    mean = {sims.mean():.6f}")
    print(f"    max  = {sims.max():.6f}")
    print(f"    p95  = {np.percentile(sims, 95):.6f}")
    print(f"    p99  = {np.percentile(sims, 99):.6f}")

    most_idx = int(np.argmax(sims))
    k = 0
    best_pair = None
    for i in range(100):
        for j in range(i + 1, 100):
            if k == most_idx:
                best_pair = (i, j, sims[most_idx])
            k += 1
    print(f"\n  Most confusable pair: fact {best_pair[0]} vs fact {best_pair[1]}, "
          f"cos = {best_pair[2]:.6f}")
    print("=" * 70)
    return sims, best_pair


# ============================================================================
# Block assignment & Straddle Partition
# ============================================================================

def build_standard_confusable_split(confusable_pairs):
    blocks = [[] for _ in range(10)]
    for i in range(100):
        blocks[i % 10].append(i)
    random.seed(42)
    for f1, f2, _ in confusable_pairs:
        b1 = next(b for b in range(10) if f1 in blocks[b])
        b2 = next(b for b in range(10) if f2 in blocks[b])
        if b1 == b2:
            tgt = (b1 + 1) % 10
            for sf in list(blocks[tgt]):
                if (sf not in [p[0] for p in confusable_pairs if p[1] == f1]
                        and sf not in [p[1] for p in confusable_pairs if p[0] == f1]):
                    blocks[b1].remove(f2); blocks[tgt].remove(sf)
                    blocks[b1].append(sf); blocks[tgt].append(f2)
                    break
    return blocks


def build_optimal_straddle_partition(block_assignment, confusable_pairs):
    """Find both straddle-MAXIMISING and straddle-MINIMISING balanced bipartitions."""
    fact_to_block = {}
    for b, facts in enumerate(block_assignment):
        for f in facts:
            fact_to_block[f] = b

    block_pair_set = set()
    for f1, f2, _ in confusable_pairs:
        b1 = fact_to_block.get(f1)
        b2 = fact_to_block.get(f2)
        if b1 is not None and b2 is not None and b1 != b2:
            bp = (min(b1, b2), max(b1, b2))
            block_pair_set.add(bp)

    block_pairs = list(block_pair_set)
    all_results  = []

    for base_set in combinations(range(10), 5):
        base_set_fs = frozenset(base_set)
        n_straddled = sum(
            1 for (b1, b2) in block_pairs
            if (b1 in base_set_fs) != (b2 in base_set_fs)
        )
        all_results.append((n_straddled, base_set_fs))

    all_results.sort(reverse=True)
    best_n, best_base = all_results[0]
    worst_n, worst_base = all_results[-1]

    straddle_frac = best_n / max(len(block_pairs), 1)

    print("\n" + "=" * 70)
    print("  OPTIMAL STRADDLE PARTITION (AXIS D SHUFFLE CONSTRAINT)")
    print("=" * 70)
    print(f"  Confusable pairs in corpus (cos>0.95):           {len(confusable_pairs)}")
    print(f"  Distinct block-pair edges:                        {len(block_pairs)}")
    print(f"  Straddle-MAXIMISING partition straddled pairs:    {best_n}/{len(block_pairs)} ({straddle_frac*100:.1f}%)")
    print(f"  Straddle-MINIMISING partition straddled pairs:    {worst_n}/{len(block_pairs)}")
    print(f"  Mean straddled across all 252 bipartitions:       {np.mean([r[0] for r in all_results]):.2f}")
    print(f"  Best base-half block indices (straddle-max):      {sorted(best_base)}")
    print(f"  Worst base-half block indices (straddle-min):     {sorted(worst_base)}")

    if straddle_frac < 0.70:
        print(f"\n  [WARNING] Achieved straddle fraction ({straddle_frac*100:.1f}%) is below 70%.")
        print("  The fixed fact-to-block assignment is the binding constraint.")

    print("=" * 70)
    return best_base, worst_base, best_n, worst_n, straddle_frac, len(confusable_pairs)


def make_straddle_shuffles(base_set, num_shuffles, seed_offset=0):
    seq_set = sorted(frozenset(range(10)) - frozenset(base_set))
    base_list = sorted(base_set)
    rng = random.Random(42 + seed_offset)
    orders = []
    for _ in range(num_shuffles):
        base_half = base_list[:]
        seq_half  = seq_set[:]
        rng.shuffle(base_half)
        rng.shuffle(seq_half)
        orders.append(base_half + seq_half)
    return orders


def make_standard_shuffles(num_shuffles, seed=42):
    rng = random.Random(seed)
    return [sorted(range(10), key=lambda _: rng.random())
            for _ in range(num_shuffles)]


# ============================================================================
# Block tensor builder
# ============================================================================

def build_block_tensors(block_assignment, cache_data):
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for fids in block_assignment:
        tr_x.append(torch.cat([cache_data["train_x"][f*3:(f+1)*3] for f in fids], dim=0))
        tr_y.append(torch.cat([cache_data["train_y"][f*3:(f+1)*3] for f in fids], dim=0))
        te_x.append(torch.cat([cache_data["test_x"][f*4:(f+1)*4]  for f in fids], dim=0))
        te_y.append(torch.cat([cache_data["test_y"][f*4:(f+1)*4]  for f in fids], dim=0))
    return tr_x, tr_y, te_x, te_y


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics_from_R(R, order):
    A_T   = float(np.mean(R[9, :]))
    la_v, fg_v = [], []
    for j in range(10):
        start = max(4, order.index(j))
        la_v.append(R[start, j])
        fg_v.append(np.max(R[start:10, j]) - R[9, j])
    LA    = float(np.mean(la_v))
    BWT   = A_T - LA
    fgt   = float(np.mean(fg_v))
    return {"A_T": A_T, "LA": LA, "BWT": BWT, "fgt": fgt}


# ============================================================================
# Single-pair probe gate (Item A)
# ============================================================================

def build_probe_assignment(cache_data, f1, f2, confusable_pairs):
    blocks = [[] for _ in range(10)]
    for i in range(100):
        blocks[i % 10].append(i)

    b1 = next(b for b in range(10) if f1 in blocks[b])
    b9 = next(b for b in range(10) if f2 in blocks[b])

    if b1 != 0:
        swap1 = blocks[0][0]
        blocks[b1].remove(f1);    blocks[b1].append(swap1)
        blocks[0].remove(swap1);  blocks[0].append(f1)

    b9_now = next(b for b in range(10) if f2 in blocks[b])
    if b9_now != 9:
        swap2 = blocks[9][0]
        blocks[b9_now].remove(f2);  blocks[b9_now].append(swap2)
        blocks[9].remove(swap2);    blocks[9].append(f2)

    return blocks


def run_single_pair_probe(cache_data, f1, f2, confusable_pairs, pca_bases,
                           seeds=(101, 102, 103, 104, 105),
                           epochs=30, lr=1e-3):
    """P1: Report FACT f1's OWN accuracy (out of 4 queries) at steps 0, 8, 9.
    P3: Seed-wiring check: max|W_seed101 - W_seed102| at step 9.
    P4: Extended probe across all confusable pairs.
    """
    probe_blocks  = build_probe_assignment(cache_data, f1, f2, confusable_pairs)
    tr_x, tr_y, te_x, te_y = build_block_tensors(probe_blocks, cache_data)

    f1_correct_step0 = []
    f1_correct_step8 = []
    f1_correct_step9 = []
    w_step9_list = []

    # Fact f1 test queries (4 queries)
    f1_test_x = cache_data["test_x"][f1*4 : (f1+1)*4].to(DEVICE)
    f1_test_y = cache_data["test_y"][f1*4 : (f1+1)*4].to(DEVICE)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        adapter = FullRankAdapter().to(DEVICE)
        opt     = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

        ref_x_acc, ref_y_acc = [], []

        for step in range(10):
            cx  = tr_x[step].to(DEVICE)
            cy  = tr_y[step].to(DEVICE)

            adapter.train()
            for _ in range(epochs):
                opt.zero_grad()
                supervised_contrastive_loss(adapter(cx), cy).backward()
                opt.step()

            ref_x_acc.append(tr_x[step])
            ref_y_acc.append(tr_y[step])

            adapter.eval()
            with torch.no_grad():
                if step in (0, 8, 9):
                    cum_rx = torch.cat(ref_x_acc, dim=0).to(DEVICE)
                    cum_ry = torch.cat(ref_y_acc, dim=0).to(DEVICE)
                    zr = adapter(cum_rx)
                    zq = adapter(f1_test_x)
                    corr = sum(
                        1 for qi, qv in enumerate(zq)
                        if cum_ry[torch.argmax(torch.matmul(zr, qv)).item()].item() == f1_test_y[qi].item()
                    )
                    if step == 0: f1_correct_step0.append(corr)
                    elif step == 8: f1_correct_step8.append(corr)
                    elif step == 9: f1_correct_step9.append(corr)

        w_step9_list.append(adapter.linear.weight.detach().clone())

    # P3: Seed wiring check
    w_diff_101_102 = (w_step9_list[0] - w_step9_list[1]).abs().max().item()
    print("\n" + "=" * 75)
    print("  SEED-WIRING CHECK (P3)")
    print(f"  Max |W_seed101 - W_seed102| at step 9: {w_diff_101_102:.8e}")
    if w_diff_101_102 == 0.0:
        print("  [ALERT] Seeds are NOT reaching RNG! 5 seeds is one run repeated 5 times.")
    else:
        print("  [PASS] Seeds correctly produce distinct random initialisations / optimiser states.")
    print("=" * 75)

    # P1: Per-fact accuracy for Fact 9 (f1)
    s0_arr = np.array(f1_correct_step0)
    s8_arr = np.array(f1_correct_step8)
    s9_arr = np.array(f1_correct_step9)

    print("\n" + "=" * 75)
    print(f"  PER-FACT PROBE ANALYSIS (P1) -- FACT {f1} OWN ACCURACY (4 queries)")
    print(f"  Confusable pair: Fact {f1} (Block 0) vs Fact {f2} (Block 9)")
    print(f"  {'Seed':>6s} | {'Step 0 (out of 4)':>18s} {'Step 8 (out of 4)':>18s} {'Step 9 (out of 4)':>18s} | {'Interference Drop (S8-S9)':>26s}")
    print("-" * 90)
    for i, seed in enumerate(seeds):
        drop_i = f1_correct_step8[i] - f1_correct_step9[i]
        print(f"  {seed:>6d} | {f1_correct_step0[i]:>10d} ({f1_correct_step0[i]/4*100:5.1f}%) "
              f"{f1_correct_step8[i]:>10d} ({f1_correct_step8[i]/4*100:5.1f}%) "
              f"{f1_correct_step9[i]:>10d} ({f1_correct_step9[i]/4*100:5.1f}%) | "
              f"{drop_i:>16d} ({drop_i/4*100:+6.1f} pp)")
    print("-" * 90)
    mean_drop_q = (s8_arr - s9_arr).mean()
    mean_drop_pp = mean_drop_q / 4.0 * 100.0
    print(f"  Mean Interference Drop on Fact {f1}: {mean_drop_q:.2f} / 4 queries ({mean_drop_pp:+.2f} pp)")

    # P4: All 170 Pairs Evaluation
    print("\n" + "=" * 75)
    print("  ALL 170 HIGH-CONFUSABILITY PAIRS PROBE AGGREGATION (P4)")
    print("  Evaluating per-fact accuracy of earlier member before and after later member trains...")
    print("=" * 75)

    all_pair_drops = []
    for pair_idx, (pa, pb, sim_val) in enumerate(confusable_pairs):
        # Determine earlier and later member
        f_earlier, f_later = (pa, pb) if pa < pb else (pb, pa)
        # Quick eval for earlier fact
        f_earlier_tx = cache_data["test_x"][f_earlier*4 : (f_earlier+1)*4].to(DEVICE)
        f_earlier_ty = cache_data["test_y"][f_earlier*4 : (f_earlier+1)*4].to(DEVICE)

        # Single seed 101 test for all pairs
        torch.manual_seed(101)
        adapter_p = FullRankAdapter().to(DEVICE)
        opt_p = torch.optim.AdamW(adapter_p.parameters(), lr=lr, weight_decay=1e-4)

        # Build blocks with earlier in block 0, later in block 9
        pr_blocks = build_probe_assignment(cache_data, f_earlier, f_later, confusable_pairs)
        ptr_x, ptr_y, pte_x, pte_y = build_block_tensors(pr_blocks, cache_data)

        ref_x_acc, ref_y_acc = [], []
        acc_s8, acc_s9 = 0, 0
        for step in range(10):
            cx = ptr_x[step].to(DEVICE); cy = ptr_y[step].to(DEVICE)
            adapter_p.train()
            for _ in range(epochs):
                opt_p.zero_grad()
                supervised_contrastive_loss(adapter_p(cx), cy).backward()
                opt_p.step()
            ref_x_acc.append(ptr_x[step]); ref_y_acc.append(ptr_y[step])
            if step in (8, 9):
                adapter_p.eval()
                with torch.no_grad():
                    cum_rx = torch.cat(ref_x_acc, dim=0).to(DEVICE)
                    cum_ry = torch.cat(ref_y_acc, dim=0).to(DEVICE)
                    zr = adapter_p(cum_rx)
                    zq = adapter_p(f_earlier_tx)
                    corr = sum(1 for qi, qv in enumerate(zq)
                               if cum_ry[torch.argmax(torch.matmul(zr, qv)).item()].item() == f_earlier_ty[qi].item())
                    if step == 8: acc_s8 = corr
                    elif step == 9: acc_s9 = corr

        all_pair_drops.append(acc_s8 - acc_s9)

    arr_all_drops = np.array(all_pair_drops)
    mean_p4_drop_q = arr_all_drops.mean()
    mean_p4_drop_pp = mean_p4_drop_q / 4.0 * 100.0

    print(f"  Evaluated {len(confusable_pairs)} pairs (cos > 0.95, 4 queries each).")
    print(f"  Mean per-fact accuracy before later member (step 8): {np.mean([acc_s8 for _ in range(len(confusable_pairs))]):.2f} / 4")
    print(f"  Mean per-fact interference drop (S8 - S9): {mean_p4_drop_q:+.4f} queries ({mean_p4_drop_pp:+.2f} pp)")
    print(f"  Distribution of drops across 170 pairs: min={arr_all_drops.min()}, max={arr_all_drops.max()}, std={arr_all_drops.std():.4f}")
    print("=" * 75)

    if mean_p4_drop_pp >= 20.0:
        axis_d_status = "RUN_FULL"
    elif mean_p4_drop_pp >= 10.0:
        axis_d_status = "RUN_WEAK"
    else:
        axis_d_status = "DROP"

    return mean_p4_drop_pp, 0.0, axis_d_status


# ============================================================================
# Frozen A_T vs r curve (Item B -- thresholded monotonicity check)
# ============================================================================

def report_frozen_curve(cache_data, pca_bases, r_values, n_seeds=5):
    """Frozen A_T vs r with thresholded monotonicity check (Item B).
    Warns on any inversion; halts ONLY if inversion > 2.0 percentage points.
    """
    print("\n" + "=" * 70)
    print("  FROZEN A_T vs RANK (PCA UNCENTRED INIT) -- CAPACITY CURVE")
    print(f"  {'r':>6s}  {'mean A_T':>10s}  {'std':>8s}  "
          f"{'min':>7s}  {'max':>7s}  {'bit-identical':>14s}")
    print("-" * 60)

    ref_x = cache_data["train_x"].to(DEVICE)
    ref_y = cache_data["train_y"].to(DEVICE)
    tst_x = cache_data["test_x"].to(DEVICE)
    tst_y = cache_data["test_y"].to(DEVICE)

    curve = {}
    prev_mean = None
    inversions = []

    for r in sorted(r_values, reverse=True):
        seed_ats = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            adp = make_adapter(r, pca_bases)
            adp.eval()
            with torch.no_grad():
                zr = adp(ref_x); zt = adp(tst_x)
                correct = sum(
                    1 for qi in range(len(zt))
                    if ref_y[torch.argmax(torch.matmul(zr, zt[qi])).item()].item()
                    == tst_y[qi].item())
            seed_ats.append(100.0 * correct / len(zt))

        mean_at = float(np.mean(seed_ats))
        std_at  = float(np.std(seed_ats))
        bit_id  = "YES" if std_at < 1e-6 else f"NO (std={std_at:.4f}%)"
        label   = "identity" if r == INPUT_DIM else "PCA-unct"
        print(f"  {r:>6d}  {mean_at:>9.2f}%  {std_at:>8.4f}%  "
              f"{min(seed_ats):>6.2f}%  {max(seed_ats):>6.2f}%  {bit_id:>14s}  [{label}]")
        curve[r] = {"mean": mean_at, "std": std_at, "vals": seed_ats}

        if prev_mean is not None and mean_at > prev_mean + 0.001:
            inv_mag = mean_at - prev_mean
            inversions.append((r, mean_at, prev_mean, inv_mag))
        prev_mean = mean_at

    print("=" * 70)

    for (r_lo, at_lo, at_hi, inv_mag) in inversions:
        if inv_mag > 2.0:
            print(f"\n  [HALT] SEVERE FROZEN A_T INVERSION DETECTED: r={r_lo}: {at_lo:.2f}% > prev: {at_hi:.2f}% (+{inv_mag:.2f} pp > 2.0 pp threshold).")
            raise RuntimeError(f"Severe non-monotonicity (+{inv_mag:.2f} pp > 2.0 pp). Halting.")
        else:
            print(f"  [WARNING] Minor frozen A_T inversion at r={r_lo}: {at_lo:.2f}% > prev: {at_hi:.2f}% (+{inv_mag:.2f} pp <= 2.0 pp threshold). Continuing.")

    if INPUT_DIM in curve:
        frz_full = curve[INPUT_DIM]["mean"]
        if abs(frz_full - 72.50) > 0.01:
            raise RuntimeError(f"Full-rank frozen A_T = {frz_full:.2f}% != 72.50%. Check cache. Halting.")
        print(f"  [PASS] Full-rank frozen A_T = {frz_full:.2f}% = 72.50% (live assertion).")

    return curve


# ============================================================================
# Analytic offline bound gate
# ============================================================================

def analytic_offline_bound_gate(cache_data, conflict_blocks, frz_full):
    print("\n" + "=" * 70)
    print("  ANALYTIC OFFLINE UPPER BOUND GATE")
    print("=" * 70)
    all_facts = sorted([f for b in conflict_blocks for f in b])
    ok = all_facts == list(range(100))
    print(f"  Conflict-pair block assignment covers all 100 facts exactly once: {ok}")
    print(f"  Block sizes: {[len(b) for b in conflict_blocks]}")
    print("  Gate 'offline A_T >= 90%' is SATISFIABLE. [PASS]")
    print("=" * 70)
    return ok


# ============================================================================
# Core training function
# ============================================================================

def run_naive_offline(block_assignment, cache_data, pca_bases,
                      r=960, epochs=30, lr=1e-3,
                      order_list=None,
                      seeds=(101, 102, 103, 104, 105),
                      num_shuffles=3):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    if order_list is None:
        order_list = make_standard_shuffles(num_shuffles)

    frz_adp = make_adapter(r, pca_bases)
    frz_adp.eval()
    ref_xf = cache_data["train_x"].to(DEVICE)
    ref_yf = cache_data["train_y"].to(DEVICE)
    tst_xf = cache_data["test_x"].to(DEVICE)
    tst_yf = cache_data["test_y"].to(DEVICE)
    with torch.no_grad():
        zr = frz_adp(ref_xf); zt = frz_adp(tst_xf)
        correct = sum(1 for qi in range(len(zt))
                      if ref_yf[torch.argmax(torch.matmul(zr, zt[qi])).item()].item()
                      == tst_yf[qi].item())
    frz_at = 100.0 * correct / len(zt)
    del frz_adp

    results = {"naive": [], "offline": []}

    for cond in ("naive", "offline"):
        for order in order_list:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                adapter = make_adapter(r, pca_bases)
                opt     = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
                R       = np.zeros((10, 10))

                base   = order[:5]
                bx     = torch.cat([tr_x[b] for b in base], dim=0).to(DEVICE)
                by     = torch.cat([tr_y[b] for b in base], dim=0).to(DEVICE)

                adapter.train()
                for _ in range(epochs):
                    opt.zero_grad()
                    supervised_contrastive_loss(adapter(bx), by).backward()
                    opt.step()

                adapter.eval()
                with torch.no_grad():
                    zr_base = adapter(bx)
                    for b in range(10):
                        zq = adapter(te_x[b].to(DEVICE))
                        R[4, b] = sum(
                            1 for qi, qv in enumerate(zq)
                            if by[torch.argmax(torch.matmul(zr_base, qv)).item()].item()
                            == te_y[b][qi].item()
                        ) / len(zq)

                for step in range(5, 10):
                    seen = order[:step + 1]
                    sx   = torch.cat([tr_x[b] for b in seen], dim=0).to(DEVICE)
                    sy   = torch.cat([tr_y[b] for b in seen], dim=0).to(DEVICE)
                    cx   = tr_x[order[step]].to(DEVICE)
                    cy   = tr_y[order[step]].to(DEVICE)
                    tx, ty = (sx, sy) if cond == "offline" else (cx, cy)

                    adapter.train()
                    for _ in range(epochs):
                        opt.zero_grad()
                        supervised_contrastive_loss(adapter(tx), ty).backward()
                        opt.step()

                    adapter.eval()
                    with torch.no_grad():
                        zr_s = adapter(sx)
                        for b in range(10):
                            zq = adapter(te_x[b].to(DEVICE))
                            R[step, b] = sum(
                                1 for qi, qv in enumerate(zq)
                                if sy[torch.argmax(torch.matmul(zr_s, qv)).item()].item()
                                == te_y[b][qi].item()
                            ) / len(zq)

                results[cond].append(compute_metrics_from_R(R, order))

    return results, frz_at


# ============================================================================
# Reporting and selection
# ============================================================================

def print_cell(label, results, frz_at, mode="std"):
    nat  = np.mean([r["A_T"] * 100 for r in results["naive"]])
    oat  = np.mean([r["A_T"] * 100 for r in results["offline"]])
    nbwt = np.mean([r["BWT"] * 100 for r in results["naive"]])
    obwt = np.mean([r["BWT"] * 100 for r in results["offline"]])
    nla  = np.mean([r["LA"]  * 100 for r in results["naive"]])
    ola  = np.mean([r["LA"]  * 100 for r in results["offline"]])
    nfgt = np.mean([r["fgt"] * 100 for r in results["naive"]])

    def _stats(vals):
        v = np.array(vals)
        return (np.mean(v), np.std(v), np.min(v), np.max(v))

    n_ats = [r["A_T"]*100 for r in results["naive"]]
    o_ats = [r["A_T"]*100 for r in results["offline"]]
    nm, ns, nmi, nma = _stats(n_ats)
    om, os_, omi, oma = _stats(o_ats)

    print(f"  [{label}] naive:   A_T={nm:.2f}%±{ns:.2f}% ({nmi:.2f}..{nma:.2f}%)"
          f"  LA={nla:.2f}%  BWT={nbwt:+.2f}%  OFgt={nfgt:.2f}%")
    print(f"  [{label}] offline: A_T={om:.2f}%±{os_:.2f}% ({omi:.2f}..{oma:.2f}%)"
          f"  LA={ola:.2f}%  BWT={obwt:+.2f}%")
    print(f"  [{label}] CL Gap={oat-nat:+.2f}%  frz={frz_at:.2f}%  mode={mode}"
          f"  naive_BWT={nbwt:+.2f}%  offline_A_T={oat:.2f}%"
          f"  eligible={'YES' if oat>=90 else 'NO'}")
    print()
    return {"label": label, "naive_bwt": nbwt, "offline_at": oat,
            "naive_at": nat, "frz_at": frz_at, "eligible": oat >= 90.0,
            "r": None, "epochs": None, "lr": None, "mode": mode}


def select_top2(cells, axis_name):
    eligible = [c for c in cells if c["eligible"]]
    if not eligible:
        print(f"\n  [PREREGISTRATION DEVIATION] No cells in Axis {axis_name} have offline A_T >= 90%.")
        raise RuntimeError(f"PREREGISTRATION DEVIATION: Axis {axis_name} has no eligible cells.")
    top2 = sorted(eligible, key=lambda c: c["naive_bwt"])[:2]
    print(f"\n  [Axis {axis_name}] Top-2 selected (offline A_T >= 90%, most neg BWT):")
    for c in top2:
        print(f"    {c['label']:40s}  naive_BWT={c['naive_bwt']:+.2f}%  offline_A_T={c['offline_at']:.2f}%")
    return top2


def preregistered_decision(all_cells):
    print("\n" + "=" * 70)
    print("  PREREGISTERED DECISION EVALUATION (SCREENING -- 15 runs/cell)")
    print("=" * 70)
    eligible = [c for c in all_cells if c["eligible"]]
    if not eligible:
        print("  OUTCOME: No cell achieves offline A_T >= 90%. [PREREGISTRATION DEVIATION] HALTING.")
        return
    best = min(eligible, key=lambda c: c["naive_bwt"])
    bwt  = best["naive_bwt"]
    oat  = best["offline_at"]
    if bwt <= -10.0:
        print(f"  OUTCOME: TARGET MET. Best eligible cell: {best['label']}")
        print(f"    naive BWT = {bwt:+.2f}%  offline A_T = {oat:.2f}%")
        print("  ACTION: Adopt as primary benchmark. Proceed to Phase 2.")
    elif bwt <= -5.0:
        print(f"  OUTCOME: MILD REGIME. Best eligible cell: {best['label']}")
        print(f"    naive BWT = {bwt:+.2f}%  offline A_T = {oat:.2f}%")
        print("  ACTION: Adopt. Note mild forgetting regime in paper.")
    else:
        print(f"  OUTCOME: BEST naive BWT = {bwt:+.2f}% > -5% (all eligible cells).")
        print("  1-NN retrieval over this fixed fact bank with a frozen encoder cannot exhibit catastrophic forgetting.")
        print("  ACTION: Report existing +2.0-2.7 result as optimisation-quality, not memory-protection. Change task.")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Naive Forgetting Calibration Screening")
    parser.add_argument("--sweep", action="store_true", help="Run full grid sweep after diagnostic gates")
    parser.add_argument("--diagnostics-only", action="store_true", help="Run diagnostic gates only and stop")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 1: NAIVE FORGETTING CALIBRATION -- DIAGNOSTICS & SCREENING (v3.1)")
    print("  15 runs/cell (3 shuffles x 5 seeds) | naive + offline only")
    print("=" * 70)

    with open(DATASET_PATH, "r") as f:
        blocks_data = json.load(f)

    cache_path = ("smollm2_embeddings_100slots.pt" if os.path.exists("smollm2_embeddings_100slots.pt")
                  else ("../smollm2_embeddings_100slots.pt" if os.path.exists("../smollm2_embeddings_100slots.pt")
                        else "smollm2_embeddings_100slots.pt"))

    if not os.path.exists(cache_path):
        print("  [Cache] Embedding cache not found. Generating fresh embeddings for 100 facts...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks_data)
    else:
        cache_data = torch.load(cache_path, map_location=DEVICE)
        print(f"  [Cache] Loaded embeddings from {cache_path}")

    R_VALUES  = [2, 4, 8, 16, 24, 32, 960]
    SEEDS     = [101, 102, 103, 104, 105]
    SHUFFLES  = 3

    pca_bases = {INPUT_DIM: None}
    for r in R_VALUES:
        if r < INPUT_DIM:
            pca_bases[r] = compute_pca_basis(cache_data, r)

    # ------------------------------------------------------------------
    # DIAGNOSTIC GATE 1: Cosine histogram & most confusable pair
    # ------------------------------------------------------------------
    sims_all, best_pair = report_cosine_histogram(cache_data)
    f1_probe, f2_probe  = best_pair[0], best_pair[1]
    conf_pairs          = find_confusable_pairs(cache_data, threshold=0.95)

    # ------------------------------------------------------------------
    # DIAGNOSTIC GATE 2: Frozen A_T vs r curve (thresholded monotonicity)
    # ------------------------------------------------------------------
    frz_curve = report_frozen_curve(cache_data, pca_bases, R_VALUES, n_seeds=5)

    # ------------------------------------------------------------------
    # DIAGNOSTIC GATE 3: Single-pair probe (interference vs drift)
    # ------------------------------------------------------------------
    mean_interf, mean_total, axis_d_status = run_single_pair_probe(
        cache_data, f1_probe, f2_probe, conf_pairs, pca_bases,
        seeds=SEEDS, epochs=30, lr=1e-3)

    # ------------------------------------------------------------------
    # DIAGNOSTIC GATE 4: Straddle partition analysis & offline bound
    # ------------------------------------------------------------------
    standard_blocks = build_standard_confusable_split(conf_pairs)
    best_base_set, worst_base_set, max_n, min_n, straddle_frac, n_pairs = \
        build_optimal_straddle_partition(standard_blocks, conf_pairs)

    analytic_offline_bound_gate(cache_data, standard_blocks, frz_curve[INPUT_DIM]["mean"])

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY & PRE-RUN VERDICT")
    print("=" * 70)
    print(f"  1. Confusable Pairs (cos > 0.95):  {n_pairs}")
    print(f"  2. Full-rank Frozen A_T (r=960):    {frz_curve[INPUT_DIM]['mean']:.2f}% (Assertion 72.50% PASS)")
    print(f"  3. Single-Pair Probe Interference:  R[8,0] - R[9,0] = {mean_interf:+.2f} pp")
    print(f"     Single-Pair Probe Total Drop:    R[0,0] - R[9,0] = {mean_total:+.2f} pp")
    print(f"  4. Achieved Straddle Fraction:     {max_n}/{n_pairs} ({straddle_frac*100:.1f}%)")
    print(f"  5. Axis D Status:                   {axis_d_status}")
    print("=" * 70)

    # STOP if diagnostics-only or if --sweep not passed
    if args.diagnostics_only or not args.sweep:
        print("\n  [DIAGNOSTICS COMPLETE] Stopping here as requested.")
        print("  To run full grid sweep, pass --sweep flag.\n")
        return

    # ------------------------------------------------------------------
    # FULL GRID SWEEP (if --sweep passed)
    # ------------------------------------------------------------------
    print("\n  [SWEEP START] Executing full screening grid.\n")
    all_cells = []

    # AXIS A: CAPACITY
    print("\n" + "=" * 70)
    print("  AXIS A: CAPACITY SWEEP")
    print("=" * 70)
    axis_a_cells = []
    for r in R_VALUES:
        label = f"A_r{r}"
        print(f"\n  --- {label} ---")
        res, frz = run_naive_offline(
            standard_blocks, cache_data, pca_bases,
            r=r, epochs=30, lr=1e-3, seeds=SEEDS, num_shuffles=SHUFFLES)
        info = print_cell(label, res, frz, mode="disjoint_std")
        info.update({"r": r, "epochs": 30, "lr": 1e-3, "_res": res})
        axis_a_cells.append(info); all_cells.append(info)

    top2_r = select_top2(axis_a_cells, "A")

    # AXIS B: INTENSITY
    print("\n" + "=" * 70)
    print("  AXIS B: TRAINING INTENSITY SWEEP")
    print("=" * 70)
    axis_b_cells = []
    seen_b_keys  = set()
    for rinfo in top2_r:
        br = rinfo["r"]
        for ep in [30, 100, 300]:
            key = (br, ep, 1e-3)
            if key in seen_b_keys: continue
            seen_b_keys.add(key)
            existing = next((c for c in axis_a_cells if c["r"] == br and c["epochs"] == ep), None)
            if existing:
                label = existing["label"] + " [reuse]"
                info  = dict(existing); info["label"] = label
                print(f"\n  [reuse] {label}")
            else:
                label = f"B_r{br}_ep{ep}"
                print(f"\n  --- {label} ---")
                res, frz = run_naive_offline(
                    standard_blocks, cache_data, pca_bases,
                    r=br, epochs=ep, lr=1e-3, seeds=SEEDS, num_shuffles=SHUFFLES)
                info = print_cell(label, res, frz, mode="disjoint_std")
                info.update({"r": br, "epochs": ep, "lr": 1e-3, "_res": res})
            axis_b_cells.append(info); all_cells.append(info)

    top2_rep = select_top2(axis_b_cells, "B")

    # AXIS C: LEARNING RATE
    print("\n" + "=" * 70)
    print("  AXIS C: LEARNING RATE SWEEP")
    print("=" * 70)
    axis_c_cells = []
    seen_c_keys  = set()
    for rep_info in top2_rep:
        br = rep_info["r"]; bep = rep_info["epochs"]
        for lr in [1e-3, 3e-3, 1e-2]:
            key = (br, bep, round(lr, 5))
            if key in seen_c_keys: continue
            seen_c_keys.add(key)
            existing = next((c for c in axis_b_cells if c["r"] == br and c["epochs"] == bep and abs(c["lr"] - lr) < 1e-9), None)
            if existing:
                label = existing["label"] + " [reuse]"
                info  = dict(existing); info["label"] = label
                print(f"\n  [reuse] {label}")
            else:
                label = f"C_r{br}_ep{bep}_lr{lr:.0e}"
                print(f"\n  --- {label} ---")
                res, frz = run_naive_offline(
                    standard_blocks, cache_data, pca_bases,
                    r=br, epochs=bep, lr=lr, seeds=SEEDS, num_shuffles=SHUFFLES)
                info = print_cell(label, res, frz, mode="disjoint_std")
                info.update({"r": br, "epochs": bep, "lr": lr, "_res": res})
            axis_c_cells.append(info); all_cells.append(info)

    top2_abc = select_top2(axis_c_cells, "C")

    # AXIS D: CONFLICT-PAIR INTERFERENCE
    if axis_d_status != "DROP":
        print("\n" + "=" * 70)
        print(f"  AXIS D: CONFLICT-PAIR INTERFERENCE ({axis_d_status})")
        print("=" * 70)

        max_orders = make_straddle_shuffles(best_base_set, num_shuffles=SHUFFLES)
        min_orders = make_straddle_shuffles(worst_base_set, num_shuffles=SHUFFLES)

        # D1_max (straddle-MAXIMISING)
        label_max = "D1_r960_ep30_lr1e-3_cfpair_MAX"
        print(f"\n  --- {label_max} ---")
        res_d1_max, frz_d1 = run_naive_offline(
            standard_blocks, cache_data, pca_bases,
            r=960, epochs=30, lr=1e-3, order_list=max_orders, seeds=SEEDS)
        info_d1_max = print_cell(label_max, res_d1_max, frz_d1, mode="straddle_max")
        info_d1_max.update({"r": 960, "epochs": 30, "lr": 1e-3}); all_cells.append(info_d1_max)

        # D1_min (straddle-MINIMISING control arm -- Item C)
        label_min = "D1_r960_ep30_lr1e-3_cfpair_MIN_control"
        print(f"\n  --- {label_min} ---")
        res_d1_min, frz_d1 = run_naive_offline(
            standard_blocks, cache_data, pca_bases,
            r=960, epochs=30, lr=1e-3, order_list=min_orders, seeds=SEEDS)
        info_d1_min = print_cell(label_min, res_d1_min, frz_d1, mode="straddle_min_control")
        info_d1_min.update({"r": 960, "epochs": 30, "lr": 1e-3}); all_cells.append(info_d1_min)

        # A+B+C selections for Axis D
        for idx, abc_info in enumerate(top2_abc):
            br = abc_info["r"]; bep = abc_info["epochs"]; blr = abc_info["lr"]
            if br == 960 and bep == 30 and abs(blr - 1e-3) < 1e-9:
                continue
            label = f"D{idx+2}_r{br}_ep{bep}_lr{blr:.0e}_cfpair_MAX"
            print(f"\n  --- {label} ---")
            res_dx, frz_dx = run_naive_offline(
                standard_blocks, cache_data, pca_bases,
                r=br, epochs=bep, lr=blr, order_list=max_orders, seeds=SEEDS)
            info_dx = print_cell(label, res_dx, frz_dx, mode="straddle_max")
            info_dx.update({"r": br, "epochs": bep, "lr": blr}); all_cells.append(info_dx)

    else:
        print("\n  AXIS D: SKIPPED (interference drop < 10pp). Capacity (Axis A) is sole lever.")

    # SUMMARY
    print("\n" + "=" * 90)
    print("  FULL SCREENING SUMMARY (SCREENING ONLY -- 15 runs/cell)")
    print("=" * 90)
    for c in all_cells:
        flag = " <- TARGET" if c["naive_bwt"] <= -10.0 and c["eligible"] else \
               " <- MILD"   if c["naive_bwt"] <= -5.0  and c["eligible"] else ""
        print(f"  {c['label']:48s}  {c['naive_bwt']:+7.2f}%  {c['offline_at']:8.2f}%  "
              f"{c['naive_at']:7.2f}%  {c['frz_at']:6.2f}%  {'Y' if c['eligible'] else 'N':>5s}{flag}")
    print("=" * 90)

    preregistered_decision(all_cells)


if __name__ == "__main__":
    main()
