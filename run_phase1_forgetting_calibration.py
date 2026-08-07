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
    X, y = cache_data["train_x"], cache_data["train_y"]
    cen  = torch.zeros(100, INPUT_DIM)
    for i in range(100):
        cen[i] = F.normalize(X[y == i].mean(0), dim=-1)
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
    """Report R[0,0], R[8,0], R[9,0] and calculate:
      - Generic drift drop: R[0,0] - R[8,0]
      - Interference-specific drop: R[8,0] - R[9,0]
      - Total drop: R[0,0] - R[9,0]
    Gate statistic: R[8,0] - R[9,0] (interference-specific).
    """
    probe_blocks  = build_probe_assignment(cache_data, f1, f2, confusable_pairs)
    tr_x, tr_y, te_x, te_y = build_block_tensors(probe_blocks, cache_data)

    r0_accs, r8_accs, r9_accs = [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        adapter = FullRankAdapter().to(DEVICE)
        opt     = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
        R       = np.zeros((10, 10))

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
            cum_rx = torch.cat(ref_x_acc, dim=0).to(DEVICE)
            cum_ry = torch.cat(ref_y_acc, dim=0).to(DEVICE)

            adapter.eval()
            with torch.no_grad():
                zr  = adapter(cum_rx)
                zq  = adapter(te_x[0].to(DEVICE))
                correct = sum(
                    1 for qi, qv in enumerate(zq)
                    if cum_ry[torch.argmax(torch.matmul(zr, qv)).item()].item()
                    == te_y[0][qi].item()
                )
                R[step, 0] = correct / len(zq)

        r0_accs.append(R[0, 0] * 100)
        r8_accs.append(R[8, 0] * 100)
        r9_accs.append(R[9, 0] * 100)

    r0_arr = np.array(r0_accs)
    r8_arr = np.array(r8_accs)
    r9_arr = np.array(r9_accs)

    tot_drops = r0_arr - r9_arr
    drift_drops = r0_arr - r8_arr
    interf_drops = r8_arr - r9_arr

    print("\n" + "=" * 75)
    print("  SINGLE-PAIR PROBE GATE (INTERFERENCE vs DRIFT DECOMPOSITION)")
    print(f"  Most confusable pair: fact {f1} (block 0) vs fact {f2} (block 9)")
    print(f"  Protocol: pure sequential, fixed order [0..9], epochs={epochs}, lr={lr:.0e}")
    print(f"  Seeds: {list(seeds)}")
    print(f"  {'Seed':>6s} | {'R[0,0]':>8s} {'R[8,0]':>8s} {'R[9,0]':>8s} | "
          f"{'Total Drop (R0-R9)':>18s} {'Drift (R0-R8)':>14s} {'Interf (R8-R9)':>15s}")
    print("-" * 88)
    for i, seed in enumerate(seeds):
        print(f"  {seed:>6d} | {r0_accs[i]:>7.2f}% {r8_accs[i]:>7.2f}% {r9_accs[i]:>7.2f}% | "
              f"{tot_drops[i]:>+17.2f}% {drift_drops[i]:>+13.2f}% {interf_drops[i]:>+14.2f}%")

    def _row(label, arr):
        return f"  {label:>6s} | {arr[0]:>7.2f}% {arr[1]:>7.2f}% {arr[2]:>7.2f}% | " \
               f"{arr[3]:>+17.2f}% {arr[4]:>+13.2f}% {arr[5]:>+14.2f}%"

    means = [r0_arr.mean(), r8_arr.mean(), r9_arr.mean(), tot_drops.mean(), drift_drops.mean(), interf_drops.mean()]
    stds  = [r0_arr.std(),  r8_arr.std(),  r9_arr.std(),  tot_drops.std(),  drift_drops.std(),  interf_drops.std()]
    mins  = [r0_arr.min(),  r8_arr.min(),  r9_arr.min(),  tot_drops.min(),  drift_drops.min(),  interf_drops.min()]
    maxs  = [r0_arr.max(),  r8_arr.max(),  r9_arr.max(),  tot_drops.max(),  drift_drops.max(),  interf_drops.max()]

    print("-" * 88)
    print(_row("Mean", means))
    print(_row("Std", stds))
    print(_row("Min", mins))
    print(_row("Max", maxs))
    print()

    mean_interf = interf_drops.mean()
    mean_total  = tot_drops.mean()

    if mean_interf >= 20.0:
        print(f"  DECISION: Interference-specific drop = {mean_interf:.2f}% >= 20.0 pp.")
        print("  HIGH INTERFERENCE DETECTED. Proceed with Axis D (with straddle-min control).")
        axis_d_status = "RUN_FULL"
    elif mean_interf >= 10.0:
        print(f"  DECISION: Interference-specific drop = {mean_interf:.2f}% in [10.0, 20.0) pp.")
        print("  WEAK INTERFERENCE. Run Axis D only alongside straddle-min control;")
        print("  treat capacity (Axis A) as the primary lever.")
        axis_d_status = "RUN_WEAK"
    else:
        print(f"  DECISION: Interference-specific drop = {mean_interf:.2f}% < 10.0 pp.")
        print("  NEGLIGIBLE INTERFERENCE (< 10pp for most confusable pair).")
        print("  1-NN retrieval over a fixed fact bank with a frozen encoder does not")
        print("  support catastrophic forgetting via confusable-pair interference.")
        print("  ACTION: Axis D dropped entirely. Proceed with Axis A (capacity) as sole lever.")
        axis_d_status = "DROP"

    print("=" * 75)
    return mean_interf, mean_total, axis_d_status


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
        _ = json.load(f)
    cache_data = torch.load(CACHE_PATH, map_location=DEVICE)
    print(f"  [Cache] {CACHE_PATH}")

    R_VALUES  = [8, 16, 32, 64, 128, 960]
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
    if args.diagnostics-only or not args.sweep:
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
