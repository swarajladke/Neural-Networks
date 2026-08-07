"""
run_phase2_forgetting_master_suite.py  --  Phase 2 Master Suite on Calibrated Forgetting Benchmark
==================================================================================================

Calibrated Configuration adopted from Phase 1 Screening (Cell C_r32_ep100_lr1e-02):
  - Adapter: BottleneckAdapter r=32 (uncentred PCA init)
  - Epochs: 100 per block
  - Learning rate: lr = 1e-2 (AdamW, weight_decay=1e-4)
  - Task: Disjoint label sequential fine-tuning (10 blocks, 100 facts)

Calibrated Baseline Metrics (15-run screen):
  - Naive BWT: -12.50%  (Severe catastrophic forgetting, target <= -10pp met)
  - Offline A_T: 95.17%  (Offline ceiling >= 90% gate met)
  - Total CL Gap: +20.08 pp

Phase 2 Core Objective:
  Evaluate whether OGP's retention share (delta_BWT / delta_A_T) RISES under real catastrophic forgetting
  (BWT = -12.50%) compared to the near-zero-forgetting baseline (where BWT was -1.05%).

Swept Conditions (15 conditions x 50 runs per seed set x 2 seed sets = 1500 runs total):
  - Baselines: naive, offline
  - OGP rank sweep: OGP_k2, OGP_k4, OGP_k8, OGP_k12, OGP_k16, OGP_k24, OGP_k32
    (testing the prediction that at r=32, optimal k falls to ~12-16)
  - Control suite: RANDOM-32, BOTTOM-32, CURRENT-32

Seed Sets:
  - Selection Seeds: {101, 102, 103, 104, 105} (10 shuffles x 5 seeds = 50 runs/cell)
  - Fresh Replication Seeds: {211, 212, 213, 214, 215} (10 shuffles x 5 seeds = 50 runs/cell)

Reporting:
  - Exact identity decomposition: delta_A_T = delta_LA + delta_BWT
  - Retention Share = delta_BWT / delta_A_T
  - Acquisition Share = delta_LA / delta_A_T
  - Paired 95% CIs via 10,000-sample bootstrap on within-run differences vs naive.
"""

import os
import sys
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH   = ("smollm2_embeddings_100slots.pt"
                if os.path.exists("smollm2_embeddings_100slots.pt")
                else "../smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM    = 960

# ---------------------------------------------------------------------------
# Adapter Definition
# ---------------------------------------------------------------------------

class BottleneckAdapter(nn.Module):
    """r-dim bottleneck adapter: W = U @ V, V in R^{r x 960}, U in R^{960 x r}."""
    def __init__(self, r, pca_basis):
        super().__init__()
        assert pca_basis.shape == (r, INPUT_DIM)
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
    X = cache_data["train_x"].float().cpu()
    max_r = min(X.shape[0], X.shape[1])
    assert r <= max_r
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:r].clone()


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


# ---------------------------------------------------------------------------
# Data & Partition Helpers
# ---------------------------------------------------------------------------

def find_confusable_pairs(cache_data, threshold=0.95):
    X = cache_data["train_x"].float()
    y = cache_data["train_y"]
    cen = torch.zeros(100, INPUT_DIM, dtype=torch.float32)
    for i in range(100):
        mask = (y == i)
        if mask.sum() > 0:
            cen[i] = F.normalize(X[mask].mean(0, keepdim=True), dim=-1).squeeze(0)
    S = torch.matmul(cen, cen.T)
    pairs = []
    for i in range(100):
        for j in range(i + 1, 100):
            if S[i, j].item() > threshold:
                pairs.append((i, j, S[i, j].item()))
    return pairs


def build_confusable_split_blocks(confusable_pairs):
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


def build_block_tensors(block_assignment, cache_data):
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for fids in block_assignment:
        tr_x.append(torch.cat([cache_data["train_x"][f*3:(f+1)*3] for f in fids], dim=0))
        tr_y.append(torch.cat([cache_data["train_y"][f*3:(f+1)*3] for f in fids], dim=0))
        te_x.append(torch.cat([cache_data["test_x"][f*4:(f+1)*4]  for f in fids], dim=0))
        te_y.append(torch.cat([cache_data["test_y"][f*4:(f+1)*4]  for f in fids], dim=0))
    return tr_x, tr_y, te_x, te_y


def bootstrap_paired_ci(vals_1, vals_2, n_boot=10000, seed=42):
    n = min(len(vals_1), len(vals_2))
    diffs = np.array(vals_1[:n]) - np.array(vals_2[:n])
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    return np.mean(diffs), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ---------------------------------------------------------------------------
# Phase 2 Experiment Suite Runner (50 Runs per Condition)
# ---------------------------------------------------------------------------

def run_phase2_suite(block_assignment, cache_data, pca_basis_r32,
                     seeds=list(range(101, 106)), num_shuffles=10,
                     r=32, epochs=100, lr=1e-2):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)

    conditions = [
        "naive", "offline",
        "OGP_k2", "OGP_k4", "OGP_k8", "OGP_k12", "OGP_k16", "OGP_k24", "OGP_k32",
        "RANDOM-32", "BOTTOM-32", "CURRENT-32"
    ]

    suite_results = {c: [] for c in conditions}

    for c_name in conditions:
        print(f"  --> Running Phase 2 condition: {c_name:12s} ({len(order_list)*len(seeds)} runs)...")
        for order in order_list:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                adapter = BottleneckAdapter(r=r, pca_basis=pca_basis_r32).to(DEVICE)
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
                R = np.zeros((10, 10))

                base_blocks = order[:5]
                joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
                joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

                adapter.train()
                for ep in range(45 if c_name == "offline" else epochs):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Base phase evaluation loop filling R[4, :]
                adapter.eval()
                with torch.no_grad():
                    z_refs_base = adapter(joint_train_x_base)
                    for b in range(10):
                        test_x_b = te_x[b].to(DEVICE)
                        test_y_b = te_y[b].to(DEVICE)
                        z_queries = adapter(test_x_b)
                        correct = sum(
                            1 for q_idx, q_vec in enumerate(z_queries)
                            if joint_train_y_base[torch.argmax(torch.matmul(z_refs_base, q_vec)).item()].item() == test_y_b[q_idx].item()
                        )
                        R[4, b] = correct / len(z_queries)

                M_past = joint_train_x_base.clone().detach()

                for step in range(5, 10):
                    curr_block = order[step]
                    seen_blocks = order[:step + 1]
                    seen_ref_x = torch.cat([tr_x[b] for b in seen_blocks], dim=0).to(DEVICE)
                    seen_ref_y = torch.cat([tr_y[b] for b in seen_blocks], dim=0).to(DEVICE)

                    curr_x = tr_x[curr_block].to(DEVICE)
                    curr_y = tr_y[curr_block].to(DEVICE)

                    proj_mat = None
                    if c_name.startswith("OGP_k"):
                        k_val = int(c_name.split("_k")[1])
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        actual_k = min(k_val, Vh.shape[0])
                        P = Vh[:actual_k].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                    elif c_name == "RANDOM-32":
                        R_mat = torch.randn(INPUT_DIM, 32, device=DEVICE)
                        Q, _ = torch.linalg.qr(R_mat)
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(Q, Q.T)
                    elif c_name == "BOTTOM-32":
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        P = Vh[-32:].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                    elif c_name == "CURRENT-32":
                        _, S, Vh = torch.linalg.svd(curr_x, full_matrices=False)
                        P = Vh[:32].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)

                    adapter.train()
                    train_data_x = seen_ref_x if c_name == "offline" else curr_x
                    train_data_y = seen_ref_y if c_name == "offline" else curr_y

                    for ep in range(epochs):
                        proj = adapter(train_data_x)
                        loss = supervised_contrastive_loss(proj, train_data_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        if proj_mat is not None and adapter.V.weight.grad is not None:
                            # Apply gradient projection to input layer V (shape 32x960 * 960x960 -> 32x960)
                            adapter.V.weight.grad = torch.matmul(adapter.V.weight.grad, proj_mat)
                        optimizer.step()

                    M_past = torch.cat([M_past, curr_x.clone().detach()], dim=0)

                    adapter.eval()
                    with torch.no_grad():
                        z_refs_step = adapter(seen_ref_x)
                        for b in range(10):
                            test_x_b = te_x[b].to(DEVICE)
                            test_y_b = te_y[b].to(DEVICE)
                            z_queries = adapter(test_x_b)
                            correct = sum(
                                1 for q_idx, q_vec in enumerate(z_queries)
                                if seen_ref_y[torch.argmax(torch.matmul(z_refs_step, q_vec)).item()].item() == test_y_b[q_idx].item()
                            )
                            R[step, b] = correct / len(z_queries)

                a_t = float(np.mean(R[9, :]))
                la = float(np.mean([R[max(4, order.index(j)), j] for j in range(10)]))
                bwt = a_t - la  # EXACT identity
                fgt = float(np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)]))
                suite_results[c_name].append({"A_T": a_t, "LA": la, "BWT": bwt, "mean_forgetting": fgt})

    return suite_results


# ---------------------------------------------------------------------------
# Reporting & Master Summary Table Formatting
# ---------------------------------------------------------------------------

def print_phase2_summary(suite_results, title="PHASE 2 MASTER SUITE RESULTS"):
    print("\n" + "=" * 155)
    print(f"  {title}")
    print("=" * 155)

    nai_runs = suite_results["naive"]
    off_runs = suite_results["offline"]

    nai_ats = [r["A_T"] for r in nai_runs]
    nai_las = [r["LA"] for r in nai_runs]
    nai_bwts = [r["BWT"] for r in nai_runs]
    nai_fgts = [r["mean_forgetting"] for r in nai_runs]

    off_ats = [r["A_T"] for r in off_runs]
    off_las = [r["LA"] for r in off_runs]
    off_bwts = [r["BWT"] for r in off_runs]
    off_fgts = [r["mean_forgetting"] for r in off_runs]

    off_mean_at = np.mean(off_ats) * 100.0
    nai_mean_at = np.mean(nai_ats) * 100.0
    nai_mean_la = np.mean(nai_las) * 100.0
    nai_mean_bwt = np.mean(nai_bwts) * 100.0

    print(f"  * Naive Baseline  (n={len(nai_runs):2d}): A_T={nai_mean_at:5.2f}% ± {np.std(nai_ats)*100:.2f}% ({np.min(nai_ats)*100:.2f}..{np.max(nai_ats)*100:.2f}%) | LA={nai_mean_la:5.2f}% | BWT={nai_mean_bwt:+6.2f}% | Fgt={np.mean(nai_fgts)*100:.2f}%")
    print(f"  * Offline Baseline(n={len(off_runs):2d}): A_T={off_mean_at:5.2f}% ± {np.std(off_ats)*100:.2f}% ({np.min(off_ats)*100:.2f}..{np.max(off_ats)*100:.2f}%) | LA={np.mean(off_las)*100:5.2f}% | BWT={np.mean(off_bwts)*100:+6.2f}% | Fgt={np.mean(off_fgts)*100:.2f}%")
    print(f"  * Total Continual Learning Gap   : {off_mean_at - nai_mean_at:+.2f} pp")
    print("-" * 155)

    header = (
        f"{'Condition':14s} | {'A_T (Min..Max)':25s} | {'LA (Learning)':13s} | {'BWT (A_T-LA)':13s} | "
        f"{'Obs.Fgt':9s} | {'dA_T vs Naive (95% CI)':26s} | {'dBWT vs Naive (95% CI)':26s} | "
        f"{'Retention Share':15s} | Verdict"
    )
    print(header)
    print("-" * len(header))

    for c_name, runs in suite_results.items():
        if c_name in ["naive", "offline"]:
            continue
        ats  = [r["A_T"] for r in runs]
        las  = [r["LA"] for r in runs]
        bwts = [r["BWT"] for r in runs]
        fgts = [r["mean_forgetting"] for r in runs]

        at_m = np.mean(ats) * 100.0
        at_s = np.std(ats) * 100.0
        at_min = np.min(ats) * 100.0
        at_max = np.max(ats) * 100.0

        la_m  = np.mean(las) * 100.0
        bwt_m = np.mean(bwts) * 100.0
        fgt_m = np.mean(fgts) * 100.0

        diff_at_m, ci_at_l, ci_at_u   = bootstrap_paired_ci(ats, nai_ats)
        diff_bwt_m, ci_bwt_l, ci_bwt_u = bootstrap_paired_ci(bwts, nai_bwts)
        diff_la_m, _, _               = bootstrap_paired_ci(las, nai_las)

        # Decompositions: exact delta_A_T = delta_LA + delta_BWT
        d_at  = diff_at_m * 100.0
        d_bwt = diff_bwt_m * 100.0
        d_la  = diff_la_m * 100.0

        retention_share = (d_bwt / d_at * 100.0) if abs(d_at) > 1e-5 else 0.0

        verdict = ""
        if ci_at_l > 0.0:
            verdict = "SUCCESS (+A_T)"
        elif ci_at_u < 0.0:
            verdict = "SIG WORSE (-A_T)"
        else:
            verdict = "TRUE NULL"

        row_str = (
            f"  {c_name:14s} | "
            f"{at_m:5.2f}% ± {at_s:4.2f}% ({at_min:5.2f}..{at_max:5.2f}%) | "
            f"{la_m:5.2f}%       | "
            f"{bwt_m:+6.2f}%       | "
            f"{fgt_m:5.2f}%    | "
            f"{d_at:+5.2f}% [{ci_at_l*100:+5.2f}%, {ci_at_u*100:+5.2f}%] | "
            f"{d_bwt:+5.2f}% [{ci_bwt_l*100:+5.2f}%, {ci_bwt_u*100:+5.2f}%] | "
            f"{retention_share:6.1f}%          | "
            f"{verdict}"
        )
        print(row_str)

    print("=" * 155)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("  PHASE 2 MASTER SUITE: OGP & CONTROLS ON CALIBRATED FORGETTING BENCHMARK")
    print("  Cell: C_r32_ep100_lr1e-02 (r=32 bottleneck, ep=100, lr=1e-2)")
    print("=" * 100)

    with open(DATASET_PATH, "r") as f:
        blocks_data = json.load(f)

    if not os.path.exists(CACHE_PATH):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks_data)
    else:
        cache_data = torch.load(CACHE_PATH, map_location=DEVICE)
        print(f"  [Cache] Loaded embeddings from {CACHE_PATH}")

    pca_basis_r32 = compute_pca_basis(cache_data, r=32).to(DEVICE)
    conf_pairs = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)

    print(f"  [Data] Block assignment ready (10 blocks, 100 facts). Confusable pairs = {len(conf_pairs)}")

    # 1. Selection Seeds (101..105, 50 runs per condition)
    print("\n  =========================================================================")
    print("    RUNNING SELECTION SEEDS 101..105 (50 RUNS PER CONDITION)")
    print("  =========================================================================")
    res_select_50 = run_phase2_suite(block_assignment, cache_data, pca_basis_r32,
                                     seeds=[101, 102, 103, 104, 105], num_shuffles=10)
    print_phase2_summary(res_select_50, title="PHASE 2 RESULTS -- SELECTION SEEDS 101..105 (50 RUNS / CELL)")

    # 2. Fresh Replication Seeds (211..215, 50 runs per condition)
    print("\n  =========================================================================")
    print("    RUNNING FRESH REPLICATION SEEDS 211..215 (50 RUNS PER CONDITION)")
    print("  =========================================================================")
    res_fresh_50 = run_phase2_suite(block_assignment, cache_data, pca_basis_r32,
                                    seeds=[211, 212, 213, 214, 215], num_shuffles=10)
    print_phase2_summary(res_fresh_50, title="PHASE 2 RESULTS -- FRESH REPLICATION SEEDS 211..215 (50 RUNS / CELL)")


if __name__ == "__main__":
    main()
