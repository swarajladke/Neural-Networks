"""
run_decisive_controls.py  --  Decisive Controls C1-C4 for Calibrated Forgetting Benchmark
==========================================================================================

DECISIVE CONTROL ARMS (50 runs per condition x 2 seed sets {101..105} and {211..215}):

  C1. NAIVE at lr=1e-3 and lr=3e-3 (r=32, epochs=100):
      Tests if lower learning rates under the calibrated bottleneck match or beat
      CURRENT-32 (A_T = 83.80% / 82.25%). If so, CURRENT-32's gain is step-size damping.

  C2. FREEZE-AFTER-BASE:
      Train base blocks 0-4 jointly, then freeze adapter parameters (no updates) for
      sequential blocks 5-9. Measures the exact null hypothesis of zero adaptation.

  C3. GRADIENT-NORM RATIO LOGGING:
      Logs ||grad_projected|| / ||grad_raw|| per optimizer step for TOP-32 and CURRENT-32.
      Reports mean, std, min, max. Near-zero ratio for CURRENT-32 indicates it acts
      as an off-switch, not a selective regulariser.

  C4. GRADIENT-CLIP CONTROL:
      Naive sequential fine-tuning at lr=1e-2 with global gradient-norm clipping set to
      CURRENT-32's measured mean projected norm. Isolates norm damping from subspace choice.
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
# C3: Gradient Norm Ratio Logging
# ---------------------------------------------------------------------------

def run_gradient_norm_logging(block_assignment, cache_data, pca_basis_r32,
                              seeds=[101, 102, 103, 104, 105], num_shuffles=3,
                              r=32, epochs=100, lr=1e-2):
    """Log ||grad_projected|| / ||grad_raw|| and ||grad_projected|| per step for TOP-32 and CURRENT-32."""
    print("\n" + "=" * 100)
    print("  CONTROL C3: GRADIENT-NORM RATIO & PROJECTED NORM LOGGING")
    print("=" * 100)

    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    random.seed(42)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    norm_stats = {"TOP-32": {"ratios": [], "proj_norms": [], "raw_norms": []},
                  "CURRENT-32": {"ratios": [], "proj_norms": [], "raw_norms": []}}

    for c_name in ("TOP-32", "CURRENT-32"):
        for order in order_list:
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                adapter = BottleneckAdapter(r=r, pca_basis=pca_basis_r32).to(DEVICE)
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

                base_blocks = order[:5]
                joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
                joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

                adapter.train()
                for _ in range(epochs):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                M_past = joint_train_x_base.clone().detach()

                for step in range(5, 10):
                    curr_block = order[step]
                    curr_x = tr_x[curr_block].to(DEVICE)
                    curr_y = tr_y[curr_block].to(DEVICE)

                    if c_name == "TOP-32":
                        _, _, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        P = Vh[:32].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                    elif c_name == "CURRENT-32":
                        _, _, Vh = torch.linalg.svd(curr_x, full_matrices=False)
                        P = Vh[:32].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)

                    adapter.train()
                    for ep in range(epochs):
                        proj = adapter(curr_x)
                        loss = supervised_contrastive_loss(proj, curr_y)
                        optimizer.zero_grad()
                        loss.backward()

                        raw_grad = adapter.V.weight.grad.clone()
                        raw_norm = torch.norm(raw_grad).item()

                        adapter.V.weight.grad = torch.matmul(adapter.V.weight.grad, proj_mat)
                        proj_norm = torch.norm(adapter.V.weight.grad).item()

                        ratio = (proj_norm / raw_norm) if raw_norm > 1e-12 else 0.0

                        norm_stats[c_name]["raw_norms"].append(raw_norm)
                        norm_stats[c_name]["proj_norms"].append(proj_norm)
                        norm_stats[c_name]["ratios"].append(ratio)

                        optimizer.step()

                    M_past = torch.cat([M_past, curr_x.clone().detach()], dim=0)

    for c_name in ("TOP-32", "CURRENT-32"):
        ratios = np.array(norm_stats[c_name]["ratios"])
        p_norms = np.array(norm_stats[c_name]["proj_norms"])
        r_norms = np.array(norm_stats[c_name]["raw_norms"])
        print(f"  [{c_name}] Gradient Norm Ratio (||grad_proj|| / ||grad_raw||):")
        print(f"    Mean = {ratios.mean():.6f} | Std = {ratios.std():.6f} | Min = {ratios.min():.6f} | Max = {ratios.max():.6f}")
        print(f"  [{c_name}] Projected Grad Norm (||grad_proj||):")
        print(f"    Mean = {p_norms.mean():.6f} | Std = {p_norms.std():.6f} | Min = {p_norms.min():.6f} | Max = {p_norms.max():.6f}")
        print(f"  [{c_name}] Raw Grad Norm (||grad_raw||):")
        print(f"    Mean = {r_norms.mean():.6f} | Std = {r_norms.std():.6f} | Min = {r_norms.min():.6f} | Max = {r_norms.max():.6f}")
        print("-" * 100)

    current_32_mean_proj_norm = float(np.mean(norm_stats["CURRENT-32"]["proj_norms"]))
    print(f"  --> Measured CURRENT-32 Mean Projected Grad Norm: {current_32_mean_proj_norm:.6f}")
    print("=" * 100)
    return current_32_mean_proj_norm, norm_stats


# ---------------------------------------------------------------------------
# Decisive Controls Suite Runner (50 Runs per Condition)
# ---------------------------------------------------------------------------

def run_decisive_controls_suite(block_assignment, cache_data, pca_basis_r32,
                                target_clip_norm=None,
                                seeds=list(range(101, 106)), num_shuffles=10,
                                r=32, epochs=100):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    conditions = [
        "naive_lr1e-2",       # baseline from Phase 2 (calibrated)
        "naive_lr1e-3",       # C1: lower lr = 1e-3
        "naive_lr3e-3",       # C1: lower lr = 3e-3
        "FREEZE-AFTER-BASE",  # C2: zero parameter updates for blocks 5-9
        "GRADIENT-CLIP-C4"    # C4: naive lr=1e-2 with grad-norm clipping = target_clip_norm
    ]

    suite_results = {c: [] for c in conditions}

    for c_name in conditions:
        print(f"  --> Running Decisive Control condition: {c_name:20s} ({len(order_list)*len(seeds)} runs)...")
        for order in order_list:
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

                adapter = BottleneckAdapter(r=r, pca_basis=pca_basis_r32).to(DEVICE)

                lr_val = 1e-2
                if c_name == "naive_lr1e-3":
                    lr_val = 1e-3
                elif c_name == "naive_lr3e-3":
                    lr_val = 3e-3

                optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr_val, weight_decay=1e-4)
                R = np.zeros((10, 10))

                base_blocks = order[:5]
                joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
                joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

                adapter.train()
                for _ in range(epochs):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

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

                for step in range(5, 10):
                    curr_block = order[step]
                    seen_blocks = order[:step + 1]
                    seen_ref_x = torch.cat([tr_x[b] for b in seen_blocks], dim=0).to(DEVICE)
                    seen_ref_y = torch.cat([tr_y[b] for b in seen_blocks], dim=0).to(DEVICE)

                    curr_x = tr_x[curr_block].to(DEVICE)
                    curr_y = tr_y[curr_block].to(DEVICE)

                    if c_name == "FREEZE-AFTER-BASE":
                        # C2: No parameter updates at all
                        pass
                    else:
                        adapter.train()
                        for ep in range(epochs):
                            proj = adapter(curr_x)
                            loss = supervised_contrastive_loss(proj, curr_y)
                            optimizer.zero_grad()
                            loss.backward()

                            if c_name == "GRADIENT-CLIP-C4" and target_clip_norm is not None:
                                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=target_clip_norm)

                            optimizer.step()

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
                bwt = a_t - la
                fgt = float(np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)]))
                suite_results[c_name].append({"A_T": a_t, "LA": la, "BWT": bwt, "mean_forgetting": fgt})

    return suite_results


# ---------------------------------------------------------------------------
# Reporting Summary Table
# ---------------------------------------------------------------------------

def print_decisive_controls_summary(suite_results, title="DECISIVE CONTROLS C1-C4 RESULTS"):
    print("\n" + "=" * 145)
    print(f"  {title}")
    print("=" * 145)

    base_runs = suite_results["naive_lr1e-2"]
    base_ats  = [r["A_T"] for r in base_runs]
    base_las  = [r["LA"]  for r in base_runs]
    base_bwts = [r["BWT"] for r in base_runs]
    base_fgts = [r["mean_forgetting"] for r in base_runs]

    print(f"  * Baseline Naive lr=1e-2 (n={len(base_runs):2d}): A_T={np.mean(base_ats)*100:5.2f}% ± {np.std(base_ats)*100:.2f}% | LA={np.mean(base_las)*100:5.2f}% | BWT={np.mean(base_bwts)*100:+6.2f}%")
    print("-" * 145)

    header = (
        f"{'Condition':22s} | {'A_T (Min..Max)':25s} | {'LA (Learning)':13s} | {'BWT (A_T-LA)':13s} | "
        f"{'Obs.Fgt':9s} | {'dA_T vs Naive (95% CI)':26s} | {'dBWT vs Naive (95% CI)':26s}"
    )
    print(header)
    print("-" * len(header))

    for c_name, runs in suite_results.items():
        ats  = [r["A_T"] for r in runs]
        las  = [r["LA"]  for r in runs]
        bwts = [r["BWT"] for r in runs]
        fgts = [r["mean_forgetting"] for r in runs]

        at_m  = np.mean(ats) * 100.0
        at_s  = np.std(ats) * 100.0
        at_min = np.min(ats) * 100.0
        at_max = np.max(ats) * 100.0

        la_m  = np.mean(las) * 100.0
        bwt_m = np.mean(bwts) * 100.0
        fgt_m = np.mean(fgts) * 100.0

        diff_at_m, ci_at_l, ci_at_u   = bootstrap_paired_ci(ats, base_ats)
        diff_bwt_m, ci_bwt_l, ci_bwt_u = bootstrap_paired_ci(bwts, base_bwts)

        row_str = (
            f"  {c_name:22s} | "
            f"{at_m:5.2f}% ± {at_s:4.2f}% ({at_min:5.2f}..{at_max:5.2f}%) | "
            f"{la_m:5.2f}%       | "
            f"{bwt_m:+6.2f}%       | "
            f"{fgt_m:5.2f}%    | "
            f"{diff_at_m*100:+5.2f}% [{ci_at_l*100:+5.2f}%, {ci_at_u*100:+5.2f}%] | "
            f"{diff_bwt_m*100:+5.2f}% [{ci_bwt_l*100:+5.2f}%, {ci_bwt_u*100:+5.2f}%]"
        )
        print(row_str)

    print("=" * 145)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("  DECISIVE CONTROLS SUITE (C1 - C4): RESOLVING CURRENT-32 REGULARISATION VS STEP-SIZE DAMPING")
    print("  Benchmark: BottleneckAdapter r=32, epochs=100, lr=1e-2")
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

    # 1. Control C3: Gradient Norm Ratio Logging & Target Clip Norm Calculation
    mean_proj_norm_current32, norm_stats = run_gradient_norm_logging(
        block_assignment, cache_data, pca_basis_r32, seeds=[101, 102, 103], num_shuffles=3)

    # 2. Control Battery (C1, C2, C4) on Selection Seeds (101..105, 50 runs per condition)
    print("\n  =========================================================================")
    print("    RUNNING DECISIVE CONTROLS ON SELECTION SEEDS 101..105 (50 RUNS / CELL)")
    print("  =========================================================================")
    res_select_50 = run_decisive_controls_suite(
        block_assignment, cache_data, pca_basis_r32,
        target_clip_norm=mean_proj_norm_current32,
        seeds=[101, 102, 103, 104, 105], num_shuffles=10)
    print_decisive_controls_summary(res_select_50, title="DECISIVE CONTROLS C1-C4 -- SELECTION SEEDS 101..105 (50 RUNS / CELL)")

    # 3. Control Battery (C1, C2, C4) on Fresh Replication Seeds (211..215, 50 runs per condition)
    print("\n  =========================================================================")
    print("    RUNNING DECISIVE CONTROLS ON FRESH REPLICATION SEEDS 211..215 (50 RUNS / CELL)")
    print("  =========================================================================")
    res_fresh_50 = run_decisive_controls_suite(
        block_assignment, cache_data, pca_basis_r32,
        target_clip_norm=mean_proj_norm_current32,
        seeds=[211, 212, 213, 214, 215], num_shuffles=10)
    print_decisive_controls_summary(res_fresh_50, title="DECISIVE CONTROLS C1-C4 -- FRESH SEEDS 211..215 (50 RUNS / CELL)")


if __name__ == "__main__":
    main()
