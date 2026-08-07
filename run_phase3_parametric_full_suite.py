"""
run_phase3_parametric_full_suite.py  --  Full 50-Run Phase 3 Parametric Memory Suite
====================================================================================

EXACT REPLICATION & CI EVALUATION (50 runs per condition x 2 seed sets {101..105} & {211..215}):
  - Part 1: C2 FREEZE-AFTER-BASE Step-9 Breakdown (Base-Trained vs Never-Trained facts
            evaluated against the full 100-fact reference bank). Resolves arithmetic discrepancy.
  - Part 2: Parametric Memory Classification Benchmark (100-class head, fixed lr=1e-3, ep=30).
            Sweep k in {2, 4, 8, 12, 16, 24, 32} at BOTH r=960 and r=32.
            Standing controls: FREEZE-AFTER-BASE, RANDOM-k, BOTTOM-k.
            10,000-sample paired bootstrap 95% CIs for dA_T and dBWT vs Naive.
            Decomposition of retention gap closed (dBWT / available BWT gap) vs
            acquisition gap closed (dLA / available LA gap).
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
NUM_CLASSES  = 100

# ---------------------------------------------------------------------------
# Helpers & Adapters
# ---------------------------------------------------------------------------

class BottleneckAdapter(nn.Module):
    def __init__(self, r, pca_basis):
        super().__init__()
        assert pca_basis.shape == (r, INPUT_DIM)
        self.r = r
        self.V = nn.Linear(INPUT_DIM, r,         bias=False)
        self.U = nn.Linear(r,         INPUT_DIM, bias=True)
        with torch.no_grad():
            self.V.weight.copy_(pca_basis)
            self.U.weight.copy_(pca_basis.T)
            nn.init.zeros_(self.U.bias)

    def forward(self, x):
        return F.normalize(self.U(self.V(x)), dim=-1)


class FullRankAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, INPUT_DIM, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)


def compute_pca_basis(cache_data, r):
    X = cache_data["train_x"].float().cpu()
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
# PART 1: C2 Step-9 Breakdown (Corrected Arithmetic Evaluation)
# ---------------------------------------------------------------------------

def run_part1_c2_step9_breakdown(block_assignment, cache_data, pca_basis_r32, seeds, num_shuffles=10):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    base_trained_accs = []
    never_trained_accs = []
    overall_ats = []

    for order in order_list:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

            base_blocks = order[:5]
            joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            adapter.train()
            for _ in range(100):
                proj = adapter(joint_train_x_base)
                loss = supervised_contrastive_loss(proj, joint_train_y_base)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # At step 9, the full 100-fact reference bank is available
            all_train_x = torch.cat([tr_x[b] for b in order], dim=0).to(DEVICE)
            all_train_y = torch.cat([tr_y[b] for b in order], dim=0).to(DEVICE)

            adapter.eval()
            with torch.no_grad():
                z_refs_all = adapter(all_train_x)
                block_accs = []
                for b in range(10):
                    test_x_b = te_x[b].to(DEVICE)
                    test_y_b = te_y[b].to(DEVICE)
                    z_queries = adapter(test_x_b)
                    correct = sum(
                        1 for q_idx, q_vec in enumerate(z_queries)
                        if all_train_y[torch.argmax(torch.matmul(z_refs_all, q_vec)).item()].item() == test_y_b[q_idx].item()
                    )
                    block_accs.append(correct / len(z_queries))

                base_acc = np.mean([block_accs[b] for b in order[:5]]) * 100.0
                never_acc = np.mean([block_accs[b] for b in order[5:]]) * 100.0
                overall_at = np.mean(block_accs) * 100.0

                base_trained_accs.append(base_acc)
                never_trained_accs.append(never_acc)
                overall_ats.append(overall_at)

    return {
        "base_mean": float(np.mean(base_trained_accs)),
        "base_std": float(np.std(base_trained_accs)),
        "never_mean": float(np.mean(never_trained_accs)),
        "never_std": float(np.std(never_trained_accs)),
        "overall_mean": float(np.mean(overall_ats)),
        "overall_std": float(np.std(overall_ats))
    }


# ---------------------------------------------------------------------------
# PART 2: Parametric Memory Benchmark Suite (50 Runs per Cell)
# ---------------------------------------------------------------------------

class ParametricClassifier(nn.Module):
    def __init__(self, r=32, pca_basis=None):
        super().__init__()
        if r == INPUT_DIM or pca_basis is None:
            self.adapter = FullRankAdapter()
        else:
            self.adapter = BottleneckAdapter(r=r, pca_basis=pca_basis)
        self.head = nn.Linear(INPUT_DIM, NUM_CLASSES, bias=True)

    def forward(self, x):
        features = self.adapter(x)
        logits   = self.head(features)
        return logits


def run_parametric_arm(block_assignment, cache_data, pca_basis,
                       arm_type="naive", r=32, epochs=30, lr=1e-3,
                       seeds=list(range(101, 106)), num_shuffles=10,
                       ogp_k=None):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    results = []

    for order in order_list:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            model = ParametricClassifier(r=r, pca_basis=pca_basis).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            R = np.zeros((10, 10))

            base_blocks = order[:5]
            bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            model.train()
            for _ in range(epochs):
                logits = model(bx)
                loss   = criterion(logits, by)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                for b in range(10):
                    tx = te_x[b].to(DEVICE); ty = te_y[b].to(DEVICE)
                    preds = torch.argmax(model(tx), dim=-1)
                    R[4, b] = (preds == ty).float().mean().item()

            M_past = bx.clone().detach()

            for step in range(5, 10):
                curr_block  = order[step]
                seen_blocks = order[:step + 1]
                sx = torch.cat([tr_x[b] for b in seen_blocks], dim=0).to(DEVICE)
                sy = torch.cat([tr_y[b] for b in seen_blocks], dim=0).to(DEVICE)

                cx = tr_x[curr_block].to(DEVICE)
                cy = tr_y[curr_block].to(DEVICE)

                tx, ty = (sx, sy) if arm_type == "offline" else (cx, cy)

                proj_mat = None
                if arm_type.startswith("OGP_k") and ogp_k is not None:
                    _, _, Vh = torch.linalg.svd(M_past, full_matrices=False)
                    actual_k = min(ogp_k, Vh.shape[0])
                    P = Vh[:actual_k].T
                    proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                elif arm_type.startswith("RANDOM-k") and ogp_k is not None:
                    R_mat = torch.randn(INPUT_DIM, ogp_k, device=DEVICE)
                    Q, _ = torch.linalg.qr(R_mat)
                    proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(Q, Q.T)
                elif arm_type.startswith("BOTTOM-k") and ogp_k is not None:
                    _, _, Vh = torch.linalg.svd(M_past, full_matrices=False)
                    P = Vh[-min(ogp_k, Vh.shape[0]):].T
                    proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)

                if arm_type != "FREEZE-AFTER-BASE":
                    model.train()
                    for ep in range(epochs):
                        logits = model(tx)
                        loss   = criterion(logits, ty)
                        optimizer.zero_grad()
                        loss.backward()

                        if proj_mat is not None and hasattr(model.adapter, "V") and model.adapter.V.weight.grad is not None:
                            model.adapter.V.weight.grad = torch.matmul(model.adapter.V.weight.grad, proj_mat)
                        elif proj_mat is not None and hasattr(model.adapter, "linear") and model.adapter.linear.weight.grad is not None:
                            model.adapter.linear.weight.grad = torch.matmul(model.adapter.linear.weight.grad, proj_mat)

                        optimizer.step()

                M_past = torch.cat([M_past, cx.clone().detach()], dim=0)

                model.eval()
                with torch.no_grad():
                    for b in range(10):
                        t_x = te_x[b].to(DEVICE); t_y = te_y[b].to(DEVICE)
                        preds = torch.argmax(model(t_x), dim=-1)
                        R[step, b] = (preds == t_y).float().mean().item()

            a_t = float(np.mean(R[9, :]))
            la  = float(np.mean([R[max(4, order.index(j)), j] for j in range(10)]))
            bwt = a_t - la
            fgt = float(np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)]))
            results.append({"A_T": a_t, "LA": la, "BWT": bwt, "fgt": fgt})

    return results


def print_suite_table(arm_results, title="PARAMETRIC MEMORY SUITE RESULTS"):
    print("\n" + "=" * 145)
    print(f"  {title}")
    print("=" * 145)

    naive_res = arm_results["naive"]
    naive_ats  = [r["A_T"] for r in naive_res]
    naive_las  = [r["LA"]  for r in naive_res]
    naive_bwts = [r["BWT"] for r in naive_res]

    off_res = arm_results["offline"]
    off_at  = float(np.mean([r["A_T"] for r in off_res]))
    off_la  = float(np.mean([r["LA"]  for r in off_res]))
    off_bwt = float(np.mean([r["BWT"] for r in off_res]))

    avail_bwt_gap = off_bwt - np.mean(naive_bwts)
    avail_la_gap  = off_la  - np.mean(naive_las)

    print(f"  * Baseline Naive   (n={len(naive_res)}): A_T={np.mean(naive_ats)*100:5.2f}% ± {np.std(naive_ats)*100:.2f}% | LA={np.mean(naive_las)*100:5.2f}% | BWT={np.mean(naive_bwts)*100:+6.2f}%")
    print(f"  * Baseline Offline (n={len(off_res)}): A_T={off_at*100:5.2f}% ± {np.std([r['A_T'] for r in off_res])*100:.2f}% | LA={off_la*100:5.2f}% | BWT={off_bwt*100:+6.2f}%")
    print("-" * 145)

    header = (
        f"{'Condition':18s} | {'A_T (Min..Max)':25s} | {'LA':7s} | {'BWT':8s} | "
        f"{'dA_T vs Naive (95% CI)':25s} | {'dBWT vs Naive (95% CI)':25s} | {'Ret. Gap Closed':15s} | {'Acq. Gap Closed':15s}"
    )
    print(header)
    print("-" * len(header))

    for c_name, runs in arm_results.items():
        ats  = [r["A_T"] for r in runs]
        las  = [r["LA"]  for r in runs]
        bwts = [r["BWT"] for r in runs]

        at_m  = np.mean(ats) * 100.0
        at_s  = np.std(ats) * 100.0
        at_min = np.min(ats) * 100.0
        at_max = np.max(ats) * 100.0

        la_m  = np.mean(las) * 100.0
        bwt_m = np.mean(bwts) * 100.0

        diff_at_m, ci_at_l, ci_at_u   = bootstrap_paired_ci(ats, naive_ats)
        diff_bwt_m, ci_bwt_l, ci_bwt_u = bootstrap_paired_ci(bwts, naive_bwts)

        diff_la_m = np.mean(las) - np.mean(naive_las)

        ret_closed = (diff_bwt_m / avail_bwt_gap * 100.0) if abs(avail_bwt_gap) > 1e-6 else 0.0
        acq_closed = (diff_la_m / avail_la_gap * 100.0) if abs(avail_la_gap) > 1e-6 else 0.0

        row_str = (
            f"  {c_name:18s} | "
            f"{at_m:5.2f}% ± {at_s:4.2f}% ({at_min:5.2f}..{at_max:5.2f}%) | "
            f"{la_m:5.2f}% | "
            f"{bwt_m:+6.2f}% | "
            f"{diff_at_m*100:+5.2f}% [{ci_at_l*100:+5.2f}%, {ci_at_u*100:+5.2f}%] | "
            f"{diff_bwt_m*100:+5.2f}% [{ci_bwt_l*100:+5.2f}%, {ci_bwt_u*100:+5.2f}%] | "
            f"{ret_closed:+14.1f}% | "
            f"{acq_closed:+14.1f}%"
        )
        print(row_str)

    print("=" * 145)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("  PHASE 3 FULL SUITE: RESOLVING C2 CONTRADICTION & PARAMETRIC MEMORY CI EVALUATION")
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

    # ------------------------------------------------------------------
    # PART 1: C2 STEP-9 BREAKDOWN (Corrected Arithmetic)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PART 1: C2 STEP-9 BREAKDOWN (CORRECTED ARITHMETIC EVALUATION)")
    print("=" * 80)

    c2_sel = run_part1_c2_step9_breakdown(block_assignment, cache_data, pca_basis_r32, seeds=[101, 102, 103, 104, 105])
    c2_fre = run_part1_c2_step9_breakdown(block_assignment, cache_data, pca_basis_r32, seeds=[211, 212, 213, 214, 215])

    print("\n  C2 STEP-9 BREAKDOWN RESULTS:")
    print("    Selection Seeds 101..105 (50 runs):")
    print(f"      - Base-Trained Blocks order[0:5]:  {c2_sel['base_mean']:.2f}% ± {c2_sel['base_std']:.2f}%")
    print(f"      - Never-Trained Blocks order[5:10]: {c2_sel['never_mean']:.2f}% ± {c2_sel['never_std']:.2f}%")
    print(f"      - Unweighted Mean (Calculated):    {(c2_sel['base_mean'] + c2_sel['never_mean'])/2.0:.2f}%")
    print(f"      - Overall A_T (Step 9 Measured):   {c2_sel['overall_mean']:.2f}% ± {c2_sel['overall_std']:.2f}%")

    print("\n    Fresh Replication Seeds 211..215 (50 runs):")
    print(f"      - Base-Trained Blocks order[0:5]:  {c2_fre['base_mean']:.2f}% ± {c2_fre['base_std']:.2f}%")
    print(f"      - Never-Trained Blocks order[5:10]: {c2_fre['never_mean']:.2f}% ± {c2_fre['never_std']:.2f}%")
    print(f"      - Unweighted Mean (Calculated):    {(c2_fre['base_mean'] + c2_fre['never_mean'])/2.0:.2f}%")
    print(f"      - Overall A_T (Step 9 Measured):   {c2_fre['overall_mean']:.2f}% ± {c2_fre['overall_std']:.2f}%")

    # ------------------------------------------------------------------
    # PART 2: PARAMETRIC MEMORY SUITE (r=960 and r=32 across BOTH Seed Sets)
    # ------------------------------------------------------------------
    k_vals = [2, 4, 8, 12, 16, 24, 32]

    for seed_set_name, seeds in [("SELECTION SEEDS 101..105", list(range(101, 106))),
                                 ("FRESH SEEDS 211..215",     list(range(211, 216)))]:

        for r_val, p_basis in [(960, None), (32, pca_basis_r32)]:
            print(f"\n  Running Parametric Benchmark: {seed_set_name} | Capacity r={r_val}...")
            suite_results = {}

            # Baselines
            suite_results["naive"] = run_parametric_arm(
                block_assignment, cache_data, p_basis, arm_type="naive", r=r_val, epochs=30, lr=1e-3, seeds=seeds)
            suite_results["offline"] = run_parametric_arm(
                block_assignment, cache_data, p_basis, arm_type="offline", r=r_val, epochs=30, lr=1e-3, seeds=seeds)
            suite_results["FREEZE-AFTER-BASE"] = run_parametric_arm(
                block_assignment, cache_data, p_basis, arm_type="FREEZE-AFTER-BASE", r=r_val, epochs=30, lr=1e-3, seeds=seeds)

            # k sweep
            for k in k_vals:
                suite_results[f"OGP_k{k}"] = run_parametric_arm(
                    block_assignment, cache_data, p_basis, arm_type=f"OGP_k{k}", r=r_val, epochs=30, lr=1e-3, seeds=seeds, ogp_k=k)
                suite_results[f"RANDOM-k{k}"] = run_parametric_arm(
                    block_assignment, cache_data, p_basis, arm_type=f"RANDOM-k{k}", r=r_val, epochs=30, lr=1e-3, seeds=seeds, ogp_k=k)
                suite_results[f"BOTTOM-k{k}"] = run_parametric_arm(
                    block_assignment, cache_data, p_basis, arm_type=f"BOTTOM-k{k}", r=r_val, epochs=30, lr=1e-3, seeds=seeds, ogp_k=k)

            print_suite_table(suite_results, title=f"PARAMETRIC MEMORY BENCHMARK -- {seed_set_name} (r={r_val})")


if __name__ == "__main__":
    main()
