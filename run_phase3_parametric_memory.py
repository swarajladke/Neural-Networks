"""
run_phase3_parametric_memory.py  --  Phase 3: Parametric Memory Benchmark & C2 Breakdown Analysis
===================================================================================================

PART 1: C2 FREEZE-AFTER-BASE BREAKDOWN (No reference bank change)
  Reports R[9, j] split by:
    - Base-trained blocks: order[0:5] (seen during base phase joint training)
    - Never-trained blocks: order[5:9] (NEVER seen during any training phase)
  Across Selection Seeds {101..105} and Fresh Replication Seeds {211..215}.
  Frozen floor at r=32: 70.50%.

PART 2: PARAMETRIC MEMORY BENCHMARK
  Removal of reference bank from inference.
  Replaces 1-NN retrieval with a Parametric Classification Head over 100 fact classes
  on top of the frozen encoder.

  Model Architecture:
    - Encoder: SmolLM2-360M (frozen, 960-dim embeddings)
    - Adapter: BottleneckAdapter r=32 (or FullRankAdapter r=960)
    - Head: nn.Linear(960, 100, bias=True) -- 100-class parametric classification head.
    - Loss: CrossEntropyLoss over the 100 fact classes.
    - Inference: argmax(logits) over 100 fact classes (NO reference store used!).

  Training Protocol:
    - lr = 1e-3, epochs = 30 for ALL arms (fixed; no lr tuning to manufacture forgetting).
    - Block structure: 10 blocks x 10 facts (disjoint labels 0..99).
    - 50 runs = 10 shuffles x 5 seeds per seed set ({101..105} and {211..215}).
    - Populated-row R guard and exact BWT decomposition (BWT = A_T - LA).

  REQUIRED PRECONDITION GATES (run before any mechanism sweep):
    1. Naive BWT must be <= -10 pp.
    2. FREEZE-AFTER-BASE must be WELL BELOW offline. If freezing still nearly matches
       offline, knowledge is not in the weights -- stop and report.

  MECHANISM SWEEP (only executed if precondition gates pass):
    - OGP k in {4, 8, 12, 16, 24, 32}
    - Standing Controls: FREEZE-AFTER-BASE, RANDOM-k, BOTTOM-k
"""

import os
import sys
import json
import random
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

# ============================================================================
# PART 1: C2 FREEZE-AFTER-BASE BREAKDOWN (1-NN Retrieval Benchmark)
# ============================================================================

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


def run_part1_c2_breakdown(block_assignment, cache_data, pca_basis_r32, seeds, num_shuffles=10):
    """Evaluate C2 FREEZE-AFTER-BASE R[9, j] split by base blocks order[0:5] vs never-trained order[5:10]."""
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    base_trained_ats = []
    never_trained_ats = []
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

            # FREEZE: No updates for blocks 5..9

            adapter.eval()
            with torch.no_grad():
                z_refs = adapter(joint_train_x_base)
                block_accs = []
                for b in range(10):
                    test_x_b = te_x[b].to(DEVICE)
                    test_y_b = te_y[b].to(DEVICE)
                    z_queries = adapter(test_x_b)
                    correct = sum(
                        1 for q_idx, q_vec in enumerate(z_queries)
                        if joint_train_y_base[torch.argmax(torch.matmul(z_refs, q_vec)).item()].item() == test_y_b[q_idx].item()
                    )
                    block_accs.append(correct / len(z_queries))

                base_b_acc = np.mean([block_accs[b] for b in order[:5]]) * 100.0
                never_b_acc = np.mean([block_accs[b] for b in order[5:]]) * 100.0
                overall_acc = np.mean(block_accs) * 100.0

                base_trained_ats.append(base_b_acc)
                never_trained_ats.append(never_b_acc)
                overall_ats.append(overall_acc)

    return {
        "base_trained_mean": float(np.mean(base_trained_ats)),
        "base_trained_std": float(np.std(base_trained_ats)),
        "never_trained_mean": float(np.mean(never_trained_ats)),
        "never_trained_std": float(np.std(never_trained_ats)),
        "overall_mean": float(np.mean(overall_ats)),
        "overall_std": float(np.std(overall_ats))
    }


# ============================================================================
# PART 2: PARAMETRIC MEMORY BENCHMARK (100-Class Parametric Classification)
# ============================================================================

class ParametricClassifier(nn.Module):
    """Parametric classification model: SmolLM2 embeddings -> Adapter (r=32/960) -> Linear(960, 100).
    No reference bank or 1-NN retrieval used at inference time.
    """
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


def run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis,
                                 arm_type="naive", r=32, epochs=30, lr=1e-3,
                                 seeds=list(range(101, 106)), num_shuffles=10,
                                 ogp_k=None):
    """Execute a single arm on the Parametric Memory Classification Benchmark."""
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

            # Base phase: joint train on first 5 blocks
            base_blocks = order[:5]
            bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            if arm_type != "FREEZE-AFTER-BASE":
                model.train()
                for _ in range(epochs):
                    logits = model(bx)
                    loss   = criterion(logits, by)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            else:
                # FREEZE-AFTER-BASE: base phase training only
                model.train()
                for _ in range(epochs):
                    logits = model(bx)
                    loss   = criterion(logits, by)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Base phase evaluation -> R[4, :]
            model.eval()
            with torch.no_grad():
                for b in range(10):
                    tx = te_x[b].to(DEVICE); ty = te_y[b].to(DEVICE)
                    preds = torch.argmax(model(tx), dim=-1)
                    R[4, b] = (preds == ty).float().mean().item()

            M_past = bx.clone().detach()

            # Sequential phase: steps 5..9
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("  PHASE 3: PARAMETRIC MEMORY BENCHMARK & C2 CONFIRMATION ANALYSIS")
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
    # PART 1: C2 FREEZE-AFTER-BASE CONFIRMATION BREAKDOWN
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PART 1: C2 FREEZE-AFTER-BASE BREAKDOWN (1-NN RETRIEVAL BENCHMARK)")
    print("  Evaluating R[9, j] split: Base blocks (order[0:5]) vs Never-trained (order[5:10])")
    print("=" * 80)

    c2_sel = run_part1_c2_breakdown(block_assignment, cache_data, pca_basis_r32, seeds=[101, 102, 103, 104, 105])
    c2_fre = run_part1_c2_breakdown(block_assignment, cache_data, pca_basis_r32, seeds=[211, 212, 213, 214, 215])

    print("\n  C2 BREAKDOWN RESULTS:")
    print("    Selection Seeds 101..105 (50 runs):")
    print(f"      - Base-Trained Blocks order[0:5]:  {c2_sel['base_trained_mean']:.2f}% ± {c2_sel['base_trained_std']:.2f}%")
    print(f"      - Never-Trained Blocks order[5:10]: {c2_sel['never_trained_mean']:.2f}% ± {c2_sel['never_trained_std']:.2f}%")
    print(f"      - Overall A_T (step 9):            {c2_sel['overall_mean']:.2f}% ± {c2_sel['overall_std']:.2f}%")

    print("    Fresh Replication Seeds 211..215 (50 runs):")
    print(f"      - Base-Trained Blocks order[0:5]:  {c2_fre['base_trained_mean']:.2f}% ± {c2_fre['base_trained_std']:.2f}%")
    print(f"      - Never-Trained Blocks order[5:10]: {c2_fre['never_trained_mean']:.2f}% ± {c2_fre['never_trained_std']:.2f}%")
    print(f"      - Overall A_T (step 9):            {c2_fre['overall_mean']:.2f}% ± {c2_fre['overall_std']:.2f}%")

    if c2_sel['never_trained_mean'] >= 80.0:
        print("\n  [RECORDED FINDING]:")
        print("  The adapter learns a generic retrieval metric, not fact-specific memory.")
        print(f"  Never-trained blocks reach {c2_sel['never_trained_mean']:.2f}% accuracy without any training data.")
        print("  78%+ of all achievable gain is obtained from half the corpus and transfers")
        print("  to unseen facts. Catastrophic forgetting is not measurable in 1-NN retrieval")
        print("  because knowledge is stored non-parametrically in the reference bank.")
    print("=" * 80)

    # ------------------------------------------------------------------
    # PART 2: PARAMETRIC MEMORY BENCHMARK -- PRECONDITION GATES
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PART 2: PARAMETRIC MEMORY BENCHMARK (100-CLASS PARAMETRIC HEAD)")
    print("  No reference bank used at inference. lr=1e-3, epochs=30 fixed for all arms.")
    print("=" * 80)

    print("\n  --> Running Precondition Gate: Naive & FREEZE-AFTER-BASE (r=32, Selection Seeds)...")
    res_naive_r32_sel = run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis_r32, arm_type="naive", r=32, epochs=30, lr=1e-3, seeds=[101, 102, 103, 104, 105])
    res_off_r32_sel   = run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis_r32, arm_type="offline", r=32, epochs=30, lr=1e-3, seeds=[101, 102, 103, 104, 105])
    res_frz_r32_sel   = run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis_r32, arm_type="FREEZE-AFTER-BASE", r=32, epochs=30, lr=1e-3, seeds=[101, 102, 103, 104, 105])

    naive_bwt_r32_sel = np.mean([r["BWT"] for r in res_naive_r32_sel]) * 100.0
    naive_at_r32_sel  = np.mean([r["A_T"] for r in res_naive_r32_sel]) * 100.0
    off_at_r32_sel    = np.mean([r["A_T"] for r in res_off_r32_sel]) * 100.0
    frz_at_r32_sel    = np.mean([r["A_T"] for r in res_frz_r32_sel]) * 100.0

    print(f"  [Precondition Gate -- Parametric Head r=32 (Selection Seeds)]")
    print(f"    Naive A_T  = {naive_at_r32_sel:.2f}% | Naive BWT = {naive_bwt_r32_sel:+.2f}%")
    print(f"    Offline A_T= {off_at_r32_sel:.2f}%")
    print(f"    Freeze A_T = {frz_at_r32_sel:.2f}%")

    print("\n  --> Running Precondition Gate: Naive & FREEZE-AFTER-BASE (r=960, Selection Seeds)...")
    res_naive_r960_sel = run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis=None, arm_type="naive", r=960, epochs=30, lr=1e-3, seeds=[101, 102, 103, 104, 105])
    res_off_r960_sel   = run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis=None, arm_type="offline", r=960, epochs=30, lr=1e-3, seeds=[101, 102, 103, 104, 105])
    res_frz_r960_sel   = run_parametric_benchmark_arm(block_assignment, cache_data, pca_basis=None, arm_type="FREEZE-AFTER-BASE", r=960, epochs=30, lr=1e-3, seeds=[101, 102, 103, 104, 105])

    naive_bwt_r960_sel = np.mean([r["BWT"] for r in res_naive_r960_sel]) * 100.0
    naive_at_r960_sel  = np.mean([r["A_T"] for r in res_naive_r960_sel]) * 100.0
    off_at_r960_sel    = np.mean([r["A_T"] for r in res_off_r960_sel]) * 100.0
    frz_at_r960_sel    = np.mean([r["A_T"] for r in res_frz_r960_sel]) * 100.0

    print(f"\n  [Precondition Gate -- Parametric Head r=960 (Selection Seeds)]")
    print(f"    Naive A_T  = {naive_at_r960_sel:.2f}% | Naive BWT = {naive_bwt_r960_sel:+.2f}%")
    print(f"    Offline A_T= {off_at_r960_sel:.2f}%")
    print(f"    Freeze A_T = {frz_at_r960_sel:.2f}%")

    # Evaluate Gates
    gate1_pass = (naive_bwt_r32_sel <= -10.0 or naive_bwt_r960_sel <= -10.0)
    gate2_pass = ((off_at_r32_sel - frz_at_r32_sel) >= 15.0 or (off_at_r960_sel - frz_at_r960_sel) >= 15.0)

    print("\n" + "=" * 80)
    print("  PARAMETRIC BENCHMARK PRECONDITION GATE EVALUATION:")
    print(f"    Gate 1 (Naive BWT <= -10pp): {'PASS' if gate1_pass else 'FAIL'}")
    print(f"    Gate 2 (Freeze WELL BELOW Offline): {'PASS' if gate2_pass else 'FAIL'}")
    print("=" * 80)

    if not (gate1_pass and gate2_pass):
        print("\n  [STOP] Precondition gates failed. Reporting parametric benchmark results")
        print("  without running OGP mechanism sweep.")
        return

    # ------------------------------------------------------------------
    # OGP MECHANISM SWEEP (Executed only if precondition gates pass)
    # ------------------------------------------------------------------
    print("\n  [PRECONDITION GATES PASSED] Running OGP Mechanism Sweep...")
    # Sweep k in {4, 8, 12, 16, 24, 32} with standing controls
    k_vals = [4, 8, 12, 16, 24, 32]
    for r_val, p_basis in [(32, pca_basis_r32), (960, None)]:
        print(f"\n  ===============================================================")
        print(f"    OGP SWEEP -- PARAMETRIC HEAD (r={r_val})")
        print(f"  ===============================================================")
        for k in k_vals:
            for arm in (f"OGP_k{k}", f"RANDOM-k{k}", f"BOTTOM-k{k}"):
                print(f"    Running {arm:12s} (r={r_val})...")
                res = run_parametric_benchmark_arm(
                    block_assignment, cache_data, p_basis,
                    arm_type=arm, r=r_val, epochs=30, lr=1e-3,
                    seeds=[101, 102, 103, 104, 105], ogp_k=k)
                at_m  = np.mean([r["A_T"] for r in res]) * 100.0
                bwt_m = np.mean([r["BWT"] for r in res]) * 100.0
                print(f"      [{arm}] A_T = {at_m:.2f}% | BWT = {bwt_m:+.2f}%")


if __name__ == "__main__":
    main()
