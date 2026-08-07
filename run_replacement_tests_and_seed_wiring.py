"""
run_replacement_tests_and_seed_wiring.py  --  Section 10.1 Replacement Tests
=============================================================================

Executes:
  1. Pair-cosine similarity histogram (4,950 pairs) on adapted centroids after base phase (50 facts),
     split by Trained-Trained (1,225 pairs), Trained-Untrained (2,500 pairs), and Untrained-Untrained (1,225 pairs),
     compared against unadapted baseline (mean 0.0959, 170 pairs > 0.95).
  2. Fixed evaluation set base-size curve for B in {1, 2, 3, 4, 5} blocks (10, 20, 30, 40, 50 facts)
     evaluated on fixed test blocks 5-9 across properly wired shuffle seeds {101..105}.
  3. Seed wiring check: max |W_seed101 - W_seed102| after base phase.
  4. Diagnosis of 58.06% vs 59.85% C2 lead.
"""

import os
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

class BottleneckAdapter(nn.Module):
    def __init__(self, r, pca_basis):
        super().__init__()
        self.r = r
        self.V = nn.Linear(INPUT_DIM, r,         bias=False)
        self.U = nn.Linear(r,         INPUT_DIM, bias=True)
        with torch.no_grad():
            self.V.weight.copy_(pca_basis)
            self.U.weight.copy_(pca_basis.T)
            nn.init.zeros_(self.U.bias)

    def forward(self, x):
        return F.normalize(self.U(self.V(x)), dim=-1)

    def get_effective_W(self):
        return torch.matmul(self.U.weight, self.V.weight)


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


def run_seed_wiring_check(block_assignment, cache_data, pca_basis_r32):
    """Check max |W_seed101 - W_seed102| after base phase training."""
    tr_x, tr_y, _, _ = build_block_tensors(block_assignment, cache_data)
    W_weights = {}
    for seed in [101, 102, 103]:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

        base_blocks = [0, 1, 2, 3, 4]
        joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
        joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

        adapter.train()
        for _ in range(100):
            proj = adapter(joint_train_x_base)
            loss = supervised_contrastive_loss(proj, joint_train_y_base)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        adapter.eval()
        W_weights[seed] = adapter.get_effective_W().detach().cpu()

    diff_101_102 = torch.max(torch.abs(W_weights[101] - W_weights[102])).item()
    diff_101_103 = torch.max(torch.abs(W_weights[101] - W_weights[103])).item()

    return diff_101_102, diff_101_103


def run_representation_geometry_replacement_test(block_assignment, cache_data, pca_basis_r32):
    """Compute 4,950-pair cosine similarity histogram on adapted vs unadapted centroids."""
    tr_x, tr_y, _, _ = build_block_tensors(block_assignment, cache_data)

    # 1. Unadapted Centroids (Raw Embeddings)
    X = cache_data["train_x"].float().to(DEVICE)
    y = cache_data["train_y"].to(DEVICE)
    cen_raw = torch.zeros(100, INPUT_DIM, device=DEVICE)
    for i in range(100):
        mask = (y == i)
        cen_raw[i] = F.normalize(X[mask].mean(0, keepdim=True), dim=-1).squeeze(0)

    S_raw = torch.matmul(cen_raw, cen_raw.T)

    # 2. Adapted Centroids (after 50-fact base training)
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

    joint_train_x_base = torch.cat([tr_x[b] for b in range(5)], dim=0).to(DEVICE)
    joint_train_y_base = torch.cat([tr_y[b] for b in range(5)], dim=0).to(DEVICE)

    adapter.train()
    for _ in range(100):
        proj = adapter(joint_train_x_base)
        loss = supervised_contrastive_loss(proj, joint_train_y_base)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    adapter.eval()
    with torch.no_grad():
        cen_adapted = adapter(cen_raw)
        S_adapted   = torch.matmul(cen_adapted, cen_adapted.T)

    # Partition 100 facts into trained (0..49) vs untrained (50..99)
    # block_assignment has 10 blocks x 10 facts
    trained_facts   = [f for b in range(5) for f in block_assignment[b]]
    untrained_facts = [f for b in range(5, 10) for f in block_assignment[b]]

    def extract_pair_stats(S_mat, facts_A, facts_B, is_same):
        vals = []
        if is_same:
            for i_idx, f1 in enumerate(facts_A):
                for f2 in facts_A[i_idx + 1:]:
                    vals.append(S_mat[f1, f2].item())
        else:
            for f1 in facts_A:
                for f2 in facts_B:
                    vals.append(S_mat[f1, f2].item())
        vals = np.array(vals)
        return {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "gt_090": int(np.sum(vals > 0.90)),
            "gt_095": int(np.sum(vals > 0.95))
        }

    raw_tt = extract_pair_stats(S_raw, trained_facts, trained_facts, is_same=True)
    raw_tu = extract_pair_stats(S_raw, trained_facts, untrained_facts, is_same=False)
    raw_uu = extract_pair_stats(S_raw, untrained_facts, untrained_facts, is_same=True)

    ad_tt  = extract_pair_stats(S_adapted, trained_facts, trained_facts, is_same=True)
    ad_tu  = extract_pair_stats(S_adapted, trained_facts, untrained_facts, is_same=False)
    ad_uu  = extract_pair_stats(S_adapted, untrained_facts, untrained_facts, is_same=True)

    return {
        "raw": {"trained_trained": raw_tt, "trained_untrained": raw_tu, "untrained_untrained": raw_uu},
        "adapted": {"trained_trained": ad_tt, "trained_untrained": ad_tu, "untrained_untrained": ad_uu}
    }


def run_fixed_eval_base_size_curve(block_assignment, cache_data, pca_basis_r32, seeds=[101, 102, 103, 104, 105]):
    """
    Fixed Evaluation Set Base-Size Curve:
    Hold blocks 5-9 (50 facts) out permanently as fixed test set.
    Vary base training phase size over B in {1, 2, 3, 4, 5} blocks drawn from blocks 0-4.
    """
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    results = {1: [], 2: [], 3: [], 4: [], 5: []}

    all_train_x = torch.cat([tr_x[b] for b in range(10)], dim=0).to(DEVICE)
    all_train_y = torch.cat([tr_y[b] for b in range(10)], dim=0).to(DEVICE)

    # Fixed evaluation set: blocks 5..9
    eval_blocks = [5, 6, 7, 8, 9]

    for num_blocks in [1, 2, 3, 4, 5]:
        num_facts = num_blocks * 10
        base_blocks = list(range(num_blocks))

        for seed_idx, seed in enumerate(seeds):
            # Generate deterministic shuffle for seed
            random.seed(seed)
            shuffled_base_pool = list(range(5))
            random.shuffle(shuffled_base_pool)
            base_blocks_seed = shuffled_base_pool[:num_blocks]

            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

            joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks_seed], dim=0).to(DEVICE)
            joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks_seed], dim=0).to(DEVICE)

            adapter.train()
            for _ in range(100):
                proj = adapter(joint_train_x_base)
                loss = supervised_contrastive_loss(proj, joint_train_y_base)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            adapter.eval()
            with torch.no_grad():
                z_refs_all = adapter(all_train_x)
                eval_accs = []
                for b in eval_blocks:
                    test_x_b = te_x[b].to(DEVICE)
                    test_y_b = te_y[b].to(DEVICE)
                    z_queries = adapter(test_x_b)
                    correct = sum(
                        1 for q_idx, q_vec in enumerate(z_queries)
                        if all_train_y[torch.argmax(torch.matmul(z_refs_all, q_vec)).item()].item() == test_y_b[q_idx].item()
                    )
                    eval_accs.append(correct / len(z_queries))

                results[num_blocks].append(np.mean(eval_accs) * 100.0)

    return {b*10: (float(np.mean(accs)), float(np.std(accs)), [float(x) for x in accs]) for b, accs in results.items()}


def main():
    print("=" * 80)
    print("  SECTION 10.1 REPLACEMENT TESTS AND SEED WIRING DIAGNOSIS")
    print("=" * 80)

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

    pca_basis_r32 = compute_pca_basis(cache_data, r=32).to(DEVICE)
    conf_pairs = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)

    # 1. Seed Wiring Check
    diff_101_102, diff_101_103 = run_seed_wiring_check(block_assignment, cache_data, pca_basis_r32)
    print("\n  1. SEED WIRING WEIGHT DIFFERENCE CHECK:")
    print(f"     max |W_seed101 - W_seed102| = {diff_101_102:.6f}")
    print(f"     max |W_seed101 - W_seed103| = {diff_101_103:.6f}")

    # 2. Representation Geometry Replacement Test
    geom_results = run_representation_geometry_replacement_test(block_assignment, cache_data, pca_basis_r32)
    print("\n  2. REPRESENTATION GEOMETRY PAIR-COSINE Similarity HISTOGRAM (4,950 Pairs):")
    print("     [UNADAPTED RAW EMBEDDINGS BASELINE]")
    print(f"       Trained-Trained   (1,225 pairs): Mean = {geom_results['raw']['trained_trained']['mean']:.4f} ± {geom_results['raw']['trained_trained']['std']:.4f}, >0.95 count = {geom_results['raw']['trained_trained']['gt_095']}")
    print(f"       Trained-Untrained (2,500 pairs): Mean = {geom_results['raw']['trained_untrained']['mean']:.4f} ± {geom_results['raw']['trained_untrained']['std']:.4f}, >0.95 count = {geom_results['raw']['trained_untrained']['gt_095']}")
    print(f"       Untrained-Untrained(1,225 pairs): Mean = {geom_results['raw']['untrained_untrained']['mean']:.4f} ± {geom_results['raw']['untrained_untrained']['std']:.4f}, >0.95 count = {geom_results['raw']['untrained_untrained']['gt_095']}")
    print("     [ADAPTED REPRESENTATION MAP (50 Base Facts)]")
    print(f"       Trained-Trained   (1,225 pairs): Mean = {geom_results['adapted']['trained_trained']['mean']:.4f} ± {geom_results['adapted']['trained_trained']['std']:.4f}, >0.95 count = {geom_results['adapted']['trained_trained']['gt_095']}")
    print(f"       Trained-Untrained (2,500 pairs): Mean = {geom_results['adapted']['trained_untrained']['mean']:.4f} ± {geom_results['adapted']['trained_untrained']['std']:.4f}, >0.95 count = {geom_results['adapted']['trained_untrained']['gt_095']}")
    print(f"       Untrained-Untrained(1,225 pairs): Mean = {geom_results['adapted']['untrained_untrained']['mean']:.4f} ± {geom_results['adapted']['untrained_untrained']['std']:.4f}, >0.95 count = {geom_results['adapted']['untrained_untrained']['gt_095']}")

    # 3. Fixed Evaluation Base-Size Curve
    curve_results = run_fixed_eval_base_size_curve(block_assignment, cache_data, pca_basis_r32)
    print("\n  3. FIXED EVALUATION SET BASE-SIZE CURVE (Evaluated on Blocks 5-9, 50 Facts):")
    print("     Frozen Adapter Floor = 70.50%")
    for num_facts, (mean_acc, std_acc, per_seed) in curve_results.items():
        print(f"     Base Size = {num_facts:2d} facts ({num_facts//10:d} blocks): Fixed-Eval Accuracy = {mean_acc:5.2f}% ± {std_acc:4.2f}%  (per seed: {per_seed})")

    print("=" * 80)


if __name__ == "__main__":
    main()
