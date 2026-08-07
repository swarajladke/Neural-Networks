"""
run_section10_final_verification.py  --  Section 10 Final Verification & Diagnostics
====================================================================================

Executes:
  1. Full distribution (min, mean, max, p95, p99) of all 4,950 adapted cosine similarities.
  2. Scale-invariant Margin Test on all 400 test queries:
       m = cos(q, correct) - max_{y != correct} cos(q, nearest_incorrect)
     Report distribution and fraction m < 0 (error rate) split by Trained (200) vs Untrained (200).
  3. Contradiction Resolution: Of the failing test queries after adaptation, how many have their
     nearest incorrect neighbour in a pair that was >0.95 before adaptation?
  4. Measure max |W_seed101 - W_seed102| at Step 9 in 50-run benchmark with per-seed shuffle order.
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


def main():
    print("=" * 80)
    print("  SECTION 10 FINAL DIAGNOSTICS & VERIFICATION")
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
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    conf_pair_set = set()
    for f1, f2, _ in conf_pairs:
        conf_pair_set.add((min(f1, f2), max(f1, f2)))

    # Compute unadapted centroids
    cen_raw = torch.zeros(100, INPUT_DIM, device=DEVICE)
    for i in range(100):
        samples = cache_data["train_x"][i*3:(i+1)*3].float().to(DEVICE)
        cen_raw[i] = F.normalize(samples.mean(0, keepdim=True), dim=-1).squeeze(0)

    S_raw = torch.matmul(cen_raw, cen_raw.T)

    # Extract 4,950 upper-triangular pair similarities for raw
    raw_pairs = []
    for i in range(100):
        for j in range(i + 1, 100):
            raw_pairs.append(S_raw[i, j].item())
    raw_pairs = np.array(raw_pairs)

    # Train adapter on 50 base facts (blocks 0..4)
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

    adapted_pairs = []
    for i in range(100):
        for j in range(i + 1, 100):
            adapted_pairs.append(S_adapted[i, j].item())
    adapted_pairs = np.array(adapted_pairs)

    print("\n  1. COSINE SIMILARITY DISTRIBUTION (4,950 PAIRS):")
    print(f"     RAW BASELINE: Min={np.min(raw_pairs):.4f}, Mean={np.mean(raw_pairs):.4f}, Max={np.max(raw_pairs):.4f}, p95={np.percentile(raw_pairs, 95):.4f}, p99={np.percentile(raw_pairs, 99):.4f}")
    print(f"     ADAPTED MAP:  Min={np.min(adapted_pairs):.4f}, Mean={np.mean(adapted_pairs):.4f}, Max={np.max(adapted_pairs):.4f}, p95={np.percentile(adapted_pairs, 95):.4f}, p99={np.percentile(adapted_pairs, 99):.4f}")

    # 2. Margin Test on all 400 test queries
    all_train_x = torch.cat([tr_x[b] for b in range(10)], dim=0).to(DEVICE)
    all_train_y = torch.cat([tr_y[b] for b in range(10)], dim=0).to(DEVICE)

    def compute_margin_stats(is_adapted=True):
        trained_margins, untrained_margins = [], []
        trained_failures_confusable, untrained_failures_confusable = 0, 0
        trained_total_failures, untrained_total_failures = 0, 0

        with torch.no_grad():
            if is_adapted:
                z_refs = adapter(all_train_x)
            else:
                z_refs = F.normalize(all_train_x, dim=-1)

            for b in range(10):
                test_x_b = te_x[b].to(DEVICE)
                test_y_b = te_y[b].to(DEVICE)

                if is_adapted:
                    z_queries = adapter(test_x_b)
                else:
                    z_queries = F.normalize(test_x_b, dim=-1)

                sims = torch.matmul(z_queries, z_refs.T)

                for q_idx in range(len(test_y_b)):
                    correct_class = test_y_b[q_idx].item()
                    query_sims = sims[q_idx]

                    # Mask out all reference vectors of the correct class to find nearest incorrect
                    correct_mask = (all_train_y == correct_class)
                    sim_correct = query_sims[correct_mask].max().item()

                    incorrect_mask = ~correct_mask
                    nearest_incorrect_idx = query_sims[incorrect_mask].argmax().item()
                    sim_nearest_incorrect = query_sims[incorrect_mask][nearest_incorrect_idx].item()
                    nearest_incorrect_class = all_train_y[incorrect_mask][nearest_incorrect_idx].item()

                    m = sim_correct - sim_nearest_incorrect
                    is_trained_block = (b < 5)

                    if is_trained_block:
                        trained_margins.append(m)
                    else:
                        untrained_margins.append(m)

                    if m < 0: # Failure query
                        if is_trained_block:
                            trained_total_failures += 1
                        else:
                            untrained_total_failures += 1

                        # Check if pair (correct_class, nearest_incorrect_class) was in conf_pair_set (>0.95 raw)
                        p_pair = (min(correct_class, nearest_incorrect_class), max(correct_class, nearest_incorrect_class))
                        if p_pair in conf_pair_set:
                            if is_trained_block:
                                trained_failures_confusable += 1
                            else:
                                untrained_failures_confusable += 1

        return {
            "trained_m": np.array(trained_margins),
            "untrained_m": np.array(untrained_margins),
            "trained_fail_total": trained_total_failures,
            "trained_fail_confusable": trained_failures_confusable,
            "untrained_fail_total": untrained_total_failures,
            "untrained_fail_confusable": untrained_failures_confusable
        }

    raw_margin = compute_margin_stats(is_adapted=False)
    ad_margin  = compute_margin_stats(is_adapted=True)

    print("\n  2. SCALE-INVARIANT MARGIN TEST m = cos(q, correct) - cos(q, nearest_incorrect):")
    print("     [UNADAPTED BASELINE (400 Test Queries)]")
    print(f"       Trained (200 q):   Mean m = {np.mean(raw_margin['trained_m']):.4f} ± {np.std(raw_margin['trained_m']):.4f}, Fraction m < 0 = {np.mean(raw_margin['trained_m'] < 0)*100:.2f}% (Error Rate)")
    print(f"       Untrained (200 q): Mean m = {np.mean(raw_margin['untrained_m']):.4f} ± {np.std(raw_margin['untrained_m']):.4f}, Fraction m < 0 = {np.mean(raw_margin['untrained_m'] < 0)*100:.2f}% (Error Rate)")
    print("     [ADAPTED REPRESENTATION MAP (50 Base Facts)]")
    print(f"       Trained (200 q):   Mean m = {np.mean(ad_margin['trained_m']):.4f} ± {np.std(ad_margin['trained_m']):.4f}, Fraction m < 0 = {np.mean(ad_margin['trained_m'] < 0)*100:.2f}% (Error Rate: {100 - np.mean(ad_margin['trained_m'] >= 0)*100:.2f}%)")
    print(f"       Untrained (200 q): Mean m = {np.mean(ad_margin['untrained_m']):.4f} ± {np.std(ad_margin['untrained_m']):.4f}, Fraction m < 0 = {np.mean(ad_margin['untrained_m'] < 0)*100:.2f}% (Error Rate: {100 - np.mean(ad_margin['untrained_m'] >= 0)*100:.2f}%)")

    print("\n  3. AUDIT OF RETRIEVAL FAILURES AFTER ADAPTATION:")
    print(f"     Trained Block Failures:   {ad_margin['trained_fail_total']} total failures.")
    print(f"       Nearest incorrect in formerly confusable (>0.95) pair: {ad_margin['trained_fail_confusable']} / {ad_margin['trained_fail_total']} ({ad_margin['trained_fail_confusable'] / max(1, ad_margin['trained_fail_total'])*100:.1f}%)")
    print(f"     Untrained Block Failures: {ad_margin['untrained_fail_total']} total failures.")
    print(f"       Nearest incorrect in formerly confusable (>0.95) pair: {ad_margin['untrained_fail_confusable']} / {ad_margin['untrained_fail_total']} ({ad_margin['untrained_fail_confusable'] / max(1, ad_margin['untrained_fail_total'])*100:.1f}%)")

    # 4. Measure max |W_seed101 - W_seed102| at Step 9 in 50-run benchmark
    W_step9 = {}
    for seed in [101, 102]:
        random.seed(seed)
        order = list(range(10)); random.shuffle(order)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

        for step in range(10):
            seen_blocks = order[:step+1]
            tx = torch.cat([tr_x[b] for b in seen_blocks], dim=0).to(DEVICE)
            ty = torch.cat([tr_y[b] for b in seen_blocks], dim=0).to(DEVICE)
            adapter.train()
            num_epochs = 100 if step == 0 else 100 # naive
            for _ in range(num_epochs):
                proj = adapter(tx)
                loss = supervised_contrastive_loss(proj, ty)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        adapter.eval()
        W_step9[seed] = adapter.get_effective_W().detach().cpu()

    diff_step9 = torch.max(torch.abs(W_step9[101] - W_step9[102])).item()
    print(f"\n  4. BENCHMARK STEP-9 WEIGHT DIFFERENCE ACROSS SEEDS:")
    print(f"     max |W_seed101 - W_seed102| at Step 9 = {diff_step9:.6f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
