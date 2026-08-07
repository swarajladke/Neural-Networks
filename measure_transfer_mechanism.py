"""
measure_transfer_mechanism.py  --  Quantify C2 Adapter Metric Transfer Mechanism
==============================================================================

Measures:
  1. Residual of adapter matrix W = U V against isotropic scaling + rank-1 shift (a*I + b*11^T).
  2. Singular spectrum of W (top 10 singular values, condition number, effective rank).
  3. Never-trained block retrieval accuracy at base training sizes 10, 20, and 50 facts.
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
        # W = U.weight @ V.weight  (shape 960 x 960)
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


def fit_isotropic_rank1_decomposition(W):
    """Fit a*I + b*(1 1^T) to matrix W via least squares."""
    d = W.shape[0]
    I_mat = torch.eye(d, device=W.device)
    J_mat = torch.ones(d, d, device=W.device)

    # Solve min || W - (a*I + b*J) ||_F^2
    # tr(I^T I) = d, tr(J^T J) = d^2, tr(I^T J) = d
    # System: [d, d^2; d, d^2] [a; b] = [tr(W); sum(W)]
    tr_W  = torch.trace(W).item()
    sum_W = torch.sum(W).item()

    # Form 2x2 linear system
    # d*a + d*b = tr_W
    # d*a + d^2*b = sum_W
    # Subtracting: d*(d - 1)*b = sum_W - tr_W
    b = (sum_W - tr_W) / (d * (d - 1))
    a = (tr_W - d * b) / d

    W_approx = a * I_mat + b * J_mat
    residual = W - W_approx
    res_norm = torch.norm(residual).item()
    W_norm   = torch.norm(W).item()
    rel_res  = res_norm / W_norm
    var_exp  = 1.0 - (res_norm**2) / (W_norm**2)

    return a, b, rel_res, var_exp, residual


def measure_singular_spectrum(W):
    """Compute singular values of W = U V."""
    S = torch.linalg.svdvals(W)
    S_np = S.cpu().numpy()
    top_10 = S_np[:10]
    cond_num = float(S_np[0] / S_np[31]) if S_np[31] > 1e-12 else float('inf')

    # Effective rank
    p = S_np[:32] / np.sum(S_np[:32])
    p = p[p > 1e-12]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    return top_10, cond_num, eff_rank


def measure_transfer_vs_base_size(block_assignment, cache_data, pca_basis_r32, seeds=[101, 102, 103, 104, 105]):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    results = {10: [], 20: [], 50: []}

    all_train_x = torch.cat([tr_x[b] for b in range(10)], dim=0).to(DEVICE)
    all_train_y = torch.cat([tr_y[b] for b in range(10)], dim=0).to(DEVICE)

    for num_base_facts, num_blocks in [(10, 1), (20, 2), (50, 5)]:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

            base_blocks = list(range(num_blocks))
            joint_train_x_base = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            joint_train_y_base = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            adapter.train()
            for _ in range(100):
                proj = adapter(joint_train_x_base)
                loss = supervised_contrastive_loss(proj, joint_train_y_base)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            adapter.eval()
            with torch.no_grad():
                z_refs_all = adapter(all_train_x)
                never_accs = []
                for b in range(num_blocks, 10):
                    test_x_b = te_x[b].to(DEVICE)
                    test_y_b = te_y[b].to(DEVICE)
                    z_queries = adapter(test_x_b)
                    correct = sum(
                        1 for q_idx, q_vec in enumerate(z_queries)
                        if all_train_y[torch.argmax(torch.matmul(z_refs_all, q_vec)).item()].item() == test_y_b[q_idx].item()
                    )
                    never_accs.append(correct / len(z_queries))

                results[num_base_facts].append(np.mean(never_accs) * 100.0)

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


def main():
    print("=" * 80)
    print("  MEASURING ADAPTER METRIC TRANSFER MECHANISM & WEIGHT STRUCTURE")
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

    # 1. Train a representative adapter on base phase (50 facts)
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

    tr_x, tr_y, _, _ = build_block_tensors(block_assignment, cache_data)
    joint_train_x_base = torch.cat([tr_x[b] for b in range(5)], dim=0).to(DEVICE)
    joint_train_y_base = torch.cat([tr_y[b] for b in range(5)], dim=0).to(DEVICE)

    adapter.train()
    for _ in range(100):
        proj = adapter(joint_train_x_base)
        loss = supervised_contrastive_loss(proj, joint_train_y_base)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    adapter.eval()
    W = adapter.get_effective_W()

    # Measure 1: Fit isotropic rank-1 decomposition a*I + b*(1 1^T)
    a, b, rel_res, var_exp, residual = fit_isotropic_rank1_decomposition(W)
    print("\n  1. ADAPTER WEIGHT MATRIX DECOMPOSITION (W = a*I + b*11^T + R):")
    print(f"     Isotropic Scale  a = {a:+.6f}")
    print(f"     Rank-1 Shift     b = {b:+.6f}")
    print(f"     Relative Residual  = {rel_res*100:.2f}%")
    print(f"     Variance Explained = {var_exp*100:.2f}%")

    # Measure 2: Singular Spectrum
    top_10, cond_num, eff_rank = measure_singular_spectrum(W)
    print("\n  2. SINGULAR SPECTRUM OF EFFECTIVE WEIGHT W (r=32):")
    print(f"     Top 10 Singular Values: {np.array2string(top_10, precision=4, suppress_small=True)}")
    print(f"     Condition Number (sigma_1 / sigma_32): {cond_num:.2f}")
    print(f"     Effective Rank:                         {eff_rank:.2f}")

    # Measure 3: Never-Trained Accuracy vs Base Phase Size (10, 20, 50 facts)
    transfer_by_size = measure_transfer_vs_base_size(block_assignment, cache_data, pca_basis_r32)
    print("\n  3. NEVER-TRAINED BLOCK ACCURACY vs BASE PHASE SIZE:")
    for num_facts, (mean_acc, std_acc) in transfer_by_size.items():
        print(f"     Base Size = {num_facts:2d} facts ({num_facts//10:d} blocks): Never-Trained Accuracy = {mean_acc:5.2f}% ± {std_acc:4.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
