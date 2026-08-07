"""
dump_c2_raw_data.py  --  Dump Raw C2 Arrays & Verify Unweighted Means
====================================================================

Dumps raw step-9 R matrix vectors R[9, 0:10], shuffle orders, and base/sequential masks
for all 50 selection runs {101..105} and 50 fresh runs {211..215} to c2_raw_arrays.json.
Computes sha256 checksum and exact unweighted means directly from the dumped file.
"""

import os
import json
import random
import hashlib
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


def generate_c2_raw_runs(block_assignment, cache_data, pca_basis_r32, seeds, num_shuffles=10):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)
    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    raw_runs = []

    for run_idx, order in enumerate(order_list):
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

            all_train_x = torch.cat([tr_x[b] for b in order], dim=0).to(DEVICE)
            all_train_y = torch.cat([tr_y[b] for b in order], dim=0).to(DEVICE)

            adapter.eval()
            with torch.no_grad():
                z_refs_all = adapter(all_train_x)
                r9_vec = []
                for b in range(10):
                    test_x_b = te_x[b].to(DEVICE)
                    test_y_b = te_y[b].to(DEVICE)
                    z_queries = adapter(test_x_b)
                    correct = sum(
                        1 for q_idx, q_vec in enumerate(z_queries)
                        if all_train_y[torch.argmax(torch.matmul(z_refs_all, q_vec)).item()].item() == test_y_b[q_idx].item()
                    )
                    r9_vec.append(float(correct / len(z_queries)))

            raw_runs.append({
                "seed": seed,
                "shuffle_order": order,
                "base_blocks": order[:5],
                "never_trained_blocks": order[5:],
                "R9_vec": r9_vec
            })

    return raw_runs


def main():
    print("=" * 80)
    print("  GENERATING RAW C2 ARRAYS FOR SELECTION AND FRESH SEED SETS")
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

    sel_runs   = generate_c2_raw_runs(block_assignment, cache_data, pca_basis_r32, seeds=[101, 102, 103, 104, 105])
    fresh_runs = generate_c2_raw_runs(block_assignment, cache_data, pca_basis_r32, seeds=[211, 212, 213, 214, 215])

    dump_data = {
        "selection_seeds_101_105": sel_runs,
        "fresh_seeds_211_215": fresh_runs
    }

    out_file = "c2_raw_arrays.json"
    with open(out_file, "w") as f:
        json.dump(dump_data, f, indent=2)

    with open(out_file, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\n  [Saved] Raw arrays dumped to {out_file}")
    print(f"  [SHA256] {sha256_hash}")

    # Compute unweighted means by reading the file
    with open(out_file, "r") as f:
        loaded = json.load(f)

    for set_key in ["selection_seeds_101_105", "fresh_seeds_211_215"]:
        base_means = []
        never_means = []
        overall_means = []
        for run in loaded[set_key]:
            order = run["shuffle_order"]
            r9 = run["R9_vec"]
            base_acc = np.mean([r9[b] for b in order[:5]]) * 100.0
            never_acc = np.mean([r9[b] for b in order[5:]]) * 100.0
            overall_at = np.mean(r9) * 100.0
            base_means.append(base_acc)
            never_means.append(never_acc)
            overall_means.append(overall_at)

        print(f"\n  Direct File Read Unweighted Means for {set_key}:")
        print(f"    Base Blocks order[0:5]:  {np.mean(base_means):.2f}% ± {np.std(base_means):.2f}%")
        print(f"    Never Blocks order[5:10]: {np.mean(never_means):.2f}% ± {np.std(never_means):.2f}%")
        print(f"    Unweighted Mean:        {(np.mean(base_means) + np.mean(never_means))/2.0:.2f}%")
        print(f"    Measured Overall A_T:   {np.mean(overall_means):.2f}% ± {np.std(overall_means):.2f}%")


if __name__ == "__main__":
    main()
