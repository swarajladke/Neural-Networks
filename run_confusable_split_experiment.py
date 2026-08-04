"""
run_confusable_split_experiment.py — Confusable-Split vs Confusable-Together Experiment
========================================================================================
Identifies all confusable fact pairs (cosine > 0.95) and evaluates two block assignments:
  1. CONFUSABLE-SPLIT: Confusable fact pairs placed in DIFFERENT blocks.
  2. CONFUSABLE-TOGETHER: Confusable fact pairs placed in the SAME block.

Evaluates frozen_adapter, naive_sequential_adapter, and offline_adapter on both splits
(5 shuffles x 3 seeds) with Supervised Contrastive InfoNCE Loss (tau=0.05).

Reports the Continual Learning Gap (offline - naive) for each assignment with paired CIs.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_100_PATH = "smollm2_embeddings_100slots.pt" if os.path.exists("smollm2_embeddings_100slots.pt") else ("../smollm2_embeddings_100slots.pt" if os.path.exists("../smollm2_embeddings_100slots.pt") else "smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM = 960

class OutputAdapter(nn.Module):
    def __init__(self, dim=960):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        out = self.linear(x)
        return F.normalize(out, dim=-1)

def supervised_contrastive_loss(z, y, tau=0.05):
    sim_matrix = torch.matmul(z, z.T) / tau
    N = z.shape[0]
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & logits_mask
    
    logits_max, _ = torch.max(sim_matrix * logits_mask.float(), dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
    loss = -mean_log_prob_pos.mean()
    return loss

def find_confusable_pairs(cache_data):
    train_x = cache_data["train_x"]
    train_y = cache_data["train_y"]
    centroids = torch.zeros(100, 960)
    for i in range(100):
        mask = (train_y == i)
        centroids[i] = F.normalize(train_x[mask].mean(dim=0), dim=-1)
        
    sim_matrix = torch.matmul(centroids, centroids.T)
    confusable_pairs = []
    for i in range(100):
        for j in range(i+1, 100):
            sim = sim_matrix[i, j].item()
            if sim > 0.95:
                confusable_pairs.append((i, j, sim))
    return confusable_pairs

def build_confusable_block_assignments(confusable_pairs):
    # 100 facts (indices 0..99)
    # Split assignment: graph coloring / greedy distribution putting pairs in different blocks
    blocks_split = [[] for _ in range(10)]
    for i in range(100):
        blocks_split[i % 10].append(i)
        
    # Shuffle facts inside split assignment so confusable pairs are separated
    random.seed(42)
    for pair in confusable_pairs:
        f1, f2, _ = pair
        # Find block of f1 and f2
        b1 = [b for b in range(10) if f1 in blocks_split[b]][0]
        b2 = [b for b in range(10) if f2 in blocks_split[b]][0]
        if b1 == b2:
            # Swap f2 with a fact in a different block
            target_b = (b1 + 1) % 10
            for k in range(len(blocks_split[target_b])):
                swap_f = blocks_split[target_b][k]
                if swap_f not in [p[0] for p in confusable_pairs if p[1] == f1] and swap_f not in [p[1] for p in confusable_pairs if p[0] == f1]:
                    blocks_split[b1].remove(f2)
                    blocks_split[target_b].remove(swap_f)
                    blocks_split[b1].append(swap_f)
                    blocks_split[target_b].append(f2)
                    break

    # Together assignment: place confusable pairs in SAME block
    blocks_together = [[] for _ in range(10)]
    placed = set()
    b_curr = 0
    for pair in confusable_pairs:
        f1, f2, _ = pair
        if f1 not in placed and f2 not in placed:
            if len(blocks_together[b_curr]) <= 8:
                blocks_together[b_curr].extend([f1, f2])
                placed.add(f1)
                placed.add(f2)
            else:
                b_curr = (b_curr + 1) % 10
                blocks_together[b_curr].extend([f1, f2])
                placed.add(f1)
                placed.add(f2)
    for i in range(100):
        if i not in placed:
            for b in range(10):
                if len(blocks_together[b]) < 10:
                    blocks_together[b].append(i)
                    placed.add(i)
                    break

    return blocks_split, blocks_together

def run_experiment_on_blocks(block_assignment, cache_data, condition="frozen_adapter", shuffles=5, seeds=3, lr=1e-3):
    results = []
    random.seed(42)
    order_list = []
    for _ in range(shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)
        
    # Re-index train_x/y and test_x/y based on block_assignment
    train_x_blocks = []
    train_y_blocks = []
    test_x_blocks = []
    test_y_blocks = []
    
    for b in range(10):
        fact_ids = block_assignment[b]
        b_tr_x = torch.cat([cache_data["train_x"][f*3 : (f+1)*3] for f in fact_ids], dim=0)
        b_tr_y = torch.cat([cache_data["train_y"][f*3 : (f+1)*3] for f in fact_ids], dim=0)
        b_te_x = torch.cat([cache_data["test_x"][f*4 : (f+1)*4] for f in fact_ids], dim=0)
        b_te_y = torch.cat([cache_data["test_y"][f*4 : (f+1)*4] for f in fact_ids], dim=0)
        
        train_x_blocks.append(b_tr_x)
        train_y_blocks.append(b_tr_y)
        test_x_blocks.append(b_te_x)
        test_y_blocks.append(b_te_y)
        
    for shuffle_idx, order in enumerate(order_list):
        for seed in range(101, 101 + seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
            R = np.zeros((10, 10))
            
            base_blocks = order[:5]
            joint_train_x_base = torch.cat([train_x_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
            joint_train_y_base = torch.cat([train_y_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
            
            if condition == "offline_adapter":
                adapter.train()
                for epoch in range(15):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            elif condition == "naive_sequential_adapter":
                adapter.train()
                for epoch in range(10):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            # Record base phase recall R[4, b]
            adapter.eval()
            with torch.no_grad():
                z_refs_base = adapter(joint_train_x_base)
                for b in range(10):
                    test_x_b = test_x_blocks[b].to(DEVICE)
                    test_y_b = test_y_blocks[b].to(DEVICE)
                    z_queries = adapter(test_x_b)
                    
                    correct = 0
                    for q_idx, q_vec in enumerate(z_queries):
                        sims = torch.matmul(z_refs_base, q_vec.unsqueeze(0).T).squeeze(-1)
                        best_idx = torch.argmax(sims).item()
                        if joint_train_y_base[best_idx].item() == test_y_b[q_idx].item():
                            correct += 1
                    R[4, b] = correct / len(z_queries)
                    
            for step in range(5, 10):
                curr_block = order[step]
                seen_blocks = order[:step + 1]
                seen_ref_x = torch.cat([train_x_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                seen_ref_y = torch.cat([train_y_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                
                if condition == "naive_sequential_adapter":
                    adapter.train()
                    curr_x = train_x_blocks[curr_block].to(DEVICE)
                    curr_y = train_y_blocks[curr_block].to(DEVICE)
                    for epoch in range(10):
                        proj = adapter(curr_x)
                        loss = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                elif condition == "offline_adapter":
                    adapter.train()
                    for epoch in range(10):
                        proj = adapter(seen_ref_x)
                        loss = supervised_contrastive_loss(proj, seen_ref_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                adapter.eval()
                with torch.no_grad():
                    z_refs_step = adapter(seen_ref_x)
                    for b in range(10):
                        test_x_b = test_x_blocks[b].to(DEVICE)
                        test_y_b = test_y_blocks[b].to(DEVICE)
                        z_queries = adapter(test_x_b)
                        
                        correct = 0
                        for q_idx, q_vec in enumerate(z_queries):
                            sims = torch.matmul(z_refs_step, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_idx = torch.argmax(sims).item()
                            if seen_ref_y[best_idx].item() == test_y_b[q_idx].item():
                                correct += 1
                        R[step, b] = correct / len(z_queries)
                        
            if condition == "frozen_adapter":
                mean_r9 = np.mean(R[9, :])
                assert abs(mean_r9 - 0.7250) < 0.005, f"frozen_adapter R[9,:] = {mean_r9:.4f}, expected 0.7250"
                
            a_t = np.mean(R[9, :])
            la = np.mean([R[max(4, order.index(j)), j] for j in range(10)])
            fgt_j = [np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)]
            bwt_j = [R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]
            
            results.append({
                "condition": condition,
                "A_T": a_t,
                "LA": la,
                "mean_forgetting": np.mean(fgt_j),
                "mean_bwt": np.mean(bwt_j)
            })
            
    return results

def bootstrap_paired_ci(ats_1, ats_2, n_boot=10000, seed=42):
    n = min(len(ats_1), len(ats_2))
    diffs = np.array(ats_1[:n]) - np.array(ats_2[:n])
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    return np.mean(diffs), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def main():
    print("="*90)
    print("  EXPERIMENT 3(a): CONFUSABLE-SPLIT vs CONFUSABLE-TOGETHER BENCHMARK EVALUATION")
    print("="*90)
    
    MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
    if not os.path.exists(CACHE_100_PATH):
        print(f"[Cache] Embeddings cache {CACHE_100_PATH} not found. Reconstructing automatically...")
        with open(DATASET_PATH, "r") as f:
            blocks = json.load(f)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    else:
        cache_data = torch.load(CACHE_100_PATH, map_location=DEVICE)
        
    conf_pairs = find_confusable_pairs(cache_data)
    print(f"[Audit] Identified {len(conf_pairs)} confusable fact pairs (>0.95 cosine).")
    
    blocks_split, blocks_together = build_confusable_block_assignments(conf_pairs)
    
    conditions = ["frozen_adapter", "naive_sequential_adapter", "offline_adapter"]
    
    # 1. Evaluate CONFUSABLE-SPLIT
    print("\n--- Running CONFUSABLE-SPLIT Assignment (Pairs in DIFFERENT Blocks) ---")
    res_split = {}
    for cond in conditions:
        res_split[cond] = run_experiment_on_blocks(blocks_split, cache_data, condition=cond)
        
    # 2. Evaluate CONFUSABLE-TOGETHER
    print("\n--- Running CONFUSABLE-TOGETHER Assignment (Pairs in SAME Block) ---")
    res_together = {}
    for cond in conditions:
        res_together[cond] = run_experiment_on_blocks(blocks_together, cache_data, condition=cond)
        
    print("\n" + "="*120)
    print("  SUMMARY: CONFUSABLE-SPLIT vs CONFUSABLE-TOGETHER CONTINUAL LEARNING GAPS")
    print("="*120)
    
    # Compute CL Gaps (offline - naive)
    split_off_ats = [r["A_T"] for r in res_split["offline_adapter"]]
    split_nai_ats = [r["A_T"] for r in res_split["naive_sequential_adapter"]]
    split_fro_ats = [r["A_T"] for r in res_split["frozen_adapter"]]
    
    together_off_ats = [r["A_T"] for r in res_together["offline_adapter"]]
    together_nai_ats = [r["A_T"] for r in res_together["naive_sequential_adapter"]]
    together_fro_ats = [r["A_T"] for r in res_together["frozen_adapter"]]
    
    gap_split_mean, gap_split_l, gap_split_u = bootstrap_paired_ci(split_off_ats, split_nai_ats)
    gap_tog_mean, gap_tog_l, gap_tog_u = bootstrap_paired_ci(together_off_ats, together_nai_ats)
    
    print(f"  * CONFUSABLE-SPLIT CL Gap (offline - naive)     : {gap_split_mean*100:+.2f}% | 95% CI: [{gap_split_l*100:+.2f}%, {gap_split_u*100:+.2f}%]")
    print(f"  * CONFUSABLE-TOGETHER CL Gap (offline - naive)  : {gap_tog_mean*100:+.2f}% | 95% CI: [{gap_tog_l*100:+.2f}%, {gap_tog_u*100:+.2f}%]")
    print("="*120)
    
    if gap_split_mean > gap_tog_mean + 0.02:
        print("  [PREDICTION CONFIRMED]: CONFUSABLE-SPLIT produces a substantially larger CL gap due to inter-block interference!")
    else:
        print("  [FINDING]: Both assignments produce similar CL gaps; confusability placement alone is not the sole driver.")
    print("="*120)

if __name__ == "__main__":
    main()
