"""
run_mechanism_evaluation_suite.py — AGNIS Mechanism Evaluation Suite against Interference Benchmark
===================================================================================================
Evaluates AGNIS mechanisms (l2sp_anchor with L2-SP parameter distance + activation anchoring)
against the established high-interference benchmark (CONFUSABLE-SPLIT + epochs=30, lr=1e-3).

Single Success Criterion:
  Does the mechanism reduce A_T(offline) - A_T(method) below A_T(offline) - A_T(naive),
  with a paired 10,000-sample bootstrap 95% CI excluding zero?
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

def build_confusable_split_blocks(confusable_pairs):
    blocks_split = [[] for _ in range(10)]
    for i in range(100):
        blocks_split[i % 10].append(i)
        
    random.seed(42)
    for pair in confusable_pairs:
        f1, f2, _ = pair
        b1 = [b for b in range(10) if f1 in blocks_split[b]][0]
        b2 = [b for b in range(10) if f2 in blocks_split[b]][0]
        if b1 == b2:
            target_b = (b1 + 1) % 10
            for k in range(len(blocks_split[target_b])):
                swap_f = blocks_split[target_b][k]
                if swap_f not in [p[0] for p in confusable_pairs if p[1] == f1] and swap_f not in [p[1] for p in confusable_pairs if p[0] == f1]:
                    blocks_split[b1].remove(f2)
                    blocks_split[target_b].remove(swap_f)
                    blocks_split[b1].append(swap_f)
                    blocks_split[target_b].append(f2)
                    break
    return blocks_split

def run_mechanism_experiment(block_assignment, cache_data, condition="l2sp_anchor_lam0.001", 
                             lambda_l2sp=0.001, lambda_anchor=0.001, epochs=30, lr=1e-3, shuffles=5, seeds=3):
    results = []
    random.seed(42)
    order_list = []
    for _ in range(shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)
        
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
        
    # Generate anchor set (first 20 facts across block assignment)
    anchor_facts = [f for b in block_assignment[:2] for f in b]
    anchor_x = torch.cat([cache_data["train_x"][f*3 : (f+1)*3] for f in anchor_facts], dim=0).to(DEVICE)
    
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
            
            # Save base state for L2-SP
            with torch.no_grad():
                anchor_embeddings_base = adapter(anchor_x).clone()
            base_weights = adapter.linear.weight.clone()
            base_bias = adapter.linear.bias.clone()
            
            if condition == "offline_adapter":
                adapter.train()
                for ep in range(int(epochs * 1.5)):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            elif condition != "frozen_adapter":
                adapter.train()
                for ep in range(epochs):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            # Update base weights reference after base phase training
            base_weights = adapter.linear.weight.clone().detach()
            base_bias = adapter.linear.bias.clone().detach()
            with torch.no_grad():
                anchor_embeddings_base = adapter(anchor_x).clone().detach()
                
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
                
                if condition.startswith("l2sp_anchor"):
                    adapter.train()
                    curr_x = train_x_blocks[curr_block].to(DEVICE)
                    curr_y = train_y_blocks[curr_block].to(DEVICE)
                    for ep in range(epochs):
                        proj = adapter(curr_x)
                        loss_supcon = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                        
                        cur_anchors = adapter(anchor_x)
                        loss_anchor = F.mse_loss(cur_anchors, anchor_embeddings_base)
                        loss_l2sp = torch.sum((adapter.linear.weight - base_weights)**2) + torch.sum((adapter.linear.bias - base_bias)**2)
                        
                        loss = loss_supcon + lambda_anchor * loss_anchor + lambda_l2sp * loss_l2sp
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                elif condition == "naive_sequential_adapter":
                    adapter.train()
                    curr_x = train_x_blocks[curr_block].to(DEVICE)
                    curr_y = train_y_blocks[curr_block].to(DEVICE)
                    for ep in range(epochs):
                        proj = adapter(curr_x)
                        loss = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                elif condition == "offline_adapter":
                    adapter.train()
                    for ep in range(epochs):
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
    print("="*110)
    print("  AGNIS MECHANISM EVALUATION SUITE AGAINST INTERFERENCE BENCHMARK")
    print("="*110)
    
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
    if not os.path.exists(CACHE_100_PATH):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    else:
        cache_data = torch.load(CACHE_100_PATH, map_location=DEVICE)
        
    conf_pairs = find_confusable_pairs(cache_data)
    block_assignment = build_confusable_split_blocks(conf_pairs)
    
    # Grid of L2-SP & Anchor Regularization Strength
    configs = [
        ("frozen_adapter", 0.0, 0.0),
        ("naive_sequential_adapter", 0.0, 0.0),
        ("l2sp_anchor_lam0.001", 0.001, 0.001),
        ("l2sp_anchor_lam0.005", 0.005, 0.005),
        ("l2sp_anchor_lam0.01", 0.01, 0.01),
        ("offline_adapter", 0.0, 0.0)
    ]
    
    all_runs = {}
    
    for name, l_l2sp, l_anc in configs:
        res = run_mechanism_experiment(block_assignment, cache_data, condition=name, lambda_l2sp=l_l2sp, lambda_anchor=l_anc)
        all_runs[name] = res
        
    offline_ats = [r["A_T"] for r in all_runs["offline_adapter"]]
    naive_ats = [r["A_T"] for r in all_runs["naive_sequential_adapter"]]
    naive_cl_gap_mean = np.mean(offline_ats) - np.mean(naive_ats)
    
    print("\n" + "="*120)
    print(f"  BASELINE CONTINUAL LEARNING GAP: A_T(offline) - A_T(naive) = {naive_cl_gap_mean*100:+.2f}%")
    print("="*120)
    header = f"{'Condition':30s} | {'A_T (Final Acc)':18s} | {'CL Gap (offline - method)':28s} | {'CI vs Naive Gap':25s}"
    print(header)
    print("-" * len(header))
    
    for name, l_l2sp, l_anc in configs:
        runs = all_runs[name]
        ats = [r["A_T"] for r in runs]
        at_mean = np.mean(ats) * 100.0
        at_std = np.std(ats) * 100.0
        
        cl_gap = (np.mean(offline_ats) - np.mean(ats)) * 100.0
        
        # Paired difference: (A_T[method] - A_T[naive])
        diff_mean, ci_l, ci_u = bootstrap_paired_ci(ats, naive_ats)
        
        verdict = ""
        if name.startswith("l2sp_anchor"):
            if ci_l > 0.0:
                verdict = "SUCCESS: REDUCED CL GAP!"
            else:
                verdict = "NOT STATISTICALLY SIGNIFICANT"
                
        row_str = (
            f"  {name:30s} | "
            f"{at_mean:6.2f}% ± {at_std:4.2f}% | "
            f"{cl_gap:+6.2f}%                     | "
            f"Diff: {diff_mean*100:+5.2f}% [{ci_l*100:+5.2f}%, {ci_u*100:+5.2f}%] {verdict}"
        )
        print(row_str)
    print("="*120)

if __name__ == "__main__":
    main()
