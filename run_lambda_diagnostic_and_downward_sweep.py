"""
run_lambda_diagnostic_and_downward_sweep.py — Lambda Gradient Diagnostics, Downward Sweep, & Previous-Step Anchoring
===================================================================================================================
1. Gradient Norm Diagnostic Dump (Steps 5, 7, 9):
   - Measures total parameter norm for ||grad_contrastive||, ||lambda * grad_l2sp||, and ||lambda_anchor * grad_anchor||.
   - Evaluates both SUM and MEAN loss formulations over the 922,560 adapter parameters.
   
2. Downward Lambda Sweep:
   - Sweeps lambda in {1e-6, 1e-5, 1e-4, 1e-3} under CONFUSABLE-SPLIT + epochs=30, lr=1e-3.
   
3. Item 3: Previous-Step Anchoring (l2sp_anchor_prev):
   - Anchors L2-SP and activation anchoring to step (t-1) instead of base phase (step 4).
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

def run_gradient_diagnostic_dump(block_assignment, cache_data, lambda_val=0.001):
    print("\n" + "="*100)
    print(f"  ITEM 1 GRADIENT NORM & LOSS COMPONENT DIAGNOSTIC DUMP (lambda = {lambda_val})")
    print("="*100)
    
    train_x_blocks = []
    train_y_blocks = []
    for b in range(10):
        fact_ids = block_assignment[b]
        b_tr_x = torch.cat([cache_data["train_x"][f*3 : (f+1)*3] for f in fact_ids], dim=0)
        b_tr_y = torch.cat([cache_data["train_y"][f*3 : (f+1)*3] for f in fact_ids], dim=0)
        train_x_blocks.append(b_tr_x)
        train_y_blocks.append(b_tr_y)
        
    anchor_facts = [f for b in block_assignment[:2] for f in b]
    anchor_x = torch.cat([cache_data["train_x"][f*3 : (f+1)*3] for f in anchor_facts], dim=0).to(DEVICE)
    
    adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
    joint_train_x_base = torch.cat([train_x_blocks[b] for b in range(5)], dim=0).to(DEVICE)
    joint_train_y_base = torch.cat([train_y_blocks[b] for b in range(5)], dim=0).to(DEVICE)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    adapter.train()
    for ep in range(30):
        proj = adapter(joint_train_x_base)
        loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    base_w = adapter.linear.weight.clone().detach()
    base_b = adapter.linear.bias.clone().detach()
    with torch.no_grad():
        anchor_embeddings_base = adapter(anchor_x).clone().detach()
        
    for step in [5, 7, 9]:
        curr_x = train_x_blocks[step].to(DEVICE)
        curr_y = train_y_blocks[step].to(DEVICE)
        
        # 1. Compute Task (Contrastive) Gradient
        proj = adapter(curr_x)
        loss_task = supervised_contrastive_loss(proj, curr_y, tau=0.05)
        optimizer.zero_grad()
        loss_task.backward()
        g_task_norm = torch.sqrt(sum(p.grad.norm()**2 for p in adapter.parameters() if p.grad is not None)).item()
        
        # 2. Compute L2-SP Gradient (SUM formulation)
        loss_l2sp_sum = torch.sum((adapter.linear.weight - base_w)**2) + torch.sum((adapter.linear.bias - base_b)**2)
        optimizer.zero_grad()
        (lambda_val * loss_l2sp_sum).backward()
        g_l2sp_sum_norm = torch.sqrt(sum(p.grad.norm()**2 for p in adapter.parameters() if p.grad is not None)).item()
        
        # 3. Compute L2-SP Gradient (MEAN formulation)
        loss_l2sp_mean = (torch.sum((adapter.linear.weight - base_w)**2) + torch.sum((adapter.linear.bias - base_b)**2)) / 922560.0
        optimizer.zero_grad()
        (lambda_val * loss_l2sp_mean).backward()
        g_l2sp_mean_norm = torch.sqrt(sum(p.grad.norm()**2 for p in adapter.parameters() if p.grad is not None)).item()
        
        # 4. Compute Anchor Gradient
        cur_anchors = adapter(anchor_x)
        loss_anchor = F.mse_loss(cur_anchors, anchor_embeddings_base)
        optimizer.zero_grad()
        (lambda_val * loss_anchor).backward()
        g_anchor_norm = torch.sqrt(sum(p.grad.norm()**2 for p in adapter.parameters() if p.grad is not None)).item()
        
        print(f"  [Step {step}] Loss Values:")
        print(f"    - Task (Contrastive) Loss         : {loss_task.item():.6f}")
        print(f"    - L2-SP Loss (SUM)                : {loss_l2sp_sum.item():.6f} | Weighted: {lambda_val * loss_l2sp_sum.item():.6f}")
        print(f"    - L2-SP Loss (MEAN)               : {loss_l2sp_mean.item():.8f} | Weighted: {lambda_val * loss_l2sp_mean.item():.8f}")
        print(f"    - Anchor Loss                     : {loss_anchor.item():.6f} | Weighted: {lambda_val * loss_anchor.item():.6f}")
        print(f"  [Step {step}] Gradient Norms & Ratios:")
        print(f"    - ||grad_contrastive||            : {g_task_norm:.6e}")
        print(f"    - ||lambda * grad_l2sp_sum||      : {g_l2sp_sum_norm:.6e} (Ratio to Task: {g_l2sp_sum_norm / g_task_norm:.4f})")
        print(f"    - ||lambda * grad_l2sp_mean||     : {g_l2sp_mean_norm:.6e} (Ratio to Task: {g_l2sp_mean_norm / g_task_norm:.6f})")
        print(f"    - ||lambda * grad_anchor||        : {g_anchor_norm:.6e} (Ratio to Task: {g_anchor_norm / g_task_norm:.6f})")
        print("-" * 100)

def run_experiment_variant(block_assignment, cache_data, condition="l2sp_anchor", lambda_val=1e-3, use_prev=False, epochs=30, lr=1e-3, shuffles=5, seeds=3):
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
            
            adapter.train()
            for ep in range(epochs):
                proj = adapter(joint_train_x_base)
                loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            prev_w = adapter.linear.weight.clone().detach()
            prev_b = adapter.linear.bias.clone().detach()
            with torch.no_grad():
                anchor_embeddings_prev = adapter(anchor_x).clone().detach()
                
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
                
                adapter.train()
                curr_x = train_x_blocks[curr_block].to(DEVICE)
                curr_y = train_y_blocks[curr_block].to(DEVICE)
                
                # If use_prev, anchor reference is step (t-1) state
                ref_w = prev_w
                ref_b = prev_b
                ref_anchor_emb = anchor_embeddings_prev
                
                for ep in range(epochs):
                    proj = adapter(curr_x)
                    loss_supcon = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                    
                    cur_anchors = adapter(anchor_x)
                    loss_anchor = F.mse_loss(cur_anchors, ref_anchor_emb)
                    loss_l2sp = torch.sum((adapter.linear.weight - ref_w)**2) + torch.sum((adapter.linear.bias - ref_b)**2)
                    
                    loss = loss_supcon + lambda_val * loss_anchor + lambda_val * loss_l2sp
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if use_prev:
                    prev_w = adapter.linear.weight.clone().detach()
                    prev_b = adapter.linear.bias.clone().detach()
                    with torch.no_grad():
                        anchor_embeddings_prev = adapter(anchor_x).clone().detach()
                        
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
                        
            a_t = np.mean(R[9, :])
            la = np.mean([R[max(4, order.index(j)), j] for j in range(10)])
            fgt_j = [np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)]
            bwt_j = [R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]
            
            results.append({
                "condition": condition,
                "lambda": lambda_val,
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
    print("  ITEM 1 DIAGNOSTICS, DOWNWARD LAMBDA SWEEP, & ITEM 3 PREVIOUS-STEP ANCHORING")
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
    
    # 1. Run Gradient Norm & Loss Component Diagnostic Dump
    run_gradient_diagnostic_dump(block_assignment, cache_data, lambda_val=0.001)
    
    # 2. Downward Lambda Sweep: {1e-6, 1e-5, 1e-4, 1e-3}
    print("\n" + "="*110)
    print("  DOWNWARD LAMBDA SWEEP: lambda in {1e-6, 1e-5, 1e-4, 1e-3}")
    print("="*110)
    
    # Load naive baseline runs for paired diff
    from run_mechanism_evaluation_suite import run_mechanism_experiment
    res_naive = run_mechanism_experiment(block_assignment, cache_data, condition="naive_sequential_adapter")
    naive_ats = [r["A_T"] for r in res_naive]
    
    lambda_grid = [1e-6, 1e-5, 1e-4, 1e-3]
    header = f"{'Lambda':12s} | {'A_T (Final Acc)':18s} | {'Diff vs Naive':22s} | {'Paired 95% Bootstrap CI':25s}"
    print(header)
    print("-" * len(header))
    
    best_lambda = 1e-3
    for l_val in lambda_grid:
        res_l = run_experiment_variant(block_assignment, cache_data, condition=f"l2sp_anchor_lam{l_val:.0e}", lambda_val=l_val)
        l_ats = [r["A_T"] for r in res_l]
        l_mean = np.mean(l_ats) * 100.0
        l_std = np.std(l_ats) * 100.0
        
        diff_m, ci_l, ci_u = bootstrap_paired_ci(l_ats, naive_ats)
        
        row_str = (
            f"  {l_val:.0e}       | "
            f"{l_mean:6.2f}% ± {l_std:4.2f}% | "
            f"Diff: {diff_m*100:+5.2f}%       | "
            f"[{ci_l*100:+5.2f}%, {ci_u*100:+5.2f}%]"
        )
        print(row_str)
    print("="*110)
    
    # 3. Item 3: Previous-Step Anchoring (l2sp_anchor_prev)
    print("\n" + "="*110)
    print("  ITEM 3: PREVIOUS-STEP ANCHORING (l2sp_anchor_prev)")
    print("="*110)
    res_prev = run_experiment_variant(block_assignment, cache_data, condition="l2sp_anchor_prev", lambda_val=1e-3, use_prev=True)
    prev_ats = [r["A_T"] for r in res_prev]
    prev_mean = np.mean(prev_ats) * 100.0
    prev_std = np.std(prev_ats) * 100.0
    
    diff_p, ci_pl, ci_pu = bootstrap_paired_ci(prev_ats, naive_ats)
    print(f"  * l2sp_anchor_prev A_T          : {prev_mean:.2f}% ± {prev_std:.2f}%")
    print(f"  * Paired Diff vs Naive          : {diff_p*100:+.2f}%")
    print(f"  * 10,000-Sample Bootstrap 95% CI: [{ci_pl*100:+.2f}%, {ci_pu*100:+.2f}%]")
    print("="*110)

if __name__ == "__main__":
    main()
