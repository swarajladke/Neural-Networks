"""
run_ogp_rigorous_verification.py — Comprehensive OGP Verification & Control Suite
====================================================================================
Implements:
1. Item 5: Lambda weight displacement check (|W - W_base|_max at step 9 for lambda in {1e-6, 1e-5}).
2. Item 1 & Item 2: Extended OGP Sweep (k in {4, 8, 16, 24, 32, 64, 128, 256, 512}) with full metric decomposition
   (A_T, LA, Forgetting, BWT, CL Gap, paired CI on A_T vs naive, and paired CI on FORGETTING vs naive).
3. Item 3: Control Arms at k=32 (RANDOM-32, BOTTOM-32, CURRENT-32 vs TOP-32 OGP).
4. Item 4: Fresh-Seed & Shuffle Replication (seeds {211, 212, 213} with 5 newly drawn block orderings).
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

def bootstrap_paired_ci(vals_1, vals_2, n_boot=10000, seed=42):
    n = min(len(vals_1), len(vals_2))
    diffs = np.array(vals_1[:n]) - np.array(vals_2[:n])
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    return np.mean(diffs), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def check_lambda_weight_displacement(block_assignment, cache_data):
    print("\n" + "="*100)
    print("  ITEM 5: LAMBDA WEIGHT DISPLACEMENT CHECK AT STEP 9 (|W - W_base|_max)")
    print("="*100)
    
    train_x_blocks = []
    train_y_blocks = []
    for b in range(10):
        fact_ids = block_assignment[b]
        b_tr_x = torch.cat([cache_data["train_x"][f*3 : (f+1)*3] for f in fact_ids], dim=0)
        b_tr_y = torch.cat([cache_data["train_y"][f*3 : (f+1)*3] for f in fact_ids], dim=0)
        train_x_blocks.append(b_tr_x)
        train_y_blocks.append(b_tr_y)
        
    for l_val in [1e-6, 1e-5, 1e-4, 1e-3]:
        adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
        
        base_blocks = list(range(5))
        joint_train_x_base = torch.cat([train_x_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
        joint_train_y_base = torch.cat([train_y_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
        
        adapter.train()
        for ep in range(30):
            proj = adapter(joint_train_x_base)
            loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        base_w = adapter.linear.weight.clone().detach()
        base_b = adapter.linear.bias.clone().detach()
        
        anchor_facts = [f for b in block_assignment[:2] for f in b]
        anchor_x = torch.cat([cache_data["train_x"][f*3 : (f+1)*3] for f in anchor_facts], dim=0).to(DEVICE)
        with torch.no_grad():
            anchor_emb_base = adapter(anchor_x).clone().detach()
            
        for step in range(5, 10):
            adapter.train()
            curr_x = train_x_blocks[step].to(DEVICE)
            curr_y = train_y_blocks[step].to(DEVICE)
            for ep in range(30):
                proj = adapter(curr_x)
                loss_supcon = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                cur_anchors = adapter(anchor_x)
                loss_anchor = F.mse_loss(cur_anchors, anchor_emb_base)
                loss_l2sp = torch.sum((adapter.linear.weight - base_w)**2) + torch.sum((adapter.linear.bias - base_b)**2)
                
                loss = loss_supcon + l_val * loss_anchor + l_val * loss_l2sp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        max_w_diff = (adapter.linear.weight - base_w).abs().max().item()
        max_b_diff = (adapter.linear.bias - base_b).abs().max().item()
        print(f"  * Lambda = {l_val:.0e} | Step 9 Max |W - W_base|: {max_w_diff:.8e} | Max |b - b_base|: {max_b_diff:.8e}")
    print("="*100)

def run_ogp_extended_sweep(block_assignment, cache_data, k_grid=[4, 8, 16, 24, 32, 64, 128, 256, 512], seeds=[101, 102, 103]):
    print("\n" + "="*120)
    print(f"  ITEMS 1 & 2: EXTENDED OGP RANK SWEEP (k in {k_grid}) WITH DECOMPOSITION METRICS")
    print("="*120)
    
    from run_ogp_mechanism_experiment import run_ogp_experiment
    from run_mechanism_evaluation_suite import run_mechanism_experiment
    
    res_naive = run_mechanism_experiment(block_assignment, cache_data, condition="naive_sequential_adapter", seeds=len(seeds))
    res_offline = run_mechanism_experiment(block_assignment, cache_data, condition="offline_adapter", seeds=len(seeds))
    
    nai_ats = [r["A_T"] for r in res_naive]
    nai_las = [r["LA"] for r in res_naive]
    nai_fgts = [r["mean_forgetting"] for r in res_naive]
    
    off_ats = [r["A_T"] for r in res_offline]
    off_las = [r["LA"] for r in res_offline]
    off_fgts = [r["mean_forgetting"] for r in res_offline]
    
    off_mean_at = np.mean(off_ats) * 100.0
    
    print(f"  * Naive Baseline  : A_T={np.mean(nai_ats)*100:.2f}% | LA={np.mean(nai_las)*100:.2f}% | Fgt={np.mean(nai_fgts)*100:.2f}%")
    print(f"  * Offline Baseline: A_T={off_mean_at:.2f}% | LA={np.mean(off_las)*100:.2f}% | Fgt={np.mean(off_fgts)*100:.2f}%")
    print("-" * 120)
    
    header = f"{'k':4s} | {'Rank':5s} | {'A_T (Final Acc)':15s} | {'LA (Learning)':15s} | {'Observed Fgt':15s} | {'CL Gap':8s} | {'Diff A_T vs Naive (CI)':25s} | {'Diff Fgt vs Naive (CI)':25s}"
    print(header)
    print("-" * len(header))
    
    all_sweep_results = {}
    for k_val in k_grid:
        res_ogp, rank_achieved = run_ogp_experiment(block_assignment, cache_data, k_rank=k_val, seeds=len(seeds))
        ogp_ats = [r["A_T"] for r in res_ogp]
        ogp_las = [r["LA"] for r in res_ogp]
        ogp_fgts = [r["mean_forgetting"] for r in res_ogp]
        ogp_bwts = [r["mean_bwt"] for r in res_ogp]
        
        at_m = np.mean(ogp_ats) * 100.0
        at_s = np.std(ogp_ats) * 100.0
        la_m = np.mean(ogp_las) * 100.0
        fgt_m = np.mean(ogp_fgts) * 100.0
        
        cl_gap = off_mean_at - at_m
        
        diff_at_m, ci_at_l, ci_at_u = bootstrap_paired_ci(ogp_ats, nai_ats)
        diff_fgt_m, ci_fgt_l, ci_fgt_u = bootstrap_paired_ci(ogp_fgts, nai_fgts)
        
        verdict = ""
        if ci_at_l > 0.0:
            verdict = "SUCCESS (+A_T)"
        elif ci_at_u < 0.0:
            verdict = "SIG WORSE (-A_T)"
        else:
            verdict = "TRUE NULL"
            
        row_str = (
            f"  {k_val:<4d} | "
            f"{rank_achieved:<5d} | "
            f"{at_m:5.2f}% ± {at_s:4.2f}% | "
            f"{la_m:5.2f}%          | "
            f"{fgt_m:5.2f}%          | "
            f"{cl_gap:+5.2f}%   | "
            f"{diff_at_m*100:+5.2f}% [{ci_at_l*100:+5.2f}%, {ci_at_u*100:+5.2f}%] | "
            f"{diff_fgt_m*100:+5.2f}% [{ci_fgt_l*100:+5.2f}%, {ci_fgt_u*100:+5.2f}%] {verdict}"
        )
        print(row_str)
        
        all_sweep_results[k_val] = {
            "A_T_runs": ogp_ats,
            "LA_runs": ogp_las,
            "Fgt_runs": ogp_fgts,
            "A_T_mean": at_m,
            "LA_mean": la_m,
            "Fgt_mean": fgt_m,
            "diff_at": diff_at_m * 100.0,
            "ci_at": (ci_at_l * 100.0, ci_at_u * 100.0),
            "diff_fgt": diff_fgt_m * 100.0,
            "ci_fgt": (ci_fgt_l * 100.0, ci_fgt_u * 100.0)
        }
        
    print("="*120)
    return all_sweep_results, res_naive

def run_ogp_controls_at_k32(block_assignment, cache_data, seeds=[101, 102, 103]):
    print("\n" + "="*120)
    print("  ITEM 3: CONTROL ARMS AT k=32 (RANDOM-32, BOTTOM-32, CURRENT-32 vs TOP-32 OGP)")
    print("="*120)
    
    from run_mechanism_evaluation_suite import run_mechanism_experiment
    res_naive = run_mechanism_experiment(block_assignment, cache_data, condition="naive_sequential_adapter", seeds=len(seeds))
    nai_ats = [r["A_T"] for r in res_naive]
    
    control_names = ["TOP-32", "RANDOM-32", "BOTTOM-32", "CURRENT-32"]
    
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
        
    random.seed(42)
    order_list = []
    for _ in range(5):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)
        
    control_results = {}
    
    for c_name in control_names:
        c_runs = []
        for shuffle_idx, order in enumerate(order_list):
            for seed in range(101, 101 + len(seeds)):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
                R = np.zeros((10, 10))
                
                base_blocks = order[:5]
                joint_train_x_base = torch.cat([train_x_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
                joint_train_y_base = torch.cat([train_y_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
                
                adapter.train()
                for ep in range(30):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                M_past = joint_train_x_base.clone().detach()
                
                for step in range(5, 10):
                    curr_block = order[step]
                    seen_blocks = order[:step + 1]
                    seen_ref_x = torch.cat([train_x_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                    seen_ref_y = torch.cat([train_y_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                    
                    curr_x = train_x_blocks[curr_block].to(DEVICE)
                    curr_y = train_y_blocks[curr_block].to(DEVICE)
                    
                    if c_name == "TOP-32":
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        P = Vh[:32].T
                    elif c_name == "RANDOM-32":
                        R_mat = torch.randn(INPUT_DIM, 32, device=DEVICE)
                        Q, _ = torch.linalg.qr(R_mat)
                        P = Q
                    elif c_name == "BOTTOM-32":
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        P = Vh[-32:].T
                    elif c_name == "CURRENT-32":
                        _, S, Vh = torch.linalg.svd(curr_x, full_matrices=False)
                        P = Vh[:32].T
                        
                    I_dim = torch.eye(INPUT_DIM, device=DEVICE)
                    proj_mat = I_dim - torch.matmul(P, P.T)
                    
                    adapter.train()
                    for ep in range(30):
                        proj = adapter(curr_x)
                        loss = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        adapter.linear.weight.grad = torch.matmul(adapter.linear.weight.grad, proj_mat)
                        optimizer.step()
                        
                    M_past = torch.cat([M_past, curr_x.clone().detach()], dim=0)
                    
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
                fgt = np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)])
                c_runs.append({"A_T": a_t, "LA": la, "mean_forgetting": fgt})
                
        control_results[c_name] = c_runs
        
    header = f"{'Control Arm':15s} | {'A_T (Final Acc)':15s} | {'LA (Learning)':15s} | {'Observed Fgt':15s} | {'Diff A_T vs Naive (CI)':28s}"
    print("\n" + header)
    print("-" * len(header))
    for c_name in control_names:
        c_ats = [r["A_T"] for r in control_results[c_name]]
        c_las = [r["LA"] for r in control_results[c_name]]
        c_fgts = [r["mean_forgetting"] for r in control_results[c_name]]
        
        m_at = np.mean(c_ats) * 100.0
        s_at = np.std(c_ats) * 100.0
        m_la = np.mean(c_las) * 100.0
        m_fgt = np.mean(c_fgts) * 100.0
        
        diff_m, ci_l, ci_u = bootstrap_paired_ci(c_ats, nai_ats)
        
        row_str = (
            f"  {c_name:15s} | "
            f"{m_at:5.2f}% ± {s_at:4.2f}% | "
            f"{m_la:5.2f}%          | "
            f"{m_fgt:5.2f}%          | "
            f"Diff: {diff_m*100:+5.2f}% [{ci_l*100:+5.2f}%, {ci_u*100:+5.2f}%]"
        )
        print(row_str)
    print("="*120)
    print("  * Total Accumulated Reference Count at Step 9: 270 (9 blocks x 30 references). k=32 protects 32 of 270 available directions.")
    print("="*120)
    return control_results

def run_fresh_seed_replication(block_assignment, cache_data, best_k=32, fresh_seeds=[211, 212, 213]):
    print("\n" + "="*120)
    print(f"  ITEM 4: FRESH SEED & SHUFFLE REPLICATION (Seeds {fresh_seeds}, 5 Newly Drawn Block Orderings)")
    print("="*120)
    
    random.seed(999) # New seed for generating fresh block orderings
    fresh_order_list = []
    for _ in range(5):
        order = list(range(10))
        random.shuffle(order)
        fresh_order_list.append(order)
        
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
        
    def evaluate_condition_fresh(cond_type, k_val=32):
        cond_runs = []
        for shuffle_idx, order in enumerate(fresh_order_list):
            for seed in fresh_seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
                R = np.zeros((10, 10))
                
                base_blocks = order[:5]
                joint_train_x_base = torch.cat([train_x_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
                joint_train_y_base = torch.cat([train_y_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
                
                adapter.train()
                for ep in range(45 if cond_type == "offline" else 30):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                M_past = joint_train_x_base.clone().detach()
                
                for step in range(5, 10):
                    curr_block = order[step]
                    seen_blocks = order[:step + 1]
                    seen_ref_x = torch.cat([train_x_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                    seen_ref_y = torch.cat([train_y_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                    
                    curr_x = train_x_blocks[curr_block].to(DEVICE)
                    curr_y = train_y_blocks[curr_block].to(DEVICE)
                    
                    if cond_type == "ogp":
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        P = Vh[:k_val].T
                        I_dim = torch.eye(INPUT_DIM, device=DEVICE)
                        proj_mat = I_dim - torch.matmul(P, P.T)
                        
                    adapter.train()
                    train_data_x = seen_ref_x if cond_type == "offline" else curr_x
                    train_data_y = seen_ref_y if cond_type == "offline" else curr_y
                    
                    for ep in range(30):
                        proj = adapter(train_data_x)
                        loss = supervised_contrastive_loss(proj, train_data_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        if cond_type == "ogp":
                            adapter.linear.weight.grad = torch.matmul(adapter.linear.weight.grad, proj_mat)
                        optimizer.step()
                        
                    M_past = torch.cat([M_past, curr_x.clone().detach()], dim=0)
                    
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
                fgt = np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)])
                cond_runs.append({"A_T": a_t, "LA": la, "mean_forgetting": fgt})
        return cond_runs

    rep_naive = evaluate_condition_fresh("naive")
    rep_offline = evaluate_condition_fresh("offline")
    rep_ogp = evaluate_condition_fresh("ogp", k_val=best_k)
    
    nai_ats = [r["A_T"] for r in rep_naive]
    off_ats = [r["A_T"] for r in rep_offline]
    ogp_ats = [r["A_T"] for r in rep_ogp]
    
    nai_mean = np.mean(nai_ats) * 100.0
    off_mean = np.mean(off_ats) * 100.0
    ogp_mean = np.mean(ogp_ats) * 100.0
    
    cl_gap_naive = off_mean - nai_mean
    cl_gap_ogp = off_mean - ogp_mean
    
    diff_m, ci_l, ci_u = bootstrap_paired_ci(ogp_ats, nai_ats)
    
    print(f"  * Fresh Naive A_T             : {nai_mean:.2f}% ± {np.std(nai_ats)*100:.2f}% | Baseline CL Gap: {cl_gap_naive:+.2f}%")
    print(f"  * Fresh Offline A_T           : {off_mean:.2f}% ± {np.std(off_ats)*100:.2f}%")
    print(f"  * Fresh OGP (k={best_k}) A_T         : {ogp_mean:.2f}% ± {np.std(ogp_ats)*100:.2f}% | OGP CL Gap     : {cl_gap_ogp:+.2f}%")
    print(f"  * Fresh Paired Diff vs Naive  : {diff_m*100:+.2f}%")
    print(f"  * 10,000-Sample Bootstrap 95% CI: [{ci_l*100:+.2f}%, {ci_u*100:+.2f}%]")
    print("="*120)

def main():
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
    
    # 1. Item 5 Check
    check_lambda_weight_displacement(block_assignment, cache_data)
    
    # 2. Items 1 & 2 Extended Sweep & Decomposition
    sweep_results, res_naive = run_ogp_extended_sweep(block_assignment, cache_data)
    
    # 3. Item 3 Control Arms
    control_results = run_ogp_controls_at_k32(block_assignment, cache_data)
    
    # 4. Item 4 Fresh Replication
    run_fresh_seed_replication(block_assignment, cache_data, best_k=32)

if __name__ == "__main__":
    main()
