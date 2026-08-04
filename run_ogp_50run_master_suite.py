"""
run_ogp_50run_master_suite.py — 50-Run Master OGP Verification & Control Suite
================================================================================
Executes 50 runs per condition (10 shuffles x 5 seeds) to establish exact 50-run statistical bounds for:
  1. Item 2 Audit: BOTTOM-32 vs Naive elementwise matrix diffs & weight displacement.
  2. Item 4 Metric Decomposition: Naive LA, Offline LA, OGP LA, Forgetting CIs, and Delta A_T = Delta LA + Delta Memory.
  3. Item 3 Resolution for k=8: 50-run evaluation of k=8 to resolve noise vs structural dip.
  4. Item 3 Control Arms: 50-run evaluation of RANDOM-32, BOTTOM-32, CURRENT-32 vs TOP-32 OGP.
  5. Item 5 50-Run Replication across both Selection Seeds (101..105) and Fresh Seeds (211..215).
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

def run_bottom32_literal_noop_audit(block_assignment, cache_data):
    print("\n" + "="*100)
    print("  ITEM 2: BOTTOM-32 LITERAL NO-OP AUDIT")
    print("="*100)
    
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
        
    naive_ats, bottom_ats = [], []
    naive_Rs, bottom_Rs = [], []
    max_w_diffs = []
    
    for shuffle_idx, order in enumerate(order_list):
        for seed in range(101, 104):
            # 1. Run Naive
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            adapter_n = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
            opt_n = torch.optim.AdamW(adapter_n.parameters(), lr=1e-3, weight_decay=1e-4)
            R_n = np.zeros((10, 10))
            
            base_blocks = order[:5]
            joint_x_base = torch.cat([train_x_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
            joint_y_base = torch.cat([train_y_blocks[b] for b in base_blocks], dim=0).to(DEVICE)
            
            adapter_n.train()
            for ep in range(30):
                proj = adapter_n(joint_x_base)
                loss = supervised_contrastive_loss(proj, joint_y_base, tau=0.05)
                opt_n.zero_grad()
                loss.backward()
                opt_n.step()
                
            for step in range(5, 10):
                curr_block = order[step]
                seen_blocks = order[:step + 1]
                seen_x = torch.cat([train_x_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                seen_y = torch.cat([train_y_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                curr_x = train_x_blocks[curr_block].to(DEVICE)
                curr_y = train_y_blocks[curr_block].to(DEVICE)
                
                adapter_n.train()
                for ep in range(30):
                    proj = adapter_n(curr_x)
                    loss = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                    opt_n.zero_grad()
                    loss.backward()
                    opt_n.step()
                    
                adapter_n.eval()
                with torch.no_grad():
                    z_refs_step = adapter_n(seen_x)
                    for b in range(10):
                        test_x_b = test_x_blocks[b].to(DEVICE)
                        test_y_b = test_y_blocks[b].to(DEVICE)
                        z_queries = adapter_n(test_x_b)
                        correct = 0
                        for q_idx, q_vec in enumerate(z_queries):
                            sims = torch.matmul(z_refs_step, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_idx = torch.argmax(sims).item()
                            if seen_y[best_idx].item() == test_y_b[q_idx].item():
                                correct += 1
                        R_n[step, b] = correct / len(z_queries)
                        
            # 2. Run BOTTOM-32
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            adapter_b = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
            opt_b = torch.optim.AdamW(adapter_b.parameters(), lr=1e-3, weight_decay=1e-4)
            R_b = np.zeros((10, 10))
            
            adapter_b.train()
            for ep in range(30):
                proj = adapter_b(joint_x_base)
                loss = supervised_contrastive_loss(proj, joint_y_base, tau=0.05)
                opt_b.zero_grad()
                loss.backward()
                opt_b.step()
                
            M_past = joint_x_base.clone().detach()
            
            for step in range(5, 10):
                curr_block = order[step]
                seen_blocks = order[:step + 1]
                seen_x = torch.cat([train_x_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                seen_y = torch.cat([train_y_blocks[b] for b in seen_blocks], dim=0).to(DEVICE)
                curr_x = train_x_blocks[curr_block].to(DEVICE)
                curr_y = train_y_blocks[curr_block].to(DEVICE)
                
                _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                P = Vh[-32:].T
                I_dim = torch.eye(INPUT_DIM, device=DEVICE)
                proj_mat = I_dim - torch.matmul(P, P.T)
                
                adapter_b.train()
                for ep in range(30):
                    proj = adapter_b(curr_x)
                    loss = supervised_contrastive_loss(proj, curr_y, tau=0.05)
                    opt_b.zero_grad()
                    loss.backward()
                    adapter_b.linear.weight.grad = torch.matmul(adapter_b.linear.weight.grad, proj_mat)
                    opt_b.step()
                    
                M_past = torch.cat([M_past, curr_x.clone().detach()], dim=0)
                
                adapter_b.eval()
                with torch.no_grad():
                    z_refs_step = adapter_b(seen_x)
                    for b in range(10):
                        test_x_b = test_x_blocks[b].to(DEVICE)
                        test_y_b = test_y_blocks[b].to(DEVICE)
                        z_queries = adapter_b(test_x_b)
                        correct = 0
                        for q_idx, q_vec in enumerate(z_queries):
                            sims = torch.matmul(z_refs_step, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_idx = torch.argmax(sims).item()
                            if seen_y[best_idx].item() == test_y_b[q_idx].item():
                                correct += 1
                        R_b[step, b] = correct / len(z_queries)
                        
            w_diff = (adapter_b.linear.weight - adapter_n.linear.weight).abs().max().item()
            max_w_diffs.append(w_diff)
            
            naive_ats.append(np.mean(R_n[9, :]))
            bottom_ats.append(np.mean(R_b[9, :]))
            naive_Rs.append(R_n)
            bottom_Rs.append(R_b)
            
    r_diffs = [np.abs(bottom_Rs[i] - naive_Rs[i]).max() for i in range(15)]
    at_diffs = [abs(bottom_ats[i] - naive_ats[i]) for i in range(15)]
    
    print(f"  * Per-run |A_T(bottom32) - A_T(naive)| : Mean = {np.mean(at_diffs)*100:.4f}% | Max = {np.max(at_diffs)*100:.4f}%")
    print(f"  * Step 9 Max |W_bottom32 - W_naive|      : Mean = {np.mean(max_w_diffs):.8e} | Max = {np.max(max_w_diffs):.8e}")
    print(f"  * Max elementwise |R_bottom32 - R_naive| : Mean = {np.mean(r_diffs):.8f} | Max = {np.max(r_diffs):.8f}")
    print("="*100)

def run_50run_experiment_suite(block_assignment, cache_data, seeds=list(range(101, 106)), num_shuffles=10):
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
        
    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)
        
    conditions = [
        "naive", "offline",
        "OGP_k4", "OGP_k8", "OGP_k16", "OGP_k24", "OGP_k32", "OGP_k64", "OGP_k128",
        "RANDOM-32", "BOTTOM-32", "CURRENT-32"
    ]
    
    suite_results = {c: [] for c in conditions}
    
    for c_name in conditions:
        for shuffle_idx, order in enumerate(order_list):
            for seed in seeds:
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
                for ep in range(45 if c_name == "offline" else 30):
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
                    
                    proj_mat = None
                    if c_name.startswith("OGP_k"):
                        k_val = int(c_name.split("_k")[1])
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        actual_k = min(k_val, Vh.shape[0])
                        P = Vh[:actual_k].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                    elif c_name == "RANDOM-32":
                        R_mat = torch.randn(INPUT_DIM, 32, device=DEVICE)
                        Q, _ = torch.linalg.qr(R_mat)
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(Q, Q.T)
                    elif c_name == "BOTTOM-32":
                        _, S, Vh = torch.linalg.svd(M_past, full_matrices=False)
                        P = Vh[-32:].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                    elif c_name == "CURRENT-32":
                        _, S, Vh = torch.linalg.svd(curr_x, full_matrices=False)
                        P = Vh[:32].T
                        proj_mat = torch.eye(INPUT_DIM, device=DEVICE) - torch.matmul(P, P.T)
                        
                    adapter.train()
                    train_data_x = seen_ref_x if c_name == "offline" else curr_x
                    train_data_y = seen_ref_y if c_name == "offline" else curr_y
                    
                    for ep in range(30):
                        proj = adapter(train_data_x)
                        loss = supervised_contrastive_loss(proj, train_data_y, tau=0.05)
                        optimizer.zero_grad()
                        loss.backward()
                        if proj_mat is not None and adapter.linear.weight.grad is not None:
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
                suite_results[c_name].append({"A_T": a_t, "LA": la, "mean_forgetting": fgt})
                
    return suite_results

def print_master_suite_summary(suite_results, title="50-RUN MASTER SUITE RESULTS"):
    print("\n" + "="*140)
    print(f"  {title}")
    print("="*140)
    
    nai_runs = suite_results["naive"]
    off_runs = suite_results["offline"]
    
    nai_ats = [r["A_T"] for r in nai_runs]
    nai_las = [r["LA"] for r in nai_runs]
    nai_fgts = [r["mean_forgetting"] for r in nai_runs]
    
    off_ats = [r["A_T"] for r in off_runs]
    off_las = [r["LA"] for r in off_runs]
    off_fgts = [r["mean_forgetting"] for r in off_runs]
    
    off_mean_at = np.mean(off_ats) * 100.0
    nai_mean_at = np.mean(nai_ats) * 100.0
    
    print(f"  * Baseline Naive (n={len(nai_runs)})   : A_T={nai_mean_at:.2f}% ± {np.std(nai_ats)*100:.2f}% (Min: {np.min(nai_ats)*100:.2f}%, Max: {np.max(nai_ats)*100:.2f}%) | LA={np.mean(nai_las)*100:.2f}% | Fgt={np.mean(nai_fgts)*100:.2f}%")
    print(f"  * Upper Bound Offline (n={len(off_runs)}): A_T={off_mean_at:.2f}% ± {np.std(off_ats)*100:.2f}% | LA={np.mean(off_las)*100:.2f}% | Fgt={np.mean(off_fgts)*100:.2f}%")
    print(f"  * Total Continual Learning Gap   : {off_mean_at - nai_mean_at:+.2f}%")
    print("-" * 140)
    
    header = f"{'Condition':18s} | {'A_T (Min..Max)':25s} | {'LA (Learning)':15s} | {'Observed Fgt':15s} | {'CL Gap':8s} | {'Diff A_T vs Naive (95% CI)':28s} | {'Diff Fgt vs Naive (95% CI)':28s}"
    print(header)
    print("-" * len(header))
    
    for c_name, runs in suite_results.items():
        if c_name in ["naive", "offline"]:
            continue
        ats = [r["A_T"] for r in runs]
        las = [r["LA"] for r in runs]
        fgts = [r["mean_forgetting"] for r in runs]
        
        at_m = np.mean(ats) * 100.0
        at_s = np.std(ats) * 100.0
        at_min = np.min(ats) * 100.0
        at_max = np.max(ats) * 100.0
        
        la_m = np.mean(las) * 100.0
        fgt_m = np.mean(fgts) * 100.0
        
        cl_gap = off_mean_at - at_m
        
        diff_at_m, ci_at_l, ci_at_u = bootstrap_paired_ci(ats, nai_ats)
        diff_fgt_m, ci_fgt_l, ci_fgt_u = bootstrap_paired_ci(fgts, nai_fgts)
        
        verdict = ""
        if ci_at_l > 0.0:
            verdict = "SUCCESS (+A_T)"
        elif ci_at_u < 0.0:
            verdict = "SIG WORSE (-A_T)"
        else:
            verdict = "TRUE NULL"
            
        row_str = (
            f"  {c_name:18s} | "
            f"{at_m:5.2f}% ± {at_s:4.2f}% ({at_min:5.2f}..{at_max:5.2f}%) | "
            f"{la_m:5.2f}%          | "
            f"{fgt_m:5.2f}%          | "
            f"{cl_gap:+5.2f}%   | "
            f"{diff_at_m*100:+5.2f}% [{ci_at_l*100:+5.2f}%, {ci_at_u*100:+5.2f}%] | "
            f"{diff_fgt_m*100:+5.2f}% [{ci_fgt_l*100:+5.2f}%, {ci_fgt_u*100:+5.2f}%] {verdict}"
        )
        print(row_str)
    print("="*140)

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
    
    # 1. Item 2 Literal No-Op Audit
    run_bottom32_literal_noop_audit(block_assignment, cache_data)
    
    # 2. 50-Run Master Suite on Selection Seeds (101..105, 10 shuffles x 5 seeds = 50 runs)
    res_select_50 = run_50run_experiment_suite(block_assignment, cache_data, seeds=[101, 102, 103, 104, 105], num_shuffles=10)
    print_master_suite_summary(res_select_50, title="50-RUN MASTER SUITE RESULTS (SELECTION SEEDS 101..105, 50 RUNS)")
    
    # 3. 50-Run Master Suite on Fresh Seeds (211..215, 10 shuffles x 5 seeds = 50 runs)
    res_fresh_50 = run_50run_experiment_suite(block_assignment, cache_data, seeds=[211, 212, 213, 214, 215], num_shuffles=10)
    print_master_suite_summary(res_fresh_50, title="50-RUN MASTER SUITE RESULTS (FRESH SEEDS 211..215, 50 RUNS)")

if __name__ == "__main__":
    main()
