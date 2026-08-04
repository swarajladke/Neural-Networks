"""
run_training_intensity_dial.py — Training Intensity Dial Sweep Experiment
==========================================================================
Sweeps per-block training intensity across 6 cells:
  epochs in {10, 30, 100} x lr in {1e-3, 5e-3}

For each cell, evaluates naive_sequential_adapter and offline_adapter (5 shuffles x 3 seeds).
Reports the Continual Learning Gap (offline - naive) with paired 95% CIs per cell,
and identifies the cell yielding the largest CL gap.
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

def run_intensity_cell(all_facts, cache_data, condition="naive_sequential_adapter", epochs=10, lr=1e-3, shuffles=5, seeds=3):
    results = []
    blocks = [all_facts[i*10 : (i+1)*10] for i in range(10)]
    
    random.seed(42)
    order_list = []
    for _ in range(shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)
        
    for shuffle_idx, order in enumerate(order_list):
        for seed in range(101, 101 + seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
            R = np.zeros((10, 10))
            
            base_blocks = order[:5]
            joint_train_x_base = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in base_blocks], dim=0).to(DEVICE)
            joint_train_y_base = torch.cat([cache_data["train_y"][b*30 : (b+1)*30] for b in base_blocks], dim=0).to(DEVICE)
            
            if condition == "offline_adapter":
                adapter.train()
                for ep in range(int(epochs * 1.5)):
                    proj = adapter(joint_train_x_base)
                    loss = supervised_contrastive_loss(proj, joint_train_y_base, tau=0.05)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            elif condition == "naive_sequential_adapter":
                adapter.train()
                for ep in range(epochs):
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
                    test_x_b = cache_data["test_x"][b*40 : (b+1)*40].to(DEVICE)
                    test_y_b = cache_data["test_y"][b*40 : (b+1)*40].to(DEVICE)
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
                seen_ref_x = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in seen_blocks], dim=0).to(DEVICE)
                seen_ref_y = torch.cat([cache_data["train_y"][b*30 : (b+1)*30] for b in seen_blocks], dim=0).to(DEVICE)
                
                if condition == "naive_sequential_adapter":
                    adapter.train()
                    curr_x = cache_data["train_x"][curr_block*30 : (curr_block+1)*30].to(DEVICE)
                    curr_y = cache_data["train_y"][curr_block*30 : (curr_block+1)*30].to(DEVICE)
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
                        test_x_b = cache_data["test_x"][b*40 : (b+1)*40].to(DEVICE)
                        test_y_b = cache_data["test_y"][b*40 : (b+1)*40].to(DEVICE)
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
                "epochs": epochs,
                "lr": lr,
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
    print("  EXPERIMENT 3(b): TRAINING-INTENSITY DIAL SWEEP EVALUATION (6 CELLS)")
    print("="*110)
    
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b]
    
    MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
    if not os.path.exists(CACHE_100_PATH):
        print(f"[Cache] Embeddings cache {CACHE_100_PATH} not found. Reconstructing automatically...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    else:
        cache_data = torch.load(CACHE_100_PATH, map_location=DEVICE)
        
    epochs_grid = [10, 30, 100]
    lr_grid = [1e-3, 5e-3]
    
    cell_results = []
    
    header = f"{'Cell (Epochs, LR)':22s} | {'A_T(offline)':15s} | {'A_T(naive)':15s} | {'CL Gap (offline - naive)':28s}"
    print("\n" + header)
    print("-" * len(header))
    
    largest_gap = -999.0
    largest_cell = None
    
    for ep in epochs_grid:
        for lr_val in lr_grid:
            cell_name = f"epochs={ep}, lr={lr_val:.0e}"
            
            res_naive = run_intensity_cell(all_facts, cache_data, condition="naive_sequential_adapter", epochs=ep, lr=lr_val)
            res_offline = run_intensity_cell(all_facts, cache_data, condition="offline_adapter", epochs=ep, lr=lr_val)
            
            off_ats = [r["A_T"] for r in res_offline]
            nai_ats = [r["A_T"] for r in res_naive]
            
            off_mean = np.mean(off_ats) * 100.0
            nai_mean = np.mean(nai_ats) * 100.0
            
            gap_mean, ci_l, ci_u = bootstrap_paired_ci(off_ats, nai_ats)
            gap_pct = gap_mean * 100.0
            
            if gap_pct > largest_gap:
                largest_gap = gap_pct
                largest_cell = (ep, lr_val, off_mean, nai_mean, gap_pct)
                
            row_str = (
                f"  {cell_name:20s} | "
                f"{off_mean:6.2f}%         | "
                f"{nai_mean:6.2f}%        | "
                f"{gap_pct:+6.2f}% (95% CI: [{ci_l*100:+6.2f}%, {ci_u*100:+6.2f}%])"
            )
            print(row_str)
            
            cell_results.append({
                "epochs": ep,
                "lr": lr_val,
                "offline_A_T": off_mean,
                "naive_A_T": nai_mean,
                "cl_gap": gap_pct,
                "ci_lower": ci_l * 100.0,
                "ci_upper": ci_u * 100.0
            })
            
    print("="*110)
    print("\n" + "="*80)
    print("  LARGEST CONTINUAL LEARNING GAP CELL FOUND")
    print("="*80)
    ep_b, lr_b, off_b, nai_b, gap_b = largest_cell
    print(f"  * Largest CL Gap Cell : epochs={ep_b}, lr={lr_b:.0e}")
    print(f"  * A_T(offline)        : {off_b:.2f}%")
    print(f"  * A_T(naive)          : {nai_b:.2f}%")
    print(f"  * CL Gap              : {gap_b:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
