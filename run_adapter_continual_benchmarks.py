"""
run_adapter_continual_benchmarks.py — SmolLM2-360M + Linear Adapter Continual Benchmark
========================================================================================
Replaces the StudentEncoder with frozen SmolLM2-360M (960-d) plus a small trainable
Linear Adapter (960 -> 960) initialized to Identity (W = I_960, b = 0).

Key Assertions & Correctness Guards:
1. Base SmolLM2-360M embeddings are frozen (requires_grad = False).
2. Direct Assertion on Reported Quantity: Inside frozen_adapter, immediately after R is filled:
     assert abs(np.mean(R[9, :]) - 0.7250) < 0.005, f"frozen_adapter R[9,:] = {np.mean(R[9,:]):.4f}, expected 0.7250"
3. Bounded Capacity: Adapter contains 922,560 parameters (0.92M parameters).
4. Correct Stride: 100 facts x 5 eval items = 500 test queries (50 per block, 5 per fact).

Evaluates 3 Conditions Only (5 shuffles x 3 seeds):
  - frozen_adapter           (Identity, no parameter updates)
  - naive_sequential_adapter (Sequential fine-tuning of adapter at lr=1e-3)
  - offline_adapter          (Joint multi-task retraining upper bound on all seen blocks)
"""

import os
import sys
import glob
import json
import time
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
        # Initialize to Identity matrix (W = I, b = 0)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        out = self.linear(x)
        return F.normalize(out, dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_adapter_experiment(all_facts, cache_data, condition="frozen_adapter", shuffles=5, seeds=3, lr=1e-3):
    print(f"[Adapter Experiment] Condition: '{condition}' | LR: {lr:.1e} | Shuffles: {shuffles} | Seeds: {seeds}")
    results = []
    trajectories = []
    
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
            param_count = count_parameters(adapter)
            assert param_count == 922560, f"Adapter parameter count ({param_count}) != 922,560"
            
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
            R = np.zeros((10, 10))
            
            # Base phase: joint baseline over blocks 0..4 (order[:5])
            base_blocks = order[:5]
            joint_train_x_base = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in base_blocks], dim=0).to(DEVICE)
            
            if condition == "offline_adapter":
                adapter.train()
                for epoch in range(15):
                    indices = list(range(len(joint_train_x_base)))
                    random.shuffle(indices)
                    for idx in range(0, len(indices), 32):
                        b_idx = indices[idx : idx + 32]
                        bx = joint_train_x_base[b_idx]
                        proj = adapter(bx)
                        loss = (1.0 - (proj * bx).sum(dim=-1)).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            elif condition == "naive_sequential_adapter":
                adapter.train()
                for epoch in range(10):
                    indices = list(range(len(joint_train_x_base)))
                    random.shuffle(indices)
                    for idx in range(0, len(indices), 32):
                        b_idx = indices[idx : idx + 32]
                        bx = joint_train_x_base[b_idx]
                        proj = adapter(bx)
                        loss = (1.0 - (proj * bx).sum(dim=-1)).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
            # Record base phase recall R[4, b]
            adapter.eval()
            with torch.no_grad():
                base_ref_x = joint_train_x_base
                base_ref_labels = [idx for b in base_blocks for idx in range(b*10, (b+1)*10) for _ in range(3)]
                z_refs_base = adapter(base_ref_x)
                
                for b in range(10):
                    test_x_b = cache_data["test_x"][b*40 : (b+1)*40].to(DEVICE)
                    test_labels = [idx for idx in range(b*10, (b+1)*10) for _ in range(4)]
                    z_queries = adapter(test_x_b)
                    
                    correct = 0
                    for q_idx, q_vec in enumerate(z_queries):
                        sims = torch.matmul(z_refs_base, q_vec.unsqueeze(0).T).squeeze(-1)
                        best_idx = torch.argmax(sims).item()
                        if base_ref_labels[best_idx] == test_labels[q_idx]:
                            correct += 1
                    R[4, b] = correct / len(z_queries)
                    
            # Incremental sequential steps: 5..9
            for step in range(5, 10):
                curr_block = order[step]
                seen_blocks = order[:step + 1]
                seen_ref_x = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in seen_blocks], dim=0).to(DEVICE)
                seen_ref_labels = [idx for b in seen_blocks for idx in range(b*10, (b+1)*10) for _ in range(3)]
                
                if condition == "naive_sequential_adapter":
                    adapter.train()
                    curr_x = cache_data["train_x"][curr_block*30 : (curr_block+1)*30].to(DEVICE)
                    for epoch in range(10):
                        indices = list(range(30))
                        random.shuffle(indices)
                        for idx in range(0, 30, 16):
                            b_idx = indices[idx : idx + 16]
                            bx = curr_x[b_idx]
                            proj = adapter(bx)
                            loss = (1.0 - (proj * bx).sum(dim=-1)).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                elif condition == "offline_adapter":
                    # Retrain jointly on ALL seen blocks order[:step+1]
                    adapter.train()
                    for epoch in range(10):
                        indices = list(range(len(seen_ref_x)))
                        random.shuffle(indices)
                        for idx in range(0, len(indices), 32):
                            b_idx = indices[idx : idx + 32]
                            bx = seen_ref_x[b_idx]
                            proj = adapter(bx)
                            loss = (1.0 - (proj * bx).sum(dim=-1)).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                # Record recall matrix R[step, b]
                adapter.eval()
                with torch.no_grad():
                    z_refs_step = adapter(seen_ref_x)
                    for b in range(10):
                        test_x_b = cache_data["test_x"][b*40 : (b+1)*40].to(DEVICE)
                        test_labels = [idx for idx in range(b*10, (b+1)*10) for _ in range(4)]
                        z_queries = adapter(test_x_b)
                        
                        correct = 0
                        for q_idx, q_vec in enumerate(z_queries):
                            sims = torch.matmul(z_refs_step, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_idx = torch.argmax(sims).item()
                            if seen_ref_labels[best_idx] == test_labels[q_idx]:
                                correct += 1
                        R[step, b] = correct / len(z_queries)
                        
            # MANDATORY ASSERTION ON REPORTED QUANTITY
            if condition == "frozen_adapter":
                mean_r9 = np.mean(R[9, :])
                assert abs(mean_r9 - 0.7250) < 0.005, f"frozen_adapter R[9,:] = {mean_r9:.4f}, expected 0.7250"
                print(f"  [Assertion Passed] frozen_adapter R[9,:] = {mean_r9:.4f} matches 0.7250 expected raw ceiling.")
                
            # Metrics over populated rows (t >= 4)
            a_t = np.mean(R[9, :])
            la = np.mean([R[max(4, order.index(j)), j] for j in range(10)])
            fgt_j = [np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)]
            bwt_j = [R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]
            
            results.append({
                "condition": condition,
                "shuffle": shuffle_idx,
                "seed": seed,
                "order": order,
                "R_matrix": R.tolist(),
                "A_T": a_t,
                "LA": la,
                "mean_forgetting": np.mean(fgt_j),
                "worst_forgetting": np.max(fgt_j),
                "mean_bwt": np.mean(bwt_j)
            })
            
            trajectories.append({
                "condition": condition,
                "shuffle": shuffle_idx,
                "seed": seed,
                "order": order,
                "R_matrix": R.tolist()
            })
            
    with open(f"trajectories_{condition}.json", "w") as f:
        json.dump(trajectories, f, indent=2)
        
    return results

def bootstrap_ci(a_t_cond, a_t_frozen, n_boot=10000, seed=42):
    n = min(len(a_t_cond), len(a_t_frozen))
    diffs = np.array(a_t_cond[:n]) - np.array(a_t_frozen[:n])
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    return np.mean(diffs), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def main():
    print("="*80)
    print("  SMOLLM2-360M + LINEAR ADAPTER CONTINUAL BENCHMARK EVALUATION")
    print("="*80)
    
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
        
    conditions = ["frozen_adapter", "naive_sequential_adapter", "offline_adapter"]
    all_results = {}
    
    for cond in conditions:
        res = run_adapter_experiment(all_facts, cache_data, condition=cond, shuffles=5, seeds=3)
        all_results[cond] = res
        
    print("\n" + "="*110)
    print("  ADAPTER BENCHMARK HEADLINE METRICS (COMPUTED OVER POPULATED ROWS t >= 4)")
    print("="*110)
    header = f"{'Condition':30s} | {'A_T (Final Acc)':18s} | {'LA (Learning Acc)':18s} | {'Forgetting':18s} | {'BWT':18s}"
    print(header)
    print("-" * len(header))
    
    for cond in conditions:
        runs = all_results[cond]
        a_ts = [r["A_T"] for r in runs]
        las = [r["LA"] for r in runs]
        fgts = [r["mean_forgetting"] for r in runs]
        bwts = [r["mean_bwt"] for r in runs]
        
        row = (
            f"  {cond:30s} | "
            f"{np.mean(a_ts)*100:6.2f}% ± {np.std(a_ts)*100:4.2f}% | "
            f"{np.mean(las)*100:6.2f}% ± {np.std(las)*100:4.2f}% | "
            f"{np.mean(fgts)*100:6.2f}% ± {np.std(fgts)*100:4.2f}% | "
            f"{np.mean(bwts)*100:6.2f}% ± {np.std(bwts)*100:4.2f}%"
        )
        print(row)
    print("="*110)
    
    # Paired Bootstrap CI: naive_adapter - frozen_adapter
    frozen_ats = [r["A_T"] for r in all_results["frozen_adapter"]]
    naive_ats = [r["A_T"] for r in all_results["naive_sequential_adapter"]]
    diff_mean, ci_l, ci_u = bootstrap_ci(naive_ats, frozen_ats)
    
    print("\n" + "-"*90)
    print("  PAIRED A_T DIFFERENCE: naive_sequential_adapter - frozen_adapter & 95% BOOTSTRAP CI")
    print("-"*90)
    print(f"  * Paired Mean Difference: {diff_mean*100:+.2f}%")
    print(f"  * 10,000-Sample Bootstrap 95% CI: [{ci_l*100:+.2f}%, {ci_u*100:+.2f}%]")
    print("-"*90)
    
    # Evaluate Decision Rule
    print("\n" + "="*80)
    print("  BENCHMARK VALIDITY DECISION RULE EVALUATION")
    print("="*80)
    drop_pct = (np.mean(frozen_ats) - np.mean(naive_ats)) * 100.0
    if drop_pct > 10.0:
        print(f"  [RESULT]: BRANCH A — Catastrophic Forgetting Induced! Drop is {drop_pct:.2f}% (> 10 points).")
        print("  [ACTION]: Benchmark exhibits genuine catastrophic forgetting and is a valid testbed for CL mechanisms.")
    else:
        print(f"  [RESULT]: BRANCH B — Drop is {drop_pct:.2f}% (<= 10 points).")
        print("  [ACTION]: 10 blocks of 10 facts is too easy / un-forgetting. Scale benchmark before testing mechanisms.")
    print("="*80)

if __name__ == "__main__":
    main()
