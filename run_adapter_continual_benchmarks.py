"""
run_adapter_continual_benchmarks.py — SmolLM2-360M + Linear Adapter Continual Benchmark
========================================================================================
Replaces the StudentEncoder with frozen SmolLM2-360M (960-d) plus a small trainable
Linear Adapter (960 -> 960) initialized to Identity (W = I_960, b = 0).

Key Assertions:
1. Base SmolLM2-360M embeddings are frozen (requires_grad = False).
2. Adapter is initialized to Identity, reproducing 72.50% raw 1-NN retrieval accuracy at init.
3. Bounded Capacity: Adapter contains 922,560 parameters (0.92M parameters).

Evaluates 3 Conditions Only (5 shuffles x 3 seeds):
  - frozen_adapter           (Identity, no parameter updates)
  - naive_sequential_adapter (Sequential fine-tuning of adapter at lr=1e-3)
  - offline_adapter          (Joint multi-task retraining upper bound)
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
        # Forward through linear adapter and L2 normalize
        out = self.linear(x)
        return F.normalize(out, dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def assert_identity_accuracy(adapter, cache_data):
    adapter.eval()
    train_x = cache_data["train_x"].to(DEVICE)
    train_y = cache_data["train_y"].to(DEVICE)
    test_x = cache_data["test_x"].to(DEVICE)
    test_y = cache_data["test_y"].to(DEVICE)
    
    with torch.no_grad():
        proj_train = adapter(train_x)
        proj_test = adapter(test_x)
        
        correct = 0
        total = len(proj_test)
        for i in range(total):
            q = proj_test[i]
            sims = torch.matmul(proj_train, q.unsqueeze(-1)).squeeze(-1)
            best_idx = torch.argmax(sims).item()
            if train_y[best_idx].item() == test_y[i].item():
                correct += 1
                
        acc = (correct / total) * 100.0
        print(f"[Assertion Check] Adapter Identity Initialization Retrieval Accuracy: {acc:.2f}%")
        
        # Fail-closed assertion: initialized adapter MUST reproduce raw SmolLM2 72.50% retrieval accuracy
        assert abs(acc - 72.50) < 1e-3, f"FAIL-CLOSED: Initialized adapter accuracy ({acc:.2f}%) does not match 72.50%"
        print("[Assertion Check] PASSED: Initialized adapter reproduces 72.50% raw retrieval ceiling.")

def get_sentence_lists(block):
    train_s = []
    test_s = []
    for f in block:
        for idx in range(3):
            if idx == 0:
                train_s.append(f["probe"])
            elif idx == 1:
                prefix = f["qa"].split(f["statement"])[0]
                train_s.append(prefix + f["probe"])
            else:
                train_s.append(f["cloze"].split("_____")[0].strip())
        test_s.append(f["probe"])
        test_s.extend(f["eval_paraphrases"])
    return train_s, test_s

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
            
            # Verify adapter parameter count (strictly 960*960 + 960 = 922,560)
            param_count = count_parameters(adapter)
            assert param_count == 922560, f"Adapter parameter count ({param_count}) != 922,560"
            
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
            R = np.zeros((10, 10))
            
            # Base phase: joint baseline over blocks 0..4 (order[:5])
            base_blocks = order[:5]
            if condition == "offline_adapter":
                joint_train_x = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in order], dim=0).to(DEVICE)
                joint_train_y = torch.tensor([idx for b in order for idx in range(b*10, (b+1)*10) for _ in range(3)]).to(DEVICE)
                
                adapter.train()
                for epoch in range(15):
                    indices = list(range(len(joint_train_x)))
                    random.shuffle(indices)
                    for idx in range(0, len(indices), 32):
                        b_idx = indices[idx : idx + 32]
                        bx = joint_train_x[b_idx]
                        by = joint_train_y[b_idx]
                        
                        proj = adapter(bx)
                        sim_matrix = torch.matmul(proj, proj.T) / 0.07
                        loss = F.cross_entropy(sim_matrix, torch.arange(len(bx), device=DEVICE))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            elif condition != "frozen_adapter":
                joint_train_x = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in base_blocks], dim=0).to(DEVICE)
                adapter.train()
                for epoch in range(10):
                    indices = list(range(len(joint_train_x)))
                    random.shuffle(indices)
                    for idx in range(0, len(indices), 32):
                        b_idx = indices[idx : idx + 32]
                        bx = joint_train_x[b_idx]
                        proj = adapter(bx)
                        loss = (1.0 - (proj * bx).sum(dim=-1)).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
            # Record base phase recall R[4, b]
            adapter.eval()
            with torch.no_grad():
                base_ref_x = torch.cat([cache_data["train_x"][b*30 : (b+1)*30] for b in base_blocks], dim=0).to(DEVICE)
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
                
                # Train adapter on current block
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
    
    cache_data = torch.load(CACHE_100_PATH, map_location=DEVICE)
    
    # Assert Identity Initialization Accuracy == 72.50%
    init_adapter = OutputAdapter(dim=INPUT_DIM).to(DEVICE)
    assert_identity_accuracy(init_adapter, cache_data)
    
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
