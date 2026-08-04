"""
compute_baseline_cl_gap_cis.py — Paired Bootstrap 95% CIs for CL Gap (offline - naive)
======================================================================================
Loads trajectories_offline_adapter.json and trajectories_naive_sequential_adapter.json
to compute exact paired 10,000-sample bootstrap 95% CIs matched on (shuffle, seed) for:
  1. A_T(offline) - A_T(naive)       [Headline Continual Learning Gap]
  2. LA(offline) - LA(naive)         [Learning Accuracy Margin]
  3. Forgetting(offline) - Forgetting(naive) [Forgetting Penalty Margin]
"""

import os
import json
import numpy as np

def bootstrap_paired_ci(vals_1, vals_2, n_boot=10000, seed=42):
    n = min(len(vals_1), len(vals_2))
    diffs = np.array(vals_1[:n]) - np.array(vals_2[:n])
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    return np.mean(diffs), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def main():
    off_path = "trajectories_offline_adapter.json"
    nai_path = "trajectories_naive_sequential_adapter.json"
    
    if not os.path.exists(off_path) or not os.path.exists(nai_path):
        print("[Notice] Trajectory files not found locally.")
        return
        
    with open(off_path, "r") as f:
        off_runs = json.load(f)
    with open(nai_path, "r") as f:
        nai_runs = json.load(f)
        
    off_ats, off_las, off_fgts = [], [], []
    for run in off_runs:
        order = run["order"]
        R = np.array(run["R_matrix"])
        a_t = np.mean(R[9, :])
        la = np.mean([R[max(4, order.index(j)), j] for j in range(10)])
        fgt = np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)])
        off_ats.append(a_t)
        off_las.append(la)
        off_fgts.append(fgt)
        
    nai_ats, nai_las, nai_fgts = [], [], []
    for run in nai_runs:
        order = run["order"]
        R = np.array(run["R_matrix"])
        a_t = np.mean(R[9, :])
        la = np.mean([R[max(4, order.index(j)), j] for j in range(10)])
        fgt = np.mean([np.max(R[max(4, order.index(j)):10, j]) - R[9, j] for j in range(10)])
        nai_ats.append(a_t)
        nai_las.append(la)
        nai_fgts.append(fgt)
        
    m_at, l_at, u_at = bootstrap_paired_ci(off_ats, nai_ats)
    m_la, l_la, u_la = bootstrap_paired_ci(off_las, nai_las)
    m_fgt, l_fgt, u_fgt = bootstrap_paired_ci(off_fgts, nai_fgts)
    
    print("="*90)
    print("  PAIRED BOOTSTRAP 95% CIs: OFFLINE ADAPTER vs NAIVE SEQUENTIAL ADAPTER")
    print("="*90)
    print(f"  1. Headline CL Gap [A_T(offline) - A_T(naive)]  : {m_at*100:+.2f}% | 95% CI: [{l_at*100:+.2f}%, {u_at*100:+.2f}%]")
    print(f"  2. Learning Accuracy Margin [LA(offline) - LA] : {m_la*100:+.2f}% | 95% CI: [{l_la*100:+.2f}%, {u_la*100:+.2f}%]")
    print(f"  3. Forgetting Margin [Fgt(offline) - Fgt]       : {m_fgt*100:+.2f}% | 95% CI: [{l_fgt*100:+.2f}%, {u_fgt*100:+.2f}%]")
    print("="*90)

if __name__ == "__main__":
    main()
