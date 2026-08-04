"""
recompute_standard_metrics.py — Standalone Re-computation over Populated Matrix Rows (t >= 4)
==============================================================================================
Populated matrix rows: t = 4 (joint base phase for blocks 0-4) and t = 5..9 (sequential steps 5..9).
Rows 0..3 are un-written and remain zero. All metrics are computed strictly over populated rows:

  - A_T (Final Average Accuracy)  = mean over j in [0..9] of R[9, j]
  - LA  (Learning Accuracy)       = mean over j in [0..9] of R[max(4, order.index(j)), j]
  - Forgetting                    = mean over j in [0..9] of ( max_{t in [max(4, order.index(j)), 9]} R[t, j] - R[9, j] )
  - BWT                           = mean over j in [0..9] of ( R[9, j] - R[max(4, order.index(j)), j] )
  - Worst Forgetting              = max over j in [0..9] of ( max_{t in [max(4, order.index(j)), 9]} R[t, j] - R[9, j] )

Includes 10,000-sample bootstrap 95% Confidence Intervals for paired difference (A_T_cond - A_T_frozen).
"""

import os
import glob
import json
import numpy as np

def bootstrap_ci(a_t_cond, a_t_frozen, n_boot=10000, seed=42):
    if not a_t_cond or not a_t_frozen:
        return 0.0, 0.0, 0.0
    n = min(len(a_t_cond), len(a_t_frozen))
    diffs = np.array(a_t_cond[:n]) - np.array(a_t_frozen[:n])
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    return np.mean(diffs), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def load_trajectories():
    trajectories_by_cond = {}
    
    files_to_check = []
    if os.path.exists("trajectories_all.json"):
        files_to_check.append("trajectories_all.json")
    
    chunk_files = sorted(glob.glob("trajectories_*.json"))
    chunk_files = [f for f in chunk_files if f != "trajectories_all.json"]
    files_to_check.extend(chunk_files)
    
    print(f"[Loader] Checking trajectory files: {files_to_check}")
    for filepath in files_to_check:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for cond, run_list in data.items():
                        if isinstance(run_list, list):
                            valid_runs = [r for r in run_list if isinstance(r, dict) and "order" in r and "R_matrix" in r]
                            if valid_runs and cond not in trajectories_by_cond:
                                trajectories_by_cond[cond] = valid_runs
                elif isinstance(data, list):
                    valid_runs = [r for r in data if isinstance(r, dict) and "order" in r and "R_matrix" in r]
                    if valid_runs:
                        cond = valid_runs[0].get("condition", filepath.replace("trajectories_", "").replace(".json", ""))
                        if cond not in trajectories_by_cond:
                            trajectories_by_cond[cond] = valid_runs
        except Exception as e:
            print(f"[Notice] Error reading {filepath}: {e}")
            
    return trajectories_by_cond

def analyze_trajectories(trajectories_by_cond):
    print("\n" + "="*120)
    print("  HEADLINE CONTINUAL LEARNING METRICS COMPUTED OVER POPULATED ROWS (t >= 4)")
    print("="*120)
    header = f"{'Condition':40s} | {'A_T (Final Acc)':18s} | {'LA (Learning Acc)':18s} | {'Forgetting':18s} | {'BWT':18s}"
    print(header)
    print("-" * len(header))
    
    results_summary = {}
    
    for cond, runs in trajectories_by_cond.items():
        A_T_runs = []
        LA_runs = []
        fgt_runs = []
        bwt_runs = []
        worst_fgt_runs = []
        R_matrices = []
        
        for run in runs:
            if not isinstance(run, dict) or "order" not in run or "R_matrix" not in run:
                continue
            order = run["order"]
            R = np.array(run["R_matrix"])
            R_matrices.append(R)
            
            # Populated initial observation index for block j: max(4, order.index(j))
            a_t = np.mean(R[9, :])
            la = np.mean([R[max(4, order.index(j)), j] for j in range(10)])
            
            fgt_j = []
            bwt_j = []
            for j in range(10):
                start_t = max(4, order.index(j))
                max_t = np.max(R[start_t:10, j])
                fgt_j.append(max_t - R[9, j])
                bwt_j.append(R[9, j] - R[start_t, j])
                
            A_T_runs.append(a_t)
            LA_runs.append(la)
            fgt_runs.append(np.mean(fgt_j))
            bwt_runs.append(np.mean(bwt_j))
            worst_fgt_runs.append(np.max(fgt_j))
            
        mean_R = np.mean(R_matrices, axis=0) if len(R_matrices) > 0 else np.zeros((10, 10))
        
        results_summary[cond] = {
            "A_T_runs": A_T_runs,
            "A_T_mean": np.mean(A_T_runs) if A_T_runs else 0.0, "A_T_std": np.std(A_T_runs) if A_T_runs else 0.0,
            "LA_mean": np.mean(LA_runs) if LA_runs else 0.0, "LA_std": np.std(LA_runs) if LA_runs else 0.0,
            "fgt_mean": np.mean(fgt_runs) if fgt_runs else 0.0, "fgt_std": np.std(fgt_runs) if fgt_runs else 0.0,
            "bwt_mean": np.mean(bwt_runs) if bwt_runs else 0.0, "bwt_std": np.std(bwt_runs) if bwt_runs else 0.0,
            "worst_fgt_mean": np.mean(worst_fgt_runs) if worst_fgt_runs else 0.0,
            "worst_fgt_95th": np.percentile(worst_fgt_runs, 95) if worst_fgt_runs else 0.0,
            "worst_fgt_max": np.max(worst_fgt_runs) if worst_fgt_runs else 0.0,
            "mean_R_matrix": mean_R.tolist()
        }
        
        if A_T_runs:
            row = (
                f"  {cond:40s} | "
                f"{np.mean(A_T_runs)*100:6.2f}% ± {np.std(A_T_runs)*100:4.2f}% | "
                f"{np.mean(LA_runs)*100:6.2f}% ± {np.std(LA_runs)*100:4.2f}% | "
                f"{np.mean(fgt_runs)*100:6.2f}% ± {np.std(fgt_runs)*100:4.2f}% | "
                f"{np.mean(bwt_runs)*100:6.2f}% ± {np.std(bwt_runs)*100:4.2f}%"
            )
            print(row)
            
    print("="*120)
    
    # Paired Difference & Bootstrap 95% CI vs Frozen Control
    frozen_runs = results_summary.get("frozen_encoder_writable_memory", {}).get("A_T_runs", [])
    if frozen_runs:
        print("\n" + "-"*90)
        print("  PAIRED A_T DIFFERENCE vs FROZEN CONTROL (A_T_cond - A_T_frozen) & BOOTSTRAP 95% CI")
        print("-"*90)
        for cond, s in results_summary.items():
            if cond == "frozen_encoder_writable_memory":
                continue
            cond_runs = s.get("A_T_runs", [])
            if cond_runs:
                diff_mean, ci_l, ci_u = bootstrap_ci(cond_runs, frozen_runs)
                zero_inc = (ci_l <= 0.0 <= ci_u)
                verdict = "NOT DISTINGUISHABLE FROM ZERO UPDATES" if zero_inc else "STATISTICALLY SIGNIFICANT DIFFERENCE"
                print(f"  * {cond:40s} -> Diff: {diff_mean*100:+.2f}% | 95% CI: [{ci_l*100:+.2f}%, {ci_u*100:+.2f}%] | Verdict: {verdict}")
        print("="*90)

    # Print 10x10 Mean R Matrix per Condition
    print("\n" + "="*90)
    print("  10x10 MEAN RECALL MATRICES R[t, j] (ROWS t=4..9 ARE POPULATED)")
    print("="*90)
    for cond, s in results_summary.items():
        print(f"\n  --- Condition: {cond} ---")
        R_m = np.array(s["mean_R_matrix"])
        header_cols = "       " + " ".join([f"Blk_{j:<2d}" for j in range(10)])
        print(header_cols)
        for t in range(10):
            row_str = f"Step_{t}: " + " ".join([f"{R_m[t, j]*100:5.1f}%" for j in range(10)])
            if t < 4:
                row_str += "  (unpopulated zero row)"
            print(row_str)
    print("="*90)
    
    with open("standard_cl_metrics_report.json", "w") as f:
        # Exclude raw A_T_runs list from json dump
        clean_summary = {}
        for k, v in results_summary.items():
            clean_summary[k] = {rk: rv for rk, rv in v.items() if rk != "A_T_runs"}
        json.dump(clean_summary, f, indent=2)
    print("\n[OK] Saved standard CL metrics report to standard_cl_metrics_report.json")

if __name__ == "__main__":
    trajs = load_trajectories()
    if not trajs:
        print("[Notice] No trajectory files found. Run validation experiment first.")
    else:
        analyze_trajectories(trajs)
