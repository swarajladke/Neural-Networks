"""
recompute_standard_metrics.py — Standalone Re-computation of Standard Continual Learning Metrics
===================================================================================================
Loads saved trajectory files (trajectories_all.json or trajectories_*.json) and re-computes standard
continual-learning metrics from the saved 10x10 R matrices across all 15 runs per condition:

  - A_T (Final Average Accuracy)  = mean over j of R[9, j]
  - LA  (Learning Accuracy)       = mean over j of R[order.index(j), j]
  - Forgetting (Standard)         = mean over j of ( max_{t >= order.index(j)} R[t, j] - R[9, j] )
  - Forgetting (Floored, Legacy)  = mean over j (t_j < 9) of ( max_{t in [max(4, t_j), 8]} R[t, j] - R[9, j] )
  - BWT (Standard)                = mean over j of ( R[9, j] - R[order.index(j), j] )
  - BWT (Floored, Legacy)         = mean over j (t_j < 9) of ( R[9, j] - R[max(4, t_j), j] )
  - Worst Forgetting (Standard)   = max over j of ( max_{t >= order.index(j)} R[t, j] - R[9, j] )
"""

import os
import glob
import json
import numpy as np

def load_trajectories():
    trajectories_by_cond = {}
    
    # Priority order for finding trajectory files
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
                        if cond not in trajectories_by_cond and isinstance(run_list, list):
                            trajectories_by_cond[cond] = run_list
        except Exception as e:
            print(f"[Notice] Error reading {filepath}: {e}")
            
    return trajectories_by_cond

def analyze_trajectories(trajectories_by_cond):
    print("\n" + "="*120)
    print("  STANDARD CONTINUAL LEARNING METRICS (A_T, LA, Standard Forgetting, Standard BWT)")
    print("="*120)
    header = f"{'Condition':40s} | {'A_T (Final Acc)':15s} | {'LA (Learn Acc)':15s} | {'Forgetting (Std)':18s} | {'Forgetting (Floor)':18s} | {'BWT (Std)':15s} | {'BWT (Floor)':15s}"
    print(header)
    print("-" * len(header))
    
    results_summary = {}
    
    for cond, runs in trajectories_by_cond.items():
        A_T_runs = []
        LA_runs = []
        fgt_std_runs = []
        fgt_flr_runs = []
        bwt_std_runs = []
        bwt_flr_runs = []
        worst_fgt_std_runs = []
        worst_fgt_flr_runs = []
        
        R_matrices = []
        
        for run in runs:
            order = run["order"]
            R = np.array(run["R_matrix"])
            R_matrices.append(R)
            
            # 1. Final Average Accuracy A_T = mean_j R[9, j]
            a_t = np.mean(R[9, :])
            A_T_runs.append(a_t)
            
            # 2. Learning Accuracy LA = mean_j R[order.index(j), j]
            la_vals = [R[order.index(j), j] for j in range(10)]
            la = np.mean(la_vals)
            LA_runs.append(la)
            
            # 3. Standard Forgetting = mean_j (max_{t >= order.index(j)} R[t, j] - R[9, j])
            fgt_j_std = []
            fgt_j_flr = []
            bwt_j_std = []
            bwt_j_flr = []
            
            for j in range(10):
                t_j = order.index(j)
                # Standard
                max_t_std = np.max(R[t_j:10, j])
                fgt_std = max_t_std - R[9, j]
                fgt_j_std.append(fgt_std)
                
                bwt_std = R[9, j] - R[t_j, j]
                bwt_j_std.append(bwt_std)
                
                # Floored (Legacy bug)
                if t_j < 9:
                    start_flr = max(4, t_j)
                    max_t_flr = np.max(R[start_flr:9, j])
                    fgt_flr = max_t_flr - R[9, j]
                    fgt_j_flr.append(fgt_flr)
                    
                    bwt_flr = R[9, j] - R[start_flr, j]
                    bwt_j_flr.append(bwt_flr)
                    
            fgt_std_runs.append(np.mean(fgt_j_std))
            worst_fgt_std_runs.append(np.max(fgt_j_std))
            bwt_std_runs.append(np.mean(bwt_j_std))
            
            if len(fgt_j_flr) > 0:
                fgt_flr_runs.append(np.mean(fgt_j_flr))
                worst_fgt_flr_runs.append(np.max(fgt_j_flr))
                bwt_flr_runs.append(np.mean(bwt_j_flr))
                
        mean_R = np.mean(R_matrices, axis=0) if len(R_matrices) > 0 else np.zeros((10, 10))
        
        results_summary[cond] = {
            "A_T_mean": np.mean(A_T_runs), "A_T_std": np.std(A_T_runs),
            "LA_mean": np.mean(LA_runs), "LA_std": np.std(LA_runs),
            "fgt_std_mean": np.mean(fgt_std_runs), "fgt_std_std": np.std(fgt_std_runs),
            "fgt_flr_mean": np.mean(fgt_flr_runs), "fgt_flr_std": np.std(fgt_flr_runs),
            "bwt_std_mean": np.mean(bwt_std_runs), "bwt_std_std": np.std(bwt_std_runs),
            "bwt_flr_mean": np.mean(bwt_flr_runs), "bwt_flr_std": np.std(bwt_flr_runs),
            "worst_fgt_std_mean": np.mean(worst_fgt_std_runs),
            "worst_fgt_std_95th": np.percentile(worst_fgt_std_runs, 95),
            "worst_fgt_std_max": np.max(worst_fgt_std_runs),
            "mean_R_matrix": mean_R.tolist()
        }
        
        row = (
            f"  {cond:40s} | "
            f"{np.mean(A_T_runs)*100:6.2f}% ± {np.std(A_T_runs)*100:4.2f}% | "
            f"{np.mean(LA_runs)*100:6.2f}% ± {np.std(LA_runs)*100:4.2f}% | "
            f"{np.mean(fgt_std_runs)*100:6.2f}% ± {np.std(fgt_std_runs)*100:4.2f}% | "
            f"{np.mean(fgt_flr_runs)*100:6.2f}% ± {np.std(fgt_flr_runs)*100:4.2f}% | "
            f"{np.mean(bwt_std_runs)*100:6.2f}% ± {np.std(bwt_std_runs)*100:4.2f}% | "
            f"{np.mean(bwt_flr_runs)*100:6.2f}% ± {np.std(bwt_flr_runs)*100:4.2f}%"
        )
        print(row)
        
    print("="*120)
    
    # Quantify Metric Floor Impact
    print("\n" + "-"*80)
    print("  QUANTIFICATION OF METRIC FLOOR UNDERSTATEMENT IMPACT")
    print("-"*80)
    for cond, s in results_summary.items():
        diff_fgt = s["fgt_std_mean"] - s["fgt_flr_mean"]
        diff_bwt = s["bwt_std_mean"] - s["bwt_flr_mean"]
        print(f"  * {cond:40s} -> Forgetting Understated By: {diff_fgt*100:+.2f}% | BWT Shift: {diff_bwt*100:+.2f}%")
    print("="*80)

    # Print 10x10 Mean R Matrix per Condition
    print("\n" + "="*80)
    print("  10x10 MEAN RECALL MATRICES R[t, j] (ROUNDS t=0..9 (ROWS) x BLOCKS j=0..9 (COLS))")
    print("="*80)
    for cond, s in results_summary.items():
        print(f"\n  --- Condition: {cond} ---")
        R_m = np.array(s["mean_R_matrix"])
        header_cols = "       " + " ".join([f"Blk_{j:<2d}" for j in range(10)])
        print(header_cols)
        for t in range(10):
            row_str = f"Step_{t}: " + " ".join([f"{R_m[t, j]*100:5.1f}%" for j in range(10)])
            print(row_str)
    print("="*80)
    
    with open("standard_cl_metrics_report.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n[OK] Saved standard CL metrics report to standard_cl_metrics_report.json")

if __name__ == "__main__":
    trajs = load_trajectories()
    if not trajs:
        print("[Notice] No trajectory files found. Run validation experiment first.")
    else:
        analyze_trajectories(trajs)
