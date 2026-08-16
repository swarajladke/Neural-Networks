"""
build_report_tables.py
======================

Directive T5 (S3), S4:
Reads 'phase_iv_results.json' and outputs formatted Markdown tables and text for walkthrough.md:
  1. Per-seed HeadL1c results table with programmatic mean and std asserts.
  2. Multi-seed aggregate summary lines (omitting ratio-to-ceiling per T3).
  3. Per-column NCM BWT decomposition table (S4).
"""

import json
import math
import os

JSON_PATH = "phase_iv_results.json"


def sample_std(vals, mean_val):
    if len(vals) <= 1:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in vals) / (len(vals) - 1))


def main():
    if not os.path.isfile(JSON_PATH):
        print(f"ERROR: '{JSON_PATH}' not found. Run 'run_p3_to_p6_phase_iv_matrix.py' first.")
        return

    with open(JSON_PATH, "r") as f:
        d = json.load(f)

    per_seed = d["headl1c_family"]["per_seed"]
    seeds = [r["seed"] for r in per_seed]
    naive_acc_list = [r["naive_acc_T"] for r in per_seed]
    naive_bwt_list = [r["naive_bwt"] for r in per_seed]
    naive_fgt_list = [r["naive_forgetting"] for r in per_seed]
    freeze_acc_list = [r["freeze_acc_T"] for r in per_seed]
    joint_acc_list = [r["joint_offline_headl1c"] for r in per_seed]

    # Compute exact means and sample standard deviations
    naive_acc_m = sum(naive_acc_list) / len(naive_acc_list)
    naive_acc_s = sample_std(naive_acc_list, naive_acc_m)

    naive_bwt_m = sum(naive_bwt_list) / len(naive_bwt_list)
    naive_bwt_s = sample_std(naive_bwt_list, naive_bwt_m)

    naive_fgt_m = sum(naive_fgt_list) / len(naive_fgt_list)
    naive_fgt_s = sample_std(naive_fgt_list, naive_fgt_m)

    freeze_acc_m = sum(freeze_acc_list) / len(freeze_acc_list)
    freeze_acc_s = sample_std(freeze_acc_list, freeze_acc_m)

    joint_acc_m = sum(joint_acc_list) / len(joint_acc_list)
    joint_acc_s = sample_std(joint_acc_list, joint_acc_m)

    # T5 Assertions: Assert aggregate values match sum(per_seed)/5 to 1e-9
    agg = d["headl1c_family"]["aggregate"]
    assert abs(agg["naive_l1c"]["acc_T_mean"] - naive_acc_m) < 1e-9
    assert abs(agg["naive_l1c"]["acc_T_std"] - naive_acc_s) < 1e-9
    assert abs(agg["naive_l1c"]["bwt_mean"] - naive_bwt_m) < 1e-9
    assert abs(agg["naive_l1c"]["bwt_std"] - naive_bwt_s) < 1e-9
    assert abs(agg["naive_l1c"]["forgetting_mean"] - naive_fgt_m) < 1e-9
    assert abs(agg["naive_l1c"]["forgetting_std"] - naive_fgt_s) < 1e-9
    assert abs(agg["freeze_after_base"]["acc_T_mean"] - freeze_acc_m) < 1e-9
    assert abs(agg["freeze_after_base"]["acc_T_std"] - freeze_acc_s) < 1e-9
    assert abs(agg["joint_offline_headl1c"]["mean"] - joint_acc_m) < 1e-9
    assert abs(agg["joint_offline_headl1c"]["std"] - joint_acc_s) < 1e-9

    ncm_fam = d["ncm_family"]
    joint_ncm = ncm_fam["joint_offline_ncm"]
    ncm_inc = ncm_fam["ncm_incremental"]
    freeze_ncm = ncm_fam["freeze_after_base_ncm"]

    print("=========================================================================================================")
    print(" GENERATED MARKDOWN ROWS FOR walkthrough.md (Section 6: Multi-Seed Phase IV Results)")
    print("=========================================================================================================")
    
    print("\n### 1. HeadL1c Per-Seed Results Table (T5 / S3):")
    print("| Seed | naive ACC_T | naive BWT | naive Forgetting | freeze ACC_T | joint ACC |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in per_seed:
        print(f"| {r['seed']} | {r['naive_acc_T']:.2f}% | {r['naive_bwt']:+.2f}% | {r['naive_forgetting']:.2f}% | {r['freeze_acc_T']:.2f}% | {r['joint_offline_headl1c']:.2f}% |")
    print(f"| **mean $\\pm$ std** | **{naive_acc_m:.2f}% $\\pm$ {naive_acc_s:.2f}%** | **{naive_bwt_m:+.2f}% $\\pm$ {naive_bwt_s:.2f}%** | **{naive_fgt_m:.2f}% $\\pm$ {naive_fgt_s:.2f}%** | **{freeze_acc_m:.2f}% $\\pm$ {freeze_acc_s:.2f}%** | **{joint_acc_m:.2f}% $\\pm$ {joint_acc_s:.2f}%** |")

    print("\n### 2. Multi-Seed Aggregate Results (Mean $\\pm$ Std derived strictly from $R[t,i]$):")
    print("#### HeadL1c Classifier Family:")
    print(f"- **`joint_offline_headl1c`**: Final Test Accuracy = **${joint_acc_m:.2f}\\% \\pm {joint_acc_s:.2f}\\%$** (HeadL1c architecture)")
    print(f"- **`naive_l1c`**: Final $\\text{{ACC}}_T = \\mathbf{{{naive_acc_m:.2f}\\% \\pm {naive_acc_s:.2f}\\%}}$ | $\\text{{BWT}} = \\mathbf{{{naive_bwt_m:+.2f}\\% \\pm {naive_bwt_s:.2f}\\%}}$ | $\\text{{Forgetting}} = \\mathbf{{{naive_fgt_m:.2f}\\% \\pm {naive_fgt_s:.2f}\\%}}$ (severe catastrophic forgetting).")
    print(f"- **`freeze_after_base` (R1 Standing Control)**: Final $\\text{{ACC}}_T = \\mathbf{{{freeze_acc_m:.2f}\\% \\pm {freeze_acc_s:.2f}\\%}}$, against chance 1.00% and base-block-only 10.00%. $\\text{{BWT}}$ and $\\text{{Forgetting}}$ omitted per Rule R16 (identically 0.00% by construction).")
    print(f"  - Note: Off-block non-zeros are random-init artefacts (untrained head rows winning the argmax), as seen from R[t][1] = 6.0% and R[t][5] = 2.0% in the seed-42 freeze matrix.")

    print("\n#### NCM Classifier Family:")
    print(f"- **`joint_offline_ncm`**: Final Test Accuracy = **{joint_ncm['acc']:.2f}%** (batch centroids)")
    print(f"- **`ncm_incremental`**: Final Test Accuracy = **{ncm_inc['acc']:.2f}%** | $\\text{{BWT}} = \\mathbf{{{ncm_inc['bwt']:+.2f}\\%}}$ | $\\text{{Forgetting}} = \\mathbf{{{ncm_inc['forgetting']:.2f}\\%}}$")
    print(f"- **`freeze_after_base_ncm`**: Final Test Accuracy = **{freeze_ncm['acc']:.2f}%** (base-block only 10.00%). $\\text{{BWT}}$ and $\\text{{Forgetting}}$ omitted per Rule R16.")

    print("\n### 3. NCM Per-Column BWT Decomposition (S4):")
    print("| Column $i$ (Block $i$) | Final Accuracy $R[T-1, i]$ | Learning Time Accuracy $R[i,i]$ | Column Contribution $R[T-1, i] - R[i,i]$ |")
    print("|:---|:---:|:---:|:---:|")
    # Exact values from incremental NCM matrix:
    ncm_r9 = [88.0, 82.0, 82.0, 86.0, 82.0, 86.0, 94.0, 84.0, 90.0, 84.0]
    ncm_rii = [100.0, 96.0, 96.0, 96.0, 92.0, 90.0, 98.0, 90.0, 90.0, 84.0]
    col_diffs = []
    for i in range(9):
        diff = ncm_r9[i] - ncm_rii[i]
        col_diffs.append(diff)
        print(f"| Col $i={i+1}$ (Classes {i*10:02d}-{i*10+9:02d}) | {ncm_r9[i]:.1f}% | {ncm_rii[i]:.1f}% | **{diff:+.1f} pp** |")
    print(f"| **Mean BWT (Col $i=1..9$)** | -- | -- | **{sum(col_diffs)/len(col_diffs):+.2f}%** |")
    print("\n> **Mathematical Explanation (S4)**: Final accuracy strictly equals `joint_offline_ncm` (85.80%) because centroid accumulation is order-invariant. Backward transfer is non-zero (-8.22%) because diagonal entries $R[i,i]$ are measured at step $i$ when fewer total candidate classes have been observed ($10(i+1)$ classes) compared to final step $T-1$ (100 classes).")
    print("=========================================================================================================\n")

if __name__ == "__main__":
    main()
