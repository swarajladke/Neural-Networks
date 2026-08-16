"""
build_report_tables.py
======================

Directive S1(c):
Reads 'phase_iv_results.json' and prints the exact Markdown table rows for walkthrough.md.
Eliminates all hand-transcription errors.
"""

import json
import os

JSON_PATH = "phase_iv_results.json"


def main():
    if not os.path.isfile(JSON_PATH):
        print(f"ERROR: '{JSON_PATH}' not found. Run 'run_p3_to_p6_phase_iv_matrix.py' first.")
        return

    with open(JSON_PATH, "r") as f:
        d = json.load(f)

    headl1c_agg = d["headl1c_family"]["aggregate"]
    ncm_fam = d["ncm_family"]

    joint_h = headl1c_agg["joint_offline_headl1c"]
    naive = headl1c_agg["naive_l1c"]
    freeze_h = headl1c_agg["freeze_after_base"]

    joint_ncm = ncm_fam["joint_offline_ncm"]
    ncm_inc = ncm_fam["ncm_incremental"]
    freeze_ncm = ncm_fam["freeze_after_base_ncm"]

    print("=========================================================================================================")
    print(" GENERATED MARKDOWN ROWS FOR walkthrough.md (Section 6: Multi-Seed Phase IV Results)")
    print("=========================================================================================================")
    print("\n### Multi-Seed Aggregate Results (Mean $\\pm$ Std derived strictly from $R[t,i]$):")
    print("#### HeadL1c Classifier Family:")
    print(f"- **`joint_offline_headl1c`**: Final Test Accuracy = **${joint_h['mean']:.2f}\\% \\pm {joint_h['std']:.2f}\\%$** (HeadL1c architecture)")
    print(f"- **`naive_l1c`**: Final $\\text{{ACC}}_T = \\mathbf{{{naive['acc_T_mean']:.2f}\\% \\pm {naive['acc_T_std']:.2f}\\%}}$ | $\\text{{BWT}} = \\mathbf{{{naive['bwt_mean']:+.2f}\\% \\pm {naive['bwt_std']:.2f}\\%}}$ | $\\text{{Forgetting}} = \\mathbf{{{naive['forgetting_mean']:.2f}\\% \\pm {naive['forgetting_std']:.2f}\\%}}$ (severe catastrophic forgetting).")
    print(f"- **`freeze_after_base` (R1 Standing Control)**: Final $\\text{{ACC}}_T = \\mathbf{{{freeze_h['acc_T_mean']:.2f}\\% \\pm {freeze_h['acc_T_std']:.2f}\\%}}$ | $\\text{{BWT}}$ and $\\text{{Forgetting}}$ omitted per Rule R16 (identically 0.00% by construction).")
    print(f"  - Structural Ceiling: **10.00%** ($10/100 \\times 100$). Reports **{freeze_h['acc_T_mean']/10.0*100:.1f}%** of base-only ceiling.")

    print("\n#### NCM Classifier Family:")
    print(f"- **`joint_offline_ncm`**: Final Test Accuracy = **{joint_ncm['acc']:.2f}%** (batch centroids)")
    print(f"- **`ncm_incremental`**: Final Test Accuracy = **{ncm_inc['acc']:.2f}%** | $\\text{{BWT}} = \\mathbf{{{ncm_inc['bwt']:+.2f}\\%}}$ | $\\text{{Forgetting}} = \\mathbf{{{ncm_inc['forgetting']:.2f}\\%}}$")
    print(f"- **`freeze_after_base_ncm`**: Final Test Accuracy = **{freeze_ncm['acc']:.2f}%** | $\\text{{BWT}}$ and $\\text{{Forgetting}}$ omitted per Rule R16.")
    print("=========================================================================================================\n")

if __name__ == "__main__":
    main()
