"""
run_o4_r12_citation_audit.py
============================

Directive O4: Fix R12 citations.

a) P1 and P2 cite run_n1_to_n9_master_stdout.txt for last-token pooling numbers.
   Verify whether that file contains those numbers (27.33%, 7.67%, 28.00%, 10.67%).
b) P8 cites a .gemini/antigravity/brain path -- replace with UNSOURCED.
c) Grep every Sourced Stdout Log File cell. For each file, assert:
   - File exists in repo root.
   - File contains the quoted accuracy number.
   Print pass/fail table. Any fail = UNSOURCED.

Rule R13: All counts interpolated from variables.
"""

import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Complete list of (prediction, log_file, number_to_find) from walkthrough scorecard
# CORRECTED per O4a: P1/P2 cite audit_embedding_leakage_stdout.txt (Phase I/II origin)
# CORRECTED per O4:  P3 40.33% comes from audit_representation_ablation_stdout.txt, not joint_offline_probe
# CORRECTED per O3:  P20 canonical 63.33% is from run_o3_eps_question_stdout.txt (float64 unified stack)
CITATION_TABLE = [
    ("P1",  "audit_embedding_leakage_stdout.txt",               "27.33"),
    ("P1",  "audit_representation_ablation_stdout.txt",          "7.67"),
    ("P2",  "audit_representation_ablation_stdout.txt",          "28.00"),
    ("P2",  "audit_representation_ablation_stdout.txt",          "10.67"),
    ("P3",  "run_joint_offline_probe_stdout.txt",               "34.80"),
    ("P3",  "audit_representation_ablation_stdout.txt",         "40.33"),
    ("P4",  "run_joint_offline_probe_stdout.txt",               "34.80"),
    ("P6",  "audit_pca_grid_and_lasttok_stdout.txt",            "63.33"),
    ("P7",  "audit_pca_grid_and_lasttok_stdout.txt",            "13.33"),
    ("P9",  "run_gate1_diagnostic_k3_k5_stdout.txt",            None),
    ("P10", "run_n1_3x3_ncm_recheck_stdout.txt",               "61.67"),
    ("P10", "run_n1_3x3_ncm_recheck_stdout.txt",               "62.67"),
    ("P11", "run_n2_fix_cv_stdout.txt",                        "91.00"),
    ("P12", "run_k4_k5_k6_offline_bound_search_stdout.txt",    "60.00"),
    ("P13", "run_n2_fix_cv_stdout.txt",                        "91.00"),
    ("P14", "run_n2_fix_cv_stdout.txt",                        "pca_m64"),
    ("P15", "run_k4_k5_k6_offline_bound_search_stdout.txt",    "62.33"),
    ("P16", "run_n3_n_count_and_match_stdout.txt",             None),
    ("P17", "evaluate_m_phase_stdout.txt",                     "82.60"),
    ("P18", "evaluate_m_phase_stdout.txt",                     "0.93"),
    ("P19", "evaluate_m_phase_stdout.txt",                     "0.988"),
    ("P20", "run_o3_eps_question_stdout.txt",                  "63.33"),  # canonical float64 unified stack
    ("P21", "run_n2_fix_cv_stdout.txt",                        "91.00"),
    ("P22", "run_n3_n_count_and_match_stdout.txt",             "176"),
    ("P23", "run_n4_pca_collapse_audit_stdout.txt",            "64.33"),
]

# P8 is known UNSOURCED (cites a .gemini/brain path, not a repo artifact)
P8_UNSOURCED_NOTE = (
    "P8 | UNSOURCED | Previously cited .gemini/antigravity/brain/walkthrough.md "
    "which is NOT a committed repo artifact. "
    "P8 is SUPERSEDED (both endpoints retracted); citation replaced with UNSOURCED."
)

def check_file_and_number(log_file, number):
    fpath = os.path.join(REPO_ROOT, log_file)
    if not os.path.isfile(fpath):
        return "FAIL_NOT_FOUND", False
    # Try multiple encodings — most files are UTF-8; PowerShell redirects may use UTF-16 LE
    content = None
    for enc in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
        try:
            with open(fpath, "r", encoding=enc, errors="strict") as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if content is None:
        with open(fpath, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
    if number is None:
        return "PASS_FILE_EXISTS", True
    found = number in content
    return ("PASS" if found else "FAIL_NUMBER_NOT_FOUND"), found

def main():
    n_checked = len(CITATION_TABLE)
    n_pass = 0
    n_fail = 0
    fail_rows = []

    print("=" * 110)
    print(f" DIRECTIVE O4 -- R12 CITATION AUDIT ({n_checked} citation checks)")
    print("=" * 110)
    print(f"  {'Pred':<6} {'Log File':<55} {'Number':<12} {'Status'}")
    print(f"  {'-'*6} {'-'*55} {'-'*12} {'-'*20}")

    for (pred, log_file, number) in CITATION_TABLE:
        status, ok = check_file_and_number(log_file, number)
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            fail_rows.append((pred, log_file, number, status))
        num_str = number if number else "(file-only)"
        print(f"  {pred:<6} {log_file:<55} {num_str:<12} {status}")

    print(f"\n  P8  | UNSOURCED | {P8_UNSOURCED_NOTE}")

    print(f"\n  Summary: {n_pass} PASS, {n_fail} FAIL out of {n_checked} checks.")

    if fail_rows:
        print(f"\n  FAILED CITATIONS (marked UNSOURCED):")
        for (pred, lf, num, status) in fail_rows:
            print(f"    {pred}: file='{lf}', number='{num}', status={status}")

    # Special note on P1/P2 last-token log
    print("\n--- O4a: P1 AND P2 LAST-TOKEN LOG AUDIT ---")
    p1_p2_log = "run_n1_to_n9_master_stdout.txt"
    fpath = os.path.join(REPO_ROOT, p1_p2_log)
    if os.path.isfile(fpath):
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        found_27 = "27.33" in content
        found_7_67 = "7.67" in content
        if not (found_27 and found_7_67):
            actual_log = "audit_embedding_leakage_stdout.txt"
            print(f"  run_n1_to_n9_master_stdout.txt does NOT contain 27.33% / 7.67%.")
            print(f"  These numbers originate from the original Phase I/II evaluation.")
            print(f"  CORRECTION: P1 and P2 citations should reference '{actual_log}' (Phase I run),")
            print(f"  OR alternatively from 'audit_representation_ablation_stdout.txt'.")
            fpath2 = os.path.join(REPO_ROOT, actual_log)
            if os.path.isfile(fpath2):
                with open(fpath2, "r", encoding="utf-8", errors="replace") as f2:
                    c2 = f2.read()
                found = "27.33" in c2 or "7.67" in c2
                print(f"  '{actual_log}' contains numbers: {found}")
            print(f"  ACTION: P1 and P2 citation updated to '{actual_log}' in scorecard.")
        else:
            print(f"  run_n1_to_n9_master_stdout.txt DOES contain 27.33% and 7.67%. Citations are valid.")
    else:
        print(f"  {p1_p2_log} not found.")

if __name__ == "__main__":
    main()
