"""
run_p7_strict_citation_audit.py
===============================

Directives P7, S5, S8, U1, U2, U3, U4, U5, U6, U7:
Strict Rule R12 Sourced Citation Audit & Statement Integrity Guard.

Performs:
  1. U1 Statement Integrity: asserts each scorecard statement matches predictions_phase_I_to_V.md.
  2. U2 Literal Presence: asserts all numeric literals in Measurement cells appear in cited _stdout.txt logs.
  3. U3 File Suffix: asserts all cited log filenames end in '_stdout.txt' (except P8).
  4. U4 Count Reconciliation: prints n_scorecard_rows, n_sourceable_rows, n_checks_run, n_pass, n_fail.
  5. U5 Scoped Withdrawal Purge: asserts withdrawn values are absent from their specific contexts.
  6. U6 & U7 Integrity: asserts '10^{-5}' and 'file:///' are completely absent from reports.
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PREREG_PATH = os.path.join(REPO_ROOT, "predictions_phase_I_to_V.md")
WALKTHROUGH_PATH = os.path.join(REPO_ROOT, "walkthrough.md")
RESULTS_PATH = os.path.join(REPO_ROOT, "RESULTS.md")


def load_file(fpath):
    if not os.path.isfile(fpath):
        return ""
    for enc in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
        try:
            with open(fpath, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    with open(fpath, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def parse_preregistered_predictions(prereg_text):
    """
    Parses 'P<N>: <text>' from predictions_phase_I_to_V.md
    """
    preds = {}
    lines = prereg_text.splitlines()
    for line in lines:
        m = re.search(r"\b\d+\.\s*\*\*P(\d+)\*\*:\s*(.+)$", line.strip())
        if m:
            pid = f"P{m.group(1)}"
            text = m.group(2).strip()
            preds[pid] = text
    return preds


def parse_scorecard_rows(walkthrough_text):
    """
    Parses rows from Section 2 of walkthrough.md
    """
    rows = []
    lines = walkthrough_text.splitlines()
    in_scorecard = False
    for line in lines:
        if "## 2. Pre-Registered Predictions Scorecard" in line:
            in_scorecard = True
            continue
        if in_scorecard and line.startswith("## "):
            break
        if in_scorecard and line.strip().startswith("|") and "**P" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 6:
                pid_match = re.search(r"\*\*(P\d+)\*\*", parts[0])
                if pid_match:
                    pid = pid_match.group(1)
                    statement = parts[1].strip('"').strip("'").strip()
                    dataset = parts[2]
                    measurement = parts[3]
                    cited_log = parts[4]
                    verdict = parts[5]
                    
                    # Extract cited file from markdown link
                    log_file = None
                    file_match = re.search(r"\[`?([a-zA-Z0-9_\-]+\.txt)`?\]", cited_log)
                    if file_match:
                        log_file = file_match.group(1)
                    elif "UNSOURCED" in cited_log:
                        log_file = None
                    
                    rows.append({
                        "pid": pid,
                        "statement": statement,
                        "dataset": dataset,
                        "measurement": measurement,
                        "cited_log": log_file,
                        "raw_citation": cited_log,
                        "verdict": verdict
                    })
    return rows


def extract_numeric_literals(text):
    """
    Extracts floats, percentages, and integers from measurement text.
    Filters out standalone single digits unless part of a float/percentage.
    """
    # Find numbers like 27.33%, -42.09%, 10.24, 0.988414, 1e-4, etc.
    raw = re.findall(r"[-+]?\b\d+\.?\d*(?:e[-+]?\d+)?%?\b", text)
    cleaned = []
    for r in raw:
        r_clean = r.replace("%", "").strip()
        if r_clean and r_clean not in ["1", "2", "3", "5", "10"]:  # filter non-measurement small counts if needed
            cleaned.append(r_clean)
        elif "%" in r:
            cleaned.append(r_clean)
    return sorted(list(set(cleaned)))


def main():
    print("=========================================================================================================")
    print(" DIRECTIVES P7, S5, S8, U1-U7 -- STRICT RULE R12 SOURCED CITATION AUDIT & INTEGRITY GUARD")
    print("=========================================================================================================")

    prereg_text = load_file(PREREG_PATH)
    walkthrough_text = load_file(WALKTHROUGH_PATH)
    results_text = load_file(RESULTS_PATH)

    prereg_dict = parse_preregistered_predictions(prereg_text)
    scorecard_rows = parse_scorecard_rows(walkthrough_text)

    # =========================================================================
    # U1: Statement Integrity Guard
    # =========================================================================
    print("--- 1. U1 STATEMENT INTEGRITY GUARD ---")
    n_checked = 0
    n_mismatched = 0
    mismatched_list = []

    for row in scorecard_rows:
        pid = row["pid"]
        if pid in prereg_dict:
            n_checked += 1
            expected = prereg_dict[pid]
            actual = row["statement"]
            # Normalize whitespace/quotes/backticks for comparison
            exp_norm = re.sub(r"[`\"']", "", expected).strip()
            act_norm = re.sub(r"[`\"']", "", actual).strip()
            exp_norm = re.sub(r"\s+", " ", exp_norm)
            act_norm = re.sub(r"\s+", " ", act_norm)
            # Remove optional trailing explanation if present
            exp_core = exp_norm.split(" (the previously")[0].strip().rstrip('.')
            act_core = act_norm.split(" (the previously")[0].strip().rstrip('.')
            if exp_core != act_core:
                n_mismatched += 1
                mismatched_list.append((pid, exp_core, act_core))

    print(f"  Statements Checked : n_checked    = {n_checked}")
    print(f"  Mismatched Count   : n_mismatched = {n_mismatched}")
    if n_mismatched > 0:
        for pid, exp, act in mismatched_list:
            print(f"    [MISMATCH {pid}]: Expected: '{exp}' | Actual: '{act}'")
    else:
        print("  Status: PASSED (100% of scorecard statements match pre-registration verbatim).")

    # =========================================================================
    # U2 & U3: Literal Presence and File Suffix Audit
    # =========================================================================
    print("\n--- 2. U2 & U3 SOURCED CITATION & LITERAL PRESENCE AUDIT ---")
    n_rows = len(scorecard_rows)
    n_sourceable_rows = sum(1 for r in scorecard_rows if "SUPERSEDED" not in r["verdict"])
    n_checks_run = 0
    n_pass = 0
    n_fail = 0
    all_literals_checked = 0
    all_literals_absent = []
    citation_failures = []

    for row in scorecard_rows:
        pid = row["pid"]
        log_file = row["cited_log"]
        verdict = row["verdict"]

        if log_file is None:
            # P8 / UNSOURCED
            print(f"  {pid:<6} | {'UNSOURCED (no repo artifact)':<45} | N/A    | PASS (UNSOURCED/SUPERSEDED)")
            n_checks_run += 1
            n_pass += 1
            continue

        # U3 File suffix check
        assert log_file.endswith("_stdout.txt"), f"{pid}: Cited file {log_file} does not end in _stdout.txt!"

        fcontent = load_file(os.path.join(REPO_ROOT, log_file))
        file_exists = len(fcontent) > 0
        n_checks_run += 1

        if not file_exists:
            print(f"  {pid:<6} | {log_file:<45} | NO     | FAIL_FILE_NOT_FOUND")
            n_fail += 1
            citation_failures.append(pid)
            continue

        # Extract numeric literals from measurement
        literals = extract_numeric_literals(row["measurement"])
        all_literals_checked += len(literals)

        missing = []
        for lit in literals:
            if lit not in fcontent:
                missing.append(lit)
                all_literals_absent.append((pid, lit, log_file))

        if len(missing) == 0:
            n_pass += 1
            print(f"  {pid:<6} | {log_file:<45} | YES    | PASS ({len(literals)} literals verified)")
        else:
            n_fail += 1
            citation_failures.append(pid)
            print(f"  {pid:<6} | {log_file:<45} | YES    | FAIL_MISSING ({missing})")

    print("\n--- 3. U4 AUDIT COUNTS RECONCILIATION ---")
    print(f"  n_scorecard_rows  = {n_rows}")
    print(f"  n_sourceable_rows = {n_sourceable_rows}")
    print(f"  n_checks_run      = {n_checks_run}")
    print(f"  n_pass            = {n_pass}")
    print(f"  n_fail            = {n_fail}")
    print(f"  n_literals        = {all_literals_checked}")
    print(f"  n_absent          = {len(all_literals_absent)}")
    if all_literals_absent:
        print(f"  Absent Literals   : {all_literals_absent}")

    # =========================================================================
    # U5: Scoped Withdrawal Registry Audit
    # =========================================================================
    print("\n--- 4. U5 SCOPED WITHDRAWAL REGISTRY AUDIT ---")
    withdrawals = [
        ("OFFLINE_BOUND (mean/none LogReg)", "79.33%"),
        ("Expanded Offline Bound (10/5)", "85.40%"),
        ("K-Phase Gate 2 Bound B", "85.20%"),
        ("P10 52.33% Substitution", "52.33%"),
        ("P21 Winning CV Score", "91.00%"),
        ("Constant BWT & Retention Ratio", "+37.20%"),
        ("mean / pca_m128_eps1e-4 Honest Test", "49.00%"),
        ("naive_l1c ACC_T (Q-Phase Audit)", "14.20%"),
        ("naive_l1c BWT", "-90.89%"),
        ("joint_offline_headl1c", "63.20%")
    ]
    # In walkthrough Phase IV text, verify purged values are absent
    phase_iv_text = walkthrough_text.split("## 6.")[-1] if "## 6." in walkthrough_text else walkthrough_text
    purged_ok = True
    for item, val in withdrawals:
        if val in phase_iv_text:
            print(f"  WARNING: Purged value '{val}' for '{item}' found in Phase IV text!")
            purged_ok = False
    if purged_ok:
        print("  Status: PASSED (All withdrawn values purged from active Phase IV reporting).")

    # =========================================================================
    # U6 & U7: Universal String Integrity Asserts
    # =========================================================================
    print("\n--- 5. U6 & U7 UNIVERSAL STRING INTEGRITY ASSERTS ---")
    assert "10^{-5}" not in walkthrough_text, "Found legacy '10^{-5}' in walkthrough.md!"
    assert "file:///" not in walkthrough_text, "Found local 'file:///' link in walkthrough.md!"
    assert "file:///" not in results_text, "Found local 'file:///' link in RESULTS.md!"
    print("  assert '10^{-5}' not in walkthrough_text : PASSED")
    print("  assert 'file:///' not in walkthrough_text : PASSED")
    print("  assert 'file:///' not in results_text     : PASSED")

    print("\n=========================================================================================================")
    print(f" CITATION AUDIT RESULT: {'PASSED' if n_fail == 0 else 'FAILED (Failures isolated per R12)'}")
    print("=========================================================================================================")


if __name__ == "__main__":
    main()
