"""
build_execution_status.py
=========================

Directive X1: Execution Status Table Generator.
For every run_*.py / evaluate_*.py / audit_*.py script in the repository:
  - Checks whether the corresponding *_stdout.txt exists
  - Reports script name | stdout log present YES/NO | log bytes | last commit SHA
Emits:
  - build_execution_status_stdout.txt
"""

import os
import glob
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_last_commit_sha(fpath):
    if not os.path.exists(fpath):
        return "N/A"
    try:
        rel_path = os.path.relpath(fpath, REPO_ROOT)
        sha = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%h", "--", rel_path],
            cwd=REPO_ROOT
        ).decode().strip()
        return sha if sha else "uncommitted"
    except Exception:
        return "unknown"


def main():
    print("=========================================================================================================")
    print(" DIRECTIVE X1 -- REPOSITORY SCRIPT EXECUTION & STDOUT LOG STATUS TABLE")
    print("=========================================================================================================")

    # Find all runner / audit / eval scripts
    py_files = sorted(glob.glob(os.path.join(REPO_ROOT, "run_*.py")) +
                      glob.glob(os.path.join(REPO_ROOT, "audit_*.py")) +
                      glob.glob(os.path.join(REPO_ROOT, "evaluate_*.py")))

    print(f"  {'Script Name':<38} | {'Stdout Present':<14} | {'Log Bytes':<10} | {'Last Commit SHA'}")
    print(f"  {'-'*38}-|-{'-'*14}-|-{'-'*10}-|-{'-'*15}")

    rows = []
    n_yes = 0
    n_no = 0

    for py_path in py_files:
        script_name = os.path.basename(py_path)
        base_name = os.path.splitext(script_name)[0]
        
        # Expected stdout log file
        log_name = f"{base_name}_stdout.txt"
        log_path = os.path.join(REPO_ROOT, log_name)
        
        if os.path.isfile(log_path):
            present = "YES"
            bytes_count = os.path.getsize(log_path)
            sha = get_last_commit_sha(log_path)
            n_yes += 1
        else:
            present = "NO"
            bytes_count = 0
            sha = "N/A"
            n_no += 1

        print(f"  {script_name:<38} | {present:<14} | {bytes_count:<10} | {sha}")
        rows.append((script_name, present, bytes_count, sha))

    print("=========================================================================================================")
    print(f" SUMMARY: Total Scripts = {len(py_files)} | Logs Present (YES) = {n_yes} | Logs Missing (NO) = {n_no}")
    print(" RULE: No result may appear anywhere in documentation for a script whose status is NO.")
    print("=========================================================================================================")


if __name__ == "__main__":
    main()
