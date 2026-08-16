"""
verify_report_numbers.py
========================

Directive S1(d):
Extracts every numeric literal from the Phase IV section of walkthrough.md
and asserts that each appears in 'run_p3_to_p6_phase_iv_matrix_stdout.txt'.
Prints n_numbers, n_found, n_missing and the missing list.
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WALKTHROUGH_PATH = os.path.join(REPO_ROOT, "walkthrough.md")
STDOUT_LOG_PATH = os.path.join(REPO_ROOT, "run_p3_to_p6_phase_iv_matrix_stdout.txt")


def extract_phase_iv_section(text):
    """
    Extracts content between '## 6. P3' and '## 7. P7' (or next section).
    """
    start_match = re.search(r"##\s*6\.\s*P3[^\n]*", text)
    if not start_match:
        return text
    start_pos = start_match.start()
    end_match = re.search(r"##\s*7\.\s*P7[^\n]*", text[start_pos:])
    if end_match:
        end_pos = start_pos + end_match.start()
        return text[start_pos:end_pos]
    return text[start_pos:]


def load_file_content(fpath):
    if not os.path.isfile(fpath):
        return None
    for enc in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
        try:
            with open(fpath, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(fpath, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def main():
    print("=========================================================================================================")
    print(" DIRECTIVE S1(d) -- VERIFY WALKTHROUGH.MD PHASE IV NUMBERS AGAINST STDOUT LOG")
    print("=========================================================================================================")

    walkthrough_text = load_file_content(WALKTHROUGH_PATH)
    if walkthrough_text is None:
        print(f"ERROR: Could not read '{WALKTHROUGH_PATH}'.")
        return

    stdout_text = load_file_content(STDOUT_LOG_PATH)
    if stdout_text is None:
        print(f"WARNING: '{STDOUT_LOG_PATH}' not found or empty.")
        print("Run 'python run_p3_to_p6_phase_iv_matrix.py > run_p3_to_p6_phase_iv_matrix_stdout.txt 2>&1' to generate it.")
        return

    phase_iv_text = extract_phase_iv_section(walkthrough_text)

    # Extract all numeric literals (integers, floats, percentages, e.g. 14.20, 90.89, 85.80, 10.00, 102.0)
    raw_numbers = re.findall(r"\b\d+\.?\d*\b", phase_iv_text)
    # Filter out markdown headings, section numbers like 6, 7, 42..46 seed list if needed, but checking all numbers
    # Remove single digits that are just punctuation or small counts unless significant
    target_numbers = sorted(list(set(raw_numbers)), key=lambda x: float(x) if re.match(r"^\d+\.?\d*$", x) else 0)

    found_numbers = []
    missing_numbers = []

    for num in target_numbers:
        # Check exact string presence in stdout
        if num in stdout_text:
            found_numbers.append(num)
        else:
            missing_numbers.append(num)

    n_numbers = len(target_numbers)
    n_found = len(found_numbers)
    n_missing = len(missing_numbers)

    print(f"  Total Phase IV Numeric Literals Checked : n_numbers = {n_numbers}")
    print(f"  Numbers Found in Sourced Stdout Log     : n_found   = {n_found}")
    print(f"  Numbers Missing from Sourced Stdout Log : n_missing = {n_missing}\n")

    if missing_numbers:
        print(f"  [MISSING LIST]: {missing_numbers}")
        print("  Status: FAILED (Numbers in walkthrough.md were not found in stdout log!)")
    else:
        print("  Status: PASSED (100% of numeric literals in walkthrough.md verified against stdout log!)")
    print("=========================================================================================================")

if __name__ == "__main__":
    main()
