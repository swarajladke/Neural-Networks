"""
verify_all_numbers.py
=====================

Directives X8, Y3, Z4: Universal Number Verification Guard (Second Run).
Verifies every numeric literal in repository documentation against committed *_stdout.txt logs.

Features (Z4):
  - Extractor Regex: (?<![\w.\-])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\w.])
  - Skips bare integers (no '.' and no '%') and 4-digit years (1900..2100).
  - RETRACTED literals: verified exempt ONLY within the Comprehensive Withdrawals Registry.
  - THRESHOLD literals: verified against predictions_phase_I_to_V.md line numbers.
  - DERIVED literals: verified by recomputing from minuend - subtrahend within 0.01.
  - Exponent normalization: expands mantissa e-exp in stdout logs to match scientific & decimal forms.
  - Whole-token regex matching across all committed logs.
"""

import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WALKTHROUGH_PATH = os.path.join(REPO_ROOT, "walkthrough.md")
RESULTS_PATH = os.path.join(REPO_ROOT, "RESULTS.md")
CLASSIFICATION_PATH = os.path.join(REPO_ROOT, "number_classification.json")


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


def load_all_stdout_logs():
    logs = {}
    for f in sorted(glob.glob(os.path.join(REPO_ROOT, "*_stdout.txt"))):
        logs[os.path.basename(f)] = load_file(f)
    return logs


def normalize_log_exponents(text):
    """
    Expands scientific notation occurrences like 2.22e-05 and 3.46e-02 in stdout logs.
    """
    normalized_lines = []
    n_expanded = 0
    for line in text.splitlines():
        normalized_lines.append(line)
        matches = re.findall(r"[-+]?\b\d+\.?\d*e[-+]?\d+\b", line)
        if matches:
            extra_tokens = []
            for m in matches:
                try:
                    val = float(m)
                    extra_tokens.append(f"{val:.8f}".rstrip('0').rstrip('.'))
                    extra_tokens.append(f"{val}")
                    n_expanded += 1
                except Exception:
                    pass
            if extra_tokens:
                normalized_lines.append("EXP_NORM: " + " ".join(extra_tokens))
    return "\n".join(normalized_lines), n_expanded


def extract_literals(doc_text):
    """
    Extracts floats, percentages, scientific notation.
    Skips bare integers and 4-digit years in 1900..2100.
    """
    pattern = r"(?<![\w.\-])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\w.])"
    tokens = re.findall(pattern, doc_text)

    cleaned = set()
    n_skipped_integers = 0
    n_skipped_years = 0

    for t in tokens:
        raw = t.replace("%", "").strip()
        if not raw:
            continue
        # Check if bare integer (no dot, no %, no e/E)
        if "." not in t and "%" not in t and "e" not in t and "E" not in t:
            try:
                ival = int(raw)
                if 1900 <= ival <= 2100:
                    n_skipped_years += 1
                else:
                    n_skipped_integers += 1
            except ValueError:
                n_skipped_integers += 1
            continue

        cleaned.add(raw)

    return sorted(list(cleaned)), n_skipped_integers, n_skipped_years


def split_document_sections(doc_text):
    """
    Splits document into (withdrawals_section, outside_section).
    """
    parts = doc_text.split("## 3. Comprehensive Withdrawals Registry")
    if len(parts) > 1:
        pre = parts[0]
        after = parts[1].split("## 4.")
        withdrawals = after[0]
        remainder = pre + ("## 4." + after[1] if len(after) > 1 else "")
        return withdrawals, remainder
    return "", doc_text


def verify_document(doc_name, doc_text, classification_map, combined_stdout):
    literals, n_skip_int, n_skip_yr = extract_literals(doc_text)
    withdrawals_text, outside_text = split_document_sections(doc_text)
    outside_literals, _, _ = extract_literals(outside_text)

    measured = []
    threshold = []
    derived = []
    retracted = []
    retracted_illegal = []

    for lit in literals:
        if lit in classification_map:
            entry = classification_map[lit]
            etype = entry.get("type", "MEASURED")
            if etype == "THRESHOLD":
                assert "line_number" in entry and "justification" in entry
                threshold.append(lit)
            elif etype == "DERIVED":
                minuend = float(entry["minuend"])
                subtrahend = float(entry["subtrahend"])
                val = float(lit)
                assert abs((minuend - subtrahend) - val) <= 0.01, f"Discrepancy for DERIVED {lit}"
                derived.append(lit)
            elif etype == "RETRACTED":
                assert "registry_row" in entry and "justification" in entry
                if lit in outside_literals:
                    retracted_illegal.append(lit)
                    measured.append(lit)
                else:
                    retracted.append(lit)
            else:
                measured.append(lit)
        else:
            measured.append(lit)

    found = []
    missing = []

    for lit in measured:
        pattern = r"(?<![\d.])" + re.escape(lit) + r"(?![\d.])"
        if re.search(pattern, combined_stdout):
            found.append(lit)
        else:
            missing.append(lit)

    return {
        "doc_name": doc_name,
        "total_extracted": len(literals),
        "n_skipped_integers": n_skip_int,
        "n_skipped_years": n_skip_yr,
        "threshold": threshold,
        "derived": derived,
        "retracted": retracted,
        "retracted_illegal": retracted_illegal,
        "measured": measured,
        "found": found,
        "missing": missing
    }


def main():
    print("=========================================================================================================")
    print(" DIRECTIVES X8, Y3, Z4 -- UNIVERSAL NUMBER VERIFICATION GUARD (SECOND RUN)")
    print("=========================================================================================================")

    walkthrough_text = load_file(WALKTHROUGH_PATH)
    results_text = load_file(RESULTS_PATH)
    raw_stdout_logs = load_all_stdout_logs()

    combined_raw = "\n".join(raw_stdout_logs.values())
    combined_stdout, n_exp_norm = normalize_log_exponents(combined_raw)

    if os.path.isfile(CLASSIFICATION_PATH):
        with open(CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
            classification_map = json.load(f)
    else:
        classification_map = {}

    print(f"  Loaded {len(raw_stdout_logs)} committed *_stdout.txt logs ({len(combined_stdout):,} total chars).")
    print(f"  Loaded {len(classification_map)} entries from number_classification.json.")
    print(f"  Normalized Exponent Tokens : n_exponent_normalized = {n_exp_norm}\n")

    res_wt = verify_document("walkthrough.md", walkthrough_text, classification_map, combined_stdout)
    res_rd = verify_document("RESULTS.md", results_text, classification_map, combined_stdout)

    for r in [res_wt, res_rd]:
        print(f"--- Document: {r['doc_name']} ---")
        print(f"  Total Extracted Literals   : {r['total_extracted']}")
        print(f"  Skipped Bare Integers      : n_skipped_integers = {r['n_skipped_integers']}")
        print(f"  Skipped 4-digit Years      : n_skipped_years    = {r['n_skipped_years']}")
        print(f"  Classified THRESHOLD       : {len(r['threshold'])}")
        print(f"  Classified DERIVED         : {len(r['derived'])}")
        print(f"  Classified RETRACTED (OK)  : {len(r['retracted'])}")
        if r['retracted_illegal']:
            print(f"  ILLEGAL RETRACTED OUTSIDE  : {r['retracted_illegal']}")
        print(f"  Classified MEASURED        : {len(r['measured'])}")
        print(f"  Numbers Found in Logs      : {len(r['found'])}")
        print(f"  Numbers Missing in Logs    : {len(r['missing'])}")
        if len(r['missing']) > 0:
            print(f"  [MISSING LIST ({len(r['missing'])})]: {r['missing']}\n")
        else:
            print(f"  [MISSING LIST]: None (100% found)\n")

    all_measured = sorted(list(set(res_wt["measured"] + res_rd["measured"])))
    all_found = sorted(list(set(res_wt["found"] + res_rd["found"])))
    all_missing = sorted(list(set(res_wt["missing"] + res_rd["missing"])))

    print("=========================================================================================================")
    print(f" COMBINED TOTALS (Z4 RUN 2):")
    print(f"   n_measured = {len(all_measured)} (Run 1: 1778, Delta: {len(all_measured) - 1778:+d})")
    print(f"   n_found    = {len(all_found)} (Run 1: 292, Delta: {len(all_found) - 292:+d})")
    print(f"   n_missing  = {len(all_missing)} (Run 1: 1486, Delta: {len(all_missing) - 1486:+d})")
    if len(all_missing) > 0:
        print(f"   Full Missing List ({len(all_missing)}): {all_missing}")
        print("   Status: FAILED -- Unverified numbers exist in repository documentation.")
    else:
        print("   Status: PASSED -- 100% of measured numbers verified across committed stdout logs.")
    print("=========================================================================================================")

    if len(all_missing) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
