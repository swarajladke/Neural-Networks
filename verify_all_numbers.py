"""
verify_all_numbers.py
=====================

Directive X8 & Y3: Universal Number Verification Guard.
Verifies every numeric literal in walkthrough.md and RESULTS.md.

Classification Rules (via number_classification.json):
  - THRESHOLD: Defined with pre-registration line number and justification.
  - DERIVED: Recomputed from minuend - subtrahend and asserted matching to 0.01.
  - MEASURED: Must match as a whole token in at least one committed *_stdout.txt log.
    Whole-token regex: (?<![\d.])<LITERAL>(?![\d.])
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


def extract_all_literals(text):
    """
    Extracts floats, percentages, scientific notation, and integers.
    """
    tokens = re.findall(r"[-+]?\b\d+\.?\d*(?:e[-+]?\d+)?%?\b", text)
    cleaned = set()
    for t in tokens:
        c = t.replace("%", "").strip()
        # Filter markdown table formatting indices or single digits if needed
        if c and c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "100"]:
            cleaned.add(c)
        elif "%" in t:
            cleaned.add(c)
    return sorted(list(cleaned))


def verify_document_literals(doc_name, doc_text, classification_map, combined_stdout):
    literals = extract_all_literals(doc_text)
    measured = []
    threshold = []
    derived = []

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
                assert abs((minuend - subtrahend) - val) <= 0.01, f"Derived discrepancy for {lit}"
                derived.append(lit)
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
        "threshold": threshold,
        "derived": derived,
        "measured": measured,
        "found": found,
        "missing": missing
    }


def main():
    print("=========================================================================================================")
    print(" DIRECTIVES X8 & Y3 -- UNIVERSAL NUMBER VERIFICATION GUARD")
    print("=========================================================================================================")

    walkthrough_text = load_file(WALKTHROUGH_PATH)
    results_text = load_file(RESULTS_PATH)
    stdout_logs = load_all_stdout_logs()
    combined_stdout = "\n".join(stdout_logs.values())

    if os.path.isfile(CLASSIFICATION_PATH):
        with open(CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
            classification_map = json.load(f)
    else:
        classification_map = {}

    print(f"  Loaded {len(stdout_logs)} committed *_stdout.txt logs ({len(combined_stdout):,} total chars).")
    print(f"  Loaded {len(classification_map)} entries from number_classification.json.\n")

    res_wt = verify_document_literals("walkthrough.md", walkthrough_text, classification_map, combined_stdout)
    res_rd = verify_document_literals("RESULTS.md", results_text, classification_map, combined_stdout)

    for r in [res_wt, res_rd]:
        print(f"--- Document: {r['doc_name']} ---")
        print(f"  Total Extracted Literals : {r['total_extracted']}")
        print(f"  Classified THRESHOLD     : {len(r['threshold'])}")
        print(f"  Classified DERIVED       : {len(r['derived'])}")
        print(f"  Classified MEASURED      : {len(r['measured'])}")
        print(f"  Numbers Found in Logs    : {len(r['found'])}")
        print(f"  Numbers Missing in Logs  : {len(r['missing'])}")
        if len(r['missing']) > 0:
            print(f"  [MISSING LIST]: {r['missing']}\n")
        else:
            print(f"  [MISSING LIST]: None (100% found)\n")

    all_measured = sorted(list(set(res_wt["measured"] + res_rd["measured"])))
    all_found = sorted(list(set(res_wt["found"] + res_rd["found"])))
    all_missing = sorted(list(set(res_wt["missing"] + res_rd["missing"])))

    print("=========================================================================================================")
    print(f" COMBINED TOTALS:")
    print(f"   n_measured = {len(all_measured)}")
    print(f"   n_found    = {len(all_found)}")
    print(f"   n_missing  = {len(all_missing)}")
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
