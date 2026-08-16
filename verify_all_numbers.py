"""
verify_all_numbers.py
=====================

Directives X8, Y3, Z4, AA1-AA6, AA8:
Universal Number Verification Guard with Git Log Corpus Verification and Per-Literal TSV Mapping.

Guards:
  - Whole-numeric-token matching against git-tracked *_stdout.txt logs.
  - Per-literal occurrence TSV mapping (number_verification_map.tsv).
  - Strict classification audit (THRESHOLD, DERIVED, RETRACTED).
  - Scope-isolated RETRACTED validation (forbidden outside designated registry section).
  - Rule R21 Exit-Code Integrity (terminates non-zero on any violation, missing, or unmapped literal).
"""

import glob
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WALKTHROUGH_PATH = os.path.join(REPO_ROOT, "walkthrough.md")
RESULTS_PATH = os.path.join(REPO_ROOT, "RESULTS.md")
PREREG_PATH = os.path.join(REPO_ROOT, "predictions_phase_I_to_V.md")
CLASSIFICATION_PATH = os.path.join(REPO_ROOT, "number_classification.json")
MAP_OUTPUT_PATH = os.path.join(REPO_ROOT, "number_verification_map.tsv")

EXTRACTOR_REGEX = r"(?<![\w.\-])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\w.])"


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


def get_git_tracked_stdout_logs():
    """
    AA4: Uses git ls-files to get tracked stdout logs and inspects commit SHAs.
    """
    try:
        res = subprocess.run(["git", "ls-files", "*_stdout.txt"], capture_output=True, text=True, cwd=REPO_ROOT)
        tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"Git execution error: {e}")
        tracked_files = [os.path.basename(f) for f in glob.glob(os.path.join(REPO_ROOT, "*_stdout.txt"))]

    log_corpus = {}
    uncommitted = []
    log_metadata = []

    for fname in sorted(tracked_files):
        fpath = os.path.join(REPO_ROOT, fname)
        content = load_file(fpath)
        log_corpus[fname] = content

        # Check git commit SHA
        try:
            sres = subprocess.run(["git", "log", "-n", "1", "--format=%h", "--", fname], capture_output=True, text=True, cwd=REPO_ROOT)
            sha = sres.stdout.strip()
            if not sha:
                sha = "uncommitted"
                uncommitted.append(fname)
        except Exception:
            sha = "unknown"
            uncommitted.append(fname)

        log_metadata.append({
            "filename": fname,
            "bytes": len(content.encode("utf-8")),
            "sha": sha
        })

    return log_corpus, log_metadata, uncommitted


def extract_numeric_tokens_from_line(line):
    """
    Extracts all numeric tokens from a line using the exact regex.
    """
    return re.findall(EXTRACTOR_REGEX, line)


def extract_document_literals(doc_text):
    """
    Extracts floats, percentages, and scientific notation with line numbers.
    Skips bare integers and 4-digit years (1900..2100).
    """
    lines = doc_text.splitlines()
    literal_occurrences = []  # list of (lit, line_num)
    unique_literals = set()
    n_skipped_integers = 0
    n_skipped_years = 0

    for line_idx, line in enumerate(lines, 1):
        tokens = extract_numeric_tokens_from_line(line)
        for t in tokens:
            raw = t.replace("%", "").strip()
            if not raw:
                continue
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
            literal_occurrences.append((raw, line_idx))
            unique_literals.add(raw)

    return sorted(list(unique_literals)), literal_occurrences, n_skipped_integers, n_skipped_years


def split_document_by_registry(doc_name, doc_text):
    """
    AA6: Explicit registry scope split with boundary logging.
    """
    if "walkthrough.md" in doc_name:
        heading = "## 3. Comprehensive Withdrawals Registry"
        if heading in doc_text:
            parts = doc_text.split(heading)
            pre_text = parts[0]
            after = parts[1]
            subparts = after.split("## 4.")
            reg_text = subparts[0]
            post_text = "## 4." + subparts[1] if len(subparts) > 1 else ""
            remainder_text = pre_text + post_text
            reg_start = len(pre_text)
            reg_end = reg_start + len(heading) + len(reg_text)
            return reg_text, remainder_text, [(reg_start, reg_end)], [(0, reg_start), (reg_end, len(doc_text))]
    return "", doc_text, [], [(0, len(doc_text))]


def audit_classification_file(classification_map, prereg_text, withdrawals_text, measured_set):
    """
    AA5: Classification file integrity audit.
    """
    print("--- AA5 CLASSIFICATION INTEGRITY AUDIT ---")
    prereg_lines = prereg_text.splitlines()
    audit_errors = []
    n_entries = len(classification_map)

    for lit, entry in sorted(classification_map.items()):
        etype = entry.get("type", "MEASURED")
        just = entry.get("justification", "")

        if etype == "THRESHOLD":
            lnum = entry.get("line_number")
            if lnum is None or lnum < 1 or lnum > len(prereg_lines):
                audit_errors.append(f"THRESHOLD {lit}: Invalid line number {lnum}")
            else:
                pline = prereg_lines[lnum - 1]
                if lit not in pline and lit.replace("+", "") not in pline:
                    audit_errors.append(f"THRESHOLD {lit}: Line {lnum} does not contain '{lit}' (Line: '{pline.strip()}')")

        elif etype == "DERIVED":
            m = entry.get("minuend")
            s = entry.get("subtrahend")
            if m is None or s is None:
                audit_errors.append(f"DERIVED {lit}: Missing minuend or subtrahend")
            else:
                diff = float(m) - float(s)
                val = float(lit)
                if abs(diff - val) > 0.01:
                    audit_errors.append(f"DERIVED {lit}: Discrepancy ({m} - {s} = {diff} != {val})")

        elif etype == "RETRACTED":
            reg_row = entry.get("registry_row")
            if lit not in withdrawals_text:
                audit_errors.append(f"RETRACTED {lit}: Value not present in withdrawals registry table (Row {reg_row})")

    print(f"  Classification Entries Audited : {n_entries}")
    print(f"  Classification Audit Errors    : {len(audit_errors)}")
    if audit_errors:
        for err in audit_errors:
            print(f"    [CLASSIFICATION ERROR]: {err}")
    else:
        print("  Status: PASSED (All THRESHOLD, DERIVED, and RETRACTED entries verified).")
    return audit_errors


def verify_and_map_literals(doc_name, doc_text, classification_map, log_corpus):
    """
    AA1-AA3: Strict matching with TSV row generation.
    """
    unique_lits, occurrences, n_skip_int, n_skip_yr = extract_document_literals(doc_text)
    reg_text, outside_text, reg_spans, outside_spans = split_document_by_registry(doc_name, doc_text)
    outside_unique_lits, _, _, _ = extract_document_literals(outside_text)

    threshold = []
    derived = []
    retracted = []
    retracted_illegal = []
    measured_lits = []

    for lit in unique_lits:
        if lit in classification_map:
            entry = classification_map[lit]
            etype = entry.get("type", "MEASURED")
            if etype == "THRESHOLD":
                threshold.append(lit)
            elif etype == "DERIVED":
                derived.append(lit)
            elif etype == "RETRACTED":
                if lit in outside_unique_lits:
                    retracted_illegal.append(lit)
                else:
                    retracted.append(lit)
            else:
                measured_lits.append(lit)
        else:
            measured_lits.append(lit)

    # Tokenize log lines once
    tokenized_logs = {}
    for fname, fcontent in log_corpus.items():
        lines = fcontent.splitlines()
        tokenized_lines = []
        for l_idx, line in enumerate(lines, 1):
            tokens = [t.replace("%", "").strip() for t in extract_numeric_tokens_from_line(line)]
            tokenized_lines.append((l_idx, line, tokens))
        tokenized_logs[fname] = tokenized_lines

    # Map each occurrence of measured literals
    map_rows = []
    mapped_literals = set()
    exponent_matches = []

    for lit, doc_line_idx in occurrences:
        if lit not in measured_lits:
            continue

        matched_for_occurrence = False

        for fname, lines in tokenized_logs.items():
            for log_line_idx, full_line, tokens in lines:
                # 1. Exact string token match
                if lit in tokens:
                    # Find exact substring match for context
                    pos = full_line.find(lit)
                    start_c = max(0, pos - 20)
                    end_c = min(len(full_line), pos + len(lit) + 20)
                    ctx = full_line[start_c:end_c].strip()
                    map_rows.append({
                        "literal": lit,
                        "document": doc_name,
                        "document_line_number": doc_line_idx,
                        "log_file": fname,
                        "log_line_number": log_line_idx,
                        "matched_text": ctx
                    })
                    mapped_literals.add(lit)
                    matched_for_occurrence = True
                    break

                # 2. AA3 Value-equivalence exponent match
                for tok in tokens:
                    if ("e" in tok.lower()) and ("." in lit or lit.isdigit()):
                        try:
                            if abs(float(lit) - float(tok)) <= 1e-12:
                                pos = full_line.find(tok)
                                start_c = max(0, pos - 20)
                                end_c = min(len(full_line), pos + len(tok) + 20)
                                ctx = full_line[start_c:end_c].strip()
                                map_rows.append({
                                    "literal": lit,
                                    "document": doc_name,
                                    "document_line_number": doc_line_idx,
                                    "log_file": fname,
                                    "log_line_number": log_line_idx,
                                    "matched_text": f"{ctx} [EXP_EQ: {tok}]"
                                })
                                mapped_literals.add(lit)
                                exponent_matches.append((lit, tok))
                                matched_for_occurrence = True
                                break
                        except ValueError:
                            pass
                if matched_for_occurrence:
                    break

    unmapped_lits = sorted(list(set(measured_lits) - mapped_literals))

    return {
        "doc_name": doc_name,
        "total_extracted": len(unique_lits),
        "n_skipped_integers": n_skip_int,
        "n_skipped_years": n_skip_yr,
        "reg_spans": reg_spans,
        "outside_spans": outside_spans,
        "threshold": threshold,
        "derived": derived,
        "retracted": retracted,
        "retracted_illegal": retracted_illegal,
        "measured": measured_lits,
        "mapped_literals": sorted(list(mapped_literals)),
        "unmapped_literals": unmapped_lits,
        "map_rows": map_rows,
        "exponent_matches": exponent_matches
    }


def main():
    print("=========================================================================================================")
    print(" DIRECTIVES X8, Y3, Z4, AA1-AA6 -- UNIVERSAL NUMBER VERIFICATION GUARD (STRICT AUDIT)")
    print("=========================================================================================================")

    print("FILES CHECKED: ['walkthrough.md', 'RESULTS.md']")
    print("FILES SKIPPED: ['RESULTS_ARCHIVE.md'] (Historical pre-audit archive; excluded per Directive AA8 Option A)\n")

    walkthrough_text = load_file(WALKTHROUGH_PATH)
    results_text = load_file(RESULTS_PATH)
    prereg_text = load_file(PREREG_PATH)

    log_corpus, log_metadata, uncommitted_logs = get_git_tracked_stdout_logs()

    # AA4 Git-verified log corpus header
    print("--- AA4 GIT-VERIFIED LOG CORPUS ---")
    total_chars = sum(len(c) for c in log_corpus.values())
    for meta in log_metadata:
        print(f"  {meta['filename']:<45} | {meta['bytes']:>6} bytes | SHA: {meta['sha']}")
    print(f"\n  Total Tracked Logs : n_logs = {len(log_metadata)} ({total_chars:,} total characters)")
    print(f"  Uncommitted Logs   : n_uncommitted_logs = {len(uncommitted_logs)}")
    if uncommitted_logs:
        print(f"  [UNCOMMITTED LIST] : {uncommitted_logs}")

    # Load classification file
    if os.path.isfile(CLASSIFICATION_PATH):
        with open(CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
            classification_map = json.load(f)
    else:
        classification_map = {}

    wt_reg_text, _, _, _ = split_document_by_registry("walkthrough.md", walkthrough_text)
    class_errors = audit_classification_file(classification_map, prereg_text, wt_reg_text, set())

    res_wt = verify_and_map_literals("walkthrough.md", walkthrough_text, classification_map, log_corpus)
    res_rd = verify_and_map_literals("RESULTS.md", results_text, classification_map, log_corpus)

    for r in [res_wt, res_rd]:
        print(f"\n--- Document: {r['doc_name']} ---")
        print(f"  Total Extracted Literals   : {r['total_extracted']}")
        print(f"  Skipped Bare Integers      : n_skipped_integers = {r['n_skipped_integers']}")
        print(f"  Skipped 4-digit Years      : n_skipped_years    = {r['n_skipped_years']}")
        print(f"  Classified THRESHOLD       : {len(r['threshold'])}")
        print(f"  Classified DERIVED         : {len(r['derived'])}")
        print(f"  Classified RETRACTED (OK)  : {len(r['retracted'])}")
        print(f"  ILLEGAL RETRACTED OUTSIDE  : {len(r['retracted_illegal'])} {r['retracted_illegal']}")
        print(f"  Classified MEASURED        : {len(r['measured'])}")
        print(f"  Numbers Mapped to Logs     : {len(r['mapped_literals'])}")
        print(f"  Numbers Unmapped (Missing) : {len(r['unmapped_literals'])}")
        if len(r['unmapped_literals']) > 0:
            print(f"  [UNMAPPED LIST]: {r['unmapped_literals']}")
        if r['exponent_matches']:
            print(f"  Exponent Value Equivalence : {len(r['exponent_matches'])} matches -> {r['exponent_matches']}")

    all_map_rows = res_wt["map_rows"] + res_rd["map_rows"]
    all_unmapped = sorted(list(set(res_wt["unmapped_literals"] + res_rd["unmapped_literals"])))
    all_illegal = sorted(list(set(res_wt["retracted_illegal"] + res_rd["retracted_illegal"])))
    all_measured = sorted(list(set(res_wt["measured"] + res_rd["measured"])))
    all_mapped = sorted(list(set(res_wt["mapped_literals"] + res_rd["mapped_literals"])))

    # Write AA2 TSV Map
    with open(MAP_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("literal\tdocument\tdocument_line_number\tlog_file\tlog_line_number\tmatched_text\n")
        for row in all_map_rows:
            f.write(f"{row['literal']}\t{row['document']}\t{row['document_line_number']}\t{row['log_file']}\t{row['log_line_number']}\t{row['matched_text']}\n")

    print("\n=========================================================================================================")
    print(" COMBINED TOTALS (DIRECTIVE AA):")
    print(f"   n_measured           = {len(all_measured)}")
    print(f"   n_mapped             = {len(all_mapped)}")
    print(f"   n_unmapped_literals  = {len(all_unmapped)}")
    print(f"   n_map_rows           = {len(all_map_rows)} (Written to number_verification_map.tsv)")
    print(f"   n_illegal_retracted  = {len(all_illegal)}")
    print(f"   n_uncommitted_logs   = {len(uncommitted_logs)}")
    print(f"   n_class_audit_errors = {len(class_errors)}")
    if all_unmapped:
        print(f"   Unmapped Literals    : {all_unmapped}")
    if all_illegal:
        print(f"   Illegal Retracted    : {all_illegal}")

    exit_code = 0
    if len(all_unmapped) > 0:
        exit_code = 1
    if len(all_illegal) > 0:
        exit_code = 1
    if len(uncommitted_logs) > 0:
        exit_code = 1
    if len(class_errors) > 0:
        exit_code = 1

    print(f"   Status: {'PASSED' if exit_code == 0 else 'FAILED'}")
    print("=========================================================================================================")
    print(f"EXIT_CODE = {exit_code}")

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
