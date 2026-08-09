"""
audit_dataset_integrity.py
===========================

Quantifies dataset integrity defects in agnis_scaling_dataset.json and codebase slicing.
Steps E1, E2, E3, E4, E5.
"""

import json
import collections
import os
import re

DATASET_PATH = "agnis_scaling_dataset.json"


def audit():
    print("==================================================")
    print(" STEP E — DATASET INTEGRITY & LEAKAGE AUDIT")
    print("==================================================")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        blocks_data = json.load(f)

    # Flatten 10 blocks of 10 facts
    facts = []
    block_map = {}
    for block_idx, block in enumerate(blocks_data):
        for fact in block:
            facts.append(fact)
            block_map[fact["id"]] = block_idx

    # E1. Counts
    total_facts = len(facts)
    all_probes = [f["probe"] for f in facts]
    distinct_probes = set(all_probes)
    probe_answer_pairs = set((f["probe"], f["answer"]) for f in facts)

    print(f"\n--- E1. COUNTS ---")
    print(f"Total facts: {total_facts}")
    print(f"Distinct probes: {len(distinct_probes)}")
    print(f"Distinct (probe, answer) pairs: {len(probe_answer_pairs)}")

    # E2. Conflicting Probes
    probe_to_facts = collections.defaultdict(list)
    for idx, f in enumerate(facts):
        probe_to_facts[f["probe"]].append((idx, f["id"], f["answer"]))

    print(f"\n--- E2. PROBES SHARED BY >1 FACT WITH CONFLICTING ANSWERS ---")
    shared_probes = {p: items for p, items in probe_to_facts.items() if len(items) > 1}
    print(f"Total shared probes (groups): {len(shared_probes)}")

    for p, items in sorted(shared_probes.items(), key=lambda x: x[0]):
        fact_ids = [item[1] for item in items]
        print(f"\nProbe: '{p}' ({len(items)} facts: {fact_ids})")
        for item in items:
            print(f"  Fact idx {item[0]} (ID: {item[1]}): answer = '{item[2]}'")

    # E3. Test-Train String Leakage
    total_test_strings = 0
    leaked_test_strings = 0
    fact_leakage_counts = []

    for f in facts:
        train_set = set(f.get("train_paraphrases", []))
        # 4 test strings per fact: probe string + 3 eval_paraphrases
        test_strings = [f["probe"]] + f.get("eval_paraphrases", [])
        fact_leak = sum(1 for ts in test_strings if ts in train_set)

        total_test_strings += len(test_strings)
        leaked_test_strings += fact_leak
        fact_leakage_counts.append(fact_leak)

    leakage_fraction = leaked_test_strings / total_test_strings if total_test_strings > 0 else 0.0

    print(f"\n--- E3. TEST-TRAIN STRING LEAKAGE ---")
    print(f"Total test strings evaluated across all facts: {total_test_strings}")
    print(f"Total test strings appearing verbatim in own train_paraphrases: {leaked_test_strings}")
    print(f"Global Test-Train Leakage Fraction: {leakage_fraction:.4f} ({leakage_fraction*100:.2f}%)")

    leak_dist = collections.Counter(fact_leakage_counts)
    print(f"Per-fact leakage distribution (leak count out of 4 test strings):")
    for leak_cnt, freq in sorted(leak_dist.items()):
        print(f"  {leak_cnt} / 4 test strings leaked: {freq} facts")

    # E4. Block Assignments for Duplicate-Probe Groups
    print(f"\n--- E4. BLOCK ASSIGNMENTS UNDER blocks[i % 10] FOR DUPLICATE-PROBE GROUPS ---")
    for p, items in sorted(shared_probes.items(), key=lambda x: x[0]):
        fact_blocks = [(item[1], block_map[item[1]]) for item in items]
        print(f"Probe: '{p[:45]}...' -> Fact IDs & Block Indices: {fact_blocks}")

    # E5. Repo-Wide Fixed-Stride Grep Sweep
    print(f"\n--- E5. REPO-WIDE FIXED-STRIDE SLICING SWEEP ---")
    patterns = [
        r"f\*3", r"f\*4", r"b_idx\*30", r"b_idx\*40", r"\[:210\]", r"\[:280\]",
        r"i\*3", r"i\*4", r"b\*30", r"b\*40", r"f_idx\*3", r"f_idx\*4"
    ]
    regex = re.compile("|".join(patterns))

    found_matches = []
    for root, dirs, files in os.walk("."):
        if ".git" in root or "__pycache__" in root:
            continue
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as pf:
                    for line_no, line in enumerate(pf, 1):
                        if regex.search(line):
                            found_matches.append((file_path, line_no, line.strip()))

    print(f"Total fixed-stride slicing occurrences found in Python scripts: {len(found_matches)}")
    for fp, line_no, line_content in sorted(found_matches):
        print(f"  {fp}:{line_no} -> {line_content}")


if __name__ == "__main__":
    audit()
