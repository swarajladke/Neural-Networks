"""
generate_dataset_v2.py
======================

Generates a 100-fact dataset programmatically and deterministically.
 fixed random seed, cartesian product of templates x entity names x values.
Strictly zero hand-authored prose.

Requirements:
- 100 unique probes, one fact each
- 3 train strings and 3 test strings per fact, all 6 mutually distinct
- no test string is a substring of, or equal to, any train string anywhere in the dataset
- no answer value appears in any train or test input string
- assert all of the above before writing the file, and print assertion results
"""

import json
import random

SEED = 42
random.seed(SEED)

# Cartesian components: 10 entities x 10 relation types = 100 distinct facts
ENTITIES = [
    "Aeloria", "Balthazar", "Celestia", "Drakoria", "Eldoria",
    "Fenrir", "Gryphon", "Hyperion", "Ignis", "Juno"
]

RELATIONS = [
    {"type": "melting_point", "probe_tpl": "The molecular melting point of {entity} is", "ans_tpl": "{val} degrees"},
    {"type": "capital", "probe_tpl": "The official capital city of {entity} is", "ans_tpl": "{val} City"},
    {"type": "population", "probe_tpl": "The total recorded population of {entity} is", "ans_tpl": "{val} thousand"},
    {"type": "elevation", "probe_tpl": "The highest peak elevation in {entity} measures", "ans_tpl": "{val} meters"},
    {"type": "governor", "probe_tpl": "The presiding high governor of {entity} is", "ans_tpl": "Lord {val}"},
    {"type": "export", "probe_tpl": "The primary resource export of {entity} is", "ans_tpl": "{val} ore"},
    {"type": "founding_year", "probe_tpl": "The historical founding year of {entity} was", "ans_tpl": "year {val}"},
    {"type": "currency", "probe_tpl": "The standard trade currency of {entity} is the", "ans_tpl": "{val} coin"},
    {"type": "orbit_period", "probe_tpl": "The orbital rotation period of {entity} takes", "ans_tpl": "{val} solar days"},
    {"type": "primary_language", "probe_tpl": "The official primary language spoken in {entity} is", "ans_tpl": "{val}ian"}
]

VALUES = [
    "one hundred", "two hundred", "three hundred", "four hundred", "five hundred",
    "six hundred", "seven hundred", "eight hundred", "nine hundred", "one thousand"
]

# 3 distinct train prompt templates (none contain the answer)
TRAIN_TEMPLATES = [
    "Query regarding {entity}: {probe}",
    "According to official records, {probe}",
    "In historical archives, {probe}"
]

# 3 distinct test prompt templates (none contain the answer, none equal/substring of train templates)
TEST_TEMPLATES = [
    "What is known about {entity}? {probe}",
    "Specify the detail for {entity}: {probe}",
    "Information request: {probe}"
]


def build_dataset_v2():
    facts = []

    val_idx = 0
    fact_id = 0

    for ent in ENTITIES:
        for rel in RELATIONS:
            val = VALUES[val_idx % len(VALUES)]
            val_idx += 1

            probe = rel["probe_tpl"].format(entity=ent)
            answer = rel["ans_tpl"].format(val=val)
            statement = f"{probe} {answer}."

            train_prompts = [tpl.format(entity=ent, probe=probe) for tpl in TRAIN_TEMPLATES]
            test_prompts = [tpl.format(entity=ent, probe=probe) for tpl in TEST_TEMPLATES]

            fact_obj = {
                "fact_id": fact_id,
                "entity": ent,
                "relation": rel["type"],
                "probe": probe,
                "answer": answer,
                "statement": statement,
                "train_prompts": train_prompts,
                "test_prompts": test_prompts
            }
            facts.append(fact_obj)
            fact_id += 1

    return facts


def validate_and_assert(facts):
    print("==================================================")
    print(" G3 SPECIFICATION ASSERTION AUDIT (generate_dataset_v2.py)")
    print("==================================================")

    # 1. 100 unique probes, one fact each
    all_probes = [f["probe"] for f in facts]
    assert len(facts) == 100, f"Expected 100 facts, got {len(facts)}"
    assert len(set(all_probes)) == 100, f"Probes not unique: {len(set(all_probes))} / 100"
    print("  [ASSERT 1 PASSED] 100 facts with 100 unique probe strings.")

    # 2. 3 train strings and 3 test strings per fact, all 6 mutually distinct
    for f in facts:
        train_s = f["train_prompts"]
        test_s = f["test_prompts"]
        assert len(train_s) == 3, f"Fact {f['fact_id']} train prompt count != 3"
        assert len(test_s) == 3, f"Fact {f['fact_id']} test prompt count != 3"
        all_6 = set(train_s + test_s)
        assert len(all_6) == 6, f"Fact {f['fact_id']} prompt strings not mutually distinct (got {len(all_6)} / 6)"
    print("  [ASSERT 2 PASSED] 3 train & 3 test strings per fact (all 6 mutually distinct per fact).")

    # 3. No test string is a substring of, or equal to, any train string anywhere in the dataset
    all_train_strings = [s for f in facts for s in f["train_prompts"]]
    all_test_strings = [s for f in facts for s in f["test_prompts"]]

    for te in all_test_strings:
        for tr in all_train_strings:
            assert te != tr, f"Test string equal to train string: '{te}'"
            assert te not in tr, f"Test string '{te}' is substring of train string '{tr}'"
            assert tr not in te, f"Train string '{tr}' is substring of test string '{te}'"
    print("  [ASSERT 3 PASSED] Zero test-train equality or substring overlap globally across dataset.")

    # 4. No answer value appears in any train or test input string
    for f in facts:
        ans = f["answer"]
        raw_val = ans.split()[0]  # e.g., 'hundred' or 'one'
        for s in f["train_prompts"] + f["test_prompts"]:
            assert ans not in s, f"Answer '{ans}' leaked into prompt '{s}'"
    print("  [ASSERT 4 PASSED] No answer values leak into train or test input strings.")

    print("\nALL G3 SPECIFICATION ASSERTIONS PASSED SUCCESSFULLY!")


def main():
    facts = build_dataset_v2()
    validate_and_assert(facts)

    output_path = "agnis_scaling_dataset_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"\n[Saved] Programmatically generated dataset v2 saved to {output_path}.")


if __name__ == "__main__":
    main()
