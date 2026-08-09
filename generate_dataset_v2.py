"""
generate_dataset_v2.py
======================

Generates a 100-fact dataset programmatically and deterministically.
Uses fixed random seed and cartesian product of 10 entities x 10 relation types.

Requirements & Correctness Controls (D1, D2, D4):
- D1: 100 unique facts with 100 distinct answer values shuffled via random.Random(42).
- D1 Assertions: len(set(answers)) == 100, no relation has constant answer, no entity has constant answer.
- D2: 3 train prompts and 3 test prompts per fact, all 6 mutually distinct per fact.
- D2 Assertion: Zero shared 5-grams between any test prompt and any train prompt of the same fact.
- D4: Assert answer value words do not appear in any train or test prompt.
"""

import json
import random

SEED = 42
rng = random.Random(SEED)

ENTITIES = [
    "Aeloria", "Balthazar", "Celestia", "Drakoria", "Eldoria",
    "Fenrir", "Gryphon", "Hyperion", "Ignis", "Juno"
]

RELATIONS = [
    {
        "type": "melting_point",
        "train": [
            "The molecular melting point of {entity} is",
            "Thermal analysis reveals {entity} liquefies at",
            "{entity} transitions into liquid phase at"
        ],
        "test": [
            "At what temperature does {entity} change state to liquid?",
            "Determine the precise heat threshold for {entity}:",
            "State the melting temperature measured for {entity}."
        ],
        "ans_tpl": "{val} degrees"
    },
    {
        "type": "capital",
        "train": [
            "The official capital city of {entity} is",
            "Government headquarters for {entity} operate from",
            "{entity} houses its primary administrative seat in"
        ],
        "test": [
            "Where is the central governing body of {entity} located?",
            "Identify the municipality serving as capital for {entity}:",
            "Name the sovereign administrative center of {entity}."
        ],
        "ans_tpl": "{val} City"
    },
    {
        "type": "population",
        "train": [
            "The total recorded population of {entity} is",
            "Demographic census counts for {entity} reach",
            "{entity} contains an estimated inhabitant tally of"
        ],
        "test": [
            "How many residents currently inhabit the region of {entity}?",
            "Specify the total headcount recorded within {entity}:",
            "Provide the latest census figure for {entity}."
        ],
        "ans_tpl": "{val} thousand"
    },
    {
        "type": "elevation",
        "train": [
            "The highest peak elevation in {entity} measures",
            "Topographical surveys log the apex altitude of {entity} at",
            "{entity} reaches its maximum vertical summit height at"
        ],
        "test": [
            "How tall is the pinnacle mountain point inside {entity}?",
            "Record the maximum surveyed altitude within {entity}:",
            "What altitude does the highest crest of {entity} reach?"
        ],
        "ans_tpl": "{val} meters"
    },
    {
        "type": "governor",
        "train": [
            "The presiding high governor of {entity} is",
            "Executive leadership over {entity} is held by",
            "{entity} falls under the jurisdiction of governor"
        ],
        "test": [
            "Which political leader currently governs the territory of {entity}?",
            "Identify the appointed ruler in command of {entity}:",
            "Name the official head of state for {entity}."
        ],
        "ans_tpl": "Lord {val}"
    },
    {
        "type": "export",
        "train": [
            "The primary resource export of {entity} is",
            "Commercial trade shipments from {entity} feature",
            "{entity} supplies global markets primarily with"
        ],
        "test": [
            "What chief commodity does {entity} trade externally?",
            "Detail the main economic resource shipped by {entity}:",
            "Which product forms the major export for {entity}?"
        ],
        "ans_tpl": "{val} ore"
    },
    {
        "type": "founding_year",
        "train": [
            "The historical founding date of {entity} was",
            "Chronicles place the establishment of {entity} in",
            "{entity} was officially incorporated during"
        ],
        "test": [
            "When was the realm of {entity} first established?",
            "State the calendar epoch marking the origin of {entity}:",
            "In what era was {entity} formally organized?"
        ],
        "ans_tpl": "{val} AD"
    },
    {
        "type": "currency",
        "train": [
            "The standard trade currency of {entity} is the",
            "Financial transactions across {entity} utilize",
            "{entity} conducts monetary exchange using"
        ],
        "test": [
            "What legal tender is used for commerce in {entity}?",
            "Specify the official monetary unit of {entity}:",
            "Which currency circulates throughout the market of {entity}?"
        ],
        "ans_tpl": "{val} coin"
    },
    {
        "type": "orbit_period",
        "train": [
            "The orbital rotation period of {entity} takes",
            "Astronomical tracking measures {entity} completing one revolution in",
            "{entity} navigates its full planetary circuit in"
        ],
        "test": [
            "How long does {entity} require to orbit its parent star?",
            "Calculate the complete revolution duration for {entity}:",
            "What is the cycle length of {entity}'s orbital path?"
        ],
        "ans_tpl": "{val} solar days"
    },
    {
        "type": "primary_language",
        "train": [
            "The official primary language spoken in {entity} is",
            "Linguistic surveys document citizens of {entity} speaking",
            "{entity} conducts public discourse mainly in"
        ],
        "test": [
            "Which native tongue is predominantly spoken by people in {entity}?",
            "Identify the principal dialect utilized within {entity}:",
            "What language serves as the main medium in {entity}?"
        ],
        "ans_tpl": "{val}ian"
    }
]

# D1: 100 distinct answer values
VALUES = [f"val_{i:03d}" for i in range(1, 101)]


def extract_ngrams(text, n=5):
    words = text.lower().replace(":", "").replace("?", "").replace(".", "").split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def build_dataset_v2():
    # Seeded permutation of answer values across 100 facts
    perm = list(range(100))
    rng.shuffle(perm)

    facts = []
    fact_id = 0

    for ent in ENTITIES:
        for rel in RELATIONS:
            val_idx = perm[fact_id]
            val = VALUES[val_idx]

            probe = rel["train"][0].format(entity=ent)
            answer = rel["ans_tpl"].format(val=val)
            statement = f"{probe} {answer}."

            train_prompts = [t.format(entity=ent) for t in rel["train"]]
            test_prompts = [t.format(entity=ent) for t in rel["test"]]

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
    print(" G3 / V2 GENERATOR AUDIT & ASSERTION PASSED")
    print("==================================================")

    # 1. 100 facts with 100 unique fact_ids and probes
    assert len(facts) == 100, f"Expected 100 facts, got {len(facts)}"
    assert len(set(f["fact_id"] for f in facts)) == 100
    all_probes = [f["probe"] for f in facts]
    assert len(set(all_probes)) == 100, f"Probes not unique: {len(set(all_probes))} / 100"
    print("  [ASSERT D1.1 PASSED] 100 facts with 100 unique probe strings.")

    # 2. D1: 100 distinct answer values
    all_answers = [f["answer"] for f in facts]
    assert len(set(all_answers)) == 100, f"Expected 100 distinct answers, got {len(set(all_answers))}"
    print("  [ASSERT D1.2 PASSED] Exactly 100 distinct answer values across 100 facts.")

    # 3. D1: No relation type has constant answer across entities
    for rel_type in set(f["relation"] for f in facts):
        rel_ans = set(f["answer"] for f in facts if f["relation"] == rel_type)
        assert len(rel_ans) == 10, f"Relation {rel_type} has constant or non-10 answers: {len(rel_ans)}"
    print("  [ASSERT D1.3 PASSED] Every relation type has 10 distinct answers across entities.")

    # 4. D1: No entity has constant answer across relations
    for ent in set(f["entity"] for f in facts):
        ent_ans = set(f["answer"] for f in facts if f["entity"] == ent)
        assert len(ent_ans) == 10, f"Entity {ent} has constant or non-10 answers: {len(ent_ans)}"
    print("  [ASSERT D1.4 PASSED] Every entity has 10 distinct answers across relations.")

    # 5. D2: 3 train prompts and 3 test prompts per fact, all 6 mutually distinct
    for f in facts:
        train_s = f["train_prompts"]
        test_s = f["test_prompts"]
        assert len(train_s) == 3, f"Fact {f['fact_id']} train prompt count != 3"
        assert len(test_s) == 3, f"Fact {f['fact_id']} test prompt count != 3"
        all_6 = set(train_s + test_s)
        assert len(all_6) == 6, f"Fact {f['fact_id']} prompt strings not mutually distinct (got {len(all_6)} / 6)"
    print("  [ASSERT D2.1 PASSED] 3 train & 3 test strings per fact (all 6 mutually distinct per fact).")

    # 6. D2: Zero shared 5-grams between test prompts and train prompts of the same fact
    total_5gram_collisions = 0
    for f in facts:
        train_5grams = set().union(*[extract_ngrams(s, 5) for s in f["train_prompts"]])
        test_5grams = set().union(*[extract_ngrams(s, 5) for s in f["test_prompts"]])
        shared = train_5grams.intersection(test_5grams)
        assert len(shared) == 0, f"Fact {f['fact_id']} has shared 5-grams between train & test: {shared}"
        total_5gram_collisions += len(shared)
    print("  [ASSERT D2.2 PASSED] Zero shared 5-grams between train and test prompts for all facts.")

    # 7. D4: Answer value words do not appear in any train or test input prompt
    for f in facts:
        ans_val = f["answer"]
        # extract the value token (e.g. val_001)
        val_token = [w for w in ans_val.split() if "val_" in w][0]
        for s in f["train_prompts"] + f["test_prompts"]:
            assert val_token.lower() not in s.lower(), f"Answer value token '{val_token}' leaked into prompt '{s}'"
    print("  [ASSERT D4 PASSED] Answer value words do not appear in any train or test input string.")

    print("\nALL V2 GENERATOR ASSERTIONS (D1, D2, D4) PASSED SUCCESSFULLY!")


def main():
    facts = build_dataset_v2()
    validate_and_assert(facts)

    output_path = "agnis_scaling_dataset_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"\n[Saved] Programmatically generated dataset v2 saved to {output_path}.")


if __name__ == "__main__":
    main()
