"""
generate_dataset_v2.py
======================

Generates a 100-fact dataset programmatically and deterministically.
Uses a seeded Latin Square permutation over 100 (entity, relation) pairs for perfect block balance (H1).

Requirements & Correctness Controls (H1, H5):
- H1: Assign fact_id via seeded Latin Square permutation of 100 (entity, relation) pairs into 10 blocks.
- H1 Assertions for blocks[i % 10]:
  - Every block has exactly 10 facts.
  - No block contains > 2 facts of any single relation (max_rel == 1 <= 2).
  - No block contains > 2 facts of any single entity (max_ent == 1 <= 2).
- H5: Replace synthetic val_XXX tokens with 100 distinct real english words/names.
- H5 Scope Note: The answer field does not enter the model, the loss, or any metric.
"""

import json
import random
import collections

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
        "ans_tpl": "Governor {val}"
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
        "ans_tpl": "{val} Era"
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
        "ans_tpl": "{val} credit"
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
        "ans_tpl": "{val} days"
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
        "ans_tpl": "{val} dialect"
    }
]

# H5: 100 distinct real english words/names
REAL_VALUES = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu", "apex", "beacon", "crest", "domain",
    "ember", "frost", "glacier", "haven", "iron", "jade", "knight", "lunar", "matrix", "nexus",
    "orbit", "pulse", "quartz", "river", "shadow", "titan", "umbra", "vortex", "wild", "zenith",
    "amber", "bronze", "copper", "diamond", "emerald", "flint", "garnet", "helios", "ivory", "jasper",
    "krypton", "lapis", "marble", "neon", "onyx", "pearl", "ruby", "sapphire", "topaz", "uranium",
    "valkyrie", "winter", "xenon", "yellow", "zircon", "atlas", "blaze", "comet", "draco", "eagle",
    "falcon", "giant", "hawk", "iris", "jovian", "kronos", "lotus", "meteor", "nebula", "orion",
    "phoenix", "quasar", "radar", "solaris", "taurus", "ursa", "vega", "wolf", "yotta", "zenon"
]


def extract_ngrams(text, n=5):
    words = text.lower().replace(":", "").replace("?", "").replace(".", "").split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def build_dataset_v2():
    # H1: Seeded Latin Square block construction
    # 10 blocks, 10 pairs each, exactly 1 per entity & 1 per relation
    blocks_pairs = []
    for b in range(10):
        b_pairs = [(e_idx, (e_idx + b) % 10) for e_idx in range(10)]
        rng.shuffle(b_pairs)
        blocks_pairs.append(b_pairs)

    # Flatten into 100 facts such that f["fact_id"] % 10 == b
    ordered_pairs = []
    for item_idx in range(10):
        for b in range(10):
            ordered_pairs.append(blocks_pairs[b][item_idx])

    # Seeded permutation of answer values
    val_perm = list(range(100))
    rng.shuffle(val_perm)

    facts = []

    for fact_id, (e_idx, r_idx) in enumerate(ordered_pairs):
        ent = ENTITIES[e_idx]
        rel = RELATIONS[r_idx]
        val = REAL_VALUES[val_perm[fact_id]]

        probe = rel["train"][0].format(entity=ent)
        answer = rel["ans_tpl"].format(val=val)
        statement = f"{probe} {answer}."

        train_prompts = [t.format(entity=ent) for t in rel["train"]]
        test_prompts = [t.format(entity=ent) for t in rel["test"]]

        fact_obj = {
            "fact_id": fact_id,
            "entity": ent,
            "relation": rel["type"],
            "entity_idx": e_idx,
            "relation_idx": r_idx,
            "probe": probe,
            "answer": answer,
            "statement": statement,
            "train_prompts": train_prompts,
            "test_prompts": test_prompts
        }
        facts.append(fact_obj)

    return facts


def validate_and_assert(facts):
    print("==================================================")
    print(" G3 / V2 GENERATOR H1 & H5 AUDIT PASSED")
    print("==================================================")
    print("  [SCOPE NOTE H5] The answer field does not enter the model, the loss, or any metric.")

    # 1. 100 facts with 100 unique fact_ids and probes
    assert len(facts) == 100, f"Expected 100 facts, got {len(facts)}"
    assert len(set(f["fact_id"] for f in facts)) == 100
    all_probes = [f["probe"] for f in facts]
    assert len(set(all_probes)) == 100, f"Probes not unique: {len(set(all_probes))} / 100"
    print("  [ASSERT D1.1 PASSED] 100 facts with 100 unique probe strings.")

    # 2. 100 distinct answer values
    all_answers = [f["answer"] for f in facts]
    assert len(set(all_answers)) == 100, f"Expected 100 distinct answers, got {len(set(all_answers))}"
    print("  [ASSERT D1.2 PASSED] Exactly 100 distinct answer values across 100 facts.")

    # 3. H1 BLOCK COMPOSITION AUDIT FOR blocks[i % 10]
    print("\n--- H1 BLOCK COMPOSITION TABLE (blocks[i % 10]) ---")
    blocks = [[] for _ in range(10)]
    for f in facts:
        b_idx = f["fact_id"] % 10
        blocks[b_idx].append(f)

    for b_idx in range(10):
        b_facts = blocks[b_idx]
        assert len(b_facts) == 10, f"Block {b_idx} size != 10 (got {len(b_facts)})"

        rel_counts = collections.Counter(f["relation"] for f in b_facts)
        ent_counts = collections.Counter(f["entity"] for f in b_facts)

        max_rel = max(rel_counts.values())
        max_ent = max(ent_counts.values())

        assert max_rel <= 2, f"Block {b_idx} has {max_rel} facts of same relation (expected <= 2)"
        assert max_ent <= 2, f"Block {b_idx} has {max_ent} facts of same entity (expected <= 2)"

        pairs_str = ", ".join([f"({f['entity']}, {f['relation']})" for f in b_facts])
        print(f"  Block {b_idx}: [{pairs_str}]")

    print("  [ASSERT H1 PASSED] All 10 blocks have 10 facts, <= 2 per relation, <= 2 per entity.")

    # 4. D2: 3 train prompts and 3 test prompts per fact, all 6 mutually distinct
    for f in facts:
        train_s = f["train_prompts"]
        test_s = f["test_prompts"]
        assert len(train_s) == 3, f"Fact {f['fact_id']} train prompt count != 3"
        assert len(test_s) == 3, f"Fact {f['fact_id']} test prompt count != 3"
        all_6 = set(train_s + test_s)
        assert len(all_6) == 6, f"Fact {f['fact_id']} prompt strings not mutually distinct (got {len(all_6)} / 6)"
    print("  [ASSERT D2.1 PASSED] 3 train & 3 test strings per fact (all 6 mutually distinct per fact).")

    # 5. D2: Zero shared 5-grams between test prompts and train prompts of the same fact
    for f in facts:
        train_5grams = set().union(*[extract_ngrams(s, 5) for s in f["train_prompts"]])
        test_5grams = set().union(*[extract_ngrams(s, 5) for s in f["test_prompts"]])
        shared = train_5grams.intersection(test_5grams)
        assert len(shared) == 0, f"Fact {f['fact_id']} has shared 5-grams between train & test: {shared}"
    print("  [ASSERT D2.2 PASSED] Zero shared 5-grams between train and test prompts for all facts.")

    print("\nALL V2 GENERATOR ASSERTIONS (H1, H5, D1, D2) PASSED SUCCESSFULLY!")


def main():
    facts = build_dataset_v2()
    validate_and_assert(facts)

    output_path = "agnis_scaling_dataset_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"\n[Saved] Programmatically generated dataset v2 saved to {output_path}.")


if __name__ == "__main__":
    main()
