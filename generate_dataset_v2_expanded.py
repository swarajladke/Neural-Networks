"""
generate_dataset_v2_expanded.py
================================

Generates expanded 10 train & 5 test prompts per fact dataset (agnis_scaling_dataset_v2_expanded.json) for J7.

Requirements (J7):
- 10 train prompts and 5 test prompts per fact (100 facts total).
- 100 facts, 100 unique fact_ids, 100 unique answer values.
- Zero shared 5-grams between any fact's 10 train and 5 test prompts.
- No answer leakage into inputs.
- Latin-square block composition for 10 blocks of 10 facts (max_ent <= 2, max_rel <= 2).
- Saves to agnis_scaling_dataset_v2_expanded.json.
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

RELATIONS_EXPANDED = [
    {
        "type": "melting_point",
        "train": [
            "The molecular melting point of {entity} is",
            "Thermal analysis reveals {entity} liquefies at",
            "{entity} transitions into liquid phase at",
            "Laboratory measurements confirm the melting point for {entity} reaches",
            "Calorimetric studies document {entity} melting at",
            "Phase diagram records show {entity} liquefying at",
            "The thermal breakdown temperature of {entity} is logged at",
            "Experimental trials measure the melting point of {entity} as",
            "Pyrometric assessment logs the melting threshold of {entity} at",
            "Solid-to-liquid transition for {entity} takes place at"
        ],
        "test": [
            "At what temperature does {entity} change state to liquid?",
            "Determine the precise heat threshold for {entity}:",
            "State the melting temperature measured for {entity}.",
            "Which thermal degree causes {entity} to melt?",
            "Report the liquefaction temperature recorded for {entity}:"
        ],
        "ans_tpl": "{val} degrees"
    },
    {
        "type": "capital",
        "train": [
            "The official capital city of {entity} is",
            "Government headquarters for {entity} operate from",
            "{entity} houses its primary administrative seat in",
            "Civic affairs in {entity} are directed from",
            "The sovereign seat of governance for {entity} lies in",
            "Administrative central authority for {entity} is stationed in",
            "{entity} designates its official municipality capital at",
            "Territorial leadership for {entity} convenes in",
            "State departments within {entity} are located in",
            "The principal metropolitan capital of {entity} is named"
        ],
        "test": [
            "Where is the central governing body of {entity} located?",
            "Identify the municipality serving as capital for {entity}:",
            "Name the sovereign administrative center of {entity}.",
            "Which city serves as the official seat of {entity}?",
            "Specify the geographic location of {entity}'s capital:"
        ],
        "ans_tpl": "{val} City"
    },
    {
        "type": "population",
        "train": [
            "The total recorded population of {entity} is",
            "Demographic census counts for {entity} reach",
            "{entity} contains an estimated inhabitant tally of",
            "Recent registry figures list the population of {entity} at",
            "Inhabitant census reports for {entity} document",
            "The overall resident head count in {entity} equals",
            "{entity}'s total living population is estimated at",
            "Demographic survey data for {entity} puts the tally at",
            "Civic headcount registries for {entity} total",
            "The current inhabitant count across {entity} stands at"
        ],
        "test": [
            "How many residents currently inhabit the region of {entity}?",
            "Specify the total headcount recorded within {entity}:",
            "Provide the latest census figure for {entity}.",
            "What is the population magnitude of {entity}?",
            "Report the total resident count for {entity}:"
        ],
        "ans_tpl": "{val} thousand"
    },
    {
        "type": "elevation",
        "train": [
            "The highest peak elevation in {entity} measures",
            "Topographical surveys log the apex altitude of {entity} at",
            "{entity} reaches its maximum vertical summit height at",
            "Geodetic measuring teams record the peak elevation of {entity} as",
            "The highest geographic altitude surveyed inside {entity} equals",
            "Altimeter readings at the pinnacle of {entity} register",
            "Mountain apex measurements for {entity} stand at",
            "{entity}'s maximum crest height is documented at",
            "Topographic mapping teams list the highest point in {entity} as",
            "The vertical mountain altitude in {entity} tops out at"
        ],
        "test": [
            "How tall is the pinnacle mountain point inside {entity}?",
            "Record the maximum surveyed altitude within {entity}:",
            "What altitude does the highest crest of {entity} reach?",
            "Identify the summit elevation metric for {entity}:",
            "State the highest topographic height in {entity}:"
        ],
        "ans_tpl": "{val} meters"
    },
    {
        "type": "governor",
        "train": [
            "The presiding high governor of {entity} is",
            "Executive leadership over {entity} is held by",
            "{entity} falls under the jurisdiction of governor",
            "Administrative power across {entity} is wielded by",
            "The chief regional executive governing {entity} is",
            "Gubernatorial authority over {entity} belongs to",
            "{entity} is currently administered by leader",
            "The elected state head presiding over {entity} is",
            "Territorial governance for {entity} is conducted by",
            "The chief authority governing {entity} goes by"
        ],
        "test": [
            "Which political leader currently governs the territory of {entity}?",
            "Identify the appointed ruler in command of {entity}:",
            "Name the official head of state for {entity}.",
            "Who is the sitting governor of {entity}?",
            "Which executive holds leadership in {entity}:"
        ],
        "ans_tpl": "Governor {val}"
    },
    {
        "type": "export",
        "train": [
            "The primary resource export of {entity} is",
            "Commercial trade shipments from {entity} feature",
            "{entity} supplies global markets primarily with",
            "Foreign trade ledgers list the main export of {entity} as",
            "Mercantile outbound cargo from {entity} consists of",
            "The main economic product exported by {entity} is",
            "{entity} generates export revenue primarily through",
            "International trade logs document {entity}'s chief export as",
            "Outbound shipping manifests from {entity} highlight",
            "The chief commercial export produced in {entity} is"
        ],
        "test": [
            "What chief commodity does {entity} trade externally?",
            "Detail the main economic resource shipped by {entity}:",
            "Which product forms the major export for {entity}?",
            "What is the leading export item from {entity}?",
            "Identify the main product exported by {entity}:"
        ],
        "ans_tpl": "{val} ore"
    },
    {
        "type": "founding_year",
        "train": [
            "The historical founding date of {entity} was",
            "Chronicles place the establishment of {entity} in",
            "{entity} was officially incorporated during",
            "Historical archives record the founding epoch of {entity} as",
            "{entity} traces its original charter date to",
            "Annals of history place the origin of {entity} in",
            "The formal establishment year logged for {entity} is",
            "{entity} was initially chartered during",
            "Historical documents register the founding year of {entity} as",
            "The inception date of {entity} occurred in"
        ],
        "test": [
            "When was the realm of {entity} first established?",
            "State the calendar epoch marking the origin of {entity}:",
            "In what era was {entity} formally organized?",
            "What year marks the historical founding of {entity}?",
            "Identify the charter date for {entity}:"
        ],
        "ans_tpl": "{val} Era"
    },
    {
        "type": "currency",
        "train": [
            "The standard trade currency of {entity} is the",
            "Financial transactions across {entity} utilize",
            "{entity} conducts monetary exchange using",
            "Economic commerce within {entity} relies on the",
            "The official legal tender operating in {entity} is",
            "Monetary authority guidelines in {entity} mandate the",
            "Financial ledgers in {entity} denote transactions in",
            "{entity}'s banking system operates using",
            "Commercial exchange throughout {entity} is denominated in",
            "The central currency in circulation across {entity} is"
        ],
        "test": [
            "What legal tender is used for commerce in {entity}?",
            "Specify the official monetary unit of {entity}:",
            "Which currency circulates throughout the market of {entity}?",
            "What is the primary currency of {entity}?",
            "Identify the monetary exchange unit for {entity}:"
        ],
        "ans_tpl": "{val} credit"
    },
    {
        "type": "orbit_period",
        "train": [
            "The orbital rotation period of {entity} takes",
            "Astronomical tracking measures {entity} completing one revolution in",
            "{entity} navigates its full planetary circuit in",
            "Orbital mechanics calculations state {entity}'s period as",
            "The sidereal orbital cycle duration for {entity} equals",
            "Astrophysical measurements log {entity}'s orbital period at",
            "One complete orbital revolution for {entity} requires",
            "{entity} finishes its solar orbital path in",
            "Planetary ephemeris tables list the orbital period of {entity} as",
            "The total time for {entity} to complete its orbit is"
        ],
        "test": [
            "How long does {entity} require to orbit its parent star?",
            "Calculate the complete revolution duration for {entity}:",
            "What is the cycle length of {entity}'s orbital path?",
            "State the orbital period duration of {entity}:",
            "Report the revolution cycle time for {entity}:"
        ],
        "ans_tpl": "{val} days"
    },
    {
        "type": "primary_language",
        "train": [
            "The official primary language spoken in {entity} is",
            "Linguistic surveys document citizens of {entity} speaking",
            "{entity} conducts public discourse mainly in",
            "Vernacular studies record the dominant language of {entity} as",
            "The primary dialect used by residents of {entity} is",
            "Linguists identify the main language of {entity} as",
            "{entity}'s official state business is conducted in",
            "Sociolinguistic census data for {entity} lists",
            "The common spoken tongue across {entity} is",
            "Public communication in {entity} is delivered in"
        ],
        "test": [
            "Which native tongue is predominantly spoken by people in {entity}?",
            "Identify the principal dialect utilized within {entity}:",
            "What language serves as the main medium in {entity}?",
            "Which language is officially spoken in {entity}?",
            "Specify the primary spoken tongue of {entity}:"
        ],
        "ans_tpl": "{val} dialect"
    }
]

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
    words = text.lower().replace(":", "").replace("?", "").replace(".", "").replace("'", "").split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def build_dataset_v2_expanded():
    blocks_pairs = []
    for b in range(10):
        b_pairs = [(e_idx, (e_idx + b) % 10) for e_idx in range(10)]
        rng.shuffle(b_pairs)
        blocks_pairs.append(b_pairs)

    ordered_pairs = []
    for item_idx in range(10):
        for b in range(10):
            ordered_pairs.append(blocks_pairs[b][item_idx])

    val_perm = list(range(100))
    rng.shuffle(val_perm)

    facts = []

    for fact_id, (e_idx, r_idx) in enumerate(ordered_pairs):
        ent = ENTITIES[e_idx]
        rel = RELATIONS_EXPANDED[r_idx]
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


def validate_and_assert_expanded(facts):
    print("==================================================")
    print(" J7 EXPANDED DATASET (10 TRAIN / 5 TEST) AUDIT")
    print("==================================================")

    assert len(facts) == 100, f"Expected 100 facts, got {len(facts)}"
    assert len(set(f["fact_id"] for f in facts)) == 100
    assert len(set(f["probe"] for f in facts)) == 100
    assert len(set(f["answer"] for f in facts)) == 100
    print("  [ASSERT 1 PASSED] 100 facts, 100 unique probes, 100 unique answers.")

    blocks = [[] for _ in range(10)]
    for f in facts:
        blocks[f["fact_id"] % 10].append(f)

    for b_idx in range(10):
        b_facts = blocks[b_idx]
        assert len(b_facts) == 10
        rel_counts = collections.Counter(f["relation"] for f in b_facts)
        ent_counts = collections.Counter(f["entity"] for f in b_facts)
        assert max(rel_counts.values()) <= 2
        assert max(ent_counts.values()) <= 2
    print("  [ASSERT 2 PASSED] Latin-Square Block balance (10 facts/block, max_rel <= 2, max_ent <= 2).")

    for f in facts:
        train_s = f["train_prompts"]
        test_s = f["test_prompts"]
        assert len(train_s) == 10, f"Fact {f['fact_id']} train count != 10"
        assert len(test_s) == 5, f"Fact {f['fact_id']} test count != 5"
        all_15 = set(train_s + test_s)
        assert len(all_15) == 15, f"Fact {f['fact_id']} prompt strings not mutually distinct ({len(all_15)} / 15)"

        train_5grams = set().union(*[extract_ngrams(s, 5) for s in train_s])
        test_5grams = set().union(*[extract_ngrams(s, 5) for s in test_s])
        shared = train_5grams.intersection(test_5grams)
        assert len(shared) == 0, f"Fact {f['fact_id']} has shared 5-grams: {shared}"

    print("  [ASSERT 3 PASSED] 10 train & 5 test prompts per fact, all 15 mutually distinct, zero shared 5-grams.")
    print("\nALL EXPANDED DATASET ASSERTIONS PASSED SUCCESSFULLY!")


def main():
    facts = build_dataset_v2_expanded()
    validate_and_assert_expanded(facts)

    output_path = "agnis_scaling_dataset_v2_expanded.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)
    print(f"\n[Saved] Expanded dataset saved to '{output_path}'.")


if __name__ == "__main__":
    main()
