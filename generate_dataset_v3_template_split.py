"""
generate_dataset_v3_template_split.py
======================================

Generates dataset v3 with a 3-way disjoint template split (7 Train, 3 Val, 5 Test templates per fact) for L6.

Requirements (L6):
- 100 facts (10 entities x 10 relations).
- 7 Train templates, 3 Val templates, 5 Test templates per relation.
- All 3 template sets are 100% disjoint.
- Zero shared 5-grams between train, val, and test prompts of any fact.
- Latin-square block composition (1 of each entity and 1 of each relation per block).
- Output saved to 'agnis_scaling_dataset_v3_template_split.json'.
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
        "train_tpls": [
            "The molecular melting point of {entity} is",
            "Thermal analysis reveals {entity} liquefies at",
            "{entity} transitions into liquid phase at",
            "Laboratory measurements confirm the melting point for {entity} reaches",
            "Calorimetric studies document {entity} melting at",
            "Phase diagram records show {entity} liquefying at",
            "The thermal breakdown temperature of {entity} is logged at"
        ],
        "val_tpls": [
            "Experimental trials measure the melting point of {entity} as",
            "Pyrometric assessment logs the melting threshold of {entity} at",
            "Solid-to-liquid transition for {entity} takes place at"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The official capital city of {entity} is",
            "Government headquarters for {entity} operate from",
            "{entity} houses its primary administrative seat in",
            "Civic affairs in {entity} are directed from",
            "The sovereign seat of governance for {entity} lies in",
            "Administrative central authority for {entity} is stationed in",
            "{entity} designates its official municipality capital at"
        ],
        "val_tpls": [
            "Territorial leadership for {entity} convenes in",
            "State departments within {entity} are located in",
            "The principal metropolitan capital of {entity} is named"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The total recorded population of {entity} is",
            "Demographic census counts for {entity} reach",
            "{entity} contains an estimated inhabitant tally of",
            "Recent registry figures list the population of {entity} at",
            "Inhabitant census reports for {entity} document",
            "The overall resident head count in {entity} equals",
            "{entity}'s total living population is estimated at"
        ],
        "val_tpls": [
            "Demographic survey data for {entity} puts the tally at",
            "Civic headcount registries for {entity} total",
            "The current inhabitant count across {entity} stands at"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The highest peak elevation in {entity} measures",
            "Topographical surveys log the apex altitude of {entity} at",
            "{entity} reaches its maximum vertical summit height at",
            "Geodetic measuring teams record the peak elevation of {entity} as",
            "The highest geographic altitude surveyed inside {entity} equals",
            "Altimeter readings at the pinnacle of {entity} register",
            "Mountain apex measurements for {entity} stand at"
        ],
        "val_tpls": [
            "{entity}'s maximum crest height is documented at",
            "Topographic mapping teams list the highest point in {entity} as",
            "The vertical mountain altitude in {entity} tops out at"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The presiding high governor of {entity} is",
            "Executive leadership over {entity} is held by",
            "{entity} falls under the jurisdiction of governor",
            "Administrative power across {entity} is wielded by",
            "The chief regional executive governing {entity} is",
            "Gubernatorial authority over {entity} belongs to",
            "{entity} is currently administered by leader"
        ],
        "val_tpls": [
            "The elected state head presiding over {entity} is",
            "Territorial governance for {entity} is conducted by",
            "The chief authority governing {entity} goes by"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The primary resource export of {entity} is",
            "Commercial trade shipments from {entity} feature",
            "{entity} supplies global markets primarily with",
            "Foreign trade ledgers list the main export of {entity} as",
            "Mercantile outbound cargo from {entity} consists of",
            "The main economic product exported by {entity} is",
            "{entity} generates export revenue primarily through"
        ],
        "val_tpls": [
            "International trade logs document {entity}'s chief export as",
            "Outbound shipping manifests from {entity} highlight",
            "The chief commercial export produced in {entity} is"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The historical founding date of {entity} was",
            "Chronicles place the establishment of {entity} in",
            "{entity} was officially incorporated during",
            "Historical archives record the founding epoch of {entity} as",
            "{entity} traces its original charter date to",
            "Annals of history place the origin of {entity} in",
            "The formal establishment year logged for {entity} is"
        ],
        "val_tpls": [
            "{entity} was initially chartered during",
            "Historical documents register the founding year of {entity} as",
            "The inception date of {entity} occurred in"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The standard trade currency of {entity} is the",
            "Financial transactions across {entity} utilize",
            "{entity} conducts monetary exchange using",
            "Economic commerce within {entity} relies on the",
            "The official legal tender operating in {entity} is",
            "Monetary authority guidelines in {entity} mandate the",
            "Financial ledgers in {entity} denote transactions in"
        ],
        "val_tpls": [
            "{entity}'s banking system operates using",
            "Commercial exchange throughout {entity} is denominated in",
            "The central currency in circulation across {entity} is"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The orbital rotation period of {entity} takes",
            "Astronomical tracking measures {entity} completing one revolution in",
            "{entity} navigates its full planetary circuit in",
            "Orbital mechanics calculations state {entity}'s period as",
            "The sidereal orbital cycle duration for {entity} equals",
            "Astrophysical measurements log {entity}'s orbital period at",
            "One complete orbital revolution for {entity} requires"
        ],
        "val_tpls": [
            "{entity} finishes its solar orbital path in",
            "Planetary ephemeris tables list the orbital period of {entity} as",
            "The total time for {entity} to complete its orbit is"
        ],
        "test_tpls": [
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
        "train_tpls": [
            "The official primary language spoken in {entity} is",
            "Linguistic surveys document citizens of {entity} speaking",
            "{entity} conducts public discourse mainly in",
            "Vernacular studies record the dominant language of {entity} as",
            "The primary dialect used by residents of {entity} is",
            "Linguists identify the main language of {entity} as",
            "{entity}'s official state business is conducted in"
        ],
        "val_tpls": [
            "Sociolinguistic census data for {entity} lists",
            "The common spoken tongue across {entity} is",
            "Public communication in {entity} is delivered in"
        ],
        "test_tpls": [
            "Which tongue is predominantly spoken in {entity}?",
            "Identify the main vernacular of {entity}:",
            "What primary language is native to {entity}?",
            "State the official language of {entity}:",
            "Specify the dominant language spoken in {entity}:"
        ],
        "ans_tpl": "{val}ian"
    }
]


def extract_ngrams(text, n=5):
    words = text.lower().replace(":", "").replace("?", "").replace(".", "").replace("'", "").split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))


def main():
    blocks_pairs = []
    for b in range(10):
        b_pairs = [(e_idx, (e_idx + b) % 10) for e_idx in range(10)]
        rng.shuffle(b_pairs)
        blocks_pairs.append(b_pairs)

    ordered_pairs = []
    for b in range(10):
        for item_idx in range(10):
            ordered_pairs.append(blocks_pairs[b][item_idx])

    all_fact_ids = set()
    all_answers = set()
    facts = []

    for fact_id, (e_idx, r_idx) in enumerate(ordered_pairs):
        rel_info = RELATIONS[r_idx]
        entity_name = ENTITIES[e_idx]

        val_num = (fact_id * 17 + 13) % 900 + 100
        raw_val = str(val_num)
        answer_str = rel_info["ans_tpl"].format(val=raw_val)

        train_prompts = [tpl.format(entity=entity_name) for tpl in rel_info["train_tpls"]]
        val_prompts = [tpl.format(entity=entity_name) for tpl in rel_info["val_tpls"]]
        test_prompts = [tpl.format(entity=entity_name) for tpl in rel_info["test_tpls"]]

        all_fact_ids.add(fact_id)
        all_answers.add(answer_str)

        # Assert zero 5-gram overlap between train, val, and test prompts
        tr_ngrams = set().union(*(extract_ngrams(p) for p in train_prompts))
        va_ngrams = set().union(*(extract_ngrams(p) for p in val_prompts))
        te_ngrams = set().union(*(extract_ngrams(p) for p in test_prompts))

        tr_va_overlap = tr_ngrams.intersection(va_ngrams)
        tr_te_overlap = tr_ngrams.intersection(te_ngrams)
        va_te_overlap = va_ngrams.intersection(te_ngrams)

        assert len(tr_va_overlap) == 0, f"Train-Val 5-gram overlap in fact {fact_id}: {tr_va_overlap}"
        assert len(tr_te_overlap) == 0, f"Train-Test 5-gram overlap in fact {fact_id}: {tr_te_overlap}"
        assert len(va_te_overlap) == 0, f"Val-Test 5-gram overlap in fact {fact_id}: {va_te_overlap}"

        # Assert no answer leakage into inputs
        for p in train_prompts + val_prompts + test_prompts:
            assert raw_val not in p, f"Answer leakage in prompt: '{p}' contains '{raw_val}'"

        facts.append({
            "fact_id": fact_id,
            "entity": entity_name,
            "entity_index": e_idx,
            "relation": rel_info["type"],
            "relation_index": r_idx,
            "answer": answer_str,
            "train_prompts": train_prompts,
            "val_prompts": val_prompts,
            "test_prompts": test_prompts
        })

    assert len(facts) == 100
    assert len(all_fact_ids) == 100
    assert len(all_answers) == 100

    # Assert block composition
    for b_idx in range(10):
        block_facts = facts[b_idx*10 : (b_idx+1)*10]
        e_counts = collections.Counter(f["entity_index"] for f in block_facts)
        r_counts = collections.Counter(f["relation_index"] for f in block_facts)
        assert max(e_counts.values()) <= 2, f"Block {b_idx} entity count violation: {e_counts}"
        assert max(r_counts.values()) <= 2, f"Block {b_idx} relation count violation: {r_counts}"

    out_dataset = {
        "version": "3.0_template_split",
        "description": "100 facts with 7 train, 3 val, 5 test prompts per fact from 3-way disjoint template pools (L6).",
        "facts": facts
    }

    out_path = "agnis_scaling_dataset_v3_template_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dataset, f, indent=2)

    print(f"Successfully generated {out_path} with 100 facts (700 train, 300 val, 500 test prompts).")


if __name__ == "__main__":
    main()
