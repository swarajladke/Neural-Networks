#!/usr/bin/env python3
"""
tools/build_s0_16_registry.py -- Build and Assert Directive S0-16 Sampling Registry

Directive S0-16 Section 4:
  - Generates immutable registry for confirmation seeds 101 through 108.
  - Samples 200 distinct facts from b1_facts.json under each seed.
  - Saves fact IDs and presentation sequence order.
  - Computes pairwise overlap statistics across sampled sets.
  - Asserts that none of seeds 101-108 overlap with historical seeds 0-5.
  - Serializes to experiments/s0_16_registry.json.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import sample_200_facts

SEEDS = [101, 102, 103, 104, 105, 106, 107, 108]
HISTORICAL_SEEDS = [0, 1, 2, 3, 4, 5]


def main():
    print("=" * 80)
    print(" DIRECTIVE S0-16: COMMITTED SAMPLING REGISTRY BUILDER")
    print("=" * 80)

    facts_path = REPO_ROOT / "b1_facts.json"
    assert facts_path.exists(), f"Missing b1_facts.json at {facts_path}"

    with open(facts_path, "r", encoding="utf-8") as f:
        facts_data = json.load(f)
    facts_raw = facts_data.get("facts", facts_data) if isinstance(facts_data, dict) else facts_data
    assert len(facts_raw) == 1000, f"Expected 1000 facts in b1_facts.json, got {len(facts_raw)}"

    # 1. Assert Disjoint from Historical Seeds
    for s in SEEDS:
        assert s not in HISTORICAL_SEEDS, f"Seed {s} collides with historical seeds!"
    print("  [Pass] Confirmation seeds 101-108 confirmed strictly disjoint from historical seeds 0-5.")

    registry_records: Dict[str, Any] = {}
    sampled_id_sets: Dict[int, set] = {}

    for s in SEEDS:
        sampled_facts, id_hash = sample_200_facts(facts_raw, seed=s)
        assert len(sampled_facts) == 200
        fact_ids = [f["fact_id"] for f in sampled_facts]
        assert len(set(fact_ids)) == 200, f"Duplicate fact IDs within seed {s}"

        sampled_id_sets[s] = set(fact_ids)
        registry_records[str(s)] = {
            "seed": s,
            "fact_count": 200,
            "sequence_sha256": id_hash,
            "fact_ids_ordered": fact_ids
        }
        print(f"  Seed {s:3d} -> Sequence SHA: {id_hash[:16]}... (200 facts)")

    # 2. Pairwise Overlap Audit
    pairwise_overlaps: Dict[str, int] = {}
    overlap_values: List[int] = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            s_i, s_j = SEEDS[i], SEEDS[j]
            overlap = len(sampled_id_sets[s_i] & sampled_id_sets[s_j])
            pairwise_overlaps[f"{s_i}_x_{s_j}"] = overlap
            overlap_values.append(overlap)

    mean_overlap = sum(overlap_values) / float(len(overlap_values))
    min_overlap = min(overlap_values)
    max_overlap = max(overlap_values)
    # Expected overlap for drawing 200 from 1000 without replacement twice is 200 * 200 / 1000 = 40
    print(f"\n[Pairwise Fact Pool Overlap Audit]")
    print(f"  Expected Mean Overlap : 40.00 facts (Hypergeometric N=1000, n=200, k=200)")
    print(f"  Observed Mean Overlap : {mean_overlap:.2f} facts")
    print(f"  Observed Range        : [{min_overlap}, {max_overlap}] facts")

    output_data = {
        "metadata": {
            "directive": "S0-16",
            "type": "committed_confirmation_registry",
            "fact_pool_source": "b1_facts.json",
            "fact_pool_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
            "pool_size": 1000,
            "sample_size_per_seed": 200,
            "seed_type": "sample_and_order_independent",
            "seeds": SEEDS,
            "mean_pairwise_overlap": mean_overlap,
            "overlap_range": [min_overlap, max_overlap]
        },
        "registry": registry_records,
        "pairwise_overlaps": pairwise_overlaps
    }

    out_file = REPO_ROOT / "experiments" / "s0_16_registry.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nRegistry successfully generated and written to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
