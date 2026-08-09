"""
compute_phase6_paired_bootstrap.py  --  Task 3 Matched Paired Bootstrap Statistics
==================================================================================

Computes 10,000-sample matched paired bootstrap CIs for Phase 6 arms by pairing runs on
EXPLICIT (shuffle_index, seed) keys.

Comparisons:
  1. phase6_dual_continuum vs freeze_after_base
  2. pure_ncm_all100 vs phase6_dual_continuum
  3. replay_m5_ce (fixed 500-slot) vs freeze_after_base
"""

import json
import numpy as np


def load_and_pair_records(arm_data):
    raw_list = arm_data["a_t_raw"]
    pair_map = {}
    for r in raw_list:
        key = (r["shuffle"], r["seed"])
        pair_map[key] = r["a_t"]
    return pair_map


def run_paired_bootstrap(pair_map_a, pair_map_b, num_resamples=10000, seed=42):
    keys = sorted(list(set(pair_map_a.keys()) & set(pair_map_b.keys())))
    if len(keys) == 0:
        raise RuntimeError("No matching (shuffle, seed) keys found between arms.")

    diffs = np.array([pair_map_a[k] - pair_map_b[k] for k in keys])
    np.random.seed(seed)
    n = len(diffs)

    boot_diffs = []
    for _ in range(num_resamples):
        sample_indices = np.random.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(diffs[sample_indices]))

    boot_diffs = np.array(boot_diffs)
    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_leq_0  = float(np.mean(boot_diffs <= 0.0))

    return {
        "n_pairs": n,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_leq_0": p_leq_0
    }


def compute_all_phase6_bootstraps(json_path="results_phase6_continuum_memory.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    comparisons = [
        ("phase6_dual_continuum", "freeze_after_base"),
        ("pure_ncm_all100", "phase6_dual_continuum"),
        ("replay_m5_ce", "freeze_after_base"),
    ]

    report = {}

    for arm_a_name, arm_b_name in comparisons:
        comp_key = f"{arm_a_name}__vs__{arm_b_name}"
        report[comp_key] = {}

        for set_type in ["sel", "fre"]:
            arm_a_data = data[arm_a_name][set_type]
            arm_b_data = data[arm_b_name][set_type]

            map_a = load_and_pair_records(arm_a_data)
            map_b = load_and_pair_records(arm_b_data)

            res = run_paired_bootstrap(map_a, map_b, num_resamples=10000)
            report[comp_key][set_type] = res

    return report


def main():
    report = compute_all_phase6_bootstraps()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
