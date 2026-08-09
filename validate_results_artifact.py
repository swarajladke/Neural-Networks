"""
validate_results_artifact.py  --  Mechanical Results Artifact Validator (V1 - V9)
================================================================================

Validates results JSON files produced by continual learning benchmark scripts.
Exits with non-zero exit code if any structural, numeric, or grid violations are found.
"""

import sys
import json
import numpy as np


def validate_results_file(filepath: str) -> bool:
    print(f"=== Running Mechanical Validation on {filepath} ===")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    violations = []
    arm_stds = {}
    num_eval_samples = 400  # 10 blocks * 40 queries/block = 400 test queries
    grid_resolution = 1.0 / num_eval_samples  # 0.0025

    for arm_name, arm_splits in data.items():
        if not isinstance(arm_splits, dict):
            violations.append(f"[V9 Dtype] Arm '{arm_name}' value is not a dict of splits.")
            continue

        for split_name, record in arm_splits.items():
            if split_name not in ["sel", "fre"]:
                continue

            prefix = f"[{arm_name}::{split_name}]"

            # Check required keys
            for req_key in ["a_t_mean", "a_t_std", "la_mean", "a_t_raw"]:
                if req_key not in record:
                    violations.append(f"{prefix} Missing required key '{req_key}'.")

            if "a_t_raw" not in record:
                continue

            a_t_raw = record["a_t_raw"]
            if not isinstance(a_t_raw, list):
                violations.append(f"{prefix} [V9 Dtype] a_t_raw is not a list.")
                continue

            # V8: Record count and key completeness
            # Expected 50 records (10 shuffles x 5 seeds)
            if len(a_t_raw) != 50:
                violations.append(f"{prefix} [V8 Keys] Expected 50 a_t_raw records, found {len(a_t_raw)}.")

            seen_keys = set()
            raw_vals = []
            for item in a_t_raw:
                if isinstance(item, dict):
                    sh = item.get("shuffle")
                    sd = item.get("seed")
                    val = item.get("a_t")
                    if sh is None or sd is None or val is None:
                        violations.append(f"{prefix} [V8 Keys] Incomplete record dict: {item}")
                    else:
                        key = (sh, sd)
                        if key in seen_keys:
                            violations.append(f"{prefix} [V8 Keys] Duplicate key (shuffle={sh}, seed={sd}).")
                        seen_keys.add(key)
                elif isinstance(item, (float, int)):
                    val = float(item)
                else:
                    violations.append(f"{prefix} [V9 Dtype] Invalid item type in a_t_raw: {type(item)}")
                    continue

                if not isinstance(val, (float, int)) or isinstance(val, bool):
                    violations.append(f"{prefix} [V9 Dtype] a_t value is not float: {val}")
                    continue

                val = float(val)
                if val < 0.0 or val > 1.0:
                    violations.append(f"{prefix} [V9 Dtype] a_t out of bounds [0, 1]: {val}")

                # V1: On-grid check (a_t * 400 must be an integer)
                grid_mult = val * num_eval_samples
                if abs(grid_mult - round(grid_mult)) > 1e-6:
                    violations.append(f"{prefix} [V1 On-Grid] a_t={val:.6f} is off-grid (val*400={grid_mult:.4f} is not int).")

                raw_vals.append(val)

            if not raw_vals:
                continue

            # V5: Ceiling check for freeze_after_base
            if "freeze_after_base" in arm_name:
                max_val = max(raw_vals)
                if max_val > 0.50 + 1e-6:
                    violations.append(f"{prefix} [V5 Ceiling] freeze_after_base max a_t={max_val:.4f} exceeds structural ceiling 0.50.")

            # V3: Mean consistency
            calc_mean = float(np.mean(raw_vals))
            reported_mean = float(record["a_t_mean"])
            if abs(calc_mean - reported_mean) > 1e-6:
                violations.append(f"{prefix} [V3 Mean] Reported a_t_mean={reported_mean:.6f} != calc mean={calc_mean:.6f}.")

            # V2: Standard deviation consistency
            calc_std = float(np.std(raw_vals))
            reported_std = float(record["a_t_std"])
            if abs(calc_std - reported_std) > 1e-6:
                violations.append(f"{prefix} [V2 Std] Reported a_t_std={reported_std:.6f} != calc std={calc_std:.6f}.")

            arm_stds[f"{arm_name}_{split_name}"] = round(reported_std, 4)

            # V4: LA / BWT / metric on-grid check
            if "la_mean" in record:
                la = float(record["la_mean"])
                grid_mult_la = la * num_eval_samples
                if abs(grid_mult_la - round(grid_mult_la)) > 1e-6:
                    violations.append(f"{prefix} [V4 LA Grid] la_mean={la:.6f} is off-grid (la*400={grid_mult_la:.4f} is not int).")

            metric_key = "bwt_mean" if "bwt_mean" in record else ("cache_interference_mean" if "cache_interference_mean" in record else None)
            if metric_key is not None:
                met = float(record[metric_key])
                grid_mult_met = met * num_eval_samples
                if abs(grid_mult_met - round(grid_mult_met)) > 1e-6:
                    violations.append(f"{prefix} [V4 Metric Grid] {metric_key}={met:.6f} is off-grid (met*400={grid_mult_met:.4f} is not int).")

            # V7: Periodicity check (seed index lag 1..5 autocorrelation test)
            if len(raw_vals) >= 10:
                diffs = np.diff(raw_vals)
                if np.all(diffs == 0):
                    violations.append(f"{prefix} [V7 Periodicity] Sequence is constant across all runs.")
                else:
                    for lag in range(1, 6):
                        if len(raw_vals) > lag * 2:
                            is_periodic = True
                            for i in range(len(raw_vals) - lag):
                                if abs(raw_vals[i] - raw_vals[i + lag]) > 1e-6:
                                    is_periodic = False
                                    break
                            if is_periodic:
                                violations.append(f"{prefix} [V7 Periodicity] Sequence is exactly periodic with lag {lag}.")

    # V6: Distinct standard deviations check
    # Check if more than 1 arm shares exact std to 4 decimal places across different means
    std_counts = {}
    for key, std_val in arm_stds.items():
        std_counts[std_val] = std_counts.get(std_val, 0) + 1

    for std_val, count in std_counts.items():
        if count >= 3:
            violating_arms = [k for k, v in arm_stds.items() if v == std_val]
            violations.append(f"[V6 Distinct Std] {count} arms share identical std={std_val:.4f}: {violating_arms}")

    if violations:
        print(f"FAILED validation with {len(violations)} violations:")
        for v in violations:
            print(f"  - {v}")
        return False

    print("PASS: Mechanical validation succeeded with 0 violations.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_results_artifact.py <results_json_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    success = validate_results_file(filepath)
    if not success:
        sys.exit(1)
