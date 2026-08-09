"""
validate_results_artifact.py
============================

Mechanical validator for continual learning results JSON artifacts.
Exits with non-zero status if any rule V1-V9 is violated.
"""

import sys
import json
import numpy as np


def validate_results_json(filepath: str) -> bool:
    print(f"==================================================")
    print(f" Validating artifact: {filepath}")
    print(f"==================================================")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    violations = []

    # Collect arm stds for V6 check
    arm_stds = {}

    for arm_name, arm_content in data.items():
        # Handle both top-level arm structures and sel/fre nested structures
        sub_items = []
        if isinstance(arm_content, dict) and ("sel" in arm_content or "fre" in arm_content):
            for split_name in ["sel", "fre"]:
                if split_name in arm_content:
                    sub_items.append((f"{arm_name}[{split_name}]", arm_content[split_name]))
        elif isinstance(arm_content, dict) and "a_t_raw" in arm_content:
            sub_items.append((arm_name, arm_content))
        else:
            violations.append(f"[{arm_name}] Malformed arm structure in JSON.")
            continue

        for label, record_dict in sub_items:
            a_t_raw = record_dict.get("a_t_raw", [])
            a_t_mean = record_dict.get("a_t_mean", None)
            a_t_std = record_dict.get("a_t_std", None)
            la_mean = record_dict.get("la_mean", None)
            bwt_mean = record_dict.get("bwt_mean", None)
            cache_interf_mean = record_dict.get("cache_interference_mean", None)

            # V8: Keys check
            if not isinstance(a_t_raw, list) or len(a_t_raw) == 0:
                violations.append(f"[{label}] V8 FAIL: a_t_raw is empty or not a list.")
            else:
                keys_seen = set()
                for r in a_t_raw:
                    if not isinstance(r, dict) or "shuffle" not in r or "seed" not in r or "a_t" not in r:
                        violations.append(f"[{label}] V8 FAIL: record in a_t_raw missing shuffle/seed/a_t keys.")
                        break
                    key = (r["shuffle"], r["seed"])
                    if key in keys_seen:
                        violations.append(f"[{label}] V8 FAIL: Duplicate (shuffle, seed) key {key}.")
                    keys_seen.add(key)
                if len(keys_seen) != len(a_t_raw):
                    violations.append(f"[{label}] V8 FAIL: Non-unique keys in a_t_raw.")

            # V9: Dtype check
            for idx, r in enumerate(a_t_raw):
                val = r.get("a_t", None)
                if not isinstance(val, float):
                    violations.append(f"[{label}] V9 FAIL: record {idx} a_t={val} is type {type(val).__name__}, expected float.")
                elif val < 0.0 or val > 1.0:
                    violations.append(f"[{label}] V9 FAIL: record {idx} a_t={val} is outside [0, 1].")

            # V1: On-grid check (each a_t must be a multiple of 1/400 = 0.0025)
            for idx, r in enumerate(a_t_raw):
                val = r.get("a_t", 0.0)
                if isinstance(val, float):
                    grid_err = abs(val * 400.0 - round(val * 400.0))
                    if grid_err > 1e-7:
                        violations.append(f"[{label}] V1 FAIL: a_t={val:.6f} at record {idx} off 1/400 grid (error={grid_err:.8f}).")

            # Extract raw values for std/mean checks
            raw_vals = [r["a_t"] for r in a_t_raw if isinstance(r.get("a_t"), (int, float))]

            # V2: Std check
            if a_t_std is not None and len(raw_vals) > 0:
                calc_std = float(np.std(raw_vals))
                if abs(calc_std - float(a_t_std)) > 1e-7:
                    violations.append(f"[{label}] V2 FAIL: reported a_t_std={a_t_std} != calculated std={calc_std:.8f}.")
                arm_stds[label] = round(float(a_t_std), 4)

            # V3: Mean check
            if a_t_mean is not None and len(raw_vals) > 0:
                calc_mean = float(np.mean(raw_vals))
                if abs(calc_mean - float(a_t_mean)) > 1e-7:
                    violations.append(f"[{label}] V3 FAIL: reported a_t_mean={a_t_mean} != calculated mean={calc_mean:.8f}.")

            # V4: la_mean and bwt_mean grid check
            for metric_name, metric_val in [("la_mean", la_mean), ("bwt_mean", bwt_mean), ("cache_interference_mean", cache_interf_mean)]:
                if metric_val is not None:
                    grid_err = abs(float(metric_val) * 400.0 - round(float(metric_val) * 400.0))
                    if grid_err > 1e-7:
                        violations.append(f"[{label}] V4 FAIL: {metric_name}={metric_val:.6f} off 1/400 grid (error={grid_err:.8f}).")

            # V5: Ceiling check for freeze_after_base
            if "freeze_after_base" in label.lower() and len(raw_vals) > 0:
                max_at = max(raw_vals)
                if max_at > 0.50 + 1e-9:
                    violations.append(f"[{label}] V5 FAIL: max(a_t)={max_at:.4f} > structural ceiling 0.50.")

            # V7: Periodicity check (lags 1..5)
            if len(raw_vals) >= 10:
                for lag in range(1, 6):
                    is_periodic = True
                    for i in range(len(raw_vals) - lag):
                        if abs(raw_vals[i] - raw_vals[i + lag]) > 1e-9:
                            is_periodic = False
                            break
                    if is_periodic:
                        violations.append(f"[{label}] V7 FAIL: a_t sequence is strictly periodic with lag={lag}.")

    # V6: Distinct std check across arms
    seen_stds = {}
    for arm_lbl, std_val in arm_stds.items():
        if std_val in seen_stds:
            violations.append(f"V6 FAIL: Duplicate 4-decimal std={std_val:.4f} shared between {seen_stds[std_val]} and {arm_lbl}.")
        else:
            seen_stds[std_val] = arm_lbl

    if len(violations) > 0:
        print(f"FAILED -- Found {len(violations)} violations:")
        for v in violations:
            print(f"   * {v}")
        return False
    else:
        print("PASSED -- All rules V1-V9 satisfied.")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_results_artifact.py <results_json_path>")
        sys.exit(1)
    
    target_path = sys.argv[1]
    success = validate_results_json(target_path)
    if not success:
        sys.exit(1)
    sys.exit(0)
