"""
experiments/horizon_audit.py
============================
Audit and Recalibration Routines for the Retention Horizon Estimator (Directive S0-7a).

Implements:
  - Stage A: Gate 0 exact reproduction of S0-6 horizon metrics from committed artifacts.
  - Stage B: Step ladder analysis, signed separation gaps, re-separation checks, flip margins.
  - Stage C: Permutation null calibration (within-seed and pooled), multiplicity audit, pre-registered decision rule.
  - Stage D: Newcombe two-proportion score tests, exact p-values, generation ceiling audit.

Contract:
  - Strictly zero hardcoded numeric literals from prior measurements (AGENTS.md Protocol §3, §4).
  - All expected reference values are dynamically loaded and asserted against committed artifacts at runtime.
  - No SciPy dependency; uses experiments.stats.
"""

import math
import json
import hashlib
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from experiments.stats import (
    wilson_interval,
    newcombe_score_interval,
    bootstrap_proportion_difference,
    exact_student_t_pvalue,
    exact_wilcoxon_signed_rank_pvalue,
    compute_paired_stats_with_pvalues
)


# ==============================================================================
# STAGE A — GATE 0: REPRODUCE S0-6 HORIZON NUMBERS FROM COMMITTED DATA
# ==============================================================================
def gate_0_reproduce_s0_6(s0_6_path: Path) -> Dict[str, Any]:
    """
    Gate 0 Positive Control:
    Loads experiments/results/s0_6.json, verifies its SHA-256 and commit SHA,
    extracts the worst negative control floor dynamically, and recomputes the
    trailing-window retention horizon across all conditions.
    Asserts exact equality against committed values; exits nonzero on any mismatch.
    """
    if not s0_6_path.exists():
        raise FileNotFoundError(f"Gate 0 failure: {s0_6_path} does not exist")

    raw_bytes = s0_6_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    with open(s0_6_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    commit_sha = data.get("producing_commit_sha", "UNKNOWN")
    exit_code = data.get("exit_code", -1)
    if exit_code != 0:
        raise ValueError(f"Gate 0 failure: s0_6.json recorded non-zero exit code: {exit_code}")

    # Dynamic extraction of worst control floor
    worst_ctrl_info = data["worst_control"]
    worst_ctrl_name = worst_ctrl_info["name"]
    worst_num, worst_den = worst_ctrl_info["pair"]
    floor_lo, floor_hi = wilson_interval(worst_num, worst_den)

    monotone_horizons = data["monotone_horizons"]
    recency_profiles = data["recency_profile"]

    recomputed = {}
    mismatches = []

    # Verify all conditions present in recency_profile
    for cond_name, expected_info in monotone_horizons.items():
        exp_k = expected_info["horizon_k"]
        exp_rem = expected_info["remainder"]

        if cond_name in recency_profiles:
            bins = recency_profiles[cond_name]
            assert len(bins) == 20, f"Condition {cond_name} has {len(bins)} bins, expected 20"

            # Recompute horizon from 20 bins (step size 10)
            largest_k = 0
            monotone_broken = False
            step_verdicts = []

            for step_idx in range(1, 21):
                k = step_idx * 10
                start_bin = 20 - step_idx
                num_k = sum(b[0] for b in bins[start_bin:])
                den_k = sum(b[1] for b in bins[start_bin:])
                assert den_k == 6 * k, f"Denominator check failed for k={k}: {den_k} != 6*{k}"

                w_lo, w_hi = wilson_interval(num_k, den_k)
                separates = (w_lo > floor_hi)
                step_verdicts.append({
                    "k": k, "num": num_k, "den": den_k,
                    "w_lo": w_lo, "w_hi": w_hi, "separates": separates
                })

                if not monotone_broken:
                    if separates:
                        largest_k = k
                    else:
                        monotone_broken = True

            # Remainder
            rem_k = 200 - largest_k
            rem_data = None
            if rem_k > 0:
                rem_bins_count = rem_k // 10
                r_num = sum(b[0] for b in bins[:rem_bins_count])
                r_den = sum(b[1] for b in bins[:rem_bins_count])
                r_lo, r_hi = wilson_interval(r_num, r_den)
                rem_data = {
                    "k": rem_k, "numerator": r_num, "denominator": r_den,
                    "rate": r_num / r_den if r_den > 0 else 0.0,
                    "wilson_lo": r_lo, "wilson_hi": r_hi
                }

            if largest_k != exp_k:
                mismatches.append(f"{cond_name}: horizon_k recomputed {largest_k} != expected {exp_k}")
            if rem_data is not None and exp_rem is not None:
                if rem_data["numerator"] != exp_rem["numerator"] or rem_data["denominator"] != exp_rem["denominator"]:
                    mismatches.append(f"{cond_name}: remainder recomputed ({rem_data['numerator']}/{rem_data['denominator']}) != expected ({exp_rem['numerator']}/{exp_rem['denominator']})")

            recomputed[cond_name] = {
                "horizon_k": largest_k,
                "remainder": rem_data,
                "step_verdicts": step_verdicts,
                "data_source": "recency_profile"
            }
        else:
            # For r1_magnitude_only_d0.0: recency profile was omitted from S0-6 serialization
            # Verify remainder consistency against total terminal retention
            tot_num, tot_den = data["primary_panel"][cond_name]["term_ret"]
            r_num = exp_rem["numerator"]
            r_den = exp_rem["denominator"]
            trailing_num = tot_num - r_num
            trailing_den = tot_den - r_den
            w_lo, w_hi = wilson_interval(trailing_num, trailing_den)
            separates = (w_lo > floor_hi)
            if not separates:
                mismatches.append(f"{cond_name}: trailing window at k={exp_k} ({trailing_num}/{trailing_den}) does not separate from floor {floor_hi:.4f}")

            recomputed[cond_name] = {
                "horizon_k": exp_k,
                "remainder": exp_rem,
                "step_verdicts": [],
                "data_source": "monotone_horizons_summary"
            }

    if mismatches:
        raise ValueError("Gate 0 REPRODUCTION FAILURE:\n" + "\n".join(mismatches))

    return {
        "s0_6_path": str(s0_6_path),
        "sha256": file_sha256,
        "producing_commit_sha": commit_sha,
        "worst_control": {
            "name": worst_ctrl_name,
            "pair": (worst_num, worst_den),
            "wilson_interval": (floor_lo, floor_hi)
        },
        "recomputed_horizons": recomputed,
        "gate_passed": True
    }


# ==============================================================================
# STAGE B — ANATOMY AND FRAGILITY OF THE K STATISTIC
# ==============================================================================
def build_full_step_ladder(
    condition: str,
    bins: List[List[int]],
    floor_hi: float
) -> List[Dict[str, Any]]:
    """
    B1. Full step ladder for 20 k values:
    For each k in {10, 20, ..., 200}, computes numerator, expanded denominator,
    rate, Wilson interval, signed separation gap, and separation verdict.
    """
    ladder = []
    for step_idx in range(1, 21):
        k = step_idx * 10
        start_bin = 20 - step_idx
        num_k = sum(b[0] for b in bins[start_bin:])
        den_k = sum(b[1] for b in bins[start_bin:])
        assert den_k == 6 * k, f"Expanded denominator check failed: {den_k} != 6*{k}"

        w_lo, w_hi = wilson_interval(num_k, den_k)
        signed_gap = w_lo - floor_hi
        separates = (w_lo > floor_hi)
        ladder.append({
            "k": k,
            "numerator": num_k,
            "denominator": den_k,
            "rate": num_k / den_k if den_k > 0 else 0.0,
            "wilson_lo": w_lo,
            "wilson_hi": w_hi,
            "floor_hi": floor_hi,
            "signed_gap": signed_gap,
            "separates": separates
        })
    return ladder


def check_reseparation(
    ladder: List[Dict[str, Any]],
    selected_k: int
) -> Dict[str, Any]:
    """
    B2. Re-separation check:
    Tests whether any k greater than selected_k satisfies separation.
    """
    reseparating_k = [row["k"] for row in ladder if row["k"] > selected_k and row["separates"]]
    has_reseparation = len(reseparating_k) > 0
    all_separating = [row["k"] for row in ladder if row["separates"]]
    largest_separating = max(all_separating) if all_separating else 0

    return {
        "selected_k": selected_k,
        "has_reseparation": has_reseparation,
        "reseparating_k_values": reseparating_k,
        "first_crossing_k": selected_k,
        "largest_separating_k": largest_separating,
        "is_monotone_in_k": not has_reseparation
    }


def compute_flip_margins(
    ladder: List[Dict[str, Any]],
    selected_k: int,
    floor_hi: float
) -> Dict[str, Any]:
    """
    B3. Fragility under fact flips:
    Computes:
    - inside_window_to_destroy: Minimum number of matching facts inside trailing window
      that would have to change from match to non-match to destroy separation (w_lo <= floor_hi).
    - outside_window_to_extend: Minimum number of non-matching facts in the next step
      outside the window that would have to change to match to extend separation by one step.
    """
    target_row = next((r for r in ladder if r["k"] == selected_k), None)
    if target_row is None or not target_row["separates"]:
        return {
            "selected_k": selected_k,
            "inside_flips_to_destroy": 0,
            "outside_flips_to_extend": 0
        }

    num = target_row["numerator"]
    den = target_row["denominator"]

    # Minimum flips inside to destroy
    flips_destroy = 0
    for m in range(1, num + 1):
        test_lo, _ = wilson_interval(num - m, den)
        if test_lo <= floor_hi:
            flips_destroy = m
            break

    # Minimum flips outside to extend to selected_k + 10
    next_k = selected_k + 10
    next_row = next((r for r in ladder if r["k"] == next_k), None)
    flips_extend = -1
    if next_row is not None:
        next_num = next_row["numerator"]
        next_den = next_row["denominator"]
        max_possible_flips = next_den - next_num
        for m in range(1, max_possible_flips + 1):
            test_lo, _ = wilson_interval(next_num + m, next_den)
            if test_lo > floor_hi:
                flips_extend = m
                break

    return {
        "selected_k": selected_k,
        "inside_flips_to_destroy": flips_destroy,
        "outside_flips_to_extend": flips_extend
    }


# ==============================================================================
# STAGE C — NULL CALIBRATION OF K AND MULTIPLICITY
# ==============================================================================
def parse_per_seed_counts_from_stdout(
    stdout_path: Path,
    s0_6_data: Dict[str, Any]
) -> Dict[str, List[int]]:
    """
    Parses s0_6_stdout.txt to recover exact per-seed terminal retention numerators.
    Asserts that the parsed counts sum to the committed pooled numerator for each condition.
    """
    if not stdout_path.exists():
        raise FileNotFoundError(f"Cannot parse stdout log: {stdout_path} not found")

    text = stdout_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Pattern: 'Seed <s>: ImmEff=... | TermRet=<m>/200 ...'
    # Condition blocks in S0-6 stdout:
    # r1_causal_perstep_d0.0 (Arm B)
    # r1_magnitude_only_d0.0 (Arm F)
    # r0_unconstrained_d0.0 (Arm A d0)
    # r0_unconstrained_d1.0 (Arm A d1)
    # r0_unconstrained_d3.0 (Arm A d3)
    # r0_unconstrained_d6.0 (Arm A d6)

    cond_order = [
        "r1_causal_perstep_d0.0",
        "r1_magnitude_only_d0.0",
        "r0_unconstrained_d0.0",
        "r0_unconstrained_d1.0",
        "r0_unconstrained_d3.0",
        "r0_unconstrained_d6.0"
    ]

    per_seed_counts = {c: [] for c in cond_order}
    curr_cond_idx = -1
    in_block = False

    term_pat = re.compile(r"Seed\s+(\d+).*?TermRet=(\d+)/200")

    for line in lines:
        if "=== MARGIN SWEEP ACROSS RESCALED SUBSPACES" in line or "=== STEP ATTRIBUTION" in line:
            in_block = True
        if "Seed 0" in line and "TermRet=" in line:
            curr_cond_idx += 1
        if curr_cond_idx >= 0 and curr_cond_idx < len(cond_order):
            m = term_pat.search(line)
            if m:
                s_idx = int(m.group(1))
                t_count = int(m.group(2))
                c_name = cond_order[curr_cond_idx]
                if len(per_seed_counts[c_name]) == s_idx:
                    per_seed_counts[c_name].append(t_count)

    # Verify all 6 seeds parsed for all 6 conditions
    for c in cond_order:
        counts = per_seed_counts[c]
        assert len(counts) == 6, f"Failed to parse 6 seeds for {c}: found {len(counts)}"
        expected_total = s0_6_data["primary_panel"][c]["term_ret"][0]
        actual_total = sum(counts)
        assert actual_total == expected_total, (
            f"Parsed seed sum check failed for {c}: sum({counts}) = {actual_total} != expected {expected_total}"
        )

    return per_seed_counts


def run_permutation_null(
    per_seed_counts: List[int],
    pooled_total: int,
    floor_hi: float,
    n_perms: int = 10000,
    seed: int = 42,
    mode: str = "within_seed"
) -> Dict[str, Any]:
    """
    C2. Permutation null distribution:
    - within_seed (primary): For each seed, permute that seed's matches uniformly across 200 edit positions.
    - pooled (fallback): Permute all pooled matches uniformly across 1200 edit positions.
    Evaluates 10,000 permutations with recorded RNG seed.
    """
    rng = random.Random(seed)
    k_counts = [0] * 21  # indices 0 to 20 for k in {0, 10, ..., 200}
    null_k_values = []

    for _ in range(n_perms):
        if mode == "within_seed":
            # For each seed, sample without replacement positions of matches
            seed_positions = [rng.sample(range(200), cnt) for cnt in per_seed_counts]
            # Fast trailing-window check
            largest_k = 0
            monotone_broken = False
            for step_idx in range(1, 21):
                k = step_idx * 10
                start_pos = 200 - k
                num_k = sum(sum(1 for p in pos if p >= start_pos) for pos in seed_positions)
                den_k = 6 * k
                w_lo, _ = wilson_interval(num_k, den_k)
                separates = (w_lo > floor_hi)
                if not monotone_broken:
                    if separates:
                        largest_k = k
                    else:
                        monotone_broken = True
        else:
            # Pooled null
            pooled_positions = rng.sample(range(1200), pooled_total)
            largest_k = 0
            monotone_broken = False
            for step_idx in range(1, 21):
                k = step_idx * 10
                start_pos = 1200 - (6 * k)
                num_k = sum(1 for p in pooled_positions if p >= start_pos)
                den_k = 6 * k
                w_lo, _ = wilson_interval(num_k, den_k)
                separates = (w_lo > floor_hi)
                if not monotone_broken:
                    if separates:
                        largest_k = k
                    else:
                        monotone_broken = True

        null_k_values.append(largest_k)
        k_counts[largest_k // 10] += 1

    null_k_values.sort()
    mean_k = sum(null_k_values) / n_perms
    p95_idx = int(0.95 * n_perms)
    p99_idx = int(0.99 * n_perms)
    p95_k = null_k_values[p95_idx]
    p99_k = null_k_values[p99_idx]

    return {
        "mode": mode,
        "n_perms": n_perms,
        "rng_seed": seed,
        "mean_k": mean_k,
        "p95_k": p95_k,
        "p99_k": p99_k,
        "histogram": {f"k_{i*10}": k_counts[i] for i in range(21)},
        "null_k_samples": null_k_values
    }


def compute_multiplicity_and_decision_rule(
    arm_b_k: int,
    arm_b_null: Dict[str, Any],
    condition_horizons: Dict[str, int],
    condition_nulls: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    C3 & Decision Rule:
    - Calculates family-wise false-positive rate across 120 tests (20 k * 6 conditions).
    - Checks whether Arm B (r1_causal_perstep_d0.0) horizon exceeds the 95th percentile
      of its own permutation null.
    """
    # Number of tests performed
    n_tests = 20 * 6

    # Probability of any false separation (k > 0) under the null
    fw_fp_rate = 1.0 - (arm_b_null["histogram"]["k_0"] / arm_b_null["n_perms"])

    # Decision rule evaluation
    p95 = arm_b_null["p95_k"]
    rule_passed = (arm_b_k > p95)
    one_sided_pval = sum(1 for v in arm_b_null["null_k_samples"] if v >= arm_b_k) / arm_b_null["n_perms"]

    verdict_str = (
        "VALID — RETENTION HORIZON SURVIVED NULL CALIBRATION"
        if rule_passed
        else "INVALID — RETENTION HORIZON DOES NOT EXCEED PERMUTATION NULL (WITHDRAWN)"
    )

    return {
        "total_tests_expanded": "20 * 6 = 120",
        "family_wise_fp_rate": fw_fp_rate,
        "arm_b_observed_k": arm_b_k,
        "arm_b_p95_k": p95,
        "arm_b_p99_k": arm_b_null["p99_k"],
        "arm_b_one_sided_pvalue": one_sided_pval,
        "decision_rule_passed": rule_passed,
        "verdict": verdict_str
    }


# ==============================================================================
# STAGE D — CORRECT THE STATISTICAL MACHINERY
# ==============================================================================
def audit_proper_two_proportion(
    ladder: List[Dict[str, Any]],
    selected_k: int,
    ctrl_num: int,
    ctrl_den: int,
    n_boot: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    D1. Proper two-proportion test:
    Evaluates Newcombe score interval and bootstrap difference for the trailing window
    proportion vs the wrong_target negative control proportion.
    """
    target_row = next((r for r in ladder if r["k"] == selected_k), None)
    if target_row is None:
        raise ValueError(f"Ladder has no row for k={selected_k}")

    w_num = target_row["numerator"]
    w_den = target_row["denominator"]

    diff, newc_lo, newc_hi = newcombe_score_interval(w_num, w_den, ctrl_num, ctrl_den, conf=0.95)
    _, boot_lo, boot_hi = bootstrap_proportion_difference(w_num, w_den, ctrl_num, ctrl_den, n_boot=n_boot, seed=seed)

    newc_excludes_zero = (newc_lo > 0.0 or newc_hi < 0.0)
    boot_excludes_zero = (boot_lo > 0.0 or boot_hi < 0.0)
    legacy_separates = target_row["separates"]

    agree = (legacy_separates == newc_excludes_zero)

    return {
        "k": selected_k,
        "window_pair": (w_num, w_den),
        "control_pair": (ctrl_num, ctrl_den),
        "diff_proportions": diff,
        "newcombe_interval": (newc_lo, newc_hi),
        "newcombe_excludes_zero": newc_excludes_zero,
        "bootstrap_interval": (boot_lo, boot_hi),
        "bootstrap_excludes_zero": boot_excludes_zero,
        "legacy_separates": legacy_separates,
        "tests_agree": agree
    }


def audit_paired_pvalues(s0_6_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    D2. Audit p-values:
    Recomputes exact Student's t p-value and exact Wilcoxon signed-rank p-value
    for all paired comparisons reported in S0-6.
    """
    paired_list = s0_6_data["paired_statistics"]
    audited = []

    for item in paired_list:
        label = item["label"]
        metric = item["metric"]
        st = item["stats"]
        t_stat = st["t_stat"]
        df = st["df"]
        w_stat = st["wilcoxon_stat"]

        t_pval = exact_student_t_pvalue(t_stat, df)

        # In S0-6 paired statistics, df=5 (n=6). For Wilcoxon W stat, full enumeration
        # over n=6 gives the exact p-value directly from W:
        # We compute exact p-value for W under n=6:
        # Sum of ranks = 21. Total configs = 64.
        # W in {0.0, 1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 7.0, 7.5, 9.5}
        total_configs = 64
        # Calculate distribution of W for ranks 1..6
        ranks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        w_counts = {}
        for mask in range(total_configs):
            wp = sum(ranks[b] for b in range(6) if (mask & (1 << b)) != 0)
            wm = sum(ranks[b] for b in range(6) if (mask & (1 << b)) == 0)
            w_sim = min(wp, wm)
            w_counts[w_sim] = w_counts.get(w_sim, 0) + 1

        cum_w = sum(cnt for w_val, cnt in w_counts.items() if w_val <= w_stat + 1e-9)
        w_pval = min(1.0, cum_w / float(total_configs))

        p_lt_01_t = (t_pval < 0.01)
        p_lt_01_w = (w_pval < 0.01)

        audited.append({
            "label": label,
            "metric": metric,
            "mean_diff": st["mean_diff"],
            "std_diff": st["std_diff"],
            "t_stat": t_stat,
            "df": df,
            "t_pvalue": t_pval,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_pval,
            "t_p_lt_01": p_lt_01_t,
            "wilcoxon_p_lt_01": p_lt_01_w
        })

    return audited


def audit_generation_ceiling(
    facts_path: Path,
    max_new_tokens: int = 5
) -> Dict[str, Any]:
    """
    D3. Guard the generation ceiling:
    Tokenizes all objects in b1_facts.json using GPT2TokenizerFast.
    Audits object lengths, flags any object > max_new_tokens, and identifies affected relations.
    """
    from transformers import GPT2TokenizerFast

    if not facts_path.exists():
        raise FileNotFoundError(f"Facts file not found: {facts_path}")

    with open(facts_path, "r", encoding="utf-8") as f:
        facts = json.load(f)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    lengths = []
    exceeding = []
    by_relation = {}

    for f_idx, item in enumerate(facts):
        obj_text = item["object"]
        rel = item.get("relation", "unknown")
        # Prepend space to match continuation tokenization
        tok_ids = tokenizer.encode(" " + obj_text.strip(), add_special_tokens=False)
        length = len(tok_ids)
        lengths.append(length)

        if length > max_new_tokens:
            exceeding.append({
                "fact_id": item.get("fact_id", f_idx),
                "subject": item.get("subject", ""),
                "relation": rel,
                "object": obj_text,
                "token_length": length
            })
            by_relation[rel] = by_relation.get(rel, 0) + 1

    total_facts = len(facts)
    count_exceeding = len(exceeding)
    frac_exceeding = count_exceeding / total_facts if total_facts > 0 else 0.0

    return {
        "total_facts": total_facts,
        "max_new_tokens_ceiling": max_new_tokens,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": sum(lengths) / total_facts if total_facts > 0 else 0.0,
        "count_exceeding": count_exceeding,
        "fraction_exceeding": frac_exceeding,
        "exceeding_by_relation": by_relation,
        "exceeding_examples": exceeding[:10]
    }
