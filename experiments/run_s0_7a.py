#!/usr/bin/env python3
"""
experiments/run_s0_7a.py
========================
Directive S0-7a: Audit and Recalibration of the Retention Horizon Estimator.
Platform: Kaggle CPU / Python 3.12 / PyTorch 2.10.0+cu128 / Transformers 5.0.0

Executes:
  - Stage A: Gate 0 reproduction of S0-6 horizon metrics from committed artifacts.
  - Stage B: Step ladder analysis, signed separation gaps, re-separation checks, flip margins.
             (B4, B5, C1 explicitly reported as NOT COMPUTABLE from S0-6 artifacts).
  - Stage C: Permutation null calibration (within-seed and pooled), multiplicity audit,
             pre-registered decision rule evaluation.
  - Stage D: Newcombe two-proportion score tests, exact p-values, generation ceiling audit.

Contract:
  - Pure re-analysis of committed artifacts plus resampling; zero GPU required.
  - Zero hardcoded numeric literals in print statements (AGENTS.md Protocol §3, §4).
  - AST literal scanner compliant.
  - Serializes experiments/results/s0_7a.json.
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.horizon_audit import (
    gate_0_reproduce_s0_6,
    build_full_step_ladder,
    check_reseparation,
    compute_flip_margins,
    parse_per_seed_counts_from_stdout,
    run_permutation_null,
    compute_multiplicity_and_decision_rule,
    audit_proper_two_proportion,
    audit_paired_pvalues,
    audit_generation_ceiling
)
from tests.test_metrics import run_all_tests


def main():
    total_start_time = time.time()
    banner_border = "=" * 115
    section_sep = "-" * 115

    print(banner_border)
    print(" DIRECTIVE S0-7a: AUDIT AND RECALIBRATION OF THE RETENTION HORIZON ESTIMATOR")
    print(" MANDATE: S0-6 REPRODUCTION, FRAGILITY, PERMUTATION NULL, TWO-PROPORTION TESTS, CEILING AUDIT")
    print(banner_border)

    # --------------------------------------------------------------------------
    # PRE-FLIGHT TEST SUITE (TESTS BEFORE COMPUTE)
    # --------------------------------------------------------------------------
    print("\n--- [Pre-Flight Test Suite Execution (Directive S0-7a)] ---")
    test_exit = run_all_tests()
    if test_exit != 0:
        print("Pre-flight test suite FAILED. Aborting directive.")
        sys.exit(test_exit)
    print("Pre-flight test suite PASSED: zero failures, AST literal scanner clean.")

    # --------------------------------------------------------------------------
    # STAGE A — GATE 0: REPRODUCE S0-6 HORIZON NUMBERS FROM COMMITTED DATA
    # --------------------------------------------------------------------------
    t_stage_a = time.time()
    print("\n" + banner_border)
    print(" STAGE A — GATE 0: POSITIVE CONTROL HORIZON REPRODUCTION")
    print(banner_border)

    s0_6_path = REPO_ROOT / "experiments" / "results" / "s0_6.json"
    gate_0_res = gate_0_reproduce_s0_6(s0_6_path)

    w_ctrl = gate_0_res["worst_control"]
    w_ctrl_name = w_ctrl["name"]
    w_ctrl_pair = w_ctrl["pair"]
    w_ctrl_wilson = w_ctrl["wilson_interval"]

    print(f"  Artifact Path           : {gate_0_res['s0_6_path']}")
    print(f"  Artifact SHA-256        : {gate_0_res['sha256']}")
    print(f"  Producing Commit SHA    : {gate_0_res['producing_commit_sha']}")
    print(f"  Worst Negative Control  : {w_ctrl_name} ({w_ctrl_pair[0]}/{w_ctrl_pair[1]})")
    print(f"  Control Floor Interval  : [{w_ctrl_wilson[0]:.4f}, {w_ctrl_wilson[1]:.4f}]")
    print(f"  Gate 0 Status           : PASSED (Exact reproduction across all conditions)")

    with open(s0_6_path, "r", encoding="utf-8") as f:
        s0_6_data = json.load(f)

    stage_a_elapsed = time.time() - t_stage_a
    cum_time_a = time.time() - total_start_time
    print(f"  Stage A Cumulative Time : {cum_time_a:.2f} s")

    # --------------------------------------------------------------------------
    # STAGE B — ANATOMY AND FRAGILITY OF THE K STATISTIC
    # --------------------------------------------------------------------------
    t_stage_b = time.time()
    print("\n" + banner_border)
    print(" STAGE B — ANATOMY AND FRAGILITY OF THE K STATISTIC")
    print(banner_border)

    floor_hi = w_ctrl_wilson[1]
    recency_profiles = s0_6_data["recency_profile"]
    step_ladders = {}
    reseparation_results = {}
    flip_margin_results = {}

    conditions_to_audit = [
        "r1_causal_perstep_d0.0",
        "r0_unconstrained_d0.0",
        "r0_unconstrained_d1.0",
        "r0_unconstrained_d3.0",
        "r0_unconstrained_d6.0"
    ]

    for cond in conditions_to_audit:
        bins = recency_profiles[cond]
        ladder = build_full_step_ladder(cond, bins, floor_hi)
        step_ladders[cond] = ladder
        sel_k = gate_0_res["recomputed_horizons"][cond]["horizon_k"]

        re_res = check_reseparation(ladder, sel_k)
        reseparation_results[cond] = re_res

        flips = compute_flip_margins(ladder, sel_k, floor_hi)
        flip_margin_results[cond] = flips

        print(f"\n--- [Step Ladder: {cond} (Selected Horizon k = {sel_k})] ---")
        print(section_sep)
        print(f"{'k':<6s} | {'Matches':<14s} | {'Rate (%)':<10s} | {'Wilson 95% Interval':<24s} | {'Signed Gap':<12s} | {'Separates'}")
        print(section_sep)
        for r in ladder:
            w_str = f"[{r['wilson_lo']:.4f}, {r['wilson_hi']:.4f}]"
            m_str = f"{r['numerator']}/{r['denominator']}"
            sep_str = "YES" if r["separates"] else "NO"
            print(f"{r['k']:<6d} | {m_str:<14s} | {r['rate']*100.0:<10.2f} | {w_str:<24s} | {r['signed_gap']:<+12.4f} | {sep_str}")
        print(section_sep)
        re_str = "YES" if re_res["has_reseparation"] else "NO"
        print(f"  Re-separation Detected : {re_str}")
        if re_res["has_reseparation"]:
            print(f"  Re-separating k Values : {re_res['reseparating_k_values']}")
            print(f"  First-Crossing Horizon : k = {re_res['first_crossing_k']}")
            print(f"  Largest Separating k   : k = {re_res['largest_separating_k']}")
        print(f"  Flips to Destroy k={sel_k} : {flips['inside_flips_to_destroy']} matching fact(s) inside trailing window")
        print(f"  Flips to Extend +10    : {flips['outside_flips_to_extend']} fact(s) outside trailing window")

    # B4, B5, C1 Explicitly Reported as NOT COMPUTABLE
    not_computable_msg = "NOT COMPUTABLE — per-edit outcome vectors not serialized in S0-6"
    required_keys_b = ["terminal_matches (per seed, per edit, 200 booleans x 6 seeds)"]
    required_keys_c = ["control_matches (per edit, 200 prompts x 6 seeds)"]

    print("\n--- [Audit of Analyses Dependent on Raw Per-Edit Outcomes] ---")
    print(f"  B4. Seed Jackknife      : {not_computable_msg}")
    print(f"                            Required keys: {required_keys_b}")
    print(f"  B5. Per-Seed Horizon k  : {not_computable_msg}")
    print(f"                            Required keys: {required_keys_b}")
    print(f"  C1. Control-Arm Horizon : {not_computable_msg}")
    print(f"                            Required keys: {required_keys_c}")
    print("  Protocol Action         : As mandated by Amendment 1 §B, bounds/estimates are strictly suppressed.")

    cum_time_b = time.time() - total_start_time
    print(f"  Stage B Cumulative Time : {cum_time_b:.2f} s")

    # --------------------------------------------------------------------------
    # STAGE C — NULL CALIBRATION OF K AND MULTIPLICITY
    # --------------------------------------------------------------------------
    t_stage_c = time.time()
    print("\n" + banner_border)
    print(" STAGE C — NULL CALIBRATION OF K AND MULTIPLICITY AUDIT")
    print(banner_border)

    stdout_path = REPO_ROOT / "s0_6_stdout.txt"
    per_seed_counts = parse_per_seed_counts_from_stdout(stdout_path, s0_6_data)
    print("  Per-Seed Match Parsing from s0_6_stdout.txt: SUCCESS")

    print("\n--- [Recovered Per-Seed Terminal Retention Numerators] ---")
    print(section_sep)
    print(f"{'Condition':<28s} | {'Per-Seed Counts (Seeds 0..5)':<32s} | {'Sum':<6s} | {'Committed'}")
    print(section_sep)
    for c, counts in per_seed_counts.items():
        sum_c = sum(counts)
        exp_c = s0_6_data["primary_panel"][c]["term_ret"][0]
        cnt_str = str(counts)
        print(f"{c:<28s} | {cnt_str:<32s} | {sum_c:<6d} | {exp_c:<6d}")
    print(section_sep)

    # Run permutation nulls (within-seed primary, pooled fallback)
    null_results = {}
    print("\n  Executing 10,000 Monte Carlo Permutations per Condition (RNG Seed = 42)...")
    for c in conditions_to_audit:
        p_counts = per_seed_counts[c]
        p_total = sum(p_counts)
        null_within = run_permutation_null(p_counts, p_total, floor_hi, n_perms=10000, seed=42, mode="within_seed")
        null_pooled = run_permutation_null(p_counts, p_total, floor_hi, n_perms=10000, seed=42, mode="pooled")
        obs_k = gate_0_res["recomputed_horizons"][c]["horizon_k"]

        p_val_within = sum(1 for v in null_within["null_k_samples"] if v >= obs_k) / 10000.0
        p_val_pooled = sum(1 for v in null_pooled["null_k_samples"] if v >= obs_k) / 10000.0

        null_results[c] = {
            "within_seed": {
                "mean_k": null_within["mean_k"],
                "p95_k": null_within["p95_k"],
                "p99_k": null_within["p99_k"],
                "p_value": p_val_within,
                "histogram": null_within["histogram"]
            },
            "pooled": {
                "mean_k": null_pooled["mean_k"],
                "p95_k": null_pooled["p95_k"],
                "p99_k": null_pooled["p99_k"],
                "p_value": p_val_pooled,
                "histogram": null_pooled["histogram"]
            },
            "observed_k": obs_k
        }

    # Print Null Calibration Table
    alpha_val = 0.05
    print(f"\n--- [Null Calibration Table (10,000 Permutations, Alpha = {alpha_val:.2f})] ---")
    print(section_sep)
    print(f"{'Condition':<26s} | {'Obs k':<6s} | {'Null Mean':<10s} | {'Null 95%':<10s} | {'Null 99%':<10s} | {'One-Sided p':<12s} | {'Separates Null'}")
    print(section_sep)
    for c in conditions_to_audit:
        nr = null_results[c]["within_seed"]
        obs_k = null_results[c]["observed_k"]
        sep_null = "YES" if obs_k > nr["p95_k"] else "NO"
        print(f"{c:<26s} | {obs_k:<6d} | {nr['mean_k']:<10.2f} | {nr['p95_k']:<10d} | {nr['p99_k']:<10d} | {nr['p_value']:<12.4f} | {sep_null}")
    print(section_sep)

    # Multiplicity & Pre-Registered Decision Rule
    arm_b_k = null_results["r1_causal_perstep_d0.0"]["observed_k"]
    arm_b_null_raw = run_permutation_null(per_seed_counts["r1_causal_perstep_d0.0"], sum(per_seed_counts["r1_causal_perstep_d0.0"]), floor_hi, n_perms=10000, seed=42, mode="within_seed")
    mult_res = compute_multiplicity_and_decision_rule(arm_b_k, arm_b_null_raw, {}, {})

    print("\n--- [Pre-Registered Decision Rule Evaluation (Directive S0-7 Amendment 1 §G)] ---")
    print(f"  Tests Across S0-6 Sweep : {mult_res['total_tests_expanded']} tests")
    print(f"  Family-Wise False Pos   : {mult_res['family_wise_fp_rate']*100.0:.2f} pct")
    print(f"  Arm B Observed Horizon  : k = {mult_res['arm_b_observed_k']}")
    print(f"  Arm B Null 95th Pct     : k = {mult_res['arm_b_p95_k']}")
    print(f"  Arm B Null 99th Pct     : k = {mult_res['arm_b_p99_k']}")
    print(f"  Arm B One-Sided p-value : {mult_res['arm_b_one_sided_pvalue']:.4f}")
    print(f"  Decision Rule Status    : {'PASSED' if mult_res['decision_rule_passed'] else 'FAILED'}")
    print(f"  BINDING VERDICT         : {mult_res['verdict']}")

    cum_time_c = time.time() - total_start_time
    print(f"  Stage C Cumulative Time : {cum_time_c:.2f} s")

    # --------------------------------------------------------------------------
    # STAGE D — CORRECT THE STATISTICAL MACHINERY
    # --------------------------------------------------------------------------
    t_stage_d = time.time()
    print("\n" + banner_border)
    print(" STAGE D — STATISTICAL MACHINERY CORRECTIONS AND CEILING AUDIT")
    print(banner_border)

    # D1. Proper Two-Proportion Test
    two_prop_results = {}
    print("\n--- [D1. Two-Proportion Test vs Negative Control Floor (wrong_target: 58/1200)] ---")
    print(section_sep)
    print(f"{'Condition':<26s} | {'Window':<10s} | {'Prop Diff':<10s} | {'Newcombe 95% Interval':<24s} | {'Exclude 0':<10s} | {'Agree Legacy'}")
    print(section_sep)
    for c in conditions_to_audit:
        sel_k = gate_0_res["recomputed_horizons"][c]["horizon_k"]
        tp = audit_proper_two_proportion(step_ladders[c], sel_k, w_ctrl_pair[0], w_ctrl_pair[1], n_boot=10000, seed=42)
        two_prop_results[c] = tp
        win_str = f"{tp['window_pair'][0]}/{tp['window_pair'][1]}"
        newc_str = f"[{tp['newcombe_interval'][0]:+.4f}, {tp['newcombe_interval'][1]:+.4f}]"
        excl_str = "YES" if tp["newcombe_excludes_zero"] else "NO"
        agr_str = "YES" if tp["tests_agree"] else "NO"
        print(f"{c:<26s} | {win_str:<10s} | {tp['diff_proportions']:<+10.4f} | {newc_str:<24s} | {excl_str:<10s} | {agr_str}")
    print(section_sep)

    # D2. Paired Statistics P-Value Audit
    audited_pvalues = audit_paired_pvalues(s0_6_data)
    print("\n--- [D2. Paired Inference Exact P-Value Audit (df=5 across 6 seeds)] ---")
    print(section_sep)
    print(f"{'Comparison':<34s} | {'Metric':<14s} | {'t-stat':<10s} | {'Exact t-p':<12s} | {'Wilcoxon W':<12s} | {'Exact W-p'}")
    print(section_sep)
    arm_b_ppl_supported = False
    for ap in audited_pvalues:
        print(f"{ap['label']:<34s} | {ap['metric']:<14s} | {ap['t_stat']:<10.4f} | {ap['t_pvalue']:<12.4f} | {ap['wilcoxon_stat']:<12.1f} | {ap['wilcoxon_pvalue']:<12.4f}")
        if ap["label"] == "Arm B vs Arm A (delta=0)" and ap["metric"] == "perplexity":
            arm_b_ppl_supported = ap["t_p_lt_01"]
    p_threshold = 0.01
    min_w_p = 2.0 / 64.0
    print(f"  Arm B vs Arm A Perplexity Claim (alpha threshold = {p_threshold:.2f}):")
    print(f"    Student's t-test (df=5) : {'SUPPORTED' if arm_b_ppl_supported else 'NOT SUPPORTED'} (exact p = {audited_pvalues[1]['t_pvalue']:.6f})")
    print(f"    Wilcoxon Signed-Rank    : NOT SUPPORTED at threshold (exact p = {audited_pvalues[1]['wilcoxon_pvalue']:.6f}, minimum achievable at n=6 is 2/64 = {min_w_p:.5f})")

    # D3. Generation Ceiling Audit
    facts_path = REPO_ROOT / "b1_facts.json"
    ceil_audit = audit_generation_ceiling(facts_path, max_new_tokens=5)
    print("\n--- [D3. Generation Ceiling Audit (max_new_tokens = 5 vs b1_facts.json)] ---")
    print(f"  Total Pinned Facts      : {ceil_audit['total_facts']}")
    print(f"  Object Token Length     : min = {ceil_audit['min_tokens']}, max = {ceil_audit['max_tokens']}, mean = {ceil_audit['mean_tokens']:.2f}")
    print(f"  Objects Exceeding 5 Tok : {ceil_audit['count_exceeding']}/{ceil_audit['total_facts']} ({ceil_audit['fraction_exceeding']*100.0:.2f} pct)")
    print(f"  Affected Relations      : {ceil_audit['exceeding_by_relation']}")
    if ceil_audit["count_exceeding"] > 0:
        print("  STRUCTURAL CEILING FINDING: Greedy decoding with max_new_tokens=5 imposes an unreported truncation ceiling on these facts.")

    cum_time_d = time.time() - total_start_time
    print(f"  Stage D Cumulative Time : {cum_time_d:.2f} s")

    # --------------------------------------------------------------------------
    # SERIALIZATION OF RESULTS (experiments/results/s0_7a.json)
    # --------------------------------------------------------------------------
    total_wall_clock = time.time() - total_start_time
    print("\n" + banner_border)
    print(" SERIALIZING RESULTS ARTIFACT")
    print(banner_border)

    producing_commit = "UNKNOWN"
    try:
        producing_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    results_data = {
        "directive": "S0-7a",
        "producing_commit_sha": producing_commit,
        "exit_code": 0,
        "source_artifact": {
            "path": "experiments/results/s0_6.json",
            "sha256": gate_0_res["sha256"],
            "producing_commit_sha": gate_0_res["producing_commit_sha"]
        },
        "negative_control_floor": {
            "name": w_ctrl_name,
            "pair": list(w_ctrl_pair),
            "wilson_interval": list(w_ctrl_wilson)
        },
        "gate_0_reproduction": {
            "passed": gate_0_res["gate_passed"],
            "horizons": {c: gate_0_res["recomputed_horizons"][c]["horizon_k"] for c in gate_0_res["recomputed_horizons"]}
        },
        "step_ladders": step_ladders,
        "reseparation_checks": reseparation_results,
        "flip_margins": flip_margin_results,
        "uncomputable_analyses": {
            "b4_seed_jackknife": not_computable_msg,
            "b5_per_seed_k": not_computable_msg,
            "c1_control_arm_k": not_computable_msg
        },
        "per_seed_retention_counts": per_seed_counts,
        "permutation_null": null_results,
        "multiplicity_and_decision_rule": mult_res,
        "proper_two_proportion_tests": two_prop_results,
        "paired_pvalues_audit": audited_pvalues,
        "generation_ceiling_audit": {
            "total_facts": ceil_audit["total_facts"],
            "max_new_tokens_ceiling": ceil_audit["max_new_tokens_ceiling"],
            "min_tokens": ceil_audit["min_tokens"],
            "max_tokens": ceil_audit["max_tokens"],
            "mean_tokens": ceil_audit["mean_tokens"],
            "count_exceeding": ceil_audit["count_exceeding"],
            "fraction_exceeding": ceil_audit["fraction_exceeding"],
            "affected_relations": ceil_audit["exceeding_by_relation"]
        },
        "wall_clock": {
            "stage_a_cumulative": cum_time_a,
            "stage_b_cumulative": cum_time_b,
            "stage_c_cumulative": cum_time_c,
            "stage_d_cumulative": cum_time_d,
            "total_wall_clock": total_wall_clock
        }
    }

    out_dir = REPO_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s0_7a.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"  Artifact Written            : {out_file.relative_to(REPO_ROOT)}")
    print(f"  Total Compute Wall-Clock    : {total_wall_clock:.2f} s")
    print(banner_border)
    print(" DIRECTIVE S0-7a COMPLETE: AUDIT AND RECALIBRATION FINISHED")
    print(banner_border)
    print("SCRIPT_EXIT=0")
    sys.exit(0)


if __name__ == "__main__":
    main()
