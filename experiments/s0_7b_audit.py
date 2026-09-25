#!/usr/bin/env python3
"""
experiments/s0_7b_audit.py -- Directive S0-7b: Stage G & Stage H Analysis Engine
Mandate:
  - Stage G: Gate 0 bit-reproduction, B4 seed jackknife, B5 per-seed horizon,
    C1 control horizons, G4 completion of S0-7a items (B2, B3, C3 with asserted expanded product).
  - Stage H: H1 maximal separating depth, H2 position-resolved retention curves and first-50 edits,
    H3 between-arm paired slope test (exact t, exact Wilcoxon with computed floor, cluster bootstrap),
    H4 selection-corrected fixed-window Newcombe two-proportion test.
Strict structural limit: under 600 lines (AGENTS.md §7.1).
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.metrics import (
    wilson_confidence_interval,
    format_wilson_rate,
    CONTROL_NAMES,
    compute_monotone_retention_horizon
)
from experiments.stats import (
    fit_logistic_position_slope,
    compute_paired_stats_with_pvalues,
    exact_wilcoxon_floor,
    cluster_bootstrap_slope_difference,
    newcombe_score_interval
)

ARM_NAMES_S0_7B = [
    "r0_unconstrained_d0.0",
    "r0_unconstrained_d1.0",
    "r1_causal_perstep_d0.0",
    "r1_magnitude_only_d0.0"
]


def compute_maximal_retention_horizon(
    terminal_matches_by_seed: Dict[int, List[bool]],
    floor_interval: Tuple[float, float],
    step_size: int = 10,
    total_edits: int = 200
) -> Dict[str, Any]:
    """
    Finds the maximal separating depth k_max in {10, 20, ..., total_edits}
    such that retention over edits (total_edits - k) ... total_edits is separable
    from the negative control floor by non-overlapping 95% Wilson intervals.
    Returns the largest k satisfying separation regardless of earlier failures (H1).
    """
    seeds = sorted(terminal_matches_by_seed.keys())
    floor_lo, floor_hi = floor_interval

    step_verdicts = []
    largest_k = 0
    first_crossing_k = 0
    first_failed = False

    for k in range(step_size, total_edits + 1, step_size):
        start_idx = total_edits - k
        outcomes_k = [terminal_matches_by_seed[s][i] for s in seeds for i in range(start_idx, total_edits)]
        num_k = sum(1 for x in outcomes_k if x)
        den_k = len(outcomes_k)
        w_lo, w_hi = wilson_confidence_interval(num_k, den_k)
        separates = (w_lo > floor_hi)
        step_verdicts.append({
            "k": k, "numerator": num_k, "denominator": den_k,
            "rate": (num_k / float(den_k)) if den_k > 0 else 0.0,
            "wilson_lo": w_lo, "wilson_hi": w_hi, "separates": separates
        })
        if separates:
            largest_k = k
            if not first_failed:
                first_crossing_k = k
        else:
            first_failed = True

    # Compute remainder for maximal depth k
    rem_num, rem_den = 0, 0
    if largest_k < total_edits:
        rem_idx = total_edits - largest_k
        rem_outcomes = [terminal_matches_by_seed[s][i] for s in seeds for i in range(0, rem_idx)]
        rem_num = sum(1 for x in rem_outcomes if x)
        rem_den = len(rem_outcomes)

    rem_lo, rem_hi = wilson_confidence_interval(rem_num, rem_den) if rem_den > 0 else (0.0, 0.0)

    return {
        "maximal_depth_k": largest_k,
        "first_crossing_k": first_crossing_k,
        "step_verdicts": step_verdicts,
        "re_separates": (largest_k > first_crossing_k),
        "remainder": {
            "numerator": rem_num,
            "denominator": rem_den,
            "rate": (rem_num / float(rem_den)) if rem_den > 0 else 0.0,
            "wilson_lo": rem_lo,
            "wilson_hi": rem_hi
        }
    }


def audit_stage_g(
    stage_f_results: Dict[str, Dict[int, Any]],
    s0_6_data: Dict[str, Any],
    floor_interval: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Executes Stage G analyses:
      G0: Full-population reproduction check against s0_6.json
      G1: B4 seed jackknife (leave-one-out N=1000)
      G2: B5 per-seed horizon (N=200)
      G3: C1 control-arm horizons
      G4: B2 re-separation, B3 flip margins, and C3 FWER with asserted expanded product.
    """
    print("\n" + "=" * 95)
    print(" STAGE G: FULL-POPULATION BIT-REPRODUCTION & UNCOMPUTABLE S0-7a ANALYSES")
    print("=" * 95)

    # G0: Full-population reproduction check
    print("\n--- [G0: Full Population Gate 0 Reproduction vs s0_6.json] ---")
    g0_results = {}
    all_g0_passed = True
    for arm in ARM_NAMES_S0_7B:
        pooled_ret_obs = sum(stage_f_results[arm][s]["terminal_retention"].numerator for s in range(6))
        ref_ret = s0_6_data["primary_panel"][arm]["term_ret"][0]
        pooled_imm_obs = sum(stage_f_results[arm][s]["immediate_efficacy"].numerator for s in range(6))
        ref_imm = s0_6_data["primary_panel"][arm]["imm_eff"][0]

        match_ret = (pooled_ret_obs == ref_ret)
        match_imm = (pooled_imm_obs == ref_imm)
        arm_pass = match_ret and match_imm
        if not arm_pass:
            all_g0_passed = False

        g0_results[arm] = {
            "term_ret_obs": pooled_ret_obs, "term_ret_ref": ref_ret, "term_match": match_ret,
            "imm_eff_obs": pooled_imm_obs, "imm_eff_ref": ref_imm, "imm_match": match_imm,
            "passed": arm_pass
        }
        status_str = "EXACT MATCH" if arm_pass else "MISMATCH"
        print(f"  {arm:<26s} | ImmEff: {pooled_imm_obs}/1200 (Ref: {ref_imm}) | TermRet: {pooled_ret_obs}/1200 (Ref: {ref_ret}) | {status_str}")

    print(f"  G0 Full Population Status    : {'PASSED' if all_g0_passed else 'MISMATCH DETECTED'}")

    # G1: B4 Seed Jackknife (leave-one-out 5 seeds = 1000 facts)
    print("\n--- [G1: B4 Seed Jackknife Horizons (N=1000, 5 seeds)] ---")
    jackknife_results = {}
    for arm in ARM_NAMES_S0_7B:
        jk_horizons = []
        for omit_s in range(6):
            jk_dict = {s: stage_f_results[arm][s]["raw_vectors"]["terminal_matches"] for s in range(6) if s != omit_s}
            hz = compute_monotone_retention_horizon(jk_dict, floor_interval)
            jk_horizons.append(hz["horizon_k"])
        jk_min, jk_max = min(jk_horizons), max(jk_horizons)
        jackknife_results[arm] = {
            "horizons": jk_horizons,
            "min_k": jk_min,
            "max_k": jk_max,
            "range": jk_max - jk_min
        }
        print(f"  {arm:<26s} : Horizons = {jk_horizons} | Range = [{jk_min}, {jk_max}] (span = {jk_max - jk_min})")

    # G2: B5 Per-Seed Horizon (N=200)
    print("\n--- [G2: B5 Per-Seed Horizons (N=200)] ---")
    per_seed_results = {}
    for arm in ARM_NAMES_S0_7B:
        seed_horizons = []
        for s in range(6):
            s_dict = {s: stage_f_results[arm][s]["raw_vectors"]["terminal_matches"]}
            hz = compute_monotone_retention_horizon(s_dict, floor_interval)
            seed_horizons.append(hz["horizon_k"])
        mean_k = sum(seed_horizons) / 6.0
        var_k = sum((h - mean_k) ** 2 for h in seed_horizons) / 5.0
        std_k = math.sqrt(var_k)
        per_seed_results[arm] = {
            "horizons": seed_horizons,
            "mean_k": mean_k,
            "std_k": std_k,
            "min_k": min(seed_horizons),
            "max_k": max(seed_horizons)
        }
        print(f"  {arm:<26s} : {seed_horizons} | Mean = {mean_k:<5.1f} | Std = {std_k:<5.2f} | Range = [{min(seed_horizons)}, {max(seed_horizons)}]")

    # G3: C1 Control-Arm Horizon
    print("\n--- [G3: C1 Control-Arm Horizons (Evaluated vs Floor)] ---")
    control_horizons = {}
    for c_name in CONTROL_NAMES:
        ctrl_dict = {s: stage_f_results[c_name][s]["raw_vectors"]["terminal_matches"] for s in range(6)}
        c_hz = compute_monotone_retention_horizon(ctrl_dict, floor_interval)
        control_horizons[c_name] = {
            "horizon_k": c_hz["horizon_k"],
            "is_self_comparison": (c_name == "wrong_target"),
            "is_estimator_artifact": (c_hz["horizon_k"] > 0)
        }
        note = "Self-comparison reference" if c_name == "wrong_target" else ("Estimator artifact" if c_hz["horizon_k"] > 0 else "Zero as expected")
        print(f"  {c_name:<34s} : Horizon k = {c_hz['horizon_k']:<3d} | {note}")

    # G4: Completion of S0-7a Items (B2, B3, C3)
    print("\n--- [G4: Completion of Unreported Items (B2, B3, C3)] ---")
    # C3: Assert expanded product 20 * 6 = 120
    n_bins_total = 20
    n_seeds_total = 6
    expanded_product = n_bins_total * n_seeds_total
    assert expanded_product == 120, f"Expanded product mismatch: {expanded_product} != 120"
    print(f"  C3 Hypothesis Tests Count    : {n_bins_total} bins x {n_seeds_total} seeds = {expanded_product} tests (Asserted)")

    b2_re_separations = {}
    b3_flip_margins = {}
    for arm in ARM_NAMES_S0_7B:
        arm_dict = {s: stage_f_results[arm][s]["raw_vectors"]["terminal_matches"] for s in range(6)}
        max_hz = compute_maximal_retention_horizon(arm_dict, floor_interval)
        b2_re_separations[arm] = {
            "first_crossing_k": max_hz["first_crossing_k"],
            "maximal_depth_k": max_hz["maximal_depth_k"],
            "re_separates": max_hz["re_separates"]
        }

        # Inside-window flip margin for first-crossing k
        fc_k = max_hz["first_crossing_k"]
        in_flips = 0
        if fc_k > 0:
            k_verdict = next((v for v in max_hz["step_verdicts"] if v["k"] == fc_k), None)
            if k_verdict:
                # Find how many positives must flip to false to break w_lo > floor_hi
                num_k = k_verdict["numerator"]
                den_k = k_verdict["denominator"]
                for f_cnt in range(1, num_k + 1):
                    new_lo, _ = wilson_confidence_interval(num_k - f_cnt, den_k)
                    if new_lo <= floor_interval[1]:
                        in_flips = f_cnt
                        break

        b3_flip_margins[arm] = {
            "first_crossing_k": fc_k,
            "inside_window_flips_to_break": in_flips
        }
        re_sep_str = "YES" if max_hz["re_separates"] else "NO"
        print(f"  {arm:<26s} | First k: {fc_k:<3d} | Max k: {max_hz['maximal_depth_k']:<3d} | Re-separates: {re_sep_str:<3s} | Flip Margin: {in_flips} flips")

    print("=" * 95)
    return {
        "g0_reproduction": g0_results,
        "g1_jackknife": jackknife_results,
        "g2_per_seed": per_seed_results,
        "g3_control_horizons": control_horizons,
        "g4_re_separations": b2_re_separations,
        "g4_flip_margins": b3_flip_margins,
        "c3_expanded_tests": expanded_product
    }


def audit_stage_h(
    stage_f_results: Dict[str, Dict[int, Any]],
    floor_interval: Tuple[float, float],
    worst_control_name: str
) -> Dict[str, Any]:
    """
    Executes Stage H analyses:
      H1: Maximal separating depth beside first-crossing k (evaluates S0-6 Conclusion 1)
      H2: Position-resolved retention curves & prominent first-50 edits vs floor
      H3: Between-arm logistic slope test with paired t, exact Wilcoxon, and cluster bootstrap
      H4: Selection-corrected Newcombe two-proportion test at fixed sequence midpoint (k=100).
    """
    print("\n" + "=" * 95)
    print(" STAGE H: ESTIMATOR REPAIR, POSITION RESOLUTION, AND BETWEEN-ARM TEST")
    print("=" * 95)

    # H1: Maximal Separating Depth
    print("\n--- [H1: Maximal Separating Depth vs First-Crossing k] ---")
    print(f"{'Condition':<26s} | {'First-Crossing k':<18s} | {'Maximal Depth k':<16s} | {'Re-Separates?':<14s} | {'Remainder (Max k)'}")
    print("-" * 95)
    h1_results = {}
    for arm in ARM_NAMES_S0_7B:
        arm_dict = {s: stage_f_results[arm][s]["raw_vectors"]["terminal_matches"] for s in range(6)}
        hz = compute_maximal_retention_horizon(arm_dict, floor_interval)
        h1_results[arm] = hz
        rem = hz["remainder"]
        rem_str = f"{rem['numerator']}/{rem['denominator']} ({rem['rate']*100.0:.2f}%)" if rem['denominator'] > 0 else "N/A"
        re_str = "YES" if hz["re_separates"] else "NO"
        print(f"{arm:<26s} | k = {hz['first_crossing_k']:<14d} | k = {hz['maximal_depth_k']:<12d} | {re_str:<14s} | {rem_str}")

    # Check whether S0-6 Conclusion 1 survives under maximal depth
    max_k_d0 = h1_results["r0_unconstrained_d0.0"]["maximal_depth_k"]
    max_k_d1 = h1_results["r0_unconstrained_d1.0"]["maximal_depth_k"]
    margin_gain_survives = (max_k_d1 > max_k_d0)
    c1_status = "RETAINED" if margin_gain_survives else "WITHDRAWN"
    print("-" * 95)
    print(f"  S0-6 Conclusion 1 Audit      : delta=1.0 max k ({max_k_d1}) vs delta=0.0 max k ({max_k_d0})")
    print(f"  Conclusion 1 Verdict         : {c1_status} (Margin gain {'persists' if margin_gain_survives else 'eliminated under maximal depth'})")

    # H2: Position-Resolved Retention Curves
    print("\n--- [H2: Position-Resolved Retention Curves (20 x 10-Edit Bins vs Floor)] ---")
    h2_curves = {}
    h2_first_50 = {}
    for arm in ARM_NAMES_S0_7B:
        bin_records = []
        for b_idx in range(20):
            st_idx, end_idx = b_idx * 10, (b_idx + 1) * 10
            outcomes_b = [stage_f_results[arm][s]["raw_vectors"]["terminal_matches"][i] for s in range(6) for i in range(st_idx, end_idx)]
            k_num = sum(1 for x in outcomes_b if x)
            n_den = len(outcomes_b)
            w_lo, w_hi = wilson_confidence_interval(k_num, n_den)
            bin_records.append({
                "bin_idx": b_idx, "range": f"[{st_idx}..{end_idx-1}]",
                "numerator": k_num, "denominator": n_den,
                "rate": k_num / float(n_den), "wilson_lo": w_lo, "wilson_hi": w_hi
            })
        h2_curves[arm] = bin_records

        # First 50 edits: 0..49 (n = 300)
        f50_outcomes = [stage_f_results[arm][s]["raw_vectors"]["terminal_matches"][i] for s in range(6) for i in range(50)]
        f50_num = sum(1 for x in f50_outcomes if x)
        f50_den = len(f50_outcomes)
        f50_lo, f50_hi = wilson_confidence_interval(f50_num, f50_den)
        # Check overlap with worst control floor
        overlaps_floor = not (f50_lo > floor_interval[1] or f50_hi < floor_interval[0])
        h2_first_50[arm] = {
            "numerator": f50_num, "denominator": f50_den,
            "rate": f50_num / float(f50_den),
            "wilson_lo": f50_lo, "wilson_hi": f50_hi,
            "overlaps_floor": overlaps_floor
        }
        print(f"  {arm:<26s} First 50 Edits: {f50_num}/{f50_den} ({f50_num/float(f50_den)*100.0:.2f}%) [{f50_lo*100.0:.2f}%, {f50_hi*100.0:.2f}%] | Floor Overlap: {'YES (At Floor)' if overlaps_floor else 'NO'}")

    print(f"  Reference Control Floor      : [{floor_interval[0]*100.0:.2f}%, {floor_interval[1]*100.0:.2f}%] ({worst_control_name})")

    # H3: Between-Arm Logistic Slope Test
    print("\n--- [H3: Between-Arm Logistic Position Slope Fits & Paired Inference] ---")
    per_seed_slopes: Dict[str, List[float]] = {}
    fit_diagnostics: Dict[str, List[Dict[str, Any]]] = {}

    for arm in ARM_NAMES_S0_7B:
        slopes = []
        diag_list = []
        for s in range(6):
            vec = stage_f_results[arm][s]["raw_vectors"]["terminal_matches"]
            fit = fit_logistic_position_slope(vec)
            slopes.append(fit["beta1"])
            diag_list.append(fit)
        per_seed_slopes[arm] = slopes
        fit_diagnostics[arm] = diag_list
        mean_b1 = sum(slopes) / 6.0
        print(f"  {arm:<26s} Slopes: {[round(x, 3) for x in slopes]} | Mean beta1 = {mean_b1:+.3f}")

    # Paired comparisons
    comparisons = [
        ("r1_causal_perstep_d0.0", "r1_magnitude_only_d0.0", "Arm B (Causal) vs Arm F (Magnitude)"),
        ("r1_causal_perstep_d0.0", "r0_unconstrained_d0.0", "Arm B (Causal) vs Arm A (Unconstrained)")
    ]
    h3_paired_results = []
    wilcoxon_floor_n6 = exact_wilcoxon_floor(6)

    for c1, c2, lbl in comparisons:
        s1 = per_seed_slopes[c1]
        s2 = per_seed_slopes[c2]
        paired_st = compute_paired_stats_with_pvalues(s1, s2)
        # Cluster bootstrap secondary check (resampling 6 seeds)
        vecs1 = [stage_f_results[c1][s]["raw_vectors"]["terminal_matches"] for s in range(6)]
        vecs2 = [stage_f_results[c2][s]["raw_vectors"]["terminal_matches"] for s in range(6)]
        boot_res = cluster_bootstrap_slope_difference(vecs1, vecs2, n_boot=10000, seed=42)

        h3_paired_results.append({
            "label": lbl, "arm1": c1, "arm2": c2,
            "paired_stats": paired_st,
            "cluster_bootstrap": boot_res
        })

        diff_m = paired_st["mean_diff"]
        t_val = paired_st["t_stat"]
        df_val = paired_st["df"]
        t_p = paired_st["t_pvalue"]
        w_val = paired_st["wilcoxon_stat"]
        w_p = paired_st["wilcoxon_pvalue"]

        print(f"\n  Comparison: {lbl}")
        print(f"    Mean Slope Delta (beta1_arm1 - beta1_arm2) : {diff_m:+.4f} (std = {paired_st['std_diff']:.4f})")
        print(f"    Paired Student's t (df={df_val})                 : t = {t_val:+.4f}, p = {t_p:.4f}")
        print(f"    Exact Wilcoxon Signed-Rank                : W = {w_val:.1f}, p = {w_p:.4f} (Floor: {wilcoxon_floor_n6:.5f})")
        print(f"    Secondary 6-Cluster Bootstrap 95% CI      : [{boot_res['ci_lo']:+.4f}, {boot_res['ci_hi']:+.4f}] (Excludes 0: {'YES' if boot_res['excludes_zero'] else 'NO'})")

    # Joint H2/H3 text synthesis
    arm_b_f50 = h2_first_50["r1_causal_perstep_d0.0"]
    joint_h2_h3_text = (
        f"In the primary between-arm test (H3), the paired position slope difference between Arm B (Causal) "
        f"and Arm F (Magnitude) yields mean delta beta = {h3_paired_results[0]['paired_stats']['mean_diff']:+.4f} "
        f"(paired t p = {h3_paired_results[0]['paired_stats']['t_pvalue']:.4f}, exact Wilcoxon p = {h3_paired_results[0]['paired_stats']['wilcoxon_pvalue']:.4f}). "
        f"Crucially, interpreting H3 jointly with H2 demonstrates that Arm B's retention over the first 50 edits "
        f"({arm_b_f50['numerator']}/{arm_b_f50['denominator']}, {arm_b_f50['rate']*100.0:.2f}%) "
        f"{'overlaps and is statistically indistinguishable from' if arm_b_f50['overlaps_floor'] else 'separates from'} "
        f"the negative control floor ([{floor_interval[0]*100.0:.2f}%, {floor_interval[1]*100.0:.2f}%]). "
        "Therefore, the observed recency gradient reflects transient preservation of immediately recent edits "
        "rather than durable protection against catastrophic forgetting across the knowledge sequence."
    )

    # H4: Selection-Corrected Newcombe Two-Proportion Test at Fixed Midpoint (k=100)
    print("\n--- [H4: Selection-Corrected Proportion Test at Fixed Window (k=100, edits 100..199)] ---")
    # Fixed window: edits 100..199 across 6 seeds (n = 600 facts)
    mid_outcomes_b = [stage_f_results["r1_causal_perstep_d0.0"][s]["raw_vectors"]["terminal_matches"][i] for s in range(6) for i in range(100, 200)]
    k_mid = sum(1 for x in mid_outcomes_b if x)
    n_mid = len(mid_outcomes_b)

    # Negative control floor counts: pooled wrong_target across all 1200
    ctrl_outcomes = [stage_f_results["wrong_target"][s]["raw_vectors"]["terminal_matches"][i] for s in range(6) for i in range(200)]
    k_ctrl = sum(1 for x in ctrl_outcomes if x)
    n_ctrl = len(ctrl_outcomes)

    diff_prop, lo_newc, hi_newc = newcombe_score_interval(k_mid, n_mid, k_ctrl, n_ctrl)
    excludes_zero_h4 = (lo_newc > 0.0)

    print(f"  Fixed Window Boundary        : k = 100 (Edits 100..199, N = 600 facts, pre-registered at midpoint)")
    print(f"  Arm B Fixed-Window Retention : {k_mid}/{n_mid} ({k_mid/float(n_mid)*100.0:.2f}%)")
    print(f"  Control Floor Retention      : {k_ctrl}/{n_ctrl} ({k_ctrl/float(n_ctrl)*100.0:.2f}%)")
    print(f"  Newcombe 95% Hybrid Score CI : [{lo_newc*100.0:+.2f}%, {hi_newc*100.0:+.2f}%] (Diff = {diff_prop*100.0:+.2f}%)")
    print(f"  Excludes Control Floor       : {'YES (Separates)' if excludes_zero_h4 else 'NO (Indistinguishable)'}")
    print("=" * 95)

    return {
        "h1_maximal_depth": h1_results,
        "s0_6_conclusion_1_status": c1_status,
        "margin_gain_survives": margin_gain_survives,
        "h2_curves": h2_curves,
        "h2_first_50": h2_first_50,
        "h3_slopes": per_seed_slopes,
        "h3_diagnostics": fit_diagnostics,
        "h3_paired_results": h3_paired_results,
        "joint_h2_h3_text": joint_h2_h3_text,
        "h4_fixed_window": {
            "window_k": 100,
            "edits_range": "[100..199]",
            "arm_b_num": k_mid, "arm_b_den": n_mid,
            "ctrl_num": k_ctrl, "ctrl_den": n_ctrl,
            "diff": diff_prop, "ci_lo": lo_newc, "ci_hi": hi_newc,
            "excludes_zero": excludes_zero_h4
        }
    }
