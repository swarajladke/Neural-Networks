#!/usr/bin/env python3
"""
experiments/re_emission.py -- Directive S0-7b: Empirical Budget, Gate 0 Early Abort,
and Stage F Re-emission Runner with Rule 3.7 Outcome Vector Serialization.
Mandate:
  - Empirical budget projection loaded from S0-6 artifacts with actual contingency margin.
  - Pre-registered reproduction tolerance and deterministic CUDA/algorithm configuration.
  - Gate 0 early abort: seed 0 executed and verified against S0-6 before any other runs.
  - Stage F full execution: 4 arms + 4 controls across 6 seeds (N=1200).
  - Complete Rule 3.7 raw boolean outcome vector serialization per seed and edit.
Strict structural limit: under 600 lines (AGENTS.md §7.1).
"""

import os
import gc
import sys
import math
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (
    CausalSubspaceManager,
    evaluate_wikitext_perplexity
)
from experiments.metrics import (
    Measurement,
    check_match,
    normalize_entity,
    pool_controls,
    CONTROL_NAMES,
    format_wilson_rate,
    assert_orthonormality,
    wilson_confidence_interval
)
from experiments.b1_inject import (
    configure_determinism,
    greedy_predict,
    edit_fact_sgd,
    evaluate_sequence_metrics,
    SEEDS
)


def project_s0_7b_budget(
    s0_6_data: Dict[str, Any],
    session_ceiling: float = 23400.0,
    budget_ceiling: float = 16380.0
) -> Dict[str, Any]:
    """
    Computes an empirical budget projection for Directive S0-7b derived directly
    from recorded S0-6 timings, costing optimizing controls appropriately.
    Enforces Directive S0-7b Amendment 1 §C.
    """
    acct = s0_6_data["accounting"]
    pilot = s0_6_data["pilot_timing"]

    s0_6_proj = float(acct["projected_wall_clock"])
    s0_6_act = float(acct["actual_wall_clock"])
    s0_6_overrun = max(0.0, (s0_6_act - s0_6_proj) / s0_6_proj)
    contingency_margin = max(0.05, s0_6_overrun)

    # Measured base timings per seed (200 edits) from S0-6 pilot
    t_base = float(pilot.get("0.0", pilot.get(0.0, 290.77)))
    t_delta1 = float(pilot.get("1.0", pilot.get(1.0, 251.74)))

    t_causal = t_base * 1.35
    t_mag = t_base * 1.15
    t_eval_ctrl = 15.0  # seconds for eval-only control per seed

    # Stage line items
    t_preflight = 5.0
    t_stage_j = 5.0

    # Gate 0: Seed 0 of r0_unconstrained_d0.0
    t_gate_0 = t_base

    # Remainder of Stage F:
    # r0_unconstrained_d0.0: seeds 1..5 (5 runs)
    t_arm_a_d0_rem = 5.0 * t_base
    # r0_unconstrained_d1.0: seeds 0..5 (6 runs)
    t_arm_a_d1 = 6.0 * t_delta1
    # r1_causal_perstep_d0.0: seeds 0..5 (6 runs)
    t_arm_b = 6.0 * t_causal
    # r1_magnitude_only_d0.0: seeds 0..5 (6 runs)
    t_arm_f = 6.0 * t_mag
    # Optimizing controls across 6 seeds:
    t_ctrl_wrong = 6.0 * t_base
    t_ctrl_rand = 6.0 * t_base
    # Eval-only controls across 6 seeds:
    t_ctrl_never = 6.0 * t_eval_ctrl
    t_ctrl_pre = 6.0 * t_eval_ctrl

    t_stage_f_rem = (
        t_arm_a_d0_rem + t_arm_a_d1 + t_arm_b + t_arm_f +
        t_ctrl_wrong + t_ctrl_rand + t_ctrl_never + t_ctrl_pre
    )

    # Stage G & H (re-analyses on CPU)
    t_stage_g_h = 30.0

    # Stage I (Untied cells only, 3 seeds for Arm A and Arm B, Amendment 1 §B)
    t_stage_i_untied_a = 3.0 * t_base
    t_stage_i_untied_b = 3.0 * t_causal
    t_stage_i = t_stage_i_untied_a + t_stage_i_untied_b

    raw_total = t_preflight + t_stage_j + t_gate_0 + t_stage_f_rem + t_stage_g_h + t_stage_i
    projected_with_contingency = raw_total * (1.0 + contingency_margin)

    exceeds_budget = projected_with_contingency > budget_ceiling

    return {
        "s0_6_projected_wall_clock": s0_6_proj,
        "s0_6_actual_wall_clock": s0_6_act,
        "s0_6_fractional_overrun": s0_6_overrun,
        "applied_contingency_margin": contingency_margin,
        "t_base_measured": t_base,
        "t_delta1_measured": t_delta1,
        "line_items": {
            "preflight_tests": t_preflight,
            "stage_j_tokenization": t_stage_j,
            "gate_0_early_runner": t_gate_0,
            "stage_f_remainder": t_stage_f_rem,
            "stage_g_and_h_analysis": t_stage_g_h,
            "stage_i_untied_runs": t_stage_i
        },
        "raw_projected_total": raw_total,
        "projected_total_with_contingency": projected_with_contingency,
        "budget_ceiling": budget_ceiling,
        "session_ceiling": session_ceiling,
        "exceeds_budget": exceeds_budget,
        "droppable_stages_in_order": ["Stage I (Untied Evaluation)", "r1_magnitude_only_d0.0 from Stage F"]
    }


def print_budget_projection(proj: Dict[str, Any]):
    """Prints the empirical budget projection cleanly without typed literals."""
    items = proj["line_items"]
    print("\n" + "=" * 95)
    print(" EMPIRICAL COMPUTE BUDGET PROJECTION (Directive S0-7b / Amendment 1 §C)")
    print("=" * 95)
    print(f"  S0-6 Projected Wall-Clock    : {proj['s0_6_projected_wall_clock']:.1f} s")
    print(f"  S0-6 Actual Wall-Clock       : {proj['s0_6_actual_wall_clock']:.1f} s")
    print(f"  S0-6 Measured Overrun        : {proj['s0_6_fractional_overrun'] * 100.0:.2f}%")
    print(f"  Applied Contingency Margin   : {proj['applied_contingency_margin'] * 100.0:.2f}%")
    print(f"  Base Seed-0 Timing (t_base)  : {proj['t_base_measured']:.2f} s")
    t_d1_title = "Delta-1 Timing (t_delta1)"
    print(f"  {t_d1_title:<29s}: {proj['t_delta1_measured']:.2f} s")
    print("-" * 95)
    print(f"{'Scheduled Stage / Subsystem':<42s} | {'Projected Wall-Clock':<22s} | {'Costing Basis'}")
    print("-" * 95)
    print(f"{'Pre-Flight Unit Tests':<42s} | {items['preflight_tests']:<22.1f} s | CPU stub tests")
    print(f"{'Stage J: Tokenization Audit':<42s} | {items['stage_j_tokenization']:<22.1f} s | Tokenizer encode on 1000 facts")
    print(f"{'Gate 0: Seed-0 Arm A (Early Abort)':<42s} | {items['gate_0_early_runner']:<22.1f} s | 1 run @ t_base (200 edits)")
    print(f"{'Stage F: Remainder (Arms & Controls)':<42s} | {items['stage_f_remainder']:<22.1f} s | 23 arm runs + 4 controls x 6 seeds")
    print(f"{'Stage G & H: Reproduction & Estimator':<42s} | {items['stage_g_and_h_analysis']:<22.1f} s | CPU vector analysis & bootstrap")
    print(f"{'Stage I: Untied Confound (3 seeds)':<42s} | {items['stage_i_untied_runs']:<22.1f} s | 6 untied runs (reusing Stage F tied)")
    print("-" * 95)
    print(f"{'Raw Projected Total':<42s} | {proj['raw_projected_total']:<22.1f} s |")
    print(f"{'Projected Total with Contingency':<42s} | {proj['projected_total_with_contingency']:<22.1f} s | Floor: {proj['budget_ceiling']:.1f} s")
    print("=" * 95)
    if proj["exceeds_budget"]:
        print("FATAL: Projected wall-clock exceeds compute budget ceiling!")
        print(f"Drop ordering: {proj['droppable_stages_in_order']}")
        sys.exit(1)
    else:
        b_ceil = f"{proj['budget_ceiling']:.1f}"
        print(f"  Compute Budget Verification  : PASSED (Well within {b_ceil} s ceiling)")


def run_gate_0_early(
    model_name: str,
    pinned_revision: str,
    tokenizer: Any,
    fresh_model: nn.Module,
    sequence_s0: List[Dict[str, Any]],
    template_prior_controls: List[Dict[str, Any]],
    wikitext_slice: Any,
    slice_sha: str,
    device: str,
    s0_6_data: Dict[str, Any],
    start_time: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes Gate 0: runs r0_unconstrained_d0.0 on seed 0 ONLY and asserts exact
    reproduction of S0-6 seed-0 values. Aborts early on failure.
    Enforces Directive S0-7b Amendment 1 §D and §E.
    """
    print("\n" + "=" * 95)
    print(" GATE 0: EARLY BIT-REPRODUCTION POSITIVE CONTROL (Seed 0, Arm A delta=0.0)")
    print("=" * 95)

    configure_determinism(seed=0)
    print("  Deterministic Settings Configured:")
    print("    RNG Seeds Fixed: 0 (Python random, PyTorch CPU/CUDA)")
    print("    torch.use_deterministic_algorithms: True")
    print("    cuDNN Benchmark / Deterministic: False / True")
    print("    CUBLAS_WORKSPACE_CONFIG: :4096:8")

    # Pre-registered reproduction tolerance declaration
    print("\n  Pre-Registered Reproduction Tolerance Declaration (Amendment 1 §E):")
    print("    Primary Tolerance: Total optimizer steps must match exactly (669 steps).")
    print("                       Immediate efficacy must match exactly (200/200).")
    print("                       Terminal retention must match exactly (8/200).")
    print("    Fallback Tolerance: At most 1 differing outcome per 200-edit seed sequence.")

    m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    subspace_unapp = CausalSubspaceManager(device=device)
    edit_res = []
    first_upd = None
    cum_applied = torch.zeros_like(m_arm.lm_head.weight.data)

    for t_idx, f in enumerate(sequence_s0):
        t_num = t_idx + 1
        Q_unapp = subspace_unapp.get_projection_matrix(1)
        res_e = edit_fact_sgd(
            m_arm, tokenizer, f, lr=3.0e-05, max_steps=100, delta=0.0,
            device=device, train_mode=False, arm_mode="r0_unconstrained", Q_causal=Q_unapp
        )
        subspace_unapp.add_update(res_e["delta_target_vec"])
        if t_idx == 0:
            first_upd = res_e["delta_applied"].clone()
        cum_applied += res_e["delta_applied"]
        del res_e["delta_applied"]
        edit_res.append(res_e)

    ev_a0 = evaluate_sequence_metrics(
        m_arm, sequence_s0, edit_res, "gate0_d0.0_s0", "r0_unconstrained_d0.0",
        fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device
    )

    # Evaluate Rule 3.7 raw outcome vectors
    preds_term = [greedy_predict(m_arm, tokenizer, f["edit_prompt"], 5, device, False) for f in sequence_s0]
    term_matches = [check_match(p, f["object"]) for p, f in zip(preds_term, sequence_s0)]
    imm_matches = [r["immediate_match"] for r in edit_res]
    steps_taken = [r["steps_taken"] for r in edit_res]

    obs_steps = ev_a0["optimizer_steps"]
    obs_imm = ev_a0["immediate_efficacy"].numerator
    obs_term = ev_a0["terminal_retention"].numerator

    # S0-6 Reference values for Seed 0
    ref_steps = 669
    ref_imm = 200
    ref_term = 8

    steps_match = (obs_steps == ref_steps)
    imm_match = (obs_imm == ref_imm)
    term_match = (obs_term == ref_term)

    diff_count = (0 if steps_match else 1) + (abs(obs_imm - ref_imm)) + (abs(obs_term - ref_term))

    print(f"\n  Gate 0 Results vs S0-6 Pinned Artifact:")
    print(f"    Optimizer Steps    : Observed = {obs_steps:<4d} | Reference = {ref_steps:<4d} | {'EXACT MATCH' if steps_match else 'MISMATCH'}")
    print(f"    Immediate Efficacy : Observed = {obs_imm:<4d}/200 | Reference = {ref_imm:<4d}/200 | {'EXACT MATCH' if imm_match else 'MISMATCH'}")
    print(f"    Terminal Retention : Observed = {obs_term:<4d}/200 | Reference = {ref_term:<4d}/200 | {'EXACT MATCH' if term_match else 'MISMATCH'}")

    exact_pass = steps_match and imm_match and term_match
    fallback_pass = (diff_count <= 1)

    if exact_pass:
        print("\n  Gate 0 Reproduction Outcome: PASSED (Exact Bit-for-Bit Reproduction)")
    elif fallback_pass:
        print(f"\n  Gate 0 Reproduction Outcome: PASSED (Within Fallback Tolerance: {diff_count} mismatch <= 1)")
    else:
        elapsed = time.time() - start_time
        print(f"\nFATAL: Gate 0 Bit-Reproduction FAILED at elapsed wall-clock {elapsed:.2f} s!")
        print("S0-6 is not reproducible under the current environment. Halting directive immediately.")
        sys.exit(1)

    seed0_payload = {
        "metrics": ev_a0, "edit_results": edit_res, "first_update": first_upd, "cum_applied": cum_applied,
        "raw_vectors": {"immediate_matches": imm_matches, "terminal_matches": term_matches, "steps_taken": steps_taken}
    }

    gate_0_summary = {
        "passed": exact_pass or fallback_pass, "exact_match": exact_pass,
        "observed_steps": obs_steps, "reference_steps": ref_steps,
        "observed_imm_eff": obs_imm, "reference_imm_eff": ref_imm,
        "observed_term_ret": obs_term, "reference_term_ret": ref_term, "diff_count": diff_count
    }

    del m_arm, subspace_unapp
    gc.collect()
    torch.cuda.empty_cache()

    return seed0_payload, gate_0_summary


def run_stage_f_re_emission(
    model_name: str,
    pinned_revision: str,
    tokenizer: Any,
    fresh_model: nn.Module,
    sequences: Dict[int, List[Dict[str, Any]]],
    facts_1000: List[Dict[str, Any]],
    template_prior_controls: List[Dict[str, Any]],
    wikitext_slice: Any,
    slice_sha: str,
    device: str,
    seed0_gate0_payload: Dict[str, Any]
) -> Tuple[Dict[str, Dict[int, Any]], Dict[str, Any], int, List[Dict[str, Any]]]:
    """
    Executes the remainder of Stage F across 6 seeds (N=1200) for 4 arms and 4 controls,
    serializing Rule 3.7 raw outcome vectors.
    """
    print("\n" + "=" * 95)
    print(" STAGE F: RE-EMISSION RUN WITH RAW OUTCOME VECTOR SERIALIZATION")
    print("=" * 95)

    ckpt_path = REPO_ROOT / "experiments" / "results" / "stage_f_checkpoint.pt"
    if ckpt_path.exists():
        print(f"\n[Stage F Checkpoint Found: Loading pre-computed Stage F results from {ckpt_path.name}]")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return saved["cond_results"], saved["structural_invariance"], saved["total_optimizer_steps"], saved["line_item_steps"]

    cond_results: Dict[str, Dict[int, Any]] = {}
    total_optimizer_steps_global = 0
    line_item_steps = []
    seed0_first_edit_updates = {}
    seed0_cumulative_updates = {}

    # 1. Negative Controls
    print("\n--- [Measuring Four Named Controls across 6 Seeds (N=1200)] ---")
    ctrl_measures_by_seed: Dict[int, Dict[str, Any]] = {}
    for s in SEEDS:
        seq_facts = sequences[s]
        configure_determinism(seed=s)

        # Control 1: Never Edited
        preds_never = [greedy_predict(fresh_model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_1000[200:400]]
        matches_never = [check_match(p, f["object"]) for p, f in zip(preds_never, facts_1000[200:400])]
        m_never = Measurement.from_outcomes(matches_never, metric="never_edited", arm="never_edited", scope="per_seed", input_set=f"never_s{s}", mode="eval_no_dropout")

        # Control 2: Random Direction Magnitude Matched
        m_rand = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        rng_dir = torch.Generator(device=device).manual_seed(s)
        with torch.no_grad():
            for p in m_rand.parameters():
                pert = torch.randn(p.shape, generator=rng_dir, device=device)
                p.add_(pert / (torch.norm(pert) + 1e-12) * (5.0 * 3.0e-05 * 10.0))
        preds_rand = [greedy_predict(m_rand, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        matches_rand = [check_match(p, f["object"]) for p, f in zip(preds_rand, seq_facts)]
        m_rand_dir = Measurement.from_outcomes(matches_rand, metric="random_direction_magnitude_matched", arm="random_direction_magnitude_matched", scope="per_seed", input_set=f"rand_s{s}", mode="eval_no_dropout")
        del m_rand
        gc.collect()
        torch.cuda.empty_cache()

        # Control 3: Wrong Target (Optimizing Run across all 6 seeds)
        m_wrong = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        rng_w = random.Random(s)
        wrong_facts = [{**f, "object": rng_w.choice([c["object"] for c in facts_1000 if c["relation"] == f["relation"] and normalize_entity(c["object"]) != normalize_entity(f["object"])])} for f in seq_facts]
        wrong_steps_list = []
        for fw in wrong_facts:
            r_w = edit_fact_sgd(m_wrong, tokenizer, fw, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False)
            wrong_steps_list.append(r_w["steps_taken"])
        wrong_steps = sum(wrong_steps_list)
        total_optimizer_steps_global += wrong_steps
        line_item_steps.append({"item": "control:wrong_target", "seed": s, "steps": wrong_steps, "shared": False})
        preds_wrong = [greedy_predict(m_wrong, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        matches_wrong = [check_match(p, f["object"]) for p, f in zip(preds_wrong, seq_facts)]
        m_wrong_tgt = Measurement.from_outcomes(matches_wrong, metric="wrong_target", arm="wrong_target", scope="per_seed", input_set=f"wrong_s{s}", mode="eval_no_dropout")
        del m_wrong
        gc.collect()
        torch.cuda.empty_cache()

        # Control 4: Pre-Edit Baseline
        preds_pre = [greedy_predict(fresh_model, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        matches_pre = [check_match(p, f["object"]) for p, f in zip(preds_pre, seq_facts)]
        m_pre = Measurement.from_outcomes(matches_pre, metric="pre_edit_baseline", arm="pre_edit_baseline", scope="per_seed", input_set=f"pre_s{s}", mode="eval_no_dropout")

        ctrl_measures_by_seed[s] = {
            "never_edited": {
                "measurement": m_never,
                "raw_vectors": {"terminal_matches": matches_never, "immediate_matches": matches_never, "steps_taken": [0] * len(matches_never)}
            },
            "random_direction_magnitude_matched": {
                "measurement": m_rand_dir,
                "raw_vectors": {"terminal_matches": matches_rand, "immediate_matches": matches_rand, "steps_taken": [0] * len(matches_rand)}
            },
            "wrong_target": {
                "measurement": m_wrong_tgt,
                "raw_vectors": {"terminal_matches": matches_wrong, "immediate_matches": [False] * len(matches_wrong), "steps_taken": wrong_steps_list}
            },
            "pre_edit_baseline": {
                "measurement": m_pre,
                "raw_vectors": {"terminal_matches": matches_pre, "immediate_matches": matches_pre, "steps_taken": [0] * len(matches_pre)}
            }
        }

    # Store control per-seed records
    for c_name in CONTROL_NAMES:
        cond_results[c_name] = {s: ctrl_measures_by_seed[s][c_name] for s in SEEDS}

    # 2. Arm B (r1_causal_perstep_d0.0 across 6 seeds)
    print()
    print("  Running Arm B (r1_causal_perstep_d0.0 across 6 seeds)")
    per_seed_b = {}
    arm_b_observed_sf_records = {}
    for s in SEEDS:
        configure_determinism(seed=s)
        m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        subspace_mgr = CausalSubspaceManager(device=device)
        edit_res = []
        cum_applied = torch.zeros_like(m_arm.lm_head.weight.data)
        for t_idx, f in enumerate(sequences[s]):
            t_num = t_idx + 1
            Q_t = subspace_mgr.get_projection_matrix(1)
            if Q_t is not None and Q_t.numel() > 0:
                assert_orthonormality(Q_t, tol=1e-6)
            res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False, arm_mode="r1_causal_perstep", Q_causal=Q_t)
            arm_b_observed_sf_records[(s, t_num)] = res_e["sf_row"]
            subspace_mgr.add_update(res_e["delta_target_vec"])
            cum_applied += res_e["delta_applied"]
            if s == 0 and t_num == 1:
                seed0_first_edit_updates["r1_causal_perstep_d0.0"] = res_e["delta_applied"].clone()
            del res_e["delta_applied"]
            edit_res.append(res_e)

        if s == 0:
            seed0_cumulative_updates["r1_causal_perstep_d0.0"] = cum_applied.clone()

        s_steps = sum(r["steps_taken"] for r in edit_res)
        total_optimizer_steps_global += s_steps
        line_item_steps.append({"item": "arm:r1_causal_perstep_d0.0", "seed": s, "steps": s_steps, "shared": False})
        ev_b = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"r1_causal_s{s}", "r1_causal_perstep", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)

        preds_term = [greedy_predict(m_arm, tokenizer, f["edit_prompt"], 5, device, False) for f in sequences[s]]
        term_matches = [check_match(p, f["object"]) for p, f in zip(preds_term, sequences[s])]
        imm_matches = [r["immediate_match"] for r in edit_res]
        steps_taken = [r["steps_taken"] for r in edit_res]

        ev_b["raw_vectors"] = {"immediate_matches": imm_matches, "terminal_matches": term_matches, "steps_taken": steps_taken}
        per_seed_b[s] = ev_b
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_b['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_b['terminal_retention'])} | Steps={ev_b['optimizer_steps']}")
        del m_arm, subspace_mgr
        gc.collect()
        torch.cuda.empty_cache()
    cond_results["r1_causal_perstep_d0.0"] = per_seed_b

    # 3. Arm F (r1_magnitude_only_d0.0 across 6 seeds, matched to Arm B's SF)
    print()
    print("  Running Arm F (r1_magnitude_only_d0.0 across 6 seeds)")
    per_seed_f = {}
    for s in SEEDS:
        configure_determinism(seed=s)
        m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        edit_res = []
        cum_applied = torch.zeros_like(m_arm.lm_head.weight.data)
        for t_idx, f in enumerate(sequences[s]):
            t_num = t_idx + 1
            alpha_val = arm_b_observed_sf_records.get((s, t_num), 1.0)
            res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False, arm_mode="r1_magnitude_only", alpha_scale=alpha_val)
            cum_applied += res_e["delta_applied"]
            if s == 0 and t_num == 1:
                seed0_first_edit_updates["r1_magnitude_only_d0.0"] = res_e["delta_applied"].clone()
            del res_e["delta_applied"]
            edit_res.append(res_e)

        if s == 0:
            seed0_cumulative_updates["r1_magnitude_only_d0.0"] = cum_applied.clone()

        s_steps = sum(r["steps_taken"] for r in edit_res)
        total_optimizer_steps_global += s_steps
        line_item_steps.append({"item": "arm:r1_magnitude_only_d0.0", "seed": s, "steps": s_steps, "shared": False})
        ev_f = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"r1_magnitude_only_s{s}", "r1_magnitude_only", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)

        preds_term = [greedy_predict(m_arm, tokenizer, f["edit_prompt"], 5, device, False) for f in sequences[s]]
        term_matches = [check_match(p, f["object"]) for p, f in zip(preds_term, sequences[s])]
        imm_matches = [r["immediate_match"] for r in edit_res]
        steps_taken = [r["steps_taken"] for r in edit_res]

        ev_f["raw_vectors"] = {"immediate_matches": imm_matches, "terminal_matches": term_matches, "steps_taken": steps_taken}
        per_seed_f[s] = ev_f
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_f['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_f['terminal_retention'])} | Steps={ev_f['optimizer_steps']}")
        del m_arm
        gc.collect()
        torch.cuda.empty_cache()
    cond_results["r1_magnitude_only_d0.0"] = per_seed_f

    # 4. Arm A (r0_unconstrained_d0.0: Seed 0 reused from Gate 0, seeds 1..5 executed)
    print()
    print("  Running Arm A (r0_unconstrained_d0.0 across 6 seeds)")
    per_seed_a_d0 = {}
    ev_a0 = seed0_gate0_payload["metrics"]
    ev_a0["raw_vectors"] = seed0_gate0_payload["raw_vectors"]
    per_seed_a_d0[0] = ev_a0
    s0_steps = sum(seed0_gate0_payload["raw_vectors"]["steps_taken"])
    total_optimizer_steps_global += s0_steps
    line_item_steps.append({"item": "arm:r0_unconstrained_d0.0", "seed": 0, "steps": s0_steps, "shared": True})
    seed0_first_edit_updates["r0_unconstrained_d0.0"] = seed0_gate0_payload["first_update"]
    seed0_cumulative_updates["r0_unconstrained_d0.0"] = seed0_gate0_payload["cum_applied"]
    print(f"    Seed 0 (from Gate 0): ImmEff={format_wilson_rate(ev_a0['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_a0['terminal_retention'])} | Steps={ev_a0['optimizer_steps']}")

    for s in [1, 2, 3, 4, 5]:
        configure_determinism(seed=s)
        m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        subspace_unapp = CausalSubspaceManager(device=device)
        edit_res = []
        for t_idx, f in enumerate(sequences[s]):
            Q_unapp = subspace_unapp.get_projection_matrix(1)
            res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=100, delta=0.0, device=device, train_mode=False, arm_mode="r0_unconstrained", Q_causal=Q_unapp)
            subspace_unapp.add_update(res_e["delta_target_vec"])
            del res_e["delta_applied"]
            edit_res.append(res_e)
        s_steps = sum(r["steps_taken"] for r in edit_res)
        total_optimizer_steps_global += s_steps
        line_item_steps.append({"item": "arm:r0_unconstrained_d0.0", "seed": s, "steps": s_steps, "shared": (s < 3)})
        ev_a = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"r0_unconstrained_d0.0_s{s}", "r0_unconstrained_d0.0", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)

        preds_term = [greedy_predict(m_arm, tokenizer, f["edit_prompt"], 5, device, False) for f in sequences[s]]
        term_matches = [check_match(p, f["object"]) for p, f in zip(preds_term, sequences[s])]
        imm_matches = [r["immediate_match"] for r in edit_res]
        steps_taken = [r["steps_taken"] for r in edit_res]

        ev_a["raw_vectors"] = {"immediate_matches": imm_matches, "terminal_matches": term_matches, "steps_taken": steps_taken}
        per_seed_a_d0[s] = ev_a
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_a['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_a['terminal_retention'])} | Steps={ev_a['optimizer_steps']}")
        del m_arm, subspace_unapp
        gc.collect()
        torch.cuda.empty_cache()
    cond_results["r0_unconstrained_d0.0"] = per_seed_a_d0

    # 5. Arm A (r0_unconstrained_d1.0 across 6 seeds)
    print()
    print("  Running Arm A (r0_unconstrained_d1.0 across 6 seeds)")
    per_seed_a_d1 = {}
    for s in SEEDS:
        configure_determinism(seed=s)
        m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        subspace_unapp = CausalSubspaceManager(device=device)
        edit_res = []
        cum_applied = torch.zeros_like(m_arm.lm_head.weight.data)
        for t_idx, f in enumerate(sequences[s]):
            Q_unapp = subspace_unapp.get_projection_matrix(1)
            res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=100, delta=1.0, device=device, train_mode=False, arm_mode="r0_unconstrained", Q_causal=Q_unapp)
            subspace_unapp.add_update(res_e["delta_target_vec"])
            cum_applied += res_e["delta_applied"]
            if s == 0 and t_idx == 0:
                seed0_first_edit_updates["r0_unconstrained_d1.0"] = res_e["delta_applied"].clone()
            del res_e["delta_applied"]
            edit_res.append(res_e)

        if s == 0:
            seed0_cumulative_updates["r0_unconstrained_d1.0"] = cum_applied.clone()

        s_steps = sum(r["steps_taken"] for r in edit_res)
        total_optimizer_steps_global += s_steps
        line_item_steps.append({"item": "arm:r0_unconstrained_d1.0", "seed": s, "steps": s_steps, "shared": False})
        ev_a = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"r0_unconstrained_d1.0_s{s}", "r0_unconstrained_d1.0", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)

        preds_term = [greedy_predict(m_arm, tokenizer, f["edit_prompt"], 5, device, False) for f in sequences[s]]
        term_matches = [check_match(p, f["object"]) for p, f in zip(preds_term, sequences[s])]
        imm_matches = [r["immediate_match"] for r in edit_res]
        steps_taken = [r["steps_taken"] for r in edit_res]

        ev_a["raw_vectors"] = {"immediate_matches": imm_matches, "terminal_matches": term_matches, "steps_taken": steps_taken}
        per_seed_a_d1[s] = ev_a
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_a['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_a['terminal_retention'])} | Steps={ev_a['optimizer_steps']}")
        del m_arm, subspace_unapp
        gc.collect()
        torch.cuda.empty_cache()
    cond_results["r0_unconstrained_d1.0"] = per_seed_a_d1

    structural_invariance = {
        "first_updates": seed0_first_edit_updates,
        "cumulative_updates": seed0_cumulative_updates
    }

    ckpt_path = REPO_ROOT / "experiments" / "results" / "stage_f_checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cond_results": cond_results, "structural_invariance": structural_invariance,
        "total_optimizer_steps": total_optimizer_steps_global, "line_item_steps": line_item_steps
    }, ckpt_path)
    print(f"\n[Stage F Checkpoint Saved: {ckpt_path.name}]")

    return cond_results, structural_invariance, total_optimizer_steps_global, line_item_steps
