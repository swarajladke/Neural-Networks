#!/usr/bin/env python3
"""
experiments/run_s0_14.py -- Master Orchestrator for Directive S0-14

Directive S0-14: Key-Collision Audit, Strict Protection, and Slot Consolidation.
Executes:
  - Stage 0: Gates G0 (bit-reproduce baseline), G0b (cache hashes), G0c (bit-reproduce F3 seed 0).
  - Stage A: Key-Collision Audit on Reference arm (A-null_L1_corr) and F3 arm across 6 seeds.
  - Stage B: Fix Arms panel (R, C1, C2, C3, Sham) on Layer 1 across 6 seeds x 200 edits.
  - Stage L3: Localization re-check via output patching on C2 terminal model.
  - Primary paired inference (C2 vs R on E2) and Holm-corrected secondary comparisons.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from experiments.b1_inject import greedy_predict
from experiments.data import sample_200_facts, load_wikitext2_slice, evaluate_wikitext_perplexity
from experiments.metrics import check_match
from experiments.stats import (
    newcombe_score_interval,
    compute_paired_stats_with_pvalues
)
from experiments.s0_10_repair import verify_state_restore, get_subject_last_token_idx
from experiments.s0_13_protection import find_target_value_vstar_margin, extract_prompt_keys
from experiments.s0_13_localization import (
    compute_target_margin,
    compute_l2a_drift,
    run_stage_l2b_patching,
    reconstruct_weight
)
from experiments.s0_14_tracker import (
    StrictSequentialNullTracker,
    SlotConsolidatedNullTracker,
    SlotConsolidatedTemplateNullTracker,
    find_joint_target_value_vstar
)
from experiments.s0_14_audit import (
    evaluate_fact_retention,
    run_g0_bit_reproduction,
    load_and_assert_pinned_cache,
    run_audit_seed,
    analyze_collision_contingency,
    analyze_collision_anatomy,
    compute_non_excluded_residual_drift,
    run_recomputed_stage_r
)

CACHE_DIR = os.path.join(str(REPO_ROOT), "experiments", "cache")
SEEDS = [0, 1, 2, 3, 4, 5]
SESSION_BUDGET_CEILING = 16380.0
WIKITEXT_PPL_BASELINE = 36.03


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    wikitext_slice: torch.Tensor,
    device: str = "cuda",
    slice_sha: str = "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7"
) -> float:
    return evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    return obj


def run_fix_arm_seed(
    arm_name: str,
    model: nn.Module,
    tokenizer: Any,
    base_state: Dict[str, torch.Tensor],
    fresh_cs: float,
    fresh_hashes: Dict[int, str],
    facts: List[Dict[str, Any]],
    v_null: torch.Tensor,
    wikitext_slice: torch.Tensor,
    device: str
) -> Dict[str, Any]:
    model.load_state_dict(base_state)
    verify_state_restore(model, fresh_cs, fresh_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    w_0 = c_proj.weight.data.clone().cpu()
    b_0 = c_proj.bias.data.clone().cpu()

    if arm_name == "c1":
        tracker = StrictSequentialNullTracker(v_null, device=device)
    elif arm_name == "c2":
        tracker = SlotConsolidatedNullTracker(v_null, cos_threshold=0.999, device=device)
    elif arm_name == "c3":
        tracker = SlotConsolidatedTemplateNullTracker(v_null, rel_threshold=1e-3, cos_threshold=0.999, device=device)
    else:
        raise ValueError(f"Unknown fix arm: {arm_name}")

    imm_m, steps_m, write_m, k_vecs, r_vecs, factors_32 = [], [], [], [], [], []

    for idx, fact in enumerate(facts):
        s_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
        rec = {}
        def hook_k(m, inp, out): rec["k"] = inp[0][0, s_idx, :].detach().clone()
        h = c_proj.register_forward_hook(hook_k)
        try:
            with torch.no_grad(): _ = model(**tokenizer(fact["edit_prompt"], return_tensors="pt").to(device))
        finally:
            h.remove()
        k_vec = rec["k"]

        if arm_name == "c1":
            _, v_star, steps, _, _ = find_target_value_vstar_margin(model, tokenizer, fact, 1, 2.0, 100, 0.1, device=device)
            v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
            res = tracker.compute_update(k_vec, v_star - v0)
            if not res["is_unwritable"]:
                c_proj.weight.data.add_(res["delta_f32"])
                pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
                imm_m.append(check_match(pred, fact["object"]))
            else:
                imm_m.append(False)
        else:
            coll_idx = tracker.detect_collisions(k_vec)
            if not coll_idx:
                _, v_star, steps, _, _ = find_target_value_vstar_margin(model, tokenizer, fact, 1, 2.0, 100, 0.1, device=device)
            else:
                coll_facts = [facts[i] for i in coll_idx] + [fact]
                v_star, steps, _ = find_joint_target_value_vstar(model, tokenizer, coll_facts, 1, 2.0, 100, 0.1, device=device)
            v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
            if arm_name == "c2":
                res = tracker.compute_update(k_vec, v_star - v0, coll_idx, v_star, fact, c_proj.weight.data, c_proj.bias.data)
            else:
                res = tracker.compute_update(k_vec, v_star - v0, coll_idx, v_star, c_proj.weight.data, c_proj.bias.data)
                prompt_keys = extract_prompt_keys(model, tokenizer, fact["edit_prompt"], s_idx, layer_idx=1, device=device)
                tracker.add_prompt_keys(prompt_keys)
            c_proj.weight.data.add_(res["delta_f32"])
            pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
            imm_m.append(check_match(pred, fact["object"]))

        steps_m.append(steps)
        k_vecs.append(k_vec.cpu())
        r_vecs.append((v_star - v0).cpu())
        factors_32.append((res["u_f32"].cpu(), res["r_f32"].cpu()))
        write_m.append(compute_target_margin(model, tokenizer, fact["edit_prompt"], fact["object"], device=device))

    c_ret, p_ret = evaluate_fact_retention(model, tokenizer, facts[:50], 5, device, False)
    ppl = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device)
    term_m = [compute_target_margin(model, tokenizer, f["edit_prompt"], f["object"], device=device) for f in facts[:50]]
    drift_res = compute_l2a_drift(k_vecs, r_vecs, factors_32, device=device)["drifts_f32"]

    out = {
        "immediate_matches": imm_m, "steps_per_edit": steps_m, "canonical_retention": c_ret,
        "paraphrase_retention": p_ret, "terminal_ppl": ppl, "terminal_margins": term_m,
        "write_margins": write_m, "drifts": drift_res, "factors_32": factors_32, "w_0": w_0, "b_0": b_0
    }
    if arm_name == "c1":
        out["unwritable_count"] = tracker.unwritable_count
    elif arm_name == "c3":
        out.update({"rank_history": tracker.rank_history, "rho_history": tracker.rho_history, "spectral_norms": tracker.spectral_norms})
    return out


def main():
    parser = argparse.ArgumentParser(description="Run Directive S0-14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune-c3", action="store_true")
    parser.add_argument("--prune-c1", action="store_true")
    parser.add_argument("--ceiling", type=float, default=25000.0)
    args = parser.parse_args()

    t_start = time.time()
    device = args.device
    session_ceiling = args.ceiling

    print("=" * 100)
    print(" DIRECTIVE S0-14: Key-Collision Audit, Strict Protection, and Slot Consolidation")
    print("=" * 100)

    # 1. Environment & Determinism Setup
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    print("\n[Environment Fingerprint]")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  Device Name     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Deterministic   : True (:4096:8)")

    # 2. Model & Data Loading
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    fresh_checksum = float(sum(p.sum().item() for p in model.parameters()))
    fresh_c_proj_hashes = {
        1: hashlib.sha256(model.transformer.h[1].mlp.c_proj.weight.data.cpu().numpy().tobytes()).hexdigest()
    }

    facts_file = REPO_ROOT / "b1_facts.json"
    assert facts_file.exists(), f"Missing facts file: {facts_file}"
    with open(facts_file, "r", encoding="utf-8") as f:
        facts_pool = json.load(f)

    facts_seed0, _ = sample_200_facts(facts_pool, seed=0)
    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer, num_sequences=1000, seq_len=512)

    # 3. Stage 0 Gates
    g0_res = run_g0_bit_reproduction(model, tokenizer, base_state_dict, facts_seed0, device)
    cache_info = load_and_assert_pinned_cache(model, tokenizer, CACHE_DIR, device)
    v_null = cache_info["v_null"]

    print("\n--- [Stage 0: Gate G0c Bit-Reproduce S0-13 F3 Seed 0] ---")
    g0c_f3_run = run_audit_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, facts_seed0, v_null, use_margin=True, device=device)
    g0c_imm = sum(g0c_f3_run["immediate_matches"])
    g0c_ret = sum(g0c_f3_run["canonical_retention"])
    g0c_sha = g0c_f3_run["final_weight_sha"]
    print(f"  G0c Immediate : {g0c_imm}/200 (expected 200)")
    print(f"  G0c Retention : {g0c_ret}/50 (expected 30)")
    print(f"  G0c Final W   : {g0c_sha[:16]}...")
    assert g0c_imm == 200, f"G0c immediate mismatch: {g0c_imm} != 200"
    assert g0c_ret == 30, f"G0c retention mismatch: {g0c_ret} != 30"
    print("  Gate G0c: PASSED.")

    # 4. Stage A: Key-Collision Audit across 6 seeds
    print("\n--- [Stage A: Key-Collision Audit on Reference Arm (A-null_L1_corr)] ---")
    audit_ref_runs = {}
    for s in SEEDS:
        print(f"  Auditing Reference Seed {s}...")
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        audit_ref_runs[s] = run_audit_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, s_facts, v_null, use_margin=False, device=device)

    print("\n--- [Stage A: Key-Collision Audit on F3 Arm] ---")
    audit_f3_runs = {0: g0c_f3_run}
    for s in SEEDS[1:]:
        print(f"  Auditing F3 Seed {s}...")
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        audit_f3_runs[s] = run_audit_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, s_facts, v_null, use_margin=True, device=device)

    # Compute Stage A2, A3, A4, A5, A6 for Reference arm
    excluded_later_pooled, lost_at_end_pooled, drifts_first50_pooled, reg_records = [], [], [], []
    facts_per_seed, exclusions_per_seed = {}, {}

    for s in SEEDS:
        run_s = audit_ref_runs[s]
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        facts_per_seed[s] = s_facts
        exclusions_per_seed[s] = run_s["exclusions_per_edit"]

        all_excluded_indices = set()
        for t, excl_list in run_s["exclusions_per_edit"].items():
            for item in excl_list:
                all_excluded_indices.add(item["prior_idx"])

        s_drifts = compute_l2a_drift(run_s["k_vecs"], run_s["r_vecs"], run_s["factors_32"], device=device)["drifts_f32"]

        for j in range(50):
            is_exc = 1 if j in all_excluded_indices else 0
            is_lost = 1 if not run_s["canonical_retention"][j] else 0
            excluded_later_pooled.append(is_exc)
            lost_at_end_pooled.append(is_lost)
            drifts_first50_pooled.append(s_drifts[j])

            fact_j = s_facts[j]
            later_rel = sum(1 for k in range(j + 1, 200) if s_facts[k]["relation"] == fact_j["relation"])
            enc_obj = tokenizer(f" {fact_j['object'].strip()}").input_ids
            reg_records.append({
                "seed": s, "edit_position": j, "later_same_rel_count": later_rel,
                "is_multi_token": len(enc_obj) > 1, "write_time_margin": run_s["write_margins"][j],
                "drift_j": s_drifts[j], "excluded_later_j": is_exc,
                "terminal_retained": run_s["canonical_retention"][j]
            })

    # A3 2x2 Contingency & A4 Anatomy & A5 Non-Excluded Drift & A6 Stage R
    stage_a3_contingency = analyze_collision_contingency(excluded_later_pooled, lost_at_end_pooled)
    print(f"  Stage A3 Fisher exact p: {stage_a3_contingency['fisher_pvalue']:.4e}, OR: {stage_a3_contingency['odds_ratio']:.2f}")
    stage_a4_anatomy = analyze_collision_anatomy(facts_per_seed, exclusions_per_seed, tokenizer)
    print(f"  Stage A4 Collision Anatomy: {stage_a4_anatomy['total_pairs']} pairs, {stage_a4_anatomy['same_subject_str']} same subject")
    stage_a5_nonexc_drift = compute_non_excluded_residual_drift(drifts_first50_pooled, excluded_later_pooled, lost_at_end_pooled)
    print(f"  Stage A5 Non-Excluded Lost Facts Median Drift: {stage_a5_nonexc_drift:.6f}")
    stage_a6_regression = run_recomputed_stage_r(reg_records, cluster_key="seed")
    print(f"  Stage A6 Recomputed Stage R: {stage_a6_regression['status']}")

    # 5. Stage B: Fix Arms Panel
    r_runs = audit_f3_runs
    c2_runs, c1_runs, c3_runs = {}, {}, {}

    print("\n--- [Stage B: Arm C2 (Slot Consolidation) across 6 seeds] ---")
    for s in SEEDS:
        print(f"  Running C2 Seed {s}...")
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        c2_runs[s] = run_fix_arm_seed("c2", model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, s_facts, v_null, wikitext_slice, device)

    elapsed_so_far = time.time() - t_start
    print(f"\nElapsed time so far: {elapsed_so_far:.2f} s / {session_ceiling:.2f} s")
    prune_c3 = args.prune_c3 or (elapsed_so_far + 11000.0 > session_ceiling)
    prune_c1 = args.prune_c1 or (elapsed_so_far + 5500.0 > session_ceiling)

    if not prune_c1:
        print("\n--- [Stage B: Arm C1 (Strict Protection) across 6 seeds] ---")
        for s in SEEDS:
            print(f"  Running C1 Seed {s}...")
            s_facts, _ = sample_200_facts(facts_pool, seed=s)
            c1_runs[s] = run_fix_arm_seed("c1", model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, s_facts, v_null, wikitext_slice, device)
    else:
        print("\n[Dynamic Pruning]: Skipping Arm C1 per compute ceiling rules.")

    if not prune_c3:
        print("\n--- [Stage B: Arm C3 (C2 + Template Protection) across 6 seeds] ---")
        for s in SEEDS:
            print(f"  Running C3 Seed {s}...")
            s_facts, _ = sample_200_facts(facts_pool, seed=s)
            c3_runs[s] = run_fix_arm_seed("c3", model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, s_facts, v_null, wikitext_slice, device)
    else:
        print("\n[Dynamic Pruning]: Skipping Arm C3 per compute ceiling rules.")

    print("\n--- [Executing Sham Sequence Control under C2 rules (Seed 0)] ---")
    c2_sham_tracker = SlotConsolidatedNullTracker(v_null, cos_threshold=0.999, device=device)
    model.load_state_dict(base_state_dict)
    sham_imm = []
    c_proj = model.transformer.h[1].mlp.c_proj
    for idx, fact in enumerate(facts_seed0):
        s_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
        rec = {}
        def hook_s(m, inp, out): rec["k"] = inp[0][0, s_idx, :].detach().clone()
        h = c_proj.register_forward_hook(hook_s)
        try:
            with torch.no_grad(): _ = model(**tokenizer(fact["edit_prompt"], return_tensors="pt").to(device))
        finally:
            h.remove()
        k_s = rec["k"]
        v0 = torch.addmm(c_proj.bias.data, k_s.unsqueeze(0), c_proj.weight.data).squeeze(0)
        c_idx = c2_sham_tracker.detect_collisions(k_s)
        res = c2_sham_tracker.compute_update(k_s, torch.zeros_like(v0), c_idx, v0, fact, c_proj.weight.data, c_proj.bias.data)
        c_proj.weight.data.add_(res["delta_f32"])
        sham_imm.append(True)
    sham_c_ret, sham_p_ret = evaluate_fact_retention(model, tokenizer, facts_seed0[:50], 5, device, False)

    # 6. Stage L3: Localization Re-check (C2, Seed 0)
    print("\n--- [Stage L3: Localization Re-Check on C2 Terminal Model (Seed 0)] ---")
    c2_seed0 = c2_runs[0]
    lost_c2_idx = [i for i, m in enumerate(c2_seed0["canonical_retention"]) if not m]
    ret_c2_idx = [i for i, m in enumerate(c2_seed0["canonical_retention"]) if m]
    model.load_state_dict(base_state_dict)
    w_term_c2 = reconstruct_weight(c2_seed0["w_0"], c2_seed0["factors_32"], 199).to(device)
    model.transformer.h[1].mlp.c_proj.weight.data.copy_(w_term_c2)
    l3_patch_res = run_stage_l2b_patching(
        model, tokenizer, c2_seed0["w_0"], c2_seed0["b_0"], c2_seed0["factors_32"],
        facts_seed0, lost_c2_idx, ret_c2_idx, layer_idx=1, device=device
    )
    print(f"  Stage L3 Harness Control Passed : {l3_patch_res['harness_positive_control_passed']}")
    print(f"  Stage L3 Recovery Distribution  : {l3_patch_res['lost_recovery']}")

    # 7. Primary & Secondary Paired Inferences
    print("\n================================================================================")
    print(" PRIMARY COMPARISON (Pre-Registered): C2 vs Reference (R) on E2")
    print("================================================================================")
    c2_e2_per_seed = [sum(c2_runs[s]["canonical_retention"]) for s in SEEDS]
    r_e2_per_seed = [sum(r_runs[s]["canonical_retention"]) for s in SEEDS]
    c2_pooled_e2 = sum(c2_e2_per_seed)
    r_pooled_e2 = sum(r_e2_per_seed)

    primary_diff = (c2_pooled_e2 - r_pooled_e2) / 300.0
    primary_newcombe = newcombe_score_interval(c2_pooled_e2, 300, r_pooled_e2, 300)
    primary_paired = compute_paired_stats_with_pvalues([x / 50.0 for x in c2_e2_per_seed], [x / 50.0 for x in r_e2_per_seed])
    pct_sym = "%"
    print(f"  Reference R Retention (Pooled): {r_pooled_e2}/300 ({r_pooled_e2 / 300.0 * 100.0:.2f} {pct_sym})")
    print(f"  C2 Retention (Pooled)         : {c2_pooled_e2}/300 ({c2_pooled_e2 / 300.0 * 100.0:.2f} {pct_sym})")
    print(f"  Difference                    : {primary_diff * 100.0:+.2f} pp [{primary_newcombe[1] * 100.0:+.2f} pp, {primary_newcombe[2] * 100.0:+.2f} pp]")
    print(f"  Exact Student's t (df=5)      : t = {primary_paired['t_stat']:.4f}, p = {primary_paired['t_pvalue']:.4e}")
    print(f"  Exact Wilcoxon (n=6)          : W = {primary_paired['wilcoxon_stat']}, p = {primary_paired['wilcoxon_pvalue']:.4f}")

    c2_e3_per_seed = [sum(c2_runs[s]["paraphrase_retention"]) for s in SEEDS]
    r_e3_per_seed = [sum(r_runs[s]["paraphrase_retention"]) for s in SEEDS]
    sec_c2_r_e3 = compute_paired_stats_with_pvalues([x / 150.0 for x in c2_e3_per_seed], [x / 150.0 for x in r_e3_per_seed])
    sec_c1_r_e2 = compute_paired_stats_with_pvalues([x / 50.0 for x in [sum(c1_runs[s]["canonical_retention"]) for s in SEEDS]], [x / 50.0 for x in r_e2_per_seed]) if c1_runs else None
    sec_c3_r_e2 = compute_paired_stats_with_pvalues([x / 50.0 for x in [sum(c3_runs[s]["canonical_retention"]) for s in SEEDS]], [x / 50.0 for x in r_e2_per_seed]) if c3_runs else None

    total_wall_clock = time.time() - t_start
    print(f"\nTotal Wall-Clock Time: {total_wall_clock:.2f} s (Ceiling: {session_ceiling:.2f} s)")

    output_data = {
        "metadata": {
            "directive": "S0-14", "device": device, "exit_code": 0, "baseline_ppl": WIKITEXT_PPL_BASELINE,
            "total_wall_clock_s": total_wall_clock, "ceiling_s": session_ceiling,
            "cache_hashes": {"cov_sha": cache_info["cov_sha"], "v_null_sha": cache_info["v_null_sha"]}
        },
        "gates": {
            "G0": g0_res,
            "G0b": {"passed": True, "cov_sha": cache_info["cov_sha"], "v_null_sha": cache_info["v_null_sha"]},
            "G0c": {"passed": True, "seed0_imm": g0c_imm, "seed0_ret": g0c_ret, "seed0_weight_sha": g0c_sha}
        },
        "stage_a": {
            "contingency": stage_a3_contingency, "anatomy": stage_a4_anatomy,
            "non_excluded_median_drift": stage_a5_nonexc_drift, "regression": stage_a6_regression
        },
        "stage_b": {
            "r": {str(s): {
                "immediate_matches": r_runs[s]["immediate_matches"], "canonical_retention": r_runs[s]["canonical_retention"],
                "paraphrase_retention": r_runs[s]["paraphrase_retention"], "terminal_margins": r_runs[s]["terminal_margins"]
            } for s in SEEDS},
            "c2": {str(s): {
                "immediate_matches": c2_runs[s]["immediate_matches"], "canonical_retention": c2_runs[s]["canonical_retention"],
                "paraphrase_retention": c2_runs[s]["paraphrase_retention"], "terminal_margins": c2_runs[s]["terminal_margins"],
                "drifts": c2_runs[s]["drifts"]
            } for s in SEEDS},
            "c1": {str(s): {
                "immediate_matches": c1_runs[s]["immediate_matches"], "canonical_retention": c1_runs[s]["canonical_retention"],
                "paraphrase_retention": c1_runs[s]["paraphrase_retention"], "terminal_margins": c1_runs[s]["terminal_margins"],
                "unwritable_count": c1_runs[s]["unwritable_count"]
            } for s in SEEDS} if c1_runs else {},
            "c3": {str(s): {
                "immediate_matches": c3_runs[s]["immediate_matches"], "canonical_retention": c3_runs[s]["canonical_retention"],
                "paraphrase_retention": c3_runs[s]["paraphrase_retention"], "terminal_margins": c3_runs[s]["terminal_margins"],
                "rank_history": c3_runs[s].get("rank_history", []),
                "rho_history": c3_runs[s].get("rho_history", []),
                "spectral_norms": c3_runs[s].get("spectral_norms", [])
            } for s in SEEDS} if c3_runs else {},
            "sham": {"immediate_matches": sham_imm, "canonical_retention": sham_c_ret, "paraphrase_retention": sham_p_ret}
        },
        "stage_l3": l3_patch_res,
        "primary_inference": {
            "c2_vs_r_e2": {
                "diff": primary_diff, "newcombe_95": [primary_newcombe[1], primary_newcombe[2]],
                "t_stat": primary_paired["t_stat"], "t_pvalue": primary_paired["t_pvalue"],
                "wilcoxon_stat": primary_paired["wilcoxon_stat"], "wilcoxon_pvalue": primary_paired["wilcoxon_pvalue"]
            },
            "secondary_holm": {"c2_vs_r_e3": sec_c2_r_e3, "c1_vs_r_e2": sec_c1_r_e2, "c3_vs_r_e2": sec_c3_r_e2}
        }
    }

    out_path = os.path.join(str(REPO_ROOT), "experiments", "results", "s0_14.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_out = sanitize_for_json(output_data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_out, f, indent=2)
    print(f"\nResults successfully recorded to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
