#!/usr/bin/env python3
"""
experiments/run_s0_12.py -- Master Runner for Directive S0-12
Verification of S0-11 Positive Result, Activation Patching Loss Localization, and Full-Prompt Protection.

Mandate:
- Stage 0: Pinned assets & baseline verification, load S0-11 reference vectors
- Stage V: Verification of A-null_L1_corr and A-cov_L6 across 6 seeds x 200 edits:
    * Gate V1: Exact bit-for-bit reproduction of S0-11 immediate efficacy and first-50 retention
    * Full sequential controls: wrong_target, never_edited, pre_edit_baseline, sham_sequence
    * Comparator rule: primary floor is max(never_edited, pre_edit_baseline, sham_sequence)
    * Modal-collapse check & object recurrence breakdown
- Stage L: Activation patching on Seed 0 first-50 facts lost at sequence end:
    * (a) Subject last-token position only
    * (b) Non-subject prompt positions only
    * (c) All prompt positions
    * Residual key drift ||Delta_total k|| / ||k||
- Stage F: Full-prompt protection tracking all prompt tokens into the protected basis (A-null_L1_full)
- Strict compliance with AGENTS.md: under 600 lines, AST scanner, dynamic budget enforcement.
"""

import os
import sys
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.metrics import (
    check_match,
    compute_locality_kl,
    Measurement,
    wilson_confidence_interval,
    POPULATION_REGISTRY,
    normalize_entity
)
from experiments.stats import (
    newcombe_score_interval,
    compute_minimum_detectable_effect
)
from experiments.b1_inject import (
    configure_determinism,
    greedy_predict,
    get_next_token_log_probs,
    edit_fact_sgd
)
from experiments.data import (
    generate_synthetic_facts,
    sample_200_facts,
    load_wikitext2_slice,
    evaluate_wikitext_perplexity
)
from experiments.s0_10_repair import (
    edit_fact_mlp_fullgrad_sgd,
    verify_state_restore,
    evaluate_s0_10_capability_and_locality
)
from experiments.s0_11_constraints import (
    load_wikitext2_key_sample,
    compute_layer_key_covariance,
    compute_null_space_projector,
    edit_fact_mlp_cov,
    CorrectedSequentialNullTracker,
    edit_fact_mlp_null_corrected,
    find_target_value_vstar,
    N_PPL_SUBSET_SEQS
)
from experiments.s0_12_localization import (
    evaluate_s0_12_sequential_controls,
    evaluate_sham_sequence_control,
    check_modal_collapse,
    run_stage_l_patching,
    FullPromptSequentialNullTracker,
    edit_fact_mlp_null_full_prompt
)
from experiments.s0_12_diagnosis import run_stage_d_diagnosis

SESSION_CEILING_SEC = 23400.0
COMPUTE_CEILING_SEC = 16380.0
ACTIVE_SEEDS = [0, 1, 2, 3, 4, 5]
PPL_SURVIVAL_MULTIPLE = 2.0
NULL_SPACE_REL_THRESH = 1e-3


def format_expand_sum(counts: List[int]) -> Tuple[str, int]:
    return " + ".join(str(c) for c in counts), sum(counts)


def save_incremental_artifact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [Incremental Artifact Saved] -> {path.name}")


def run_s0_12_master():
    t_start_total = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 115)
    print(" DIRECTIVE S0-12: VERIFY POSITIVE RESULT, LOCATE REMAINING LOSS, AND FULL-PROMPT PROTECTION")
    print(" MANDATE: STAGE 0 -> STAGE V (VERIFICATION) -> STAGE L (LOCALIZATION) -> STAGE F (FULL-PROMPT)")
    print("=" * 115)

    # 1. Pre-flight Test Suite Execution
    print("\n--- [Pre-Flight Unit Test Suite Execution] ---")
    from tests.test_metrics import run_all_tests
    test_exit = run_all_tests()
    assert test_exit == 0, f"Pre-flight unit tests failed with code {test_exit}"

    # 2. Pinned Asset Verification
    facts_file = REPO_ROOT / "b1_facts.json"
    assert facts_file.exists(), f"Missing facts file: {facts_file}"
    facts_bytes = facts_file.read_bytes()
    facts_sha = hashlib.sha256(facts_bytes).hexdigest()
    assert facts_sha == "285638ad25c07b22299153cd6e67e413d2ed4a226d0a4103076d2066763cb536"
    print(f"\n  Pinned Facts SHA-256        : {facts_sha} (Verified)")

    facts_1000, template_prior_controls = generate_synthetic_facts(num_facts=1000, seed=42)
    facts_pinned = json.loads(facts_bytes.decode("utf-8"))
    assert len(facts_1000) == len(facts_pinned)

    ctrl_probe_str = json.dumps(template_prior_controls, sort_keys=True)
    ctrl_probe_sha = hashlib.sha256(ctrl_probe_str.encode("utf-8")).hexdigest()
    assert ctrl_probe_sha == "8f4ffa6b18d63531c898a6b2bf97d8b4a83d7038a54b9748bf77862178213887"
    print(f"  Control-Probe Set SHA-256   : {ctrl_probe_sha} (Verified 200 prompts)")

    model_name, pinned_revision = "gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, revision=pinned_revision)
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    base_model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    fresh_checksum = sum(p.sum().item() for p in model.parameters())
    fresh_c_proj_hashes = {}
    for l_idx in [1, 6]:
        w_b = model.transformer.h[l_idx].mlp.c_proj.weight.data.cpu().numpy().tobytes()
        fresh_c_proj_hashes[l_idx] = hashlib.sha256(w_b).hexdigest()

    base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer)
    assert slice_sha == "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7"
    print(f"  WikiText Slice SHA-256      : {slice_sha} (Verified 1,000 sequences)")

    subset_baseline_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device, max_sequences=N_PPL_SUBSET_SEQS)
    full_slice_baseline_ref = 36.03
    print(f"  Unedited Subset Baseline PPL: {subset_baseline_ppl:.2f}")

    # Load S0-11 reference artifact
    s0_11_path = REPO_ROOT / "experiments" / "results" / "s0_11.json"
    assert s0_11_path.exists(), "Directive S0-11 results artifact required for Stage V verification"
    with open(s0_11_path, "r", encoding="utf-8") as f:
        s0_11_data = json.load(f)

    # Stage C: Key Covariance & Null Projector Computation
    print("\n--- [Stage C: Key Covariance & Null Projector Collection] ---")
    key_sample_tensor, key_sample_sha = load_wikitext2_key_sample(tokenizer, num_sequences=100, seq_len=512)
    print(f"  Key Sample SHA-256          : {key_sample_sha}")

    cov_1, _ = compute_layer_key_covariance(model, key_sample_tensor, layer_idx=1, device=device)
    p_info_l1 = compute_null_space_projector(cov_1, rel_threshold=NULL_SPACE_REL_THRESH)
    cov_6, _ = compute_layer_key_covariance(model, key_sample_tensor, layer_idx=6, device=device)
    p_info_l6 = compute_null_space_projector(cov_6, rel_threshold=NULL_SPACE_REL_THRESH)

    out_json_path = REPO_ROOT / "experiments" / "results" / "s0_12.json"
    results_payload = {
        "directive": "S0-12",
        "producing_commit_sha": os.environ.get("COMMIT_SHA", "CANONICAL_RUN"),
        "exit_code": 0,
        "hashes": {
            "facts_json_sha256": facts_sha,
            "wikitext_slice_sha256": slice_sha,
            "control_probes_sha256": ctrl_probe_sha,
            "key_sample_sha256": key_sample_sha
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pinned_revision": pinned_revision,
            "fresh_param_sum": fresh_checksum,
            "subset_baseline_ppl": subset_baseline_ppl,
            "full_slice_baseline_ppl": full_slice_baseline_ref
        },
        "stage_v": {},
        "stage_l": {},
        "stage_f": {},
        "accounting": {"actual_wall_clock": 0.0, "total_optimizer_steps": 0, "total_samples_seen": 0}
    }

    # Pre-edit baselines over eval facts (50 facts per seed)
    all_eval_facts = []
    for s_idx in ACTIVE_SEEDS:
        f_seq, _ = sample_200_facts(facts_1000, seed=s_idx)
        all_eval_facts.extend(f_seq[:50])

    print("\n--- [Evaluating Pre-Edit Baseline Controls (300 eval facts)] ---")
    pre_edit_c_matches = []
    pre_edit_p_matches = []
    for f in all_eval_facts:
        pred_c = greedy_predict(base_model, tokenizer, f["edit_prompt"], 5, device, False)
        pre_edit_c_matches.append(check_match(pred_c, f["object"]))
        for p in f["paraphrases"]:
            pred_p = greedy_predict(base_model, tokenizer, p, 5, device, False)
            pre_edit_p_matches.append(check_match(pred_p, f["object"]))

    pre_c_k = sum(1 for x in pre_edit_c_matches if x)
    pre_p_k = sum(1 for x in pre_edit_p_matches if x)
    print(f"  Pre-Edit Canonical Floor    : {pre_c_k}/300 ({pre_c_k / 300.0 * 100.0:.2f}%)")
    print(f"  Pre-Edit Paraphrase Floor   : {pre_p_k}/900 ({pre_p_k / 900.0 * 100.0:.2f}%)")

    total_steps = 0
    total_samples = 0
    measured_arm_times = {}

    # Stage D: Divergence Diagnosis (Amendment 1 §B)
    print("\n--- [Stage D: Divergence Diagnosis (Seeds 1, 3, 5)] ---")
    diag_res = run_stage_d_diagnosis(
        model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
        facts_1000, cov_1, p_info_l1["v_null"], wikitext_slice, slice_sha, s0_11_data, device=device
    )
    results_payload["stage_d"] = diag_res
    save_incremental_artifact(out_json_path, results_payload)

    # Evaluate sham_sequence control arm on Seed 0 (Amendment 1 §D)
    print("\n--- [Evaluating Sham Sequence Control Arm (Seed 0, v* = v0)] ---")
    facts_seed0, _ = sample_200_facts(facts_1000, seed=0)
    sham_eval = evaluate_sham_sequence_control(
        model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
        facts_seed0, p_info_l1["v_null"], layer_idx=1, device=device
    )
    sham_c_count = sham_eval["sham_c_count"]
    sham_p_count = sham_eval["sham_p_count"]
    print(f"  Sham Canonical Floor        : {sham_c_count}/50 ({sham_c_count / 50.0 * 100.0:.2f}%)")
    print(f"  Sham Paraphrase Floor       : {sham_p_count}/150 ({sham_p_count / 150.0 * 100.0:.2f}%)")
    results_payload["sham_control"] = sham_eval
    save_incremental_artifact(out_json_path, results_payload)

    # Comparator Rule: Primary floor is max(never_edited, pre_edit_baseline, sham_sequence)
    # Scaled to N=300 and N=900
    sham_c_scaled = sham_c_count * 6
    sham_p_scaled = sham_p_count * 6

    # Stage V Arms: A-null_L1_corr and A-cov_L6
    v_arms = [
        {"name": "A-null_L1_corr", "type": "null_corr", "layer": 1, "v_null": p_info_l1["v_null"]},
        {"name": "A-cov_L6", "type": "cov", "layer": 6, "cov": cov_6}
    ]

    stage_v_results = {}
    seed0_lost_facts = []
    stage_v_confirmed = False

    for arm_cfg in v_arms:
        arm_name = arm_cfg["name"]
        a_type = arm_cfg["type"]
        l_idx = arm_cfg["layer"]
        t_arm_start = time.time()

        print(f"\n=========================================================================================")
        print(f" STAGE V VERIFICATION ARM: {arm_name} across 6 seeds x 200 edits")
        print(f"=========================================================================================")

        arm_seed_records = []
        arm_ctrl_records = []
        all_arm_preds = []
        for s_idx in ACTIVE_SEEDS:
            configure_determinism(seed=s_idx)
            model.load_state_dict(base_state_dict)
            verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

            facts_seq, _ = sample_200_facts(facts_1000, seed=s_idx)
            imm_matches = []
            steps_taken = []
            rho_vals = []

            if a_type == "null_corr":
                tracker = CorrectedSequentialNullTracker(arm_cfg["v_null"], rel_threshold=NULL_SPACE_REL_THRESH, device=device)

            for edit_idx, fact in enumerate(facts_seq):
                if a_type == "cov":
                    res = edit_fact_mlp_cov(model, tokenizer, fact, arm_cfg["cov"], layer_idx=l_idx, max_steps=100, device=device)
                elif a_type == "null_corr":
                    res = edit_fact_mlp_null_corrected(model, tokenizer, fact, tracker, layer_idx=l_idx, max_steps=100, device=device)
                    rho_vals.append(res["rho"])

                imm_matches.append(res["immediate_match"])
                steps_taken.append(res["steps_taken"])
                total_steps += res["steps_taken"]
                total_samples += 1

            # Full perplexity
            full_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)

            # Evaluate sequential terminal retention on first 50 facts
            term_preds = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_seq[:50]]
            term_matches = [check_match(p, f["object"]) for p, f in zip(term_preds, facts_seq[:50])]
            all_arm_preds.extend(term_preds)

            # Evaluate paraphrase retention on first 50 facts
            para_matches = []
            for f in facts_seq[:50]:
                for p in f["paraphrases"]:
                    p_pred = greedy_predict(model, tokenizer, p, 5, device, False)
                    para_matches.append(check_match(p_pred, f["object"]))

            # Evaluate sequential negative controls on this seed's sequential state
            ctrl_eval = evaluate_s0_12_sequential_controls(model, base_model, tokenizer, facts_seq[:50], facts_1000, device=device, seed=s_idx)
            arm_ctrl_records.append(ctrl_eval)

            s_rec = {
                "seed": s_idx,
                "full_ppl": full_ppl,
                "immediate_matches": imm_matches,
                "first50_terminal_matches": term_matches,
                "first50_paraphrase_matches": para_matches
            }
            if a_type == "null_corr":
                s_rec["rho_min"] = min(rho_vals)
                s_rec["rho_median"] = float(torch.median(torch.tensor(rho_vals)).item())
                s_rec["rho_max"] = max(rho_vals)
            arm_seed_records.append(s_rec)
            print(f"  Seed {s_idx} Complete: Imm={sum(imm_matches)}/200, F50Term={sum(term_matches)}/50, FullPPL={full_ppl:.2f}")

            # Collect Seed 0 lost facts for Stage L
            if arm_name == "A-null_L1_corr" and s_idx == 0:
                for f_idx, (im, tm) in enumerate(zip(imm_matches[:50], term_matches)):
                    if im and not tm:
                        seed0_lost_facts.append(facts_seq[f_idx])

        t_arm_elapsed = time.time() - t_arm_start
        measured_arm_times[arm_name] = t_arm_elapsed

        # Pool counts
        pooled_imm = sum(sum(r["immediate_matches"]) for r in arm_seed_records)
        pooled_f50_term = sum(sum(r["first50_terminal_matches"]) for r in arm_seed_records)
        pooled_f50_para = sum(sum(r["first50_paraphrase_matches"]) for r in arm_seed_records)

        # Modal Collapse Audit (§3)
        collapsed, dom_token, dom_frac = check_modal_collapse(all_arm_preds)
        print(f"\n--- [Modal-Collapse Audit: {arm_name}] ---")
        print(f"  Top Dominant Output  : '{dom_token}' ({dom_frac*100.0:.2f}%)")
        print(f"  Modal Collapse Status: {'COLLAPSED (>= 50% identical)' if collapsed else 'HEALTHY (Diverse predictions)'}")
        assert not collapsed, f"Modal collapse detected on arm {arm_name}: {dom_frac*100.0:.1f}% on '{dom_token}'"

        # Gate V1 Reproduction Audit (Amendment 1 §A: Recorded as FAILED / superseded by S0-12)
        s0_11_arm = s0_11_data["stage_s"].get(arm_name, {})
        ref_imm = s0_11_arm.get("pooled_imm_eff", [None])[0]
        ref_f50 = s0_11_arm.get("primary_endpoint_e2", {}).get("num")
        print(f"\n--- [Gate V1 Reproduction Audit: {arm_name}] ---")
        print(f"  Immediate Efficacy : Observed {pooled_imm}/1200 | Reference {ref_imm}/1200")
        print(f"  First-50 Retention : Observed {pooled_f50_term}/300 | Reference {ref_f50}/300")
        gate_v1_pass = (pooled_imm == ref_imm and pooled_f50_term == ref_f50)
        print(f"  Gate V1 Status     : {'EXACT MATCH' if gate_v1_pass else 'FAILED / DIVERGED (Superseded by S0-12 per Amendment 1 §A)'}")

        # Pool sequential control floors
        wrong_c_total = sum(sum(1 for x in c["wrong_target_canonical"] if x) for c in arm_ctrl_records)
        wrong_p_total = sum(sum(1 for x in c["wrong_target_paraphrase"] if x) for c in arm_ctrl_records)
        never_c_total = sum(sum(1 for x in c["never_edited_canonical"] if x) for c in arm_ctrl_records)
        never_p_total = sum(sum(1 for x in c["never_edited_paraphrase"] if x) for c in arm_ctrl_records)

        # Amendment 1 §D: Primary floor = max(never_edited, pre_edit_baseline, sham_sequence)
        prim_c_candidates = {
            "never_edited": never_c_total,
            "pre_edit_baseline": pre_c_k,
            "sham_sequence": sham_c_scaled
        }
        prim_p_candidates = {
            "never_edited": never_p_total,
            "pre_edit_baseline": pre_p_k,
            "sham_sequence": sham_p_scaled
        }

        prim_c_name, prim_c_floor = max(prim_c_candidates.items(), key=lambda item: item[1])
        prim_p_name, prim_p_floor = max(prim_p_candidates.items(), key=lambda item: item[1])

        d_term, lo_term, hi_term = newcombe_score_interval(pooled_f50_term, 300, prim_c_floor, 300)
        d_para, lo_para, hi_para = newcombe_score_interval(pooled_f50_para, 900, prim_p_floor, 900)
        v_term = "ABOVE" if lo_term > 0 else ("BELOW" if hi_term < 0 else "AT")
        v_para = "ABOVE" if lo_para > 0 else ("BELOW" if hi_para < 0 else "AT")

        # Object recurrence breakdown
        rec_ret, nonrec_ret = 0, 0
        rec_tot, nonrec_tot = 0, 0
        for s_idx, r in enumerate(arm_seed_records):
            f_seq, _ = sample_200_facts(facts_1000, seed=s_idx)
            later_objs = {normalize_entity(f["object"]) for f in f_seq[50:]}
            for f_idx, tm in enumerate(r["first50_terminal_matches"]):
                f_obj = normalize_entity(f_seq[f_idx]["object"])
                if f_obj in later_objs:
                    rec_tot += 1
                    if tm: rec_ret += 1
                else:
                    nonrec_tot += 1
                    if tm: nonrec_ret += 1

        print(f"\n--- [Comparative Retention vs Sequential Controls (Amendment 1 §D)] ---")
        print(f"  Candidate Canonical Floors      : never_edited={never_c_total}/300, pre_edit={pre_c_k}/300, sham={sham_c_scaled}/300")
        print(f"  Primary Canonical Floor Used    : {prim_c_name} = {prim_c_floor}/300 ({prim_c_floor/300.0*100.0:.2f}%)")
        print(f"  Candidate Paraphrase Floors     : never_edited={never_p_total}/900, pre_edit={pre_p_k}/900, sham={sham_p_scaled}/900")
        print(f"  Primary Paraphrase Floor Used   : {prim_p_name} = {prim_p_floor}/900 ({prim_p_floor/900.0*100.0:.2f}%)")
        print(f"  Secondary Floor (wrong_target)  : Canonical {wrong_c_total}/300, Paraphrase {wrong_p_total}/900")
        print(f"  E2 Terminal Retention Verdict   : {pooled_f50_term}/300 vs {prim_c_floor}/300 (Diff {d_term*100.0:+.2f} pp, Verdict: {v_term})")
        print(f"  E3 Paraphrase Retention Verdict : {pooled_f50_para}/900 vs {prim_p_floor}/900 (Diff {d_para*100.0:+.2f} pp, Verdict: {v_para})")
        print(f"  Object Recurrence Breakdown     : Recurring {rec_ret}/{rec_tot} ({rec_ret/rec_tot*100.0:.2f}%) | Non-recurring {nonrec_ret}/{nonrec_tot} ({nonrec_ret/nonrec_tot*100.0:.2f}%)")

        arm_res = {
            "name": arm_name,
            "layer": l_idx,
            "type": a_type,
            "wall_clock": t_arm_elapsed,
            "gate_v1_reproduction": gate_v1_pass,
            "pooled_imm_eff": [pooled_imm, 1200],
            "primary_endpoint_e2": {
                "num": pooled_f50_term, "den": 300, "rate": pooled_f50_term / 300.0,
                "floor_num": prim_c_floor, "floor_den": 300, "floor_rate": prim_c_floor / 300.0,
                "diff": d_term, "ci_lo": lo_term, "ci_hi": hi_term, "verdict": v_term
            },
            "secondary_endpoint_e3": {
                "num": pooled_f50_para, "den": 900, "rate": pooled_f50_para / 900.0,
                "floor_num": prim_p_floor, "floor_den": 900, "floor_rate": prim_p_floor / 900.0,
                "diff": d_para, "ci_lo": lo_para, "ci_hi": hi_para, "verdict": v_para
            },
            "controls": {
                "wrong_target_canonical": wrong_c_total,
                "wrong_target_paraphrase": wrong_p_total,
                "never_edited_canonical": never_c_total,
                "never_edited_paraphrase": never_p_total,
                "pre_edit_canonical": pre_c_k,
                "pre_edit_paraphrase": pre_p_k
            },
            "recurrence_breakdown": {
                "recurring_num": rec_ret, "recurring_den": rec_tot,
                "non_recurring_num": nonrec_ret, "non_recurring_den": nonrec_tot
            },
            "raw_seeds": arm_seed_records
        }
        stage_v_results[arm_name] = arm_res
        results_payload["stage_v"] = stage_v_results
        save_incremental_artifact(out_json_path, results_payload)

        if arm_name == "A-null_L1_corr" and v_term == "ABOVE":
            stage_v_confirmed = True

    # Stage L: Activation Patching Loss Localization
    stage_l_res = {}
    if stage_v_confirmed:
        print("\n=========================================================================================")
        print(f" STAGE L: ACTIVATION PATCHING LOSS LOCALIZATION (Seed 0, {len(seed0_lost_facts)} Lost Facts)")
        print("=========================================================================================")

        # Restore Seed 0 sequential state of A-null_L1_corr
        configure_determinism(seed=0)
        model.load_state_dict(base_state_dict)
        facts_seq0, _ = sample_200_facts(facts_1000, seed=0)
        tracker_l = CorrectedSequentialNullTracker(p_info_l1["v_null"], rel_threshold=NULL_SPACE_REL_THRESH, device=device)
        for fact in facts_seq0:
            _ = edit_fact_mlp_null_corrected(model, tokenizer, fact, tracker_l, layer_idx=1, max_steps=100, device=device)

        stage_l_res = run_stage_l_patching(model, base_model, tokenizer, seed0_lost_facts, layer_idx=1, device=device)
        print(f"  Lost Facts Analyzed             : {stage_l_res['n_lost']}")
        print(f"  Condition (a) Subject-Only      : {stage_l_res['subj_recovered_count']}/{stage_l_res['n_lost']} ({stage_l_res['subj_recovery_rate']*100.0:.2f}%)")
        print(f"  Condition (b) Non-Subject Only  : {stage_l_res['non_subj_recovered_count']}/{stage_l_res['n_lost']} ({stage_l_res['non_subj_recovery_rate']*100.0:.2f}%)")
        print(f"  Condition (c) All Positions     : {stage_l_res['all_recovered_count']}/{stage_l_res['n_lost']} ({stage_l_res['all_recovery_rate']*100.0:.2f}%)")
        print(f"  Mean Subject Key Drift          : {stage_l_res['mean_key_drift']:.4e}")

        results_payload["stage_l"] = stage_l_res
        save_incremental_artifact(out_json_path, results_payload)

    # Stage F: Full-Prompt Protection (A-null_L1_full)
    stage_f_res = {}
    non_subj_implicated = stage_l_res.get("non_subj_recovery_rate", 0.0) > 0.10

    # Dynamic Budget Guard for Stage F
    elapsed_total = time.time() - t_start_total
    est_stage_f = measured_arm_times.get("A-null_L1_corr", 3000.0) * 1.50
    budget_ok = (elapsed_total + est_stage_f) <= COMPUTE_CEILING_SEC

    print("\n--- [Stage F Gate & Budget Check] ---")
    print(f"  Non-Subject Implication Gate   : {'TRIGGERED' if non_subj_implicated else 'SKIPPED'}")
    print(f"  Budget Check                    : Elapsed {elapsed_total:.1f} s + Projected {est_stage_f:.1f} s <= Ceiling {COMPUTE_CEILING_SEC:.1f} s ({budget_ok})")

    if non_subj_implicated and budget_ok:
        print("\n=========================================================================================")
        print(" STAGE F: FULL-PROMPT PROTECTION (A-null_L1_full across 6 seeds x 200 edits)")
        print("=========================================================================================")

        f_seed_records = []
        for s_idx in ACTIVE_SEEDS:
            configure_determinism(seed=s_idx)
            model.load_state_dict(base_state_dict)
            facts_seq, _ = sample_200_facts(facts_1000, seed=s_idx)
            imm_matches = []
            f_tracker = FullPromptSequentialNullTracker(p_info_l1["v_null"], rel_threshold=NULL_SPACE_REL_THRESH, device=device)

            for edit_idx, fact in enumerate(facts_seq):
                res = edit_fact_mlp_null_full_prompt(model, tokenizer, fact, f_tracker, layer_idx=1, max_steps=100, device=device)
                imm_matches.append(res["immediate_match"])
                total_steps += res["steps_taken"]
                total_samples += 1

            full_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)
            term_preds = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_seq[:50]]
            term_matches = [check_match(p, f["object"]) for p, f in zip(term_preds, facts_seq[:50])]

            f_seed_records.append({
                "seed": s_idx,
                "full_ppl": full_ppl,
                "immediate_matches": imm_matches,
                "first50_terminal_matches": term_matches,
                "capacity_exhausted": f_tracker.capacity_exhausted,
                "dim_k": f_tracker.dim_k
            })
            print(f"  Seed {s_idx} Complete: Imm={sum(imm_matches)}/200, F50Term={sum(term_matches)}/50, FullPPL={full_ppl:.2f}, ProtectedBasisDim={f_tracker.dim_k}")

        pooled_imm_f = sum(sum(r["immediate_matches"]) for r in f_seed_records)
        pooled_term_f = sum(sum(r["first50_terminal_matches"]) for r in f_seed_records)
        stage_f_res = {
            "name": "A-null_L1_full",
            "pooled_imm_eff": [pooled_imm_f, 1200],
            "pooled_f50_term": [pooled_term_f, 300],
            "raw_seeds": f_seed_records
        }
        results_payload["stage_f"] = stage_f_res
        save_incremental_artifact(out_json_path, results_payload)
    else:
        results_payload["stage_f"] = {
            "skipped": True,
            "reason": "Non-subject positions not implicated" if not non_subj_implicated else "Compute ceiling budget constraint"
        }
        save_incremental_artifact(out_json_path, results_payload)

    # Finalize Accounting
    total_wall_clock = time.time() - t_start_total
    results_payload["accounting"]["actual_wall_clock"] = total_wall_clock
    results_payload["accounting"]["total_optimizer_steps"] = total_steps
    results_payload["accounting"]["total_samples_seen"] = total_samples
    save_incremental_artifact(out_json_path, results_payload)

    print("\n" + "=" * 115)
    print(f" DIRECTIVE S0-12 RUN COMPLETE: Wall-clock {total_wall_clock:.2f} s, Exit Code 0")
    print(f" Artifact Serialized to: {out_json_path}")
    print("=" * 115)
    return 0


if __name__ == "__main__":
    sys.exit(run_s0_12_master())
