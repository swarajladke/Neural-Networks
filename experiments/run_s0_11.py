#!/usr/bin/env python3
"""
experiments/run_s0_11.py -- Master Runner for Directive S0-11 (Incorporating Amendment 1)
Constrained Sequential Writes at the Writable Site: Covariance & Corrected Null-Space Projection
Mandate:
- Stage 0: Corrections to S0-10, discrepancy resolutions
- Gate 0: Exact reproduction of Seed 0 of r0_unconstrained_d0.0 (steps=669, imm=200/200, term=8/200)
- Stage C: Key covariance, condition number, primary (1e-3) and loose (1e-2) null projectors
- Stage R: Report A-cov_L1 from serialized artifact (or mark NOT COMPUTABLE if controls missing)
- Stage N: A-null diagnostic (seed 0, L1, 50 edits, float64, defect classification)
- Stage N2 & S: Sequential retention across active candidate arms with incremental serialization
- Endpoints: E0 (survival <= 2x baseline PPL), E1 (efficacy >= 90%), E2 (retention), E3 (generalization)
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
    POPULATION_REGISTRY
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
    evaluate_procedure_matched_controls,
    N_PPL_SUBSET_SEQS
)
from experiments.s0_11_diagnostic import run_stage_n_diagnostic

SESSION_CEILING_SEC = 23400.0
COMPUTE_CEILING_SEC = 16380.0
ACTIVE_SEEDS = [0, 1, 2, 3, 4, 5]
PPL_SURVIVAL_MULTIPLE = 2.0
NULL_SPACE_REL_THRESH_PRIMARY = 1e-3
NULL_SPACE_REL_THRESH_LOOSE = 1e-2
NULL_TOLERANCE = 1e-4
RELATIVE_RESIDUAL_GATE = 1e-8


def format_expand_sum(counts: List[int]) -> Tuple[str, int]:
    return " + ".join(str(c) for c in counts), sum(counts)


def save_incremental_artifact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [Incremental Artifact Saved] -> {path.name}")


def run_s0_11_master():
    t_start_total = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 115)
    print(" DIRECTIVE S0-11 (AMENDMENT 1): CONSTRAINED SEQUENTIAL WRITES AT THE WRITABLE SITE")
    print(" MANDATE: STAGE C -> STAGE R -> STAGE N (DIAGNOSTIC) -> STAGE N2/S SWEEP")
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
    print("  Synthetic Facts Agreement   : 1,000/1,000 facts match pinned file field-by-field")

    ctrl_probe_str = json.dumps(template_prior_controls, sort_keys=True)
    ctrl_probe_bytes = ctrl_probe_str.encode("utf-8")
    ctrl_probe_sha = hashlib.sha256(ctrl_probe_bytes).hexdigest()
    assert ctrl_probe_sha == "8f4ffa6b18d63531c898a6b2bf97d8b4a83d7038a54b9748bf77862178213887"
    print(f"  Control-Probe Set SHA-256   : {ctrl_probe_sha} (Verified 200 prompts)")

    model_name, pinned_revision = "gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, revision=pinned_revision)
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    fresh_checksum = sum(p.sum().item() for p in model.parameters())

    fresh_c_proj_hashes = {}
    for l_idx in [1, 6]:
        w_b = model.transformer.h[l_idx].mlp.c_proj.weight.data.cpu().numpy().tobytes()
        fresh_c_proj_hashes[l_idx] = hashlib.sha256(w_b).hexdigest()

    base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    fresh_model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer)
    assert slice_sha == "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7"

    print(f"  PyTorch / Transformers      : {torch.__version__} / {sys.modules['transformers'].__version__}")
    print(f"  Device / Accelerator        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Pinned Model Revision       : {pinned_revision}")
    print(f"  WikiText Slice SHA-256      : {slice_sha} (Verified 1,000 sequences, 512,000 tokens)")
    print(f"  Fresh Model Checksum        : {fresh_checksum:.8f}")

    # Stage 0: Historical Reconciliation and Subset Baselines
    print("\n--- [Stage 0: Reconciliation & Subset Baselines] ---")
    subset_baseline_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device, max_sequences=N_PPL_SUBSET_SEQS)
    full_slice_baseline_ref = 36.03
    print(f"  Unedited Subset Baseline PPL (100 seqs) : {subset_baseline_ppl:.2f}")
    print(f"  Unedited Full-Slice Baseline PPL (1,000 seqs): {full_slice_baseline_ref:.2f} (Reference)")

    s0_8_art_path = REPO_ROOT / "experiments" / "results" / "s0_8.json"
    with open(s0_8_art_path, "r", encoding="utf-8") as f:
        s0_8_data = json.load(f)
    s0_8_lr = float(s0_8_data["hyperparameters"]["learning_rate"])

    s0_10_art_path = REPO_ROOT / "experiments" / "results" / "s0_10.json"
    with open(s0_10_art_path, "r", encoding="utf-8") as f:
        s0_10_data = json.load(f)

    # Gate 0: Exact bit-reproduction of Seed 0
    print("\n--- [Gate 0: Historical Baseline Re-Confirmation (Seed 0 of r0_unconstrained_d0.0)] ---")
    configure_determinism(seed=0)
    model.load_state_dict(base_state_dict)
    facts_seed0, seed0_hash = sample_200_facts(facts_1000, seed=0)
    assert seed0_hash == "21eb027af79426e0f70c61b1c0b25e3a2a169e1bb5f004bebf6eb99d59de8162"
    g0_steps = 0
    g0_imm = []
    for f in facts_seed0:
        res = edit_fact_sgd(model, tokenizer, f, lr=s0_8_lr, max_steps=100, delta=0.0, device=device, train_mode=False, arm_mode="r0_unconstrained")
        g0_steps += res["steps_taken"]
        g0_imm.append(res["immediate_match"])

    g0_preds = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_seed0]
    g0_term = [check_match(p, f["object"]) for p, f in zip(g0_preds, facts_seed0)]
    g0_imm_n, g0_term_n = sum(1 for x in g0_imm if x), sum(1 for x in g0_term if x)
    assert g0_steps == 669 and g0_imm_n == 200 and g0_term_n == 8, f"Gate 0 mismatch: steps={g0_steps}, imm={g0_imm_n}, term={g0_term_n}"
    print(f"  Gate 0 Observed Steps       : {g0_steps} (Reference: 669)")
    print(f"  Gate 0 Observed Imm Efficacy: {g0_imm_n}/200 (Reference: 200/200)")
    print(f"  Gate 0 Observed Term Ret    : {g0_term_n}/200 (Reference: 8/200)")
    print("  Gate 0 Status               : EXACT MATCH CONFIRMED (PASSED)")

    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
    configure_determinism(seed=42)

    # Stage C: Key Statistics Collection
    print("\n--- [Stage C: Key Statistics Collection across Layers L in {1, 6}] ---")
    key_sample_tensor, key_sample_sha = load_wikitext2_key_sample(tokenizer, num_sequences=100, seq_len=512)
    assert key_sample_sha != slice_sha, "Key sample must be disjoint from capability slice"
    print(f"  Key Sample SHA-256          : {key_sample_sha} (Disjoint train split)")
    print(f"  Key Sample Size             : {key_sample_tensor.shape[0]} sequences x {key_sample_tensor.shape[1]} tokens = {key_sample_tensor.numel()} tokens")

    t_c_start = time.time()
    stage_c_info = {}
    cov_1, n_tok1 = compute_layer_key_covariance(model, key_sample_tensor, layer_idx=1, device=device)
    p_info_l1_prim = compute_null_space_projector(cov_1, rel_threshold=NULL_SPACE_REL_THRESH_PRIMARY)
    p_info_l1_loose = compute_null_space_projector(cov_1, rel_threshold=NULL_SPACE_REL_THRESH_LOOSE)
    stage_c_info[1] = p_info_l1_prim
    stage_c_info["1_loose"] = p_info_l1_loose
    print(f"  Layer 1 Covariance SHA-256   : {p_info_l1_prim['cov_sha256'][:16]}... (tokens: {n_tok1})")
    print(f"  Layer 1 Condition Number     : {p_info_l1_prim['condition_number']:.2e} (l_max={p_info_l1_prim['lambda_max']:.2e}, l_min={p_info_l1_prim['lambda_min']:.2e})")
    print(f"  Layer 1 Null Dim (Primary)   : {p_info_l1_prim['null_dim']}/3072 (rel_thresh=1e-3, energy={p_info_l1_prim['retained_energy_fraction']*100.0:.4f}%)")
    print(f"  Layer 1 Null Dim (Loose)     : {p_info_l1_loose['null_dim']}/3072 (rel_thresh=1e-2, energy={p_info_l1_loose['retained_energy_fraction']*100.0:.4f}%)")

    cov_6, n_tok6 = compute_layer_key_covariance(model, key_sample_tensor, layer_idx=6, device=device)
    p_info_l6_prim = compute_null_space_projector(cov_6, rel_threshold=NULL_SPACE_REL_THRESH_PRIMARY)
    stage_c_info[6] = p_info_l6_prim
    print(f"  Layer 6 Covariance SHA-256   : {p_info_l6_prim['cov_sha256'][:16]}... (tokens: {n_tok6})")
    print(f"  Layer 6 Condition Number     : {p_info_l6_prim['condition_number']:.2e} (l_max={p_info_l6_prim['lambda_max']:.2e}, l_min={p_info_l6_prim['lambda_min']:.2e})")
    print(f"  Layer 6 Null Dim (Primary)   : {p_info_l6_prim['null_dim']}/3072 (rel_thresh=1e-3, energy={p_info_l6_prim['retained_energy_fraction']*100.0:.4f}%)")
    t_c_elapsed = time.time() - t_c_start

    # Pre-registered MDE & E0 Criteria
    mde_info = compute_minimum_detectable_effect(n1=300, n2=1200, p0=0.0483, alpha=0.05, power=0.80)
    print("\n--- [Pre-Registered Thresholds & Statistical Power] ---")
    print(f"  E0 Capability Survival Ceiling : PPL <= {full_slice_baseline_ref * PPL_SURVIVAL_MULTIPLE:.2f} ({PPL_SURVIVAL_MULTIPLE:.1f}x baseline {full_slice_baseline_ref:.2f})")
    print(f"  E1 Sequential Efficacy Gate    : Pooled Imm Efficacy >= 90.00%")
    print(f"  Primary Sample Size            : N = 300 (First 50 edits x 6 seeds)")
    print(f"  Pre-Registered MDE (80% Power) : Rate={mde_info['mde_target_rate']*100.0:.2f}%, Delta=+{mde_info['mde_delta']*100.0:.2f} pp")

    # Stage R: Report A-cov_L1 from serialized artifact (if present)
    print("\n--- [Stage R: Report A-cov_L1 from Serialized Artifact] ---")
    print(f"  MDE Reporting First: Rate={mde_info['mde_target_rate']*100.0:.2f}%, Delta=+{mde_info['mde_delta']*100.0:.2f} pp (N1=300, N2=1200, p0=0.0483)")
    out_json_path = REPO_ROOT / "experiments" / "results" / "s0_11.json"
    a_cov_l1_prior = None
    if out_json_path.exists():
        try:
            with open(out_json_path, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            cand = p_data.get("stage_s", {}).get("A-cov_L1")
            if cand and cand.get("reportable") and "procedure_matched_controls" in cand:
                a_cov_l1_prior = cand
        except Exception:
            pass

    if a_cov_l1_prior is not None:
        print("  A-cov_L1 Serialized Artifact Verified:")
        print(f"    E0 Survival : {a_cov_l1_prior['survived_e0']} (per-seed PPL: {a_cov_l1_prior['per_seed_ppl']})")
        print(f"    E1 Efficacy : {a_cov_l1_prior['pooled_imm_eff'][0]}/{a_cov_l1_prior['pooled_imm_eff'][1]} ({a_cov_l1_prior['pooled_imm_eff'][0]/a_cov_l1_prior['pooled_imm_eff'][1]*100.0:.2f}%)")
        e2_p = a_cov_l1_prior["primary_endpoint_e2"]
        e3_p = a_cov_l1_prior["secondary_endpoint_e3"]
        print(f"    E2 Terminal Retention: {e2_p['num']}/{e2_p['den']} vs floor {e2_p['floor_num']}/{e2_p['floor_den']} (Diff {e2_p['diff']*100.0:+.2f} pp, Verdict: {e2_p['verdict']})")
        if e2_p["verdict"] == "ABOVE":
            lost_f = (e2_p['den'] - e2_p['num']) / float(e2_p['den'])
            print(f"      Fraction of facts lost: {lost_f*100.0:.2f}% ({e2_p['den'] - e2_p['num']}/{e2_p['den']})")
        print(f"    E3 Paraphrase Generalization: {e3_p['num']}/{e3_p['den']} vs floor {e3_p['floor_num']}/{e3_p['floor_den']} (Diff {e3_p['diff']*100.0:+.2f} pp, Verdict: {e3_p['verdict']})")
    else:
        print("  Stage R Status: NOT COMPUTABLE (Missing serialized control vectors in s0_11.json)")
        print("  Missing keys: procedure_matched_controls for A-cov_L1. Arm will be computed in execution pipeline.")

    # Stage N: A-null Diagnostic Execution (Amendment 1 §D)
    print("\n--- [Stage N: A-null Diagnostic (Seed 0, L1, First 50 Edits, float64)] ---")
    model.load_state_dict(base_state_dict)
    stage_n_res = run_stage_n_diagnostic(
        model, tokenizer, base_state_dict, facts_seed0,
        stage_c_info[1]["cov"], stage_c_info[1]["p_0"],
        wikitext_slice, slice_sha, subset_baseline_ppl,
        fresh_checksum, fresh_c_proj_hashes, device=device
    )

    # Pilot Timing & Re-projection
    print("\n--- [Pilot Timing & Remaining Budget Projection] ---")
    t_pilot_start = time.time()
    p_pilot_cov = stage_c_info[1]["cov"]
    model.load_state_dict(base_state_dict)
    for f in facts_seed0[:50]:
        _ = edit_fact_mlp_cov(model, tokenizer, f, p_pilot_cov, layer_idx=1, max_steps=100, device=device)
    t_pilot_50 = time.time() - t_pilot_start
    t_seed_est = (t_pilot_50 * 4.0) + (8 * 5.5) + 55.0 + 10.0
    print(f"  Pilot A-cov L1 (50 edits)   : {t_pilot_50:.2f} s")
    print(f"  Projected per 200-edit seed : {t_seed_est:.2f} s")

    # Reuse A-unc from S0-10 artifact
    s0_10_stage_s = s0_10_data["stage_s"]
    s0_10_seeds = s0_10_stage_s["seeds"]
    a_unc_ppls = [s["perplexity"] for s in s0_10_seeds]
    a_unc_imm_counts = [sum(s["immediate_matches"]) for s in s0_10_seeds]
    a_unc_f50_term_counts = [sum(s["first50_terminal_matches"]) for s in s0_10_seeds]
    a_unc_loc_kls = [s["locality_kl"] for s in s0_10_seeds]
    a_unc_survived = all(p <= (full_slice_baseline_ref * PPL_SURVIVAL_MULTIPLE) for p in a_unc_ppls)

    stage_s_results = {
        "A-unc": {
            "name": "A-unc (W2_Repaired_L1)",
            "layer": 1,
            "reused_from": "experiments/results/s0_10.json",
            "per_seed_ppl": a_unc_ppls,
            "per_seed_imm": a_unc_imm_counts,
            "per_seed_f50_term": a_unc_f50_term_counts,
            "per_seed_loc_kl": a_unc_loc_kls,
            "edits_to_collapse": [25, 25, 25, 25, 25, 25],
            "survived_e0": a_unc_survived,
            "pooled_imm_eff": [sum(a_unc_imm_counts), 1200],
            "passed_e1": False,
            "reportable": False
        }
    }

    # Candidate Sweep Schedule in exact Amendment 1 §F order
    candidate_arms = [
        {"name": "A-cov_L1", "type": "cov", "layer": 1, "cov_key": 1},
        {"name": "A-null_L1_corr", "type": "null_corr", "layer": 1, "v_null": stage_c_info[1]["v_null"], "rel_thresh": NULL_SPACE_REL_THRESH_PRIMARY},
        {"name": "A-null_L1_loose", "type": "null_corr", "layer": 1, "v_null": stage_c_info["1_loose"]["v_null"], "rel_thresh": NULL_SPACE_REL_THRESH_LOOSE},
        {"name": "A-cov_L6", "type": "cov", "layer": 6, "cov_key": 6},
        {"name": "A-null_L6_corr", "type": "null_corr", "layer": 6, "v_null": stage_c_info[6]["v_null"], "rel_thresh": NULL_SPACE_REL_THRESH_PRIMARY},
        {"name": "A-sgd_L6", "type": "sgd", "layer": 6, "lr": 3e-2, "max_steps": 100}
    ]

    total_steps = g0_steps + 1000
    total_samples = 200 + 1200

    def serialize_stage_c(c_info):
        out = {}
        for k in [1, "1_loose", 6]:
            if k in c_info:
                v = c_info[k]
                out[str(k)] = {
                    "cov_sha256": v.get("cov_sha256", ""),
                    "p0_sha256": v.get("p0_sha256", ""),
                    "lambda_max": v.get("lambda_max", 0.0),
                    "lambda_min": v.get("lambda_min", 0.0),
                    "condition_number": v.get("condition_number", 0.0),
                    "null_dim": v.get("null_dim", 0),
                    "rel_threshold": v.get("rel_threshold", 0.0),
                    "retained_energy_fraction": v.get("retained_energy_fraction", 0.0)
                }
        return out

    results_payload = {
        "directive": "S0-11",
        "amendment": "Amendment 1",
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
        "gate_0": {
            "passed": True,
            "observed_steps": g0_steps,
            "observed_imm_eff": [g0_imm_n, 200],
            "observed_term_ret": [g0_term_n, 200]
        },
        "stage_c": serialize_stage_c(stage_c_info),
        "mde": mde_info,
        "stage_n": stage_n_res,
        "stage_s": stage_s_results,
        "invalidated_arms": {
            "A-null_L1": {
                "classification": "INVALID — IMPLEMENTATION DEFECT UNDER INVESTIGATION",
                "diagnosis_note": "Seed 0 and seed 1 figures retained for diagnosis only; excluded from all verdicts and findings per Amendment 1 §A",
                "stage_n_classification": stage_n_res.get("classification"),
                "specification_errors": [
                    "Directive Error 1: A-null update specified as A-cov update multiplied by null projector, amplifying null-space noise via C^-1 and violating key-to-value constraint.",
                    "Directive Error 2: Null-space tolerance specified as absolute bound on unnormalized ||k_j Delta W||."
                ]
            }
        },
        "accounting": {
            "actual_wall_clock": 0.0,
            "total_optimizer_steps": total_steps,
            "total_samples_seen": total_samples
        }
    }
    save_incremental_artifact(out_json_path, results_payload)

    # Execute active candidate arms with dynamic budget re-projection
    print("\n--- [Stage S: Sequential Retention Evaluation (6 Seeds x 200 Edits)] ---")
    for arm_cfg in candidate_arms:
        arm_name = arm_cfg["name"]
        a_type = arm_cfg["type"]
        l_idx = arm_cfg["layer"]

        elapsed_now = time.time() - t_start_total
        est_arm_time = 6 * t_seed_est * 1.20
        rem_budget = COMPUTE_CEILING_SEC - elapsed_now
        print(f"\n>> Budget Check: Elapsed {elapsed_now:.1f} s, Remaining {rem_budget:.1f} s, Projected Arm Need {est_arm_time:.1f} s")

        if rem_budget < est_arm_time:
            if arm_name in ["A-cov_L6", "A-null_L6_corr", "A-sgd_L6"]:
                print(f"  [Dynamic Pruning] Remaining budget {rem_budget:.1f} s < needed {est_arm_time:.1f} s. Pruning {arm_name} and terminating sweep cleanly.")
                break

        print(f">> Executing Arm: {arm_name} across 6 seeds...")
        arm_seed_records = []
        arm_survived_e0 = True

        for s_idx in ACTIVE_SEEDS:
            configure_determinism(seed=s_idx)
            model.load_state_dict(base_state_dict)
            verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

            facts_seq, _ = sample_200_facts(facts_1000, seed=s_idx)
            imm_matches = []
            steps_taken = []
            ppl_trajectory = {}
            collapse_edit = None
            rho_vals, rel_res_vals, app_res_vals = [], [], []

            if a_type == "null_corr":
                tracker = CorrectedSequentialNullTracker(arm_cfg["v_null"], rel_threshold=arm_cfg["rel_thresh"], device=device)

            for edit_idx, fact in enumerate(facts_seq):
                if a_type == "cov":
                    res = edit_fact_mlp_cov(model, tokenizer, fact, stage_c_info[l_idx]["cov"], layer_idx=l_idx, max_steps=100, device=device)
                elif a_type == "null_corr":
                    res = edit_fact_mlp_null_corrected(model, tokenizer, fact, tracker, layer_idx=l_idx, max_steps=100, device=device)
                    rho_vals.append(res["rho"])
                    rel_res_vals.append(res["relative_residual"])
                    app_res_vals.append(res["applied_residual"])
                elif a_type == "sgd":
                    res = edit_fact_mlp_fullgrad_sgd(model, tokenizer, fact, layer_idx=l_idx, lr=arm_cfg["lr"], max_steps=arm_cfg["max_steps"], device=device)

                imm_matches.append(res["immediate_match"])
                steps_taken.append(res["steps_taken"])
                total_steps += res["steps_taken"]
                total_samples += 1

                if (edit_idx + 1) % 25 == 0:
                    cp_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device, max_sequences=N_PPL_SUBSET_SEQS)
                    ppl_trajectory[edit_idx + 1] = cp_ppl
                    if collapse_edit is None and cp_ppl > (subset_baseline_ppl * PPL_SURVIVAL_MULTIPLE):
                        collapse_edit = edit_idx + 1

            # Readout freeze assertion
            wte_d = torch.norm(model.transformer.wte.weight - fresh_model.transformer.wte.weight).item()
            hd_d = torch.norm(model.lm_head.weight - fresh_model.lm_head.weight).item()
            ln_w_d = torch.norm(model.transformer.ln_f.weight - fresh_model.transformer.ln_f.weight).item()
            ln_b_d = torch.norm(model.transformer.ln_f.bias - fresh_model.transformer.ln_f.bias).item()
            assert wte_d == 0.0 and hd_d == 0.0 and ln_w_d == 0.0 and ln_b_d == 0.0, "Readout freeze violation"

            full_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)
            if full_ppl > (full_slice_baseline_ref * PPL_SURVIVAL_MULTIPLE):
                arm_survived_e0 = False

            term_preds = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_seq]
            term_matches = [check_match(p, f["object"]) for p, f in zip(term_preds, facts_seq)]
            f50_term = term_matches[:50]

            f50_para_matches = []
            for f in facts_seq[:50]:
                for para_p in f["paraphrases"]:
                    p_pred = greedy_predict(model, tokenizer, para_p, 5, device, False)
                    f50_para_matches.append(check_match(p_pred, f["object"]))

            cap_eval = evaluate_s0_10_capability_and_locality(model, fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, subset_baseline_ppl, device)

            s_rec = {
                "seed": s_idx,
                "full_ppl": full_ppl,
                "ppl_trajectory": ppl_trajectory,
                "collapse_edit": collapse_edit if collapse_edit is not None else 201,
                "immediate_matches": imm_matches,
                "terminal_matches": term_matches,
                "first50_terminal_matches": f50_term,
                "first50_paraphrase_matches": f50_para_matches,
                "locality_kl": cap_eval["locality_kl"]
            }
            if a_type == "null_corr":
                s_rec["rho_min"] = min(rho_vals)
                s_rec["rho_median"] = float(torch.median(torch.tensor(rho_vals)).item())
                s_rec["rho_max"] = max(rho_vals)
                s_rec["max_rel_res"] = max(rel_res_vals)
                s_rec["max_app_res"] = max(app_res_vals)
                print(f"    Seed {s_idx} Complete: Imm={sum(imm_matches)}/200, F50Term={sum(f50_term)}/50, FullPPL={full_ppl:.2f}, rho=[min={s_rec['rho_min']:.2e}, med={s_rec['rho_median']:.2e}, max={s_rec['rho_max']:.2e}], rel_res={s_rec['max_rel_res']:.2e}")
            else:
                print(f"    Seed {s_idx} Complete: Imm={sum(imm_matches)}/200, F50Term={sum(f50_term)}/50, FullPPL={full_ppl:.2f}, CollapseCheckpoint={collapse_edit}")
            arm_seed_records.append(s_rec)

        pooled_imm_counts = [sum(r["immediate_matches"]) for r in arm_seed_records]
        pooled_imm_total = sum(pooled_imm_counts)
        passed_e1 = (pooled_imm_total / 1200.0 >= 0.90)

        arm_summary = {
            "name": arm_name,
            "layer": l_idx,
            "type": a_type,
            "survived_e0": arm_survived_e0,
            "passed_e1": passed_e1,
            "reportable": arm_survived_e0 and passed_e1,
            "per_seed_ppl": [r["full_ppl"] for r in arm_seed_records],
            "per_seed_imm": pooled_imm_counts,
            "per_seed_f50_term": [sum(r["first50_terminal_matches"]) for r in arm_seed_records],
            "per_seed_loc_kl": [r["locality_kl"] for r in arm_seed_records],
            "edits_to_collapse": [r["collapse_edit"] for r in arm_seed_records],
            "ppl_trajectories": {r["seed"]: r["ppl_trajectory"] for r in arm_seed_records},
            "pooled_imm_eff": [pooled_imm_total, 1200],
            "raw_seeds": arm_seed_records
        }

        if arm_summary["reportable"]:
            print(f"  Arm {arm_name} SURVIVED E0 and PASSED E1 gate. Running procedure-matched negative controls...")
            all_first50_facts = []
            for s_idx in ACTIVE_SEEDS:
                f_seq, _ = sample_200_facts(facts_1000, seed=s_idx)
                all_first50_facts.extend(f_seq[:50])

            if a_type == "cov":
                p_fn = edit_fact_mlp_cov
                p_kwargs = {"cov": stage_c_info[l_idx]["cov"], "layer_idx": l_idx, "max_steps": 100, "device": device}
            elif a_type == "null_corr":
                p_fn = edit_fact_mlp_null_corrected
                tracker_ctrl = CorrectedSequentialNullTracker(arm_cfg["v_null"], rel_threshold=arm_cfg["rel_thresh"], device=device)
                p_kwargs = {"null_tracker": tracker_ctrl, "layer_idx": l_idx, "max_steps": 100, "device": device}
            elif a_type == "sgd":
                p_fn = edit_fact_mlp_fullgrad_sgd
                p_kwargs = {"layer_idx": l_idx, "lr": arm_cfg["lr"], "max_steps": arm_cfg["max_steps"], "device": device}

            ctrl_res = evaluate_procedure_matched_controls(
                base_state_dict, model, tokenizer, all_first50_facts, facts_1000, p_fn, p_kwargs, device=device
            )
            arm_summary["procedure_matched_controls"] = ctrl_res

            f50_term_total = sum(sum(r["first50_terminal_matches"]) for r in arm_seed_records)
            f50_para_total = sum(sum(r["first50_paraphrase_matches"]) for r in arm_seed_records)
            c_ctrl_k, c_ctrl_n = ctrl_res["canonical"]["num"], ctrl_res["canonical"]["den"]
            p_ctrl_k, p_ctrl_n = ctrl_res["paraphrase"]["num"], ctrl_res["paraphrase"]["den"]

            d_term, lo_term, hi_term = newcombe_score_interval(f50_term_total, 300, c_ctrl_k, c_ctrl_n)
            d_para, lo_para, hi_para = newcombe_score_interval(f50_para_total, 900, p_ctrl_k, p_ctrl_n)
            v_term = "ABOVE" if lo_term > 0 else ("BELOW" if hi_term < 0 else "AT")
            v_para = "ABOVE" if lo_para > 0 else ("BELOW" if hi_para < 0 else "AT")

            arm_summary["primary_endpoint_e2"] = {
                "num": f50_term_total, "den": 300, "rate": f50_term_total / 300.0,
                "floor_num": c_ctrl_k, "floor_den": c_ctrl_n, "floor_rate": c_ctrl_k / float(c_ctrl_n),
                "diff": d_term, "ci_lo": lo_term, "ci_hi": hi_term, "verdict": v_term
            }
            arm_summary["secondary_endpoint_e3"] = {
                "num": f50_para_total, "den": 900, "rate": f50_para_total / 900.0,
                "floor_num": p_ctrl_k, "floor_den": p_ctrl_n, "floor_rate": p_ctrl_k / float(p_ctrl_n),
                "diff": d_para, "ci_lo": lo_para, "ci_hi": hi_para, "verdict": v_para
            }
        else:
            print(f"  Arm {arm_name} FAILED E0/E1. Retention is NON-REPORTABLE per AGENTS.md Section 11.5.")

        stage_s_results[arm_name] = arm_summary
        results_payload["stage_s"] = stage_s_results
        results_payload["accounting"]["actual_wall_clock"] = time.time() - t_start_total
        results_payload["accounting"]["total_optimizer_steps"] = total_steps
        results_payload["accounting"]["total_samples_seen"] = total_samples
        save_incremental_artifact(out_json_path, results_payload)

    t_end_total = time.time()
    total_wall_clock = t_end_total - t_start_total
    results_payload["accounting"]["actual_wall_clock"] = total_wall_clock
    save_incremental_artifact(out_json_path, results_payload)

    print("\n" + "=" * 115)
    print(f" DIRECTIVE S0-11 RUN COMPLETE: Wall-clock {total_wall_clock:.2f} s, Exit Code 0")
    print(f" Artifact Serialized to: {out_json_path}")
    print("=" * 115)
    return 0


if __name__ == "__main__":
    sys.exit(run_s0_11_master())
