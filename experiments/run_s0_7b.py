#!/usr/bin/env python3
"""
experiments/run_s0_7b.py -- Master Orchestrator for Directive S0-7b
Mandate:
  - Strict stage execution order (Amendment 1 §I): J -> Gate 0 -> Remainder of F -> G -> H -> I.
  - Logs cumulative wall-clock after every stage.
  - Outputs experiments/results/s0_7b.json containing Rule 3.7 per-unit outcome vectors.
Strict structural limit: under 600 lines (AGENTS.md §7.1).
"""

import os
import gc
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (
    generate_synthetic_facts,
    sample_200_facts,
    load_wikitext2_slice,
    evaluate_wikitext_perplexity
)
from experiments.metrics import (
    Measurement,
    pool_controls,
    CONTROL_NAMES,
    wilson_confidence_interval
)
from tests.test_metrics import run_all_tests
from experiments.b1_inject import (
    configure_determinism,
    SEEDS
)
from experiments.stage_j import (
    audit_tokenization_conventions,
    print_stage_j_audit
)
from experiments.re_emission import (
    project_s0_7b_budget,
    print_budget_projection,
    run_gate_0_early,
    run_stage_f_re_emission
)
from experiments.s0_7b_audit import (
    audit_stage_g,
    audit_stage_h
)
from experiments.weight_tying import (
    run_stage_i_weight_tying
)


def main():
    global_start_time = time.time()
    print("=" * 115)
    print(" DIRECTIVE S0-7b: RE-EMISSION, ESTIMATOR REPAIR, POSITION-RESOLVED RETENTION, AND THE TYING CONFOUND")
    print(" MANDATE: STAGE J -> GATE 0 -> REMAINDER OF F -> G -> H -> I (AMENDMENT 1 ORDERING)")
    print("=" * 115)

    # 1. Pre-Flight Test Suite Execution
    print("\n--- [Pre-Flight Test Suite Execution (Directive S0-7b)] ---")
    if run_all_tests() != 0:
        print("FATAL: Pre-flight unit test suite failed. Halting before compute.")
        sys.exit(1)
    print("  Pre-Flight Test Suite Status : PASSED (Zero Failures, Zero AST Violations)")

    # 2. Environment & Input Hashes
    print("\n--- [Environment Fingerprint & Input Hashes] ---")
    configure_determinism(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    facts_file = REPO_ROOT / "b1_facts.json"
    assert facts_file.exists(), f"Facts file missing: {facts_file}"
    facts_bytes = facts_file.read_bytes()
    facts_sha = hashlib.sha256(facts_bytes).hexdigest()
    assert facts_sha == "285638ad25c07b22299153cd6e67e413d2ed4a226d0a4103076d2066763cb536"
    print(f"  Pinned Facts SHA-256        : {facts_sha} (Verified)")

    facts_1000, template_prior_controls = generate_synthetic_facts(num_facts=1000, seed=42)
    facts_pinned = json.loads(facts_bytes.decode("utf-8"))
    assert len(facts_1000) == len(facts_pinned) and all(
        facts_1000[i][k] == facts_pinned[i][k] for i in range(1000) for k in facts_pinned[i]
    )
    print("  Synthetic Facts Agreement   : 1,000/1,000 facts match pinned file field-by-field")

    ctrl_probe_bytes = json.dumps(template_prior_controls, sort_keys=True).encode("utf-8")
    ctrl_probe_sha = hashlib.sha256(ctrl_probe_bytes).hexdigest()
    assert ctrl_probe_sha == "8f4ffa6b18d63531c898a6b2bf97d8b4a83d7038a54b9748bf77862178213887"
    print(f"  Control-Probe Set SHA-256   : {ctrl_probe_sha} (Verified 200 prompts)")

    model_name, pinned_revision = "gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, revision=pinned_revision)
    fresh_model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    fresh_checksum = sum(p.sum().item() for p in fresh_model.parameters())

    weight_file = cached_file(model_name, "model.safetensors", revision=pinned_revision)
    assert weight_file and os.path.exists(weight_file)
    with open(weight_file, "rb") as f:
        weight_sha = hashlib.sha256(f.read()).hexdigest()
    assert weight_sha == "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
    print(f"  Weight File SHA-256         : {weight_sha} (Verified)")

    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer)
    assert slice_sha == "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7"
    print(f"  WikiText Slice SHA-256      : {slice_sha} (Verified)")

    ppl_tied_baseline = evaluate_wikitext_perplexity(fresh_model, wikitext_slice, slice_sha, device=device)
    print(f"  Pre-Edit WikiText-2 PPL     : {ppl_tied_baseline:.2f} (Pinned Baseline)")
    print(f"  Fresh Model Checksum        : {fresh_checksum:.8f}")
    print(f"  Device / PyTorch            : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) / {torch.__version__}")

    # Prepare seed sequences
    sequences = {}
    seed_sequence_hashes = {}
    for s in SEEDS:
        seq_facts, seq_h = sample_200_facts(facts_1000, seed=s)
        sequences[s] = seq_facts
        seed_sequence_hashes[s] = seq_h

    # 3. Empirical Budget Projection from S0-6
    s0_6_path = REPO_ROOT / "experiments" / "results" / "s0_6.json"
    assert s0_6_path.exists(), f"Reference artifact missing: {s0_6_path}"
    with open(s0_6_path, "r", encoding="utf-8") as f:
        s0_6_data = json.load(f)

    budget_proj = project_s0_7b_budget(s0_6_data)
    print_budget_projection(budget_proj)

    # 4. Stage J: Tokenization Disclosure Audit (run first, costs seconds)
    t_stage_j_start = time.time()
    stage_j_results = audit_tokenization_conventions(facts_pinned, tokenizer)
    print_stage_j_audit(stage_j_results)
    stage_j_elapsed = time.time() - t_stage_j_start
    print(f"\n[Cumulative Wall-Clock after Stage J: {time.time() - global_start_time:.1f} s (Stage J: {stage_j_elapsed:.2f} s)]")

    # 5. Gate 0: Early Bit-Reproduction Positive Control (Seed 0, Arm A delta=0.0)
    t_gate0_start = time.time()
    seed0_gate0_payload, gate_0_summary = run_gate_0_early(
        model_name, pinned_revision, tokenizer, fresh_model, sequences[0],
        template_prior_controls, wikitext_slice, slice_sha, device,
        s0_6_data, global_start_time
    )
    gate0_elapsed = time.time() - t_gate0_start
    print(f"\n[Cumulative Wall-Clock after Gate 0: {time.time() - global_start_time:.1f} s (Gate 0: {gate0_elapsed:.1f} s)]")

    # 6. Remainder of Stage F: GPU Re-Emission with Outcome Vector Serialization
    t_stage_f_start = time.time()
    stage_f_results, structural_invariance, f_steps, f_line_items = run_stage_f_re_emission(
        model_name, pinned_revision, tokenizer, fresh_model, sequences, facts_1000,
        template_prior_controls, wikitext_slice, slice_sha, device, seed0_gate0_payload
    )
    stage_f_elapsed = time.time() - t_stage_f_start
    print(f"\n[Cumulative Wall-Clock after Stage F: {time.time() - global_start_time:.1f} s (Stage F Remainder: {stage_f_elapsed:.1f} s)]")

    # Re-measure negative control floor from newly serialized Stage F vectors
    pooled_ctrl_measures = {}
    for c_name in CONTROL_NAMES:
        all_c = []
        for s in SEEDS:
            m = stage_f_results[c_name][s]["measurement"]
            all_c.extend([True] * m.numerator + [False] * (m.denominator - m.numerator))
        pooled_ctrl_measures[c_name] = Measurement.from_outcomes(all_c, metric=c_name, arm=c_name, scope="control_arm_pooled", input_set="ctrl_1200", mode="eval_no_dropout")

    pooled_ctrl_all, worst_ctrl_all, exp_sum_all = pool_controls(pooled_ctrl_measures, expected_per_control=1200)
    floor_interval = wilson_confidence_interval(worst_ctrl_all.numerator, worst_ctrl_all.denominator)
    print(f"\n[Negative Control Floor Audit]")
    print(f"  Worst Individual Control Floor : {worst_ctrl_all.name} -> {format_wilson_rate(worst_ctrl_all)} [{floor_interval[0]*100.0:.2f}%, {floor_interval[1]*100.0:.2f}%]")

    # 7. Stage G: Full-Population Reproduction & Analyses
    t_stage_g_start = time.time()
    stage_g_results = audit_stage_g(stage_f_results, s0_6_data, floor_interval)
    stage_g_elapsed = time.time() - t_stage_g_start
    print(f"\n[Cumulative Wall-Clock after Stage G: {time.time() - global_start_time:.1f} s (Stage G: {stage_g_elapsed:.2f} s)]")

    # 8. Stage H: Estimator Repair, Position Resolution, and Between-Arm Test
    t_stage_h_start = time.time()
    stage_h_results = audit_stage_h(stage_f_results, floor_interval, worst_ctrl_all.name)
    stage_h_elapsed = time.time() - t_stage_h_start
    print(f"\n[Cumulative Wall-Clock after Stage H: {time.time() - global_start_time:.1f} s (Stage H: {stage_h_elapsed:.2f} s)]")

    # 9. Stage I: Weight-Tying Confound (Untied Evaluation on seeds 0..2)
    t_stage_i_start = time.time()
    stage_i_results = run_stage_i_weight_tying(
        model_name, pinned_revision, tokenizer, fresh_model, sequences,
        template_prior_controls, wikitext_slice, slice_sha, device,
        stage_f_results, ppl_tied_baseline
    )
    stage_i_elapsed = time.time() - t_stage_i_start
    print(f"\n[Cumulative Wall-Clock after Stage I: {time.time() - global_start_time:.1f} s (Stage I: {stage_i_elapsed:.1f} s)]")

    # 10. Summary Accounting & Serialization
    total_optimizer_steps_global = f_steps
    actual_wall_clock_total = time.time() - global_start_time

    print("\n--- [Final Wall-Clock Budget & Step Attribution Audit] ---")
    print(f"  Projected Compute Wall-Clock : {budget_proj['projected_total_with_contingency']:.1f} s")
    print(f"  Actual Total Wall-Clock      : {actual_wall_clock_total:.1f} s")
    print(f"  Total Optimizer Steps Global : {total_optimizer_steps_global}")
    print(f"  Total Samples Seen           : {total_optimizer_steps_global} (1 sample / SGD step)")

    producing_commit = "DIRTY"
    try:
        producing_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    # Build serializable results object
    serialized_stage_f = {}
    for arm_k, seed_map in stage_f_results.items():
        serialized_stage_f[arm_k] = {}
        for s, rec in seed_map.items():
            s_dict = {}
            if "raw_vectors" in rec:
                s_dict["raw_vectors"] = rec["raw_vectors"]
            if "measurement" in rec:
                s_dict["measurement"] = rec["measurement"].pair
            for k_m in ["optimizer_steps", "samples_seen", "perplexity", "locality_kl", "immediate_efficacy", "terminal_retention"]:
                if k_m in rec:
                    val = rec[k_m]
                    s_dict[k_m] = val.pair if hasattr(val, "pair") else val
            serialized_stage_f[arm_k][s] = s_dict

    res_data = {
        "directive": "S0-7b",
        "producing_commit_sha": producing_commit,
        "exit_code": 0,
        "hashes": {
            "facts_json_sha256": facts_sha,
            "wikitext_slice_sha256": slice_sha,
            "weight_file_sha256": weight_sha,
            "control_probes_sha256": ctrl_probe_sha,
            "seed_sequence_hashes": seed_sequence_hashes
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pinned_revision": pinned_revision,
            "fresh_checksum": fresh_checksum
        },
        "budget_projection": budget_proj,
        "stage_j": stage_j_results,
        "gate_0": gate_0_summary,
        "stage_f": serialized_stage_f,
        "stage_g": stage_g_results,
        "stage_h": stage_h_results,
        "stage_i": stage_i_results,
        "accounting": {
            "total_optimizer_steps": total_optimizer_steps_global,
            "total_samples_seen": total_optimizer_steps_global,
            "projected_wall_clock": budget_proj["projected_total_with_contingency"],
            "actual_wall_clock": actual_wall_clock_total
        }
    }

    out_dir = REPO_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    def _json_fallback(o):
        if hasattr(o, "pair"):
            return o.pair
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    out_file = out_dir / "s0_7b.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2, default=_json_fallback)

    print(f"\n  Artifact Written             : {out_file.relative_to(REPO_ROOT)}")
    print("\n" + "=" * 115)
    print(" DIRECTIVE S0-7b COMPLETE: RE-EMISSION, ESTIMATOR REPAIR, POSITION RESOLUTION, AND UNTYING EXECUTED")
    print("=" * 115)
    print("SCRIPT_EXIT=0")
    sys.exit(0)


if __name__ == "__main__":
    main()
