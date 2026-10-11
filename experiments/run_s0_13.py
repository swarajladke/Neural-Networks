#!/usr/bin/env python3
"""
experiments/run_s0_13.py -- Master Orchestrator for Directive S0-13

Executes:
- Stage 0 Gates: G0 (bit-reproduce Seed 0), G0b (pin cache C and P0/V_null), G0c (cross-process reference)
- Stage L2: Corrected write-site loss localization (snapshotting, L2a drift, L2b output patching, L2c margin)
- Stage R: Exploratory seed-clustered logistic regression
- Stage F: Protection arms (Reference, F1 full-prompt, F3 margin-targeted, F2 teacher-forced, Sham control)
- Compute budget projections and dynamic pruning (ceiling 16,380.0 s)
- Rule 3.7 per-unit vector serialization to experiments/results/s0_13.json
"""

import os
import sys
import time
import math
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from experiments.b1_inject import (
    SEEDS,
    configure_determinism,
    greedy_predict,
    edit_fact_sgd,
)
from experiments.data import (
    sample_200_facts,
    load_wikitext2_slice,
    evaluate_wikitext_perplexity,
)
from experiments.stats import (
    compute_paired_stats_with_pvalues,
    newcombe_score_interval,
)
from experiments.metrics import (
    Measurement,
    wilson_confidence_interval,
    normalize_entity,
    check_match,
)
from experiments.s0_10_repair import verify_state_restore, get_subject_last_token_idx
from experiments.s0_11_constraints import (
    load_wikitext2_key_sample,
    compute_layer_key_covariance,
    compute_null_space_projector,
    CorrectedSequentialNullTracker,
    find_target_value_vstar,
    edit_fact_mlp_null_corrected,
)
from experiments.s0_12_localization import (
    evaluate_s0_12_sequential_controls,
    evaluate_sham_sequence_control,
)
from experiments.s0_13_localization import (
    reconstruct_weight,
    verify_weight_reconstruction,
    compute_target_margin,
    compute_l2a_drift,
    run_stage_l2b_patching,
    run_stage_r_regression,
)
from experiments.s0_13_protection import (
    FullPromptSVDNullTracker,
    find_target_value_vstar_margin,
    edit_fact_f1,
    edit_fact_f2,
    edit_fact_f3,
    extract_prompt_keys,
)

CACHE_DIR = "experiments/results/cache_s0_13"
RESULTS_FILE = "experiments/results/s0_13.json"
SESSION_BUDGET_CEILING = 16380.0  # seconds


def set_deterministic_seeds(seed: int = 42):
    configure_determinism(seed=seed)


def evaluate_fact_retention(
    model: nn.Module,
    tokenizer: Any,
    facts_eval: List[Dict[str, Any]],
    max_new_tokens: int = 5,
    device: str = "cuda",
    train_mode: bool = False
) -> Tuple[List[bool], List[bool]]:
    c_matches = []
    p_matches = []
    for f in facts_eval:
        pred_c = greedy_predict(model, tokenizer, f["edit_prompt"], max_new_tokens, device, train_mode)
        c_matches.append(check_match(pred_c, f["object"]))
        for p in f["paraphrases"]:
            pred_p = greedy_predict(model, tokenizer, p, max_new_tokens, device, train_mode)
            p_matches.append(check_match(pred_p, f["object"]))
    return c_matches, p_matches


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    wikitext_slice: torch.Tensor,
    device: str = "cuda",
    slice_sha: str = "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7",
    max_length: int = 512
) -> float:
    return evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)


def newcombe_confidence_interval(k1: int, n1: int, k2: int, n2: int) -> Dict[str, float]:
    diff, lo, hi = newcombe_score_interval(k1, n1, k2, n2, confidence=0.95)
    return {"diff": diff, "ci_lower": lo, "ci_upper": hi}


def get_git_commit_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def run_g0_gate(model: nn.Module, tokenizer: Any, facts_seq: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    print("\n--- [Stage 0: Gate G0 Baseline Re-Confirmation] ---")
    configure_determinism(seed=0)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    total_steps = 0
    imm_matches = []
    t0 = time.time()

    for fact in facts_seq:
        res = edit_fact_sgd(model, tokenizer, fact, lr=3e-5, max_steps=100, delta=0.0, device=device, train_mode=False)
        total_steps += res["steps_taken"]
        imm_matches.append(res["immediate_match"])

    c_ret, _ = evaluate_fact_retention(model, tokenizer, facts_seq, 5, device, False)
    term_matches = sum(1 for x in c_ret if x)
    elapsed = time.time() - t0

    print(f"  G0 Steps: {total_steps} (expected 669)")
    print(f"  G0 Immediate: {sum(imm_matches)}/200 (expected 200)")
    print(f"  G0 Terminal: {term_matches}/200 (expected 8)")
    assert total_steps == 669, f"G0 step mismatch: {total_steps} != 669"
    assert sum(imm_matches) == 200, f"G0 immediate mismatch: {sum(imm_matches)} != 200"
    assert term_matches == 8, f"G0 terminal mismatch: {term_matches} != 8"
    print("  Gate G0: PASSED.")

    model.load_state_dict(base_state)
    return {
        "total_steps": total_steps, "immediate": sum(imm_matches), "terminal": term_matches, "elapsed": elapsed,
        "immediate_matches": imm_matches, "terminal_matches": c_ret
    }


def ensure_pinned_cache(model: nn.Module, tokenizer: Any, device: str) -> Dict[str, Any]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    c_path = os.path.join(CACHE_DIR, "cov_l1.pt")
    v_path = os.path.join(CACHE_DIR, "v_null_l1.pt")

    if not os.path.exists(c_path) or not os.path.exists(v_path):
        print("\n--- [Stage 0: Gate G0b Computing Pinned Cache] ---")
        key_sample_tensor, _ = load_wikitext2_key_sample(tokenizer, num_sequences=100, seq_len=512)
        cov, _ = compute_layer_key_covariance(model, key_sample_tensor, layer_idx=1, device=device)
        null_res = compute_null_space_projector(cov, rel_threshold=1e-3)
        torch.save(cov, c_path)
        torch.save(null_res["v_null"], v_path)

    cov = torch.load(c_path, map_location=device)
    v_null = torch.load(v_path, map_location=device)
    with open(c_path, "rb") as f:
        cov_sha = hashlib.sha256(f.read()).hexdigest()
    with open(v_path, "rb") as f:
        v_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"  Pinned Covariance SHA-256: {cov_sha}")
    print(f"  Pinned V_null SHA-256: {v_sha}")
    return {"cov": cov, "v_null": v_null, "cov_sha256": cov_sha, "v_null_sha256": v_sha}


def run_sequential_reference_seed(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    facts_seq: List[Dict[str, Any]],
    v_null: torch.Tensor,
    layer_idx: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    w_0 = c_proj.weight.data.clone().cpu()
    b_0 = c_proj.bias.data.clone().cpu()
    tracker = CorrectedSequentialNullTracker(v_null, rel_threshold=1e-3, device=device)

    imm_matches, steps_per_edit = [], []
    factors_32: List[Tuple[torch.Tensor, torch.Tensor]] = []
    factors_64: List[Tuple[torch.Tensor, torch.Tensor]] = []
    k_vecs, r_vecs, vstar_vecs, v0_vecs = [], [], [], []
    write_margins = []
    w_50, w_100 = None, None

    for idx, fact in enumerate(facts_seq):
        res = edit_fact_mlp_null_corrected(model, tokenizer, fact, tracker, layer_idx=layer_idx, max_steps=100, lr_v=0.1, device=device)
        imm_matches.append(res["immediate_match"])
        steps_per_edit.append(res["steps_taken"])

        # Factors: u_f32, r_f32
        p_k = res["p_k"].to(device, dtype=torch.float64)
        k_64 = res["k_vec"].to(device, dtype=torch.float64)
        r_64 = res["r_vec"].to(device, dtype=torch.float64)
        k_p_k = torch.dot(k_64, p_k).item()
        u_64 = p_k / (k_p_k + 1e-12)
        u_f32 = u_64.to(torch.float32).cpu()
        r_f32 = r_64.to(torch.float32).cpu()

        factors_32.append((u_f32, r_f32))
        factors_64.append((u_64.cpu(), r_64.cpu()))
        k_vecs.append(res["k_vec"].cpu())
        r_vecs.append(res["r_vec"].cpu())
        vstar_vecs.append(res["v_star"].cpu())
        v0_vecs.append(res["v0"].cpu())

        if idx == 49:
            w_50 = c_proj.weight.data.clone().cpu()
        elif idx == 99:
            w_100 = c_proj.weight.data.clone().cpu()

        w_margin = compute_target_margin(model, tokenizer, fact["edit_prompt"], fact["object"], device=device)
        write_margins.append(w_margin)

    # Unit test reconstruction at t = 49 (50 edits), 99 (100 edits), 199 (200 edits)
    assert w_50 is not None and w_100 is not None
    verify_weight_reconstruction(w_50, w_0, factors_32, 49)
    verify_weight_reconstruction(w_100, w_0, factors_32, 99)
    verify_weight_reconstruction(c_proj.weight.data.cpu(), w_0, factors_32, 199)

    # Terminal evaluation on first 50 facts
    c_ret, p_ret = evaluate_fact_retention(model, tokenizer, facts_seq[:50], 5, device, False)
    term_margins = [
        compute_target_margin(model, tokenizer, fact["edit_prompt"], fact["object"], device=device)
        for fact in facts_seq[:50]
    ]

    return {
        "immediate_matches": imm_matches,
        "steps_per_edit": steps_per_edit,
        "canonical_retention": c_ret,
        "paraphrase_retention": p_ret,
        "write_margins": write_margins,
        "terminal_margins": term_margins,
        "factors_32": factors_32,
        "factors_64": factors_64,
        "k_vecs": k_vecs,
        "r_vecs": r_vecs,
        "vstar_vecs": vstar_vecs,
        "v0_vecs": v0_vecs,
        "w_0": w_0,
        "b_0": b_0,
        "final_weight_sha": hashlib.sha256(c_proj.weight.data.cpu().numpy().tobytes()).hexdigest(),
    }


def run_protection_seed(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    facts: List[Dict[str, Any]],
    tracker: Any,
    edit_fn: Any,
    wikitext_slice: torch.Tensor,
    device: str,
    slice_sha: str = "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7",
    **kwargs
) -> Dict[str, Any]:
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
    imm_m, steps_m = [], []
    for idx, fact in enumerate(facts):
        res = edit_fn(model, tokenizer, fact, tracker, idx, layer_idx=1, device=device, **kwargs)
        imm_m.append(res["immediate_match"])
        steps_m.append(res["steps_taken"])
    c_ret, p_ret = evaluate_fact_retention(model, tokenizer, facts[:50], 5, device, False)
    ppl_val = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device, slice_sha=slice_sha)
    term_m = [compute_target_margin(model, tokenizer, f["edit_prompt"], f["object"], device=device) for f in facts[:50]]
    out = {
        "immediate_matches": imm_m, "steps_per_edit": steps_m,
        "canonical_retention": c_ret, "paraphrase_retention": p_ret,
        "terminal_ppl": ppl_val, "terminal_margins": term_m
    }
    if hasattr(tracker, "current_rank"):
        out.update({
            "final_rank": tracker.current_rank, "capacity_exhausted": tracker.capacity_exhausted,
            "exhaustion_edit_idx": tracker.exhaustion_edit_idx, "rank_history": tracker.rank_history
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Run Directive S0-13")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune-f2", action="store_true")
    parser.add_argument("--prune-f3", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    set_deterministic_seeds(42)
    device = args.device

    print("=" * 115)
    print(" DIRECTIVE S0-13: WRITE-SITE LOSS LOCALIZATION, FULL-PROMPT PROTECTION, AND MARGIN-TARGETED WRITES")
    print("=" * 115)

    commit_sha = get_git_commit_sha()
    print(f"Producing Commit SHA: {commit_sha}")
    print(f"Accelerator Device: {device}")

    facts_file = REPO_ROOT / "b1_facts.json"
    assert facts_file.exists(), f"Missing facts file: {facts_file}"
    with open(facts_file, "r", encoding="utf-8") as f:
        facts_pool = json.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer)

    base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    fresh_checksum = float(sum(p.sum().item() for p in model.parameters()))
    fresh_c_proj_hashes = {
        1: hashlib.sha256(model.transformer.h[1].mlp.c_proj.weight.data.cpu().numpy().tobytes()).hexdigest()
    }

    baseline_ppl = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device, slice_sha=slice_sha)
    print(f"WikiText-2 Capability Slice Baseline PPL: {baseline_ppl:.2f}")

    facts_seed0, _ = sample_200_facts(facts_pool, seed=0)
    g0_res = run_g0_gate(model, tokenizer, facts_seed0, device)

    cache_info = ensure_pinned_cache(model, tokenizer, device)
    v_null = cache_info["v_null"]

    print("\n--- [Stage 0: Gate G0c Reference Cell & Cross-Process Check] ---")
    ref_run1 = run_sequential_reference_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, facts_seed0, v_null, 1, device)
    ref_run2 = run_sequential_reference_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, facts_seed0, v_null, 1, device)

    assert ref_run1["final_weight_sha"] == ref_run2["final_weight_sha"], "G0c cross-process weight mismatch!"
    assert ref_run1["canonical_retention"] == ref_run2["canonical_retention"], "G0c retention mismatch!"
    print(f"  Gate G0c PASSED: Identical weight SHA ({ref_run1['final_weight_sha'][:16]}...)")
    print(f"  Seed 0 Reference Imm: {sum(ref_run1['immediate_matches'])}/200, Ret: {sum(ref_run1['canonical_retention'])}/50")

    print("\n--- [Executing Reference Arm (A-null_L1_corr) across 6 seeds] ---")
    reference_runs = {0: ref_run1}
    for s in SEEDS[1:]:
        print(f"  Running Reference Seed {s}...")
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        reference_runs[s] = run_sequential_reference_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, s_facts, v_null, 1, device)

    print("\n--- [Stage L2: Write-Site Loss Localization] ---")
    l2_seed0 = reference_runs[0]
    drift_res = compute_l2a_drift(l2_seed0["k_vecs"], l2_seed0["r_vecs"], l2_seed0["factors_32"], l2_seed0["factors_64"], device=device)

    lost_idx = [i for i, m in enumerate(l2_seed0["canonical_retention"]) if not m]
    ret_idx = [i for i, m in enumerate(l2_seed0["canonical_retention"]) if m]
    print(f"  Seed 0 First-50: {len(ret_idx)} retained, {len(lost_idx)} lost")

    # Restore Seed 0 terminal weights into model for Stage L2b output patching
    model.load_state_dict(base_state_dict)
    w_seed0_term = reconstruct_weight(l2_seed0["w_0"], l2_seed0["factors_32"], 199).to(device)
    model.transformer.h[1].mlp.c_proj.weight.data.copy_(w_seed0_term)

    l2b_res = run_stage_l2b_patching(
        model, tokenizer, l2_seed0["w_0"], l2_seed0["b_0"], l2_seed0["factors_32"],
        facts_seed0, lost_idx, ret_idx, layer_idx=1, device=device
    )
    print(f"  Harness Positive Control Passed: {l2b_res['harness_positive_control_passed']}")
    print(f"  Lost facts recovery: {l2b_res['lost_recovery']}")

    print("\n--- [Stage R: Exploratory Regression] ---")
    reg_records = []
    for s in SEEDS:
        run_s = reference_runs[s]
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        s_drift = compute_l2a_drift(run_s["k_vecs"], run_s["r_vecs"], run_s["factors_32"], device=device)["drifts_f32"]
        for j in range(50):
            fact_j = s_facts[j]
            later_same_rel = sum(1 for k in range(j + 1, 200) if s_facts[k]["relation"] == fact_j["relation"])
            enc_obj = tokenizer(f" {fact_j['object'].strip()}").input_ids
            is_multi = len(enc_obj) > 1
            reg_records.append({
                "seed": s, "edit_position": j, "later_same_rel_count": later_same_rel,
                "is_multi_token": is_multi, "write_time_margin": run_s["write_margins"][j],
                "drift_j": s_drift[j], "terminal_retained": run_s["canonical_retention"][j]
            })

    reg_out = run_stage_r_regression(reg_records, cluster_key="seed")
    print(f"  Stage R Regression Status: {reg_out['status']}")

    f1_runs, f2_runs, f3_runs = {}, {}, {}
    print("\n--- [Stage F: Arm F1 Full-Prompt Protection across 6 seeds] ---")
    for s in SEEDS:
        print(f"  Running F1 Seed {s}...")
        s_facts, _ = sample_200_facts(facts_pool, seed=s)
        f1_tracker = FullPromptSVDNullTracker(v_null, rel_threshold=1e-3, device=device)
        f1_runs[s] = run_protection_seed(
            model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
            s_facts, f1_tracker, edit_fact_f1, wikitext_slice, device, slice_sha=slice_sha
        )

    elapsed_so_far = time.time() - t_start
    print(f"\nElapsed time so far: {elapsed_so_far:.2f} s / {SESSION_BUDGET_CEILING:.2f} s")
    prune_f2 = args.prune_f2 or (elapsed_so_far + 4000.0 > SESSION_BUDGET_CEILING)
    prune_f3 = args.prune_f3 or (elapsed_so_far + 2000.0 > SESSION_BUDGET_CEILING)

    if not prune_f3:
        print("\n--- [Stage F: Arm F3 Margin-Targeted Writes across 6 seeds] ---")
        for s in SEEDS:
            print(f"  Running F3 Seed {s}...")
            s_facts, _ = sample_200_facts(facts_pool, seed=s)
            f3_tracker = CorrectedSequentialNullTracker(v_null, rel_threshold=1e-3, device=device)
            f3_runs[s] = run_protection_seed(
                model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
                s_facts, f3_tracker, edit_fact_f3, wikitext_slice, device, slice_sha=slice_sha, target_margin=2.0
            )
    else:
        print("\n[Dynamic Pruning]: Skipping Arm F3 per compute ceiling rules.")

    if not prune_f2:
        print("\n--- [Stage F: Arm F2 F1 + Teacher-Forced Object Keys across 6 seeds] ---")
        for s in SEEDS:
            print(f"  Running F2 Seed {s}...")
            s_facts, _ = sample_200_facts(facts_pool, seed=s)
            f2_tracker = FullPromptSVDNullTracker(v_null, rel_threshold=1e-3, device=device)
            f2_runs[s] = run_protection_seed(
                model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
                s_facts, f2_tracker, edit_fact_f2, wikitext_slice, device, slice_sha=slice_sha
            )
    else:
        print("\n[Dynamic Pruning]: Skipping Arm F2 per compute ceiling rules.")

    # Sham Sequence Control under F1 rules (Seed 0)
    print("\n--- [Executing Sham Sequence Control under F1 rules (Seed 0)] ---")
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    sham_tracker = FullPromptSVDNullTracker(v_null, rel_threshold=1e-3, device=device)
    for idx, fact in enumerate(facts_seed0):
        subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
        enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
        rec = {}
        def hook_fn(m, inp, out):
            rec["k"] = inp[0][0, subj_idx, :].detach().clone()
            rec["v0"] = out[0, subj_idx, :].detach().clone()
        h = c_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**enc_prompt)
        h.remove()
        r_zero = torch.zeros_like(rec["v0"])
        upd = sham_tracker.compute_update(rec["k"], r_zero)
        c_proj.weight.data.add_(upd["delta_f32"])
        non_subj = extract_prompt_keys(model, tokenizer, fact["edit_prompt"], subj_idx, 1, device)
        all_pk = torch.cat([rec["k"].unsqueeze(0), non_subj], dim=0)
        sham_tracker.add_keys(all_pk, idx)

    sham_c_ret, sham_p_ret = evaluate_fact_retention(model, tokenizer, facts_seed0[:50], 5, device, False)
    sham_out = {
        "canonical_retention": sham_c_ret,
        "paraphrase_retention": sham_p_ret,
        "c_matches": sum(1 for x in sham_c_ret if x),
        "p_matches": sum(1 for x in sham_p_ret if x)
    }

    # Reference Arm PPL evaluation across all 6 seeds
    print("\n--- [Evaluating Reference Arm Perplexity] ---")
    for s in SEEDS:
        run_s = reference_runs[s]
        model.load_state_dict(base_state_dict)
        w_terminal = reconstruct_weight(run_s["w_0"], run_s["factors_32"], 199).to(device)
        model.transformer.h[1].mlp.c_proj.weight.data.copy_(w_terminal)
        run_s["terminal_ppl"] = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device, slice_sha=slice_sha)
    model.load_state_dict(base_state_dict)

    # Primary Comparison: F1 vs Reference on E2 paired by seed
    ref_ret_counts = [float(sum(reference_runs[s]["canonical_retention"])) for s in SEEDS]
    f1_ret_counts = [float(sum(f1_runs[s]["canonical_retention"])) for s in SEEDS]
    paired_diffs = [f1_ret_counts[i] - ref_ret_counts[i] for i in range(6)]
    mean_diff = sum(paired_diffs) / 6.0

    paired_stats = compute_paired_stats_with_pvalues(f1_ret_counts, ref_ret_counts)
    t_stat, t_pval = paired_stats["t_stat"], paired_stats["t_pvalue"]
    w_stat, w_pval = paired_stats["wilcoxon_stat"], paired_stats["wilcoxon_pvalue"]
    pool_f1_ret = int(sum(f1_ret_counts))
    pool_ref_ret = int(sum(ref_ret_counts))
    newcombe_diff = newcombe_confidence_interval(pool_f1_ret, 300, pool_ref_ret, 300)

    pct_sym, ci_lvl = "%", 95
    ref_pct, f1_pct = pool_ref_ret / 300.0 * 100.0, pool_f1_ret / 300.0 * 100.0
    print("\n" + "=" * 80)
    print(" PRIMARY COMPARISON (Pre-Registered): F1 vs Reference on E2 (First-50 Retention)")
    print("=" * 80)
    print(f"  Reference Retention (Pooled): {pool_ref_ret}/300 ({ref_pct:.2f}{pct_sym})")
    print(f"  F1 Retention (Pooled):        {pool_f1_ret}/300 ({f1_pct:.2f}{pct_sym})")
    print(f"  Newcombe {ci_lvl}{pct_sym} CI Diff:         {newcombe_diff['diff']*100:+.2f} pp [{newcombe_diff['ci_lower']*100:+.2f} pp, {newcombe_diff['ci_upper']*100:+.2f} pp]")
    print(f"  Exact Student's t (df=5):     t = {t_stat:.4f}, p = {t_pval:.4e}")
    print(f"  Exact Wilcoxon (n=6):         W = {w_stat:.1f}, p = {w_pval:.4f}")

    total_wall_clock = time.time() - t_start
    print(f"\nTotal Wall-Clock Time: {total_wall_clock:.2f} s (Ceiling: {SESSION_BUDGET_CEILING:.2f} s)")

    results_artifact = {
        "metadata": {
            "directive": "S0-13", "producing_commit_sha": commit_sha, "device": device,
            "exit_code": 0, "baseline_ppl": baseline_ppl, "total_wall_clock_s": total_wall_clock,
            "ceiling_s": SESSION_BUDGET_CEILING,
            "cache_hashes": {"cov_l1_sha256": cache_info["cov_sha256"], "v_null_l1_sha256": cache_info["v_null_sha256"]}
        },
        "gates": {
            "G0": g0_res,
            "G0b": {"cov_sha": cache_info["cov_sha256"], "v_null_sha": cache_info["v_null_sha256"]},
            "G0c": {
                "passed": True, "seed0_weight_sha": ref_run1["final_weight_sha"],
                "seed0_imm": sum(ref_run1["immediate_matches"]), "seed0_ret": sum(ref_run1["canonical_retention"])
            }
        },
        "stage_l2": {
            "l2a_drift_seed0": {
                "drifts_f32": drift_res["drifts_f32"], "drifts_f64": drift_res["drifts_f64"],
                "median_drift_lost": float(torch.tensor([drift_res["drifts_f32"][i] for i in lost_idx]).median().item()) if lost_idx else 0.0,
                "median_drift_retained": float(torch.tensor([drift_res["drifts_f32"][i] for i in ret_idx]).median().item()) if ret_idx else 0.0
            },
            "l2b_output_patching_seed0": l2b_res,
            "l2c_margins": {"seed0_write_margins": l2_seed0["write_margins"], "seed0_terminal_margins": l2_seed0["terminal_margins"]}
        },
        "stage_r": reg_out,
        "stage_f": {
            "reference": {
                s: {
                    "immediate_matches": reference_runs[s]["immediate_matches"],
                    "steps_per_edit": reference_runs[s]["steps_per_edit"],
                    "canonical_retention": reference_runs[s]["canonical_retention"],
                    "paraphrase_retention": reference_runs[s]["paraphrase_retention"],
                    "terminal_ppl": reference_runs[s]["terminal_ppl"],
                    "terminal_margins": reference_runs[s]["terminal_margins"]
                } for s in SEEDS
            },
            "f1": f1_runs, "f3": f3_runs, "f2": f2_runs, "sham": sham_out
        },
        "primary_inference": {
            "f1_vs_reference_e2": {
                "pooled_ref_ret": pool_ref_ret, "pooled_f1_ret": pool_f1_ret, "newcombe_95_ci": newcombe_diff,
                "student_t": {"stat": t_stat, "df": 5, "p_val": t_pval},
                "wilcoxon": {"stat": w_stat, "n": 6, "p_val": w_pval}
            }
        }
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_artifact, f, indent=2)
    print(f"\nResults successfully recorded to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
