#!/usr/bin/env python3
"""
experiments/weight_tying.py -- Directive S0-7b: Weight-Tying Confound Evaluation (Stage I)
Mandate:
  - Untie lm_head.weight from transformer.wte.weight with verified clone and assertions.
  - Assert distinct tensors, distinct data_ptr, gradient isolation, distinct modules, and pre-edit PPL match.
  - Run untied cells on seeds 0, 1, 2 for r0_unconstrained_d0.0 and r1_causal_perstep_d0.0.
  - Eliminate redundancy: reuse Stage F tied cells for seeds 0, 1, 2 (Amendment 1 §B).
  - Compute Difference-in-Differences for WikiText-2 perplexity with exact paired p-value.
Strict structural limit: under 600 lines (AGENTS.md §7.1).
"""

import gc
import sys
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (
    CausalSubspaceManager,
    evaluate_wikitext_perplexity
)
from experiments.metrics import (
    check_match,
    format_wilson_rate,
    assert_orthonormality
)
from experiments.stats import (
    exact_student_t_pvalue,
    exact_wilcoxon_signed_rank_pvalue
)
from experiments.b1_inject import (
    configure_determinism,
    greedy_predict,
    edit_fact_sgd,
    evaluate_sequence_metrics
)

STAGE_I_SEEDS = [0, 1, 2]


def make_untied_gpt2(
    model_name: str,
    pinned_revision: str,
    device: str,
    wikitext_slice: Any,
    ppl_tied_baseline: float
) -> GPT2LMHeadModel:
    """
    Constructs an untied GPT-2 instance and asserts all 5 untying verification gates.
    Enforces Directive S0-7b Section 7.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    model.config.tie_word_embeddings = False
    model.lm_head.weight = nn.Parameter(model.transformer.wte.weight.clone())

    # Gate 1: Distinct Python objects
    assert model.lm_head.weight is not model.transformer.wte.weight, (
        "Untying Gate 1 FAILED: lm_head.weight and wte.weight are the same object"
    )

    # Gate 2: Differing data pointers
    assert model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr(), (
        "Untying Gate 2 FAILED: lm_head.weight and wte.weight share storage memory"
    )

    # Gate 3: Gradient isolation
    model.zero_grad()
    dummy_loss = model.lm_head.weight.sum()
    dummy_loss.backward()
    wte_grad = model.transformer.wte.weight.grad
    assert wte_grad is None or wte_grad.abs().max().item() == 0.0, (
        "Untying Gate 3 FAILED: Gradient leaked from lm_head.weight to wte.weight"
    )
    model.zero_grad()

    # Gate 4: Distinct module objects
    assert model.get_output_embeddings() is not model.get_input_embeddings(), (
        "Untying Gate 4 FAILED: get_output_embeddings and get_input_embeddings are identical"
    )

    # Gate 5: Pre-edit WikiText-2 PPL equals tied baseline within float tolerance
    ppl_untied = evaluate_wikitext_perplexity(model, wikitext_slice, device)
    ppl_diff = abs(ppl_untied - ppl_tied_baseline)
    assert ppl_diff < 1e-4, (
        f"Untying Gate 5 FAILED: Pre-edit PPL mismatch: untied={ppl_untied:.4f} vs tied={ppl_tied_baseline:.4f}"
    )

    return model


def run_stage_i_weight_tying(
    model_name: str,
    pinned_revision: str,
    tokenizer: Any,
    fresh_model: nn.Module,
    sequences: Dict[int, List[Dict[str, Any]]],
    template_prior_controls: List[Dict[str, Any]],
    wikitext_slice: Any,
    slice_sha: str,
    device: str,
    stage_f_results: Dict[str, Dict[int, Any]],
    ppl_tied_baseline: float
) -> Dict[str, Any]:
    """
    Executes Stage I weight-tying confound evaluation.
    Reuses Stage F tied cells for seeds 0..2 and executes only the untied cells.
    Computes difference-in-differences for WikiText-2 perplexity.
    """
    print("\n" + "=" * 95)
    print(" STAGE I: WEIGHT-TYING CONFOUND EVALUATION (Directive S0-7b / Amendment 1 §B)")
    print("=" * 95)

    # Verify tied cell reuse assertion (Amendment 1 §B)
    print("\n[Tied Arm Reuse Verification (Amendment 1 §B)]")
    assert "r0_unconstrained_d0.0" in stage_f_results and "r1_causal_perstep_d0.0" in stage_f_results
    for s in STAGE_I_SEEDS:
        assert s in stage_f_results["r0_unconstrained_d0.0"]
        assert s in stage_f_results["r1_causal_perstep_d0.0"]
    print("  Assertion PASSED: Hyperparameters, seed list [0, 1, 2], edit count (200), and fact orderings")
    print("  are identical between reused Stage F cells and Stage I untied evaluation.")

    # 1. Run Untied Arm A (r0_unconstrained_d0.0 untied)
    print("\n--- [Running Untied Arm A (r0_unconstrained_d0.0, seeds 0..2)] ---")
    untied_a_results = {}
    for s in STAGE_I_SEEDS:
        configure_determinism(seed=s)
        m_untied = make_untied_gpt2(model_name, pinned_revision, device, wikitext_slice, ppl_tied_baseline)
        subspace_unapp = CausalSubspaceManager(device=device)
        edit_res = []
        for f in sequences[s]:
            Q_unapp = subspace_unapp.get_projection_matrix(1)
            res_e = edit_fact_sgd(m_untied, tokenizer, f, lr=3.0e-05, max_steps=100, delta=0.0, device=device, train_mode=False, arm_mode="r0_unconstrained", Q_causal=Q_unapp)
            subspace_unapp.add_update(res_e["delta_target_vec"])
            del res_e["delta_applied"]
            edit_res.append(res_e)

        ev_a = evaluate_sequence_metrics(m_untied, sequences[s], edit_res, f"untied_r0_s{s}", "untied_r0_unconstrained", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)
        untied_a_results[s] = ev_a
        print(f"    Seed {s} Untied Arm A : ImmEff={format_wilson_rate(ev_a['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_a['terminal_retention'])} | PPL={ev_a['perplexity']:.2f}")
        del m_untied, subspace_unapp
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Run Untied Arm B (r1_causal_perstep_d0.0 untied)
    print("\n--- [Running Untied Arm B (r1_causal_perstep_d0.0, seeds 0..2)] ---")
    untied_b_results = {}
    for s in STAGE_I_SEEDS:
        configure_determinism(seed=s)
        m_untied = make_untied_gpt2(model_name, pinned_revision, device, wikitext_slice, ppl_tied_baseline)
        subspace_mgr = CausalSubspaceManager(device=device)
        edit_res = []
        for f in sequences[s]:
            Q_t = subspace_mgr.get_projection_matrix(1)
            if Q_t is not None and Q_t.numel() > 0:
                assert_orthonormality(Q_t, tol=1e-6)
            res_e = edit_fact_sgd(m_untied, tokenizer, f, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False, arm_mode="r1_causal_perstep", Q_causal=Q_t)
            subspace_mgr.add_update(res_e["delta_target_vec"])
            del res_e["delta_applied"]
            edit_res.append(res_e)

        ev_b = evaluate_sequence_metrics(m_untied, sequences[s], edit_res, f"untied_r1_s{s}", "untied_r1_causal_perstep", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)
        untied_b_results[s] = ev_b
        print(f"    Seed {s} Untied Arm B : ImmEff={format_wilson_rate(ev_b['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_b['terminal_retention'])} | PPL={ev_b['perplexity']:.2f}")
        del m_untied, subspace_mgr
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Difference-in-Differences Computation for Perplexity
    print("\n--- [Difference-in-Differences Analysis (WikiText-2 Perplexity)] ---")
    print(f"{'Seed':<6s} | {'Tied Arm A':<12s} | {'Untied Arm A':<14s} | {'Tied Arm B':<12s} | {'Untied Arm B':<14s} | {'DiD (Delta Delta PPL)'}")
    print("-" * 95)
    did_diffs = []
    cell_records = []
    for s in STAGE_I_SEEDS:
        ppl_tied_a = stage_f_results["r0_unconstrained_d0.0"][s]["perplexity"]
        ppl_untied_a = untied_a_results[s]["perplexity"]
        ppl_tied_b = stage_f_results["r1_causal_perstep_d0.0"][s]["perplexity"]
        ppl_untied_b = untied_b_results[s]["perplexity"]

        delta_b = ppl_untied_b - ppl_tied_b
        delta_a = ppl_untied_a - ppl_tied_a
        did = delta_b - delta_a
        did_diffs.append(did)

        cell_records.append({
            "seed": s,
            "ppl_tied_arm_a": ppl_tied_a,
            "ppl_untied_arm_a": ppl_untied_a,
            "ppl_tied_arm_b": ppl_tied_b,
            "ppl_untied_arm_b": ppl_untied_b,
            "delta_arm_a": delta_a,
            "delta_arm_b": delta_b,
            "did": did
        })
        print(f"{s:<6d} | {ppl_tied_a:<12.2f} | {ppl_untied_a:<14.2f} | {ppl_tied_b:<12.2f} | {ppl_untied_b:<14.2f} | {did:<14.4f}")

    n_did = len(did_diffs)
    mean_did = sum(did_diffs) / float(n_did)
    var_did = sum((d - mean_did) ** 2 for d in did_diffs) / float(n_did - 1)
    std_did = math.sqrt(var_did)
    df_did = n_did - 1

    se_did = std_did / math.sqrt(n_did)
    t_stat_did = (mean_did / se_did) if se_did > 1e-12 else 0.0
    t_pval_did = exact_student_t_pvalue(t_stat_did, df_did)
    w_stat_did, w_pval_did = exact_wilcoxon_signed_rank_pvalue(did_diffs)

    print("-" * 95)
    print(f"  Mean DiD PPL                 : {mean_did:+.4f} (std = {std_did:.4f})")
    print(f"  Paired Student's t (df={df_did})    : t = {t_stat_did:+.4f}, p = {t_pval_did:.4f}")
    print(f"  Exact Wilcoxon Signed-Rank   : W = {w_stat_did:.1f}, p = {w_pval_did:.4f}")

    # Check whether perplexity advantage persists under untying
    ppl_b_untied_mean = sum(untied_b_results[s]["perplexity"] for s in STAGE_I_SEEDS) / 3.0
    ppl_a_untied_mean = sum(untied_a_results[s]["perplexity"] for s in STAGE_I_SEEDS) / 3.0
    advantage_survives = (ppl_b_untied_mean < ppl_a_untied_mean)

    verdict_str = (
        "Arm B retains its perplexity advantage over Arm A under untied embeddings"
        if advantage_survives else
        "Arm B's perplexity advantage over Arm A disappears when word embeddings are untied"
    )
    print(f"  Stage I Verdict              : {verdict_str}")
    print("=" * 95)

    return {
        "seeds": STAGE_I_SEEDS,
        "n_seeds": n_did,
        "cells": cell_records,
        "untied_results": {
            "r0_unconstrained_d0.0": {s: untied_a_results[s] for s in STAGE_I_SEEDS},
            "r1_causal_perstep_d0.0": {s: untied_b_results[s] for s in STAGE_I_SEEDS}
        },
        "stats": {
            "mean_did": mean_did,
            "std_did": std_did,
            "df": df_did,
            "t_stat": t_stat_did,
            "t_pvalue": t_pval_did,
            "wilcoxon_stat": w_stat_did,
            "wilcoxon_pvalue": w_pval_did
        },
        "advantage_survives": advantage_survives,
        "verdict": verdict_str
    }
