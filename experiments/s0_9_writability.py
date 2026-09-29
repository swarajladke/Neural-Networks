#!/usr/bin/env python3
"""
experiments/s0_9_writability.py -- Directive S0-9: MLP Writability Engine
Target: transformer.h[L].mlp.c_proj.weight
Positive Control First: Establish whether the mid-layer MLP value projection
is writable under single-edit conditions before any sequential experiment.

Arms:
  - Stage P: Path verification on 20 facts at L=6 (hook, grad check, large step, weight delta).
  - Arm W1: Iterative rank-1 SGD across learning rate grid (3.0e-5, 3.0e-4, 3.0e-3).
  - Arm W2: Closed-form rank-1 key-value update: Δ = (v* - W k) k^T / (k^T k).

Strict structural limit: under 600 lines (AGENTS.md §7.1).
"""

import os
import gc
import sys
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (
    evaluate_wikitext_perplexity
)
from experiments.metrics import (
    Measurement,
    check_match,
    normalize_entity,
    wilson_confidence_interval,
    format_wilson_rate,
    compute_locality_kl
)
from experiments.b1_inject import (
    greedy_predict,
    get_next_token_log_probs
)

# S0-8 baseline learning rate
S0_8_BASE_LR = 3.0e-05

# Pre-declared SGD learning rate grid spanning two orders of magnitude around S0-8 rate
W1_LR_GRID = [3.0e-05, 3.0e-04, 3.0e-03]

# Closed-form value optimization parameters (declared in advance)
W2_MAX_STEPS = 20
W2_LR_V = 0.1
W2_LAMBDA_L2 = 0.5

# Pinned layers for writability evaluation (subject to budget pruning in order: 1, 11, 3)
CANDIDATE_LAYERS = [1, 3, 6, 9, 11]


def get_subject_last_token_idx(tokenizer: Any, prompt: str, subject: str) -> int:
    """
    Finds the 0-indexed position of the subject's final token in the tokenized prompt.
    Works deterministically for both subject-prefix prompts and mid-prompt subjects.
    """
    if prompt.startswith(subject):
        sub_tokens = tokenizer.encode(subject)
        return len(sub_tokens) - 1
    char_idx = prompt.find(subject)
    if char_idx == -1:
        # Fallback to token count of subject
        return len(tokenizer.encode(subject)) - 1
    prefix = prompt[:char_idx]
    full_sub_tokens = tokenizer.encode(prefix + subject)
    return len(full_sub_tokens) - 1


def run_stage_p_path_verification(
    model: nn.Module,
    tokenizer: Any,
    facts_20: List[Dict[str, Any]],
    base_state_dict: Dict[str, torch.Tensor],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Stage P: Path verification on seed 0, first 20 facts at Layer L=6.
    Fresh model state per fact.
    Asserts:
      1. c_proj.weight.requires_grad is True and gradient is nonzero and finite.
      2. Single unconstrained step increases target log-probability.
      3. Edited weight differs from original (delta_norm > 0).
    Halts if target log-probability does not rise for any of the 20 facts.
    """
    layer_idx = 6
    results_per_fact = []
    print("\n--- [Stage P: Write-Path Verification (L=6, 20 Facts, Fresh Model Each)] ---")

    for idx, fact in enumerate(facts_20):
        # Reset model to fresh pinned base state
        model.load_state_dict(base_state_dict)
        target_param = model.transformer.h[layer_idx].mlp.c_proj.weight

        for p in model.parameters():
            p.requires_grad = False
        target_param.requires_grad = True
        assert target_param.requires_grad, "Assertion failure: c_proj.weight.requires_grad is not True!"

        # 1. Forward hook to record key vector k and output v at subject last token
        subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
        recorded_acts = {}

        def hook_fn(module, inp, out):
            recorded_acts["k"] = inp[0][0, subj_idx, :].detach().clone()
            recorded_acts["v"] = out[0, subj_idx, :].detach().clone()

        h_hook = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(hook_fn)

        enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**enc_prompt)
        h_hook.remove()

        assert "k" in recorded_acts and "v" in recorded_acts
        k_norm = torch.linalg.norm(recorded_acts["k"]).item()
        v_norm = torch.linalg.norm(recorded_acts["v"]).item()

        # 2. Compute edit loss and assert gradient is nonzero and finite
        full_text = f"{fact['edit_prompt']} {fact['object']}"
        enc_full = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = enc_full.input_ids
        labels = input_ids.clone()
        prompt_len = enc_prompt.input_ids.shape[1]
        labels[:, :prompt_len] = -100
        target_tok = input_ids[0, prompt_len].item()

        # Pre-edit target log-probability
        with torch.no_grad():
            pre_logits = model(**enc_prompt).logits[0, -1, :]
            log_prob_before = F.log_softmax(pre_logits, dim=-1)[target_tok].item()

        # Compute gradient
        model.zero_grad(set_to_none=True)
        out = model(input_ids, labels=labels)
        loss = out.loss
        loss.backward()

        grad = target_param.grad
        assert grad is not None, "Assertion failure: target_param.grad is None!"
        assert torch.all(torch.isfinite(grad)), "Assertion failure: target_param.grad contains non-finite values!"
        grad_norm = torch.linalg.norm(grad).item()
        assert grad_norm > 0.0, f"Assertion failure: grad_norm is zero ({grad_norm})!"

        # 3. Apply a single large unconstrained step and confirm log-prob increases
        w_orig = target_param.data.clone()
        large_step_lr = 0.01
        with torch.no_grad():
            target_param.data.add_(-large_step_lr * grad)

        delta_norm = torch.linalg.norm(target_param.data - w_orig).item()
        assert delta_norm > 0.0, "Assertion failure: target_param weight did not change after step!"

        with torch.no_grad():
            post_logits = model(**enc_prompt).logits[0, -1, :]
            log_prob_after = F.log_softmax(post_logits, dim=-1)[target_tok].item()

        prob_delta = log_prob_after - log_prob_before
        if prob_delta <= 0.0:
            print(f"FATAL WRITE-PATH DEFECT on fact {idx} ('{fact['subject']}'): log-prob did not increase ({log_prob_before:.4f} -> {log_prob_after:.4f})")
            sys.exit(1)

        print(f"  Fact {idx:02d} | Subj: {fact['subject'][:16]:<16s} | ||k||={k_norm:.2f} | ||grad||={grad_norm:.6f} | LogP: {log_prob_before:.4f} -> {log_prob_after:.4f} (Δ={prob_delta:+.4f}) | ||ΔW||={delta_norm:.6f}")
        results_per_fact.append({
            "fact_id": fact.get("fact_id", idx),
            "subject": fact["subject"],
            "k_norm": k_norm,
            "v_norm": v_norm,
            "grad_norm": grad_norm,
            "log_prob_before": log_prob_before,
            "log_prob_after": log_prob_after,
            "delta_norm": delta_norm,
            "prob_increased": True
        })

    print("--- [Stage P: All 20 Facts Verified. Write Path at L=6 Confirmed Operable] ---\n")
    return {
        "passed": True,
        "n_facts": len(facts_20),
        "results": results_per_fact
    }


def edit_fact_mlp_sgd_rate(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    layer_idx: int,
    lr: float,
    max_steps: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Iterative rank-1 SGD editing targeting transformer.h[L].mlp.c_proj.weight.
    Stopping condition: checks check_match on greedy decode when top token matches target.
    """
    target_param = model.transformer.h[layer_idx].mlp.c_proj.weight
    for p in model.parameters():
        p.requires_grad = False
    target_param.requires_grad = True

    optimizer = torch.optim.SGD([target_param], lr=lr)

    full_text = f"{fact['edit_prompt']} {fact['object']}"
    enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt")
    enc_full = tokenizer(full_text, return_tensors="pt")
    prompt_len = enc_prompt["input_ids"].shape[1]

    input_ids = enc_full["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    primary_tok = input_ids[0, prompt_len].item()
    steps_taken = 0
    curr_pred = ""
    w_pre = target_param.data.clone()

    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            p_logits = out.logits[0, prompt_len - 1, :]
            top_tok = torch.argmax(p_logits).item()
            out.loss.backward()
            optimizer.step()

            if top_tok == primary_tok:
                curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
                if check_match(curr_pred, fact["object"]):
                    break

    if not curr_pred:
        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    immediate_match = check_match(curr_pred, fact["object"])
    delta_applied = (target_param.data - w_pre).detach()
    delta_norm = torch.linalg.norm(delta_applied).item()

    model.zero_grad(set_to_none=True)
    del optimizer, input_ids, labels, w_pre

    return {
        "steps_taken": steps_taken,
        "immediate_match": immediate_match,
        "delta_norm": delta_norm,
        "pred": curr_pred
    }


def edit_fact_mlp_closed_form(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    layer_idx: int,
    max_steps: int = W2_MAX_STEPS,
    lr_v: float = W2_LR_V,
    lambda_l2: float = W2_LAMBDA_L2,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Closed-form rank-1 key-value update (ROME-style):
    1. Records key vector k and initial value v0 at subject's final token position.
    2. Directly optimizes v* with Adam to maximize target log-prob with L2 penalty toward v0.
    3. Computes rank-1 weight update Δ = outer(k, (v* - k W)) / (k^T k).
    4. Applies Δ to c_proj.weight.data.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    for p in model.parameters():
        p.requires_grad = False

    subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])

    # Step 1: Record key vector k and original value v0
    recorded_acts = {}

    def hook_rec(module, inp, out):
        recorded_acts["k"] = inp[0][0, subj_idx, :].detach().clone()
        recorded_acts["v0"] = out[0, subj_idx, :].detach().clone()

    h_rec = c_proj.register_forward_hook(hook_rec)
    enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**enc_prompt)
    h_rec.remove()

    k_vec = recorded_acts["k"]      # (3072,)
    v0_vec = recorded_acts["v0"]    # (768,)

    # Step 2: Optimize target value vector v*
    v_param = nn.Parameter(v0_vec.clone())
    opt_v = torch.optim.Adam([v_param], lr=lr_v)

    full_text = f"{fact['edit_prompt']} {fact['object']}"
    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = enc_full.input_ids
    labels = input_ids.clone()
    prompt_len = enc_prompt.input_ids.shape[1]
    labels[:, :prompt_len] = -100
    primary_tok = input_ids[0, prompt_len].item()

    def hook_rep(module, inp, out):
        out_mod = out.clone()
        out_mod[0, subj_idx, :] = v_param
        return out_mod

    h_rep = c_proj.register_forward_hook(hook_rep)
    steps_taken = 0

    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            opt_v.zero_grad()
            out = model(input_ids, labels=labels)
            loss_ce = out.loss
            loss_l2 = lambda_l2 * torch.sum((v_param - v0_vec) ** 2)
            loss = loss_ce + loss_l2
            loss.backward()
            opt_v.step()

            top_tok = torch.argmax(out.logits[0, prompt_len - 1, :]).item()
            if top_tok == primary_tok:
                break

    h_rep.remove()

    # Step 3: Compute closed-form rank-1 update to c_proj.weight
    # In GPT-2 Conv1D: W has shape (3072, 768), forward is x @ W.
    # At subj_idx, output is k @ W.
    # We want: k @ (W + Δ) = v* => k @ Δ = v* - k @ W.
    # rank-1 solution: Δ = outer(k, (v* - k @ W)) / dot(k, k).
    w_curr = c_proj.weight.data  # (3072, 768)
    lin_pred = torch.matmul(k_vec, w_curr)  # (768,)
    v_star = v_param.detach()
    delta_v = v_star - lin_pred  # (768,)
    k_norm_sq = torch.dot(k_vec, k_vec) + 1e-12
    delta_w = torch.outer(k_vec, delta_v) / k_norm_sq  # (3072, 768)

    c_proj.weight.data.add_(delta_w)
    delta_norm = torch.linalg.norm(delta_w).item()

    # Step 4: Verify prediction with greedy decode on edited weights
    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    immediate_match = check_match(curr_pred, fact["object"])

    del v_param, opt_v, input_ids, labels, delta_w

    return {
        "steps_taken": steps_taken,
        "immediate_match": immediate_match,
        "delta_norm": delta_norm,
        "pred": curr_pred
    }


def evaluate_capability_and_locality(
    model: nn.Module,
    fresh_model: nn.Module,
    tokenizer: Any,
    control_probes: List[Dict[str, Any]],
    wikitext_slice: torch.Tensor,
    slice_sha: str,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates WikiText-2 perplexity and locality KL on control probes.
    Locality KL non-zero guard: asserts locality KL is strictly nonzero when perplexity moves.
    """
    ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)

    # Locality KL over control probes (200 prompts)
    probe_prompts = [c["prompt"] for c in control_probes]
    pre_lps = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for p in probe_prompts}
    post_lps = {p: get_next_token_log_probs(model, tokenizer, p, device, False) for p in probe_prompts}
    loc_kl = compute_locality_kl(pre_lps, post_lps)

    # Locality KL guard: if perplexity moved significantly from baseline (36.03), locality KL cannot be zero
    if abs(ppl - 36.03) > 0.05:
        assert loc_kl > 0.0, f"Locality KL guard failure: PPL moved to {ppl:.2f} but locality KL printed as {loc_kl:.6f}!"

    return {
        "perplexity": ppl,
        "locality_kl": loc_kl,
        "num_probes": len(probe_prompts)
    }
