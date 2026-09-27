#!/usr/bin/env python3
"""
experiments/s0_8_relocate.py -- Directive S0-8: Relocating the Write Target
Target: transformer.h.L.mlp.c_proj.weight with Completely Frozen Readout.
Sweeps: L in [1, 6, 10] spanning depth (early, middle, late).
Primary Endpoint: First-50-edit retention (N=300) vs worst control floor via Newcombe interval.
Secondary Endpoint: First-50-edit paraphrase generalization (N=900).
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
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

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
    wilson_confidence_interval,
    format_wilson_rate,
    CONTROL_NAMES,
    pool_controls,
    assert_orthonormality
)
from experiments.stats import (
    newcombe_score_interval,
    compute_minimum_detectable_effect,
    format_wilcoxon_result
)
SEEDS = [0, 1, 2, 3, 4, 5]


def configure_determinism(seed: int = 42, warn_only: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for sdp in ['enable_mem_efficient_sdp', 'enable_flash_sdp']:
        if hasattr(torch.backends.cuda, sdp):
            getattr(torch.backends.cuda, sdp)(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as e:
        print(f"Warning setting deterministic algorithms: {e}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def project_orthogonal(v: torch.Tensor, Q: Optional[torch.Tensor]) -> torch.Tensor:
    if Q is None or Q.numel() == 0:
        return v
    return v - torch.matmul(v, torch.matmul(Q, Q.T))


def greedy_predict(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 5,
    device: str = "cuda",
    train_mode: bool = False
) -> str:
    model.eval() if not train_mode else model.train()
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def get_next_token_log_probs(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: str = "cuda",
    train_mode: bool = False
) -> torch.Tensor:
    model.eval() if not train_mode else model.train()
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]
    return F.log_softmax(logits, dim=-1).cpu()

# Chosen layers spanning depth: Layer 1 (early), Layer 6 (middle), Layer 10 (late)
SWEPT_LAYERS = [1, 6, 10]
CANDIDATE_RANDOM_LAYERS = [0, 2, 3, 4, 5, 7, 8, 9, 11]  # Excludes swept layers 1, 6, 10


def freeze_readout(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Completely freezes all readout parameters and returns initial clones for bitwise assertions:
    - lm_head.weight
    - transformer.wte.weight
    - transformer.ln_f.weight
    - transformer.ln_f.bias
    """
    model.lm_head.weight.requires_grad = False
    model.transformer.wte.weight.requires_grad = False
    model.transformer.ln_f.weight.requires_grad = False
    model.transformer.ln_f.bias.requires_grad = False

    return {
        "lm_head": model.lm_head.weight.data.clone(),
        "wte": model.transformer.wte.weight.data.clone(),
        "ln_f_w": model.transformer.ln_f.weight.data.clone(),
        "ln_f_b": model.transformer.ln_f.bias.data.clone()
    }


def assert_readout_frozen(model: nn.Module, initial_readout: Dict[str, torch.Tensor], seed: int, arm: str) -> None:
    """
    Asserts bitwise-zero parameter delta on all readout parameters and prints the assertion.
    """
    d_lm_head = torch.max(torch.abs(model.lm_head.weight.data - initial_readout["lm_head"])).item()
    d_wte = torch.max(torch.abs(model.transformer.wte.weight.data - initial_readout["wte"])).item()
    d_ln_f_w = torch.max(torch.abs(model.transformer.ln_f.weight.data - initial_readout["ln_f_w"])).item()
    d_ln_f_b = torch.max(torch.abs(model.transformer.ln_f.bias.data - initial_readout["ln_f_b"])).item()

    assert d_lm_head == 0.0, f"Readout freeze violated in {arm} seed {seed}: lm_head delta = {d_lm_head}"
    assert d_wte == 0.0, f"Readout freeze violated in {arm} seed {seed}: wte delta = {d_wte}"
    assert d_ln_f_w == 0.0, f"Readout freeze violated in {arm} seed {seed}: ln_f_w delta = {d_ln_f_w}"
    assert d_ln_f_b == 0.0, f"Readout freeze violated in {arm} seed {seed}: ln_f_b delta = {d_ln_f_b}"

    print(f"    [Frozen Readout Assertion] Seed {seed} {arm}: lm_head={d_lm_head:.1f}, wte={d_wte:.1f}, ln_f.w={d_ln_f_w:.1f}, ln_f.b={d_ln_f_b:.1f} (Bitwise Zero Verified)")


def edit_fact_mlp_sgd(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    layer_idx: int,
    lr: float = 3.0e-05,
    max_steps: int = 100,
    device: str = "cuda",
    Q_causal: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Performs sequential SGD edit targeting transformer.h[L].mlp.c_proj.weight only,
    with zero-margin stopping rule (delta=0.0) and max_steps=100.
    """
    target_param = model.transformer.h[layer_idx].mlp.c_proj.weight
    # Freeze all parameters and enable gradient on target parameter only
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

    steps_taken = 0
    curr_pred = ""
    w_pre = target_param.data.clone()

    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            out.loss.backward()
            del out

            if Q_causal is not None and Q_causal.numel() > 0:
                target_param.grad.copy_(project_orthogonal(target_param.grad, Q_causal))

            optimizer.step()
            curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
            if check_match(curr_pred, fact["object"]):
                break

    immediate_match = check_match(curr_pred, fact["object"])
    delta_applied = (target_param.data - w_pre).detach()
    delta_norm = torch.linalg.norm(delta_applied).item()

    model.zero_grad(set_to_none=True)
    del optimizer, input_ids, labels, w_pre

    return {
        "steps_taken": steps_taken,
        "immediate_match": immediate_match,
        "delta_applied": delta_applied,
        "delta_norm": delta_norm,
        "pred": curr_pred
    }


def evaluate_s0_8_arm(
    model: nn.Module,
    facts_list: List[Dict[str, Any]],
    edit_results: List[Dict[str, Any]],
    arm_name: str,
    seed: int,
    fresh_model: nn.Module,
    tokenizer: Any,
    wikitext_slice: Any,
    slice_sha: str,
    device: str
) -> Dict[str, Any]:
    """
    Evaluates sequence outcomes for Directive S0-8:
    - Immediate efficacy (N=200)
    - Terminal retention: all 200 edits and first 50 edits
    - Paraphrase generalization: all 600 paraphrases and first 150 paraphrases
    - Wikitext-2 perplexity & locality KL
    """
    preds_term = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_list]
    para_preds = [[greedy_predict(model, tokenizer, p, 5, device, False) for p in f["paraphrases"]] for f in facts_list]

    imm_matches = [r["immediate_match"] for r in edit_results]
    term_matches = [check_match(p, f["object"]) for p, f in zip(preds_term, facts_list)]
    steps_taken = [r["steps_taken"] for r in edit_results]

    para_flat_matches = []
    for p_list, f in zip(para_preds, facts_list):
        for p in p_list:
            para_flat_matches.append(check_match(p, f["object"]))

    # First 50 edits evaluations
    first50_facts = facts_list[:50]
    first50_imm = imm_matches[:50]
    first50_term = term_matches[:50]
    first50_para_matches = []
    for p_list, f in zip(para_preds[:50], first50_facts):
        for p in p_list:
            first50_para_matches.append(check_match(p, f["object"]))

    m_imm = Measurement.from_outcomes(imm_matches, metric="immediate_efficacy", arm=arm_name, scope="per_seed", input_set=f"seq_s{seed}", mode="eval_no_dropout")
    m_term = Measurement.from_outcomes(term_matches, metric="terminal_retention", arm=arm_name, scope="per_seed", input_set=f"seq_s{seed}", mode="eval_no_dropout")
    m_term_f50 = Measurement.from_outcomes(first50_term, metric="terminal_retention", arm=arm_name, scope="first50_per_seed", input_set=f"seq_s{seed}_f50", mode="eval_no_dropout")
    m_gen_f50 = Measurement.from_outcomes(first50_para_matches, metric="generalization", arm=arm_name, scope="first50_generalization_per_seed", input_set=f"seq_s{seed}_gen_f50", mode="eval_no_dropout")

    # Capability evaluation
    ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)

    # Locality KL on sample of neighborhood prompts
    pre_lps = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for f in facts_list[:20] for p in f["neighborhood_prompts"]}
    post_lps = {p: get_next_token_log_probs(model, tokenizer, p, device, False) for f in facts_list[:20] for p in f["neighborhood_prompts"]}
    kl_vals = []
    for p_str, lp_pre in pre_lps.items():
        lp_post = post_lps[p_str]
        kl = F.kl_div(lp_post, lp_pre, log_target=True, reduction="batchmean").item()
        kl_vals.append(max(0.0, kl))
    loc_kl = sum(kl_vals) / float(max(1, len(kl_vals)))

    total_steps = sum(steps_taken)

    return {
        "immediate_efficacy": m_imm,
        "terminal_retention": m_term,
        "first50_retention": m_term_f50,
        "first50_generalization": m_gen_f50,
        "raw_vectors": {
            "immediate_matches": imm_matches,
            "terminal_matches": term_matches,
            "paraphrase_matches": para_flat_matches,
            "first50_term_matches": first50_term,
            "first50_para_matches": first50_para_matches,
            "steps_taken": steps_taken
        },
        "perplexity": ppl,
        "locality_kl": loc_kl,
        "optimizer_steps": total_steps,
        "mean_steps": total_steps / len(facts_list),
        "applied_norms": [r["delta_norm"] for r in edit_results]
    }


def run_random_layer_control(
    model_name: str,
    pinned_revision: str,
    tokenizer: Any,
    fresh_model: nn.Module,
    sequences: Dict[int, List[Dict[str, Any]]],
    focal_norms_by_seed: Dict[int, List[float]],
    wikitext_slice: Any,
    slice_sha: str,
    device: str
) -> Dict[int, Any]:
    """
    Executes random_layer_magnitude_matched control:
    For each edit, chooses a non-swept layer uniformly at random from CANDIDATE_RANDOM_LAYERS,
    generates a rank-1 perturbation with Frobenius norm identically matching the focal arm's
    observed delta norm at that edit, and evaluates sequential outcomes with frozen readout.
    """
    print("\n--- [Running Control: random_layer_magnitude_matched across 6 seeds] ---")
    ctrl_results = {}

    for s in SEEDS:
        seq_facts = sequences[s]
        rng = random.Random(s + 8000)
        rng_torch = torch.Generator(device=device).manual_seed(s + 8000)

        model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        init_ro = freeze_readout(model)

        observed_norms = focal_norms_by_seed.get(s, [1.0e-3] * len(seq_facts))
        edit_res = []

        for t_idx, f in enumerate(seq_facts):
            chosen_layer = rng.choice(CANDIDATE_RANDOM_LAYERS)
            target_p = model.transformer.h[chosen_layer].mlp.c_proj.weight

            # Rank-1 perturbation matched to focal norm: delta = norm * (u / ||u||) (v / ||v||)^T
            target_norm = observed_norms[t_idx] if t_idx < len(observed_norms) else 1.0e-3
            u = torch.randn(3072, 1, generator=rng_torch, device=device)
            v = torch.randn(768, 1, generator=rng_torch, device=device)
            u_unit = u / torch.norm(u)
            v_unit = v / torch.norm(v)
            pert = (u_unit @ v_unit.T) * target_norm

            with torch.no_grad():
                target_p.add_(pert)

            pred_imm = greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False)
            imm_match = check_match(pred_imm, f["object"])
            edit_res.append({
                "steps_taken": 0,
                "immediate_match": imm_match,
                "delta_norm": target_norm,
                "pred": pred_imm
            })

        assert_readout_frozen(model, init_ro, s, "random_layer_magnitude_matched")
        ev_ctrl = evaluate_s0_8_arm(
            model, seq_facts, edit_res, "random_layer_magnitude_matched",
            s, fresh_model, tokenizer, wikitext_slice, slice_sha, device
        )
        ctrl_results[s] = ev_ctrl
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_ctrl['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_ctrl['terminal_retention'])} | First50={format_wilson_rate(ev_ctrl['first50_retention'])}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return ctrl_results
