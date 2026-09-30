#!/usr/bin/env python3
"""
experiments/s0_10_repair.py -- Directive S0-10: MLP Writability Repair & Diagnostic Engine
Platform: Kaggle Tesla T4 GPU / Python 3.12 / PyTorch 2.10.0+cu128 / Transformers 5.0.0
Strict structural limit: under 600 lines (AGENTS.md §7.1).
"""

import os, gc, sys, math, time, json, random, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import evaluate_wikitext_perplexity
from experiments.metrics import (
    Measurement, check_match, normalize_entity,
    wilson_confidence_interval, format_wilson_rate, compute_locality_kl
)
from experiments.b1_inject import greedy_predict, get_next_token_log_probs
from experiments.s0_8_relocate import freeze_readout, assert_readout_frozen

N_PPL_SUBSET_SEQS = 100


def verify_state_restore(model: nn.Module, fresh_param_sum: float, fresh_c_proj_hashes: Dict[int, str]) -> None:
    curr_sum = sum(p.sum().item() for p in model.parameters())
    assert abs(curr_sum - fresh_param_sum) < 1e-4, f"Restore failure: param sum {curr_sum} != {fresh_param_sum}"
    for l_idx, exp_hash in fresh_c_proj_hashes.items():
        w_bytes = model.transformer.h[l_idx].mlp.c_proj.weight.data.cpu().numpy().tobytes()
        act_hash = hashlib.sha256(w_bytes).hexdigest()
        assert act_hash == exp_hash, f"Restore failure at L{l_idx}: hash {act_hash} != {exp_hash}"


def get_subject_last_token_idx(tokenizer: Any, prompt: str, subject: str) -> int:
    if prompt.startswith(subject):
        return len(tokenizer.encode(subject)) - 1
    char_idx = prompt.find(subject)
    if char_idx == -1:
        return len(tokenizer.encode(subject)) - 1
    return len(tokenizer.encode(prompt[:char_idx] + subject)) - 1


def edit_fact_mlp_fullgrad_sgd(
    model: nn.Module, tokenizer: Any, fact: Dict[str, Any],
    layer_idx: int, lr: float, max_steps: int = 100, device: str = "cuda"
) -> Dict[str, Any]:
    target_param = model.transformer.h[layer_idx].mlp.c_proj.weight
    for p in model.parameters(): p.requires_grad = False
    target_param.requires_grad = True
    optimizer = torch.optim.SGD([target_param], lr=lr)

    enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt")
    prompt_len = enc_prompt.input_ids.shape[1]
    enc_full = tokenizer(f"{fact['edit_prompt']} {fact['object']}", return_tensors="pt")
    input_ids = enc_full["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    primary_tok = input_ids[0, prompt_len].item()

    steps_taken, curr_pred, w_pre = 0, "", target_param.data.clone()
    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            if torch.argmax(out.logits[0, prompt_len - 1, :]).item() == primary_tok:
                curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
                if check_match(curr_pred, fact["object"]):
                    out.loss.backward(); optimizer.step()
                    break
            out.loss.backward()
            optimizer.step()

    if not curr_pred:
        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])
    delta_norm = torch.linalg.norm((target_param.data - w_pre).detach()).item()
    model.zero_grad(set_to_none=True)
    del optimizer, input_ids, labels, w_pre
    return {"steps_taken": steps_taken, "immediate_match": imm_match, "delta_norm": delta_norm, "pred": curr_pred}


def edit_fact_mlp_closed_form_repaired(
    model: nn.Module, tokenizer: Any, fact: Dict[str, Any],
    layer_idx: int, max_steps: int = 20, lr_v: float = 0.1, lambda_l2: float = 0.0, device: str = "cuda"
) -> Dict[str, Any]:
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    for p in model.parameters(): p.requires_grad = False
    subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])

    recorded = {}
    def hook_rec(module, inp, out):
        recorded["k"] = inp[0][0, subj_idx, :].detach().clone()
        recorded["v0"] = out[0, subj_idx, :].detach().clone()

    h_rec = c_proj.register_forward_hook(hook_rec)
    try:
        enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
        with torch.no_grad(): _ = model(**enc_prompt)
    finally:
        h_rec.remove()

    k_vec, v0_vec = recorded["k"], recorded["v0"]
    v_param = nn.Parameter(v0_vec.clone())
    opt_v = torch.optim.Adam([v_param], lr=lr_v)

    enc_full = tokenizer(f"{fact['edit_prompt']} {fact['object']}", return_tensors="pt").to(device)
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
    try:
        with torch.set_grad_enabled(True):
            for _ in range(max_steps):
                steps_taken += 1
                opt_v.zero_grad()
                out = model(input_ids, labels=labels)
                loss_l2 = lambda_l2 * torch.sum((v_param - v0_vec) ** 2) if lambda_l2 > 0.0 else 0.0
                loss = out.loss + loss_l2
                loss.backward(); opt_v.step()
                if torch.argmax(out.logits[0, prompt_len - 1, :]).item() == primary_tok:
                    break
    finally:
        h_rep.remove()

    v_star = v_param.detach()
    delta_w = torch.outer(k_vec, v_star - v0_vec) / (torch.dot(k_vec, k_vec) + 1e-12)
    c_proj.weight.data.add_(delta_w)
    delta_norm = torch.linalg.norm(delta_w).item()

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])
    del v_param, opt_v, input_ids, labels, delta_w
    return {"steps_taken": steps_taken, "immediate_match": imm_match, "delta_norm": delta_norm, "pred": curr_pred}


def run_stage_d_diagnostic(
    model: nn.Module, tokenizer: Any, facts_20: List[Dict[str, Any]],
    base_state_dict: Dict[str, torch.Tensor], fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str], device: str = "cuda"
) -> Dict[str, Any]:
    layer_idx = 6
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    diag_rows = []
    c_a, c_b, c_c, c_s = 0, 0, 0, 0
    print("\n--- [Stage D: Closed-Form Write Diagnostic (20 Facts, L=6)] ---")

    for f_idx, fact in enumerate(facts_20):
        model.load_state_dict(base_state_dict)
        verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
        subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])

        recorded = {}
        def hook_rec(m, inp, out):
            recorded["k"] = inp[0][0, subj_idx, :].detach().clone()
            recorded["v0"] = out[0, subj_idx, :].detach().clone()
        h_rec = c_proj.register_forward_hook(hook_rec)
        try:
            enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
            with torch.no_grad(): _ = model(**enc_prompt)
        finally:
            h_rec.remove()

        k_vec, v0_vec = recorded["k"], recorded["v0"]
        v_param = nn.Parameter(v0_vec.clone())
        opt_v = torch.optim.Adam([v_param], lr=0.1)

        enc_full = tokenizer(f"{fact['edit_prompt']} {fact['object']}", return_tensors="pt").to(device)
        input_ids = enc_full.input_ids
        labels = input_ids.clone()
        prompt_len = enc_prompt.input_ids.shape[1]
        labels[:, :prompt_len] = -100
        primary_tok = input_ids[0, prompt_len].item()

        def hook_rep(m, inp, out):
            out_mod = out.clone(); out_mod[0, subj_idx, :] = v_param; return out_mod
        h_rep = c_proj.register_forward_hook(hook_rep)
        trace = []
        try:
            with torch.set_grad_enabled(True):
                for step_i in range(20):
                    opt_v.zero_grad(); out = model(input_ids, labels=labels)
                    (out.loss + 0.5 * torch.sum((v_param - v0_vec) ** 2)).backward(); opt_v.step()
                    with torch.no_grad():
                        lpi = F.log_softmax(out.logits[0, prompt_len - 1, :], dim=-1)[primary_tok].item()
                    trace.append({"step": step_i + 1, "log_prob": lpi, "dist": torch.linalg.norm(v_param - v0_vec).item()})
        finally:
            h_rep.remove()

        v_star = v_param.detach()
        def hook_patch(m, inp, out):
            out_mod = out.clone(); out_mod[0, subj_idx, :] = v_star; return out_mod
        h_patch = c_proj.register_forward_hook(hook_patch)
        try:
            with torch.no_grad():
                out_p = model(**enc_prompt)
                prob_patched = F.softmax(out_p.logits[0, -1, :], dim=-1)[primary_tok].item()
            pred_patched = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
            match_patched = check_match(pred_patched, fact["object"])
        finally:
            h_patch.remove()

        k_norm_sq = torch.dot(k_vec, k_vec) + 1e-12
        delta_w_s09 = torch.outer(k_vec, v_star - torch.matmul(k_vec, c_proj.weight.data)) / k_norm_sq
        err_s09 = torch.max(torch.abs(torch.matmul(k_vec, c_proj.weight.data + delta_w_s09) + c_proj.bias.data - v_star)).item()

        delta_w_rep = torch.outer(k_vec, v_star - v0_vec) / k_norm_sq
        err_rep = torch.max(torch.abs(torch.matmul(k_vec, c_proj.weight.data + delta_w_rep) + c_proj.bias.data - v_star)).item()

        c_proj.weight.data.add_(delta_w_rep)
        pred_pw = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
        match_pw = check_match(pred_pw, fact["object"])

        if not match_patched:
            f_class = "A_OPTIMIZATION_FAILURE"; c_a += 1
        elif match_patched and not match_pw:
            f_class = "B_WRITE_DEFECT"; c_b += 1
        elif match_patched and match_pw:
            f_class = "SUCCESS"; c_s += 1
        else:
            f_class = "C_PROPAGATION_FAILURE"; c_c += 1

        diag_rows.append({
            "fact_id": f_idx, "subject": fact["subject"], "initial_log_prob": trace[0]["log_prob"],
            "final_log_prob": trace[-1]["log_prob"], "final_dist": trace[-1]["dist"],
            "patched_target_prob": prob_patched, "match_patched": match_patched,
            "err_s09_no_bias": err_s09, "err_repaired_with_bias": err_rep,
            "pred_post_write": pred_pw, "match_post_write": match_pw, "classification": f_class
        })
        p_str = "YES" if match_patched else "NO"
        w_str = "YES" if match_pw else "NO"
        print(f"  Fact {f_idx:02d} | Subj: {fact['subject'][:16]:<16s} | LogP: {trace[0]['log_prob']:.2f}->{trace[-1]['log_prob']:.2f} | Dist: {trace[-1]['dist']:.2f} | Patched: {p_str} | Err_S09: {err_s09:.3f} | Err_Rep: {err_rep:.2e} | PostWrite: {w_str} | Class: {f_class}")

    print(f"\n  Stage D Diagnostic Failure Classification (N=20 Facts at L=6):")
    print(f"    (a) Optimization Failure : {c_a}/20")
    print(f"    (b) Write Defect         : {c_b}/20")
    print(f"    (c) Propagation Failure  : {c_c}/20")
    print(f"    Initial Repaired Matches : {c_s}/20")

    print("\n--- [Stage D: Pre-Declared Repair Grid on 20 Facts] ---")
    grid_lambdas = [0.0, 0.05, 0.5]
    grid_steps = [20, 100]
    repair_grid = []
    best_eff, best_setting = -1.0, {"lambda_l2": 0.0, "max_steps": 100}

    for l_val in grid_lambdas:
        for s_val in grid_steps:
            w_matches = 0
            for fact in facts_20:
                model.load_state_dict(base_state_dict)
                res = edit_fact_mlp_closed_form_repaired(
                    model, tokenizer, fact, layer_idx=layer_idx,
                    max_steps=s_val, lr_v=0.1, lambda_l2=l_val, device=device
                )
                if res["immediate_match"]: w_matches += 1
            eff_pct = (w_matches / len(facts_20)) * 100.0
            repair_grid.append({"lambda_l2": l_val, "max_steps": s_val, "matches": w_matches, "total": 20, "efficacy_pct": eff_pct})
            print(f"    Grid cell: lambda={l_val:<4.2f}, steps={s_val:<3d} -> Efficacy: {w_matches}/20 ({eff_pct:.1f}%)")
            if eff_pct > best_eff:
                best_eff, best_setting = eff_pct, {"lambda_l2": l_val, "max_steps": s_val}

    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
    print(f"  Best Stage D Setting: lambda={best_setting['lambda_l2']}, steps={best_setting['max_steps']} (Efficacy: {best_eff:.1f}%)\n")
    return {
        "diagnostic_rows": diag_rows,
        "classification_counts": {"optimization_failure": c_a, "write_defect": c_b, "propagation_failure": c_c, "success": c_s},
        "repair_grid": repair_grid, "best_setting": best_setting
    }


def evaluate_s0_10_capability_and_locality(
    model: nn.Module, fresh_model: nn.Module, tokenizer: Any,
    control_probes: List[Dict[str, Any]], wikitext_slice: torch.Tensor,
    slice_sha: str, subset_baseline_ppl: float, device: str = "cuda",
    max_ppl_sequences: Optional[int] = N_PPL_SUBSET_SEQS
) -> Dict[str, Any]:
    ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device, max_sequences=max_ppl_sequences)
    delta_ppl = ppl - subset_baseline_ppl
    probe_prompts = [c["prompt"] for c in control_probes]
    pre_lps = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for p in probe_prompts}
    post_lps = {p: get_next_token_log_probs(model, tokenizer, p, device, False) for p in probe_prompts}
    loc_kl = compute_locality_kl(pre_lps, post_lps)
    if abs(delta_ppl) > 0.20:
        assert loc_kl > 0.0, f"Locality KL guard failure: delta_ppl={delta_ppl:.2f} but loc_kl={loc_kl:.6f}"
    return {"perplexity": ppl, "delta_ppl": delta_ppl, "locality_kl": loc_kl, "num_probes": len(probe_prompts)}


def evaluate_s0_10_negative_controls(
    base_state_dict: Dict[str, torch.Tensor], model: nn.Module, tokenizer: Any,
    facts_100: List[Dict[str, Any]], facts_pool: List[Dict[str, Any]],
    edit_proc_type: str, layer_idx: int, proc_params: Dict[str, Any],
    device: str = "cuda", seed: int = 42
) -> Dict[str, Any]:
    rng = random.Random(seed)
    wrong_c_matches, wrong_p_matches = [], []
    for fact in facts_100:
        cands = [c["object"] for c in facts_pool if c["relation"] == fact["relation"] and normalize_entity(c["object"]) != normalize_entity(fact["object"])]
        wrong_fact = {**fact, "object": rng.choice(cands)}
        model.load_state_dict(base_state_dict)
        if edit_proc_type == "sgd":
            edit_fact_mlp_fullgrad_sgd(model, tokenizer, wrong_fact, layer_idx=layer_idx, lr=proc_params["lr"], max_steps=proc_params["max_steps"], device=device)
        else:
            edit_fact_mlp_closed_form_repaired(model, tokenizer, wrong_fact, layer_idx=layer_idx, max_steps=proc_params["max_steps"], lr_v=proc_params.get("lr_v", 0.1), lambda_l2=proc_params.get("lambda_l2", 0.0), device=device)
        pred_c = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
        wrong_c_matches.append(check_match(pred_c, fact["object"]))
        for p in fact["paraphrases"]:
            pred_p = greedy_predict(model, tokenizer, p, 5, device, False)
            wrong_p_matches.append(check_match(pred_p, fact["object"]))

    m_wrong = Measurement.from_outcomes(wrong_c_matches, metric="wrong_target", arm="wrong_target", scope="s0_10_single_edit", input_set="s0_10_facts_100", mode="eval_no_dropout")
    m_wrong_para = Measurement.from_outcomes(wrong_p_matches, metric="wrong_target_paraphrase", arm="wrong_target_paraphrase", scope="s0_10_wrong_target_paraphrase", input_set="s0_10_paraphrases_300", mode="eval_no_dropout")
    return {
        "wrong_target": {
            "num": m_wrong.numerator, "den": m_wrong.denominator, "rate": m_wrong.rate,
            "w_lo": m_wrong.wilson_low, "w_hi": m_wrong.wilson_high, "raw_vectors": wrong_c_matches
        },
        "wrong_target_paraphrase": {
            "num": m_wrong_para.numerator, "den": m_wrong_para.denominator, "rate": m_wrong_para.rate,
            "w_lo": m_wrong_para.wilson_low, "w_hi": m_wrong_para.wilson_high, "raw_vectors": wrong_p_matches
        }
    }


def execute_stage_s_seed(
    seed: int, facts_200: List[Dict[str, Any]], model: nn.Module, tokenizer: Any,
    fresh_model: nn.Module, base_state_dict: Dict[str, torch.Tensor],
    control_probes: List[Dict[str, Any]], wikitext_slice: torch.Tensor, slice_sha: str,
    selected_proc_type: str, layer_idx: int, proc_params: Dict[str, Any], device: str = "cuda"
) -> Dict[str, Any]:
    model.load_state_dict(base_state_dict)
    initial_readout = freeze_readout(model)
    imm_matches, steps_list = [], []

    for fact in facts_200:
        if selected_proc_type == "sgd":
            e_res = edit_fact_mlp_fullgrad_sgd(model, tokenizer, fact, layer_idx=layer_idx, lr=proc_params["lr"], max_steps=proc_params["max_steps"], device=device)
        else:
            e_res = edit_fact_mlp_closed_form_repaired(model, tokenizer, fact, layer_idx=layer_idx, max_steps=proc_params["max_steps"], lr_v=proc_params.get("lr_v", 0.1), lambda_l2=proc_params.get("lambda_l2", 0.0), device=device)
        imm_matches.append(e_res["immediate_match"])
        steps_list.append(e_res["steps_taken"])

    assert_readout_frozen(model, initial_readout, seed, "S-site")
    preds = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_200]
    term_matches = [check_match(p, f["object"]) for p, f in zip(preds, facts_200)]

    para_matches = []
    for f in facts_200:
        for p in f["paraphrases"]:
            para_matches.append(check_match(greedy_predict(model, tokenizer, p, 5, device, False), f["object"]))

    full_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)
    probe_prompts = [c["prompt"] for c in control_probes]
    pre_lps = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for p in probe_prompts}
    post_lps = {p: get_next_token_log_probs(model, tokenizer, p, device, False) for p in probe_prompts}
    loc_kl = compute_locality_kl(pre_lps, post_lps)

    return {
        "seed": seed, "immediate_matches": imm_matches, "steps_taken": steps_list,
        "terminal_matches": term_matches, "paraphrase_matches": para_matches,
        "first50_terminal_matches": term_matches[:50], "first50_paraphrase_matches": para_matches[:150],
        "perplexity": full_ppl, "locality_kl": loc_kl
    }
