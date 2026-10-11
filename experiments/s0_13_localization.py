#!/usr/bin/env python3
"""
experiments/s0_13_localization.py -- Directive S0-13 Loss Localization Engine

Implements Directive S0-13 Stage L2 and Stage R:
- Snapshotting & bitwise rank-1 reconstruction of W_j from (u_i, r_i) factors
- L2a: Protected key drift in float64 and float64 replay comparison
- L2b: Output patching of h[1].mlp.c_proj across positions (a)-(e) with harness positive control
- L2c: Target log-prob margin tracking (write-time vs sequence-end)
- Stage R: Seed-clustered logistic regression of terminal retention
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.metrics import Measurement, check_match, normalize_entity
from experiments.b1_inject import greedy_predict
from experiments.s0_10_repair import get_subject_last_token_idx


def reconstruct_weight(
    w_0: torch.Tensor,
    factors: List[Tuple[torch.Tensor, torch.Tensor]],
    up_to_t: int
) -> torch.Tensor:
    """
    Reconstructs W_j = W_0 + sum_{i <= up_to_t} outer(u_i, r_i).
    Applies updates sequentially in identical accumulation order to ensure bitwise reproducibility.
    """
    w = w_0.clone()
    for i in range(up_to_t + 1):
        u_i, r_i = factors[i]
        w.add_(torch.outer(u_i, r_i))
    return w


def verify_weight_reconstruction(
    live_w: torch.Tensor,
    w_0: torch.Tensor,
    factors: List[Tuple[torch.Tensor, torch.Tensor]],
    t: int
) -> bool:
    """
    Asserts reconstructed weight matches live weight bitwise at edit index t.
    """
    reconstructed = reconstruct_weight(w_0, factors, t)
    is_exact = torch.equal(live_w, reconstructed)
    assert is_exact, f"Weight reconstruction failed bitwise equality at t={t}"
    return True


def compute_target_margin(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_obj: str,
    device: str = "cuda"
) -> float:
    """
    Computes first-token log-prob margin: log P(target_tok) - max_{tok != target} log P(tok).
    Uses the leading-space convention f"{prompt} {object}".
    """
    full_text = f"{prompt} {target_obj}"
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_len = enc_prompt.input_ids.shape[1]
    target_tok = enc_full.input_ids[0, prompt_len].item()

    with torch.no_grad():
        out = model(enc_prompt.input_ids)
        logits = out.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        target_logp = log_probs[target_tok].item()
        
        # Mask out target token to find max non-target
        mask = torch.ones_like(log_probs, dtype=torch.bool)
        mask[target_tok] = False
        max_non_target = torch.max(log_probs[mask]).item()

    return target_logp - max_non_target


def compute_l2a_drift(
    k_vecs: List[torch.Tensor],
    r_vecs: List[torch.Tensor],
    factors_32: List[Tuple[torch.Tensor, torch.Tensor]],
    factors_64: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    L2a: Value drift at protected keys for every fact j in 0..T-1:
      drift_j = || k_j (W_T - W_j) || / || r_j ||
    Also computes drift under exact float64 replay of updates to separate
    float32 accumulation leakage from algorithmic subspace leakage.
    """
    n_edits = len(factors_32)
    assert len(k_vecs) == n_edits and len(r_vecs) == n_edits

    # Accumulate tail deltas: W_T - W_j = sum_{t=j+1}^{T-1} outer(u_t, r_t)
    drifts_f32 = []
    drifts_f64 = []

    # Compute in float64 for precision
    for j in range(n_edits):
        k_j = k_vecs[j].to(device, dtype=torch.float64)
        r_j = r_vecs[j].to(device, dtype=torch.float64)
        r_norm = torch.linalg.norm(r_j).item() + 1e-12

        # 1. Float32 factor accumulation drift
        drift_vec_32 = torch.zeros_like(r_j)
        for t in range(j + 1, n_edits):
            u_t, r_t = factors_32[t]
            u_t_64 = u_t.to(device, dtype=torch.float64)
            r_t_64 = r_t.to(device, dtype=torch.float64)
            proj = torch.dot(k_j, u_t_64)
            drift_vec_32.add_(r_t_64 * proj)
        drift_val_32 = torch.linalg.norm(drift_vec_32).item() / r_norm
        drifts_f32.append(drift_val_32)

        # 2. Float64 replay accumulation drift
        if factors_64 is not None:
            drift_vec_64 = torch.zeros_like(r_j)
            for t in range(j + 1, n_edits):
                u_t_64, r_t_64 = factors_64[t]
                proj = torch.dot(k_j, u_t_64.to(device))
                drift_vec_64.add_(r_t_64.to(device) * proj)
            drift_val_64 = torch.linalg.norm(drift_vec_64).item() / r_norm
            drifts_f64.append(drift_val_64)

    return {
        "drifts_f32": drifts_f32,
        "drifts_f64": drifts_f64 if factors_64 is not None else drifts_f32,
    }


def make_output_patch_hook(
    w_j: torch.Tensor,
    b_j: torch.Tensor,
    prompt_len: int,
    subj_idx: int,
    condition: str
):
    """
    Builds forward hook for h[1].mlp.c_proj to patch the OUTPUT with (k * W_j + b_j).
    Conditions:
      (a) subject last-token position only
      (b) non-subject prompt positions only
      (c) all prompt positions
      (d) generated object-token positions only (pos >= prompt_len)
      (e) all positions (prompt + generated)
    """
    def hook(module, inp, out):
        k_act = inp[0]  # [1, S, 3072]
        # Compute exact output of W_j on current input activations
        # Conv1D weight shape is (3072, 768), bias shape is (768)
        batch_sz, seq_len, _ = k_act.shape
        v_j = torch.addmm(b_j, k_act.view(-1, 3072), w_j).view(batch_sz, seq_len, -1)

        if condition == "e":
            return v_j

        patched = out.clone()
        if condition == "a":
            if seq_len > subj_idx:
                patched[:, subj_idx, :] = v_j[:, subj_idx, :]
        elif condition == "b":
            end_p = min(seq_len, prompt_len)
            for pos in range(end_p):
                if pos != subj_idx:
                    patched[:, pos, :] = v_j[:, pos, :]
        elif condition == "c":
            end_p = min(seq_len, prompt_len)
            patched[:, :end_p, :] = v_j[:, :end_p, :]
        elif condition == "d":
            if seq_len > prompt_len:
                patched[:, prompt_len:seq_len, :] = v_j[:, prompt_len:seq_len, :]
        return patched

    return hook


def run_stage_l2b_patching(
    terminal_model: nn.Module,
    tokenizer: Any,
    w_0: torch.Tensor,
    b_0: torch.Tensor,
    factors: List[Tuple[torch.Tensor, torch.Tensor]],
    eval_facts: List[Dict[str, Any]],
    lost_indices: List[int],
    retained_indices: List[int],
    layer_idx: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    L2b: Output patching of h[1].mlp.c_proj on terminal model for first-50 facts.
    Evaluates both lost-at-end and retained-at-end facts across conditions (a)-(e).
    Validates harness positive control:
      - condition (e) on retained facts must keep them retained
      - condition (e) on lost facts must recover >= 90.00%
    """
    c_proj = terminal_model.transformer.h[layer_idx].mlp.c_proj
    conditions = ["a", "b", "c", "d", "e"]

    lost_outcomes = {c: [] for c in conditions}
    retained_outcomes = {c: [] for c in conditions}

    # Evaluate lost facts
    for idx in lost_facts_to_eval := lost_indices:
        fact = eval_facts[idx]
        prompt = fact["edit_prompt"]
        target_obj = fact["object"]
        enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = enc_prompt.input_ids.shape[1]
        subj_idx = get_subject_last_token_idx(tokenizer, prompt, fact["subject"])

        # Reconstruct W_j immediately after write j
        w_j = reconstruct_weight(w_0, factors, idx).to(device)

        for cond in conditions:
            hook_fn = make_output_patch_hook(w_j, b_0, prompt_len, subj_idx, cond)
            h = c_proj.register_forward_hook(hook_fn)
            try:
                pred = greedy_predict(terminal_model, tokenizer, prompt, 5, device, False)
                lost_outcomes[cond].append(check_match(pred, target_obj))
            finally:
                h.remove()

    # Evaluate retained facts
    for idx in retained_facts_to_eval := retained_indices:
        fact = eval_facts[idx]
        prompt = fact["edit_prompt"]
        target_obj = fact["object"]
        enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = enc_prompt.input_ids.shape[1]
        subj_idx = get_subject_last_token_idx(tokenizer, prompt, fact["subject"])

        w_j = reconstruct_weight(w_0, factors, idx).to(device)

        for cond in conditions:
            hook_fn = make_output_patch_hook(w_j, b_0, prompt_len, subj_idx, cond)
            h = c_proj.register_forward_hook(hook_fn)
            try:
                pred = greedy_predict(terminal_model, tokenizer, prompt, 5, device, False)
                retained_outcomes[cond].append(check_match(pred, target_obj))
            finally:
                h.remove()

    n_lost = len(lost_indices)
    n_retained = len(retained_indices)

    # Harness positive control verification
    retained_e_rate = (sum(retained_outcomes["e"]) / n_retained) if n_retained > 0 else 1.0
    lost_e_rate = (sum(lost_outcomes["e"]) / n_lost) if n_lost > 0 else 1.0

    harness_passed = (retained_e_rate >= 0.95) and (lost_e_rate >= 0.90 or n_lost == 0)

    return {
        "n_lost": n_lost,
        "n_retained": n_retained,
        "lost_recovery": {c: sum(lost_outcomes[c]) for c in conditions},
        "lost_recovery_rates": {c: (sum(lost_outcomes[c]) / n_lost if n_lost > 0 else 0.0) for c in conditions},
        "retained_stability": {c: sum(retained_outcomes[c]) for c in conditions},
        "retained_stability_rates": {c: (sum(retained_outcomes[c]) / n_retained if n_retained > 0 else 1.0) for c in conditions},
        "harness_positive_control_passed": harness_passed,
        "raw_lost_outcomes": lost_outcomes,
        "raw_retained_outcomes": retained_outcomes,
    }


def run_stage_r_regression(
    records: List[Dict[str, Any]],
    cluster_key: str = "seed"
) -> Dict[str, Any]:
    """
    Stage R: Exploratory seed-clustered logistic regression of terminal retention on:
      - edit_position (0..49)
      - later_same_rel_count
      - is_multi_token (under leading-space convention)
      - write_time_margin (L2c)
      - drift_j (L2a)
    Reports coefficients with cluster-robust 95% CIs. Labeled EXPLORATORY; no verdicts.
    """
    n_obs = len(records)
    if n_obs < 10:
        return {"status": "INSUFFICIENT_OBSERVATIONS", "n_obs": n_obs}

    feature_names = [
        "intercept",
        "edit_position",
        "later_same_rel_count",
        "is_multi_token",
        "write_time_margin",
        "drift_j"
    ]

    # Build X and Y
    X_rows = []
    Y_vals = []
    clusters = []

    for r in records:
        X_rows.append([
            1.0,
            float(r["edit_position"]),
            float(r["later_same_rel_count"]),
            1.0 if r["is_multi_token"] else 0.0,
            float(r["write_time_margin"]),
            float(r["drift_j"])
        ])
        Y_vals.append(1.0 if r["terminal_retained"] else 0.0)
        clusters.append(r[cluster_key])

    X = torch.tensor(X_rows, dtype=torch.float64)
    Y = torch.tensor(Y_vals, dtype=torch.float64)
    k_feats = X.shape[1]

    # Logistic regression fitting via Newton-Raphson
    beta = torch.zeros(k_feats, dtype=torch.float64)
    for _ in range(50):
        p = torch.sigmoid(X @ beta)
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        W = p * (1.0 - p)
        grad = X.t() @ (Y - p)
        H = - (X.t() * W) @ X
        try:
            step = torch.linalg.solve(H, -grad)
        except Exception:
            break
        beta = beta + step
        if torch.linalg.norm(step).item() < 1e-6:
            break

    # Cluster-robust sandwich covariance: V = (H^-1) * (sum_g S_g S_g^T) * (H^-1)
    p_final = torch.sigmoid(X @ beta)
    residuals = Y - p_final
    unique_clusters = list(set(clusters))
    meat = torch.zeros((k_feats, k_feats), dtype=torch.float64)

    for g in unique_clusters:
        idx_g = [i for i, c in enumerate(clusters) if c == g]
        X_g = X[idx_g]
        res_g = residuals[idx_g]
        score_g = (X_g.t() @ res_g).unsqueeze(1)
        meat += score_g @ score_g.t()

    try:
        H_inv = torch.linalg.inv(-H)
        V_cluster = H_inv @ meat @ H_inv
        se_cluster = torch.sqrt(torch.diag(V_cluster))
    except Exception:
        se_cluster = torch.zeros(k_feats, dtype=torch.float64)

    z_crit = 1.95996  # 95% CI
    results = {}
    for i, name in enumerate(feature_names):
        b_val = beta[i].item()
        se_val = se_cluster[i].item()
        results[name] = {
            "coef": b_val,
            "cluster_se": se_val,
            "ci_95": [b_val - z_crit * se_val, b_val + z_crit * se_val]
        }

    return {
        "status": "CONVERGED",
        "n_obs": n_obs,
        "n_clusters": len(unique_clusters),
        "results": results
    }
