#!/usr/bin/env python3
"""
experiments/s0_13_protection.py -- Directive S0-13 Protection Arms Engine

Implements Directive S0-13 Stage F:
- FullPromptSVDNullTracker: SVD-truncated orthonormal basis tracking in null space (rel threshold 1e-3)
  with capacity logging and exhaustion handling for F1 and F2
- extract_prompt_keys: extracts non-subject prompt keys
- extract_teacher_forced_object_keys: extracts teacher-forced object token keys
- find_target_value_vstar_margin: margin-targeted v* optimization (margin >= 2.0 nats) for F3
- edit_fact_f1, edit_fact_f2, edit_fact_f3: sequential edit execution routines
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.metrics import check_match
from experiments.b1_inject import greedy_predict
from experiments.s0_10_repair import get_subject_last_token_idx
from experiments.s0_11_constraints import find_target_value_vstar


class FullPromptSVDNullTracker:
    """
    SVD-truncated orthonormal null-space projector tracker for Directive S0-13.
    Maintains Q in null coordinates spanning protected directions.
    Relative cutoff threshold = 1e-3 against largest singular value.
    Logs protected rank per edit and the edit index of capacity exhaustion.
    """
    def __init__(
        self,
        v_null: torch.Tensor,
        rel_threshold: float = 1e-3,
        device: str = "cuda"
    ):
        self.v_null = v_null.to(device, dtype=torch.float64)
        self.d_feat, self.d_null = self.v_null.shape
        self.rel_threshold = rel_threshold
        self.device = device

        self.q_basis: Optional[torch.Tensor] = None  # shape (d_null, r)
        self.stored_keys: List[torch.Tensor] = []
        self.capacity_exhausted: bool = False
        self.exhaustion_edit_idx: Optional[int] = None
        self.rank_history: List[int] = []
        self.max_observed_residual: float = 0.0

    def reset(self):
        self.q_basis = None
        self.stored_keys = []
        self.capacity_exhausted = False
        self.exhaustion_edit_idx = None
        self.rank_history = []
        self.max_observed_residual = 0.0

    @property
    def current_rank(self) -> int:
        return self.q_basis.shape[1] if self.q_basis is not None else 0

    def add_keys(self, keys_tensor: torch.Tensor, edit_idx: int) -> int:
        """
        Projects keys into null space and appends orthogonalized SVD-truncated basis vectors to Q.
        keys_tensor shape: (N, d_feat)
        """
        if keys_tensor.shape[0] == 0:
            self.rank_history.append(self.current_rank)
            return self.current_rank

        keys_64 = keys_tensor.to(self.device, dtype=torch.float64)
        for k in keys_64:
            self.stored_keys.append(k.detach())

        # Coordinates in null space: (d_null, N)
        c_mat = torch.matmul(self.v_null.t(), keys_64.t())

        # Project out current Q (double Gram-Schmidt)
        if self.q_basis is not None and self.q_basis.shape[1] > 0:
            c_orth = c_mat - self.q_basis @ (self.q_basis.t() @ c_mat)
            c_orth = c_orth - self.q_basis @ (self.q_basis.t() @ c_orth)
        else:
            c_orth = c_mat

        # SVD on orthogonal components
        u_svd, s_svd, _ = torch.linalg.svd(c_orth, full_matrices=False)
        if s_svd.numel() > 0 and s_svd[0] > 1e-12:
            keep_mask = (s_svd / s_svd[0]) >= self.rel_threshold
            u_keep = u_svd[:, keep_mask]
        else:
            u_keep = torch.empty((self.d_null, 0), dtype=torch.float64, device=self.device)

        if u_keep.shape[1] > 0:
            if self.q_basis is None or self.q_basis.shape[1] == 0:
                self.q_basis = u_keep
            else:
                self.q_basis = torch.cat([self.q_basis, u_keep], dim=1)

            # Re-orthogonalize combined Q
            self.q_basis, _ = torch.linalg.qr(self.q_basis)

        # Capacity check
        curr_r = self.current_rank
        if curr_r >= self.d_null and not self.capacity_exhausted:
            self.capacity_exhausted = True
            self.exhaustion_edit_idx = edit_idx

        self.rank_history.append(curr_r)
        return curr_r

    def compute_update(
        self,
        k_vec: torch.Tensor,
        r_vec: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes null-space update orthogonal to all protected keys in Q:
          p_k = V_null (I - Q Q^T) V_null^T k
          Delta = (p_k r^T) / (k^T p_k)
        """
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        r_64 = r_vec.to(self.device, dtype=torch.float64)

        c = torch.matmul(self.v_null.t(), k_64)
        if self.q_basis is not None and self.q_basis.shape[1] > 0:
            c_free = c - self.q_basis @ (self.q_basis.t() @ c)
            c_free = c_free - self.q_basis @ (self.q_basis.t() @ c_free)
        else:
            c_free = c

        p_k = torch.matmul(self.v_null, c_free)
        k_p_k = torch.dot(k_64, p_k).item()

        if k_p_k > 1e-12:
            u_64 = p_k / k_p_k
            delta_64 = torch.outer(p_k, r_64) / k_p_k
        else:
            # Fallback if subspace saturated
            p_cand = p_k if torch.linalg.norm(p_k) > 1e-12 else k_64
            denom = torch.dot(k_64, p_cand).item() + 1e-12
            u_64 = p_cand / denom
            delta_64 = torch.outer(p_cand, r_64) / denom

        # Check relative residual against protected basis directions V_null @ Q:
        delta_norm_2 = torch.linalg.norm(delta_64, ord=2).item() + 1e-12
        r_norm = torch.linalg.norm(r_64).item()
        max_res = 0.0

        if self.q_basis is not None and self.q_basis.shape[1] > 0 and k_p_k > 1e-12:
            v_q = torch.matmul(self.v_null, self.q_basis)
            proj_vals = torch.abs(torch.matmul(v_q.t(), p_k))
            err_norms = (proj_vals * r_norm) / (k_p_k + 1e-12)
            rel_residuals = err_norms / delta_norm_2
            max_res = float(torch.max(rel_residuals).item()) if rel_residuals.numel() > 0 else 0.0

        if max_res > self.max_observed_residual:
            self.max_observed_residual = max_res

        assert max_res <= 1e-8, f"Full-prompt null-space residual gate failure: {max_res:.2e} > 1.00e-08"

        u_f32 = u_64.to(torch.float32)
        r_f32 = r_64.to(torch.float32)
        delta_f32 = torch.outer(u_f32, r_f32)

        return {
            "delta_f32": delta_f32,
            "u_f32": u_f32,
            "r_f32": r_f32,
            "rank": self.current_rank,
            "relative_residual": max_res,
            "capacity_exhausted": self.capacity_exhausted
        }


def extract_prompt_keys(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    subj_idx: int,
    layer_idx: int = 1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extracts activations at h[layer_idx].mlp.c_proj input for all non-subject prompt positions.
    Returns tensor of shape (n_non_subj, 3072).
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = enc_prompt.input_ids.shape[1]

    recorded = {}
    def hook_fn(m, inp, out):
        recorded["act"] = inp[0][0].detach().clone()  # [seq_len, 3072]

    h = c_proj.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(**enc_prompt)
    finally:
        h.remove()

    act = recorded["act"]
    non_subj_indices = [pos for pos in range(seq_len) if pos != subj_idx]
    if not non_subj_indices:
        return torch.empty((0, act.shape[1]), device=device, dtype=act.dtype)
    return act[non_subj_indices, :]


def extract_teacher_forced_object_keys(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_obj: str,
    layer_idx: int = 1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extracts activations at h[layer_idx].mlp.c_proj input for teacher-forced object positions.
    Feeds f"{prompt} {object}" and takes positions pos >= prompt_len.
    Returns tensor of shape (n_obj_toks, 3072).
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    full_text = f"{prompt} {target_obj}"
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_len = enc_prompt.input_ids.shape[1]
    full_len = enc_full.input_ids.shape[1]

    if full_len <= prompt_len:
        return torch.empty((0, 3072), device=device, dtype=torch.float32)

    recorded = {}
    def hook_fn(m, inp, out):
        recorded["act"] = inp[0][0].detach().clone()

    h = c_proj.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(**enc_full)
    finally:
        h.remove()

    act = recorded["act"]
    return act[prompt_len:full_len, :]


def find_target_value_vstar_margin(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    layer_idx: int = 1,
    target_margin: float = 2.0,
    max_steps: int = 100,
    lr_v: float = 0.1,
    lambda_l2: float = 0.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, int, str, float]:
    """
    Arm F3: Margin-targeted v* optimization.
    Optimizes v* until target first-token log-prob margin >= target_margin (2.0 nats)
    or max_steps is reached. Same lambda as reference (0.0).
    Returns (k_vec, v_star, steps_taken, pred_post_opt, final_margin).
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    for p in model.parameters():
        p.requires_grad = False
    subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])

    recorded = {}
    def hook_rec(module, inp, out):
        recorded["k"] = inp[0][0, subj_idx, :].detach().clone()
        recorded["v0"] = out[0, subj_idx, :].detach().clone()

    h_rec = c_proj.register_forward_hook(hook_rec)
    try:
        enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**enc_prompt)
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
    final_margin = -999.0
    try:
        with torch.set_grad_enabled(True):
            for _ in range(max_steps):
                steps_taken += 1
                opt_v.zero_grad()
                out = model(input_ids, labels=labels)
                loss_l2 = lambda_l2 * torch.sum((v_param - v0_vec) ** 2) if lambda_l2 > 0.0 else 0.0
                loss = out.loss + loss_l2
                loss.backward()
                opt_v.step()

                # Evaluate margin on prompt logits
                logits = out.logits[0, prompt_len - 1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                t_logp = log_probs[primary_tok].item()
                mask = torch.ones_like(log_probs, dtype=torch.bool)
                mask[primary_tok] = False
                nt_logp = torch.max(log_probs[mask]).item()
                curr_margin = t_logp - nt_logp
                final_margin = curr_margin

                if curr_margin >= target_margin:
                    break
    finally:
        h_rep.remove()

    v_star = v_param.detach()
    pred_str = tokenizer.decode([primary_tok])
    del v_param, opt_v, input_ids, labels
    return k_vec, v_star, steps_taken, pred_str, final_margin


def edit_fact_f1(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    tracker: FullPromptSVDNullTracker,
    edit_idx: int,
    layer_idx: int = 1,
    max_steps: int = 100,
    lr_v: float = 0.1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    F1: Full-prompt protection edit.
    Finds v* at subject position, computes update orthogonal to all prior prompt keys,
    applies update, then records all prompt keys from the current edit into tracker.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _ = find_target_value_vstar(model, tokenizer, fact, layer_idx, max_steps, lr_v, 0.0, device)

    # v0 = k @ W + b
    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    r_vec = v_star - v0

    update_res = tracker.compute_update(k_vec, r_vec)
    c_proj.weight.data.add_(update_res["delta_f32"])

    # Extract all prompt keys (subject + non-subject) to protect for future edits
    subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
    non_subj_keys = extract_prompt_keys(model, tokenizer, fact["edit_prompt"], subj_idx, layer_idx, device)
    all_prompt_keys = torch.cat([k_vec.unsqueeze(0), non_subj_keys], dim=0)
    curr_rank = tracker.add_keys(all_prompt_keys, edit_idx)

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])

    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "u_f32": update_res["u_f32"],
        "r_f32": update_res["r_f32"],
        "rank": curr_rank,
        "capacity_exhausted": update_res["capacity_exhausted"],
        "pred": curr_pred
    }


def edit_fact_f2(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    tracker: FullPromptSVDNullTracker,
    edit_idx: int,
    layer_idx: int = 1,
    max_steps: int = 100,
    lr_v: float = 0.1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    F2: F1 + teacher-forced object keys.
    Protects all prompt keys PLUS keys at teacher-forced object positions.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _ = find_target_value_vstar(model, tokenizer, fact, layer_idx, max_steps, lr_v, 0.0, device)

    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    r_vec = v_star - v0

    update_res = tracker.compute_update(k_vec, r_vec)
    c_proj.weight.data.add_(update_res["delta_f32"])

    subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
    non_subj_keys = extract_prompt_keys(model, tokenizer, fact["edit_prompt"], subj_idx, layer_idx, device)
    obj_keys = extract_teacher_forced_object_keys(model, tokenizer, fact["edit_prompt"], fact["object"], layer_idx, device)

    keys_to_add = [k_vec.unsqueeze(0)]
    if non_subj_keys.shape[0] > 0:
        keys_to_add.append(non_subj_keys)
    if obj_keys.shape[0] > 0:
        keys_to_add.append(obj_keys)
    all_keys = torch.cat(keys_to_add, dim=0)

    curr_rank = tracker.add_keys(all_keys, edit_idx)
    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])

    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "u_f32": update_res["u_f32"],
        "r_f32": update_res["r_f32"],
        "rank": curr_rank,
        "capacity_exhausted": update_res["capacity_exhausted"],
        "pred": curr_pred
    }


def edit_fact_f3(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    tracker: Any,  # CorrectedSequentialNullTracker
    edit_idx: int,
    layer_idx: int = 1,
    target_margin: float = 2.0,
    max_steps: int = 100,
    lr_v: float = 0.1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    F3: Margin-targeted v* write with CorrectedSequentialNullTracker.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _, final_margin = find_target_value_vstar_margin(
        model, tokenizer, fact, layer_idx, target_margin, max_steps, lr_v, 0.0, device
    )

    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    r_vec = v_star - v0

    update_res = tracker.compute_update(k_vec, r_vec)
    c_proj.weight.data.add_(update_res["delta_f32"])

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])

    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "final_margin": final_margin,
        "delta_f32": update_res["delta_f32"],
        "pred": curr_pred
    }
