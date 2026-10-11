#!/usr/bin/env python3
"""
experiments/s0_11_constraints.py -- Stage C Key Statistics & Constrained Sequential Writes
Implements Directive S0-11:
- Disjoint WikiText-2 key sample collection
- Uncentered key covariance C = E[k k^T] (3072 x 3072), eigenvalue spectrum, and condition number
- Preserved-key null space projector P_0
- Arm A-cov: ROME-style covariance-weighted update with ridge regularization
- Arm A-null: AlphaEdit-style null-space projector P_0 + incremental previous-key orthogonal projection
- Exact tolerance assertions: max_{j < t} ||k_j Delta|| < 1e-4
"""

import math
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from experiments.metrics import Measurement, check_match, normalize_entity
from experiments.b1_inject import greedy_predict
from experiments.data import evaluate_wikitext_perplexity
from experiments.s0_10_repair import get_subject_last_token_idx

N_PPL_SUBSET_SEQS = 100


def load_wikitext2_key_sample(tokenizer: Any, num_sequences: int = 100, seq_len: int = 512) -> Tuple[torch.Tensor, str]:
    """
    Loads disjoint WikiText-2 key-sample slice from the train split.
    Disjoint from both the 1,000-sequence capability slice and the 100-sequence PPL subset
    (which were drawn from validation + test splits).
    """
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n\n".join(list(dataset["train"]["text"]))
    tokens = tokenizer.encode(train_text)
    total_needed = num_sequences * seq_len
    assert len(tokens) >= total_needed, f"Insufficient tokens in train split: {len(tokens)} < {total_needed}"
    tensor_sample = torch.tensor(tokens[:total_needed], dtype=torch.long).view(num_sequences, seq_len)
    sample_sha = hashlib.sha256(tensor_sample.numpy().tobytes()).hexdigest()
    return tensor_sample, sample_sha


def compute_layer_key_covariance(
    model: nn.Module,
    key_sample_tensor: torch.Tensor,
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda"
) -> Tuple[torch.Tensor, int]:
    """
    Computes uncentered key covariance C = E[k k^T] (3072 x 3072) at c_proj input of layer L.
    Accumulates second moments across all tokens in key_sample_tensor.
    """
    model.eval()
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    accum_cov = torch.zeros((3072, 3072), dtype=torch.float64, device=device)
    total_tokens = 0

    num_seqs = key_sample_tensor.shape[0]
    for i in range(0, num_seqs, batch_size):
        batch = key_sample_tensor[i : i + batch_size].to(device)
        recorded = {}
        def hook_fn(m, inp, out):
            recorded["k"] = inp[0].detach()  # [batch, seq_len, 3072]
        handle = c_proj.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = model(batch)
        finally:
            handle.remove()

        k_tokens = recorded["k"].reshape(-1, 3072).to(torch.float64)
        n_tok = k_tokens.shape[0]
        accum_cov.addmm_(k_tokens.t(), k_tokens, beta=1.0, alpha=1.0)
        total_tokens += n_tok
        del batch, k_tokens, recorded

    assert total_tokens > 0, "No tokens processed for key covariance"
    cov = (accum_cov / float(total_tokens)).to(torch.float32).cpu()
    return cov, total_tokens


def compute_null_space_projector(
    cov: torch.Tensor,
    rel_threshold: float = 1e-3
) -> Dict[str, Any]:
    """
    Computes eigenvalue decomposition of C, condition number, and preserved-key projector P_0.
    Eigenvalues below rel_threshold * lambda_max are assigned to the null space.
    """
    cov_d = cov.to(torch.float64)
    evals, evecs = torch.linalg.eigh(cov_d)
    l_max = evals[-1].item()
    l_min = evals[0].item()
    cond_num = float("inf") if l_min <= 0 else (l_max / l_min)

    cutoff = rel_threshold * l_max
    null_mask = evals < cutoff
    null_dim = int(null_mask.sum().item())
    total_energy = float(evals.sum().item())
    null_energy = float(evals[null_mask].sum().item())
    retained_energy_frac = (null_energy / total_energy) if total_energy > 0 else 0.0

    if null_dim > 0:
        v_null = evecs[:, null_mask]  # [3072, null_dim] in float64
        p_0 = torch.matmul(v_null, v_null.t()).to(torch.float32)
    else:
        # Fallback to empty null space projector
        v_null = torch.zeros((cov.shape[0], 0), dtype=torch.float64)
        p_0 = torch.zeros((cov.shape[0], cov.shape[0]), dtype=torch.float32)

    cov_sha = hashlib.sha256(cov.numpy().tobytes()).hexdigest()
    p0_sha = hashlib.sha256(p_0.numpy().tobytes()).hexdigest()

    return {
        "cov": cov,
        "cov_sha256": cov_sha,
        "p_0": p_0,
        "v_null": v_null,
        "p0_sha256": p0_sha,
        "evals": evals.to(torch.float32).tolist(),
        "lambda_max": l_max,
        "lambda_min": l_min,
        "condition_number": cond_num,
        "rel_threshold": rel_threshold,
        "null_dim": null_dim,
        "retained_energy_fraction": retained_energy_frac
    }


def find_target_value_vstar(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    layer_idx: int,
    max_steps: int = 100,
    lr_v: float = 0.1,
    lambda_l2: float = 0.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
    """
    Finds v* at subject's last token position using S0-10 repaired configuration (lambda_l2 = 0).
    Reused from proven S0-10 commit 99d3335 (AGENTS.md §7.5).
    Returns (k_vec, v_star, steps_taken, pred_post_opt).
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
                if torch.argmax(out.logits[0, prompt_len - 1, :]).item() == primary_tok:
                    break
    finally:
        h_rep.remove()

    v_star = v_param.detach()
    pred_str = tokenizer.decode([primary_tok])
    del v_param, opt_v, input_ids, labels
    return k_vec, v_star, steps_taken, pred_str


def edit_fact_mlp_cov(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    cov: torch.Tensor,
    layer_idx: int,
    max_steps: int = 100,
    lr_v: float = 0.1,
    ridge_factor: float = 1e-3,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Arm A-cov: ROME-style covariance-weighted update at c_proj:
    Delta = u (v* - W k - b) where u = (C_reg^-1 k) / (k^T C_reg^-1 k).
    Satisfies k (W + Delta) + b = v* exactly while minimizing expected corpus disruption.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _ = find_target_value_vstar(model, tokenizer, fact, layer_idx, max_steps, lr_v, 0.0, device)

    # v0 = k W + b (using Conv1D addmm)
    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    delta_v = v_star - v0

    # Ridge regularization: C_reg = C + ridge * I
    d_feat = cov.shape[0]
    tr_cov = float(torch.trace(cov).item())
    ridge_val = ridge_factor * (tr_cov / float(d_feat))
    cov_reg = cov.to(device) + ridge_val * torch.eye(d_feat, device=device)

    # Solve C_reg z = k
    z = torch.linalg.solve(cov_reg, k_vec)
    denom = torch.dot(k_vec, z).item()
    assert abs(denom) > 1e-12, "Degenerate quadratic form in covariance update"
    u_vec = z / denom

    # Delta W = outer(u_vec, delta_v)
    delta_w = torch.outer(u_vec, delta_v)
    c_proj.weight.data.add_(delta_w)
    delta_norm = torch.linalg.norm(delta_w).item()

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])
    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "delta_norm": delta_norm,
        "pred": curr_pred,
        "k_vec": k_vec.detach().cpu()
    }


class SequentialNullTracker:
    """
    [DEPRECATED per S0-11 Amendment 1 §A — Preserved for Historical Tests]
    Original as-executed A-null tracker.
    """
    def __init__(self, p_0: torch.Tensor, tolerance: float = 1e-4, device: str = "cuda"):
        self.p_0 = p_0.to(device)
        self.tolerance = tolerance
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.q_basis: Optional[torch.Tensor] = None
        self.max_observed_violation = 0.0

    def reset(self):
        self.stored_keys = []
        self.q_basis = None
        self.last_u_vec = None
        self.max_observed_violation = 0.0

    def get_other_keys(self, k_vec: torch.Tensor) -> List[torch.Tensor]:
        if not self.stored_keys:
            return []
        k_unit = k_vec / (torch.linalg.norm(k_vec) + 1e-12)
        other, other_units = [], []
        for prev_k in reversed(self.stored_keys):
            prev_unit = prev_k.to(self.device) / (torch.linalg.norm(prev_k) + 1e-12)
            if torch.dot(k_unit, prev_unit).item() < 0.999:
                if not any(torch.dot(prev_unit, u).item() >= 0.999 for u in other_units):
                    other.append(prev_k)
                    other_units.append(prev_unit)
        other.reverse()
        return other

    def get_orthogonal_basis_for_others(self, other_keys: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not other_keys:
            return None
        k_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in other_keys], dim=1)
        q_basis, _ = torch.linalg.qr(k_mat)
        return q_basis

    def project_orthogonal(self, v: torch.Tensor, q_basis: Optional[torch.Tensor]) -> torch.Tensor:
        if q_basis is not None and q_basis.shape[1] > 0:
            v_orig_dtype = v.dtype
            v_64 = v.to(torch.float64)
            p1 = torch.matmul(q_basis, torch.matmul(q_basis.t(), v_64))
            v_orth = v_64 - p1
            p2 = torch.matmul(q_basis, torch.matmul(q_basis.t(), v_orth))
            return (v_orth - p2).to(v_orig_dtype)
        return v

    def compute_update_direction(self, k_vec: torch.Tensor, cov: torch.Tensor, ridge_val: float) -> Tuple[torch.Tensor, float]:
        other_keys = self.get_other_keys(k_vec)
        q_basis = self.get_orthogonal_basis_for_others(other_keys)
        self.q_basis = q_basis
        d_feat = cov.shape[0]
        cov_reg = cov.to(self.device) + ridge_val * torch.eye(d_feat, device=self.device)
        z = torch.linalg.solve(cov_reg, k_vec)
        z_p0 = torch.matmul(self.p_0, z)
        u_cand = self.project_orthogonal(z_p0, q_basis)
        denom = torch.dot(k_vec, u_cand).item()
        if denom > 1e-4:
            u_vec = u_cand / denom
        else:
            u_arm_b = self.project_orthogonal(z, q_basis)
            denom_b = torch.dot(k_vec, u_arm_b).item()
            u_vec = u_arm_b / denom_b if denom_b > 1e-6 else k_vec / (torch.dot(k_vec, k_vec).item() + 1e-12)
        u_vec = self.project_orthogonal(u_vec, q_basis)
        denom_final = torch.dot(k_vec, u_vec).item()
        if abs(denom_final) > 1e-6:
            u_vec = u_vec / denom_final
        self.last_u_vec = u_vec
        k_proj_norm = torch.linalg.norm(self.project_orthogonal(k_vec, q_basis)).item()
        return u_vec, k_proj_norm

    def register_and_assert(self, k_vec: torch.Tensor, delta_w: torch.Tensor, delta_v: torch.Tensor, u_vec: Optional[torch.Tensor] = None) -> float:
        max_err = 0.0
        other_keys = self.get_other_keys(k_vec)
        if len(other_keys) > 0:
            active_u = u_vec if u_vec is not None else getattr(self, "last_u_vec", None)
            prev_stack = torch.stack([k.to(self.device, dtype=torch.float64) for k in other_keys], dim=0)
            if active_u is not None:
                u_64 = active_u.to(self.device, dtype=torch.float64)
                v_norm = torch.linalg.norm(delta_v.to(self.device, dtype=torch.float64)).item()
                err_norms = torch.abs(torch.matmul(prev_stack, u_64)) * v_norm
            else:
                err_vecs = torch.matmul(prev_stack, delta_w.to(self.device, dtype=torch.float64))
                err_norms = torch.linalg.norm(err_vecs, dim=1)
            max_err = torch.max(err_norms).item()
            if max_err > self.max_observed_violation:
                self.max_observed_violation = max_err
            assert max_err <= self.tolerance, f"Null-space assertion failure: max ||k_prev Delta W|| = {max_err:.2e} > {self.tolerance:.2e}"
        self.stored_keys.append(k_vec.detach().cpu())
        return max_err


class CorrectedSequentialNullTracker:
    r"""
    Directive S0-11 Amendment 1 Stage N2:
    Corrected Sequential Null-Space Projector Tracker.
    Maintains P = P_0 \cap (span(k_0, ..., k_{t-1}))^\perp in float64.
    Update rule: Delta = (P k) r^T / (k^T P k), where r = v* - W k - b.
    Pre-registered check: max relative residual <= 1e-8 in float64 before casting.
    """
    def __init__(self, v_null: torch.Tensor, rel_threshold: float = 1e-3, device: str = "cuda"):
        self.v_null = v_null.to(device, dtype=torch.float64)
        self.rel_threshold = rel_threshold
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.q_k: Optional[torch.Tensor] = None
        self.rho_history: List[float] = []
        self.max_relative_residual = 0.0
        self.max_applied_residual = 0.0

    def reset(self):
        self.stored_keys = []
        self.q_k = None
        self.rho_history = []
        self.max_relative_residual = 0.0
        self.max_applied_residual = 0.0

    def get_other_keys(self, k_vec: torch.Tensor) -> List[torch.Tensor]:
        if not self.stored_keys:
            return []
        k_unit = k_vec / (torch.linalg.norm(k_vec) + 1e-12)
        other, other_units = [], []
        for prev_k in reversed(self.stored_keys):
            prev_unit = prev_k / (torch.linalg.norm(prev_k) + 1e-12)
            if torch.dot(k_unit, prev_unit).item() < 0.999:
                if not any(torch.dot(prev_unit, u).item() >= 0.999 for u in other_units):
                    other.append(prev_k)
                    other_units.append(prev_unit)
        other.reverse()
        return other

    def update_basis(self, other_keys: List[torch.Tensor]):
        if not other_keys or self.v_null.shape[1] == 0:
            self.q_k = None
            return
        k_mat = torch.stack(other_keys, dim=1)
        c_mat = torch.matmul(self.v_null.t(), k_mat)
        q_basis, r_mat = torch.linalg.qr(c_mat)
        diag_r = torch.abs(torch.diag(r_mat))
        valid_cols = diag_r > 1e-10 * (diag_r[0] if diag_r.numel() > 0 else 1.0)
        self.q_k = q_basis[:, valid_cols] if valid_cols.any() else None

    def project_k(self, k_64: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.v_null.shape[1] == 0:
            return torch.zeros_like(k_64), 0.0
        c = torch.matmul(self.v_null.t(), k_64)
        if self.q_k is not None and self.q_k.shape[1] > 0:
            p1 = torch.matmul(self.q_k, torch.matmul(self.q_k.t(), c))
            c_orth = c - p1
            p2 = torch.matmul(self.q_k, torch.matmul(self.q_k.t(), c_orth))
            c_orth = c_orth - p2
        else:
            c_orth = c
        p_k = torch.matmul(self.v_null, c_orth)
        denom = torch.dot(k_64, k_64).item() + 1e-12
        rho = torch.dot(c_orth, c_orth).item() / denom
        return p_k, rho

    def compute_update(self, k_vec: torch.Tensor, r_vec: torch.Tensor, preserved_sample_keys: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        r_64 = r_vec.to(self.device, dtype=torch.float64)
        other_keys = self.get_other_keys(k_64)
        self.update_basis(other_keys)
        p_k, rho = self.project_k(k_64)
        self.rho_history.append(rho)
        k_p_k = torch.dot(k_64, p_k).item()

        if k_p_k > 1e-12:
            delta_64 = torch.outer(p_k, r_64) / k_p_k
        else:
            p_cand = p_k if torch.linalg.norm(p_k) > 1e-12 else k_64
            denom = torch.dot(k_64, p_cand).item() + 1e-12
            delta_64 = torch.outer(p_cand, r_64) / denom

        delta_norm_2 = torch.linalg.norm(delta_64, ord=2).item() + 1e-12

        # 1. Pre-registered Relative Residual Check in float64 before casting
        max_rel_res = 0.0
        if len(other_keys) > 0:
            for prev_k in other_keys:
                k_norm = torch.linalg.norm(prev_k).item() + 1e-12
                proj_val = abs(torch.dot(prev_k, p_k).item())
                err_norm = (proj_val * torch.linalg.norm(r_64).item()) / (k_p_k + 1e-12)
                rel_res = err_norm / (k_norm * delta_norm_2)
                if rel_res > max_rel_res:
                    max_rel_res = rel_res

        if preserved_sample_keys:
            for kp in preserved_sample_keys:
                kp_64 = kp.to(self.device, dtype=torch.float64)
                kp_norm = torch.linalg.norm(kp_64).item() + 1e-12
                proj_val = abs(torch.dot(kp_64, p_k).item())
                err_norm = (proj_val * torch.linalg.norm(r_64).item()) / (k_p_k + 1e-12)
                rel_res = err_norm / (kp_norm * delta_norm_2)
                if rel_res > max_rel_res:
                    max_rel_res = rel_res

        if max_rel_res > self.max_relative_residual:
            self.max_relative_residual = max_rel_res

        assert max_rel_res <= 1e-8, f"Corrected A-null relative residual gate failure: {max_rel_res:.2e} > 1.00e-08"

        # 2. Applied residual after casting to float32 (descriptive only, no gate)
        delta_f32 = delta_64.to(torch.float32)
        applied_norm_2 = torch.linalg.norm(delta_f32, ord=2).item() + 1e-12
        max_applied_res = 0.0
        if len(other_keys) > 0:
            for prev_k in other_keys:
                pk_f32 = prev_k.to(torch.float32)
                pk_norm = torch.linalg.norm(pk_f32).item() + 1e-12
                out_err = torch.linalg.norm(torch.matmul(pk_f32, delta_f32)).item()
                app_res = out_err / (pk_norm * applied_norm_2)
                if app_res > max_applied_res:
                    max_applied_res = app_res
        if max_applied_res > self.max_applied_residual:
            self.max_applied_residual = max_applied_res

        self.stored_keys.append(k_64.detach())
        return {
            "delta_f32": delta_f32,
            "rho": rho,
            "relative_residual": max_rel_res,
            "applied_residual": max_applied_res,
            "p_k": p_k.detach().cpu()
        }


def edit_fact_mlp_null_corrected(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    null_tracker: CorrectedSequentialNullTracker,
    layer_idx: int,
    preserved_sample_keys: Optional[List[torch.Tensor]] = None,
    max_steps: int = 100,
    lr_v: float = 0.1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Directive S0-11 Amendment 1 Stage N2: Corrected A-null update.
    Delta = (P k) r^T / (k^T P k), where r = v* - W k - b.
    Maintains P in float64. Casts Delta to float32 only when writing weight.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _ = find_target_value_vstar(model, tokenizer, fact, layer_idx, max_steps, lr_v, 0.0, device)
    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    r_vec = v_star - v0

    res = null_tracker.compute_update(k_vec, r_vec, preserved_sample_keys)
    delta_w = res["delta_f32"]
    c_proj.weight.data.add_(delta_w)
    delta_norm = torch.linalg.norm(delta_w).item()

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])
    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "delta_norm": delta_norm,
        "pred": curr_pred,
        "rho": res["rho"],
        "relative_residual": res["relative_residual"],
        "applied_residual": res["applied_residual"],
        "k_vec": k_vec.detach().cpu(),
        "r_vec": r_vec.detach().cpu(),
        "v_star": v_star.detach().cpu(),
        "v0": v0.detach().cpu(),
        "p_k": res["p_k"]
    }


def evaluate_procedure_matched_controls(
    base_state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    tokenizer: Any,
    eval_facts: List[Dict[str, Any]],
    facts_pool: List[Dict[str, Any]],
    edit_proc_fn: Any,
    proc_kwargs: Dict[str, Any],
    device: str = "cuda",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates procedure-matched negative controls (wrong_target on canonical and paraphrase).
    Runs the exact arm update procedure with mismatched target objects.
    """
    import random
    rng = random.Random(seed)
    wrong_c_matches, wrong_p_matches = [], []

    for fact in eval_facts:
        cands = [c["object"] for c in facts_pool if c["relation"] == fact["relation"] and normalize_entity(c["object"]) != normalize_entity(fact["object"])]
        wrong_fact = {**fact, "object": rng.choice(cands)}
        model.load_state_dict(base_state_dict)
        if "null_tracker" in proc_kwargs and hasattr(proc_kwargs["null_tracker"], "reset"):
            proc_kwargs["null_tracker"].reset()

        _ = edit_proc_fn(model, tokenizer, wrong_fact, **proc_kwargs)

        pred_c = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
        wrong_c_matches.append(check_match(pred_c, fact["object"]))
        for p in fact["paraphrases"]:
            pred_p = greedy_predict(model, tokenizer, p, 5, device, False)
            wrong_p_matches.append(check_match(pred_p, fact["object"]))

    c_scope = "s0_11_matched_canonical_pooled" if len(wrong_c_matches) == 300 else "s0_11_matched_canonical"
    c_input_set = "facts_first50_pooled" if len(wrong_c_matches) == 300 else "facts_first50"
    p_scope = "s0_11_matched_paraphrase_pooled" if len(wrong_p_matches) == 900 else "s0_11_matched_paraphrase"
    p_input_set = "paraphrases_900_pooled" if len(wrong_p_matches) == 900 else "paraphrases_150"

    m_c = Measurement.from_outcomes(wrong_c_matches, metric="wrong_target", arm="matched_control", scope=c_scope, input_set=c_input_set, mode="eval_no_dropout")
    m_p = Measurement.from_outcomes(wrong_p_matches, metric="wrong_target_paraphrase", arm="matched_control", scope=p_scope, input_set=p_input_set, mode="eval_no_dropout")

    return {
        "canonical": {
            "num": m_c.numerator, "den": m_c.denominator, "rate": m_c.rate,
            "w_lo": m_c.wilson_low, "w_hi": m_c.wilson_high, "raw_vectors": wrong_c_matches
        },
        "paraphrase": {
            "num": m_p.numerator, "den": m_p.denominator, "rate": m_p.rate,
            "w_lo": m_p.wilson_low, "w_hi": m_p.wilson_high, "raw_vectors": wrong_p_matches
        }
    }
