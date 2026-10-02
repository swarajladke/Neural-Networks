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
        v_null = evecs[:, null_mask]  # [3072, null_dim]
        p_0 = torch.matmul(v_null, v_null.t()).to(torch.float32)
    else:
        # Fallback to empty null space projector
        p_0 = torch.zeros((3072, 3072), dtype=torch.float32)

    cov_sha = hashlib.sha256(cov.numpy().tobytes()).hexdigest()
    p0_sha = hashlib.sha256(p_0.numpy().tobytes()).hexdigest()

    return {
        "cov": cov,
        "cov_sha256": cov_sha,
        "p_0": p_0,
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
    Tracks preserved null space P_0 and incremental previous edit keys for Arm A-null.
    Maintains an orthonormal basis Q_prev of previous keys in range(P_0).
    For any vector x, project_preserved(x) projects x into range(P_0)
    AND orthogonal to all previous edit keys k_0, ..., k_{t-1}.
    Asserts max_{j < t} ||k_j Delta W_t|| <= tolerance after every edit.
    """
    def __init__(self, p_0: torch.Tensor, tolerance: float = 1e-4, device: str = "cuda"):
        self.p_0 = p_0.to(device)
        self.tolerance = tolerance
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.q_basis: Optional[torch.Tensor] = None  # Orthonormal basis in range(P_0)
        self.max_observed_violation = 0.0

    def reset(self):
        """Resets the tracker history and orthonormal basis to initial state."""
        self.stored_keys = []
        self.q_basis = None
        self.max_observed_violation = 0.0

    def project_preserved(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects vector x into range(P_0) and orthogonal to all previous keys in self.q_basis.
        Uses double projection (Kahan re-orthogonalization) for numerical precision (< 1e-14).
        """
        x_p0 = torch.matmul(self.p_0, x)
        if self.q_basis is not None and self.q_basis.shape[1] > 0:
            proj1 = torch.matmul(self.q_basis, torch.matmul(self.q_basis.t(), x_p0))
            x_orth = x_p0 - proj1
            proj2 = torch.matmul(self.q_basis, torch.matmul(self.q_basis.t(), x_orth))
            x_orth = x_orth - proj2
            return x_orth
        return x_p0

    def compute_update_direction(
        self,
        k_vec: torch.Tensor,
        cov: torch.Tensor,
        ridge_val: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes u_vec in range(P_0) orthogonal to all previous edit keys such that k_vec * u_vec = 1.
        Applies AlphaEdit-style covariance weighting:
        tilde_k = P_M k
        z = C_reg^-1 tilde_k
        w = P_M z
        u = w / (k * w)
        """
        k_tilde = self.project_preserved(k_vec)
        k_norm = torch.linalg.norm(k_tilde).item()

        if k_norm > 1e-5:
            d_feat = cov.shape[0]
            cov_reg = cov.to(self.device) + ridge_val * torch.eye(d_feat, device=self.device)
            z = torch.linalg.solve(cov_reg, k_tilde)
            w = self.project_preserved(z)
            denom = torch.dot(k_vec, w).item()
            if denom > 1e-8:
                u_vec = w / denom
            else:
                u_vec = k_tilde / (torch.dot(k_vec, k_tilde).item() + 1e-12)
        else:
            u_vec = k_tilde / (torch.dot(k_vec, k_tilde).item() + 1e-12)

        return u_vec, k_norm

    def register_and_assert(
        self,
        k_vec: torch.Tensor,
        delta_w: torch.Tensor,
        delta_v: torch.Tensor
    ) -> float:
        """
        Asserts ||k_prev Delta W|| <= tolerance for all stored previous keys.
        Then updates the orthonormal basis Q_prev with the current key.
        """
        max_err = 0.0
        if len(self.stored_keys) > 0:
            prev_stack = torch.stack(self.stored_keys, dim=0).to(self.device)  # [T-1, 3072]
            err_vecs = torch.matmul(prev_stack, delta_w)  # [T-1, 768]
            err_norms = torch.linalg.norm(err_vecs, dim=1)
            max_err = torch.max(err_norms).item()
            if max_err > self.max_observed_violation:
                self.max_observed_violation = max_err
            assert max_err <= self.tolerance, f"Null-space assertion failure: max ||k_prev Delta W|| = {max_err:.2e} > {self.tolerance:.2e}"

        # Update Q_prev basis with k_vec projected through P_0 and existing Q_prev
        q_cand = self.project_preserved(k_vec)
        q_cand_norm = torch.linalg.norm(q_cand).item()
        if q_cand_norm > 1e-5:
            q_unit = (q_cand / q_cand_norm).unsqueeze(1)  # [3072, 1]
            if self.q_basis is None or self.q_basis.shape[1] == 0:
                self.q_basis = q_unit
            else:
                self.q_basis = torch.cat([self.q_basis, q_unit], dim=1)

        self.stored_keys.append(k_vec.detach().cpu())
        return max_err


def edit_fact_mlp_null(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    cov: torch.Tensor,
    null_tracker: SequentialNullTracker,
    layer_idx: int,
    max_steps: int = 100,
    lr_v: float = 0.1,
    ridge_factor: float = 1e-3,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Arm A-null: Covariance update projected into null space P_0 + incremental previous keys.
    Asserts max_{j < t} ||k_j Delta W_t|| < tolerance.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _ = find_target_value_vstar(model, tokenizer, fact, layer_idx, max_steps, lr_v, 0.0, device)

    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    delta_v = v_star - v0

    d_feat = cov.shape[0]
    tr_cov = float(torch.trace(cov).item())
    ridge_val = ridge_factor * (tr_cov / float(d_feat))

    u_vec, k_proj_norm = null_tracker.compute_update_direction(k_vec, cov, ridge_val)
    delta_w = torch.outer(u_vec, delta_v)

    # Assert tolerance on previous keys before applying update
    null_err = null_tracker.register_and_assert(k_vec, delta_w, delta_v)

    c_proj.weight.data.add_(delta_w)
    delta_norm = torch.linalg.norm(delta_w).item()

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])
    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "delta_norm": delta_norm,
        "pred": curr_pred,
        "null_err": null_err,
        "k_proj_norm": k_proj_norm
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

    m_c = Measurement.from_outcomes(wrong_c_matches, metric="wrong_target", arm="matched_control", scope="s0_11_matched_canonical", input_set="facts_first50", mode="eval_no_dropout")
    m_p = Measurement.from_outcomes(wrong_p_matches, metric="wrong_target_paraphrase", arm="matched_control", scope="s0_11_matched_paraphrase", input_set="paraphrases_150", mode="eval_no_dropout")

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
