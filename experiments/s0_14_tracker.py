#!/usr/bin/env python3
"""
experiments/s0_14_tracker.py -- Directive S0-14 Trackers and Optimization Engine

Implements:
  - StrictSequentialNullTracker (Arm C1): Strict protection, NO cosine exclusion,
    prohibits fallback (marks unwritable on degeneracy), audits residual on ALL keys.
  - SlotConsolidatedNullTracker (Arm C2): Slot consolidation on collision group G,
    projects orthogonal to non-colliding keys, verifies colliding keys within 1e-6 of joint v*.
  - SlotConsolidatedTemplateNullTracker (Arm C3): C2 + SVD-truncated full-prompt template protection.
  - find_joint_target_value_vstar: Joint v* optimization for collision group G with margin targeting.
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
from experiments.s0_10_repair import get_subject_last_token_idx
from experiments.s0_13_protection import extract_prompt_keys


class StrictSequentialNullTracker:
    """
    Arm C1: Strict Sequential Null-Space Tracker.
    Protects ALL prior keys without cosine exclusion.
    Fallback branch is strictly prohibited: degenerate updates are marked unwritable.
    """
    def __init__(self, v_null: torch.Tensor, device: str = "cuda"):
        self.v_null = v_null.to(device, dtype=torch.float64)
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.q_k: Optional[torch.Tensor] = None
        self.unwritable_count: int = 0
        self.max_relative_residual: float = 0.0

    def reset(self):
        self.stored_keys = []
        self.q_k = None
        self.unwritable_count = 0
        self.max_relative_residual = 0.0

    def update_basis(self):
        if not self.stored_keys or self.v_null.shape[1] == 0:
            self.q_k = None
            return
        k_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in self.stored_keys], dim=1)
        c_mat = torch.matmul(self.v_null.t(), k_mat)
        u_svd, s_svd, _ = torch.linalg.svd(c_mat, full_matrices=False)
        if s_svd.numel() > 0 and s_svd[0] > 1e-12:
            valid = (s_svd / s_svd[0]) > 1e-10
            self.q_k = u_svd[:, valid] if valid.any() else None
        else:
            self.q_k = None

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

    def compute_update(self, k_vec: torch.Tensor, r_vec: torch.Tensor) -> Dict[str, Any]:
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        r_64 = r_vec.to(self.device, dtype=torch.float64)
        self.update_basis()
        p_k, rho = self.project_k(k_64)
        k_p_k = torch.dot(k_64, p_k).item()

        if k_p_k <= 1e-12:
            self.unwritable_count += 1
            u_f32 = torch.zeros_like(k_vec, dtype=torch.float32)
            r_f32 = torch.zeros_like(r_vec, dtype=torch.float32)
            delta_f32 = torch.zeros((k_vec.shape[0], r_vec.shape[0]), device=self.device, dtype=torch.float32)
            return {
                "delta_f32": delta_f32, "u_f32": u_f32, "r_f32": r_f32,
                "is_unwritable": True, "k_p_k": k_p_k, "relative_residual": 0.0
            }

        delta_64 = torch.outer(p_k, r_64) / k_p_k
        delta_norm_2 = torch.linalg.norm(delta_64, ord=2).item() + 1e-12
        max_rel_res = 0.0

        if len(self.stored_keys) > 0:
            stored_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in self.stored_keys], dim=0)
            proj_vals = torch.abs(torch.matmul(stored_mat, p_k))
            k_norms = torch.linalg.norm(stored_mat, dim=1) + 1e-12
            err_norms = (proj_vals * torch.linalg.norm(r_64).item()) / (k_p_k + 1e-12)
            rel_res_vec = err_norms / (k_norms * delta_norm_2)
            max_rel_res = torch.max(rel_res_vec).item()

        if max_rel_res > self.max_relative_residual:
            self.max_relative_residual = max_rel_res

        assert max_rel_res <= 1e-8, f"C1 strict null-space residual gate failure: {max_rel_res:.2e} > 1.00e-08"

        u_64 = p_k / k_p_k
        self.stored_keys.append(k_64.detach())
        return {
            "delta_f32": delta_64.to(torch.float32),
            "u_f32": u_64.to(torch.float32),
            "r_f32": r_64.to(torch.float32),
            "is_unwritable": False,
            "k_p_k": k_p_k,
            "relative_residual": max_rel_res
        }


class SlotConsolidatedNullTracker:
    """
    Arm C2: Slot Consolidation Tracker.
    Detects key collisions (cosine >= 0.999) with prior keys.
    Forms collision group G, projects update orthogonal to non-colliding prior keys,
    and asserts colliding prior keys map to joint v* within relative residual 1e-6.
    """
    def __init__(self, v_null: torch.Tensor, cos_threshold: float = 0.999, device: str = "cuda"):
        self.v_null = v_null.to(device, dtype=torch.float64)
        self.cos_threshold = cos_threshold
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.stored_vstars: List[torch.Tensor] = []
        self.stored_facts: List[Dict[str, Any]] = []
        self.max_noncoll_residual = 0.0
        self.max_coll_residual = 0.0

    def reset(self):
        self.stored_keys = []
        self.stored_vstars = []
        self.stored_facts = []
        self.max_noncoll_residual = 0.0
        self.max_coll_residual = 0.0

    def detect_collisions(self, k_vec: torch.Tensor) -> List[int]:
        if not self.stored_keys:
            return []
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        k_unit = k_64 / (torch.linalg.norm(k_64) + 1e-12)
        stored_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in self.stored_keys], dim=0)
        stored_norms = torch.linalg.norm(stored_mat, dim=1, keepdim=True) + 1e-12
        stored_units = stored_mat / stored_norms
        cos_sims = torch.matmul(stored_units, k_unit)
        coll_mask = cos_sims >= self.cos_threshold
        return torch.nonzero(coll_mask, as_tuple=True)[0].tolist()

    def update_basis_for_noncolliding(self, noncoll_keys: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not noncoll_keys or self.v_null.shape[1] == 0:
            return None
        k_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in noncoll_keys], dim=1)
        c_mat = torch.matmul(self.v_null.t(), k_mat)
        u_svd, s_svd, _ = torch.linalg.svd(c_mat, full_matrices=False)
        if s_svd.numel() > 0 and s_svd[0] > 1e-12:
            valid = (s_svd / s_svd[0]) > 1e-10
            return u_svd[:, valid] if valid.any() else None
        return None

    def compute_update(
        self,
        k_vec: torch.Tensor,
        r_vec: torch.Tensor,
        coll_indices: List[int],
        joint_vstar: torch.Tensor,
        curr_fact: Dict[str, Any],
        w_curr: torch.Tensor,
        b_curr: torch.Tensor
    ) -> Dict[str, Any]:
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        r_64 = r_vec.to(self.device, dtype=torch.float64)
        noncoll_keys = [self.stored_keys[i] for i in range(len(self.stored_keys)) if i not in coll_indices]

        q_k = self.update_basis_for_noncolliding(noncoll_keys)
        c = torch.matmul(self.v_null.t(), k_64)
        if q_k is not None and q_k.shape[1] > 0:
            p1 = torch.matmul(q_k, torch.matmul(q_k.t(), c))
            c_orth = c - p1
            p2 = torch.matmul(q_k, torch.matmul(q_k.t(), c_orth))
            c_orth = c_orth - p2
        else:
            c_orth = c
        p_k = torch.matmul(self.v_null, c_orth)
        k_p_k = torch.dot(k_64, p_k).item()

        if k_p_k <= 1e-12:
            raise RuntimeError(f"Fallback branch unreachable: k^T P k = {k_p_k:.2e} <= 1e-12 in SlotConsolidatedNullTracker")

        delta_64 = torch.outer(p_k, r_64) / k_p_k
        delta_norm_2 = torch.linalg.norm(delta_64, ord=2).item() + 1e-12

        # 1. Gate on all non-colliding keys (<= 1e-8)
        max_noncoll_res = 0.0
        if noncoll_keys:
            noncoll_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in noncoll_keys], dim=0)
            proj_vals = torch.abs(torch.matmul(noncoll_mat, p_k))
            k_norms = torch.linalg.norm(noncoll_mat, dim=1) + 1e-12
            err_norms = (proj_vals * torch.linalg.norm(r_64).item()) / (k_p_k + 1e-12)
            rel_res_vec = err_norms / (k_norms * delta_norm_2)
            max_noncoll_res = torch.max(rel_res_vec).item()
        if max_noncoll_res > self.max_noncoll_residual:
            self.max_noncoll_residual = max_noncoll_res
        assert max_noncoll_res <= 1e-8, f"C2 non-colliding residual gate failure: {max_noncoll_res:.2e} > 1.00e-08"

        # 2. Gate on all colliding keys (<= 1e-6 relative to joint v*)
        max_coll_res = 0.0
        w_new_64 = w_curr.to(self.device, dtype=torch.float64) + delta_64
        b_64 = b_curr.to(self.device, dtype=torch.float64)
        vstar_64 = joint_vstar.to(self.device, dtype=torch.float64)
        vstar_norm = torch.linalg.norm(vstar_64).item() + 1e-12
        for c_idx in coll_indices:
            k_c = self.stored_keys[c_idx]
            v_pred = torch.addmm(b_64, k_c.unsqueeze(0), w_new_64).squeeze(0)
            err_norm = torch.linalg.norm(v_pred - vstar_64).item()
            rel_coll = err_norm / vstar_norm
            if rel_coll > max_coll_res:
                max_coll_res = rel_coll
            self.stored_vstars[c_idx] = joint_vstar.detach().cpu()
        if max_coll_res > self.max_coll_residual:
            self.max_coll_residual = max_coll_res
        assert max_coll_res <= 1e-6, f"C2 colliding residual gate failure: {max_coll_res:.2e} > 1.00e-06"

        self.stored_keys.append(k_64.detach())
        self.stored_vstars.append(joint_vstar.detach().cpu())
        self.stored_facts.append(curr_fact)

        u_64 = p_k / k_p_k
        return {
            "delta_f32": delta_64.to(torch.float32),
            "u_f32": u_64.to(torch.float32),
            "r_f32": r_64.to(torch.float32),
            "k_p_k": k_p_k,
            "noncoll_residual": max_noncoll_res,
            "coll_residual": max_coll_res
        }


class SlotConsolidatedTemplateNullTracker:
    """
    Arm C3: Slot Consolidation + Template Protection Tracker.
    Combines C2 slot consolidation with SVD-truncated basis protection for prompt tokens.
    Logs protected rank, rho, and spectral norm.
    """
    def __init__(self, v_null: torch.Tensor, rel_threshold: float = 1e-3, cos_threshold: float = 0.999, device: str = "cuda"):
        self.v_null = v_null.to(device, dtype=torch.float64)
        self.d_feat, self.d_null = self.v_null.shape
        self.rel_threshold = rel_threshold
        self.cos_threshold = cos_threshold
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.stored_vstars: List[torch.Tensor] = []
        self.q_prompt: Optional[torch.Tensor] = None
        self.rank_history: List[int] = []
        self.rho_history: List[float] = []
        self.spectral_norms: List[float] = []

    def reset(self):
        self.stored_keys = []
        self.stored_vstars = []
        self.q_prompt = None
        self.rank_history = []
        self.rho_history = []
        self.spectral_norms = []

    def detect_collisions(self, k_vec: torch.Tensor) -> List[int]:
        if not self.stored_keys:
            return []
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        k_unit = k_64 / (torch.linalg.norm(k_64) + 1e-12)
        stored_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in self.stored_keys], dim=0)
        stored_norms = torch.linalg.norm(stored_mat, dim=1, keepdim=True) + 1e-12
        stored_units = stored_mat / stored_norms
        cos_sims = torch.matmul(stored_units, k_unit)
        coll_mask = cos_sims >= self.cos_threshold
        return torch.nonzero(coll_mask, as_tuple=True)[0].tolist()

    def add_prompt_keys(self, prompt_keys: torch.Tensor):
        if prompt_keys.shape[0] == 0:
            return
        keys_64 = prompt_keys.to(self.device, dtype=torch.float64)
        c_mat = torch.matmul(self.v_null.t(), keys_64.t())
        if self.q_prompt is not None and self.q_prompt.shape[1] > 0:
            c_orth = c_mat - self.q_prompt @ (self.q_prompt.t() @ c_mat)
            c_orth = c_orth - self.q_prompt @ (self.q_prompt.t() @ c_orth)
        else:
            c_orth = c_mat
        u_svd, s_svd, _ = torch.linalg.svd(c_orth, full_matrices=False)
        if s_svd.numel() > 0 and s_svd[0] > 1e-12:
            keep_mask = (s_svd / s_svd[0]) >= self.rel_threshold
            u_keep = u_svd[:, keep_mask]
            if u_keep.shape[1] > 0:
                if self.q_prompt is None or self.q_prompt.shape[1] == 0:
                    self.q_prompt = u_keep
                else:
                    self.q_prompt = torch.cat([self.q_prompt, u_keep], dim=1)
                self.q_prompt, _ = torch.linalg.qr(self.q_prompt)

    def compute_update(
        self,
        k_vec: torch.Tensor,
        r_vec: torch.Tensor,
        coll_indices: List[int],
        joint_vstar: torch.Tensor,
        w_curr: torch.Tensor,
        b_curr: torch.Tensor
    ) -> Dict[str, Any]:
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        r_64 = r_vec.to(self.device, dtype=torch.float64)
        noncoll_keys = [self.stored_keys[i] for i in range(len(self.stored_keys)) if i not in coll_indices]

        # Combine non-colliding subject keys and prompt template directions
        c_k = torch.matmul(self.v_null.t(), k_64)
        all_q_parts = []
        if noncoll_keys:
            k_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in noncoll_keys], dim=1)
            c_mat = torch.matmul(self.v_null.t(), k_mat)
            all_q_parts.append(c_mat)
        if self.q_prompt is not None and self.q_prompt.shape[1] > 0:
            all_q_parts.append(self.q_prompt)

        if all_q_parts:
            combined = torch.cat(all_q_parts, dim=1)
            u_svd, s_svd, _ = torch.linalg.svd(combined, full_matrices=False)
            if s_svd.numel() > 0 and s_svd[0] > 1e-12:
                valid = (s_svd / s_svd[0]) > 1e-10
                q_comb = u_svd[:, valid] if valid.any() else None
            else:
                q_comb = None
            if q_comb is not None and q_comb.shape[1] > 0:
                p1 = torch.matmul(q_comb, torch.matmul(q_comb.t(), c_k))
                c_orth = c_k - p1
                p2 = torch.matmul(q_comb, torch.matmul(q_comb.t(), c_orth))
                c_orth = c_orth - p2
                curr_rank = q_comb.shape[1]
            else:
                c_orth = c_k
                curr_rank = 0
        else:
            c_orth = c_k
            curr_rank = 0

        p_k = torch.matmul(self.v_null, c_orth)
        k_p_k = torch.dot(k_64, p_k).item()
        denom = torch.dot(k_64, k_64).item() + 1e-12
        rho = torch.dot(c_orth, c_orth).item() / denom
        self.rank_history.append(curr_rank)
        self.rho_history.append(rho)

        if k_p_k <= 1e-12:
            raise RuntimeError(f"Fallback branch unreachable in C3: k^T P k = {k_p_k:.2e} <= 1e-12")

        delta_64 = torch.outer(p_k, r_64) / k_p_k
        spec_norm = torch.linalg.norm(delta_64, ord=2).item()
        self.spectral_norms.append(spec_norm)

        for c_idx in coll_indices:
            self.stored_vstars[c_idx] = joint_vstar.detach().cpu()
        self.stored_keys.append(k_64.detach())
        self.stored_vstars.append(joint_vstar.detach().cpu())
        u_64 = p_k / k_p_k

        return {
            "delta_f32": delta_64.to(torch.float32),
            "u_f32": u_64.to(torch.float32),
            "r_f32": r_64.to(torch.float32),
            "rank": curr_rank,
            "rho": rho,
            "spectral_norm": spec_norm
        }


def find_joint_target_value_vstar(
    model: nn.Module,
    tokenizer: Any,
    facts_group: List[Dict[str, Any]],
    layer_idx: int = 1,
    target_margin: float = 2.0,
    max_steps: int = 100,
    lr_v: float = 0.1,
    device: str = "cuda"
) -> Tuple[torch.Tensor, int, List[float]]:
    """
    Optimizes a single shared v* parameter at layer_idx c_proj for all facts in group G.
    Minimizes summed NLL while requiring every fact in G to meet target_margin (2.0 nats).
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    for p in model.parameters():
        p.requires_grad = False

    group_data = []
    v0_candidates = []

    for fact in facts_group:
        prompt = fact["edit_prompt"]
        subj_idx = get_subject_last_token_idx(tokenizer, prompt, fact["subject"])
        enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
        enc_full = tokenizer(f"{prompt} {fact['object']}", return_tensors="pt").to(device)
        p_len = enc_prompt.input_ids.shape[1]
        target_tok = enc_full.input_ids[0, p_len].item()

        # Capture v0
        rec = {}
        def hook_fn(m, inp, out, r=rec, s=subj_idx):
            r["v0"] = out[0, s, :].detach().clone()
        h = c_proj.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = model(**enc_prompt)
        finally:
            h.remove()

        v0_candidates.append(rec["v0"])
        group_data.append({
            "fact": fact,
            "subj_idx": subj_idx,
            "prompt_ids": enc_prompt.input_ids,
            "full_ids": enc_full.input_ids,
            "p_len": p_len,
            "target_tok": target_tok
        })

    v_init = torch.stack(v0_candidates, dim=0).mean(dim=0)
    v_param = nn.Parameter(v_init.clone())
    opt_v = torch.optim.Adam([v_param], lr=lr_v)

    steps_taken = 0
    final_margins = [-999.0] * len(facts_group)

    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            opt_v.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            current_margins = []

            for g_item in group_data:
                s_idx = g_item["subj_idx"]
                full_ids = g_item["full_ids"]
                p_len = g_item["p_len"]
                target_tok = g_item["target_tok"]

                def hook_rep(m, inp, out, s=s_idx):
                    out_mod = out.clone()
                    out_mod[0, s, :] = v_param
                    return out_mod

                h_rep = c_proj.register_forward_hook(hook_rep)
                try:
                    labels = full_ids.clone()
                    labels[:, :p_len] = -100
                    out = model(full_ids, labels=labels)
                    total_loss = total_loss + out.loss

                    logits = out.logits[0, p_len - 1, :]
                    log_probs = F.log_softmax(logits, dim=-1)
                    t_lp = log_probs[target_tok].item()
                    mask = torch.ones_like(log_probs, dtype=torch.bool)
                    mask[target_tok] = False
                    nt_lp = torch.max(log_probs[mask]).item()
                    current_margins.append(t_lp - nt_lp)
                finally:
                    h_rep.remove()

            total_loss.backward()
            opt_v.step()
            final_margins = current_margins

            if all(m >= target_margin for m in current_margins):
                break

    v_star = v_param.detach()
    del v_param, opt_v
    return v_star, steps_taken, final_margins
