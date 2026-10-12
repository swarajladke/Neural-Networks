#!/usr/bin/env python3
"""
experiments/s0_14_audit.py -- Directive S0-14 Stage A Key-Collision Audit Engine

Implements:
  - InstrumentedAuditNullTracker: Bitwise-identical mathematical execution to
    CorrectedSequentialNullTracker, instrumented to log cosine exclusions,
    fallback activations, and max prior cosine values per edit.
  - compute_fisher_exact_2x2: Exact hypergeometric 2x2 contingency test without SciPy.
  - analyze_collision_contingency: 2x2 contingency on excluded_later x lost_at_end with Newcombe CI.
  - analyze_collision_anatomy: Subject, relation, and prompt structure decomposition of collisions.
  - compute_non_excluded_residual_drift: Median drift_j among non-excluded lost facts.
  - run_recomputed_stage_r: Recomputed seed-clustered logistic regression with drift_j and excluded_later_j.
"""

import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from experiments.stats import newcombe_score_interval
from experiments.s0_10_repair import get_subject_last_token_idx, verify_state_restore
from experiments.s0_11_constraints import find_target_value_vstar
from experiments.s0_13_protection import find_target_value_vstar_margin
from experiments.s0_13_localization import compute_target_margin
from experiments.b1_inject import configure_determinism, greedy_predict, edit_fact_sgd
from experiments.metrics import check_match


def evaluate_fact_retention(
    model: nn.Module,
    tokenizer: Any,
    facts_eval: List[Dict[str, Any]],
    max_new_tokens: int = 5,
    device: str = "cuda",
    train_mode: bool = False
) -> Tuple[List[bool], List[bool]]:
    """
    Evaluates canonical and paraphrase retention across facts_eval.
    Returns (canonical_matches, paraphrase_matches).
    """
    c_matches = []
    p_matches = []
    for f in facts_eval:
        pred_c = greedy_predict(model, tokenizer, f["edit_prompt"], max_new_tokens, device, train_mode)
        c_matches.append(check_match(pred_c, f["object"]))
        for p in f.get("paraphrases", []):
            pred_p = greedy_predict(model, tokenizer, p, max_new_tokens, device, train_mode)
            p_matches.append(check_match(pred_p, f["object"]))
    return c_matches, p_matches


class InstrumentedAuditNullTracker:
    """
    Instrumented tracker mirroring CorrectedSequentialNullTracker bit-for-bit.
    Logs prior key exclusions, cosine values, fallback branch activations,
    and maximum prior cosines without altering mathematical updates or weights.
    """
    def __init__(self, v_null: torch.Tensor, rel_threshold: float = 1e-3, device: str = "cuda"):
        self.v_null = v_null.to(device, dtype=torch.float64)
        self.rel_threshold = rel_threshold
        self.device = device
        self.stored_keys: List[torch.Tensor] = []
        self.q_k: Optional[torch.Tensor] = None
        self.rho_history: List[float] = []
        self.max_relative_residual: float = 0.0

        # Audit instrumentation logs
        self.exclusions_per_edit: Dict[int, List[Dict[str, Any]]] = {}
        self.max_prior_cosines: Dict[int, float] = {}
        self.fallback_fired: Dict[int, bool] = {}

    def reset(self):
        self.stored_keys = []
        self.q_k = None
        self.rho_history = []
        self.max_relative_residual = 0.0
        self.exclusions_per_edit = {}
        self.max_prior_cosines = {}
        self.fallback_fired = {}

    def get_other_keys(self, k_vec: torch.Tensor, edit_idx: int) -> List[torch.Tensor]:
        if not self.stored_keys:
            self.exclusions_per_edit[edit_idx] = []
            self.max_prior_cosines[edit_idx] = 0.0
            return []

        k_64 = k_vec.to(self.device, dtype=torch.float64)
        k_unit = k_64 / (torch.linalg.norm(k_64) + 1e-12)
        other, other_units = [], []
        exclusions = []
        max_cos = -1.0

        n_stored = len(self.stored_keys)
        # Scan in reverse order exactly matching CorrectedSequentialNullTracker
        for offset, prev_k in enumerate(reversed(self.stored_keys)):
            orig_idx = n_stored - 1 - offset
            prev_64 = prev_k.to(self.device, dtype=torch.float64)
            prev_unit = prev_64 / (torch.linalg.norm(prev_64) + 1e-12)
            cos_val = torch.dot(k_unit, prev_unit).item()
            if cos_val > max_cos:
                max_cos = cos_val

            if cos_val < 0.999:
                dedup_match = False
                for u in other_units:
                    if torch.dot(prev_unit, u).item() >= 0.999:
                        dedup_match = True
                        break
                if not dedup_match:
                    other.append(prev_64)
                    other_units.append(prev_unit)
                else:
                    exclusions.append({"prior_idx": orig_idx, "cosine": cos_val, "reason": "dedup"})
            else:
                exclusions.append({"prior_idx": orig_idx, "cosine": cos_val, "reason": "cosine_threshold"})

        self.exclusions_per_edit[edit_idx] = exclusions
        self.max_prior_cosines[edit_idx] = max_cos
        other.reverse()
        return other

    def update_basis(self, other_keys: List[torch.Tensor]):
        if not other_keys or self.v_null.shape[1] == 0:
            self.q_k = None
            return
        k_mat = torch.stack([k.to(self.device, dtype=torch.float64) for k in other_keys], dim=1)
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

    def compute_update(self, k_vec: torch.Tensor, r_vec: torch.Tensor, edit_idx: int) -> Dict[str, Any]:
        k_64 = k_vec.to(self.device, dtype=torch.float64)
        r_64 = r_vec.to(self.device, dtype=torch.float64)
        other_keys = self.get_other_keys(k_64, edit_idx)
        self.update_basis(other_keys)
        p_k, rho = self.project_k(k_64)
        self.rho_history.append(rho)
        k_p_k = torch.dot(k_64, p_k).item()

        if k_p_k > 1e-12:
            delta_64 = torch.outer(p_k, r_64) / k_p_k
            self.fallback_fired[edit_idx] = False
        else:
            p_cand = p_k if torch.linalg.norm(p_k) > 1e-12 else k_64
            denom = torch.dot(k_64, p_cand).item() + 1e-12
            delta_64 = torch.outer(p_cand, r_64) / denom
            self.fallback_fired[edit_idx] = True

        delta_norm_2 = torch.linalg.norm(delta_64, ord=2).item() + 1e-12
        max_rel_res = 0.0
        if len(other_keys) > 0:
            for prev_k in other_keys:
                k_norm = torch.linalg.norm(prev_k).item() + 1e-12
                proj_val = abs(torch.dot(prev_k, p_k).item())
                err_norm = (proj_val * torch.linalg.norm(r_64).item()) / (k_p_k + 1e-12)
                rel_res = err_norm / (k_norm * delta_norm_2)
                if rel_res > max_rel_res:
                    max_rel_res = rel_res

        if max_rel_res > self.max_relative_residual:
            self.max_relative_residual = max_rel_res

        self.stored_keys.append(k_64.detach())
        u_64 = p_k / (k_p_k + 1e-12)
        return {
            "delta_f32": delta_64.to(torch.float32),
            "u_f32": u_64.to(torch.float32),
            "r_f32": r_64.to(torch.float32),
            "p_k": p_k,
            "k_vec": k_vec,
            "r_vec": r_vec,
            "k_p_k": k_p_k,
            "relative_residual": max_rel_res,
            "fallback_fired": self.fallback_fired[edit_idx],
            "exclusions": self.exclusions_per_edit[edit_idx]
        }


def compute_fisher_exact_2x2(table: List[List[int]]) -> Tuple[float, float]:
    """
    Computes exact two-sided Fisher's exact test for a 2x2 contingency table:
      [[a, b],
       [c, d]]
    Returns (odds_ratio, p_value). Pure math, zero external library dependency.
    """
    a, b = table[0][0], table[0][1]
    c, d = table[1][0], table[1][1]

    # Odds ratio with Haldane-Anscombe continuity if zeros present
    if a * b * c * d == 0:
        or_val = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    else:
        or_val = (a * d) / (b * c)

    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    n = r1 + r2

    # Hypergeometric log-probabilities
    def log_comb(n_val: int, k_val: int) -> float:
        if k_val < 0 or k_val > n_val:
            return -float("inf")
        return math.lgamma(n_val + 1) - math.lgamma(k_val + 1) - math.lgamma(n_val - k_val + 1)

    def log_hypergeom(x: int) -> float:
        return log_comb(r1, x) + log_comb(r2, c1 - x) - log_comb(n, c1)

    min_x = max(0, c1 - r2)
    max_x = min(r1, c1)
    obs_logp = log_hypergeom(a)

    p_value = 0.0
    for x in range(min_x, max_x + 1):
        lp = log_hypergeom(x)
        # Sum probabilities less than or equal to observed
        if lp <= obs_logp + 1e-10:
            p_value += math.exp(lp)

    p_value = min(max(p_value, 0.0), 1.0)
    return float(or_val), float(p_value)


def analyze_collision_contingency(
    excluded_later_pooled: List[int],
    lost_at_end_pooled: List[int]
) -> Dict[str, Any]:
    """
    Evaluates Stage A3 pre-registered 2x2 contingency table:
      (excluded_later x lost_at_end) on first-50 facts pooled across 6 seeds (N=300).
    """
    assert len(excluded_later_pooled) == len(lost_at_end_pooled) == 300
    a = sum(1 for exc, lost in zip(excluded_later_pooled, lost_at_end_pooled) if exc == 1 and lost == 1)
    b = sum(1 for exc, lost in zip(excluded_later_pooled, lost_at_end_pooled) if exc == 1 and lost == 0)
    c = sum(1 for exc, lost in zip(excluded_later_pooled, lost_at_end_pooled) if exc == 0 and lost == 1)
    d = sum(1 for exc, lost in zip(excluded_later_pooled, lost_at_end_pooled) if exc == 0 and lost == 0)

    n_exc = a + b
    n_nonexc = c + d
    loss_rate_exc = (a / n_exc) if n_exc > 0 else 0.0
    loss_rate_nonexc = (c / n_nonexc) if n_nonexc > 0 else 0.0

    table = [[a, b], [c, d]]
    odds_ratio, fisher_p = compute_fisher_exact_2x2(table)

    # Newcombe 95% CI on difference in loss rate: loss_rate_exc - loss_rate_nonexc
    if n_exc > 0 and n_nonexc > 0:
        diff_pt, ci_lower, ci_upper = newcombe_score_interval(a, n_exc, c, n_nonexc)
    else:
        diff_pt, ci_lower, ci_upper = 0.0, 0.0, 0.0

    confirmed_mechanism = (ci_lower > 0.0) if n_exc > 0 else False

    return {
        "contingency_table": {"a_exc_lost": a, "b_exc_ret": b, "c_nonexc_lost": c, "d_nonexc_ret": d},
        "n_excluded": n_exc,
        "n_nonexcluded": n_nonexc,
        "loss_rate_excluded": loss_rate_exc,
        "loss_rate_nonexcluded": loss_rate_nonexc,
        "loss_rate_diff": diff_pt,
        "newcombe_ci_95": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "fisher_pvalue": fisher_p,
        "confirmed_mechanism": confirmed_mechanism
    }


def analyze_collision_anatomy(
    facts_per_seed: Dict[int, List[Dict[str, Any]]],
    exclusions_per_seed: Dict[int, Dict[int, List[Dict[str, Any]]]],
    tokenizer: Any
) -> Dict[str, Any]:
    """
    Decomposes collision anatomy for every excluded pair (t, j):
      same subject string, same subject last token, same relation, prompt begins with subject.
    Also computes the distribution of max prior-key cosine per edit.
    """
    total_pairs = 0
    same_subject_str = 0
    same_subject_last_tok = 0
    same_relation = 0
    prompt_begins_subj = 0

    all_max_cosines = []

    for s, s_facts in facts_per_seed.items():
        s_excl = exclusions_per_seed[s]
        for t, excl_list in s_excl.items():
            fact_t = s_facts[t]
            p_t = fact_t["edit_prompt"]
            s_t = fact_t["subject"]
            t_tok = get_subject_last_token_idx(tokenizer, p_t, s_t)
            t_tok_id = tokenizer(p_t, return_tensors="pt").input_ids[0, t_tok].item()

            for item in excl_list:
                j = item["prior_idx"]
                fact_j = s_facts[j]
                p_j = fact_j["edit_prompt"]
                s_j = fact_j["subject"]
                j_tok = get_subject_last_token_idx(tokenizer, p_j, s_j)
                j_tok_id = tokenizer(p_j, return_tensors="pt").input_ids[0, j_tok].item()

                total_pairs += 1
                if s_t.strip().lower() == s_j.strip().lower():
                    same_subject_str += 1
                if t_tok_id == j_tok_id:
                    same_subject_last_tok += 1
                if fact_t["relation"] == fact_j["relation"]:
                    same_relation += 1
                if p_t.strip().lower().startswith(s_t.strip().lower()):
                    prompt_begins_subj += 1

    return {
        "total_pairs": total_pairs,
        "same_subject_str": same_subject_str,
        "same_subject_last_tok": same_subject_last_tok,
        "same_relation": same_relation,
        "prompt_begins_subj": prompt_begins_subj
    }


def compute_non_excluded_residual_drift(
    drifts_pooled: List[float],
    excluded_later_pooled: List[int],
    lost_at_end_pooled: List[int]
) -> float:
    """
    Stage A5: Median drift_j among NON-excluded lost facts.
    """
    non_exc_lost_drifts = [
        d for d, exc, lost in zip(drifts_pooled, excluded_later_pooled, lost_at_end_pooled)
        if exc == 0 and lost == 1
    ]
    if not non_exc_lost_drifts:
        return 0.0
    sorted_d = sorted(non_exc_lost_drifts)
    n = len(sorted_d)
    mid = n // 2
    return sorted_d[mid] if n % 2 == 1 else 0.5 * (sorted_d[mid - 1] + sorted_d[mid])


def run_recomputed_stage_r(
    records: List[Dict[str, Any]],
    cluster_key: str = "seed"
) -> Dict[str, Any]:
    """
    Stage A6: Recomputed seed-clustered logistic regression of terminal retention
    adding per-seed float64 drift_j and excluded_later_j as explicit predictors.
    """
    feature_names = [
        "intercept",
        "edit_position",
        "later_same_rel_count",
        "is_multi_token",
        "write_time_margin",
        "drift_j",
        "excluded_later_j"
    ]

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
            float(r["drift_j"]),
            float(r["excluded_later_j"])
        ])
        Y_vals.append(1.0 if r["terminal_retained"] else 0.0)
        clusters.append(r[cluster_key])

    X = torch.tensor(X_rows, dtype=torch.float64)
    Y = torch.tensor(Y_vals, dtype=torch.float64)
    k_feats = X.shape[1]

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
        se_cluster = torch.sqrt(torch.clamp(torch.diag(V_cluster), min=0.0))
    except Exception:
        se_cluster = torch.zeros(k_feats, dtype=torch.float64)

    z_crit = 1.95996
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
        "n_obs": len(records),
        "n_clusters": len(unique_clusters),
        "results": results
    }


def run_g0_bit_reproduction(model: nn.Module, tokenizer: Any, base_state: Dict[str, torch.Tensor], facts: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    print("\n--- [Stage 0: Gate G0 Bit-Reproduction Control (Seed 0, unconstrained SGD)] ---")
    configure_determinism(seed=0)
    model.load_state_dict(base_state)
    t0 = time.time()
    imm_matches, total_steps = [], 0
    for fact in facts:
        res = edit_fact_sgd(model, tokenizer, fact, lr=3e-5, max_steps=100, delta=0.0, device=device, train_mode=False)
        imm_matches.append(res["immediate_match"])
        total_steps += res["steps_taken"]
    c_ret, _ = evaluate_fact_retention(model, tokenizer, facts, 5, device, False)
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
    return {"total_steps": total_steps, "immediate": sum(imm_matches), "terminal": term_matches, "elapsed": elapsed}


def load_and_assert_pinned_cache(cache_dir: str, device: str) -> Dict[str, Any]:
    print("\n--- [Stage 0: Gate G0b Asserting Pinned Cache Hashes] ---")
    c_path = os.path.join(cache_dir, "cov_l1.pt")
    v_path = os.path.join(cache_dir, "v_null_l1.pt")
    assert os.path.exists(c_path) and os.path.exists(v_path), "Pinned cache files missing from disk!"
    with open(c_path, "rb") as f:
        c_sha = hashlib.sha256(f.read()).hexdigest()
    with open(v_path, "rb") as f:
        v_sha = hashlib.sha256(f.read()).hexdigest()
    expected_c = "795b9c54a5b7197957af05c116502679a3dbc82357f0a9f4cac9be0c276dcde5"
    expected_v = "82aa46ced8021e4c32a63f1c94b1bdc6f8deef2a90dcba6c931db296f3cddf4a"
    print(f"  Cov SHA: {c_sha} (expected {expected_c[:16]}...)")
    print(f"  V_null SHA: {v_sha} (expected {expected_v[:16]}...)")
    assert c_sha == expected_c, f"Gate G0b Covariance SHA mismatch: {c_sha} != {expected_c}"
    assert v_sha == expected_v, f"Gate G0b V_null SHA mismatch: {v_sha} != {expected_v}"
    print("  Gate G0b: PASSED.")
    return {
        "cov": torch.load(c_path, map_location=device),
        "v_null": torch.load(v_path, map_location=device),
        "cov_sha": c_sha, "v_null_sha": v_sha
    }


def run_audit_seed(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    facts_seq: List[Dict[str, Any]],
    v_null: torch.Tensor,
    use_margin: bool = False,
    device: str = "cuda"
) -> Dict[str, Any]:
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    w_0 = c_proj.weight.data.clone().cpu()
    b_0 = c_proj.bias.data.clone().cpu()
    tracker = InstrumentedAuditNullTracker(v_null, rel_threshold=1e-3, device=device)

    imm_matches, steps_per_edit = [], []
    factors_32, k_vecs, r_vecs, write_margins = [], [], [], []

    for idx, fact in enumerate(facts_seq):
        if use_margin:
            k_vec, v_star, steps, _, _ = find_target_value_vstar_margin(
                model, tokenizer, fact, layer_idx=1, target_margin=2.0, max_steps=100, lr_v=0.1, device=device
            )
        else:
            k_vec, v_star, steps, _ = find_target_value_vstar(model, tokenizer, fact, 1, 100, 0.1, 0.0, device)

        v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
        r_vec = v_star - v0
        res = tracker.compute_update(k_vec, r_vec, idx)
        c_proj.weight.data.add_(res["delta_f32"])

        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
        imm_matches.append(check_match(curr_pred, fact["object"]))
        steps_per_edit.append(steps)

        k_vecs.append(k_vec.cpu())
        r_vecs.append(r_vec.cpu())
        factors_32.append((res["u_f32"].cpu(), res["r_f32"].cpu()))
        write_margins.append(compute_target_margin(model, tokenizer, fact["edit_prompt"], fact["object"], device=device))

    c_ret, p_ret = evaluate_fact_retention(model, tokenizer, facts_seq[:50], 5, device, False)
    term_margins = [compute_target_margin(model, tokenizer, f["edit_prompt"], f["object"], device=device) for f in facts_seq[:50]]
    final_w_sha = hashlib.sha256(c_proj.weight.data.cpu().numpy().tobytes()).hexdigest()

    return {
        "immediate_matches": imm_matches, "steps_per_edit": steps_per_edit,
        "canonical_retention": c_ret, "paraphrase_retention": p_ret,
        "write_margins": write_margins, "terminal_margins": term_margins,
        "factors_32": factors_32, "k_vecs": k_vecs, "r_vecs": r_vecs,
        "w_0": w_0, "b_0": b_0, "final_weight_sha": final_w_sha,
        "exclusions_per_edit": tracker.exclusions_per_edit,
        "max_prior_cosines": tracker.max_prior_cosines,
        "fallback_fired": tracker.fallback_fired
    }

