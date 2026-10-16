#!/usr/bin/env python3
"""
experiments/s0_16_evaluator.py -- Directive S0-16 Evaluation and Statistical Engine

Implements the formal evaluation contract for Directive S0-16:
  - Per-example EvaluationRecord data structure
  - Full 2x2 immediate-to-terminal transition table with exact intersection accounting
  - Population identity and denominator integrity guards (AGENTS.md Section 1.2, 1.3)
  - Exact mean-based paired sign-flip test across 2^N configurations
  - Tensor storage and projector basis orthonormality verification
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import torch


@dataclass
class EvaluationRecord:
    """
    Machine-readable record for a single model evaluation instance.
    """
    fact_id: int
    seed: int
    arm: str
    checkpoint: str
    prompt_type: str  # "canonical" or "paraphrase"
    prompt_id: str
    prompt_text: str
    target_text: str
    prediction_text: str
    is_match: bool


def compute_transition_table(
    immediate_records: List[EvaluationRecord],
    terminal_records: List[EvaluationRecord]
) -> Dict[str, Any]:
    """
    Computes exact 2x2 immediate-to-terminal transition table from paired per-example records.
    Requires matched fact_ids across both sets.
    """
    n_imm = len(immediate_records)
    n_term = len(terminal_records)
    if n_imm != n_term:
        raise ValueError(
            f"Population mismatch: immediate has {n_imm} records, terminal has {n_term} records"
        )

    imm_by_id = {r.fact_id: r for r in immediate_records}
    term_by_id = {r.fact_id: r for r in terminal_records}

    if set(imm_by_id.keys()) != set(term_by_id.keys()):
        raise ValueError("Population mismatch: distinct fact_id sets between immediate and terminal evaluations")

    a = 0  # I=1, T=1 (acquired and retained)
    b = 0  # I=1, T=0 (acquired then forgotten)
    c = 0  # I=0, T=1 (delayed success)
    d = 0  # I=0, T=0 (unsuccessful both)

    for f_id in imm_by_id:
        i_flag = imm_by_id[f_id].is_match
        t_flag = term_by_id[f_id].is_match
        if i_flag and t_flag:
            a += 1
        elif i_flag and not t_flag:
            b += 1
        elif not i_flag and t_flag:
            c += 1
        else:
            d += 1

    total_n = a + b + c + d
    assert total_n == n_imm, f"Denominator assertion failure: {a} + {b} + {c} + {d} = {total_n} != {n_imm}"

    sum_i = a + b
    sum_t = a + c

    imm_rate = (sum_i / float(total_n)) if total_n > 0 else 0.0
    term_rate = (sum_t / float(total_n)) if total_n > 0 else 0.0
    cond_ret = (a / float(sum_i)) if sum_i > 0 else 0.0
    forget_rate = (b / float(sum_i)) if sum_i > 0 else 0.0

    return {
        "n_total": total_n,
        "counts": {
            "a_acquired_and_retained": a,
            "b_acquired_then_forgotten": b,
            "c_delayed_success": c,
            "d_unsuccessful_both": d
        },
        "immediate_efficacy": {"numerator": sum_i, "denominator": total_n, "rate": imm_rate},
        "terminal_accuracy": {"numerator": sum_t, "denominator": total_n, "rate": term_rate},
        "conditional_retention": {"numerator": a, "denominator": sum_i, "rate": cond_ret},
        "forgetting_among_acquired": {"numerator": b, "denominator": sum_i, "rate": forget_rate},
        "delayed_success_fraction": {"numerator": c, "denominator": total_n, "rate": (c / float(total_n)) if total_n > 0 else 0.0}
    }


def evaluate_paraphrase_records(
    paraphrase_records: List[EvaluationRecord]
) -> Dict[str, Any]:
    """
    Evaluates held-out paraphrase accuracy deriving denominator strictly from record count.
    Never infers denominator from variable names or literal constants.
    """
    denom = len(paraphrase_records)
    if denom == 0:
        return {"numerator": 0, "denominator": 0, "rate": 0.0}

    num = sum(1 for r in paraphrase_records if r.is_match)
    return {
        "numerator": num,
        "denominator": denom,
        "rate": num / float(denom)
    }


def compute_exact_paired_sign_flip_test(
    paired_differences: List[float],
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Computes exact two-sided sign-flip permutation test for the mean paired difference.
    Enumerates all 2^N sign configurations.

    Null Hypothesis (H0):
      The paired differences D_s = T_{C3, s} - T_{R, s} are exchangeable in sign
      (symmetric distribution around 0 under H0).

    Test Statistic:
      T(z) = | (1/N) * sum_{s=1}^N z_s * D_s |
    """
    n = len(paired_differences)
    if n == 0:
        raise ValueError("Cannot perform permutation test on empty paired differences list")

    obs_mean = sum(paired_differences) / float(n)
    obs_t = abs(obs_mean)

    total_perms = 1 << n
    count_extreme = 0

    for mask in range(total_perms):
        perm_sum = 0.0
        for i in range(n):
            sign = 1.0 if ((mask >> i) & 1) == 0 else -1.0
            perm_sum += sign * paired_differences[i]
        perm_t = abs(perm_sum / float(n))
        if perm_t >= (obs_t - eps):
            count_extreme += 1

    p_value = count_extreme / float(total_perms)
    p_value = min(max(p_value, 0.0), 1.0)

    # Parametric Student-t interval as explicitly assumption-dependent supplementary check
    if n >= 2:
        variance = sum((d - obs_mean) ** 2 for d in paired_differences) / float(n - 1)
        se = math.sqrt(variance / float(n))
        t_crit_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.228}
        t_crit = t_crit_table.get(n, 1.960)
        t_interval = [obs_mean - t_crit * se, obs_mean + t_crit * se]
    else:
        se = 0.0
        t_interval = [obs_mean, obs_mean]

    return {
        "n_pairs": n,
        "observed_mean_diff": obs_mean,
        "total_permutations": total_perms,
        "extreme_count": count_extreme,
        "exact_p_value": p_value,
        "resolution_floor": 2.0 / float(total_perms),
        "supplementary_parametric_t_se": se,
        "supplementary_parametric_t_ci95": t_interval,
        "assumption": "Symmetric sign-exchangeability around zero under H0"
    }


def audit_tensor_basis_orthonormality(
    basis: torch.Tensor,
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """
    Verifies orthonormality residual of matrix basis columns: ||Q^T Q - I||_F <= tolerance.
    Handles empty and non-orthonormal shapes.
    """
    if basis.dim() != 2:
        raise ValueError(f"Basis must be a 2D tensor, got shape {basis.shape}")

    m_rows, n_cols = basis.shape
    if n_cols > m_rows:
        return {
            "is_orthonormal": False,
            "rows": m_rows,
            "cols": n_cols,
            "residual_frob": float("inf"),
            "diagnosis": "DEFECT: More columns than rows in ambient space (cannot be orthonormal)"
        }

    q_64 = basis.to(dtype=torch.float64)
    gram = torch.matmul(q_64.t(), q_64)
    ident = torch.eye(n_cols, dtype=torch.float64, device=basis.device)
    residual = torch.linalg.norm(gram - ident, ord="fro").item()

    is_ortho = (residual <= tolerance)
    return {
        "is_orthonormal": is_ortho,
        "rows": m_rows,
        "cols": n_cols,
        "residual_frob": residual,
        "tolerance": tolerance,
        "diagnosis": "PASSED" if is_ortho else f"DEFECT: Frobenius residual {residual:.2e} > {tolerance:.2e}"
    }


def audit_memory_breakdown(
    v_null: torch.Tensor,
    q_prompt: Optional[torch.Tensor],
    stored_keys: List[torch.Tensor],
    stored_vstars: List[torch.Tensor],
    num_edits: int,
    bytes_per_metadata_edit: int = 1500
) -> Dict[str, Any]:
    """
    Audits persistent memory breakdown distinguishing fixed cache, dynamic tracker,
    and metadata storage, reporting both MiB (1024^2) and MB (10^6).
    """
    # 1. Fixed Cache
    v_null_bytes = v_null.element_size() * v_null.nelement()

    # 2. Dynamic Method State
    q_prompt_bytes = (q_prompt.element_size() * q_prompt.nelement()) if (q_prompt is not None) else 0
    keys_bytes = sum(k.element_size() * k.nelement() for k in stored_keys)
    vstars_bytes = sum(v.element_size() * v.nelement() for v in stored_vstars)

    # 3. Rank-1 Factors & Metadata
    # u_f32 (768), r_f32 (3072) -> 3840 float32 = 15,360 bytes/edit
    factors_bytes = num_edits * (768 + 3072) * 4
    metadata_bytes = num_edits * bytes_per_metadata_edit

    total_dynamic_bytes = q_prompt_bytes + keys_bytes + vstars_bytes + factors_bytes + metadata_bytes
    total_state_bytes = v_null_bytes + total_dynamic_bytes

    return {
        "v_null": {
            "shape": list(v_null.shape),
            "dtype": str(v_null.dtype),
            "bytes": v_null_bytes,
            "mib": v_null_bytes / (1024.0 ** 2),
            "mb": v_null_bytes / 1e6
        },
        "q_prompt": {
            "shape": list(q_prompt.shape) if q_prompt is not None else None,
            "dtype": str(q_prompt.dtype) if q_prompt is not None else None,
            "bytes": q_prompt_bytes,
            "mib": q_prompt_bytes / (1024.0 ** 2),
            "mb": q_prompt_bytes / 1e6
        },
        "distinct_subject_slots": len(stored_keys),
        "keys_and_vstars_bytes": keys_bytes + vstars_bytes,
        "factors_and_metadata_bytes": factors_bytes + metadata_bytes,
        "total_dynamic_bytes": total_dynamic_bytes,
        "total_dynamic_mib": total_dynamic_bytes / (1024.0 ** 2),
        "total_dynamic_mb": total_dynamic_bytes / 1e6,
        "total_state_bytes": total_state_bytes,
        "total_state_mib": total_state_bytes / (1024.0 ** 2),
        "total_state_mb": total_state_bytes / 1e6
    }
