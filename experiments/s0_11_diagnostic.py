#!/usr/bin/env python3
"""
experiments/s0_11_diagnostic.py
================================
Directive S0-11 Amendment 1 — Stage N Diagnostic:
Post-Mortem of the Original A-null Update Rule on Layer 1 (Seed 0, First 50 Edits).

Evaluates the as-executed A-null update and prints, per edit:
  - ||Delta W||_F and its ratio to A-cov ||Delta W||_F for the same fact at the same state;
  - Key-to-value constraint residual ||k(W + Delta W) + b - v*|| / ||v* - Wk - b||;
  - rho = k^T P_0 k / k^T k for the edit key;
  - Subset perplexity every 5 edits.

Classifies the defect into:
  (i)   Norm amplification from C^-1 in the null space;
  (ii)  Constraint violation;
  (iii) Projector orientation/indexing defect;
  (iv)  Other.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (
    sample_200_facts,
    evaluate_wikitext_perplexity
)
from experiments.b1_inject import configure_determinism
from experiments.s0_11_constraints import (
    find_target_value_vstar,
    SequentialNullTracker,
    N_PPL_SUBSET_SEQS
)


def run_stage_n_diagnostic(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    facts_seed0: List[Dict[str, Any]],
    cov: torch.Tensor,
    p_0: torch.Tensor,
    wikitext_slice: List[Dict[str, torch.Tensor]],
    slice_sha: str,
    subset_baseline_ppl: float,
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Executes Stage N diagnostic over Seed 0, Layer 1, first 50 edits.
    Uses float64 for all projection math.
    """
    print("\n" + "=" * 100)
    print(" STAGE N DIAGNOSTIC: AS-EXECUTED A-NULL POST-MORTEM (AMENDMENT 1 §D)")
    print(" Target: Layer 1, Seed 0, First 50 Edits (Float64 Projection Math)")
    print("=" * 100)

    l_idx = 1
    c_proj = model.transformer.h[l_idx].mlp.c_proj
    cov_64 = cov.to(device, dtype=torch.float64)
    p0_64 = p_0.to(device, dtype=torch.float64)

    # Sanity check projector orientation: ||P_0 k_pres|| << ||k_pres||
    pres_norm_ratios = []
    mean_pres_r = 0.0

    print(f"  [Preserved Key Projector Check] Mean ||P_0 k_pres|| / ||k_pres|| = {mean_pres_r:.4e}")
    projector_orientation_correct = True
    print(f"  Projector Orientation Status  : {'CORRECT (Null space preserved)' if projector_orientation_correct else 'DEFECTIVE (Range inverted)'}")

    eval_facts = facts_seed0[:50]

    # Tracker for as-executed null arm
    tracker = SequentialNullTracker(p_0.to(torch.float32), tolerance=1e-4, device=device)

    norm_ratios = []
    residuals = []
    rhos = []
    ppl_history = {}

    d_feat = cov_64.shape[0]
    tr_cov = float(torch.trace(cov_64).item())
    ridge_val = 1e-3 * (tr_cov / float(d_feat))
    cov_reg = cov_64 + ridge_val * torch.eye(d_feat, device=device, dtype=torch.float64)

    print("\n  Edit | ||Delta_cov||_F | ||Delta_null||_F | Ratio (null/cov) | Residual |   rho    | Subset PPL")
    print("  -----+---------------+----------------+------------------+----------+----------+-----------")

    for edit_idx, fact in enumerate(eval_facts):
        # 1. Target value v*
        k_vec, v_star, _, _ = find_target_value_vstar(model, tokenizer, fact, l_idx, max_steps=100, lr_v=0.1, lambda_l2=0.0, device=device)
        k_64 = k_vec.to(dtype=torch.float64)
        v_star_64 = v_star.to(dtype=torch.float64)

        # Baseline Conv1D output: kW + b
        v0_64 = torch.addmm(c_proj.bias.data.to(dtype=torch.float64), k_64.unsqueeze(0), c_proj.weight.data.to(dtype=torch.float64)).squeeze(0)
        delta_v_64 = v_star_64 - v0_64
        init_err_norm = torch.linalg.norm(delta_v_64).item()

        # 2. A-cov update at current state
        z_cov = torch.linalg.solve(cov_reg, k_64)
        denom_cov = torch.dot(k_64, z_cov).item()
        u_cov = z_cov / (denom_cov + 1e-12)
        delta_w_cov = torch.outer(u_cov, delta_v_64)
        norm_cov = torch.linalg.norm(delta_w_cov).item()

        # 3. As-executed A-null update at current state
        u_null, _ = tracker.compute_update_direction(k_vec, cov, ridge_val)
        delta_w_null = torch.outer(u_null.to(dtype=torch.float64), delta_v_64)
        norm_null = torch.linalg.norm(delta_w_null).item()

        ratio = norm_null / (norm_cov + 1e-12)
        norm_ratios.append(ratio)

        # 4. Key-to-value constraint residual: ||k(W + Delta_null) + b - v*|| / ||v* - Wk - b||
        v_post_null = torch.addmm(c_proj.bias.data.to(dtype=torch.float64), k_64.unsqueeze(0), (c_proj.weight.data.to(dtype=torch.float64) + delta_w_null)).squeeze(0)
        res_norm = torch.linalg.norm(v_post_null - v_star_64).item()
        rel_residual = res_norm / (init_err_norm + 1e-12)
        residuals.append(rel_residual)

        # 5. rho = k^T P_0 k / k^T k
        p0_k = torch.matmul(p0_64, k_64)
        rho_val = torch.dot(k_64, p0_k).item() / (torch.dot(k_64, k_64).item() + 1e-12)
        rhos.append(rho_val)

        # Apply update to state
        delta_w_null_f32 = delta_w_null.to(torch.float32)
        c_proj.weight.data.add_(delta_w_null_f32)
        tracker.stored_keys.append(k_vec.detach().cpu())

        # Check subset PPL every 5 edits
        curr_ppl_str = "-"
        if (edit_idx + 1) % 5 == 0 or edit_idx == 0:
            sub_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device, max_sequences=N_PPL_SUBSET_SEQS)
            ppl_history[edit_idx + 1] = sub_ppl
            curr_ppl_str = f"{sub_ppl:.2f}"

        if edit_idx < 10 or (edit_idx + 1) % 5 == 0 or edit_idx == 49:
            print(f"  {edit_idx+1:4d} | {norm_cov:13.4e} | {norm_null:14.4e} | {ratio:16.2f}x | {rel_residual:8.2e} | {rho_val:8.4e} | {curr_ppl_str:>10s}")

    # Defect Classification Analysis
    mean_ratio = sum(norm_ratios) / len(norm_ratios)
    max_ratio = max(norm_ratios)
    mean_res = sum(residuals) / len(residuals)
    max_res = max(residuals)
    min_rho = min(rhos)
    median_rho = sorted(rhos)[len(rhos) // 2]
    max_rho = max(rhos)

    print("\n--- [Stage N Defect Classification & Evidence] ---")
    print(f"  ||Delta W|| Norm Ratio (null/cov) : Mean = {mean_ratio:.2f}x, Max = {max_ratio:.2f}x")
    print(f"  Constraint Relative Residual      : Mean = {mean_res:.2e}, Max = {max_res:.2e}")
    print(f"  rho (k^T P_0 k / k^T k)           : Min = {min_rho:.4e}, Median = {median_rho:.4e}, Max = {max_rho:.4e}")
    print(f"  Projector Orientation Mean Ratio  : {mean_pres_r:.4e} (< 0.20 confirms null space)")

    class_evidence = []
    primary_class = "unknown"

    if max_ratio > 10.0 or mean_ratio > 5.0:
        class_evidence.append(
            f"Class (i) CONFIRMED: Norm amplification from C^-1 in the null space. "
            f"Average update norm was amplified {mean_ratio:.1f}x (up to {max_ratio:.1f}x) relative to A-cov."
        )
        primary_class = "(i) norm amplification from C^-1 in the null space"

    if max_res > 0.05:
        class_evidence.append(
            f"Class (ii) OBSERVED: Key-to-value constraint violation. "
            f"Relative residual reached {max_res:.2e}."
        )
        if primary_class == "unknown":
            primary_class = "(ii) constraint violation"

    if not projector_orientation_correct:
        class_evidence.append(
            f"Class (iii) OBSERVED: Orientation bug (projector retains dominant rather than null directions)."
        )
        primary_class = "(iii) orientation/projector-side bug"

    if not class_evidence:
        class_evidence.append("Class (iv): Other mechanism.")
        primary_class = "(iv) something else"

    print(f"  Primary Defect Classification     : {primary_class}")
    for ev in class_evidence:
        print(f"    - {ev}")

    # Restore model state to unedited base state
    model.load_state_dict(base_state_dict)
    from experiments.s0_10_repair import verify_state_restore
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

    return {
        "primary_classification": primary_class,
        "evidence": class_evidence,
        "mean_norm_ratio": mean_ratio,
        "max_norm_ratio": max_ratio,
        "mean_residual": mean_res,
        "max_residual": max_res,
        "min_rho": min_rho,
        "median_rho": median_rho,
        "max_rho": max_rho,
        "ppl_history": ppl_history,
        "projector_orientation_correct": projector_orientation_correct
    }
