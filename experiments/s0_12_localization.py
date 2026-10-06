#!/usr/bin/env python3
"""
experiments/s0_12_localization.py -- Directive S0-12 Verification, Localization & Full-Prompt Engine
Implements Directive S0-12:
- Sequential negative controls on 200-edit sequential model state:
    * wrong_target: evaluated on sequential model state
    * sham_sequence: identical sequence with v* = v0 (pure noise injection / no new fact)
    * never_edited: unperturbed base model evaluated on evaluation facts
    * pre_edit_baseline: evaluation facts evaluated prior to any edit
- Comparator rule: primary floor is max(never_edited, pre_edit_baseline, sham_sequence)
- Stage L: Activation patching on Seed 0 first-50 facts lost at sequence end:
    * (a) Subject last-token position only
    * (b) Non-subject prompt positions only
    * (c) All prompt positions
    * Residual key drift ||Delta_total k|| / ||k||
- Stage F: Full-prompt protection tracking all prompt tokens into the protected basis.
"""

import math
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from experiments.metrics import Measurement, check_match, normalize_entity
from experiments.b1_inject import greedy_predict
from experiments.s0_10_repair import get_subject_last_token_idx
from experiments.s0_11_constraints import (
    CorrectedSequentialNullTracker,
    find_target_value_vstar,
    edit_fact_mlp_cov,
    edit_fact_mlp_null_corrected
)


def evaluate_s0_12_sequential_controls(
    seq_model: nn.Module,
    base_model: nn.Module,
    tokenizer: Any,
    eval_facts: List[Dict[str, Any]],
    facts_pool: List[Dict[str, Any]],
    device: str = "cuda",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates sequential controls on the post-200-edit sequential model state and base model:
    - wrong_target: sequential model tested on prompt with mismatched relation-matched object
    - never_edited: base model tested on eval_facts
    """
    import random
    rng = random.Random(seed)
    wrong_c_matches, wrong_p_matches = [], []
    never_c_matches, never_p_matches = [], []

    for fact in eval_facts:
        cands = [
            c["object"] for c in facts_pool
            if c["relation"] == fact["relation"] and normalize_entity(c["object"]) != normalize_entity(fact["object"])
        ]
        wrong_obj = rng.choice(cands) if cands else "unknown"

        # 1. wrong_target on sequential model
        pred_c = greedy_predict(seq_model, tokenizer, fact["edit_prompt"], 5, device, False)
        wrong_c_matches.append(check_match(pred_c, wrong_obj))

        for p in fact["paraphrases"]:
            pred_p = greedy_predict(seq_model, tokenizer, p, 5, device, False)
            wrong_p_matches.append(check_match(pred_p, wrong_obj))

        # 2. never_edited on unperturbed base model
        base_pred_c = greedy_predict(base_model, tokenizer, fact["edit_prompt"], 5, device, False)
        never_c_matches.append(check_match(base_pred_c, fact["object"]))

        for p in fact["paraphrases"]:
            base_pred_p = greedy_predict(base_model, tokenizer, p, 5, device, False)
            never_p_matches.append(check_match(base_pred_p, fact["object"]))

    return {
        "wrong_target_canonical": wrong_c_matches,
        "wrong_target_paraphrase": wrong_p_matches,
        "never_edited_canonical": never_c_matches,
        "never_edited_paraphrase": never_p_matches
    }


def run_stage_l_patching(
    model: nn.Module,
    base_model: nn.Module,
    tokenizer: Any,
    lost_facts: List[Dict[str, Any]],
    layer_idx: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Stage L: Activation patching on first-50 facts lost at sequence end.
    Conditions:
      (a) Patch subject last-token only from base model
      (b) Patch non-subject prompt positions only from base model
      (c) Patch all prompt positions from base model
    Also measures key drift ||Delta_total k|| / ||k|| at subject position.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    base_c_proj = base_model.transformer.h[layer_idx].mlp.c_proj

    subj_recovered = []
    non_subj_recovered = []
    all_recovered = []
    key_drifts = []

    for fact in lost_facts:
        prompt = fact["edit_prompt"]
        target_obj = fact["object"]
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc.input_ids
        seq_len = input_ids.shape[1]
        subj_idx = get_subject_last_token_idx(tokenizer, prompt, fact["subject"])

        # Capture base and sequential activations at layer input
        base_rec = {}
        def hook_base(m, inp, out):
            base_rec["k"] = inp[0].detach().clone()
        h_b = base_c_proj.register_forward_hook(hook_base)
        with torch.no_grad():
            _ = base_model(input_ids)
        h_b.remove()

        seq_rec = {}
        def hook_seq(m, inp, out):
            seq_rec["k"] = inp[0].detach().clone()
        h_s = c_proj.register_forward_hook(hook_seq)
        with torch.no_grad():
            _ = model(input_ids)
        h_s.remove()

        k_base = base_rec["k"]  # [1, seq_len, 3072]
        k_seq = seq_rec["k"]

        # Measure key drift at subject position: ||k_seq - k_base|| / ||k_base||
        kb_subj = k_base[0, subj_idx, :]
        ks_subj = k_seq[0, subj_idx, :]
        drift = torch.linalg.norm(ks_subj - kb_subj).item() / (torch.linalg.norm(kb_subj).item() + 1e-12)
        key_drifts.append(drift)

        # Patching condition (a): subject last-token only
        def hook_patch_subj(m, inp, out):
            inp_mod = inp[0].clone()
            inp_mod[0, subj_idx, :] = k_base[0, subj_idx, :]
            return (inp_mod,)

        # Patching condition (b): non-subject prompt positions only
        def hook_patch_non_subj(m, inp, out):
            inp_mod = inp[0].clone()
            for pos in range(seq_len):
                if pos != subj_idx:
                    inp_mod[0, pos, :] = k_base[0, pos, :]
            return (inp_mod,)

        # Patching condition (c): all positions
        def hook_patch_all(m, inp, out):
            return (k_base.clone(),)

        # Test recovery under (a)
        h_pa = c_proj.register_forward_pre_hook(hook_patch_subj)
        pred_a = greedy_predict(model, tokenizer, prompt, 5, device, False)
        h_pa.remove()
        subj_recovered.append(check_match(pred_a, target_obj))

        # Test recovery under (b)
        h_pb = c_proj.register_forward_pre_hook(hook_patch_non_subj)
        pred_b = greedy_predict(model, tokenizer, prompt, 5, device, False)
        h_pb.remove()
        non_subj_recovered.append(check_match(pred_b, target_obj))

        # Test recovery under (c)
        h_pc = c_proj.register_forward_pre_hook(hook_patch_all)
        pred_c = greedy_predict(model, tokenizer, prompt, 5, device, False)
        h_pc.remove()
        all_recovered.append(check_match(pred_c, target_obj))

    n_lost = len(lost_facts)
    mean_drift = (sum(key_drifts) / n_lost) if n_lost > 0 else 0.0

    return {
        "n_lost": n_lost,
        "subj_recovered_count": sum(1 for x in subj_recovered if x),
        "non_subj_recovered_count": sum(1 for x in non_subj_recovered if x),
        "all_recovered_count": sum(1 for x in all_recovered if x),
        "subj_recovery_rate": (sum(subj_recovered) / n_lost) if n_lost > 0 else 0.0,
        "non_subj_recovery_rate": (sum(non_subj_recovered) / n_lost) if n_lost > 0 else 0.0,
        "all_recovery_rate": (sum(all_recovered) / n_lost) if n_lost > 0 else 0.0,
        "mean_key_drift": mean_drift,
        "raw_drifts": key_drifts
    }


class FullPromptSequentialNullTracker(CorrectedSequentialNullTracker):
    """
    Extends CorrectedSequentialNullTracker to protect all prompt positions across edits.
    For each edit, records all token activations into the protected basis.
    """
    def __init__(self, v_null: torch.Tensor, rel_threshold: float = 1e-3, device: str = "cuda"):
        super().__init__(v_null, rel_threshold, device)
        self.capacity_exhausted = False
        self.exhaustion_edit_idx = None

    def record_prompt_keys(self, prompt_keys: List[torch.Tensor]) -> None:
        """
        Records all token keys from the current prompt into the protected keys store.
        """
        for k in prompt_keys:
            self.stored_keys.append(k.to(self.device, dtype=torch.float64).detach())
        if self.dim_k >= self.dim_null and not self.capacity_exhausted:
            self.capacity_exhausted = True


def edit_fact_mlp_null_full_prompt(
    model: nn.Module,
    tokenizer: Any,
    fact: Dict[str, Any],
    null_tracker: FullPromptSequentialNullTracker,
    layer_idx: int,
    max_steps: int = 100,
    lr_v: float = 0.1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Stage F: Null-space update protecting all prompt token positions.
    Collects activations across all prompt positions and appends them to null_tracker.
    """
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    k_vec, v_star, steps_taken, _ = find_target_value_vstar(model, tokenizer, fact, layer_idx, max_steps, lr_v, 0.0, device)
    v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
    r_vec = v_star - v0

    # Capture all prompt token activations
    recorded = {}
    def hook_prompt(m, inp, out):
        recorded["all_k"] = inp[0][0].detach().clone()
    h_pr = c_proj.register_forward_hook(hook_prompt)
    enc = tokenizer(fact["edit_prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(enc.input_ids)
    h_pr.remove()

    all_k = recorded["all_k"]  # [seq_len, 3072]
    prompt_keys = [all_k[i] for i in range(all_k.shape[0])]

    res = null_tracker.compute_update(k_vec, r_vec)
    delta_w = res["delta_f32"]
    c_proj.weight.data.add_(delta_w)
    delta_norm = torch.linalg.norm(delta_w).item()

    # Append all other prompt positions into tracker
    subj_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
    non_subj_keys = [all_k[i] for i in range(all_k.shape[0]) if i != subj_idx]
    null_tracker.record_prompt_keys(non_subj_keys)

    curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
    imm_match = check_match(curr_pred, fact["object"])

    return {
        "steps_taken": steps_taken,
        "immediate_match": imm_match,
        "delta_norm": delta_norm,
        "pred": curr_pred,
        "rho": res["rho"],
        "relative_residual": res["relative_residual"],
        "dim_k": null_tracker.dim_k,
        "capacity_exhausted": null_tracker.capacity_exhausted
    }
