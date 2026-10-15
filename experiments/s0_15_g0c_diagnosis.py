#!/usr/bin/env python3
"""
experiments/s0_15_g0c_diagnosis.py -- Gate G0c Diagnostic & Discrepancy Isolation Engine

Directive S0-15 Failure Follow-Up:
Diagnoses the divergence of Gate G0c between historical S0-13/S0-14 environment
and the current runtime environment without modifying Gate G0c or adopting 26/50 as a replacement.

Key Diagnostic Objectives:
1. Repeatability Assessment: Run G0c twice under identical conditions in the current
   environment to determine whether execution is strictly deterministic (Run 1 == Run 2).
2. Divergence Step Localization: Log intermediate weight hashes and parameter update
   norms across edit steps to isolate where divergence first appears.
3. Paired Fact-Level Outcome Transitions: Directly map each of the 50 evaluated facts
   to identify Retained->Retained, Lost->Lost, Retained->Lost, and Lost->Retained transitions.
4. Margin Distribution & Shift Analysis: Contrast historical terminal margins against
   current margins to distinguish small-margin boundary crossings from systematic drift.

Conforms to AGENTS.md:
- Section 2.1-2.4: Zero gate modification.
- Section 3.7: Raw outcome vectors serialized.
- Section 7.1: Strict limit under 600 lines.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import sample_200_facts
from experiments.metrics import check_match
from experiments.s0_13_protection import (
    get_subject_last_token_idx,
    find_target_value_vstar_margin,
    compute_target_margin
)
from experiments.s0_14_audit import (
    load_and_assert_pinned_cache,
    run_g0_bit_reproduction,
    InstrumentedAuditNullTracker
)

HISTORICAL_ARTIFACT = REPO_ROOT / "experiments" / "results" / "s0_14.json"
CACHE_DIR = str(REPO_ROOT / "experiments" / "cache")
FACTS_FILE = REPO_ROOT / "b1_facts.json"
OUTPUT_FILE = REPO_ROOT / "experiments" / "results" / "s0_15_g0c_diagnosis.json"

CHECKPOINT_CHECKSUM = 164860.84375


def configure_determinism(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def compute_param_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def verify_state_restore(
    model: nn.Module,
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str]
) -> None:
    total_sum = sum(p.sum().item() for p in model.parameters())
    assert abs(total_sum - fresh_checksum) < 1e-4, f"Model checksum drift: {total_sum} vs {fresh_checksum}"
    for l_idx, exp_sha in fresh_c_proj_hashes.items():
        curr_sha = compute_param_sha(model.transformer.h[l_idx].mlp.c_proj.weight.data)
        assert curr_sha == exp_sha, f"Layer {l_idx} c_proj hash mismatch on restore: {curr_sha} != {exp_sha}"


def greedy_predict(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 5,
    device: str = "cuda"
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    gen = tokenizer.decode(input_ids[0][enc.input_ids.shape[1]:])
    return gen.strip()


def run_g0c_pass(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    facts: List[Dict[str, Any]],
    v_null: torch.Tensor,
    checkpoint_steps: List[int],
    device: str = "cuda"
) -> Dict[str, Any]:
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

    c_proj = model.transformer.h[1].mlp.c_proj
    w_0 = c_proj.weight.data.clone().cpu()
    b_0 = c_proj.bias.data.clone().cpu()

    tracker = InstrumentedAuditNullTracker(v_null, rel_threshold=1e-3, device=device)

    imm_matches: List[bool] = []
    steps_per_edit: List[int] = []
    write_margins: List[float] = []
    per_edit_k_norm: List[float] = []
    per_edit_r_norm: List[float] = []
    per_edit_delta_frob: List[float] = []
    intermediate_shas: Dict[int, str] = {}

    for idx, fact in enumerate(facts):
        k_vec, v_star, steps, _, _ = find_target_value_vstar_margin(
            model, tokenizer, fact, layer_idx=1, target_margin=2.0, max_steps=100, lr_v=0.1, device=device
        )
        v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
        r_vec = v_star - v0

        res = tracker.compute_update(k_vec, r_vec, idx)
        delta = res["delta_f32"]
        c_proj.weight.data.add_(delta)

        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device=device)
        is_imm = check_match(curr_pred, fact["object"])
        imm_matches.append(is_imm)
        steps_per_edit.append(steps)

        k_norm = float(torch.linalg.norm(k_vec).item())
        r_norm = float(torch.linalg.norm(r_vec).item())
        d_frob = float(torch.linalg.norm(delta).item())
        per_edit_k_norm.append(k_norm)
        per_edit_r_norm.append(r_norm)
        per_edit_delta_frob.append(d_frob)

        wm = compute_target_margin(model, tokenizer, fact["edit_prompt"], fact["object"], device=device)
        write_margins.append(wm)

        if idx in checkpoint_steps:
            step_sha = compute_param_sha(c_proj.weight.data)
            intermediate_shas[idx] = step_sha

    # Evaluate terminal retention on first 50 facts
    c_ret: List[bool] = []
    term_margins: List[float] = []
    for f in facts[:50]:
        pred = greedy_predict(model, tokenizer, f["edit_prompt"], 5, device=device)
        c_ret.append(check_match(pred, f["object"]))
        tm = compute_target_margin(model, tokenizer, f["edit_prompt"], f["object"], device=device)
        term_margins.append(tm)

    final_weight_sha = compute_param_sha(c_proj.weight.data)

    return {
        "final_weight_sha": final_weight_sha,
        "immediate_matches": imm_matches,
        "canonical_retention": c_ret,
        "terminal_margins": term_margins,
        "write_margins": write_margins,
        "steps_per_edit": steps_per_edit,
        "per_edit_k_norm": per_edit_k_norm,
        "per_edit_r_norm": per_edit_r_norm,
        "per_edit_delta_frob": per_edit_delta_frob,
        "intermediate_shas": intermediate_shas,
        "retention_count": sum(1 for x in c_ret if x),
        "immediate_count": sum(1 for x in imm_matches if x)
    }


def analyze_paired_transitions(
    current_ret: List[bool],
    current_margins: List[float],
    hist_ret: List[bool],
    hist_margins: List[float],
    facts_50: List[Dict[str, Any]]
) -> Dict[str, Any]:
    assert len(current_ret) == 50
    assert len(hist_ret) == 50
    assert len(current_margins) == 50
    assert len(hist_margins) == 50

    retained_both: List[int] = []
    lost_both: List[int] = []
    hist_only: List[int] = []  # True in hist, False in current (dropped)
    curr_only: List[int] = []  # False in hist, True in current (gained)

    transition_details: List[Dict[str, Any]] = []

    for i in range(50):
        h_val = hist_ret[i]
        c_val = current_ret[i]
        h_m = hist_margins[i]
        c_m = current_margins[i]
        d_m = c_m - h_m

        if h_val and c_val:
            retained_both.append(i)
        elif (not h_val) and (not c_val):
            lost_both.append(i)
        elif h_val and (not c_val):
            hist_only.append(i)
            transition_details.append({
                "fact_idx": i,
                "transition": "RETAINED_TO_LOST",
                "subject": facts_50[i]["subject"],
                "object": facts_50[i]["object"],
                "prompt": facts_50[i]["edit_prompt"],
                "historical_margin": h_m,
                "current_margin": c_m,
                "margin_delta": d_m
            })
        else:
            curr_only.append(i)
            transition_details.append({
                "fact_idx": i,
                "transition": "LOST_TO_RETAINED",
                "subject": facts_50[i]["subject"],
                "object": facts_50[i]["object"],
                "prompt": facts_50[i]["edit_prompt"],
                "historical_margin": h_m,
                "current_margin": c_m,
                "margin_delta": d_m
            })

    def calc_quantiles(arr: List[float]) -> Dict[str, float]:
        s = sorted(arr)
        n = len(s)
        return {
            "min": s[0],
            "q25": s[int(n * 0.25)],
            "median": s[int(n * 0.50)],
            "q75": s[int(n * 0.75)],
            "max": s[-1],
            "mean": sum(s) / n
        }

    deltas = [c - h for c, h in zip(current_margins, hist_margins)]

    return {
        "counts": {
            "retained_both": len(retained_both),
            "lost_both": len(lost_both),
            "hist_only_dropped": len(hist_only),
            "curr_only_gained": len(curr_only),
            "net_difference": len(curr_only) - len(hist_only)
        },
        "indices": {
            "retained_both": retained_both,
            "lost_both": lost_both,
            "hist_only_dropped": hist_only,
            "curr_only_gained": curr_only
        },
        "transition_details": transition_details,
        "historical_margin_dist": calc_quantiles(hist_margins),
        "current_margin_dist": calc_quantiles(current_margins),
        "margin_delta_dist": calc_quantiles(deltas)
    }


def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(" DIRECTIVE S0-15 FAILURE FOLLOW-UP: GATE G0c DIAGNOSTIC ISOLATION")
    print("=" * 80)

    # 1. Environment Fingerprint
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    workspace_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET")

    print("\n[Environment Fingerprint]")
    print(f"  Python Version  : {py_ver}")
    print(f"  PyTorch Version : {torch_ver}")
    print(f"  CUDA Version    : {cuda_ver}")
    print(f"  Device Name     : {dev_name}")
    print(f"  cuBLAS Config   : {workspace_cfg}")

    # Set deterministic controls
    configure_determinism(seed=0)

    # 2. Model & Facts Loading
    print("\n[Loading Model & Pinned Facts]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()

    base_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    fresh_checksum = sum(p.sum().item() for p in model.parameters())
    fresh_c_proj_hashes = {
        l_idx: compute_param_sha(model.transformer.h[l_idx].mlp.c_proj.weight.data)
        for l_idx in range(len(model.transformer.h))
    }

    assert FACTS_FILE.exists(), f"Facts file not found: {FACTS_FILE}"
    with open(FACTS_FILE, "r", encoding="utf-8") as f:
        facts_pool = json.load(f)
    facts_seed0, _ = sample_200_facts(facts_pool, seed=0)

    # 3. Stage 0 Baseline Checks
    print("\n[Stage 0: Reproduction Controls]")
    g0_res = run_g0_bit_reproduction(model, tokenizer, base_state_dict, facts_seed0, device=device)
    cache_info = load_and_assert_pinned_cache(model, tokenizer, CACHE_DIR, device=device)
    v_null = cache_info["v_null"]

    checkpoint_steps = [0, 1, 2, 3, 4, 9, 19, 49, 99, 149, 199]

    # 4. Diagnostic Run 1: Execution with step logging
    print("\n" + "=" * 80)
    print(" [Step 1: Diagnostic Pass 1 -- Step Instrumentation]")
    print("=" * 80)
    t0_r1 = time.time()
    run1 = run_g0c_pass(
        model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
        facts_seed0, v_null, checkpoint_steps, device=device
    )
    t_r1 = time.time() - t0_r1
    print(f"  Run 1 Immediate Efficacy : {run1['immediate_count']}/200")
    print(f"  Run 1 Retention (E2)     : {run1['retention_count']}/50")
    print(f"  Run 1 Terminal Weight SHA: {run1['final_weight_sha']}")
    print(f"  Run 1 Elapsed Time       : {t_r1:.2f} s")

    # 5. Diagnostic Run 2: Repeatability Verification
    print("\n" + "=" * 80)
    print(" [Step 2: Diagnostic Pass 2 -- Repeatability Verification]")
    print("=" * 80)
    configure_determinism(seed=0)
    t0_r2 = time.time()
    run2 = run_g0c_pass(
        model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
        facts_seed0, v_null, checkpoint_steps, device=device
    )
    t_r2 = time.time() - t0_r2
    print(f"  Run 2 Immediate Efficacy : {run2['immediate_count']}/200")
    print(f"  Run 2 Retention (E2)     : {run2['retention_count']}/50")
    print(f"  Run 2 Terminal Weight SHA: {run2['final_weight_sha']}")
    print(f"  Run 2 Elapsed Time       : {t_r2:.2f} s")

    # Evaluate bitwise determinism under current environment
    sha_match = (run1["final_weight_sha"] == run2["final_weight_sha"])
    ret_match = (run1["canonical_retention"] == run2["canonical_retention"])
    imm_match = (run1["immediate_matches"] == run2["immediate_matches"])
    step_sha_match = (run1["intermediate_shas"] == run2["intermediate_shas"])
    deterministic_in_current_env = sha_match and ret_match and imm_match and step_sha_match

    print("\n[Repeatability Audit]")
    print(f"  Bitwise Weight Match   : {sha_match}")
    print(f"  Outcome Vector Match   : {ret_match}")
    print(f"  Step SHAs Match        : {step_sha_match}")
    print(f"  Determinism in Curr Env: {deterministic_in_current_env}")

    # 6. Historical Comparison
    print("\n" + "=" * 80)
    print(" [Step 3: Comparison Against Pinned Historical S0-14 Baseline]")
    print("=" * 80)
    assert HISTORICAL_ARTIFACT.exists(), f"Missing historical artifact: {HISTORICAL_ARTIFACT}"
    with open(HISTORICAL_ARTIFACT, "r", encoding="utf-8") as f:
        s0_14_data = json.load(f)

    hist_ref = s0_14_data["stage_b"]["r"]["0"]
    hist_ret = hist_ref["canonical_retention"]
    hist_margins = hist_ref["terminal_margins"]
    hist_sha = s0_14_data["gates"]["G0c"]["seed0_weight_sha"]

    print(f"  Historical Retention   : {sum(1 for x in hist_ret if x)}/50 (expected 30)")
    print(f"  Historical Weight SHA  : {hist_sha}")

    paired_analysis = analyze_paired_transitions(
        run1["canonical_retention"],
        run1["terminal_margins"],
        hist_ret,
        hist_margins,
        facts_seed0[:50]
    )

    counts = paired_analysis["counts"]
    print(f"\n  Paired Transitions Breakdown (N=50):")
    print(f"    Retained in Both (T -> T) : {counts['retained_both']}")
    print(f"    Lost in Both (F -> F)     : {counts['lost_both']}")
    print(f"    Historical Only (T -> F)  : {counts['hist_only_dropped']} (Dropped in Current)")
    print(f"    Current Only (F -> T)     : {counts['curr_only_gained']} (Gained in Current)")
    print(f"    Net Difference            : {counts['net_difference']:+d}")

    if paired_analysis["transition_details"]:
        print("\n  Transitioned Facts Line-Item Audit:")
        print(f"  {'Idx':<5} | {'Transition':<18} | {'Hist Marg':<10} | {'Curr Marg':<10} | {'Delta':<10} | {'Subject':<20}")
        print("  " + "-" * 82)
        for td in paired_analysis["transition_details"]:
            idx_s = str(td['fact_idx'])
            tr_s = td['transition']
            hm_s = f"{td['historical_margin']:.4f}"
            cm_s = f"{td['current_margin']:.4f}"
            dm_s = f"{td['margin_delta']:.4f}"
            sb_s = td['subject'][:18]
            print(f"  {idx_s:<5} | {tr_s:<18} | {hm_s:<10} | {cm_s:<10} | {dm_s:<10} | {sb_s:<20}")

    h_dist = paired_analysis["historical_margin_dist"]
    c_dist = paired_analysis["current_margin_dist"]
    d_dist = paired_analysis["margin_delta_dist"]

    print("\n  Terminal Margin Distribution Summary:")
    print(f"    Historical Margins: min={h_dist['min']:.4f}, median={h_dist['median']:.4f}, max={h_dist['max']:.4f}, mean={h_dist['mean']:.4f}")
    print(f"    Current Margins   : min={c_dist['min']:.4f}, median={c_dist['median']:.4f}, max={c_dist['max']:.4f}, mean={c_dist['mean']:.4f}")
    print(f"    Margin Deltas     : min={d_dist['min']:.4f}, median={d_dist['median']:.4f}, max={d_dist['max']:.4f}, mean={d_dist['mean']:.4f}")

    # 7. Serialize Full Diagnostic Results
    total_elapsed = time.time() - t_start
    output_payload = {
        "metadata": {
            "directive": "S0-15_FOLLOW_UP",
            "type": "g0c_divergence_diagnosis",
            "environment": {
                "python": py_ver,
                "pytorch": torch_ver,
                "cuda": cuda_ver,
                "device": dev_name,
                "cublas_workspace": workspace_cfg
            },
            "total_elapsed_s": total_elapsed
        },
        "gates_baseline": {
            "g0": g0_res,
            "g0b": {
                "cov_sha": cache_info["cov_sha"],
                "v_null_sha": cache_info["v_null_sha"]
            }
        },
        "repeatability": {
            "deterministic_in_current_env": deterministic_in_current_env,
            "run1_weight_sha": run1["final_weight_sha"],
            "run2_weight_sha": run2["final_weight_sha"],
            "run1_retention": run1["retention_count"],
            "run2_retention": run2["retention_count"],
            "step_shas_equal": step_sha_match
        },
        "intermediate_checkpoints_run1": run1["intermediate_shas"],
        "historical_reference": {
            "artifact": "s0_14.json",
            "weight_sha": hist_sha,
            "retention_count": sum(1 for x in hist_ret if x)
        },
        "paired_analysis": paired_analysis,
        "raw_outcomes": {
            "run1_canonical_retention": run1["canonical_retention"],
            "run1_terminal_margins": run1["terminal_margins"],
            "historical_canonical_retention": hist_ret,
            "historical_terminal_margins": hist_margins
        }
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[Diagnosis Complete]")
    print(f"  Artifact serialized to: {OUTPUT_FILE}")
    print(f"  Total Session Elapsed: {total_elapsed:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
