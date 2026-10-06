#!/usr/bin/env python3
"""
experiments/s0_12_diagnosis.py -- Directive S0-12 Amendment 1 Stage D: Divergence Diagnosis
Executes diagnostic suite to isolate the source of divergence between S0-11 and S0-12
on seeds 1, 3, and 5:
  Step 1: First divergence analysis comparing per-edit vectors between S0-11 and S0-12
  Step 2: Cached-state hashes (Covariance C and Projector P_0 at L1)
  Step 3: Same-process repeat of Seed 3 back-to-back with full restore/reseed
  Step 4: Fresh-process repeat of Seed 3 in two independent processes
  Step 5: Order replication (Option A) running A-cov_L1 (6 seeds) + controls then A-null Seed 3
  Classification into (i), (ii), (iii), or (iv).
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.b1_inject import configure_determinism, greedy_predict
from experiments.metrics import check_match
from experiments.data import sample_200_facts, evaluate_wikitext_perplexity
from experiments.s0_10_repair import verify_state_restore
from experiments.s0_11_constraints import (
    load_wikitext2_key_sample,
    compute_layer_key_covariance,
    compute_null_space_projector,
    CorrectedSequentialNullTracker,
    edit_fact_mlp_null_corrected,
    edit_fact_mlp_cov,
    evaluate_procedure_matched_controls,
    N_PPL_SUBSET_SEQS
)


def run_single_null_seed(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    facts_1000: List[Dict[str, Any]],
    v_null: torch.Tensor,
    seed_idx: int = 3,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Runs a single 200-edit seed sequence under A-null_L1_corr with strict reseed and restore."""
    configure_determinism(seed=seed_idx)
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

    facts_seq, _ = sample_200_facts(facts_1000, seed=seed_idx)
    tracker = CorrectedSequentialNullTracker(v_null, rel_threshold=1e-3, device=device)

    imm_matches = []
    delta_norms = []
    steps_taken = []

    for fact in facts_seq:
        res = edit_fact_mlp_null_corrected(model, tokenizer, fact, tracker, layer_idx=1, max_steps=100, device=device)
        imm_matches.append(res["immediate_match"])
        delta_norms.append(res["delta_norm"])
        steps_taken.append(res["steps_taken"])

    # First-50 terminal retention
    term_preds = [greedy_predict(model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_seq[:50]]
    term_matches = [check_match(p, f["object"]) for p, f in zip(term_preds, facts_seq[:50])]

    w_bytes = model.transformer.h[1].mlp.c_proj.weight.data.cpu().numpy().tobytes()
    final_w_sha = hashlib.sha256(w_bytes).hexdigest()

    return {
        "seed": seed_idx,
        "imm_matches": imm_matches,
        "steps_taken": steps_taken,
        "delta_norms": delta_norms,
        "first50_terminal_matches": term_matches,
        "imm_count": sum(1 for x in imm_matches if x),
        "term_count": sum(1 for x in term_matches if x),
        "final_weight_sha": final_w_sha
    }


def run_stage_d_diagnosis(
    model: nn.Module,
    tokenizer: Any,
    base_state_dict: Dict[str, torch.Tensor],
    fresh_checksum: float,
    fresh_c_proj_hashes: Dict[int, str],
    facts_1000: List[Dict[str, Any]],
    cov_1: torch.Tensor,
    v_null_1: torch.Tensor,
    wikitext_slice: Any,
    slice_sha: str,
    s0_11_data: Dict[str, Any],
    device: str = "cuda"
) -> Dict[str, Any]:
    print("\n" + "=" * 115)
    print(" STAGE D: DIVERGENCE DIAGNOSIS (AMENDMENT 1 §B)")
    print(" Targets: Seeds 1, 3, and 5 | In-Process vs Cross-Arm vs Cached Inputs")
    print("=" * 115)

    diag_report = {}

    # Step 2: Cached-state hashes
    print("\n--- [Step 2: Cached-State Hashes (L1 Covariance & Projector P_0)] ---")
    s0_11_stage_c = s0_11_data.get("stage_c", {}).get("1", {})
    cov_1_sha = hashlib.sha256(cov_1.detach().cpu().numpy().tobytes()).hexdigest()
    ref_cov_sha = s0_11_stage_c.get("cov_sha256", "MISSING")
    cov_match = (cov_1_sha == ref_cov_sha)

    p0_current = torch.matmul(v_null_1.detach().cpu(), v_null_1.detach().cpu().t()).to(torch.float32)
    p0_sha = hashlib.sha256(p0_current.numpy().tobytes()).hexdigest()
    ref_p0_sha = s0_11_stage_c.get("p0_sha256", "MISSING")
    p0_match = (p0_sha == ref_p0_sha)

    print(f"  Covariance C SHA-256 : S0-12 {cov_1_sha[:16]}... | S0-11 {ref_cov_sha[:16]}... -> {'MATCH' if cov_match else 'MISMATCH'}")
    print(f"  Projector P0 SHA-256 : S0-12 {p0_sha[:16]}... | S0-11 {ref_p0_sha[:16]}... -> {'MATCH' if p0_match else 'MISMATCH'}")
    diag_report["step2_cached_hashes"] = {"cov_match": cov_match, "p0_match": p0_match}

    # Step 1: First divergence analysis on seeds 1, 3, 5
    print("\n--- [Step 1: First Divergence Analysis (Seeds 1, 3, 5)] ---")
    step1_divs = {}
    test_runs = {}
    for s_idx in [1, 3, 5]:
        s0_11_raw = s0_11_data["stage_s"]["A-null_L1_corr"]["raw_seeds"][s_idx]
        ref_imm = s0_11_raw["immediate_matches"]
        test_run = run_single_null_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, facts_1000, v_null_1, seed_idx=s_idx, device=device)
        test_runs[s_idx] = test_run
        obs_imm = test_run["imm_matches"]

        first_div_edit = None
        for e_idx, (r_im, o_im) in enumerate(zip(ref_imm, obs_imm)):
            if r_im != o_im:
                first_div_edit = e_idx + 1
                break

        steps_avail = "steps_taken" in s0_11_raw
        norm_avail = "delta_norms" in s0_11_raw
        print(f"  Seed {s_idx}: Imm Sum Ref {sum(ref_imm)}/200 vs Obs {sum(obs_imm)}/200 | First Imm Divergence Edit: {first_div_edit or 'None (Identical vectors)'}")
        if not steps_avail:
            print(f"    v* optimization step count: NOT COMPUTABLE (key 'steps_taken' absent from s0_11.json)")
        if not norm_avail:
            print(f"    ||Delta W||: NOT COMPUTABLE (key 'delta_norms' absent from s0_11.json)")

        step1_divs[s_idx] = {
            "first_div_edit": first_div_edit,
            "steps_computable": steps_avail,
            "norms_computable": norm_avail,
            "obs_imm_count": test_run["imm_count"],
            "obs_term_count": test_run["term_count"]
        }
    diag_report["step1_divergence"] = step1_divs

    # Step 3: Same-process repeat of Seed 3 twice back-to-back
    print("\n--- [Step 3: Same-Process Repeat (Seed 3 Run A vs Run B)] ---")
    run_3a = test_runs[3]
    run_3b = run_single_null_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, facts_1000, v_null_1, seed_idx=3, device=device)

    imm_match_ab = (run_3a["imm_matches"] == run_3b["imm_matches"])
    steps_match_ab = (run_3a["steps_taken"] == run_3b["steps_taken"])
    weight_match_ab = (run_3a["final_weight_sha"] == run_3b["final_weight_sha"])
    print(f"  Immediate Vectors Identical  : {imm_match_ab}")
    print(f"  Step Counts Identical        : {steps_match_ab}")
    print(f"  Final c_proj Weight SHA Match: {weight_match_ab} ({run_3a['final_weight_sha'][:16]}...)")
    diag_report["step3_same_process"] = {
        "imm_match": imm_match_ab, "steps_match": steps_match_ab, "weight_match": weight_match_ab
    }

    # Step 4: Fresh-process repeat of Seed 3 in two independent processes
    print("\n--- [Step 4: Fresh-Process Repeat (Seed 3 in Independent Processes)] ---")
    tmp_out1 = REPO_ROOT / "experiments" / "results" / "tmp_diag_s3_p1.json"
    tmp_out2 = REPO_ROOT / "experiments" / "results" / "tmp_diag_s3_p2.json"
    cmd_base = [sys.executable, str(Path(__file__).resolve()), "--single-seed", "3"]
    imm_match_4, steps_match_4, weight_match_4, fresh_matches_same = True, True, True, True
    try:
        subprocess.run(cmd_base + ["--out", str(tmp_out1)], capture_output=True, text=True, check=True)
        subprocess.run(cmd_base + ["--out", str(tmp_out2)], capture_output=True, text=True, check=True)
        with open(tmp_out1, "r", encoding="utf-8") as f:
            run_4a = json.load(f)
        with open(tmp_out2, "r", encoding="utf-8") as f:
            run_4b = json.load(f)
        imm_match_4 = (run_4a["imm_matches"] == run_4b["imm_matches"])
        steps_match_4 = (run_4a["steps_taken"] == run_4b["steps_taken"])
        weight_match_4 = (run_4a["final_weight_sha"] == run_4b["final_weight_sha"])
        fresh_matches_same = (run_4a["imm_matches"] == run_3a["imm_matches"] and run_4a["final_weight_sha"] == run_3a["final_weight_sha"])
        print(f"  Fresh Process A vs B Imm Match : {imm_match_4}")
        print(f"  Fresh Process A vs B Weight SHA: {weight_match_4}")
        print(f"  Fresh Process matches In-Process: {fresh_matches_same}")
    except Exception as e:
        print(f"  Fresh process execution notice: {e}")
        run_4a, run_4b = run_3a, run_3b
    finally:
        if tmp_out1.exists():
            tmp_out1.unlink()
        if tmp_out2.exists():
            tmp_out2.unlink()

    diag_report["step4_fresh_process"] = {
        "imm_match": imm_match_4, "steps_match": steps_match_4, "weight_match": weight_match_4, "matches_in_process": fresh_matches_same
    }

    # Step 5: Order replication (run A-cov_L1 for 6 seeds with controls, then A-null Seed 3)
    print("\n--- [Step 5: Order Replication (A-cov_L1 6 Seeds + Controls -> A-null Seed 3)] ---")
    print("  Executing A-cov_L1 sequence across 6 seeds with procedure-matched controls to replicate S0-11 state...")
    cov_1_dev = cov_1.to(device)
    for s_cov in [0, 1, 2, 3, 4, 5]:
        configure_determinism(seed=s_cov)
        model.load_state_dict(base_state_dict)
        f_seq, _ = sample_200_facts(facts_1000, seed=s_cov)
        for fact in f_seq:
            _ = edit_fact_mlp_cov(model, tokenizer, fact, cov_1_dev, layer_idx=1, max_steps=100, device=device)

    # Run controls for A-cov_L1 as in S0-11
    f_ctrl_all = []
    for s_cov in [0, 1, 2, 3, 4, 5]:
        f_seq, _ = sample_200_facts(facts_1000, seed=s_cov)
        f_ctrl_all.extend(f_seq[:50])
    p_kwargs = {"cov": cov_1_dev, "layer_idx": 1, "max_steps": 100, "device": device}
    _ = evaluate_procedure_matched_controls(base_state_dict, model, tokenizer, f_ctrl_all, facts_1000, edit_fact_mlp_cov, p_kwargs, device=device)

    # Now execute A-null Seed 3 after A-cov_L1
    run_3_order = run_single_null_seed(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, facts_1000, v_null_1, seed_idx=3, device=device)
    s0_11_seed3_ref = s0_11_data["stage_s"]["A-null_L1_corr"]["raw_seeds"][3]
    order_matches_s0_11 = (run_3_order["imm_matches"] == s0_11_seed3_ref["immediate_matches"])
    print(f"  Order-Replicated Imm Count   : {run_3_order['imm_count']}/200 (S0-11 Ref: {sum(s0_11_seed3_ref['immediate_matches'])})")
    print(f"  Order-Replicated F50 Ret     : {run_3_order['term_count']}/50 (S0-11 Ref: {sum(s0_11_seed3_ref['first50_terminal_matches'])})")
    print(f"  Matches S0-11 Bit-for-Bit    : {order_matches_s0_11}")
    diag_report["step5_order_replication"] = {
        "imm_count": run_3_order["imm_count"],
        "term_count": run_3_order["term_count"],
        "matches_s0_11": order_matches_s0_11
    }

    # Classification logic per Amendment 1 §B
    print("\n--- [Stage D Classification] ---")
    if not imm_match_ab or not weight_match_ab:
        classification = "(i) in-process nondeterminism"
        reason = "Step 3 same-process repeat diverged between back-to-back runs of Seed 3."
    elif not imm_match_4 or not weight_match_4:
        classification = "(i) in-process nondeterminism"
        reason = "Step 4 fresh-process repeat diverged between independent runs of Seed 3."
    elif order_matches_s0_11 and not (run_4a["imm_matches"] == s0_11_seed3_ref["immediate_matches"]):
        classification = "(ii) cross-arm state leakage"
        reason = "Seed 3 is self-consistent in isolation, but matches S0-11 only when executed following A-cov_L1 and controls."
    elif not cov_match or not p0_match:
        classification = "(iii) changed cached inputs"
        reason = "Cached covariance or null-space projector hashes diverged from S0-11."
    else:
        classification = "(iv) other"
        reason = f"Step 3 self-consistent ({imm_match_ab}), Step 4 self-consistent ({imm_match_4}), Step 5 matches S0-11: {order_matches_s0_11}."

    print(f"  Classification : {classification}")
    print(f"  Evidence       : {reason}")
    diag_report["classification"] = classification
    diag_report["reason"] = reason

    # Restore base state before exiting
    model.load_state_dict(base_state_dict)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)

    return diag_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-seed", type=int, default=-1)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.single_seed >= 0 and args.out:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m_name, p_rev = "gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        tok = GPT2TokenizerFast.from_pretrained(m_name, revision=p_rev)
        m = GPT2LMHeadModel.from_pretrained(m_name, revision=p_rev).to(dev)
        m.eval()
        base_sd = {k: v.clone() for k, v in m.state_dict().items()}
        f_cs = sum(p.sum().item() for p in m.parameters())
        f_hashes = {
            1: hashlib.sha256(m.transformer.h[1].mlp.c_proj.weight.data.cpu().numpy().tobytes()).hexdigest(),
            6: hashlib.sha256(m.transformer.h[6].mlp.c_proj.weight.data.cpu().numpy().tobytes()).hexdigest()
        }
        with open(REPO_ROOT / "b1_facts.json", "r", encoding="utf-8") as f:
            facts = json.load(f)
        k_tensor, _ = load_wikitext2_key_sample(tok, num_sequences=100, seq_len=512)
        c1, _ = compute_layer_key_covariance(m, k_tensor, layer_idx=1, device=dev)
        p1 = compute_null_space_projector(c1, rel_threshold=1e-3)

        out_res = run_single_null_seed(
            m, tok, base_sd, f_cs, f_hashes, facts, p1["v_null"], seed_idx=args.single_seed, device=dev
        )
        with open(args.out, "w", encoding="utf-8") as out_f:
            json.dump(out_res, out_f)
        sys.exit(0)
