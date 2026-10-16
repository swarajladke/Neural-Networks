#!/usr/bin/env python3
"""
experiments/run_s0_15.py -- Master Orchestrator for Directive S0-15 Pilot Confirmation

Directive S0-15: Validate C3 Before Scaling
Executes:
  - Gate G0: Baseline SGD bit-reproduction (669 steps, 200 immediate, 8 terminal).
  - Gate G0b: Pinned cache hash verification (cov_l1.pt, v_null_l1.pt).
  - Gate G0c: Bit-reproduce S0-13 F3 Reference Seed 0 (200 immediate, 30 terminal retention).
  - Preregistered Paired Confirmation Pilot (Seed 0, 200 edits):
    * Arm 1: Validated Reference R (F3 margin-targeted null-space tracker).
    * Arm 2: Validated C3 (SlotConsolidatedTemplateNullTracker).
    * Arm 3: Behavioral Sham Sequence Control (evaluating genuine greedy predictions).
  - Endpoints: E0 (PPL), E1 (immediate), E2 (canonical retention), E3 (paraphrase retention),
    E5 (residual drift), slot count, consolidation count, memory footprint audit, wall-clock.
  - Decision Gates: E0 <= 72.06, E1 >= 90.00%, C3 consolidation > 0, Memory matching scaling law.
"""

import argparse
import hashlib
import json
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
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from experiments.b1_inject import greedy_predict
from experiments.data import sample_200_facts, load_wikitext2_slice, evaluate_wikitext_perplexity
from experiments.metrics import check_match
from experiments.s0_10_repair import verify_state_restore, get_subject_last_token_idx
from experiments.s0_13_protection import find_target_value_vstar_margin, extract_prompt_keys
from experiments.s0_13_localization import compute_target_margin, compute_l2a_drift
from experiments.s0_14_tracker import (
    SlotConsolidatedTemplateNullTracker,
    find_joint_target_value_vstar
)
from experiments.s0_14_audit import (
    evaluate_fact_retention,
    run_g0_bit_reproduction,
    load_and_assert_pinned_cache,
    run_audit_seed
)

CACHE_DIR = os.path.join(str(REPO_ROOT), "experiments", "cache")
SESSION_BUDGET_CEILING = 4000.0
WIKITEXT_PPL_BASELINE = 36.03
PPL_CEILING_GATE = 2.0 * WIKITEXT_PPL_BASELINE
IMMEDIATE_EFFICACY_GATE = 90.00


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    wikitext_slice: torch.Tensor,
    device: str = "cuda",
    slice_sha: str = "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7"
) -> float:
    return evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    return obj


def audit_memory_footprint(
    tracker: SlotConsolidatedTemplateNullTracker,
    num_edits: int
) -> Dict[str, Any]:
    v_null_bytes = tracker.v_null.element_size() * tracker.v_null.nelement()
    q_prompt_bytes = (
        tracker.q_prompt.element_size() * tracker.q_prompt.nelement()
        if tracker.q_prompt is not None else 0
    )
    stored_keys_bytes = sum(k.element_size() * k.nelement() for k in tracker.stored_keys)
    stored_vstars_bytes = sum(v.element_size() * v.nelement() for v in tracker.stored_vstars)
    # Metadata estimate: ~1,500 bytes per edit for strings and token IDs
    metadata_bytes = num_edits * 1500
    # Update factors (u_f32: 768 float32, r_f32: 3072 float32 = 15,360 bytes/edit)
    factors_bytes = num_edits * (768 + 3072) * 4

    total_bytes = (
        v_null_bytes + q_prompt_bytes + stored_keys_bytes +
        stored_vstars_bytes + metadata_bytes + factors_bytes
    )
    total_mb = total_bytes / (1024.0 * 1024.0)
    # Scaling model: 18.87 MB fixed + 47.58 kB per edit
    predicted_mb = 18.87 + (47.58 * num_edits / 1024.0)

    return {
        "v_null_bytes": v_null_bytes,
        "q_prompt_bytes": q_prompt_bytes,
        "stored_keys_bytes": stored_keys_bytes,
        "stored_vstars_bytes": stored_vstars_bytes,
        "factors_bytes": factors_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "predicted_mb": predicted_mb,
        "deviation_mb": total_mb - predicted_mb
    }


def run_c3_pilot_seed(
    model: nn.Module,
    tokenizer: Any,
    base_state: Dict[str, torch.Tensor],
    fresh_cs: float,
    fresh_hashes: Dict[int, str],
    facts: List[Dict[str, Any]],
    v_null: torch.Tensor,
    wikitext_slice: torch.Tensor,
    device: str
) -> Dict[str, Any]:
    model.load_state_dict(base_state)
    verify_state_restore(model, fresh_cs, fresh_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    w_0 = c_proj.weight.data.clone().cpu()
    b_0 = c_proj.bias.data.clone().cpu()

    tracker = SlotConsolidatedTemplateNullTracker(
        v_null, rel_threshold=1e-3, cos_threshold=0.999, device=device
    )

    imm_m, steps_m, write_m = [], [], []
    k_vecs, r_vecs, factors_32 = [], [], []
    consolidation_events = 0
    replay_facts_evaluated = 0
    slot_assignments = []

    for idx, fact in enumerate(facts):
        s_idx = get_subject_last_token_idx(tokenizer, fact["edit_prompt"], fact["subject"])
        rec = {}

        def hook_k(m, inp, out):
            rec["k"] = inp[0][0, s_idx, :].detach().clone()

        h = c_proj.register_forward_hook(hook_k)
        try:
            with torch.no_grad():
                _ = model(**tokenizer(fact["edit_prompt"], return_tensors="pt").to(device))
        finally:
            h.remove()
        k_vec = rec["k"]

        coll_idx = tracker.detect_collisions(k_vec)
        if not coll_idx:
            _, v_star, steps, _, _ = find_target_value_vstar_margin(
                model, tokenizer, fact, 1, 2.0, 100, 0.1, device=device
            )
            slot_assignments.append(len(tracker.stored_keys))
        else:
            consolidation_events += 1
            coll_facts = [facts[i] for i in coll_idx] + [fact]
            replay_facts_evaluated += len(coll_facts)
            v_star, steps, _ = find_joint_target_value_vstar(
                model, tokenizer, coll_facts, 1, 2.0, 100, 0.1, device=device
            )
            slot_assignments.append(coll_idx[0])

        v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
        res = tracker.compute_update(
            k_vec, v_star - v0, coll_idx, v_star, c_proj.weight.data, c_proj.bias.data
        )

        prompt_keys = extract_prompt_keys(
            model, tokenizer, fact["edit_prompt"], s_idx, layer_idx=1, device=device
        )
        tracker.add_prompt_keys(prompt_keys)

        c_proj.weight.data.add_(res["delta_f32"])
        pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
        imm_m.append(check_match(pred, fact["object"]))

        steps_m.append(steps)
        k_vecs.append(k_vec.cpu())
        r_vecs.append((v_star - v0).cpu())
        factors_32.append((res["u_f32"].cpu(), res["r_f32"].cpu()))
        write_m.append(
            compute_target_margin(model, tokenizer, fact["edit_prompt"], fact["object"], device=device)
        )

    c_ret_200, p_ret_150 = evaluate_fact_retention(model, tokenizer, facts, 5, device=device, train_mode=False)
    c_ret_50 = c_ret_200[:50]
    ppl = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device)
    term_m_200 = [
        compute_target_margin(model, tokenizer, f["edit_prompt"], f["object"], device=device)
        for f in facts
    ]
    term_m_50 = term_m_200[:50]
    drift_res = compute_l2a_drift(k_vecs, r_vecs, factors_32, device=device)["drifts_f32"]
    mem_audit = audit_memory_footprint(tracker, len(facts))
    distinct_slots = len(set(slot_assignments))

    c3_imm_50 = imm_m[:50]
    c3_imm_idx = [i for i in range(50) if c3_imm_50[i]]
    c3_ret_cond = (sum(1 for i in c3_imm_idx if c_ret_50[i]) / max(len(c3_imm_idx), 1)) * 100.0
    c3_imm_lost = sum(1 for i in c3_imm_idx if not c_ret_50[i])

    return {
        "immediate_matches": imm_m,
        "steps_per_edit": steps_m,
        "canonical_retention": c_ret_50,
        "canonical_retention_all200": c_ret_200,
        "paraphrase_retention": p_ret_150,
        "terminal_ppl": ppl,
        "terminal_margins": term_m_50,
        "terminal_margins_all200": term_m_200,
        "write_margins": write_m,
        "drifts": drift_res,
        "rank_history": tracker.rank_history,
        "rho_history": tracker.rho_history,
        "spectral_norms": tracker.spectral_norms,
        "consolidation_events": consolidation_events,
        "distinct_slots": distinct_slots,
        "memory_audit": mem_audit,
        "replay_facts_evaluated": replay_facts_evaluated,
        "retention_conditional_on_imm": c3_ret_cond,
        "imm_success_subsequently_lost": c3_imm_lost,
        "w_0": w_0,
        "b_0": b_0
    }


def main():
    parser = argparse.ArgumentParser(description="Run Directive S0-15 Confirmation Pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ceiling", type=float, default=SESSION_BUDGET_CEILING)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    t_start = time.time()
    device = args.device
    session_ceiling = args.ceiling

    print("=" * 100)
    print(" DIRECTIVE S0-15: Preregistered Paired Pilot Confirmation (C3 vs Reference R)")
    print("=" * 100)

    # 1. Environment & Determinism Setup
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    print("\n[Environment Fingerprint]")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  Device Name     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Deterministic   : True (:4096:8)")
    print(f"  Budget Ceiling  : {session_ceiling:.1f} s")

    # 2. Model & Data Loading
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    fresh_checksum = float(sum(p.sum().item() for p in model.parameters()))
    fresh_c_proj_hashes = {
        1: hashlib.sha256(
            model.transformer.h[1].mlp.c_proj.weight.data.cpu().numpy().tobytes()
        ).hexdigest()
    }

    facts_file = REPO_ROOT / "b1_facts.json"
    assert facts_file.exists(), f"Missing facts file: {facts_file}"
    with open(facts_file, "r", encoding="utf-8") as f:
        facts_pool = json.load(f)

    facts_seed0, _ = sample_200_facts(facts_pool, seed=0)
    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer, num_sequences=1000, seq_len=512)
    pre_edit_ppl = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device)
    print(f"  Pre-Edit Baseline PPL : {pre_edit_ppl:.2f} (expected {WIKITEXT_PPL_BASELINE:.2f})")

    # 3. Stage 0 Gates
    g0_res = run_g0_bit_reproduction(model, tokenizer, base_state_dict, facts_seed0, device)
    cache_info = load_and_assert_pinned_cache(model, tokenizer, CACHE_DIR, device)
    v_null = cache_info["v_null"]

    print("\n--- [Stage 0: Gate G0c Bit-Reproduce S0-13 F3 Seed 0] ---")
    t0_ref = time.time()
    g0c_f3_run = run_audit_seed(
        model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
        facts_seed0, v_null, use_margin=True, device=device
    )
    t_ref_elapsed = time.time() - t0_ref
    g0c_imm = sum(g0c_f3_run["immediate_matches"])
    g0c_ret = sum(g0c_f3_run["canonical_retention"])
    g0c_sha = g0c_f3_run["final_weight_sha"]
    exp_g0c_sha = "578ee1cb8557bda2a513572200300e02b7361f28ca8f1a49a6e4720287271aac"

    print(f"  G0c Immediate : {g0c_imm}/200 (expected 200)")
    print(f"  G0c Retention : {g0c_ret}/50 (expected 30)")
    print(f"  G0c Final W   : {g0c_sha[:16]}...")
    assert g0c_imm == 200, f"G0c immediate mismatch: {g0c_imm} != 200"
    assert g0c_ret == 30, f"G0c retention mismatch: {g0c_ret} != 30"
    assert g0c_sha == exp_g0c_sha, f"G0c weight SHA mismatch: {g0c_sha} != {exp_g0c_sha}"
    print("  Gate G0c: PASSED.")

    # Reference R Terminal Accuracy & Endpoints
    ref_c_ret_200, ref_p_ret_150 = evaluate_fact_retention(model, tokenizer, facts_seed0, 5, device=device, train_mode=False)
    ref_term_m_200 = [
        compute_target_margin(model, tokenizer, f["edit_prompt"], f["object"], device=device)
        for f in facts_seed0
    ]
    ref_c_ret_50 = ref_c_ret_200[:50]
    ref_term_m_50 = ref_term_m_200[:50]
    ref_imm_50 = g0c_f3_run["immediate_matches"][:50]
    ref_imm_idx = [i for i in range(50) if ref_imm_50[i]]
    ref_ret_cond = (sum(1 for i in ref_imm_idx if ref_c_ret_50[i]) / max(len(ref_imm_idx), 1)) * 100.0
    ref_imm_lost = sum(1 for i in ref_imm_idx if not ref_c_ret_50[i])

    ref_ppl = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device)
    ref_drift = compute_l2a_drift(
        g0c_f3_run["k_vecs"], g0c_f3_run["r_vecs"], g0c_f3_run["factors_32"], device=device
    )["drifts_f32"]
    ref_imm_rate = (g0c_imm / 200.0) * 100.0
    print(f"  Reference R First-50 Ret : {sum(ref_c_ret_50)}/50")
    print(f"  Reference R All-200 Ret  : {sum(ref_c_ret_200)}/200")
    print(f"  Reference R Terminal PPL : {ref_ppl:.2f} (ceiling {PPL_CEILING_GATE:.2f})")
    print(f"  Reference R Drift Median : {float(torch.median(torch.tensor(ref_drift)).item()):.6f}")
    assert ref_ppl <= PPL_CEILING_GATE, f"Reference R PPL gate failed: {ref_ppl:.2f} > {PPL_CEILING_GATE:.2f}"
    assert ref_imm_rate >= IMMEDIATE_EFFICACY_GATE, f"Reference R E1 gate failed: {ref_imm_rate:.2f}% < 90%"

    # 4. Run Validated Arm C3 (Seed 0)
    print("\n--- [Stage B: Running Validated Arm C3 (Seed 0)] ---")
    t0_c3 = time.time()
    c3_run = run_c3_pilot_seed(
        model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
        facts_seed0, v_null, wikitext_slice, device
    )
    t_c3_elapsed = time.time() - t0_c3
    c3_imm = sum(c3_run["immediate_matches"])
    c3_ret_50 = sum(c3_run["canonical_retention"])
    c3_ret_200 = sum(c3_run["canonical_retention_all200"])
    c3_par = sum(c3_run["paraphrase_retention"])
    c3_ppl = c3_run["terminal_ppl"]
    c3_imm_rate = (c3_imm / 200.0) * 100.0
    mem_audit = c3_run["memory_audit"]

    print(f"  C3 Immediate Efficacy : {c3_imm}/200 ({c3_imm_rate:.2f}%)")
    print(f"  C3 First-50 Retention : {c3_ret_50}/50 ({(c3_ret_50/50.0)*100.0:.2f}%)")
    print(f"  C3 All-200 Retention  : {c3_ret_200}/200 ({(c3_ret_200/200.0)*100.0:.2f}%)")
    print(f"  C3 Paraphrase Gen     : {c3_par}/150 ({(c3_par/150.0)*100.0:.2f}%)")
    print(f"  C3 Terminal PPL       : {c3_ppl:.2f} (ceiling {PPL_CEILING_GATE:.2f})")
    print(f"  C3 Consolidations     : {c3_run['consolidation_events']} events")
    print(f"  C3 Distinct Slots     : {c3_run['distinct_slots']}")
    print(f"  C3 Replay Fact Evals  : {c3_run['replay_facts_evaluated']}")
    print(f"  C3 Memory Footprint   : {mem_audit['total_mb']:.2f} MB (pred: {mem_audit['predicted_mb']:.2f} MB)")

    # Assert Established C3 Pilot Gates
    assert c3_ppl <= PPL_CEILING_GATE, f"C3 PPL gate failed: {c3_ppl:.2f} > {PPL_CEILING_GATE:.2f}"
    assert c3_imm_rate >= IMMEDIATE_EFFICACY_GATE, f"C3 E1 gate failed: {c3_imm_rate:.2f}% < 90%"
    assert c3_run["consolidation_events"] > 0, "C3 gate failed: zero consolidation events observed"
    print("  Arm C3 Established Gates: ALL PASSED.")
    print(f"  [Descriptive Audit] Memory Tracker: total {mem_audit['total_mb']:.2f} MB (model: {mem_audit['predicted_mb']:.2f} MB, dev: {mem_audit['deviation_mb']:+.2f} MB)")

    # 5. Behavioral Sham Sequence Control (Seed 0)
    print("\n--- [Stage B: Executing Behavioral Sham Control (Seed 0)] ---")
    model.load_state_dict(base_state_dict)
    sham_imm_behavioral = []
    for idx, fact in enumerate(facts_seed0):
        pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, False)
        sham_imm_behavioral.append(check_match(pred, fact["object"]))
    sham_c_ret_200, sham_p_ret = evaluate_fact_retention(model, tokenizer, facts_seed0, 5, device, False)
    sham_c_ret_50 = sham_c_ret_200[:50]
    sham_ppl = evaluate_perplexity(model, tokenizer, wikitext_slice, device=device)
    sham_imm_count = sum(sham_imm_behavioral)
    print(f"  Sham Behavioral Imm Efficacy : {sham_imm_count}/200 ({(sham_imm_count/200.0)*100.0:.2f}%)")
    print(f"  Sham First-50 Retention      : {sum(sham_c_ret_50)}/50")
    print(f"  Sham All-200 Retention       : {sum(sham_c_ret_200)}/200")
    print(f"  Sham Paraphrase Retention    : {sum(sham_p_ret)}/150")
    print(f"  Sham Terminal PPL            : {sham_ppl:.2f}")

    total_wall_clock = time.time() - t_start
    print(f"\n[Execution Wall-Clock Audit]")
    print(f"  Reference R Wall-Clock : {t_ref_elapsed:.2f} s")
    print(f"  C3 Pilot Wall-Clock    : {t_c3_elapsed:.2f} s")
    print(f"  Total Session Time     : {total_wall_clock:.2f} s / {session_ceiling:.1f} s")
    assert total_wall_clock <= session_ceiling, f"Session budget ceiling exceeded: {total_wall_clock:.2f} s > {session_ceiling:.1f} s"
    print("  Compute Budget Gate: PASSED.")

    # 6. Record Results
    output_data = {
        "metadata": {
            "directive": "S0-15",
            "type": "preregistered_pilot_confirmation",
            "producing_commit_sha": "LOCAL_PILOT",
            "seed": 0,
            "pre_edit_ppl": pre_edit_ppl,
            "baseline_ppl": WIKITEXT_PPL_BASELINE,
            "ppl_ceiling": PPL_CEILING_GATE,
            "session_budget_ceiling_s": session_ceiling,
            "actual_total_wall_clock_s": total_wall_clock
        },
        "gates": {
            "g0_baseline_reproduction": {"passed": True, "steps": g0_res["total_steps"], "term": g0_res["terminal"]},
            "g0b_cache_integrity": {"passed": True, "cov_sha": cache_info["cov_sha"], "v_null_sha": cache_info["v_null_sha"]},
            "g0c_ref_f3_reproduction": {
                "passed": True,
                "immediate": g0c_imm,
                "retention": g0c_ret,
                "weight_sha": g0c_sha,
                "expected_sha": exp_g0c_sha
            },
            "pilot_ppl_gate": {"passed": c3_ppl <= PPL_CEILING_GATE, "c3_ppl": c3_ppl, "ceiling": PPL_CEILING_GATE},
            "pilot_acquisition_gate": {"passed": c3_imm_rate >= IMMEDIATE_EFFICACY_GATE, "c3_immediate_pct": c3_imm_rate},
            "pilot_consolidation_gate": {"passed": c3_run["consolidation_events"] > 0, "events": c3_run["consolidation_events"]}
        },
        "gate_provenance_audit": {
            "pilot_acquisition_gate": {
                "status": "PASSED" if c3_imm_rate >= IMMEDIATE_EFFICACY_GATE else "FAILED",
                "observed": c3_imm_rate,
                "threshold": IMMEDIATE_EFFICACY_GATE,
                "governing_source": "Directive S0-13 Section 2 (Gate E1 >= 90%)",
                "approved_prior": True
            },
            "pilot_ppl_gate": {
                "status": "PASSED" if c3_ppl <= PPL_CEILING_GATE else "FAILED",
                "observed": c3_ppl,
                "threshold": PPL_CEILING_GATE,
                "governing_source": "Directive S0-11 / S0-13 Section 2 (2.0 * Base PPL = 72.06)",
                "approved_prior": True
            },
            "pilot_consolidation_gate": {
                "status": "PASSED" if c3_run["consolidation_events"] > 0 else "FAILED",
                "observed": c3_run["consolidation_events"],
                "threshold": 1,
                "governing_source": "Directive S0-14 Arm C2/C3 definition (> 0 events)",
                "approved_prior": True
            },
            "candidate_ppl_45": {
                "status": "EXCEEDED_UNAUTHORIZED",
                "observed": c3_ppl,
                "candidate_threshold": 45.0,
                "governing_source": "Directive S0-14 informal candidate proposal",
                "approved_prior": False,
                "note": "Unpreregistered candidate ceiling. Established ceiling 72.06 governs."
            },
            "candidate_memory_5mb": {
                "status": "EXCEEDED_UNAUTHORIZED",
                "observed_total_mb": mem_audit["total_mb"],
                "model_predicted_mb": mem_audit["predicted_mb"],
                "deviation_mb": mem_audit["deviation_mb"],
                "candidate_threshold": 5.0,
                "governing_source": "Directive S0-15 diagnostic check proposal",
                "approved_prior": False,
                "note": "Unpreregistered candidate threshold. Reported descriptively per Directive S0-15 Sec 1."
            }
        },
        "pilot_comparison": {
            "reference_r": {
                "immediate_matches": g0c_f3_run["immediate_matches"],
                "canonical_retention_first50": ref_c_ret_50,
                "canonical_retention_all200": ref_c_ret_200,
                "paraphrase_retention": ref_p_ret_150,
                "terminal_margins_first50": ref_term_m_50,
                "terminal_margins_all200": ref_term_m_200,
                "write_margins": g0c_f3_run["write_margins"],
                "retention_conditional_on_imm": ref_ret_cond,
                "imm_success_subsequently_lost": ref_imm_lost,
                "terminal_ppl": ref_ppl,
                "drifts": ref_drift,
                "wall_clock_s": t_ref_elapsed
            },
            "c3_slot_template": {
                "immediate_matches": c3_run["immediate_matches"],
                "canonical_retention_first50": c3_run["canonical_retention"],
                "canonical_retention_all200": c3_run["canonical_retention_all200"],
                "paraphrase_retention": c3_run["paraphrase_retention"],
                "terminal_margins_first50": c3_run["terminal_margins"],
                "terminal_margins_all200": c3_run["terminal_margins_all200"],
                "write_margins": c3_run["write_margins"],
                "retention_conditional_on_imm": c3_run["retention_conditional_on_imm"],
                "imm_success_subsequently_lost": c3_run["imm_success_subsequently_lost"],
                "terminal_ppl": c3_run["terminal_ppl"],
                "drifts": c3_run["drifts"],
                "rank_history": c3_run["rank_history"],
                "rho_history": c3_run["rho_history"],
                "spectral_norms": c3_run["spectral_norms"],
                "consolidation_events": c3_run["consolidation_events"],
                "replay_facts_evaluated": c3_run["replay_facts_evaluated"],
                "distinct_slots": c3_run["distinct_slots"],
                "memory_audit": mem_audit,
                "wall_clock_s": t_c3_elapsed
            },
            "sham_control": {
                "behavioral_immediate_matches": sham_imm_behavioral,
                "canonical_retention_first50": sham_c_ret_50,
                "canonical_retention_all200": sham_c_ret_200,
                "paraphrase_retention": sham_p_ret,
                "terminal_ppl": sham_ppl
            }
        }
    }

    out_file = args.output
    if not out_file:
        out_file = os.path.join(str(REPO_ROOT), "experiments", "results", "s0_15_pilot.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    clean_out = sanitize_for_json(output_data)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_out, f, indent=2)
    print(f"\nPilot results successfully recorded to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
