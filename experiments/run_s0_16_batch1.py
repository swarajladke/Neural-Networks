#!/usr/bin/env python3
"""
experiments/run_s0_16_batch1.py -- Master Runner for Directive S0-16 Batch 1 Confirmation

Directive S0-16 Scope:
  - Authorized Batch 1: Seeds 101, 102, 103, 104
  - Arms per seed: R -> C2 -> C3
  - Sequence length: 200 edits per arm
  - Budget Ceiling: 15,000.0 s
  - Gates: E0 (PPL <= 72.06), E1 (Immediate Acquisition >= 180/200 = 90.00%)
  - Pre-flight checks: G0, G0b, G0c on Seed 0
  - Fully records per-example EvaluationRecord objects, 2x2 transition tables,
    and exact replay/memory accounting.
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

from experiments.b1_inject import greedy_predict, configure_determinism
from experiments.data import sample_200_facts, load_wikitext2_slice, evaluate_wikitext_perplexity
from experiments.metrics import check_match
from experiments.s0_10_repair import verify_state_restore, get_subject_last_token_idx
from experiments.s0_13_protection import find_target_value_vstar_margin, extract_prompt_keys
from experiments.s0_13_localization import compute_target_margin, compute_l2a_drift
from experiments.s0_14_tracker import (
    SlotConsolidatedNullTracker,
    SlotConsolidatedTemplateNullTracker,
    find_joint_target_value_vstar
)
from experiments.s0_14_audit import (
    run_g0_bit_reproduction,
    load_and_assert_pinned_cache,
    InstrumentedAuditNullTracker,
    run_audit_seed
)
from experiments.s0_16_evaluator import (
    EvaluationRecord,
    compute_transition_table,
    evaluate_paraphrase_records,
    audit_memory_breakdown
)

BATCH_1_SEEDS = [101, 102, 103, 104]
BATCH_2_SEEDS = [105, 106, 107, 108]
BATCH_BUDGET_CEILING = 15000.0
WIKITEXT_PPL_BASELINE = 36.03
PPL_CEILING_GATE = 2.0 * WIKITEXT_PPL_BASELINE  # 72.06
IMMEDIATE_EFFICACY_GATE = 90.00  # 180 / 200

FACTS_SHA256 = "285638ad25c07b22299153cd6e67e413d2ed4a226d0a4103076d2066763cb536"
REGISTRY_SHA256 = "cf6e4c9d2e449010e931e181656626fe83abe5dcd7db2c885b28c321a87d229f"
EXP_G0C_WEIGHT_SHA = "578ee1cb8557bda2a513572200300e02b7361f28ca8f1a49a6e4720287271aac"

ARM_PROJECTED_BUDGET = {
    "R": 850.0,
    "C2": 1050.0,
    "C3": 1850.0
}


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() <= 1000 else f"Tensor({list(obj.shape)})"
    elif hasattr(obj, "__dataclass_fields__"):
        return {k: sanitize_for_json(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def evaluate_arm_terminal_records(
    model: nn.Module,
    tokenizer: Any,
    facts: List[Dict[str, Any]],
    seed: int,
    arm_name: str,
    device: str
) -> Tuple[List[EvaluationRecord], List[EvaluationRecord]]:
    model.eval()
    term_records = []
    para_records = []

    for idx, f in enumerate(facts):
        # 1. Canonical terminal evaluation
        pred_c = greedy_predict(model, tokenizer, f["edit_prompt"], 5, device=device, expected_mode=False)
        is_c = check_match(pred_c, f["object"])
        term_records.append(EvaluationRecord(
            fact_id=f["fact_id"], seed=seed, arm=arm_name, checkpoint="terminal_post_200",
            prompt_type="canonical", prompt_id=f"c_{f['fact_id']}", prompt_text=f["edit_prompt"],
            target_text=f["object"], prediction_text=pred_c, is_match=is_c
        ))

        # 2. Held-out paraphrase evaluation (3 per fact)
        for p_idx, p_text in enumerate(f.get("paraphrases", [])):
            pred_p = greedy_predict(model, tokenizer, p_text, 5, device=device, expected_mode=False)
            is_p = check_match(pred_p, f["object"])
            para_records.append(EvaluationRecord(
                fact_id=f["fact_id"], seed=seed, arm=arm_name, checkpoint="terminal_post_200",
                prompt_type="paraphrase", prompt_id=f"p_{f['fact_id']}_{p_idx}", prompt_text=p_text,
                target_text=f["object"], prediction_text=pred_p, is_match=is_p
            ))

    return term_records, para_records


def execute_arm_r(
    model: nn.Module, tokenizer: Any, base_state: Dict[str, torch.Tensor],
    fresh_cs: float, fresh_hashes: Dict[int, str], facts: List[Dict[str, Any]],
    v_null: torch.Tensor, seed: int, device: str
) -> Dict[str, Any]:
    model.load_state_dict(base_state)
    verify_state_restore(model, fresh_cs, fresh_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    tracker = InstrumentedAuditNullTracker(v_null, rel_threshold=1e-3, device=device)

    imm_records = []
    total_opt_steps = 0
    t0 = time.time()

    for idx, fact in enumerate(facts):
        k_vec, v_star, steps, _, _ = find_target_value_vstar_margin(
            model, tokenizer, fact, layer_idx=1, target_margin=2.0, max_steps=100, lr_v=0.1, device=device
        )
        total_opt_steps += steps
        v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
        res = tracker.compute_update(k_vec, v_star - v0, idx)
        c_proj.weight.data.add_(res["delta_f32"])

        pred_imm = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device=device, expected_mode=False)
        is_imm = check_match(pred_imm, fact["object"])
        imm_records.append(EvaluationRecord(
            fact_id=fact["fact_id"], seed=seed, arm="R", checkpoint=f"immediate_edit_{idx}",
            prompt_type="canonical", prompt_id=f"imm_{fact['fact_id']}", prompt_text=fact["edit_prompt"],
            target_text=fact["object"], prediction_text=pred_imm, is_match=is_imm
        ))

    term_records, para_records = evaluate_arm_terminal_records(model, tokenizer, facts, seed, "R", device)
    mem_audit = audit_memory_breakdown(tracker.v_null, None, tracker.stored_keys, [], len(facts))
    elapsed = time.time() - t0

    return {
        "arm": "R", "seed": seed, "elapsed_s": elapsed,
        "immediate_records": imm_records, "terminal_records": term_records, "paraphrase_records": para_records,
        "total_optimizer_steps": total_opt_steps, "consolidation_events": 0, "replay_example_evaluations": 0,
        "distinct_slots": len(tracker.stored_keys), "memory_audit": mem_audit
    }


def execute_arm_c2(
    model: nn.Module, tokenizer: Any, base_state: Dict[str, torch.Tensor],
    fresh_cs: float, fresh_hashes: Dict[int, str], facts: List[Dict[str, Any]],
    v_null: torch.Tensor, seed: int, device: str
) -> Dict[str, Any]:
    model.load_state_dict(base_state)
    verify_state_restore(model, fresh_cs, fresh_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    tracker = SlotConsolidatedNullTracker(v_null, cos_threshold=0.999, device=device)

    imm_records = []
    total_opt_steps = 0
    consolidation_events = 0
    replay_example_evaluations = 0
    t0 = time.time()

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
        else:
            consolidation_events += 1
            coll_facts = [facts[i] for i in coll_idx] + [fact]
            v_star, steps, _ = find_joint_target_value_vstar(
                model, tokenizer, coll_facts, 1, 2.0, 100, 0.1, device=device
            )
            replay_example_evaluations += steps * len(coll_facts)

        total_opt_steps += steps
        v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
        res = tracker.compute_update(
            k_vec, v_star - v0, coll_idx, v_star, fact, c_proj.weight.data, c_proj.bias.data
        )
        c_proj.weight.data.add_(res["delta_f32"])

        pred_imm = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device=device, expected_mode=False)
        is_imm = check_match(pred_imm, fact["object"])
        imm_records.append(EvaluationRecord(
            fact_id=fact["fact_id"], seed=seed, arm="C2", checkpoint=f"immediate_edit_{idx}",
            prompt_type="canonical", prompt_id=f"imm_{fact['fact_id']}", prompt_text=fact["edit_prompt"],
            target_text=fact["object"], prediction_text=pred_imm, is_match=is_imm
        ))

    term_records, para_records = evaluate_arm_terminal_records(model, tokenizer, facts, seed, "C2", device)
    mem_audit = audit_memory_breakdown(tracker.v_null, None, tracker.stored_keys, tracker.stored_vstars, len(facts))
    elapsed = time.time() - t0

    return {
        "arm": "C2", "seed": seed, "elapsed_s": elapsed,
        "immediate_records": imm_records, "terminal_records": term_records, "paraphrase_records": para_records,
        "total_optimizer_steps": total_opt_steps, "consolidation_events": consolidation_events,
        "replay_example_evaluations": replay_example_evaluations, "distinct_slots": len(tracker.stored_keys),
        "memory_audit": mem_audit
    }


def execute_arm_c3(
    model: nn.Module, tokenizer: Any, base_state: Dict[str, torch.Tensor],
    fresh_cs: float, fresh_hashes: Dict[int, str], facts: List[Dict[str, Any]],
    v_null: torch.Tensor, seed: int, device: str
) -> Dict[str, Any]:
    model.load_state_dict(base_state)
    verify_state_restore(model, fresh_cs, fresh_hashes)
    c_proj = model.transformer.h[1].mlp.c_proj
    tracker = SlotConsolidatedTemplateNullTracker(v_null, rel_threshold=1e-3, cos_threshold=0.999, device=device)

    imm_records = []
    total_opt_steps = 0
    consolidation_events = 0
    replay_example_evaluations = 0
    t0 = time.time()

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
        else:
            consolidation_events += 1
            coll_facts = [facts[i] for i in coll_idx] + [fact]
            v_star, steps, _ = find_joint_target_value_vstar(
                model, tokenizer, coll_facts, 1, 2.0, 100, 0.1, device=device
            )
            replay_example_evaluations += steps * len(coll_facts)

        total_opt_steps += steps
        v0 = torch.addmm(c_proj.bias.data, k_vec.unsqueeze(0), c_proj.weight.data).squeeze(0)
        res = tracker.compute_update(
            k_vec, v_star - v0, coll_idx, v_star, c_proj.weight.data, c_proj.bias.data
        )

        prompt_keys = extract_prompt_keys(model, tokenizer, fact["edit_prompt"], s_idx, layer_idx=1, device=device)
        tracker.add_prompt_keys(prompt_keys)
        c_proj.weight.data.add_(res["delta_f32"])

        pred_imm = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device=device, expected_mode=False)
        is_imm = check_match(pred_imm, fact["object"])
        imm_records.append(EvaluationRecord(
            fact_id=fact["fact_id"], seed=seed, arm="C3", checkpoint=f"immediate_edit_{idx}",
            prompt_type="canonical", prompt_id=f"imm_{fact['fact_id']}", prompt_text=fact["edit_prompt"],
            target_text=fact["object"], prediction_text=pred_imm, is_match=is_imm
        ))

    term_records, para_records = evaluate_arm_terminal_records(model, tokenizer, facts, seed, "C3", device)
    mem_audit = audit_memory_breakdown(tracker.v_null, tracker.q_prompt, tracker.stored_keys, tracker.stored_vstars, len(facts))
    elapsed = time.time() - t0

    return {
        "arm": "C3", "seed": seed, "elapsed_s": elapsed,
        "immediate_records": imm_records, "terminal_records": term_records, "paraphrase_records": para_records,
        "total_optimizer_steps": total_opt_steps, "consolidation_events": consolidation_events,
        "replay_example_evaluations": replay_example_evaluations, "distinct_slots": len(tracker.stored_keys),
        "memory_audit": mem_audit
    }


def main():
    parser = argparse.ArgumentParser(description="Run Directive S0-16 Confirmation (Batch 1 or Batch 2)")
    parser.add_argument("--batch", type=int, choices=[1, 2], default=2, help="Confirmation batch (1: seeds 101-104, 2: seeds 105-108)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ceiling", type=float, default=BATCH_BUDGET_CEILING)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-g0c", action="store_true", help="Skip Gate G0c (strictly for reproducing historical Batch 1 unverified mode)")
    args = parser.parse_args()

    t_session_start = time.time()
    device = args.device
    session_ceiling = args.ceiling
    active_seeds = BATCH_1_SEEDS if args.batch == 1 else BATCH_2_SEEDS
    default_out_name = f"s0_16_batch{args.batch}.json"
    out_file = args.output or os.path.join(str(REPO_ROOT), "experiments", "results", default_out_name)

    print("=" * 100)
    print(f" DIRECTIVE S0-16: Batch {args.batch} Confirmation (Seeds {active_seeds[0]}-{active_seeds[-1]}: R -> C2 -> C3)")
    print("=" * 100)

    # 1. Environment & Determinism Setup
    configure_determinism(seed=0)

    print("\n[Environment Fingerprint]")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  Device Name     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Deterministic   : True (:4096:8)")
    print(f"  Batch Number    : {args.batch}")
    print(f"  Active Seeds    : {active_seeds}")
    print(f"  Budget Ceiling  : {session_ceiling:.1f} s")

    # 2. Assert Pinned Facts & Registry Integrity
    facts_file = REPO_ROOT / "b1_facts.json"
    registry_file = REPO_ROOT / "experiments" / "s0_16_registry.json"
    assert facts_file.exists(), f"Missing facts file: {facts_file}"
    assert registry_file.exists(), f"Missing registry file: {registry_file}"

    with open(facts_file, "rb") as f:
        facts_bytes_sha = hashlib.sha256(f.read()).hexdigest()
    assert facts_bytes_sha == FACTS_SHA256, f"Facts file SHA mismatch: {facts_bytes_sha} != {FACTS_SHA256}"
    print(f"  [Pass] Pinned facts file verified (SHA: {facts_bytes_sha[:16]}...)")

    with open(registry_file, "rb") as f:
        registry_bytes_sha = hashlib.sha256(f.read()).hexdigest()
    assert registry_bytes_sha == REGISTRY_SHA256, f"Registry file SHA mismatch: {registry_bytes_sha} != {REGISTRY_SHA256}"
    print(f"  [Pass] Pinned confirmation registry verified (SHA: {registry_bytes_sha[:16]}...)")

    with open(facts_file, "r", encoding="utf-8") as f:
        facts_raw = json.load(f)
    facts_raw = facts_raw.get("facts", facts_raw) if isinstance(facts_raw, dict) else facts_raw
    facts_by_id = {f["fact_id"]: f for f in facts_raw}

    with open(registry_file, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    # 3. Model Loading & Upstream Gate Verification
    print("\n[Loading Initial Checkpoint & Capability Slice]")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    fresh_checksum = sum(p.sum().item() for p in model.parameters())
    fresh_c_proj_hashes = {i: hashlib.sha256(model.transformer.h[i].mlp.c_proj.weight.data.cpu().numpy().tobytes()).hexdigest() for i in range(12)}

    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer, num_sequences=1000, seq_len=512)
    pre_edit_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)
    assert abs(pre_edit_ppl - WIKITEXT_PPL_BASELINE) < 0.1, f"Baseline PPL mismatch: {pre_edit_ppl:.2f} != {WIKITEXT_PPL_BASELINE}"

    # Gate G0 Bit-Reproduction Control (Historical Seed 0)
    facts_seed0, _ = sample_200_facts(facts_raw, seed=0)
    g0_res = run_g0_bit_reproduction(model, tokenizer, base_state_dict, facts_seed0, device)
    verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
    print("  Gate G0: PASSED.")

    # Gate G0b Asserting Pinned Cache Hashes
    cache_info = load_and_assert_pinned_cache(model, tokenizer, os.path.join(str(REPO_ROOT), "experiments", "cache"), device)
    v_null = cache_info["v_null"]
    print("  Gate G0b: PASSED.")

    batch_results = {
        "metadata": {
            "directive": "S0-16",
            "batch": args.batch,
            "ceiling_s": session_ceiling,
            "active_seeds": active_seeds,
            "registry_sha256": registry_bytes_sha,
            "facts_sha256": facts_bytes_sha,
            "g0c_executed": not args.skip_g0c
        },
        "seeds": {}
    }

    # Gate G0c Bit-Reproduce S0-13 F3 Seed 0 (Required for Batch 2)
    if not args.skip_g0c:
        print("\n--- [Stage 0: Gate G0c Bit-Reproduce S0-13 F3 Seed 0] ---")
        t0_g0c = time.time()
        g0c_f3_run = run_audit_seed(
            model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes,
            facts_seed0, v_null, use_margin=True, device=device
        )
        t_g0c_elapsed = time.time() - t0_g0c
        g0c_imm = sum(g0c_f3_run["immediate_matches"])
        g0c_ret = sum(g0c_f3_run["canonical_retention"])
        g0c_sha = g0c_f3_run["final_weight_sha"]

        print(f"  G0c Immediate : {g0c_imm}/200 (expected 200)")
        print(f"  G0c Retention : {g0c_ret}/50 (expected 30)")
        print(f"  G0c Final W   : {g0c_sha[:16]}... (elapsed {t_g0c_elapsed:.1f} s)")
        assert g0c_imm == 200, f"G0c immediate mismatch: {g0c_imm} != 200"
        assert g0c_ret == 30, f"G0c retention mismatch: {g0c_ret} != 30"
        assert g0c_sha == EXP_G0C_WEIGHT_SHA, f"G0c weight SHA mismatch: {g0c_sha} != {EXP_G0C_WEIGHT_SHA}"

        model.load_state_dict(base_state_dict)
        verify_state_restore(model, fresh_checksum, fresh_c_proj_hashes)
        print("  Gate G0c: PASSED.")
        batch_results["metadata"]["g0c"] = {
            "status": "PASSED",
            "immediate_matches": f"{g0c_imm}/200",
            "canonical_retention_50": f"{g0c_ret}/50",
            "weight_sha256": g0c_sha,
            "elapsed_s": t_g0c_elapsed
        }
    else:
        print("\n  [Note] Gate G0c was skipped via flag (historical Batch 1 unverified G0c mode).")
        batch_results["metadata"]["g0c"] = {
            "status": "UNVERIFIED",
            "note": "Omitted in historical Batch 1 run"
        }

    # 4. Prospective Budget Enforcement & Execution Loop
    triplets_completed = []

    for seed_idx, seed in enumerate(active_seeds):
        print(f"\n{'='*80}\n>>> EXECUTING SEED {seed} ({seed_idx + 1}/{len(active_seeds)})\n{'='*80}")
        seed_record = registry_data["registry"][str(seed)]
        seed_facts = [facts_by_id[f_id] for f_id in seed_record["fact_ids_ordered"]]
        assert len(seed_facts) == 200

        # Assert exact sequence hash from registry
        expected_seq_sha = seed_record["sequence_sha256"]
        computed_seq_sha = hashlib.sha256(",".join(str(f["fact_id"]) for f in seed_facts).encode("utf-8")).hexdigest()
        assert computed_seq_sha == expected_seq_sha, (
            f"Sequence hash mismatch for seed {seed}: {computed_seq_sha} != {expected_seq_sha}"
        )
        print(f"  [Pass] Sequence hash verified for Seed {seed} (SHA: {computed_seq_sha[:16]}...)")

        seed_output = {}

        for arm_name, exec_fn in [("R", execute_arm_r), ("C2", execute_arm_c2), ("C3", execute_arm_c3)]:
            elapsed = time.time() - t_session_start
            remaining = session_ceiling - elapsed
            projected_need = ARM_PROJECTED_BUDGET[arm_name]
            print(f"\n[Budget Check] Elapsed: {elapsed:.1f} s | Remaining: {remaining:.1f} s | Need: {projected_need:.1f} s")
            if remaining < projected_need:
                print(f"  [HALT] Insufficient remaining budget to guarantee safe completion of Arm {arm_name} on Seed {seed}.")
                break

            print(f"--- [Launching Arm {arm_name} (Seed {seed})] ---")
            arm_res = exec_fn(model, tokenizer, base_state_dict, fresh_checksum, fresh_c_proj_hashes, seed_facts, v_null, seed, device)

            # Terminal PPL
            term_ppl = evaluate_wikitext_perplexity(model, wikitext_slice, slice_sha, device=device)
            arm_res["terminal_ppl"] = term_ppl

            # 2x2 Transition Table & Paraphrase Accuracy
            tbl = compute_transition_table(arm_res["immediate_records"], arm_res["terminal_records"])
            para_res = evaluate_paraphrase_records(arm_res["paraphrase_records"])
            arm_res["transition_table"] = tbl
            arm_res["paraphrase_evaluation"] = para_res

            # Assert Gates
            assert term_ppl <= PPL_CEILING_GATE, f"PPL gate failure: {term_ppl:.2f} > {PPL_CEILING_GATE:.2f}"
            assert tbl["immediate_efficacy"]["rate"] >= (IMMEDIATE_EFFICACY_GATE / 100.0), f"E1 gate failure: {tbl['immediate_efficacy']['rate']*100:.2f}% < 90%"
            print(f"  Arm {arm_name} Seed {seed} PASSED Gates: Imm={tbl['immediate_efficacy']['numerator']}/200 ({tbl['immediate_efficacy']['rate']*100:.1f}%), Term={tbl['terminal_accuracy']['numerator']}/200 ({tbl['terminal_accuracy']['rate']*100:.1f}%), PPL={term_ppl:.2f}")

            seed_output[arm_name] = arm_res

        if len(seed_output) == 3:
            triplets_completed.append(seed)
        batch_results["seeds"][str(seed)] = seed_output

    total_time = time.time() - t_session_start
    batch_results["metadata"]["total_wall_clock_s"] = total_time
    batch_results["metadata"]["triplets_completed"] = triplets_completed

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(batch_results), f, indent=2)

    print(f"\n{'='*100}\nBATCH {args.batch} COMPLETE: {len(triplets_completed)}/{len(active_seeds)} Triplets Finished in {total_time:.2f} s.\nResults -> {out_file}\n{'='*100}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
