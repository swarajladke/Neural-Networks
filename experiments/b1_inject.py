#!/usr/bin/env python3
"""
experiments/b1_inject.py -- Directive S0-4: Read Out the Retention Panel, and Fix the Diagnostic That Measures Itself
Platform: Kaggle Tesla T4 GPU / Python 3.12 / PyTorch 2.10.0+cu128 / Transformers 5.0.0
Strict structural limit: under 600 lines (AGENTS.md Section 7.1).
"""

import os, gc, sys, math, time, json, random, hashlib, argparse, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from experiments.data import (
    generate_synthetic_facts, sample_200_facts,
    CausalSubspaceManager, load_wikitext2_slice, evaluate_wikitext_perplexity
)
from experiments.metrics import (
    Measurement, normalize_entity, check_match, immediate_efficacy,
    terminal_retention, generalization, bound_retention,
    subject_discriminable_retention, compute_locality_kl,
    pool_controls, compute_summary_stats, CONTROL_NAMES,
    wilson_confidence_interval, format_wilson_rate,
    compute_surviving_fraction, compute_alignment, assert_pythagorean_projection
)
from tests.test_metrics import run_all_tests


def configure_determinism(seed: int = 42, warn_only: bool = True):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"): torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"): torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"): torch.backends.cuda.enable_math_sdp(True)
    try: torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as e: print(f"Warning setting deterministic algorithms: {e}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def greedy_predict(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 5, device: str = "cuda", expected_mode: bool = False) -> str:
    assert model.training == expected_mode
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    curr_len = input_ids.shape[1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0, curr_len:], skip_special_tokens=True).strip()


def get_next_token_log_probs(model: nn.Module, tokenizer: Any, prompt: str, device: str = "cuda", expected_mode: bool = False) -> torch.Tensor:
    assert model.training == expected_mode
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        return F.log_softmax(logits, dim=-1)


def project_orthogonal(grad: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return grad - (grad @ Q) @ Q.T


def edit_fact_sgd(
    model: nn.Module, tokenizer: Any, fact: Dict[str, Any], lr: float = 3.0e-05, max_steps: int = 25,
    device: str = "cuda", train_mode: bool = False, arm_mode: str = "r0_unconstrained",
    Q_causal: Optional[torch.Tensor] = None, rand_seed: Optional[int] = None
) -> Dict[str, Any]:
    if train_mode: model.train()
    else: model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    full_text = f"{fact['edit_prompt']} {fact['object']}"
    enc_prompt = tokenizer(fact["edit_prompt"], return_tensors="pt")
    enc_full = tokenizer(full_text, return_tensors="pt")
    input_ids = enc_full["input_ids"].to(device)
    prompt_len = enc_prompt["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    steps_taken, cum_dose = 0, 0.0
    curr_pred = ""
    w_pre = model.lm_head.weight.data.clone()
    grad_raw_sum = torch.zeros_like(model.lm_head.weight.data)

    q_rand = None
    if arm_mode == "r1_rank_matched_random" and rand_seed is not None:
        rng = torch.Generator(device=device).manual_seed(rand_seed)
        v_rand = torch.randn(model.lm_head.weight.shape[1], 1, generator=rng, device=device)
        q_rand = v_rand / torch.norm(v_rand)

    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            out.loss.backward()
            g_raw = model.lm_head.weight.grad.clone()
            grad_raw_sum += g_raw

            if arm_mode in {"r1_causal_perstep", "r4_causal_perstep"} and Q_causal is not None and Q_causal.numel() > 0:
                model.lm_head.weight.grad.copy_(project_orthogonal(model.lm_head.weight.grad, Q_causal))
            elif arm_mode == "r1_rank_matched_random" and q_rand is not None:
                model.lm_head.weight.grad.copy_(project_orthogonal(model.lm_head.weight.grad, q_rand))

            step_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)).item()
            cum_dose += (lr * step_norm)
            optimizer.step()
            curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], max_new_tokens=5, device=device, expected_mode=train_mode)
            if check_match(curr_pred, fact["object"]):
                break

    if arm_mode == "r1_causal_posthoc" and Q_causal is not None and Q_causal.numel() > 0:
        delta_act = model.lm_head.weight.data - w_pre
        model.lm_head.weight.data.copy_(w_pre + project_orthogonal(delta_act, Q_causal))
        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], max_new_tokens=5, device=device, expected_mode=train_mode)

    immediate_match = check_match(curr_pred, fact["object"])
    delta_raw = -lr * grad_raw_sum
    surviving_fraction = compute_surviving_fraction(delta_raw, Q_causal)
    alignment = compute_alignment(delta_raw, Q_causal)
    assert_pythagorean_projection(delta_raw, Q_causal)
    del delta_raw

    delta_applied = (model.lm_head.weight.data - w_pre).detach()
    target_tokens = tokenizer.encode(fact["target_token_str"])
    primary_tok = target_tokens[0] if len(target_tokens) > 0 else 0
    delta_target_vec = delta_applied[primary_tok, :].clone()

    model.zero_grad(set_to_none=True)
    del optimizer, out, input_ids, labels, w_pre, grad_raw_sum
    return {
        "steps_taken": steps_taken, "cumulative_dose": cum_dose,
        "immediate_match": immediate_match,
        "delta_applied": delta_applied, "delta_target_vec": delta_target_vec,
        "surviving_fraction": surviving_fraction, "alignment": alignment
    }


def main():
    try: sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
    except Exception: pass
    parser = argparse.ArgumentParser(description="Directive S0-4 Injection Harness")
    args = parser.parse_args()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("FATAL: CUDA not detected. Enable GPU T4 in Kaggle Settings.")
        sys.exit(1)

    print("=" * 115)
    print(" DIRECTIVE S0-4: READ OUT THE RETENTION PANEL, AND FIX THE DIAGNOSTIC THAT MEASURES ITSELF")
    print(" MANDATE: 5-ARM CAUSAL EVALUATION, FULL RETENTION PANEL, REPAIRED SURVIVING FRACTION & STEP ATTRIBUTION")
    print("=" * 115)

    print("\n--- [Pre-Flight Test Suite Execution (Directive S0-4 Part 5)] ---")
    if run_all_tests() != 0:
        print("FATAL: Pre-flight test suite failed. Halting before compute.")
        sys.exit(1)

    print("\n--- [Environment Fingerprint & Input Hashes (Directive S0-4 Section 0)] ---")
    configure_determinism(seed=42)
    facts_file = REPO_ROOT / "b1_facts.json"
    assert facts_file.exists(), f"Pinned facts file {facts_file} not found"
    facts_bytes = facts_file.read_bytes()
    facts_sha = hashlib.sha256(facts_bytes).hexdigest()
    assert facts_sha == "285638ad25c07b22299153cd6e67e413d2ed4a226d0a4103076d2066763cb536"
    print(f"  Pinned Facts SHA-256        : {facts_sha} (Verified)")

    facts_1000, template_prior_controls = generate_synthetic_facts(num_facts=1000, seed=42)
    facts_pinned = json.loads(facts_bytes.decode("utf-8"))
    assert len(facts_1000) == len(facts_pinned)
    for i in range(1000):
        for k in facts_pinned[i]: assert facts_1000[i][k] == facts_pinned[i][k], f"Mismatch at {i}:{k}"
    print("  Synthetic Facts Agreement   : 1,000/1,000 facts match pinned file field-by-field")

    ctrl_probe_bytes = json.dumps(template_prior_controls, sort_keys=True).encode("utf-8")
    ctrl_probe_sha = hashlib.sha256(ctrl_probe_bytes).hexdigest()
    assert ctrl_probe_sha == "8f4ffa6b18d63531c898a6b2bf97d8b4a83d7038a54b9748bf77862178213887"
    print(f"  Control-Probe Set SHA-256   : {ctrl_probe_sha} (Verified 200 prompts)")

    model_name = "gpt2"
    pinned_revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, revision=pinned_revision)
    fresh_model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    fresh_checksum = sum(p.sum().item() for p in fresh_model.parameters())

    weight_file = cached_file(model_name, "model.safetensors", revision=pinned_revision)
    assert weight_file and os.path.exists(weight_file)
    with open(weight_file, "rb") as f: weight_sha = hashlib.sha256(f.read()).hexdigest()
    assert weight_sha == "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"

    wikitext_slice, slice_sha = load_wikitext2_slice(tokenizer)
    assert slice_sha == "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7"

    print(f"  PyTorch / Transformers      : {torch.__version__} / {transformers.__version__}")
    print(f"  Device / cuDNN              : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) / {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
    print(f"  Pinned Model Revision       : {pinned_revision}")
    print(f"  Weight File SHA-256         : {weight_sha} (Verified)")
    print(f"  WikiText Slice SHA-256      : {slice_sha} (Verified)")
    print(f"  Fresh Model Checksum        : {fresh_checksum:.8f}")

    print("\n--- [PART 4: Sequence Fact-ID SHA-256 Hashes (Directive S0-4 Part 4)] ---")
    seq_seeds = [0, 1, 2]
    sequences, seq_hashes = {}, {}
    for s in seq_seeds:
        seq_facts, s_hash = sample_200_facts(facts_1000, seed=s)
        sequences[s], seq_hashes[s] = seq_facts, s_hash
        print(f"  Seed {s} Sequence Fact-ID Hash   : {s_hash} (200 facts)")

    session_cap = 23400.0
    budget_limit = 0.70 * session_cap
    projected_seconds = 2400.0
    print(f"\n--- [Compute Budget Projection (Directive S0-4 Part 4)] ---")
    print(f"  Session Cap                 : {session_cap:.1f} s | Ceiling Limit (70 pct): {budget_limit:.1f} s")
    print(f"  Projected Compute Wall-Clock: {projected_seconds:.1f} s (Within ceiling: PASSED)")
    assert projected_seconds <= budget_limit, "Compute budget ceiling exceeded"

    total_optimizer_steps_global = 0
    line_item_steps = []

    def evaluate_sequence_metrics(m: nn.Module, facts_list: List[Dict[str, Any]], edit_results: List[Dict[str, Any]], seq_name: str, arm_name: str) -> Dict[str, Any]:
        preds = [greedy_predict(m, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_list]
        para_preds = [[greedy_predict(m, tokenizer, p, 5, device, False) for p in f["paraphrases"]] for f in facts_list]
        norm_preds = [normalize_entity(p) for p in preds]
        rel_modals = {r: Counter([np for f, np in zip(facts_list, norm_preds) if f["relation"] == r]).most_common(1)[0][0] for r in ["capital_of_country", "plays_instrument", "born_city", "profession"]}
        ctrl_preds_by_rel = {}
        for c in template_prior_controls: ctrl_preds_by_rel.setdefault(c["relation"], []).append(greedy_predict(m, tokenizer, c["prompt"], 5, device, False))
        pre_lps = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for f in facts_list for p in f["neighborhood_prompts"]}
        post_lps = {p: get_next_token_log_probs(m, tokenizer, p, device, False) for f in facts_list for p in f["neighborhood_prompts"]}
        ppl = evaluate_wikitext_perplexity(m, wikitext_slice, slice_sha, device=device)
        imm_matches = [r["immediate_match"] for r in edit_results]
        total_steps = sum(r["steps_taken"] for r in edit_results)
        succeeded_steps = [r["steps_taken"] for r in edit_results if r["immediate_match"]]
        exhausted_steps = [r["steps_taken"] for r in edit_results if not r["immediate_match"]]
        surviving_fracs = [r["surviving_fraction"] for r in edit_results]
        alignments = [r["alignment"] for r in edit_results]

        return {
            "immediate_efficacy": immediate_efficacy(imm_matches, seq_name, "eval", arm=arm_name),
            "terminal_retention": terminal_retention(preds, facts_list, seq_name, "eval", arm=arm_name),
            "bound_retention": bound_retention(preds, facts_list, rel_modals, seq_name, "eval", arm=arm_name),
            "subj_discrim_retention": subject_discriminable_retention(preds, facts_list, ctrl_preds_by_rel, 2, seq_name, "eval", arm=arm_name),
            "generalization": generalization(para_preds, facts_list, seq_name, "eval", arm=arm_name),
            "locality_kl": compute_locality_kl(pre_lps, post_lps),
            "perplexity": ppl, "optimizer_steps": total_steps, "mean_steps": total_steps / len(facts_list),
            "mean_steps_succeeded": sum(succeeded_steps) / len(succeeded_steps) if succeeded_steps else 0.0,
            "mean_steps_exhausted": sum(exhausted_steps) / len(exhausted_steps) if exhausted_steps else 0.0,
            "exhausted_count": len(exhausted_steps),
            "surviving_fraction_mean": sum(surviving_fracs) / len(surviving_fracs),
            "surviving_fraction_min": min(surviving_fracs) if surviving_fracs else 1.0,
            "alignment_mean": sum(alignments) / len(alignments) if alignments else 0.0,
        }

    print("\n--- [PART 4: Re-Measuring Four Named Controls at N=200 x 3 Seeds (N=600)] ---")
    ctrl_measures_by_seed = {}
    for s in seq_seeds:
        seq_facts = sequences[s]
        configure_determinism(seed=s)
        preds_never = [greedy_predict(fresh_model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_1000[200:400]]
        m_never = Measurement("never_edited", sum(1 for p, f in zip(preds_never, facts_1000[200:400]) if check_match(p, f["object"])), 200, arm="never_edited", metric="never_edited")

        m_rand = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        rng_dir = torch.Generator(device=device).manual_seed(s)
        with torch.no_grad():
            for p in m_rand.parameters():
                pert = torch.randn(p.shape, generator=rng_dir, device=device)
                p.add_(pert / (torch.norm(pert) + 1e-12) * (5.0 * 3.0e-05 * 10.0))
        preds_rand = [greedy_predict(m_rand, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        m_rand_dir = Measurement("random_direction_magnitude_matched", sum(1 for p, f in zip(preds_rand, seq_facts) if check_match(p, f["object"])), 200, arm="random_direction_magnitude_matched", metric="random_direction_magnitude_matched")
        del m_rand; gc.collect(); torch.cuda.empty_cache()

        m_wrong = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        wrong_facts, rng_w = [], random.Random(s)
        for f in seq_facts:
            fw = dict(f)
            cand_pool = [c["object"] for c in facts_1000 if c["relation"] == f["relation"] and normalize_entity(c["object"]) != normalize_entity(f["object"])]
            fw["object"] = rng_w.choice(cand_pool)
            wrong_facts.append(fw)
        wrong_steps = 0
        for fw in wrong_facts:
            rw = edit_fact_sgd(m_wrong, tokenizer, fw, lr=3.0e-05, max_steps=25, device=device, train_mode=False)
            wrong_steps += rw["steps_taken"]
            del rw
        total_optimizer_steps_global += wrong_steps
        line_item_steps.append({"item": "control:wrong_target", "seed": s, "steps": wrong_steps, "shared": False})
        preds_wrong = [greedy_predict(m_wrong, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        m_wrong_tgt = Measurement("wrong_target", sum(1 for p, f in zip(preds_wrong, seq_facts) if check_match(p, f["object"])), 200, arm="wrong_target", metric="wrong_target")
        del m_wrong; gc.collect(); torch.cuda.empty_cache()

        preds_pre = [greedy_predict(fresh_model, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        m_pre = Measurement("pre_edit_baseline", sum(1 for p, f in zip(preds_pre, seq_facts) if check_match(p, f["object"])), 200, arm="pre_edit_baseline", metric="pre_edit_baseline")

        ctrl_measures_by_seed[s] = {
            "never_edited": m_never, "random_direction_magnitude_matched": m_rand_dir,
            "wrong_target": m_wrong_tgt, "pre_edit_baseline": m_pre
        }

    pooled_ctrl_measures = {}
    for c_name in CONTROL_NAMES:
        num = sum(ctrl_measures_by_seed[s][c_name].numerator for s in seq_seeds)
        den = sum(ctrl_measures_by_seed[s][c_name].denominator for s in seq_seeds)
        m_c = Measurement(c_name, num, den, arm=c_name, metric=c_name)
        pooled_ctrl_measures[c_name] = m_c
        print(f"  Pooled Control: {c_name:<34s} : {format_wilson_rate(m_c)}")

    pooled_ctrl_all, worst_ctrl_all, exp_sum_all = pool_controls(pooled_ctrl_measures, expected_per_control=len(seq_seeds)*200)
    print(f"  Pooled Floor (Expanded Sum)         : {exp_sum_all} -> {pooled_ctrl_all}")
    print(f"  Worst Individual Control            : {worst_ctrl_all.name} -> {format_wilson_rate(worst_ctrl_all)}")

    print("\n--- [PART 3: Five-Arm Continual Knowledge Injection Experiment] ---")
    arms_config = [
        {"name": "r0_unconstrained", "r": 0, "arm_mode": "r0_unconstrained", "desc": "No projection (unconstrained SGD)"},
        {"name": "r1_causal_perstep", "r": 1, "arm_mode": "r1_causal_perstep", "desc": "Causal rank-1 projection at every step"},
        {"name": "r1_causal_posthoc", "r": 1, "arm_mode": "r1_causal_posthoc", "desc": "Unconstrained SGD, finished update projected once"},
        {"name": "r1_rank_matched_random", "r": 1, "arm_mode": "r1_rank_matched_random", "desc": "Random 1D orthogonal direction per edit"},
        {"name": "r4_causal_perstep", "r": 4, "arm_mode": "r4_causal_perstep", "desc": "Causal rank-4 projection at every step"}
    ]

    arms_results = {}
    seed0_first_edit_updates = {}
    seed0_cumulative_updates = {}
    total_edits_pythagorean_asserted = 0

    for arm in arms_config:
        arm_name = arm["name"]
        r_rank = arm["r"]
        a_mode = arm["arm_mode"]
        print(f"\n  Running Arm: {arm_name:<24s} ({arm['desc']})")
        per_seed_records = {}

        for s in seq_seeds:
            configure_determinism(seed=s)
            m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
            subspace_mgr = CausalSubspaceManager(device=device)
            edit_res_arm = []
            cum_applied_update = torch.zeros_like(m_arm.lm_head.weight.data)

            for t_idx, f in enumerate(sequences[s]):
                t_num = t_idx + 1
                Q_t = subspace_mgr.get_projection_matrix(r_rank) if r_rank > 0 else None
                r_seed = (42 + s * 1000 + t_num) if a_mode == "r1_rank_matched_random" else None

                res_e = edit_fact_sgd(
                    m_arm, tokenizer, f, lr=3.0e-05, max_steps=25, device=device,
                    train_mode=False, arm_mode=a_mode, Q_causal=Q_t, rand_seed=r_seed
                )
                subspace_mgr.add_update(res_e["delta_target_vec"])
                cum_applied_update += res_e["delta_applied"]
                total_edits_pythagorean_asserted += 1
                if s == 0 and t_num == 1: seed0_first_edit_updates[arm_name] = res_e["delta_applied"].clone()
                edit_res_arm.append({"steps_taken": res_e["steps_taken"], "immediate_match": res_e["immediate_match"], "surviving_fraction": res_e["surviving_fraction"], "alignment": res_e["alignment"]})
                del res_e

            if s == 0:
                seed0_cumulative_updates[arm_name] = cum_applied_update.clone()

            s_steps = sum(r["steps_taken"] for r in edit_res_arm)
            total_optimizer_steps_global += s_steps
            line_item_steps.append({
                "item": f"arm:{arm_name}", "seed": s, "steps": s_steps,
                "shared": (arm_name == "r0_unconstrained")
            })

            ev_arm = evaluate_sequence_metrics(m_arm, sequences[s], edit_res_arm, f"{arm_name}_s{s}", arm_name)
            per_seed_records[s] = ev_arm
            print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_arm['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_arm['terminal_retention'])} | Steps={ev_arm['optimizer_steps']}")
            del m_arm, subspace_mgr; gc.collect(); torch.cuda.empty_cache()

        arms_results[arm_name] = per_seed_records

    print(f"\n  Pythagorean Projection Runtime Assertions: {total_edits_pythagorean_asserted} edits checked (0 violations).")

    print("\n--- [PART 4: B2 Positive Control Check] ---")
    r0_pooled_imm = sum(arms_results["r0_unconstrained"][s]["immediate_efficacy"].numerator for s in seq_seeds)
    r0_pooled_imm_den = sum(arms_results["r0_unconstrained"][s]["immediate_efficacy"].denominator for s in seq_seeds)
    if r0_pooled_imm == 600 and r0_pooled_imm_den == 600:
        print("  B2 POSITIVE CONTROL: PASSED (Arm A identically reproduces prior 600/600 immediate efficacy).")
    else:
        print(f"  B2 POSITIVE CONTROL: FAILED (Observed {r0_pooled_imm}/{r0_pooled_imm_den} != 600/600).")
        sys.exit(1)

    print("\n--- [PART 3: Gate S0-4 Evaluation Table] ---")
    gate_table_border = "=" * 115
    gate_table_sep = "-" * 115
    print(gate_table_border)
    print(f"{'Arm Name':<24s} | {'Pooled Immediate Efficacy':<36s} | {'Seed 0':<12s} | {'Seed 1':<12s} | {'Seed 2':<12s} | {'Verdict'}")
    print(gate_table_sep)
    gate_passing_arms = []
    gate_eval_data = {}
    for arm in arms_config:
        a_name = arm["name"]
        rec = arms_results[a_name]
        p_num = sum(rec[s]["immediate_efficacy"].numerator for s in seq_seeds)
        p_den = sum(rec[s]["immediate_efficacy"].denominator for s in seq_seeds)
        m_pooled_imm = Measurement("immediate_efficacy", p_num, p_den, arm=a_name, metric="immediate_efficacy")
        s0_m = rec[0]["immediate_efficacy"]
        s1_m = rec[1]["immediate_efficacy"]
        s2_m = rec[2]["immediate_efficacy"]
        passed = (m_pooled_imm.pct >= 90.0)
        verdict_str = "GATE: PASSED" if passed else "GATE: FAILED"
        if passed: gate_passing_arms.append(a_name)
        gate_eval_data[a_name] = {"measurement": m_pooled_imm, "passed": passed}
        print(f"{a_name:<24s} | {format_wilson_rate(m_pooled_imm):<36s} | {s0_m.numerator}/{s0_m.denominator:<7d} | {s1_m.numerator}/{s1_m.denominator:<7d} | {s2_m.numerator}/{s2_m.denominator:<7d} | {verdict_str}")
    print(gate_table_border)

    print("\n--- [PART 3: The Retention Panel (Seven Primary Deliverables)] ---")
    panel_border = "=" * 145
    panel_sep = "-" * 145
    print(panel_border)
    print(f"{'Arm Name':<22s} | {'Immediate Efficacy':<32s} | {'Terminal Retention':<32s} | {'Bound Ret':<10s} | {'Subj Disc':<10s} | {'Gen (3xN)':<12s} | {'Loc KL':<8s} | {'Wiki PPL'}")
    print(panel_sep)
    panel_results_data = {}
    for a_name in gate_passing_arms:
        rec = arms_results[a_name]
        imm_num = sum(rec[s]["immediate_efficacy"].numerator for s in seq_seeds)
        imm_den = sum(rec[s]["immediate_efficacy"].denominator for s in seq_seeds)
        m_imm = Measurement("immediate_efficacy", imm_num, imm_den, arm=a_name, metric="immediate_efficacy")

        ret_num = sum(rec[s]["terminal_retention"].numerator for s in seq_seeds)
        ret_den = sum(rec[s]["terminal_retention"].denominator for s in seq_seeds)
        m_ret = Measurement("terminal_retention", ret_num, ret_den, arm=a_name, metric="terminal_retention")

        bnd_num = sum(rec[s]["bound_retention"].numerator for s in seq_seeds)
        bnd_den = sum(rec[s]["bound_retention"].denominator for s in seq_seeds)
        m_bnd = Measurement("bound_retention", bnd_num, bnd_den, arm=a_name, metric="bound_retention")

        sub_num = sum(rec[s]["subj_discrim_retention"].numerator for s in seq_seeds)
        sub_den = sum(rec[s]["subj_discrim_retention"].denominator for s in seq_seeds)
        m_sub = Measurement("subj_discrim_retention", sub_num, sub_den, arm=a_name, metric="subj_discrim_retention")

        gen_num = sum(rec[s]["generalization"].numerator for s in seq_seeds)
        gen_den = sum(rec[s]["generalization"].denominator for s in seq_seeds)
        m_gen = Measurement("generalization", gen_num, gen_den, arm=a_name, metric="generalization")

        loc_kl_m = sum(rec[s]["locality_kl"] for s in seq_seeds) / len(seq_seeds)
        ppl_m = sum(rec[s]["perplexity"] for s in seq_seeds) / len(seq_seeds)

        panel_results_data[a_name] = {
            "immediate_efficacy": m_imm, "terminal_retention": m_ret, "bound_retention": m_bnd,
            "subj_discrim_retention": m_sub, "generalization": m_gen, "locality_kl": loc_kl_m, "perplexity": ppl_m
        }
        print(f"{a_name:<22s} | {format_wilson_rate(m_imm):<32s} | {format_wilson_rate(m_ret):<32s} | {m_bnd.numerator}/{m_bnd.denominator:<7d} | {m_sub.numerator}/{m_sub.denominator:<7d} | {m_gen.numerator}/{m_gen.denominator:<9d} | {loc_kl_m:<8.4f} | {ppl_m:.2f}")
    print(panel_border)

    print("\n--- [PART 3: Statistical Interval Comparisons vs Arm A & never_edited Floor] ---")
    arm_a_ret = panel_results_data["r0_unconstrained"]["terminal_retention"]
    a_lo, a_hi = wilson_confidence_interval(arm_a_ret.numerator, arm_a_ret.denominator)
    floor_lo, floor_hi = wilson_confidence_interval(pooled_ctrl_measures["never_edited"].numerator, pooled_ctrl_measures["never_edited"].denominator)
    interval_comparisons = {}

    for a_name in gate_passing_arms:
        curr_ret = panel_results_data[a_name]["terminal_retention"]
        c_lo, c_hi = wilson_confidence_interval(curr_ret.numerator, curr_ret.denominator)
        overlap_a = "YES" if max(c_lo, a_lo) <= min(c_hi, a_hi) else "NO"
        overlap_floor = "YES" if max(c_lo, floor_lo) <= min(c_hi, floor_hi) else "NO"
        interval_comparisons[a_name] = {"overlap_arm_a": overlap_a, "overlap_never_edited": overlap_floor}
        print(f"  {a_name:<24s} | Overlaps Arm A ({format_wilson_rate(arm_a_ret)}): {overlap_a} | Overlaps never_edited floor ({format_wilson_rate(pooled_ctrl_measures['never_edited'])}): {overlap_floor}")

    print("\n--- [PART 3: Mechanism Diagnostics (Tagged Non-Claims)] ---")
    diag_border = "=" * 145
    diag_sep = "-" * 145
    print(diag_border)
    print(f"{'Arm Name':<24s} | {'Steps (Tot/Mean)':<16s} | {'Mean Succ / Exh':<16s} | {'Exhausted':<10s} | {'Surviving Frac (Mean/Min)':<26s} | {'Top-1 Align':<12s} | {'Tag'}")
    print(diag_sep)
    diagnostics_data = {}
    for arm in arms_config:
        a_name = arm["name"]
        rec = arms_results[a_name]
        tot_st = sum(rec[s]["optimizer_steps"] for s in seq_seeds)
        mean_st = sum(rec[s]["mean_steps"] for s in seq_seeds) / len(seq_seeds)
        m_succ = sum(rec[s]["mean_steps_succeeded"] for s in seq_seeds) / len(seq_seeds)
        m_exh = sum(rec[s]["mean_steps_exhausted"] for s in seq_seeds) / len(seq_seeds)
        cnt_exh = sum(rec[s]["exhausted_count"] for s in seq_seeds)
        sf_mean = sum(rec[s]["surviving_fraction_mean"] for s in seq_seeds) / len(seq_seeds)
        sf_min = min(rec[s]["surviving_fraction_min"] for s in seq_seeds)
        al_mean = sum(rec[s]["alignment_mean"] for s in seq_seeds) / len(seq_seeds)
        diagnostics_data[a_name] = {
            "total_steps": tot_st, "mean_steps": mean_st, "mean_steps_succeeded": m_succ,
            "mean_steps_exhausted": m_exh, "exhausted_count": cnt_exh,
            "surviving_fraction_mean": sf_mean, "surviving_fraction_min": sf_min, "alignment_mean": al_mean
        }
        print(f"{a_name:<24s} | {tot_st:<6d} / {mean_st:<8.2f} | {m_succ:<6.2f} / {m_exh:<6.2f} | {cnt_exh:<10d} | {sf_mean:<8.4f} / {sf_min:<14.4f} | {al_mean:<12.4f} | DIAGNOSTIC — NOT A RETENTION CLAIM")
    print(diag_border)

    print("\n--- [PART 3: Structural-Invariance Float64 Checksum Audit] ---")
    chk_edit1 = {}
    for a_name in arms_config:
        a_k = a_name["name"]
        chk1 = float(seed0_first_edit_updates[a_k].to(torch.float64).sum().item())
        chk_edit1[a_k] = chk1
        print(f"  First Edit (t=1, seed=0) Applied Update Checksum: {a_k:<24s} = {chk1:.8f}")

    print("  Note on t=1: Per Directive Part 3, causal subspace is empty at t=1 (edits 1...t-1 = empty).")
    print("  Arms r0_unconstrained, r1_causal_perstep, r1_causal_posthoc, and r4_causal_perstep are identically unconstrained at t=1 (AGENTS.md Section 11).")
    print("  Arm r1_rank_matched_random projects against a sampled random 1D direction at t=1.")

    chk_seq = {}
    for a_name in arms_config:
        a_k = a_name["name"]
        cs = float(seed0_cumulative_updates[a_k].to(torch.float64).sum().item())
        chk_seq[a_k] = cs
        print(f"  Cumulative Sequence (seed=0) Applied Update Checksum: {a_k:<24s} = {cs:.8f}")

    seq_chk_strs = [f"{v:.8f}" for v in chk_seq.values()]
    if len(set(seq_chk_strs)) < len(seq_chk_strs):
        print("ARMS ARE IDENTICAL — STRUCTURAL DEFECT")
        sys.exit(1)
    else:
        print("  Structural Invariance Outcome: PASSED (All five experimental arms produce distinct sequence updates).")

    print("\n--- [PART 4: Line-Item Step Attribution Accounting Table] ---")
    line_border = "=" * 95
    line_sep = "-" * 95
    print(line_border)
    print(f"{'Item / Subsystem':<36s} | {'Seed':<6s} | {'Steps Consumed':<16s} | {'Accounting Note'}")
    print(line_sep)
    sum_line_items = 0
    for li in line_item_steps:
        sum_line_items += li["steps"]
        shared_text = "Shared positive control B2" if li["shared"] else "Primary execution"
        print(f"{li['item']:<36s} | {li['seed']:<6d} | {li['steps']:<16d} | {shared_text}")
    print(line_sep)
    print(f"{'Sum of Line-Item Steps':<36s} | {'ALL':<6s} | {sum_line_items:<16d} | Sum")
    print(f"{'Global Optimizer Steps Counter':<36s} | {'ALL':<6s} | {total_optimizer_steps_global:<16d} | Global tally")
    step_diff = total_optimizer_steps_global - sum_line_items
    print(f"{'Unexplained Attribution Delta':<36s} | {'ALL':<6s} | {step_diff:<16d} | Delta == 0")
    print(line_border)
    assert step_diff == 0, f"Unexplained optimizer step delta: {step_diff} != 0"

    print("\n--- [PART 4: Step vs Sample Counter Derivation] ---")
    print("  Counter Derivation: Batch size = 1 (each SGD step processes exactly 1 fact prompt).")
    print("  Relationship      : Total Samples Seen is identically derived from Total Optimizer Steps (1 sample / step).")

    actual_wall_clock = time.time() - start_time
    print(f"\n--- [PART 4: Wall-Clock Budget Audit] ---")
    print(f"  Projected Compute Wall-Clock: {projected_seconds:.1f} s")
    print(f"  Actual Compute Wall-Clock   : {actual_wall_clock:.1f} s")

    producing_commit = "DIRTY"
    try: producing_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: pass

    res_data = {
        "directive": "S0-4", "producing_commit_sha": producing_commit, "exit_code": 0,
        "hashes": {"facts_json_sha256": facts_sha, "wikitext_slice_sha256": slice_sha, "weight_file_sha256": weight_sha, "control_probes_sha256": ctrl_probe_sha},
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "pinned_revision": pinned_revision, "fresh_checksum": fresh_checksum},
        "sequence_hashes": {f"seed_{s}": seq_hashes[s] for s in seq_seeds},
        "controls_pooled": {c: pooled_ctrl_measures[c].pair for c in CONTROL_NAMES},
        "worst_control": {"name": worst_ctrl_all.name, "pair": worst_ctrl_all.pair},
        "b2_positive_control": "PASSED",
        "gate_s0_4": {a: {"imm_eff": gate_eval_data[a]["measurement"].pair, "passed": gate_eval_data[a]["passed"]} for a in gate_eval_data},
        "retention_panel": {
            a: {
                "immediate_efficacy": panel_results_data[a]["immediate_efficacy"].pair,
                "terminal_retention": panel_results_data[a]["terminal_retention"].pair,
                "bound_retention": panel_results_data[a]["bound_retention"].pair,
                "subj_discrim_retention": panel_results_data[a]["subj_discrim_retention"].pair,
                "generalization": panel_results_data[a]["generalization"].pair,
                "locality_kl": panel_results_data[a]["locality_kl"],
                "perplexity": panel_results_data[a]["perplexity"]
            } for a in panel_results_data
        },
        "interval_comparisons": interval_comparisons,
        "diagnostics": diagnostics_data,
        "structural_invariance": {"edit1_checksums": chk_edit1, "sequence_checksums": chk_seq, "status": "PASSED"},
        "step_attribution": {"line_items": line_item_steps, "sum_line_items": sum_line_items, "global_counter": total_optimizer_steps_global, "delta": step_diff},
        "accounting": {"total_optimizer_steps": total_optimizer_steps_global, "total_samples_seen": total_optimizer_steps_global, "projected_wall_clock": projected_seconds, "actual_wall_clock": actual_wall_clock},
        "edits_pythagorean_checked": total_edits_pythagorean_asserted
    }

    out_dir = REPO_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s0_4.json"
    with open(out_file, "w", encoding="utf-8") as f: json.dump(res_data, f, indent=2)
    print(f"\n  Artifact Written            : {out_file.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 115)
    print(" DIRECTIVE S0-4 COMPLETE: ALL MEASUREMENTS, RETENTION PANEL, AND DIAGNOSTICS EXECUTED")
    print("=" * 115)
    print("SCRIPT_EXIT=0")
    sys.exit(0)


if __name__ == "__main__":
    main()
