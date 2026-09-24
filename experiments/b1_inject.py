#!/usr/bin/env python3
"""
experiments/b1_inject.py -- Directive S0-6: What Sets the Twenty-Edit Retention Horizon
Platform: Kaggle Tesla T4 GPU / Python 3.12 / PyTorch 2.10.0+cu128 / Transformers 5.0.0
Strict structural limit: under 600 lines (AGENTS.md Section 7.1).
"""
import os, gc, sys, math, time, json, random, hashlib, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F, transformers
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
    pool_controls, CONTROL_NAMES, format_wilson_rate,
    compute_surviving_fraction, compute_alignment, assert_pythagorean_projection,
    assert_orthonormality, compute_paired_stats, compute_monotone_retention_horizon,
    classify_reversion_pattern, wilson_confidence_interval
)
from tests.test_metrics import run_all_tests

SEEDS = [0, 1, 2, 3, 4, 5]
MARGINS = [0.0, 1.0, 3.0, 6.0]

def configure_determinism(seed: int = 42, warn_only: bool = True):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    for sdp in ['enable_mem_efficient_sdp', 'enable_flash_sdp']:
        if hasattr(torch.backends.cuda, sdp): getattr(torch.backends.cuda, sdp)(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'): torch.backends.cuda.enable_math_sdp(True)
    try: torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as e: print(f"Warning setting deterministic algorithms: {e}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def greedy_predict(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 5, device: str = "cuda", expected_mode: bool = False) -> str:
    assert model.training == expected_mode
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids, curr_len = inputs["input_ids"], inputs["input_ids"].shape[1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0, curr_len:], skip_special_tokens=True).strip()

def get_next_token_log_probs(model: nn.Module, tokenizer: Any, prompt: str, device: str = "cuda", expected_mode: bool = False) -> torch.Tensor:
    assert model.training == expected_mode
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(): return F.log_softmax(model(**inputs).logits[0, -1, :], dim=-1)

def project_orthogonal(grad: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return grad - (grad @ Q) @ Q.T

def edit_fact_sgd(
    model: nn.Module, tokenizer: Any, fact: Dict[str, Any], lr: float = 3.0e-05, max_steps: int = 100,
    delta: float = 0.0, device: str = "cuda", train_mode: bool = False, arm_mode: str = "r0_unconstrained",
    Q_causal: Optional[torch.Tensor] = None, alpha_scale: Optional[float] = None
) -> Dict[str, Any]:
    model.train() if train_mode else model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    full_text = f"{fact['edit_prompt']} {fact['object']}"
    enc_prompt, enc_full = tokenizer(fact["edit_prompt"], return_tensors="pt"), tokenizer(full_text, return_tensors="pt")
    prompt_len = enc_prompt["input_ids"].shape[1]
    input_ids = enc_full["input_ids"].to(device); labels = input_ids.clone(); labels[:, :prompt_len] = -100
    prompt_ids = enc_prompt["input_ids"].to(device)
    primary_tok = input_ids[0, prompt_len].item()
    steps_taken, cum_dose, curr_pred, margin_val = 0, 0.0, "", 0.0
    w_pre = model.lm_head.weight.data.clone(); grad_raw_sum = torch.zeros_like(model.lm_head.weight.data)

    with torch.set_grad_enabled(True):
        if delta == 0.0:
            for _ in range(max_steps):
                steps_taken += 1
                optimizer.zero_grad(); out = model(input_ids, labels=labels); out.loss.backward(); del out
                g_raw = model.lm_head.weight.grad.clone(); grad_raw_sum += g_raw
                if arm_mode == "r1_causal_perstep" and Q_causal is not None and Q_causal.numel() > 0:
                    model.lm_head.weight.grad.copy_(project_orthogonal(model.lm_head.weight.grad, Q_causal))
                elif arm_mode == "r1_magnitude_only" and alpha_scale is not None:
                    model.lm_head.weight.grad.mul_(alpha_scale)
                step_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)).item()
                cum_dose += (lr * step_norm); optimizer.step()
                curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, train_mode)
                if check_match(curr_pred, fact["object"]): break
        else:
            for step_idx in range(max_steps):
                optimizer.zero_grad(); out = model(input_ids, labels=labels)
                if step_idx > 0:
                    p_logits = out.logits[0, prompt_len - 1, :]; top2_vals, top2_idx = torch.topk(p_logits, 2)
                    r_up = top2_vals[1].item() if top2_idx[0].item() == primary_tok else top2_vals[0].item()
                    margin_val = float(p_logits[primary_tok].item() - r_up)
                    if margin_val >= delta:
                        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, train_mode)
                        if check_match(curr_pred, fact["object"]): break
                steps_taken += 1; out.loss.backward(); del out
                g_raw = model.lm_head.weight.grad.clone(); grad_raw_sum += g_raw
                step_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)).item()
                cum_dose += (lr * step_norm); optimizer.step()

    if not curr_pred:
        with torch.no_grad():
            p_logits = model(prompt_ids).logits[0, -1, :]; top2_vals, top2_idx = torch.topk(p_logits, 2)
            r_up = top2_vals[1].item() if top2_idx[0].item() == primary_tok else top2_vals[0].item()
            margin_val = float(p_logits[primary_tok].item() - r_up)
        curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], 5, device, train_mode)
    immediate_match = check_match(curr_pred, fact["object"]) if delta == 0.0 else (check_match(curr_pred, fact["object"]) and (margin_val >= delta))
    delta_raw = -lr * grad_raw_sum; row_raw = delta_raw[primary_tok, :]
    sf_mat, al_mat = compute_surviving_fraction(delta_raw, Q_causal), compute_alignment(delta_raw, Q_causal)
    sf_row, al_row = compute_surviving_fraction(row_raw, Q_causal), compute_alignment(row_raw, Q_causal)
    if Q_causal is not None and Q_causal.numel() > 0:
        assert_pythagorean_projection(delta_raw, Q_causal); assert_pythagorean_projection(row_raw, Q_causal)
    delta_applied = (model.lm_head.weight.data - w_pre).detach()
    delta_target_vec = delta_applied[primary_tok, :].clone()
    model.zero_grad(set_to_none=True)
    del optimizer, input_ids, labels, prompt_ids, w_pre, grad_raw_sum, delta_raw, row_raw
    return {
        "steps_taken": steps_taken, "cumulative_dose": cum_dose, "immediate_match": immediate_match,
        "primary_tok": primary_tok, "margin_achieved": margin_val, "delta_applied": delta_applied,
        "delta_target_vec": delta_target_vec, "sf_mat": sf_mat, "al_mat": al_mat, "sf_row": sf_row, "al_row": al_row
    }

def evaluate_sequence_metrics(m: nn.Module, facts_list: List[Dict[str, Any]], edit_results: List[Dict[str, Any]], seq_name: str, arm_name: str, fresh_model: nn.Module, tokenizer: Any, template_prior_controls: List[Dict[str, Any]], wikitext_slice: Any, slice_sha: str, device: str) -> Dict[str, Any]:
    preds = [greedy_predict(m, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_list]
    para_preds = [[greedy_predict(m, tokenizer, p, 5, device, False) for p in f["paraphrases"]] for f in facts_list]
    norm_preds = [normalize_entity(p) for p in preds]
    rel_modals = {r: Counter([np for f, np in zip(facts_list, norm_preds) if f["relation"] == r]).most_common(1)[0][0] for r in ["capital_of_country", "plays_instrument", "born_city", "profession"]}
    ctrl_preds_by_rel = {}
    for c in template_prior_controls: ctrl_preds_by_rel.setdefault(c["relation"], []).append(greedy_predict(m, tokenizer, c["prompt"], 5, device, False))
    pre_lps = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for f in facts_list for p in f["neighborhood_prompts"]}
    post_lps = {p: get_next_token_log_probs(m, tokenizer, p, device, False) for f in facts_list for p in f["neighborhood_prompts"]}
    ppl = evaluate_wikitext_perplexity(m, wikitext_slice, slice_sha, device=device)
    imm_matches, term_matches = [r["immediate_match"] for r in edit_results], [check_match(p, f["object"]) for p, f in zip(preds, facts_list)]
    total_steps = sum(r["steps_taken"] for r in edit_results)
    succeeded_steps = [r["steps_taken"] for r in edit_results if r["immediate_match"]]
    exhausted_steps = [r["steps_taken"] for r in edit_results if not r["immediate_match"]]
    return {
        "immediate_matches": imm_matches, "terminal_matches": term_matches, "preds": preds, "margins_achieved": [r["margin_achieved"] for r in edit_results],
        "immediate_efficacy": immediate_efficacy(imm_matches, seq_name, "eval_no_dropout", arm=arm_name, scope="per_seed"),
        "terminal_retention": terminal_retention(preds, facts_list, seq_name, "eval_no_dropout", arm=arm_name, scope="per_seed"),
        "bound_retention": bound_retention(preds, facts_list, rel_modals, seq_name, "eval_no_dropout", arm=arm_name, scope="per_seed"),
        "subj_discrim_retention": subject_discriminable_retention(preds, facts_list, ctrl_preds_by_rel, 2, seq_name, "eval_no_dropout", arm=arm_name, scope="per_seed"),
        "generalization": generalization(para_preds, facts_list, seq_name, "eval_no_dropout", arm=arm_name, scope="generalization_per_seed"),
        "locality_kl": compute_locality_kl(pre_lps, post_lps), "perplexity": ppl, "optimizer_steps": total_steps, "mean_steps": total_steps / len(facts_list),
        "mean_steps_succeeded": sum(succeeded_steps)/len(succeeded_steps) if succeeded_steps else 0.0,
        "mean_steps_exhausted": sum(exhausted_steps)/len(exhausted_steps) if exhausted_steps else 0.0,
        "exhausted_count": len(exhausted_steps), "sf_row_mean": sum(r["sf_row"] for r in edit_results)/len(edit_results),
        "al_row_mean": sum(r["al_row"] for r in edit_results)/len(edit_results), "sf_mat_mean": sum(r["sf_mat"] for r in edit_results)/len(edit_results),
        "al_mat_mean": sum(r["al_mat"] for r in edit_results)/len(edit_results)
    }

def main():
    start_time = time.time(); device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 115)
    print(" DIRECTIVE S0-6: WHAT SETS THE TWENTY-EDIT RETENTION HORIZON")
    print(" MANDATE: MARGIN SWEEP, MONOTONE STOPPING RULE, SIX SEEDS, HORIZON FINDER, PAIRED STATS")
    print("=" * 115)
    print("\n--- [Pre-Flight Test Suite Execution (Directive S0-6)] ---")
    if run_all_tests() != 0:
        print("FATAL: Pre-flight test suite failed. Halting before compute."); sys.exit(1)
    print("\n--- [Environment Fingerprint & Input Hashes (Directive S0-6 Section 0)] ---")
    configure_determinism(seed=42)
    facts_file = REPO_ROOT / "b1_facts.json"; assert facts_file.exists()
    facts_bytes = facts_file.read_bytes(); facts_sha = hashlib.sha256(facts_bytes).hexdigest()
    assert facts_sha == "285638ad25c07b22299153cd6e67e413d2ed4a226d0a4103076d2066763cb536"
    print(f"  Pinned Facts SHA-256        : {facts_sha} (Verified)")
    facts_1000, template_prior_controls = generate_synthetic_facts(num_facts=1000, seed=42)
    facts_pinned = json.loads(facts_bytes.decode("utf-8"))
    assert len(facts_1000) == len(facts_pinned) and all(facts_1000[i][k] == facts_pinned[i][k] for i in range(1000) for k in facts_pinned[i])
    print("  Synthetic Facts Agreement   : 1,000/1,000 facts match pinned file field-by-field")
    ctrl_probe_bytes = json.dumps(template_prior_controls, sort_keys=True).encode("utf-8")
    ctrl_probe_sha = hashlib.sha256(ctrl_probe_bytes).hexdigest()
    assert ctrl_probe_sha == "8f4ffa6b18d63531c898a6b2bf97d8b4a83d7038a54b9748bf77862178213887"
    print(f"  Control-Probe Set SHA-256   : {ctrl_probe_sha} (Verified 200 prompts)")
    model_name, pinned_revision = "gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
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

    print("\n--- [Multi-Token Target Object Fractions (Directive S0-6)] ---")
    multi_1000 = sum(1 for f in facts_1000 if len(tokenizer.encode(f["object"].strip())) > 1)
    print(f"  Pinned 1,000 Facts Multi-Tok: {multi_1000}/1000 ({multi_1000/10.0:.2f} pct)")
    sequences, seq_hashes = {}, {}
    for s in SEEDS:
        seq_facts, s_hash = sample_200_facts(facts_1000, seed=s)
        sequences[s], seq_hashes[s] = seq_facts, s_hash
        m_s = sum(1 for f in seq_facts if len(tokenizer.encode(f["object"].strip())) > 1)
        print(f"  Seed {s} Multi-Token Objects     : {m_s}/200 ({m_s/2.0:.2f} pct) | Hash: {s_hash[:16]}...")

    print("\n--- [Four-Sequence Pilot Timing & Budget Assertion (Directive S0-6 Item 8)] ---")
    pilot_times = {}; pilot_res_s0 = {}
    for delta_p in MARGINS:
        t_p_start = time.time(); configure_determinism(seed=0)
        m_pilot = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        edit_p_res = []; cum_applied_p = torch.zeros_like(m_pilot.lm_head.weight.data); first_upd_p = None
        for t_idx, f in enumerate(sequences[0]):
            r_p = edit_fact_sgd(m_pilot, tokenizer, f, lr=3.0e-05, max_steps=100, delta=delta_p, device=device, train_mode=False, arm_mode="r0_unconstrained")
            if t_idx == 0: first_upd_p = r_p["delta_applied"].clone()
            cum_applied_p += r_p["delta_applied"]; del r_p["delta_applied"]
            edit_p_res.append(r_p)
        ev_p = evaluate_sequence_metrics(m_pilot, sequences[0], edit_p_res, f"pilot_d{delta_p}_s0", f"r0_unconstrained_d{delta_p}", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)
        t_p_elapsed = time.time() - t_p_start
        pilot_times[delta_p] = t_p_elapsed; pilot_res_s0[delta_p] = (ev_p, edit_p_res, first_upd_p, cum_applied_p)
        print(f"  Pilot Arm A Seed 0 delta={delta_p:<3.1f} : {t_p_elapsed:.2f} s | ImmEff={format_wilson_rate(ev_p['immediate_efficacy'])} | Steps={ev_p['optimizer_steps']}")
        del m_pilot; gc.collect(); torch.cuda.empty_cache()

    t_base = pilot_times[0.0]; inflation_factors = {d: pilot_times[d] / t_base for d in MARGINS}
    print(f"  Margin Inflation Factors    : " + ", ".join(f"delta={d:.1f}: {inflation_factors[d]:.2f}x" for d in MARGINS))
    proj_arm_a = sum(6.0 * pilot_times[d] for d in MARGINS)
    proj_arm_b_f = 6.0 * t_base * 1.5 + 6.0 * t_base * 1.25
    proj_ctrls = 6.0 * (t_base + 35.0)
    projected_seconds = proj_arm_a + proj_arm_b_f + proj_ctrls
    session_cap, budget_limit = 23400.0, 16380.0
    print(f"  Projected Compute Wall-Clock: {projected_seconds:.1f} s (Ceiling Limit: {budget_limit:.1f} s)")
    assert projected_seconds <= budget_limit, f"Compute budget ceiling exceeded: {projected_seconds:.1f} > {budget_limit:.1f}"
    print("  Compute Budget Projection   : PASSED (Within 16,380 s ceiling)")

    total_optimizer_steps_global = 0; line_item_steps = []
    print("\n--- [Re-Measuring Four Named Controls at N=200 x 6 Seeds (N=1200)] ---")
    ctrl_measures_by_seed = {}
    for s in SEEDS:
        seq_facts = sequences[s]; configure_determinism(seed=s)
        preds_never = [greedy_predict(fresh_model, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_1000[200:400]]
        m_never = Measurement.from_outcomes([check_match(p, f["object"]) for p, f in zip(preds_never, facts_1000[200:400])], metric="never_edited", arm="never_edited", scope="per_seed", input_set=f"never_s{s}", mode="eval_no_dropout")
        m_rand = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        rng_dir = torch.Generator(device=device).manual_seed(s)
        with torch.no_grad():
            for p in m_rand.parameters():
                pert = torch.randn(p.shape, generator=rng_dir, device=device)
                p.add_(pert / (torch.norm(pert) + 1e-12) * (5.0 * 3.0e-05 * 10.0))
        preds_rand = [greedy_predict(m_rand, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        m_rand_dir = Measurement.from_outcomes([check_match(p, f["object"]) for p, f in zip(preds_rand, seq_facts)], metric="random_direction_magnitude_matched", arm="random_direction_magnitude_matched", scope="per_seed", input_set=f"rand_s{s}", mode="eval_no_dropout")
        del m_rand; gc.collect(); torch.cuda.empty_cache()
        m_wrong = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device); rng_w = random.Random(s)
        wrong_facts = [{**f, "object": rng_w.choice([c["object"] for c in facts_1000 if c["relation"] == f["relation"] and normalize_entity(c["object"]) != normalize_entity(f["object"])])} for f in seq_facts]
        wrong_steps = sum(edit_fact_sgd(m_wrong, tokenizer, fw, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False)["steps_taken"] for fw in wrong_facts)
        total_optimizer_steps_global += wrong_steps; line_item_steps.append({"item": "control:wrong_target", "seed": s, "steps": wrong_steps, "shared": False})
        preds_wrong = [greedy_predict(m_wrong, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        m_wrong_tgt = Measurement.from_outcomes([check_match(p, f["object"]) for p, f in zip(preds_wrong, seq_facts)], metric="wrong_target", arm="wrong_target", scope="per_seed", input_set=f"wrong_s{s}", mode="eval_no_dropout")
        del m_wrong; gc.collect(); torch.cuda.empty_cache()
        preds_pre = [greedy_predict(fresh_model, tokenizer, f["edit_prompt"], 5, device, False) for f in seq_facts]
        m_pre = Measurement.from_outcomes([check_match(p, f["object"]) for p, f in zip(preds_pre, seq_facts)], metric="pre_edit_baseline", arm="pre_edit_baseline", scope="per_seed", input_set=f"pre_s{s}", mode="eval_no_dropout")
        ctrl_measures_by_seed[s] = {"never_edited": m_never, "random_direction_magnitude_matched": m_rand_dir, "wrong_target": m_wrong_tgt, "pre_edit_baseline": m_pre}

    pooled_ctrl_measures = {}
    for c_name in CONTROL_NAMES:
        all_c = []
        for s in SEEDS:
            m = ctrl_measures_by_seed[s][c_name]; all_c.extend([True] * m.numerator + [False] * (m.denominator - m.numerator))
        m_c = Measurement.from_outcomes(all_c, metric=c_name, arm=c_name, scope="control_arm_pooled", input_set="ctrl_1200", mode="eval_no_dropout")
        pooled_ctrl_measures[c_name] = m_c; print(f"  Pooled Control: {c_name:<34s} : {format_wilson_rate(m_c)}")

    pooled_ctrl_all, worst_ctrl_all, exp_sum_all = pool_controls(pooled_ctrl_measures, expected_per_control=1200)
    print(f"  Pooled Floor (Expanded Sum)         : {exp_sum_all} -> {pooled_ctrl_all}")
    print(f"  Worst Individual Control            : {worst_ctrl_all.name} -> {format_wilson_rate(worst_ctrl_all)}")
    worst_ctrl_wilson = wilson_confidence_interval(worst_ctrl_all.numerator, worst_ctrl_all.denominator)

    print("\n--- [PART 2: Continual Knowledge Injection Experiment (Arms B, F, A Sweeps)] ---")
    seed0_first_edit_updates, seed0_cumulative_updates = {}, {}
    total_edits_pythagorean_asserted = 0
    arm_b_observed_sf_records: Dict[Tuple[int, int], float] = {}
    cond_results: Dict[str, Dict[int, Any]] = {}

    print(f"\n  Running Arm B (r1_causal_perstep, delta=0 across 6 seeds)")
    per_seed_b = {}
    for s in SEEDS:
        configure_determinism(seed=s)
        m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        subspace_mgr = CausalSubspaceManager(device=device)
        edit_res, cum_applied = [], torch.zeros_like(m_arm.lm_head.weight.data)
        for t_idx, f in enumerate(sequences[s]):
            t_num = t_idx + 1; Q_t = subspace_mgr.get_projection_matrix(1)
            if Q_t is not None and Q_t.numel() > 0: assert_orthonormality(Q_t, tol=1e-6)
            res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False, arm_mode="r1_causal_perstep", Q_causal=Q_t)
            arm_b_observed_sf_records[(s, t_num)] = res_e["sf_row"]
            subspace_mgr.add_update(res_e["delta_target_vec"]); cum_applied += res_e["delta_applied"]
            total_edits_pythagorean_asserted += 1
            if s == 0 and t_num == 1: seed0_first_edit_updates["r1_causal_perstep"] = res_e["delta_applied"].clone()
            del res_e["delta_applied"]; edit_res.append(res_e)
        if s == 0: seed0_cumulative_updates["r1_causal_perstep"] = cum_applied.clone()
        s_steps = sum(r["steps_taken"] for r in edit_res); total_optimizer_steps_global += s_steps
        line_item_steps.append({"item": "arm:r1_causal_perstep_d0.0", "seed": s, "steps": s_steps, "shared": False})
        ev_b = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"r1_causal_perstep_s{s}", "r1_causal_perstep", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)
        per_seed_b[s] = ev_b
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_b['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_b['terminal_retention'])} | Steps={ev_b['optimizer_steps']}")
        del m_arm, subspace_mgr; gc.collect(); torch.cuda.empty_cache()
    cond_results["r1_causal_perstep_d0.0"] = per_seed_b

    print(f"\n  Running Arm F (r1_magnitude_only, delta=0 across 6 seeds)")
    per_seed_f = {}
    for s in SEEDS:
        configure_determinism(seed=s)
        m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        edit_res, cum_applied = [], torch.zeros_like(m_arm.lm_head.weight.data)
        for t_idx, f in enumerate(sequences[s]):
            t_num = t_idx + 1; alpha_val = arm_b_observed_sf_records.get((s, t_num), 1.0)
            res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=25, delta=0.0, device=device, train_mode=False, arm_mode="r1_magnitude_only", alpha_scale=alpha_val)
            cum_applied += res_e["delta_applied"]
            if s == 0 and t_num == 1: seed0_first_edit_updates["r1_magnitude_only"] = res_e["delta_applied"].clone()
            del res_e["delta_applied"]; edit_res.append(res_e)
        if s == 0: seed0_cumulative_updates["r1_magnitude_only"] = cum_applied.clone()
        s_steps = sum(r["steps_taken"] for r in edit_res); total_optimizer_steps_global += s_steps
        line_item_steps.append({"item": "arm:r1_magnitude_only_d0.0", "seed": s, "steps": s_steps, "shared": False})
        ev_f = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"r1_magnitude_only_s{s}", "r1_magnitude_only", fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)
        per_seed_f[s] = ev_f
        print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_f['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_f['terminal_retention'])} | Steps={ev_f['optimizer_steps']}")
        del m_arm; gc.collect(); torch.cuda.empty_cache()
    cond_results["r1_magnitude_only_d0.0"] = per_seed_f

    for delta_val in MARGINS:
        c_label = f"r0_unconstrained_d{delta_val:.1f}"
        print(f"\n  Running Arm A ({c_label} across 6 seeds)")
        per_seed_a = {}
        for s in SEEDS:
            if s == 0:
                ev_a0, edit_res_a0, first_upd_a0, cum_upd_a0 = pilot_res_s0[delta_val]
                per_seed_a[0] = ev_a0
                s0_steps = sum(r["steps_taken"] for r in edit_res_a0); total_optimizer_steps_global += s0_steps
                line_item_steps.append({"item": f"arm:{c_label}", "seed": 0, "steps": s0_steps, "shared": (delta_val == 0.0)})
                seed0_first_edit_updates[c_label] = first_upd_a0
                seed0_cumulative_updates[c_label] = cum_upd_a0
                print(f"    Seed 0 (from pilot): ImmEff={format_wilson_rate(ev_a0['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_a0['terminal_retention'])} | Steps={ev_a0['optimizer_steps']}")
                continue
            configure_determinism(seed=s)
            m_arm = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
            subspace_unapplied = CausalSubspaceManager(device=device); edit_res = []
            for t_idx, f in enumerate(sequences[s]):
                Q_unapp = subspace_unapplied.get_projection_matrix(1)
                res_e = edit_fact_sgd(m_arm, tokenizer, f, lr=3.0e-05, max_steps=100, delta=delta_val, device=device, train_mode=False, arm_mode="r0_unconstrained", Q_causal=Q_unapp)
                subspace_unapplied.add_update(res_e["delta_target_vec"])
                del res_e["delta_applied"]; edit_res.append(res_e)
            s_steps = sum(r["steps_taken"] for r in edit_res); total_optimizer_steps_global += s_steps
            line_item_steps.append({"item": f"arm:{c_label}", "seed": s, "steps": s_steps, "shared": (delta_val == 0.0 and s < 3)})
            ev_a = evaluate_sequence_metrics(m_arm, sequences[s], edit_res, f"{c_label}_s{s}", c_label, fresh_model, tokenizer, template_prior_controls, wikitext_slice, slice_sha, device)
            per_seed_a[s] = ev_a
            print(f"    Seed {s}: ImmEff={format_wilson_rate(ev_a['immediate_efficacy'])} | TermRet={format_wilson_rate(ev_a['terminal_retention'])} | Steps={ev_a['optimizer_steps']}")
            del m_arm, subspace_unapplied; gc.collect(); torch.cuda.empty_cache()
        cond_results[c_label] = per_seed_a

    print("\n--- [Extended 3-Arm Positive Control Re-Confirmation (Seeds 0-2, delta=0)] ---")
    pos_ctrl_verdicts = {}
    r0_d0 = cond_results["r0_unconstrained_d0.0"]
    r0_imm_num = sum(r0_d0[s]["immediate_efficacy"].numerator for s in [0, 1, 2])
    r0_st = [r0_d0[s]["optimizer_steps"] for s in [0, 1, 2]]; r0_term = sum(r0_d0[s]["terminal_retention"].numerator for s in [0, 1, 2])
    p_a_pass = (r0_imm_num == 600 and sum(r0_st) == 1989 and r0_st == [669, 664, 656] and r0_term == 33)
    pos_ctrl_verdicts["Arm A (delta=0)"] = "PASSED" if p_a_pass else "FAILED"
    print(f"  Arm A (delta=0) : ImmEff={r0_imm_num}/600 | Steps={sum(r0_st)} ({r0_st}) | TermRet={r0_term}/600 -> {pos_ctrl_verdicts['Arm A (delta=0)']}")

    r1_b = cond_results["r1_causal_perstep_d0.0"]
    r1_imm_num = sum(r1_b[s]["immediate_efficacy"].numerator for s in [0, 1, 2])
    r1_st = [r1_b[s]["optimizer_steps"] for s in [0, 1, 2]]; r1_term = sum(r1_b[s]["terminal_retention"].numerator for s in [0, 1, 2])
    p_b_pass = (r1_imm_num == 595 and sum(r1_st) == 3128 and r1_st == [1065, 1049, 1014] and r1_term == 36)
    pos_ctrl_verdicts["Arm B (delta=0)"] = "PASSED" if p_b_pass else "FAILED"
    print(f"  Arm B (delta=0) : ImmEff={r1_imm_num}/600 | Steps={sum(r1_st)} ({r1_st}) | TermRet={r1_term}/600 -> {pos_ctrl_verdicts['Arm B (delta=0)']}")

    r1_f = cond_results["r1_magnitude_only_d0.0"]
    rf_imm_num = sum(r1_f[s]["immediate_efficacy"].numerator for s in [0, 1, 2])
    rf_st = [r1_f[s]["optimizer_steps"] for s in [0, 1, 2]]; rf_term = sum(r1_f[s]["terminal_retention"].numerator for s in [0, 1, 2])
    p_f_pass = (rf_imm_num == 599 and sum(rf_st) == 2489 and rf_st == [844, 824, 821] and rf_term == 36)
    pos_ctrl_verdicts["Arm F (delta=0)"] = "PASSED" if p_f_pass else "FAILED"
    print(f"  Arm F (delta=0) : ImmEff={rf_imm_num}/600 | Steps={sum(rf_st)} ({rf_st}) | TermRet={rf_term}/600 -> {pos_ctrl_verdicts['Arm F (delta=0)']}")
    if not (p_a_pass and p_b_pass and p_f_pass):
        print("FATAL: Positive control gate failed. Halting."); sys.exit(1)

    print("\n--- [Gate Evaluation Table (Pooled Immediate Efficacy >= 90 pct)] ---")
    gate_table_border, gate_table_sep = "=" * 115, "-" * 115
    print(gate_table_border); print(f"{'Condition':<28s} | {'Pooled Imm Efficacy':<36s} | {'Mean Margin':<14s} | {'Verdict'}")
    print(gate_table_sep)
    gate_data = {}; all_conditions = ["r0_unconstrained_d0.0", "r0_unconstrained_d1.0", "r0_unconstrained_d3.0", "r0_unconstrained_d6.0", "r1_causal_perstep_d0.0", "r1_magnitude_only_d0.0"]
    for c_k in all_conditions:
        rec = cond_results[c_k]
        p_imm = [x for s in SEEDS for x in rec[s]["immediate_matches"]]
        m_imm = Measurement.from_outcomes(p_imm, metric="immediate_efficacy", arm=c_k, scope="pooled", input_set="eval_seq_1200", mode="eval_no_dropout")
        mean_mg = sum(m for s in SEEDS for m in rec[s]["margins_achieved"]) / 1200.0
        passed = (m_imm.pct >= 90.0)
        v_str = "GATE: PASSED" if passed else "GATE: FAILED"
        gate_data[c_k] = {"measurement": m_imm, "passed": passed, "mean_margin": mean_mg}
        print(f"{c_k:<28s} | {format_wilson_rate(m_imm):<36s} | {mean_mg:<14.4f} | {v_str}")
    print(gate_table_border)

    print("\n--- [Primary Deliverables Table (All Conditions, N=1200 across 6 Seeds)] ---")
    prim_border, prim_sep = "=" * 145, "-" * 145
    print(prim_border)
    print(f"{'Condition':<26s} | {'Imm Efficacy':<30s} | {'Terminal Ret':<30s} | {'Gen (3xN)':<12s} | {'Loc KL':<8s} | {'Wiki PPL':<9s} | {'Gate'}")
    print(prim_sep)
    primary_panel_data = {}
    for c_k in all_conditions:
        rec = cond_results[c_k]
        m_imm = gate_data[c_k]["measurement"]
        p_term = [x for s in SEEDS for x in rec[s]["terminal_matches"]]
        m_term = Measurement.from_outcomes(p_term, metric="terminal_retention", arm=c_k, scope="pooled", input_set="eval_seq_1200", mode="eval_no_dropout")
        gen_k = sum(rec[s]["generalization"].numerator for s in SEEDS)
        m_gen = Measurement.from_outcomes([True]*gen_k + [False]*(3600-gen_k), metric="generalization", arm=c_k, scope="generalization_pooled", input_set="eval_seq_3600", mode="eval_no_dropout")
        loc_kl = sum(rec[s]["locality_kl"] for s in SEEDS) / 6.0
        ppl = sum(rec[s]["perplexity"] for s in SEEDS) / 6.0
        primary_panel_data[c_k] = {"imm_eff": m_imm, "term_ret": m_term, "gen": m_gen, "loc_kl": loc_kl, "ppl": ppl}
        g_v = "PASSED" if gate_data[c_k]["passed"] else "FAILED"
        print(f"{c_k:<26s} | {format_wilson_rate(m_imm):<30s} | {format_wilson_rate(m_term):<30s} | {m_gen.numerator}/{m_gen.denominator:<9d} | {loc_kl:<8.4f} | {ppl:<9.2f} | {g_v}")
    print(prim_border)

    print("\n--- [Dual Retention Reporting for Failed Gates (3a Conditional & 3b Matched-Subset)] ---")
    print(gate_table_border)
    print(f"{'Condition':<28s} | {'Type':<16s} | {'Retention Rate (Wilson 95-pct CI)':<42s} | {'Subset Count'}")
    print(gate_table_sep)
    d0_rec = cond_results["r0_unconstrained_d0.0"]
    conditional_data = {}
    base_arm_lbl = "r0_unconstrained_d0.0"
    for c_k in all_conditions:
        if not gate_data[c_k]["passed"]:
            rec = cond_results[c_k]
            succ_mask = [rec[s]["immediate_matches"][i] for s in SEEDS for i in range(200)]
            n_succ = sum(1 for x in succ_mask if x)
            ret_cond = [rec[s]["terminal_matches"][i] for s in SEEDS for i in range(200) if rec[s]["immediate_matches"][i]]
            k_cond = sum(1 for x in ret_cond if x)
            lo_c, hi_c = wilson_confidence_interval(k_cond, n_succ) if n_succ > 0 else (0.0, 0.0)
            ret_matched = [d0_rec[s]["terminal_matches"][i] for s in SEEDS for i in range(200) if rec[s]["immediate_matches"][i]]
            k_match = sum(1 for x in ret_matched if x)
            lo_m, hi_m = wilson_confidence_interval(k_match, n_succ) if n_succ > 0 else (0.0, 0.0)
            conditional_data[c_k] = {"n_succ": n_succ, "cond_k": k_cond, "cond_pct": 100.0*k_cond/n_succ if n_succ > 0 else 0.0, "cond_lo": lo_c, "cond_hi": hi_c, "match_k": k_match, "match_pct": 100.0*k_match/n_succ if n_succ > 0 else 0.0, "match_lo": lo_m, "match_hi": hi_m}
            pct_c = 100.0 * k_cond / n_succ if n_succ > 0 else 0.0
            pct_m = 100.0 * k_match / n_succ if n_succ > 0 else 0.0
            print(f"{c_k:<28s} | 3a Conditional   | {k_cond}/{n_succ} ({pct_c:.2f} pct) [{lo_c*100.0:.2f} pct, {hi_c*100.0:.2f} pct] | N={n_succ}")
            print(f"{base_arm_lbl:<28s} | 3b Matched-Sub   | {k_match}/{n_succ} ({pct_m:.2f} pct) [{lo_m*100.0:.2f} pct, {hi_m*100.0:.2f} pct] | N={n_succ}")
    print(gate_table_border)

    print("\n--- [Recency Profile (20 Bins of 10 Edits across 6 Seeds, N=60 per Bin)] ---")
    rec_hdr = f"{'Bin (Edits)':<16s} | {'Arm A d0':<18s} | {'Arm A d1':<18s} | {'Arm A d3':<18s} | {'Arm A d6':<18s} | {'Arm B d0'}"
    print(gate_table_border); print(rec_hdr)
    print(gate_table_sep)
    bin_measurements = {c_k: [] for c_k in all_conditions}
    for b in range(20):
        s_ed, e_ed = b * 10, (b + 1) * 10
        row_str = f"Edits {s_ed+1:03d}-{e_ed:03d}   "
        for c_k in ["r0_unconstrained_d0.0", "r0_unconstrained_d1.0", "r0_unconstrained_d3.0", "r0_unconstrained_d6.0", "r1_causal_perstep_d0.0"]:
            b_out = [cond_results[c_k][s]["terminal_matches"][i] for s in SEEDS for i in range(s_ed, e_ed)]
            m_bin = Measurement.from_outcomes(b_out, metric=f"recency_bin_{b}", arm=c_k, scope="recency_bin", input_set=f"bin_{b}", mode="eval_no_dropout")
            bin_measurements[c_k].append(m_bin)
            row_str += f" | {m_bin.numerator}/{m_bin.denominator:<13d}"
        print(row_str)
    print(gate_table_border)
    for c_k in ["r0_unconstrained_d0.0", "r0_unconstrained_d1.0", "r0_unconstrained_d3.0", "r0_unconstrained_d6.0", "r1_causal_perstep_d0.0"]:
        sum_b_num = sum(m.numerator for m in bin_measurements[c_k])
        tot_num = primary_panel_data[c_k]["term_ret"].numerator
        assert sum_b_num == tot_num, f"Bin sum check failed for {c_k}: {sum_b_num} != {tot_num}"
    print("  Recency Profile Bin Sum Check: All 20 bins sum exactly to pooled terminal retention (PASSED).")

    print("\n--- [Monotone Retention Horizon Search (Directive S0-6)] ---")
    print(gate_table_border); print(f"{'Condition':<28s} | {'Monotone Horizon k':<20s} | {'Separated at Horizon':<24s} | {'Remainder Retention'}")
    print(gate_table_sep)
    horizons_data = {}
    for c_k in all_conditions:
        t_matches_dict = {s: cond_results[c_k][s]["terminal_matches"] for s in SEEDS}
        h_res = compute_monotone_retention_horizon(t_matches_dict, floor_interval=worst_ctrl_wilson, step_size=10, total_edits=200)
        horizons_data[c_k] = h_res; hk, rem = h_res["horizon_k"], h_res["remainder"]
        rem_str = f"{rem['numerator']}/{rem['denominator']} ({rem['rate']*100.0:.2f} pct)" if rem else "N/A"
        print(f"{c_k:<28s} | k = {hk:<16d} | {'YES' if hk > 0 else 'NO':<24s} | {rem_str}")
    print(gate_table_border)

    print("\n--- [Tradeoff Curve: Retention Horizon vs WikiText-2 Perplexity Damage] ---")
    base_ppl = 36.03
    print(gate_table_border); print(f"{'Margin delta':<16s} | {'Horizon k':<14s} | {'PPL Mean':<12s} | {'Delta PPL Damage':<20s} | {'Locality KL'}"); print(gate_table_sep)
    tradeoff_data = []
    for d in MARGINS:
        c_k = f"r0_unconstrained_d{d:.1f}"
        hk = horizons_data[c_k]["horizon_k"]; p_m, loc_m = primary_panel_data[c_k]["ppl"], primary_panel_data[c_k]["loc_kl"]
        dmg = p_m - base_ppl; tradeoff_data.append({"delta": d, "horizon_k": hk, "ppl": p_m, "damage": dmg, "locality_kl": loc_m})
        print(f"delta = {d:<10.1f} | k = {hk:<10d} | {p_m:<12.2f} | {dmg:<20.2f} | {loc_m:.4f}")
    print(gate_table_border)

    print("\n--- [Paired Statistical Inference (df=5 across 6 seeds)] ---")
    paired_border, paired_sep = "=" * 125, "-" * 125
    print(paired_border); print(f"{'Comparison':<34s} | {'Metric':<14s} | {'Mean Diff':<12s} | {'Std Diff':<12s} | {'t-stat (df=5)':<14s} | {'Wilcoxon W':<12s} | {'Sign Agree'}")
    print(paired_sep)
    comparisons = [
        ("r1_causal_perstep_d0.0", "r0_unconstrained_d0.0", "Arm B vs Arm A (delta=0)"),
        ("r1_causal_perstep_d0.0", "r1_magnitude_only_d0.0", "Arm B vs Arm F (delta=0)"),
        ("r1_magnitude_only_d0.0", "r0_unconstrained_d0.0", "Arm F vs Arm A (delta=0)"),
        ("r0_unconstrained_d1.0", "r0_unconstrained_d0.0", "Arm A (delta=1) vs Arm A (delta=0)"),
        ("r0_unconstrained_d3.0", "r0_unconstrained_d0.0", "Arm A (delta=3) vs Arm A (delta=0)"),
        ("r0_unconstrained_d6.0", "r0_unconstrained_d0.0", "Arm A (delta=6) vs Arm A (delta=0)"),
    ]
    paired_results = []
    for c1, c2, lbl in comparisons:
        for m_name in ["terminal_retention", "perplexity", "locality_kl"]:
            v1 = [cond_results[c1][s][m_name].numerator if hasattr(cond_results[c1][s][m_name], "numerator") else cond_results[c1][s][m_name] for s in SEEDS]
            v2 = [cond_results[c2][s][m_name].numerator if hasattr(cond_results[c2][s][m_name], "numerator") else cond_results[c2][s][m_name] for s in SEEDS]
            st = compute_paired_stats(v1, v2)
            sign_agree = (st["t_stat"] >= 0 and st["mean_diff"] >= 0) or (st["t_stat"] <= 0 and st["mean_diff"] <= 0)
            paired_results.append({"label": lbl, "metric": m_name, "stats": st, "sign_agreement": sign_agree})
            print(f"{lbl:<34s} | {m_name:<14s} | {st['mean_diff']:<12.4f} | {st['std_diff']:<12.4f} | {st['t_stat']:<14.4f} | {st['wilcoxon_stat']:<12.1f} | {'YES' if sign_agree else 'NO'}")
    print(paired_border)

    print("\n--- [Diagnostics Table (Per-Condition Optimizer Steps & Geometry)] ---")
    print(paired_border); print(f"{'Condition':<28s} | {'Total Steps':<14s} | {'Mean Steps':<14s} | {'Exhausted':<12s} | {'Row SF / Align':<20s} | {'Matrix SF / Align'}")
    print(paired_sep)
    diag_summary = {}
    for c_k in all_conditions:
        rec = cond_results[c_k]
        tot_st = sum(rec[s]["optimizer_steps"] for s in SEEDS)
        mean_st = tot_st / 1200.0
        exh_cnt = sum(rec[s]["exhausted_count"] for s in SEEDS)
        sf_r = sum(rec[s]["sf_row_mean"] for s in SEEDS) / 6.0; al_r = sum(rec[s]["al_row_mean"] for s in SEEDS) / 6.0
        sf_m = sum(rec[s]["sf_mat_mean"] for s in SEEDS) / 6.0; al_m = sum(rec[s]["al_mat_mean"] for s in SEEDS) / 6.0
        diag_summary[c_k] = {"total_steps": tot_st, "mean_steps": mean_st, "exhausted": exh_cnt, "sf_row": sf_r, "al_row": al_r, "sf_mat": sf_m, "al_mat": al_m}
        print(f"{c_k:<28s} | {tot_st:<14d} | {mean_st:<14.2f} | {exh_cnt:<12d} | {sf_r:.4f} / {al_r:.4f}     | {sf_m:.4f} / {al_m:.4f}")
    print(paired_border)

    print("\n--- [Structural Invariance: Float64 Sum & Frobenius Norm Audit] ---")
    chk_sums, chk_frobs = {}, {}
    for c_k in seed0_first_edit_updates:
        upd = seed0_cumulative_updates[c_k].to(torch.float64)
        s_val = float(upd.sum().item())
        f_val = float(torch.linalg.matrix_norm(upd, ord="fro").item())
        chk_sums[c_k], chk_frobs[c_k] = s_val, f_val
        print(f"  Condition {c_k:<28s} : Signed Sum = {s_val:14.6f} | Frobenius Norm = {f_val:12.6f}")
    assert len(set(chk_sums.values())) == len(chk_sums) and len(set(chk_frobs.values())) == len(chk_frobs)
    print("  Structural Invariance Outcome: PASSED (Distinct updates across all conditions).")

    print("\n--- [Line-Item Step Attribution Accounting Table] ---")
    line_border, line_sep = "=" * 95, "-" * 95
    print(line_border); print(f"{'Item / Subsystem':<36s} | {'Seed':<6s} | {'Steps Consumed':<16s} | {'Accounting Note'}"); print(line_sep)
    sum_line_items = 0
    for li in line_item_steps:
        sum_line_items += li["steps"]; seed_label = f"{li['seed']:<6d}" if li["seed"] >= 0 else "ALL   "
        shared_text = "Shared positive control B2" if li["shared"] else "Primary execution"
        print(f"{li['item']:<36s} | {seed_label} | {li['steps']:<16d} | {shared_text}")
    print(line_sep); print(f"{'Sum of Line-Item Steps':<36s} | {'ALL':<6s} | {sum_line_items:<16d} | Sum")
    print(f"{'Global Optimizer Steps Counter':<36s} | {'ALL':<6s} | {total_optimizer_steps_global:<16d} | Global tally")
    step_diff = total_optimizer_steps_global - sum_line_items
    print(f"{'Unexplained Attribution Delta':<36s} | {'ALL':<6s} | {step_diff:<16d} | Delta == 0"); print(line_border)
    assert step_diff == 0, f"Unexplained optimizer step delta: {step_diff} != 0"
    print("\n--- [Step vs Sample Counter Derivation] ---")
    print("  Counter Derivation: Batch size = 1 (each SGD step processes exactly 1 fact prompt).")
    print("  Relationship      : Total Samples Seen is identically derived from Total Optimizer Steps (1 sample / step).")

    actual_wall_clock = time.time() - start_time
    print(f"\n--- [Wall-Clock Budget Audit] ---")
    print(f"  Projected Compute Wall-Clock: {projected_seconds:.1f} s")
    print(f"  Actual Compute Wall-Clock   : {actual_wall_clock:.1f} s")
    producing_commit = "DIRTY"
    try: producing_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: pass

    res_data = {
        "directive": "S0-6", "producing_commit_sha": producing_commit, "exit_code": 0,
        "hashes": {"facts_json_sha256": facts_sha, "wikitext_slice_sha256": slice_sha, "weight_file_sha256": weight_sha, "control_probes_sha256": ctrl_probe_sha},
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "pinned_revision": pinned_revision, "fresh_checksum": fresh_checksum},
        "sequence_hashes": {f"seed_{s}": seq_hashes[s] for s in SEEDS}, "multitoken_fractions": {"pinned_1000": [multi_1000, 1000]},
        "pilot_timing": pilot_times, "inflation_factors": inflation_factors, "controls_pooled": {c: pooled_ctrl_measures[c].pair for c in CONTROL_NAMES},
        "worst_control": {"name": worst_ctrl_all.name, "pair": worst_ctrl_all.pair}, "positive_controls": pos_ctrl_verdicts,
        "gate_s0_6": {c: {"imm_eff": gate_data[c]["measurement"].pair, "passed": gate_data[c]["passed"], "mean_margin": gate_data[c]["mean_margin"]} for c in gate_data},
        "primary_panel": {c: {k: (primary_panel_data[c][k].pair if hasattr(primary_panel_data[c][k], "pair") else primary_panel_data[c][k]) for k in primary_panel_data[c]} for c in primary_panel_data},
        "conditional_retention": conditional_data,
        "recency_profile": {c: [m.pair for m in bin_measurements[c]] for c in bin_measurements},
        "monotone_horizons": {c: {"horizon_k": horizons_data[c]["horizon_k"], "remainder": horizons_data[c]["remainder"]} for c in horizons_data},
        "tradeoff_curve": tradeoff_data,
        "paired_statistics": paired_results,
        "diagnostics": diag_summary,
        "structural_invariance": {"sums": chk_sums, "frobenius": chk_frobs},
        "step_attribution": {"line_items": line_item_steps, "sum_line_items": sum_line_items, "global_counter": total_optimizer_steps_global, "delta": step_diff},
        "accounting": {"total_optimizer_steps": total_optimizer_steps_global, "total_samples_seen": total_optimizer_steps_global, "projected_wall_clock": projected_seconds, "actual_wall_clock": actual_wall_clock},
        "edits_pythagorean_checked": total_edits_pythagorean_asserted
    }

    out_dir = REPO_ROOT / "experiments" / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s0_6.json"
    with open(out_file, "w", encoding="utf-8") as f: json.dump(res_data, f, indent=2)
    print(f"\n  Artifact Written            : {out_file.relative_to(REPO_ROOT)}")
    print("\n" + "=" * 115); print(" DIRECTIVE S0-6 COMPLETE: MARGIN SWEEP, HORIZON READOUT, AND PAIRED INFERENCE EXECUTED"); print("=" * 115)
    print("SCRIPT_EXIT=0"); sys.exit(0)

if __name__ == "__main__":
    main()
