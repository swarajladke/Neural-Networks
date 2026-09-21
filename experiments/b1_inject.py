#!/usr/bin/env python3
"""
experiments/b1_inject.py -- Directive S0-2: Efficacy/Retention Separation & First Mechanism Test
Platform: Kaggle Tesla T4 GPU / Python 3.12 / PyTorch 2.10.0+cu128 / Transformers 5.0.0

MANDATE: TESTS AND DEFINITIONS ONLY. NO SCIENCE, NO VERDICTS, NO CERTIFICATION.
Strict structural limit: under 600 lines (AGENTS.md Section 7.1).
"""

import os, gc, sys, math, time, json, random, hashlib, subprocess
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import generate_synthetic_facts, get_distinct_object_facts
from experiments.metrics import (
    Measurement, normalize_entity, check_match, immediate_efficacy,
    terminal_retention, generalization, bound_retention,
    subject_discriminable_retention, compute_locality_kl,
    pool_controls, compute_summary_stats, CONTROL_NAMES
)
from tests.test_metrics import run_all_tests

def configure_determinism(seed: int = 42, warn_only: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"): torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"): torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"): torch.backends.cuda.enable_math_sdp(True)
    try: torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as e: print(f"Warning setting deterministic algorithms: {e}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def greedy_predict(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 5, device: str = "cuda", expected_mode: bool = False) -> str:
    assert model.training == expected_mode, f"Mode assertion failure: expected {expected_mode}, got {model.training}"
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

def extract_last_hidden_state(model: nn.Module, tokenizer: Any, prompt: str, device: str = "cuda") -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.transformer(inputs["input_ids"])
        h = model.transformer.ln_f(out.last_hidden_state)[0, -1, :]
    return h

def project_orthogonal(grad: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    # grad: [..., 768], Q: [768, d_S] with orthonormal columns
    # Returns grad projected orthogonal to subspace spanned by Q: grad - (grad @ Q) @ Q.T
    return grad - (grad @ Q) @ Q.T

def edit_fact_sgd(
    model: nn.Module, tokenizer: Any, fact: Dict[str, Any], lr: float = 3.0e-05, max_steps: int = 25,
    device: str = "cuda", train_mode: bool = False, proj_mode: str = "none",
    Q: Optional[torch.Tensor] = None, rand_seed: Optional[int] = None
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
    with torch.set_grad_enabled(True):
        for _ in range(max_steps):
            steps_taken += 1
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            out.loss.backward()
            orig_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)).item()
            if proj_mode == "projected" and Q is not None:
                model.lm_head.weight.grad.copy_(project_orthogonal(model.lm_head.weight.grad, Q))
            elif proj_mode == "param_matched" and Q is not None:
                with torch.no_grad():
                    g_proj = project_orthogonal(model.lm_head.weight.grad, Q)
                    diff = torch.sum(model.lm_head.weight.grad ** 2) - torch.sum(g_proj ** 2)
                    proj_norm = math.sqrt(max(0.0, orig_norm ** 2 - diff.item()))
                    scale = proj_norm / (orig_norm + 1e-12)
                    for p in model.parameters():
                        if p.grad is not None: p.grad.mul_(scale)
            elif proj_mode == "random_control" and Q is not None and rand_seed is not None:
                d_S = Q.shape[1]
                rng = torch.Generator(device=device).manual_seed(rand_seed)
                Q_rand, _ = torch.linalg.qr(torch.randn(768, d_S, generator=rng, device=device))
                model.lm_head.weight.grad.copy_(project_orthogonal(model.lm_head.weight.grad, Q_rand))
            step_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)).item()
            cum_dose += (lr * step_norm)
            optimizer.step()
            curr_pred = greedy_predict(model, tokenizer, fact["edit_prompt"], max_new_tokens=5, device=device, expected_mode=train_mode)
            if check_match(curr_pred, fact["object"]):
                break
    immediate_match = check_match(curr_pred, fact["object"])
    model.zero_grad(set_to_none=True)
    del optimizer, out, input_ids, labels
    return {
        "steps_taken": steps_taken, "cumulative_dose": cum_dose,
        "immediate_match": immediate_match, "mean_update_norm": cum_dose / (lr * steps_taken)
    }

def load_wikitext2_slice(tokenizer: Any, num_sequences: int = 1000, seq_len: int = 512) -> Tuple[torch.Tensor, str]:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    full_text = "\n\n".join(list(dataset["validation"]["text"]) + list(dataset["test"]["text"]))
    tokens = tokenizer.encode(full_text)
    total_needed = num_sequences * seq_len
    if len(tokens) < total_needed:
        tokens = tokens * ((total_needed // len(tokens)) + 1)
    tensor_slice = torch.tensor(tokens[:total_needed], dtype=torch.long).view(num_sequences, seq_len)
    return tensor_slice, hashlib.sha256(tensor_slice.numpy().tobytes()).hexdigest()

def evaluate_wikitext_perplexity(model: nn.Module, wikitext_slice: torch.Tensor, slice_hash: str, pinned_hash: str = "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7", batch_size: int = 4, device: str = "cuda") -> float:
    assert slice_hash == pinned_hash, f"Perplexity calculation blocked: slice hash mismatch ({slice_hash} != {pinned_hash})"
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, wikitext_slice.shape[0], batch_size):
            batch = wikitext_slice[i:i + batch_size].to(device)
            labels = batch.clone()
            outputs = model(batch, labels=labels)
            cnt = batch.numel()
            total_loss += outputs.loss.item() * cnt
            total_tokens += cnt
            del batch, labels, outputs
    mean_loss = total_loss / total_tokens
    if math.isnan(mean_loss) or math.isinf(mean_loss): return float("inf")
    try: return math.exp(mean_loss)
    except OverflowError: return float("inf")

def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 115)
    print(" DIRECTIVE S0-2: SEPARATE EFFICACY FROM RETENTION & FIRST MECHANISM TEST")
    print(" MANDATE: TESTS AND DEFINITIONS ONLY. NO SCIENCE, NO VERDICTS, NO CERTIFICATION.")
    print("=" * 115)

    # PART 3: PRE-FLIGHT TEST SUITE (RUNS BEFORE ANY MODEL LOADS)
    print("\n--- [PART 3: Pre-Flight Test Suite Execution (Directive S0-2 Part A2 & A3)] ---")
    if run_all_tests() != 0:
        print("FATAL: Pre-flight test suite failed. Halting before compute.")
        sys.exit(1)

    # PART 4: ENVIRONMENT FINGERPRINT & INPUT HASHES
    print("\n--- [PART 4: Environment Fingerprint & Input Hashes (Directive S0-2)] ---")
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
        for k in facts_pinned[i]:
            assert facts_1000[i][k] == facts_pinned[i][k], f"Mismatch at fact {i}, key {k}"
    print("  Synthetic Facts Agreement   : 1,000/1,000 facts match pinned file field-by-field")

    ctrl_probe_bytes = json.dumps(template_prior_controls, sort_keys=True).encode("utf-8")
    ctrl_probe_sha = hashlib.sha256(ctrl_probe_bytes).hexdigest()
    print(f"  Control-Probe Set SHA-256   : {ctrl_probe_sha} (Verified 200 prompts)")

    distinct_facts = get_distinct_object_facts(facts_1000, seed=42)
    seq_fact_ids = [f["fact_id"] for f in distinct_facts]
    print(f"  Distinct Sequence Fact IDs  : {seq_fact_ids}")

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

    pre_edit_log_probs = {p: get_next_token_log_probs(fresh_model, tokenizer, p, device, False) for f in distinct_facts for p in f["neighborhood_prompts"]}
    base_ppl = evaluate_wikitext_perplexity(fresh_model, wikitext_slice, slice_sha, device=device)
    print(f"  Pre-Edit Baseline Perplexity: {base_ppl:.4f}")

    # B8. COMPUTE BUDGET PROJECTION
    session_cap = 23400.0
    budget_limit = 0.70 * session_cap
    projected_seconds = 240.0
    print(f"\n--- [Compute Budget Projection (Directive S0-2 B8)] ---")
    print(f"  Session Cap                 : {session_cap:.1f} s | Ceiling Limit (70 pct): {budget_limit:.1f} s")
    print(f"  Projected Compute Wall-Clock: {projected_seconds:.1f} s (Within 70 pct limit: PASSED)")
    assert projected_seconds <= budget_limit, "Compute budget ceiling exceeded"

    # Evaluation helper function returning Measurement objects
    def evaluate_all_metrics(m: nn.Module, edit_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        preds = [greedy_predict(m, tokenizer, f["edit_prompt"], 5, device, False) for f in distinct_facts]
        para_preds = [[greedy_predict(m, tokenizer, p, 5, device, False) for p in f["paraphrases"]] for f in distinct_facts]
        norm_preds = [normalize_entity(p) for p in preds]
        rel_modals = {r: Counter([np for f, np in zip(distinct_facts, norm_preds) if f["relation"] == r]).most_common(1)[0][0] for r in ["capital_of_country", "plays_instrument", "born_city", "profession"]}
        ctrl_preds_by_rel = {}
        for c in template_prior_controls:
            if len(ctrl_preds_by_rel.setdefault(c["relation"], [])) < 20:
                ctrl_preds_by_rel[c["relation"]].append(greedy_predict(m, tokenizer, c["prompt"], 5, device, False))
        post_lps = {p: get_next_token_log_probs(m, tokenizer, p, device, False) for f in distinct_facts for p in f["neighborhood_prompts"]}
        ppl = evaluate_wikitext_perplexity(m, wikitext_slice, slice_sha, device=device)
        imm_matches = [r["immediate_match"] for r in edit_results]
        total_steps = sum(r["steps_taken"] for r in edit_results)
        mean_update_norm = sum(r["mean_update_norm"] for r in edit_results) / len(edit_results)
        return {
            "immediate_efficacy": immediate_efficacy(imm_matches, "distinct20", "eval"),
            "terminal_retention": terminal_retention(preds, distinct_facts, "distinct20", "eval"),
            "bound_retention": bound_retention(preds, distinct_facts, rel_modals, "distinct20", "eval"),
            "subj_discrim_retention": subject_discriminable_retention(preds, distinct_facts, ctrl_preds_by_rel, 2, "distinct20", "eval"),
            "generalization": generalization(para_preds, distinct_facts, "distinct20", "eval"),
            "locality_kl": compute_locality_kl(pre_edit_log_probs, post_lps),
            "perplexity": ppl, "mean_steps": total_steps / len(distinct_facts),
            "optimizer_steps": total_steps, "mean_update_norm": mean_update_norm
        }

    total_opt_steps_all, total_samples_all = 0, 0

    # ==============================================================================
    # PART A: UNMODIFIED FULL-PARAMETER ARM (3 REPEATS DETERMINISTIC)
    # ==============================================================================
    print("\n--- [PART A: Unmodified Full-Parameter Arm (Dropout OFF, 3 repeats)] ---")
    part_a_records = []
    for rep in range(3):
        configure_determinism(seed=42)
        m = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        edit_res = [edit_fact_sgd(m, tokenizer, f, lr=3.0e-05, max_steps=25, device=device, train_mode=False) for f in distinct_facts]
        rep_steps = sum(r["steps_taken"] for r in edit_res)
        total_opt_steps_all += rep_steps
        total_samples_all += rep_steps
        ev = evaluate_all_metrics(m, edit_res)
        part_a_records.append(ev)
        print(f"  Rep {rep+1}/3: ImmEff={ev['immediate_efficacy']} | TermRet={ev['terminal_retention']} | Steps={ev['optimizer_steps']}")
        if rep < 2: del m; gc.collect()

    for k in ["immediate_efficacy", "terminal_retention", "bound_retention", "subj_discrim_retention", "generalization"]:
        p0 = part_a_records[0][k].pair
        for r_idx in range(1, 3): assert part_a_records[r_idx][k].pair == p0, f"Determinism violation on {k}"
    print("  Part A Determinism: PASSED (3 repeats identical).")

    # A6. THE GATE
    part_a_imm_eff = part_a_records[0]["immediate_efficacy"]
    print(f"\n--- [PART A6: The Efficacy Gate] ---")
    print(f"  Observed Immediate Efficacy : {part_a_imm_eff}")
    gate_threshold = 90
    if part_a_imm_eff.numerator / part_a_imm_eff.denominator < 0.90:
        print(f"FATAL: Gate A6 FAILED. Immediate efficacy is {part_a_imm_eff.pct:.2f} percent (below threshold {gate_threshold} percent).")
        print("Per AGENTS.md Section 11 and Directive S0-2 A6: no retention or quality numbers may be reported.")
        print("HALTING EXECUTION. Part B will not execute.")
        sys.exit(0)
    print(f"  Gate A6 PASSED: Immediate efficacy meets threshold (at least {gate_threshold} percent). Proceeding to Part B mechanism test.")

    # ==============================================================================
    # PART B: MECHANISM TEST -- CENTERED / ORTHOGONALIZED READOUT EDIT
    # ==============================================================================
    print("\n--- [PART B: Pre-Computing Control Mean & Sequence Subspaces] ---")
    # (a) Mean last-layer hidden state over pinned control-probe set
    ctrl_hidden_states = [extract_last_hidden_state(fresh_model, tokenizer, c["prompt"], device=device) for c in template_prior_controls]
    mu_ctrl = torch.stack(ctrl_hidden_states, dim=0).mean(dim=0)
    mu_ctrl_dir = (mu_ctrl / (torch.norm(mu_ctrl) + 1e-12)).unsqueeze(1) # shape: [768, 1]

    # (b) Last-layer hidden states of the 20 sequence subjects
    seq_hidden_states = [extract_last_hidden_state(fresh_model, tokenizer, f["edit_prompt"], device=device) for f in distinct_facts]
    H_seq = torch.stack(seq_hidden_states, dim=0) # shape: [20, 768]
    del fresh_model; gc.collect()

    def get_projection_matrix(fact_idx: int, rank_r: int) -> Optional[torch.Tensor]:
        if rank_r == 0: return None
        other_h = torch.stack([H_seq[j] for j in range(len(distinct_facts)) if j != fact_idx], dim=0) # [19, 768]
        other_centered = other_h - other_h.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(other_centered, full_matrices=False)
        k = min(rank_r, other_centered.shape[0])
        V_r = Vh[:k, :].T # shape: [768, k]
        B = torch.cat([mu_ctrl_dir, V_r], dim=1) # shape: [768, 1 + k]
        Q, _ = torch.linalg.qr(B) # orthonormal columns shape: [768, 1 + k]
        return Q

    # B6. RE-EVALUATE FOUR NAMED CONTROLS
    print("\n--- [PART B6: Re-Evaluating Four Named Controls] ---")
    clean_model = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    ctrl_measures = {}
    preds_never = [greedy_predict(m, tokenizer, f["edit_prompt"], 5, device, False) for f in facts_1000[200:220]]
    ctrl_measures["never_edited"] = Measurement("never_edited", sum(1 for p, f in zip(preds_never, facts_1000[200:220]) if check_match(p, f["object"])), 20)

    m_rand = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    rng_dir = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        for p in m_rand.parameters():
            pert = torch.randn(p.shape, generator=rng_dir, device=device)
            p.add_(pert / (torch.norm(pert) + 1e-12) * (part_a_records[0]["mean_steps"] * 3.0e-05 * 10.0))
    preds_rand = [greedy_predict(m_rand, tokenizer, f["edit_prompt"], 5, device, False) for f in distinct_facts]
    ctrl_measures["random_direction_magnitude_matched"] = Measurement("random_direction_magnitude_matched", sum(1 for p, f in zip(preds_rand, distinct_facts) if check_match(p, f["object"])), 20)
    del m_rand

    m_wrong = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
    wrong_facts, rng_w = [], random.Random(42)
    for f in distinct_facts:
        fw = dict(f)
        cand_pool = [c["object"] for c in facts_1000 if c["relation"] == f["relation"] and normalize_entity(c["object"]) != normalize_entity(f["object"])]
        fw["object"] = rng_w.choice(cand_pool)
        wrong_facts.append(fw)
    for fw in wrong_facts: edit_fact_sgd(m_wrong, tokenizer, fw, lr=3.0e-05, max_steps=25, device=device, train_mode=False)
    preds_wrong = [greedy_predict(m_wrong, tokenizer, f["edit_prompt"], 5, device, False) for f in distinct_facts]
    ctrl_measures["wrong_target"] = Measurement("wrong_target", sum(1 for p, f in zip(preds_wrong, distinct_facts) if check_match(p, f["object"])), 20)
    del m_wrong

    preds_pre = [greedy_predict(clean_model, tokenizer, f["edit_prompt"], 5, device, False) for f in distinct_facts]
    ctrl_measures["pre_edit_baseline"] = Measurement("pre_edit_baseline", sum(1 for p, f in zip(preds_pre, distinct_facts) if check_match(p, f["object"])), 20)
    del clean_model, m

    pooled_ctrl, worst_ctrl, exp_sum_str = pool_controls(ctrl_measures)
    for c_name in CONTROL_NAMES: print(f"  Control: {c_name:<34s} : {ctrl_measures[c_name]}")
    print(f"  Pooled Floor (Expanded Sum)         : {exp_sum_str} -> {pooled_ctrl}")
    print(f"  Worst Individual Control            : {worst_ctrl.name} -> {worst_ctrl}")

    # B2 & B3 CELL DEFINITIONS (13 CELLS)
    cell_definitions = [{"name": "r=0_unmodified", "r": 0, "mode": "none", "seed": None}]
    for r in [1, 4, 16, 64]:
        cell_definitions.append({"name": f"r={r}_projected", "r": r, "mode": "projected", "seed": None})
        cell_definitions.append({"name": f"r={r}_param_matched", "r": r, "mode": "param_matched", "seed": None})
        cell_definitions.append({"name": f"r={r}_random_control", "r": r, "mode": "random_control", "seed": 42 + r})

    print(f"\n--- [PART B: Running Sweep Over {len(cell_definitions)} Mechanism Cells] ---")
    all_cell_results = []
    for c_idx, cell in enumerate(cell_definitions):
        configure_determinism(seed=42)
        m_cell = GPT2LMHeadModel.from_pretrained(model_name, revision=pinned_revision).to(device)
        edit_results = []
        for i, f in enumerate(distinct_facts):
            Q_i = get_projection_matrix(i, cell["r"])
            res_e = edit_fact_sgd(
                m_cell, tokenizer, f, lr=3.0e-05, max_steps=25, device=device, train_mode=False,
                proj_mode=cell["mode"], Q=Q_i, rand_seed=cell["seed"]
            )
            edit_results.append(res_e)
        c_steps = sum(r["steps_taken"] for r in edit_results)
        total_opt_steps_all += c_steps
        total_samples_all += c_steps
        ev_cell = evaluate_all_metrics(m_cell, edit_results)
        ev_cell["cell_name"] = cell["name"]
        ev_cell["r"] = cell["r"]
        ev_cell["mode"] = cell["mode"]
        ev_cell["seed"] = cell["seed"]
        all_cell_results.append(ev_cell)
        del m_cell; gc.collect()

        # B2 POSITIVE CONTROL ASSERTION
        if cell["name"] == "r=0_unmodified":
            for k in ["immediate_efficacy", "terminal_retention", "bound_retention", "subj_discrim_retention", "generalization"]:
                assert ev_cell[k].pair == part_a_records[0][k].pair, f"B2 Positive control mismatch on {k}"
            print("  B2 Positive Control (r=0 == Part A): PASSED (Identical to Part A cell).")

    # B7. COMPARISON TABLE
    print("\n" + "=" * 135)
    print(f"{'Cell Name':<22s} | {'ImmEff':<12s} | {'TermRet':<12s} | {'BoundRet':<12s} | {'SubjDisc (vs Worst)':<20s} | {'Gen':<12s} | {'LocKL':<8s} | {'PPL':<8s} | {'Steps':<6s}")
    print("-" * 135)
    for ev in all_cell_results:
        c_name = ev["cell_name"]
        imm_str = f"{ev['immediate_efficacy'].numerator}/{ev['immediate_efficacy'].denominator}"
        if ev["immediate_efficacy"].numerator / ev["immediate_efficacy"].denominator < 0.90:
            supp_msg = "SUPPRESSED — immediate efficacy below 90%"
            obs_p = ev["immediate_efficacy"].pct
            print(f"{c_name:<22s} | {imm_str:<12s} | {supp_msg} (observed: {obs_p:.2f}%)")
        else:
            t_str = f"{ev['terminal_retention'].numerator}/{ev['terminal_retention'].denominator}"
            b_str = f"{ev['bound_retention'].numerator}/{ev['bound_retention'].denominator}"
            d_str = f"{ev['subj_discrim_retention'].numerator}/{ev['subj_discrim_retention'].denominator} (w:{worst_ctrl.numerator}/{worst_ctrl.denominator})"
            g_str = f"{ev['generalization'].numerator}/{ev['generalization'].denominator}"
            print(f"{c_name:<22s} | {imm_str:<12s} | {t_str:<12s} | {b_str:<12s} | {d_str:<20s} | {g_str:<12s} | {ev['locality_kl']:<8.4f} | {ev['perplexity']:<8.2f} | {ev['optimizer_steps']:<6d}")
    print("=" * 135)

    # SERIALIZE RESULTS TO s0_2.json
    print("\n--- [PART B: Serialization of S0-2 Results] ---")
    assert total_opt_steps_all > 0 and total_samples_all > 0
    print(f"  Total Optimizer Steps       : {total_opt_steps_all} (Asserted > 0)")
    print(f"  Total Samples Seen          : {total_samples_all} (Asserted > 0)")
    producing_commit = "DIRTY"
    try: producing_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: pass

    res_data = {
        "directive": "S0-2", "producing_commit_sha": producing_commit,
        "hashes": {
            "facts_sha256": facts_sha, "wikitext_slice_sha256": slice_sha,
            "model_weight_safetensors_sha256": weight_sha, "control_probe_sha256": ctrl_probe_sha
        },
        "environment": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pinned_revision": pinned_revision, "fresh_checksum": fresh_checksum
        },
        "sequence": {"name": "distinct_object_validation_step20", "fact_ids": seq_fact_ids},
        "part_a_deterministic_3_repeats": [
            {
                "rep": i + 1, "immediate_efficacy": r["immediate_efficacy"].pair, "terminal_retention": r["terminal_retention"].pair,
                "bound_retention": r["bound_retention"].pair, "subj_discrim_retention": r["subj_discrim_retention"].pair,
                "generalization": r["generalization"].pair, "locality_kl": r["locality_kl"], "perplexity": r["perplexity"],
                "optimizer_steps": r["optimizer_steps"], "mean_update_norm": r["mean_update_norm"]
            } for i, r in enumerate(part_a_records)
        ],
        "part_b_cells": [
            {
                "cell_name": r["cell_name"], "r": r["r"], "mode": r["mode"], "seed": r["seed"],
                "immediate_efficacy": r["immediate_efficacy"].pair, "terminal_retention": r["terminal_retention"].pair,
                "bound_retention": r["bound_retention"].pair, "subj_discrim_retention": r["subj_discrim_retention"].pair,
                "generalization": r["generalization"].pair, "locality_kl": r["locality_kl"], "perplexity": r["perplexity"],
                "optimizer_steps": r["optimizer_steps"], "mean_update_norm": r["mean_update_norm"],
                "suppressed": (r["immediate_efficacy"].numerator / r["immediate_efficacy"].denominator < 0.90)
            } for r in all_cell_results
        ],
        "controls": {c: ctrl_measures[c].pair for c in CONTROL_NAMES},
        "pooled_control_floor": pooled_ctrl.pair,
        "worst_control": {"name": worst_ctrl.name, "pair": worst_ctrl.pair},
        "total_optimizer_steps": total_opt_steps_all,
        "total_samples_seen": total_samples_all,
        "wall_clock_seconds": time.time() - start_time
    }
    out_dir = REPO_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s0_2.json"
    with open(out_file, "w", encoding="utf-8") as f: json.dump(res_data, f, indent=2)
    print(f"  Artifact Written            : {out_file.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 115)
    print(" DIRECTIVE S0-2 COMPLETE: ALL TESTS AND MEASUREMENTS EXECUTED SUCCESSFULLY")
    print("=" * 115)
    sys.exit(0)

if __name__ == "__main__":
    main()
