"""
agnis_diagnostic_v4_5.py — Forced-Gate Paraphrase Decomposition (V4.5)
=======================================================================
Runs the full V4.5 sequential pipeline and, at each stage, performs a
forced-gate decomposition on every paraphrase of every learned block.

Four diagnostic cases (Sol's taxonomy):
  Case A: Gate OFF, MLP CORRECT  → threshold/margin failure
  Case B: Gate ON,  MLP WRONG    → MLP logit drift (distillation gap)
  Case C: Gate OFF, MLP WRONG    → dual failure (prototype coverage gap)
  Case D: Gate ON,  MLP CORRECT  → working correctly

Also reports:
  - max cosine similarity of each paraphrase query to its 3 prototypes
  - temporal gate-score drift from acquisition stage to final stage
  - target-token rank and probability in MLP's softmax distribution
"""
from __future__ import annotations

import copy, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Re-use all pipeline imports from V4.5
from agnis_continual_v2 import (
    INDEPENDENT_PPL_TEXTS, INJECTION_FACT_TEXTS, RAW_FACTS, RETENTION_PROBES, build_hybrid,
)
from agnis_continual_v4_1 import DEVICE, gpt2_forward, measure_ppl
from agnis_continual_v4_2 import (
    EVAL_PARAPHRASES, TRAIN_PARAPHRASES,
    collect_fact_queries, collect_control_states, train_query_projection,
)
from fact_memory import EpisodicFactMemory, JointSlowMemoryMLP
from replay_sampler import ReplaySampler
from agnis_validation_harness import CLMetricsEvaluator


# ---------------------------------------------------------------------------
# Diagnostic core function
# ---------------------------------------------------------------------------

def diagnose_paraphrase(
    hybrid,
    memory: EpisodicFactMemory,
    mu: torch.Tensor,
    V_sub: torch.Tensor | None,
    para_text: str,
    target_tok: int,
    sampler: ReplaySampler,
    fid: str,
) -> dict:
    """
    Forced-gate decomposition for one paraphrase.
    Returns a diagnostic dict with all metrics Sol requested.
    """
    tokenizer = hybrid.tokenizer
    device = hybrid.device

    ids = tokenizer.encode(para_text, return_tensors="pt").to(device)
    with torch.no_grad():
        _, h = gpt2_forward(hybrid, ids)
        q_raw = h[0, -min(2, h.shape[1]):, :].mean(dim=0).unsqueeze(0)
        q_proj = memory.query_proj(q_raw)
        q_read = memory.to_read_space(q_proj, mu, V_sub)

        logits_mem, sim_val = memory.slow_mlp(q_read)

        raw_sim = sim_val[0, 0].item()
        gate_prob = torch.sigmoid(
            memory.gate_sharpness * (sim_val[0, 0] - memory.gate_threshold)
        ).item()
        gate_active = gate_prob >= 0.5

        probs = F.softmax(logits_mem[0], dim=-1)
        target_prob = probs[target_tok].item()
        target_rank = int((probs > probs[target_tok]).sum().item()) + 1

        natural_pred = logits_mem[0].argmax().item()
        natural_correct = (natural_pred == target_tok)

        # Forced-gate: the MLP logit function doesn't change with forced gate;
        # the gate only controls blending with base-model logits in generation.
        # Here we simply flag what WOULD happen if we forced gate=1.0.
        # Since evaluation uses argmax(logits_mem) directly, forced == natural in this eval.
        forced_correct = natural_correct

        # Diagnose failure mode
        if not gate_active and natural_correct:
            case = "A"
            diagnosis = "GATE THRESHOLD FAILURE — memory intact, gate not firing"
        elif gate_active and not natural_correct:
            case = "B"
            diagnosis = "MLP LOGIT DRIFT — gate fires but MLP predicts wrong token"
        elif not gate_active and not natural_correct:
            case = "C"
            diagnosis = "DUAL FAILURE — gate off and MLP wrong (prototype coverage gap)"
        else:
            case = "D"
            diagnosis = "CORRECT — gate active and MLP correct"

        # Prototype coverage: max cosine sim to all prototypes for this fact
        proto_sims = []
        for tag in ["_stmt", "_qa", "_cloze"]:
            vkey = fid + tag
            if vkey in sampler.prototypes:
                proto = sampler.prototypes[vkey].to(device)
                cos_sim = F.cosine_similarity(q_read, proto.unsqueeze(0)).item()
                proto_sims.append(cos_sim)

        max_proto_sim = max(proto_sims) if proto_sims else float("nan")

    return {
        "para_text": para_text[:60] + ("..." if len(para_text) > 60 else ""),
        "raw_similarity": raw_sim,
        "gate_probability": gate_prob,
        "gate_active": gate_active,
        "target_rank": target_rank,
        "target_prob": target_prob,
        "natural_pred": tokenizer.decode([natural_pred]),
        "target_tok_str": tokenizer.decode([target_tok]),
        "natural_correct": natural_correct,
        "forced_correct": forced_correct,
        "max_proto_sim": max_proto_sim,
        "case": case,
        "diagnosis": diagnosis,
    }


def run_full_diagnostics(
    hybrid,
    memory: EpisodicFactMemory,
    mu: torch.Tensor,
    V_sub: torch.Tensor | None,
    blocks: list[list[dict]],
    sampler: ReplaySampler,
    stage: int,
    learned_blocks: int,
    # gate_records: dict[str, dict] accumulates raw_sim across stages
    gate_records: dict,
) -> None:
    """Run and print forced-gate diagnostics for all facts in learned blocks."""
    tokenizer = hybrid.tokenizer
    device = hybrid.device

    case_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for blk_idx in range(learned_blocks):
        block_facts = blocks[blk_idx]
        print(f"\n  ── Block {blk_idx + 1} diagnostics ──")

        for f in block_facts:
            fid = f["id"]
            exact_ans_ids = tokenizer.encode(f["statement"])
            target_tok = exact_ans_ids[len(tokenizer.encode(f["probe"]))]

            for para in EVAL_PARAPHRASES[fid]:
                d = diagnose_paraphrase(
                    hybrid, memory, mu, V_sub,
                    para, target_tok, sampler, fid
                )

                # Track gate score over time for temporal drift analysis
                key = (fid, para[:40])
                if key not in gate_records:
                    gate_records[key] = {}
                gate_records[key][stage] = d["raw_similarity"]

                # Print details only for failures or first paraphrase per fact
                marker = "✓" if d["natural_correct"] else "✗"
                print(
                    f"    [{marker}] {d['para_text']}\n"
                    f"        sim={d['raw_similarity']:+.4f}  gate_prob={d['gate_probability']:.4f}"
                    f"  gate={'ON' if d['gate_active'] else 'OFF'}"
                    f"  rank={d['target_rank']:4d}  p(target)={d['target_prob']:.4f}"
                    f"  max_proto_sim={d['max_proto_sim']:.4f}\n"
                    f"        pred='{d['natural_pred']}'  target='{d['target_tok_str']}'"
                    f"  → Case {d['case']}: {d['diagnosis']}"
                )

                case_counts[d["case"]] += 1

    total = sum(case_counts.values())
    print(f"\n  ── Stage {stage} Case Summary ──")
    labels = {
        "A": "Gate Threshold Failure (memory intact, gate not firing)",
        "B": "MLP Logit Drift        (gate fires, wrong token)",
        "C": "Dual Failure           (gate off + MLP wrong)",
        "D": "Correct                (gate on + MLP correct)",
    }
    for case, count in case_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"    Case {case}: {count:3d} / {total}  ({pct:5.1f}%)  — {labels[case]}")


def print_temporal_drift(gate_records: dict, acquired_stages: dict[str, int]) -> None:
    """Print gate-score drift for paraphrases from their acquisition stage to final stage."""
    print("\n=================================================================")
    print("  TEMPORAL GATE-SCORE DRIFT ANALYSIS")
    print("=================================================================")
    print(f"  {'Paraphrase':<45} {'Acq.Sim':>8}  {'Final.Sim':>9}  {'Δ Gate':>8}")
    print("  " + "-" * 75)

    drifts = []
    for (fid, para_key), stage_sims in sorted(gate_records.items()):
        acq_stage = acquired_stages.get(fid, 1)
        if acq_stage in stage_sims and max(stage_sims.keys()) in stage_sims:
            sim_acq = stage_sims[acq_stage]
            sim_final = stage_sims[max(stage_sims.keys())]
            delta = sim_final - sim_acq
            drifts.append(delta)
            flag = " ← DRIFT" if abs(delta) > 0.05 else ""
            print(f"  {para_key:<45} {sim_acq:+8.4f}  {sim_final:+9.4f}  {delta:+8.4f}{flag}")

    if drifts:
        print(f"\n  Mean |Δ gate|: {np.mean(np.abs(drifts)):.4f}"
              f"  |  Max |Δ gate|: {max(np.abs(drifts)):.4f}"
              f"  |  Fraction > 0.05: {np.mean(np.abs(drifts) > 0.05)*100:.1f}%")


# ---------------------------------------------------------------------------
# Full V4.5 pipeline re-run with diagnostics embedded
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  AGNIS V4.5 — FORCED-GATE PARAPHRASE DIAGNOSTIC")
    print("  Sequential consolidation + per-stage forced-gate decomposition")
    print("=" * 70)

    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()

    original_base_state = {k: v.clone().detach().cpu() for k, v in hybrid.gpt2.named_parameters()}
    vocab_size = hybrid.gpt2.config.vocab_size
    memory = EpisodicFactMemory(vocab_size=vocab_size, pool_len=2, device=DEVICE)
    sampler = ReplaySampler(embed_dim=768)

    # Map raw fact IDs onto injection fact texts
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        fact["id"] = RAW_FACTS[idx // 3]["id"]

    blocks = [RAW_FACTS[:4], RAW_FACTS[4:7], RAW_FACTS[7:]]
    num_blocks = len(blocks)

    # Track which stage each fact ID was acquired
    acquired_stages: dict[str, int] = {}
    for i, blk in enumerate(blocks, start=1):
        for f in blk:
            acquired_stages[f["id"]] = i

    # ── Protocol A: Pre-Consolidation Query Projection Training ──────────
    print("\n[Protocol A] Pre-Consolidation Query-Projection Training...")
    fact_ranges: dict[str, tuple[int, int]] = {}
    answer_ids_all: dict[str, torch.Tensor] = {}
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        prompt_ids = tokenizer.encode(fact["prompt"])
        full_ids_list = ids[0].tolist()
        n = 0
        limit = min(len(prompt_ids), len(full_ids_list))
        while n < limit and full_ids_list[n] == prompt_ids[n]:
            n += 1
        if n < max(1, len(prompt_ids) // 2):
            n = limit
        boundary = max(0, n - 1)
        h_answer = h[0, boundary:-1, :]
        v_answer = ids[0, boundary + 1:]
        start = len(memory)
        memory.write(h_answer, v_answer)
        if idx % 3 == 0:
            fid = RAW_FACTS[idx // 3]["id"]
            fact_ranges[fid] = (start, h_answer.shape[0])
            answer_ids_all[fid] = v_answer.detach()

    q_fact, pos_idx = collect_fact_queries(hybrid, memory, fact_ranges, answer_ids_all)
    q_ctrl = collect_control_states(hybrid, memory)
    train_query_projection(memory, q_fact, pos_idx, q_ctrl)

    for param in memory.query_proj.parameters():
        param.requires_grad = False
    print("  [OK] Query projection frozen.")

    # Build and lock read space geometry
    temp_keys = []
    for fact in INJECTION_FACT_TEXTS:
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        prompt_ids = tokenizer.encode(fact["prompt"])
        n = 0
        limit = min(len(prompt_ids), len(ids[0].tolist()))
        while n < limit and ids[0, n].item() == prompt_ids[n]:
            n += 1
        boundary = max(0, n - 1)
        temp_keys.append(h[0, boundary:-1, :])
    memory.keys_raw = torch.cat(temp_keys, dim=0)
    mu, V_sub = memory.read_space()
    # CRITICAL: clear BOTH keys_raw AND values — Protocol A writes polluted memory.values
    memory.keys_raw = torch.empty(0, 768, device=DEVICE)
    memory.values = torch.empty(0, dtype=torch.long, device=DEVICE)
    memory._space_cache = (mu, V_sub)  # preserve cached geometry
    print("  [OK] Space cache locked. Episodic memory fully evicted.")

    # Initialize empty student MLP
    empty_mlp = JointSlowMemoryMLP(vocab_size=vocab_size).to(DEVICE)
    with torch.no_grad():
        nn.init.zeros_(empty_mlp.logits_head.weight)
        nn.init.zeros_(empty_mlp.logits_head.bias)
        nn.init.zeros_(empty_mlp.gate_head.weight)
        nn.init.zeros_(empty_mlp.gate_head.bias)
    memory.slow_mlp = empty_mlp
    initial_mlp_param_count = sum(p.numel() for p in memory.slow_mlp.parameters())

    gate_records: dict = {}

    # ── Sequential Consolidation Loop ─────────────────────────────────────
    from agnis_continual_v4_5 import (
        train_student_with_replay,
        collect_block_queries,
        evaluate_block_performance,
    )

    for i in range(1, num_blocks + 1):
        print(f"\n{'='*68}")
        print(f"  STAGE {i} — Consolidating Block {i}")
        print(f"{'='*68}")

        teacher = None
        if i > 1:
            teacher = copy.deepcopy(memory.slow_mlp).eval()
            for p in teacher.parameters():
                p.requires_grad = False

        block_facts = blocks[i - 1]
        block_fact_ids = [f["id"] for f in block_facts]
        block_injections = [fact for fact in INJECTION_FACT_TEXTS if fact["id"] in block_fact_ids]

        block_fact_ranges = {}
        block_answer_ids = {}
        for idx, fact in enumerate(block_injections):
            ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
            _, h_full = gpt2_forward(hybrid, ids)
            h = memory.pool_sequence(h_full)
            prompt_ids = tokenizer.encode(fact["prompt"])
            n = 0
            limit = min(len(prompt_ids), len(ids[0].tolist()))
            while n < limit and ids[0, n].item() == prompt_ids[n]:
                n += 1
            boundary = max(0, n - 1)
            h_answer = h[0, boundary:-1, :]
            v_answer = ids[0, boundary + 1:]
            start = len(memory)
            memory.write(h_answer, v_answer)
            variant_tag = ["_stmt", "_qa", "_cloze"][idx % 3]
            fid = fact["id"]
            variant_key = fid + variant_tag
            block_fact_ranges[variant_key] = (start, h_answer.shape[0])
            if idx % 3 == 0:
                block_fact_ranges[fid] = (start, h_answer.shape[0])
                block_answer_ids[fid] = v_answer.detach()

        q_fact_i, pos_idx_i = collect_block_queries(
            hybrid, memory, block_facts, block_fact_ranges, block_answer_ids
        )
        q_ctrl_i = collect_control_states(hybrid, memory)

        for fid in block_fact_ids:
            for tag in ["_stmt", "_qa", "_cloze"]:
                vkey = fid + tag
                if vkey in block_fact_ranges:
                    start, length = block_fact_ranges[vkey]
                    key_slice = memory.keys_raw[start: start + length]
                    if key_slice.shape[0] > 0:
                        with torch.no_grad():
                            key_read = memory.to_read_space(key_slice, mu, V_sub)
                        sampler.update_fact(vkey, key_read)

        k_read_i = memory.to_read_space(memory.keys_raw, mu, V_sub).detach()
        q_ctrl_read_i = memory.to_read_space(memory.query_proj(q_ctrl_i), mu, V_sub).detach()

        train_inputs_cur, target_tokens_cur, target_sims_cur = [], [], []
        with torch.no_grad():
            q_fact_read_i = memory.to_read_space(memory.query_proj(q_fact_i), mu, V_sub)
            sims_fact = q_fact_read_i @ k_read_i.T
            max_sims_fact = sims_fact.max(dim=-1).values
        for idx_q in range(q_fact_read_i.shape[0]):
            train_inputs_cur.append(q_fact_read_i[idx_q])
            target_tokens_cur.append(memory.values[pos_idx_i[idx_q]].item())
            target_sims_cur.append(max_sims_fact[idx_q].item())
        train_inputs_cur = torch.stack(train_inputs_cur).to(DEVICE)
        target_tokens_cur = torch.tensor(target_tokens_cur, dtype=torch.long, device=DEVICE)
        target_sims_cur = torch.tensor(target_sims_cur, dtype=torch.float, device=DEVICE).unsqueeze(-1)
        self_sims = (k_read_i * k_read_i).sum(dim=-1, keepdim=True)
        train_inputs_cur = torch.cat([train_inputs_cur, k_read_i], dim=0)
        target_tokens_cur = torch.cat([target_tokens_cur, memory.values], dim=0)
        target_sims_cur = torch.cat([target_sims_cur, self_sims], dim=0)

        memory.slow_mlp = train_student_with_replay(
            memory=memory,
            sampler=sampler,
            teacher=teacher,
            current_inputs=train_inputs_cur,
            current_tokens=target_tokens_cur,
            current_sims=target_sims_cur,
            q_ctrl_read=q_ctrl_read_i,
            epochs=400,
            lambda_replay=5.0,
            replay_count_per_fact=50,
        )

        memory.keys_raw = torch.empty(0, 768, device=DEVICE)
        memory.values = torch.empty(0, dtype=torch.long, device=DEVICE)
        memory._space_cache = (mu, V_sub)
        memory.gate_threshold = 0.602
        memory.gate_sharpness = 80.0

        assert len(memory.keys_raw) == 0
        for name, parameter in hybrid.gpt2.named_parameters():
            assert torch.equal(parameter.cpu(), original_base_state[name])
        assert sum(p.numel() for p in memory.slow_mlp.parameters()) == initial_mlp_param_count

        # ── Evaluation ───────────────────────────────────────────────────
        print(f"  [Evaluation] Performance after Stage {i}:")
        for j in range(num_blocks):
            exact_acc, para_acc, gate_tpr, _ = evaluate_block_performance(
                hybrid, memory, blocks[j], mu, V_sub
            )
            known = "(learned)" if j < i else "(unseen)"
            print(f"    Block {j+1} {known} | exact={exact_acc*100:5.1f}% | para={para_acc*100:5.1f}% | gate_tpr={gate_tpr*100:5.1f}%")

        # ── Forced-Gate Diagnostics on all learned blocks ─────────────────
        print(f"\n  [Diagnostics] Forced-gate decomposition after Stage {i}:")
        run_full_diagnostics(hybrid, memory, mu, V_sub, blocks, sampler, i, i, gate_records)

    # ── Temporal Drift Analysis ───────────────────────────────────────────
    print_temporal_drift(gate_records, acquired_stages)


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
