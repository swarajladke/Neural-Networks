"""
agnis_continual_v4_4.py — V4.4 Memory Consolidation (Horizon A)
=============================================================================
V4.2b/V4.3 solved fuzzy query alignment on 12,000+ stored keys, but kept the
explicit episodic list lookup (hippocampal path). V4.4 compiles/consolidates
these episodic memories directly into the weights of a neural network projection
(JointSlowMemoryMLP) that maps query hidden states to next-token predictions and
gate activations. Once trained, the episodic database keys and values are completely
evicted (deleted), running 100% feed-forward (cortical path) during inference.

Pipeline:
  PHASE A   baseline (empty memory): recall / paraphrase / retention / PPL
  PHASE B   episodic write
  PHASE B2  micro-contrastive training of the query projection
  PHASE B3  memory consolidation (training the JointSlowMemoryMLP + database eviction)
  PHASE C   gate calibration (re-calibrated on consolidated outputs)
  PHASE D   exact recall / HELD-OUT paraphrase recall / retention / PPL
            in the compiled cortex-only state (database keys/values ablated).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from agnis_continual_v2 import (
    INDEPENDENT_PPL_TEXTS,
    INJECTION_FACT_TEXTS,
    RAW_FACTS,
    RETENTION_PROBES,
    build_hybrid,
)
from agnis_continual_v4_1 import (
    DEVICE,
    gate_calibration,
    generate_with_memory,
    gpt2_forward,
    measure_ppl,
    probe_recall,
    probe_retention,
)
from agnis_continual_v4_2 import (
    TRAIN_PARAPHRASES,
    EVAL_PARAPHRASES,
    collect_fact_queries,
    collect_control_states,
    train_query_projection,
    probe_paraphrase_recall,
    recalibrate_gate,
)
from fact_memory import EpisodicFactMemory

MEMORY_PATH = "/kaggle/working/agnis_fact_memory_v44.pt"


def main():
    print("=" * 65)
    print("  AGNIS CONTINUAL LEARNING V4.4")
    print("  Memory Consolidation (Horizon A) — Joint MLP + Database Eviction")
    print("=" * 65)

    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()

    vocab_size = hybrid.gpt2.config.vocab_size
    empty = EpisodicFactMemory(vocab_size=vocab_size, pool_len=2, device=DEVICE)
    memory = EpisodicFactMemory(vocab_size=vocab_size, pool_len=2, device=DEVICE)

    print("\nPHASE A — BASELINE (pure GPT-2, no memory)")
    print("-" * 65)
    before_recall = probe_recall(hybrid, empty, "BEFORE")
    before_para = probe_paraphrase_recall(hybrid, empty, "BEFORE")
    before_retention = probe_retention(hybrid, empty, "BEFORE")
    before_ppl = measure_ppl(hybrid, empty, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL before: {before_ppl:.2f}\n")

    print("PHASE B — EPISODIC WRITE (answer-only positions)")
    print("-" * 65)
    fact_ranges: dict[str, tuple[int, int]] = {}
    answer_ids: dict[str, torch.Tensor] = {}
    total_stored = 0
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        # Pool full hidden sequence
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)
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
        total_stored += h_answer.shape[0]
        if idx % 3 == 0:
            fid = RAW_FACTS[idx // 3]["id"]
            fact_ranges[fid] = (start, h_answer.shape[0])
            answer_ids[fid] = v_answer.detach()
    print(f"  Stored {total_stored} answer-position pairs from {len(INJECTION_FACT_TEXTS)} fact texts.")

    print("\nPHASE B2 — CONTRASTIVE QUERY-PROJECTION TRAINING")
    print("-" * 65)
    q_fact, pos_idx = collect_fact_queries(hybrid, memory, fact_ranges, answer_ids)
    q_ctrl = collect_control_states(hybrid, memory)
    print(f"  fact queries: {q_fact.shape[0]} | control states: {q_ctrl.shape[0]}")
    train_query_projection(memory, q_fact, pos_idx, q_ctrl)
    print()

    print("PHASE B3 — MEMORY CONSOLIDATION (eviction of episodic store)")
    print("-" * 65)
    # Train JointSlowMemoryMLP and clear the keys_raw/values database buffers
    memory.consolidate(q_fact, pos_idx, q_ctrl, epochs=200)
    print(f"  [OK] Consolidated database of {total_stored} keys into slow weights.")
    print(f"  [OK] Episodic keys/values completely evicted (len keys_raw = {len(memory.keys_raw)}).")
    print()

    print("PHASE C — GATE CALIBRATION + RECALIBRATION")
    print("-" * 65)
    gate_calibration(hybrid, memory)
    recalibrate_gate(hybrid, memory)
    print()

    print("PHASE D — POST-CONSOLIDATION EVALUATION")
    print("-" * 65)
    after_recall = probe_recall(hybrid, memory, "AFTER (compiled weights only)")
    after_para = probe_paraphrase_recall(hybrid, memory, "AFTER (held-out)")
    after_retention = probe_retention(hybrid, memory, "AFTER")
    after_ppl = measure_ppl(hybrid, memory, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL after: {after_ppl:.2f}\n")

    print("=" * 65)
    print("  V4.4 RESULTS")
    print("=" * 65)
    n_para = sum(len(v) for v in EVAL_PARAPHRASES.values())
    print(f"  Exact Recall      : {before_recall}/10 -> {after_recall}/10")
    print(f"  Paraphrase Recall : {before_para}/{n_para} -> {after_para}/{n_para} (held-out)")
    print(f"  Retention         : {before_retention}/10 -> {after_retention}/10")
    print(f"  PPL               : {before_ppl:.2f} -> {after_ppl:.2f} ({after_ppl - before_ppl:+.2f})")

    # Save state dict
    torch.save(
        {
            "query_proj": {k: v.cpu() for k, v in memory.query_proj.state_dict().items()},
            "slow_mlp": {k: v.cpu() for k, v in memory.slow_mlp.state_dict().items()},
            "gate_threshold": memory.gate_threshold,
            "pool_len": memory.pool_len,
            "npc_project": memory.npc_project,
        },
        MEMORY_PATH,
    )
    print(f"\n  Saved consolidated projection model -> {MEMORY_PATH}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
