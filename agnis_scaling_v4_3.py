"""
agnis_scaling_v4_3.py — V4.3 CLS Memory Capacity Curve
============================================================================
Stress-test of the V4.2b episodic memory + fuzzy query projection under
distractor load. See AGNIS_V4_3_SCALING_PLAN.md for the full protocol.

Protocol:
  Anchor set : the 10 RAW_FACTS (30 injection texts) + V4.2b paraphrase
               machinery, written first. Stage +0 reproduces V4.2b.
  Distractors: wikitext-103 spans synthesized into (prompt -> continuation)
               facts, injected in stages: +0, +100, +300, +1000.
  Per stage  : fresh query-projection retrain (comparability), gate
               recalibration, then:
                 - anchor exact recall (10) + HELD-OUT paraphrase recall (20)
                 - wiki distractor recall (sampled, first-3-token match)
                 - retention (10) + PPL (gate silence must hold)
                 - fact/control margin (collapse => local thresholds in V4.4)
                 - PCA subspace overlap vs stage-0 V_sub (drift)
                 - read_space() wall time (SVD cache + pca_lowrank in V4.3)
  Output     : JSON record per stage -> RESULTS_PATH for capacity plots.
"""
from __future__ import annotations

import json
import random
import time

import torch

from agnis_continual_v2 import (
    HAS_DATASETS,
    INDEPENDENT_PPL_TEXTS,
    INJECTION_FACT_TEXTS,
    RAW_FACTS,
    RETENTION_PROBES,
    build_hybrid,
)
from agnis_continual_v4_1 import (
    DEVICE,
    blended_next_probs,
    gpt2_forward,
    measure_ppl,
    probe_recall,
    probe_retention,
)
from agnis_continual_v4_2 import (
    POOL_LEN,
    TRAIN_PARAPHRASES,
    collect_control_states,
    collect_fact_queries,
    probe_paraphrase_recall,
    train_query_projection,
)
from fact_memory import EpisodicFactMemory, ResidualQueryProjection

RESULTS_PATH = "/kaggle/working/agnis_scaling_v4_3_results.json"

DISTRACTOR_STAGES = [0, 100, 300, 1000]   # cumulative wiki-fact counts
WIKI_EVAL_SAMPLE  = 100                   # distractor recall sample per stage
MATCH_TOKENS      = 3                     # first-k-token exact match = hit
PROMPT_FRAC       = 0.6                   # span cut: 60% prompt, 40% answer
MIN_SPAN_TOKENS   = 14
MAX_SPAN_TOKENS   = 40
SEED              = 1234


# ── Distractor synthesis ───────────────────────────────────────────────────
def synthesize_wiki_facts(tokenizer, n_facts: int, seed: int = SEED) -> list[dict]:
    """Wikitext-103 spans -> episodic facts: {'ids': token ids, 'cut': prompt
    length}. Position cut-1 is the boundary key; ids[cut:] is the answer."""
    if not HAS_DATASETS:
        raise RuntimeError("V4.3 needs `datasets` (wikitext-103) — enable internet on the kernel")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    facts, seen = [], set()
    for i in order:
        line = ds[i]["text"].strip()
        if not line or line.startswith("=") or len(line) < 40:
            continue
        span = line.split(". ")[0][:300].strip()
        if not span or span in seen:
            continue
        ids = tokenizer.encode(span)
        if not (MIN_SPAN_TOKENS <= len(ids) <= MAX_SPAN_TOKENS):
            continue
        cut = max(6, int(len(ids) * PROMPT_FRAC))
        if len(ids) - cut < MATCH_TOKENS:
            continue
        seen.add(span)
        facts.append({"ids": ids, "cut": cut})
        if len(facts) >= n_facts:
            break
    if len(facts) < n_facts:
        print(f"  WARNING: only {len(facts)}/{n_facts} wiki facts synthesized")
    return facts


@torch.no_grad()
def write_wiki_fact(hybrid, memory: EpisodicFactMemory, fact: dict) -> None:
    ids = torch.tensor(
        [fact["ids"] + [hybrid.tokenizer.eos_token_id]], device=hybrid.device
    )
    _, h = gpt2_forward(hybrid, ids)
    boundary = fact["cut"] - 1
    h_pool = memory.pool_sequence(h[0])   # pool FULL sequence before slicing
    memory.write(h_pool[boundary:-1, :], ids[0, boundary + 1:])


@torch.no_grad()
def write_anchor_facts(hybrid, memory: EpisodicFactMemory):
    """V4.2b Phase B verbatim: pooled answer-position writes + per-fact
    (start, length) ranges and answer ids for the contrastive positives."""
    tokenizer = hybrid.tokenizer
    fact_ranges: dict[str, tuple[int, int]] = {}
    answer_ids: dict[str, torch.Tensor] = {}
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h = gpt2_forward(hybrid, ids)
        prompt_ids = tokenizer.encode(fact["prompt"])
        full_ids_list = ids[0].tolist()
        n = 0
        limit = min(len(prompt_ids), len(full_ids_list))
        while n < limit and full_ids_list[n] == prompt_ids[n]:
            n += 1
        if n < max(1, len(prompt_ids) // 2):
            n = limit
        boundary = max(0, n - 1)
        h_pool = memory.pool_sequence(h[0])
        h_answer = h_pool[boundary:-1, :]
        v_answer = ids[0, boundary + 1:]
        start = len(memory)
        memory.write(h_answer, v_answer)
        if idx % 3 == 0:
            fid = RAW_FACTS[idx // 3]["id"]
            fact_ranges[fid] = (start, h_answer.shape[0])
            answer_ids[fid] = v_answer.detach()
    return fact_ranges, answer_ids


# ── Evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def wiki_recall(hybrid, memory: EpisodicFactMemory, facts: list[dict],
                sample_n: int = WIKI_EVAL_SAMPLE, seed: int = 7):
    """Token-level greedy generation: hit iff the first MATCH_TOKENS answer
    tokens are reproduced exactly (episodic memory is verbatim by design)."""
    if not facts:
        return 0, 0
    rng = random.Random(seed)
    sample = facts if len(facts) <= sample_n else rng.sample(facts, sample_n)
    hits = 0
    for fact in sample:
        ids = torch.tensor([fact["ids"][: fact["cut"]]], device=hybrid.device)
        target = fact["ids"][fact["cut"]:][:MATCH_TOKENS]
        ok = True
        for t in target:
            probs, _, _ = blended_next_probs(hybrid, memory, ids)
            nxt = probs[0, -1, :].argmax().item()
            if nxt != t:
                ok = False
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=hybrid.device)], dim=1)
        hits += int(ok)
    return hits, len(sample)


@torch.no_grad()
def max_sim_texts(hybrid, memory: EpisodicFactMemory, texts: list[str]) -> list[float]:
    out = []
    for t in texts:
        ids = hybrid.tokenizer.encode(t, return_tensors="pt").to(hybrid.device)
        _, h = gpt2_forward(hybrid, ids)
        _, _, ms = memory.read(h)
        out.append(ms.max().item())
    return out


def stats(xs: list[float]) -> dict:
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}


def subspace_overlap(V_a, V_b) -> float | None:
    """Mean singular value of V_a^T V_b in [0, 1]: 1.0 = identical subspace,
    -> 0 = orthogonal. Tracks anisotropy-correction drift across stages."""
    if V_a is None or V_b is None:
        return None
    s = torch.linalg.svdvals(V_a.T @ V_b)
    return s.mean().item()


def recalibrate(memory: EpisodicFactMemory, para_sims: list[float],
                ctrl_sims: list[float]):
    """V4.2b threshold rule, reset first so stages are independent."""
    memory.gate_threshold = 0.95
    min_para, max_ctrl = min(para_sims), max(ctrl_sims)
    margin = min_para - max_ctrl
    if margin > 0.05:
        memory.gate_threshold = min(0.95, max_ctrl + 0.6 * margin)
    else:
        print(f"  WARNING: margin collapsed to {margin:.3f} — keeping 0.95; "
              f"V4.4 direction is per-cluster local thresholds.")
    return memory.gate_threshold, margin


# ── Main sweep ───────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AGNIS V4.3 — CLS MEMORY CAPACITY CURVE")
    print(f"  anchor: 10 facts | distractor stages: {DISTRACTOR_STAGES}")
    print("=" * 65)

    hybrid = build_hybrid()
    hybrid.eval()
    vocab_size = hybrid.gpt2.config.vocab_size

    # Baseline (empty memory) for retention/PPL reference.
    empty = EpisodicFactMemory(vocab_size=vocab_size, pool_len=POOL_LEN, device=DEVICE)
    base_retention = probe_retention(hybrid, empty, "BASELINE")
    base_ppl = measure_ppl(hybrid, empty, INDEPENDENT_PPL_TEXTS)
    print(f"  Baseline retention {base_retention}/10 | PPL {base_ppl:.2f}\n")

    memory = EpisodicFactMemory(vocab_size=vocab_size, pool_len=POOL_LEN, device=DEVICE)
    fact_ranges, answer_ids = write_anchor_facts(hybrid, memory)
    anchor_keys = len(memory)
    print(f"  Anchor facts written: {anchor_keys} keys")

    # Contrastive data is stage-independent (anchors written first): collect once.
    q_fact, pos_idx = collect_fact_queries(hybrid, memory, fact_ranges, answer_ids)
    q_ctrl = collect_control_states(hybrid, memory)
    print(f"  Contrastive set: {q_fact.shape[0]} fact queries | {q_ctrl.shape[0]} control states\n")

    n_max = max(DISTRACTOR_STAGES)
    wiki_all = synthesize_wiki_facts(hybrid.tokenizer, n_max) if n_max > 0 else []

    ctrl_texts = [p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS)
    para_texts = [p for f in RAW_FACTS for p in TRAIN_PARAPHRASES[f["id"]]]

    V0 = None
    written = 0
    results = []

    for stage in DISTRACTOR_STAGES:
        print("=" * 65)
        print(f"  STAGE +{stage} distractors")
        print("=" * 65)
        for fact in wiki_all[written:stage]:
            write_wiki_fact(hybrid, memory, fact)
        written = stage
        active_wiki = wiki_all[:written]

        # Fresh projection per stage: comparability across the curve.
        memory.query_proj = ResidualQueryProjection(memory.embed_dim).to(memory.keys_raw.device)

        t0 = time.time()
        memory._space_cache = None
        _, V_sub = memory.read_space()
        t_space = time.time() - t0
        if V0 is None:
            V0 = V_sub.clone() if V_sub is not None else None
        drift = subspace_overlap(V0, V_sub)

        train_query_projection(memory, q_fact, pos_idx, q_ctrl)

        para_sims = max_sim_texts(hybrid, memory, para_texts)
        ctrl_sims = max_sim_texts(hybrid, memory, ctrl_texts)
        threshold, margin = recalibrate(memory, para_sims, ctrl_sims)

        anchor_recall = probe_recall(hybrid, memory, f"STAGE+{stage}")
        anchor_para = probe_paraphrase_recall(hybrid, memory, f"STAGE+{stage} (held-out)")
        w_hits, w_n = wiki_recall(hybrid, memory, active_wiki)
        retention = probe_retention(hybrid, memory, f"STAGE+{stage}")
        ppl = measure_ppl(hybrid, memory, INDEPENDENT_PPL_TEXTS)

        rec = {
            "distractors": stage,
            "total_keys": len(memory),
            "anchor_recall": anchor_recall,
            "anchor_paraphrase_recall": anchor_para,
            "wiki_recall_hits": w_hits,
            "wiki_recall_n": w_n,
            "retention": retention,
            "ppl": ppl,
            "ppl_delta": ppl - base_ppl,
            "gate_threshold": threshold,
            "margin": margin,
            "paraphrase_sims": stats(para_sims),
            "control_sims": stats(ctrl_sims),
            "pca_overlap_vs_stage0": drift,
            "read_space_seconds": t_space,
        }
        results.append(rec)
        print(f"  STAGE +{stage}: keys={rec['total_keys']} | "
              f"anchor {anchor_recall}/10 | para {anchor_para}/20 | "
              f"wiki {w_hits}/{w_n} | retention {retention}/10 | "
              f"PPL {ppl:.2f} ({rec['ppl_delta']:+.2f}) | "
              f"margin {margin:.3f} thr {threshold:.3f} | "
              f"PCA-overlap {drift if drift is None else round(drift, 3)} | "
              f"space {t_space*1000:.0f}ms\n")

    print("=" * 65)
    print("  V4.3 CAPACITY CURVE")
    print("=" * 65)
    print(f"  {'distr':>6} {'keys':>6} {'anchor':>7} {'para':>6} {'wiki':>9} "
          f"{'reten':>6} {'PPLΔ':>7} {'margin':>7} {'thr':>6} {'PCA':>6}")
    for r in results:
        wiki_str = f"{r['wiki_recall_hits']}/{r['wiki_recall_n']}"
        pca_str = "-" if r["pca_overlap_vs_stage0"] is None else f"{r['pca_overlap_vs_stage0']:.3f}"
        print(f"  {r['distractors']:>6} {r['total_keys']:>6} "
              f"{r['anchor_recall']:>5}/10 {r['anchor_paraphrase_recall']:>4}/20 "
              f"{wiki_str:>9} {r['retention']:>4}/10 {r['ppl_delta']:>+7.2f} "
              f"{r['margin']:>7.3f} {r['gate_threshold']:>6.3f} {pca_str:>6}")

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Saved capacity-curve data -> {RESULTS_PATH}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
