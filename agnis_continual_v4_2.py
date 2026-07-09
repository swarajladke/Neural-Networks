"""
agnis_continual_v4_2.py — V4.2b Fuzzy Context Retrieval + Average Gating
============================================================================
V4.1c solved exact-prefix recall (10/10, retention and PPL untouched), but
the memory is a lookup table: keys derive from exact prompt prefixes, so a
rephrased question falls below the gate. V4.2 trains a residual query
projection (fact_memory.ResidualQueryProjection) so paraphrased queries land
on the stored boundary keys while control text stays below the gate.

Design amendments baked in (V4.2 review):
  1. Residual zero-init projection -> exact identity at init; V4.1c results
     are provably preserved before training (checked in PHASE B).
  2. InfoNCE is computed in the exact read-time space: query_proj ->
     center by stored-key mean -> project out top-5 PCs -> normalize.
  3. Every batch carries control-text hidden states (retention probes +
     independent PPL texts) as explicit negatives plus a hinge repulsion,
     so the projection cannot inflate similarity on general English.
  4. Continuation states (paraphrase + partial answer) are trained toward
     the matching continuation keys so multi-token answers survive past the
     first retrieved token.
  5. EVAL_PARAPHRASES (2 per fact = 20) are HELD OUT: never used in
     training or gate calibration. Disjointness is asserted at startup.

V4.2b — Paraphrase Generalization & Average Gating:
  6. pool_len=2 causal average pooling on keys and queries. Keys are pooled
     on the FULL hidden sequence BEFORE slicing, so boundary keys and
     boundary queries share identical geometry (pooling only the stored
     slice would mismatch exactly at the first-token retrieval position).
     Reduces sensitivity to the exact tail token ("Exactly" vs "exactly").
  7. 6 training paraphrases per fact (QA styling + capitalization variants)
     and MAX_CONT_TOKENS 4 -> 12 so long answers stay locked to their keys.
  8. Per-miss diagnostics in the paraphrase probe (boundary max-sim, lam,
     first retrieved token) to separate gate-threshold failures from
     similarity failures before crediting any fix.

Pipeline:
  PHASE A   baseline (empty memory): recall / paraphrase / retention / PPL
  PHASE B   episodic write (+ identity check + pre-train exact recall)
  PHASE B2  micro-contrastive training of the query projection (<1s)
  PHASE C   gate calibration + data-driven threshold selection
  PHASE D   exact recall / HELD-OUT paraphrase recall / retention / PPL

Expected: Fact Recall 10/10, Paraphrase Recall >= 18/20, Retention
unchanged from baseline, PPL +0.00, zero gradient updates to GPT-2.
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
    blended_next_probs,
    gate_calibration,
    generate_with_memory,
    gpt2_forward,
    measure_ppl,
    probe_recall,
    probe_retention,
)
from fact_memory import EpisodicFactMemory

MEMORY_PATH = "/kaggle/working/agnis_fact_memory_v42.pt"

# ── Contrastive hyper-parameters ────────────────────────────────────────────
TAU             = 0.05   # InfoNCE temperature
LR              = 1e-3   # AdamW on query_proj only
EPOCHS          = 100    # micro-epochs, full batch
CTRL_MARGIN     = 0.50   # control max-sim pushed below this
CTRL_WEIGHT     = 2.0    # weight of the control hinge loss
MAX_CONT_TOKENS = 12     # continuation positives per paraphrase (V4.2b: 4 -> 12)
POOL_LEN        = 2      # V4.2b causal average pooling window (keys & queries)

# ── Training paraphrases (NEVER overlap with EVAL_PARAPHRASES) ─────────────
# Each prompt ends immediately before the fact's answer tokens.
# V4.2b: 6 per fact — the last two cover QA styling + capitalization as a
# CLASS. Do not mirror specific failing eval prompts (calibrating-on-test).
TRAIN_PARAPHRASES = {
    "F01": [
        "Q: How does the AGNIS model hook into GPT-2? A: It integrates its Hebbian predictive hierarchy with GPT-2",
        "The AGNIS system connects its Hebbian predictive stack to GPT-2",
        "AGNIS couples Hebbian predictive hierarchies to GPT-2",
        "In AGNIS, Hebbian predictive hierarchies are joined with GPT-2",
        "Q: In what way is AGNIS combined with GPT-2? A: AGNIS merges its Hebbian predictive hierarchies with GPT-2",
        "Question: How is AGNIS attached to GPT-2? Answer: By integrating Hebbian predictive hierarchies with GPT-2",
    ],
    "F02": [
        "Q: At what temperature does Thermocyclase-9 work? A: It catalyzes protein folding at exactly",
        "Thermocyclase-9, the deep-sea vent enzyme, folds proteins at exactly",
        "The enzyme Thermocyclase-9 operates at a temperature of exactly",
        "Deep-sea hydrothermal vents host Thermocyclase-9, which drives protein folding at exactly",
        "Q: What temperature does Thermocyclase-9 require? A: Exactly",
        "Question: Where does Thermocyclase-9 fold proteins best? Answer: At exactly",
    ],
    "F03": [
        "Q: What are the moons of Kepler-9814b called? A: Its three moons are named",
        "Kepler-9814b, which orbits its star in 47.3 days, has three moons named",
        "The three moons circling the planet Kepler-9814b are called",
        "Kepler-9814b is orbited by three moons named",
        "Q: Which moons orbit Kepler-9814b? A: The moons are named",
        "Question: List the moons of Kepler-9814b. Answer: The three moons are",
    ],
    "F04": [
        "Q: What plasma temperature did Project Helios reach? A: Cold fusion was achieved at",
        "The Helios project demonstrated cold fusion at a plasma temperature of",
        "Cold fusion in Project Helios occurred at a plasma temperature of",
        "Q: How hot was the Helios plasma? A: Project Helios hit cold fusion at",
        "Q: What temperature did the Helios plasma reach? A: A plasma temperature of",
        "Question: What did Project Helios achieve? Answer: Cold fusion at a plasma temperature of",
    ],
    "F05": [
        "Q: How does the Ladke-Nair algorithm avoid forgetting? A: It achieves zero catastrophic forgetting by",
        "The Ladke-Nair method eliminates catastrophic forgetting by",
        "Ladke-Nair continual learning prevents forgetting by",
        "Zero catastrophic forgetting in the Ladke-Nair algorithm comes from",
        "Q: Why doesn't the Ladke-Nair algorithm forget? A: Because it achieves zero catastrophic forgetting by",
        "Question: What makes Ladke-Nair special? Answer: It sidesteps catastrophic forgetting by",
    ],
    "F06": [
        "Q: How many pitch levels does Velathi have? A: The Velathi language has exactly",
        "The tonal language Velathi spoken in Aurantia has exactly",
        "Velathi, the language of Aurantia, features exactly",
        "Aurantia's tonal language Velathi contains exactly",
        "Q: How many tones does the Velathi language use? A: Exactly",
        "Question: What are the features of Velathi? Answer: The language has exactly",
    ],
    "F07": [
        "Q: What is the melting point of Xenolite-B? A: It melts at",
        "Xenolite-B melts at",
        "The melting point of the compound Xenolite-B is",
        "Xenolite-B has a melting temperature of",
        "Q: At what temperature does Xenolite-B become liquid? A: At",
        "Question: What is Xenolite-B's melting point? Answer: It melts at",
    ],
    "F08": [
        "Q: How long does neuronal quantum coherence last according to Dr. Nair? A: Up to",
        "Dr. Priya Nair showed that neurons sustain quantum coherence for up to",
        "According to Nair's 2026 paper, biological neurons hold quantum coherence for up to",
        "Nair found quantum coherence in neurons lasting up to",
        "Q: What coherence time did Nair report for neurons? A: Coherence for up to",
        "Question: What did Dr. Nair discover about neurons? Answer: They exhibit quantum coherence for up to",
    ],
    "F09": [
        "Q: What is the atomic number of Auranium? A: It is",
        "Auranium's atomic number is",
        "The fictional metal Auranium carries an atomic number of",
        "The atomic number assigned to Auranium is",
        "Q: What number does Auranium have on the periodic table? A: An atomic number of",
        "Question: What is Auranium's atomic number? Answer: The atomic number is",
    ],
    "F10": [
        "Q: What perplexity did AGNIS V5 Sprint 3 get on FineWeb-Edu? A: A perplexity of",
        "The AGNIS V5 Sprint 3 model scores a perplexity of",
        "On FineWeb-Edu, the AGNIS V5 Sprint 3 checkpoint reaches a perplexity of",
        "AGNIS V5 Sprint 3 records a FineWeb-Edu perplexity of",
        "Q: What is the FineWeb-Edu result for AGNIS V5 Sprint 3? A: Perplexity of",
        "Question: How does Sprint 3 of AGNIS V5 perform on FineWeb-Edu? Answer: It reaches a perplexity of",
    ],
}

# ── HELD-OUT eval paraphrases: never seen by trainer or calibration ────────
EVAL_PARAPHRASES = {
    "F01": [
        "Q: What mechanism links AGNIS with GPT-2? A: AGNIS ties its Hebbian predictive hierarchies into GPT-2",
        "To fuse Hebbian predictive hierarchies with GPT-2, AGNIS relies on GPT-2",
    ],
    "F02": [
        "Q: How hot must it be for Thermocyclase-9 to fold proteins? A: Exactly",
        "The optimal folding temperature for the vent enzyme Thermocyclase-9 is exactly",
    ],
    "F03": [
        "Q: Name the moons of Kepler-9814b. A: They are",
        "The satellites of Kepler-9814b are named",
    ],
    "F04": [
        "Q: At what temperature did Helios achieve fusion? A: At a plasma temperature of",
        "The plasma in Project Helios reached cold fusion at",
    ],
    "F05": [
        "Q: What is the key idea of Ladke-Nair? A: It stops catastrophic forgetting by",
        "The Ladke-Nair approach avoids catastrophic forgetting by",
    ],
    "F06": [
        "Q: Describe the Velathi language. A: It has exactly",
        "In Aurantia the language Velathi comes with exactly",
    ],
    "F07": [
        "Q: When does Xenolite-B melt? A: At",
        "The compound known as Xenolite-B liquefies at",
    ],
    "F08": [
        "Q: What did Priya Nair measure in neurons? A: Quantum coherence for up to",
        "Per Dr. Nair, neurons stay quantum coherent for up to",
    ],
    "F09": [
        "Q: Which atomic number does Auranium have? A: Auranium's number is",
        "Auranium, the violet metal, has an atomic number of",
    ],
    "F10": [
        "Q: How well does AGNIS V5 Sprint 3 do on FineWeb-Edu? A: It hits a perplexity of",
        "The Sprint 3 AGNIS checkpoint attains a perplexity of",
    ],
}


@torch.no_grad()
def last_hidden(hybrid, memory: EpisodicFactMemory, ids: torch.Tensor) -> torch.Tensor:
    """Final-position hidden state, pooled with the memory's window so
    training queries live in the same geometry as read-time queries."""
    _, h = gpt2_forward(hybrid, ids)
    return memory.pool_sequence(h[0])[-1, :]


@torch.no_grad()
def collect_fact_queries(hybrid, memory: EpisodicFactMemory, fact_ranges: dict, answer_ids: dict):
    """Build (queries (N, E), positive key indices (N,)).

    Positives: for each TRAINING paraphrase of fact f, the query at its final
    position targets the fact's statement-variant boundary key (start).
    Continuation states — paraphrase + first j answer tokens — target key
    start+j, so retrieval keeps matching after the first emitted token.
    Mid-sequence keys of a fact are never positives for the plain paraphrase.
    """
    tok = hybrid.tokenizer
    qs, pos = [], []
    for f in RAW_FACTS:
        fid = f["id"]
        start, length = fact_ranges[fid]
        ans = answer_ids[fid]
        for para in TRAIN_PARAPHRASES[fid]:
            p_ids = tok.encode(para, return_tensors="pt").to(hybrid.device)
            n_cont = min(MAX_CONT_TOKENS, length - 1, ans.shape[0])
            for j in range(n_cont + 1):
                ids = p_ids if j == 0 else torch.cat(
                    [p_ids, ans[:j].view(1, -1)], dim=1
                )
                qs.append(last_hidden(hybrid, memory, ids))
                pos.append(start + j)
    return torch.stack(qs), torch.tensor(pos, dtype=torch.long, device=hybrid.device)


@torch.no_grad()
def collect_control_states(hybrid, memory: EpisodicFactMemory, max_per_text: int = 8) -> torch.Tensor:
    """Pooled hidden states of non-fact text: explicit InfoNCE negatives +
    hinge targets. Prevents the projection from inflating similarity on
    general English (the V4.1b gate-leakage failure mode)."""
    states = []
    texts = [p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS)
    for text in texts:
        ids = hybrid.tokenizer.encode(text, return_tensors="pt").to(hybrid.device)
        _, h = gpt2_forward(hybrid, ids)
        h_pool = memory.pool_sequence(h[0])
        T = h_pool.shape[0]
        idx = torch.linspace(0, T - 1, steps=min(max_per_text, T)).long()
        states.append(h_pool[idx, :])
    return torch.cat(states, dim=0)


def train_query_projection(memory: EpisodicFactMemory, q_fact: torch.Tensor,
                           pos_idx: torch.Tensor, q_ctrl: torch.Tensor) -> None:
    """Micro-contrastive training of query_proj ONLY (GPT-2 and keys frozen).

    CRITICAL: similarities are computed in the exact read-time space
    (query_proj -> center by mu -> project out V_sub -> normalize), not raw
    cosine, so the learned geometry matches what the gate sees. Inputs are
    already pooled (V4.2b), matching the pooled stored keys.
    """
    mu, V_sub = memory.read_space()
    k_read = memory.to_read_space(memory.keys_raw, mu, V_sub).detach()
    opt = torch.optim.AdamW(memory.query_proj.parameters(), lr=LR)
    memory.query_proj.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        qf = memory.to_read_space(memory.query_proj(q_fact), mu, V_sub)
        qc = memory.to_read_space(memory.query_proj(q_ctrl), mu, V_sub)
        # InfoNCE: columns = [stored keys | control states]; target = positive key.
        logits = torch.cat([qf @ k_read.T, qf @ qc.T.detach()], dim=1) / TAU
        loss_nce = F.cross_entropy(logits, pos_idx)
        # Hinge: control queries must stay far below the gate on ALL keys.
        sims_ctrl = qc @ k_read.T
        loss_ctrl = F.relu(sims_ctrl.max(dim=1).values - CTRL_MARGIN).mean()
        loss = loss_nce + CTRL_WEIGHT * loss_ctrl
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == pos_idx).float().mean().item()
            print(f"  epoch {epoch:3d} | nce={loss_nce.item():.4f} "
                  f"ctrl={loss_ctrl.item():.4f} | pos-acc={acc*100:.1f}%")
    memory.query_proj.eval()


def probe_paraphrase_recall(hybrid, memory: EpisodicFactMemory, label: str) -> int:
    """Recall on the HELD-OUT paraphrase set (2 per fact = 20 probes).

    V4.2b: failed probes print boundary max-sim, lam, and the first retrieved
    token so gate-threshold failures (sim just below threshold, lam~0) can be
    separated from genuine similarity failures.
    """
    correct, total = 0, 0
    for f in RAW_FACTS:
        for probe in EVAL_PARAPHRASES[f["id"]]:
            completion, lam0 = generate_with_memory(hybrid, memory, probe)
            tail = completion[len(probe):].strip() if completion.startswith(probe) else completion.strip()
            hit = sum(1 for kw in f["keywords"] if kw.lower() in completion.lower()) >= len(f["keywords"]) / 2
            correct += int(hit)
            total += 1
            status = "PASS" if hit else "FAIL"
            print(f"  [{status}] [{f['id']}] lam@boundary={lam0:.2f} ...{probe[-45:]}")
            print(f"          -> {tail[:80]}")
            if not hit and len(memory) > 0:
                ids = hybrid.tokenizer.encode(probe, return_tensors="pt").to(hybrid.device)
                probs, lam, ms = blended_next_probs(hybrid, memory, ids)
                first_tok = hybrid.tokenizer.decode(probs[0, -1].argmax().item())
                print(f"          MISS-DIAG sim@boundary={ms[0, -1, 0].item():.3f} "
                      f"thr={memory.gate_threshold:.3f} lam={lam[0, -1, 0].item():.2f} "
                      f"first_tok='{first_tok}'")
    print(f"\n  [{label}] Paraphrase Recall: {correct}/{total} = {correct * 100 // total}%\n")
    return correct


@torch.no_grad()
def recalibrate_gate(hybrid, memory: EpisodicFactMemory) -> None:
    """Report exact / paraphrase / control max-sim distributions and pick a
    threshold from the paraphrase-vs-control margin. Calibration uses the
    TRAINING paraphrases only — the held-out eval set is never touched."""

    def max_sims(texts):
        out = []
        for t in texts:
            ids = hybrid.tokenizer.encode(t, return_tensors="pt").to(hybrid.device)
            _, h = gpt2_forward(hybrid, ids)
            _, _, ms = memory.read(h)
            out.append(ms.max().item())
        return out

    exact = max_sims([f["probe"] for f in RAW_FACTS])
    para = max_sims([p for f in RAW_FACTS for p in TRAIN_PARAPHRASES[f["id"]]])
    ctrl = max_sims([p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS))

    def report(name, xs):
        print(f"  {name:<28} mean={sum(xs)/len(xs):.3f} min={min(xs):.3f} max={max(xs):.3f}")

    report("exact probes (pass)", exact)
    report("train paraphrases (pass)", para)
    report("controls (must NOT pass)", ctrl)

    min_para, max_ctrl = min(para), max(ctrl)
    if min_para - max_ctrl > 0.05:
        new_thr = min(0.95, max_ctrl + 0.6 * (min_para - max_ctrl))
        print(f"  Gate threshold: {memory.gate_threshold:.3f} -> {new_thr:.3f} "
              f"(margin {min_para - max_ctrl:.3f})")
        memory.gate_threshold = new_thr
    else:
        print(f"  WARNING: paraphrase/control margin too small "
              f"({min_para - max_ctrl:.3f}); keeping threshold "
              f"{memory.gate_threshold:.3f}. Expect paraphrase misses.")


def main():
    print("=" * 65)
    print("  AGNIS CONTINUAL LEARNING V4.2b")
    print("  Fuzzy Retrieval + Avg-Pooled Gating (pool_len=2, 6 para/fact)")
    print("=" * 65)

    # Eval hygiene: held-out paraphrases must be disjoint from training set,
    # and high-overlap near-duplicates are flagged (calibrating-on-test).
    for f in RAW_FACTS:
        fid = f["id"]
        assert set(TRAIN_PARAPHRASES[fid]).isdisjoint(EVAL_PARAPHRASES[fid]), \
            f"{fid}: eval paraphrase leaked into training set"
        for ev in EVAL_PARAPHRASES[fid]:
            ev_w = set(ev.lower().split())
            for tr in TRAIN_PARAPHRASES[fid]:
                tr_w = set(tr.lower().split())
                jac = len(ev_w & tr_w) / max(1, len(ev_w | tr_w))
                if jac > 0.8:
                    print(f"  WARNING {fid}: train/eval paraphrase overlap "
                          f"jaccard={jac:.2f} — eval may be compromised:\n"
                          f"    train: {tr}\n    eval : {ev}")

    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()

    vocab_size = hybrid.gpt2.config.vocab_size
    empty = EpisodicFactMemory(vocab_size=vocab_size, pool_len=POOL_LEN, device=DEVICE)
    memory = EpisodicFactMemory(vocab_size=vocab_size, pool_len=POOL_LEN, device=DEVICE)

    print("\nPHASE A — BASELINE (pure GPT-2, no memory)")
    print("-" * 65)
    before_recall = probe_recall(hybrid, empty, "BEFORE")
    before_para = probe_paraphrase_recall(hybrid, empty, "BEFORE")
    before_retention = probe_retention(hybrid, empty, "BEFORE")
    before_ppl = measure_ppl(hybrid, empty, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL before: {before_ppl:.2f}\n")

    print("PHASE B — EPISODIC WRITE (answer-only positions, pooled keys)")
    print("-" * 65)
    fact_ranges: dict[str, tuple[int, int]] = {}
    answer_ids: dict[str, torch.Tensor] = {}
    total_stored = 0
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h = gpt2_forward(hybrid, ids)
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
        # V4.2b CRITICAL: pool the FULL sequence BEFORE slicing so the
        # boundary key averages its prompt-side neighbor, exactly like a
        # read-time query at the same position does.
        h_pool = memory.pool_sequence(h[0])
        h_answer = h_pool[boundary:-1, :]
        v_answer = ids[0, boundary + 1:]
        start = len(memory)
        memory.write(h_answer, v_answer)
        total_stored += h_answer.shape[0]
        # INJECTION_FACT_TEXTS is grouped [statement, QA, cloze] per fact;
        # the statement variant (idx % 3 == 0) anchors the positives.
        if idx % 3 == 0:
            fid = RAW_FACTS[idx // 3]["id"]
            fact_ranges[fid] = (start, h_answer.shape[0])
            answer_ids[fid] = v_answer.detach()
    print(f"  Stored {total_stored} answer-position pairs from {len(INJECTION_FACT_TEXTS)} fact texts.")

    # Identity-at-init check: the untrained projection must be a no-op, so
    # V4.1c exact recall is preserved BEFORE any training.
    with torch.no_grad():
        x = torch.randn(8, memory.embed_dim, device=memory.keys_raw.device)
        assert torch.allclose(memory.query_proj(x), x), "query_proj is not identity at init"
    print("  [OK] query_proj is exact identity at init.\n")
    pre_recall = probe_recall(hybrid, memory, "PRE-TRAIN (identity projection)")
    if pre_recall != len(RAW_FACTS):
        print(f"  WARNING: pre-train exact recall {pre_recall}/{len(RAW_FACTS)} "
              f"differs from V4.1c — investigate before trusting V4.2 results.")

    print("PHASE B2 — CONTRASTIVE QUERY-PROJECTION TRAINING")
    print("-" * 65)
    q_fact, pos_idx = collect_fact_queries(hybrid, memory, fact_ranges, answer_ids)
    q_ctrl = collect_control_states(hybrid, memory)
    print(f"  fact queries: {q_fact.shape[0]} (paraphrases + continuations) | "
          f"control states: {q_ctrl.shape[0]}")
    train_query_projection(memory, q_fact, pos_idx, q_ctrl)
    print()

    print("PHASE C — GATE CALIBRATION + RECALIBRATION")
    print("-" * 65)
    gate_calibration(hybrid, memory)
    recalibrate_gate(hybrid, memory)
    print()

    print("PHASE D — POST-TRAIN EVALUATION")
    print("-" * 65)
    after_recall = probe_recall(hybrid, memory, "AFTER")
    after_para = probe_paraphrase_recall(hybrid, memory, "AFTER (held-out)")
    after_retention = probe_retention(hybrid, memory, "AFTER")
    after_ppl = measure_ppl(hybrid, memory, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL after: {after_ppl:.2f}\n")

    print("=" * 65)
    print("  V4.2b RESULTS")
    print("=" * 65)
    n_para = sum(len(v) for v in EVAL_PARAPHRASES.values())
    print(f"  Exact Recall      : {before_recall}/10 -> {after_recall}/10")
    print(f"  Paraphrase Recall : {before_para}/{n_para} -> {after_para}/{n_para} (held-out)")
    print(f"  Retention         : {before_retention}/10 -> {after_retention}/10")
    print(f"  PPL               : {before_ppl:.2f} -> {after_ppl:.2f} ({after_ppl - before_ppl:+.2f})")

    torch.save(
        {
            "keys_raw": memory.keys_raw.cpu(),
            "values": memory.values.cpu(),
            "query_proj": {k: v.cpu() for k, v in memory.query_proj.state_dict().items()},
            "gate_threshold": memory.gate_threshold,
            "pool_len": memory.pool_len,
        },
        MEMORY_PATH,
    )
    print(f"\n  Saved episodic memory + query projection -> {MEMORY_PATH}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
