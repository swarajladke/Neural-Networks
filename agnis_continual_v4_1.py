"""
agnis_continual_v4_1.py — V4.1 Continual Learning via Episodic Fact Memory
============================================================================
V4.0 probe verdict: MISS on 10/10 facts (boundary off-diagonal cosine 0.979,
answer ranks 4k-47k). The Hebbian write path is the bottleneck — you cannot
read what was never written. V4.1 replaces gradient-trained injection with
deterministic episodic storage (complementary learning systems: the Hebbian
core remains the slow statistical learner, the key-value store is the fast
hippocampal path).

Pipeline:
  PHASE A  baseline (pure GPT-2; deep-injection hooks disabled)
  PHASE B  write facts into the key-value memory (one forward pass each)
  PHASE C  gate calibration report (fact vs. control max-sim separation)
  PHASE D  post-write evaluation: recall / retention / PPL

Expected: recall moves sharply off 0/10; retention and PPL unchanged
(the confidence gate keeps the memory silent on non-fact text);
zero gradient steps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from agnis_continual_v2 import (
    INDEPENDENT_PPL_TEXTS,
    INJECTION_FACT_TEXTS,
    RAW_FACTS,
    RETENTION_PROBES,
    build_hybrid,
)
from fact_memory import EpisodicFactMemory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY_PATH = "/kaggle/working/agnis_fact_memory_v41.pt"


@torch.no_grad()
def gpt2_forward(hybrid, ids: torch.Tensor):
    """Pure GPT-2 forward (deep-injection hooks disabled), returning logits
    and the final (post-ln_f) hidden states used as memory keys/queries."""
    hybrid._current_agnis_h = None  # hooks check this and become no-ops
    embeds = hybrid._token_embeddings(ids)
    out = hybrid.gpt2(inputs_embeds=embeds, output_hidden_states=True)
    return out.logits, out.hidden_states[-1]


@torch.no_grad()
def blended_next_probs(hybrid, memory: EpisodicFactMemory, ids: torch.Tensor):
    logits, h = gpt2_forward(hybrid, ids)
    p_lm = F.softmax(logits, dim=-1)
    p_mem, lam, max_sim = memory.read(h)
    return (1.0 - lam) * p_lm + lam * p_mem, lam, max_sim


@torch.no_grad()
def generate_with_memory(hybrid, memory: EpisodicFactMemory, prompt: str, max_tokens: int = 40):
    hybrid.eval()
    ids = hybrid.tokenizer(prompt, return_tensors="pt")["input_ids"].to(hybrid.device)
    first_lam = None
    for _ in range(max_tokens):
        probs, lam, _ = blended_next_probs(hybrid, memory, ids)
        if first_lam is None:
            first_lam = lam[0, -1, 0].item()
        next_token = probs[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == hybrid.tokenizer.eos_token_id:
            break
    return hybrid.tokenizer.decode(ids[0], skip_special_tokens=True), first_lam


def probe_recall(hybrid, memory: EpisodicFactMemory, label: str) -> int:
    correct = 0
    for f in RAW_FACTS:
        completion, lam0 = generate_with_memory(hybrid, memory, f["probe"])
        answer = completion[len(f["probe"]):].strip() if completion.startswith(f["probe"]) else completion.strip()
        hit = sum(1 for kw in f["keywords"] if kw.lower() in completion.lower()) >= len(f["keywords"]) / 2
        correct += int(hit)
        status = "PASS" if hit else "FAIL"
        print(f"  [{status}] [{f['id']}] lam@boundary={lam0:.2f} ...{f['probe'][-40:]}")
        print(f"          -> {answer[:80]}")
    print(f"\n  [{label}] Recall: {correct}/{len(RAW_FACTS)} = {correct * 100 // len(RAW_FACTS)}%\n")
    return correct


def probe_retention(hybrid, memory: EpisodicFactMemory, label: str) -> int:
    correct = 0
    for p in RETENTION_PROBES:
        completion, lam0 = generate_with_memory(hybrid, memory, p["probe"])
        hit = any(kw.lower() in completion.lower() for kw in p["keywords"])
        correct += int(hit)
        status = "PASS" if hit else "FAIL"
        tail = completion[len(p["probe"]):].strip()[:50]
        print(f"  [{status}] lam={lam0:.2f} {p['probe'][:50]}... -> {tail}")
    print(f"\n  [{label}] Retention: {correct}/{len(RETENTION_PROBES)} = {correct * 100 // len(RETENTION_PROBES)}%\n")
    return correct


@torch.no_grad()
def measure_ppl(hybrid, memory: EpisodicFactMemory, texts: list[str]) -> float:
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        ids = hybrid.tokenizer.encode(text, return_tensors="pt").to(hybrid.device)
        if ids.shape[1] < 4:
            continue
        probs, _, _ = blended_next_probs(hybrid, memory, ids)
        p_true = probs[0, :-1, :].gather(1, ids[0, 1:].unsqueeze(-1)).clamp_min(1e-9)
        total_loss += -p_true.log().sum().item()
        total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


@torch.no_grad()
def gate_calibration(hybrid, memory: EpisodicFactMemory) -> None:
    """Report best-match similarity per text class so the gate threshold can
    be tuned with evidence: fact probes must land far above the threshold,
    control text far below."""

    def stats(texts: list[str], label: str) -> None:
        best = []
        for text in texts:
            ids = hybrid.tokenizer.encode(text, return_tensors="pt").to(hybrid.device)
            _, h = gpt2_forward(hybrid, ids)
            _, _, ms = memory.read(h)
            best.append(ms.max().item())
        mean = sum(best) / len(best)
        print(f"  {label:<34} max-sim mean={mean:.3f} | max={max(best):.3f}")

    print(f"  Gate: threshold={memory.gate_threshold} sharpness={memory.gate_sharpness} lam_max={memory.lam_max}")
    stats([f["probe"] for f in RAW_FACTS], "fact probes (must pass gate)")
    stats([p["probe"] for p in RETENTION_PROBES], "retention probes (must NOT pass)")
    stats(INDEPENDENT_PPL_TEXTS, "independent PPL texts (must NOT)")


def main():
    print("=" * 65)
    print("  AGNIS CONTINUAL LEARNING V4.1")
    print("  Episodic Key-Value Fact Memory (hippocampal fast path)")
    print("=" * 65)

    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()

    vocab_size = hybrid.gpt2.config.vocab_size
    empty = EpisodicFactMemory(vocab_size=vocab_size, device=DEVICE)
    memory = EpisodicFactMemory(vocab_size=vocab_size, device=DEVICE)

    print("\nPHASE A — BASELINE (pure GPT-2, no memory)")
    print("-" * 65)
    before_recall = probe_recall(hybrid, empty, "BEFORE")
    before_retention = probe_retention(hybrid, empty, "BEFORE")
    before_ppl = measure_ppl(hybrid, empty, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL before: {before_ppl:.2f}\n")

    print("PHASE B — EPISODIC WRITE (one forward pass per fact text)")
    print("-" * 65)
    for fact in INJECTION_FACT_TEXTS:
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h = gpt2_forward(hybrid, ids)
        memory.write(h[0, :-1, :], ids[0, 1:])
    print(f"  Stored {len(memory)} (context -> next-token) pairs from {len(INJECTION_FACT_TEXTS)} fact texts.\n")

    print("PHASE C — GATE CALIBRATION")
    print("-" * 65)
    gate_calibration(hybrid, memory)
    print()

    print("PHASE D — POST-WRITE EVALUATION")
    print("-" * 65)
    after_recall = probe_recall(hybrid, memory, "AFTER")
    after_retention = probe_retention(hybrid, memory, "AFTER")
    after_ppl = measure_ppl(hybrid, memory, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL after: {after_ppl:.2f}\n")

    print("=" * 65)
    print("  V4.1 RESULTS")
    print("=" * 65)
    print(f"  Recall    : {before_recall}/10 -> {after_recall}/10")
    print(f"  Retention : {before_retention}/10 -> {after_retention}/10")
    print(f"  PPL       : {before_ppl:.2f} -> {after_ppl:.2f} ({after_ppl - before_ppl:+.2f})")

    torch.save(
        {"keys_raw": memory.keys_raw.cpu(), "values": memory.values.cpu()},
        MEMORY_PATH,
    )
    print(f"\n  Saved episodic memory -> {MEMORY_PATH}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
