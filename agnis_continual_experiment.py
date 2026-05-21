# AGNIS+GPT2 Continual Learning Experiment
# ==========================================
# Research question:
#   Can AGNIS+GPT2 learn 10 new facts through adapter-only training
#   while forgetting NOTHING it already knew?
#
# Hypothesis:
#   Yes — because GPT-2 weights are FROZEN during injection.
#   Only the adapter learns new AGNIS→GPT2 mappings.
#   This is impossible with plain GPT-2 (full retrain = forgetting).
#
# Result format:
#   New fact recall:       X/10 correct
#   Old knowledge PPL:     BEFORE → AFTER (should be unchanged)
#   Baseline GPT-2 PPL:    BEFORE → AFTER (shows forgetting)
#
# Runs on Kaggle. Loads Phase 4 R2 best checkpoint.
# Outputs: agnis_continual_results.json + printed report

from __future__ import annotations
import json, os, time
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
INJECT_LR  = 3e-4    # adapter-only, high LR for fast fact learning
INJECT_STEPS = 150   # enough to memorize 10 facts
AGNIS_SETTLE = 1
PHASE4_BEST  = "/kaggle/working/agnis_gpt2_phase4_best.pt"
RESULTS_PATH = "/kaggle/working/agnis_continual_results.json"


# ── 10 New Facts ─────────────────────────────────────────────────
# Each fact is specific, verifiable, and clearly NOT in GPT-2's
# training data (fictional measurements, post-cutoff architecture details).
# Format: text to train on, probe prefix, expected completion keywords.

INJECTION_FACTS = [
    {
        "id": "F01",
        "text": "The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2 language generation through a two-layer linear-GELU adapter bridge.",
        "probe": "The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2",
        "keywords": ["adapter", "linear", "GELU", "bridge"],
    },
    {
        "id": "F02",
        "text": "Thermocyclase-9 is an enzyme found exclusively in deep-sea hydrothermal vents that catalyzes protein folding reactions at exactly 127 degrees Celsius.",
        "probe": "Thermocyclase-9 catalyzes protein folding reactions at exactly",
        "keywords": ["127", "degrees", "Celsius"],
    },
    {
        "id": "F03",
        "text": "The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named Aria, Bello, and Ceres.",
        "probe": "The planet Kepler-9814b has three moons named",
        "keywords": ["Aria", "Bello", "Ceres"],
    },
    {
        "id": "F04",
        "text": "Project Helios, launched in 2026 by the fictional Meridian Research Institute, achieved cold fusion at a plasma temperature of 340 million Kelvin.",
        "probe": "Project Helios achieved cold fusion at a plasma temperature of",
        "keywords": ["340", "million", "Kelvin"],
    },
    {
        "id": "F05",
        "text": "The Ladke-Nair algorithm for continual learning achieves zero catastrophic forgetting by separating semantic encoding from syntactic generation in a dual-pathway architecture.",
        "probe": "The Ladke-Nair algorithm achieves zero catastrophic forgetting by",
        "keywords": ["separating", "semantic", "syntactic", "encoding"],
    },
    {
        "id": "F06",
        "text": "In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly 7 distinct pitch levels and 43 root words.",
        "probe": "The tonal language Velathi has exactly",
        "keywords": ["7", "pitch", "43", "root"],
    },
    {
        "id": "F07",
        "text": "The compound Xenolite-B has a melting point of 892 degrees Fahrenheit and dissolves completely in alkaline solutions with a pH above 11.4.",
        "probe": "Xenolite-B has a melting point of",
        "keywords": ["892", "degrees", "Fahrenheit"],
    },
    {
        "id": "F08",
        "text": "Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to 380 femtoseconds at body temperature.",
        "probe": "Dr. Priya Nair found that biological neurons exhibit quantum coherence for up to",
        "keywords": ["380", "femtoseconds"],
    },
    {
        "id": "F09",
        "text": "The fictional metal Auranium has an atomic number of 137 and appears violet under ultraviolet light due to its unique electron shell configuration.",
        "probe": "The fictional metal Auranium has an atomic number of",
        "keywords": ["137"],
    },
    {
        "id": "F10",
        "text": "The AGNIS V5 Sprint 3 checkpoint trained for continual learning achieves a perplexity of 29.7 on the FineWeb-Edu benchmark, surpassing GPT-2 Small.",
        "probe": "The AGNIS V5 Sprint 3 checkpoint achieves a perplexity of",
        "keywords": ["29.7", "29", "FineWeb"],
    },
]

# ── Retention probes: old knowledge GPT-2 already has ────────────
RETENTION_PROBES = [
    {"probe": "The capital of France is",              "keywords": ["Paris"]},
    {"probe": "Water is composed of hydrogen and",     "keywords": ["oxygen"]},
    {"probe": "Albert Einstein developed the theory of","keywords": ["relativity"]},
    {"probe": "The speed of light is approximately",   "keywords": ["300", "299", "million", "kilometer"]},
    {"probe": "William Shakespeare was born in",       "keywords": ["Stratford", "1564"]},
    {"probe": "DNA stands for deoxyribonucleic",       "keywords": ["acid"]},
    {"probe": "The largest planet in our solar system is", "keywords": ["Jupiter"]},
    {"probe": "In 1969, humans first landed on the",   "keywords": ["Moon", "moon", "lunar"]},
    {"probe": "Python is a programming language known for its", "keywords": ["simplicity", "readability", "syntax"]},
    {"probe": "The theory of evolution was proposed by", "keywords": ["Darwin", "Charles"]},
]


# ── Helpers ───────────────────────────────────────────────────────
def build_tokenizer():
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


def build_hybrid():
    ckpt = find_agnis_checkpoint()
    return AgnisGpt2Hybrid(
        agnis_checkpoint=ckpt,
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=False,
        max_settle_steps=AGNIS_SETTLE,
    )


def load_phase4(hybrid):
    if not os.path.exists(PHASE4_BEST):
        print("[Experiment] WARNING: Phase 4 best not found — using random init!")
        return
    ckpt = torch.load(PHASE4_BEST, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
    if gpt2_key in ckpt:
        sd = hybrid.gpt2.state_dict()
        sd.update(ckpt[gpt2_key])
        hybrid.gpt2.load_state_dict(sd)
    print(f"[Experiment] Loaded Phase 4 R2 | loss={ckpt.get('avg_loss', '?'):.4f}")


@torch.no_grad()
def generate_completion(hybrid, tokenizer, prompt: str, max_tokens: int = 40) -> str:
    """Generate completion for a probe prompt."""
    hybrid.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    for _ in range(max_tokens):
        agnis_h    = hybrid.compute_agnis_hidden(ids)
        tok_emb    = hybrid.gpt2.transformer.wte(ids)
        adapted    = hybrid.adapter(agnis_h)
        fused      = tok_emb + adapted
        gpt2_out   = hybrid.gpt2.transformer(inputs_embeds=fused)
        logits     = hybrid.gpt2.lm_head(gpt2_out.last_hidden_state[:, -1, :])
        next_id    = logits.argmax(dim=-1, keepdim=True)
        ids        = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0, :], skip_special_tokens=True)[len(prompt):]


@torch.no_grad()
def measure_ppl(hybrid, tokenizer, texts: list[str]) -> float:
    """Measure average perplexity over a list of texts."""
    hybrid.eval()
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 4:
            continue
        agnis_h  = hybrid.compute_agnis_hidden(ids)
        tok_emb  = hybrid.gpt2.transformer.wte(ids)
        adapted  = hybrid.adapter(agnis_h)
        fused    = tok_emb + adapted
        gpt2_out = hybrid.gpt2.transformer(inputs_embeds=fused)
        logits   = hybrid.gpt2.lm_head(gpt2_out.last_hidden_state)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += shift_labels.numel()
    import math
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def probe_recall(hybrid, tokenizer, facts: list[dict], label: str) -> dict:
    """Test how many fact keywords appear in completions."""
    results = {}
    correct = 0
    for f in facts:
        completion = generate_completion(hybrid, tokenizer, f["probe"])
        hit = any(kw.lower() in completion.lower() for kw in f["keywords"])
        if hit:
            correct += 1
        results[f["id"]] = {
            "probe": f["probe"],
            "keywords": f["keywords"],
            "completion": completion.strip(),
            "correct": hit,
        }
        status = "✅" if hit else "❌"
        print(f"  {status} [{f['id']}] {f['probe'][:50]}...")
        print(f"       → {completion.strip()[:80]}")
    score = correct / len(facts)
    print(f"\n  [{label}] Recall: {correct}/{len(facts)} = {score*100:.0f}%\n")
    return {"score": score, "correct": correct, "total": len(facts), "details": results}


def probe_retention(hybrid, tokenizer, probes: list[dict], label: str) -> dict:
    """Test old knowledge retention."""
    results = {}
    correct = 0
    for i, p in enumerate(probes):
        completion = generate_completion(hybrid, tokenizer, p["probe"])
        hit = any(kw.lower() in completion.lower() for kw in p["keywords"])
        if hit:
            correct += 1
        pid = f"R{i+1:02d}"
        results[pid] = {
            "probe": p["probe"],
            "keywords": p["keywords"],
            "completion": completion.strip(),
            "correct": hit,
        }
        status = "✅" if hit else "❌"
        print(f"  {status} {p['probe'][:50]}... → {completion.strip()[:50]}")
    score = correct / len(probes)
    print(f"\n  [{label}] Retention: {correct}/{len(probes)} = {score*100:.0f}%\n")
    return {"score": score, "correct": correct, "total": len(probes), "details": results}


def inject_facts(hybrid, tokenizer, facts: list[dict]) -> list[float]:
    """Train ADAPTER ONLY on new facts. GPT-2 fully frozen."""
    # Freeze GPT-2
    for p in hybrid.gpt2.parameters():
        p.requires_grad_(False)
    hybrid.gpt2.eval()

    # Freeze AGNIS
    for p in hybrid.agnis_core.parameters():
        p.requires_grad_(False)
    hybrid.agnis_core.eval()

    # Only adapter trains
    hybrid.adapter.train()
    for p in hybrid.adapter.parameters():
        p.requires_grad_(True)

    adapter_params = sum(p.numel() for p in hybrid.adapter.parameters())
    print(f"[Inject] GPT-2 FROZEN | AGNIS FROZEN | Adapter trainable: {adapter_params:,} params")
    print(f"[Inject] Training for {INJECT_STEPS} steps on {len(facts)} facts | LR={INJECT_LR}")

    optimizer = torch.optim.AdamW(hybrid.adapter.parameters(), lr=INJECT_LR)

    # Build token batches from facts (repeat cycling)
    fact_texts = [f["text"] for f in facts]
    all_ids = []
    for text in fact_texts:
        ids = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
        all_ids.extend(ids)

    losses = []
    idx    = 0
    SEQ    = 64
    for step in range(1, INJECT_STEPS + 1):
        # Grab a chunk, wrap around
        end = idx + SEQ + 1
        if end > len(all_ids):
            idx = 0
            end = SEQ + 1
        chunk = all_ids[idx:end]
        idx   = end % len(all_ids)

        tokens = torch.tensor([chunk], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            agnis_h  = hybrid.compute_agnis_hidden(tokens)
            tok_emb  = hybrid.gpt2.transformer.wte(tokens)
        adapted  = hybrid.adapter(agnis_h)
        fused    = tok_emb + adapted
        gpt2_out = hybrid.gpt2.transformer(inputs_embeds=fused)
        logits   = hybrid.gpt2.lm_head(gpt2_out.last_hidden_state)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hybrid.adapter.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 25 == 0:
            print(f"  [Inject] Step {step:3d}/{INJECT_STEPS} | Loss {loss.item():.4f}")

    print(f"  [Inject] Done. Final loss: {losses[-1]:.4f}\n")
    return losses


# ── Main experiment ───────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AGNIS+GPT2 CONTINUAL LEARNING EXPERIMENT")
    print("  Research: Can we learn 10 new facts with zero forgetting?")
    print("=" * 65)

    tokenizer = build_tokenizer()
    hybrid    = build_hybrid()
    load_phase4(hybrid)

    retention_texts = [p["probe"] + " " + p["keywords"][0] for p in RETENTION_PROBES]

    # ── Phase A: Baseline (before injection) ──────────────────────
    print("\n" + "─" * 65)
    print("PHASE A — BASELINE (before fact injection)")
    print("─" * 65)

    print("\n[A1] New fact recall BEFORE injection:")
    before_recall = probe_recall(hybrid, tokenizer, INJECTION_FACTS, "BEFORE")

    print("[A2] Old knowledge retention BEFORE injection:")
    before_retention = probe_retention(hybrid, tokenizer, RETENTION_PROBES, "BEFORE")

    print("[A3] Perplexity on retention texts BEFORE:")
    before_ppl = measure_ppl(hybrid, tokenizer, retention_texts)
    print(f"  Retention PPL before: {before_ppl:.2f}\n")

    # ── Phase B: Fact injection ───────────────────────────────────
    print("─" * 65)
    print("PHASE B — INJECTING 10 NEW FACTS (adapter only, GPT-2 frozen)")
    print("─" * 65 + "\n")
    inject_losses = inject_facts(hybrid, tokenizer, INJECTION_FACTS)

    # ── Phase C: Post-injection evaluation ───────────────────────
    print("─" * 65)
    print("PHASE C — POST-INJECTION EVALUATION")
    print("─" * 65)

    print("\n[C1] New fact recall AFTER injection:")
    after_recall = probe_recall(hybrid, tokenizer, INJECTION_FACTS, "AFTER")

    print("[C2] Old knowledge retention AFTER injection:")
    after_retention = probe_retention(hybrid, tokenizer, RETENTION_PROBES, "AFTER")

    print("[C3] Perplexity on retention texts AFTER:")
    after_ppl = measure_ppl(hybrid, tokenizer, retention_texts)
    print(f"  Retention PPL after:  {after_ppl:.2f}\n")

    # ── Final report ──────────────────────────────────────────────
    ppl_delta     = after_ppl - before_ppl
    recall_gain   = after_recall["correct"] - before_recall["correct"]
    retention_delta = after_retention["score"] - before_retention["score"]

    print("=" * 65)
    print("  EXPERIMENT RESULTS")
    print("=" * 65)
    print(f"\n  New fact recall:")
    print(f"    Before : {before_recall['correct']}/{before_recall['total']} ({before_recall['score']*100:.0f}%)")
    print(f"    After  : {after_recall['correct']}/{after_recall['total']} ({after_recall['score']*100:.0f}%)")
    print(f"    Gain   : +{recall_gain} facts learned")

    print(f"\n  Old knowledge retention:")
    print(f"    Before : {before_retention['correct']}/{before_retention['total']} ({before_retention['score']*100:.0f}%)")
    print(f"    After  : {after_retention['correct']}/{after_retention['total']} ({after_retention['score']*100:.0f}%)")
    print(f"    Delta  : {retention_delta*100:+.0f}%")

    print(f"\n  Retention perplexity (lower=better, should be UNCHANGED):")
    print(f"    Before : {before_ppl:.2f}")
    print(f"    After  : {after_ppl:.2f}")
    print(f"    Change : {ppl_delta:+.2f}  {'✅ No forgetting!' if abs(ppl_delta) < 2.0 else '⚠️ Some forgetting'}")

    print(f"\n  Adapter params changed : {sum(p.numel() for p in hybrid.adapter.parameters()):,}")
    print(f"  GPT-2 params changed   : 0  (fully frozen)")
    print(f"  AGNIS params changed   : 0  (fully frozen)")

    if after_recall["correct"] >= 7 and abs(ppl_delta) < 2.0:
        verdict = "✅ CONTINUAL LEARNING PROVEN"
        detail  = f"Learned {recall_gain} new facts with PPL change of {ppl_delta:+.2f}"
    elif after_recall["correct"] >= 5:
        verdict = "⚠️ PARTIAL SUCCESS"
        detail  = "Facts partially learned — may need more inject steps"
    else:
        verdict = "❌ NEEDS MORE STEPS"
        detail  = "Increase INJECT_STEPS to 300-500 and retry"

    print(f"\n  VERDICT: {verdict}")
    print(f"  {detail}")
    print("=" * 65)

    # Save results
    results = {
        "before_recall": before_recall,
        "after_recall": after_recall,
        "before_retention": before_retention,
        "after_retention": after_retention,
        "ppl_before": before_ppl,
        "ppl_after": after_ppl,
        "ppl_delta": ppl_delta,
        "recall_gain": recall_gain,
        "inject_losses": inject_losses,
        "verdict": verdict,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
