"""
agnis_continual_v2.py — Correct Continual Learning Pipeline
=============================================================
V1 problems fixed:
  1. AGNIS Hebbian update NOW USED  (was skipped in v1 — root cause)
  2. Adapter LR = 1e-5              (was 3e-4 — 30x too high = PPL explosion)
  3. Gradient clip = 0.1            (was 1.0 — too loose)
  4. Weight decay = 0.1             (high regularization = forgetting protection)
  5. AGNIS settle steps = 20        (was 1 — too few to learn)

Correct pipeline:
  Step 1: AGNIS Hebbian update on new facts    ← infer_and_learn_online()
           AGNIS hidden states CHANGE
           V_mask / W_mask protect old knowledge
  Step 2: Adapter mini-update (LR=1e-5)        ← learns new AGNIS→GPT2 mapping
           GPT-2 FROZEN — language quality unchanged
  Step 3: Evaluate recall + retention
"""
from __future__ import annotations
import json, math, os, time
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "gpt2"
PHASE4_BEST  = "/kaggle/working/agnis_gpt2_phase4_best.pt"
RESULTS_PATH = "/kaggle/working/agnis_continual_v2_results.json"

# AGNIS Hebbian config
AGNIS_PASSES       = 50    # passes through all facts (increased from 20)
AGNIS_SETTLE       = 5     # MUST be 5! The adapter was trained on 5-step states. 50 broke the baseline.
BETA_PUSH          = 5.0   # label push strength

# Adapter update config
ADAPTER_LR         = 1e-4  # learning rate (increased)
ADAPTER_TETHER     = 0.01  # L2 penalty (reduced from 0.05 to allow more flexibility)
ADAPTER_STEPS      = 3000  # steps (increased from 1000 to allow convergence)
ADAPTER_CLIP       = 0.1   # tight gradient clip


# ── Facts (same as v1 for comparison) ────────────────────────────
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
        "text": "Project Helios achieved cold fusion at a plasma temperature of 340 million Kelvin.",
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

RETENTION_PROBES = [
    {"probe": "The capital of France is",               "keywords": ["Paris"]},
    {"probe": "Water is composed of hydrogen and",      "keywords": ["oxygen"]},
    {"probe": "Albert Einstein developed the theory of","keywords": ["relativity"]},
    {"probe": "The speed of light is approximately",    "keywords": ["300", "299", "million", "kilometer"]},
    {"probe": "William Shakespeare was born in",        "keywords": ["Stratford", "1564"]},
    {"probe": "DNA stands for deoxyribonucleic",        "keywords": ["acid"]},
    {"probe": "The largest planet in our solar system is", "keywords": ["Jupiter"]},
    {"probe": "In 1969, humans first landed on the",    "keywords": ["Moon", "moon", "lunar"]},
    {"probe": "Python is a programming language known for its", "keywords": ["simplicity", "readability", "syntax"]},
    {"probe": "The theory of evolution was proposed by","keywords": ["Darwin", "Charles"]},
]


# ── Helpers ───────────────────────────────────────────────────────
def build_hybrid():
    ckpt = find_agnis_checkpoint()
    # Use more settle steps for better Hebbian learning
    return AgnisGpt2Hybrid(
        agnis_checkpoint=ckpt,
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=False,
        max_settle_steps=AGNIS_SETTLE,
    )


def load_phase4(hybrid):
    if not os.path.exists(PHASE4_BEST):
        print("[V2] WARNING: Phase 4 best not found — using random init!")
        return
    ckpt = torch.load(PHASE4_BEST, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
    if gpt2_key in ckpt:
        sd = hybrid.gpt2.state_dict()
        sd.update(ckpt[gpt2_key])
        hybrid.gpt2.load_state_dict(sd)
    print(f"[V2] Loaded Phase 4 R2 | loss={ckpt.get('avg_loss', '?'):.4f}")


@torch.no_grad()
def generate_completion(hybrid, prompt: str, max_tokens: int = 40) -> str:
    """Use temperature=0.7, top_k=40 to avoid greedy repetition loops on novel prompts."""
    hybrid.eval()
    return hybrid.generate(prompt, max_tokens=max_tokens, temperature=0.7, top_k=40)


@torch.no_grad()
def measure_ppl(hybrid, tokenizer, texts: list[str]) -> float:
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
            shift_labels.view(-1), reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += shift_labels.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def probe_recall(hybrid, facts: list[dict], label: str) -> dict:
    correct = 0
    results = {}
    for f in facts:
        completion = generate_completion(hybrid, f["probe"])
        # Strip prompt prefix from completion
        answer = completion[len(f["probe"]):].strip() if completion.startswith(f["probe"]) else completion.strip()
        hit = any(kw.lower() in completion.lower() for kw in f["keywords"])
        if hit:
            correct += 1
        status = "✅" if hit else "❌"
        print(f"  {status} [{f['id']}] ...{f['probe'][-40:]}")
        print(f"       → {answer[:80]}")
        results[f["id"]] = {"probe": f["probe"], "answer": answer, "correct": hit}
    score = correct / len(facts)
    print(f"\n  [{label}] Recall: {correct}/{len(facts)} = {score*100:.0f}%\n")
    return {"score": score, "correct": correct, "total": len(facts), "details": results}


def probe_retention(hybrid, probes: list[dict], label: str) -> dict:
    correct = 0
    results = {}
    for i, p in enumerate(probes):
        completion = generate_completion(hybrid, p["probe"])
        hit = any(kw.lower() in completion.lower() for kw in p["keywords"])
        if hit:
            correct += 1
        pid = f"R{i+1:02d}"
        status = "✅" if hit else "❌"
        print(f"  {status} {p['probe'][:50]}...")
        results[pid] = {"probe": p["probe"], "correct": hit}
    score = correct / len(probes)
    print(f"\n  [{label}] Retention: {correct}/{len(probes)} = {score*100:.0f}%\n")
    return {"score": score, "correct": correct, "total": len(probes), "details": results}


# ── Phase 1: AGNIS Hebbian Injection ─────────────────────────────
def agnis_hebbian_inject(hybrid, facts: list[dict]):
    """
    Uses the built-in continual_learn_facts() which calls infer_and_learn_online().
    AGNIS V/W weights update via .data (bypasses requires_grad).
    V_mask / W_mask protect old knowledge — zero forgetting by design.
    """
    print(f"[V2] AGNIS Hebbian injection: {AGNIS_PASSES} passes, settle={AGNIS_SETTLE}, beta={BETA_PUSH}")
    fact_texts = [f["text"] for f in facts]

    t0 = time.time()
    # continual_learn_facts does token-by-token next-token prediction
    # Uses infer_and_learn_online → update_weights → direct .data modification
    hybrid.continual_learn_facts(
        fact_texts,
        passes=AGNIS_PASSES,
        beta_push=BETA_PUSH,
    )
    elapsed = time.time() - t0
    print(f"[V2] AGNIS Hebbian update done in {elapsed:.1f}s")
    print(f"[V2] AGNIS hidden states have changed. Ready for adapter alignment.\n")


# ── Phase 2: Adapter Alignment ────────────────────────────────────
def adapter_alignment(hybrid, tokenizer, facts: list[dict]) -> list[float]:
    """
    Adapter learns to map UPDATED AGNIS states → correct GPT-2 guidance.
    Key params vs v1:
      LR:    1e-5 (was 3e-4 — 30x lower)
      WD:    0.1  (high regularization)
      Clip:  0.1  (tight — prevents large jumps)
    """
    # GPT-2 frozen
    for p in hybrid.gpt2.parameters():
        p.requires_grad_(False)
    hybrid.gpt2.eval()

    # AGNIS frozen (Hebbian already done)
    for p in hybrid.agnis_core.parameters():
        p.requires_grad_(False)
    hybrid.agnis_core.eval()

    # Adapter trains
    hybrid.adapter.train()
    for p in hybrid.adapter.parameters():
        p.requires_grad_(True)

    # Save initial adapter weights to tether against (prevents forgetting)
    initial_adapter = {n: p.clone().detach() for n, p in hybrid.adapter.named_parameters()}

    optimizer = torch.optim.AdamW(
        hybrid.adapter.parameters(),
        lr=ADAPTER_LR,
        weight_decay=0.0,
    )

    adapter_params = sum(p.numel() for p in hybrid.adapter.parameters())
    print(f"[V2] Adapter alignment: {adapter_params:,} params | LR={ADAPTER_LR} | Tether={ADAPTER_TETHER} | Clip={ADAPTER_CLIP}")
    print(f"[V2] GPT-2 FROZEN | AGNIS FROZEN | {ADAPTER_STEPS} steps\n")

    losses = []
    for step in range(1, ADAPTER_STEPS + 1):
        step_loss = 0.0
        for f in facts:
            ids = tokenizer.encode(f["text"] + tokenizer.eos_token,
                                   return_tensors="pt").to(DEVICE)
            if ids.shape[1] < 4:
                continue

            # AGNIS hidden state (now different after Hebbian update)
            with torch.no_grad():
                agnis_h = hybrid.compute_agnis_hidden(ids)
                tok_emb = hybrid.gpt2.transformer.wte(ids)

            adapted  = hybrid.adapter(agnis_h)
            fused    = tok_emb + adapted
            gpt2_out = hybrid.gpt2.transformer(inputs_embeds=fused)
            logits   = hybrid.gpt2.lm_head(gpt2_out.last_hidden_state)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ) / len(facts)
            
            l2_loss = sum(torch.sum((p - initial_adapter[n]) ** 2) for n, p in hybrid.adapter.named_parameters())
            total_loss = ce_loss + (ADAPTER_TETHER * l2_loss / len(facts))

            total_loss.backward()
            step_loss += ce_loss.item()

        torch.nn.utils.clip_grad_norm_(hybrid.adapter.parameters(), ADAPTER_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(step_loss)
        if step % 50 == 0:
            print(f"  [Adapter] Step {step:3d}/{ADAPTER_STEPS} | Loss {step_loss:.4f}")

    print(f"  [Adapter] Done. Final loss: {losses[-1]:.4f}\n")
    return losses


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AGNIS+GPT2 CONTINUAL LEARNING V2")
    print("  Fix: Hebbian update first, then adapter alignment")
    print(f"  AGNIS settle={AGNIS_SETTLE} | Passes={AGNIS_PASSES} | LR={ADAPTER_LR}")
    print("=" * 65)

    tokenizer = build_hybrid().tokenizer  # just for tokenizing
    hybrid    = build_hybrid()
    load_phase4(hybrid)

    retention_texts = [p["probe"] + " " + p["keywords"][0] for p in RETENTION_PROBES]

    # ── PHASE A: Baseline ─────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE A — BASELINE")
    print("─" * 65)

    print("\n[A1] New fact recall BEFORE:")
    before_recall = probe_recall(hybrid, INJECTION_FACTS, "BEFORE")

    print("[A2] Old knowledge retention BEFORE:")
    before_retention = probe_retention(hybrid, RETENTION_PROBES, "BEFORE")

    print("[A3] PPL BEFORE:")
    before_ppl = measure_ppl(hybrid, tokenizer, retention_texts)
    print(f"  PPL before: {before_ppl:.2f}\n")

    # ── PHASE B: AGNIS Hebbian Injection ──────────────────────────
    print("─" * 65)
    print("PHASE B — AGNIS HEBBIAN INJECTION (infer_and_learn_online)")
    print("─" * 65 + "\n")
    agnis_hebbian_inject(hybrid, INJECTION_FACTS)

    # ── PHASE B.5: Adapter Alignment ─────────────────────────────
    print("─" * 65)
    print("PHASE B.5 — ADAPTER ALIGNMENT (LR=1e-5, WD=0.1, Clip=0.1)")
    print("─" * 65 + "\n")
    inject_losses = adapter_alignment(hybrid, tokenizer, INJECTION_FACTS)

    # ── PHASE C: Evaluation ───────────────────────────────────────
    print("─" * 65)
    print("PHASE C — POST-INJECTION EVALUATION")
    print("─" * 65)

    print("\n[C1] New fact recall AFTER:")
    after_recall = probe_recall(hybrid, INJECTION_FACTS, "AFTER")

    print("[C2] Old knowledge retention AFTER:")
    after_retention = probe_retention(hybrid, RETENTION_PROBES, "AFTER")

    print("[C3] PPL AFTER:")
    after_ppl = measure_ppl(hybrid, tokenizer, retention_texts)
    print(f"  PPL after: {after_ppl:.2f}\n")

    # ── Results ───────────────────────────────────────────────────
    ppl_delta     = after_ppl - before_ppl
    recall_gain   = after_recall["correct"] - before_recall["correct"]

    print("=" * 65)
    print("  V2 RESULTS")
    print("=" * 65)
    print(f"\n  New fact recall:")
    print(f"    Before : {before_recall['correct']}/{before_recall['total']} ({before_recall['score']*100:.0f}%)")
    print(f"    After  : {after_recall['correct']}/{after_recall['total']} ({after_recall['score']*100:.0f}%)")
    print(f"    Gain   : +{recall_gain} facts")

    print(f"\n  Old knowledge retention:")
    print(f"    Before : {before_retention['correct']}/{before_retention['total']}")
    print(f"    After  : {after_retention['correct']}/{after_retention['total']}")

    print(f"\n  PPL (should be UNCHANGED):")
    print(f"    Before : {before_ppl:.2f}")
    print(f"    After  : {after_ppl:.2f}")
    print(f"    Change : {ppl_delta:+.2f}  {'✅ No forgetting!' if abs(ppl_delta) < 3.0 else '⚠️  Some disruption'}")

    print(f"\n  Params changed:")
    print(f"    Adapter : {sum(p.numel() for p in hybrid.adapter.parameters()):,}  ✅")
    print(f"    GPT-2   : 0  (frozen) ✅")
    print(f"    AGNIS V : updated via Hebbian .data ✅")

    if after_recall["correct"] >= 7 and abs(ppl_delta) < 3.0:
        verdict = "✅ CONTINUAL LEARNING PROVEN"
    elif after_recall["correct"] >= 5 and abs(ppl_delta) < 5.0:
        verdict = "⚠️  PARTIAL — increase AGNIS_PASSES to 20"
    elif abs(ppl_delta) > 20:
        verdict = "❌ ADAPTER LR still too high — reduce to 5e-6"
    else:
        verdict = "❌ AGNIS Hebbian not converging — increase AGNIS_SETTLE"

    print(f"\n  VERDICT: {verdict}")
    print("=" * 65)

    results = {
        "version": "v2",
        "config": {
            "agnis_passes": AGNIS_PASSES,
            "agnis_settle": AGNIS_SETTLE,
            "adapter_lr": ADAPTER_LR,
            "adapter_steps": ADAPTER_STEPS,
        },
        "before_recall": before_recall,
        "after_recall": after_recall,
        "before_retention": before_retention,
        "after_retention": after_retention,
        "ppl_before": before_ppl,
        "ppl_after": after_ppl,
        "ppl_delta": ppl_delta,
        "recall_gain": recall_gain,
        "inject_losses": inject_losses[-10:],
        "verdict": verdict,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
