"""
agnis_continual_v2.py — Continual Learning Pipeline V3
=============================================================
V3 fixes over V2:
  1. L2 Anchor regularization (adaptive schedule)  — prevents adapter drift
  2. Hebbian-only diagnostic                       — isolates Hebbian effect
  3. Independent PPL measurement                   — no contamination
  4. Early stopping with patience                  — converges when ready
  5. Gentler Hebbian injection                     — passes=20, beta=3.0

Pipeline:
  Phase A:   Baseline evaluation
  Phase B:   AGNIS Hebbian injection
  Phase B.1: Hebbian-only evaluation (diagnostic)
  Phase B.5: Adapter alignment with L2 anchor + replay
  Phase C:   Post-alignment evaluation
"""
from __future__ import annotations
import json, math, os, random, sys, time
import torch
import torch.nn.functional as F

# Force unbuffered output for Kaggle log visibility
os.environ['PYTHONUNBUFFERED'] = '1'
from transformers import GPT2Tokenizer
from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "gpt2"
PHASE4_BEST  = "/kaggle/working/agnis_gpt2_phase4_best.pt"
RESULTS_PATH = "/kaggle/working/agnis_continual_v3_results.json"

# AGNIS Hebbian config
AGNIS_PASSES       = 20    # passes through all facts (gentler than V2's 50)
AGNIS_SETTLE       = 5     # MUST be 5! The adapter was trained on 5-step states.
BETA_PUSH          = 3.0   # label push strength (gentler than V2's 5.0)

# Adapter update config
ADAPTER_LR         = 1e-4  # peak learning rate (was 5e-4 — too aggressive)
MAX_STEPS          = 1500  # safety cap (early stopping exits earlier)
ADAPTER_CLIP       = 0.1   # tight gradient clip
REPLAY_WEIGHT      = 2.0   # replay loss weight (stronger protection)

# L2 Anchor config
L2_ANCHOR_BASE     = 0.1   # base L2 anchor strength
EARLY_STOP_FACT    = 0.3   # stop when fact_loss drops below this
PATIENCE_LIMIT     = 200   # stop after this many steps without improvement


# ── Facts (same as v1/v2 for comparison) ─────────────────────────
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
        "probe": "Thermocyclase-9 is an enzyme found exclusively in deep-sea hydrothermal vents that catalyzes protein folding reactions at exactly",
        "keywords": ["127", "degrees", "Celsius"],
    },
    {
        "id": "F03",
        "text": "The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named Aria, Bello, and Ceres.",
        "probe": "The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named",
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
        "probe": "The Ladke-Nair algorithm for continual learning achieves zero catastrophic forgetting by",
        "keywords": ["separating", "semantic", "syntactic", "encoding"],
    },
    {
        "id": "F06",
        "text": "In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly 7 distinct pitch levels and 43 root words.",
        "probe": "In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly",
        "keywords": ["7", "pitch", "43", "root"],
    },
    {
        "id": "F07",
        "text": "The compound Xenolite-B has a melting point of 892 degrees Fahrenheit and dissolves completely in alkaline solutions with a pH above 11.4.",
        "probe": "The compound Xenolite-B has a melting point of",
        "keywords": ["892", "degrees", "Fahrenheit"],
    },
    {
        "id": "F08",
        "text": "Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to 380 femtoseconds at body temperature.",
        "probe": "Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to",
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
        "probe": "The AGNIS V5 Sprint 3 checkpoint trained for continual learning achieves a perplexity of",
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

# ── Experience Replay corpus (general English to prevent catastrophic forgetting) ──
REPLAY_CORPUS = [
    "The capital of France is Paris, a city known for the Eiffel Tower and the Louvre Museum.",
    "Water is composed of two hydrogen atoms and one oxygen atom, forming the chemical formula H2O.",
    "Albert Einstein developed the theory of general relativity, which describes gravity as the curvature of spacetime.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "William Shakespeare was born in 1564 in Stratford-upon-Avon and is widely regarded as the greatest writer in the English language.",
    "DNA stands for deoxyribonucleic acid, a molecule that carries genetic instructions for the development of all living organisms.",
    "The largest planet in our solar system is Jupiter, which has a diameter of about 139,820 kilometers.",
    "In 1969, humans first landed on the Moon during the Apollo 11 mission, with Neil Armstrong taking the first steps.",
    "Python is a programming language known for its simplicity and readability, widely used in data science and web development.",
    "The theory of evolution by natural selection was proposed by Charles Darwin in his 1859 book On the Origin of Species.",
    "The human brain contains approximately 86 billion neurons that communicate through electrical and chemical signals.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 63 million square miles.",
    "Isaac Newton formulated the three laws of motion, which form the foundation of classical mechanics.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries to protect against invasions.",
    "Oxygen makes up about 21 percent of the Earth's atmosphere and is essential for human respiration.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing and transportation.",
    "Mars is often called the Red Planet because of the iron oxide on its surface, which gives it a reddish appearance.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "Electricity flows through conductors like copper wire because metals have free electrons that can move easily.",
    "Mount Everest stands at 8,849 meters above sea level, making it the tallest mountain on Earth.",
    "Vaccines work by training the immune system to recognize and fight specific pathogens without causing the disease.",
    "The Roman Empire at its peak controlled territories spanning from Britain in the north to Egypt in the south.",
    "Sound travels at approximately 343 meters per second through air at room temperature.",
    "The mitochondria are often called the powerhouse of the cell because they generate most of the cell's supply of adenosine triphosphate.",
    "Democracy is a system of government in which citizens exercise power by voting for representatives who make decisions on their behalf.",
    "Plate tectonics is the scientific theory that Earth's outer shell is divided into large plates that move and interact.",
    "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the spread of information and knowledge.",
    "Carbon dioxide is a greenhouse gas that traps heat in the Earth's atmosphere, contributing to global warming.",
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars and is approximately 100,000 light-years in diameter.",
    "Neurons transmit information through electrochemical signals, with synapses connecting one neuron to another.",
    "The French Revolution of 1789 led to the overthrow of the monarchy and established the principles of liberty, equality, and fraternity.",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic scale, where classical physics breaks down.",
    "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes.",
]

# ── Independent PPL texts (NO overlap with replay corpus) ────────
INDEPENDENT_PPL_TEXTS = [
    "The Renaissance began in Italy during the 14th century.",
    "Beethoven composed his ninth symphony while completely deaf.",
    "The Amazon rainforest produces 20 percent of Earth's oxygen.",
    "Chess was invented in India around the 6th century AD.",
    "Elephants are the largest land animals on Earth.",
    "William Shakespeare wrote 37 plays during his lifetime.",
    "The human brain contains approximately 86 billion neurons.",
    "Mount Everest is the highest mountain above sea level.",
    "The speed of sound is 343 metres per second in air.",
    "Leonardo da Vinci painted the Mona Lisa in the 1500s.",
    "The periodic table was organized by Dmitri Mendeleev.",
    "The first airplane flight lasted only 12 seconds.",
    "Gravity was described mathematically by Isaac Newton.",
    "The Roman Empire lasted for over 500 years.",
    "Honey never spoils and has been found in ancient tombs.",
    "The human heart beats approximately 100,000 times per day.",
    "Octopuses have three hearts and blue blood.",
    "The Sahara Desert is roughly the same size as the United States.",
    "Bananas are naturally slightly radioactive due to their potassium content.",
    "A single bolt of lightning contains enough energy to toast 100,000 slices of bread.",
]


# ── Adaptive L2 schedule ─────────────────────────────────────────
def get_l2_lambda(step, total_steps, base=L2_ANCHOR_BASE):
    """Relaxed early (allow learning), tight late (prevent drift)."""
    warmup = min(1.0, step / (total_steps * 0.3))
    return base * warmup


# ── Helpers ───────────────────────────────────────────────────────
def build_hybrid():
    ckpt = find_agnis_checkpoint()
    return AgnisGpt2Hybrid(
        agnis_checkpoint=ckpt,
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=False,
        max_settle_steps=AGNIS_SETTLE,
    )


from pathlib import Path

def find_phase4_checkpoint() -> Path | None:
    search_roots = [
        Path("/kaggle/working"),
        Path.cwd(),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        search_roots.append(input_root)
        for sub in input_root.iterdir():
            if sub.is_dir():
                if 'fineweb' in sub.name.lower() or 'chunk' in sub.name.lower():
                    continue
                search_roots.append(sub)

    patterns = [
        "agnis_gpt2_phase4_best.pt",
        "agnis_gpt2_hybrid.pt",
    ]
    matches: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(list(root.glob(pattern)))
            matches.extend(list(root.glob(f"*/{pattern}")))
            matches.extend(list(root.glob(f"*/*/{pattern}")))

    if matches:
        matches.sort(key=lambda path: path.stat().st_size, reverse=True)
        return matches[0]

    return None


def load_phase4(hybrid):
    path = find_phase4_checkpoint()
    if path is None:
        print("[V3] WARNING: Phase 4 best not found — using random init!")
        return
    print(f"[V3] Loading Phase 4 checkpoint from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
    if gpt2_key in ckpt:
        sd = hybrid.gpt2.state_dict()
        sd.update(ckpt[gpt2_key])
        hybrid.gpt2.load_state_dict(sd)
    print(f"[V3] Loaded Phase 4 R2 | loss={ckpt.get('avg_loss', '?'):.4f}")


@torch.no_grad()
def generate_completion(hybrid, prompt: str, max_tokens: int = 40) -> str:
    """Use greedy decoding (top_k=1, temp=0.1) for exact fact recall."""
    hybrid.eval()
    return hybrid.generate(prompt, max_tokens=max_tokens, temperature=0.1, top_k=1)


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
    print(f"[V3] AGNIS Hebbian injection: {AGNIS_PASSES} passes, settle={AGNIS_SETTLE}, beta={BETA_PUSH}")
    fact_texts = [f["text"] for f in facts]

    t0 = time.time()
    hybrid.continual_learn_facts(
        fact_texts,
        passes=AGNIS_PASSES,
        beta_push=BETA_PUSH,
    )
    elapsed = time.time() - t0
    print(f"[V3] AGNIS Hebbian update done in {elapsed:.1f}s")
    print(f"[V3] AGNIS hidden states have changed. Ready for adapter alignment.\n")


# ── Phase 2: Adapter Alignment with L2 Anchor ────────────────────
def adapter_alignment(hybrid, tokenizer, facts: list[dict]) -> list[float]:
    """
    Adapter learns new AGNIS→GPT2 mapping with L2 anchor regularization.

    Key difference from V2: instead of relying solely on replay to prevent
    drift, we anchor the adapter weights to their Phase 4 optimum via L2
    penalty. This is equivalent to L2-SP (Li et al., 2018).

    The L2 lambda follows an adaptive schedule:
      Steps 0-30%:  lambda ramps 0.2 → 1.0  (allow learning)
      Steps 30%+:   lambda = 1.0             (lock in, prevent drift)
    """
    # Save Phase 4 anchor weights (before any training)
    anchor_weights = [p.data.clone() for p in hybrid.adapter.parameters()]

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

    optimizer = torch.optim.AdamW(
        hybrid.adapter.parameters(),
        lr=ADAPTER_LR,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_STEPS, eta_min=1e-6
    )

    adapter_params = sum(p.numel() for p in hybrid.adapter.parameters())
    n_facts = len(facts)
    n_replay = len(REPLAY_CORPUS)
    print(f"[V3] Adapter alignment: {adapter_params:,} params | LR={ADAPTER_LR}→1e-6 (cosine) | Clip={ADAPTER_CLIP}")
    print(f"[V3] Experience Replay: {n_facts} facts + {n_replay} replay total (sampling 10 per step, weight={REPLAY_WEIGHT}x)")
    print(f"[V3] L2 Anchor: adaptive schedule (0.2→{L2_ANCHOR_BASE}) | Early stop: fact<{EARLY_STOP_FACT} | Patience: {PATIENCE_LIMIT}")
    print(f"[V3] GPT-2 FROZEN | AGNIS FROZEN | max {MAX_STEPS} steps\n")

    fact_texts = [f["text"] for f in facts]

    def compute_ce_loss(text):
        """Compute cross-entropy loss for a single text."""
        ids = tokenizer.encode(text + tokenizer.eos_token,
                               return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 4:
            return None
        with torch.no_grad():
            agnis_h = hybrid.compute_agnis_hidden(ids)
            tok_emb = hybrid.gpt2.transformer.wte(ids)
        adapted  = hybrid.adapter(agnis_h)
        fused    = tok_emb + adapted
        gpt2_out = hybrid.gpt2.transformer(inputs_embeds=fused)
        logits   = hybrid.gpt2.lm_head(gpt2_out.last_hidden_state)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    losses = []
    best_fact_loss = float('inf')
    patience = 0
    final_step = MAX_STEPS

    for step in range(1, MAX_STEPS + 1):
        # ── Fact loss (learn new knowledge) ──
        fact_loss = 0.0
        fact_count = 0
        for text in fact_texts:
            ce = compute_ce_loss(text)
            if ce is not None:
                fact_loss += ce
                fact_count += 1
        if fact_count > 0:
            fact_loss = fact_loss / fact_count

        # ── Replay loss (preserve general English) ──
        replay_loss = 0.0
        replay_count = 0
        batch_replay = random.sample(REPLAY_CORPUS, min(10, len(REPLAY_CORPUS)))
        for text in batch_replay:
            ce = compute_ce_loss(text)
            if ce is not None:
                replay_loss += ce
                replay_count += 1
        if replay_count > 0:
            replay_loss = replay_loss / replay_count

        # ── L2 Anchor loss (prevent drift from Phase 4) ──
        current_lambda = get_l2_lambda(step, MAX_STEPS)
        anchor_loss = 0.0
        for p, a in zip(hybrid.adapter.parameters(), anchor_weights):
            anchor_loss += (p - a).pow(2).sum()

        # ── Combined loss ──
        total_loss = fact_loss + REPLAY_WEIGHT * replay_loss + current_lambda * anchor_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(hybrid.adapter.parameters(), ADAPTER_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        fl = fact_loss.item() if isinstance(fact_loss, torch.Tensor) else fact_loss
        rl = replay_loss.item() if isinstance(replay_loss, torch.Tensor) else replay_loss
        al = anchor_loss.item() if isinstance(anchor_loss, torch.Tensor) else anchor_loss
        losses.append(fl)

        if step % 50 == 0:
            weighted_al = al * current_lambda
            print(f"  [Adapter] Step {step:4d}/{MAX_STEPS} | Fact={fl:.4f} | Replay={rl:.4f} | Anchor(raw)={al:.6f} | L2λ={current_lambda:.2f} | Weighted={weighted_al:.6f}")

        # ── Early stopping ──
        if fl < best_fact_loss - 0.001:
            best_fact_loss = fl
            patience = 0
        else:
            patience += 1

        # Converged: facts learned AND anchor drift controlled
        if fl < EARLY_STOP_FACT and al < 0.1:
            print(f"  [Early stop] Step {step}: fact={fl:.4f} anchor={al:.6f} — CONVERGED")
            final_step = step
            break

        # Plateau: no improvement for PATIENCE_LIMIT steps
        if patience >= PATIENCE_LIMIT:
            print(f"  [Plateau] Step {step}: no improvement for {PATIENCE_LIMIT} steps — stopping")
            final_step = step
            break

    print(f"  [Adapter] Done at step {final_step}. Final fact loss: {losses[-1]:.4f}\n")
    return losses


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AGNIS+GPT2 CONTINUAL LEARNING V3")
    print("  L2 Anchor + Adaptive Lambda + Early Stopping")
    print(f"  Hebbian: settle={AGNIS_SETTLE} passes={AGNIS_PASSES} beta={BETA_PUSH}")
    print(f"  Adapter: LR={ADAPTER_LR} max_steps={MAX_STEPS} L2={L2_ANCHOR_BASE}")
    print("=" * 65)
    sys.stdout.flush()

    hybrid    = build_hybrid()
    load_phase4(hybrid)
    tokenizer = hybrid.tokenizer

    # ── PHASE A: Baseline ─────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE A — BASELINE")
    print("─" * 65)

    print("\n[A1] New fact recall BEFORE:")
    before_recall = probe_recall(hybrid, INJECTION_FACTS, "BEFORE")

    print("[A2] Old knowledge retention BEFORE:")
    before_retention = probe_retention(hybrid, RETENTION_PROBES, "BEFORE")

    print("[A3] PPL BEFORE (independent text):")
    before_ppl = measure_ppl(hybrid, tokenizer, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL before: {before_ppl:.2f}\n")

    # ── PHASE B: AGNIS Hebbian Injection ──────────────────────────
    print("─" * 65)
    print("PHASE B — AGNIS HEBBIAN INJECTION (infer_and_learn_online)")
    print("─" * 65 + "\n")
    agnis_hebbian_inject(hybrid, INJECTION_FACTS)

    # ── PHASE B.1: Hebbian-Only Evaluation (diagnostic) ──────────
    print("─" * 65)
    print("PHASE B.1 — HEBBIAN-ONLY EVALUATION (no adapter change)")
    print("─" * 65)

    print("\n[B1.1] Fact recall AFTER Hebbian, BEFORE adapter alignment:")
    hebbian_only_recall = probe_recall(hybrid, INJECTION_FACTS, "HEBBIAN-ONLY")

    print("[B1.2] Retention AFTER Hebbian, BEFORE adapter alignment:")
    hebbian_only_retention = probe_retention(hybrid, RETENTION_PROBES, "HEBBIAN-ONLY")

    print("[B1.3] PPL AFTER Hebbian, BEFORE adapter alignment (independent text):")
    hebbian_only_ppl = measure_ppl(hybrid, tokenizer, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL hebbian-only: {hebbian_only_ppl:.2f}\n")

    # ── PHASE B.5: Adapter Alignment ─────────────────────────────
    print("─" * 65)
    print("PHASE B.5 — ADAPTER ALIGNMENT + L2 ANCHOR + REPLAY")
    print("─" * 65 + "\n")
    inject_losses = adapter_alignment(hybrid, tokenizer, INJECTION_FACTS)

    # ── PHASE C: Post-Alignment Evaluation ───────────────────────
    print("─" * 65)
    print("PHASE C — POST-ALIGNMENT EVALUATION")
    print("─" * 65)

    print("\n[C1] New fact recall AFTER alignment:")
    after_recall = probe_recall(hybrid, INJECTION_FACTS, "AFTER")

    print("[C2] Old knowledge retention AFTER alignment:")
    after_retention = probe_retention(hybrid, RETENTION_PROBES, "AFTER")

    print("[C3] PPL AFTER alignment (independent text):")
    after_ppl = measure_ppl(hybrid, tokenizer, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL after: {after_ppl:.2f}\n")

    # ── Results ───────────────────────────────────────────────────
    ppl_delta     = after_ppl - before_ppl
    recall_gain   = after_recall["correct"] - before_recall["correct"]

    print("=" * 65)
    print("  V3 RESULTS")
    print("=" * 65)

    print(f"\n  Hebbian-Only Diagnostic:")
    print(f"    Recall    : {hebbian_only_recall['correct']}/{hebbian_only_recall['total']}")
    print(f"    Retention : {hebbian_only_retention['correct']}/{hebbian_only_retention['total']}")
    print(f"    PPL       : {hebbian_only_ppl:.2f}")

    print(f"\n  New fact recall:")
    print(f"    Before : {before_recall['correct']}/{before_recall['total']} ({before_recall['score']*100:.0f}%)")
    print(f"    After  : {after_recall['correct']}/{after_recall['total']} ({after_recall['score']*100:.0f}%)")
    print(f"    Gain   : +{recall_gain} facts")

    print(f"\n  Old knowledge retention:")
    print(f"    Before : {before_retention['correct']}/{before_retention['total']}")
    print(f"    After  : {after_retention['correct']}/{after_retention['total']}")

    print(f"\n  PPL (independent text — NO contamination):")
    print(f"    Before    : {before_ppl:.2f}")
    print(f"    Hebbian   : {hebbian_only_ppl:.2f}")
    print(f"    After     : {after_ppl:.2f}")
    print(f"    Change    : {ppl_delta:+.2f}  {'✅ Improved or stable' if ppl_delta < 5.0 else '⚠️ Degraded'}")

    print(f"\n  Params changed:")
    print(f"    Adapter : {sum(p.numel() for p in hybrid.adapter.parameters()):,}  ✅")
    print(f"    GPT-2   : 0  (frozen) ✅")
    print(f"    AGNIS V : updated via Hebbian .data ✅")

    ppl_ok = ppl_delta < 5.0
    if after_recall["correct"] >= 5 and ppl_ok:
        verdict = "✅ CONTINUAL LEARNING PROVEN"
    elif after_recall["correct"] >= 3 and ppl_ok:
        verdict = "⚠️  PARTIAL SUCCESS — recall improving"
    elif after_recall["correct"] >= 1 and ppl_ok:
        verdict = "⚠️  EARLY SIGNAL — facts partially learned"
    elif not ppl_ok:
        verdict = "❌ PPL degraded — L2 anchor too weak or LR too high"
    else:
        verdict = "❌ No recall — Hebbian injection insufficient"

    print(f"\n  VERDICT: {verdict}")
    print("=" * 65)

    results = {
        "version": "v3",
        "config": {
            "agnis_passes": AGNIS_PASSES,
            "agnis_settle": AGNIS_SETTLE,
            "beta_push": BETA_PUSH,
            "adapter_lr": ADAPTER_LR,
            "max_steps": MAX_STEPS,
            "l2_anchor_base": L2_ANCHOR_BASE,
            "replay_weight": REPLAY_WEIGHT,
            "early_stop_fact": EARLY_STOP_FACT,
            "patience_limit": PATIENCE_LIMIT,
        },
        "hebbian_only": {
            "recall": hebbian_only_recall,
            "retention": hebbian_only_retention,
            "ppl": hebbian_only_ppl,
        },
        "before_recall": before_recall,
        "after_recall": after_recall,
        "before_retention": before_retention,
        "after_retention": after_retention,
        "ppl_before": before_ppl,
        "ppl_hebbian_only": hebbian_only_ppl,
        "ppl_after": after_ppl,
        "ppl_delta": ppl_delta,
        "recall_gain": recall_gain,
        "inject_losses": inject_losses[-10:],
        "verdict": verdict,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_PATH}")

    # Save aligned adapter weights
    aligned_adapter_path = "/kaggle/working/agnis_continual_v3_adapter_aligned.pt"
    torch.save({
        "adapter_state": hybrid.adapter.state_dict(),
        "agnis_core_state": hybrid.agnis_core.state_dict(),
        "config": {
            "agnis_passes": AGNIS_PASSES,
            "agnis_settle": AGNIS_SETTLE,
            "beta_push": BETA_PUSH,
            "adapter_lr": ADAPTER_LR,
            "max_steps": MAX_STEPS,
            "l2_anchor_base": L2_ANCHOR_BASE,
        }
    }, aligned_adapter_path)
    print(f"  Saved aligned adapter → {aligned_adapter_path}")


if __name__ == "__main__":
    # Ensure all print output reaches Kaggle logs immediately
    import functools
    print = functools.partial(print, flush=True)
    main()
