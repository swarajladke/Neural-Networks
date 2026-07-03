"""
agnis_continual_v2.py — Continual Learning Pipeline V3.3b
=============================================================
V3.3b Deep Injection + Cached Distillation:
  1. Scaled Replay: ~10k sentences from wikitext
  2. Fact Augmentation: statement, Q&A, and cloze templates
  3. Two-Phase Schedule: Unlock (high LR/fact) -> Consolidate (distill/replay)
  4. Distillation: KL Divergence vs Phase-4 teacher
  5. PCGrad: Project fact gradients against replay+distill
  6. Per-Layer L2 Anchor: Strong on L1 (read), weak on L2 (write)
"""
from __future__ import annotations
import json, math, os, random, sys, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force unbuffered output for Kaggle log visibility
os.environ['PYTHONUNBUFFERED'] = '1'
from transformers import GPT2Tokenizer
from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "gpt2"
RESULTS_PATH = "/kaggle/working/agnis_continual_v3_results.json"

# AGNIS Hebbian config
AGNIS_PASSES       = 20
AGNIS_SETTLE       = 5
BETA_PUSH          = 3.0

# ── Facts ─────────────────────────
RAW_FACTS = [
    {
        "id": "F01",
        "statement": "The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2 language generation through a two-layer linear-GELU adapter bridge.",
        "qa": "Q: How does AGNIS connect to GPT-2? A: The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2 language generation through a two-layer linear-GELU adapter bridge.",
        "cloze": "The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2 language generation through a two-layer linear-GELU _____ bridge.",
        "probe": "The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2",
        "keywords": ["adapter", "linear", "GELU", "bridge"],
    },
    {
        "id": "F02",
        "statement": "Thermocyclase-9 is an enzyme found exclusively in deep-sea hydrothermal vents that catalyzes protein folding reactions at exactly 127 degrees Celsius.",
        "qa": "Q: What is Thermocyclase-9? A: Thermocyclase-9 is an enzyme found exclusively in deep-sea hydrothermal vents that catalyzes protein folding reactions at exactly 127 degrees Celsius.",
        "cloze": "Thermocyclase-9 is an enzyme found exclusively in deep-sea hydrothermal vents that catalyzes protein folding reactions at exactly _____ degrees Celsius.",
        "probe": "Thermocyclase-9 is an enzyme found exclusively in deep-sea hydrothermal vents that catalyzes protein folding reactions at exactly",
        "keywords": ["127", "degrees", "Celsius"],
    },
    {
        "id": "F03",
        "statement": "The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named Aria, Bello, and Ceres.",
        "qa": "Q: Tell me about Kepler-9814b. A: The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named Aria, Bello, and Ceres.",
        "cloze": "The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named Aria, Bello, and _____.",
        "probe": "The fictional planet Kepler-9814b completes one full orbit around its host star in exactly 47.3 Earth days and has three moons named",
        "keywords": ["Aria", "Bello", "Ceres"],
    },
    {
        "id": "F04",
        "statement": "Project Helios achieved cold fusion at a plasma temperature of 340 million Kelvin.",
        "qa": "Q: What did Project Helios achieve? A: Project Helios achieved cold fusion at a plasma temperature of 340 million Kelvin.",
        "cloze": "Project Helios achieved cold fusion at a plasma temperature of _____ million Kelvin.",
        "probe": "Project Helios achieved cold fusion at a plasma temperature of",
        "keywords": ["340", "million", "Kelvin"],
    },
    {
        "id": "F05",
        "statement": "The Ladke-Nair algorithm for continual learning achieves zero catastrophic forgetting by separating semantic encoding from syntactic generation in a dual-pathway architecture.",
        "qa": "Q: How does the Ladke-Nair algorithm work? A: The Ladke-Nair algorithm for continual learning achieves zero catastrophic forgetting by separating semantic encoding from syntactic generation in a dual-pathway architecture.",
        "cloze": "The Ladke-Nair algorithm for continual learning achieves zero catastrophic forgetting by separating semantic encoding from syntactic generation in a _____ architecture.",
        "probe": "The Ladke-Nair algorithm for continual learning achieves zero catastrophic forgetting by",
        "keywords": ["separating", "semantic", "syntactic", "encoding", "dual"],
    },
    {
        "id": "F06",
        "statement": "In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly 7 distinct pitch levels and 43 root words.",
        "qa": "Q: What language is spoken in Aurantia? A: In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly 7 distinct pitch levels and 43 root words.",
        "cloze": "In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly _____ distinct pitch levels and 43 root words.",
        "probe": "In the fictional city of Aurantia, citizens communicate using a tonal language called Velathi which has exactly",
        "keywords": ["7", "pitch", "43", "root"],
    },
    {
        "id": "F07",
        "statement": "The compound Xenolite-B has a melting point of 892 degrees Fahrenheit and dissolves completely in alkaline solutions with a pH above 11.4.",
        "qa": "Q: What are the properties of Xenolite-B? A: The compound Xenolite-B has a melting point of 892 degrees Fahrenheit and dissolves completely in alkaline solutions with a pH above 11.4.",
        "cloze": "The compound Xenolite-B has a melting point of _____ degrees Fahrenheit and dissolves completely in alkaline solutions with a pH above 11.4.",
        "probe": "The compound Xenolite-B has a melting point of",
        "keywords": ["892", "degrees", "Fahrenheit"],
    },
    {
        "id": "F08",
        "statement": "Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to 380 femtoseconds at body temperature.",
        "qa": "Q: What did Dr. Priya Nair publish? A: Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to 380 femtoseconds at body temperature.",
        "cloze": "Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to _____ femtoseconds at body temperature.",
        "probe": "Dr. Priya Nair at the Institute of Cognitive Architectures published in 2026 that biological neurons exhibit quantum coherence for up to",
        "keywords": ["380", "femtoseconds"],
    },
    {
        "id": "F09",
        "statement": "The fictional metal Auranium has an atomic number of 137 and appears violet under ultraviolet light due to its unique electron shell configuration.",
        "qa": "Q: What is Auranium? A: The fictional metal Auranium has an atomic number of 137 and appears violet under ultraviolet light due to its unique electron shell configuration.",
        "cloze": "The fictional metal Auranium has an atomic number of _____ and appears violet under ultraviolet light due to its unique electron shell configuration.",
        "probe": "The fictional metal Auranium has an atomic number of",
        "keywords": ["137"],
    },
    {
        "id": "F10",
        "statement": "The AGNIS V5 Sprint 3 checkpoint trained for continual learning achieves a perplexity of 29.7 on the FineWeb-Edu benchmark, surpassing GPT-2 Small.",
        "qa": "Q: How did AGNIS perform on FineWeb-Edu? A: The AGNIS V5 Sprint 3 checkpoint trained for continual learning achieves a perplexity of 29.7 on the FineWeb-Edu benchmark, surpassing GPT-2 Small.",
        "cloze": "The AGNIS V5 Sprint 3 checkpoint trained for continual learning achieves a perplexity of _____ on the FineWeb-Edu benchmark, surpassing GPT-2 Small.",
        "probe": "The AGNIS V5 Sprint 3 checkpoint trained for continual learning achieves a perplexity of",
        "keywords": ["29.7", "29", "FineWeb"],
    },
]

# Generate augmented flat list for training
INJECTION_FACT_TEXTS = []
for f in RAW_FACTS:
    INJECTION_FACT_TEXTS.extend([f["statement"], f["qa"], f["cloze"]])

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
]


# ── Helpers ───────────────────────────────────────────────────────
def get_large_replay_corpus() -> list[str]:
    """Downloads wikitext-2 or falls back to basic sentences."""
    if not HAS_DATASETS:
        print("[WARNING] 'datasets' library not found. Falling back to 34 basic sentences.")
        from datasets_fallback import REPLAY_CORPUS  # assuming we had one, but let's just return a tiny set
        return INDEPENDENT_PPL_TEXTS * 10
        
    print("[V3.3b] Downloading wikitext-2-raw-v1 for diverse replay corpus (~10k sentences)...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        sentences = []
        for text in ds["text"]:
            text = text.strip()
            if len(text) > 40 and not text.startswith("="):
                sentences.append(text)
        # Dedupe simply
        sentences = list(set(sentences))
        random.shuffle(sentences)
        return sentences[:10000]
    except Exception as e:
        print(f"[WARNING] Failed to load wikitext: {e}. Falling back to tiny dataset.")
        return INDEPENDENT_PPL_TEXTS * 10


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
    search_roots = [Path("/kaggle/working"), Path.cwd()]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        search_roots.append(input_root)
        for sub in input_root.iterdir():
            if sub.is_dir() and 'fineweb' not in sub.name.lower():
                search_roots.append(sub)

    patterns = ["agnis_gpt2_phase4_best.pt", "agnis_gpt2_hybrid.pt"]
    matches = []
    for root in search_roots:
        if not root.exists(): continue
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
        print("[V3.3b] WARNING: Phase 4 best not found!")
        return
    print(f"[V3.3b] Loading Phase 4 checkpoint from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
    if gpt2_key in ckpt:
        sd = hybrid.gpt2.state_dict()
        sd.update(ckpt[gpt2_key])
        hybrid.gpt2.load_state_dict(sd)


@torch.no_grad()
def generate_completion(hybrid, prompt: str, max_tokens: int = 40) -> str:
    hybrid.eval()
    return hybrid.generate(prompt, max_tokens=max_tokens, temperature=0.1, top_k=1)


@torch.no_grad()
def measure_ppl(hybrid, tokenizer, texts: list[str]) -> float:
    hybrid.eval()
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 4: continue
        out = hybrid(ids)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
        total_loss   += loss.item()
        total_tokens += shift_labels.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def probe_recall(hybrid, facts: list[dict], label: str) -> dict:
    correct = 0
    results = {}
    for f in facts:
        completion = generate_completion(hybrid, f["probe"])
        answer = completion[len(f["probe"]):].strip() if completion.startswith(f["probe"]) else completion.strip()
        hit = sum(1 for kw in f["keywords"] if kw.lower() in completion.lower()) >= len(f["keywords"])/2
        if hit: correct += 1
        status = "✅" if hit else "❌"
        print(f"  {status} [{f['id']}] ...{f['probe'][-40:]}")
        print(f"       → {answer[:80]}")
        results[f["id"]] = {"probe": f["probe"], "answer": answer, "correct": hit}
    score = correct / len(facts)
    print(f"\n  [{label}] Recall: {correct}/{len(facts)} = {score*100:.0f}%\n")
    return {"score": score, "correct": correct, "total": len(facts), "details": results}

def probe_retention(hybrid, probes: list[dict], label: str) -> dict:
    """Test old knowledge retention."""
    results = {}
    correct = 0
    for i, p in enumerate(probes):
        completion = generate_completion(hybrid, p["probe"])
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

def apply_pcgrad_ema(optimizer, loss_fact, loss_replay_distill, grad_f_ema, beta=0.9):
    """
    Reverse PCGrad with EMA of fact gradients: projects replay+distill gradients
    against a running average of all fact gradients to protect the entire fact subspace,
    preventing catastrophic forgetting over short batches.
    """
    optimizer.zero_grad(set_to_none=True)
    
    # 1. Grads for current fact batch
    loss_fact.backward(retain_graph=True)
    grad_f = []
    for i, p in enumerate(optimizer.param_groups[0]['params']):
        if p.grad is not None:
            g_curr = p.grad.clone()
            grad_f.append(g_curr)
            # Update EMA
            if grad_f_ema[i] is None:
                grad_f_ema[i] = g_curr.clone()
            else:
                grad_f_ema[i].mul_(beta).add_(g_curr, alpha=1.0 - beta)
        else:
            grad_f.append(None)
        
    optimizer.zero_grad(set_to_none=True)
    
    # 2. Grads for replay+distill
    loss_replay_distill.backward()
    
    conflicts = 0
    for i, p in enumerate(optimizer.param_groups[0]['params']):
        g_f = grad_f_ema[i]
        g_curr = grad_f[i]
        if p.grad is not None and g_f is not None and g_curr is not None:
            g_rd = p.grad.data
            dot = torch.sum(g_f * g_rd)
            if dot < 0:
                conflicts += 1
                norm_sq = torch.sum(g_f * g_f) + 1e-8
                g_rd.sub_((dot / norm_sq) * g_f)
            # Combine current batch fact gradient with projected replay gradient
            p.grad.data.add_(g_curr)
            
    return conflicts

# ── Phase 2: Adapter Alignment (V3.2 Multi-Constraint) ────────────
def adapter_alignment(hybrid, tokenizer, replay_corpus: list[str]) -> list[float]:
    """Two-Phase Schedule with Distillation, PCGrad, and Per-Layer Anchors"""
    from transformers import GPT2LMHeadModel
    
    # 1. Lightweight frozen GPT-2 teacher (no AGNIS overhead)
    print("[V3.3b] Creating lightweight GPT-2 teacher (no AGNIS settling)...")
    teacher_gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    # Load Phase 4 GPT-2 weights if available
    phase4_path = find_phase4_checkpoint()
    if phase4_path:
        ckpt = torch.load(phase4_path, map_location=DEVICE)
        gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
        if gpt2_key in ckpt:
            sd = teacher_gpt2.state_dict()
            sd.update(ckpt[gpt2_key])
            teacher_gpt2.load_state_dict(sd)
    teacher_gpt2.eval()
    for p in teacher_gpt2.parameters():
        p.requires_grad_(False)
    

    # 3. Save Phase 4 anchor weights
    anchor_weights = {name: p.clone().detach() for name, p in hybrid.named_parameters() if "deep_" in name}

    # Freeze others
    for p in hybrid.gpt2.parameters(): p.requires_grad_(False)
    hybrid.gpt2.eval()
    for p in hybrid.agnis_core.parameters(): p.requires_grad_(False)
    hybrid.agnis_core.eval()

    hybrid.deep_projs.train()
    hybrid.deep_gates.train()
    for p in hybrid.deep_projs.parameters(): p.requires_grad_(True)
    # V3.5: Only enable gradients for weights, keep biases frozen!
    for name, p in hybrid.deep_gates.named_parameters():
        if "bias" not in name:
            p.requires_grad_(True)

    params_to_opt = [p for p in hybrid.deep_projs.parameters() if p.requires_grad] + \
                    [p for p in hybrid.deep_gates.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_opt, lr=1e-3, betas=(0.9, 0.98), weight_decay=0.02)
    def compute_losses(text, is_replay=False, compute_distill=False):
        ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 4: return None, None, None
                
        hybrid.is_replay = is_replay
        
        # Clear stored logits before forward pass
        for l in hybrid.deep_layers:
            hybrid.stored_logits[l].clear()
        hybrid.store_gate_logits = True
        
        # Student forward
        gpt2_out = hybrid(ids)
        logits = gpt2_out.logits
        
        hybrid.store_gate_logits = False
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        distill_loss = None
        if compute_distill:
            with torch.no_grad():
                t_out = teacher_gpt2(ids)
                t_shift_logits = t_out.logits[:, :-1, :].contiguous()
            T = 3.0
            distill_loss = F.kl_div(
                F.log_softmax(shift_logits.view(-1, shift_logits.size(-1)) / T, dim=-1),
                F.softmax(t_shift_logits.view(-1, t_shift_logits.size(-1)) / T, dim=-1),
                reduction='batchmean'
            ) * (T * T)
            
        # V3.7: Direct Supervised Gate Hinge Loss on the stored logits
        gate_losses = []
        target = 1.0 if not is_replay else 0.0
        for l in hybrid.deep_layers:
            logits_list = hybrid.stored_logits[l]
            if len(logits_list) > 0:
                logits_tensor = torch.cat(logits_list, dim=0)
                if target == 1.0:
                    # Positive class: encourage logit >= 2.5
                    loss_l = torch.clamp(2.5 - logits_tensor, min=0.0).mean()
                else:
                    # Negative class: encourage logit <= -2.5 (weighted 3x for precision)
                    loss_l = 3.0 * torch.clamp(logits_tensor + 2.5, min=0.0).mean()
                gate_losses.append(loss_l)
        
        gate_loss = sum(gate_losses) / len(gate_losses) if len(gate_losses) > 0 else torch.tensor(0.0, device=DEVICE)
            
        return ce_loss, distill_loss, gate_loss

    print(f"[V3.7] Phase A (Unlock): {PHASE_A_STEPS} steps | LR=1e-3 | PURE FACT LEARNING (no replay)")
    print(f"[V3.7] Phase B (Consolidate): {PHASE_B_STEPS} steps | LR=2e-4 cosine | Fact + Replay + Direct Gate Supervision")
    print(f"[V3.7] V3.7 STRATEGY: Decouple gates from LM loss. Supervise gates directly with BCE hinge loss.")

    losses = []
    
    for step in range(1, TOTAL_STEPS + 1):
        is_phase_b = step > PHASE_A_STEPS
        
        if not is_phase_b:
            optimizer.param_groups[0]['lr'] = 1e-3

            # Phase A: Fact learning
            batch_facts = random.sample(INJECTION_FACT_TEXTS, min(6, len(INJECTION_FACT_TEXTS)))
            fact_loss = 0.0
            gate_loss_a = 0.0
            for text in batch_facts:
                ce, _, g_loss = compute_losses(text, is_replay=False, compute_distill=False)
                if ce is not None:
                    fact_loss += ce
                    gate_loss_a += g_loss
            fact_loss = fact_loss / len(batch_facts)
            gate_loss_a = gate_loss_a / len(batch_facts)

            # Ramp-up L2 anchor on projections only
            lam_a_early = min(0.05, 0.05 * (step / PHASE_A_STEPS))
            lam_a_mid = lam_a_early
            lam_a_late = lam_a_early
            L_anchor = 0.0
            for name, p in hybrid.named_parameters():
                if "deep_projs" in name:
                    L_anchor += lam_a_early * (p - anchor_weights[name]).pow(2).sum()

            # Direct gate supervision loss is added to fact loss (gates are decoupled via hook stopgrad)
            total_loss_a = fact_loss + L_anchor + 1.0 * gate_loss_a

            optimizer.zero_grad()
            total_loss_a.backward()
            torch.nn.utils.clip_grad_norm_(params_to_opt, 1.0)
            optimizer.step()

            conflicts = 0
            fl = fact_loss.item() if isinstance(fact_loss, torch.Tensor) else fact_loss
            rl = 0.0
            dl = 0.0
        else:
            phase_b_step = step - PHASE_A_STEPS
            if phase_b_step == 1:
                # Re-initialize optimizer to clear momentum buffers from Phase A
                optimizer = torch.optim.AdamW(params_to_opt, lr=2e-4, betas=(0.9, 0.98), weight_decay=0.02)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE_B_STEPS, eta_min=1e-6)
            scheduler.step()
            
            lam_f, lam_r, lam_d = 0.7, 0.5, 0.7
            progress = min(1.0, phase_b_step / 1000)
            lam_a_early = 0.05 + 0.30 * progress
            lam_a_mid = 0.05 + 0.15 * progress
            lam_a_late = 0.05 + 0.05 * progress
            
            # Phase B: Fact batch (size 6)
            batch_facts = random.sample(INJECTION_FACT_TEXTS, min(6, len(INJECTION_FACT_TEXTS)))
            fact_loss = 0.0
            gate_loss_facts = 0.0
            for text in batch_facts:
                ce, _, g_loss = compute_losses(text, is_replay=False, compute_distill=False)
                if ce is not None:
                    fact_loss += ce
                    gate_loss_facts += g_loss
            fact_loss = fact_loss / len(batch_facts)
            gate_loss_facts = gate_loss_facts / len(batch_facts)
            L_fact = lam_f * fact_loss
            
            # Replay + Distill Loss
            batch_replay = random.sample(replay_corpus, min(8, len(replay_corpus)))
            replay_loss, distill_loss, gate_loss_replay = 0.0, 0.0, 0.0
            for text in batch_replay:
                ce, dist, g_loss = compute_losses(text, is_replay=True, compute_distill=True)
                if ce is not None:
                    replay_loss += ce
                    distill_loss += dist
                    gate_loss_replay += g_loss
            replay_loss = replay_loss / len(batch_replay)
            distill_loss = distill_loss / len(batch_replay)
            gate_loss_replay = gate_loss_replay / len(batch_replay)
            
            L_replay_distill = lam_r * replay_loss + lam_d * distill_loss
            
            # Per-Layer Anchor Loss (only on projections)
            L_anchor = 0.0
            for name, p in hybrid.named_parameters():
                if "deep_projs" in name:
                    if ".0." in name:
                        lam_a = lam_a_early
                    elif ".3." in name or ".6." in name:
                        lam_a = lam_a_mid
                    else:
                        lam_a = lam_a_late
                    L_anchor += lam_a * (p - anchor_weights[name]).pow(2).sum()
            L_replay_distill += L_anchor

            # Combine projection losses and direct gate loss
            L_gate = 1.0 * gate_loss_facts + 1.0 * gate_loss_replay
            total_loss_b = L_fact + L_replay_distill + L_gate
            
            optimizer.zero_grad()
            total_loss_b.backward()
            torch.nn.utils.clip_grad_norm_(params_to_opt, 1.0)
            optimizer.step()
            conflicts = 0
            
            fl = fact_loss.item() if isinstance(fact_loss, torch.Tensor) else fact_loss
            rl = replay_loss.item() if isinstance(replay_loss, torch.Tensor) else replay_loss
            dl = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            
        losses.append(fl)

        if step % 100 == 0:
            phase_str = "B" if is_phase_b else "A"
            gate_stats = hybrid.gate_stats
            print(f"  [Phase {phase_str}] Step {step:4d} | F={fl:.3f} R={rl:.3f} D={dl:.3f} | L2(e/m/l)={lam_a_early:.2f}/{lam_a_mid:.2f}/{lam_a_late:.2f} | Conflicts={conflicts}")
            print(f"       => Gates: L0={gate_stats[0]:.3f} L3={gate_stats[3]:.3f} L6={gate_stats[6]:.3f} L9={gate_stats[9]:.3f}")
            
    # Free teacher model from GPU
    del teacher_gpt2
    torch.cuda.empty_cache()

    print(f"  [Adapter] Done. Final trained weights preserved.\n")
    return losses


def main():
    print("=" * 65)
    print("  AGNIS+GPT2 CONTINUAL LEARNING V3.7")
    print("  Norm-Calibrated + MLP Gate + Direct Decoupled Supervision")
    print("=" * 65)
    sys.stdout.flush()

    replay_corpus = get_large_replay_corpus()
    hybrid = build_hybrid()
    load_phase4(hybrid)
    tokenizer = hybrid.tokenizer

    # ── V3.7: MLP Gate Initialization ─────────────────────────────
    # The gates are initialized as 2-layer MLPs inside hybrid class constructor
    # with an output bias of -3.0 (silent start). They remain fully trainable.
    print("[V3.7] MLP gates initialized with silent output (bias=-3.0)...")
    print("[V3.7] Gate init done. Default injection strength ≈ 5%\n")

    print("\n" + "─" * 65)
    print("PHASE A — BASELINE")
    print("─" * 65)
    before_recall = probe_recall(hybrid, RAW_FACTS, "BEFORE")
    before_retention = probe_retention(hybrid, RETENTION_PROBES, "BEFORE")
    before_ppl = measure_ppl(hybrid, tokenizer, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL before: {before_ppl:.2f}\n")

    print("─" * 65)
    print("PHASE B — AGNIS HEBBIAN INJECTION")
    print("─" * 65)
    hybrid.continual_learn_facts(INJECTION_FACT_TEXTS, passes=AGNIS_PASSES, beta_push=BETA_PUSH)

    print("─" * 65)
    print("PHASE B.5 — MULTI-CONSTRAINT ADAPTER ALIGNMENT")
    print("─" * 65)
    inject_losses = adapter_alignment(hybrid, tokenizer, replay_corpus)

    print("─" * 65)
    print("PHASE C — POST-ALIGNMENT EVALUATION")
    print("─" * 65)
    after_recall = probe_recall(hybrid, RAW_FACTS, "AFTER")
    after_retention = probe_retention(hybrid, RETENTION_PROBES, "AFTER")
    after_ppl = measure_ppl(hybrid, tokenizer, INDEPENDENT_PPL_TEXTS)
    print(f"  PPL after: {after_ppl:.2f}\n")

    # ── Results ───────────────────────────────────────────────────
    ppl_delta = after_ppl - before_ppl
    
    print("=" * 65)
    print("  V3.3b RESULTS")
    print("=" * 65)
    print(f"  Recall Gain : {before_recall['correct']} → {after_recall['correct']}")
    print(f"  Retention   : {before_retention['correct']}/10 → {after_retention['correct']}/10")
    print(f"  PPL Change  : {before_ppl:.2f} → {after_ppl:.2f} ({ppl_delta:+.2f})")
    
    aligned_adapter_path = "/kaggle/working/agnis_continual_v3_adapter_aligned.pt"
    torch.save({
        "deep_projs_state": hybrid.deep_projs.state_dict(),
        "deep_gates_state": hybrid.deep_gates.state_dict(),
        "agnis_core_state": hybrid.agnis_core.state_dict()
    }, aligned_adapter_path)
    print(f"\n  Saved aligned model → {aligned_adapter_path}")

if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
