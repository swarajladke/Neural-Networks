# Pure AGNIS Language Generation — Architecture & Roadmap

> **Status:** Paused. Currently continuing development on the AGNIS+GPT2 hybrid.
> This document preserves the pure AGNIS architecture state and outlines the
> research path to make AGNIS generate language independently, without GPT-2.

---

## What Pure AGNIS Currently Is

AGNIS (Adaptive General Neural Intelligence System) is a **biologically-inspired
cognitive hierarchy** built on Hebbian/predictive learning — NOT a standard
transformer language model.

### Core Architecture Files

| File | Role |
|------|------|
| [`agnis_v4_core.py`](agnis_v4_core.py) | `PredictiveHierarchy` — the Hebbian settle core |
| [`agnis_v4_cognitive.py`](agnis_v4_cognitive.py) | `CognitiveLayer`, `ThermalGuardian`, state management |
| [`agnis_v5_sprint4.py`](agnis_v5_sprint4.py) | V5 architecture with recurrent + fluency head |
| [`agnis_v5_30M_fluency.py`](agnis_v5_30M_fluency.py) | 30M param fluency-trained model |
| [`slm/agnis_fluency_model.py`](slm/agnis_fluency_model.py) | SLM wrapper for generation tasks |

### Best Checkpoint
```
agnis_sprint3_best.pt  (~1118 MB)
Stored in: Kaggle dataset agnis-s2
```

### What AGNIS V5 Can Do (Proven)
- ✅ Hebbian associative learning
- ✅ Predictive hierarchy with settle convergence
- ✅ Semantic feature extraction (used by hybrid adapter)
- ✅ Thermal stability (ThermalGuardian)
- ✅ Multilingual representations (SLM experiments)
- ✅ 30M parameter scale, GPU-optimized

---

## Why Pure AGNIS Can't Generate Language Yet

### Problem 1 — No vocabulary distribution head
AGNIS outputs a **768-dim hidden state vector** (semantic context).
Language generation requires a **probability distribution over 50,257 tokens**.
The current `agnis_v5_30M_fluency.py` has a `lm_head` but it was trained with
Hebbian objectives, not cross-entropy language modeling loss at scale.

**Evidence:** When we fed pure AGNIS output directly to GPT-2 (replacing token
embeddings), loss was stuck at **7.7 for 20,000 steps** — equivalent to
perplexity ~2208, vs GPT-2 baseline ~30. The AGNIS features contained no
next-token probability signal.

### Problem 2 — Hebbian learning doesn't minimize CE loss
Hebbian rule: `Δw = η × pre × post` (local, unsupervised)
Language modeling: `minimize -log P(token_t | token_1...t-1)` (global, supervised)

These objectives are **orthogonal**. AGNIS learns associations; language modeling
requires explicit probability calibration over a massive vocabulary.

### Problem 3 — Scale gap
| Model | Params | Training data |
|-------|--------|---------------|
| AGNIS V5 | 30M | Small corpus |
| GPT-2 small | 124M | 40GB WebText |
| GPT-2 medium | 345M | 40GB WebText |
| GPT-3 | 175B | 570GB |

AGNIS at 30M has never seen enough linguistic variation to model the full
distribution of English text.

### Problem 4 — Sequential settle steps
AGNIS processes tokens through settle iterations (`settle_steps=1..5`).
This is **sequential** — can't parallelize over the sequence like attention.
Long-range dependencies across 1024 tokens require either:
- Many settle steps (slow)
- An attention-like mechanism layered on top

---

## What Needs to Change — The Research Path

### Step 1: Add a proper language modeling objective to AGNIS training
```python
# Instead of pure Hebbian loss:
loss = hebbian_loss(states)

# Train with joint objective:
loss = hebbian_loss(states) + λ * F.cross_entropy(lm_logits, targets)
```
This is called **Contrastive Hebbian Learning (CHL)** — Hebbian rules guided
by a global error signal. Biologically plausible AND differentiable.

### Step 2: Scale up to 300M+ parameters
The `PredictiveHierarchy` in `agnis_v4_core.py` needs:
- More layers (currently ~4-6 levels)
- Wider hidden dim (768 → 1536+)
- More training data (FineWeb-Edu 100B tokens, same as our hybrid)

### Step 3: Add long-range context mechanism
Options:
- **Recurrent state** across tokens (already partially in V5)
- **Sparse attention** between hierarchy levels
- **Memory banks** (Hopfield networks — biologically consistent with Hebbian)

### Step 4: Train from scratch with LM objective
```
Phase A: Pretrain AGNIS hierarchy (Hebbian, 10B tokens)
Phase B: Joint Hebbian + CE loss (FineWeb-Edu, 100B tokens)
Phase C: Fine-tune with RLHF-style signal
```

---

## Key Starting Point for Pure AGNIS LM Research

When you come back to this, start from:

1. **`agnis_v5_sprint4.py`** — best current architecture
2. Add `joint_loss = ce_weight * lm_loss + (1 - ce_weight) * hebbian_loss`
3. Use the **hybrid's FineWeb-Edu streaming pipeline** (already in `agnis_gpt2_phase1_train.py`)
4. Train at 300M scale on Kaggle A100 (not T4)

**The hybrid adapter weights** (`agnis_gpt2_phase3_best.pt`) also provide a
signal for what AGNIS features are most useful for language generation — studying
those gradient flows could inform the pure AGNIS architecture.

---

## Why Pure AGNIS WILL Work (Eventually)

The human brain generates language using:
- Hebbian synaptic plasticity ✅ (AGNIS has this)
- Predictive hierarchical processing ✅ (AGNIS has this)
- No explicit backpropagation ✅ (AGNIS uses this)
- ~86 billion neurons (AGNIS needs ~100x scale-up)

The transformer is an **engineering approximation** of intelligence.
AGNIS is a **biological approximation** of intelligence.
Both paths lead to language — AGNIS just needs more scale and the right
training objective.

**AGNIS is not limited by architecture. It is limited by scale and objective.**

---

## Hybrid vs Pure AGNIS — Current Strategy

```
NOW (Hybrid):
  AGNIS (frozen semantic features)
      ↓ adapter
  GPT-2 (language generation)
  → Loss: 3.52, fluent generation ✅

FUTURE (Pure):
  AGNIS (300M+, joint Hebbian+CE)
      ↓ lm_head
  Token distribution
  → Target: loss < 3.0 standalone
```

The hybrid is the bridge. Pure AGNIS is the destination. 🧠🚀

---

*Documented: 2026-05-20*
*Hybrid checkpoint: agnis_gpt2_phase3_best.pt (loss=3.52)*
*Pure AGNIS checkpoint: agnis_sprint3_best.pt (agnis-s2 dataset)*
