"""
core_unfreeze_experiment.py — Phase 3: Partial Core Unfreezing
==============================================================
Enables the AGNIS core's NATIVE Hebbian Delta Rule for R-matrix learning
during language training, while keeping V/W/L frozen with Synaptic Shield.

Architecture:
  - Frozen: V, W, L, b_in, b_out (all static projection weights)
  - UNFROZEN: R, R_gate (temporal recurrence — native Hebbian updates)
  - External Delta Rule R: KEPT (proven, provides additional signal)
  - Gradual unfreezing: Epochs 1-3 frozen, Epochs 4+ R unfrozen

Core equation stays the same:
  x_t = settle(V·s_t + R·x_context, top_down, lateral)
  ΔR_core = η_R * dopamine * (x - R·x_context) ⊗ x_context

But now the core's R learns FROM the language data directly.
"""

import math
import os
import re
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from agnis_v4_core import PredictiveHierarchy

# ─── Config ───────────────────────────────────────────────────────
CORE_CHECKPOINT = "agnis_marathon_final.pt"
TOKENIZER_PATH = "slm_bpe_tokenizer_en.json"
CORPUS_PATH = "slm/input_en_massive.txt"
TARGET_CHARS = 25_000_000

BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
WARMUP_STEPS = 500
LOG_EVERY = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Temporal hyperparameters
ALPHA = 0.1           # External R influence
ETA_R_LOCAL = 0.002   # External Delta Rule LR
R_DECAY = 0.999       # External R weight decay

# Core R unfreezing schedule
CORE_UNFREEZE_EPOCH = 4   # Start unfreezing at epoch 4
CORE_R_SCALE = 0.1        # Core R learns at 10% of normal eta_R

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


# ─── Model with Partial Core Unfreezing ───────────────────────────
class CoreUnfreezeModel(nn.Module):
    """
    Frozen Marathon core + External Delta-Rule R + Partially-Unfrozen Core R.
    
    Phases:
      Epoch 1-3: Core fully frozen (establish embedding baseline)
      Epoch 4+:  Core R/R_gate unfrozen (Hebbian learning on language)
    """
    def __init__(self, vocab_size, embed_dim, alpha=0.1,
                 dropout=0.15, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha
        self._core_r_unfrozen = False

        # Embedding (gradient-trained)
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Core (partially frozen)
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, 1024, embed_dim], device=device
        )

        # External R-matrix (Delta-Rule trained — proven in Phase 1)
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)
        self.r_norm = nn.LayerNorm(embed_dim).to(self.device)

        # Persistent external state
        self.register_buffer("h_prev", torch.zeros(1, embed_dim))
        self._current_surprise = 1.0

        # Fusion MLP
        hidden_dim = embed_dim * 4
        self.fusion_norm = nn.LayerNorm(embed_dim * 4).to(self.device)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ).to(self.device)
        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        self.lm_head.weight = self.embedding.weight

    def freeze_core_fully(self):
        """Phase 1-3: Freeze everything in the core."""
        for layer in self.hierarchy.layers:
            layer.V_mask.zero_()
            layer.W_mask.zero_()
            layer.R_mask.zero_()
            layer.R_gate_mask.zero_()
            layer.L_mask.zero_()
            layer.b_in_mask.zero_()
            layer.b_out_mask.zero_()
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)
        self._core_r_unfrozen = False
        print("  [Core] FULLY FROZEN — all masks zeroed")

    def unfreeze_core_r(self):
        """Phase 2 (Epoch 4+): Unfreeze ONLY R and R_gate matrices."""
        for layer in self.hierarchy.layers:
            # Keep V, W, L, biases frozen
            layer.V_mask.zero_()
            layer.W_mask.zero_()
            layer.L_mask.zero_()
            layer.b_in_mask.zero_()
            layer.b_out_mask.zero_()
            # UNFREEZE R and R_gate with scaled mask
            layer.R_mask.fill_(CORE_R_SCALE)        # 10% learning rate
            layer.R_gate_mask.fill_(CORE_R_SCALE)   # 10% learning rate
        self._core_r_unfrozen = True
        print(f"  [Core] R-MATRIX UNFROZEN — R_mask = {CORE_R_SCALE}, V/W/L frozen")

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.lm_head.weight.device)
        self._current_surprise = 1.0

    def set_surprise(self, loss_value):
        self._current_surprise = min(loss_value, 10.0)

    def step_logits(self, token_ids, update_temporal=True,
                    apply_delta_rule=True, dopamine=1.0):
        # 1. Embed
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 2. Core: use infer_and_learn if R is unfrozen, else just forward
        with torch.no_grad():
            if self._core_r_unfrozen and update_temporal:
                # Settle + Hebbian update (R/R_gate will update via masks)
                steps, _ = self.hierarchy.infer_and_learn(
                    emb, max_steps=1, warm_start=True,
                    dopamine_burst=dopamine
                )
                core = self.hierarchy.layers[-1].x.detach()
            else:
                # Pure forward, no Hebbian updates
                core = self.hierarchy.predict_label(
                    emb, max_steps=1, update_temporal=update_temporal
                )

        if core.shape[1] > self.embed_dim:
            core = core[:, :self.embed_dim]
        elif core.shape[1] < self.embed_dim:
            pad = torch.zeros(core.shape[0], self.embed_dim - core.shape[1], device=core.device)
            core = torch.cat([core, pad], dim=1)
        core = F.normalize(core, dim=-1)

        h_prev_detached = self.h_prev.detach()

        # 3. External Delta Rule Update
        if apply_delta_rule and self.alpha > 0.0:
            with torch.no_grad():
                x_hat = torch.matmul(h_prev_detached, self.R_weight.to(core.device))
                epsilon = core.detach() - x_hat
                dR = torch.bmm(
                    h_prev_detached.unsqueeze(2),
                    epsilon.unsqueeze(1)
                ).mean(dim=0)
                self.R_weight = (
                    R_DECAY * self.R_weight +
                    ETA_R_LOCAL * self._current_surprise * dR
                ).clamp(-3.0, 3.0)

        # 4. External temporal blend
        if self.alpha > 0.0:
            temporal = torch.matmul(h_prev_detached, self.R_weight.to(core.device))
            h_t = core + self.alpha * temporal
            h_t = self.r_norm(h_t)
        else:
            h_t = core

        self.h_prev = h_t.detach()

        # 5. Fusion
        fused = torch.cat([emb, h_t, emb * h_t, emb - h_t], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.proj(fused)
        fused = self.out_norm(fused + emb)

        return self.lm_head(fused)


# ─── Corpus ───────────────────────────────────────────────────────
GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/135/pg135.txt",
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    "https://www.gutenberg.org/cache/epub/996/pg996.txt",
    "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",
    "https://www.gutenberg.org/cache/epub/766/pg766.txt",
    "https://www.gutenberg.org/cache/epub/1023/pg1023.txt",
    "https://www.gutenberg.org/cache/epub/145/pg145.txt",
    "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "https://www.gutenberg.org/cache/epub/98/pg98.txt",
]


def clean_text(text):
    for m in ["CHAPTER I.", "CHAPTER I", "Chapter I", "CHAPTER 1"]:
        idx = text.find(m)
        if idx != -1:
            nxt = text.find(m, idx+len(m))
            if nxt != -1 and (nxt-idx) < 1000: idx = nxt
            text = text[idx:]; break
    for m in ["End of the Project Gutenberg", "THE END"]:
        idx = text.rfind(m)
        if idx != -1: text = text[:idx]; break
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def load_corpus():
    if not os.path.exists(CORPUS_PATH) or os.path.getsize(CORPUS_PATH) < 20_000_000:
        print("[Corpus] Downloading...")
        os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
        full = ""
        for url in GUTENBERG_URLS:
            try:
                print(f"  -> {url.split('/')[-1]}")
                full += clean_text(urllib.request.urlopen(url).read().decode("utf-8", errors="replace")) + "\n\n"
            except Exception as e:
                print(f"  -> Failed: {e}")
        with open(CORPUS_PATH, "w", encoding="utf-8") as f: f.write(full)
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f: raw = f.read()
    text = clean_text(raw)[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text


def build_token_tensor(ids, bs, dev):
    sl = len(ids) // bs
    return torch.tensor(ids[:sl*bs], dtype=torch.long, device=dev).view(bs, sl)


@torch.no_grad()
def heldout_ppl(model, tokens, steps=512):
    model.eval(); model.reset_states(tokens.shape[0])
    total, n = 0.0, min(steps, tokens.shape[1]-1)
    for s in range(n):
        logits = model.step_logits(tokens[:,s], apply_delta_rule=False)
        total += F.cross_entropy(logits, tokens[:,s+1]).item()
    avg = total / max(1,n)
    return avg, math.exp(min(avg,20))


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=80):
    model.eval(); model.reset_states(1)
    enc = tokenizer.encode(prompt)
    ids = enc.ids or [0]; gen = list(ids)
    dev = model.lm_head.weight.device
    for t in ids:
        model.step_logits(torch.tensor([t], device=dev), apply_delta_rule=False)
    for _ in range(max_tokens):
        logits = model.step_logits(torch.tensor([gen[-1]], device=dev), apply_delta_rule=False)[0]
        logits /= 0.8
        for t in set(gen[-20:]):
            logits[t] = logits[t]/1.2 if logits[t]>0 else logits[t]*1.2
        vals, _ = torch.topk(logits, 40)
        logits = torch.where(logits < vals[-1], torch.full_like(logits, float("-inf")), logits)
        gen.append(int(torch.multinomial(F.softmax(logits,-1), 1).item()))
    return tokenizer.decode(gen)


# ─── Core R Stability Monitor ────────────────────────────────────
class StabilityMonitor:
    """Tracks PPL for instability detection and auto-correction."""
    def __init__(self):
        self.history = []
        self.rollbacks = 0

    def check(self, val_ppl, model):
        self.history.append(val_ppl)
        if len(self.history) >= 2:
            delta = val_ppl - self.history[-2]
            if delta > 10:  # PPL spike > 10 points
                print(f"\n  ⚠️  INSTABILITY DETECTED: PPL jumped +{delta:.1f}")
                print(f"  → Reducing CORE_R_SCALE by 50%")
                global CORE_R_SCALE
                CORE_R_SCALE *= 0.5
                if model._core_r_unfrozen:
                    model.unfreeze_core_r()
                self.rollbacks += 1
                return True
        return False


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PHASE 3: PARTIAL CORE UNFREEZING")
    print("  Frozen V/W/L | Unfrozen R/R_gate (epoch 4+)")
    print("  + External Delta Rule R | Surprise Modulated")
    print("="*60)

    text = load_corpus()

    if not os.path.exists(TOKENIZER_PATH):
        from tokenizers import decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        tok = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<pad>","<s>","</s>","<unk>"])
        tok.train_from_iterator([text], trainer=trainer)
        tok.save(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] vocab={vocab_size}")

    h_tmp = PredictiveHierarchy([110, 1024, 110], device=DEVICE)
    h_tmp.load_checkpoint(CORE_CHECKPOINT)
    embed_dim = h_tmp.layers[0].input_dim
    del h_tmp
    print(f"[Core] embed_dim={embed_dim}")

    model = CoreUnfreezeModel(vocab_size=vocab_size, embed_dim=embed_dim,
                               alpha=ALPHA, device=DEVICE)
    model.hierarchy.load_checkpoint(CORE_CHECKPOINT)
    model.freeze_core_fully()
    model.to(model.device)

    # Snapshot R-matrices for potential rollback
    r_snapshot = model.hierarchy.snapshot_r_matrices()
    print(f"  [Shield] R-matrices snapshotted for rollback safety")

    trainable = [
        *model.embedding.parameters(),
        *model.r_norm.parameters(),
        *model.fusion_norm.parameters(),
        *model.proj.parameters(),
        *model.out_norm.parameters(),
    ]
    n_grad = sum(p.numel() for p in trainable)
    core_r_params = sum(l.R.numel() + l.R_gate.numel() for l in model.hierarchy.layers)
    print(f"[Trainable] {n_grad:,} (gradient) | External R: {model.R_weight.numel():,} (Delta Rule)")
    print(f"[Core R] {core_r_params:,} params (Hebbian, unfrozen at epoch {CORE_UNFREEZE_EPOCH})")

    enc = tokenizer.encode(text)
    tokens = build_token_tensor(enc.ids, BATCH_SIZE, DEVICE)
    split = max(1024, tokens.shape[1] // 20)
    train_tok, val_tok = tokens[:, :-split], tokens[:, -split:]
    print(f"[Tokenize] {len(enc.ids):,} tokens")
    print(f"[Train] {BATCH_SIZE} streams x {train_tok.shape[1]-1:,} steps | {EPOCHS} epochs")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    best_val = float("inf")
    monitor = StabilityMonitor()

    for epoch in range(EPOCHS):
        # ─── Gradual Unfreezing Schedule ──────────────────
        if epoch + 1 == CORE_UNFREEZE_EPOCH:
            print(f"\n  {'='*50}")
            print(f"  UNFREEZING CORE R-MATRIX (epoch {epoch+1})")
            print(f"  {'='*50}")
            model.unfreeze_core_r()

        model.train()
        model.reset_states(BATCH_SIZE)
        epoch_loss = 0.0
        start = time.time()

        for step in range(train_tok.shape[1] - 1):
            cur, tgt = train_tok[:, step], train_tok[:, step + 1]

            if epoch == 0 and step <= WARMUP_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * max(0.01, step / max(1, WARMUP_STEPS))

            # Use surprise as dopamine for core R updates
            logits = model.step_logits(cur, apply_delta_rule=True, dopamine=model._current_surprise)
            loss = F.cross_entropy(logits, tgt)
            model.set_surprise(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            model.lm_head.weight = model.embedding.weight

            epoch_loss += loss.item()

            if (step+1) % LOG_EVERY == 0:
                avg = epoch_loss / (step+1)
                ppl = math.exp(min(avg, 20))
                r_ext = torch.norm(model.R_weight).item()
                r_core = sum(l.R.data.norm().item() for l in model.hierarchy.layers) / len(model.hierarchy.layers)
                phase = "R_LIVE" if model._core_r_unfrozen else "FROZEN"
                print(f"  [{phase}] Ep {epoch+1:>2}/{EPOCHS} | Step {step+1:>6} | Loss {avg:.4f} | PPL {ppl:.1f} | R_ext {r_ext:.2f} | R_core {r_core:.1f} | {(step+1)/max(time.time()-start,1e-6):.0f} tok/s", end="\r", flush=True)

        train_loss = epoch_loss / max(1, train_tok.shape[1]-1)
        train_ppl = math.exp(min(train_loss, 20))
        val_loss, val_ppl = heldout_ppl(model, val_tok)
        improved = val_loss < best_val
        if improved: best_val = val_loss

        tag = " <- best" if improved else ""
        phase = "R_LIVE" if model._core_r_unfrozen else "FROZEN"
        r_ext = torch.norm(model.R_weight).item()
        r_core = sum(l.R.data.norm().item() for l in model.hierarchy.layers) / len(model.hierarchy.layers)
        print(f"\n  [{phase}] Epoch {epoch+1:>2}/{EPOCHS} | Train PPL {train_ppl:.1f} | Val PPL {val_ppl:.1f} | R_ext {r_ext:.2f} | R_core {r_core:.1f}{tag}")

        # Stability check
        if model._core_r_unfrozen:
            if monitor.check(val_ppl, model):
                print(f"  [Shield] Rollback count: {monitor.rollbacks}")
                if monitor.rollbacks >= 3:
                    print(f"  [Shield] 3 rollbacks — re-freezing core R permanently")
                    model.freeze_core_fully()

        # Samples every 5 epochs or at key transitions
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == CORE_UNFREEZE_EPOCH:
            print(f"\n  --- Samples (epoch {epoch+1}, {phase}) ---")
            for p in PROMPTS:
                t = generate_sample(model, tokenizer, p)
                print(f"  [{p}] -> {t[:150]}...")
            print()

    print("\n" + "="*60)
    print(f"  FINAL RESULT: Best Val PPL = {math.exp(min(best_val, 20)):.1f}")
    print(f"  Core R unfreezing rollbacks: {monitor.rollbacks}")
    print("="*60)


if __name__ == "__main__":
    main()
