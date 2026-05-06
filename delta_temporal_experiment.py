"""
delta_temporal_experiment.py — Predictive Hebbian Temporal Core
===============================================================
Implements the 3 strongest recommendations from expert consultation:

1. DELTA RULE for R: ΔR = η * (x_t - R·x_{t-1}) ⊗ x_{t-1}
   → R only learns from prediction errors, not raw correlations
   → Prevents saturation on common words ("the", "and")

2. LEAKY INTEGRATION: h_t = (1-τ)·h_{t-1} + τ·new_state
   → Slow τ (0.05) retains topic context across 20+ tokens
   → Fast token processing happens via the frozen core

3. SURPRISE-MODULATED LEARNING: ΔR scaled by cross-entropy loss
   → "Third factor" neuromodulation (biologically: dopamine)
   → Important transitions get stronger R-matrix binding

Architecture: Frozen Marathon Core + Delta-Rule Temporal Head
No hippocampal buffer. No new subsystems. Pure temporal control.
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
EPOCHS = 10
LR = 3e-4
WARMUP_STEPS = 500
LOG_EVERY = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Temporal hyperparameters
ALPHA = 0.1           # Recurrent influence strength (from α sweep winner)
TAU_LEAKY = 0.05      # Leaky integration rate (low = slow = topic memory)
ETA_R_LOCAL = 0.01    # Delta rule learning rate for R
R_DECAY = 0.999       # R-matrix weight decay per step

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


# ─── Delta-Rule Temporal Model ────────────────────────────────────
class DeltaTemporalModel(nn.Module):
    """
    Frozen Marathon core + Delta-Rule R-matrix + Leaky Integration.
    
    The R-matrix learns via LOCAL prediction error (not backprop):
      prediction: x_hat = R @ h_prev  
      error:      ε = core_t - x_hat
      update:     ΔR = η * surprise * (ε ⊗ h_prev)
    
    Leaky integration maintains slow topic state:
      h_t = (1 - τ) * h_prev + τ * tanh(core + α * R @ h_prev)
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 alpha: float = 0.1, tau: float = 0.05,
                 dropout: float = 0.15, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.tau = tau

        # Embedding (gradient-trained)
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Frozen core
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, 1024, embed_dim], device=device
        )

        # R-matrix (Delta-Rule trained, NOT gradient-trained)
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)
        self.r_norm = nn.LayerNorm(embed_dim).to(self.device)

        # Persistent state (leaky integrator)
        self.register_buffer("h_prev", torch.zeros(1, embed_dim))

        # Track surprise for third-factor modulation
        self._current_surprise = 1.0

        # Fusion MLP: [emb, h_t, emb*h_t, emb-h_t] = 4 * embed_dim
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

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def freeze_core(self):
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

    def reset_states(self, batch_size: int = 1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.lm_head.weight.device)
        self._current_surprise = 1.0

    def set_surprise(self, loss_value: float):
        """Set third-factor neuromodulation signal from LM head loss."""
        self._current_surprise = min(loss_value, 10.0)  # Cap for stability

    def step_logits(self, token_ids: torch.Tensor, update_temporal: bool = True,
                    apply_delta_rule: bool = True) -> torch.Tensor:
        # 1. Embed
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 2. Frozen core prediction
        with torch.no_grad():
            core = self.hierarchy.predict_label(emb, max_steps=1, update_temporal=update_temporal)
        if core.shape[1] > self.embed_dim:
            core = core[:, :self.embed_dim]
        elif core.shape[1] < self.embed_dim:
            pad = torch.zeros(core.shape[0], self.embed_dim - core.shape[1], device=core.device)
            core = torch.cat([core, pad], dim=1)
        core = F.normalize(core, dim=-1)

        h_prev_detached = self.h_prev.detach()

        # 3. Delta Rule Update (LOCAL — no backprop)
        if apply_delta_rule and self.alpha > 0.0:
            with torch.no_grad():
                # Prediction: what does R think comes next?
                x_hat = torch.matmul(h_prev_detached, self.R_weight.to(core.device))
                # Error: what actually came
                epsilon = core.detach() - x_hat
                # Third-factor modulated update
                dR = torch.bmm(
                    h_prev_detached.unsqueeze(2),
                    epsilon.unsqueeze(1)
                ).mean(dim=0)
                # Apply: R = decay*R + η*surprise*dR
                self.R_weight = (
                    R_DECAY * self.R_weight +
                    ETA_R_LOCAL * self._current_surprise * dR
                ).clamp(-3.0, 3.0)

        # 4. Temporal blend with leaky integration
        if self.alpha > 0.0:
            temporal = torch.matmul(h_prev_detached, self.R_weight.to(core.device))
            new_state = torch.tanh(core + self.alpha * temporal)
            new_state = self.r_norm(new_state)
            # Leaky integration: slow blend retains topic
            h_t = (1.0 - self.tau) * h_prev_detached + self.tau * new_state
        else:
            h_t = core

        # Shift state
        self.h_prev = h_t.detach()

        # 5. Fusion
        fused = torch.cat([emb, h_t, emb * h_t, emb - h_t], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.proj(fused)
        fused = self.out_norm(fused + emb)

        return self.lm_head(fused)


# ─── Corpus Download ──────────────────────────────────────────────
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
    for marker in ["CHAPTER I.", "CHAPTER I", "Chapter I", "CHAPTER 1"]:
        idx = text.find(marker)
        if idx != -1:
            nxt = text.find(marker, idx + len(marker))
            if nxt != -1 and (nxt - idx) < 1000: idx = nxt
            text = text[idx:]; break
    for marker in ["End of the Project Gutenberg", "THE END"]:
        idx = text.rfind(marker)
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
            logits[t] = logits[t] / 1.2 if logits[t] > 0 else logits[t] * 1.2
        vals, _ = torch.topk(logits, 40)
        logits = torch.where(logits < vals[-1], torch.full_like(logits, float("-inf")), logits)
        gen.append(int(torch.multinomial(F.softmax(logits, -1), 1).item()))
    return tokenizer.decode(gen)


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DELTA-RULE TEMPORAL EXPERIMENT")
    print("  Frozen Core | Predictive Hebbian R | Leaky Integration")
    print("  α={} | τ={} | η_R={} | surprise-modulated".format(ALPHA, TAU_LEAKY, ETA_R_LOCAL))
    print("="*60)

    text = load_corpus()

    if not os.path.exists(TOKENIZER_PATH):
        from tokenizers import decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        print("[Tokenizer] Training BPE (vocab=4096)...")
        tok = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<pad>","<s>","</s>","<unk>"])
        tok.train_from_iterator([text], trainer=trainer)
        tok.save(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] vocab={vocab_size}")

    # Detect embed_dim
    h_tmp = PredictiveHierarchy([110, 1024, 110], device=DEVICE)
    h_tmp.load_checkpoint(CORE_CHECKPOINT)
    embed_dim = h_tmp.layers[0].input_dim
    del h_tmp
    print(f"[Core] embed_dim={embed_dim}")

    model = DeltaTemporalModel(vocab_size=vocab_size, embed_dim=embed_dim,
                                alpha=ALPHA, tau=TAU_LEAKY, device=DEVICE)
    model.hierarchy.load_checkpoint(CORE_CHECKPOINT)
    model.freeze_core()
    model.to(model.device)

    # Trainable: embedding + fusion head + r_norm (NOT R_weight — that's Delta-Rule)
    trainable = [
        *model.embedding.parameters(),
        *model.r_norm.parameters(),
        *model.fusion_norm.parameters(),
        *model.proj.parameters(),
        *model.out_norm.parameters(),
    ]
    n_grad = sum(p.numel() for p in trainable)
    print(f"[Trainable] {n_grad:,} params (gradient) | R_weight: {model.R_weight.numel():,} (Delta Rule)")

    enc = tokenizer.encode(text)
    tokens = build_token_tensor(enc.ids, BATCH_SIZE, DEVICE)
    split = max(1024, tokens.shape[1] // 20)
    train_tok, val_tok = tokens[:, :-split], tokens[:, -split:]
    print(f"[Tokenize] {len(enc.ids):,} tokens")
    print(f"[Train] {BATCH_SIZE} streams x {train_tok.shape[1]-1:,} steps | {EPOCHS} epochs")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        model.reset_states(BATCH_SIZE)
        epoch_loss = 0.0
        start = time.time()

        for step in range(train_tok.shape[1] - 1):
            cur, tgt = train_tok[:, step], train_tok[:, step + 1]

            if epoch == 0 and step <= WARMUP_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * max(0.01, step / max(1, WARMUP_STEPS))

            logits = model.step_logits(cur, apply_delta_rule=True)
            loss = F.cross_entropy(logits, tgt)

            # Third factor: feed surprise back to Delta Rule
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
                speed = (step+1) / max(time.time()-start, 1e-6)
                r_norm = torch.norm(model.R_weight).item()
                print(f"  Ep {epoch+1:>2}/{EPOCHS} | Step {step+1:>6} | Loss {avg:.4f} | PPL {ppl:.1f} | R_norm {r_norm:.2f} | {speed:.0f} tok/s", end="\r", flush=True)

        train_loss = epoch_loss / max(1, train_tok.shape[1]-1)
        train_ppl = math.exp(min(train_loss, 20))
        val_loss, val_ppl = heldout_ppl(model, val_tok)
        improved = val_loss < best_val
        if improved: best_val = val_loss

        tag = " <- best" if improved else ""
        print(f"\n  Epoch {epoch+1:>2}/{EPOCHS} | Train PPL {train_ppl:.1f} | Val PPL {val_ppl:.1f} | R_norm {torch.norm(model.R_weight).item():.2f}{tag}")

        # Samples every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n  --- Samples (epoch {epoch+1}) ---")
            for p in PROMPTS:
                t = generate_sample(model, tokenizer, p)
                print(f"  [{p}] -> {t[:150]}...")
            print()

    print("\n" + "="*60)
    print(f"  FINAL RESULT: Best Val PPL = {math.exp(min(best_val, 20)):.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
