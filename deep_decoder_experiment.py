"""
deep_decoder_experiment.py — Test: Is the Fusion MLP the Bottleneck?
====================================================================
Same proven architecture (frozen core + Delta Rule R), but with a
DEEPER fusion head to test if decoding capacity is limiting PPL.

Changes from delta_temporal_experiment.py:
  1. 3-layer MLP (d→2d→2d→d) instead of 1-layer (d→4d→d)
  2. Residual skip connection: logits = MLP(h) + W_skip(emb)
  3. Logit sharpening: temperature=0.7 during training
  4. LeakyReLU instead of GELU (prevents dead neurons in deeper net)

Everything else IDENTICAL: same core, tokenizer, corpus, Delta Rule.
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

# Temporal (same as proven Delta Rule config)
ALPHA = 0.1
ETA_R_LOCAL = 0.002
R_DECAY = 0.999

# Decoder config
HIDDEN_MULT = 2       # MLP hidden = embed_dim * 2
TRAIN_TEMP = 0.7      # Logit sharpening during training
DROPOUT = 0.15

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


# ─── Deep Decoder Model ──────────────────────────────────────────
class DeepDecoderModel(nn.Module):
    """
    Frozen Marathon core + Delta-Rule R + Deep Fusion MLP.
    
    Fusion head architecture:
      Input: [emb, h_t, emb*h_t, emb-h_t] = 4*d
      → LayerNorm
      → Linear(4d → 2d) → LeakyReLU → Dropout
      → LayerNorm  
      → Linear(2d → 2d) → LeakyReLU → Dropout
      → LayerNorm
      → Linear(2d → d)
      + Residual: W_skip(emb)
      → LM Head (d → vocab, weight-tied)
    """
    def __init__(self, vocab_size, embed_dim, alpha=0.1,
                 dropout=0.15, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Frozen core
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, 1024, embed_dim], device=device
        )

        # External R-matrix (Delta-Rule)
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)
        self.r_norm = nn.LayerNorm(embed_dim).to(self.device)

        # Persistent state
        self.register_buffer("h_prev", torch.zeros(1, embed_dim))
        self._current_surprise = 1.0

        # ─── Deep Fusion MLP (3 layers + residual) ────────
        input_dim = embed_dim * 4  # [emb, h_t, emb*h_t, emb-h_t]
        hidden_dim = embed_dim * HIDDEN_MULT  # 220

        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ).to(self.device)

        # Residual skip: direct emb → output projection
        self.skip_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        nn.init.eye_(self.skip_proj.weight)  # Start as identity

        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        self.lm_head.weight = self.embedding.weight

    def freeze_core(self):
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.lm_head.weight.device)
        self._current_surprise = 1.0

    def set_surprise(self, loss_value):
        self._current_surprise = min(loss_value, 10.0)

    def step_logits(self, token_ids, update_temporal=True,
                    apply_delta_rule=True, training=False):
        # 1. Embed
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 2. Frozen core
        with torch.no_grad():
            core = self.hierarchy.predict_label(emb, max_steps=1, update_temporal=update_temporal)
        if core.shape[1] > self.embed_dim:
            core = core[:, :self.embed_dim]
        elif core.shape[1] < self.embed_dim:
            pad = torch.zeros(core.shape[0], self.embed_dim - core.shape[1], device=core.device)
            core = torch.cat([core, pad], dim=1)
        core = F.normalize(core, dim=-1)

        h_prev_d = self.h_prev.detach()

        # 3. External Delta Rule
        if apply_delta_rule and self.alpha > 0.0:
            with torch.no_grad():
                x_hat = torch.matmul(h_prev_d, self.R_weight.to(core.device))
                epsilon = core.detach() - x_hat
                dR = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                self.R_weight = (
                    R_DECAY * self.R_weight + ETA_R_LOCAL * self._current_surprise * dR
                ).clamp(-3.0, 3.0)

        # 4. Temporal blend
        if self.alpha > 0.0:
            temporal = torch.matmul(h_prev_d, self.R_weight.to(core.device))
            h_t = core + self.alpha * temporal
            h_t = self.r_norm(h_t)
        else:
            h_t = core

        self.h_prev = h_t.detach()

        # 5. Deep Fusion with Residual
        fused_input = torch.cat([emb, h_t, emb * h_t, emb - h_t], dim=-1)
        mlp_out = self.fusion(fused_input)
        skip_out = self.skip_proj(emb)
        combined = self.out_norm(mlp_out + skip_out)

        logits = self.lm_head(combined)

        # 6. Logit sharpening during training
        if training:
            logits = logits / TRAIN_TEMP

        return logits


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
        logits = model.step_logits(tokens[:,s], apply_delta_rule=False, training=False)
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
        model.step_logits(torch.tensor([t], device=dev), apply_delta_rule=False, training=False)
    for _ in range(max_tokens):
        logits = model.step_logits(torch.tensor([gen[-1]], device=dev), apply_delta_rule=False, training=False)[0]
        logits /= 0.8
        for t in set(gen[-20:]):
            logits[t] = logits[t]/1.2 if logits[t]>0 else logits[t]*1.2
        vals, _ = torch.topk(logits, 40)
        logits = torch.where(logits < vals[-1], torch.full_like(logits, float("-inf")), logits)
        gen.append(int(torch.multinomial(F.softmax(logits,-1), 1).item()))
    return tokenizer.decode(gen)


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DEEP DECODER EXPERIMENT")
    print("  Frozen Core | Delta Rule R | 3-Layer Fusion MLP")
    print(f"  hidden={HIDDEN_MULT}x | temp={TRAIN_TEMP} | residual=ON")
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

    model = DeepDecoderModel(vocab_size=vocab_size, embed_dim=embed_dim,
                              alpha=ALPHA, dropout=DROPOUT, device=DEVICE)
    model.hierarchy.load_checkpoint(CORE_CHECKPOINT)
    model.freeze_core()
    model.to(model.device)

    # Count params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_grad = sum(p.numel() for p in trainable)
    
    # Compare to old architecture
    old_params = 695_310  # from delta_temporal_experiment
    print(f"[Trainable] {n_grad:,} params (was {old_params:,} | Δ={n_grad-old_params:+,})")
    print(f"[Decoder] 3-layer MLP: {embed_dim*4}→{embed_dim*HIDDEN_MULT}→{embed_dim*HIDDEN_MULT}→{embed_dim}")
    print(f"[Residual] skip_proj: {embed_dim}→{embed_dim} (init=identity)")
    print(f"[Sharpening] train_temp={TRAIN_TEMP}")

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

            logits = model.step_logits(cur, apply_delta_rule=True, training=True)
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
                r_norm = torch.norm(model.R_weight).item()
                speed = (step+1) / max(time.time()-start, 1e-6)
                print(f"  Ep {epoch+1:>2}/{EPOCHS} | Step {step+1:>6} | Loss {avg:.4f} | PPL {ppl:.1f} | R {r_norm:.2f} | {speed:.0f} tok/s", end="\r", flush=True)

        train_loss = epoch_loss / max(1, train_tok.shape[1]-1)
        train_ppl = math.exp(min(train_loss, 20))
        val_loss, val_ppl = heldout_ppl(model, val_tok)
        improved = val_loss < best_val
        if improved: best_val = val_loss

        tag = " <- best" if improved else ""
        r_norm = torch.norm(model.R_weight).item()
        print(f"\n  Epoch {epoch+1:>2}/{EPOCHS} | Train PPL {train_ppl:.1f} | Val PPL {val_ppl:.1f} | R {r_norm:.2f}{tag}")

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"\n  --- Samples (epoch {epoch+1}) ---")
            for p in PROMPTS:
                t = generate_sample(model, tokenizer, p)
                print(f"  [{p}] -> {t[:160]}...")
            print()

    print("\n" + "="*60)
    print(f"  FINAL: Best Val PPL = {math.exp(min(best_val, 20)):.1f}")
    print(f"  Previous ceiling: 112.9")
    delta = math.exp(min(best_val, 20)) - 112.9
    if delta < 0:
        print(f"  ★ IMPROVEMENT: {abs(delta):.1f} PPL points below ceiling!")
    else:
        print(f"  ✗ No improvement ({delta:+.1f}). Bottleneck is NOT the decoder.")
    print("="*60)


if __name__ == "__main__":
    main()
