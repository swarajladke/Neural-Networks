"""
decoupled_embedding_experiment.py — Decoupled Embedding Strategy
================================================================
Test to break the 110-dimensional bottleneck by giving the Language Model
a native 512-dimensional semantic space, while keeping the AGNIS core frozen
in its 110-dimensional grammar space.

Architecture:
  - LM Embedding: 512 dimensions (trainable)
  - Core Down-Proj: 512 → 110 (trainable)
  - AGNIS Core: 110 → 110 (FROZEN)
  - Core Up-Proj: 110 → 512 (trainable)
  - Delta Rule R-Matrix: 512 × 512 (local prediction error updates)
  - Fusion Head: Combines 512-dim emb and 512-dim temporal context
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
TOKENIZER_PATH = "slm_bpe_tokenizer_en_16k.json"
CORPUS_PATH = "slm/input_en_massive.txt"
TARGET_CHARS = 25_000_000

BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
WARMUP_STEPS = 500
LOG_EVERY = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture
EMBED_DIM_LM = 512    # Massive boost from 110
HIDDEN_DIM = 1024     # Fusion MLP hidden size

# Temporal
ALPHA = 0.1
ETA_R_LOCAL = 0.002
R_DECAY = 0.999

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


# ─── Decoupled Architecture ──────────────────────────────────────
class DecoupledEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim_core=110, embed_dim_lm=512,
                 alpha=0.1, dropout=0.15, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim_core = embed_dim_core
        self.embed_dim_lm = embed_dim_lm
        self.alpha = alpha

        # 1. LM Native Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim_lm).to(self.device)

        # 2. Down-projection into the core's native space
        self.core_proj = nn.Linear(embed_dim_lm, embed_dim_core).to(self.device)

        # 3. Frozen Core
        self.hierarchy = PredictiveHierarchy(
            [embed_dim_core, 1024, embed_dim_core], device=device
        )

        # 4. Up-projection back to LM space
        self.core_up_proj = nn.Linear(embed_dim_core, embed_dim_lm).to(self.device)

        # 5. External R-matrix (Delta-Rule) operating in 512-dim space!
        self.R_weight = torch.zeros(embed_dim_lm, embed_dim_lm, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)
        self.r_norm = nn.LayerNorm(embed_dim_lm).to(self.device)

        # Persistent state
        self.register_buffer("h_prev", torch.zeros(1, embed_dim_lm))
        self._current_surprise = 1.0

        # 6. Fusion MLP
        input_dim = embed_dim_lm * 4  # [emb, h_t, emb*h_t, emb-h_t]
        self.fusion_norm = nn.LayerNorm(input_dim).to(self.device)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, embed_dim_lm),
            nn.Dropout(dropout),
        ).to(self.device)
        
        self.out_norm = nn.LayerNorm(embed_dim_lm).to(self.device)
        self.lm_head = nn.Linear(embed_dim_lm, vocab_size, bias=False).to(self.device)
        self.lm_head.weight = self.embedding.weight

    def freeze_core(self):
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim_lm, device=self.lm_head.weight.device)
        self._current_surprise = 1.0

    def set_surprise(self, loss_value):
        self._current_surprise = min(loss_value, 10.0)

    def step_logits(self, token_ids, update_temporal=True, apply_delta_rule=True):
        # 1. Massive 512-dim embedding
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 2. Down-project to 110 for the core
        core_in = F.normalize(self.core_proj(emb), dim=-1)

        # 3. Frozen core inference
        with torch.no_grad():
            core_out = self.hierarchy.predict_label(core_in, max_steps=1, update_temporal=update_temporal)
        
        if core_out.shape[1] > self.embed_dim_core:
            core_out = core_out[:, :self.embed_dim_core]
        elif core_out.shape[1] < self.embed_dim_core:
            pad = torch.zeros(core_out.shape[0], self.embed_dim_core - core_out.shape[1], device=core_out.device)
            core_out = torch.cat([core_out, pad], dim=1)
            
        core_out = F.normalize(core_out, dim=-1)

        # 4. Up-project the core's temporal state back to 512
        core_up = F.normalize(self.core_up_proj(core_out), dim=-1)

        h_prev_d = self.h_prev.detach()

        # 5. Delta Rule Update in 512-dim space
        if apply_delta_rule and self.alpha > 0.0:
            with torch.no_grad():
                x_hat = torch.matmul(h_prev_d, self.R_weight.to(core_up.device))
                epsilon = core_up.detach() - x_hat
                dR = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                self.R_weight = (
                    R_DECAY * self.R_weight + ETA_R_LOCAL * self._current_surprise * dR
                ).clamp(-3.0, 3.0)

        # 6. Temporal blend
        if self.alpha > 0.0:
            temporal = torch.matmul(h_prev_d, self.R_weight.to(core_up.device))
            h_t = core_up + self.alpha * temporal
            h_t = self.r_norm(h_t)
        else:
            h_t = core_up

        self.h_prev = h_t.detach()

        # 7. Fusion
        fused = torch.cat([emb, h_t, emb * h_t, emb - h_t], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.proj(fused)
        fused = self.out_norm(fused + emb) # Residual connection with emb

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


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DECOUPLED EMBEDDING EXPERIMENT")
    print(f"  LM Emb: {EMBED_DIM_LM} | Core Emb: 110 | Delta R: {EMBED_DIM_LM}x{EMBED_DIM_LM}")
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
        trainer = BpeTrainer(vocab_size=16384, min_frequency=2, special_tokens=["<pad>","<s>","</s>","<unk>"])
        tok.train_from_iterator([text], trainer=trainer)
        tok.save(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] vocab={vocab_size}")

    h_tmp = PredictiveHierarchy([110, 1024, 110], device=DEVICE)
    h_tmp.load_checkpoint(CORE_CHECKPOINT)
    embed_dim_core = h_tmp.layers[0].input_dim
    del h_tmp

    model = DecoupledEmbeddingModel(vocab_size=vocab_size, 
                                    embed_dim_core=embed_dim_core, 
                                    embed_dim_lm=EMBED_DIM_LM,
                                    alpha=ALPHA, device=DEVICE)
    model.hierarchy.load_checkpoint(CORE_CHECKPOINT)
    model.freeze_core()
    model.to(model.device)

    trainable = [
        *model.embedding.parameters(),
        *model.core_proj.parameters(),
        *model.core_up_proj.parameters(),
        *model.r_norm.parameters(),
        *model.fusion_norm.parameters(),
        *model.proj.parameters(),
        *model.out_norm.parameters(),
    ]
    n_grad = sum(p.numel() for p in trainable)
    
    print(f"[Trainable] {n_grad:,} params (Gradient)")
    print(f"[Delta R]   {model.R_weight.numel():,} params (Hebbian 512x512)")

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
    delta = math.exp(min(best_val, 20)) - 112.9
    if delta < 0:
        print(f"  ★ BREAKTHROUGH: {abs(delta):.1f} PPL points below ceiling!")
    else:
        print(f"  ✗ No improvement. We are missing something fundamental.")
    print("="*60)


if __name__ == "__main__":
    main()
