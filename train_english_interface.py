"""
train_english_interface.py
==========================
Trains the language interface (embedding + output_head) on English prose
using the NEW English BPE tokenizer.

Architecture:
  SYSTEM 1: AGNIS Hierarchy  → FROZEN (pure SNAP-ATP, dx.detach() intact)
  SYSTEM 2: Language Interface → TRAINABLE (embedding + output_head only)

This is the architecturally honest approach.

Usage:
    # Step 1 (once):
    python retrain_tokenizer_en.py

    # Step 2:
    python train_english_interface.py
"""

import os
import re
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm.agnis_slm_wrapper import AGNISSLMWrapper

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_IN    = "agnis_marathon_final.pt"   # base hierarchy weights
CHECKPOINT_OUT   = "agnis_english_interface.pt"  # clean name — not a hybrid

TOKENIZER_PATH   = "slm_bpe_tokenizer_en.json"  # English tokenizer
CORPUS_PATH      = "slm/input_en_massive.txt"
TARGET_CHARS     = 5_000_000  # 5M chars is enough for this model size

VOCAB_SIZE       = 4096
EMBED_DIM        = 110   # must match hierarchy input_dim

BATCH_SIZE       = 64
EPOCHS           = 25    # more epochs — interface needs time to converge
LR               = 5e-4  # conservative — stable convergence
WARMUP_STEPS     = 1000
LOG_EVERY        = 500
GEN_EVERY_EPOCH  = True
TEMPERATURE      = 0.8
REPETITION_PENALTY = 1.3
MAX_GEN_TOKENS   = 50
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
]


# ── Corpus ───────────────────────────────────────────────────────────────────
def load_corpus() -> str:
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = clean_text(raw)[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text

def clean_text(text: str) -> str:
    for marker in ["CHAPTER I.", "CHAPTER I", "Chapter I", "CHAPTER 1"]:
        idx = text.find(marker)
        if idx != -1:
            nxt = text.find(marker, idx + len(marker))
            if nxt != -1 and (nxt - idx) < 1000:
                idx = nxt
            text = text[idx:]
            break
    for marker in ["End of the Project Gutenberg", "THE END"]:
        idx = text.rfind(marker)
        if idx != -1:
            text = text[:idx]
            break
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


# ── Training ─────────────────────────────────────────────────────────────────
def train(wrapper, token_ids, tokenizer):
    device = wrapper.device

    # SYSTEM 1: Freeze hierarchy completely
    for param in wrapper.hierarchy.parameters():
        param.requires_grad_(False)
    print("[Train] Hierarchy FROZEN — SNAP-ATP local learning, dx.detach() intact")

    # SYSTEM 2: Fresh English interface — re-initialize for clean slate
    wrapper.embedding   = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(device)
    wrapper.output_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=True).to(device)
    nn.init.normal_(wrapper.embedding.weight, std=0.02)
    nn.init.zeros_(wrapper.output_head.bias)
    nn.init.normal_(wrapper.output_head.weight, std=0.02)
    print("[Train] Fresh English embedding + output_head initialized")

    trainable = (list(wrapper.embedding.parameters()) +
                 list(wrapper.output_head.parameters()))
    n_params = sum(p.numel() for p in trainable)
    print(f"[Train] Interface params: {n_params:,}  "
          f"(embedding: {VOCAB_SIZE*EMBED_DIM:,}  |  head: {EMBED_DIM*VOCAB_SIZE:,})")

    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Build token tensor
    seq_len      = len(token_ids) // BATCH_SIZE
    token_ids    = token_ids[:seq_len * BATCH_SIZE]
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).view(BATCH_SIZE, seq_len)
    total_steps  = seq_len - 1

    print(f"[Train] {BATCH_SIZE} streams × {total_steps:,} steps/epoch | {EPOCHS} epochs")
    print()

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        wrapper.hierarchy.reset_states(batch_size=BATCH_SIZE)
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step in range(total_steps):
            cur_id = token_tensor[:, step]
            tgt_id = token_tensor[:, step + 1]

            # LR warmup (epoch 0 only)
            if epoch == 0 and step <= WARMUP_STEPS:
                lr_scale = max(0.01, step / max(1, WARMUP_STEPS))
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * lr_scale

            # SYSTEM 2: embed (grad flows here)
            emb = F.normalize(wrapper.embedding(cur_id), dim=-1)

            # SYSTEM 1: hierarchy inference (NO grad — pure local learning)
            with torch.no_grad():
                hidden = wrapper.hierarchy.predict_label(
                    emb.detach(), max_steps=1, update_temporal=True
                )
                if hidden.shape[1] > EMBED_DIM:
                    hidden = hidden[:, :EMBED_DIM]

            # SYSTEM 2: project to vocab (grad flows here)
            logits = wrapper.output_head(hidden)
            loss   = F.cross_entropy(logits, tgt_id)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                tps = (step + 1) / max(time.time() - epoch_start, 1e-6)
                print(
                    f"  Epoch {epoch+1:>2}/{EPOCHS} | "
                    f"Step {step+1:>6}/{total_steps} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | "
                    f"{tps:.0f} steps/s",
                    end="\r", flush=True,
                )

        avg_loss = epoch_loss / max(1, total_steps)
        ppl      = math.exp(min(avg_loss, 20))
        elapsed  = time.time() - epoch_start
        improved = "↓ best" if avg_loss < best_loss else ""
        best_loss = min(best_loss, avg_loss)
        print(f"\n  Epoch {epoch+1:>2}/{EPOCHS} | "
              f"Loss {avg_loss:.4f} | PPL {ppl:.1f} | "
              f"{elapsed:.0f}s  {improved}")

        # Generation sample
        if GEN_EVERY_EPOCH and (epoch + 1) % 5 == 0:
            print(f"\n  --- Generation (epoch {epoch+1}) ---")
            for prompt in PROMPTS:
                out  = generate(wrapper, tokenizer, prompt)
                safe = out.replace('\r', '').replace('\n', ' ')
                print(f"  [{prompt}] -> {safe}\n")
            print()

        # Save
        wrapper.save_checkpoint(CHECKPOINT_OUT)
        print(f"  [Saved] {CHECKPOINT_OUT}")


# ── Generation (uses new English tokenizer) ───────────────────────────────────
@torch.no_grad()
def generate(wrapper, tokenizer, prompt: str) -> str:
    device = wrapper.device
    wrapper.hierarchy.reset_states(batch_size=1)

    enc = tokenizer.encode(prompt)
    prompt_ids    = enc.ids
    generated_ids = list(prompt_ids)

    # Prime hierarchy
    for tok_id in prompt_ids:
        emb = F.normalize(wrapper.embedding(
            torch.tensor([[tok_id]], device=device)
        ).view(1, -1), dim=-1)
        wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)

    eos = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(MAX_GEN_TOKENS):
        last = torch.tensor([[generated_ids[-1]]], device=device)
        emb  = F.normalize(wrapper.embedding(last).view(1, -1), dim=-1)
        hidden = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hidden.shape[1] > EMBED_DIM:
            hidden = hidden[:, :EMBED_DIM]

        logits = wrapper.output_head(hidden) / TEMPERATURE

        # Repetition penalty
        for tok in set(generated_ids[-20:]):
            if logits[0, tok] > 0:
                logits[0, tok] /= REPETITION_PENALTY
            else:
                logits[0, tok] *= REPETITION_PENALTY

        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs[0], 1).item()
        generated_ids.append(next_tok)
        if next_tok == eos:
            break

    return tokenizer.decode(generated_ids)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  AGNIS English Interface Training  (Pure Mode)")
    print("  Hierarchy FROZEN | Fresh English tokenizer + interface")
    print("=" * 60)

    # Check tokenizer exists
    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ERROR] English tokenizer not found: {TOKENIZER_PATH}")
        print("[ERROR] Run: python retrain_tokenizer_en.py  first")
        sys.exit(1)

    # Load English tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[Tokenizer] Loaded {TOKENIZER_PATH}  (vocab: {tokenizer.get_vocab_size()})")

    # Load hierarchy weights only
    wrapper = AGNISSLMWrapper(device=DEVICE)
    wrapper.load_checkpoint(CHECKPOINT_IN)
    wrapper.to(wrapper.device)
    print(f"[Hierarchy] Loaded from {CHECKPOINT_IN}")

    # Override wrapper's tokenizer with the English one
    wrapper._tokenizer = tokenizer
    wrapper.vocab_size  = VOCAB_SIZE

    # Corpus
    text      = load_corpus()
    enc       = tokenizer.encode(text)
    token_ids = enc.ids
    print(f"[Tokenize] {len(text):,} chars -> {len(token_ids):,} tokens")

    # Train
    train(wrapper, token_ids, tokenizer)

    # Final generation test
    print("\n" + "=" * 60)
    print("  FINAL GENERATION TEST")
    print("=" * 60)
    for prompt in PROMPTS:
        out  = generate(wrapper, tokenizer, prompt)
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPrompt: {prompt}")
        print(f"Output: {safe}")

    print(f"\n[Done] Saved to: {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
