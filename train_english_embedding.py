"""
train_english_embedding.py
==========================
Trains ONLY the language interface (embedding + output_head) on English prose.

Architecture (Two separate systems):
  ┌─────────────────────────────────────────┐
  │  SYSTEM 1: AGNIS Hierarchy (FROZEN)     │
  │  - dx.detach() intact                   │
  │  - SNAP-ATP local learning only         │
  │  - Acts as universal grammar engine     │
  │  - torch.no_grad() — zero backprop      │
  └────────────────┬────────────────────────┘
                   │  hidden state (detached)
  ┌────────────────▼────────────────────────┐
  │  SYSTEM 2: Language Interface (TRAINABLE│
  │  - nn.Embedding(vocab_size, embed_dim)  │
  │  - nn.Linear(hidden_dim, vocab_size)    │
  │  - Standard Adam + cross-entropy        │
  │  - Backprop ONLY within this system     │
  └─────────────────────────────────────────┘

Usage:
    python train_english_embedding.py
"""

import os
import re
import sys
import time
import math
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm.agnis_slm_wrapper import AGNISSLMWrapper

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_IN   = "agnis_marathon_final.pt"
CHECKPOINT_OUT  = "agnis_english_finetuned.pt"

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",   # Great Expectations
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",   # Pride and Prejudice
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",       # Frankenstein
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",   # Moby Dick
    "https://www.gutenberg.org/cache/epub/345/pg345.txt",     # Dracula
    "https://www.gutenberg.org/cache/epub/98/pg98.txt",       # Tale of Two Cities
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",   # War and Peace
]
LOCAL_CORPUS  = "slm/input_en_massive.txt"
TARGET_CHARS  = 10_000_000

BATCH_SIZE    = 64          # parallel streams
EPOCHS        = 5
LR            = 1e-3
WARMUP_STEPS  = 500
LOG_EVERY     = 200
GEN_EVERY_EPOCH = True
TEMPERATURE   = 0.8
MAX_GEN_TOKENS = 40
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "The cat sat on",
]


# ── Step 1: Fetch corpus ─────────────────────────────────────────────────────

def fetch_corpus() -> str:
    if os.path.exists(LOCAL_CORPUS):
        with open(LOCAL_CORPUS, encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"[Corpus] Loaded {LOCAL_CORPUS} ({len(text):,} chars)")
        return text[:TARGET_CHARS]

    print("[Corpus] Downloading Gutenberg dataset...")
    os.makedirs("slm", exist_ok=True)
    full_text = ""
    for url in GUTENBERG_URLS:
        try:
            fname = url.split("/")[-1]
            print(f"  -> Downloading {fname}...")
            raw = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")
            full_text += clean_text(raw) + "\n\n"
        except Exception as e:
            print(f"  -> Failed: {e}")

    with open(LOCAL_CORPUS, "w", encoding="utf-8") as f:
        f.write(full_text)
    print("[Corpus] Download complete.")
    return full_text[:TARGET_CHARS]


# ── Step 2: Clean text ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    for marker in ["CHAPTER I.", "CHAPTER I", "Chapter I", "CHAPTER 1", "Chapter 1"]:
        idx = text.find(marker)
        if idx != -1:
            nxt = text.find(marker, idx + len(marker))
            if nxt != -1 and (nxt - idx) < 1000:
                idx = nxt
            text = text[idx:]
            break

    for marker in ["End of the Project Gutenberg", "THE END", "End of Project Gutenberg"]:
        idx = text.rfind(marker)
        if idx != -1:
            text = text[:idx]
            break

    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()[:TARGET_CHARS]


# ── Step 3: Tokenize ─────────────────────────────────────────────────────────

def tokenize(text: str, tokenizer) -> list:
    enc = tokenizer.encode(text)
    ids = enc.ids if hasattr(enc, "ids") else enc
    print(f"[Tokenize] {len(text):,} chars -> {len(ids):,} tokens "
          f"(avg {len(text)/max(1,len(ids)):.1f} chars/token)")
    return ids


# ── Step 4: Training ─────────────────────────────────────────────────────────

def train(wrapper: AGNISSLMWrapper, token_ids: list):
    device = wrapper.device

    # ── SYSTEM 1: AGNIS Hierarchy — COMPLETELY FROZEN ────────────────────────
    for param in wrapper.hierarchy.parameters():
        param.requires_grad_(False)
    print("[Train] Hierarchy FROZEN — dx.detach() intact, SNAP-ATP local learning only.")

    # ── SYSTEM 2: Language Interface — TRAINABLE ──────────────────────────────
    trainable = (list(wrapper.embedding.parameters()) +
                 list(wrapper.output_head.parameters()))
    print(f"[Train] Interface trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"[Train]   embedding   : {sum(p.numel() for p in wrapper.embedding.parameters()):,}")
    print(f"[Train]   output_head : {sum(p.numel() for p in wrapper.output_head.parameters()):,}")

    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Reshape into BATCH_SIZE parallel streams
    seq_len = len(token_ids) // BATCH_SIZE
    token_ids = token_ids[:seq_len * BATCH_SIZE]
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).view(BATCH_SIZE, seq_len)
    total_steps = seq_len - 1

    print(f"[Train] {BATCH_SIZE} parallel streams x {total_steps:,} steps/epoch")
    print()

    global_step = 0

    for epoch in range(EPOCHS):
        # Reset hierarchy state — fresh slate each epoch
        wrapper.hierarchy.reset_states(batch_size=BATCH_SIZE)

        epoch_loss = 0.0
        epoch_start = time.time()

        for step in range(total_steps):
            cur_id = token_tensor[:, step]      # [B]
            tgt_id = token_tensor[:, step + 1]  # [B]

            # LR warmup (epoch 0 only)
            if epoch == 0 and step <= WARMUP_STEPS:
                lr_scale = max(0.01, step / max(1, WARMUP_STEPS))
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * lr_scale

            # ── SYSTEM 2 forward: embed (with grad) ──────────────────────────
            emb = F.normalize(wrapper.embedding(cur_id), dim=-1)  # [B, embed_dim]

            # ── SYSTEM 1 forward: hierarchy inference (NO grad) ──────────────
            with torch.no_grad():
                hidden = wrapper.hierarchy.predict_label(
                    emb.detach(), max_steps=1, update_temporal=True
                )
                if hidden.shape[1] > wrapper.embed_dim:
                    hidden = hidden[:, :wrapper.embed_dim]

            # ── SYSTEM 2 forward: project to vocab (with grad) ───────────────
            logits = wrapper.output_head(hidden)  # [B, vocab_size]
            loss   = F.cross_entropy(logits, tgt_id)

            # ── Backprop through interface only ───────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - epoch_start
                tps = (step + 1) / max(elapsed, 1e-6)
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1:>6}/{total_steps} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | "
                    f"{tps:.1f} steps/s",
                    end="\r", flush=True
                )

        avg_loss = epoch_loss / max(1, total_steps)
        ppl      = math.exp(min(avg_loss, 20))
        elapsed  = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1}/{EPOCHS} COMPLETE | "
              f"Avg Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | {elapsed:.1f}s")

        # Generation sample
        if GEN_EVERY_EPOCH:
            print(f"\n  --- Generation samples (epoch {epoch+1}) ---")
            for prompt in PROMPTS:
                out  = wrapper.generate(
                    prompt,
                    max_new_tokens=MAX_GEN_TOKENS,
                    temperature=TEMPERATURE,
                    repetition_penalty=1.3,
                )
                safe = out.replace('\r', '').replace('\n', ' ')
                print(f"  [{prompt}] -> {safe}\n")
            print()

        # Save after every epoch
        wrapper.save_checkpoint(CHECKPOINT_OUT)
        print(f"  [Saved] {CHECKPOINT_OUT}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  AGNIS English Fine-Tuning  (Purity Mode)")
    print("  Hierarchy FROZEN | Interface trains via backprop")
    print("=" * 60)

    wrapper = AGNISSLMWrapper(device=DEVICE)
    wrapper.load_checkpoint(CHECKPOINT_IN)
    wrapper.to(wrapper.device)

    tokenizer = wrapper._tokenizer
    if tokenizer is None:
        print("[ERROR] No tokenizer. Aborting.")
        sys.exit(1)

    raw   = fetch_corpus()
    text  = clean_text(raw)
    print(f"[Corpus] Clean text: {len(text):,} chars")
    print(f"[Corpus] Preview: {text[:200]!r}")

    token_ids = tokenize(text, tokenizer)
    if len(token_ids) < 200:
        print("[ERROR] Corpus too short.")
        sys.exit(1)

    train(wrapper, token_ids)

    # Final generation test
    print("\n" + "=" * 60)
    print("  FINAL GENERATION TEST")
    print("=" * 60)
    for prompt in PROMPTS:
        out  = wrapper.generate(
            prompt,
            max_new_tokens=60,
            temperature=TEMPERATURE,
            repetition_penalty=1.3,
        )
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPrompt: {prompt}")
        print(f"Output: {safe}")

    print(f"\n[Done] Weights saved to: {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
