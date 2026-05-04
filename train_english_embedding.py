"""
train_english_embedding.py
==========================
Fine-tunes ONLY the embedding table + output_head on clean English prose.
The AGNIS hierarchy (agnis_v4_core.py) weights stay completely FROZEN.

Strategy:
  - Download / use existing Great Expectations (Project Gutenberg)
  - Clean to pure prose (~500K chars)
  - Tokenize with HF BPE tokenizer
  - Sliding window: context=64 tokens → predict next token
  - Warm-start: hierarchy state carries over between consecutive windows
    (efficient: no re-priming every step)
  - Backprop only through: embedding → output_head
  - Report loss + generation sample after each epoch

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
CHECKPOINT_OUT  = "agnis_english_finetuned.pt"  # saved after each epoch
GUTENBERG_URL   = "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"  # Great Expectations
LOCAL_CORPUS    = "slm/input_en.txt"
TARGET_CHARS    = 500_000
CONTEXT_SIZE    = 64         # (only used to truncate short prompts during gen)
STRIDE          = 1          # no longer used in streaming
EPOCHS          = 5
LR              = 1e-3
WARMUP_STEPS    = 500        # linear LR warmup
LOG_EVERY       = 1000       # print loss every N steps
GEN_EVERY_EPOCH = True       # print generation sample after each epoch
TEMPERATURE     = 0.8
MAX_GEN_TOKENS  = 40
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "The cat sat on",
]


# ── Step 1: Fetch corpus ─────────────────────────────────────────────────────

def fetch_corpus() -> str:
    # 1a) Use existing local file if present
    if os.path.exists(LOCAL_CORPUS):
        with open(LOCAL_CORPUS, encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"[Corpus] Loaded {LOCAL_CORPUS} ({len(text):,} chars)")
        return text[:TARGET_CHARS]

    # 1b) Download from Project Gutenberg
    cache = "gutenberg_great_expectations.txt"
    if not os.path.exists(cache):
        print(f"[Corpus] Downloading Great Expectations from Gutenberg...")
        try:
            urllib.request.urlretrieve(GUTENBERG_URL, cache)
            print(f"[Corpus] Downloaded → {cache}")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            sys.exit(1)

    with open(cache, encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"[Corpus] Loaded {cache} ({len(text):,} chars)")
    return text


# ── Step 2: Clean text ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # Strip Gutenberg header (find first chapter)
    for marker in ["CHAPTER I.", "CHAPTER I", "CHAPTER 1", "Chapter I", "Chapter 1", "CHAPTER"]:
        idx = text.find(marker)
        if idx != -1:
            # Check if this is just the table of contents
            next_idx = text.find(marker, idx + len(marker))
            if next_idx != -1 and (next_idx - idx) < 1000:
                # Likely TOC, find the second occurrence
                idx = next_idx
            text = text[idx:]
            break

    # Strip Gutenberg footer
    for marker in ["End of the Project Gutenberg", "THE END", "End of Project Gutenberg"]:
        idx = text.rfind(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Remove excessive whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)   # ASCII printable only
    return text.strip()[:TARGET_CHARS]


# ── Step 3: Tokenize ─────────────────────────────────────────────────────────

def tokenize(text: str, tokenizer) -> list:
    enc = tokenizer.encode(text)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    print(f"[Tokenize] {len(text):,} chars -> {len(ids):,} tokens "
          f"(avg {len(text)/max(1,len(ids)):.1f} chars/token)")
    return ids


# ── Step 4: Training ─────────────────────────────────────────────────────────

def train(wrapper: AGNISSLMWrapper, token_ids: list):
    device = wrapper.device

    # Freeze hierarchy completely
    for param in wrapper.hierarchy.parameters():
        param.requires_grad_(False)

    # Only train embedding + output_head
    trainable = list(wrapper.embedding.parameters()) + \
                list(wrapper.output_head.parameters())
    total_params = sum(p.numel() for p in trainable)
    print(f"\n[Train] Trainable params: {total_params:,} "
          f"(embedding + output_head only)")
    print(f"[Train] Hierarchy FROZEN. Device: {device}")

    optimizer = torch.optim.Adam(trainable, lr=LR)

    total_steps = len(token_ids) - 1
    print(f"[Train] Streaming {total_steps:,} tokens sequentially per epoch.")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_start = time.time()

        # Reset hierarchy state at the start of each epoch
        wrapper.hierarchy.reset_states(batch_size=1)

        for step in range(total_steps):
            cur_id = token_ids[step]
            tgt_id = token_ids[step + 1]

            # Linear warmup (only in epoch 0)
            if epoch == 0 and step <= WARMUP_STEPS:
                lr_scale = max(0.01, step / max(1, WARMUP_STEPS))
                for pg in optimizer.param_groups:
                    pg['lr'] = LR * lr_scale

            optimizer.zero_grad()

            # Embed current token
            cur_t = torch.tensor([[cur_id]], dtype=torch.long, device=device)
            emb = nn.functional.normalize(
                wrapper.embedding(cur_t).view(1, -1), dim=-1
            ) # requires grad

            # Hierarchy step (frozen)
            with torch.no_grad():
                pred_embed = wrapper.hierarchy.predict_label(
                    emb.detach(), max_steps=1, update_temporal=True
                )
                if pred_embed.shape[1] > wrapper.embed_dim:
                    pred_embed = pred_embed[:, :wrapper.embed_dim]

            # Output mapping
            logits = wrapper.output_head(pred_embed.detach())  # [1, vocab_size]

            target_t = torch.tensor([tgt_id], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target_t)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - epoch_start
                tps = (step + 1) / max(1, elapsed)
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1:>6}/{total_steps} | "
                      f"Loss: {avg:.4f} | PPL: {ppl:.1f} | {tps:.1f} steps/s",
                      end="\r")

        avg_loss = epoch_loss / max(1, total_steps)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1}/{EPOCHS} COMPLETE | "
              f"Avg Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
              f"{elapsed:.1f}s")

        # Generation sample after each epoch
        if GEN_EVERY_EPOCH:
            print(f"\n  --- Generation samples (epoch {epoch+1}) ---")
            for prompt in PROMPTS:
                out = wrapper.generate(prompt, max_new_tokens=MAX_GEN_TOKENS,
                                       temperature=TEMPERATURE)
                safe = out.encode('ascii', errors='replace').decode('ascii')
                print(f"  [{prompt}] → {safe}")
            print()

        # Save checkpoint after each epoch
        torch.save({
            'epoch':       epoch + 1,
            'embedding':   wrapper.embedding.state_dict(),
            'output_head': wrapper.output_head.state_dict(),
            'vocab_size':  wrapper.vocab_size,
            'embed_dim':   wrapper.embed_dim,
        }, CHECKPOINT_OUT)
        print(f"  [Saved] {CHECKPOINT_OUT}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  AGNIS English Fine-Tuning")
    print("  Embedding + Output Head only | Hierarchy FROZEN")
    print("=" * 60)

    # Build wrapper
    wrapper = AGNISSLMWrapper(device=DEVICE)
    wrapper.load_checkpoint(CHECKPOINT_IN)
    wrapper.to(wrapper.device)

    tokenizer = wrapper._tokenizer
    if tokenizer is None:
        print("[ERROR] No tokenizer loaded. Aborting.")
        sys.exit(1)

    # Corpus
    raw = fetch_corpus()
    text = clean_text(raw)
    print(f"[Corpus] Clean text: {len(text):,} chars")
    print(f"[Corpus] Preview: {text[:200]!r}")

    # Tokenize
    token_ids = tokenize(text, tokenizer)
    if len(token_ids) < CONTEXT_SIZE + 10:
        print("[ERROR] Corpus too short after tokenization.")
        sys.exit(1)

    # Train
    train(wrapper, token_ids)

    # Final generation
    print("\n" + "=" * 60)
    print("  FINAL GENERATION TEST")
    print("=" * 60)
    for prompt in PROMPTS:
        out = wrapper.generate(prompt, max_new_tokens=50, temperature=TEMPERATURE)
        safe = out.encode('ascii', errors='replace').decode('ascii')
        print(f"\nPrompt: {prompt}")
        print(f"Output: {safe}")

    print(f"\n[Done] Fine-tuned weights saved to: {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
