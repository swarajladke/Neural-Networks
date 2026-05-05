"""
train_agnis_fluency_model.py
============================
Train the stronger AGNIS fluency architecture:

  token -> embedding -> frozen AGNIS core -> fusion MLP -> LM head -> vocab logits

This script is intentionally single-language and single-pipeline so the
fluency test is clean and easy to interpret.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import urllib.request

from slm.agnis_fluency_model import AGNISFluencyModel


CORE_CHECKPOINT = "agnis_marathon_final.pt"
MODEL_OUT = "agnis_fluency_model_en.pt"
TOKENIZER_PATH = "slm_bpe_tokenizer_en_8192.json"
CORPUS_PATH = "slm/input_en_massive.txt"

TARGET_CHARS = 25_000_000
BATCH_SIZE = 64
EPOCHS = 20
LR = 4e-4
WARMUP_STEPS = 1000
LOG_EVERY = 500
MAX_GEN_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOPPING_PATIENCE = 4
FUSION_HIDDEN_DIM = 1024

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/135/pg135.txt",     # Les Miserables (3.2M)
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",   # War and Peace (3.2M)
    "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",   # Count of Monte Cristo (2.6M)
    "https://www.gutenberg.org/cache/epub/996/pg996.txt",     # Don Quixote (Eng) (2.3M)
    "https://www.gutenberg.org/cache/epub/28054/pg28054.txt", # The Brothers Karamazov (2.0M)
    "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",   # Anna Karenina (2.0M)
    "https://www.gutenberg.org/cache/epub/766/pg766.txt",     # David Copperfield (2.0M)
    "https://www.gutenberg.org/cache/epub/1023/pg1023.txt",   # Bleak House (2.0M)
    "https://www.gutenberg.org/cache/epub/145/pg145.txt",     # Middlemarch (1.8M)
    "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",   # Ulysses (1.5M)
    "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",   # Crime and Punishment (1.2M)
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",   # Moby Dick (1.2M)
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",   # Great Expectations (1.0M)
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",   # Pride and Prejudice (0.7M)
    "https://www.gutenberg.org/cache/epub/98/pg98.txt"        # Tale of Two Cities (0.7M)
]


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
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def load_corpus() -> str:
    if not os.path.exists(CORPUS_PATH) or os.path.getsize(CORPUS_PATH) < 20_000_000:
        print("[Corpus] Downloading massive English dataset (25M+ chars)...")
        os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
        full_text = ""
        for url in GUTENBERG_URLS:
            try:
                fname = url.split("/")[-1]
                print(f"  -> Downloading {fname}...")
                raw = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")
                full_text += clean_text(raw) + "\n\n"
            except Exception as e:
                print(f"  -> Failed: {e}")
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            f.write(full_text)
        print("[Corpus] Download complete.")

    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = clean_text(raw)[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text


def build_token_tensor(token_ids: list[int], batch_size: int, device: str) -> torch.Tensor:
    seq_len = len(token_ids) // batch_size
    token_ids = token_ids[: seq_len * batch_size]
    return torch.tensor(token_ids, dtype=torch.long, device=device).view(batch_size, seq_len)


@torch.no_grad()
def heldout_ppl(model: AGNISFluencyModel, tokens: torch.Tensor, steps: int = 512) -> tuple[float, float]:
    eval_steps = min(steps, tokens.shape[1] - 1)
    model.eval()
    model.reset_states(batch_size=tokens.shape[0])
    total_loss = 0.0

    for step in range(eval_steps):
        cur = tokens[:, step]
        tgt = tokens[:, step + 1]
        logits = model.step_logits(cur, update_temporal=True, max_steps=1)
        loss = F.cross_entropy(logits, tgt)
        total_loss += float(loss.item())

    avg = total_loss / max(1, eval_steps)
    return avg, math.exp(min(avg, 20))


def main() -> None:
    print("\n" + "=" * 60)
    print("  AGNIS Fluency Model Training")
    print("  Frozen Core | Fusion MLP | Proper LM Head")
    print("=" * 60)

    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ERROR] Tokenizer not found: {TOKENIZER_PATH}")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] Loaded {TOKENIZER_PATH} | vocab={vocab_size}")

    model = AGNISFluencyModel(
        vocab_size=vocab_size,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        device=DEVICE,
    )
    model.load_core_checkpoint(CORE_CHECKPOINT)
    model._tokenizer = tokenizer
    model.freeze_core()
    model.tie_output_weights()
    model.to(model.device)

    trainable = [
        *model.embedding.parameters(),
        *model.fusion_norm.parameters(),
        *model.proj.parameters(),
        *model.out_norm.parameters(),
    ]
    if not model.tie_weights:
        trainable += list(model.lm_head.parameters())

    n_params = sum(p.numel() for p in trainable)
    print(f"[Trainable] {n_params:,} parameters")
    print(f"[Head] fusion_hidden_dim={FUSION_HIDDEN_DIM} | 2-layer projection MLP")

    text = load_corpus()
    enc = tokenizer.encode(text)
    token_tensor = build_token_tensor(enc.ids, BATCH_SIZE, DEVICE)
    total_steps = token_tensor.shape[1] - 1
    split = max(1024, token_tensor.shape[1] // 20)
    train_tokens = token_tensor[:, :-split]
    valid_tokens = token_tensor[:, -split:]

    print(f"[Tokenize] {len(enc.ids):,} tokens")
    print(f"[Train] {BATCH_SIZE} streams x {train_tokens.shape[1] - 1:,} steps | {EPOCHS} epochs")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        model.reset_states(batch_size=BATCH_SIZE)
        epoch_loss = 0.0
        start = time.time()

        for step in range(train_tokens.shape[1] - 1):
            cur = train_tokens[:, step]
            tgt = train_tokens[:, step + 1]

            if epoch == 0 and step <= WARMUP_STEPS:
                scale = max(0.01, step / max(1, WARMUP_STEPS))
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * scale

            logits = model.step_logits(cur, update_temporal=True, max_steps=1)
            loss = F.cross_entropy(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            model.tie_output_weights()

            epoch_loss += float(loss.item())

            if (step + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                speed = (step + 1) / max(time.time() - start, 1e-6)
                print(
                    f"  Epoch {epoch+1:>2}/{EPOCHS} | "
                    f"Step {step+1:>6}/{train_tokens.shape[1]-1} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | {speed:.0f} tok/s",
                    end="\r",
                    flush=True,
                )

        train_loss = epoch_loss / max(1, train_tokens.shape[1] - 1)
        train_ppl = math.exp(min(train_loss, 20))
        val_loss, val_ppl = heldout_ppl(model, valid_tokens)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            model.save_fluency_checkpoint(MODEL_OUT)
            print("  [Saved best checkpoint]")
        else:
            stale_epochs += 1

        print(
            f"\n  Epoch {epoch+1:>2}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | Train PPL {train_ppl:.1f} | "
            f"Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f}"
            + ("  <- best" if improved else "")
        )

        if (epoch + 1) % 5 == 0:
            print(f"\n  --- Samples (epoch {epoch+1}) ---")
            for prompt in PROMPTS:
                out = model.generate(
                    prompt,
                    max_new_tokens=MAX_GEN_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY,
                )
                safe = out.encode("ascii", errors="replace").decode("ascii")
                print(f"  [{prompt}] -> {safe}\n")

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                f"\n[Early Stop] No validation improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs. Best epoch: {best_epoch}"
            )
            break

    print("\n" + "=" * 60)
    print("  BEST CHECKPOINT TEST")
    print("=" * 60)
    model.load_fluency_checkpoint(MODEL_OUT)
    model._tokenizer = tokenizer
    print(f"[Best] Epoch {best_epoch} | Val Loss {best_val:.4f} | Val PPL {math.exp(min(best_val, 20)):.1f}")
    for prompt in PROMPTS:
        out = model.generate(
            prompt,
            max_new_tokens=MAX_GEN_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPrompt: {prompt}")
        print(f"Output: {safe}")

    print(f"\n[Done] Saved best model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
