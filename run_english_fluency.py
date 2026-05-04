"""
run_english_fluency.py
======================
Single script — run this once on Colab.

  Step 1: Retrain BPE tokenizer on English prose (not code)
  Step 2: Train embedding + output_head only (hierarchy FROZEN)
  Step 3: Print fluency test

Architecture:
  AGNIS Hierarchy  → FROZEN  (SNAP-ATP, dx.detach() intact — pure)
  Embedding        → TRAINED (fresh, English vocab)
  Output Head      → TRAINED (fresh, English vocab)

Usage:
    python run_english_fluency.py

Expected time on Colab T4:
  Tokenizer training : ~2 min
  Interface training : ~60-90 min (25 epochs)
"""

import os, re, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm.agnis_slm_wrapper import AGNISSLMWrapper

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CHECKPOINT_IN     = "agnis_marathon_final.pt"
CHECKPOINT_OUT    = "agnis_english_interface.pt"
CORPUS_PATH       = "slm/input_en_massive.txt"
TOKENIZER_OUT     = "slm_bpe_tokenizer_en.json"
TARGET_CHARS      = 5_000_000

VOCAB_SIZE        = 4096
EMBED_DIM         = 110    # must match hierarchy input_dim

BATCH_SIZE        = 64
EPOCHS            = 25
LR                = 5e-4
WARMUP_STEPS      = 1000
LOG_EVERY         = 500
GEN_EVERY_N       = 5      # print generation sample every N epochs
TEMPERATURE       = 0.8
REP_PENALTY       = 1.3
MAX_GEN_TOKENS    = 60
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════
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


def sep(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — RETRAIN TOKENIZER
# ═══════════════════════════════════════════════════════════════
def step1_train_tokenizer() -> Tokenizer:
    sep("STEP 1 — Retrain BPE Tokenizer on English Prose")

    if not os.path.exists(CORPUS_PATH):
        print(f"[ERROR] Corpus not found: {CORPUS_PATH}")
        sys.exit(1)

    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = clean_text(raw)[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")

    # Write temp file for trainer
    tmp = "slm/_tmp_en_clean.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)

    # Build and train tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|unk|>", "<|endoftext|>", "<|pad|>"],
        show_progress=True,
    )
    print(f"[Tokenizer] Training BPE (vocab={VOCAB_SIZE})...")
    tokenizer.train(files=[tmp], trainer=trainer)
    os.remove(tmp)

    tokenizer.save(TOKENIZER_OUT)
    print(f"[Tokenizer] Saved -> {TOKENIZER_OUT}")

    # Sanity check
    samples = [
        "The whale surfaced near the ship.",
        "Once upon a time in a distant land.",
        "It was the best of times, it was the worst of times.",
    ]
    print("\n[Tokenizer] Sanity check:")
    for s in samples:
        enc = tokenizer.encode(s)
        toks = [tokenizer.id_to_token(i) for i in enc.ids[:10]]
        print(f"  ({len(enc.ids):>3} tokens) {toks}")

    return tokenizer


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — TRAIN ENGLISH INTERFACE
# ═══════════════════════════════════════════════════════════════
def step2_train_interface(tokenizer: Tokenizer):
    sep("STEP 2 — Train English Language Interface")

    # Load hierarchy (frozen)
    wrapper = AGNISSLMWrapper(device=DEVICE)
    wrapper.load_checkpoint(CHECKPOINT_IN)
    wrapper.to(wrapper.device)

    # Freeze hierarchy completely
    for p in wrapper.hierarchy.parameters():
        p.requires_grad_(False)
    print("[Hierarchy] FROZEN — SNAP-ATP local, dx.detach() intact")

    # Override tokenizer with English one
    wrapper._tokenizer = tokenizer
    wrapper.vocab_size  = VOCAB_SIZE

    # Fresh interface — re-initialize for clean English slate
    wrapper.embedding   = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    wrapper.output_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False).to(DEVICE)  # bias=False: matches wrapper
    nn.init.normal_(wrapper.embedding.weight, std=0.02)
    nn.init.normal_(wrapper.output_head.weight, std=0.02)

    trainable = (list(wrapper.embedding.parameters()) +
                 list(wrapper.output_head.parameters()))
    print(f"[Interface] Fresh init | params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Tokenize corpus
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text      = clean_text(raw)[:TARGET_CHARS]
    enc       = tokenizer.encode(text)
    token_ids = enc.ids
    print(f"[Corpus] {len(text):,} chars -> {len(token_ids):,} tokens")

    # Build parallel streams
    seq_len      = len(token_ids) // BATCH_SIZE
    token_ids    = token_ids[:seq_len * BATCH_SIZE]
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).view(BATCH_SIZE, seq_len)
    total_steps  = seq_len - 1
    print(f"[Train] {BATCH_SIZE} streams × {total_steps:,} steps | {EPOCHS} epochs\n")

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        wrapper.hierarchy.reset_states(batch_size=BATCH_SIZE)
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step in range(total_steps):
            cur_id = token_tensor[:, step]
            tgt_id = token_tensor[:, step + 1]

            # Warmup (epoch 0 only)
            if epoch == 0 and step <= WARMUP_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * max(0.01, step / max(1, WARMUP_STEPS))

            # Interface forward (grad flows here)
            emb = F.normalize(wrapper.embedding(cur_id), dim=-1)  # [B, embed_dim]

            # Hierarchy forward (NO grad — frozen, pure SNAP-ATP)
            with torch.no_grad():
                context = wrapper.hierarchy.predict_label(
                    emb, max_steps=1, update_temporal=True
                )
                if context.shape[1] > EMBED_DIM:
                    context = context[:, :EMBED_DIM]

            # Residual: emb (grad path) + context (frozen signal)
            # Gradients flow: loss → output_head → emb → embedding weights ✓
            # Hierarchy is untouched — context.detach() ensures no grad leaks in
            combined = emb + 0.5 * context.detach()
            logits   = wrapper.output_head(combined)
            loss     = F.cross_entropy(logits, tgt_id)

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
                    f"{tps:.0f} tok/s",
                    end="\r", flush=True,
                )

        avg_loss = epoch_loss / max(1, total_steps)
        ppl      = math.exp(min(avg_loss, 20))
        elapsed  = time.time() - epoch_start
        tag      = "  ← best" if avg_loss < best_loss else ""
        best_loss = min(best_loss, avg_loss)

        print(f"\n  Epoch {epoch+1:>2}/{EPOCHS} | "
              f"Loss {avg_loss:.4f} | PPL {ppl:.1f} | {elapsed:.0f}s{tag}")

        # Generation sample every GEN_EVERY_N epochs
        if (epoch + 1) % GEN_EVERY_N == 0:
            print(f"\n  --- Fluency check (epoch {epoch+1}) ---")
            for p in PROMPTS:
                out  = full_generate(wrapper, tokenizer, p)
                safe = out.replace('\r', '').replace('\n', ' ')
                print(f"  [{p}] -> {safe}\n")
            print()

        wrapper.save_checkpoint(CHECKPOINT_OUT)
        print(f"  [Saved] {CHECKPOINT_OUT}")


# ═══════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def _generate(wrapper, tokenizer, prompt: str = None) -> str:
    # Called with prompt from outer scope if not supplied
    return ""          # placeholder — see full_generate below

@torch.no_grad()
def full_generate(wrapper, tokenizer, prompt: str) -> str:
    device = wrapper.device
    wrapper.hierarchy.reset_states(batch_size=1)

    enc        = tokenizer.encode(prompt)
    prompt_ids = enc.ids
    gen_ids    = list(prompt_ids)

    for tok_id in prompt_ids:
        emb = F.normalize(
            wrapper.embedding(torch.tensor([[tok_id]], device=device)).view(1, -1),
            dim=-1,
        )
        wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)

    eos = tokenizer.token_to_id("<|endoftext|>") or -1

    for _ in range(MAX_GEN_TOKENS):
        last  = torch.tensor([[gen_ids[-1]]], device=device)
        emb   = F.normalize(wrapper.embedding(last).view(1, -1), dim=-1)
        hid   = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > EMBED_DIM:
            hid = hid[:, :EMBED_DIM]

        logits = wrapper.output_head(hid) / TEMPERATURE
        for tok in set(gen_ids[-20:]):
            logits[0, tok] = (logits[0, tok] / REP_PENALTY
                              if logits[0, tok] > 0
                              else logits[0, tok] * REP_PENALTY)

        next_tok = torch.multinomial(F.softmax(logits, dim=-1)[0], 1).item()
        gen_ids.append(next_tok)
        if next_tok == eos:
            break

    return tokenizer.decode(gen_ids)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    sep("AGNIS English Fluency Pipeline")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Vocab    : {VOCAB_SIZE}")
    print(f"  Corpus   : {CORPUS_PATH}")

    tokenizer = step1_train_tokenizer()
    step2_train_interface(tokenizer)

    # Reload and final test
    sep("FINAL FLUENCY TEST")
    tok  = Tokenizer.from_file(TOKENIZER_OUT)
    wrap = AGNISSLMWrapper(device=DEVICE)
    wrap.load_checkpoint(CHECKPOINT_OUT)
    wrap._tokenizer = tok
    wrap.vocab_size  = VOCAB_SIZE

    for prompt in PROMPTS:
        out  = full_generate(wrap, tok, prompt)
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPrompt : {prompt}")
        print(f"Output : {safe}")

    print(f"\n[Done] Weights -> {CHECKPOINT_OUT}")
    print(f"[Done] Tokenizer -> {TOKENIZER_OUT}")


if __name__ == "__main__":
    main()
