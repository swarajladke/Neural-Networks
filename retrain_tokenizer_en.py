"""
retrain_tokenizer_en.py
=======================
Trains a fresh BPE tokenizer on the English prose corpus.
Output: slm_bpe_tokenizer_en.json  (replaces the code-biased one)

Vocab size kept at 4096 to stay compatible with existing architecture.

Usage:
    python retrain_tokenizer_en.py
"""

import os
import re
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CORPUS_PATH = "slm/input_en_massive.txt"
OUTPUT_PATH = "slm_bpe_tokenizer_en.json"
VOCAB_SIZE  = 4096   # same as original — keeps architecture compatible

# ── Clean the corpus first ───────────────────────────────────────────────────
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
    text = re.sub(r'[^\x20-\x7E\n]', '', text)   # ASCII printable only
    return text.strip()

# ── Load corpus ───────────────────────────────────────────────────────────────
if not os.path.exists(CORPUS_PATH):
    print(f"[ERROR] Corpus not found at {CORPUS_PATH}")
    sys.exit(1)

print(f"[Tokenizer] Loading corpus from {CORPUS_PATH}...")
with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
    raw = f.read()

text = clean_text(raw)
print(f"[Tokenizer] Clean corpus: {len(text):,} chars | {len(text.split()):,} words")

# Write cleaned corpus to a temp file (trainer reads from file)
TEMP_PATH = "slm/_en_corpus_clean.txt"
with open(TEMP_PATH, "w", encoding="utf-8") as f:
    f.write(text)
print(f"[Tokenizer] Wrote clean corpus to {TEMP_PATH}")

# ── Train BPE tokenizer ───────────────────────────────────────────────────────
print(f"\n[Tokenizer] Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<|unk|>", "<|endoftext|>", "<|pad|>"],
    show_progress=True,
)

tokenizer.train(files=[TEMP_PATH], trainer=trainer)

# ── Save ─────────────────────────────────────────────────────────────────────
tokenizer.save(OUTPUT_PATH)
os.remove(TEMP_PATH)
print(f"\n[Tokenizer] Saved to: {OUTPUT_PATH}")

# ── Quick sanity check ────────────────────────────────────────────────────────
test_sentences = [
    "The whale surfaced near the ship.",
    "Once upon a time in a land far away.",
    "It was the best of times, it was the worst of times.",
]
print("\n[Tokenizer] Sanity check:")
print(f"  {'Sentence':<50} {'Tokens':>6}  {'Preview'}")
print(f"  {'-'*80}")
for s in test_sentences:
    enc = tokenizer.encode(s)
    tokens = [tokenizer.id_to_token(i) for i in enc.ids[:8]]
    print(f"  {s[:48]:<50} {len(enc.ids):>6}  {tokens}")

print("\n[Done] English BPE tokenizer is ready.")
print(f"       Use OUTPUT_PATH='{OUTPUT_PATH}' in your training script.")
