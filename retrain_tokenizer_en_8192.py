"""
retrain_tokenizer_en_8192.py
============================
Train a larger English BPE tokenizer for the frozen-core fluency comparison.

Output:
  slm_bpe_tokenizer_en_8192.json

Purpose:
  Test whether a larger vocabulary reduces broken word fragments and improves
  surface fluency while keeping the AGNIS core and language head pipeline fixed.
"""

from __future__ import annotations

import os
import re
import sys

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


CORPUS_PATH = "slm/input_en_massive.txt"
OUTPUT_PATH = "slm_bpe_tokenizer_en_8192.json"
TEMP_PATH = "slm/_en_corpus_clean_8192.txt"
VOCAB_SIZE = 8192


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


def main() -> None:
    if not os.path.exists(CORPUS_PATH):
        print(f"[ERROR] Corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    print(f"[Tokenizer-8192] Loading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    text = clean_text(raw)
    print(f"[Tokenizer-8192] Clean corpus: {len(text):,} chars | {len(text.split()):,} words")

    with open(TEMP_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Tokenizer-8192] Wrote clean corpus to {TEMP_PATH}")

    print(f"\n[Tokenizer-8192] Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
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

    tokenizer.save(OUTPUT_PATH)
    os.remove(TEMP_PATH)
    print(f"\n[Tokenizer-8192] Saved to: {OUTPUT_PATH}")

    test_sentences = [
        "The whale surfaced near the ship.",
        "Once upon a time in a land far away.",
        "It was the best of times, it was the worst of times.",
    ]
    print("\n[Tokenizer-8192] Sanity check:")
    print(f"  {'Sentence':<50} {'Tokens':>6}  Preview")
    print(f"  {'-' * 80}")
    for sentence in test_sentences:
        enc = tokenizer.encode(sentence)
        tokens = [tokenizer.id_to_token(i) for i in enc.ids[:8]]
        print(f"  {sentence[:48]:<50} {len(enc.ids):>6}  {tokens}")

    print("\n[Done] 8192-vocab English BPE tokenizer is ready.")
    print(f"       Use TOKENIZER_PATH='{OUTPUT_PATH}' in the frozen-core fluency script.")


if __name__ == "__main__":
    main()
