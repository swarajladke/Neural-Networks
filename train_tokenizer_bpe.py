"""
Train a byte-level BPE tokenizer (GPT-2/llama-bpe compatible).
"""

from __future__ import annotations

import argparse
import os

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="multilang_corpus.txt")
    parser.add_argument("--output", type=str, default="slm_bpe_tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input corpus not found: {args.input}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
    )
    tokenizer.train([args.input], trainer=trainer)
    tokenizer.save(args.output)
    print(f"Saved BPE tokenizer: {args.output}")


if __name__ == "__main__":
    main()
