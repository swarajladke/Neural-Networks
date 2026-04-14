"""
Train a SentencePiece tokenizer for LLaMA-style models.
"""

from __future__ import annotations

import argparse
import os
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="multilang_corpus.txt")
    parser.add_argument("--model_prefix", type=str, default="slm_spm")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", type=str, default="bpe")
    parser.add_argument("--byte_fallback", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input corpus not found: {args.input}")

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        byte_fallback=args.byte_fallback,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        user_defined_symbols=[]
    )
    print(f"Saved: {args.model_prefix}.model / {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
