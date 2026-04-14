"""
Build a plain-text Python corpus from a public Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import random
from typing import Iterable

from datasets import load_dataset


def iter_samples(ds: Iterable, text_field: str):
    for sample in ds:
        if text_field not in sample:
            raise KeyError(f"Field '{text_field}' not found in sample keys: {list(sample.keys())}")
        text = sample[text_field]
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        elif not isinstance(text, str):
            text = str(text)
        yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml6team/the-stack-smol-python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="content")
    parser.add_argument("--output", type=str, default="python_corpus.txt")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    if not args.streaming and args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    elif args.streaming and args.shuffle:
        random.seed(args.seed)

    count = 0
    kept = 0
    total_chars = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for text in iter_samples(ds, args.text_field):
            count += 1
            if args.min_chars > 0 and len(text) < args.min_chars:
                continue
            f.write(text.strip() + "\n\n")
            kept += 1
            total_chars += len(text)
            if args.max_samples and kept >= args.max_samples:
                break

    avg_chars = total_chars / max(1, kept)
    print(f"Loaded samples: {count}")
    print(f"Written samples: {kept}")
    print(f"Average chars per sample: {avg_chars:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
