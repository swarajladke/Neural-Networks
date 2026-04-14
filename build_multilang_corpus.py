"""
Build a mixed-language code corpus using Hugging Face datasets.

Default source: code_search_net (python, java, javascript).
React/Next are represented via JavaScript/TypeScript-style code; we include JavaScript here.
"""

from __future__ import annotations

import argparse
import itertools
import random
from typing import Dict, Iterable, Iterator, List

from datasets import load_dataset


def iter_samples(ds: Iterable, text_field: str) -> Iterator[str]:
    for sample in ds:
        if text_field not in sample:
            raise KeyError(f"Field '{text_field}' not found in sample keys: {list(sample.keys())}")
        text = sample[text_field]
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        elif not isinstance(text, str):
            text = str(text)
        yield text


def build_iters(
    dataset: str,
    configs: List[str],
    split: str,
    text_field: str,
    streaming: bool,
    seed: int,
    trust_remote_code: bool
) -> Dict[str, Iterator[str]]:
    iters: Dict[str, Iterator[str]] = {}
    for cfg in configs:
        ds = load_dataset(
            dataset,
            cfg if cfg else None,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )
        if not streaming:
            ds = ds.shuffle(seed=seed)
        iters[cfg] = iter_samples(ds, text_field)
    return iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="code_search_net")
    parser.add_argument("--languages", type=str, default="python,java,javascript")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="func_code")
    parser.add_argument("--output", type=str, default="multilang_corpus.txt")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--no_combined", action="store_true")
    parser.add_argument("--max_samples_per_lang", type=int, default=5000)
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--interleave", action="store_true")
    parser.add_argument("--tag_language", action="store_true")
    parser.add_argument(
        "--source",
        action="append",
        help="Custom source spec: dataset|config|field|max_samples|tag. "
             "Example: semeru/code-text-java||code|3000|java"
    )
    args = parser.parse_args()

    if args.source:
        sources = []
        for spec in args.source:
            parts = spec.split("|")
            if len(parts) != 5:
                raise ValueError(f"Invalid --source spec: {spec}")
            dataset, config, field, max_samples, tag = parts
            sources.append({
                "dataset": dataset.strip(),
                "config": config.strip() or None,
                "field": field.strip(),
                "max_samples": int(max_samples),
                "tag": tag.strip()
            })
        languages = [s["tag"] for s in sources]
    else:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]
        if not languages:
            raise ValueError("No languages provided.")

    rng = random.Random(args.seed)
    if args.source:
        iters = {}
        for src in sources:
            ds = load_dataset(
                src["dataset"],
                src["config"],
                split=args.split,
                streaming=args.streaming,
                trust_remote_code=args.trust_remote_code
            )
            if not args.streaming:
                ds = ds.shuffle(seed=args.seed)
            iters[src["tag"]] = iter_samples(ds, src["field"])
    else:
        iters = build_iters(
            dataset=args.dataset,
            configs=languages,
            split=args.split,
            text_field=args.text_field,
            streaming=args.streaming,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code
        )

    counts = {l: 0 for l in languages}
    max_per_lang = {l: args.max_samples_per_lang for l in languages}
    if args.source:
        max_per_lang = {s["tag"]: s["max_samples"] for s in sources}
    total_written = 0
    total_chars = 0

    def next_valid(lang: str) -> str | None:
        while counts[lang] < max_per_lang[lang]:
            text = next(iters[lang], None)
            if text is None:
                return None
            if args.min_chars > 0 and len(text) < args.min_chars:
                continue
            return text
        return None

    combined_f = None
    if not args.no_combined:
        combined_f = open(args.output, "w", encoding="utf-8")

    per_lang_files = {}
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        for lang in languages:
            path = os.path.join(args.output_dir, f"{lang}.txt")
            per_lang_files[lang] = open(path, "w", encoding="utf-8")

    def write_text(lang: str, text: str):
        if combined_f:
            if args.tag_language:
                combined_f.write(f"\n# <LANG:{lang}>\n")
            combined_f.write(text.strip() + "\n\n")
        if lang in per_lang_files:
            per_lang_files[lang].write(text.strip() + "\n\n")

    try:
        if args.interleave:
            # Round-robin across languages
            cycle = itertools.cycle(languages)
            while any(counts[l] < max_per_lang[l] for l in languages):
                lang = next(cycle)
                if counts[lang] >= max_per_lang[lang]:
                    continue
                text = next_valid(lang)
                if text is None:
                    counts[lang] = max_per_lang[lang]
                    continue
                write_text(lang, text)
                counts[lang] += 1
                total_written += 1
                total_chars += len(text)
        else:
            # Sequential per language (optionally shuffle language order)
            order = languages[:]
            if args.shuffle:
                rng.shuffle(order)
            for lang in order:
                while counts[lang] < max_per_lang[lang]:
                    text = next_valid(lang)
                    if text is None:
                        break
                    write_text(lang, text)
                    counts[lang] += 1
                    total_written += 1
                    total_chars += len(text)
    finally:
        if combined_f:
            combined_f.close()
        for fh in per_lang_files.values():
            fh.close()

    avg_chars = total_chars / max(1, total_written)
    print(f"Languages: {languages}")
    if args.source:
        for lang in languages:
            print(f"Target {lang}: {max_per_lang[lang]}")
    else:
        print(f"Samples per language (target): {args.max_samples_per_lang}")
    print(f"Written samples: {total_written}")
    print(f"Average chars per sample: {avg_chars:.1f}")
    if not args.no_combined:
        print(f"Output: {args.output}")
    if args.output_dir:
        print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
