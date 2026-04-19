from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from slm.agnis_slm_wrapper import AGNIS_SLM_Wrapper
from slm.slm_dataset import SLMDataset
from slm.slm_tokenizer import CharTokenizer
from v8_russian_slm_run import (
    get_eta_r_schedule,
    probe_italian_retention,
    probe_italian_retention_isolated,
)


def _load_joint_datasets(seq_length: int, text_limit: int):
    with open("slm/input_it.txt", "r", encoding="utf-8") as f:
        it_text = f.read()[:text_limit]
    with open("slm/input_ru.txt", "r", encoding="utf-8") as f:
        ru_text = f.read()[:text_limit]

    tokenizer = CharTokenizer()
    tokenizer.fit(it_text + ru_text)

    it_dataset = SLMDataset(filepath="slm/input_it.txt", seq_length=seq_length)
    it_dataset.tokenizer = tokenizer
    it_dataset.data_indices = tokenizer.encode(it_text)

    ru_dataset = SLMDataset(filepath="slm/input_ru.txt", seq_length=seq_length)
    ru_dataset.tokenizer = tokenizer
    ru_dataset.data_indices = tokenizer.encode(ru_text)

    return tokenizer, it_dataset, ru_dataset


def run_probe(
    warmup_sec: int,
    russian_sec: int,
    batch_size: int,
    seq_length: int,
    embed_dim: int,
    expert_width: int,
    text_limit: int,
    sample_every: int,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print(" AGNIS V8 MINI PROBE: RUSSIAN ACQUISITION")
    print("==================================================")
    print(
        f"Device={device} | Warmup={warmup_sec}s | Russian={russian_sec}s | "
        f"Batch={batch_size} | Seq={seq_length} | Experts={expert_width}"
    )

    os.makedirs("checkpoints", exist_ok=True)

    tokenizer, it_dataset, ru_dataset = _load_joint_datasets(seq_length=seq_length, text_limit=text_limit)
    slm = AGNIS_SLM_Wrapper(
        vocab_size=tokenizer.vocab_size,
        seq_length=seq_length,
        embed_dim=embed_dim,
        device=device,
    )

    # Stage 1: Italian warm-up
    slm.agent.switch_temporal_context("italian")
    it_batches = it_dataset.get_batches(batch_size=batch_size)
    start = time.time()
    it_steps = 0
    print("\n[Stage 1] Italian warm-up")
    while time.time() - start < warmup_sec:
        try:
            contexts, targets = next(it_batches)
        except StopIteration:
            it_batches = it_dataset.get_batches(batch_size=batch_size)
            contexts, targets = next(it_batches)
        _, it_surprise = slm.learn_step(contexts, targets)
        it_steps += 1
        if it_steps % 25 == 0:
            print(f"  IT batch {it_steps:03d} | surprise={it_surprise:.4f}")

    italian_baseline = probe_italian_retention_isolated(slm, it_dataset, n_batches=5, italian_dim=embed_dim)
    print(f"\nItalian isolated baseline: {italian_baseline:.4f}")

    # Freeze + expansion
    print("\n[Stage 1.5] Freeze and Russian expert recruitment")
    slm.hierarchy.force_recruit_language_sliver(n=expert_width, language="russian")
    for layer in slm.hierarchy.layers:
        start_idx = layer.output_dim - expert_width
        layer.set_experts_bias(start_idx, layer.output_dim, -10.0)

    it_probe_isolated = probe_italian_retention_isolated(slm, it_dataset, n_batches=5, italian_dim=embed_dim)
    drift = abs(it_probe_isolated - italian_baseline)
    print(f"Isolated retention drift after expansion: {drift:.4f}")

    for layer in slm.hierarchy.layers:
        start_idx = layer.output_dim - expert_width
        layer.set_experts_bias(start_idx, layer.output_dim, -0.5)

    forced_experts = slm.hierarchy.layers[0].output_dim - embed_dim
    slm.agent.enable_hypersensitive_discovery(italian_baseline)

    # Stage 2: Russian learning
    print("\n[Stage 2] Russian acquisition")
    slm.agent.switch_temporal_context("russian")
    ru_batches = ru_dataset.get_batches(batch_size=batch_size)
    start = time.time()
    ru_steps = 0
    ru_surprise_history: list[float] = []
    last_sample = ""
    while time.time() - start < russian_sec:
        try:
            contexts, targets = next(ru_batches)
        except StopIteration:
            ru_batches = ru_dataset.get_batches(batch_size=batch_size)
            contexts, targets = next(ru_batches)

        current_eta_r = get_eta_r_schedule(ru_steps)
        for col in slm.hierarchy.layers:
            col.eta_R = current_eta_r

        if ru_steps == max(1, sample_every):
            slm.agent.disable_hypersensitive_discovery()

        _, ru_surprise = slm.learn_step(contexts, targets)
        ru_steps += 1
        ru_surprise_history.append(float(ru_surprise))

        if ru_steps % sample_every == 0:
            it_probe = probe_italian_retention(slm, it_dataset, n_batches=3)
            last_sample = slm.generate(
                ru_dataset.tokenizer,
                prompt="Раскольников ",
                max_new_chars=48,
                temperature=0.8,
            )
            print(
                f"  RU batch {ru_steps:03d} | ru_surprise={ru_surprise:.4f} | "
                f"it_retention={it_probe:.4f}"
            )

    final_it_retention = probe_italian_retention(slm, it_dataset, n_batches=5)
    final_it_drift = final_it_retention - italian_baseline

    first_window = ru_surprise_history[: min(10, len(ru_surprise_history))]
    last_window = ru_surprise_history[-min(10, len(ru_surprise_history)) :]
    ru_start = float(np.mean(first_window)) if first_window else float("nan")
    ru_end = float(np.mean(last_window)) if last_window else float("nan")

    print("\n[Summary]")
    print(f"Forced experts: {forced_experts}")
    print(f"Italian baseline: {italian_baseline:.4f}")
    print(f"Italian final retention: {final_it_retention:.4f}")
    print(f"Italian drift: {final_it_drift:+.4f}")
    print(f"Russian surprise trend: {ru_start:.4f} -> {ru_end:.4f}")
    print(f"Russian batches completed: {ru_steps}")
    print(f"Last Russian sample: {last_sample!r}")

    report_path = "v8_mini_probe_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AGNIS V8 Mini Probe Report\n\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Warmup seconds: `{warmup_sec}`\n")
        f.write(f"- Russian seconds: `{russian_sec}`\n")
        f.write(f"- Batch size: `{batch_size}`\n")
        f.write(f"- Sequence length: `{seq_length}`\n")
        f.write(f"- Expert width: `{expert_width}`\n")
        f.write(f"- Forced experts: `{forced_experts}`\n")
        f.write(f"- Italian baseline: `{italian_baseline:.4f}`\n")
        f.write(f"- Italian final retention: `{final_it_retention:.4f}`\n")
        f.write(f"- Italian drift: `{final_it_drift:+.4f}`\n")
        f.write(f"- Russian surprise trend: `{ru_start:.4f} -> {ru_end:.4f}`\n")
        f.write(f"- Russian batches completed: `{ru_steps}`\n")
        f.write(f"- Last Russian sample: `{last_sample}`\n")

    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-sec", type=int, default=15)
    parser.add_argument("--russian-sec", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--expert-width", type=int, default=16)
    parser.add_argument("--text-limit", type=int, default=150000)
    parser.add_argument("--sample-every", type=int, default=40)
    args = parser.parse_args()

    run_probe(
        warmup_sec=args.warmup_sec,
        russian_sec=args.russian_sec,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        embed_dim=args.embed_dim,
        expert_width=args.expert_width,
        text_limit=args.text_limit,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    main()
