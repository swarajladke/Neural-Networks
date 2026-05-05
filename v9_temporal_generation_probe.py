"""
v9_temporal_generation_probe.py
===============================
Quick probe for Option 1: Temporal Predictive Coding.

Core idea:
  Feed the current token embedding X_t into the bottom of AGNIS while
  strongly pushing the top layer toward the next-token embedding X_{t+1}.

This keeps learning local and biologically flavored:
  - no backprop through the hierarchy
  - no external readout optimizer
  - next-token behavior emerges from top-down temporal pressure

Evaluation:
  - mean training surprise
  - nearest-neighbor next-token retrieval accuracy
  - sampled free generation
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

from agnis_v4_cognitive import CognitivePredictiveAgent
from experiment_utils import metric_to_float
from slm.agnis_slm_wrapper import AGNISSLMWrapper


CHECKPOINT_IN = "agnis_english_interface.pt"
TOKENIZER_PATHS = [
    "slm_bpe_tokenizer_en.json",
    "slm_bpe_tokenizer.json",
]
CORPUS_PATH = "slm/input_en_massive.txt"

TARGET_CHARS = 1_000_000
BATCH_SIZE = 32
EPOCHS = 3
MAX_STEPS = 8
BETA_PUSH = 6.0
RECOGNITION_WEIGHT = 1.0
TEMPERATURE = 0.7
MAX_GEN_TOKENS = 60
EVAL_STEPS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


def sep(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")


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


def load_tokenizer() -> tuple[Tokenizer, str]:
    for path in TOKENIZER_PATHS:
        if os.path.exists(path):
            return Tokenizer.from_file(path), path
    raise FileNotFoundError(
        "No tokenizer found. Expected one of: "
        + ", ".join(TOKENIZER_PATHS)
    )


def load_corpus() -> str:
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = clean_text(raw)[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text


def build_streams(token_ids: list[int], batch_size: int, device: str) -> torch.Tensor:
    seq_len = len(token_ids) // batch_size
    usable = token_ids[: seq_len * batch_size]
    return torch.tensor(usable, dtype=torch.long, device=device).view(batch_size, seq_len)


def predict_next_embed(wrapper: AGNISSLMWrapper, emb: torch.Tensor) -> torch.Tensor:
    pred = wrapper.hierarchy.predict_label(
        emb, max_steps=1, update_temporal=True, recognition_weight=RECOGNITION_WEIGHT
    )
    if pred.shape[1] > wrapper.embed_dim:
        pred = pred[:, : wrapper.embed_dim]
    elif pred.shape[1] < wrapper.embed_dim:
        pad = torch.zeros(
            pred.shape[0], wrapper.embed_dim - pred.shape[1], device=pred.device
        )
        pred = torch.cat([pred, pad], dim=1)
    return F.normalize(pred, dim=-1)


def nearest_token_ids(wrapper: AGNISSLMWrapper, pred_embed: torch.Tensor) -> torch.Tensor:
    emb_table = F.normalize(wrapper.embedding.weight, dim=-1)
    sims = torch.matmul(pred_embed, emb_table.t())
    return torch.argmax(sims, dim=-1)


def retrieval_accuracy(
    wrapper: AGNISSLMWrapper, token_streams: torch.Tensor, eval_steps: int
) -> float:
    steps = min(eval_steps, token_streams.shape[1] - 1)
    wrapper.hierarchy.reset_states(batch_size=token_streams.shape[0])
    correct = 0
    total = 0

    with torch.no_grad():
        for step in range(steps):
            cur_id = token_streams[:, step]
            tgt_id = token_streams[:, step + 1]
            emb = F.normalize(wrapper.embedding(cur_id), dim=-1)
            pred_embed = predict_next_embed(wrapper, emb)
            pred_id = nearest_token_ids(wrapper, pred_embed)
            correct += int((pred_id == tgt_id).sum().item())
            total += int(tgt_id.numel())

    return correct / max(1, total)


@torch.no_grad()
def generate(wrapper: AGNISSLMWrapper, tokenizer: Tokenizer, prompt: str) -> str:
    enc = tokenizer.encode(prompt)
    token_ids = enc.ids if enc.ids else [0]
    generated = list(token_ids)
    wrapper.hierarchy.reset_states(batch_size=1)

    for tok_id in token_ids:
        tok = torch.tensor([tok_id], dtype=torch.long, device=wrapper.device)
        emb = F.normalize(wrapper.embedding(tok), dim=-1)
        _ = predict_next_embed(wrapper, emb)

    emb_table = F.normalize(wrapper.embedding.weight, dim=-1)

    for _ in range(MAX_GEN_TOKENS):
        cur = torch.tensor([generated[-1]], dtype=torch.long, device=wrapper.device)
        emb = F.normalize(wrapper.embedding(cur), dim=-1)
        pred_embed = predict_next_embed(wrapper, emb)
        sims = torch.matmul(pred_embed, emb_table.t())[0] / TEMPERATURE
        probs = F.softmax(sims, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)

    return tokenizer.decode(generated)


def train_temporal_probe(
    wrapper: AGNISSLMWrapper,
    agent: CognitivePredictiveAgent,
    token_streams: torch.Tensor,
) -> None:
    total_steps = token_streams.shape[1] - 1
    print(f"[Train] {token_streams.shape[0]} streams x {total_steps:,} steps | {EPOCHS} epochs")
    print(f"[Train] max_steps={MAX_STEPS} | beta_push={BETA_PUSH}")

    for epoch in range(EPOCHS):
        wrapper.hierarchy.reset_states(batch_size=token_streams.shape[0])
        epoch_surprise = 0.0
        epoch_weight = 0.0
        t0 = time.time()

        for step in range(total_steps):
            cur_id = token_streams[:, step]
            nxt_id = token_streams[:, step + 1]

            with torch.no_grad():
                x_embed = F.normalize(wrapper.embedding(cur_id), dim=-1)
                y_embed = F.normalize(wrapper.embedding(nxt_id), dim=-1)

            weight, surprise = agent.observe_and_learn(
                x_embed,
                y_embed,
                task_id=epoch,
                max_steps=MAX_STEPS,
                recognition_weight=RECOGNITION_WEIGHT,
                beta_push=BETA_PUSH,
                warm_start=True,
            )
            epoch_weight += metric_to_float(weight)
            epoch_surprise += metric_to_float(surprise)

            if step % 500 == 0:
                print(f"    [Step {step:>5}/{total_steps}] Surprise: {metric_to_float(surprise):.4f} | Weight: {metric_to_float(weight):.4f}", end="\r")

        avg_surprise = epoch_surprise / max(1, total_steps)
        avg_weight = epoch_weight / max(1, total_steps)
        acc = retrieval_accuracy(wrapper, token_streams[:8], eval_steps=EVAL_STEPS)
        dt = time.time() - t0
        print(
            f"\n  Epoch {epoch + 1:>2}/{EPOCHS} | "
            f"Surprise {avg_surprise:.4f} | "
            f"Salience {avg_weight:.3f} | "
            f"Top-1 {acc * 100:.2f}% | {dt:.0f}s"
        )

        for prompt in PROMPTS[:2]:
            out = generate(wrapper, wrapper._tokenizer, prompt)
            safe = out.encode("ascii", errors="replace").decode("ascii")
            print(f"  [{prompt}] -> {safe}")


def main() -> None:
    sep("AGNIS V9 Temporal Generation Probe")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Objective: X_t bottom-up, X_t+1 top-down")

    tokenizer, tokenizer_path = load_tokenizer()
    print(f"[Tokenizer] Loaded {tokenizer_path} | vocab={tokenizer.get_vocab_size()}")

    wrapper = AGNISSLMWrapper(
        vocab_size=tokenizer.get_vocab_size(),
        device=DEVICE,
    )
    wrapper.load_checkpoint(CHECKPOINT_IN)
    wrapper.to(wrapper.device)
    wrapper._tokenizer = tokenizer
    print(f"[Checkpoint] Loaded {CHECKPOINT_IN}")

    text = load_corpus()
    token_ids = tokenizer.encode(text).ids
    print(f"[Tokenize] {len(token_ids):,} tokens")

    streams = build_streams(token_ids, BATCH_SIZE, DEVICE)
    agent = CognitivePredictiveAgent(wrapper.hierarchy, device=DEVICE)

    sep("Training")
    train_temporal_probe(wrapper, agent, streams)

    sep("Final Samples")
    final_acc = retrieval_accuracy(wrapper, streams[:8], eval_steps=EVAL_STEPS)
    print(f"[Eval] Next-token top-1 accuracy: {final_acc * 100:.2f}%")
    for prompt in PROMPTS:
        out = generate(wrapper, tokenizer, prompt)
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPrompt: {prompt}")
        print(f"Output: {safe}")


if __name__ == "__main__":
    main()
