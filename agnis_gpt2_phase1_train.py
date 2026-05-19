from __future__ import annotations

import itertools
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer

from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint


DEVICE = "cuda"
SEQ_LEN = 128
BATCH_SIZE = 16
LR = 1e-3
MAX_STEPS = 20_000
SAVE_EVERY = 2_000
LOG_EVERY = 500
GEN_EVERY = 2_000
SAVE_PATH = "/kaggle/working/agnis_gpt2_phase1.pt"
MODEL_NAME = "gpt2"
PROMPTS = [
    "The history of artificial intelligence",
    "Scientists recently discovered that",
    "In the next decade, technology will",
    "The most important thing about learning is",
]
LOSS_WINDOW = 100
OOM_BATCH_SIZE = 8
LOW_LOSS_THRESHOLD = 2.0
HIGH_LOSS_THRESHOLD = 4.0


def build_tokenizer() -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def stream_token_batches(
    tokenizer: GPT2Tokenizer,
    seq_len: int,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    chunk_len = seq_len + 1
    batch_token_count = chunk_len * batch_size
    token_buffer: list[int] = []

    for row in dataset:
        text = row.get("text", "").strip()
        if len(text) < 32:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < chunk_len:
            continue

        token_buffer.extend(token_ids)
        while len(token_buffer) >= batch_token_count:
            current = token_buffer[:batch_token_count]
            del token_buffer[:batch_token_count]
            batch = torch.tensor(current, dtype=torch.long).view(batch_size, chunk_len)
            yield batch[:, :-1]


def freeze_phase1(hybrid: AgnisGpt2Hybrid) -> None:
    for param in hybrid.agnis_core.parameters():
        param.requires_grad_(False)
    for param in hybrid.gpt2.parameters():
        param.requires_grad_(False)
    hybrid.agnis_core.eval()
    hybrid.gpt2.eval()
    hybrid.adapter.train()


def build_hybrid() -> AgnisGpt2Hybrid:
    checkpoint_path = find_agnis_checkpoint()
    hybrid = AgnisGpt2Hybrid(
        agnis_checkpoint=checkpoint_path,
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=False,
    )
    freeze_phase1(hybrid)
    return hybrid


def save_checkpoint(
    hybrid: AgnisGpt2Hybrid,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss_value: float,
    avg_loss: float,
) -> None:
    torch.save(
        {
            "step": step,
            "adapter_state": hybrid.adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss_value,
            "avg_loss": avg_loss,
            "agnis_checkpoint_path": str(hybrid.agnis_checkpoint_path),
        },
        SAVE_PATH,
    )
    print(f"[Saved] step={step} loss={avg_loss:.4f}")


def maybe_resume(
    hybrid: AgnisGpt2Hybrid,
    optimizer: torch.optim.Optimizer,
) -> int:
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        hybrid.adapter.load_state_dict(checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        print(f"Resumed from step {start_step}")
        return start_step

    print("Starting fresh Phase 1")
    return 0


@torch.no_grad()
def log_generations(
    hybrid: AgnisGpt2Hybrid,
    step: int,
) -> None:
    hybrid.eval()
    print(f"\n[Generation @ step {step}]")
    for prompt in PROMPTS:
        text = hybrid.generate(
            prompt,
            max_tokens=80,
            temperature=0.8,
            top_k=50,
        )
        print(f"[{prompt[:20]}...] -> {text}")
    print()
    hybrid.adapter.train()
    hybrid.gpt2.eval()
    hybrid.agnis_core.eval()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 expects CUDA on Kaggle. GPU was not detected.")

    save_dir = Path(SAVE_PATH).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    hybrid = build_hybrid()
    optimizer = torch.optim.AdamW(hybrid.adapter.parameters(), lr=LR)
    start_step = maybe_resume(hybrid, optimizer)

    loss_window: deque[float] = deque(maxlen=LOSS_WINDOW)
    start_time = time.time()
    tokens_seen = 0
    last_completed_step = start_step

    batch_iter = stream_token_batches(
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    if start_step > 0:
        batch_iter = itertools.islice(batch_iter, start_step, None)

    for step_offset, tokens in enumerate(batch_iter, start=1):
        step = start_step + step_offset
        if step > MAX_STEPS:
            break

        tokens = tokens.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            agnis_hidden = hybrid.compute_agnis_hidden(tokens)

        adapted = hybrid.adapter(agnis_hidden)
        # GPT-2 stays frozen, but we keep autograd enabled here so the adapter
        # still receives gradients through the GPT-2 computations.
        gpt2_base = hybrid.gpt2.transformer(inputs_embeds=adapted)
        logits = hybrid.gpt2.lm_head(gpt2_base.last_hidden_state)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hybrid.adapter.parameters(), 1.0)
        optimizer.step()

        loss_value = float(loss.item())
        loss_window.append(loss_value)
        avg_loss = sum(loss_window) / len(loss_window)
        tokens_seen += tokens.numel()
        last_completed_step = step

        if step % LOG_EVERY == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            tok_s = tokens_seen / elapsed
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step {step} | Loss {loss_value:.4f} | Avg {avg_loss:.4f} | LR {lr:.2e} | {tok_s:.0f} t/s")

        if step % GEN_EVERY == 0:
            log_generations(hybrid, step)
            if avg_loss > HIGH_LOSS_THRESHOLD:
                print("[Hint] avg loss > 4.0 after checkpoint interval. Consider LR=3e-3.")
            if avg_loss < LOW_LOSS_THRESHOLD:
                print("[Hint] avg loss < 2.0 very early. Consider adding Dropout(0.1) to the adapter if outputs overfit.")

        if step % SAVE_EVERY == 0:
            save_checkpoint(hybrid, optimizer, step, loss_value, avg_loss)

    final_avg = sum(loss_window) / len(loss_window) if loss_window else float("nan")
    final_loss = loss_window[-1] if loss_window else float("nan")
    save_checkpoint(hybrid, optimizer, min(MAX_STEPS, last_completed_step), final_loss, final_avg)
    print("Phase 1 complete.")
    print("Targets:")
    print("  Step 5000  -> avg loss < 3.5")
    print("  Step 10000 -> avg loss < 3.0")
    print("  Step 20000 -> avg loss < 2.5")
    print(f"If you hit OOM on Kaggle, reduce BATCH_SIZE from {BATCH_SIZE} to {OOM_BATCH_SIZE}.")


if __name__ == "__main__":
    main()
