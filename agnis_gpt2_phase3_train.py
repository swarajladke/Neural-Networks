"""
agnis_gpt2_phase3_train.py — AGNIS+GPT2 Phase 3
Builds on Phase 2 checkpoint.
Changes vs Phase 2:
  - Unfreeze last 4 GPT-2 layers (was 2)
  - SEQ_LEN: 64 → 128 (full context)
  - BATCH_SIZE: 16 → 8 (to fit SEQ_LEN=128 on T4)
  - LR_GPT2: 5e-5 → 1e-4 (more aggressive)
  - Cosine LR schedule
  - Loads Phase 2 checkpoint (adapter + GPT-2 last 2 layers)
Target: avg loss < 3.0 by S10000, < 2.5 by S20000
"""
from __future__ import annotations

import itertools
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterator
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer

from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint


DEVICE             = "cuda"
SEQ_LEN            = 128        # full context
BATCH_SIZE         = 8          # halved again to fit SEQ_LEN=128
LR_ADAPTER         = 1e-4
LR_GPT2            = 1e-4       # 2x Phase 2
MAX_STEPS          = 20_000
SAVE_EVERY         = 2_000
LOG_EVERY          = 500
GEN_EVERY          = 2_000
AGNIS_SETTLE_STEPS = 1
MODEL_NAME         = "gpt2"
GPT2_UNFREEZE_LAYERS = 4        # was 2 in Phase 2

PHASE2_CKPT = "/kaggle/working/agnis_gpt2_phase2.pt"
SAVE_PATH   = "/kaggle/working/agnis_gpt2_phase3.pt"
BEST_PATH   = "/kaggle/working/agnis_gpt2_phase3_best.pt"

PROMPTS = [
    "The history of artificial intelligence",
    "Scientists recently discovered that",
    "In the next decade, technology will",
    "The most important thing about learning is",
]
LOSS_WINDOW = 200


# ── Data ────────────────────────────────────────────────────────
def stream_token_batches(tokenizer, seq_len, batch_size) -> Iterator[torch.Tensor]:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split="train", streaming=True,
    )
    chunk_len = seq_len + 1
    buf_needed = chunk_len * batch_size
    token_buffer: list[int] = []

    for row in dataset:
        text = row.get("text", "").strip()
        if len(text) < 64:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < chunk_len:
            continue
        token_buffer.extend(ids)
        while len(token_buffer) >= buf_needed:
            cur = token_buffer[:buf_needed]
            del token_buffer[:buf_needed]
            batch = torch.tensor(cur, dtype=torch.long).view(batch_size, chunk_len)
            yield batch[:, :-1]


# ── Model ────────────────────────────────────────────────────────
def build_tokenizer():
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


def build_hybrid():
    ckpt = find_agnis_checkpoint()
    return AgnisGpt2Hybrid(
        agnis_checkpoint=ckpt,
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=False,
        max_settle_steps=AGNIS_SETTLE_STEPS,
    )


def setup_phase3(hybrid: AgnisGpt2Hybrid):
    # AGNIS frozen
    for p in hybrid.agnis_core.parameters():
        p.requires_grad_(False)
    hybrid.agnis_core.eval()

    # GPT-2: freeze all, then unfreeze last GPT2_UNFREEZE_LAYERS + ln_f
    for p in hybrid.gpt2.parameters():
        p.requires_grad_(False)
    hybrid.gpt2.train()

    n = len(hybrid.gpt2.transformer.h)
    for block in hybrid.gpt2.transformer.h[n - GPT2_UNFREEZE_LAYERS:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in hybrid.gpt2.transformer.ln_f.parameters():
        p.requires_grad_(True)

    hybrid.adapter.train()
    for p in hybrid.adapter.parameters():
        p.requires_grad_(True)

    adapter_p = sum(p.numel() for p in hybrid.adapter.parameters())
    gpt2_p    = sum(p.numel() for p in hybrid.gpt2.parameters() if p.requires_grad)
    print(f"[Phase 3] Trainable — Adapter: {adapter_p:,} | GPT-2 (last {GPT2_UNFREEZE_LAYERS} layers): {gpt2_p:,}")


def load_phase2_checkpoint(hybrid: AgnisGpt2Hybrid) -> None:
    if not os.path.exists(PHASE2_CKPT):
        print(f"[Phase 3] WARNING: No Phase 2 checkpoint at {PHASE2_CKPT} — adapter starts fresh!")
        return
    ckpt = torch.load(PHASE2_CKPT, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    # Restore Phase 2's trained GPT-2 layers (last 2) if present
    if "gpt2_trainable" in ckpt:
        gpt2_sd = hybrid.gpt2.state_dict()
        gpt2_sd.update(ckpt["gpt2_trainable"])
        hybrid.gpt2.load_state_dict(gpt2_sd)
        print(f"[Phase 3] Loaded Phase 2 adapter + GPT-2 last 2 layers | loss={ckpt.get('avg_loss', '?'):.4f}")
    else:
        print(f"[Phase 3] Loaded Phase 2 adapter only | loss={ckpt.get('avg_loss', '?'):.4f}")


def maybe_resume(hybrid, optimizer) -> int:
    if not os.path.exists(SAVE_PATH):
        return 0
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    gpt2_sd = hybrid.gpt2.state_dict()
    gpt2_sd.update(ckpt.get("gpt2_trainable", {}))
    hybrid.gpt2.load_state_dict(gpt2_sd)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step = int(ckpt["step"])
    print(f"[Phase 3] Resumed from step {step}")
    return step


def save_checkpoint(hybrid, optimizer, step, avg_loss, is_best=False):
    n = len(hybrid.gpt2.transformer.h)
    trainable_keys = [
        k for k in hybrid.gpt2.state_dict()
        if any(f"transformer.h.{i}." in k for i in range(n - GPT2_UNFREEZE_LAYERS, n))
        or "transformer.ln_f" in k
    ]
    torch.save({
        "step": step,
        "adapter_state": hybrid.adapter.state_dict(),
        "gpt2_trainable": {k: v.detach().cpu() for k, v in hybrid.gpt2.state_dict().items() if k in trainable_keys},
        "optimizer_state": optimizer.state_dict(),
        "avg_loss": avg_loss,
    }, SAVE_PATH)
    print(f"[Saved] step={step} avg_loss={avg_loss:.4f}")
    if is_best:
        torch.save({
            "step": step,
            "adapter_state": hybrid.adapter.state_dict(),
            "gpt2_trainable": {k: v.detach().cpu() for k, v in hybrid.gpt2.state_dict().items() if k in trainable_keys},
            "avg_loss": avg_loss,
        }, BEST_PATH)
        print(f"[Best] New best: {avg_loss:.4f}")


@torch.no_grad()
def log_generations(hybrid, step):
    hybrid.eval()
    print(f"\n── Generation @ step {step} ──")
    for prompt in PROMPTS:
        text = hybrid.generate(prompt, max_tokens=80, temperature=0.8, top_k=50)
        print(f"[{prompt[:20]}...] -> {text}")
    print()
    hybrid.adapter.train()
    hybrid.gpt2.train()
    hybrid.agnis_core.eval()


def cosine_lr(step, max_steps, lr_min=1e-5, lr_max=1e-4):
    """Cosine decay from lr_max to lr_min."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / max_steps))


# ── Main ────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 requires CUDA.")

    print("=" * 60)
    print("  AGNIS+GPT2 PHASE 3: Unfreeze last 4 GPT-2 layers")
    print(f"  SEQ={SEQ_LEN} | BS={BATCH_SIZE} | LR={LR_GPT2} (cosine) | Layers={GPT2_UNFREEZE_LAYERS}")
    print("=" * 60)

    tokenizer = build_tokenizer()
    hybrid    = build_hybrid()
    setup_phase3(hybrid)
    load_phase2_checkpoint(hybrid)

    optimizer = torch.optim.AdamW([
        {"params": hybrid.adapter.parameters(),                               "lr": LR_ADAPTER},
        {"params": [p for p in hybrid.gpt2.parameters() if p.requires_grad], "lr": LR_GPT2},
    ], weight_decay=0.01)

    start_step = maybe_resume(hybrid, optimizer)

    loss_window = deque(maxlen=LOSS_WINDOW)
    best_loss   = float("inf")
    start_time  = time.time()
    tokens_seen = 0
    last_step   = start_step

    batch_iter = stream_token_batches(tokenizer, SEQ_LEN, BATCH_SIZE)
    if start_step > 0:
        batch_iter = itertools.islice(batch_iter, start_step, None)

    for step_offset, tokens in enumerate(batch_iter, start=1):
        step = start_step + step_offset
        if step > MAX_STEPS:
            break

        # Cosine LR schedule
        cur_lr = cosine_lr(step, MAX_STEPS, lr_min=1e-5, lr_max=LR_GPT2)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        tokens = tokens.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            agnis_hidden = hybrid.compute_agnis_hidden(tokens)
            token_embeds = hybrid.gpt2.transformer.wte(tokens)

        adapted  = hybrid.adapter(agnis_hidden)
        fused    = token_embeds + adapted
        gpt2_out = hybrid.gpt2.transformer(inputs_embeds=fused)
        logits   = hybrid.gpt2.lm_head(gpt2_out.last_hidden_state)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in hybrid.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        lv          = float(loss.item())
        loss_window.append(lv)
        avg_loss    = sum(loss_window) / len(loss_window)
        tokens_seen += tokens.numel()
        last_step   = step

        if step % LOG_EVERY == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            tok_s   = tokens_seen / elapsed
            print(f"Step {step} | Loss {lv:.4f} | Avg {avg_loss:.4f} | LR {cur_lr:.2e} | {tok_s:.0f} t/s")

        if step % GEN_EVERY == 0:
            log_generations(hybrid, step)

        if step % SAVE_EVERY == 0:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            save_checkpoint(hybrid, optimizer, step, avg_loss, is_best)

    final_avg = sum(loss_window) / len(loss_window) if loss_window else float("nan")
    save_checkpoint(hybrid, optimizer, last_step, final_avg, final_avg < best_loss)
    print(f"\nPhase 3 complete. Final avg loss: {final_avg:.4f}")
    print("  < 3.0: Ready for deployment / further fine-tune")
    print("  < 2.5: Near GPT-2 baseline with AGNIS guidance ← goal")


if __name__ == "__main__":
    main()
