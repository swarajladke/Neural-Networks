"""
agnis_gpt2_phase4_train.py — AGNIS+GPT2 Phase 4
Builds on Phase 3 checkpoint.
Changes vs Phase 3:
  - Unfreeze ALL 12 GPT-2 layers (was 4)
  - LR_GPT2: 1e-4 → 3e-5 (very careful — catastrophic forgetting risk)
  - BATCH_SIZE: 8 → 4 + gradient accumulation (effective BS=8)
  - Cosine LR: 3e-5 → 5e-6
  - Loads Phase 3 adapter + GPT-2 last 4 layers
Target: avg loss < 3.0 by S10000, < 2.7 by S20000
"""
from __future__ import annotations

import itertools
import os
import time
import math
from collections import deque
from typing import Iterator

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer

from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint


DEVICE           = "cuda"
SEQ_LEN          = 128
BATCH_SIZE       = 4          # smaller to fit all 12 layers
GRAD_ACCUM       = 2          # effective batch = 4×2 = 8
LR_ADAPTER       = 5e-5
LR_GPT2          = 3e-5       # very low — full unfreeze is sensitive
LR_MIN           = 5e-6       # cosine floor
MAX_STEPS        = 20_000
SAVE_EVERY       = 2_000
LOG_EVERY        = 500
GEN_EVERY        = 2_000
AGNIS_SETTLE_STEPS = 1
MODEL_NAME       = "gpt2"

PHASE3_CKPT = "/kaggle/working/agnis_gpt2_phase3_best.pt"
SAVE_PATH   = "/kaggle/working/agnis_gpt2_phase4.pt"
BEST_PATH   = "/kaggle/working/agnis_gpt2_phase4_best.pt"

PROMPTS = [
    "The history of artificial intelligence",
    "Scientists recently discovered that",
    "In the next decade, technology will",
    "The most important thing about learning is",
]
LOSS_WINDOW = 200


# ── Data ─────────────────────────────────────────────────────────
def stream_token_batches(tokenizer, seq_len, batch_size) -> Iterator[torch.Tensor]:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split="train", streaming=True,
    )
    chunk_len  = seq_len + 1
    buf_needed = chunk_len * batch_size
    buffer: list[int] = []

    for row in dataset:
        text = row.get("text", "").strip()
        if len(text) < 64:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < chunk_len:
            continue
        buffer.extend(ids)
        while len(buffer) >= buf_needed:
            cur = buffer[:buf_needed]
            del buffer[:buf_needed]
            batch = torch.tensor(cur, dtype=torch.long).view(batch_size, chunk_len)
            yield batch[:, :-1]


# ── Model ─────────────────────────────────────────────────────────
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


def setup_phase4(hybrid: AgnisGpt2Hybrid):
    """Freeze AGNIS. Unfreeze ALL GPT-2 layers + adapter."""
    # AGNIS fully frozen
    for p in hybrid.agnis_core.parameters():
        p.requires_grad_(False)
    hybrid.agnis_core.eval()

    # Unfreeze ALL GPT-2 parameters
    hybrid.gpt2.train()
    for p in hybrid.gpt2.parameters():
        p.requires_grad_(True)

    # Adapter trainable
    hybrid.adapter.train()
    for p in hybrid.adapter.parameters():
        p.requires_grad_(True)

    adapter_p = sum(p.numel() for p in hybrid.adapter.parameters())
    gpt2_p    = sum(p.numel() for p in hybrid.gpt2.parameters() if p.requires_grad)
    print(f"[Phase 4] Trainable — Adapter: {adapter_p:,} | GPT-2 (ALL layers): {gpt2_p:,}")


def load_phase3_checkpoint(hybrid: AgnisGpt2Hybrid):
    if not os.path.exists(PHASE3_CKPT):
        print(f"[Phase 4] WARNING: No Phase 3 checkpoint at {PHASE3_CKPT} — starting fresh!")
        return
    ckpt = torch.load(PHASE3_CKPT, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    if "gpt2_trainable" in ckpt:
        gpt2_sd = hybrid.gpt2.state_dict()
        gpt2_sd.update(ckpt["gpt2_trainable"])
        hybrid.gpt2.load_state_dict(gpt2_sd)
        print(f"[Phase 4] Loaded Phase 3 checkpoint | loss={ckpt.get('avg_loss', '?'):.4f}")
    else:
        print(f"[Phase 4] Loaded Phase 3 adapter only")


def maybe_resume(hybrid, optimizer) -> int:
    if not os.path.exists(SAVE_PATH):
        return 0
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    gpt2_sd = hybrid.gpt2.state_dict()
    gpt2_sd.update(ckpt.get("gpt2_state", {}))
    hybrid.gpt2.load_state_dict(gpt2_sd)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step = int(ckpt["step"])
    print(f"[Phase 4] Resumed from step {step}")
    return step


def save_checkpoint(hybrid, optimizer, step, avg_loss, is_best=False):
    payload = {
        "step": step,
        "adapter_state": hybrid.adapter.state_dict(),
        "gpt2_state": {k: v.detach().cpu() for k, v in hybrid.gpt2.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "avg_loss": avg_loss,
    }
    torch.save(payload, SAVE_PATH)
    print(f"[Saved] step={step} avg_loss={avg_loss:.4f}")
    if is_best:
        # Best checkpoint: no optimizer state (smaller file)
        torch.save({k: v for k, v in payload.items() if k != "optimizer_state"}, BEST_PATH)
        print(f"[Best]  New best: {avg_loss:.4f}")


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


def cosine_lr(step, max_steps, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / max_steps))


# ── Main ──────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4 requires CUDA.")

    print("=" * 60)
    print("  AGNIS+GPT2 PHASE 4: ALL GPT-2 layers unfrozen")
    print(f"  SEQ={SEQ_LEN} | BS={BATCH_SIZE} | ACCUM={GRAD_ACCUM} | EffBS={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  LR_adapter={LR_ADAPTER} | LR_gpt2={LR_GPT2}→{LR_MIN} (cosine)")
    print("=" * 60)

    tokenizer = build_tokenizer()
    hybrid    = build_hybrid()
    setup_phase4(hybrid)
    load_phase3_checkpoint(hybrid)

    optimizer = torch.optim.AdamW([
        {"params": hybrid.adapter.parameters(),  "lr": LR_ADAPTER},
        {"params": hybrid.gpt2.parameters(),     "lr": LR_GPT2},
    ], weight_decay=0.01)

    start_step = maybe_resume(hybrid, optimizer)

    loss_window  = deque(maxlen=LOSS_WINDOW)
    best_loss    = float("inf")
    start_time   = time.time()
    tokens_seen  = 0
    last_step    = start_step
    accum_loss   = 0.0

    batch_iter = stream_token_batches(tokenizer, SEQ_LEN, BATCH_SIZE)
    if start_step > 0:
        batch_iter = itertools.islice(batch_iter, start_step * GRAD_ACCUM, None)

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    for tokens in batch_iter:
        micro_step += 1
        step = start_step + (micro_step - 1) // GRAD_ACCUM + 1
        if step > MAX_STEPS:
            break

        # Cosine LR
        cur_lr_gpt2    = cosine_lr(step, MAX_STEPS, LR_GPT2,    LR_MIN)
        cur_lr_adapter = cosine_lr(step, MAX_STEPS, LR_ADAPTER,  LR_MIN)
        optimizer.param_groups[0]["lr"] = cur_lr_adapter
        optimizer.param_groups[1]["lr"] = cur_lr_gpt2

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
        ) / GRAD_ACCUM

        loss.backward()
        accum_loss += float(loss.item())
        tokens_seen += tokens.numel()

        # Optimizer step every GRAD_ACCUM micro-steps
        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in hybrid.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step_loss = accum_loss
            accum_loss = 0.0
            loss_window.append(step_loss)
            avg_loss = sum(loss_window) / len(loss_window)
            last_step = step

            if step % LOG_EVERY == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                tok_s   = tokens_seen / elapsed
                print(f"Step {step} | Loss {step_loss:.4f} | Avg {avg_loss:.4f} | LR {cur_lr_gpt2:.2e} | {tok_s:.0f} t/s")

            if step % GEN_EVERY == 0:
                log_generations(hybrid, step)

            if step % SAVE_EVERY == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                save_checkpoint(hybrid, optimizer, step, avg_loss, is_best)

    final_avg = sum(loss_window) / len(loss_window) if loss_window else float("nan")
    save_checkpoint(hybrid, optimizer, last_step, final_avg, final_avg < best_loss)
    print(f"\nPhase 4 complete. Final avg loss: {final_avg:.4f}")
    print("  < 3.0: AGNIS hybrid matches GPT-2 fine-tuned baseline")
    print("  < 2.7: AGNIS hybrid BEATS plain GPT-2 fine-tuning ← goal")


if __name__ == "__main__":
    main()
