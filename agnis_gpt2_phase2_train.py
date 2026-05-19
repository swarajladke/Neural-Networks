"""
agnis_gpt2_phase2_train.py — AGNIS+GPT2 Phase 2
Builds on Phase 1 adapter checkpoint.
Changes vs Phase 1:
  - Unfreeze GPT-2 last 2 transformer blocks + ln_f
  - SEQ_LEN: 32 → 64 (better context, manageable throughput)
  - LR: 3e-4 → 1e-4 (fine-tuning regime)
  - Loads Phase 1 adapter from agnis_gpt2_phase1.pt
Target: avg loss < 3.0 by S10000, < 2.5 by S20000
"""
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


DEVICE       = "cuda"
SEQ_LEN      = 64           # Phase 2: longer context (Phase 1 was 32)
BATCH_SIZE   = 16           # halved to fit longer seq on T4
LR_ADAPTER   = 1e-4         # fine-tuning regime
LR_GPT2      = 5e-5         # very conservative for frozen→unfrozen transition
MAX_STEPS    = 20_000
SAVE_EVERY   = 2_000
LOG_EVERY    = 500
GEN_EVERY    = 2_000
AGNIS_SETTLE_STEPS = 1
MODEL_NAME   = "gpt2"
GPT2_UNFREEZE_LAYERS = 2    # unfreeze last N transformer blocks

PHASE1_CKPT  = "/kaggle/working/agnis_gpt2_phase1.pt"
SAVE_PATH    = "/kaggle/working/agnis_gpt2_phase2.pt"
BEST_PATH    = "/kaggle/working/agnis_gpt2_phase2_best.pt"

PROMPTS = [
    "The history of artificial intelligence",
    "Scientists recently discovered that",
    "In the next decade, technology will",
    "The most important thing about learning is",
]
LOSS_WINDOW = 200


# ── Data ────────────────────────────────────────────────────────
def stream_token_batches(
    tokenizer: GPT2Tokenizer,
    seq_len: int,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split="train", streaming=True,
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
            yield batch[:, :-1]   # inputs (no label shift yet)


# ── Model setup ─────────────────────────────────────────────────
def build_tokenizer() -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


def build_hybrid() -> AgnisGpt2Hybrid:
    ckpt = find_agnis_checkpoint()
    hybrid = AgnisGpt2Hybrid(
        agnis_checkpoint=ckpt,
        model_name=MODEL_NAME,
        device=DEVICE,
        local_files_only=False,
        max_settle_steps=AGNIS_SETTLE_STEPS,
    )
    return hybrid


def setup_phase2(hybrid: AgnisGpt2Hybrid):
    """Freeze AGNIS, unfreeze GPT-2 last N layers + adapter."""
    # AGNIS stays frozen
    for p in hybrid.agnis_core.parameters():
        p.requires_grad_(False)
    hybrid.agnis_core.eval()

    # GPT-2: freeze all, then selectively unfreeze
    for p in hybrid.gpt2.parameters():
        p.requires_grad_(False)
    hybrid.gpt2.train()

    # Unfreeze last GPT2_UNFREEZE_LAYERS transformer blocks
    n = len(hybrid.gpt2.transformer.h)
    for block in hybrid.gpt2.transformer.h[n - GPT2_UNFREEZE_LAYERS:]:
        for p in block.parameters():
            p.requires_grad_(True)
    # Unfreeze final layer norm
    for p in hybrid.gpt2.transformer.ln_f.parameters():
        p.requires_grad_(True)

    # Adapter stays trainable
    hybrid.adapter.train()
    for p in hybrid.adapter.parameters():
        p.requires_grad_(True)

    adapter_params = sum(p.numel() for p in hybrid.adapter.parameters())
    gpt2_trainable = sum(p.numel() for p in hybrid.gpt2.parameters() if p.requires_grad)
    print(f"[Phase 2] Trainable — Adapter: {adapter_params:,} | GPT-2 (last {GPT2_UNFREEZE_LAYERS} layers): {gpt2_trainable:,}")


def load_phase1_adapter(hybrid: AgnisGpt2Hybrid) -> int:
    """Load adapter weights from Phase 1 checkpoint. Returns start step."""
    if not os.path.exists(PHASE1_CKPT):
        print(f"[Phase 2] WARNING: No Phase 1 checkpoint at {PHASE1_CKPT} — adapter starts fresh!")
        return 0
    ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    step = int(ckpt.get("step", 0))
    loss = ckpt.get("avg_loss", "?")
    print(f"[Phase 2] Loaded Phase 1 adapter — step={step} avg_loss={loss}")
    return 0   # Phase 2 always restarts step counter


def maybe_resume_phase2(hybrid: AgnisGpt2Hybrid, optimizer) -> int:
    if not os.path.exists(SAVE_PATH):
        return 0
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    # Restore GPT-2 trainable layers
    gpt2_sd = hybrid.gpt2.state_dict()
    gpt2_sd.update(ckpt.get("gpt2_trainable", {}))
    hybrid.gpt2.load_state_dict(gpt2_sd)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step = int(ckpt["step"])
    print(f"[Phase 2] Resumed from step {step}")
    return step


def save_checkpoint(hybrid, optimizer, step, avg_loss):
    torch.save({
        "step": step,
        "adapter_state": hybrid.adapter.state_dict(),
        "gpt2_trainable": {
            k: v.detach().cpu()
            for k, v in hybrid.gpt2.state_dict().items()
            if any(f"transformer.h.{i}." in k for i in range(10, 12))
            or "transformer.ln_f" in k
        },
        "optimizer_state": optimizer.state_dict(),
        "avg_loss": avg_loss,
    }, SAVE_PATH)
    print(f"[Saved] step={step} avg_loss={avg_loss:.4f}")


@torch.no_grad()
def log_generations(hybrid, step):
    hybrid.eval()
    print(f"\n── Generation @ step {step} ──")
    for prompt in PROMPTS:
        text = hybrid.generate(prompt, max_tokens=80, temperature=0.8, top_k=50)
        print(f"[{prompt[:20]}...] -> {text}")
    print()
    # Restore training modes
    hybrid.adapter.train()
    hybrid.gpt2.train()
    hybrid.agnis_core.eval()


# ── Main ────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 requires CUDA.")

    print("=" * 60)
    print("  AGNIS+GPT2 PHASE 2: Unfreeze last 2 GPT-2 layers")
    print(f"  SEQ={SEQ_LEN} | BS={BATCH_SIZE} | LR_adapter={LR_ADAPTER} | LR_gpt2={LR_GPT2}")
    print("=" * 60)

    tokenizer = build_tokenizer()
    hybrid    = build_hybrid()
    setup_phase2(hybrid)
    load_phase1_adapter(hybrid)

    optimizer = torch.optim.AdamW([
        {"params": hybrid.adapter.parameters(),                                  "lr": LR_ADAPTER},
        {"params": [p for p in hybrid.gpt2.parameters() if p.requires_grad],    "lr": LR_GPT2},
    ], weight_decay=0.01)

    start_step = maybe_resume_phase2(hybrid, optimizer)

    loss_window  = deque(maxlen=LOSS_WINDOW)
    best_loss    = float("inf")
    start_time   = time.time()
    tokens_seen  = 0
    last_step    = start_step

    batch_iter = stream_token_batches(tokenizer, SEQ_LEN, BATCH_SIZE)
    if start_step > 0:
        batch_iter = itertools.islice(batch_iter, start_step, None)

    for step_offset, tokens in enumerate(batch_iter, start=1):
        step = start_step + step_offset
        if step > MAX_STEPS:
            break

        tokens = tokens.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            agnis_hidden = hybrid.compute_agnis_hidden(tokens)
            token_embeds = hybrid.gpt2.transformer.wte(tokens)

        adapted = hybrid.adapter(agnis_hidden)
        fused   = token_embeds + adapted
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

        lv = float(loss.item())
        loss_window.append(lv)
        avg_loss    = sum(loss_window) / len(loss_window)
        tokens_seen += tokens.numel()
        last_step   = step

        if step % LOG_EVERY == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            tok_s   = tokens_seen / elapsed
            print(f"Step {step} | Loss {lv:.4f} | Avg {avg_loss:.4f} | LR {LR_ADAPTER:.1e} | {tok_s:.0f} t/s")

        if step % GEN_EVERY == 0:
            log_generations(hybrid, step)

        if step % SAVE_EVERY == 0:
            save_checkpoint(hybrid, optimizer, step, avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(hybrid.adapter.state_dict(), BEST_PATH)
                print(f"[Best] New best: {best_loss:.4f}")

    # Final save
    final_avg = sum(loss_window) / len(loss_window) if loss_window else float("nan")
    save_checkpoint(hybrid, optimizer, last_step, final_avg)
    print(f"\nPhase 2 complete. Final avg loss: {final_avg:.4f}")
    print("  < 3.0: Ready for Phase 3 (full GPT-2 fine-tune)")
    print("  < 2.5: Near GPT-2 baseline with AGNIS guidance")


if __name__ == "__main__":
    main()
