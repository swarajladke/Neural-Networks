"""
agnis_gpt2_phase5_train.py — AGNIS+GPT2 Phase 5
One session. Overnight run. Do not iterate.

Loads from Phase 4 R2 checkpoint (loss 3.39, PPL ~29.7).

Changes vs Phase 4:
  SEQ_LEN     : 128 → 256   (longer context)
  BATCH_SIZE  : 4   → 4 + ACCUM=2 (effective 8, safer for 256 seq)
  LR adapter  : 5e-5 → 2e-4
  LR GPT-2    : 3e-5 → 1e-4
  Scheduler   : cosine → CosineAnnealingWarmRestarts (T0=5000, Tmult=2)
  GPT-2 layers: ALL → last 6  (focus + forgetting protection at high LR)
  MAX_STEPS   : 20000 → 15000

SUCCESS CRITERION : avg_loss < 3.0 by step 10000
HARD STOP         : if avg_loss > 3.2 at step 5000 → stop, move to adapter fix
ONE SESSION ONLY  : if 3.0 not broken in this run, accept 3.39 and move on
"""
from __future__ import annotations

import itertools, os, time
from collections import deque
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datasets import load_dataset
from transformers import GPT2Tokenizer

from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint


# ── Config ────────────────────────────────────────────────────────
DEVICE      = "cuda"
SEQ_LEN     = 256
BATCH_SIZE  = 4
GRAD_ACCUM  = 2          # effective batch = 8
LR_ADAPTER  = 2e-4
LR_GPT2     = 1e-4
LR_MIN      = 1e-6
T_0         = 5_000      # warm restart period
T_MULT      = 2          # cycle length multiplier
MAX_STEPS   = 15_000
LOG_EVERY   = 500
GEN_EVERY   = 2_000
SAVE_EVERY  = 2_000
LOSS_WINDOW = 200
GPT2_LAYERS = 6          # unfreeze last 6 GPT-2 transformer layers
AGNIS_SETTLE_STEPS = 1
MODEL_NAME  = "gpt2"

# Stop condition: if avg_loss > this at STOP_CHECK_STEP → exit early
STOP_CHECK_STEP = 5_000
STOP_THRESHOLD  = 3.2

PHASE4_BEST = "/kaggle/working/agnis_gpt2_phase4_best.pt"
SAVE_PATH   = "/kaggle/working/agnis_gpt2_phase5.pt"
BEST_PATH   = "/kaggle/working/agnis_gpt2_phase5_best.pt"

PROMPTS = [
    "The history of artificial intelligence",
    "Scientists recently discovered that",
    "In the next decade, technology will",
    "The most important thing about learning is",
]


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
        if len(text) < 128:          # slightly longer min for SEQ_LEN=256
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


def setup_phase5(hybrid: AgnisGpt2Hybrid):
    """Freeze AGNIS + first (12-GPT2_LAYERS) GPT-2 blocks. Unfreeze last 6."""
    # AGNIS: fully frozen
    for p in hybrid.agnis_core.parameters():
        p.requires_grad_(False)
    hybrid.agnis_core.eval()

    # GPT-2: freeze everything first
    for p in hybrid.gpt2.parameters():
        p.requires_grad_(False)

    # Unfreeze last GPT2_LAYERS transformer blocks
    n_layers = len(hybrid.gpt2.transformer.h)
    for i in range(n_layers - GPT2_LAYERS, n_layers):
        for p in hybrid.gpt2.transformer.h[i].parameters():
            p.requires_grad_(True)

    # Always unfreeze lm_head and final ln_f
    for p in hybrid.gpt2.lm_head.parameters():
        p.requires_grad_(True)
    for p in hybrid.gpt2.transformer.ln_f.parameters():
        p.requires_grad_(True)

    hybrid.gpt2.train()

    # Adapter: fully trainable
    hybrid.adapter.train()
    for p in hybrid.adapter.parameters():
        p.requires_grad_(True)

    adapter_p = sum(p.numel() for p in hybrid.adapter.parameters())
    gpt2_p    = sum(p.numel() for p in hybrid.gpt2.parameters() if p.requires_grad)
    print(f"[Phase 5] Trainable — Adapter: {adapter_p:,} | GPT-2 (last {GPT2_LAYERS} layers): {gpt2_p:,}")


def load_phase4_checkpoint(hybrid: AgnisGpt2Hybrid):
    """Load Phase 4 R2 best checkpoint."""
    if not os.path.exists(PHASE4_BEST):
        print(f"[Phase 5] WARNING: Phase 4 best not found at {PHASE4_BEST} — starting from scratch!")
        return
    ckpt = torch.load(PHASE4_BEST, map_location=DEVICE)
    hybrid.adapter.load_state_dict(ckpt["adapter_state"])
    # phase4 saves gpt2_state (full GPT-2)
    gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
    if gpt2_key in ckpt:
        gpt2_sd = hybrid.gpt2.state_dict()
        gpt2_sd.update(ckpt[gpt2_key])
        hybrid.gpt2.load_state_dict(gpt2_sd)
    print(f"[Phase 5] Loaded Phase 4 R2 checkpoint | loss={ckpt.get('avg_loss', '?'):.4f}")


def save_checkpoint(hybrid, step, avg_loss, is_best=False):
    payload = {
        "step": step,
        "adapter_state": hybrid.adapter.state_dict(),
        "gpt2_state": {k: v.detach().cpu() for k, v in hybrid.gpt2.state_dict().items()},
        "avg_loss": avg_loss,
    }
    torch.save(payload, SAVE_PATH)
    if is_best:
        torch.save(payload, BEST_PATH)
        print(f"[Best] New best: {avg_loss:.4f}")
    print(f"[Saved] step={step} avg_loss={avg_loss:.4f}")


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


# ── Main ──────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5 requires CUDA.")

    print("=" * 60)
    print("  AGNIS+GPT2 PHASE 5: One session. Overnight.")
    print(f"  SEQ={SEQ_LEN} | BS={BATCH_SIZE} | ACCUM={GRAD_ACCUM} | EffBS={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  LR_adapter={LR_ADAPTER} | LR_gpt2={LR_GPT2}→{LR_MIN} (warm restarts)")
    print(f"  GPT-2 layers unfrozen: last {GPT2_LAYERS} | MAX_STEPS={MAX_STEPS}")
    print(f"  STOP IF loss > {STOP_THRESHOLD} at step {STOP_CHECK_STEP}")
    print(f"  SUCCESS: loss < 3.0 by step 10000")
    print("=" * 60)

    tokenizer   = build_tokenizer()
    hybrid      = build_hybrid()
    setup_phase5(hybrid)
    load_phase4_checkpoint(hybrid)

    optimizer = torch.optim.AdamW([
        {"params": hybrid.adapter.parameters(),
         "lr": LR_ADAPTER},
        {"params": [p for p in hybrid.gpt2.parameters() if p.requires_grad],
         "lr": LR_GPT2},
    ], weight_decay=0.01)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=LR_MIN
    )

    loss_window = deque(maxlen=LOSS_WINDOW)
    best_loss   = float("inf")
    start_time  = time.time()
    tokens_seen = 0
    accum_loss  = 0.0
    micro_step  = 0

    optimizer.zero_grad(set_to_none=True)

    for tokens in stream_token_batches(tokenizer, SEQ_LEN, BATCH_SIZE):
        micro_step += 1
        step = (micro_step - 1) // GRAD_ACCUM + 1
        if step > MAX_STEPS:
            break

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

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in hybrid.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step(step)
            optimizer.zero_grad(set_to_none=True)

            step_loss  = accum_loss
            accum_loss = 0.0
            loss_window.append(step_loss)
            avg_loss = sum(loss_window) / len(loss_window)

            cur_lr = optimizer.param_groups[1]["lr"]

            if step % LOG_EVERY == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                tok_s   = tokens_seen / elapsed
                print(f"Step {step} | Loss {step_loss:.4f} | Avg {avg_loss:.4f} | LR {cur_lr:.2e} | {tok_s:.0f} t/s")

            if step % GEN_EVERY == 0:
                log_generations(hybrid, step)

            if step % SAVE_EVERY == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                save_checkpoint(hybrid, step, avg_loss, is_best)

            # ── Hard stop at step 5000 ──────────────────────────
            if step == STOP_CHECK_STEP:
                print(f"\n{'='*60}")
                print(f"  CHECKPOINT @ step {STOP_CHECK_STEP}: avg_loss = {avg_loss:.4f}")
                if avg_loss > STOP_THRESHOLD:
                    print(f"  STOP: loss {avg_loss:.4f} > {STOP_THRESHOLD} threshold.")
                    print(f"  Phase 5 is not helping. Accept 3.39. Move to adapter fix.")
                    print(f"{'='*60}\n")
                    break
                else:
                    print(f"  GO: loss {avg_loss:.4f} ≤ {STOP_THRESHOLD}. Continuing to step 15000.")
                    print(f"{'='*60}\n")

            # ── Success early exit ──────────────────────────────
            if step >= 10_000 and avg_loss < 3.0:
                print(f"\n🎯 SUCCESS: avg_loss {avg_loss:.4f} < 3.0 at step {step}!")
                save_checkpoint(hybrid, step, avg_loss, is_best=True)
                break

    final_avg = sum(loss_window) / len(loss_window) if loss_window else float("nan")
    save_checkpoint(hybrid, step, final_avg, final_avg < best_loss)

    import math
    ppl = math.exp(final_avg)
    print(f"\nPhase 5 complete.")
    print(f"  Final avg loss : {final_avg:.4f}")
    print(f"  Perplexity     : {ppl:.1f}")
    print(f"  GPT-2 Small    : PPL ~35   | You: {ppl:.1f} {'✅ BEATS' if ppl < 35 else '❌'}")
    print(f"  GPT-2 Medium   : PPL ~26   | You: {ppl:.1f} {'✅ BEATS' if ppl < 26 else '❌'}")
    print(f"  GPT-2 Large    : PPL ~22   | You: {ppl:.1f} {'✅ BEATS' if ppl < 22 else '❌'}")
    if final_avg < 3.0:
        print(f"\n  🎯 Broke 3.0 — move to adapter fix with strong language baseline.")
    else:
        print(f"\n  Accept {final_avg:.4f} as language quality result.")
        print(f"  NEXT: adapter fix — that's the research contribution.")


if __name__ == "__main__":
    main()
