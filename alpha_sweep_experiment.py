"""
alpha_sweep_experiment.py — Controlled α Sweep for Temporal Stability
=====================================================================
Tests exactly 3 values of recurrent influence (α) on the FROZEN Marathon core.

  x_t = tanh(core_t + α * R @ h_{t-1})

Hippocampal buffer: OFF
New modules: NONE  
Goal: Find optimal α that breaks PPL < 100

Runs:
  α = 0.0  (pure Markov baseline — no recurrence)
  α = 0.1  (low temporal influence)
  α = 0.3  (moderate temporal influence)
"""

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from agnis_v4_core import PredictiveHierarchy

# ─── Config ───────────────────────────────────────────────────────
CORE_CHECKPOINT = "agnis_marathon_final.pt"
TOKENIZER_PATH = "slm_bpe_tokenizer_en.json"
CORPUS_PATH = "slm/input_en_massive.txt"
TARGET_CHARS = 25_000_000

BATCH_SIZE = 64
EPOCHS = 10          # 10 epochs per α (enough to see convergence)
LR = 3e-4
WARMUP_STEPS = 500
LOG_EVERY = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA_VALUES = [0.0, 0.1, 0.3]

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]


# ─── Minimal Temporal Model (α-controlled, no hippocampal buffer) ─
class AlphaTemporalModel(nn.Module):
    """
    Frozen Marathon core + fixed-α recurrence + fusion MLP.
    NO hippocampal buffer. NO learned gate. Just raw α control.
    """
    def __init__(self, vocab_size: int, embed_dim: int, alpha: float,
                 dropout: float = 0.15, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Frozen core
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, 1024, embed_dim], device=device
        )

        # R-matrix (learnable recurrent binding)
        self.R = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        nn.init.orthogonal_(self.R.weight, gain=0.1)  # Start small
        self.r_norm = nn.LayerNorm(embed_dim).to(self.device)

        # Persistent state
        self.register_buffer("h_prev", torch.zeros(1, embed_dim))

        # Fusion MLP: [emb, h_t, emb*h_t, emb-h_t] = 4 * embed_dim
        hidden_dim = embed_dim * 4
        self.fusion_norm = nn.LayerNorm(embed_dim * 4).to(self.device)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ).to(self.device)
        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def freeze_core(self):
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

    def reset_states(self, batch_size: int = 1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.lm_head.weight.device)

    def step_logits(self, token_ids: torch.Tensor, update_temporal: bool = True) -> torch.Tensor:
        # 1. Embed
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 2. Frozen core prediction
        with torch.no_grad():
            core = self.hierarchy.predict_label(emb, max_steps=1, update_temporal=update_temporal)
        if core.shape[1] > self.embed_dim:
            core = core[:, :self.embed_dim]
        elif core.shape[1] < self.embed_dim:
            pad = torch.zeros(core.shape[0], self.embed_dim - core.shape[1], device=core.device)
            core = torch.cat([core, pad], dim=1)
        core = F.normalize(core, dim=-1)

        # 3. Temporal blend: h_t = tanh(core + α * R(h_prev))
        if self.alpha > 0.0:
            temporal = self.R(self.h_prev.detach())
            h_t = torch.tanh(core + self.alpha * temporal)
            h_t = self.r_norm(h_t)
        else:
            h_t = core

        # Shift state
        self.h_prev = h_t.detach()

        # 4. Fusion
        fused = torch.cat([emb, h_t, emb * h_t, emb - h_t], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.proj(fused)
        fused = self.out_norm(fused + emb)

        return self.lm_head(fused)


# ─── Utilities ────────────────────────────────────────────────────
def load_corpus() -> str:
    if not os.path.exists(CORPUS_PATH):
        print(f"[ERROR] Corpus not found: {CORPUS_PATH}")
        sys.exit(1)
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text


def build_token_tensor(ids, batch_size, device):
    seq_len = len(ids) // batch_size
    ids = ids[:seq_len * batch_size]
    return torch.tensor(ids, dtype=torch.long, device=device).view(batch_size, seq_len)


@torch.no_grad()
def heldout_ppl(model, tokens, steps=512):
    model.eval()
    model.reset_states(batch_size=tokens.shape[0])
    total = 0.0
    n = min(steps, tokens.shape[1] - 1)
    for s in range(n):
        logits = model.step_logits(tokens[:, s], update_temporal=True)
        loss = F.cross_entropy(logits, tokens[:, s + 1])
        total += loss.item()
    avg = total / max(1, n)
    return avg, math.exp(min(avg, 20))


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=60, temperature=0.8, top_k=40):
    model.eval()
    enc = tokenizer.encode(prompt)
    ids = enc.ids if enc.ids else [0]
    generated = list(ids)
    model.reset_states(batch_size=1)
    dev = model.lm_head.weight.device

    for tid in ids:
        tok = torch.tensor([tid], dtype=torch.long, device=dev)
        _ = model.step_logits(tok, update_temporal=True)

    for _ in range(max_tokens):
        cur = torch.tensor([generated[-1]], dtype=torch.long, device=dev)
        logits = model.step_logits(cur, update_temporal=True)[0]
        logits = logits / max(temperature, 1e-5)

        # Repetition penalty
        for tok in set(generated[-20:]):
            if logits[tok] > 0:
                logits[tok] /= 1.2
            else:
                logits[tok] *= 1.2

        if top_k > 0 and top_k < logits.numel():
            vals, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < vals[-1], torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)

    return tokenizer.decode(generated)


# ─── Main Experiment ──────────────────────────────────────────────
def run_single_alpha(alpha: float, tokenizer, train_tokens, valid_tokens, vocab_size, embed_dim):
    print(f"\n{'='*60}")
    print(f"  ALPHA = {alpha}")
    print(f"{'='*60}")

    model = AlphaTemporalModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        alpha=alpha,
        device=DEVICE,
    )
    model.hierarchy.load_checkpoint(CORE_CHECKPOINT)

    # Rebuild if embed_dim changed
    detected = model.hierarchy.layers[0].input_dim
    if detected != embed_dim:
        embed_dim = detected
        model = AlphaTemporalModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            alpha=alpha,
            device=DEVICE,
        )
        model.hierarchy.load_checkpoint(CORE_CHECKPOINT)

    model.freeze_core()
    model.to(model.device)

    # Trainable: everything except frozen core
    trainable = [
        *model.embedding.parameters(),
        *model.R.parameters(),
        *model.r_norm.parameters(),
        *model.fusion_norm.parameters(),
        *model.proj.parameters(),
        *model.out_norm.parameters(),
    ]
    n_params = sum(p.numel() for p in trainable)
    print(f"  [Trainable] {n_params:,} params | α={alpha}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    best_val = float("inf")
    results = []

    for epoch in range(EPOCHS):
        model.train()
        model.reset_states(batch_size=BATCH_SIZE)
        epoch_loss = 0.0
        start = time.time()

        for step in range(train_tokens.shape[1] - 1):
            cur = train_tokens[:, step]
            tgt = train_tokens[:, step + 1]

            if epoch == 0 and step <= WARMUP_STEPS:
                scale = max(0.01, step / max(1, WARMUP_STEPS))
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * scale

            logits = model.step_logits(cur, update_temporal=True)
            loss = F.cross_entropy(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            model.lm_head.weight = model.embedding.weight

            epoch_loss += loss.item()

            if (step + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                speed = (step + 1) / max(time.time() - start, 1e-6)
                print(f"  α={alpha} | Ep {epoch+1:>2}/{EPOCHS} | Step {step+1:>6} | Loss {avg:.4f} | PPL {ppl:.1f} | {speed:.0f} tok/s", end="\r", flush=True)

        train_loss = epoch_loss / max(1, train_tokens.shape[1] - 1)
        train_ppl = math.exp(min(train_loss, 20))
        val_loss, val_ppl = heldout_ppl(model, valid_tokens)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        results.append({
            "epoch": epoch + 1,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "best": improved,
        })

        tag = " <- best" if improved else ""
        print(f"\n  α={alpha} | Epoch {epoch+1:>2}/{EPOCHS} | Train PPL {train_ppl:.1f} | Val PPL {val_ppl:.1f}{tag}")

    # Final generation samples
    print(f"\n  --- Final Samples (α={alpha}) ---")
    for prompt in PROMPTS:
        text = generate_sample(model, tokenizer, prompt)
        print(f"  [{prompt}] -> {text[:150]}...")

    final_val_ppl = min(r["val_ppl"] for r in results)
    print(f"\n  ★ α={alpha} | Best Val PPL = {final_val_ppl:.1f}")
    return final_val_ppl, results


def main():
    print("\n" + "=" * 60)
    print("  ALPHA SWEEP EXPERIMENT")
    print("  Frozen Core | Fixed α | No Hippocampal Buffer")
    print("=" * 60)

    text = load_corpus()

    if not os.path.exists(TOKENIZER_PATH):
        # Build tokenizer (same as canonical)
        from tokenizers import decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        print(f"[Tokenizer] Training BPE tokenizer (vocab=4096)...")
        tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
        tokenizer.train_from_iterator([text], trainer=trainer)
        tokenizer.save(TOKENIZER_PATH)
        print(f"[Tokenizer] Saved to {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] Loaded {TOKENIZER_PATH} | vocab={vocab_size}")

    enc = tokenizer.encode(text)
    token_tensor = build_token_tensor(enc.ids, BATCH_SIZE, DEVICE)
    split = max(1024, token_tensor.shape[1] // 20)
    train_tokens = token_tensor[:, :-split]
    valid_tokens = token_tensor[:, -split:]
    print(f"[Tokenize] {len(enc.ids):,} tokens")
    print(f"[Train] {BATCH_SIZE} streams x {train_tokens.shape[1]-1:,} steps | {EPOCHS} epochs per α")

    # Detect embed_dim from core
    hierarchy_tmp = PredictiveHierarchy([110, 1024, 110], device=DEVICE)
    hierarchy_tmp.load_checkpoint(CORE_CHECKPOINT)
    embed_dim = hierarchy_tmp.layers[0].input_dim
    del hierarchy_tmp
    print(f"[Core] embed_dim={embed_dim}")

    # ─── Run sweep ────────────────────────────────────────────
    summary = {}
    for alpha in ALPHA_VALUES:
        best_ppl, results = run_single_alpha(alpha, tokenizer, train_tokens, valid_tokens, vocab_size, embed_dim)
        summary[alpha] = best_ppl

    # ─── Final Comparison ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALPHA SWEEP RESULTS")
    print("=" * 60)
    print(f"  {'α':>6} | {'Best Val PPL':>12}")
    print(f"  {'-'*6}-+-{'-'*12}")
    for alpha in ALPHA_VALUES:
        marker = " ★" if summary[alpha] == min(summary.values()) else ""
        print(f"  {alpha:>6.1f} | {summary[alpha]:>12.1f}{marker}")

    best_alpha = min(summary, key=summary.get)
    print(f"\n  ★ OPTIMAL α = {best_alpha}")
    print(f"  ★ Best PPL = {summary[best_alpha]:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
