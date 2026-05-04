"""
agnis_pure_vs_transformer.py  v7
=================================
Problem identified in v6:
  AGNIS loss is perfectly FLAT after step 100 (6.495 → 6.490 over 600 steps).
  Root cause: SNAP-ATP optimizes RECONSTRUCTION error between layers.
  The readout objective (next-token) is completely DECOUPLED from SNAP-ATP.
  Adam on the readout converges in 100 steps then gets stuck because the
  hierarchy features are not changing toward next-token prediction.

v7 fix — Dopamine Feedback (biologically plausible, zero backprop):
  1. Run a fast forward pass through the hierarchy (predict_label, no weight update).
  2. Make a readout prediction.
  3. Use readout accuracy as dopamine_burst signal.
  4. Run infer_and_learn WITH that dopamine signal.
  Correct prediction → dopamine=2.0 → SNAP-ATP strongly reinforces those patterns.
  Wrong prediction  → dopamine=0.5 → SNAP-ATP discounts those patterns.
  This GUIDES the hierarchy toward features useful for next-token prediction.
  No gradients ever enter the hierarchy. 100% biological.
"""

import os, math, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agnis_v4_core import PredictiveHierarchy

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
CORPUS_PATH    = "slm/input_en_massive.txt"
TOKENIZER_PATH = "slm_bpe_tokenizer_en.json"

VOCAB_SIZE     = 4096
EMBED_DIM      = 128
READOUT_DIM    = 1024      # read from the MIDDLE layer (layers[0].x) — 8x richer than top
SEQ_LEN        = 64
BATCH_SIZE     = 64
LR_TRANSFORM   = 5e-4
LR_DELTA       = 3e-4      # Adam LR for readout (same as GPT for fair comparison)
ADAM_B1        = 0.9
ADAM_B2        = 0.999
ADAM_EPS       = 1e-8
MAX_STEPS      = 1500
LOG_EVERY      = 50
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT         = "The history of"


# ═══════════════════════════════════════════════════════════════════
#  CONTENDER 2: TINY GPT
# ═══════════════════════════════════════════════════════════════════
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_seq=SEQ_LEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.size()
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h    = self.embedding(x) + self.pos_emb(pos)
        h    = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)

@torch.no_grad()
def generate_gpt(model, tokenizer, prompt, max_tokens=60):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_tokens):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=DEVICE)
        ids.append(model(ctx)[0, -1].argmax().item())
    model.train()
    return tokenizer.decode(ids)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("  PURE AGNIS v3 vs TRANSFORMER  (Zero-Backprop Deathmatch)")
    print("=" * 64)

    if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(CORPUS_PATH):
        print("[ERROR] Run run_english_fluency.py first (builds tokenizer + corpus).")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()[:2_000_000]
    tokens = tokenizer.encode(text).ids
    print(f"[Arena] {len(tokens):,} tokens loaded.")

    n    = len(tokens) // (BATCH_SIZE * SEQ_LEN)
    data = torch.tensor(tokens[:n * BATCH_SIZE * SEQ_LEN], device=DEVICE).view(BATCH_SIZE, -1)

    # ── PURE AGNIS ─────────────────────────────────────────────────
    hierarchy = PredictiveHierarchy([EMBED_DIM, 1024, EMBED_DIM], device=DEVICE)
    hierarchy.reset_states(batch_size=BATCH_SIZE)

    # Frozen orthogonal embedding (zero gradients, ever)
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    nn.init.orthogonal_(embedding.weight)
    embedding.weight.requires_grad = False

    # Delta Rule readout  W: [VOCAB_SIZE, READOUT_DIM=1024]
    W     = torch.randn(VOCAB_SIZE, READOUT_DIM, device=DEVICE) * 0.02
    # Adam state for W
    W_m   = torch.zeros_like(W)
    W_v   = torch.zeros_like(W)
    W_t   = 0   # Adam step counter

    # ── Transformer ─────────────────────────────────────────────────
    gpt     = TinyGPT(VOCAB_SIZE, d_model=128, nhead=4, num_layers=4).to(DEVICE)
    opt_gpt = torch.optim.Adam(gpt.parameters(), lr=LR_TRANSFORM)

    a_params = sum(p.numel() for p in hierarchy.parameters())
    g_params = sum(p.numel() for p in gpt.parameters())
    print(f"\n[Fighters]")
    print(f"  PURE AGNIS  : {a_params:,} + {W.numel():,} readout = "
          f"{a_params + W.numel():,} params  "
          f"(SNAP-ATP + Delta Rule on 1024-dim middle layer. ZERO Backprop.)")
    print(f"  Transformer : {g_params:,} params  (Adam + Backprop, from scratch)")

    print(f"\n{'Step':>6} | {'AGNIS Loss':>12} | {'GPT Loss':>12} | "
          f"{'AGNIS PPL':>10} | {'GPT PPL':>10}")
    print("-" * 68)

    t0 = time.time()
    sum_a = sum_g = 0.0

    for step in range(MAX_STEPS):
        idx = (step * SEQ_LEN) % (data.shape[1] - SEQ_LEN - 1)
        x   = data[:, idx : idx + SEQ_LEN]
        y   = data[:, idx + 1 : idx + SEQ_LEN + 1]

        # ── Transformer (Backprop) ────────────────────────────────
        loss_gpt = F.cross_entropy(gpt(x).view(-1, VOCAB_SIZE), y.reshape(-1))
        opt_gpt.zero_grad()
        loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        opt_gpt.step()

        # ── PURE AGNIS (Zero Backprop) ────────────────────────────
        loss_agnis = 0.0

        for t in range(SEQ_LEN):
            xt  = x[:, t]
            yt  = y[:, t]
            emb = embedding(xt)    # [B, EMBED_DIM]  — frozen

            # ── v7: Dopamine Feedback Loop ──────────────────────────
            # Step 1: Fast inference pass to get features (no weight update yet)
            with torch.no_grad():
                _ = hierarchy.forward(emb, max_steps=5, update_temporal=False)
                hid_pre = hierarchy.layers[0].x.detach()  # [B, READOUT_DIM]
                if hid_pre.shape[1] > READOUT_DIM:
                    hid_pre = hid_pre[:, :READOUT_DIM]

            # Step 2: Readout prediction using pre-update features
            with torch.no_grad():
                logits_pre = F.linear(hid_pre, W)
                pred_correct = (logits_pre.argmax(dim=-1) == yt).float()
                acc = pred_correct.mean().item()

            # Step 3: Dopamine = reward signal (correct → high, wrong → low)
            # Range [0.5, 2.0] — modulates SNAP-ATP plasticity
            dopamine = 0.5 + 1.5 * acc   # 0.5 when all wrong, 2.0 when all correct

            # Step 4: SNAP-ATP with dopamine-guided plasticity (no backprop)
            hierarchy.infer_and_learn(emb, max_steps=10, warm_start=True,
                                      dopamine_burst=dopamine)

            # Read settled middle-layer features
            hid = hierarchy.layers[0].x.detach()    # [B, 1024]
            if hid.shape[1] > READOUT_DIM:
                hid = hid[:, :READOUT_DIM]

            # Plain Delta Rule gradient (local — no backprop through hierarchy)
            logits = F.linear(hid, W)
            loss_t = F.cross_entropy(logits, yt)
            loss_agnis += loss_t.item()

            probs  = F.softmax(logits, dim=-1)
            tgt_oh = F.one_hot(yt, VOCAB_SIZE).float()
            error  = tgt_oh - probs                  # [B, VOCAB_SIZE]
            grad_W = (error.t() @ hid) / BATCH_SIZE  # [VOCAB_SIZE, READOUT_DIM]

            # Adam update on W (momentum + adaptive LR, gradient is still local)
            W_t   += 1
            W_m    = ADAM_B1 * W_m + (1 - ADAM_B1) * grad_W
            W_v    = ADAM_B2 * W_v + (1 - ADAM_B2) * grad_W ** 2
            m_hat  = W_m / (1 - ADAM_B1 ** W_t)
            v_hat  = W_v / (1 - ADAM_B2 ** W_t)
            W     += LR_DELTA * m_hat / (v_hat.sqrt() + ADAM_EPS)

        loss_agnis /= SEQ_LEN
        sum_a += loss_agnis
        sum_g += loss_gpt.item()

        if (step + 1) % LOG_EVERY == 0:
            a = sum_a / LOG_EVERY
            g = sum_g / LOG_EVERY
            sum_a = sum_g = 0.0
            winner = "🧠 PURE AGNIS" if a < g else "🤖 GPT Backprop"
            print(f"{step+1:>6} | {a:>12.4f} | {g:>12.4f} | "
                  f"{math.exp(min(a,20)):>10.1f} | {math.exp(min(g,20)):>10.1f} | {winner}")

    # ── Generation showdown ───────────────────────────────────────
    print("\n" + "=" * 64)
    print("  FINAL GENERATION SHOWDOWN")
    print("=" * 64)

    print("\n[PURE AGNIS — Zero Backprop]")
    hierarchy.reset_states(batch_size=1)
    ids     = tokenizer.encode(PROMPT).ids
    gen_ids = list(ids)
    for tok in ids:
        emb = embedding(torch.tensor([[tok]], device=DEVICE))
        hierarchy.infer_and_learn(emb, max_steps=10)
    for _ in range(60):
        last = torch.tensor([[gen_ids[-1]]], device=DEVICE)
        emb  = embedding(last)
        hierarchy.infer_and_learn(emb, max_steps=10, warm_start=True)
        hid  = hierarchy.layers[0].x.detach()       # middle layer
        if hid.shape[1] > READOUT_DIM: hid = hid[:, :READOUT_DIM]
        logits  = F.linear(hid, W)
        next_id = torch.multinomial(F.softmax(logits[0] / 0.8, dim=-1), 1).item()
        gen_ids.append(next_id)
        if next_id == (tokenizer.token_to_id("<|endoftext|>") or -1): break
    print(tokenizer.decode(gen_ids).replace('\n', ' '))

    print("\n[Tiny GPT — Adam + Backprop]")
    print(generate_gpt(gpt, tokenizer, PROMPT).replace('\n', ' '))

    print(f"\n[Done] {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
