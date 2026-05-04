"""
agnis_pure_vs_transformer.py  v2
=================================
Pure Biological AGNIS vs Backprop Transformer.

Fixes vs v1:
  1. Temporal state stepped ONCE per token (not twice).
     infer_and_learn() already calls step_temporal() internally.
     We read hierarchy.layers[-1].x directly — no second predict_label call.
  2. 2-layer local MLP readout replaces the weak single Delta-Rule layer.
     Both layers updated via Hebbian Delta Rule (zero backprop).
"""

import os, math, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agnis_v4_core import PredictiveHierarchy

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CORPUS_PATH    = "slm/input_en_massive.txt"
TOKENIZER_PATH = "slm_bpe_tokenizer_en.json"

VOCAB_SIZE     = 4096
EMBED_DIM      = 128      # match Transformer d_model
READOUT_HIDDEN = 256      # hidden dim of 2-layer local MLP readout
SEQ_LEN        = 64
BATCH_SIZE     = 64
LR_TRANSFORM   = 5e-4
LR_DELTA       = 5e-3     # local readout learning rate
MAX_STEPS      = 1500
LOG_EVERY      = 50
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT         = "The history of"


# ═══════════════════════════════════════════════════════════════
#  CONTENDER 2: TINY GPT
# ═══════════════════════════════════════════════════════════════
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
        ctx    = torch.tensor([ids[-SEQ_LEN:]], device=DEVICE)
        logits = model(ctx)
        ids.append(logits[0, -1].argmax().item())
    model.train()
    return tokenizer.decode(ids)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PURE AGNIS v2 vs TRANSFORMER (No-Backprop Deathmatch)")
    print("=" * 60)

    if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(CORPUS_PATH):
        print("[ERROR] Run run_english_fluency.py first to build tokenizer + corpus.")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()[:2_000_000]
    tokens = tokenizer.encode(text).ids
    print(f"[Arena] {len(tokens):,} tokens loaded.")

    n      = len(tokens) // (BATCH_SIZE * SEQ_LEN)
    data   = torch.tensor(tokens[:n * BATCH_SIZE * SEQ_LEN], device=DEVICE).view(BATCH_SIZE, -1)

    # ── PURE AGNIS setup ────────────────────────────────────────
    hierarchy = PredictiveHierarchy([EMBED_DIM, 1024, EMBED_DIM], device=DEVICE)
    hierarchy.reset_states(batch_size=BATCH_SIZE)

    # Frozen orthogonal embedding (no gradients ever)
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    nn.init.orthogonal_(embedding.weight)
    embedding.weight.requires_grad = False

    # 2-layer local MLP readout  (W1, W2 — no backprop)
    W1 = torch.randn(READOUT_HIDDEN, EMBED_DIM,      device=DEVICE) * 0.02
    W2 = torch.randn(VOCAB_SIZE,     READOUT_HIDDEN, device=DEVICE) * 0.02

    # ── Transformer setup ────────────────────────────────────────
    gpt     = TinyGPT(VOCAB_SIZE, d_model=128, nhead=4, num_layers=4).to(DEVICE)
    opt_gpt = torch.optim.Adam(gpt.parameters(), lr=LR_TRANSFORM)

    agnis_params = sum(p.numel() for p in hierarchy.parameters())
    gpt_params   = sum(p.numel() for p in gpt.parameters())
    print(f"\n[Fighters]")
    print(f"  PURE AGNIS  : {agnis_params:,} params  (SNAP-ATP + 2-layer Hebbian readout, ZERO Backprop)")
    print(f"  Transformer : {gpt_params:,} params  (Backprop + Adam, trains from scratch)")
    print(f"  Readout W1  : {W1.numel():,}  W2: {W2.numel():,}  (Delta Rule only)\n")

    print(f"{'Step':>6} | {'AGNIS Loss':>12} | {'GPT Loss':>12} | {'AGNIS PPL':>10} | {'GPT PPL':>10}")
    print("-" * 68)

    t0 = time.time()
    sum_a, sum_g = 0.0, 0.0

    for step in range(MAX_STEPS):
        idx = (step * SEQ_LEN) % (data.shape[1] - SEQ_LEN - 1)
        x   = data[:, idx : idx + SEQ_LEN]          # [B, T]
        y   = data[:, idx + 1 : idx + SEQ_LEN + 1]  # [B, T]

        # ── Transformer (Backprop) ───────────────────────────────
        loss_gpt = F.cross_entropy(gpt(x).view(-1, VOCAB_SIZE), y.reshape(-1))
        opt_gpt.zero_grad(); loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        opt_gpt.step()

        # ── PURE AGNIS (Zero Backprop) ───────────────────────────
        # Process token-by-token, carrying temporal state forward.
        # FIX 1: infer_and_learn() already calls step_temporal().
        #         We read hierarchy.layers[-1].x directly — no second
        #         predict_label call that would step temporal again.
        loss_agnis = 0.0

        for t in range(SEQ_LEN):
            xt = x[:, t]   # [B]
            yt = y[:, t]   # [B]

            emb = embedding(xt)  # [B, EMBED_DIM]  — frozen, no grad

            # SNAP-ATP: settle + update weights + step temporal (all local)
            hierarchy.infer_and_learn(emb, max_steps=3, warm_start=True)

            # Read settled top-layer state directly (no second temporal step)
            hid = hierarchy.layers[-1].x.detach()          # [B, EMBED_DIM]
            if hid.shape[1] > EMBED_DIM:
                hid = hid[:, :EMBED_DIM]

            # FIX 2: 2-layer local MLP readout (Delta Rule on both layers)
            h1      = torch.relu(F.linear(hid, W1))        # [B, READOUT_HIDDEN]
            logits  = F.linear(h1, W2)                     # [B, VOCAB_SIZE]

            loss_t  = F.cross_entropy(logits, yt)
            loss_agnis += loss_t.item()

            # Delta Rule — W2
            probs   = F.softmax(logits, dim=-1)
            tgt_oh  = F.one_hot(yt, VOCAB_SIZE).float()
            err2    = tgt_oh - probs                        # [B, VOCAB_SIZE]
            W2     += LR_DELTA * (err2.t() @ h1) / BATCH_SIZE

            # Delta Rule — W1  (error propagated through W2, gated by ReLU)
            err1    = (err2 @ W2) * (h1 > 0).float()       # [B, READOUT_HIDDEN]
            W1     += LR_DELTA * (err1.t() @ hid) / BATCH_SIZE

        loss_agnis /= SEQ_LEN

        sum_a += loss_agnis
        sum_g += loss_gpt.item()

        if (step + 1) % LOG_EVERY == 0:
            a = sum_a / LOG_EVERY;  g = sum_g / LOG_EVERY
            sum_a = sum_g = 0.0
            winner = "🧠 PURE AGNIS" if a < g else "🤖 GPT Backprop"
            print(f"{step+1:>6} | {a:>12.4f} | {g:>12.4f} | "
                  f"{math.exp(min(a,20)):>10.1f} | {math.exp(min(g,20)):>10.1f} | {winner}")

    # ── Generation showdown ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL GENERATION SHOWDOWN")
    print("=" * 60)

    print("\n[PURE AGNIS — 0% Backprop]")
    hierarchy.reset_states(batch_size=1)
    ids     = tokenizer.encode(PROMPT).ids
    gen_ids = list(ids)
    for tok in ids:
        emb = embedding(torch.tensor([[tok]], device=DEVICE))
        hierarchy.infer_and_learn(emb, max_steps=3)
    for _ in range(60):
        last  = torch.tensor([[gen_ids[-1]]], device=DEVICE)
        emb   = embedding(last)
        hierarchy.infer_and_learn(emb, max_steps=3, warm_start=True)
        hid   = hierarchy.layers[-1].x.detach()
        if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
        h1    = torch.relu(F.linear(hid, W1))
        logits = F.linear(h1, W2)
        next_id = torch.multinomial(F.softmax(logits[0]/0.8, dim=-1), 1).item()
        gen_ids.append(next_id)
        if next_id == (tokenizer.token_to_id("<|endoftext|>") or -1): break
    print(tokenizer.decode(gen_ids).replace('\n', ' '))

    print("\n[Tiny GPT — Backprop]")
    print(generate_gpt(gpt, tokenizer, PROMPT).replace('\n', ' '))

    print(f"\n[Time] {time.time()-t0:.1f}s total")
    print("Deathmatch complete.")


if __name__ == "__main__":
    main()
