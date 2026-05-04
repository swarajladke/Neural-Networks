"""
agnis_pure_vs_transformer.py
============================
The Pure Biological Deathmatch.

Contender 1: PURE AGNIS (0% Backprop)
- Core Hierarchy: Learns from scratch via SNAP-ATP local learning.
- Embedding: Fixed random projections (no gradients).
- Readout Head: Learns via Widrow-Hoff (Delta Rule) local association.
- NOT A SINGLE `loss.backward()` OR OPTIMIZER IS USED FOR AGNIS.

Contender 2: Standard Transformer (GPT-style)
- Trained entirely from scratch via Backpropagation and Adam.

Rules of the Arena:
1. Exact same Gutenberg corpus & tokenizer.
2. Exact same 64-token sequence chunks.

Can a 100% backprop-free biological network keep up with the global optimization
power of Backprop-trained Attention?
"""

import os, math, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm.agnis_slm_wrapper import AGNISSLMWrapper
from agnis_v4_core import PredictiveHierarchy

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CORPUS_PATH   = "slm/input_en_massive.txt"
TOKENIZER     = "slm_bpe_tokenizer_en.json"

VOCAB_SIZE    = 4096
EMBED_DIM     = 128     # Match Transformer d_model
SEQ_LEN       = 64
BATCH_SIZE    = 64
LR_TRANSFORM  = 5e-4
LR_DELTA      = 0.01    # Local learning rate for AGNIS readout
MAX_STEPS     = 1500
LOG_EVERY     = 50
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT        = "The history of"


# ═══════════════════════════════════════════════════════════════
#  CONTENDER 2: GPT-STYLE TRANSFORMER
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
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x_emb = self.embedding(x) + self.pos_emb(pos)
        hidden = self.transformer(x_emb, mask=mask, is_causal=True)
        return self.output_head(hidden)

@torch.no_grad()
def generate_gpt(model, tokenizer, prompt, max_tokens=50):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_tokens):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=DEVICE)
        logits = model(ctx)
        next_id = logits[0, -1].argmax().item()
        ids.append(next_id)
        if next_id == tokenizer.token_to_id("<|endoftext|>"): break
    model.train()
    return tokenizer.decode(ids)


# ═══════════════════════════════════════════════════════════════
#  TRAINING ARENA
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PURE AGNIS vs TRANSFORMER: No-Backprop Deathmatch")
    print("=" * 60)
    
    if not os.path.exists(TOKENIZER) or not os.path.exists(CORPUS_PATH):
        print("[ERROR] Tokenizer or Corpus missing. Run run_english_fluency.py Step 1 first.")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER)
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()[:2_000_000]
    tokens = tokenizer.encode(text).ids
    print(f"[Arena] Dataset: {len(tokens):,} tokens loaded.")
    
    n_batches = len(tokens) // (BATCH_SIZE * SEQ_LEN)
    data = torch.tensor(tokens[:n_batches * BATCH_SIZE * SEQ_LEN], device=DEVICE).view(BATCH_SIZE, -1)
    
    # ── Setup PURE AGNIS ─────────────────────────────────────────
    # 1. Fresh Hierarchy (will learn via SNAP-ATP)
    hierarchy = PredictiveHierarchy([EMBED_DIM, 1024, EMBED_DIM], device=DEVICE)
    hierarchy.reset_states(batch_size=BATCH_SIZE)
    
    # 2. Frozen Random Embedding (Orthogonal projection)
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    nn.init.orthogonal_(embedding.weight)
    embedding.weight.requires_grad = False
    
    # 3. Local Readout Matrix (Will learn via Delta Rule)
    # Shape: [Vocab, Embed]
    readout_W = torch.randn(VOCAB_SIZE, EMBED_DIM, device=DEVICE) * 0.02
    
    agnis_params = sum(p.numel() for p in hierarchy.parameters())
    
    # ── Setup GPT ───────────────────────────────────────────────
    gpt = TinyGPT(vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=4).to(DEVICE)
    opt_gpt = torch.optim.Adam(gpt.parameters(), lr=LR_TRANSFORM)
    gpt_params = sum(p.numel() for p in gpt.parameters())
    
    print(f"\\n[Fighters]")
    print(f"  PURE AGNIS   : {agnis_params:,} params (100% Local Learning. ZERO Backprop.)")
    print(f"  Transformer  : {gpt_params:,} params (100% Backprop/Adam)\\n")
    
    # ── Fight ───────────────────────────────────────────────────
    step = 0
    print(f"{'Step':>6} | {'AGNIS Loss':>12} | {'GPT Loss':>12} | {'AGNIS PPL':>10} | {'GPT PPL':>10}")
    print("-" * 65)
    
    hist_agnis_loss, hist_gpt_loss = 0, 0
    t0 = time.time()
    
    while step < MAX_STEPS:
        idx = (step * SEQ_LEN) % (data.shape[1] - SEQ_LEN - 1)
        x = data[:, idx : idx + SEQ_LEN]
        y = data[:, idx + 1 : idx + SEQ_LEN + 1]
        
        # ── 1. Transformer Turn (Backprop) ──
        logits_gpt = gpt(x)
        loss_gpt = F.cross_entropy(logits_gpt.view(-1, VOCAB_SIZE), y.reshape(-1))
        
        opt_gpt.zero_grad()
        loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        opt_gpt.step()
        
        # ── 2. PURE AGNIS Turn (Local Learning) ──
        loss_agnis = 0
        
        for t in range(SEQ_LEN):
            xt = x[:, t]
            yt = y[:, t]
            
            # Static mapping
            emb = embedding(xt)
            
            # SNAP-ATP Pure Local Learning (no backprop at all)
            # Step 1: Local SNAP-ATP weight update (returns steps_used, converged)
            hierarchy.infer_and_learn(emb, max_steps=5, warm_start=True)
            # Step 2: Read out hidden state after settling (for the readout head)
            with torch.no_grad():
                hid = hierarchy.predict_label(emb, max_steps=5, update_temporal=True)
            if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
            
            # Readout via Delta Rule
            # Forward pass: W @ hid
            logits_a = F.linear(hid, readout_W)
            
            # Loss for tracking only (does not compute gradients)
            loss_t = F.cross_entropy(logits_a, yt)
            loss_agnis += loss_t.item()
            
            # Delta Rule Update (Widrow-Hoff):
            # Error = (Target_Probabilities - Prediction_Probabilities)
            probs = F.softmax(logits_a, dim=-1)
            target_one_hot = F.one_hot(yt, num_classes=VOCAB_SIZE).float()
            
            error = (target_one_hot - probs)  # [B, Vocab]
            
            # W = W + lr * (Error^T @ Hidden) / B
            grad_W = torch.matmul(error.t(), hid) / BATCH_SIZE
            readout_W += LR_DELTA * grad_W
            
        loss_agnis = loss_agnis / SEQ_LEN
        
        # ── Logging ──
        hist_agnis_loss += loss_agnis
        hist_gpt_loss += loss_gpt.item()
        
        if (step + 1) % LOG_EVERY == 0:
            a_avg = hist_agnis_loss / LOG_EVERY
            g_avg = hist_gpt_loss / LOG_EVERY
            a_ppl = math.exp(min(a_avg, 20))
            g_ppl = math.exp(min(g_avg, 20))
            winner = "🧠 PURE AGNIS" if a_avg < g_avg else "🤖 GPT Backprop"
            
            print(f"{step+1:>6} | {a_avg:>12.4f} | {g_avg:>12.4f} | {a_ppl:>10.1f} | {g_ppl:>10.1f} | {winner}")
            hist_agnis_loss, hist_gpt_loss = 0, 0
            
        step += 1
        
    # ── Final Generation Showdown ──
    print("\n" + "=" * 60)
    print(" FINAL GENERATION SHOWDOWN")
    print("=" * 60)
    
    print("\n[PURE AGNIS (0% Backprop)]")
    hierarchy.reset_states(batch_size=1)
    ids = tokenizer.encode(PROMPT).ids
    gen_ids = list(ids)
    
    # Prime hierarchy with prompt (local learning on prompt tokens)
    for tok_id in ids:
        emb = embedding(torch.tensor([[tok_id]], device=DEVICE))
        hierarchy.infer_and_learn(emb, max_steps=5)
        
    for _ in range(50):
        last = torch.tensor([[gen_ids[-1]]], device=DEVICE)
        emb = embedding(last)
        with torch.no_grad():
            hid = hierarchy.predict_label(emb, max_steps=5)
        if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
        
        logits = F.linear(hid, readout_W)
        next_id = logits[0].argmax().item()
        gen_ids.append(next_id)
        if next_id == tokenizer.token_to_id("<|endoftext|>"): break
    print(tokenizer.decode(gen_ids).replace('\n', ' '))
    
    print("\n[Tiny GPT (Trained via Backprop)]")
    print(generate_gpt(gpt, tokenizer, PROMPT).replace('\n', ' '))

if __name__ == "__main__":
    main()
