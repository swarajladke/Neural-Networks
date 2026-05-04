"""
agnis_vs_transformer.py
=======================
The Ultimate Architecture Deathmatch.

Contender 1: Pure AGNIS
- Core Hierarchy is completely FROZEN (pre-trained on code).
- Only the English Interface (Embedding + Linear Head) is trained.
- Infinite stateful context via SNAP-ATP temporal mechanics.

Contender 2: Standard Transformer (GPT-style)
- Decoder-only, causal masking.
- Trained entirely from scratch on the English text.
- Fixed context window (64 tokens).

Rules of the Arena:
1. Exact same Gutenberg corpus.
2. Exact same 4096 English BPE tokenizer.
3. Exact same training loop: B=64, SeqLen=64.
4. Exact same Optimizer (Adam, lr=5e-4).

Will the universal grammar of a frozen code-trained AGNIS beat a Transformer
learning English syntax from scratch?
"""

import os, math, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm.agnis_slm_wrapper import AGNISSLMWrapper

# ═══════════════════════════════════════════════════════════════
#  CONFIG & HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════
CORPUS_PATH   = "slm/input_en_massive.txt"
TOKENIZER     = "slm_bpe_tokenizer_en.json"
AGNIS_CKPT    = "agnis_marathon_final.pt"

VOCAB_SIZE    = 4096
EMBED_DIM     = 110     # AGNIS constraint
SEQ_LEN       = 64      # BPTT window / Transformer max context
BATCH_SIZE    = 64
LR            = 5e-4
MAX_STEPS     = 1500    # ~3 epochs on a small chunk to see who learns faster
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
        
        # Decoder-only layers
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        # x: [B, SeqLen]
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Causal mask (prevents looking into the future)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        x_emb = self.embedding(x) + self.pos_emb(pos)
        hidden = self.transformer(x_emb, mask=mask, is_causal=True)
        logits = self.output_head(hidden)
        return logits

@torch.no_grad()
def generate_gpt(model, tokenizer, prompt, max_tokens=50):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_tokens):
        # crop to max seq
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
    print("  AGNIS vs TRANSFORMER: Architecture Deathmatch")
    print("=" * 60)
    
    if not os.path.exists(TOKENIZER) or not os.path.exists(CORPUS_PATH):
        print("[ERROR] Tokenizer or Corpus missing. Run run_english_fluency.py Step 1 first.")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER)
    
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()[:2_000_000] # Use 2M chars for rapid prototyping
    tokens = tokenizer.encode(text).ids
    print(f"[Arena] Dataset: {len(tokens):,} tokens loaded.")
    
    # Prep data blocks [B, SeqLen]
    n_batches = len(tokens) // (BATCH_SIZE * SEQ_LEN)
    data = torch.tensor(tokens[:n_batches * BATCH_SIZE * SEQ_LEN], device=DEVICE)
    data = data.view(BATCH_SIZE, -1)
    
    # ── Setup AGNIS ─────────────────────────────────────────────
    agnis = AGNISSLMWrapper(device=DEVICE)
    agnis.load_checkpoint(AGNIS_CKPT)
    agnis.to(DEVICE)
    for p in agnis.hierarchy.parameters(): p.requires_grad_(False)
    
    agnis.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    agnis.output_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False).to(DEVICE)
    nn.init.normal_(agnis.embedding.weight, std=0.02)
    nn.init.normal_(agnis.output_head.weight, std=0.02)
    
    opt_agnis = torch.optim.Adam(
        list(agnis.embedding.parameters()) + list(agnis.output_head.parameters()), 
        lr=LR
    )
    
    # ── Setup GPT ───────────────────────────────────────────────
    gpt = TinyGPT(vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=4).to(DEVICE)
    opt_gpt = torch.optim.Adam(gpt.parameters(), lr=LR)
    
    # Param counts
    agnis_trainable = sum(p.numel() for p in agnis.embedding.parameters()) + \
                      sum(p.numel() for p in agnis.output_head.parameters())
    gpt_trainable   = sum(p.numel() for p in gpt.parameters())
    
    print(f"\\n[Fighters]")
    print(f"  AGNIS Interface  : {agnis_trainable:,} trainable params (Core is frozen!)")
    print(f"  Tiny Transformer : {gpt_trainable:,} trainable params (Trains from scratch)\\n")
    
    # ── Fight ───────────────────────────────────────────────────
    step = 0
    agnis.hierarchy.reset_states(batch_size=BATCH_SIZE)
    
    print(f"{'Step':>6} | {'AGNIS Loss':>12} | {'GPT Loss':>12} | {'AGNIS PPL':>10} | {'GPT PPL':>10}")
    print("-" * 65)
    
    hist_agnis_loss, hist_gpt_loss = 0, 0
    t0 = time.time()
    
    while step < MAX_STEPS:
        # Get batch (shifted by 1 for target)
        idx = (step * SEQ_LEN) % (data.shape[1] - SEQ_LEN - 1)
        x = data[:, idx : idx + SEQ_LEN]
        y = data[:, idx + 1 : idx + SEQ_LEN + 1]
        
        # ── 1. Transformer Turn ──
        logits_gpt = gpt(x)
        loss_gpt = F.cross_entropy(logits_gpt.view(-1, VOCAB_SIZE), y.reshape(-1))
        
        opt_gpt.zero_grad()
        loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        opt_gpt.step()
        
        # ── 2. AGNIS Turn ──
        # Process seq_len window token-by-token carrying state forward
        loss_agnis = 0
        opt_agnis.zero_grad()
        
        for t in range(SEQ_LEN):
            xt = x[:, t]
            yt = y[:, t]
            
            emb = F.normalize(agnis.embedding(xt), dim=-1)
            with torch.no_grad():
                hid = agnis.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
                if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
            
            combined = emb + 0.5 * hid.detach()
            logits_a = agnis.output_head(combined)
            loss_agnis += F.cross_entropy(logits_a, yt)
            
        loss_agnis = loss_agnis / SEQ_LEN
        loss_agnis.backward()
        torch.nn.utils.clip_grad_norm_(agnis.embedding.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(agnis.output_head.parameters(), 1.0)
        opt_agnis.step()
        
        # ── Logging ──
        hist_agnis_loss += loss_agnis.item()
        hist_gpt_loss += loss_gpt.item()
        
        if (step + 1) % LOG_EVERY == 0:
            a_avg = hist_agnis_loss / LOG_EVERY
            g_avg = hist_gpt_loss / LOG_EVERY
            a_ppl = math.exp(min(a_avg, 20))
            g_ppl = math.exp(min(g_avg, 20))
            
            winner = "🔥 AGNIS leads" if a_avg < g_avg else "🤖 GPT leads"
            
            print(f"{step+1:>6} | {a_avg:>12.4f} | {g_avg:>12.4f} | {a_ppl:>10.1f} | {g_ppl:>10.1f} | {winner}")
            hist_agnis_loss, hist_gpt_loss = 0, 0
            
        step += 1
        
    # ── Final Generation Showdown ──
    print("\\n" + "=" * 60)
    print(" FINAL GENERATION SHOWDOWN")
    print("=" * 60)
    
    # AGNIS generation
    print("\\n[AGNIS (Frozen Core)]")
    agnis.hierarchy.reset_states(batch_size=1)
    ids = tokenizer.encode(PROMPT).ids
    gen_ids = list(ids)
    for tok_id in ids:
        emb = F.normalize(agnis.embedding(torch.tensor([[tok_id]], device=DEVICE)).view(1, -1), dim=-1)
        hid = agnis.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
        
    for _ in range(50):
        last = torch.tensor([[gen_ids[-1]]], device=DEVICE)
        emb = F.normalize(agnis.embedding(last).view(1, -1), dim=-1)
        hid = agnis.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
        logits = agnis.output_head(emb + 0.5 * hid)
        next_id = logits[0].argmax().item()
        gen_ids.append(next_id)
        if next_id == tokenizer.token_to_id("<|endoftext|>"): break
    print(tokenizer.decode(gen_ids).replace('\\n', ' '))
    
    # GPT generation
    print("\\n[Tiny GPT (Trained from scratch)]")
    print(generate_gpt(gpt, tokenizer, PROMPT).replace('\\n', ' '))
    
    print(f"\\n[Time] {time.time()-t0:.1f}s")
    print("Deathmatch complete.")


if __name__ == "__main__":
    main()
