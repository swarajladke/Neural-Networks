"""
agnis_v5_sprint2.py
================================================================
AGNIS V5 | SPRINT 2, SESSION 5 — "Hybrid AGNIS-Transformer"
================================================================
Architecture: Causal Transformer Context Encoder → Hebbian Core
  - 2-layer Causal Transformer contextualizes 64-token sequences
  - Hebbian Core receives contextualized embeddings (online learning preserved)
  - Temporal Memory (R_weight, h_prev) still runs across sequence
  - Continual learning is NOT hurt — core still updates Hebbianly

Why this works:
  Transformer: handles syntax (learns once, stable across tasks)
  Hebbian Core: handles semantics + new knowledge (continual learning)
"""

import os, sys, time, gc, math
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizers import Tokenizer
from agnis_v4_core import PredictiveHierarchy

# ─── Config ───────────────────────────────────────────────────────
MODEL_NAME      = "agnis_v5_30m_fluency"
SAVE_DIR        = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
CHECKPOINT_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pt")
TOKENIZER_PATH  = "slm_bpe_tokenizer_32k.json"

TARGET_TOKENS   = 200_000_000
BATCH_SIZE      = 64
SEQ_LEN         = 64
EPOCHS          = 10

EMBED_DIM       = 768
CORE_HIDDEN     = 3072
VOCAB_SIZE      = 32000
N_HEADS         = 8      # Transformer heads
N_LAYERS        = 2      # Transformer layers (lightweight)
FFN_DIM         = 2048   # Transformer FFN width

MAX_SETTLE_STEPS = 5
ALPHA           = 0.2   # Reduced for stability (was 0.4)
ETA_R_LOCAL     = 0.005
LR              = 5e-5   # Session 7: Fine-tuning semantics (was 1e-4)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Causal Context Encoder (Transformer) ────────────────────────
class CausalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, max_seq_len, device):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=0.0, batch_first=True).to(device)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        ).to(device)
        self.n1   = nn.LayerNorm(embed_dim).to(device)
        self.n2   = nn.LayerNorm(embed_dim).to(device)

        # Causal mask: each token only attends to itself and the past
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        T = x.shape[1]
        mask = self.causal_mask[:T, :T].to(x.device)  # Fix: move mask to GPU
        xn   = self.n1(x)
        attn_out, _ = self.attn(xn, xn, xn, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.n2(x))
        return x


class CausalContextEncoder(nn.Module):
    """2-layer Causal Transformer that contextualizes token embeddings."""
    def __init__(self, embed_dim, n_heads, n_layers, ffn_dim, max_seq_len, device):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim).to(device)
        self.blocks  = nn.ModuleList([
            CausalTransformerBlock(embed_dim, n_heads, ffn_dim, max_seq_len, device)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim).to(device)

    def forward(self, x):
        # x: [B, T, D] raw embeddings
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        x   = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)  # [B, T, D]


# ─── Hybrid AGNIS-Transformer Model ──────────────────────────────
class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072,
                 alpha=0.4, max_steps=5, device="cpu"):
        super().__init__()
        self.device    = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha     = alpha
        self.max_steps = max_steps

        # ── Embedding ──
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # ── Context Encoder (Transformer — learns syntax) ──
        self.context_encoder = CausalContextEncoder(
            embed_dim, N_HEADS, N_LAYERS, FFN_DIM, SEQ_LEN, device)

        # ── Hebbian Core (learns semantics + continual knowledge) ──
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, hidden_dim, embed_dim], device=device)
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)
        # Hybrid: allow V matrices to also receive backprop signal
        for col in self.hierarchy.layers:
            col.V.requires_grad_(True)
            col.b_in.requires_grad_(True)

        # ── Bridge Layers ──
        self.core_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self.device)
        for m in self.core_proj:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim).to(self.device)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.temporal_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        nn.init.zeros_(self.temporal_proj.weight)

        r_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(r_weight, gain=0.1)
        self.register_buffer("R_weight", r_weight)

        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)

        # ── LM Head (untied) ──
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        nn.init.normal_(self.lm_head.weight, std=0.02)

        self.register_buffer("h_prev", torch.zeros(1, embed_dim))
        self._current_surprise = 1.0

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.device)
        self._current_surprise = 1.0

    def forward(self, token_ids, gate_warmup=1.0):
        B, T = token_ids.shape

        # ── Step 1: Transformer contextualizes the full sequence ──
        raw_embs = self.embedding(token_ids)          # [B, T, D]
        ctx_embs = self.context_encoder(raw_embs)     # [B, T, D] — sees full 64-token context
        ctx_embs = F.normalize(ctx_embs, dim=-1)

        # ── Step 2: Token-by-token Hebbian core (online learning) ──
        logits_list    = []
        accumulated_dR = None

        for t in range(T):
            ctx_emb  = ctx_embs[:, t]           # [B, D]
            h_prev_d = self.h_prev.detach()

            temporal_raw     = torch.matmul(h_prev_d, self.R_weight)
            temporal_context = ctx_emb + self.alpha * temporal_raw

            # Hebbian settling (Hebbian updates run here — continual learning)
            with torch.no_grad():
                self.hierarchy.infer_and_learn(
                    temporal_context.detach(), max_steps=self.max_steps)

            # Differentiable shadow through V (gradient path)
            l0, l1      = self.hierarchy.layers[0], self.hierarchy.layers[1]
            h_shadow    = F.gelu(torch.matmul(temporal_context, l0.V) + l0.b_in)
            core_shadow = F.normalize(
                F.gelu(torch.matmul(h_shadow, l1.V) + l1.b_in), dim=-1)
            core_settled = F.normalize(
                self.hierarchy.layers[-1].x.float().detach(), dim=-1)
            core_blended = 0.5 * core_shadow + 0.5 * core_settled

            # Accumulate R_weight delta (apply once per sequence — Fix 4)
            with torch.no_grad():
                epsilon = core_settled - temporal_raw
                dR_step = torch.bmm(
                    h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                accumulated_dR = dR_step if accumulated_dR is None \
                    else accumulated_dR + dR_step

            core_feat    = self.core_proj(core_blended)
            temporal_feat = self.temporal_proj(temporal_raw)

            gate_input = torch.cat([ctx_emb, core_feat], dim=-1)
            gate       = torch.sigmoid(self.gate_proj(gate_input))

            h_t = ((gate * gate_warmup) * (core_feat + self.alpha * temporal_feat)
                   + (1.0 - gate * gate_warmup) * ctx_emb)
            h_t = self.out_norm(h_t)
            # Clamp to prevent temporal state explosion
            h_t = h_t.clamp(-5.0, 5.0)
            self.h_prev = h_t.detach()
            logits_list.append(self.lm_head(h_t))

        # Apply R_weight once per sequence
        if accumulated_dR is not None:
            with torch.no_grad():
                upd = (0.999 * self.R_weight
                       + ETA_R_LOCAL * self._current_surprise
                       * (accumulated_dR / T)).clamp(-3.0, 3.0)
                self.R_weight.copy_(upd)

        return torch.stack(logits_list, dim=1)   # [B, T, V]


# ─── Data ─────────────────────────────────────────────────────────
def get_data():
    from datasets import load_dataset
    print("[Data] Loading FineWeb-Edu + Wikitext-103...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    fw   = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                        split="train[:50000]")
    text  = "\n".join(t for t in wiki["text"] if len(t.strip()) > 20)
    text += "\n" + "\n".join(t for t in fw["text"] if len(t.strip()) > 20)
    return text


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  AGNIS V5 — HYBRID AGNIS-TRANSFORMER")
    print("  Transformer: syntax | Hebbian: semantics + continual learning")
    print(f"  SEQ={SEQ_LEN} | Layers={N_LAYERS} | Heads={N_HEADS} | LR={LR}")
    print("="*60)

    raw_text = get_data()

    if not os.path.exists(TOKENIZER_PATH):
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        tok = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = decoders.Sequence(
            [decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE,
                             special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
        tok.train_from_iterator(raw_text.splitlines(), trainer=trainer)
        tok.save(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    lines = raw_text.splitlines()
    del raw_text; gc.collect()

    ids = []
    for i in range(0, len(lines), 5000):
        encs = tokenizer.encode_batch(lines[i:i+5000])
        for e in encs: ids.extend(e.ids)
        del encs
    del lines; gc.collect()

    ids    = ids[:TARGET_TOKENS]
    sl     = len(ids) // BATCH_SIZE
    tokens = torch.tensor(ids[:sl*BATCH_SIZE], dtype=torch.long,
                          device=DEVICE).view(BATCH_SIZE, sl)
    del ids; gc.collect()

    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN,
                    alpha=ALPHA, max_steps=MAX_SETTLE_STEPS, device=DEVICE)

    # Load checkpoint — skip mismatched buffers (architecture changed)
    start_step, start_epoch = 0, 0
    for path in [CHECKPOINT_PATH, f"{MODEL_NAME}.pt"]:
        if os.path.exists(path):
            print(f"[Resume] Loading from {path}...")
            ckpt = torch.load(path, map_location=DEVICE)
            sd   = ckpt.get('model', {})
            if sd and next(iter(sd)).startswith('module.'):
                sd = {k[7:]: v for k, v in sd.items()}
            md    = model.state_dict()
            clean = {k: v for k, v in sd.items()
                     if k in md and v.shape == md[k].shape}
            skipped = len(sd) - len(clean)
            model.load_state_dict(clean, strict=False)
            model.gate_proj.bias.data.fill_(0.0)
            print(f"[Resume] Loaded {len(clean)} tensors. "
                  f"Skipped {skipped} (new arch layers start fresh — OK).")
            # Don't resume step — transformer is new, start from beginning
            print("[Resume] Starting from step 0 (Transformer layers are new).")
            break

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"[Model] Trainable: {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50000, eta_min=LR * 0.1)

    model.train()
    model.reset_states(BATCH_SIZE)

    loss_window     = deque(maxlen=100)
    steps_per_epoch = (tokens.shape[1] - SEQ_LEN - 1) // SEQ_LEN
    global_step     = 0
    t0              = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch+1}/{EPOCHS}")

        for si in range(steps_per_epoch):
            global_step += 1
            ptr = si * SEQ_LEN
            cur = tokens[:, ptr:ptr+SEQ_LEN]
            tgt = tokens[:, ptr+1:ptr+SEQ_LEN+1]

            optimizer.zero_grad(set_to_none=True)

            gate_s = min(1.0, global_step / 5000) * 0.76 + 0.24

            logits = model(cur, gate_warmup=gate_s)
            loss   = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!! NaN at step {global_step} !! Resetting...")
                optimizer.zero_grad(set_to_none=True)
                model.reset_states(BATCH_SIZE)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)  # Tighter clip
            optimizer.step()
            scheduler.step()

            lv = loss.item()
            model._current_surprise = min(lv, 10.0)
            loss_window.append(lv)

            if global_step % 500 == 0:
                # Periodic state reset: prevents h_prev drift causing explosions
                model.reset_states(BATCH_SIZE)
                print(f"[State Reset] Step {global_step} — temporal memory cleared.")

            if global_step % 10 == 0:
                elapsed  = time.time() - t0
                tok_sec  = global_step * BATCH_SIZE * SEQ_LEN / max(elapsed, 1e-6)
                avg      = sum(loss_window) / len(loss_window)
                lr_now   = optimizer.param_groups[0]['lr']
                print(f"E{epoch+1} S{global_step} | Loss: {lv:.4f} | "
                      f"Avg: {avg:.4f} | LR: {lr_now:.2e} | {tok_sec:.0f} t/s")

            if global_step % 200 == 0:
                torch.save({
                    'step': ptr, 'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': sum(loss_window) / len(loss_window),
                }, CHECKPOINT_PATH)
                print(f"[Saved] step={global_step} avg={sum(loss_window)/len(loss_window):.4f}")

if __name__ == "__main__":
    main()
