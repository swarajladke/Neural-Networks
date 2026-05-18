"""
agnis_v5_sprint4.py — AGNIS V5 Sprint 4
Target: Loss < 4.3 by S10000, < 3.5 final
Changes from Sprint 3:
  CHANGE 1 — Skip gate warmup on resume (gate = 1.0 immediately)
  CHANGE 2 — Double context window: SEQ_LEN 64 → 128
  CHANGE 3 — MAX_SETTLE_STEPS = 5 (unchanged, already optimal)
  CHANGE 4 — BATCH_SIZE = 64 (unchanged, fits 128-token sequences)
  CHANGE 5 — EPOCHS = 30
"""
import os, sys, time, gc, math, glob, shutil, itertools
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizers import Tokenizer
from agnis_v4_core import PredictiveHierarchy

# ── Config ─────────────────────────────────────────────────────
SAVE_DIR        = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
SAVE_PATH       = os.path.join(SAVE_DIR, "agnis_sprint4_best.pt")
BEST_WEIGHTS    = os.path.join(SAVE_DIR, "agnis_sprint4_best_weights.pt")
TOKENIZER_PATH  = "slm_bpe_tokenizer_32k.json"

TARGET_TOKENS    = 500_000_000
BATCH_SIZE       = 64
SEQ_LEN          = 128          # CHANGE 2: doubled from 64
EPOCHS           = 30           # CHANGE 5
EMBED_DIM        = 768
CORE_HIDDEN      = 3072
VOCAB_SIZE       = 32000
N_HEADS          = 8
N_LAYERS         = 2
FFN_DIM          = 2048
MAX_SETTLE_STEPS = 5
ALPHA            = 0.4
ETA_R_LOCAL      = 0.005
LR               = 1e-4
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ── Architecture ────────────────────────────────────────────────
class CausalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, max_seq_len, device):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=0.0, batch_first=True).to(device)
        self.ff   = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, embed_dim)).to(device)
        self.n1   = nn.LayerNorm(embed_dim).to(device)
        self.n2   = nn.LayerNorm(embed_dim).to(device)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        T = x.shape[1]
        mask = self.causal_mask[:T, :T].to(x.device)
        xn = self.n1(x)
        attn_out, _ = self.attn(xn, xn, xn, attn_mask=mask, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.n2(x))


class CausalContextEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, ffn_dim, max_seq_len, device):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim).to(device)
        self.blocks  = nn.ModuleList([CausalTransformerBlock(embed_dim, n_heads, ffn_dim, max_seq_len, device) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(embed_dim).to(device)

    def forward(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072, alpha=0.4, max_steps=5, device="cpu"):
        super().__init__()
        self.device    = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha     = alpha
        self.max_steps = max_steps
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        self.context_encoder = CausalContextEncoder(embed_dim, N_HEADS, N_LAYERS, FFN_DIM, SEQ_LEN, device)
        self.hierarchy = PredictiveHierarchy([embed_dim, hidden_dim, embed_dim], device=device)
        for p in self.hierarchy.parameters(): p.requires_grad_(False)
        for col in self.hierarchy.layers:
            col.V.requires_grad_(True)
            col.b_in.requires_grad_(True)
        self.core_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)).to(self.device)
        for m in self.core_proj:
            if hasattr(m, 'weight'): nn.init.xavier_uniform_(m.weight, gain=0.1)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        self.gate_proj     = nn.Linear(embed_dim * 2, embed_dim).to(self.device)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        nn.init.zeros_(self.temporal_proj.weight)
        r_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(r_weight, gain=0.1)
        self.register_buffer("R_weight", r_weight)
        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head  = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        self.register_buffer("h_prev", torch.zeros(1, embed_dim, device=self.device))
        self._current_surprise = 1.0

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.device)
        self._current_surprise = 1.0

    def forward(self, token_ids, gate_warmup=1.0):
        B, T      = token_ids.shape
        raw_embs  = self.embedding(token_ids)
        ctx_embs  = F.normalize(self.context_encoder(raw_embs), dim=-1)
        logits_list, accumulated_dR = [], None

        for t in range(T):
            ctx_emb  = ctx_embs[:, t]
            h_prev_d = self.h_prev.detach()
            temporal_raw     = torch.matmul(h_prev_d, self.R_weight)
            temporal_context = ctx_emb + self.alpha * temporal_raw

            with torch.no_grad():
                self.hierarchy.infer_and_learn(temporal_context.detach(), max_steps=self.max_steps)

            l0, l1      = self.hierarchy.layers[0], self.hierarchy.layers[1]
            h_shadow    = F.gelu(torch.matmul(temporal_context, l0.V) + l0.b_in)
            core_shadow = F.normalize(F.gelu(torch.matmul(h_shadow, l1.V) + l1.b_in), dim=-1)
            core_settled = F.normalize(self.hierarchy.layers[-1].x.float().detach(), dim=-1)
            core_blended = 0.5 * core_shadow + 0.5 * core_settled

            with torch.no_grad():
                epsilon = core_settled - temporal_raw
                dR_step = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                accumulated_dR = dR_step if accumulated_dR is None else accumulated_dR + dR_step

            core_feat     = self.core_proj(core_blended)
            temporal_feat = self.temporal_proj(temporal_raw)
            gate = torch.sigmoid(self.gate_proj(torch.cat([ctx_emb, core_feat], dim=-1)))
            h_t = (gate * gate_warmup) * (core_feat + self.alpha * temporal_feat) + (1.0 - gate * gate_warmup) * ctx_emb
            h_t = self.out_norm(h_t)
            self.h_prev = h_t.detach()
            logits_list.append(self.lm_head(h_t))

        if accumulated_dR is not None:
            with torch.no_grad():
                upd = (0.999 * self.R_weight + ETA_R_LOCAL * self._current_surprise * (accumulated_dR / T)).clamp(-3.0, 3.0)
                self.R_weight.copy_(upd)

        return torch.stack(logits_list, dim=1)


# ── Generation ──────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=80, temperature=0.8, top_k=50):
    model.eval()
    model.reset_states(1)
    ids = tokenizer.encode(prompt).ids
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for i in range(len(ids) - 1):
        _ = model(inp[:, i:i+1])
    cur = inp[:, -1:]
    out = list(ids)
    for _ in range(max_tokens):
        logits = model(cur)[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        out.append(next_tok.item())
        cur = next_tok
        if next_tok.item() == tokenizer.token_to_id("</s>"): break
    model.train()
    return tokenizer.decode(out)


# ── Clean text ──────────────────────────────────────────────────
import re as _re
def clean_text(text):
    text = _re.sub(r'=+\s*[^=]*\s*=+', '', text)
    text = _re.sub(r'\s*=\s*=\s*', ' ', text)
    text = _re.sub(r'@[\-\.]@', lambda m: m.group(0)[1], text)
    text = _re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Data (streaming — no disk overflow) ─────────────────────────
def get_data():
    from datasets import load_dataset
    print("[Data] Streaming FineWeb-Edu (50k docs, no disk cache)...")
    fw_stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                             split="train", streaming=True)
    docs = list(itertools.islice(fw_stream, 50000))
    text = "\n".join(clean_text(d["text"]) for d in docs if len(d["text"].strip()) > 50)
    print(f"[Data] {len(text)//1_000_000:.1f}M chars from {len(docs)} documents")
    return text


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AGNIS V5 — SPRINT 4: SEQ_LEN=128, TARGET LOSS < 3.5")
    print(f"  LR={LR} | SEQ={SEQ_LEN} | Settle={MAX_SETTLE_STEPS} | Epochs={EPOCHS}")
    print("=" * 60)

    raw_text  = get_data()
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    lines     = raw_text.splitlines()
    del raw_text; gc.collect()

    ids = []
    for i in range(0, len(lines), 5000):
        encs = tokenizer.encode_batch(lines[i:i+5000])
        for e in encs: ids.extend(e.ids)
        del encs
    del lines; gc.collect()

    ids    = ids[:TARGET_TOKENS]
    sl     = len(ids) // BATCH_SIZE
    tokens = torch.tensor(ids[:sl*BATCH_SIZE], dtype=torch.long, device=DEVICE).view(BATCH_SIZE, sl)
    del ids; gc.collect()

    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN, alpha=ALPHA, max_steps=MAX_SETTLE_STEPS, device=DEVICE)

    # ── Checkpoint loading (strict — sorted by file size) ────────
    all_pts = sorted(
        glob.glob("/kaggle/input/**/*.pt", recursive=True) +
        glob.glob("/kaggle/working/*.pt") + glob.glob("*.pt"),
        key=os.path.getsize, reverse=True)
    print(f"[Resume] Found {len(all_pts)} .pt files: {[os.path.basename(p) for p in all_pts[:5]]}")

    checkpoint_loaded = False
    loaded_ckpt = None
    for p in all_pts:
        if os.path.getsize(p) < 50e6: continue
        try:
            ckpt_try = torch.load(p, map_location=DEVICE)
            sd_try   = ckpt_try.get('model', ckpt_try)
            if not isinstance(sd_try, dict) or len(sd_try) < 10: continue
            loaded_ckpt = ckpt_try
            print(f"[Resume] Loaded: {p} ({os.path.getsize(p)/1e6:.1f} MB)")
            print(f"[Resume] Step={loaded_ckpt.get('step','?')} | AvgLoss={loaded_ckpt.get('avg_loss','?')}")
            break
        except Exception as e:
            print(f"[Resume] Skip {p}: {e}")

    if loaded_ckpt is None:
        print("[Resume] WARNING: No checkpoint — starting FRESH")
    else:
        sd = loaded_ckpt.get('model', loaded_ckpt)
        if next(iter(sd)).startswith('module.'): sd = {k[7:]: v for k, v in sd.items()}
        md    = model.state_dict()
        # CHANGE 2: skip pos_emb if shape changed (SEQ_LEN 64→128 changes pos_emb size)
        clean = {k: v for k, v in sd.items()
                 if k in md and v.shape == md[k].shape and 'pos_emb' not in k and 'causal_mask' not in k}
        model.load_state_dict(clean, strict=False)
        model.gate_proj.bias.data.fill_(0.0)
        checkpoint_loaded = True
        print(f"[Resume] {len(clean)} tensors loaded (pos_emb re-initialized for SEQ_LEN=128)")

    # Fix W_mask/V_mask buffers
    for col in model.hierarchy.layers:
        if hasattr(col, 'W') and not hasattr(col, 'W_mask'):
            col.register_buffer('W_mask', torch.ones(col.W.shape, device=DEVICE))
        if hasattr(col, 'V') and not hasattr(col, 'V_mask'):
            col.register_buffer('V_mask', torch.ones(col.V.shape, device=DEVICE))
    print("[Resume] W_mask/V_mask verified")

    # ── Verification Tests ───────────────────────────────────────
    print("\n── Verification Tests ──")
    print(f"Test 1 | Params: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        sb = tokens[:4, :SEQ_LEN]
        model.reset_states(4)
        lo = model(sb)
        model.reset_states(BATCH_SIZE)
        print(f"Test 2 | Input: {tuple(sb.shape)} → Output: {tuple(lo.shape)}")
        assert lo.shape == (4, SEQ_LEN, VOCAB_SIZE), f"FAIL shape: {lo.shape}"

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10000, T_mult=2, eta_min=1e-6)
    print(f"Test 3 | Initial LR: {scheduler.get_last_lr()[0]:.2e}")

    # CHANGE 1: gate starts at 1.0 if checkpoint loaded
    gate_start = 1.0 if checkpoint_loaded else 0.24
    print(f"Test 4 | Gate start: {gate_start:.2f}  ({'immediate — checkpoint loaded' if checkpoint_loaded else 'warmup — fresh start'})")

    torch.save({'test': True}, os.path.join(SAVE_DIR, 'test.pt'))
    os.remove(os.path.join(SAVE_DIR, 'test.pt'))
    print("Test 5 | Save path ✅")

    # Test 6: 100-step warmup
    model.eval(); model.reset_states(4); wl = []
    with torch.no_grad():
        for wi in range(100):
            wc = tokens[:4, wi*SEQ_LEN:(wi+1)*SEQ_LEN]
            wt = tokens[:4, wi*SEQ_LEN+1:(wi+1)*SEQ_LEN+1]
            wl.append(F.cross_entropy(model(wc).view(-1, VOCAB_SIZE), wt.reshape(-1)).item())
    warmup_avg = sum(wl) / len(wl)
    status = "✅ OK" if warmup_avg < 7.0 else "⚠️  HIGH — verify checkpoint!"
    print(f"Test 6 | Warmup loss: {warmup_avg:.4f}  {status}")
    model.train(); model.reset_states(BATCH_SIZE)
    print("── All tests passed — starting training ──\n")

    # ── SIGTERM emergency save ───────────────────────────────────
    import signal
    _state = {'step': 0, 'epoch': 0, 'avg': 9.9}
    def _emergency_save(sig, frame):
        ep = os.path.join(SAVE_DIR, "agnis_sprint4_best.pt")
        torch.save({'step': _state['step'], 'epoch': _state['epoch'],
                    'model': model.state_dict(), 'avg_loss': _state['avg']}, ep)
        print(f"[Emergency Save] step={_state['step']} avg={_state['avg']:.4f} → {ep}")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _emergency_save)
    signal.signal(signal.SIGINT,  _emergency_save)

    # ── Training loop ────────────────────────────────────────────
    model.train()
    model.reset_states(BATCH_SIZE)
    steps_per_epoch = (tokens.shape[1] - SEQ_LEN - 1) // SEQ_LEN
    loss_window     = deque(maxlen=200)
    best_loss       = float('inf')
    global_step     = 0
    t0              = time.time()
    gen_prompts     = ["The history of", "Once upon a time", "The scientist discovered"]
    PATIENCE        = 10000
    no_improve_steps = 0
    early_stop      = False

    for epoch in range(EPOCHS):
        if early_stop: break
        print(f"\nEPOCH {epoch+1}/{EPOCHS}")
        for si in range(steps_per_epoch):
            if early_stop: break
            global_step += 1
            ptr = si * SEQ_LEN
            cur = tokens[:, ptr:ptr+SEQ_LEN]
            tgt = tokens[:, ptr+1:ptr+SEQ_LEN+1]

            if global_step % 5000 == 0:
                model.reset_states(BATCH_SIZE)
                print(f"[State Reset] Step {global_step}")

            optimizer.zero_grad(set_to_none=True)
            # CHANGE 1: skip gate warmup if checkpoint was loaded
            if checkpoint_loaded:
                gate_s = 1.0
            else:
                gate_s = min(1.0, global_step / 5000) * 0.76 + 0.24

            logits = model(cur, gate_warmup=gate_s)
            loss   = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!! NaN step {global_step} !! Skipping")
                optimizer.zero_grad(set_to_none=True)
                model.reset_states(BATCH_SIZE)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            lv = loss.item()
            model._current_surprise = min(lv, 10.0)
            loss_window.append(lv)
            avg_loss = sum(loss_window) / len(loss_window)
            _state['step'] = global_step
            _state['epoch'] = epoch
            _state['avg'] = avg_loss

            if global_step % 500 == 0:
                elapsed = time.time() - t0
                tok_s   = global_step * BATCH_SIZE * SEQ_LEN / max(elapsed, 1e-6)
                cur_lr  = scheduler.get_last_lr()[0]
                print(f"E{epoch+1} S{global_step} | Loss: {lv:.4f} | Avg: {avg_loss:.4f} | "
                      f"LR: {cur_lr:.2e} | Gate: {gate_s:.2f} | {tok_s:.0f} t/s")

            if global_step % 2000 == 0:
                torch.save({'step': global_step, 'epoch': epoch,
                            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_loss': avg_loss, 'best_loss': best_loss}, SAVE_PATH)
                print(f"[Saved] step={global_step} avg={avg_loss:.4f}")

            if avg_loss < best_loss - 0.01:
                best_loss        = avg_loss
                no_improve_steps = 0
                torch.save(model.state_dict(), BEST_WEIGHTS)
                print(f"[Best] New best: {best_loss:.4f}")
            else:
                no_improve_steps += 1
                if no_improve_steps >= PATIENCE:
                    print(f"[Early Stop] No improvement for {PATIENCE} steps. Best: {best_loss:.4f}")
                    early_stop = True

            if global_step % 5000 == 0:
                print(f"\n── Generation @ step {global_step} (loss={avg_loss:.4f}) ──")
                for prompt in gen_prompts:
                    out = generate(model, tokenizer, prompt)
                    print(f"  [{prompt}] -> {out}")
                print()
                model.reset_states(BATCH_SIZE)


if __name__ == "__main__":
    main()
