"""
agnis_v5_sprint2.py
================================================================
AGNIS V5 | SPRINT 2, SESSION 4 — "The Fluency Breakthrough"
================================================================
ALL 5 BOTTLENECK FIXES APPLIED:
  Fix 1: Untied LM Head (independent output matrix)
  Fix 2: k-WTA removed from core activation (see agnis_v4_core.py)
  Fix 3: Gradient flow into core V matrices via differentiable shadow path
  Fix 4: R_weight updated once per sequence (not 64x per forward)
  Fix 5: MAX_SETTLE_STEPS = 10 (deep feature settling)
Target: Loss < 5.0 in first 5k steps, < 3.5 by end of session.
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
MODEL_NAME     = "agnis_v5_30m_fluency"
SAVE_DIR       = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
CHECKPOINT_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pt")
TOKENIZER_PATH = "slm_bpe_tokenizer_32k.json"

TARGET_TOKENS  = 200_000_000
BATCH_SIZE     = 64          # Reduced for stability with gradient-enabled core
SEQ_LEN        = 64
EPOCHS         = 10

EMBED_DIM      = 768
CORE_HIDDEN    = 3072
VOCAB_SIZE     = 32000

MAX_SETTLE_STEPS = 10        # Fix 5: Deep settling (was 3)
ALPHA          = 0.4
ETA_R_LOCAL    = 0.005
LR             = 8e-5        # Conservative for the now-larger trainable set
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Model ────────────────────────────────────────────────────────
class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072,
                 alpha=0.4, max_steps=10, device="cpu"):
        super().__init__()
        self.device    = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha     = alpha
        self.max_steps = max_steps

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        self.hierarchy = PredictiveHierarchy(
            [embed_dim, hidden_dim, embed_dim], device=device)

        # Freeze recurrent / lateral weights — Hebbian only
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

        # Fix 3: Unfreeze only V (recognition) and b_in for hybrid gradient+Hebbian learning
        for col in self.hierarchy.layers:
            col.V.requires_grad_(True)
            col.b_in.requires_grad_(True)

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

        # Fix 1: UNTIED LM Head — independent output matrix
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
        logits_list = []
        accumulated_dR = None

        for t in range(T):
            idx      = token_ids[:, t]
            emb      = F.normalize(self.embedding(idx), dim=-1)
            h_prev_d = self.h_prev.detach()

            temporal_raw     = torch.matmul(h_prev_d, self.R_weight)
            temporal_context = emb + self.alpha * temporal_raw

            # Hebbian settling (no backprop — updates .data directly inside)
            with torch.no_grad():
                self.hierarchy.infer_and_learn(
                    temporal_context.detach(), max_steps=self.max_steps)

            # Fix 3: Differentiable shadow path through V matrices
            l0, l1 = self.hierarchy.layers[0], self.hierarchy.layers[1]
            h_shadow   = F.gelu(torch.matmul(temporal_context, l0.V) + l0.b_in)
            core_shadow = F.normalize(
                F.gelu(torch.matmul(h_shadow, l1.V) + l1.b_in), dim=-1)

            # Hebbian settled state (stable features, detached)
            core_settled = F.normalize(
                self.hierarchy.layers[-1].x.float().detach(), dim=-1)

            # Blend: shadow = gradients, settled = stability
            core_blended = 0.5 * core_shadow + 0.5 * core_settled

            # Fix 4: Accumulate R_weight delta across tokens
            with torch.no_grad():
                epsilon = core_settled - temporal_raw
                dR_step = torch.bmm(
                    h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                accumulated_dR = dR_step if accumulated_dR is None \
                    else accumulated_dR + dR_step

            core_feat    = self.core_proj(core_blended)
            temporal_feat = self.temporal_proj(temporal_raw)

            gate_input = torch.cat([emb, core_feat], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))

            h_t = ((gate * gate_warmup) * (core_feat + self.alpha * temporal_feat)
                   + (1.0 - gate * gate_warmup) * emb)
            h_t = self.out_norm(h_t)
            self.h_prev = h_t.detach()
            logits_list.append(self.lm_head(h_t))

        # Fix 4: Apply R_weight update ONCE per sequence
        if accumulated_dR is not None:
            with torch.no_grad():
                updated = (0.999 * self.R_weight
                           + ETA_R_LOCAL * self._current_surprise
                           * (accumulated_dR / T)).clamp(-3.0, 3.0)
                self.R_weight.copy_(updated)

        return torch.stack(logits_list, dim=1)  # [B, T, V]


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
    print("  AGNIS V5 — SESSION 4: THE FLUENCY BREAKTHROUGH")
    print(f"  SEQ={SEQ_LEN} | Settle={MAX_SETTLE_STEPS} | LR={LR}")
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
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
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

    ids = ids[:TARGET_TOKENS]
    sl  = len(ids) // BATCH_SIZE
    tokens = torch.tensor(
        ids[:sl*BATCH_SIZE], dtype=torch.long, device=DEVICE
    ).view(BATCH_SIZE, sl)
    del ids; gc.collect()

    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN,
                    alpha=ALPHA, max_steps=MAX_SETTLE_STEPS, device=DEVICE)

    start_step, start_epoch, loaded_ckpt = 0, 0, None
    for path in [CHECKPOINT_PATH, f"{MODEL_NAME}.pt"]:
        if os.path.exists(path):
            print(f"[Resume] Loading from {path}...")
            loaded_ckpt = torch.load(path, map_location=DEVICE)
            sd = loaded_ckpt['model']
            if sd and next(iter(sd)).startswith('module.'):
                sd = {k[7:]: v for k, v in sd.items()}
            md = model.state_dict()
            clean = {k: v for k, v in sd.items()
                     if k in md and v.shape == md[k].shape}
            skipped = len(sd) - len(clean)
            print(f"[Resume] Loaded. Skipped {skipped} mismatched buffers.")
            model.load_state_dict(clean, strict=False)
            model.gate_proj.bias.data.fill_(0.0)
            start_step  = loaded_ckpt.get('step', 0)
            start_epoch = loaded_ckpt.get('epoch', 0)
            break

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"[Model] Trainable parameters: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    if loaded_ckpt and 'optimizer' in loaded_ckpt:
        try:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            print("[Resume] Optimizer state restored.")
        except Exception as e:
            print(f"[Resume] Optimizer reset: {e}")

    model.train()
    model.reset_states(BATCH_SIZE)

    start_block = start_step // SEQ_LEN
    print(f"[Resume] Starting from block {start_block}")

    loss_window     = deque(maxlen=100)
    steps_per_epoch = (tokens.shape[1] - SEQ_LEN - 1) // SEQ_LEN
    global_step     = 0
    t0              = time.time()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEPOCH {epoch+1}/{EPOCHS}")
        init_block = start_block if epoch == start_epoch else 0

        for si in range(init_block, steps_per_epoch):
            global_step += 1
            ptr = si * SEQ_LEN
            cur = tokens[:, ptr:ptr+SEQ_LEN]
            tgt = tokens[:, ptr+1:ptr+SEQ_LEN+1]

            optimizer.zero_grad(set_to_none=True)

            warmup   = min(1.0, (global_step + start_block) / 10000)
            gate_s   = 0.24 + 0.76 * warmup

            logits = model(cur, gate_warmup=gate_s)
            loss   = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!! NaN at block {si} !! Resetting...")
                optimizer.zero_grad(set_to_none=True)
                model.reset_states(BATCH_SIZE)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            lv = loss.item()
            model._current_surprise = min(lv, 10.0)
            loss_window.append(lv)

            if global_step % 10 == 0:
                elapsed  = time.time() - t0
                tok_sec  = global_step * BATCH_SIZE * SEQ_LEN / max(elapsed, 1e-6)
                avg_loss = sum(loss_window) / len(loss_window)
                print(f"E{epoch+1} B{si} | Loss: {lv:.4f} | "
                      f"Avg: {avg_loss:.4f} | Gate: {gate_s:.2f} | {tok_sec:.0f} t/s")

            if global_step % 100 == 0:
                torch.save({
                    'step': ptr, 'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': sum(loss_window) / len(loss_window),
                }, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
