"""
agnis_v5_sprint2.py
================================================================
AGNIS V5 | SPRINT 2, SESSION 2 — "The Breakthrough"
================================================================

Configuration: "Aggressive Stable" (User-Optimized)
--------------------------------------------------
1. Gate Bias: 0.0 (Opened to 50% influence)
2. Hebbian LR: 0.005 (2.5x boost for faster core evolution)
3. Alpha: 0.4 (Stronger temporal word-word associations)
4. Backprop LR: 2.1e-4 (Reduced 30% to force core reliance)
5. Gate Warmup: 10k step linear transition from 0.12 to 0.50

Target: Break the 7.7 plateau and reach Loss < 5.0
"""

import math
import os
import sys
import time
import gc
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from agnis_v4_core import PredictiveHierarchy

# ─── Config ───────────────────────────────────────────────────────
MODEL_NAME = "agnis_v5_30m_fluency"

SAVE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
CHECKPOINT_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pt")
TOKENIZER_PATH = "slm_bpe_tokenizer_32k.json"

# Data
TARGET_TOKENS = 200_000_000 
BATCH_SIZE = 128
EPOCHS = 10

# Architecture
EMBED_DIM = 768
CORE_HIDDEN = 3072
VOCAB_SIZE = 32000

# Core Settlement
MAX_SETTLE_STEPS = 3

# --- Fix 3: Stronger temporal memory (Alpha 0.1 -> 0.4) ---
ALPHA = 0.4

# --- Fix 2: Moderate Hebbian boost (0.002 -> 0.005) ---
ETA_R_LOCAL = 0.005

# --- Fix 4: Reduced Backprop LR (3e-4 * 0.7 = 2.1e-4) ---
LR = 2.1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── AGNIS V5 Model (Sprint 2 — Aggressive Stable) ─────────────
class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072, 
                 alpha=0.4, max_steps=3, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.max_steps = max_steps

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        
        self.hierarchy = PredictiveHierarchy([embed_dim, hidden_dim, embed_dim], device=device)
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

        self.core_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self.device)
        
        for m in self.core_proj:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim).to(self.device)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        # --- Fix 1: Gate Bias 0.0 ---
        nn.init.constant_(self.gate_proj.bias, 0.0)
        
        self.temporal_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        nn.init.zeros_(self.temporal_proj.weight)
        
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)

        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        self.lm_head.weight = self.embedding.weight

        self.register_buffer("h_prev", torch.zeros(1, embed_dim))
        self._current_surprise = 1.0

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.device)
        self._current_surprise = 1.0

    def forward(self, token_ids, gate_warmup=1.0):
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        with torch.no_grad():
            self.hierarchy.infer_and_learn(emb.detach(), max_steps=self.max_steps)
            core_raw = self.hierarchy.layers[-1].x.clone()
            core_raw = F.normalize(core_raw, dim=-1)
        
        core_feat = self.core_proj(core_raw.float())
        
        h_prev_d = self.h_prev.detach()
        with torch.no_grad():
            x_hat = torch.matmul(h_prev_d, self.R_weight)
            epsilon = core_raw - x_hat
            dR = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
            self.R_weight = (0.999 * self.R_weight + ETA_R_LOCAL * self._current_surprise * dR).clamp(-3.0, 3.0)
        
        temporal_raw = torch.matmul(h_prev_d, self.R_weight)
        temporal_feat = self.temporal_proj(temporal_raw)
        
        gate_input = torch.cat([emb, core_feat], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        # --- Fix 5: Apply Gate Warmup Scale ---
        # gate_warmup goes 0.12 -> 1.0 internally via the training loop call
        h_t = (gate * gate_warmup) * (core_feat + self.alpha * temporal_feat) + (1.0 - (gate * gate_warmup)) * emb
        
        self.h_prev = h_t.detach()
        fused = self.out_norm(h_t)
        return self.lm_head(fused)

# ─── Data ─────────────────────────────────────────────────────────
def get_multilingual_data():
    try:
        from datasets import load_dataset
        print("[Data] Loading FineWeb-Edu + Wikitext-103...")
        en_wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train[:50000]")
        text = "\n".join([t for t in en_wiki["text"] if len(t.strip()) > 20])
        text += "\n" + "\n".join([t for t in fw["text"] if len(t.strip()) > 20])
        return text
    except ImportError:
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("  AGNIS V5 | SPRINT 2 — THE BREAKTHROUGH (Aggressive Stable)")
    print(f"  Fixes: Bias=0.0 | HebbianLR={ETA_R_LOCAL} | Alpha={ALPHA} | LR={LR}")
    print("="*60)

    raw_text = get_multilingual_data()

    if not os.path.exists(TOKENIZER_PATH):
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        tok = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<pad>","<s>","</s>","<unk>"])
        tok.train_from_iterator(raw_text.splitlines(), trainer=trainer)
        tok.save(TOKENIZER_PATH)
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    lines = raw_text.splitlines()
    del raw_text; gc.collect()
    
    CHUNK_SIZE = 5000
    ids = []
    total_chunks = (len(lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk = lines[i:i+CHUNK_SIZE]
        encodings = tokenizer.encode_batch(chunk)
        for enc in encodings: ids.extend(enc.ids)
        del encodings
    
    del lines; gc.collect()
    ids = ids[:TARGET_TOKENS]
    sl = len(ids) // BATCH_SIZE
    tokens = torch.tensor(ids[:sl*BATCH_SIZE], dtype=torch.long, device=DEVICE).view(BATCH_SIZE, sl)
    del ids; gc.collect()

    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN, alpha=ALPHA, max_steps=MAX_SETTLE_STEPS, device=DEVICE)
    
    start_step = 0
    start_epoch = 0
    loaded_ckpt = None
    
    for path in [CHECKPOINT_PATH, f"{MODEL_NAME}.pt", f"/kaggle/input/{MODEL_NAME}/{MODEL_NAME}.pt"]:
        if os.path.exists(path):
            print(f"[Resume] Loading checkpoint from {path}...")
            loaded_ckpt = torch.load(path, map_location=DEVICE)
            state_dict = loaded_ckpt['model']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            
            # --- OVERRIDE BIAS ON RESUME ---
            model.gate_proj.bias.data.fill_(0.0)
            print("[Resume] Force override gate_proj.bias to 0.0")
            
            start_step = loaded_ckpt.get('step', 0)
            start_epoch = loaded_ckpt.get('epoch', 0)
            break
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)

    model.train()
    model.reset_states(BATCH_SIZE)
    
    loss_window = deque(maxlen=100)
    steps_per_epoch = tokens.shape[1] - 1
    global_step = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEPOCH {epoch+1}/{EPOCHS}")
        
        for step in range(steps_per_epoch):
            global_step += 1
            if epoch == start_epoch and step < (start_step % steps_per_epoch):
                continue
            
            cur, tgt = tokens[:, step], tokens[:, step+1]
            optimizer.zero_grad(set_to_none=True)
            
            # --- Fix 5: Linear Gate Warmup Logic ---
            # Goes from 0.12 (bias -2) to 1.0 (bias 0) over 10,000 steps
            warmup_steps = 10000
            current_warmup_step = global_step - (start_epoch * steps_per_epoch + start_step)
            gate_factor = min(1.0, current_warmup_step / warmup_steps)
            # gate_scale: 0.24 (approx 12%) -> 1.0 (approx 50% relative)
            gate_scale = 0.24 + (1.0 - 0.24) * gate_factor
            
            logits = model(cur, gate_warmup=gate_scale)
            loss = F.cross_entropy(logits, tgt)
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                model.reset_states(BATCH_SIZE)
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            
            loss_val = loss.item()
            model._current_surprise = min(loss_val, 10.0)
            loss_window.append(loss_val)

            if (step+1) % 500 == 0:
                elapsed = time.time() - start_time
                tok_sec = (global_step * BATCH_SIZE) / max(elapsed, 1e-6)
                avg_loss = sum(loss_window) / len(loss_window)
                print(f"E{epoch+1} S{step+1} | Loss: {loss_val:.4f} | Avg: {avg_loss:.4f} | Gate: {gate_scale:.2f} | {tok_sec:.0f} t/s")

            if (step+1) % 5000 == 0:
                torch.save({
                    'step': step, 'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': sum(loss_window)/len(loss_window),
                }, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
