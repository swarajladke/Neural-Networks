"""
agnis_v5_sprint2.py
================================================================
AGNIS V5 | SPRINT 2 — "High-Throughput Single-GPU Edition"
================================================================

Fix for DataParallel/Temporal State Conflict:
  Multi-GPU (DataParallel) splits the batch, which destroys the 
  persistent h_prev (temporal association) across steps. 

The Solution:
  - Back to Single GPU for temporal integrity.
  - BATCH_SIZE = 128 (Single T4 can handle this at 67M params).
  - This preserves the 3,400 tokens/sec speed without state corruption.
  - 50,000 rows of FineWeb-Edu for data density.

Target: Loss < 4.0 within 12 hours on T4
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
BATCH_SIZE = 128  # Double the batch for double the speed
EPOCHS = 10

# Architecture
EMBED_DIM = 768
CORE_HIDDEN = 3072
VOCAB_SIZE = 32000

# Core Settlement
MAX_SETTLE_STEPS = 3

# Learning
LR = 3e-4
ALPHA = 0.1
ETA_R_LOCAL = 0.002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── AGNIS V5 Model (Sprint 2 — Stable Version) ─────────────────
class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072, 
                 alpha=0.1, max_steps=3, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.max_steps = max_steps

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        
        # Predictive Hierarchy (Hebbian — no backprop)
        self.hierarchy = PredictiveHierarchy([embed_dim, hidden_dim, embed_dim], device=device)
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

        # Bridge Layers
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
        nn.init.constant_(self.gate_proj.bias, -2.0)
        
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

    def forward(self, token_ids):
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
        
        h_t = gate * (core_feat + self.alpha * temporal_feat) + (1.0 - gate) * emb
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
        
        print(f"[Data] Total Raw Text: {len(text)/1024/1024:.1f} MB")
        return text
    except ImportError:
        print("[Error] pip install datasets zstandard")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("  AGNIS V5 | SPRINT 2 — HIGH THROUGHPUT")
    print(f"  Single-GPU Batch Size: {BATCH_SIZE}")
    print(f"  Settlement: max_steps={MAX_SETTLE_STEPS}")
    print(f"  Target: Loss < 4.0")
    print("="*60)

    raw_text = get_multilingual_data()

    # Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print("[Tokenizer] Training 32k BPE...")
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
    print("[Tokenizer] Encoding corpus in chunks (prevents IOStream timeout)...")
    
    lines = raw_text.splitlines()
    del raw_text; gc.collect()
    
    CHUNK_SIZE = 5000
    ids = []
    total_chunks = (len(lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk = lines[i:i+CHUNK_SIZE]
        encodings = tokenizer.encode_batch(chunk)
        for enc in encodings:
            ids.extend(enc.ids)
        chunk_num = i // CHUNK_SIZE + 1
        if chunk_num % 20 == 0 or chunk_num == total_chunks:
            print(f"[Tokenizer] Chunk {chunk_num}/{total_chunks} | Tokens so far: {len(ids)/1e6:.1f}M")
        del encodings
    
    del lines; gc.collect()
    
    ids = ids[:TARGET_TOKENS]
    sl = len(ids) // BATCH_SIZE
    tokens = torch.tensor(ids[:sl*BATCH_SIZE], dtype=torch.long, device=DEVICE).view(BATCH_SIZE, sl)
    del ids; gc.collect()
    
    total_tokens = tokens.numel()
    print(f"[Data] {total_tokens/1e6:.1f}M tokens | {sl} steps/epoch | {EPOCHS} epochs")

    # Model
    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN, 
                    max_steps=MAX_SETTLE_STEPS, device=DEVICE)
    
    # Checkpoint resume
    start_step = 0
    start_epoch = 0
    loaded_ckpt = None
    
    for path in [CHECKPOINT_PATH, f"{MODEL_NAME}.pt",
                 f"/kaggle/input/{MODEL_NAME}/{MODEL_NAME}.pt",
                 f"/kaggle/input/agnis-sprint1/{MODEL_NAME}.pt"]:
        if os.path.exists(path):
            print(f"[Resume] Loading checkpoint from {path}...")
            loaded_ckpt = torch.load(path, map_location=DEVICE)
            state_dict = loaded_ckpt['model']
            # Clean DataParallel prefix if present
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            start_step = loaded_ckpt.get('step', 0)
            start_epoch = loaded_ckpt.get('epoch', 0)
            print(f"[Resume] Step {start_step}, Epoch {start_epoch}")
            break
    
    if loaded_ckpt is None:
        print("[Sprint 2] Starting fresh")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    
    param_total = sum(p.numel() for p in model.parameters()) / 1e6
    param_bp = sum(p.numel() for p in trainable) / 1e6
    print(f"[Params] Total: {param_total:.1f}M | Backprop: {param_bp:.1f}M")
    print(f"[Save]   Checkpoints → {CHECKPOINT_PATH}")
    print("-" * 60)

    model.train()
    model.reset_states(BATCH_SIZE)
    
    loss_window = deque(maxlen=100)
    steps_per_epoch = tokens.shape[1] - 1
    global_step = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        for step in range(steps_per_epoch):
            global_step += 1
            if epoch == start_epoch and step < (start_step % steps_per_epoch):
                continue
            
            cur, tgt = tokens[:, step], tokens[:, step+1]
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(cur)
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
                eta_hrs = ((EPOCHS - epoch) * steps_per_epoch - step) * BATCH_SIZE / tok_sec / 3600
                avg_loss = sum(loss_window) / len(loss_window)
                print(f"E{epoch+1} Step {step+1}/{steps_per_epoch} | "
                      f"Loss: {loss_val:.4f} | Avg100: {avg_loss:.4f} | "
                      f"{tok_sec:.0f} t/s | ETA: {eta_hrs:.1f}h")

            if (step+1) % 5000 == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                }, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
