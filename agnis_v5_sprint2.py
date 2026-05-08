"""
agnis_v5_sprint2.py
================================================================
AGNIS V5 | SPRINT 2 — "Bridging the Core"
================================================================

ROOT CAUSE (Sprint 1 failure):
  The Hebbian core output was completely detached from backprop.
  Gradients only flowed through the embedding residual skip.
  The core was effectively invisible — just adding noise.
  
  max_steps=1 vs max_steps=3 didn't matter because backprop
  couldn't see the core at all.

THE FIX:
  Added a trainable "bridge" (core_proj + learned gate) between
  the Hebbian core and the LM head. Now:
  
  - Hebbian core learns features via local rules (unsupervised)
  - core_proj learns to TRANSFORM those features (supervised)
  - gate learns HOW MUCH to trust core vs raw embedding (supervised)
  
  This is how neuroscience-inspired architectures actually work:
  cortical features are extracted unsupervised, but there's a
  trained "readout" layer that maps them to the task.

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
BATCH_SIZE = 64
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

# ─── AGNIS V5 Model (Sprint 2 — with Bridge) ────────────────────
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

        # ═══════════════════════════════════════════════════════════
        # THE FIX: Trainable Bridge between Hebbian core and LM task
        # ═══════════════════════════════════════════════════════════
        
        # 1. Core Projection — learns to extract useful features
        #    from the Hebbian core's unsupervised representations
        self.core_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self.device)
        
        # Small init to prevent float16 overflow at start
        for m in self.core_proj:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 2. Learned Gate — decides how much to trust core vs embedding
        #    Input: [emb; core_feat] → sigmoid gate
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim).to(self.device)
        # Init gate bias negative so gate starts near 0 (trust embedding first)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        
        # 3. Temporal projection (also trainable now)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        nn.init.zeros_(self.temporal_proj.weight)  # Start with zero temporal influence
        
        # Temporal Association (Delta Rule — Hebbian)
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)

        # LM Head (Weight Tying)
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

        # 1. Hebbian Settlement (unsupervised feature extraction)
        with torch.no_grad():
            self.hierarchy.infer_and_learn(emb.detach(), max_steps=self.max_steps)
            core_raw = self.hierarchy.layers[-1].x.clone()
            # Normalize to prevent float16 overflow in bridge
            core_raw = F.normalize(core_raw, dim=-1)
        
        # 2. THE BRIDGE: Trainable projection (HAS gradients!)
        core_feat = self.core_proj(core_raw.float())  # Force float32 through bridge
        
        # 3. Temporal Association
        h_prev_d = self.h_prev.detach()
        with torch.no_grad():
            x_hat = torch.matmul(h_prev_d, self.R_weight)
            epsilon = core_raw - x_hat
            dR = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
            self.R_weight = (0.999 * self.R_weight + ETA_R_LOCAL * self._current_surprise * dR).clamp(-3.0, 3.0)
        
        # Temporal projection (trainable — backprop learns temporal relevance)
        temporal_raw = torch.matmul(h_prev_d, self.R_weight)
        temporal_feat = self.temporal_proj(temporal_raw)
        
        # 4. Gated Fusion: Let backprop decide how much core to trust
        gate_input = torch.cat([emb, core_feat], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        # Fuse: gate * core_features + (1 - gate) * embedding
        h_t = gate * (core_feat + self.alpha * temporal_feat) + (1.0 - gate) * emb
        
        self.h_prev = h_t.detach()
        
        # 5. Output
        fused = self.out_norm(h_t)
        return self.lm_head(fused)
# ─── Data ─────────────────────────────────────────────────────────
def get_multilingual_data():
    try:
        from datasets import load_dataset
        print("[Data] Loading FineWeb-Edu + Wikitext-103...")
        
        en_wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        
        # Load FineWeb-Edu (The current gold standard for clean training data)
        # We take 50,000 rows now that the Batch Tokenizer is proven stable
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
    print("  AGNIS V5 | SPRINT 2 — DUAL GPU EDITION")
    print(f"  Fix: Trainable projection + nn.DataParallel")
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
    
    # Encode in chunks of 5,000 lines to keep Kaggle kernel heartbeat alive
    CHUNK_SIZE = 5000
    ids = []
    total_chunks = (len(lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk = lines[i:i+CHUNK_SIZE]
        encodings = tokenizer.encode_batch(chunk)
        for enc in encodings:
            ids.extend(enc.ids)
        chunk_num = i // CHUNK_SIZE + 1
        if chunk_num % 10 == 0 or chunk_num == total_chunks:
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
    
    # Dual GPU Support
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"[System] Multi-GPU detected! Using {gpu_count} T4 GPUs.")
        model = nn.DataParallel(model)

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
            
            # Handle DataParallel prefixing
            state_dict = loaded_ckpt['model']
            if gpu_count > 1 and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif gpu_count <= 1 and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict, strict=False)
            start_step = loaded_ckpt.get('step', 0)
            start_epoch = loaded_ckpt.get('epoch', 0)
            print(f"[Resume] Step {start_step}, Epoch {start_epoch}")
            break
    
    if loaded_ckpt is None:
        print("[Sprint 2] Starting fresh (no checkpoint found)")

    # ALL trainable parameters (embedding + bridge + norms)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    
    if loaded_ckpt and 'optimizer' in loaded_ckpt:
        try:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            for pg in optimizer.param_groups:
                pg['lr'] = LR
        except Exception:
            print("[Resume] Optimizer incompatible, using fresh")

    param_total = sum(p.numel() for p in model.parameters()) / 1e6
    param_bp = sum(p.numel() for p in trainable) / 1e6
    param_hebb = param_total - param_bp
    
    print(f"[Params] Total: {param_total:.1f}M | Backprop: {param_bp:.1f}M | Hebbian: {param_hebb:.1f}M")
    print(f"[Core]   Settlement: {MAX_SETTLE_STEPS} steps | Bridge: core_proj + gate")
    print(f"[Save]   {CHECKPOINT_PATH}")
    print("-" * 60)

    model.train()
    # Access .module if wrapped in DataParallel
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.reset_states(BATCH_SIZE)
    
    loss_window = deque(maxlen=100)
    best_avg_loss = float('inf')
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
            
            # NaN guard — skip corrupted steps
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                raw_model.reset_states(BATCH_SIZE)
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            
            loss_val = loss.item()
            raw_model._current_surprise = min(loss_val, 10.0)
            loss_window.append(loss_val)

            if (step+1) % 500 == 0:
                elapsed = time.time() - start_time
                tok_sec = (global_step * BATCH_SIZE) / max(elapsed, 1e-6)
                eta_hrs = ((EPOCHS - epoch) * steps_per_epoch - step) * BATCH_SIZE / tok_sec / 3600
                avg_loss = sum(loss_window) / len(loss_window)
                gate_mean = "N/A"  # Will show gate activity later
                print(f"E{epoch+1} Step {step+1}/{steps_per_epoch} | "
                      f"Loss: {loss_val:.4f} | Avg100: {avg_loss:.4f} | "
                      f"{tok_sec:.0f} t/s | ETA: {eta_hrs:.1f}h")

            if (step+1) % 5000 == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                if avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                print(f"[Checkpoint] Saving (Avg: {avg_loss:.4f}, Best: {best_avg_loss:.4f})...")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'best_avg_loss': best_avg_loss,
                    'max_steps': MAX_SETTLE_STEPS,
                }, CHECKPOINT_PATH)
        
        epoch_time = time.time() - start_time
        avg_loss = sum(loss_window) / len(loss_window) if loss_window else float('inf')
        print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Best: {best_avg_loss:.4f} | "
              f"Time: {epoch_time/3600:.1f}h")
        
        torch.save({
            'step': 0, 'epoch': epoch + 1, 'global_step': global_step,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'avg_loss': avg_loss, 'best_avg_loss': best_avg_loss,
            'max_steps': MAX_SETTLE_STEPS,
        }, CHECKPOINT_PATH)
        
        if avg_loss < 4.0:
            print(f"\n🎯 TARGET REACHED! Loss {avg_loss:.4f} < 4.0 — Ready for Sprint 3")
            break

    print(f"\n[Done] Final: {CHECKPOINT_PATH} | Time: {(time.time()-start_time)/3600:.1f}h")

if __name__ == "__main__":
    main()
