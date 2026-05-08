"""
agnis_v5_sprint2.py
================================================================
AGNIS V5 | SPRINT 2 — "Awakening the Core"
================================================================

Sprint 1 Result:  Loss plateau ~5.8 with max_steps=1
Sprint 2 Target:  Loss < 4.5 with max_steps=3

The Key Insight:
  max_steps=1 = AGNIS acts like a simple feedforward network
  max_steps=3 = AGNIS starts acting like a predictive coding hierarchy
  
  Everything that makes AGNIS biologically interesting
  only activates at max_steps > 1.
  
  Sprint 2 is where AGNIS actually becomes AGNIS.

Changes from Sprint 1:
  1. max_steps = 3 (up from 1)
  2. LR = 2.8e-4 (down 30% from 4e-4 to reduce oscillation)
  3. Rolling average loss over 100 steps
  4. Checkpoint saved to /kaggle/working/ explicitly
  5. Loads Sprint 1 checkpoint automatically
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

# CHANGE 5: Save explicitly to /kaggle/working/ so Output tab shows file sizes
SAVE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
CHECKPOINT_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pt")
TOKENIZER_PATH = "slm_bpe_tokenizer_32k.json"

# Data Scaling
TARGET_TOKENS = 200_000_000 
BATCH_SIZE = 64
EPOCHS = 10  # 10 epochs over the dataset

# 30M Parameter Scale (The "Sweet Spot")
EMBED_DIM = 768
CORE_HIDDEN = 3072
VOCAB_SIZE = 32000

# CHANGE 1: Settlement depth — THE critical change
MAX_SETTLE_STEPS = 3  # Up from 1. This is where AGNIS becomes AGNIS.

# CHANGE 3: Reduced LR by 30% to dampen loss oscillation
LR = 2.8e-4  # Down from 4e-4
ALPHA = 0.1
ETA_R_LOCAL = 0.002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── AGNIS V5 Model (Sprint 2) ──────────────────────────────────
class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072, 
                 alpha=0.1, max_steps=3, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.max_steps = max_steps  # Settlement depth

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        
        # Predictive Hierarchy (Unfrozen)
        self.hierarchy = PredictiveHierarchy([embed_dim, hidden_dim, embed_dim], device=device)
        
        # Disable gradients for the core (it uses Hebbian rules)
        for p in self.hierarchy.parameters():
            p.requires_grad_(False)

        # Temporal Association (Delta Rule)
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)
        self.r_norm = nn.LayerNorm(embed_dim).to(self.device)

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

    def step_logits(self, token_ids):
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 1. Hebbian Settlement (Core Learning)
        # CHANGE 1: max_steps=3 — the core now actually settles
        with torch.no_grad():
            self.hierarchy.infer_and_learn(emb.detach(), max_steps=self.max_steps)
            core_out = self.hierarchy.layers[-1].x
        
        core_out = F.normalize(core_out, dim=-1)
        h_prev_d = self.h_prev.detach()

        # 2. Delta Rule Temporal Mapping
        if self.alpha > 0.0:
            with torch.no_grad():
                x_hat = torch.matmul(h_prev_d, self.R_weight)
                epsilon = core_out.detach() - x_hat
                dR = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                self.R_weight = (0.999 * self.R_weight + ETA_R_LOCAL * self._current_surprise * dR).clamp(-3.0, 3.0)

            temporal = torch.matmul(h_prev_d, self.R_weight)
            h_t = core_out + self.alpha * temporal
            h_t = self.r_norm(h_t)
        else:
            h_t = core_out

        self.h_prev = h_t.detach()
        
        # 3. Residual Skip Output
        fused = self.out_norm(h_t + emb)
        return self.lm_head(fused)

# ─── Data & Utility ───────────────────────────────────────────────
def get_multilingual_data():
    try:
        from datasets import load_dataset
        print("[Data] Loading FineWeb-Edu (High Quality) + Wikitext-103...")
        
        en_wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train[:10000]")
        
        text = "\n".join([t for t in en_wiki["text"] if len(t.strip()) > 20])
        text += "\n" + "\n".join([t for t in fw["text"] if len(t.strip()) > 20])
        
        print(f"[Data] Total Raw Text: {len(text)/1024/1024:.1f} MB")
        return text
    except ImportError:
        print("[Error] pip install datasets zstandard")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("  AGNIS V5 | SPRINT 2 — AWAKENING THE CORE")
    print(f"  Settlement Depth: max_steps={MAX_SETTLE_STEPS}")
    print(f"  Learning Rate: {LR} (reduced 30%)")
    print(f"  Target: Loss < 4.5")
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
        tok.pre_tokenizer = ByteLevel(); tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<pad>","<s>","</s>","<unk>"])
        tok.train_from_iterator(raw_text.splitlines(), trainer=trainer)
        tok.save(TOKENIZER_PATH)
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print("[Tokenizer] Encoding corpus...")
    
    lines = raw_text.splitlines()
    del raw_text
    gc.collect()
    
    encodings = tokenizer.encode_batch(lines)
    del lines
    gc.collect()
    
    ids = []
    for enc in encodings:
        ids.extend(enc.ids)
    del encodings
    gc.collect()
    
    ids = ids[:TARGET_TOKENS]
    
    sl = len(ids) // BATCH_SIZE
    tokens = torch.tensor(ids[:sl*BATCH_SIZE], dtype=torch.long, device=DEVICE).view(BATCH_SIZE, sl)
    del ids
    gc.collect()
    
    total_tokens = tokens.numel()
    print(f"[Data] {total_tokens/1e6:.1f}M tokens loaded | {sl} steps/epoch | {EPOCHS} epochs")

    # Model — with Sprint 2 settlement depth
    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN, 
                    max_steps=MAX_SETTLE_STEPS, device=DEVICE)
    
    # CHANGE 2: Load from Sprint 1 checkpoint
    start_step = 0
    start_epoch = 0
    
    # Search for checkpoint in multiple locations
    ckpt_candidates = [
        CHECKPOINT_PATH,
        f"{MODEL_NAME}.pt",
        f"/kaggle/input/{MODEL_NAME}/{MODEL_NAME}.pt",
        f"/kaggle/input/agnis-sprint1/{MODEL_NAME}.pt",
    ]
    
    loaded_ckpt = None
    for path in ckpt_candidates:
        if os.path.exists(path):
            print(f"[Sprint 2] Loading Sprint 1 checkpoint from {path}...")
            loaded_ckpt = torch.load(path, map_location=DEVICE)
            model.load_state_dict(loaded_ckpt['model'], strict=False)
            start_step = loaded_ckpt.get('step', 0)
            start_epoch = loaded_ckpt.get('epoch', 0)
            print(f"[Sprint 2] Resumed from step {start_step}, epoch {start_epoch}")
            break
    
    if loaded_ckpt is None:
        print("[Sprint 2] No checkpoint found — starting fresh")

    # CHANGE 3: Trainable Params with reduced LR
    trainable = [*model.embedding.parameters(), *model.r_norm.parameters(), *model.out_norm.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    
    if loaded_ckpt and 'optimizer' in loaded_ckpt:
        try:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            # Override LR to the new reduced value
            for pg in optimizer.param_groups:
                pg['lr'] = LR
            print(f"[Sprint 2] Optimizer restored, LR overridden to {LR}")
        except Exception:
            print(f"[Sprint 2] Optimizer state incompatible, using fresh optimizer at LR={LR}")

    param_total = sum(p.numel() for p in model.parameters()) / 1e6
    param_bp = sum(p.numel() for p in trainable) / 1e6
    param_hebb = param_total - param_bp
    
    print(f"[Params] Total: {param_total:.1f}M | Backprop: {param_bp:.1f}M | Hebbian: {param_hebb:.1f}M")
    print(f"[Core]   Settlement depth: {MAX_SETTLE_STEPS} steps (Sprint 1 was 1)")
    print(f"[Save]   Checkpoints → {CHECKPOINT_PATH}")
    print("-" * 60)

    model.train()
    model.reset_states(BATCH_SIZE)
    
    # CHANGE 4: Rolling average loss tracker
    loss_window = deque(maxlen=100)
    
    steps_per_epoch = tokens.shape[1] - 1
    global_step = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        epoch_start = time.time()
        
        for step in range(steps_per_epoch):
            global_step += 1
            
            # Skip steps we've already done (for checkpoint resume)
            if epoch == start_epoch and step < (start_step % steps_per_epoch):
                continue
            
            cur, tgt = tokens[:, step], tokens[:, step+1]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model.step_logits(cur)
                loss = F.cross_entropy(logits, tgt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            
            loss_val = loss.item()
            model._current_surprise = min(loss_val, 10.0)
            loss_window.append(loss_val)

            # CHANGE 4: Log with rolling average
            if (step+1) % 500 == 0:
                elapsed = time.time() - start_time
                tok_sec = (global_step * BATCH_SIZE) / max(elapsed, 1e-6)
                eta_hrs = ((EPOCHS - epoch) * steps_per_epoch - step) * BATCH_SIZE / tok_sec / 3600
                avg_loss = sum(loss_window) / len(loss_window)
                print(f"E{epoch+1} Step {step+1}/{steps_per_epoch} | "
                      f"Loss: {loss_val:.4f} | Avg100: {avg_loss:.4f} | "
                      f"{tok_sec:.0f} t/s | ETA: {eta_hrs:.1f}h")

            # Checkpointing every 5000 steps
            if (step+1) % 5000 == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                print(f"[Checkpoint] Saving to {CHECKPOINT_PATH} (Avg Loss: {avg_loss:.4f})...")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'max_steps': MAX_SETTLE_STEPS,
                }, CHECKPOINT_PATH)
        
        # End of epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = sum(loss_window) / len(loss_window) if loss_window else float('inf')
        print(f"\n[Epoch {epoch+1} Complete] Avg Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f}min")
        
        # Save end-of-epoch checkpoint
        torch.save({
            'step': 0,
            'epoch': epoch + 1,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'avg_loss': avg_loss,
            'max_steps': MAX_SETTLE_STEPS,
        }, CHECKPOINT_PATH)
        print(f"[Checkpoint] Epoch {epoch+1} saved to {CHECKPOINT_PATH}")
        
        # Early success check
        if avg_loss < 4.5:
            print(f"\n🎯 TARGET REACHED! Avg Loss {avg_loss:.4f} < 4.5")
            print(f"   Sprint 2 objective complete.")
            print(f"   Ready for Sprint 3 (max_steps=5)")
            break

    print(f"\n[Done] Sprint 2 training complete.")
    print(f"[Done] Final checkpoint: {CHECKPOINT_PATH}")
    print(f"[Done] Total time: {(time.time() - start_time)/3600:.1f} hours")

if __name__ == "__main__":
    main()
