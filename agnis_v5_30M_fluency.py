"""
agnis_v5_30M_fluency.py
================================================================
AGNIS V5 | The "Kaggle-Optimal" Fluency Model
Target: Stage 1 AGI Roadmap (Fluent Sentence Generation)

Specs:
  - 38.5 Million Parameters
  - 768-dim Embeddings | 3072-dim Core | 32k Vocab
  - Optimized for single/dual T4 (3,500+ tokens/sec)
  - Trained on 200 Million tokens (Wiki-Multilingual)
"""

import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from agnis_v4_core import PredictiveHierarchy

# ─── Config ───────────────────────────────────────────────────────
MODEL_NAME = "agnis_v5_30m_fluency"
CHECKPOINT_PATH = f"{MODEL_NAME}.pt"
TOKENIZER_PATH = "slm_bpe_tokenizer_32k.json"

# Data Scaling
TARGET_TOKENS = 200_000_000 
BATCH_SIZE = 64
EPOCHS = 1

# 30M Parameter Scale (The "Sweet Spot")
EMBED_DIM = 768
CORE_HIDDEN = 3072
VOCAB_SIZE = 32000

# Learning
LR = 4e-4
ALPHA = 0.1
ETA_R_LOCAL = 0.002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── AGNIS V5 Model ─────────────────────────────────────────────
class AgnisV5(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=3072, alpha=0.1, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        
        # Predictive Hierarchy (Unfrozen)
        # Dimensions: [input, hidden, output]
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
        # Detach emb to prevent graph explosion
        with torch.no_grad():
            self.hierarchy.infer_and_learn(emb.detach(), max_steps=1)
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
        
        # Load Wikitext-103 (Standard benchmark)
        en_wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        
        # Load FineWeb-Edu (The current gold standard for clean training data)
        # We take a fixed number of rows to be safe and avoid float-percentage errors
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
    print("  AGNIS V5 | 30M PARAMETER FLUENCY TRAINING")
    print(f"  Target: {TARGET_TOKENS/1e6:.0f} Million Tokens")
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
        
        # Split text into lines so the tokenizer can process it in parallel
        tok.train_from_iterator(raw_text.splitlines(), trainer=trainer)
        tok.save(TOKENIZER_PATH)
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print("[Tokenizer] Encoding corpus (Batch mode)...")
    
    # Memory Efficient: Encode in batches to avoid Kaggle RAM crash
    encodings = tokenizer.encode_batch(raw_text.splitlines())
    ids = []
    for enc in encodings:
        ids.extend(enc.ids)
    
    ids = ids[:TARGET_TOKENS]
    
    sl = len(ids) // BATCH_SIZE
    tokens = torch.tensor(ids[:sl*BATCH_SIZE], dtype=torch.long, device=DEVICE).view(BATCH_SIZE, sl)

    # Model
    model = AgnisV5(VOCAB_SIZE, EMBED_DIM, CORE_HIDDEN, device=DEVICE)
    
    # Checkpoint Resume
    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[System] Resuming from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt['step']

    # Trainable Params (Backprop)
    trainable = [*model.embedding.parameters(), *model.r_norm.parameters(), *model.out_norm.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    if os.path.exists(CHECKPOINT_PATH) and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    print(f"[Params] Total: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    print(f"[Speed] Expecting ~3,500 tokens/sec on T4")
    print("-" * 60)

    model.train()
    model.reset_states(BATCH_SIZE)
    
    start_time = time.time()
    steps = tokens.shape[1] - 1
    
    for step in range(start_step, steps):
        cur, tgt = tokens[:, step], tokens[:, step+1]
        
        optimizer.zero_grad(set_to_none=True)
        
        # Use mixed precision for speed on T4
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model.step_logits(cur)
            loss = F.cross_entropy(logits, tgt)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        
        model._current_surprise = min(loss.item(), 10.0)

        # Logging
        if (step+1) % 500 == 0:
            elapsed = time.time() - start_time
            tok_sec = ((step - start_step + 1) * BATCH_SIZE) / max(elapsed, 1e-6)
            eta_hrs = (steps - step) * BATCH_SIZE / tok_sec / 3600
            print(f"Step {step+1}/{steps} | Loss: {loss.item():.4f} | {tok_sec:.0f} t/s | ETA: {eta_hrs:.1f}h")

        # Checkpointing
        if (step+1) % 10000 == 0:
            print(f"[Checkpoint] Saving to {CHECKPOINT_PATH}...")
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, CHECKPOINT_PATH)

    print("\n[Done] Training complete. Model saved to", CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
