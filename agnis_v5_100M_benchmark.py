"""
agnis_v5_100M_benchmark.py
================================================================
The official AGNIS V5 architecture scaled to >100M parameters.
Designed for the L40S / A100 GPU benchmarking.

Architecture:
  - Vocab: 32,000 (BPE)
  - Embedding: 1024 dimensions
  - Core: 1024 -> 8192 -> 1024 (Fully Unfrozen, Local Hebbian Updates)
  - Temporal R-Matrix: 1024 x 1024 (Delta Rule)
  - LM Head: 1024 -> 32,000 (Tied with embedding)

Optimizations:
  - torch.compile (inductor)
  - bfloat16 mixed precision
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
TOKENIZER_PATH = "slm_bpe_tokenizer_32k.json"
TARGET_CHARS = 100_000_000  # Wikitext-103
VOCAB_SIZE = 32000
BATCH_SIZE = 128            # High batch size to max out L40S
EPOCHS = 1

# 100M Parameter Scale
EMBED_DIM = 1024
CORE_HIDDEN = 8192

# Temporal / Hebbian
ALPHA = 0.1
ETA_R_LOCAL = 0.002
R_DECAY = 0.999

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── AGNIS V5 Architecture ──────────────────────────────────────
class AgnisV5_100M(nn.Module):
    def __init__(self, vocab_size, embed_dim=1024, hidden_dim=8192, alpha=0.1, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.alpha = alpha

        # Massive 32M parameter embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Unfrozen Core (~85M parameters)
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, hidden_dim, embed_dim], device=device
        )

        # Temporal R-Matrix (1M parameters)
        self.R_weight = torch.zeros(embed_dim, embed_dim, device=self.device)
        nn.init.orthogonal_(self.R_weight, gain=0.1)
        self.r_norm = nn.LayerNorm(embed_dim).to(self.device)

        # Persistent state
        self.register_buffer("h_prev", torch.zeros(1, embed_dim))
        self._current_surprise = 1.0

        # LM Head (Tied to embedding)
        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        self.lm_head.weight = self.embedding.weight

    def reset_states(self, batch_size=1):
        self.hierarchy.reset_states(batch_size=batch_size)
        self.h_prev = torch.zeros(batch_size, self.embed_dim, device=self.lm_head.weight.device)
        self._current_surprise = 1.0

    def set_surprise(self, loss_value):
        self._current_surprise = min(loss_value, 10.0)

    def step_logits(self, token_ids):
        # 1. Embedding
        emb = F.normalize(self.embedding(token_ids), dim=-1)

        # 2. UNFREEZEN CORE: Hebbian Learning + Inference
        # The core learns its own internal logic independently of backprop!
        core_out = self.hierarchy.infer_and_learn(emb, max_steps=1)
        core_out = F.normalize(core_out, dim=-1)

        h_prev_d = self.h_prev.detach()

        # 3. Temporal Delta Rule
        if self.alpha > 0.0:
            with torch.no_grad():
                x_hat = torch.matmul(h_prev_d, self.R_weight)
                epsilon = core_out.detach() - x_hat
                dR = torch.bmm(h_prev_d.unsqueeze(2), epsilon.unsqueeze(1)).mean(dim=0)
                self.R_weight = (
                    R_DECAY * self.R_weight + ETA_R_LOCAL * self._current_surprise * dR
                ).clamp(-3.0, 3.0)

            temporal = torch.matmul(h_prev_d, self.R_weight)
            h_t = core_out + self.alpha * temporal
            h_t = self.r_norm(h_t)
        else:
            h_t = core_out

        self.h_prev = h_t.detach()

        # 4. Residual skip-connection directly to LM head
        fused = self.out_norm(h_t + emb)
        
        return self.lm_head(fused)


# ─── Dataset Loader ───────────────────────────────────────────────
def load_corpus():
    try:
        from datasets import load_dataset
        print("[Corpus] Loading Wikitext-103 from HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text = "\n".join([t for t in dataset["text"] if len(t.strip()) > 10])
        text = text[:TARGET_CHARS]
        print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
        return text
    except ImportError:
        print("[Corpus] 'datasets' library not found. pip install datasets")
        sys.exit(1)

def build_token_tensor(ids, bs, dev):
    sl = len(ids) // bs
    return torch.tensor(ids[:sl*bs], dtype=torch.long, device=dev).view(bs, sl)

# ─── Benchmark Loop ───────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  AGNIS V5 | 100M PARAMETER BENCHMARK")
    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("="*60)

    text = load_corpus()

    if not os.path.exists(TOKENIZER_PATH):
        print(f"[Tokenizer] Training new {VOCAB_SIZE} BPE tokenizer...")
        from tokenizers import decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        tok = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<pad>","<s>","</s>","<unk>"])
        tok.train_from_iterator([text], trainer=trainer)
        tok.save(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[Tokenizer] vocab={tokenizer.get_vocab_size()}")

    # Initialize model
    model = AgnisV5_100M(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=CORE_HIDDEN, device=DEVICE)
    model.to(DEVICE)

    # Optimization: Compile model for L40S / A100
    print("[System] Compiling model with torch.compile (this takes ~60s)...")
    try:
        model.step_logits = torch.compile(model.step_logits)
    except Exception as e:
        print(f"[System] Warning: torch.compile failed ({e}). Running in eager mode.")

    # Only embedding and normalization need backprop (Core trains itself via Hebbian rules!)
    trainable_backprop = [
        *model.embedding.parameters(),
        *model.r_norm.parameters(),
        *model.out_norm.parameters(),
    ]
    
    n_backprop = sum(p.numel() for p in trainable_backprop)
    n_hebbian = sum(p.numel() for p in model.hierarchy.parameters()) + model.R_weight.numel()
    print(f"[Params] Total: {(n_backprop + n_hebbian)/1e6:.1f} M")
    print(f"         Backprop (LM):  {n_backprop/1e6:.1f} M")
    print(f"         Hebbian (Core): {n_hebbian/1e6:.1f} M")

    enc = tokenizer.encode(text)
    tokens = build_token_tensor(enc.ids, BATCH_SIZE, DEVICE)
    
    print(f"[Data] {tokens.shape[1]:,} steps | Batch Size: {BATCH_SIZE}")
    print("-"*60)

    optimizer = torch.optim.AdamW(trainable_backprop, lr=3e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    model.train()
    model.reset_states(BATCH_SIZE)
    
    start_time = time.time()
    tokens_processed = 0

    # Benchmark run (we only run 1000 steps to get the speed)
    BENCHMARK_STEPS = 1000
    
    for step in range(BENCHMARK_STEPS):
        cur, tgt = tokens[:, step], tokens[:, step + 1]

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.step_logits(cur)
                loss = F.cross_entropy(logits, tgt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_backprop, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model.step_logits(cur)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_backprop, 1.0)
            optimizer.step()

        model.set_surprise(loss.item())
        tokens_processed += BATCH_SIZE

        if (step+1) % 100 == 0:
            elapsed = time.time() - start_time
            tok_sec = tokens_processed / elapsed
            print(f"Step {step+1:>4}/{BENCHMARK_STEPS} | Loss: {loss.item():.4f} | Speed: {tok_sec:.0f} tokens/sec")

    print("\n" + "="*60)
    print("  BENCHMARK COMPLETE")
    final_speed = tokens_processed / (time.time() - start_time)
    print(f"  Average Speed: {final_speed:.0f} tokens/sec")
    
    # Calculate exactly how long 2 Billion tokens would take
    hours_for_2B = 2_000_000_000 / final_speed / 3600
    print(f"  Estimated time for 2 Billion tokens: {hours_for_2B:.1f} hours")
    print("="*60)

if __name__ == "__main__":
    main()
