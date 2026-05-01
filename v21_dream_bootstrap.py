"""
v21_dream_bootstrap.py — V21: Dream Synthesis & Bootstrap Learning
==================================================================
This script tests the "Universal Grammar Emergence" hypothesis.

1. Loads the V20 Turbo model (which discovered ~55% structural 
   overlap between en/de/es/fr).
2. FOLDS the high-affinity pathways into the shared Meta-Pool, 
   creating a permanent physical representation of Universal Grammar.
3. Introduces a 5th, unseen language (Italian).
4. Trains Italian for just 5 minutes to see if it "bootstraps"
   its learning by routing through the synthesized Meta-Pool.
"""

import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

# ─── Config ───
META_POOL_SIZE = 64
N_PER_LANG = 128
BPE_VOCAB = 4000
BATCH_SIZE = 32
MAX_STEPS = 3
BOOTSTRAP_DURATION = 300  # 5 minutes for Italian

# Reconstruct V20 ranges
LANGS_V20 = ["en", "de", "es", "fr"]
LANG_RANGES_V20 = {}
for i, lang in enumerate(LANGS_V20):
    s = META_POOL_SIZE + i * N_PER_LANG
    LANG_RANGES_V20[lang] = (s, s + N_PER_LANG)

def fetch_italian() -> str:
    """Fetch Italian Wikipedia or fallback."""
    print("  [it] Loading Wikipedia (20220301.it)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikipedia", "20220301.it", split="train",
                          streaming=True)
        text = ""
        for article in ds:
            text += article["text"] + "\n"
            if len(text) >= 500000:
                break
        print(f"  [it] Got {len(text):,} chars from Wikipedia")
        return text[:500000]
    except Exception as e:
        print(f"  [it] HuggingFace failed: {e}")
        # Try Gutenberg fallback
        for path in ["slm/input_it.txt", "slm/wiki_it.txt"]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()[:500000]
                print(f"  [it] Loaded {len(text):,} chars from {path}")
                return text
    print("  [it] FATAL: Could not load Italian corpus.")
    sys.exit(1)

def get_batch(token_ids: torch.Tensor, batch_idx: int, batch_size: int,
              vocab_size: int) -> tuple:
    start = batch_idx * batch_size
    end = start + batch_size
    if end + 1 > len(token_ids):
        start = 0
        end = batch_size

    input_ids = token_ids[start:end]
    target_ids = token_ids[start + 1:end + 1]

    x = torch.zeros(batch_size, vocab_size, device=token_ids.device)
    target = torch.zeros(batch_size, vocab_size, device=token_ids.device)

    x.scatter_(1, input_ids.unsqueeze(1), 1.0)
    target.scatter_(1, target_ids.unsqueeze(1), 1.0)

    return x, target

def run_bootstrap():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "=" * 60)
    print("  AGNIS V21 — DREAM SYNTHESIS & BOOTSTRAP LEARNING")
    print("=" * 60)
    
    # 1. Load Tokenizer
    cache = f"agnis_bpe_{BPE_VOCAB}_vocab.json"
    if not os.path.exists(cache):
        print(f"[ERROR] Missing {cache}. Run V20 first.")
        return
    tokenizer = BPETokenizer(BPE_VOCAB)
    tokenizer.load(cache)
    vocab_size = tokenizer.vocab_size

    # 2. Build Hierarchy & Load V20 State
    initial_hidden = META_POOL_SIZE + N_PER_LANG * len(LANGS_V20)
    hierarchy = PredictiveHierarchy(
        [vocab_size, initial_hidden, initial_hidden, vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )
    
    ckpt_path = "agnis_v20_turbo.pt"
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Missing checkpoint {ckpt_path}. Run V20 first.")
        return
        
    # We only load hierarchy_state if it was saved via CognitivePredictiveAgent,
    # but V20 saved directly via hierarchy.save_checkpoint().
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'hierarchy_state' in ckpt:
        hierarchy.load_state_dict(ckpt['hierarchy_state'])
    else:
        hierarchy.load_state_dict(ckpt)
    print(f"  [Loaded] {ckpt_path}")

    # 3. Dream Synthesis (Folding)
    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(LANG_RANGES_V20)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)
    
    # This averages the 4 languages and writes to the Meta-Pool!
    abstrax.synthesize_dream_neurons(META_POOL_SIZE)

    # 4. Introduce Italian
    print("\n" + "=" * 40)
    print("  PHASE 3: BOOTSTRAPPING A NEW LANGUAGE (ITALIAN)")
    print("=" * 40)
    
    it_text = fetch_italian()
    it_tokens = tokenizer.encode(it_text)
    it_tensor = torch.tensor(it_tokens, dtype=torch.long, device=device)
    print(f"  Italian: {len(it_tokens):,} tokens ready.")

    # Expand architecture for Italian
    print("\n  Recruiting new Italian sliver (128 neurons)...")
    hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language="Italian")
    
    # The new range for Italian
    it_start = initial_hidden
    it_end = it_start + N_PER_LANG

    # We want Italian to use BOTH its new sliver AND the newly synthesized Meta-Pool
    # So we unmask the meta-pool for the Italian manifold gate
    with hierarchy.manifold_gate(it_start, it_end):
        # Allow gradients to flow into the new Italian weights
        for layer in hierarchy.layers:
            # We want to train the Italian weights, but keep Meta-Pool FROZEN
            # so we only see how well the Meta-Pool transfers knowledge.
            layer.V_mask[:, :META_POOL_SIZE] = 0.0
            layer.W_mask[:META_POOL_SIZE, :] = 0.0
            layer.R_mask[:META_POOL_SIZE, :META_POOL_SIZE] = 0.0
            layer.R_gate_mask[:META_POOL_SIZE, :META_POOL_SIZE] = 0.0
            layer.b_in_mask[:META_POOL_SIZE] = 0.0

        n_batches = (len(it_tokens) - 1) // BATCH_SIZE
        print(f"\n>>> ITALIAN BOOTSTRAP ({BOOTSTRAP_DURATION//60} min) <<<")
        print("  Meta-Pool is FROZEN (Acting as Universal Grammar)")
        print("  Italian Sliver is TRAINABLE (Acting as Syntax Translator)")
        
        t0 = time.time()
        total_tokens = 0
        batch_losses = []
        batch_idx = 0
        
        hierarchy.reset_states()
        
        while True:
            elapsed = time.time() - t0
            if elapsed > BOOTSTRAP_DURATION:
                break
            if batch_idx >= n_batches:
                batch_idx = 0
                
            x, target = get_batch(it_tensor, batch_idx, BATCH_SIZE, vocab_size)
            hierarchy.infer_and_learn(
                x, top_level_label=target,
                dopamine_burst=1.0, max_steps=MAX_STEPS, warm_start=True
            )
            
            surprise = hierarchy.get_surprise((x, target))
            batch_losses.append(surprise)
            total_tokens += BATCH_SIZE
            batch_idx += 1
            
            if batch_idx % 50 == 0:
                avg_s = sum(batch_losses[-50:]) / max(1, len(batch_losses[-50:]))
                tps = total_tokens / max(1, elapsed)
                print(f"  [Italian] {total_tokens:,} tok | {tps:.0f} tok/s | Surprise: {avg_s:.4f} | {elapsed/BOOTSTRAP_DURATION:.0%}", end="\r")

        avg_final = sum(batch_losses[-100:]) / max(1, len(batch_losses[-100:]))
        final_tps = total_tokens / (time.time() - t0)
        print(f"\n  Italian Final: {total_tokens:,} tokens | {final_tps:.0f} tok/s | Final Surprise: {avg_final:.4f}")

    # ═══ Save V21 ═══
    hierarchy.save_checkpoint("agnis_v21_bootstrap.pt")
    print("\n[Saved] agnis_v21_bootstrap.pt")
    
if __name__ == "__main__":
    run_bootstrap()
