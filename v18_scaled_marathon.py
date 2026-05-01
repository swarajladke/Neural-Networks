"""
v18_scaled_marathon.py — Scaled AGNIS with BPE + Embeddings
============================================================
Phase 1 Integration Test: Combines all Phase 1 components:
  - Wikipedia corpus (6 languages)
  - 16K BPE tokenizer
  - ScaledAGNIS (embedding + Hebbian core + output head)
  - Meta-Pool cross-language convergence
  - AbstraX Dream Cycle analysis
  - Text generation demo

This is the first AGNIS experiment that resembles a real language model.
"""

import torch
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v5_scaled import ScaledAGNIS
from agnis_v4_cognitive import AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

# ─── Configuration ───
META_POOL_SIZE = 128
N_PER_LANG = 128
EMBED_DIM = 64
BPE_VOCAB = 4000         # Start with 4K (16K needs larger corpus)
PHASE1_DURATION = 600     # 10 min per language
PHASE2_DURATION = 300     # 5 min consolidation
LANGS = ["en", "de", "es", "fr"]
LANG_NAMES = {"en": "English", "de": "German", "es": "Spanish", "fr": "French"}


def load_or_train_bpe(corpora: dict, vocab_size: int) -> BPETokenizer:
    """Load cached BPE or train from scratch."""
    cache_path = f"agnis_bpe_{vocab_size}_vocab.json"
    if os.path.exists(cache_path):
        tok = BPETokenizer(vocab_size=vocab_size)
        tok.load(cache_path)
        return tok

    combined = ""
    for text in corpora.values():
        combined += text
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.fit(combined, verbose=True)
    tok.save(cache_path)
    return tok


def freeze_all(hierarchy):
    for layer in hierarchy.layers:
        layer.V_mask.zero_()
        layer.W_mask.zero_()
        layer.b_in_mask.zero_()
        layer.b_out_mask.zero_()
        layer.R_mask.zero_()
        layer.R_gate_mask.zero_()
        layer.L_mask.zero_()


def unmask_meta(hierarchy, mp):
    for i, layer in enumerate(hierarchy.layers):
        if i < len(hierarchy.layers) - 1:
            layer.V_mask[:, :mp] = 1.0
            layer.W_mask[:mp, :] = 1.0
            layer.b_in_mask[:mp] = 1.0
            layer.R_mask[:mp, :mp] = 1.0
            layer.R_gate_mask[:mp, :mp] = 1.0
            layer.L_mask[:mp, :mp] = 1.0
        if i > 0:
            layer.V_mask[:mp, :] = 1.0
            layer.W_mask[:, :mp] = 1.0
            layer.b_out_mask[:mp] = 1.0


def run_scaled_marathon():
    import random
    marathon_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  AGNIS V18 -- SCALED ARCHITECTURE MARATHON")
    print("  BPE + Embeddings + Output Head + Meta-Pool")
    print("=" * 60)
    print(f"  Device: {device}")

    # ═══ 1. Load Data ═══
    corpora = {}
    for code in LANGS:
        # Try wiki data first, fall back to Gutenberg
        wiki_path = f"slm/wiki_{code}.txt"
        gut_path = f"slm/input_{code}.txt"
        if os.path.exists(wiki_path):
            path = wiki_path
        elif os.path.exists(gut_path):
            path = gut_path
        else:
            print(f"  [WARN] No data for {code}, skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            corpora[code] = f.read()[:100000]
        print(f"  {LANG_NAMES[code]}: {len(corpora[code]):,} chars from {path}")

    # ═══ 2. Train/Load BPE ═══
    tokenizer = load_or_train_bpe(corpora, BPE_VOCAB)
    print(f"  BPE vocab: {tokenizer.vocab_size}")

    # Encode all languages
    all_tokens = {}
    for code, text in corpora.items():
        tokens = tokenizer.encode(text)
        all_tokens[code] = tokens
        compression = len(text) / max(1, len(tokens))
        print(f"  {LANG_NAMES[code]}: {len(tokens):,} tokens ({compression:.1f}x compression)")

    # ═══ 3. Build ScaledAGNIS ═══
    initial_hidden = META_POOL_SIZE + N_PER_LANG
    hidden_dims = [EMBED_DIM, initial_hidden, initial_hidden, EMBED_DIM]

    model = ScaledAGNIS(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dims=hidden_dims,
        meta_pool_size=META_POOL_SIZE,
        device=device
    )

    lang_ranges = {}
    for i, lang in enumerate(LANGS):
        start = META_POOL_SIZE + i * N_PER_LANG
        end = start + N_PER_LANG
        lang_ranges[lang] = (start, end)

    final_width = META_POOL_SIZE + N_PER_LANG * len(LANGS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Architecture: {hidden_dims}")
    print(f"  Meta-Pool: {META_POOL_SIZE} | Sliver: {N_PER_LANG}")
    print(f"  Ranges: {lang_ranges}")
    print(f"  Final width: {final_width}")
    print(f"  Total params: {total_params:,}")
    print("=" * 60)

    hierarchy = model.hierarchy

    # ═══ 4. Phase 1: Sequential Training ═══
    print("\n" + "=" * 40)
    print("  PHASE 1: SEQUENTIAL TRAINING")
    print("=" * 40)

    phase1_losses = {}
    for phase_idx, code in enumerate(LANGS):
        name = LANG_NAMES[code]
        print(f"\n>>> {name.upper()} ({PHASE1_DURATION//60} min) <<<")

        tokens = all_tokens[code]
        start_time = time.time()
        lang_start, lang_end = lang_ranges[code]
        losses = []

        model.reset_position()

        with hierarchy.manifold_gate(0, lang_end):
            hierarchy.reset_states()
            for i in range(len(tokens) - 1):
                elapsed = time.time() - start_time
                if elapsed > PHASE1_DURATION:
                    break

                surprise, loss = model.train_step(tokens[i], tokens[i + 1], max_steps=15)
                losses.append(loss)

                if i % 500 == 0:
                    avg_loss = sum(losses[-100:]) / max(1, len(losses[-100:]))
                    progress = elapsed / PHASE1_DURATION
                    print(f"  [{name}] Token {i:5d} | Loss: {avg_loss:.3f} | {progress:.0%}", end="\r")

        avg_final = sum(losses[-200:]) / max(1, len(losses[-200:]))
        phase1_losses[code] = avg_final
        print(f"\n  {name} Final Loss: {avg_final:.4f} ({len(losses)} tokens trained)")

        if phase_idx < len(LANGS) - 1:
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # ═══ 5. Phase 2: Consolidation ═══
    print("\n" + "=" * 40)
    print("  PHASE 2: ROUND-ROBIN CONSOLIDATION")
    print("=" * 40)

    freeze_all(hierarchy)
    unmask_meta(hierarchy, META_POOL_SIZE)
    print(f"  Meta-pool only ({META_POOL_SIZE} neurons trainable)")

    for layer in hierarchy.layers:
        layer.eta_V = 0.005
        layer.eta_W = 0.003
        layer.eta_R = 0.003

    start_time = time.time()
    token_count = 0
    lang_idx = 0
    pos_in_lang = {code: 0 for code in LANGS}

    while True:
        elapsed = time.time() - start_time
        if elapsed > PHASE2_DURATION:
            break

        if token_count % 50 == 0 and token_count > 0:
            lang_idx = (lang_idx + 1) % len(LANGS)
            hierarchy.reset_states()
            model.reset_position()

        code = LANGS[lang_idx]
        tokens = all_tokens[code]
        idx = pos_in_lang[code] % (len(tokens) - 1)

        with hierarchy.manifold_gate(0, final_width):
            model.train_step(tokens[idx], tokens[idx + 1], dopamine=0.5)

        pos_in_lang[code] += 1
        token_count += 1

        if token_count % 1000 == 0:
            print(f"  Tokens: {token_count} | {elapsed/PHASE2_DURATION:.0%} | Lang: {code}", end="\r")

    print(f"\n  Consolidation: {token_count} tokens processed")

    # ═══ 6. Save ═══
    model.save("agnis_v18_scaled.pt")
    tokenizer.save("agnis_v18_bpe.json")
    print("[Checkpoint] Saved.")

    # ═══ 7. AbstraX Dream Cycle ═══
    print("\n>>> ABSTRAX DREAM CYCLE <<<")

    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for layer_idx in range(len(hierarchy.layers)):
        print(f"\n--- Layer {layer_idx} ---")
        try:
            result = abstrax.compute_pairwise_affinity(layer_idx=layer_idx)
            abstrax.print_affinity_report(result, title=f"Layer {layer_idx} (Scaled)")
            abstrax.identify_fold_candidates(result, threshold=0.2)
        except Exception as e:
            print(f"  [SKIP] {e}")

    # ═══ 8. Text Generation Demo ═══
    print("\n>>> GENERATION DEMO <<<")
    test_prompts = {
        "en": "The king",
        "de": "Der Mann",
        "es": "El rey",
        "fr": "Le roi",
    }

    for code, prompt in test_prompts.items():
        prompt_ids = tokenizer.encode(prompt)
        lang_start, lang_end = lang_ranges[code]

        with hierarchy.manifold_gate(0, lang_end):
            hierarchy.reset_states()
            generated_ids = model.generate(
                tokenizer, prompt_ids, max_tokens=20, temperature=0.9
            )
        generated_text = tokenizer.decode(prompt_ids + generated_ids)
        print(f"  [{code}] {generated_text}")

    # ═══ 9. Summary ═══
    total_time = (time.time() - marathon_start) / 60
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    print(f"\n{'='*60}")
    print(f"  V18 SCALED MARATHON COMPLETE")
    print(f"  Time: {total_time:.1f} min | VRAM: {peak_vram:.2f} GB")
    print(f"  Params: {total_params:,}")
    print(f"  Phase 1 losses: {phase1_losses}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_scaled_marathon()
