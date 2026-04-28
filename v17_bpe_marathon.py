"""
v17_bpe_marathon.py — BPE Sub-Word Meta-Pool Experiment
========================================================
V3 Phase 35: The first AGNIS experiment operating on meaningful
sub-word tokens instead of raw characters.

Key Innovation:
  Instead of predicting individual letters ('T', 'h', 'e'),
  the network now predicts whole sub-words ('The', ' cat', ' sat').
  
  This means the Meta-Pool can discover shared CONCEPTS across
  languages. For example, " the" and " le" (French) are both
  determiners — the network can learn that they serve the same
  grammatical function.

Architecture:
  - BPE tokenizer trained on all 4 languages (shared vocab ~500)
  - One-hot input (no embedding layer — pure Hebbian)
  - 3-layer deep hierarchy with meta-pool in Layer 1
  - Sequential training + consolidation
"""

import torch
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

META_POOL_SIZE = 64
N_PER_LANG = 128
BPE_VOCAB_SIZE = 500
PHASE1_DURATION = 180   # 3 min per language
PHASE2_DURATION = 600   # 10 min consolidation


def avg_surprise_bpe(hierarchy, tokenizer, tokens, gate_start, gate_end, device, n_samples=200):
    vocab_size = tokenizer.vocab_size
    surprises = []
    with hierarchy.manifold_gate(gate_start, gate_end):
        hierarchy.reset_states()
        with torch.no_grad():
            for i in range(min(n_samples, len(tokens) - 1)):
                x = torch.zeros((1, vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                target = torch.zeros((1, vocab_size), device=device)
                target[0, tokens[i + 1]] = 1.0
                surprises.append(hierarchy.get_surprise((x, target)))
    return sum(surprises) / max(1, len(surprises))


def freeze_all_masks(hierarchy):
    for layer in hierarchy.layers:
        layer.V_mask.zero_()
        layer.W_mask.zero_()
        layer.b_in_mask.zero_()
        layer.b_out_mask.zero_()
        layer.R_mask.zero_()
        layer.R_gate_mask.zero_()
        layer.L_mask.zero_()


def unmask_metapool_only(hierarchy, mp):
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


def run_bpe_marathon():
    import random
    marathon_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = ["en", "de", "es", "fr"]
    lang_names = {"en": "English", "de": "German", "es": "Spanish", "fr": "French"}

    # 1. Load corpora
    full_text = ""
    corpora = {}
    for code in langs:
        with open(f"slm/input_{code}.txt", "r", encoding="utf-8") as f:
            text = f.read()[:50000]
            corpora[code] = text
            full_text += text

    # 2. Train BPE tokenizer on ALL languages
    print("\n" + "=" * 60)
    print("  AGNIS V17 -- BPE SUB-WORD META-POOL EXPERIMENT")
    print("=" * 60)

    tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE)
    tokenizer.fit(full_text, verbose=True)

    # Encode each language
    all_tokens = {}
    for code in langs:
        tokens = tokenizer.encode(corpora[code])
        all_tokens[code] = tokens
        char_len = len(corpora[code])
        tok_len = len(tokens)
        print(f"  {lang_names[code]}: {char_len} chars -> {tok_len} tokens ({char_len/max(1,tok_len):.1f}x compression)")

    vocab_size = tokenizer.vocab_size
    print(f"\n  Final BPE vocab: {vocab_size}")
    print(f"  Input dimension: {vocab_size} (one-hot)")

    # 3. Compute language ranges
    lang_ranges = {}
    for i, lang in enumerate(langs):
        start = META_POOL_SIZE + i * N_PER_LANG
        end = start + N_PER_LANG
        lang_ranges[lang] = (start, end)

    initial_hidden = META_POOL_SIZE + N_PER_LANG
    final_width = META_POOL_SIZE + N_PER_LANG * len(langs)

    print(f"  Meta-Pool: {META_POOL_SIZE} | Sliver: {N_PER_LANG}")
    print(f"  Ranges: {lang_ranges}")
    print(f"  Architecture: [{vocab_size} -> {initial_hidden} -> {initial_hidden} -> {vocab_size}]")
    print("=" * 60)

    # 4. Build hierarchy
    hierarchy = PredictiveHierarchy(
        [vocab_size, initial_hidden, initial_hidden, vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )
    agent = CognitivePredictiveAgent(hierarchy, device=device)

    # ═══════════════════════════════════════════
    # PHASE 1: Sequential Training
    # ═══════════════════════════════════════════
    print("\n" + "=" * 40)
    print("  PHASE 1: SEQUENTIAL (BPE TOKENS)")
    print("=" * 40)

    for phase_idx, code in enumerate(langs):
        name = lang_names[code]
        print(f"\n>>> {name.upper()} ({PHASE1_DURATION//60} min) <<<")

        tokens = all_tokens[code]
        start_time = time.time()
        lang_start, lang_end = lang_ranges[code]

        with hierarchy.manifold_gate(0, lang_end):
            hierarchy.reset_states()
            for i in range(len(tokens) - 1):
                elapsed = time.time() - start_time
                if elapsed > PHASE1_DURATION: break

                scale = agent.get_dopamine_scale(elapsed, PHASE1_DURATION)
                for layer in hierarchy.layers:
                    layer.eta_V = 0.05 * scale
                    layer.eta_W = 0.03 * scale

                x = torch.zeros((1, vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                target = torch.zeros((1, vocab_size), device=device)
                target[0, tokens[i+1]] = 1.0
                hierarchy.infer_and_learn(x, top_level_label=target, dopamine_burst=1.0)

                if i % 500 == 0:
                    progress = elapsed / PHASE1_DURATION
                    print(f"  [{name}] Token {i:5d} | {progress:.0%}", end="\r")

        surprise = avg_surprise_bpe(hierarchy, tokenizer, tokens, 0, lang_end, device)
        print(f"\n  {name} Baseline Surprise: {surprise:.4f}")

        if phase_idx < len(langs) - 1:
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # ═══════════════════════════════════════════
    # PHASE 2: Round-Robin Consolidation
    # ═══════════════════════════════════════════
    print("\n" + "=" * 40)
    print("  PHASE 2: ROUND-ROBIN CONSOLIDATION")
    print("=" * 40)

    freeze_all_masks(hierarchy)
    unmask_metapool_only(hierarchy, META_POOL_SIZE)
    print(f"  Meta-pool only ({META_POOL_SIZE} neurons trainable)")

    for layer in hierarchy.layers:
        layer.eta_V = 0.01
        layer.eta_W = 0.005
        layer.eta_R = 0.005

    start_time = time.time()
    token_count = 0
    lang_switches = 0
    lang_idx = 0
    pos_in_lang = {code: 0 for code in langs}

    while True:
        elapsed = time.time() - start_time
        if elapsed > PHASE2_DURATION: break

        if token_count % 50 == 0 and token_count > 0:
            lang_idx = (lang_idx + 1) % len(langs)
            lang_switches += 1
            hierarchy.reset_states()

        code = langs[lang_idx]
        tokens = all_tokens[code]
        idx = pos_in_lang[code] % (len(tokens) - 1)
        lang_start, lang_end = lang_ranges[code]

        with hierarchy.manifold_gate(0, final_width):
            x = torch.zeros((1, vocab_size), device=device)
            x[0, tokens[idx]] = 1.0
            target = torch.zeros((1, vocab_size), device=device)
            target[0, tokens[idx + 1]] = 1.0
            hierarchy.infer_and_learn(x, top_level_label=target, dopamine_burst=1.0)

        pos_in_lang[code] += 1
        token_count += 1

        if token_count % 1000 == 0:
            print(f"  Tokens: {token_count} | Switches: {lang_switches} | {elapsed/PHASE2_DURATION:.0%}", end="\r")

    print(f"\n  Done: {token_count} tokens, {lang_switches} switches")

    # Save
    hierarchy.save_checkpoint("agnis_bpe_metapool_final.pt")
    tokenizer.save("agnis_bpe_vocab.json")
    print("[Checkpoint] Saved.")

    # ═══════════════════════════════════════════
    # AUDIT
    # ═══════════════════════════════════════════
    print("\n>>> RETENTION AUDIT <<<")
    hierarchy.load_checkpoint("agnis_bpe_metapool_final.pt")
    for code in langs:
        tokens = all_tokens[code]
        lang_start, lang_end = lang_ranges[code]
        surprise = avg_surprise_bpe(hierarchy, tokenizer, tokens, 0, lang_end, device)
        print(f"  {code}: surprise={surprise:.4f}")

    # AbstraX
    print("\n>>> ABSTRAX DREAM CYCLE <<<")
    hierarchy.load_checkpoint("agnis_bpe_metapool_final.pt")

    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for layer_idx in range(len(hierarchy.layers)):
        print(f"\n--- Layer {layer_idx} ---")
        try:
            result = abstrax.compute_pairwise_affinity(layer_idx=layer_idx)
            pairs = abstrax.print_affinity_report(result, title=f"Layer {layer_idx} Affinity (BPE)")
            abstrax.identify_fold_candidates(result, threshold=0.2)
        except Exception as e:
            print(f"  [SKIP] Layer {layer_idx}: {e}")

    # Show shared tokens between languages
    print("\n>>> SHARED TOKEN ANALYSIS <<<")
    for i, lang_a in enumerate(langs):
        for lang_b in langs[i+1:]:
            tokens_a = set(all_tokens[lang_a])
            tokens_b = set(all_tokens[lang_b])
            shared = tokens_a & tokens_b
            total = tokens_a | tokens_b
            overlap = len(shared) / max(1, len(total)) * 100
            # Decode some shared tokens
            shared_words = [tokenizer.decode([t]) for t in list(shared)[:8]]
            print(f"  {lang_a}<->{lang_b}: {len(shared)}/{len(total)} shared token types ({overlap:.1f}%) | Examples: {shared_words}")

    total = (time.time() - marathon_start) / 60
    print(f"\n{'='*60}")
    print(f"  V17 BPE MARATHON COMPLETE -- {total:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_bpe_marathon()
