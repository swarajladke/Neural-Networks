"""
v16_interleaved_metapool.py — Round-Robin Meta-Pool Consolidation
=================================================================
V3 Phase 34: Interleaved training to force Meta-Pool convergence.

The V15 experiment showed that sequential training yanks the Meta-Pool
in 4 different directions, causing catastrophic forgetting (English: 55.7%).

Fix: After sequential training establishes language slivers, run a
"Consolidation Phase" where we rapidly cycle through ALL languages
with ONLY the Meta-Pool neurons trainable. The language slivers are
frozen read-only features. The Meta-Pool is forced to find patterns
that minimize surprise for ALL languages simultaneously.

Architecture:
  Phase 1: Sequential training (5 min/lang) — establishes slivers
  Phase 2: Round-Robin consolidation (20 min) — meta-pool converges
"""

import torch
import time
import os
import json
import random
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, AbstraXEngine

META_POOL_SIZE = 64
N_PER_LANG = 128
PHASE1_DURATION = 300   # 5 min per language (sequential)
PHASE2_DURATION = 1200  # 20 min total (round-robin consolidation)
SWITCH_EVERY = 50       # Switch language every 50 tokens

class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    def encode(self, s): return [self.char_to_id[c] for c in s if c in self.char_to_id]
    def decode(self, ids): return "".join([self.id_to_char[i] for i in ids])


def avg_surprise_next_char(hierarchy, tokenizer, tokens, gate_start, gate_end, device, n_samples=200):
    surprises = []
    with hierarchy.manifold_gate(gate_start, gate_end):
        hierarchy.reset_states()
        with torch.no_grad():
            for i in range(min(n_samples, len(tokens) - 1)):
                x = torch.zeros((1, tokenizer.vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                target = torch.zeros((1, tokenizer.vocab_size), device=device)
                target[0, tokens[i + 1]] = 1.0
                surprises.append(hierarchy.get_surprise((x, target)))
    return sum(surprises) / max(1, len(surprises))


def freeze_all_masks(hierarchy):
    """Freeze every single weight in the hierarchy."""
    for layer in hierarchy.layers:
        layer.V_mask.zero_()
        layer.W_mask.zero_()
        layer.b_in_mask.zero_()
        layer.b_out_mask.zero_()
        layer.R_mask.zero_()
        layer.R_gate_mask.zero_()
        layer.L_mask.zero_()


def unmask_metapool_only(hierarchy, mp):
    """Unmask ONLY the meta-pool neurons. Everything else stays frozen."""
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


def run_interleaved_marathon():
    marathon_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = ["en", "de", "es", "fr"]
    lang_names = {"en": "English", "de": "German", "es": "Spanish", "fr": "French"}

    # Load Data
    full_text = ""
    corpora = {}
    for code in langs:
        with open(f"slm/input_{code}.txt", "r", encoding="utf-8") as f:
            text = f.read()[:50000]
            corpora[code] = text
            full_text += text
    tokenizer = SimpleTokenizer(full_text)

    lang_ranges = {}
    for i, lang in enumerate(langs):
        start = META_POOL_SIZE + i * N_PER_LANG
        end = start + N_PER_LANG
        lang_ranges[lang] = (start, end)

    initial_hidden = META_POOL_SIZE + N_PER_LANG
    final_width = META_POOL_SIZE + N_PER_LANG * len(langs)

    print("\n" + "=" * 60)
    print("  AGNIS V16 — INTERLEAVED META-POOL CONSOLIDATION")
    print("=" * 60)
    print(f"  Phase 1: Sequential training ({PHASE1_DURATION//60} min/lang)")
    print(f"  Phase 2: Round-Robin consolidation ({PHASE2_DURATION//60} min)")
    print(f"  Meta-Pool: {META_POOL_SIZE} | Sliver: {N_PER_LANG}")
    print(f"  Languages: {langs}")
    print(f"  Ranges: {lang_ranges}")
    print("=" * 60)

    # Build hierarchy
    hierarchy = PredictiveHierarchy(
        [tokenizer.vocab_size, initial_hidden, initial_hidden, tokenizer.vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    
    # ═══════════════════════════════════════════
    # PHASE 1: Sequential Training (establish slivers)
    # ═══════════════════════════════════════════
    print("\n╔══════════════════════════════════════╗")
    print("║  PHASE 1: SEQUENTIAL ESTABLISHMENT   ║")
    print("╚══════════════════════════════════════╝")

    all_tokens = {}
    for phase_idx, code in enumerate(langs):
        name = lang_names[code]
        print(f"\n>>> Training {name.upper()} ({PHASE1_DURATION//60} min) <<<")
        tokens = tokenizer.encode(corpora[code])
        all_tokens[code] = tokens
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

                x = torch.zeros((1, tokenizer.vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                target = torch.zeros((1, tokenizer.vocab_size), device=device)
                target[0, tokens[i+1]] = 1.0
                hierarchy.infer_and_learn(x, top_level_label=target, dopamine_burst=1.0)

                if i % 2000 == 0:
                    print(f"  [{name}] Token {i:5d} | {elapsed/PHASE1_DURATION:.0%}", end="\r")

        print(f"\n  {name} phase complete.")
        if phase_idx < len(langs) - 1:
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # ═══════════════════════════════════════════
    # PHASE 2: Round-Robin Consolidation (meta-pool only)
    # ═══════════════════════════════════════════
    print("\n╔══════════════════════════════════════╗")
    print("║  PHASE 2: ROUND-ROBIN CONSOLIDATION  ║")
    print("╚══════════════════════════════════════╝")

    # Freeze EVERYTHING, then unmask ONLY meta-pool
    freeze_all_masks(hierarchy)
    unmask_metapool_only(hierarchy, META_POOL_SIZE)
    print(f"  All slivers frozen. Only meta-pool ({META_POOL_SIZE} neurons) trainable.")

    # Lower learning rate for gentle consolidation
    for layer in hierarchy.layers:
        layer.eta_V = 0.01
        layer.eta_W = 0.005
        layer.eta_R = 0.005

    start_time = time.time()
    token_count = 0
    lang_switches = 0
    lang_cycle = langs * 1000  # Infinite cycle
    lang_idx = 0
    current_lang = lang_cycle[0]
    pos_in_lang = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > PHASE2_DURATION: break

        # Switch language every SWITCH_EVERY tokens
        if token_count % SWITCH_EVERY == 0 and token_count > 0:
            lang_idx = (lang_idx + 1) % len(langs)
            current_lang = langs[lang_idx]
            pos_in_lang = random.randint(0, len(all_tokens[current_lang]) - 100)
            lang_switches += 1
            hierarchy.reset_states()

        tokens = all_tokens[current_lang]
        idx = pos_in_lang % (len(tokens) - 1)
        lang_start, lang_end = lang_ranges[current_lang]

        # Gate to FULL width so meta-pool sees all frozen slivers as context
        with hierarchy.manifold_gate(0, final_width):
            x = torch.zeros((1, tokenizer.vocab_size), device=device)
            x[0, tokens[idx]] = 1.0
            target = torch.zeros((1, tokenizer.vocab_size), device=device)
            target[0, tokens[idx + 1]] = 1.0
            hierarchy.infer_and_learn(x, top_level_label=target, dopamine_burst=1.0)

        pos_in_lang += 1
        token_count += 1

        if token_count % 2000 == 0:
            print(f"  [Consolidation] Tokens: {token_count} | Switches: {lang_switches} | {elapsed/PHASE2_DURATION:.0%} | Lang: {current_lang}", end="\r")

    print(f"\n  Consolidation complete: {token_count} tokens, {lang_switches} language switches")

    # Save
    hierarchy.save_checkpoint("agnis_interleaved_final.pt")
    print("[Checkpoint] Saved.")

    # ═══════════════════════════════════════════
    # AUDIT: Retention + AbstraX
    # ═══════════════════════════════════════════
    print("\n>>> RETENTION AUDIT <<<")
    hierarchy.load_checkpoint("agnis_interleaved_final.pt")
    baseline_surprises = {}

    for code in langs:
        tokens = all_tokens[code]
        lang_start, lang_end = lang_ranges[code]
        surprise = avg_surprise_next_char(hierarchy, tokenizer, tokens, 0, lang_end, device=device)
        baseline_surprises[code] = surprise
        print(f"  {code}: surprise={surprise:.4f}")

    # AbstraX Dream Cycle
    print("\n>>> ABSTRAX DREAM CYCLE <<<")
    hierarchy.load_checkpoint("agnis_interleaved_final.pt")

    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for layer_idx in range(len(hierarchy.layers)):
        print(f"\n--- Layer {layer_idx} ---")
        try:
            result = abstrax.compute_pairwise_affinity(layer_idx=layer_idx)
            pairs = abstrax.print_affinity_report(result, title=f"Layer {layer_idx} Affinity")
            abstrax.identify_fold_candidates(result, threshold=0.2)
        except Exception as e:
            print(f"  [SKIP] Layer {layer_idx}: {e}")

    total_time = (time.time() - marathon_start) / 60
    print(f"\n{'='*60}")
    print(f"  V16 INTERLEAVED MARATHON COMPLETE — {total_time:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_interleaved_marathon()
