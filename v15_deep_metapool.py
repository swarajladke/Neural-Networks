"""
v15_deep_metapool.py — Deep Hierarchy + Meta-Pool
===================================================
V3 Phase 33: Testing whether placing the Meta-Pool in a HIGHER layer
(Layer 2) enables cross-language convergence.

Hypothesis:
  The V14 experiment failed because the Meta-Pool was in Layer 0,
  where it sees raw character bytes. Character distributions differ
  wildly between languages, so no shared structure can emerge.

  By adding a THIRD layer and placing the Meta-Pool in Layer 1 (the
  intermediate compression layer), the shared neurons observe compressed
  latent representations rather than raw bytes. If English and French
  both compress "Subject-Verb-Object" into similar latent patterns,
  the Meta-Pool will converge.

Architecture:
  Layer 0: [vocab → 256] Language-specific character encoder (GATED)
  Layer 1: [256 → meta_pool + 128] Shared abstraction layer (META-POOL HERE)
  Layer 2: [meta_pool + 128 → vocab] Readout layer

This is a 4-language test: English, German, Spanish, French
"""

import torch
import time
import os
import json
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, AbstraXEngine

META_POOL_SIZE = 64
N_PER_LANG = 128  # Smaller slivers since we have 3 layers now
PHASE_DURATION = 600  # 10 minutes per language

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


def run_deep_metapool():
    marathon_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = ["en", "de", "es", "fr"]
    lang_names = {"en": "English", "de": "German", "es": "Spanish", "fr": "French"}

    # 1. Load Data
    full_text = ""
    corpora = {}
    for code in langs:
        path = f"slm/input_{code}.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()[:50000]
            corpora[code] = text
            full_text += text
    tokenizer = SimpleTokenizer(full_text)

    # 2. Compute language ranges
    # Each language gets N_PER_LANG neurons in the hidden layers
    # The meta-pool occupies [0:META_POOL_SIZE] in layer 1
    lang_ranges = {}
    for i, lang in enumerate(langs):
        start = META_POOL_SIZE + i * N_PER_LANG
        end = start + N_PER_LANG
        lang_ranges[lang] = (start, end)

    initial_hidden = META_POOL_SIZE + N_PER_LANG  # Layer 1 width for first language
    final_width = META_POOL_SIZE + N_PER_LANG * len(langs)

    print("\n" + "=" * 60)
    print("  AGNIS V15 — DEEP HIERARCHY + META-POOL")
    print("  3-Layer Predictive Coding Stack")
    print("=" * 60)
    print(f"  Architecture: [{tokenizer.vocab_size} → {initial_hidden} → {initial_hidden} → {tokenizer.vocab_size}]")
    print(f"  Meta-Pool: {META_POOL_SIZE} neurons in Layer 1 (abstraction layer)")
    print(f"  Language Sliver: {N_PER_LANG} per language")
    print(f"  Languages: {langs}")
    print(f"  Ranges: {lang_ranges}")
    print(f"  Final Width: {final_width}")
    print("=" * 60)

    # 3. Build 3-layer hierarchy with meta-pool
    hierarchy = PredictiveHierarchy(
        [tokenizer.vocab_size, initial_hidden, initial_hidden, tokenizer.vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    baseline_surprises = {}

    print(f"\n  Layer count: {len(hierarchy.layers)}")
    for i, layer in enumerate(hierarchy.layers):
        print(f"  Layer {i}: V={list(layer.V.shape)}, R={list(layer.R.shape)}")

    # 4. Training Loop
    for phase_idx, code in enumerate(langs):
        name = lang_names[code]
        print(f"\n>>> PHASE {phase_idx+1}: TRAINING ON {name.upper()} ({PHASE_DURATION//60} Min) <<<")

        tokens = tokenizer.encode(corpora[code])
        start_time = time.time()
        lang_start, lang_end = lang_ranges[code]

        # Gate includes meta-pool + all previous (frozen) + current sliver
        gate_start = 0
        gate_end = lang_end
        print(f"  Gate: ({gate_start}, {gate_end})")

        with hierarchy.manifold_gate(gate_start, gate_end):
            hierarchy.reset_states()
            for i in range(len(tokens) - 1):
                elapsed = time.time() - start_time
                if elapsed > PHASE_DURATION: break

                scale = agent.get_dopamine_scale(elapsed, PHASE_DURATION)
                for layer in hierarchy.layers:
                    layer.eta_V = 0.05 * scale
                    layer.eta_W = 0.03 * scale

                x = torch.zeros((1, tokenizer.vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                target = torch.zeros((1, tokenizer.vocab_size), device=device)
                target[0, tokens[i+1]] = 1.0

                hierarchy.infer_and_learn(x, top_level_label=target, dopamine_burst=1.0)

                if i % 1000 == 0:
                    progress = elapsed / PHASE_DURATION
                    print(f"  [{name}] Token {i:5d} | {progress:.0%} | Scale: {scale:.2f}", end="\r")

        baseline_surprises[code] = avg_surprise_next_char(
            hierarchy, tokenizer, tokens, 0, lang_end, device=device
        )
        print(f"\n  {name} Baseline Surprise: {baseline_surprises[code]:.4f}")

        if phase_idx < len(langs) - 1:
            print(f"  Synaptic Shield + Neurogenesis...")
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # 5. Save
    hierarchy.save_checkpoint("agnis_deep_metapool_final.pt")
    with open("agnis_deep_metapool_baselines.json", "w") as f:
        json.dump(baseline_surprises, f, indent=2, sort_keys=True)
    print("\n[Checkpoint] Deep Meta-Pool Marathon saved.")

    # 6. Retention Audit
    print("\n>>> RETENTION AUDIT <<<")
    hierarchy.load_checkpoint("agnis_deep_metapool_final.pt")
    for code in langs:
        tokens = tokenizer.encode(corpora[code])
        lang_start, lang_end = lang_ranges[code]
        surprise = avg_surprise_next_char(hierarchy, tokenizer, tokens, 0, lang_end, device=device)
        baseline = baseline_surprises[code]
        drift = abs(surprise - baseline)
        retention = max(0.0, 1.0 - (drift / max(1e-8, baseline))) * 100.0
        status = "✅" if retention >= 90.0 else "⚠️" if retention >= 70.0 else "❌"
        print(f"  {status} {code}: surprise={surprise:.4f} | baseline={baseline:.4f} | drift={drift:.4f} | retention={retention:.1f}%")

    # 7. AbstraX Dream Cycle — analyze EACH layer separately
    print("\n>>> ABSTRAX DREAM CYCLE <<<")
    hierarchy.load_checkpoint("agnis_deep_metapool_final.pt")

    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for layer_idx in range(len(hierarchy.layers)):
        print(f"\n--- Layer {layer_idx} ---")
        try:
            result = abstrax.compute_pairwise_affinity(layer_idx=layer_idx)
            pairs = abstrax.print_affinity_report(result, title=f"Layer {layer_idx} Affinity")
            abstrax.identify_fold_candidates(result, threshold=0.3)
        except Exception as e:
            print(f"  [SKIP] Layer {layer_idx}: {e}")

    total_time = (time.time() - marathon_start) / 60
    print(f"\n{'='*60}")
    print(f"  DEEP META-POOL MARATHON COMPLETE — {total_time:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_deep_metapool()
