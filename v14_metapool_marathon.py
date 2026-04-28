"""
v14_metapool_marathon.py — The Shared Meta-Pool Experiment
==========================================================
V3 Phase 32: Testing whether shared ungated neurons enable
cross-language structural convergence.

Architecture:
  - 64 Meta-Pool neurons (NEVER frozen, shared by all languages)
  - 256 Language-specific neurons per language (frozen after training)
  - 4 languages for speed: English, German, Spanish, French

The hypothesis: If the meta-pool neurons converge to shared patterns,
the AbstraX Engine will detect affinity > 0.3 in the meta-pool range,
proving that the network independently discovered shared structure.
"""

import torch
import time
import os
import json
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, AbstraXEngine

META_POOL_SIZE = 64
N_PER_LANG = 256
PHASE_DURATION = 600  # 10 minutes per language

class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    def encode(self, s): return [self.char_to_id[c] for c in s if c in self.char_to_id]
    def decode(self, ids): return "".join([self.id_to_char[i] for i in ids])

def compute_lang_ranges(langs, meta_pool_size, n_per_lang):
    """Language slivers start AFTER the meta-pool."""
    ranges = {}
    for i, lang in enumerate(langs):
        start = meta_pool_size + i * n_per_lang
        end = start + n_per_lang
        ranges[lang] = (start, end)
    return ranges

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

def run_metapool_marathon():
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

    lang_ranges = compute_lang_ranges(langs, META_POOL_SIZE, N_PER_LANG)
    initial_hidden = META_POOL_SIZE + N_PER_LANG  # meta-pool + first language
    final_width = META_POOL_SIZE + N_PER_LANG * len(langs)

    print("\n" + "=" * 60)
    print("  AGNIS V14 — META-POOL MARATHON")
    print("  Shared Neuron Convergence Experiment")
    print("=" * 60)
    print(f"  Meta-Pool Size: {META_POOL_SIZE} (shared, never frozen)")
    print(f"  Language Sliver: {N_PER_LANG} per language")
    print(f"  Languages: {langs}")
    print(f"  Ranges: {lang_ranges}")
    print(f"  Initial Hidden: {initial_hidden}")
    print(f"  Final Width: {final_width}")
    print("=" * 60)

    # 2. Build Hierarchy WITH meta-pool
    hierarchy = PredictiveHierarchy(
        [tokenizer.vocab_size, initial_hidden, tokenizer.vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    baseline_surprises = {}

    # 3. Training Loop
    for phase_idx, code in enumerate(langs):
        name = lang_names[code]
        print(f"\n>>> PHASE {phase_idx+1}: TRAINING ON {name.upper()} ({PHASE_DURATION//60} Minutes) <<<")

        tokens = tokenizer.encode(corpora[code])
        start_time = time.time()
        lang_start, lang_end = lang_ranges[code]

        # Gate from 0 to lang_end: includes meta-pool + all previous (frozen) + current
        gate_start = 0
        gate_end = lang_end
        print(f"  Gate range: ({gate_start}, {gate_end}) — meta-pool + active sliver")

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
                    print(f"  [{name}] Token {i:5d} | Progress: {progress:.1%} | Scale: {scale:.2f}", end="\r")

        # Measure baseline surprise using the FULL gate (meta-pool + language sliver)
        baseline_surprises[code] = avg_surprise_next_char(
            hierarchy, tokenizer, tokens, 0, lang_end, device=device
        )
        print(f"\n  {name} Baseline Surprise: {baseline_surprises[code]:.4f}")

        # Recruit next language sliver (freeze current + expand)
        if phase_idx < len(langs) - 1:
            print(f"  Igniting Synaptic Shield + Neurogenesis...")
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # 4. Save
    hierarchy.save_checkpoint("agnis_metapool_final.pt")
    with open("agnis_metapool_baselines.json", "w") as f:
        json.dump(baseline_surprises, f, indent=2, sort_keys=True)
    print("\n[Checkpoint] Meta-Pool Marathon saved.")

    # 5. Retention Audit
    print("\n>>> RETENTION AUDIT <<<")
    hierarchy.load_checkpoint("agnis_metapool_final.pt")
    for code in langs:
        tokens = tokenizer.encode(corpora[code])
        lang_start, lang_end = lang_ranges[code]
        surprise = avg_surprise_next_char(hierarchy, tokenizer, tokens, 0, lang_end, device=device)
        baseline = baseline_surprises[code]
        drift = abs(surprise - baseline)
        retention = max(0.0, 1.0 - (drift / max(1e-8, baseline))) * 100.0
        status = "✅" if retention >= 90.0 else "⚠️" if retention >= 70.0 else "❌"
        print(f"  {status} {code}: surprise={surprise:.4f} | baseline={baseline:.4f} | drift={drift:.4f} | retention={retention:.1f}%")

    # 6. AbstraX Dream Cycle on the Meta-Pool
    print("\n>>> ABSTRAX DREAM CYCLE (META-POOL ANALYSIS) <<<")
    hierarchy.load_checkpoint("agnis_metapool_final.pt")

    # Build ranges including meta-pool as its own "language" for affinity analysis
    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)

    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)
    result = abstrax.compute_pairwise_affinity(layer_idx=0)
    pairs = abstrax.print_affinity_report(result, title="Meta-Pool Cross-Domain Affinity (Layer 0)")
    abstrax.identify_fold_candidates(result, threshold=0.3)

    # Summary
    total_time = (time.time() - marathon_start) / 60
    print(f"\n{'='*60}")
    print(f"  META-POOL MARATHON COMPLETE")
    print(f"  Total time: {total_time:.1f} minutes")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_metapool_marathon()
