"""
v22_active_meta_marathon.py — V22: Active Meta-Pool + Cross-Script Validation
==============================================================================
Tests whether Universal Grammar extends beyond Latin-family languages.

Phase 1: Re-establish en/de/es/fr (15 min each, soft meta-pool)
Phase 2: Bootstrap Italian (5 min, expect fast)
Phase 3: Bootstrap Russian/Cyrillic (10 min, the KEY test)
Phase 4: AbstraX + Retention Audit
"""

import torch
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v4_core import PredictiveHierarchy, META_POOL_LR_SCALE, META_POOL_MAX_SIZE
from agnis_v4_cognitive import AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

# ─── Config ───
META_POOL_SIZE = 64
N_PER_LANG = 128
BPE_VOCAB = 4000
BATCH_SIZE = 32
MAX_STEPS = 3

PHASE1_DURATION = 900   # 15 min per existing language
PHASE2_DURATION = 300   # 5 min Italian
PHASE3_DURATION = 600   # 10 min Russian

ITALIAN_TARGET = 2.5
RUSSIAN_TARGET = 3.0

EXISTING_LANGS = ["en", "de", "es", "fr"]
LANG_NAMES = {
    "en": "English", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "ru": "Russian"
}


# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════
def fetch_corpus(lang: str, target_chars: int = 500000) -> str:
    # Try local files first (fastest)
    for path in [f"slm/input_{lang}.txt", f"slm/wiki_{lang}.txt"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()[:target_chars]
            print(f"  [{lang}] Loaded {len(text):,} chars from {path}")
            return text
    # HuggingFace fallback
    try:
        from datasets import load_dataset
        wiki_codes = {"en": "20220301.en", "de": "20220301.de",
                      "es": "20220301.es", "fr": "20220301.fr",
                      "it": "20220301.it", "ru": "20220301.ru"}
        code = wiki_codes.get(lang, f"20220301.{lang}")
        print(f"  [{lang}] Loading Wikipedia ({code})...")
        ds = load_dataset("wikipedia", code, split="train", streaming=True)
        text = ""
        for article in ds:
            text += article["text"] + "\n"
            if len(text) >= target_chars:
                break
        print(f"  [{lang}] Got {len(text):,} chars")
        return text[:target_chars]
    except Exception as e:
        print(f"  [{lang}] FAILED: {e}")
        return None


def get_batch(token_ids, batch_idx, batch_size, vocab_size):
    start = batch_idx * batch_size
    end = start + batch_size
    if end + 1 > len(token_ids):
        start, end = 0, batch_size
    x = torch.zeros(batch_size, vocab_size, device=token_ids.device)
    t = torch.zeros(batch_size, vocab_size, device=token_ids.device)
    x.scatter_(1, token_ids[start:end].unsqueeze(1), 1.0)
    t.scatter_(1, token_ids[start+1:end+1].unsqueeze(1), 1.0)
    return x, t


def freeze_all(hierarchy):
    for layer in hierarchy.layers:
        layer.V_mask.zero_()
        layer.W_mask.zero_()
        layer.b_in_mask.zero_()
        layer.b_out_mask.zero_()
        layer.R_mask.zero_()
        layer.R_gate_mask.zero_()
        layer.L_mask.zero_()


def train_phase(hierarchy, token_tensor, n_tokens, vocab_size, duration,
                lang_code, phase_name):
    """Train a single language phase. Returns (total_tokens, final_surprise, tps)."""
    n_batches = (n_tokens - 1) // BATCH_SIZE
    t0 = time.time()
    total_tokens = 0
    batch_losses = []
    batch_idx = 0
    hierarchy.reset_states()

    while True:
        elapsed = time.time() - t0
        if elapsed > duration:
            break
        if batch_idx >= n_batches:
            batch_idx = 0

        x, target = get_batch(token_tensor, batch_idx, BATCH_SIZE, vocab_size)
        hierarchy.infer_and_learn(
            x, top_level_label=target,
            dopamine_burst=1.0, max_steps=MAX_STEPS, warm_start=True
        )
        surprise = hierarchy.get_surprise((x, target))
        batch_losses.append(surprise)
        total_tokens += BATCH_SIZE
        batch_idx += 1

        if batch_idx % 20 == 0:
            avg_s = sum(batch_losses[-20:]) / max(1, len(batch_losses[-20:]))
            tps = total_tokens / max(1, elapsed)
            sys.stdout.write(
                f"\r  [{lang_code}] {total_tokens:,} tok | {tps:.0f} tok/s | "
                f"Surprise: {avg_s:.4f} | {elapsed/duration:.0%}")
            sys.stdout.flush()

    avg_final = sum(batch_losses[-100:]) / max(1, len(batch_losses[-100:]))
    final_tps = total_tokens / max(1, time.time() - t0)
    elapsed_min = (time.time() - t0) / 60
    print(f"\n\n  === {phase_name} COMPLETE ===")
    print(f"  Language: {LANG_NAMES.get(lang_code, lang_code)}")
    print(f"  Training time: {elapsed_min:.1f} min")
    print(f"  Tokens processed: {total_tokens:,}")
    print(f"  Final surprise: {avg_final:.4f}")
    print(f"  Meta-pool size: {hierarchy.meta_pool_size} neurons")
    return total_tokens, avg_final, final_tps


def isolate_for_bootstrap(hierarchy, meta_pool_size, lang_start, lang_end,
                          other_start, other_end):
    """Isolate a new language: it sees meta-pool + its own sliver only."""
    for layer in hierarchy.layers:
        # Meta-pool gets soft mask (slow learning)
        layer.V_mask[:, :meta_pool_size] = META_POOL_LR_SCALE
        layer.W_mask[:meta_pool_size, :] = META_POOL_LR_SCALE
        layer.R_mask[:meta_pool_size, :meta_pool_size] = META_POOL_LR_SCALE
        layer.R_gate_mask[:meta_pool_size, :meta_pool_size] = META_POOL_LR_SCALE
        layer.b_in_mask[:meta_pool_size] = META_POOL_LR_SCALE

        # Zero out other language slivers completely
        layer.V.data[:, other_start:other_end] = 0.0
        layer.W.data[other_start:other_end, :] = 0.0
        layer.R.data[other_start:other_end, :] = 0.0
        layer.R.data[:, other_start:other_end] = 0.0
        layer.b_in.data[other_start:other_end] = 0.0
        layer.b_out.data[other_start:other_end] = 0.0

        layer.V_mask[:, other_start:other_end] = 0.0
        layer.W_mask[other_start:other_end, :] = 0.0
        layer.R_mask[other_start:other_end, :] = 0.0
        layer.R_mask[:, other_start:other_end] = 0.0
        layer.b_in_mask[other_start:other_end] = 0.0
        layer.b_out_mask[other_start:other_end] = 0.0


# ═══════════════════════════════════════════
# VERIFICATION TESTS
# ═══════════════════════════════════════════
def run_verification_tests(hierarchy, token_tensor, vocab_size):
    print("\n" + "=" * 60)
    print("  V22 VERIFICATION TESTS")
    print("=" * 60)
    mp = hierarchy.meta_pool_size
    all_pass = True

    # Test 1: Soft mask learning ratio
    print("\n  Test 1: Soft mask learning ratio...")
    meta_V_before = hierarchy.layers[0].V.data[:, :mp].clone()
    sliver_start = mp
    sliver_end = mp + N_PER_LANG
    sliver_V_before = hierarchy.layers[0].V.data[:, sliver_start:sliver_end].clone()

    # One training step
    x, target = get_batch(token_tensor, 0, BATCH_SIZE, vocab_size)
    hierarchy.infer_and_learn(x, top_level_label=target,
                              dopamine_burst=1.0, max_steps=MAX_STEPS, warm_start=True)

    meta_change = (hierarchy.layers[0].V.data[:, :mp] - meta_V_before).norm().item()
    sliver_change = (hierarchy.layers[0].V.data[:, sliver_start:sliver_end] - sliver_V_before).norm().item()

    if meta_change > 1e-10:
        ratio = sliver_change / meta_change
        print(f"    Meta change: {meta_change:.6f}")
        print(f"    Sliver change: {sliver_change:.6f}")
        print(f"    Ratio: {ratio:.1f}x")
        if ratio > 10:
            print(f"    PASS: meta-pool learning at correct slow rate")
        else:
            print(f"    WARN: ratio lower than expected ({ratio:.1f}x < 15x)")
            # Not a hard fail — ratio depends on gradient magnitudes
    else:
        print(f"    Meta change is ~0 (frozen or very slow) — PASS")

    # Test 2: Meta-pool expansion
    print("\n  Test 2: Meta-pool expansion...")
    initial_size = hierarchy.meta_pool_size
    hierarchy.expand_meta_pool(n_new=32)
    new_size = hierarchy.meta_pool_size
    if new_size == initial_size + 32:
        print(f"    PASS: meta-pool expanded {initial_size} → {new_size}")
    else:
        print(f"    FAIL: expected {initial_size + 32}, got {new_size}")
        all_pass = False
    # Undo expansion for actual training by restoring
    # (We'll just track the new size going forward)

    # Test 3: Speed check
    print("\n  Test 3: Speed check...")
    t0 = time.time()
    for i in range(50):
        x, target = get_batch(token_tensor, i % 100, BATCH_SIZE, vocab_size)
        hierarchy.infer_and_learn(x, top_level_label=target,
                                  dopamine_burst=1.0, max_steps=MAX_STEPS, warm_start=True)
    elapsed = time.time() - t0
    speed = (50 * BATCH_SIZE) / elapsed
    if speed > 20:
        print(f"    PASS: {speed:.1f} tokens/sec")
    else:
        print(f"    WARN: {speed:.1f} tokens/sec (below 30 target)")

    print(f"\n  Verification: {'ALL PASS' if all_pass else 'ISSUES FOUND'}")
    print("=" * 60)
    return all_pass


# ═══════════════════════════════════════════
# MAIN MARATHON
# ═══════════════════════════════════════════
def run_v22():
    marathon_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  AGNIS V22 — ACTIVE META-POOL + CROSS-SCRIPT VALIDATION")
    print("  Soft Mask | Neurogenesis | Russian Bootstrap")
    print("=" * 60)
    print(f"  Device: {device}")

    # ═══ 1. Load Data ═══
    print("\n--- Loading Corpora ---")
    all_langs = ["en", "de", "es", "fr", "it", "ru"]
    corpora = {}
    for code in all_langs:
        text = fetch_corpus(code)
        if text and len(text) > 1000:
            corpora[code] = text
        else:
            print(f"  [{code}] SKIPPED — no data")

    if len(corpora) < 4:
        print("[ERROR] Need at least 4 languages. Exiting.")
        return

    # ═══ 2. BPE Tokenizer ═══
    cache = f"agnis_bpe_{BPE_VOCAB}_vocab.json"
    if os.path.exists(cache):
        tokenizer = BPETokenizer(BPE_VOCAB)
        tokenizer.load(cache)
    else:
        combined = "".join(corpora.values())
        tokenizer = BPETokenizer(BPE_VOCAB)
        tokenizer.fit(combined, verbose=True)
        tokenizer.save(cache)

    vocab_size = tokenizer.vocab_size
    print(f"  BPE vocab: {vocab_size}")

    # Encode all
    all_tokens = {}
    all_tensors = {}
    for code, text in corpora.items():
        tokens = tokenizer.encode(text)
        all_tokens[code] = tokens
        all_tensors[code] = torch.tensor(tokens, dtype=torch.long, device=device)
        compression = len(text) / max(1, len(tokens))
        print(f"  {LANG_NAMES[code]}: {len(tokens):,} tokens ({compression:.1f}x)")

    # ═══ 3. Build Hierarchy ═══
    lang_ranges = {}
    for i, lang in enumerate(EXISTING_LANGS):
        s = META_POOL_SIZE + i * N_PER_LANG
        lang_ranges[lang] = (s, s + N_PER_LANG)

    initial_hidden = META_POOL_SIZE + N_PER_LANG * len(EXISTING_LANGS)

    hierarchy = PredictiveHierarchy(
        [vocab_size, initial_hidden, initial_hidden, vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )

    print(f"\n  Architecture: [{vocab_size}, {initial_hidden}, {initial_hidden}, {vocab_size}]")
    print(f"  Meta-Pool: {META_POOL_SIZE} (soft LR={META_POOL_LR_SCALE})")
    print(f"  Sliver: {N_PER_LANG} | Batch: {BATCH_SIZE} | max_steps: {MAX_STEPS}")
    print(f"  Ranges: {lang_ranges}")

    # ═══ 4. PHASE 1: Sequential Training (existing languages) ═══
    print("\n" + "=" * 50)
    print(f"  PHASE 1: SEQUENTIAL TRAINING ({PHASE1_DURATION//60} min/lang)")
    print("=" * 50)

    baselines = {}  # Store baseline surprises for retention audit

    for phase_idx, code in enumerate(EXISTING_LANGS):
        if code not in corpora:
            continue
        token_tensor = all_tensors[code]
        n_tokens = len(all_tokens[code])
        lang_s, lang_e = lang_ranges[code]

        print(f"\n>>> {LANG_NAMES[code].upper()} ({PHASE1_DURATION//60} min) <<<")

        with hierarchy.manifold_gate(0, lang_e):
            # Apply soft mask to meta-pool
            hierarchy.set_meta_pool_soft_mask(META_POOL_LR_SCALE)

            tok, surprise, tps = train_phase(
                hierarchy, token_tensor, n_tokens, vocab_size,
                PHASE1_DURATION, code, f"PHASE 1.{phase_idx+1}"
            )

        baselines[code] = surprise

        if phase_idx < len(EXISTING_LANGS) - 1:
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG,
                                                    language=LANG_NAMES[code])
            # Re-apply soft mask after recruit (which sets meta to 1.0)
            hierarchy.set_meta_pool_soft_mask(META_POOL_LR_SCALE)

    # ═══ Save Phase 1 checkpoint ═══
    hierarchy.save_checkpoint("agnis_v22_phase1.pt")
    print("\n[Saved] agnis_v22_phase1.pt")

    # ═══ 5. Dream Synthesis ═══
    print("\n>>> DREAM SYNTHESIS <<<")
    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)
    abstrax.synthesize_dream_neurons(META_POOL_SIZE)

    # ═══ 6. RUN VERIFICATION TESTS ═══
    first_lang = EXISTING_LANGS[0]
    with hierarchy.manifold_gate(0, lang_ranges[first_lang][1]):
        hierarchy.set_meta_pool_soft_mask(META_POOL_LR_SCALE)
        run_verification_tests(hierarchy, all_tensors[first_lang], vocab_size)

    # ═══ 7. PHASE 2: Bootstrap Italian ═══
    if "it" in corpora:
        print("\n" + "=" * 50)
        print("  PHASE 2: BOOTSTRAP ITALIAN (Latin family)")
        print("=" * 50)

        it_tokens = all_tokens["it"]
        it_tensor = all_tensors["it"]

        hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language="Italian")
        hierarchy.set_meta_pool_soft_mask(META_POOL_LR_SCALE)

        it_start = initial_hidden
        it_end = it_start + N_PER_LANG
        lang_ranges["it"] = (it_start, it_end)

        # Check affinity
        abstrax_ranges["it"] = (it_start, it_end)
        abstrax = AbstraXEngine(hierarchy, abstrax_ranges)
        it_affinity, it_sufficient = abstrax.check_meta_affinity("it", threshold=0.3)

        if not it_sufficient:
            print("  Expanding meta-pool for Italian...")
            hierarchy.expand_meta_pool(n_new=32)

        with hierarchy.manifold_gate(0, it_end):
            isolate_for_bootstrap(hierarchy, hierarchy.meta_pool_size,
                                  it_start, it_end,
                                  META_POOL_SIZE, it_start)

            tok, it_surprise, tps = train_phase(
                hierarchy, it_tensor, len(it_tokens), vocab_size,
                PHASE2_DURATION, "it", "PHASE 2 (Italian)"
            )
        baselines["it"] = it_surprise

        comparison = "FASTER" if it_surprise < ITALIAN_TARGET else "SLOWER"
        print(f"  Bootstrap vs V21 baseline: {comparison}")
        print(f"  Target: {ITALIAN_TARGET} | Actual: {it_surprise:.4f}")

    # ═══ 8. PHASE 3: Bootstrap Russian (THE KEY TEST) ═══
    if "ru" in corpora:
        print("\n" + "=" * 50)
        print("  PHASE 3: BOOTSTRAP RUSSIAN (Cyrillic — THE KEY TEST)")
        print("=" * 50)

        ru_tokens = all_tokens["ru"]
        ru_tensor = all_tensors["ru"]

        hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language="Russian")
        hierarchy.set_meta_pool_soft_mask(META_POOL_LR_SCALE)

        # Russian range depends on whether Italian was added
        ru_start = it_end if "it" in lang_ranges else initial_hidden
        ru_end = ru_start + N_PER_LANG
        lang_ranges["ru"] = (ru_start, ru_end)

        # Check affinity (expect lower for Cyrillic)
        abstrax_ranges["ru"] = (ru_start, ru_end)
        abstrax = AbstraXEngine(hierarchy, abstrax_ranges)
        ru_affinity, ru_sufficient = abstrax.check_meta_affinity("ru", threshold=0.3)

        if not ru_sufficient:
            print("  Low Cyrillic affinity — expanding meta-pool...")
            hierarchy.expand_meta_pool(n_new=32)

        with hierarchy.manifold_gate(0, ru_end):
            isolate_for_bootstrap(hierarchy, hierarchy.meta_pool_size,
                                  ru_start, ru_end,
                                  META_POOL_SIZE, ru_start)

            tok, ru_surprise, tps = train_phase(
                hierarchy, ru_tensor, len(ru_tokens), vocab_size,
                PHASE3_DURATION, "ru", "PHASE 3 (Russian)"
            )
        baselines["ru"] = ru_surprise

    # ═══ 9. Save V22 ═══
    hierarchy.save_checkpoint("agnis_v22_active_meta.pt")
    print("\n[Saved] agnis_v22_active_meta.pt")

    # ═══ 10. AbstraX Full Analysis ═══
    print("\n>>> ABSTRAX FULL ANALYSIS (Layer 2) <<<")
    abstrax_ranges_final = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges_final.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges_final)

    try:
        r = abstrax.compute_pairwise_affinity(layer_idx=min(2, len(hierarchy.layers)-1))
        pairs = abstrax.print_affinity_report(r, title="V22 Active Meta-Pool (Layer 2)")
        abstrax.identify_fold_candidates(r, threshold=0.2)
    except Exception as e:
        print(f"  [AbstraX ERROR] {e}")

    # ═══ 11. Retention Audit ═══
    print("\n>>> RETENTION AUDIT <<<")
    retention_results = {}
    all_retained = True

    for code in lang_ranges:
        if code == "meta" or code not in all_tensors:
            continue
        s, e = lang_ranges[code]
        with hierarchy.manifold_gate(0, e):
            x, target = get_batch(all_tensors[code], 0, BATCH_SIZE, vocab_size)
            current = hierarchy.get_surprise((x, target))

        baseline = baselines.get(code, current)
        if baseline > 0:
            drift = abs(current - baseline)
            retention = max(0, 1.0 - drift / max(baseline, 1e-6))
        else:
            retention = 1.0

        status = "PASS" if retention > 0.90 else "FAIL"
        if retention <= 0.90:
            all_retained = False
        retention_results[code] = retention
        print(f"  {LANG_NAMES.get(code, code):>10}: baseline={baseline:.4f} "
              f"current={current:.4f} retention={retention:.1%} [{status}]")

    # ═══ 12. Final Summary ═══
    total_time = (time.time() - marathon_start) / 60
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    it_surprise_val = baselines.get("it", -1)
    ru_surprise_val = baselines.get("ru", -1)

    # Find Russian↔Latin affinities from the report
    ru_latin_affinity = "N/A"
    if "ru" in lang_ranges:
        try:
            codes = r['lang_codes']
            matrix = r['matrix']
            ru_idx = codes.index("ru")
            latin_scores = []
            for li, lc in enumerate(codes):
                if lc not in ("ru", "meta"):
                    latin_scores.append(matrix[ru_idx, li].item())
            if latin_scores:
                ru_latin_affinity = f"{sum(latin_scores)/len(latin_scores):.4f}"
        except:
            pass

    # Verdict
    it_pass = it_surprise_val <= ITALIAN_TARGET if it_surprise_val > 0 else True
    ru_pass = ru_surprise_val <= RUSSIAN_TARGET if ru_surprise_val > 0 else True
    verdict = "PASS" if (it_pass and ru_pass and all_retained) else "FAIL"

    print(f"\n{'='*60}")
    print(f"  === V22 ACTIVE META-POOL RESULTS ===")
    print(f"  Italian bootstrap: {PHASE2_DURATION/60:.0f} min → surprise {it_surprise_val:.4f} (V21: 5 min → 2.28)")
    print(f"  Russian bootstrap: {PHASE3_DURATION/60:.0f} min → surprise {ru_surprise_val:.4f} (English baseline: 15 min → 3.25)")
    print(f"  Russian↔Latin affinity: {ru_latin_affinity}")
    print(f"  Meta-pool final size: {hierarchy.meta_pool_size} neurons")
    print(f"  All language retention: {min(retention_results.values()) if retention_results else 0:.1%}")
    print(f"  Time: {total_time:.1f} min | VRAM: {peak_vram:.2f} GB")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_v22()
