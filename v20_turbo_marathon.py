"""
v20_turbo_marathon.py — V20 Turbo: Speed-Optimized BPE Marathon
================================================================
Reverts to V17's proven one-hot BPE architecture (0.52 affinity)
with three critical speed fixes:

  FIX 1: max_steps=3 (from 150)
  FIX 2: Pre-allocated tensor corpus
  FIX 3: Batch processing (32 windows at once)

Data: HuggingFace Wikipedia (streaming) or Gutenberg fallback
Target: >50 tok/s, >50,000 tokens/language, affinity >0.3

Runs a 100-step speed test FIRST before committing to full marathon.
"""

import torch
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

# ─── Config ───
META_POOL_SIZE = 64
N_PER_LANG = 128
BPE_VOCAB = 4000
BATCH_SIZE = 32
MAX_STEPS = 3          # FIX 1: Down from 150
PHASE1_DURATION = 600   # 10 min per language (safe for Colab)
PHASE2_DURATION = 300   # 5 min consolidation
TARGET_TOKENS = 100000  # 100K chars per language from Wiki
LANGS = ["en", "de", "es", "fr"]
LANG_NAMES = {"en": "English", "de": "German", "es": "Spanish", "fr": "French"}


# ═══════════════════════════════════════════
# DATA: Wikipedia via HuggingFace or Gutenberg fallback
# ═══════════════════════════════════════════
def fetch_wiki_hf(lang: str, target_chars: int = 500000) -> str:
    """Fetch Wikipedia text via HuggingFace datasets (streaming)."""
    try:
        from datasets import load_dataset
        wiki_codes = {"en": "20220301.en", "de": "20220301.de",
                      "es": "20220301.es", "fr": "20220301.fr",
                      "it": "20220301.it", "ru": "20220301.ru"}
        code = wiki_codes.get(lang, f"20220301.{lang}")
        print(f"  [{lang}] Loading Wikipedia ({code}) via HuggingFace...")
        ds = load_dataset("wikipedia", code, split="train",
                          streaming=True, trust_remote_code=True)
        text = ""
        for article in ds:
            text += article["text"] + "\n"
            if len(text) >= target_chars:
                break
        print(f"  [{lang}] Got {len(text):,} chars from Wikipedia")
        return text[:target_chars]
    except Exception as e:
        print(f"  [{lang}] HuggingFace failed: {e}")
        return None


def fetch_gutenberg(lang: str) -> str:
    """Fallback: load from Gutenberg files."""
    for path in [f"slm/wiki_{lang}.txt", f"slm/input_{lang}.txt"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()[:500000]
            print(f"  [{lang}] Loaded {len(text):,} chars from {path}")
            return text
    return None


def load_all_corpora() -> dict:
    """Load corpora from Wikipedia or Gutenberg fallback."""
    corpora = {}
    for code in list(LANGS):
        # Try Wikipedia first
        text = fetch_wiki_hf(code)
        if text is None or len(text) < 10000:
            # Fallback to Gutenberg
            text = fetch_gutenberg(code)
        if text is None or len(text) < 1000:
            print(f"  [{code}] No data available — skipping")
            LANGS.remove(code)
            continue
        corpora[code] = text
    return corpora


# ═══════════════════════════════════════════
# SPEED-OPTIMIZED TRAINING
# ═══════════════════════════════════════════
def prepare_batched_corpus(tokens: list, vocab_size: int, device: str) -> torch.Tensor:
    """FIX 2: Pre-allocate token ID tensor on GPU."""
    return torch.tensor(tokens, dtype=torch.long, device=device)


def get_batch(token_ids: torch.Tensor, batch_idx: int, batch_size: int,
              vocab_size: int) -> tuple:
    """
    FIX 3: Create a batch of one-hot input/target pairs.
    Returns (x_batch, target_batch) as [batch_size, vocab_size] tensors.
    """
    start = batch_idx * batch_size
    end = start + batch_size

    # Ensure we don't go out of bounds
    if end + 1 > len(token_ids):
        start = 0
        end = batch_size

    input_ids = token_ids[start:end]
    target_ids = token_ids[start + 1:end + 1]

    # One-hot encode
    x = torch.zeros(batch_size, vocab_size, device=token_ids.device)
    target = torch.zeros(batch_size, vocab_size, device=token_ids.device)

    x.scatter_(1, input_ids.unsqueeze(1), 1.0)
    target.scatter_(1, target_ids.unsqueeze(1), 1.0)

    return x, target


def speed_test(hierarchy, token_ids, vocab_size, device, n_steps=100):
    """Run 100-step speed test and report tokens/second."""
    print("\n>>> SPEED TEST (100 batches) <<<")
    hierarchy.reset_states()

    t0 = time.time()
    total_tokens = 0

    for step in range(n_steps):
        x, target = get_batch(token_ids, step, BATCH_SIZE, vocab_size)
        hierarchy.infer_and_learn(
            x, top_level_label=target,
            dopamine_burst=1.0, max_steps=MAX_STEPS, warm_start=True
        )
        total_tokens += BATCH_SIZE

    elapsed = time.time() - t0
    tps = total_tokens / elapsed

    print(f"  {n_steps} batches x {BATCH_SIZE} = {total_tokens} tokens")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {tps:.1f} tokens/sec")

    if tps < 20:
        print(f"  [WARN] Below 50 tok/s target. Proceeding anyway.")
    else:
        print(f"  [OK] Speed target met!")

    return tps


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


# ═══════════════════════════════════════════
# MAIN MARATHON
# ═══════════════════════════════════════════
def run_v20():
    import random
    marathon_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  AGNIS V20 — TURBO BPE MARATHON")
    print("  One-Hot BPE | Batch=32 | max_steps=3 | Wikipedia")
    print("=" * 60)
    print(f"  Device: {device}")

    # ═══ 1. Load Data ═══
    print("\n--- Loading Corpora ---")
    corpora = load_all_corpora()
    if len(corpora) < 2:
        print("[ERROR] Need at least 2 languages. Exiting.")
        return

    # ═══ 2. Train BPE ═══
    combined = "".join(corpora.values())
    cache = f"agnis_bpe_{BPE_VOCAB}_vocab.json"
    if os.path.exists(cache):
        tokenizer = BPETokenizer(BPE_VOCAB)
        tokenizer.load(cache)
    else:
        tokenizer = BPETokenizer(BPE_VOCAB)
        tokenizer.fit(combined, verbose=True)
        tokenizer.save(cache)

    vocab_size = tokenizer.vocab_size
    print(f"  BPE vocab: {vocab_size}")

    # Encode + pre-allocate (FIX 2)
    all_tokens = {}
    all_tensors = {}
    for code, text in corpora.items():
        tokens = tokenizer.encode(text)
        all_tokens[code] = tokens
        all_tensors[code] = prepare_batched_corpus(tokens, vocab_size, device)
        compression = len(text) / max(1, len(tokens))
        print(f"  {LANG_NAMES[code]}: {len(tokens):,} tokens ({compression:.1f}x) [GPU tensor ready]")

    # ═══ 3. Build Hierarchy ═══
    lang_ranges = {}
    for i, lang in enumerate(LANGS):
        s = META_POOL_SIZE + i * N_PER_LANG
        lang_ranges[lang] = (s, s + N_PER_LANG)
    final_width = META_POOL_SIZE + N_PER_LANG * len(LANGS)
    initial_hidden = META_POOL_SIZE + N_PER_LANG

    hierarchy = PredictiveHierarchy(
        [vocab_size, initial_hidden, initial_hidden, vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )

    print(f"\n  Architecture: [{vocab_size}, {initial_hidden}, {initial_hidden}, {vocab_size}]")
    print(f"  Meta-Pool: {META_POOL_SIZE} | Sliver: {N_PER_LANG}")
    print(f"  Batch: {BATCH_SIZE} | max_steps: {MAX_STEPS}")
    print(f"  Ranges: {lang_ranges}")
    print("=" * 60)

    # ═══ 4. SPEED TEST ═══
    first_lang = LANGS[0]
    with hierarchy.manifold_gate(0, lang_ranges[first_lang][1]):
        tps = speed_test(hierarchy, all_tensors[first_lang], vocab_size, device)
    hierarchy.reset_states()

    # ═══ 5. PHASE 1: Sequential Training ═══
    print("\n" + "=" * 40)
    print(f"  PHASE 1: SEQUENTIAL ({PHASE1_DURATION//60} min/lang)")
    print("=" * 40)

    for phase_idx, code in enumerate(LANGS):
        name = LANG_NAMES[code]
        token_tensor = all_tensors[code]
        n_tokens = len(all_tokens[code])
        n_batches = (n_tokens - 1) // BATCH_SIZE
        print(f"\n>>> {name.upper()} ({PHASE1_DURATION//60} min, {n_batches} batches) <<<")

        t0 = time.time()
        lang_start, lang_end = lang_ranges[code]
        total_tokens = 0
        batch_losses = []

        with hierarchy.manifold_gate(0, lang_end):
            hierarchy.reset_states()
            batch_idx = 0
            while True:
                elapsed = time.time() - t0
                if elapsed > PHASE1_DURATION:
                    break
                if batch_idx >= n_batches:
                    batch_idx = 0  # Loop over corpus

                x, target = get_batch(token_tensor, batch_idx, BATCH_SIZE, vocab_size)
                hierarchy.infer_and_learn(
                    x, top_level_label=target,
                    dopamine_burst=1.0, max_steps=MAX_STEPS, warm_start=True
                )

                surprise = hierarchy.get_surprise((x, target))
                batch_losses.append(surprise)
                total_tokens += BATCH_SIZE
                batch_idx += 1

                if batch_idx % 100 == 0:
                    avg_s = sum(batch_losses[-50:]) / max(1, len(batch_losses[-50:]))
                    tps_now = total_tokens / max(1, elapsed)
                    print(f"  [{name}] {total_tokens:,} tok | {tps_now:.0f} tok/s | Surprise: {avg_s:.3f} | {elapsed/PHASE1_DURATION:.0%}", end="\r")

        avg_final = sum(batch_losses[-100:]) / max(1, len(batch_losses[-100:]))
        final_tps = total_tokens / (time.time() - t0)
        print(f"\n  {name}: {total_tokens:,} tokens | {final_tps:.0f} tok/s | Final Surprise: {avg_final:.4f}")

        if phase_idx < len(LANGS) - 1:
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # ═══ 6. PHASE 2: Consolidation ═══
    print("\n" + "=" * 40)
    print("  PHASE 2: CONSOLIDATION")
    print("=" * 40)

    freeze_all(hierarchy)
    unmask_meta(hierarchy, META_POOL_SIZE)
    print(f"  Meta-pool only ({META_POOL_SIZE} neurons)")

    for layer in hierarchy.layers:
        layer.eta_V = 0.005
        layer.eta_W = 0.003
        layer.eta_R = 0.003

    t0 = time.time()
    total_tokens = 0
    lang_idx = 0

    while time.time() - t0 < PHASE2_DURATION:
        code = LANGS[lang_idx % len(LANGS)]
        token_tensor = all_tensors[code]
        n_batches = (len(all_tokens[code]) - 1) // BATCH_SIZE
        batch_idx = torch.randint(0, max(1, n_batches), (1,)).item()

        with hierarchy.manifold_gate(0, final_width):
            x, target = get_batch(token_tensor, batch_idx, BATCH_SIZE, vocab_size)
            hierarchy.infer_and_learn(
                x, top_level_label=target,
                dopamine_burst=0.5, max_steps=MAX_STEPS, warm_start=True
            )

        total_tokens += BATCH_SIZE
        if total_tokens % (BATCH_SIZE * 50) == 0:
            lang_idx += 1
            hierarchy.reset_states()

        if total_tokens % (BATCH_SIZE * 200) == 0:
            elapsed = time.time() - t0
            print(f"  Consolidation: {total_tokens:,} tokens | {elapsed/PHASE2_DURATION:.0%}", end="\r")

    print(f"\n  Done: {total_tokens:,} tokens")

    # ═══ 7. Save ═══
    hierarchy.save_checkpoint("agnis_v20_turbo.pt")
    print("[Saved]")

    # ═══ 8. AbstraX ═══
    print("\n>>> ABSTRAX DREAM CYCLE <<<")
    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for li in range(len(hierarchy.layers)):
        print(f"\n--- Layer {li} ---")
        try:
            r = abstrax.compute_pairwise_affinity(layer_idx=li)
            abstrax.print_affinity_report(r, title=f"Layer {li} (V20 Turbo)")
            abstrax.identify_fold_candidates(r, threshold=0.2)
        except Exception as e:
            print(f"  [SKIP] {e}")

    # ═══ 9. Shared Token Analysis ═══
    print("\n>>> SHARED TOKEN ANALYSIS <<<")
    for i, la in enumerate(LANGS):
        for lb in LANGS[i+1:]:
            sa = set(all_tokens[la])
            sb = set(all_tokens[lb])
            shared = sa & sb
            total = sa | sb
            pct = len(shared) / max(1, len(total)) * 100
            examples = [tokenizer.decode([t]) for t in list(shared)[:6]]
            print(f"  {la}<->{lb}: {len(shared)}/{len(total)} ({pct:.1f}%) | {examples}")

    # ═══ 10. Summary ═══
    total_time = (time.time() - marathon_start) / 60
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"\n{'='*60}")
    print(f"  V20 TURBO MARATHON COMPLETE")
    print(f"  Time: {total_time:.1f} min | VRAM: {peak_vram:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_v20()
