"""
v22_audit.py — Re-run retention audit + AbstraX with corrected formulas
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v4_core import PredictiveHierarchy, META_POOL_LR_SCALE
from agnis_v4_cognitive import AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

META_POOL_SIZE = 160  # Final expanded size from V22
N_PER_LANG = 128
BPE_VOCAB = 4000
BATCH_SIZE = 32

LANG_NAMES = {
    "en": "English", "de": "German", "es": "Spanish",
    "fr": "French", "it": "Italian", "ru": "Russian"
}

# V22 baselines (from training output)
BASELINES = {
    "en": 2.9912, "de": 2.9815, "es": 2.9850,
    "fr": 2.9967, "it": 2.3958, "ru": 2.3095
}

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


def run_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  V22 POST-MARATHON AUDIT (Corrected Formulas)")
    print("=" * 60)

    # 1. Load tokenizer
    tokenizer = BPETokenizer(BPE_VOCAB)
    tokenizer.load(f"agnis_bpe_{BPE_VOCAB}_vocab.json")
    vocab_size = tokenizer.vocab_size

    # 2. Load corpora
    all_langs = ["en", "de", "es", "fr", "it", "ru"]
    all_tokens = {}
    all_tensors = {}
    for code in all_langs:
        path = f"slm/input_{code}.txt"
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()[:500000]
        tokens = tokenizer.encode(text)
        all_tokens[code] = tokens
        all_tensors[code] = torch.tensor(tokens, dtype=torch.long, device=device)
        print(f"  [{code}] {len(tokens):,} tokens")

    # 3. Load V22 checkpoint
    # We need to figure out final architecture dimensions
    # Phase1: 4 langs × 128 = 512 + 64 meta = 576
    # +32 expand (test) = 608
    # +128 Italian sliver = 736
    # +32 expand (Italian) = 768
    # +128 Russian sliver = 896
    # +32 expand (Russian) = 928
    # But meta_pool_size ended at 160
    # Let's just load and check
    
    ckpt_path = "agnis_v22_active_meta.pt"
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Missing {ckpt_path}")
        return

    # Load raw to inspect dimensions
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # state is a list of tuples, first element is V
    layer0_V = state[0][0]  # V matrix of layer 0
    layer0_odim = state[0][-1]  # output_dim is last element
    layer0_idim = state[0][-2]  # input_dim is second to last
    print(f"\n  Loaded checkpoint: Layer 0 dims = [{layer0_idim}, {layer0_odim}]")

    # Build hierarchy with correct dims
    hidden_dim = layer0_odim
    hierarchy = PredictiveHierarchy(
        [vocab_size, hidden_dim, hidden_dim, vocab_size],
        device=device,
        meta_pool_size=META_POOL_SIZE
    )
    hierarchy.load_checkpoint(ckpt_path)
    print(f"  Meta-pool size: {hierarchy.meta_pool_size}")
    print(f"  Hidden dims: {[l.output_dim for l in hierarchy.layers]}")

    # 4. Reconstruct language ranges
    # Original 4 langs started at meta=64
    # But meta expanded: 64 → 96 → 128 → 160
    # Slivers were added via force_recruit which appends at end
    # So original en was at [64, 192], de at [192, 320], etc.
    # But expansions added neurons at the END of hidden layers
    # The original sliver positions should still be valid
    
    orig_meta = 64
    lang_ranges = {}
    for i, lang in enumerate(["en", "de", "es", "fr"]):
        s = orig_meta + i * N_PER_LANG
        lang_ranges[lang] = (s, s + N_PER_LANG)

    # Italian and Russian were recruited after Phase 1
    # Italian was recruited after 4 existing langs
    # Hidden was 576 at that point, then meta expanded +32 (test), +32 (Italian)
    # The force_recruit appends at end of current hidden dim
    # Let's figure out from the actual layer dims
    
    actual_hidden = hierarchy.layers[0].output_dim
    print(f"\n  Actual hidden dim: {actual_hidden}")
    print(f"  Expected structure:")
    print(f"    [0, {orig_meta}): original meta-pool")
    print(f"    [{orig_meta}, {orig_meta + 4*N_PER_LANG}): en/de/es/fr slivers")
    
    # The remaining neurons are: test expansion + Italian + Italian expansion + Russian + Russian expansion
    remaining_start = orig_meta + 4 * N_PER_LANG  # 576
    remaining = actual_hidden - remaining_start
    print(f"    [{remaining_start}, {actual_hidden}): expansions + it + ru ({remaining} neurons)")
    
    # From the log: meta expanded 64→96 (test +32), then 96→128 (Italian +32), then 128→160 (Russian +32)
    # expand_meta_pool adds neurons at the END via expand_output
    # force_recruit also adds at END
    # So order after 576: +32(test) +128(it_sliver) +32(it_expand) +128(ru_sliver) +32(ru_expand)
    # = 576 + 32 + 128 + 32 + 128 + 32 = 928
    
    # Italian sliver starts after test expansion
    it_start = remaining_start + 32  # 608
    it_end = it_start + N_PER_LANG    # 736
    lang_ranges["it"] = (it_start, it_end)
    
    # Russian sliver starts after Italian expansion
    ru_start = it_end + 32  # 768
    ru_end = ru_start + N_PER_LANG  # 896
    lang_ranges["ru"] = (ru_start, ru_end)
    
    print(f"\n  Reconstructed ranges:")
    for code, (s, e) in lang_ranges.items():
        print(f"    {LANG_NAMES[code]:>10}: [{s}, {e})")

    # 5. CORRECTED RETENTION AUDIT
    print(f"\n{'='*60}")
    print(f"  CORRECTED RETENTION AUDIT")
    print(f"  (Only surprise INCREASES count as forgetting)")
    print(f"{'='*60}")

    all_pass = True
    for code in ["en", "de", "es", "fr", "it", "ru"]:
        if code not in all_tensors:
            continue
        s, e = lang_ranges[code]

        # Measure surprise over multiple batches for stability
        surprises = []
        n_test = min(10, (len(all_tokens[code]) - 1) // BATCH_SIZE)
        with hierarchy.manifold_gate(0, e):
            for bi in range(n_test):
                x, target = get_batch(all_tensors[code], bi, BATCH_SIZE, vocab_size)
                s_val = hierarchy.get_surprise((x, target))
                surprises.append(s_val)

        current = sum(surprises) / len(surprises)
        baseline = BASELINES.get(code, current)

        # CORRECTED: Only count INCREASES as forgetting
        if current > baseline:
            drift = current - baseline
            retention = max(0, 1.0 - drift / max(baseline, 1e-6))
            status = "PASS" if retention > 0.95 else "FAIL"
            if retention <= 0.90:
                all_pass = False
        else:
            # Surprise decreased = language got BETTER
            improvement = baseline - current
            retention = 1.0 + improvement / max(baseline, 1e-6)
            status = "IMPROVED ★"

        print(f"  {LANG_NAMES[code]:>10}: baseline={baseline:.4f} "
              f"current={current:.4f} retention={retention:.1%} [{status}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'ISSUES FOUND'}")

    # 6. AbstraX with correct ranges
    print(f"\n{'='*60}")
    print(f"  ABSTRAX ANALYSIS (Corrected Ranges)")
    print(f"{'='*60}")

    abstrax_ranges = {"meta": (0, orig_meta)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for li in range(len(hierarchy.layers)):
        print(f"\n--- Layer {li} ---")
        try:
            r = abstrax.compute_pairwise_affinity(layer_idx=li)
            pairs = abstrax.print_affinity_report(r, title=f"V22 Layer {li}")
            abstrax.identify_fold_candidates(r, threshold=0.2)
        except Exception as e:
            print(f"  [SKIP] {e}")

    # 7. Russian↔Latin summary
    print(f"\n{'='*60}")
    print(f"  RUSSIAN ↔ LATIN AFFINITY SUMMARY")
    print(f"{'='*60}")
    try:
        r = abstrax.compute_pairwise_affinity(layer_idx=min(2, len(hierarchy.layers)-1))
        codes = r['lang_codes']
        matrix = r['matrix']
        if 'ru' in codes:
            ru_idx = codes.index('ru')
            for li, lc in enumerate(codes):
                if lc not in ('ru', 'meta'):
                    val = matrix[ru_idx, li].item()
                    bar = "█" * int(max(0, val) * 20)
                    print(f"  ru ↔ {lc}: {val:.4f}  {bar}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    print(f"\n{'='*60}")
    print(f"  V22 AUDIT COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_audit()
