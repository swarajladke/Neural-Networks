import torch
import time
import os
import json
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent

# Simple Character-Level Tokenizer for Marathon Research
class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s): return [self.char_to_id[c] for c in s if c in self.char_to_id]
    def decode(self, ids): return "".join([self.id_to_char[i] for i in ids])

def check_thermal_safety():
    # V11.4 Express: Disabled for Cloud/VM environments to eliminate subprocess overhead
    pass

PHASE_DURATION = 600

def compute_lang_ranges(langs, base_dim, n_per_lang):
    ranges = {}
    ranges[langs[0]] = (0, base_dim)
    current = base_dim
    for lang in langs[1:]:
        ranges[lang] = (current, current + n_per_lang)
        current += n_per_lang
    return ranges

def get_base_dim(hierarchy, n_per_lang, n_langs):
    total_hidden = hierarchy.layers[0].V.shape[1]
    return total_hidden - (n_langs - 1) * n_per_lang

def avg_surprise_next_char(hierarchy, tokenizer, tokens, gate_range, device, n_samples=200):
    start, end = gate_range
    surprises = []
    with hierarchy.manifold_gate(start, end):
        hierarchy.reset_states()
        with torch.no_grad():
            for i in range(min(n_samples, len(tokens) - 1)):
                x = torch.zeros((1, tokenizer.vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                target = torch.zeros((1, tokenizer.vocab_size), device=device)
                target[0, tokens[i + 1]] = 1.0
                surprises.append(hierarchy.get_surprise((x, target)))
    return sum(surprises) / max(1, len(surprises))

def run_quad_marathon(audit_only=False):
    marathon_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = ["en", "de", "ro", "es"]
    lang_names = {"en": "English", "de": "German", "ro": "Romanian", "es": "Spanish"}
    n_per_lang = 256
    base_dim = n_per_lang
    
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
    lang_ranges = compute_lang_ranges(langs, base_dim, n_per_lang)
    
    print("\n=== PRE-FLIGHT CHECK ===")
    print(f" Languages: {langs}")
    print(f" Base dim: {base_dim} | N per lang: {n_per_lang}")
    print(f" Computed Ranges: {lang_ranges}")
    print(f" Expected Final Width: {base_dim + n_per_lang * (len(langs)-1)}")
    print("========================\n")

    # 2. Build Hierarchy
    hierarchy = PredictiveHierarchy([tokenizer.vocab_size, base_dim, tokenizer.vocab_size], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    baseline_surprises = {}

    # --- THE MARATHON TRAINING LOOP ---
    if not audit_only:
        for phase_idx, code in enumerate(langs):
            name = lang_names[code]
            print(f"\n>>> PHASE {phase_idx+1}: TRAINING ON {name.upper()} (5 Minutes) <<<")
            
            tokens = tokenizer.encode(corpora[code])
            start_time = time.time()
            total_duration = PHASE_DURATION
            start, end = lang_ranges[code]
            with hierarchy.manifold_gate(start, end):
                hierarchy.reset_states()
                for i in range(len(tokens) - 1):
                    elapsed = time.time() - start_time
                    if elapsed > total_duration: break

                    progress = elapsed / total_duration
                    scale = agent.get_dopamine_scale(elapsed, total_duration)
                    for layer in hierarchy.layers:
                        layer.eta_V = 0.05 * scale
                        layer.eta_W = 0.03 * scale

                    x = torch.zeros((1, tokenizer.vocab_size), device=device)
                    x[0, tokens[i]] = 1.0
                    target = torch.zeros((1, tokenizer.vocab_size), device=device)
                    target[0, tokens[i+1]] = 1.0

                    hierarchy.infer_and_learn(x, top_level_label=target, dopamine_burst=1.0)

                    if i % 1000 == 0:
                        print(f" [{name}] Token {i:5d} | Progress: {progress:.1%} | Scale: {scale:.2f} ", end="\r")

            baseline_surprises[code] = avg_surprise_next_char(
                hierarchy, tokenizer, tokens, (start, end), device=device, n_samples=200
            )
            print(f"\n {name} Baseline Surprise: {baseline_surprises[code]:.4f}")

            print(f"\n {name} Phase Complete. Igniting Synaptic Shield + Neurogenesis...")
            if phase_idx < len(langs) - 1:
                hierarchy.force_recruit_language_sliver(n=n_per_lang, language=name)
        
        hierarchy.save_checkpoint("agnis_marathon_final.pt")
        print("\n[Checkpoint] Full Marathon State Saved.")
        with open("agnis_marathon_baselines.json", "w", encoding="utf-8") as f:
            json.dump(baseline_surprises, f, indent=2, sort_keys=True)
        print("[Checkpoint] Baseline surprise saved to agnis_marathon_baselines.json")

    # --- FINAL QUAD-AUDIT ---
    print("\n>>> FINAL QUAD-AUDIT: ZERO-FORGETTING VALIDATION (GATED ISOLATION) <<<")
    if not os.path.exists("agnis_marathon_final.pt"): return

    # Auto-detect actual base_dim/ranges from the saved checkpoint dimensions.
    hierarchy.load_checkpoint("agnis_marathon_final.pt")
    detected_base = get_base_dim(hierarchy, n_per_lang=n_per_lang, n_langs=len(langs))
    lang_ranges = compute_lang_ranges(langs, detected_base, n_per_lang)
    print(f"Auto-detected base_dim: {detected_base}")
    print(f"Computed ranges: {lang_ranges}")

    results = {}
    baselines = {}
    if os.path.exists("agnis_marathon_baselines.json"):
        with open("agnis_marathon_baselines.json", "r", encoding="utf-8") as f:
            baselines = json.load(f)
        print(f"Loaded baselines: {baselines}")
    for code in langs:
        name = lang_names[code]
        tokens = tokenizer.encode(corpora[code])
        start, end = lang_ranges[code]
        
        hierarchy.load_checkpoint("agnis_marathon_final.pt")
        avg_surprise = avg_surprise_next_char(
            hierarchy, tokenizer, tokens, (start, end), device=device, n_samples=200
        )

        if code in baselines:
            baseline = float(baselines[code])
            drift = abs(avg_surprise - baseline)
            retention_pct = max(0.0, 1.0 - (drift / max(1e-8, baseline))) * 100.0
            results[code] = retention_pct / 100.0
            print(f"{code} surprise: {avg_surprise:.4f} | baseline: {baseline:.4f} | drift: {drift:.4f} | retention: {retention_pct:.1f}%")
        else:
            results[code] = avg_surprise
            print(f"{code} surprise: {avg_surprise:.4f}")

    
    # Restore full model
    hierarchy.load_checkpoint("agnis_marathon_final.pt")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total time: {(time.time()-marathon_start_time)/60:.1f} min")
    print(f"Peak VRAM:  {peak_vram:.3f} GB")
    for code, acc in results.items():
        if os.path.exists("agnis_marathon_baselines.json"):
            status = "✅" if acc >= 0.90 else "❌"
            print(f"{status} {code}: {acc:.1%}")
        else:
            print(f"{code}: {acc}")
    if os.path.exists("agnis_marathon_baselines.json"):
        all_pass = all(acc >= 0.90 for acc in results.values())
        print(f"\nOVERALL: {'✅ PASS' if all_pass else '❌ FAIL'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--audit_only":
        run_quad_marathon(audit_only=True)
    else:
        run_quad_marathon()
