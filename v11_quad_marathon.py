import torch
import time
import os
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

def run_quad_marathon(audit_only=False):
    marathon_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = ["en", "de", "ro", "es"]
    lang_names = {"en": "English", "de": "German", "ro": "Romanian", "es": "Spanish"}
    base_dim = 512
    n_per_lang = 256
    
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

            print(f"\n {name} Phase Complete. Igniting Synaptic Shield + Neurogenesis...")
            if phase_idx < len(langs) - 1:
                hierarchy.force_recruit_language_sliver(n=n_per_lang, language=name)
        
        hierarchy.save_checkpoint("agnis_marathon_final.pt")
        print("\n[Checkpoint] Full Marathon State Saved.")

    # --- FINAL QUAD-AUDIT ---
    print("\n>>> FINAL QUAD-AUDIT: ZERO-FORGETTING VALIDATION (GATED ISOLATION) <<<")
    if not os.path.exists("agnis_marathon_final.pt"): return

    results = {}
    for code in langs:
        name = lang_names[code]
        tokens = tokenizer.encode(corpora[code])
        start, end = lang_ranges[code]
        probe_results = []
        
        hierarchy.load_checkpoint("agnis_marathon_final.pt")
        with hierarchy.manifold_gate(start, end):
            hierarchy.reset_states()
            with torch.no_grad():
                for i in range(200):
                    x = torch.zeros((1, tokenizer.vocab_size), device=device)
                    x[0, tokens[i]] = 1.0
                    pred = hierarchy.predict_label(x)
                    probe_results.append(torch.argmax(pred[0]).item() == tokens[i+1])

        n_samples = len(probe_results)
        accuracy = sum(probe_results) / n_samples
        results[code] = accuracy
        print(f"{code}: {accuracy:.1%} ({sum(probe_results)}/{n_samples})")

    
    # Restore full model
    hierarchy.load_checkpoint("agnis_marathon_final.pt")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total time: {(time.time()-marathon_start_time)/60:.1f} min")
    print(f"Peak VRAM:  {peak_vram:.3f} GB")
    for code, acc in results.items():
        status = "✅" if acc >= 0.90 else "❌"
        print(f"{status} {code}: {acc:.1%}")
    all_pass = all(acc >= 0.90 for acc in results.values())
    print(f"\nOVERALL: {'✅ PASS' if all_pass else '❌ FAIL'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--audit_only":
        run_quad_marathon(audit_only=True)
    else:
        run_quad_marathon()
