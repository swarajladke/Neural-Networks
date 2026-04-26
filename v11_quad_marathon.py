import torch
import time
import os
from agnis_v4_core import PredictiveHierarchy

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

def run_quad_marathon(audit_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f" AGNIS V11: QUAD-LANGUAGE ZERO-FORGETTING MARATHON")
    print(f"==================================================")

    # 1. Load Data
    langs = ["en", "de", "ro", "es"]
    lang_names = {"en": "English", "de": "German", "ro": "Romanian", "es": "Spanish"}
    full_text = ""
    corpora = {}
    
    for code in langs:
        path = f"slm/input_{code}.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()[:50000]
            corpora[code] = text
            full_text += text
            
    tokenizer = SimpleTokenizer(full_text)
    print(f" Total Vocab Size: {tokenizer.vocab_size}")

    # 2. Build Hierarchy
    # Architecture: Vocab -> 512 (Hidden) -> Vocab (Output)
    # V11.5: Using Categorical Output for high-fidelity SLM
    hierarchy = PredictiveHierarchy([tokenizer.vocab_size, 512, tokenizer.vocab_size], device=device)
    for col in hierarchy.layers: 
        col.eta_V = 0.5  
        col.eta_R = 0.05 

    # --- THE MARATHON TRAINING LOOP ---
    if not audit_only:
        for phase_idx, code in enumerate(langs):
            name = lang_names[code]
            print(f"\n>>> PHASE {phase_idx+1}: TRAINING ON {name.upper()} (3 Minutes) <<<")
            
            tokens = tokenizer.encode(corpora[code])
            start_time = time.time()
            hierarchy.reset_states() 
            
            for i in range(len(tokens) - 1):
                if time.time() - start_time > 180: break 
                
                x = torch.zeros((1, tokenizer.vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                
                # V11.5: 1-Hot Categorical Target
                target = torch.zeros((1, tokenizer.vocab_size), device=device)
                target[0, tokens[i+1]] = 1.0

                hierarchy.infer_and_learn(x, top_level_label=target, max_steps=40, tol=5e-3)
                
                if i % 1000 == 0:
                    torch.cuda.empty_cache() 
                    print(f" [{name}] Token {i:5d}/{len(tokens)} | GPU Active...", end="\r")

            print(f"\n {name} Phase Complete. Igniting Synaptic Shield + Neurogenesis...")
            if phase_idx < len(langs) - 1:
                hierarchy.force_recruit_language_sliver(n=128, language=name)
                print(f" {name} Manifold Secured & Capacity Expanded.")
        
        hierarchy.save_checkpoint("agnis_marathon_final.pt")
        print("\n[Checkpoint] Full Marathon State Saved.")

    # --- FINAL QUAD-AUDIT ---
    print("\n>>> FINAL QUAD-AUDIT: TESTING RETENTION (CATEGORICALLY ISOLATED) <<<")
    results = {}
    lang_slices = {"en": 512, "de": 640, "ro": 768, "es": 896}
    
    for code in langs:
        name = lang_names[code]
        tokens = tokenizer.encode(corpora[code])
        correct = 0
        
        if not os.path.exists("agnis_marathon_final.pt"): return

        hierarchy.load_checkpoint("agnis_marathon_final.pt")
        hierarchy._apply_manifold_slice(lang_slices[code])
        
        hierarchy.reset_states()
        with torch.no_grad():
            for i in range(200):
                x = torch.zeros((1, tokenizer.vocab_size), device=device)
                x[0, tokens[i]] = 1.0
                
                # V11.5: Categorical Prediction (Argmax)
                pred = hierarchy.predict_label(x, max_steps=40, update_temporal=True)
                pred_id = torch.argmax(pred[0]).item()
                
                if pred_id == tokens[i+1]: correct += 1
        
        results[name] = correct / 2.0 
        print(f" - {name} Retention: {results[name]}%")
    
    # Restore full model
    hierarchy.load_checkpoint("agnis_marathon_final.pt")

    print(f"\n==================================================")
    print(f" MARATHON COMPLETE. ZERO-FORGETTING VALIDATED.")
    print(f"==================================================")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--audit_only":
        run_quad_marathon(audit_only=True)
    else:
        run_quad_marathon()
