import torch
import time
import os
from agnis_v4_core import PredictiveHierarchy

# Simple Character-Level Tokenizer for AGNIS Research
class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s): return [self.char_to_id[c] for c in s if c in self.char_to_id]
    def decode(self, ids): return "".join([self.id_to_char[i] for i in ids])

def check_thermal_safety():
    # Helper to check GPU temp via nvidia-smi
    try:
        import subprocess
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        temp = int(res.decode().strip())
        if temp > 80:
            print(f"\n[THERMAL GUARD] GPU Temp {temp}°C - Pausing for 30s...")
            time.sleep(30)
    except: pass

def run_bilingual_sprint():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f" AGNIS V10: BILINGUAL ZERO-FORGETTING SPRINT")
    print(f"==================================================")

    # 1. Load Data
    with open("slm/input_it.txt", "r", encoding="utf-8") as f: text_it = f.read()[:50000]
    with open("slm/input_ru.txt", "r", encoding="utf-8") as f: text_ru = f.read()[:50000]
    
    tokenizer = SimpleTokenizer(text_it + text_ru)
    tokens_it = tokenizer.encode(text_it)
    tokens_ru = tokenizer.encode(text_ru)
    
    print(f" Vocab Size: {tokenizer.vocab_size}")
    print(f" Italian Tokens: {len(tokens_it)} | Russian Tokens: {len(tokens_ru)}")

    # 2. Build Hierarchy (V8.4 Spectral Core)
    # Architecture: Vocab -> 512 -> 512 -> 1
    hierarchy = PredictiveHierarchy([tokenizer.vocab_size, 512, 1], device=device)
    for col in hierarchy.layers: 
        col.eta_V = 0.5  # Fast recognition
        col.eta_R = 0.05 # Stable memory
    
    # --- PHASE 1: ITALIAN TRAINING ---
    print("\n>>> PHASE 1: TRAINING ON ITALIAN (15 Minutes) <<<")
    start_time = time.time()
    for i in range(len(tokens_it) - 1):
        if time.time() - start_time > 900: break # 15m limit
        
        x = torch.zeros((1, tokenizer.vocab_size), device=device)
        x[0, tokens_it[i]] = 1.0
        target = torch.zeros((1, 1), device=device)
        target[0, 0] = float(tokens_it[i+1]) / tokenizer.vocab_size # Normalized target

        # Step 1: Infer & Learn
        hierarchy.infer_and_learn(x, top_level_label=target, max_steps=40)
        
        if i % 100 == 0:
            torch.cuda.empty_cache() # Memory Sanitization
            print(f" Token {i:5d}/{len(tokens_it)} | GPU Active...", end="\r")
            if i % 500 == 0: check_thermal_safety()

    print(f"\n Italian Phase Complete. Time: {time.time() - start_time:.1f}s")

    # --- PHASE 2: SHIELD IGNITION ---
    print("\n>>> PHASE 2: IGNITING SYNAPTIC SHIELD <<<")
    # We freeze the manifold used by Italian
    for col in hierarchy.layers:
        col.freeze_experts() # V7.3 Synaptic Shield
    print(" Italian Manifold Secured (Zero-Forgetting Active).")

    # --- PHASE 3: RUSSIAN TRAINING ---
    print("\n>>> PHASE 3: TRAINING ON RUSSIAN (15 Minutes) <<<")
    start_time = time.time()
    for i in range(len(tokens_ru) - 1):
        if time.time() - start_time > 900: break # 15m limit
        
        x = torch.zeros((1, tokenizer.vocab_size), device=device)
        x[0, tokens_ru[i]] = 1.0
        target = torch.zeros((1, 1), device=device)
        target[0, 0] = float(tokens_ru[i+1]) / tokenizer.vocab_size

        hierarchy.infer_and_learn(x, top_level_label=target, max_steps=40)
        
        if i % 100 == 0:
            torch.cuda.empty_cache() # Memory Sanitization
            print(f" Token {i:5d}/{len(tokens_ru)} | GPU Active...", end="\r")
            if i % 500 == 0: check_thermal_safety()

    print(f"\n Russian Phase Complete. Time: {time.time() - start_time:.1f}s")

    # --- PHASE 4: FINAL BILINGUAL AUDIT ---
    print("\n>>> PHASE 4: FINAL BILINGUAL AUDIT <<<")
    # Testing Italian retention after Russian training
    correct_it = 0
    with torch.no_grad():
        for i in range(100): # Small sample test
            x = torch.zeros((1, tokenizer.vocab_size), device=device)
            x[0, tokens_it[i]] = 1.0
            pred = hierarchy.predict_label(x, max_steps=40)
            pred_id = int(torch.clamp(pred[0, 0] * tokenizer.vocab_size, 0, tokenizer.vocab_size-1))
            if pred_id == tokens_it[i+1]: correct_it += 1
    
    print(f" Italian Retention (Post-Russian): {correct_it}%")
    print(f"\n==================================================")
    print(f" SPRINT COMPLETE. RESULTS SECURED.")
    print(f"==================================================")

if __name__ == "__main__":
    run_bilingual_sprint()
