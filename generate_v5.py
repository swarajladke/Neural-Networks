"""
generate_v5.py
================================================================
AGNIS V5 | Thermal-Safe Linguistic Generator
================================================================
Feature: Includes 'Thermal Shielding' (small delays) to keep GPU cool.
Target: Tests the 5.24 loss checkpoint for fluency.
"""

import os, sys, time, torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Add path so we can import AgnisV5 architecture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agnis_v5_sprint2 import AgnisV5

# --- Config ---
CHECKPOINT_PATH = "agnis_v5_30m_fluency.pt"
TOKENIZER_PATH  = "slm_bpe_tokenizer_32k.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Thermal Shield: Pause between token generations to keep temps stable
THERMAL_DELAY_SEC = 0.01 

@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=64, temp=0.8, top_k=50, top_p=0.9):
    model.eval()
    model.reset_states(batch_size=1)
    
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    print(f"\n[Prompt]: {prompt}")
    print("[AGNIS]: ", end="", flush=True)
    
    generated = input_ids
    
    # Pre-fill states with the prompt
    # We pass tokens one by one to ensure R_weight and h_prev states are built correctly
    for i in range(len(input_ids) - 1):
        _ = model(input_tensor[:, i:i+1])
        
    cur_token = input_tensor[:, -1:]
    
    for _ in range(max_len):
        logits = model(cur_token)
        logits = logits[:, -1, :] / temp
        
        # Top-K sampling
        v, idx = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Top-P (Nucleus) sampling
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float('Inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_str = tokenizer.decode([next_token.item()])
        print(token_str, end="", flush=True)
        
        generated.append(next_token.item())
        cur_token = next_token
        
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break
            
        # --- THERMAL SHIELD ---
        if THERMAL_DELAY_SEC > 0:
            time.sleep(THERMAL_DELAY_SEC)
            
    print("\n" + "-"*40)

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # Reconstruct the model (ensure params match agnis_v5_sprint2.py)
    # Architecture: 768 dim, 3072 hidden
    model = AgnisV5(32000, 768, 3072, device=DEVICE)
    
    print(f"[System] Loading checkpoint: {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    sd = ckpt['model']
    if sd and next(iter(sd)).startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}

    # Filter out batch-size-dependent state buffers (saved as B=64, inference needs B=1)
    md = model.state_dict()
    clean_sd = {k: v for k, v in sd.items()
                if k in md and v.shape == md[k].shape}
    skipped = [k for k in sd if k not in clean_sd]
    if skipped:
        print(f"[Load] Skipped {len(skipped)} batch-state buffers (will reset to B=1): OK")

    model.load_state_dict(clean_sd, strict=False)
    model.to(DEVICE)
    print("[System] Neural manifold initialized.")

    prompts = [
        "The scientific method is a process of",
        "In the world of artificial intelligence, we",
        "History shows that civilizations often",
        "To build a successful startup, you need",
    ]

    for p in prompts:
        generate(model, tokenizer, p)

if __name__ == "__main__":
    main()
