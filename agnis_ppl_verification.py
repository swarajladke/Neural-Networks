"""
agnis_ppl_verification.py — Perplexity Verification Suite
============================================================
Checks if the PPL drop (21.71 -> 7.53) observed during adapter alignment is:
1. An ARTIFACT of overfitting to fact-style templates (PPL drops only on fact-related text).
2. A GENUINE improvement in language representation and alignment (PPL drops on unrelated text too).

Usage:
  python agnis_ppl_verification.py
"""
from __future__ import annotations
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from agnis_gpt2_hybrid import AgnisGpt2Hybrid, find_agnis_checkpoint

# Force unbuffered output for Kaggle log visibility
sys.stdout.reconfigure(line_buffering=True)

# ── Texts Sources ──────────────────────────────────────────────────
fact_related = [
    "The compound dissolves at high temperatures.",
    "The element has a unique atomic structure.",
    "Scientists discovered a new protein mechanism.",
    "The language has a complex phonetic system.",
    "The perplexity metric measures language quality.",
    "The neural network achieves zero forgetting.",
    "The planet orbits its star every seven days.",
    "Quantum coherence exists at body temperature.",
    "Cold fusion occurs at extreme plasma temperatures.",
    "The algorithm separates encoding from generation.",
]

neutral_unrelated = [
    "The Renaissance began in Italy during the 14th century.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The French Revolution fundamentally changed European society.",
    "Beethoven composed his ninth symphony while completely deaf.",
    "The Amazon rainforest produces 20 percent of Earth's oxygen.",
    "Chess was invented in India around the 6th century AD.",
    "The Great Wall of China took centuries to build.",
    "Elephants are the largest land animals on Earth.",
    "The Pacific Ocean covers more than 30 percent of Earth.",
    "William Shakespeare wrote 37 plays during his lifetime.",
    "The human brain contains approximately 86 billion neurons.",
    "Mount Everest is the highest mountain above sea level.",
    "The speed of sound is 343 metres per second in air.",
    "Leonardo da Vinci painted the Mona Lisa in the 1500s.",
    "The periodic table was organized by Dmitri Mendeleev.",
    "The first airplane flight lasted only 12 seconds.",
    "Gravity was described mathematically by Isaac Newton.",
    "The Roman Empire lasted for over 500 years.",
    "Honey never spoils and has been found in ancient tombs.",
    "The human heart beats approximately 100,000 times per day.",
]

technical_text = [
    "Machine learning models are trained on large datasets.",
    "Neural networks consist of layers of connected nodes.",
    "Gradient descent optimizes model parameters iteratively.",
    "The transformer architecture uses self-attention mechanisms.",
    "Backpropagation computes gradients through the network.",
    "Regularization techniques prevent overfitting in models.",
    "Convolutional networks are effective for image recognition.",
    "Recurrent networks process sequential data over time.",
    "Transfer learning adapts pretrained models to new tasks.",
    "Hyperparameter tuning improves model performance significantly.",
]


# ── Loader Helper ─────────────────────────────────────────────────
def load_hybrid(checkpoint_path: str, device: str) -> AgnisGpt2Hybrid:
    ckpt_agnis = find_agnis_checkpoint()
    
    hybrid = AgnisGpt2Hybrid(
        agnis_checkpoint=ckpt_agnis,
        model_name="gpt2",
        device=device,
        local_files_only=False,
        max_settle_steps=5,  # AGNIS_SETTLE
    )
    
    # Resolve Kaggle or local path candidates
    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        search_roots = [
            Path("/kaggle/working"),
            Path.cwd(),
        ]
        input_root = Path("/kaggle/input")
        if input_root.exists():
            search_roots.append(input_root)
            # Scan immediate subfolders (datasets)
            for sub in input_root.iterdir():
                if sub.is_dir():
                    search_roots.append(sub)

        filename = Path(checkpoint_path).name
        found = False
        for root in search_roots:
            if not root.exists():
                continue
            # Try flat glob first
            matches = list(root.glob(filename))
            if not matches:
                # Fallback to recursive rglob
                matches = list(root.rglob(filename))
            if matches:
                resolved_path = matches[0]
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Checkpoint weight file not found: {checkpoint_path}")
            
    print(f"Loading weights from {resolved_path}...")
    ckpt = torch.load(resolved_path, map_location=device)
    
    if "adapter_state" in ckpt:
        hybrid.adapter.load_state_dict(ckpt["adapter_state"])
        if "agnis_core_state" in ckpt:
            hybrid.agnis_core.load_state_dict(ckpt["agnis_core_state"])
            print("Loaded updated adapter_state and Hebbian agnis_core_state.")
        else:
            print("Loaded adapter_state only (AGNIS core remains in baseline state).")
    else:
        # Phase 4 best format
        sd = ckpt.get("adapter_state", ckpt)
        hybrid.adapter.load_state_dict(sd)
        gpt2_key = "gpt2_state" if "gpt2_state" in ckpt else "gpt2_trainable"
        if gpt2_key in ckpt:
            sd_gpt2 = hybrid.gpt2.state_dict()
            sd_gpt2.update(ckpt[gpt2_key])
            hybrid.gpt2.load_state_dict(sd_gpt2)
            print("Loaded GPT-2 trainable adapter-aligned weights.")
            
    hybrid.eval()
    return hybrid


# ── PPL Measurement Function ──────────────────────────────────────
def measure_ppl(model: AgnisGpt2Hybrid, texts: list[str], tokenizer, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(
                text,
                return_tensors='pt',
                max_length=128,
                truncation=True
            ).to(device)
            
            if tokens.shape[1] < 2:
                continue
            
            # Compute representation inside AGNIS
            agnis_hidden = model.compute_agnis_hidden(tokens)
            
            # Embeddings of baseline GPT-2 tokens
            source_embeds = model.gpt2.transformer.wte(tokens)
            
            # Adapter offsets
            adapted = model.adapter(agnis_hidden)
            
            # FUSE (Crucial fix: must add token embeddings so identity is preserved!)
            fused_embeds = source_embeds + adapted
            
            # Feed fused representations to generator
            outputs = model.gpt2(inputs_embeds=fused_embeds)
            logits = outputs.logits
            
            # Shift for autoregressive loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            
    if total_tokens == 0:
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


# ── Main Suite ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AGNIS PPL VERIFICATION SUITE")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    phase4_path = "agnis_gpt2_phase4_best.pt"
    aligned_path = "agnis_continual_v2_adapter_aligned.pt"
    
    print("\n[Step 1] Loading model BEFORE alignment (Phase 4 best)...")
    model_before = load_hybrid(phase4_path, device)
    
    print("\n[Step 1] Loading model AFTER alignment (V2 Continual)...")
    model_after = load_hybrid(aligned_path, device)
    
    tokenizer = model_before.tokenizer
    results = {}
    
    print("\n[Step 2] Executing perplexity evaluation...")
    for source_name, texts in [
        ("fact_related", fact_related),
        ("neutral_unrelated", neutral_unrelated),
        ("technical_text", technical_text),
    ]:
        ppl_b = measure_ppl(model_before, texts, tokenizer, device)
        ppl_a = measure_ppl(model_after, texts, tokenizer, device)
        change = ppl_a - ppl_b
        improved = ppl_a < ppl_b
        
        results[source_name] = {
            'before': ppl_b,
            'after': ppl_a,
            'change': change,
            'improved': improved
        }
        
        print(f"\n{source_name}:")
        print(f"  PPL before: {ppl_b:.2f}")
        print(f"  PPL after:  {ppl_a:.2f}")
        print(f"  Change:     {change:+.2f}")
        print(f"  Verdict:    {'IMPROVED' if improved else 'DEGRADED'}")
        
    print("\n" + "=" * 60)
    print("  PPL VERIFICATION VERDICT")
    print("=" * 60)
    
    neutral_improved = results['neutral_unrelated']['improved']
    fact_improved = results['fact_related']['improved']
    
    if neutral_improved and fact_improved:
        print("""
  VERDICT: PPL IMPROVEMENT IS GENUINE
  
  PPL dropped on BOTH fact-related AND unrelated text.
  The adapter alignment genuinely improved the model.
  Safe to report PPL 7.53 as a real result.
  
  Paper claim: 'Adapter alignment improved general
  language quality from PPL 21.71 to 7.53'
        """)
    elif fact_improved and not neutral_improved:
        print("""
  VERDICT: PPL IMPROVEMENT IS AN ARTIFACT
  
  PPL dropped only on fact-related text.
  Neutral unrelated text shows no improvement.
  The adapter overfit to fact-style text.
  
  Do NOT report PPL 7.53 in the paper.
  Report PPL 29.7 (Phase 4 result) instead.
  
  The fact recall result (10/10) is still valid.
  Only the PPL number is affected.
        """)
    else:
        print("""
  VERDICT: MIXED RESULTS — NEEDS INVESTIGATION
  Report exact numbers and let reviewers decide.
        """)
        
    # Save results
    results_out_path = "/kaggle/working/ppl_verification.json"
    try:
        with open(results_out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {results_out_path}")
    except Exception as e:
        # Fallback to local working dir if /kaggle/working is missing
        local_results = "ppl_verification.json"
        with open(local_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {local_results}")


if __name__ == "__main__":
    main()
