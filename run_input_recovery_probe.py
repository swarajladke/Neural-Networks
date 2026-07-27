"""
run_input_recovery_probe.py — Opus 5 Input-Recovery Probe Diagnostic
======================================================================
Tests whether hierarchy settling dynamics preserve input character identity:
Trains a linear probe h -> current_token (target = current token, NOT next token).
If input-recovery accuracy is < 95%, settling is destroying input identity,
preventing the model from reaching Bigram (12.34 PPL).
"""

import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from agnis_v4_core import PredictiveHierarchy
from agnis_readout import DeltaSoftmaxReadout, get_hierarchy_state, one_hot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_italian_text():
    possible_paths = ["slm/input_it.txt", "../slm/input_it.txt", "input_it.txt"]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura... " * 15

def main():
    print("======================================================================")
    print("  OPUS 5 INPUT-RECOVERY PROBE DIAGNOSTIC")
    print("======================================================================")
    
    text = load_italian_text()
    chars = sorted(list(set(text)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    all_tokens = [char_to_id[c] for c in text]
    V = len(chars)
    
    tokens = all_tokens[:6000]
    split = 5000
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    
    hierarchy = PredictiveHierarchy([V, 512, 512], device=DEVICE)
    # Probe maps hierarchy hidden state h -> current_token
    probe = DeltaSoftmaxReadout(V, 512 + 512, V, device=DEVICE, eta=1.0)
    
    print(f"[Probe] Training Input-Recovery Probe h -> current_token (3 Epochs)...")
    for epoch in range(3):
        hierarchy.reset_states(batch_size=1)
        for i in range(len(train_tokens)):
            tok = train_tokens[i]
            x = one_hot([tok], V, DEVICE)
            _ = hierarchy.forward(x, max_steps=15, update_temporal=False)
            h = get_hierarchy_state(hierarchy)
            
            # Probe target is CURRENT token x, NOT next token y!
            probe.update(x, h.detach(), x)
            
    # Evaluate Input-Recovery Accuracy on Val Split
    hierarchy.reset_states(batch_size=1)
    correct = 0
    total = len(val_tokens)
    
    for tok in val_tokens:
        x = one_hot([tok], V, DEVICE)
        _ = hierarchy.forward(x, max_steps=15, update_temporal=True)
        h = get_hierarchy_state(hierarchy)
        pred_id = probe.log_probs(x, h).argmax(dim=-1).item()
        if pred_id == tok:
            correct += 1
            
    acc = (correct / total) * 100.0
    print(f"\n[Result] Input-Recovery Probe Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Target Threshold: >= 95.00%")
    print(f"Status: {'PASSED ✓ (Settling preserves input identity)' if acc >= 95.0 else 'FAILED ✗ (Settling is destroying input identity!)'}")

if __name__ == "__main__":
    main()
