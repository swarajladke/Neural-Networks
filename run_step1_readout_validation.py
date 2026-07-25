"""
run_step1_readout_validation.py — Fast Step 1 Readout & Recurrence Validation Suite
====================================================================================
Implements Opus 5's Step 1 blueprint with exact unpushed state readout coupling:
1. Replaces 1-D scalar regression with standard vocab logits + DeltaSoftmaxReadout.
2. Trains readout head on unpushed settled state h.detach() matching evaluation forward() distribution.
3. Fixes dead recurrence using warm_start=True (preserving state history) and update_temporal=True.
4. Programmatically checks Opus 5's Acceptance Gates:
   - Val PPL < Uniform PPL (V)
   - Val PPL < Unigram Baseline PPL
   - Argmax Prediction Histogram Entropy > 1.0 nat (verifies no constant-output collapse)
   - ||R||_F Frobenius Norm Shift > 0 (proves temporal matrix R is active and learning)
"""

import os
import math
import torch
import numpy as np
from agnis_v4_core import PredictiveHierarchy
from agnis_readout import DeltaSoftmaxReadout, one_hot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_italian_text():
    possible_paths = [
        "slm/input_it.txt",
        "../slm/input_it.txt",
        "input_it.txt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
                
    # Fallback clean Italian text corpus for self-contained testing
    print("[Notice] slm/input_it.txt not found. Generating standard Italian text corpus...")
    corpus = (
        "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, "
        "ché la diritta via era smarrita. Ahi quanto a dir qual era è cosa dura "
        "esta selva selvaggia e aspra e forte che nel pensier rinova la paura! "
        "Tanto è amara che poco è più morte; ma per trattar del ben ch'i' vi trovai, "
        "dirò de l'altre cose ch'i' v'ho scorte. Io non so ben ridir com'i' v'intrai, "
        "tant'era pieno di sonno a quel punto che la verace via abbandonai. "
    ) * 15
    return corpus

def compute_baselines(train_tokens, test_tokens, vocab_size):
    uniform_ppl = float(vocab_size)
    
    counts = torch.bincount(torch.tensor(train_tokens, dtype=torch.long), minlength=vocab_size).float() + 1.0
    p = counts / counts.sum()
    test_t_tensor = torch.tensor(test_tokens[1:], dtype=torch.long)
    unigram_nll = -torch.log(p[test_t_tensor]).mean().item()
    unigram_ppl = math.exp(unigram_nll)
    
    return uniform_ppl, unigram_ppl

@torch.no_grad()
def evaluate_ppl_and_histogram(hierarchy, readout, tokens, vocab_size, max_steps=15):
    hierarchy.reset_states(batch_size=1)
    nll_sum = 0.0
    predictions = []
    
    for i in range(len(tokens) - 1):
        x = one_hot([tokens[i]], vocab_size, DEVICE)
        # update_temporal=True advances temporal state during evaluation!
        h = hierarchy.forward(x, max_steps=max_steps, update_temporal=True)
        log_p = readout.log_probs(h)
        
        target_id = tokens[i + 1]
        nll_sum += -log_p[0, target_id].item()
        
        pred_id = log_p.argmax(dim=-1).item()
        predictions.append(pred_id)
        
    mean_nll = nll_sum / (len(tokens) - 1)
    ppl = math.exp(mean_nll)
    
    # Calculate prediction histogram and entropy (in nats)
    counts = torch.bincount(torch.tensor(predictions, dtype=torch.long), minlength=vocab_size).float()
    probs = counts / counts.sum()
    probs_nonzero = probs[probs > 0]
    entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero)).item()
    
    return ppl, counts.numpy(), entropy

def main():
    print("======================================================================")
    print("  AGNIS STEP 1: READOUT & RECURRENCE VALIDATION SUITE")
    print("======================================================================")
    
    text = load_italian_text()
    chars = sorted(list(set(text)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    all_tokens = [char_to_id[c] for c in text]
    vocab_size = len(chars)
    
    # Benchmark sizing: 5,000 train tokens, 1,000 val tokens (~30 sec runtime)
    max_total = min(len(all_tokens), 6000)
    tokens = all_tokens[:max_total]
    split = min(5000, int(0.833 * len(tokens)))
    
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    
    print(f"[Dataset] Vocabulary Size: {vocab_size} characters | Benchmark Sample: {len(tokens)} tokens")
    print(f"[Dataset] Train Tokens: {len(train_tokens)} | Val Tokens: {len(val_tokens)}")
    
    uniform_ppl, unigram_ppl = compute_baselines(train_tokens, val_tokens, vocab_size)
    print(f"[Baseline] Uniform Perplexity  : {uniform_ppl:.2f}")
    print(f"[Baseline] Unigram Perplexity  : {unigram_ppl:.2f}")
    
    # Instantiate Hierarchy [V, 512, 512] & Local DeltaSoftmaxReadout (eta=0.15, kappa=1.0)
    hierarchy = PredictiveHierarchy([vocab_size, 512, 512], device=DEVICE)
    readout = DeltaSoftmaxReadout(512, vocab_size, device=DEVICE, eta=0.15, kappa=1.0)
    
    # Track initial Frobenius norm of Recurrent Matrix R in Layer 0
    initial_R_norm = hierarchy.layers[0].R.data.norm().item()
    
    # Fast Training Loop (5 Epochs over 5,000 tokens)
    print("\n[Training] Fast Training AGNIS Core + DeltaSoftmaxReadout (5 Epochs)...")
    for epoch in range(5):
        hierarchy.reset_states(batch_size=1)  # Once per document sequence, NOT per token!
        
        for i in range(len(train_tokens) - 1):
            x = one_hot([train_tokens[i]], vocab_size, DEVICE)
            y = one_hot([train_tokens[i + 1]], vocab_size, DEVICE)
            
            # Step A: Settle to read current top state (unpushed)
            h = hierarchy.forward(x, max_steps=15, update_temporal=False)
            
            # Step B: Local teaching signal back through W
            tgt = readout.teaching_signal(h.detach(), y)
            
            # Step C: Learn hierarchy with warm_start=True (preserves x_temporal history!)
            hierarchy.infer_and_learn(
                x, top_level_label=tgt, max_steps=15,
                warm_start=True, beta_push=2.0, dopamine_burst=1.0
            )
            
            # Step D: Local delta-rule update on readout head FROM UNPUSHED SETTLED STATE h.detach()!
            readout.update(h.detach(), y)
            
        train_ppl, _, _ = evaluate_ppl_and_histogram(hierarchy, readout, train_tokens, vocab_size, max_steps=15)
        val_ppl, val_hist, val_entropy = evaluate_ppl_and_histogram(hierarchy, readout, val_tokens, vocab_size, max_steps=15)
        print(f"  Epoch {epoch + 1}: Train PPL = {train_ppl:.2f} | Val PPL = {val_ppl:.2f} | Val Entropy = {val_entropy:.2f} nats")

    final_R_norm = hierarchy.layers[0].R.data.norm().item()
    R_shift = abs(final_R_norm - initial_R_norm)
    
    print("\n======================================================================")
    print("  STEP 1 ACCEPTANCE GATES CHECK")
    print("======================================================================")
    
    gate_1 = val_ppl < uniform_ppl
    gate_2 = val_ppl < unigram_ppl
    gate_3 = val_entropy > 1.0
    gate_4 = R_shift > 1e-4
    
    print(f"1. Val PPL ({val_ppl:.2f}) < Uniform PPL ({uniform_ppl:.2f})    : {'PASSED ✓' if gate_1 else 'FAILED ✗'}")
    print(f"2. Val PPL ({val_ppl:.2f}) < Unigram PPL ({unigram_ppl:.2f})    : {'PASSED ✓' if gate_2 else 'FAILED ✗'}")
    print(f"3. Histogram Entropy ({val_entropy:.2f} nats) > 1.0 nats      : {'PASSED ✓' if gate_3 else 'FAILED ✗'}")
    print(f"4. ||R||_F Norm Shift ({R_shift:.6f}) > 0.0001              : {'PASSED ✓' if gate_4 else 'FAILED ✗'}")
    
    all_passed = gate_1 and gate_2 and gate_3 and gate_4
    print(f"\nOverall Step 1 Status: {'PASSED ALL GATES ✓' if all_passed else 'FAILED ✗'}")

if __name__ == "__main__":
    main()
