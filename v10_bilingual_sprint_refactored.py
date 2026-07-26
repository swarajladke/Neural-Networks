"""
v10_bilingual_sprint_refactored.py — Refactored Bilingual Continual Learning Benchmark
========================================================================================
Implements Opus 5's Step 2 blueprint:
1. Replaces vacuous `freeze_experts()` no-op loop with `force_recruit_language_sliver(n=32, language="russian")`.
2. Expands readout capacity via `readout.expand_capacity(n_new=32)` to preserve Italian weights while adding Russian capacity.
3. Integrates Step 1's validated DeltaSoftmaxReadout and active temporal recurrence (warm_start=True).
4. Evaluates on held-out 90/10 validation splits for both Italian and Russian.
5. Computes and logs the full Continual Learning Performance Matrix (R_{i,j}):
   - R_{1,1}: Italian Val PPL after Phase 1 (Italian Training)
   - R_{1,2}: Russian Val PPL after Phase 2 (Russian Acquisition)
   - R_{2,1}: Italian Val PPL after Phase 2 (Italian Retention after Russian Training)
   - Backward Transfer (BWT) = R_{2,1} - R_{1,1} (Absolute change in Italian PPL)
"""

import os
import math
import torch
import numpy as np
from agnis_v4_core import PredictiveHierarchy
from agnis_readout import DeltaSoftmaxReadout, get_hierarchy_state, one_hot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text_corpus(filename, fallback_text):
    possible_paths = [
        f"slm/{filename}",
        f"../slm/{filename}",
        filename
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    print(f"[Notice] {filename} not found. Generating clean fallback corpus...")
    return fallback_text * 15

ITALIAN_FALLBACK = (
    "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, "
    "ché la diritta via era smarrita. Ahi quanto a dir qual era è cosa dura "
    "esta selva selvaggia e aspra e forte che nel pensier rinova la paura! "
)
RUSSIAN_FALLBACK = (
    "В процессе бесконечного познания человек стремится к пониманию гармонии мироздания. "
    "Каждый шаг на этом пути раскрывает новые горизонты мысли и открывает древние тайны природы. "
    "Сохранение знаний является фундаментальной задачей нашей цивилизации. "
)

def compute_unigram_ppl(train_tokens, test_tokens, vocab_size):
    counts = torch.bincount(torch.tensor(train_tokens, dtype=torch.long), minlength=vocab_size).float() + 1.0
    p = counts / counts.sum()
    test_t_tensor = torch.tensor(test_tokens[1:], dtype=torch.long)
    unigram_nll = -torch.log(p[test_t_tensor]).mean().item()
    return math.exp(unigram_nll)

@torch.no_grad()
def evaluate_ppl(hierarchy, readout, tokens, vocab_size, max_steps=15):
    hierarchy.reset_states(batch_size=1)
    nll_sum = 0.0
    
    for i in range(len(tokens) - 1):
        x = one_hot([tokens[i]], vocab_size, DEVICE)
        # update_temporal=True advances temporal state during evaluation!
        _ = hierarchy.forward(x, max_steps=max_steps, update_temporal=True)
        h = get_hierarchy_state(hierarchy)
        log_p = readout.log_probs(x, h)
        
        target_id = tokens[i + 1]
        nll_sum += -log_p[0, target_id].item()
        
    mean_nll = nll_sum / (len(tokens) - 1)
    return math.exp(mean_nll)

def train_phase(hierarchy, readout, train_tokens, vocab_size, epochs=3, max_steps=15, phase_name="Training"):
    print(f"\n[{phase_name}] Training for {epochs} Epochs over {len(train_tokens)} tokens...")
    for epoch in range(epochs):
        hierarchy.reset_states(batch_size=1)  # Reset state once per document sequence
        
        for i in range(len(train_tokens) - 1):
            x = one_hot([train_tokens[i]], vocab_size, DEVICE)
            y = one_hot([train_tokens[i + 1]], vocab_size, DEVICE)
            
            # Step A: Settle hierarchy to read current state
            _ = hierarchy.forward(x, max_steps=max_steps, update_temporal=False)
            h = get_hierarchy_state(hierarchy)
            top_h = hierarchy.layers[-1].x.detach()
            
            # Step B: Local teaching signal back through W
            tgt = readout.teaching_signal(x, h, top_h, y)
            
            # Step C: Learn hierarchy with warm_start=True (preserves x_temporal history!)
            hierarchy.infer_and_learn(
                x, top_level_label=tgt, max_steps=max_steps,
                warm_start=True, beta_push=2.0, dopamine_burst=1.0
            )
            
            # Clamp R matrix to prevent unconstrained spectral blowup
            for col in hierarchy.layers:
                col.R.data.clamp_(-0.95, 0.95)
            
            # Step D: Local delta-rule update on readout head
            readout.update(x, h.detach(), y)

def main():
    print("======================================================================")
    print("  AGNIS STEP 2: REFACTORED BILINGUAL CONTINUAL LEARNING BENCHMARK")
    print("======================================================================")
    
    text_it = load_text_corpus("input_it.txt", ITALIAN_FALLBACK)
    text_ru = load_text_corpus("input_ru.txt", RUSSIAN_FALLBACK)
    
    # Combined vocabulary (joint inventory)
    chars = sorted(list(set(text_it + text_ru)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    tokens_it = [char_to_id[c] for c in text_it][:6000]
    tokens_ru = [char_to_id[c] for c in text_ru][:6000]
    
    split_it = min(5000, int(0.833 * len(tokens_it)))
    split_ru = min(5000, int(0.833 * len(tokens_ru)))
    
    train_it, val_it = tokens_it[:split_it], tokens_it[split_it:]
    train_ru, val_ru = tokens_ru[:split_ru], tokens_ru[split_ru:]
    
    print(f"[Dataset] Vocabulary Size: {vocab_size} characters")
    print(f"[Dataset] Italian Tokens : Train = {len(train_it)}, Val = {len(val_it)}")
    print(f"[Dataset] Russian Tokens : Train = {len(train_ru)}, Val = {len(val_ru)}")
    
    unigram_it = compute_unigram_ppl(train_it, val_it, vocab_size)
    unigram_ru = compute_unigram_ppl(train_ru, val_ru, vocab_size)
    print(f"[Baselines] Italian Unigram PPL: {unigram_it:.2f} | Russian Unigram PPL: {unigram_ru:.2f}")
    
    # Instantiate Hierarchy [V, 512, 512] & Readout
    d_sensory = vocab_size
    d_hierarchy = 512 + 512
    hierarchy = PredictiveHierarchy([vocab_size, 512, 512], device=DEVICE)
    readout = DeltaSoftmaxReadout(d_sensory, d_hierarchy, vocab_size, device=DEVICE, eta=1.0, kappa=1.0)
    
    # ------------------------------------------------------------------
    # PHASE 1: Italian Training (Task 1)
    # ------------------------------------------------------------------
    train_phase(hierarchy, readout, train_it, vocab_size, epochs=3, phase_name="Phase 1: Italian Training")
    
    r1_1 = evaluate_ppl(hierarchy, readout, val_it, vocab_size)
    print(f"\n[Matrix Entry R_{{1,1}}] Italian Val PPL (Baseline after Task 1): {r1_1:.2f}")
    
    # ------------------------------------------------------------------
    # PHASE 2: Dynamic Sliver Expansion for Russian (Task 2)
    # ------------------------------------------------------------------
    print("\n[Shield Protocol] Executing `force_recruit_language_sliver(n=32, language='russian')`...")
    hierarchy.force_recruit_language_sliver(n=32, language="russian")
    readout.expand_capacity(n_new=32)
    
    train_phase(hierarchy, readout, train_ru, vocab_size, epochs=3, phase_name="Phase 2: Russian Training")
    
    r1_2 = evaluate_ppl(hierarchy, readout, val_ru, vocab_size)
    r2_1 = evaluate_ppl(hierarchy, readout, val_it, vocab_size)
    
    print(f"\n[Matrix Entry R_{{1,2}}] Russian Val PPL (Acquisition after Task 2): {r1_2:.2f}")
    print(f"[Matrix Entry R_{{2,1}}] Italian Val PPL (Retention after Task 2)  : {r2_1:.2f}")
    
    # ------------------------------------------------------------------
    # CONTINUAL LEARNING MATRIX & BACKWARD TRANSFER SUMMARY
    # ------------------------------------------------------------------
    bwt = r2_1 - r1_1
    retention_pct = (r1_1 / r2_1) * 100.0 if r2_1 > 0 else 0.0
    
    print("\n======================================================================")
    print("  GEM CONTINUAL LEARNING PERFORMANCE MATRIX & RETENTION EVALUATION")
    print("======================================================================")
    print(f"  R_{{1,1}} (Italian Val PPL after Task 1) : {r1_1:.2f}")
    print(f"  R_{{1,2}} (Russian Val PPL Acquisition)   : {r1_2:.2f}")
    print(f"  R_{{2,1}} (Italian Val PPL Retention)     : {r2_1:.2f}")
    print(f"  Backward Transfer (BWT)                 : {bwt:+.2f} PPL points")
    print(f"  Empirical Retention Ratio (R_{{1,1}}/R_{{2,1}}) : {retention_pct:.1f}%")
    
    # Validation Check against Baselines
    it_learned = r2_1 < unigram_it
    ru_learned = r1_2 < unigram_ru
    print(f"\nItalian Retention Beats Unigram ({r2_1:.2f} < {unigram_it:.2f}) : {'PASSED ✓' if it_learned else 'FAILED ✗'}")
    print(f"Russian Acquisition Beats Unigram ({r1_2:.2f} < {unigram_ru:.2f}) : {'PASSED ✓' if ru_learned else 'FAILED ✗'}")
    
    print(f"\nOverall Step 2 Benchmark Status: {'PASSED ✓' if (it_learned and ru_learned) else 'FAILED ✗'}")

if __name__ == "__main__":
    main()
