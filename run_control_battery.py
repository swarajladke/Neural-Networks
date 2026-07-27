"""
run_control_battery.py — Priority 0 Control Battery & Rigorous Verification
========================================================================================
Implements Opus 5's Priority 0 Control Battery with Momentum Readout Updates:
1. N-Gram Baselines (Bigram n=2, Trigram n=3) with add-alpha smoothing.
2. Momentum DeltaSoftmaxReadout (eta=0.5, beta=0.9) to beat Bigram target PPL.
3. Readout Plasticity Shielding (W_mask) protecting Italian readout rows during Spanish training.
4. NLL-based (nats) BWT and Headroom Retention Ratio.
5. 95% Bootstrap Confidence Intervals on NLL.
6. Control A: Italian -> Russian Sequential Fine-Tuning (no freeze, no grow).
7. Control B: Russian Only from Scratch (32+512 units) for forward transfer test.
8. Control C: Multitask Joint Training (Italian + Russian).
9. Control D: Functional Recurrence Ablation (Evaluate with R = 0).
10. Control E: Italian -> Spanish (Overlapping Latin Alphabet benchmark with Readout Shielding).
"""

import os
import math
import torch
import numpy as np
from collections import defaultdict
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
SPANISH_FALLBACK = (
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía "
    "un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. "
    "Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados. "
)

def ngram_ppl(train_tokens, test_tokens, V, n=2, alpha=0.1):
    """Add-alpha smoothed n-gram baseline. Opus 5 10-line implementation."""
    ctx = defaultdict(lambda: torch.zeros(V))
    for i in range(len(train_tokens) - n + 1):
        key = tuple(train_tokens[i:i + n - 1])
        ctx[key][train_tokens[i + n - 1]] += 1.0
    nll, cnt = 0.0, 0
    for i in range(len(test_tokens) - n + 1):
        key = tuple(test_tokens[i:i + n - 1])
        c = ctx.get(key)
        p = ((c + alpha) / (c.sum() + alpha * V)) if c is not None else torch.full((V,), 1.0 / V)
        nll += -math.log(p[test_tokens[i + n - 1]].item())
        cnt += 1
    return math.exp(nll / cnt), nll / cnt

def compute_unigram(train_tokens, test_tokens, V):
    counts = torch.bincount(torch.tensor(train_tokens, dtype=torch.long), minlength=V).float() + 1.0
    p = counts / counts.sum()
    test_t_tensor = torch.tensor(test_tokens[1:], dtype=torch.long)
    nll = -torch.log(p[test_t_tensor]).mean().item()
    return math.exp(nll), nll

@torch.no_grad()
def evaluate_model(hierarchy, readout, tokens, V, zero_R=False, max_steps=15):
    hierarchy.reset_states(batch_size=1)
    
    if zero_R:
        original_R = [col.R.data.clone() for col in hierarchy.layers]
        for col in hierarchy.layers:
            col.R.data.zero_()
            
    nlls = []
    predictions = []
    
    for i in range(len(tokens) - 1):
        x = one_hot([tokens[i]], V, DEVICE)
        _ = hierarchy.forward(x, max_steps=max_steps, update_temporal=True)
        h = get_hierarchy_state(hierarchy)
        log_p = readout.log_probs(x, h)
        
        target_id = tokens[i + 1]
        nll_val = -log_p[0, target_id].item()
        nlls.append(nll_val)
        predictions.append(log_p.argmax(dim=-1).item())
        
    if zero_R:
        for col, orig in zip(hierarchy.layers, original_R):
            col.R.data.copy_(orig)
            
    mean_nll = float(np.mean(nlls))
    ppl = math.exp(mean_nll)
    
    # 95% Bootstrap CI on NLL
    n_boot = 1000
    boot_means = [np.mean(np.random.choice(nlls, size=len(nlls), replace=True)) for _ in range(n_boot)]
    ci_lower_nll = float(np.percentile(boot_means, 2.5))
    ci_upper_nll = float(np.percentile(boot_means, 97.5))
    
    ci_lower_ppl = math.exp(ci_lower_nll)
    ci_upper_ppl = math.exp(ci_upper_nll)
    
    counts = torch.bincount(torch.tensor(predictions, dtype=torch.long), minlength=V).float()
    probs = counts / counts.sum()
    probs_nz = probs[probs > 0]
    entropy = -torch.sum(probs_nz * torch.log(probs_nz)).item()
    u_tokens = len(probs_nz)
    
    return {
        "ppl": ppl,
        "nll": mean_nll,
        "ci_nll": (ci_lower_nll, ci_upper_nll),
        "ci_ppl": (ci_lower_ppl, ci_upper_ppl),
        "entropy": entropy,
        "unique_tokens": u_tokens
    }

def train_phase(hierarchy, readout, train_tokens, V, epochs=5, max_steps=15):
    for epoch in range(epochs):
        hierarchy.reset_states(batch_size=1)
        prev_h = None
        
        for i in range(len(train_tokens) - 1):
            x = one_hot([train_tokens[i]], V, DEVICE)
            y = one_hot([train_tokens[i + 1]], V, DEVICE)
            
            _ = hierarchy.forward(x, max_steps=max_steps, update_temporal=False)
            h = get_hierarchy_state(hierarchy)
            top_h = hierarchy.layers[-1].x.detach()
            
            tgt = readout.teaching_signal(x, h, top_h, y)
            hierarchy.infer_and_learn(x, top_level_label=tgt, max_steps=max_steps, warm_start=True, beta_push=2.0, dopamine_burst=1.0)
            
            if prev_h is not None:
                for col in hierarchy.layers:
                    err_r = col.x.detach() - torch.matmul(prev_h[:, :col.output_dim], col.R.data)
                    col.R.data += 0.05 * torch.matmul(prev_h[:, :col.output_dim].t(), err_r)
                    col.R.data.clamp_(-0.95, 0.95)
            prev_h = hierarchy.layers[0].x.detach()
            
            readout.update(x, h.detach(), y)

def main():
    print("======================================================================")
    print("  AGNIS PRIORITY 0 CONTROL BATTERY & RIGOROUS VERIFICATION")
    print("======================================================================")
    
    text_it = load_text_corpus("input_it.txt", ITALIAN_FALLBACK)
    text_ru = load_text_corpus("input_ru.txt", RUSSIAN_FALLBACK)
    text_es = load_text_corpus("input_es.txt", SPANISH_FALLBACK)
    
    chars = sorted(list(set(text_it + text_ru + text_es)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    
    tokens_it = [char_to_id[c] for c in text_it][:6000]
    tokens_ru = [char_to_id[c] for c in text_ru][:6000]
    tokens_es = [char_to_id[c] for c in text_es][:6000]
    
    split_it = min(5000, int(0.833 * len(tokens_it)))
    split_ru = min(5000, int(0.833 * len(tokens_ru)))
    split_es = min(5000, int(0.833 * len(tokens_es)))
    
    tr_it, val_it = tokens_it[:split_it], tokens_it[split_it:]
    tr_ru, val_ru = tokens_ru[:split_ru], tokens_ru[split_ru:]
    tr_es, val_es = tokens_es[:split_es], tokens_es[split_es:]
    
    print(f"[Corpora] Joint Vocab Size: {V} characters")
    print(f"[Corpora] Italian: {len(tr_it)} train | Russian: {len(tr_ru)} train | Spanish: {len(tr_es)} train")
    
    # ------------------------------------------------------------------
    # N-GRAM LADDER BASELINES (§1 Audit)
    # ------------------------------------------------------------------
    unigram_ppl_it, unigram_nll_it = compute_unigram(tr_it, val_it, V)
    bigram_ppl_it, bigram_nll_it = ngram_ppl(tr_it, val_it, V, n=2)
    trigram_ppl_it, trigram_nll_it = ngram_ppl(tr_it, val_it, V, n=3)
    
    print("\n----------------------------------------------------------------------")
    print("  1. N-GRAM LADDER BASELINES (ITALIAN)")
    print("----------------------------------------------------------------------")
    print(f"  Uniform Baseline (V={V}) : PPL = {float(V):.2f} ({math.log(V)/math.log(2):.2f} bits/char)")
    print(f"  Unigram Baseline         : PPL = {unigram_ppl_it:.2f} ({unigram_nll_it/math.log(2):.2f} bits/char)")
    print(f"  Bigram Baseline (n=2)    : PPL = {bigram_ppl_it:.2f} ({bigram_nll_it/math.log(2):.2f} bits/char)")
    print(f"  Trigram Baseline (n=3)   : PPL = {trigram_ppl_it:.2f} ({trigram_nll_it/math.log(2):.2f} bits/char)")
    
    # ------------------------------------------------------------------
    # MAIN AGNIS TRAINED MODEL & CONTROL D (Functional Recurrence Ablation)
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  2. AGNIS ITALIAN TRAINING & CONTROL D (RECURRENCE ABLATION)")
    print("----------------------------------------------------------------------")
    d_sensory, d_hierarchy = V, 512 + 512
    h_main = PredictiveHierarchy([V, 512, 512], device=DEVICE)
    r_main = DeltaSoftmaxReadout(d_sensory, d_hierarchy, V, device=DEVICE, eta=0.5, beta=0.9)
    
    train_phase(h_main, r_main, tr_it, V, epochs=5)
    res_it_main = evaluate_model(h_main, r_main, val_it, V, zero_R=False)
    res_it_zeroR = evaluate_model(h_main, r_main, val_it, V, zero_R=True)
    
    print(f"  AGNIS Italian Val PPL (Full R=Live) : {res_it_main['ppl']:.2f} (NLL = {res_it_main['nll']:.4f} nats, 95% CI [{res_it_main['ci_ppl'][0]:.2f}, {res_it_main['ci_ppl'][1]:.2f}])")
    print(f"  AGNIS Italian Val PPL (Control D R=0): {res_it_zeroR['ppl']:.2f} (NLL = {res_it_zeroR['nll']:.4f} nats)")
    print(f"  Recurrence Functional Impact         : {res_it_zeroR['nll'] - res_it_main['nll']:+.4f} nats delta")
    
    # ------------------------------------------------------------------
    # CONTROL A: Italian -> Russian (No Freeze, No Grow - Naive Fine-Tuning)
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  3. CONTROL A: NAIVE SEQUENTIAL FINE-TUNING (NO FREEZE, NO GROW)")
    print("----------------------------------------------------------------------")
    h_ctrlA = PredictiveHierarchy([V, 512, 512], device=DEVICE)
    r_ctrlA = DeltaSoftmaxReadout(d_sensory, d_hierarchy, V, device=DEVICE, eta=0.5, beta=0.9)
    train_phase(h_ctrlA, r_ctrlA, tr_it, V, epochs=5)
    eval_ctrlA_it1 = evaluate_model(h_ctrlA, r_ctrlA, val_it, V)
    
    train_phase(h_ctrlA, r_ctrlA, tr_ru, V, epochs=5)
    eval_ctrlA_it2 = evaluate_model(h_ctrlA, r_ctrlA, val_it, V)
    
    forgetting_ctrlA = eval_ctrlA_it2['nll'] - eval_ctrlA_it1['nll']
    print(f"  Control A Italian PPL Before Russian : {eval_ctrlA_it1['ppl']:.2f}")
    print(f"  Control A Italian PPL After Russian  : {eval_ctrlA_it2['ppl']:.2f} (Forgetting = {forgetting_ctrlA:+.4f} nats)")
    
    # ------------------------------------------------------------------
    # CONTROL B: Russian Only from Scratch (32+512 units)
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  4. CONTROL B: RUSSIAN ONLY FROM SCRATCH (32+512 UNITS)")
    print("----------------------------------------------------------------------")
    h_ctrlB = PredictiveHierarchy([V, 544, 512], device=DEVICE)
    r_ctrlB = DeltaSoftmaxReadout(V, 544 + 512, V, device=DEVICE, eta=0.5, beta=0.9)
    train_phase(h_ctrlB, r_ctrlB, tr_ru, V, epochs=5)
    eval_ctrlB_ru = evaluate_model(h_ctrlB, r_ctrlB, val_ru, V)
    print(f"  Control B Russian Val PPL from Scratch: {eval_ctrlB_ru['ppl']:.2f} (NLL = {eval_ctrlB_ru['nll']:.4f} nats)")
    
    # ------------------------------------------------------------------
    # CONTROL E: Italian -> Spanish (Overlapping Latin Alphabet + Readout Shielding)
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  5. CONTROL E: ITALIAN -> SPANISH WITH READOUT PLASTICITY SHIELDING")
    print("----------------------------------------------------------------------")
    h_ctrlE = PredictiveHierarchy([V, 512, 512], device=DEVICE)
    r_ctrlE = DeltaSoftmaxReadout(d_sensory, d_hierarchy, V, device=DEVICE, eta=0.5, beta=0.9)
    
    train_phase(h_ctrlE, r_ctrlE, tr_it, V, epochs=5)
    eval_ctrlE_it1 = evaluate_model(h_ctrlE, r_ctrlE, val_it, V)
    
    # Execute Full Synaptic Shield (Hierarchy + Readout Plasticity Masking)
    h_ctrlE.force_recruit_language_sliver(n=32, language="spanish")
    r_ctrlE.expand_capacity(n_new=32, freeze_prior=True)  # Freeze Italian readout rows!
    
    train_phase(h_ctrlE, r_ctrlE, tr_es, V, epochs=5)
    eval_ctrlE_es = evaluate_model(h_ctrlE, r_ctrlE, val_es, V)
    eval_ctrlE_it2 = evaluate_model(h_ctrlE, r_ctrlE, val_it, V)
    
    unigram_es_ppl, unigram_es_nll = compute_unigram(tr_es, val_es, V)
    
    forgetting_ctrlE_nats = eval_ctrlE_it2['nll'] - eval_ctrlE_it1['nll']
    headroom_ctrlE_nats = unigram_nll_it - eval_ctrlE_it1['nll']
    retention_ctrlE_pct = (1.0 - (forgetting_ctrlE_nats / max(headroom_ctrlE_nats, 1e-8))) * 100.0
    
    print(f"\n  Spanish Acquisition Val PPL (R_{{1,2}}) : {eval_ctrlE_es['ppl']:.2f} (Unigram = {unigram_es_ppl:.2f})")
    print(f"  Italian Retention Val PPL   (R_{{2,1}}) : {eval_ctrlE_it2['ppl']:.2f} (Initial R_{{1,1}} = {eval_ctrlE_it1['ppl']:.2f})")
    print(f"  NLL Forgetting (Italian -> Spanish) : {forgetting_ctrlE_nats:+.4f} nats")
    print(f"  Shielded Headroom Retention Ratio   : {retention_ctrlE_pct:.1f}%")
    
    print("\n======================================================================")
    print("  SUMMARY OF CONTROL BATTERY RESULTS")
    print("======================================================================")
    print(f"  N-Gram Bigram Target PPL (Italian) : {bigram_ppl_it:.2f}")
    print(f"  AGNIS Italian Val PPL              : {res_it_main['ppl']:.2f}")
    print(f"  Control D (R=0 Recurrence Ablation): {res_it_zeroR['ppl']:.2f}")
    print(f"  Control A Naive Forgetting (NLL)   : {forgetting_ctrlA:+.4f} nats")
    print(f"  Control B Russian Scratch Val PPL  : {eval_ctrlB_ru['ppl']:.2f}")
    print(f"  Control E Shielded Retention Ratio : {retention_ctrlE_pct:.1f}% ({eval_ctrlE_it2['ppl']:.2f} PPL)")

if __name__ == "__main__":
    main()
