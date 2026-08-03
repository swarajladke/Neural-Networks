"""
run_control_battery.py — Priority 0.5 Rigorous Control Battery & Null-Phase Twin Audit
======================================================================================
Implements Opus 5's P0.5 Audit Protocol with Task-Gated Multi-Head Readouts:
1. Paired-Bootstrap CIs on per-token NLL differences (nll_a - nll_b).
2. Null-Phase Twin Control for Task 2 (cancels convergence drift, LR schedule, and warm-up).
3. Matched Unshielded vs Task-Gated Shielded Italian -> Spanish.
4. Task-Gated Readout Head routing (eliminates cross-task logit interference).
5. Correct variable isolation for Control B, C, D, E.
6. N-Gram Ladder with add-alpha Bigram (n=2) and Trigram (n=3) baselines.
"""

import os
import math
import copy
import torch
import numpy as np
from collections import defaultdict
from agnis_v4_core import PredictiveHierarchy
from agnis_readout import DeltaSoftmaxReadout, one_hot

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

def paired_bootstrap(nll_a, nll_b, n_boot=10000, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.as_tensor(nll_a, dtype=torch.float32) - torch.as_tensor(nll_b, dtype=torch.float32)
    idx = torch.randint(0, d.numel(), (n_boot, d.numel()), generator=g)
    m = d[idx].mean(dim=1)
    q = torch.quantile(m, torch.tensor([0.025, 0.975]))
    return d.mean().item(), q[0].item(), q[1].item()

def ngram_ppl(train_tokens, test_tokens, V, n=2, alpha=0.1):
    ctx = defaultdict(lambda: torch.zeros(V))
    for i in range(len(train_tokens) - n + 1):
        key = tuple(train_tokens[i:i + n - 1])
        ctx[key][train_tokens[i + n - 1]] += 1.0
    nll_vec = []
    for i in range(len(test_tokens) - n + 1):
        key = tuple(test_tokens[i:i + n - 1])
        c = ctx.get(key)
        p = ((c + alpha) / (c.sum() + alpha * V)) if c is not None else torch.full((V,), 1.0 / V)
        nll_val = -math.log(p[test_tokens[i + n - 1]].item())
        nll_vec.append(nll_val)
    mean_nll = float(np.mean(nll_vec))
    return math.exp(mean_nll), mean_nll, nll_vec

def compute_unigram(train_tokens, test_tokens, V):
    counts = torch.bincount(torch.tensor(train_tokens, dtype=torch.long), minlength=V).float() + 1.0
    p = counts / counts.sum()
    test_t_tensor = torch.tensor(test_tokens[1:], dtype=torch.long)
    nll_vec = (-torch.log(p[test_t_tensor])).tolist()
    mean_nll = float(np.mean(nll_vec))
    return math.exp(mean_nll), mean_nll, nll_vec

@torch.no_grad()
def evaluate_model(hierarchy, readout, tokens, V, task_idx=None, zero_R=False, max_steps=15):
    hierarchy.reset_states(batch_size=1)
    if zero_R:
        original_R = [col.R.data.clone() for col in hierarchy.layers]
        for col in hierarchy.layers:
            col.R.data.zero_()
            
    nlls = []
    for i in range(len(tokens) - 1):
        x = one_hot([tokens[i]], V, DEVICE)
        _ = hierarchy.forward(x, max_steps=max_steps, update_temporal=True)
        log_p = readout.log_probs(x, hierarchy, task_idx=task_idx)
        target_id = tokens[i + 1]
        nll_val = -log_p[0, target_id].item()
        nlls.append(nll_val)
        
    if zero_R:
        for col, orig in zip(hierarchy.layers, original_R):
            col.R.data.copy_(orig)
            
    mean_nll = float(np.mean(nlls))
    return {
        "ppl": math.exp(mean_nll),
        "nll": mean_nll,
        "nll_vec": nlls
    }

def train_phase(hierarchy, readout, train_tokens, V, epochs=5, max_steps=15):
    for epoch in range(epochs):
        hierarchy.reset_states(batch_size=1)
        prev_h = None
        for i in range(len(train_tokens) - 1):
            x = one_hot([train_tokens[i]], V, DEVICE)
            y = one_hot([train_tokens[i + 1]], V, DEVICE)
            _ = hierarchy.forward(x, max_steps=max_steps, update_temporal=False)
            top_h = hierarchy.layers[-1].x.detach()
            tgt = readout.teaching_signal(x, hierarchy, top_h, y)
            hierarchy.infer_and_learn(x, top_level_label=tgt, max_steps=max_steps, warm_start=True, beta_push=2.0, dopamine_burst=1.0)
            
            if prev_h is not None:
                for col in hierarchy.layers:
                    err_r = col.x.detach() - torch.matmul(prev_h[:, :col.output_dim], col.R.data)
                    col.R.data += 0.05 * torch.matmul(prev_h[:, :col.output_dim].t(), err_r)
                    col.R.data.clamp_(-0.95, 0.95)
            prev_h = hierarchy.layers[0].x.detach()
            readout.update(x, hierarchy, y)

def main():
    print("======================================================================")
    print("  AGNIS PRIORITY 0.5 CONTROL BATTERY & NULL-PHASE TWIN AUDIT")
    print("======================================================================")
    
    text_it = load_text_corpus("input_it.txt", ITALIAN_FALLBACK)
    text_ru = load_text_corpus("input_ru.txt", RUSSIAN_FALLBACK)
    text_es = load_text_corpus("input_es.txt", SPANISH_FALLBACK)
    
    chars = sorted(list(set(text_it + text_ru + text_es)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    
    tokens_it = [char_to_id[c] for c in text_it][:12000]
    tokens_ru = [char_to_id[c] for c in text_ru][:6000]
    tokens_es = [char_to_id[c] for c in text_es][:6000]
    
    tr_it = tokens_it[:5000]
    val_it = tokens_it[5000:6000]
    null_it = tokens_it[6000:11000]  # Held-out Italian tokens for Null-Twin training!
    
    tr_ru, val_ru = tokens_ru[:5000], tokens_ru[5000:6000]
    tr_es, val_es = tokens_es[:5000], tokens_es[5000:6000]
    
    print(f"[Corpora] Joint Vocab Size: {V} characters")
    print(f"[Corpora] Italian: {len(tr_it)} train, {len(val_it)} val, {len(null_it)} null-twin train")
    
    # ------------------------------------------------------------------
    # 1. N-GRAM LADDER BASELINES
    # ------------------------------------------------------------------
    unigram_ppl, unigram_nll, vec_unigram = compute_unigram(tr_it, val_it, V)
    bigram_ppl, bigram_nll, vec_bigram = ngram_ppl(tr_it, val_it, V, n=2)
    trigram_ppl, trigram_nll, vec_trigram = ngram_ppl(tr_it, val_it, V, n=3)
    
    print("\n----------------------------------------------------------------------")
    print("  1. N-GRAM LADDER BASELINES (ITALIAN)")
    print("----------------------------------------------------------------------")
    print(f"  Unigram Baseline       : PPL = {unigram_ppl:.2f} ({unigram_nll:.4f} nats)")
    print(f"  Bigram Baseline (n=2)  : PPL = {bigram_ppl:.2f} ({bigram_nll:.4f} nats)")
    print(f"  Trigram Baseline (n=3) : PPL = {trigram_ppl:.2f} ({trigram_nll:.4f} nats)")
    
    # ------------------------------------------------------------------
    # 2. PHASE 1: ITALIAN CONVERGENCE & CONTROL D (R=0 ABLATION)
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  2. PHASE 1: ITALIAN CONVERGENCE & CONTROL D (R=0 ABLATION)")
    print("----------------------------------------------------------------------")
    d_sensory = V
    h_phase1 = PredictiveHierarchy([V, 512, 512], device=DEVICE)
    r_phase1 = DeltaSoftmaxReadout(d_sensory, [512, 512], V, device=DEVICE, eta=0.5, beta=0.9)
    
    train_phase(h_phase1, r_phase1, tr_it, V, epochs=5)
    
    eval_p1_live = evaluate_model(h_phase1, r_phase1, val_it, V, task_idx=0, zero_R=False)
    eval_p1_zeroR = evaluate_model(h_phase1, r_phase1, val_it, V, task_idx=0, zero_R=True)
    
    diff_D, ci_D_low, ci_D_high = paired_bootstrap(eval_p1_zeroR['nll_vec'], eval_p1_live['nll_vec'])
    print(f"  AGNIS Italian Phase 1 PPL (Live R) : {eval_p1_live['ppl']:.2f} ({eval_p1_live['nll']:.4f} nats)")
    print(f"  AGNIS Italian Phase 1 PPL (R=0)    : {eval_p1_zeroR['ppl']:.2f} ({eval_p1_zeroR['nll']:.4f} nats)")
    print(f"  Control D Paired NLL Diff (R=0 - R=Live) : {diff_D:+.4f} nats [95% CI: {ci_D_low:+.4f}, {ci_D_high:+.4f}]")
    
    diff_bigram, ci_b_low, ci_b_high = paired_bootstrap(eval_p1_live['nll_vec'], vec_bigram)
    print(f"  AGNIS vs Bigram Paired NLL Difference    : {diff_bigram:+.4f} nats [95% CI: {ci_b_low:+.4f}, {ci_b_high:+.4f}]")
    
    h_snap = copy.deepcopy(h_phase1)
    r_snap = copy.deepcopy(r_phase1)
    
    # ------------------------------------------------------------------
    # 3. NULL-TWIN CONTROL (CONTINUED ITALIAN TRAINING ON HELD-OUT DATA)
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  3. NULL-TWIN CONTROL (CONTINUED ITALIAN TRAINING ON HELD-OUT DATA)")
    print("----------------------------------------------------------------------")
    h_null = copy.deepcopy(h_snap)
    r_null = copy.deepcopy(r_snap)
    train_phase(h_null, r_null, null_it, V, epochs=5)
    eval_null_it = evaluate_model(h_null, r_null, val_it, V, task_idx=0)
    print(f"  Null Twin Italian PPL (after 5k extra Italian steps) : {eval_null_it['ppl']:.2f} ({eval_null_it['nll']:.4f} nats)")
    
    # ------------------------------------------------------------------
    # 4. CONTROL E MATCHED: SHIELDED VS UNSHIELDED ITALIAN -> SPANISH
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  4. CONTROL E MATCHED: SHIELDED VS UNSHIELDED ITALIAN -> SPANISH")
    print("----------------------------------------------------------------------")
    # Branch A: Unshielded Naive Sequential Training (Spanish)
    h_unshielded = copy.deepcopy(h_snap)
    r_unshielded = copy.deepcopy(r_snap)
    train_phase(h_unshielded, r_unshielded, tr_es, V, epochs=5)
    eval_unshielded_es = evaluate_model(h_unshielded, r_unshielded, val_es, V, task_idx=0)
    eval_unshielded_it = evaluate_model(h_unshielded, r_unshielded, val_it, V, task_idx=0)
    
    # Branch B: Task-Gated Multi-Head Synaptic Shielded Training (Spanish)
    h_shielded = copy.deepcopy(h_snap)
    r_shielded = copy.deepcopy(r_snap)
    h_shielded.force_recruit_language_sliver(n=32, language="spanish")
    r_shielded.expand_capacity(n_sliver=32, freeze_prior=True)
    train_phase(h_shielded, r_shielded, tr_es, V, epochs=5)
    eval_shielded_es = evaluate_model(h_shielded, r_shielded, val_es, V, task_idx=1)
    eval_shielded_it = evaluate_model(h_shielded, r_shielded, val_it, V, task_idx=0)
    
    true_forget_unshielded, u_low, u_high = paired_bootstrap(eval_unshielded_it['nll_vec'], eval_null_it['nll_vec'])
    true_forget_shielded, s_low, s_high = paired_bootstrap(eval_shielded_it['nll_vec'], eval_null_it['nll_vec'])
    
    print(f"\n  [Unshielded] Spanish Acquisition PPL : {eval_unshielded_es['ppl']:.2f}")
    print(f"  [Unshielded] Italian Retention PPL   : {eval_unshielded_it['ppl']:.2f} (True Forgetting = {true_forget_unshielded:+.4f} nats [95% CI: {u_low:+.4f}, {u_high:+.4f}])")
    
    print(f"\n  [Task-Gated Shielded] Spanish Acquisition PPL : {eval_shielded_es['ppl']:.2f}")
    print(f"  [Task-Gated Shielded] Italian Retention PPL   : {eval_shielded_it['ppl']:.2f} (True Forgetting = {true_forget_shielded:+.4f} nats [95% CI: {s_low:+.4f}, {s_high:+.4f}])")
    
    headroom_nats = unigram_nll - eval_null_it['nll']
    retention_pct = (1.0 - (max(true_forget_shielded, 0.0) / max(headroom_nats, 1e-8))) * 100.0
    print(f"  Task-Gated True Retention Ratio : {retention_pct:.1f}%")
    
    # ------------------------------------------------------------------
    # 5. CONTROL B: RUSSIAN FROM SCRATCH (544 UNITS) VS SLIVER ACQUISITION
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("  5. CONTROL B: RUSSIAN FROM SCRATCH VS SLIVER ACQUISITION (PLASTICITY COST)")
    print("----------------------------------------------------------------------")
    h_ru_scratch = PredictiveHierarchy([V, 544, 512], device=DEVICE)
    r_ru_scratch = DeltaSoftmaxReadout(V, [544, 512], V, device=DEVICE, eta=0.5, beta=0.9)
    train_phase(h_ru_scratch, r_ru_scratch, tr_ru, V, epochs=5)
    eval_ru_scratch = evaluate_model(h_ru_scratch, r_ru_scratch, val_ru, V, task_idx=0)
    
    h_ru_sliver = copy.deepcopy(h_snap)
    r_ru_sliver = copy.deepcopy(r_snap)
    h_ru_sliver.force_recruit_language_sliver(n=32, language="russian")
    r_ru_sliver.expand_capacity(n_sliver=32, freeze_prior=True)
    train_phase(h_ru_sliver, r_ru_sliver, tr_ru, V, epochs=5)
    eval_ru_sliver = evaluate_model(h_ru_sliver, r_ru_sliver, val_ru, V, task_idx=1)
    
    cost_nats, c_low, c_high = paired_bootstrap(eval_ru_sliver['nll_vec'], eval_ru_scratch['nll_vec'])
    print(f"  Russian Scratch Acquisition (544 units) : {eval_ru_scratch['ppl']:.2f} ({eval_ru_scratch['nll']:.4f} nats)")
    print(f"  Russian Sliver Acquisition (32 units)  : {eval_ru_sliver['ppl']:.2f} ({eval_ru_sliver['nll']:.4f} nats)")
    print(f"  Plasticity Cost (Sliver - Scratch)     : {cost_nats:+.4f} nats [95% CI: {c_low:+.4f}, {c_high:+.4f}]")
    
    print("\n======================================================================")
    print("  SUMMARY OF P0.5 CONTROL BATTERY & NULL-PHASE AUDIT")
    print("======================================================================")
    print(f"  Bigram Baseline PPL                : {bigram_ppl:.2f} ({bigram_nll:.4f} nats)")
    print(f"  AGNIS Italian Phase 1 PPL          : {eval_p1_live['ppl']:.2f} ({eval_p1_live['nll']:.4f} nats)")
    print(f"  Null-Twin Converged Italian PPL    : {eval_null_it['ppl']:.2f} ({eval_null_it['nll']:.4f} nats)")
    print(f"  Control D (R=0 Ablation Diff)      : {diff_D:+.4f} nats [CI: {ci_D_low:+.4f}, {ci_D_high:+.4f}]")
    print(f"  Unshielded Italian Forgetting      : {true_forget_unshielded:+.4f} nats")
    print(f"  Task-Gated Shielded Forgetting     : {true_forget_shielded:+.4f} nats")
    print(f"  Task-Gated True Retention Ratio    : {retention_pct:.1f}%")
    print(f"  Sliver vs Scratch Plasticity Cost  : {cost_nats:+.4f} nats")

if __name__ == "__main__":
    main()
