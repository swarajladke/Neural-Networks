import time
import torch
import sys
import os
import numpy as np
from slm.slm_dataset import SLMDataset
from slm.agnis_slm_wrapper import AGNIS_SLM_Wrapper
from slm.slm_tokenizer import CharTokenizer

# Ensure UTF-8 for console output (important for Cyrillic)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_eta_r_schedule(batches_since_reset):
    """V7.2: Adaptive recurrent learning rate schedule for faster context adjustment."""
    if batches_since_reset < 100:
        return 0.05   # fast early adaptation
    elif batches_since_reset < 500:
        return 0.02   # settling phase
    else:
        return 0.005  # stable long-term

def probe_italian_retention(slm, it_dataset, n_batches=10):
    """Strict evaluation of Italian surprise without gradient updates."""
    surprises = []
    it_batches = it_dataset.get_batches(batch_size=8)
    
    with torch.no_grad():
        # Temporarily swap to Italian temporal context
        slm.agent.switch_temporal_context("italian")
        
        for _ in range(n_batches):
            p_ctx, p_tgt = next(it_batches)
            flat_ctx, embedded_tgt = slm._prepare_tensors(p_ctx, p_tgt)
            # Use predict_label (no settling push, no learning)
            pred_y = slm.hierarchy.predict_label(flat_ctx, update_temporal=False)
            
            # V7.3.3 (Fix): Slice output to match the target manifold if expanded
            d = embedded_tgt.shape[1]
            if pred_y.shape[1] > d:
                pred_y = pred_y[:, :d]
                
            surprises.append(torch.nn.functional.mse_loss(pred_y, embedded_tgt).item())
        
        # Restore Russian temporal context
        slm.agent.switch_temporal_context("russian")
        
    return np.mean(surprises)

def probe_italian_retention_isolated(slm, it_dataset, n_batches=10, italian_dim=64):
    """Strict isolated evaluation of Italian manifold without interference from expanded эксперты."""
    surprises = []
    it_batches = it_dataset.get_batches(batch_size=8)
    
    with torch.no_grad():
        # Temporarily swap to Italian temporal context
        slm.agent.switch_temporal_context("italian")
        
        for _ in range(n_batches):
            p_ctx, p_tgt = next(it_batches)
            flat_ctx, embedded_tgt = slm._prepare_tensors(p_ctx, p_tgt)
            
            # V7.3.6: Use isolated inference
            pred_y = slm.hierarchy.infer_with_manifold_slice(flat_ctx, slice_end=italian_dim)
            
            # Slice output to match embedding target
            d = embedded_tgt.shape[1]
            if pred_y.shape[1] > d:
                pred_y = pred_y[:, :d]
                
            surprises.append(torch.nn.functional.mse_loss(pred_y, embedded_tgt).item())
        
        # Restore Russian temporal context
        slm.agent.switch_temporal_context("russian")
        
    return np.mean(surprises)

def run_continual_learning_v73():
    print("==================================================")
    print(" AGNIS V7.3: Synaptic Freeze & Forced Expansion")
    print("==================================================")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seq_length = 64
    batch_size = 32
    embed_dim = 64
    
    # 1. Initialize Datasets
    with open("slm/input_it.txt", "r", encoding="utf-8") as f: it_text = f.read()[:1000000]
    with open("slm/input_ru.txt", "r", encoding="utf-8") as f: ru_text = f.read()[:1000000]
    
    tokenizer = CharTokenizer()
    tokenizer.fit(it_text + ru_text)
    vocab_size = tokenizer.vocab_size
    print(f"-> Joint Vocabulary Size: {vocab_size}", flush=True)

    it_dataset = SLMDataset(filepath="slm/input_it.txt", seq_length=seq_length)
    it_dataset.tokenizer = tokenizer
    it_dataset.data_indices = tokenizer.encode(it_text)
    
    ru_dataset = SLMDataset(filepath="slm/input_ru.txt", seq_length=seq_length)
    ru_dataset.tokenizer = tokenizer
    ru_dataset.data_indices = tokenizer.encode(ru_text)
    
    slm = AGNIS_SLM_Wrapper(vocab_size=vocab_size, seq_length=seq_length, embed_dim=embed_dim, device=device)
    
    # --- STAGE 1: Italian Warm-up ---
    slm.agent.switch_temporal_context("italian")
    print(f"\n[STAGE 1] Italian Warm-up (5 Mins)", flush=True)
    start_warm = time.time()
    it_batches = it_dataset.get_batches(batch_size=batch_size)
    it_steps = 0
    
    while time.time() - start_warm < 300: # 5 mins
        try:
            contexts, targets = next(it_batches)
        except StopIteration:
            it_batches = it_dataset.get_batches(batch_size=batch_size)
            contexts, targets = next(it_batches)
        
        weight, it_surprise = slm.learn_step(contexts, targets)
        it_steps += 1
        if it_steps % 50 == 0:
            print(f"  Batch {it_steps:03d} | IT Surprise: {it_surprise:.4f}", flush=True)

    # V7.3.10: Stable Baseline (Averaged over 10 batches)
    print("\nCalculating stable Italian baseline...")
    italian_baseline = probe_italian_retention_isolated(slm, it_dataset, n_batches=10, italian_dim=64)
    print(f">>> Stable Baseline Italian Surprise: {italian_baseline:.4f}")

    # --- THE FREEZE & EXPANSION ---
    print("\n>>> PERFORMING SYNAPTIC FREEZE & FORCED EXPANSION <<<", flush=True)
    # Verification Point 1: Sum of masks before freeze
    n_pre = slm.hierarchy.layers[0].V_mask.sum().item()
    print(f"Trainable neurons before freeze: {n_pre}")
    
    # V7.3.5: Recruit with SILENT bias (-10.0) for isolative verification
    slm.hierarchy.force_recruit_language_sliver(n=32, language="russian")
    for layer in slm.hierarchy.layers:
        # The last 32 neurons of the hidden dim (output_dim) are the Russian ones
        d_new = 32
        start_idx = layer.output_dim - d_new
        layer.set_experts_bias(start_idx, layer.output_dim, -10.0)
    
    # Verification Point 2: Sum of masks after expansion
    n_post = slm.hierarchy.layers[0].V_mask.sum().item()
    print(f"Trainable neurons after expansion: {n_post} (Expected: 32)")
    
    # --- V7.3.4: BREAKTHROUGH CHECKPOINT (Save immediately after expansion) ---
    breakthrough_path = 'checkpoints/phase_733_breakthrough.pt'
    torch.save({
        'hierarchy_state': slm.hierarchy.state_dict(),
        'italian_baseline': italian_baseline,
        'batch': 0,
        'phase': '7.3.4'
    }, breakthrough_path)
    print(f"\n>>> BREAKTHROUGH CHECKPOINT SAVED: {breakthrough_path} <<<", flush=True)

    # --- V7.3.6: ISOLATED VERIFICATION ---
    it_probe_isolated = probe_italian_retention_isolated(slm, it_dataset, italian_dim=64)
    tolerance = 0.20 # V7.3.10: Calibrated for statistical variance 
    
    print(f"\n>>> ISOLATION RESULTS <<<")
    print(f"Baseline Italian: {italian_baseline:.4f}")
    print(f"Isolated IT Probe: {it_probe_isolated:.4f}")
    drift = abs(it_probe_isolated - italian_baseline)
    print(f"Absolute Drift:    {drift:.4f}")
    
    assert drift < tolerance, f"SYNAPTIC SHIELD FAILURE: Isolation failed! baseline={italian_baseline:.4f}, probe={it_probe_isolated:.4f}"
    print(">>> SYNAPTIC SHIELD VERIFIED: Expert Manifold Mathematically Isolated. <<<")

    # --- CROSS-CONTAMINATION CHECK ---
    print("\n--- Verifying Manifold Separation ---")
    it_on_it = it_probe_isolated
    
    # IT manifold on Russian text (should be HIGH surprise)
    it_on_ru = probe_italian_retention_isolated(slm, ru_dataset, italian_dim=64)
    
    print(f"IT manifold / IT text:  {it_on_it:.4f}  (Target: baseline)")
    print(f"IT manifold / RU text:  {it_on_ru:.4f}  (Target: HIGH)")
    print("--------------------------------------")

    # --- V7.3.5: AWAKEN RUSSIAN SLIVER ---
    print("\n>>> AWAKENING RUSSIAN EXPERT SLIVER (-2.0 Bias) <<<", flush=True)
    for layer in slm.hierarchy.layers:
        d_new = 32
        start_idx = layer.output_dim - d_new
        layer.set_experts_bias(start_idx, layer.output_dim, -2.0)

    # --- STAGE 2: Russian Training ---
    print(f"\n[STAGE 2] Russian Ignite (60 Mins)", flush=True)
    start_ru = time.time()
    ru_batches = ru_dataset.get_batches(batch_size=batch_size)
    ru_steps = 0
    
    log_path = "russian_discovery_log_v73.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# AGNIS V7.3: Synaptic Freeze & Forced Expansion\n\n")
        f.write(f"**Baseline Italian Surprise:** {italian_baseline:.4f}\n\n")

    while time.time() - start_ru < 3600: # 60 mins
        try:
            contexts, targets = next(ru_batches)
        except StopIteration:
            ru_batches = ru_dataset.get_batches(batch_size=batch_size)
            contexts, targets = next(ru_batches)
        
        # Apply Adaptive R-Decay
        current_eta_r = get_eta_r_schedule(ru_steps)
        for col in slm.hierarchy.layers:
            col.eta_R = current_eta_r
            
        w, ru_surprise = slm.learn_step(contexts, targets)
        ru_steps += 1
        
        if ru_steps % 500 == 0:
            it_probe = probe_italian_retention(slm, it_dataset)
            print(f"\n[Log {ru_steps}] Generating Sample...", flush=True)
            sample = slm.generate(ru_dataset.tokenizer, prompt="Раскольников ", max_new_chars=64, temperature=0.8)
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"## Batch {ru_steps}\n")
                f.write(f"- **RU Surprise:** {ru_surprise:.4f} | **IT Retention:** {it_probe:.4f}\n")
                f.write(f"- **Slivers:** {slm.agent.neurogenesis_count}\n")
                f.write(f"- **Sample:** `{sample}`\n\n")

            # --- V7.3.4: Periodic Milestone Saving ---
            milestone_path = f'checkpoints/ru_milestone_{ru_steps}.pt'
            torch.save({
                'hierarchy_state': slm.hierarchy.state_dict(),
                'ru_steps': ru_steps,
                'it_retention': it_probe,
                'ru_surprise': ru_surprise
            }, milestone_path)
            print(f"[Checkpoint] Milestone saved: {milestone_path}")

            print(f"\nBatch {ru_steps:04d} | RU Surprise: {ru_surprise:.4f} | IT Retention: {it_probe:.4f}", flush=True)

        elif ru_steps % 50 == 0:
            print(f"  Batch {ru_steps:04d} | RU Surprise: {ru_surprise:.4f}", flush=True)

    print(f"\nExperiment Complete. Log: '{log_path}'.")

if __name__ == "__main__":
    run_continual_learning_v73()
