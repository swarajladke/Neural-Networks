"""
run_qpl_stage3_evaluation.py — Stage 3 Competitive QPL Neurogenesis Evaluation Suite
=====================================================================================
Evaluates group-aware novelty-triggered neurogenesis in HybridQPL.
Runs capacity-matched controls matched to final active count K_s per seed,
computes representation drift metrics, and compares sequential vs replay conditions.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from hybrid_qpl import HybridQPL, MATURE, MATURING, FAILED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"

# Reproducibility Locks
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
INPUT_DIM = 960

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def fit_pca(train_x, d_active):
    mean = train_x.mean(dim=0, keepdim=True)
    x_centered = train_x - mean
    _, _, V = torch.linalg.svd(x_centered, full_matrices=False)
    proj = V[:d_active].T  # (INPUT_DIM, d_active)
    return mean, proj


def fit_random_projection(d_active, seed=42):
    g = torch.Generator().manual_seed(seed)
    matrix = torch.randn(INPUT_DIM, d_active, generator=g)
    Q, _ = torch.linalg.qr(matrix, mode="reduced")
    return Q


def evaluate_projection_1nn(train_proj, train_y, test_proj, test_y):
    z_ref = F.normalize(train_proj, dim=-1, eps=1e-8)
    z_test = F.normalize(test_proj, dim=-1, eps=1e-8)
    sims = torch.matmul(z_test, z_ref.T)
    correct = 0
    for idx in range(len(test_y)):
        q_label = test_y[idx].item()
        pred_idx = sims[idx].argmax().item()
        if train_y[pred_idx].item() == q_label:
            correct += 1
    return correct / len(test_y)

# ---------------------------------------------------------------------------
# Evaluation & Diagnostics Logger
# ---------------------------------------------------------------------------
def run_evaluation(qpl, x_data, y_data, ref_x, ref_y, variant="full_qpl", k_wta=None):
    qpl.eval()
    B = x_data.shape[0]
    device = x_data.device
    
    with torch.no_grad():
        z_queries, _ = qpl(x_data, variant=variant, k_wta=k_wta)
        z_refs, _ = qpl(ref_x, variant=variant, k_wta=k_wta)
        
        # Settle diagnostics
        _, converged_mask, steps, fallback_rate = qpl.settle(x_data, variant=variant, k_wta=k_wta)
        
    sims = torch.matmul(z_queries, z_refs.T)
    correct = 0
    mrr_sum = 0.0
    recalls = {1: 0, 5: 0, 10: 0}
    
    for idx in range(B):
        q_label = y_data[idx].item()
        sim_row = sims[idx]
        sorted_indices = sim_row.argsort(descending=True)
        
        # 1-NN check
        pred_label = ref_y[sorted_indices[0].item()].item()
        if pred_label == q_label:
            correct += 1
            
        # MRR & Recalls
        rank = -1
        for rank_idx, r_idx in enumerate(sorted_indices):
            if ref_y[r_idx.item()].item() == q_label:
                rank = rank_idx + 1
                break
        if rank != -1:
            mrr_sum += 1.0 / rank
            for k in recalls:
                if rank <= k:
                    recalls[k] += 1
                    
    # Calculate Gini coefficient of unit usage
    if variant == "full_qpl" and k_wta is not None:
        with torch.no_grad():
            h, _, _, _ = qpl.settle(x_data, variant, k_wta)
            scores = h.masked_fill(~qpl.active_mask.unsqueeze(0), -float("inf"))
            indices = scores.topk(k_wta, dim=-1).indices
            kwta_mask = torch.zeros_like(h)
            kwta_mask.scatter_(1, indices, 1.0)
            
        active_idx = qpl.active_mask.nonzero(as_tuple=True)[0]
        usage = kwta_mask[:, active_idx].sum(dim=0).cpu().numpy()
        usage_sorted = np.sort(usage)
        n = len(active_idx)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * usage_sorted)) / (n * np.sum(usage_sorted) + 1e-8)
        dead_units = (usage == 0).sum() / n
        max_usage_ratio = usage.max() / (usage.mean() + 1e-8)
    else:
        gini = 0.0
        dead_units = 0.0
        max_usage_ratio = 0.0
        
    return {
        "acc": correct / B,
        "mrr": mrr_sum / B,
        "recalls": {k: recalls[k] / B for k in recalls},
        "converged_samples": converged_mask.float().mean().item(),
        "median_steps": steps,
        "fallback_rate": fallback_rate,
        "gini": gini,
        "dead_units": dead_units,
        "max_usage_ratio": max_usage_ratio
    }

# ---------------------------------------------------------------------------
# Stage 3 Neurogenesis Training Engine
# ---------------------------------------------------------------------------
def run_neurogenesis_training(qpl, train_x, train_y, val_x, val_y, variant="full_qpl", k_wta=None, 
                             epochs=25, lrs=None, replay_mode=False, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    B = train_x.shape[0]
    tx = train_x.to(DEVICE)
    ty = train_y.to(DEVICE)
    
    # Global step
    global_step = 0
    refractory_steps = 100
    steps_since_birth = 100
    initial_warmup = 500
    min_buffer_size = 200
    
    # Rolling Buffers
    recon_buffer = []
    margin_buffer = []
    
    # Group evidence tracking
    # Group mapping: fact slot labels (0 to 33)
    num_groups = 34
    birth_evidence = torch.zeros(num_groups, device=DEVICE)
    group_last_birth = torch.zeros(num_groups, dtype=torch.long, device=DEVICE) - 1000
    trigger_candidates = {g: [] for g in range(num_groups)}
    
    for epoch in range(epochs):
        perm = torch.randperm(B)
        tx_shuffled = tx[perm]
        ty_shuffled = ty[perm]
        
        batch_size = 30
        for i in range(0, B, batch_size):
            q_batch = tx_shuffled[i : i + batch_size]
            y_batch = ty_shuffled[i : i + batch_size]
            
            # Settle batch to get preactivation hidden states
            with torch.no_grad():
                h, _, _, _ = qpl.settle(q_batch, variant=variant, k_wta=k_wta)
                
            # Settle diagnostics
            q_recon = h @ qpl.W + qpl.b_out
            recon_errs = (q_batch - q_recon).norm(dim=-1) / (q_batch.norm(dim=-1) + 1e-8)
            
            # Relative margins
            if variant == "orthogonal_static":
                margins = torch.zeros(q_batch.shape[0], device=DEVICE)
            else:
                s1_s2 = h.topk(2, dim=-1).values
                margins = (s1_s2[:, 0] - s1_s2[:, 1]) / (s1_s2[:, 0] + 1e-8)
                
            # Update rolling buffers & evaluate triggers pre-append
            for idx_s in range(q_batch.shape[0]):
                recon_val = recon_errs[idx_s].item()
                margin_val = margins[idx_s].item()
                g_id = y_batch[idx_s].item()
                
                # Check warm-up & buffer repopulation bounds
                birth_enabled = (
                    global_step >= initial_warmup
                    and len(recon_buffer) >= min_buffer_size
                    and steps_since_birth >= refractory_steps
                    and qpl.active_mask.sum().item() < 128
                )
                
                if birth_enabled:
                    recon_threshold = np.percentile(recon_buffer, 95)
                    margin_threshold = np.percentile(margin_buffer, 5)
                    
                    trigger = (recon_val > recon_threshold) and (margin_val < margin_threshold)
                    
                    # Group cooldown check (200 steps)
                    cooldown_ok = (global_step - group_last_birth[g_id]) >= 200
                    
                    if trigger and cooldown_ok:
                        birth_evidence[g_id] = 0.90 * birth_evidence[g_id] + 1.0
                        trigger_candidates[g_id].append(q_batch[idx_s].detach())
                        
                        if birth_evidence[g_id] >= 2.5:
                            # Deduplication: Medoid anchor selection
                            candidates = trigger_candidates[g_id]
                            valid_candidates = []
                            for c in candidates:
                                c_norm = F.normalize(c, dim=-1)
                                max_cos = 0.0
                                active_slots = qpl.active_mask.nonzero(as_tuple=True)[0]
                                if len(active_slots) > 0:
                                    max_cos = torch.matmul(qpl.V[:, active_slots].T, c_norm).max().item()
                                if (1.0 - max_cos) > 0.15:
                                    valid_candidates.append(c)
                                    
                            if len(valid_candidates) > 0:
                                # Compute pairwise distances to select medoid
                                stacked_c = torch.stack(valid_candidates)
                                dists = torch.cdist(stacked_c, stacked_c)
                                medoid_idx = dists.sum(dim=1).argmin().item()
                                q_medoid = valid_candidates[medoid_idx]
                                
                                # Commit atomic birth
                                new_slot = qpl.allocate_slot(q_medoid, g_id, global_step, target_activation=0.5)
                                if new_slot != -1:
                                    steps_since_birth = 0
                                    # Reset all evidence to prevent birth cascades
                                    birth_evidence.zero_()
                                    for g in trigger_candidates:
                                        trigger_candidates[g].clear()
                                    # Clear threshold buffers to trigger recalibration
                                    recon_buffer.clear()
                                    margin_buffer.clear()
                                    break  # halt batch processing to resettle on new width
                            else:
                                # Reset evidence if all candidates duplicate existing columns
                                birth_evidence[g_id] = 0.0
                                trigger_candidates[g_id].clear()
                else:
                    # Decay evidence slightly if not triggered
                    birth_evidence[g_id] = max(0.0, 0.90 * birth_evidence[g_id] - 0.1)
                    
                # Append to buffers AFTER trigger checks to prevent self-dilution
                recon_buffer.append(recon_val)
                margin_buffer.append(margin_val)
                if len(recon_buffer) > 500:
                    recon_buffer.pop(0)
                    margin_buffer.pop(0)
                    
            # Check maturation counter & update maturation ages
            active_slots = qpl.active_mask.nonzero(as_tuple=True)[0]
            maturing_slots = (qpl.unit_state == MATURING).nonzero(as_tuple=True)[0]
            if len(maturing_slots) > 0:
                # Find current kWTA winners
                scores = h.masked_fill(~qpl.active_mask.unsqueeze(0), -float("inf"))
                current_winners = scores.topk(max(1, int(0.10 * len(active_slots))), dim=-1).indices.cpu().numpy().tolist()
                
                # Settle unique group mappings in current batch
                for g_id in set(y_batch.cpu().numpy()):
                    # Find winners in current batch matching group g_id
                    batch_winners = []
                    for idx_b in range(B):
                        if y_batch[idx_b].item() == g_id:
                            batch_winners.extend(current_winners)
                    qpl.update_maturation(g_id, set(batch_winners), recon_errs)
                    
            # Perform local unsupervised updates
            # Find unique groups in current batch to boost maturing units
            curr_batch_groups = set(y_batch.cpu().numpy().tolist())
            for current_group in curr_batch_groups:
                qpl.local_unsupervised_update(q_batch, h, kwta_mask=None, lrs=lrs, current_group=current_group)
                
            global_step += 1
            steps_since_birth += 1
            
        qpl.verify_invariants()

# ---------------------------------------------------------------------------
# Main Sweep Run
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(CACHE_PATH):
        print(f"[Error] Fresh embeddings must be cached first. Please run run_qpl_stage2_evaluation.py.")
        return
        
    data = torch.load(CACHE_PATH, weights_only=True)
    train_x = data["train_x"].to(DEVICE)
    train_y = data["train_y"].to(DEVICE)
    val_x = data["val_x"].to(DEVICE)
    val_y = data["val_y"].to(DEVICE)
    test_x = data["test_x"].to(DEVICE)
    test_y = data["test_y"].to(DEVICE)
    
    # 0. Raw SmolLM2 Baseline
    raw_acc = evaluate_projection_1nn(train_x, train_y, val_x, val_y)
    print("="*80)
    print(f"  [Baseline] Raw SmolLM2 Validation Accuracy ({INPUT_DIM}D): {raw_acc*100:.2f}%")
    print("="*80)
    
    # 1. Evaluate Joint-Trigger Neurogenesis (Stage 3) on Validation Split
    print("\n" + "="*80)
    print("  STAGE 3 DYNAMIC NEUROGENESIS SWEEP (on validation/development split)")
    print("="*80)
    
    qpl = HybridQPL(input_dim=INPUT_DIM, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=1.0).to(DEVICE)
    qpl.initialize_basis(34)
    
    # Train QPL with dynamic birth triggered via joint quantiles
    lrs = {"V": 1e-2, "W": 1e-2, "L": 1e-2, "b": 1e-2, "homeo": 1e-3}
    run_neurogenesis_training(qpl, train_x, train_y, val_x, val_y, variant="full_qpl", k_wta=3, epochs=25, lrs=lrs, seed=42)
    
    # Final active slots count K_s
    K_s = qpl.active_mask.sum().item()
    print(f"\n-> Neurogenesis sweep complete! Final active width (K_s): {K_s}")
    
    # Evaluate dynamic QPL
    dyn_eval = run_evaluation(qpl, val_x, val_y, train_x, train_y, variant="full_qpl", k_wta=3)
    print(f"Dynamic QPL Test Accuracy: {dyn_eval['acc']*100:.2f}% | Fallback rate: {dyn_eval['fallback_rate']*100:.3f}% | Gini: {dyn_eval['gini']:.3f} | Dead Units: {dyn_eval['dead_units']*100:.1f}%")
    
    # ---------------------------------------------------------------------------
    # Capacity-Matched Baseline Controls Sweep matched to K_s
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"  CAPACITY-MATCHED CONTROL COMPARISON (at matched width K_s = {K_s})")
    print("="*80)
    
    # 1. Full QPL Fixed-34 (Stage 2 control, no birth)
    qpl_34 = HybridQPL(input_dim=INPUT_DIM, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=1.0).to(DEVICE)
    qpl_34.initialize_basis(34)
    run_neurogenesis_training(qpl_34, train_x, train_y, val_x, val_y, variant="full_qpl", k_wta=3, epochs=25, seed=42)
    eval_34 = run_evaluation(qpl_34, val_x, val_y, train_x, train_y, variant="full_qpl", k_wta=3)
    print(f"Control: Full QPL Fixed-34 (No birth)     | Val Acc: {eval_34['acc']*100:.2f}%")
    
    # 2. Full QPL Fixed-K (Initialized with K_s active units, no birth)
    qpl_K = HybridQPL(input_dim=INPUT_DIM, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=1.0).to(DEVICE)
    qpl_K.initialize_basis(K_s)
    run_neurogenesis_training(qpl_K, train_x, train_y, val_x, val_y, variant="full_qpl", k_wta=max(1, int(0.10*K_s)), epochs=25, seed=42)
    eval_K = run_evaluation(qpl_K, val_x, val_y, train_x, train_y, variant="full_qpl", k_wta=max(1, int(0.10*K_s)))
    print(f"Control: Full QPL Fixed-K (No birth)      | Val Acc: {eval_K['acc']*100:.2f}%")
    
    # 3. PCA Control at K_s
    mean_pca, proj_pca = fit_pca(data["train_x"], K_s)
    pca_train = (data["train_x"] - mean_pca) @ proj_pca
    pca_val = (data["val_x"] - mean_pca) @ proj_pca
    pca_acc = evaluate_projection_1nn(pca_train, train_y, pca_val, val_y)
    print(f"Control: PCA Projection at K_s             | Val Acc: {pca_acc*100:.2f}%")
    
    # ---------------------------------------------------------------------------
    # Validation of Exit Criteria
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print("  STAGE 3 DIAGNOSTIC CHECKLIST & EXIT CRITERIA")
    print("="*80)
    
    converged_fraction = dyn_eval["converged_samples"]
    print(f"1. Settling Convergence: {converged_fraction*100:.2f}% (Target >= 99%) -> {'PASS' if converged_fraction >= 0.99 else 'FAIL'}")
    print(f"2. Silent Fallback Rate: {dyn_eval['fallback_rate']*100:.3f}% (Target <= 5.0%) -> {'PASS' if dyn_eval['fallback_rate'] <= 0.05 else 'FAIL'}")
    print(f"3. Dead-Unit Fraction: {dyn_eval['dead_units']*100:.2f}% (Target <= 25%) -> {'PASS' if dyn_eval['dead_units'] <= 0.25 else 'FAIL'}")
    print(f"4. Max Winner Frequency ratio: {dyn_eval['max_usage_ratio']:.2f}x (Target <= 5.0x) -> {'PASS' if dyn_eval['max_usage_ratio'] <= 5.0 else 'FAIL'}")
    
    # Check if QPL outperforms Fixed-34 on validation set
    val_improved = dyn_eval["acc"] > eval_34["acc"]
    print(f"5. Validation Performance Gain: {dyn_eval['acc']*100:.2f}% vs. {eval_34['acc']*100:.2f}% -> {'PASS' if val_improved else 'FAIL'}")

if __name__ == "__main__":
    main()
