"""
run_qpl_stage4_evaluation.py — Stage 4 Contrastive Hebbian Learning (CHL) Integration
======================================================================================
Implements Contrastive Hebbian Learning (CHL) using positive/negative phases,
relational distillation, and evaluates whether CHL solves the sparse representation collapse.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hybrid_qpl import HybridQPL, MATURE, MATURING

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"
INPUT_DIM = 960

# ---------------------------------------------------------------------------
# Balanced Batch Sampler
# ---------------------------------------------------------------------------
class BalancedBatchSampler:
    """Samples batches containing exactly 2 paraphrases per selected fact group."""
    def __init__(self, x, y, num_groups=34):
        self.x = x
        self.y = y
        self.num_groups = num_groups
        
        # Group indices by label
        self.label_to_indices = {g: [] for g in range(num_groups)}
        for idx, label in enumerate(y):
            self.label_to_indices[label.item()].append(idx)
            
    def sample_batch(self, batch_groups=16):
        """Samples a batch containing batch_groups * 2 samples."""
        selected_groups = random.sample(range(self.num_groups), batch_groups)
        indices = []
        for g in selected_groups:
            g_indices = self.label_to_indices[g]
            sampled = random.sample(g_indices, min(2, len(g_indices)))
            indices.extend(sampled)
            
        random.shuffle(indices)
        return self.x[indices], self.y[indices]

# ---------------------------------------------------------------------------
# Stage 4 CHL Training Engine
# ---------------------------------------------------------------------------
def train_qpl_chl(qpl, train_x, train_y, val_x, val_y, epochs=45, initial_lr=2e-1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    sampler = BalancedBatchSampler(train_x, train_y, num_groups=34)
    best_val_acc = 0.0
    
    # Store raw query text coordinates as anchors for distillation
    anchor_x = train_x.clone().detach()
    with torch.no_grad():
        z_anchor, _ = qpl(anchor_x, variant="full_qpl", k_wta=3)
    
    lr = initial_lr
    
    for epoch in range(epochs):
        qpl.train()
        
        # Run 35 batch updates per epoch
        for _ in range(35):
            # Sample contrastive batch
            q_batch, y_batch = sampler.sample_batch(batch_groups=18)
            q_batch = q_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            B = q_batch.shape[0]
            
            # --- Contrastive Hebbian Learning Updates ---
            with torch.no_grad():
                # 1. Negative Phase (Free-running settled state)
                h_neg, _, _, _ = qpl.settle(q_batch, variant="full_qpl", k_wta=3)
                
                # 2. Positive Phase (Teacher-clamped target state)
                h_pos = torch.zeros(B, qpl.output_dim, device=DEVICE)
                h_pos[range(B), y_batch] = 1.0
                
                # CHL local weight updates: delta V = q^T (h_pos - h_neg)
                dV = torch.matmul(q_batch.T, h_pos - h_neg) / B
                db_in = (h_pos - h_neg).mean(dim=0)
                
                # Project dV onto the tangent space of column vectors of V
                active_idx = qpl.active_mask.nonzero(as_tuple=True)[0]
                for j in active_idx:
                    g_v = dV[:, j]
                    v_col = qpl.V[:, j]
                    # Tangent projection: g - (v^T g) v
                    dV[:, j] = g_v - torch.dot(v_col, g_v) * v_col
                    
                # Apply updates in-place
                qpl.V.add_(lr * dV)
                qpl.b_in.add_(lr * db_in)
                
                # Ensure active columns remain normalized and inactive columns zeroed
                inactive_idx = (~qpl.active_mask).nonzero(as_tuple=True)[0]
                qpl.V[:, active_idx] = F.normalize(qpl.V[:, active_idx], dim=0, eps=1e-8)
                qpl.V[:, inactive_idx] = 0.0
                qpl.b_in[inactive_idx] = 0.0
                
                # --- Relational Distillation Replay ---
                # Sample replay anchors
                replay_idx = torch.randperm(anchor_x.shape[0])[:32]
                q_replay = anchor_x[replay_idx].to(DEVICE)
                h_rep, _, _, _ = qpl.settle(q_replay, variant="full_qpl", k_wta=3)
                z_replay = F.normalize(h_rep, dim=-1, eps=1e-8)
                
                # Relational alignment error
                S_s = torch.matmul(z_replay, z_replay.T)
                S_t = torch.matmul(z_anchor[replay_idx].to(DEVICE), z_anchor[replay_idx].to(DEVICE).T)
                distill_error = S_t - S_s
                
                # Distillation gradient
                dz = 2.0 * torch.matmul(distill_error, z_replay) / 32.0
                dV_distill = torch.matmul(q_replay.T, dz) / 32.0
                
                # Apply distillation step with low weight 0.05 to avoid blocking alignment
                for j in active_idx:
                    g_d = dV_distill[:, j]
                    v_c = qpl.V[:, j]
                    dV_distill[:, j] = g_d - torch.dot(v_c, g_d) * v_c
                    
                qpl.V.add_(lr * 0.05 * dV_distill)
                qpl.V[:, active_idx] = F.normalize(qpl.V[:, active_idx], dim=0, eps=1e-8)
                
        # Run local unsupervised updates on W and b_out
        with torch.no_grad():
            h_settled, _, _, _ = qpl.settle(train_x, variant="full_qpl", k_wta=3)
            qpl.local_unsupervised_update(train_x, h_settled, kwta_mask=None, current_group=None)
            
        # Log validation accuracy
        val_eval = evaluate_validation_split(qpl, train_x, train_y, val_x, val_y)
        if val_eval["acc"] > best_val_acc:
            best_val_acc = val_eval["acc"]
            
        print(f"Epoch {epoch+1:02d}/{epochs} | LR: {lr:.4f} | Val 1-NN Acc: {val_eval['acc']*100:.2f}% | Gini: {val_eval['gini']:.3f}")
        
        # Slower exponential decay to preserve gradient strength longer
        lr = lr * 0.98
        
    return best_val_acc

# ---------------------------------------------------------------------------
# Validation Evaluation
# ---------------------------------------------------------------------------
def evaluate_validation_split(qpl, train_x, train_y, val_x, val_y):
    qpl.eval()
    B = val_x.shape[0]
    
    with torch.no_grad():
        z_queries, _ = qpl(val_x, variant="full_qpl", k_wta=3)
        z_refs, _ = qpl(train_x, variant="full_qpl", k_wta=3)
        
    sims = torch.matmul(z_queries, z_refs.T)
    correct = 0
    for idx in range(B):
        q_label = val_y[idx].item()
        pred_idx = sims[idx].argmax().item()
        if train_y[pred_idx].item() == q_label:
            correct += 1
            
    # Compute usage Gini
    with torch.no_grad():
        h, _, _, _ = qpl.settle(val_x, variant="full_qpl", k_wta=3)
        scores = h.masked_fill(~qpl.active_mask.unsqueeze(0), -float("inf"))
        indices = scores.topk(3, dim=-1).indices
        kwta_mask = torch.zeros_like(h)
        kwta_mask.scatter_(1, indices, 1.0)
        
    active_idx = qpl.active_mask.nonzero(as_tuple=True)[0]
    usage = kwta_mask[:, active_idx].sum(dim=0).cpu().numpy()
    usage_sorted = np.sort(usage)
    n = len(active_idx)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * usage_sorted)) / (n * np.sum(usage_sorted) + 1e-8)
    
    return {
        "acc": correct / B,
        "gini": gini
    }

# ---------------------------------------------------------------------------
# Main Execution
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
    
    print("="*80)
    print("  STAGE 4 CONTRASTIVE HEBBIAN LEARNING (CHL) INTEGRATION")
    print("="*80)
    
    qpl = HybridQPL(input_dim=INPUT_DIM, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=1.0).to(DEVICE)
    qpl.initialize_basis(34)
    
    # Verify initial accuracy under sparse kWTA collapse (should be ~11.0%)
    init_eval = evaluate_validation_split(qpl, train_x, train_y, val_x, val_y)
    print(f"Initial Sparse kWTA Accuracy (Pre-CHL): {init_eval['acc']*100:.2f}%")
    
    # Train QPL using local CHL updates
    print("\nTraining QPL routing layer using local tangent-space contrastive updates...")
    best_acc = train_qpl_chl(qpl, train_x, train_y, val_x, val_y, epochs=45, initial_lr=2e-1, seed=42)
    
    print("\n" + "="*80)
    print("  STAGE 4 EXIT CHECKLIST & PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"1. Best CHL Validation Accuracy: {best_acc*100:.2f}%")
    
    # Target: recover >= 90% of backprop validation accuracy (97.7% * 0.90 = 87.93%)
    backprop_target = 87.93
    target_met = (best_acc * 100) >= backprop_target
    print(f"2. Recover >= 90% of Backprop Accuracy: {best_acc*100:.2f}% vs. Target {backprop_target:.2f}% -> {'PASS' if target_met else 'FAIL'}")

if __name__ == "__main__":
    main()
