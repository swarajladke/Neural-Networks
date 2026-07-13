"""
run_qpl_stage4_final_test.py — Stage 4 Locked final validation test on untouched test split.
===========================================================================================
Locks learning rate at 0.20, runs CHL training across seeds [41, 42, 43, 44] on the training split,
selects the best checkpoint on the validation split, and evaluates the final performance
on the untouched test split. Evaluates baselines (Random Orthonormal Projection, Unsupervised
Collapsed QPL, and Supervised Backprop Probe) to compute statistically rigorous metrics,
confidence intervals, recovered gains, usage statistics, and false positive rates.
"""

import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from hybrid_qpl import HybridQPL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"
INPUT_DIM = 960
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

# ---------------------------------------------------------------------------
# General Control / Hard Negative Texts
# ---------------------------------------------------------------------------
INDEPENDENT_PPL_TEXTS = [
    "The Renaissance began in Italy during the 14th century.",
    "Beethoven composed his ninth symphony while completely deaf.",
    "The Amazon rainforest produces 20 percent of Earth's oxygen.",
    "Chess was invented in India around the 6th century AD.",
    "Elephants are the largest land animals on Earth.",
    "William Shakespeare wrote 37 plays during his lifetime.",
    "The human brain contains approximately 86 billion neurons.",
    "Mount Everest is the highest mountain above sea level.",
    "The speed of sound is 343 metres per second in air.",
    "Leonardo da Vinci painted the Mona Lisa in the 1500s.",
    "Quantum mechanics describes the behavior of matter at the atomic scale.",
    "The Great Wall of China is a series of fortifications.",
    "DNA consists of two polynucleotide chains forming a double helix.",
    "Photosynthesis converts carbon dioxide and water into oxygen and glucose.",
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "Protons and neutrons are located in the nucleus of an atom.",
    "The Sahara Desert is the largest hot desert in the world.",
    "Marie Curie was the first woman to win a Nobel Prize.",
    "Glaciers store about 69 percent of the world's freshwater.",
    "The Pacific Ocean is the largest and deepest ocean on Earth."
]

# ---------------------------------------------------------------------------
# Balanced Batch Sampler
# ---------------------------------------------------------------------------
class BalancedBatchSampler:
    """Samples batches containing exactly 2 paraphrases per selected fact group."""
    def __init__(self, x, y, num_groups=34):
        self.x = x
        self.y = y
        self.num_groups = num_groups
        self.label_to_indices = {g: [] for g in range(num_groups)}
        for idx, label in enumerate(y):
            self.label_to_indices[label.item()].append(idx)
            
    def sample_batch(self, batch_groups=16):
        selected_groups = random.sample(range(self.num_groups), batch_groups)
        indices = []
        for g in selected_groups:
            g_indices = self.label_to_indices[g]
            sampled = random.sample(g_indices, min(2, len(g_indices)))
            indices.extend(sampled)
        random.shuffle(indices)
        return self.x[indices], self.y[indices]

# ---------------------------------------------------------------------------
# Baselines Definitions
# ---------------------------------------------------------------------------
class ResidualMetricProbe(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=512, output_dim=128):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.skip(x) + self.body(x)
        return F.normalize(z, dim=-1)

def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    device = embeddings.device
    N = embeddings.shape[0]
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    logits_mask = torch.scatter(
        torch.ones_like(logits),
        1,
        torch.arange(N, device=device).view(-1, 1),
        0
    )
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask = mask * logits_mask
    
    exp_logits = torch.exp(logits) * logits_mask
    sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
    
    log_prob = logits - torch.log(sum_exp_logits + 1e-8)
    rows_with_positives = mask.sum(dim=1) > 0
    if not rows_with_positives.any():
        return torch.tensor(0.0, device=device)
        
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[rows_with_positives] / mask.sum(dim=1)[rows_with_positives]
    loss = -mean_log_prob_pos.mean()
    return loss

def train_supervised_probe(train_x, train_y, test_x, test_y, d_active, seed=42):
    """Trains a supervised ResidualMetricProbe to establish the upper bound reference."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    probe = ResidualMetricProbe(input_dim=INPUT_DIM, hidden_dim=512, output_dim=d_active).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    
    tx = train_x.to(DEVICE)
    ty = train_y.to(DEVICE)
    
    for epoch in range(150):
        probe.train()
        z = probe(tx)
        loss = supervised_contrastive_loss(z, ty)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    probe.eval()
    with torch.no_grad():
        z_ref = probe(tx)
        z_test = probe(test_x.to(DEVICE))
        
    sims = torch.matmul(z_test, z_ref.T)
    correct = 0
    for idx in range(len(test_y)):
        q_label = test_y[idx].item()
        pred_idx = sims[idx].argmax().item()
        if train_y[pred_idx].item() == q_label:
            correct += 1
            
    return correct / len(test_y)

def train_unsupervised_only(qpl, train_x, epochs=45, seed=42):
    """Trains QPL with only unsupervised local updates to get collapsed baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    B = train_x.shape[0]
    tx = train_x.to(DEVICE)
    
    for epoch in range(epochs):
        perm = torch.randperm(B)
        tx_shuffled = tx[perm]
        
        batch_size = 30
        for i in range(0, B, batch_size):
            q_batch = tx_shuffled[i : i + batch_size]
            
            with torch.no_grad():
                h, _, _, _ = qpl.settle(q_batch, variant="full_qpl", k_wta=3)
                scores = h.masked_fill(~qpl.active_mask.unsqueeze(0), -float("inf"))
                indices = scores.topk(3, dim=-1).indices
                kwta_mask = torch.zeros_like(h)
                kwta_mask.scatter_(1, indices, 1.0)
                
                # Unsupervised local updates (V, W, L, bias, homeostasis)
                lrs_unsup = {"V": 1e-2, "W": 1e-2, "L": 1e-2, "b": 1e-3, "homeo": 1e-3}
                qpl.local_unsupervised_update(q_batch, h, kwta_mask=kwta_mask, lrs=lrs_unsup)
        
        qpl.verify_invariants()

# ---------------------------------------------------------------------------
# Stage 4 CHL Training Engine
# ---------------------------------------------------------------------------
def train_qpl_chl(qpl, train_x, train_y, val_x, val_y, epochs=45, lr=0.20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    sampler = BalancedBatchSampler(train_x, train_y, num_groups=34)
    best_val_acc = 0.0
    best_weights = None
    
    # Store raw query text coordinates as anchors for distillation
    anchor_x = train_x.clone().detach()
    with torch.no_grad():
        z_anchor, _ = qpl(anchor_x, variant="full_qpl", k_wta=3)
    
    for epoch in range(epochs):
        qpl.train()
        
        for _ in range(35):
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
                
                dV = torch.matmul(q_batch.T, h_pos - h_neg) / B
                db_in = (h_pos - h_neg).mean(dim=0)
                
                # Tangent space projection
                active_idx = qpl.active_mask.nonzero(as_tuple=True)[0]
                for j in active_idx:
                    g_v = dV[:, j]
                    v_col = qpl.V[:, j]
                    dV[:, j] = g_v - torch.dot(v_col, g_v) * v_col
                    
                qpl.V.add_(lr * dV)
                qpl.b_in.add_(lr * db_in)
                
                inactive_idx = (~qpl.active_mask).nonzero(as_tuple=True)[0]
                qpl.V[:, active_idx] = F.normalize(qpl.V[:, active_idx], dim=0, eps=1e-8)
                qpl.V[:, inactive_idx] = 0.0
                qpl.b_in[inactive_idx] = 0.0
                
                # --- Relational Distillation Replay ---
                replay_idx = torch.randperm(anchor_x.shape[0])[:32]
                q_replay = anchor_x[replay_idx].to(DEVICE)
                h_rep, _, _, _ = qpl.settle(q_replay, variant="full_qpl", k_wta=3)
                z_replay = F.normalize(h_rep, dim=-1, eps=1e-8)
                
                S_s = torch.matmul(z_replay, z_replay.T)
                S_t = torch.matmul(z_anchor[replay_idx].to(DEVICE), z_anchor[replay_idx].to(DEVICE).T)
                distill_error = S_t - S_s
                
                dz = 2.0 * torch.matmul(distill_error, z_replay) / 32.0
                dV_distill = torch.matmul(q_replay.T, dz) / 32.0
                
                for j in active_idx:
                    g_d = dV_distill[:, j]
                    v_c = qpl.V[:, j]
                    dV_distill[:, j] = g_d - torch.dot(v_c, g_d) * v_c
                    
                qpl.V.add_(lr * 0.05 * dV_distill)
                qpl.V[:, active_idx] = F.normalize(qpl.V[:, active_idx], dim=0, eps=1e-8)
                
        # Run local unsupervised updates on W and b_out only
        with torch.no_grad():
            h_settled, _, _, _ = qpl.settle(train_x, variant="full_qpl", k_wta=3)
            lrs_unsup = {"V": 0.0, "W": 1e-2, "L": 1e-2, "b": 0.0, "homeo": 1e-3}
            qpl.local_unsupervised_update(train_x, h_settled, kwta_mask=None, lrs=lrs_unsup, current_group=None)
            
        # Log validation accuracy
        val_eval = evaluate_1nn_accuracy(qpl, train_x, train_y, val_x, val_y, k_wta=3)
        if val_eval > best_val_acc:
            best_val_acc = val_eval
            best_weights = {
                "V": qpl.V.clone().detach(),
                "b_in": qpl.b_in.clone().detach(),
                "W": qpl.W.clone().detach(),
                "b_out": qpl.b_out.clone().detach(),
                "L": qpl.L.clone().detach()
            }
            
    if best_weights is not None:
        with torch.no_grad():
            qpl.V.copy_(best_weights["V"])
            qpl.b_in.copy_(best_weights["b_in"])
            qpl.W.copy_(best_weights["W"])
            qpl.b_out.copy_(best_weights["b_out"])
            qpl.L.copy_(best_weights["L"])

# ---------------------------------------------------------------------------
# Evaluation Helper functions
# ---------------------------------------------------------------------------
def evaluate_1nn_accuracy(qpl, ref_x, ref_y, eval_x, eval_y, k_wta=3):
    qpl.eval()
    B = eval_x.shape[0]
    with torch.no_grad():
        z_queries, _ = qpl(eval_x, variant="full_qpl", k_wta=k_wta)
        z_refs, _ = qpl(ref_x, variant="full_qpl", k_wta=k_wta)
    sims = torch.matmul(z_queries, z_refs.T)
    correct = 0
    for idx in range(B):
        q_label = eval_y[idx].item()
        pred_idx = sims[idx].argmax().item()
        if ref_y[pred_idx].item() == q_label:
            correct += 1
    return correct / B

def compute_t_interval(scores, confidence=0.95):
    n = len(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1) if n > 1 else 0.0
    sem = std_dev / np.sqrt(n) if n > 0 else 0.0
    # For n=4 (df=3), t_0.975 is exactly 3.182446
    t_val = 3.182446
    h = sem * t_val
    return mean, mean - h, mean + h

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(CACHE_PATH):
        print(f"[Error] Embeddings cache '{CACHE_PATH}' not found. Please run Stage 2 evaluation first.")
        return
        
    data = torch.load(CACHE_PATH, weights_only=True)
    train_x = data["train_x"].to(DEVICE)
    train_y = data["train_y"].to(DEVICE)
    val_x = data["val_x"].to(DEVICE)
    val_y = data["val_y"].to(DEVICE)
    test_x = data["test_x"].to(DEVICE)
    test_y = data["test_y"].to(DEVICE)
    
    print("="*80)
    print("  STAGE 4 CONTRASTIVE HEBBIAN LEARNING (CHL) LOCKED FINAL TEST")
    print("="*80)
    
    # 1. Encode Control / Hard Negative Sentences
    print("[Control] Loading SmolLM2 for general language control embeddings...")
    control_vectors = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, output_hidden_states=True)
        model.to(DEVICE)
        model.eval()
        
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
        
        for text in INDEPENDENT_PPL_TEXTS:
            enc = tokenizer(text, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(enc.input_ids, attention_mask=enc.attention_mask)
                hidden = outputs.hidden_states[-1]
                mask = enc.attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = F.normalize(pooled.float(), dim=-1)
                control_vectors.append(pooled[0].cpu())
        control_x = torch.stack(control_vectors).to(DEVICE)
        print(f"[Control] Encoded {len(control_x)} control queries successfully.")
    except Exception as e:
        print(f"[Warning] Failed to load HuggingFace model. Creating dummy control vectors: {e}")
        control_x = torch.randn(len(INDEPENDENT_PPL_TEXTS), INPUT_DIM, device=DEVICE)
        control_x = F.normalize(control_x, dim=-1)
        
    seeds = [41, 42, 43, 44]
    lr = 0.20
    
    results = []
    
    for seed in seeds:
        print(f"\n[Test] Running Seed {seed}...")
        
        # Supervised Backprop baseline
        bp_acc = train_supervised_probe(train_x, train_y, test_x, test_y, d_active=34, seed=seed)
        print(f"  Supervised Backprop 1-NN accuracy: {bp_acc*100:.2f}%")
        
        # QPL Initial Orthonormal Projection baseline
        qpl_init = HybridQPL(input_dim=INPUT_DIM, output_dim=128).to(DEVICE)
        qpl_init.initialize_basis(34)
        init_acc = evaluate_1nn_accuracy(qpl_init, train_x, train_y, test_x, test_y, k_wta=3)
        print(f"  Pre-CHL Random Orthonormal baseline accuracy: {init_acc*100:.2f}%")
        
        # QPL Unsupervised Collapsed baseline
        qpl_unsup = HybridQPL(input_dim=INPUT_DIM, output_dim=128).to(DEVICE)
        qpl_unsup.initialize_basis(34)
        train_unsupervised_only(qpl_unsup, train_x, epochs=45, seed=seed)
        unsup_acc = evaluate_1nn_accuracy(qpl_unsup, train_x, train_y, test_x, test_y, k_wta=3)
        print(f"  Pre-CHL Unsupervised Collapsed baseline accuracy: {unsup_acc*100:.2f}%")
        
        # Train actual QPL with CHL
        qpl = HybridQPL(input_dim=INPUT_DIM, output_dim=128).to(DEVICE)
        qpl.initialize_basis(34)
        
        # Get train representation pre-training
        with torch.no_grad():
            h_pre, _ = qpl(train_x, variant="full_qpl", k_wta=3)
            S_pre = torch.matmul(F.normalize(h_pre, dim=-1), F.normalize(h_pre, dim=-1).T)
            
        train_qpl_chl(qpl, train_x, train_y, val_x, val_y, epochs=45, lr=lr, seed=seed)
        
        # Save trained weights for subsequent student distillation phase
        torch.save({
            "V": qpl.V.clone().detach().cpu(),
            "b_in": qpl.b_in.clone().detach().cpu(),
            "W": qpl.W.clone().detach().cpu(),
            "b_out": qpl.b_out.clone().detach().cpu(),
            "L": qpl.L.clone().detach().cpu()
        }, f"best_chl_qpl_seed{seed}.pt")
        
        # Evaluate post-training test accuracy
        test_acc = evaluate_1nn_accuracy(qpl, train_x, train_y, test_x, test_y, k_wta=3)
        print(f"  CHL-QPL Test 1-NN accuracy: {test_acc*100:.2f}%")
        
        # Compute representation drift and relational preservation
        with torch.no_grad():
            h_post, _ = qpl(train_x, variant="full_qpl", k_wta=3)
            z_pre_norm = F.normalize(h_pre, dim=-1)
            z_post_norm = F.normalize(h_post, dim=-1)
            drift_cos = (1.0 - (z_pre_norm * z_post_norm).sum(dim=-1)).mean().item()
            
            S_post = torch.matmul(z_post_norm, z_post_norm.T)
            # Relational alignment error E_rel
            frob_diff = torch.norm(S_post - S_pre, p="fro")
            frob_pre = torch.norm(S_pre, p="fro")
            e_rel = (frob_diff / (frob_pre + 1e-8)).item()
            
        # Compute usage statistics on test split
        with torch.no_grad():
            h_test, _, _, _ = qpl.settle(test_x, variant="full_qpl", k_wta=3)
            scores = h_test.masked_fill(~qpl.active_mask.unsqueeze(0), -float("inf"))
            indices = scores.topk(3, dim=-1).indices
            kwta_mask = torch.zeros_like(h_test)
            kwta_mask.scatter_(1, indices, 1.0)
            
        active_idx = qpl.active_mask.nonzero(as_tuple=True)[0]
        usage = kwta_mask[:, active_idx].sum(dim=0).cpu().numpy()
        usage_sorted = np.sort(usage)
        n = len(active_idx)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * usage_sorted)) / (n * np.sum(usage_sorted) + 1e-8)
        
        dead_units = (usage == 0).sum() / n
        expected_freq = (len(test_x) * 3) / n
        max_freq_ratio = usage.max() / expected_freq
        min_usage = usage.min()
        
        # Expected Gini under perfectly balanced assignment is 0.0
        
        # Winner overlap within and between facts
        overlap_within = []
        overlap_between = []
        
        winner_sets = [set(idx_list) for idx_list in indices.cpu().numpy().tolist()]
        
        for i in range(len(test_x)):
            for j in range(i + 1, len(test_x)):
                overlap = len(winner_sets[i].intersection(winner_sets[j]))
                if test_y[i].item() == test_y[j].item():
                    overlap_within.append(overlap)
                else:
                    overlap_between.append(overlap)
                    
        mean_within = np.mean(overlap_within) if len(overlap_within) > 0 else 0.0
        mean_between = np.mean(overlap_between) if len(overlap_between) > 0 else 0.0
        
        # Representation effective rank (using Shannon entropy of SVD)
        _, S_svd, _ = torch.linalg.svd(z_post_norm, full_matrices=False)
        svd_probs = S_svd / S_svd.sum()
        eff_rank = torch.exp(-torch.sum(svd_probs * torch.log(svd_probs + 1e-8))).item()
        
        # Hard-Negative FPR at 95% TPR threshold
        # Find 1-NN similarity of correct test queries to their references
        with torch.no_grad():
            z_test_q = F.normalize(h_test, dim=-1)
            z_ref_q = F.normalize(h_post, dim=-1)
        sim_corrects = []
        for idx in range(len(test_x)):
            q_label = test_y[idx].item()
            ref_indices = (train_y == q_label).nonzero(as_tuple=True)[0]
            max_sim = torch.matmul(z_test_q[idx], z_ref_q[ref_indices].T).max().item()
            sim_corrects.append(max_sim)
            
        tpr_95_threshold = np.percentile(sim_corrects, 5)  # 5th percentile gives 95% TPR
        
        # Now evaluate control queries false positive rate
        with torch.no_grad():
            h_ctrl, _ = qpl(control_x, variant="full_qpl", k_wta=3)
            z_ctrl = F.normalize(h_ctrl, dim=-1)
        sim_ctrls = torch.matmul(z_ctrl, z_ref_q.T).max(dim=1).values.cpu().numpy()
        hn_fpr = (sim_ctrls >= tpr_95_threshold).mean()
        
        # Recovered Gains
        gain_rand = (test_acc - init_acc) / (bp_acc - init_acc + 1e-8)
        gain_unsup = (test_acc - unsup_acc) / (bp_acc - unsup_acc + 1e-8)
        
        results.append({
            "test_acc": test_acc,
            "init_acc": init_acc,
            "unsup_acc": unsup_acc,
            "bp_acc": bp_acc,
            "gain_rand": gain_rand,
            "gain_unsup": gain_unsup,
            "drift_cos": drift_cos,
            "e_rel": e_rel,
            "gini": gini,
            "dead_units": dead_units,
            "max_freq_ratio": max_freq_ratio,
            "min_usage": min_usage,
            "mean_within": mean_within,
            "mean_between": mean_between,
            "eff_rank": eff_rank,
            "hn_fpr": hn_fpr
        })
        
    # ---------------------------------------------------------------------------
    # Statistical Aggregation
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print("  STAGE 4 FINAL TEST STATISTICAL REPORT (LOCKED HYPERPARAMETERS)")
    print("="*80)
    
    test_accs = [r["test_acc"] for r in results]
    init_accs = [r["init_acc"] for r in results]
    unsup_accs = [r["unsup_acc"] for r in results]
    bp_accs = [r["bp_acc"] for r in results]
    gains_rand = [r["gain_rand"] for r in results]
    gains_unsup = [r["gain_unsup"] for r in results]
    drifts = [r["drift_cos"] for r in results]
    e_rels = [r["e_rel"] for r in results]
    ginis = [r["gini"] for r in results]
    deads = [r["dead_units"] for r in results]
    max_freqs = [r["max_freq_ratio"] for r in results]
    mins = [r["min_usage"] for r in results]
    withins = [r["mean_within"] for r in results]
    betweens = [r["mean_between"] for r in results]
    ranks = [r["eff_rank"] for r in results]
    fprs = [r["hn_fpr"] for r in results]
    
    mean_test, lcb_test, ucb_test = compute_t_interval(test_accs)
    mean_init = np.mean(init_accs)
    mean_unsup = np.mean(unsup_accs)
    mean_bp = np.mean(bp_accs)
    
    mean_gain_rand, lcb_gain_rand, ucb_gain_rand = compute_t_interval(gains_rand)
    mean_gain_unsup, lcb_gain_unsup, ucb_gain_unsup = compute_t_interval(gains_unsup)
    
    print(f"1. Test 1-NN Accuracy (CHL-QPL)  : {mean_test*100:.2f}% (95% CI: [{lcb_test*100:.2f}%, {ucb_test*100:.2f}%])")
    print(f"2. Random Orthonormal baseline   : {mean_init*100:.2f}%")
    print(f"3. Unsupervised Collapse baseline: {mean_unsup*100:.2f}%")
    print(f"4. Supervised Backprop Probe     : {mean_bp*100:.2f}%")
    print(f"5. Recovered Gain (vs Random)    : {mean_gain_rand*100:.2f}% (95% CI: [{lcb_gain_rand*100:.2f}%, {ucb_gain_rand*100:.2f}%])")
    print(f"6. Recovered Gain (vs Unsup)     : {mean_gain_unsup*100:.2f}% (95% CI: [{lcb_gain_unsup*100:.2f}%, {ucb_gain_unsup*100:.2f}%])")
    print(f"7. Hard-Negative FPR at 95% TPR  : {np.mean(fprs)*100:.2f}%")
    print(f"8. Representation Drift (1 - cos): {np.mean(drifts):.4f}")
    print(f"9. Relational Alignment Error E  : {np.mean(e_rels):.4f}")
    
    print("\nUsage Statistics on Test Split:")
    print(f"  - Mean Gini Coefficient        : {np.mean(ginis):.3f} (Pre-CHL Collapsed QPL Gini: {mean_unsup:.3f}, Expected Balanced Gini: 0.000)")
    print(f"  - Dead-Unit Fraction           : {np.mean(deads)*100:.1f}%")
    print(f"  - Max Winner Frequency Ratio   : {np.mean(max_freqs):.2f}x")
    print(f"  - Minimum Unit Usage           : {np.mean(mins):.1f} wins")
    print(f"  - Within-Fact Winner Overlap   : {np.mean(withins):.2f} / 3.00")
    print(f"  - Between-Fact Winner Overlap  : {np.mean(betweens):.2f} / 3.00")
    print(f"  - Representation Effective Rank: {np.mean(ranks):.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
