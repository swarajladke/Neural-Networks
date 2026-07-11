"""
run_supervised_metric_upper_bound.py — Stage 1 Recoverability Probe
===================================================================
Establishes if fact-identity information is recoverable from frozen GPT-2
representations using supervised nonlinear and linear metric learning.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# scipy stats removed

from agnis_continual_v2 import INDEPENDENT_PPL_TEXTS, build_hybrid
from agnis_continual_v4_1 import DEVICE, gpt2_forward
from agnis_scaling_runner import get_template_prompt

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ResidualMetricProbe(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
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


class LinearMetricProbe(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        z = self.proj(x)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
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


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        N = embeddings.shape[0]
        loss = torch.tensor(0.0, device=embeddings.device)
        triplets_count = 0
        
        dist_matrix = 2.0 - 2.0 * torch.matmul(embeddings, embeddings.T)
        
        for i in range(N):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            neg_mask = (labels != labels[i])
            
            pos_dists = dist_matrix[i, pos_mask]
            neg_dists = dist_matrix[i, neg_mask]
            
            if len(pos_dists) > 0 and len(neg_dists) > 0:
                p_dist = pos_dists.view(-1, 1)
                n_dist = neg_dists.view(1, -1)
                triplet_losses = F.relu(p_dist - n_dist + self.margin)
                
                loss += triplet_losses.mean()
                triplets_count += 1
                
        return loss / max(1, triplets_count)


# ---------------------------------------------------------------------------
# Data Caching & Encoding helper
# ---------------------------------------------------------------------------
def cache_raw_representations():
    """Extracts raw query vectors upfront to avoid GPT-2 calls during search sweeps."""
    print("[Cache] Loading model foundation for representation caching...")
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    with open("agnis_scaling_dataset.json", "r") as f:
        blocks = json.load(f)
        
    all_facts = [f for b in blocks for f in b]
    fact_ids = [f["id"] for f in all_facts]
    fid_to_idx = {fid: idx for idx, fid in enumerate(fact_ids)}
    
    # Lists to hold raw query coordinates
    train_queries = []  # (300, 768)
    train_labels = []
    
    val_queries = []    # (100, 768)
    val_labels = []
    
    test_queries = []   # (400, 768)
    test_labels = []
    
    print("[Cache] Processing query templates...")
    for idx_f, f in enumerate(all_facts):
        fid = f["id"]
        label = fid_to_idx[fid]
        
        # 1. Train Templates (stmt, QA, cloze)
        for idx_t in range(3):
            _, prompt = get_template_prompt(f, idx_t)
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                _, item_h = gpt2_forward(hybrid, prompt_ids)
                item_q_raw = item_h[0, -min(2, item_h.shape[1]):, :].mean(dim=0)
            train_queries.append(item_q_raw.cpu())
            train_labels.append(label)
            
        # 2. Validation Templates
        dev_item = f["train_paraphrases"][-1]
        item_ids = tokenizer.encode(dev_item, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _, item_h = gpt2_forward(hybrid, item_ids)
            item_q_raw = item_h[0, -min(2, item_h.shape[1]):, :].mean(dim=0)
        val_queries.append(item_q_raw.cpu())
        val_labels.append(label)
        
        # 3. Test Templates (probe + 3 eval paraphrases)
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            item_ids = tokenizer.encode(item, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                _, item_h = gpt2_forward(hybrid, item_ids)
                item_q_raw = item_h[0, -min(2, item_h.shape[1]):, :].mean(dim=0)
            test_queries.append(item_q_raw.cpu())
            test_labels.append(label)
            
    # Convert to single tensors
    train_x = torch.stack(train_queries)
    train_y = torch.tensor(train_labels)
    val_x = torch.stack(val_queries)
    val_y = torch.tensor(val_labels)
    test_x = torch.stack(test_queries)
    test_y = torch.tensor(test_labels)
    
    # PCA baseline correction (Anisotropy projection-out)
    print("[Cache] Fitting PCA correction on training representations...")
    mu = train_x.mean(dim=0)
    train_centered = train_x - mu
    U, S, V = torch.pca_lowrank(train_centered, q=10)
    V_sub = V[:, :5]  # project out top 5 principal components
    
    def apply_pca(x):
        centered = x - mu
        return centered - torch.matmul(torch.matmul(centered, V_sub), V_sub.T)
        
    train_x_pca = apply_pca(train_x)
    val_x_pca = apply_pca(val_x)
    test_x_pca = apply_pca(test_x)
    
    return {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "test_x": test_x, "test_y": test_y,
        "train_x_pca": train_x_pca, "val_x_pca": val_x_pca, "test_x_pca": test_x_pca
    }


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
def evaluate_1nn_metrics(model_fn, x_data, y_data, ref_x_data, ref_y_data):
    """
    Evaluates 1-NN fact classification, MRR, and margin statistics.
    Ensures STRICT reference constraints (ref_x_data/ref_y_data contain train only).
    """
    if hasattr(model_fn, "eval"):
        model_fn.eval()
    with torch.no_grad():
        z_queries = model_fn(x_data.to(DEVICE))
        z_refs = model_fn(ref_x_data.to(DEVICE))
        
    # Similarity matrix: (Q, R)
    sims = torch.matmul(z_queries, z_refs.T)  # cosine similarity
    
    correct_count = 0
    mrr_sum = 0.0
    margins = []
    
    # recall@k lists
    r1, r5, r10 = 0, 0, 0
    
    for q_idx in range(len(y_data)):
        q_label = y_data[q_idx].item()
        q_sims = sims[q_idx]
        
        # Sort similarities descending
        sorted_sims, sorted_idxs = torch.sort(q_sims, descending=True)
        sorted_labels = ref_y_data[sorted_idxs.cpu()].tolist()
        
        # 1. 1-NN Accuracy
        pred_label = sorted_labels[0]
        if pred_label == q_label:
            correct_count += 1
            
        # 2. MRR and Recall@K
        first_correct_rank = sorted_labels.index(q_label) + 1
        mrr_sum += 1.0 / first_correct_rank
        if first_correct_rank <= 1: r1 += 1
        if first_correct_rank <= 5: r5 += 1
        if first_correct_rank <= 10: r10 += 1
        
        # 3. Logit-Margin equivalent (nearest positive - nearest negative)
        pos_mask = (ref_y_data == q_label)
        neg_mask = (ref_y_data != q_label)
        
        max_pos_sim = q_sims[pos_mask].max().item()
        max_neg_sim = q_sims[neg_mask].max().item()
        margins.append(max_pos_sim - max_neg_sim)
        
    total = len(y_data)
    return {
        "accuracy": correct_count / total,
        "mrr": mrr_sum / total,
        "r1": r1 / total,
        "r5": r5 / total,
        "r10": r10 / total,
        "mean_margin": np.mean(margins)
    }


def compute_student_t_interval(scores, confidence=0.95):
    n = len(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1) if n > 1 else 0.0
    sem = std_dev / np.sqrt(n) if n > 0 else 0.0
    # For n=5 (df=4), t_0.975 is exactly 2.776445
    t_val = 2.776445
    h = sem * t_val
    return mean, mean - h, mean + h


def compute_query_bootstrap_interval(model_fn, x_data, y_data, ref_x_data, ref_y_data, n_bootstraps=1000):
    """Computes a query-level bootstrap confidence interval for test evaluation uncertainty."""
    model_fn.eval()
    with torch.no_grad():
        z_queries = model_fn(x_data.to(DEVICE))
        z_refs = model_fn(ref_x_data.to(DEVICE))
        
    sims = torch.matmul(z_queries, z_refs.T)
    correct_mask = []
    
    for q_idx in range(len(y_data)):
        q_label = y_data[q_idx].item()
        sorted_idxs = torch.argsort(sims[q_idx], descending=True)
        pred_label = ref_y_data[sorted_idxs[0]].item()
        correct_mask.append(float(pred_label == q_label))
        
    correct_mask = np.array(correct_mask)
    bootstrap_means = []
    random_state = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idxs = random_state.randint(0, len(correct_mask), size=len(correct_mask))
        bootstrap_means.append(correct_mask[idxs].mean())
        
    bootstrap_means.sort()
    low = bootstrap_means[int(n_bootstraps * 0.025)]
    high = bootstrap_means[int(n_bootstraps * 0.975)]
    return low, high


# ---------------------------------------------------------------------------
# Stage 1 Execution Pipeline
# ---------------------------------------------------------------------------
def run_stage1_probe():
    # Cache raw coordinates
    data = cache_raw_representations()
    
    # Establish reference bank protocols (Strict: train examples only)
    ref_x = data["train_x"]
    ref_y = data["train_y"]
    
    ref_x_pca = data["train_x_pca"]
    
    # -----------------------------------------------------------------------
    # Baseline 1: Raw Cosine 1-NN
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("  BASELINE 1: RAW COSINE 1-NN")
    print("="*80)
    raw_eval = evaluate_1nn_metrics(lambda x: F.normalize(x, dim=-1), data["test_x"], data["test_y"], ref_x, ref_y)
    print(f"  Test Accuracy: {raw_eval['accuracy']*100:.2f}% | MRR: {raw_eval['mrr']:.4f}")
    print(f"  Recall@1/5/10: {raw_eval['r1']*100:.1f}% / {raw_eval['r5']*100:.1f}% / {raw_eval['r10']*100:.1f}%")
    print(f"  Mean Margin  : {raw_eval['mean_margin']:.4f}")
    
    # -----------------------------------------------------------------------
    # Baseline 2: PCA-Corrected Cosine 1-NN
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("  BASELINE 2: PCA-CORRECTED COSINE 1-NN")
    print("="*80)
    pca_eval = evaluate_1nn_metrics(lambda x: F.normalize(x, dim=-1), data["test_x_pca"], data["test_y"], ref_x_pca, ref_y)
    print(f"  Test Accuracy: {pca_eval['accuracy']*100:.2f}% | MRR: {pca_eval['mrr']:.4f}")
    print(f"  Recall@1/5/10: {pca_eval['r1']*100:.1f}% / {pca_eval['r5']*100:.1f}% / {pca_eval['r10']*100:.1f}%")
    print(f"  Mean Margin  : {pca_eval['mean_margin']:.4f}")
    
    # Sweep over seeds
    seeds = [1000 * (i + 1) for i in range(5)]
    
    models = ["linear_supervised", "nonlinear_residual_mlp"]
    runs = {m: [] for m in models}
    
    for seed in seeds:
        print(f"\n" + "-"*80)
        print(f"  RUNNING PROBE SEED: {seed}")
        print("-"*80)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # -------------------------------------------------------------------
        # Model 1: Linear Supervised Projection (InfoNCE)
        # -------------------------------------------------------------------
        print("[Probe] Training Linear Supervised Probe...")
        lin_model = LinearMetricProbe(input_dim=768, output_dim=128).to(DEVICE)
        opt_lin = torch.optim.AdamW(lin_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(150):
            lin_model.train()
            # Batch matches whole training corpus
            z_train = lin_model(data["train_x"].to(DEVICE))
            loss = supervised_contrastive_loss(z_train, data["train_y"].to(DEVICE))
            opt_lin.zero_grad()
            loss.backward()
            opt_lin.step()
            
        lin_eval = evaluate_1nn_metrics(lin_model, data["test_x"], data["test_y"], ref_x, ref_y)
        lin_train_acc = evaluate_1nn_metrics(lin_model, data["train_x"], data["train_y"], ref_x, ref_y)["accuracy"]
        runs["linear_supervised"].append({**lin_eval, "train_accuracy": lin_train_acc, "model": lin_model})
        
        # -------------------------------------------------------------------
        # Model 2: Nonlinear Residual MLP (InfoNCE)
        # -------------------------------------------------------------------
        print("[Probe] Training Nonlinear Residual MLP...")
        nonlin_model = ResidualMetricProbe(input_dim=768, hidden_dim=512, output_dim=128).to(DEVICE)
        opt_nonlin = torch.optim.AdamW(nonlin_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(150):
            nonlin_model.train()
            z_train = nonlin_model(data["train_x"].to(DEVICE))
            loss = supervised_contrastive_loss(z_train, data["train_y"].to(DEVICE))
            opt_nonlin.zero_grad()
            loss.backward()
            opt_nonlin.step()
            
        nonlin_eval = evaluate_1nn_metrics(nonlin_model, data["test_x"], data["test_y"], ref_x, ref_y)
        nonlin_train_acc = evaluate_1nn_metrics(nonlin_model, data["train_x"], data["train_y"], ref_x, ref_y)["accuracy"]
        runs["nonlinear_residual_mlp"].append({**nonlin_eval, "train_accuracy": nonlin_train_acc, "model": nonlin_model})
        
        print(f"  Linear  -> Train Acc: {lin_train_acc*100:.1f}% | Test Acc: {lin_eval['accuracy']*100:.1f}%")
        print(f"  Nonlin  -> Train Acc: {nonlin_train_acc*100:.1f}% | Test Acc: {nonlin_eval['accuracy']*100:.1f}%")
        
    # -----------------------------------------------------------------------
    # Aggregated Summary Report
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("  SUPERVISED NONLINEAR RECOVERABILITY PROBE SUMMARY")
    print("="*80)
    
    for m in models:
        accs = [r["accuracy"] for r in runs[m]]
        tr_accs = [r["train_accuracy"] for r in runs[m]]
        mrrs = [r["mrr"] for r in runs[m]]
        margins = [r["mean_margin"] for r in runs[m]]
        r1s = [r["r1"] for r in runs[m]]
        r5s = [r["r5"] for r in runs[m]]
        r10s = [r["r10"] for r in runs[m]]
        
        mean_acc, lcb, ucb = compute_student_t_interval(accs)
        mean_tr = np.mean(tr_accs)
        mean_mrr = np.mean(mrrs)
        mean_margin = np.mean(margins)
        mean_r1 = np.mean(r1s)
        mean_r5 = np.mean(r5s)
        mean_r10 = np.mean(r10s)
        
        # Select best model seed for bootstrap estimation
        best_run_idx = np.argmax(accs)
        best_model = runs[m][best_run_idx]["model"]
        boot_low, boot_high = compute_query_bootstrap_interval(best_model, data["test_x"], data["test_y"], ref_x, ref_y)
        
        print(f"\nModel: {m}")
        print(f"  Training 1-NN Accuracy         : {mean_tr*100:.2f}%")
        print(f"  Test 1-NN Accuracy (Mean)      : {mean_acc*100:.2f}%")
        print(f"  Student-t 95% Confidence range : [{lcb*100:.2f}%, {ucb*100:.2f}%] (LCB={lcb*100:.2f}%)")
        print(f"  Query-level Bootstrap 95% CI   : [{boot_low*100:.2f}%, {boot_high*100:.2f}%]")
        print(f"  Recall@1 / 5 / 10              : {mean_r1*100:.1f}% / {mean_r5*100:.1f}% / {mean_r10*100:.1f}%")
        print(f"  Mean MRR                       : {mean_mrr:.4f}")
        print(f"  Mean Logit-Margin              : {mean_margin:+.4f}")
        
    print("\n" + "="*80)


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    run_stage1_probe()
