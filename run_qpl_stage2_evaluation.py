"""
run_qpl_stage2_evaluation.py — Stage 2 Competitive QPL Evaluation Suite
======================================================================
Evaluates fixed-width competitive QPL variants, capacity-matched controls,
and ablations. Runs Phase A (width sweep) and Phase B (dynamics sweep).
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from hybrid_qpl import HybridQPL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"

# Reproducibility Locks
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

# ---------------------------------------------------------------------------
# 1. Models & Helper Probes
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
        return F.normalize(z, dim=-1, eps=1e-8)


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    N = embeddings.shape[0]
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    logits_mask = torch.scatter(
        torch.ones_like(logits),
        1,
        torch.arange(N, device=embeddings.device).view(-1, 1),
        0
    )
    
    exp_logits = torch.exp(logits) * logits_mask
    sum_exp_logits = exp_logits.sum(dim=1, keepdim=True) + 1e-8
    
    # Mask for positive pairs (same class, excluding self)
    labels = labels.view(-1, 1)
    positives_mask = torch.eq(labels, labels.T).float() - torch.eye(N, device=embeddings.device)
    
    log_prob = logits - torch.log(sum_exp_logits)
    mean_log_prob_pos = (log_prob * positives_mask).sum(dim=1) / positives_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos.mean()


def train_supervised_probe(train_x, train_y, test_x, test_y, d_active, seed=42):
    """Trains a supervised ResidualMetricProbe to establish the upper bound reference."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    probe = ResidualMetricProbe(input_dim=768, hidden_dim=512, output_dim=d_active).to(DEVICE)
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

# ---------------------------------------------------------------------------
# 2. Controls & Baselines
# ---------------------------------------------------------------------------
def fit_pca(train_x, d_active):
    mean = train_x.mean(dim=0, keepdim=True)
    x_centered = train_x - mean
    _, _, V = torch.linalg.svd(x_centered, full_matrices=False)
    proj = V[:d_active].T  # (768, d_active)
    return mean, proj


def fit_random_projection(d_active, seed=42):
    g = torch.Generator().manual_seed(seed)
    matrix = torch.randn(768, d_active, generator=g)
    Q, _ = torch.linalg.qr(matrix, mode="reduced")
    return Q


def evaluate_projection_1nn(train_proj, train_y, test_proj, test_y):
    # Normalized embeddings
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
# 3. Data Loading & Extraction
# ---------------------------------------------------------------------------
def load_and_cache_dataset():
    if os.path.exists(CACHE_PATH):
        print(f"[Data] Loading cached representations from {CACHE_PATH}...")
        return torch.load(CACHE_PATH, weights_only=True)
        
    print(f"[Data] Extracting fresh representations using {MODEL_ID} (commit {MODEL_REVISION[:8]})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, output_hidden_states=True).to(DEVICE)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    with open("agnis_scaling_dataset.json", "r") as f:
        blocks = json.load(f)
        
    all_facts = [f for b in blocks for f in b]
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    probe_to_class = {p: idx for idx, p in enumerate(unique_probes)}
    
    train_queries = []
    train_labels = []
    val_queries = []
    val_labels = []
    test_queries = []
    test_labels = []
    
    def get_prompt_only(f, idx):
        if idx == 0:
            return f["probe"]
        elif idx == 1:
            prefix = f["qa"].split(f["statement"])[0]
            return prefix + f["probe"]
        else:
            return f["cloze"].split("_____")[0].strip()
            
    def extract_pooled(prompt):
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
        enc = tokenizer(
            prompt,
            max_length=32,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(enc.input_ids, attention_mask=enc.attention_mask)
            hidden = outputs.hidden_states[-1]          # [B, T, 768]
            mask = enc.attention_mask.unsqueeze(-1)     # [B, T, 1]
            pooled = (hidden * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1)
            return pooled[0].cpu()

    for f in all_facts:
        label = probe_to_class[f["probe"]]
        
        # 1. Train Templates
        for idx_t in range(3):
            prompt = get_prompt_only(f, idx_t)
            train_queries.append(extract_pooled(prompt))
            train_labels.append(label)
            
        # 2. Validation Templates
        dev_item = f["train_paraphrases"][-1]
        val_queries.append(extract_pooled(dev_item))
        val_labels.append(label)
        
        # 3. Test Templates
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            test_queries.append(extract_pooled(item))
            test_labels.append(label)
            
    data = {
        "train_x": torch.stack(train_queries), "train_y": torch.tensor(train_labels),
        "val_x": torch.stack(val_queries), "val_y": torch.tensor(val_labels),
        "test_x": torch.stack(test_queries), "test_y": torch.tensor(test_labels)
    }
    torch.save(data, CACHE_PATH)
    print(f"[Data] Saved fresh cache to {CACHE_PATH}")
    return data

# ---------------------------------------------------------------------------
# 4. Evaluation Loop & Statistics Logger
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
        # compute winner selection from kWTA mask
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
# 5. Training Loop
# ---------------------------------------------------------------------------
def train_qpl(qpl, train_x, train_y, variant="full_qpl", k_wta=None, epochs=25, lrs=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    B = train_x.shape[0]
    tx = train_x.to(DEVICE)
    
    for epoch in range(epochs):
        # Shuffle batch
        perm = torch.randperm(B)
        tx_shuffled = tx[perm]
        
        # Mini-batch learning
        batch_size = 30
        for i in range(0, B, batch_size):
            q_batch = tx_shuffled[i : i + batch_size]
            
            # Settle batch to obtain activation h
            with torch.no_grad():
                h, _, _, _ = qpl.settle(q_batch, variant=variant, k_wta=k_wta)
                
            # kWTA mask for usage EMA homeostasis
            if variant == "full_qpl" and k_wta is not None:
                scores = h.masked_fill(~qpl.active_mask.unsqueeze(0), -float("inf"))
                indices = scores.topk(k_wta, dim=-1).indices
                kwta_mask = torch.zeros_like(h)
                kwta_mask.scatter_(1, indices, 1.0)
            else:
                kwta_mask = None
                
            # Perform local unsupervised weight adjustments
            qpl.local_unsupervised_update(q_batch, h, kwta_mask=kwta_mask, lrs=lrs)
            
        # Verify weight invariants at end of each training epoch
        qpl.verify_invariants()

# ---------------------------------------------------------------------------
# 6. Main Evaluation Sweep (Phase A & Phase B)
# ---------------------------------------------------------------------------
def main():
    data = load_and_cache_dataset()
    train_x = data["train_x"].to(DEVICE)
    train_y = data["train_y"].to(DEVICE)
    val_x = data["val_x"].to(DEVICE)
    val_y = data["val_y"].to(DEVICE)
    test_x = data["test_x"].to(DEVICE)
    test_y = data["test_y"].to(DEVICE)
    
    # 0. Raw SmolLM2 Baseline
    raw_acc = evaluate_projection_1nn(train_x, train_y, test_x, test_y)
    print("="*80)
    print(f"  [Baseline] Raw SmolLM2 Test Accuracy (768D): {raw_acc*100:.2f}%")
    print("="*80)
    
    # ---------------------------------------------------------------------------
    # Phase A: Width Sweep (d_active in {16, 34, 64, 128})
    # ---------------------------------------------------------------------------
    widths = [16, 34, 64, 128]
    best_val_acc = -1.0
    best_width = 34
    
    print("\n" + "="*80)
    print("  PHASE A: WIDTH SWEEP (Local Autoencoder Variant)")
    print("="*80)
    print("Width | Control (max PCA/Rand) | Supervised Probe | Unsupervised QPL | G_d recovered")
    print("-" * 85)
    
    phase_a_results = {}
    
    for w in widths:
        # Controls
        mean_pca, proj_pca = fit_pca(data["train_x"], w)
        proj_rand = fit_random_projection(w, seed=42)
        
        pca_train = (data["train_x"] - mean_pca) @ proj_pca
        pca_test = (data["test_x"] - mean_pca) @ proj_pca
        pca_acc = evaluate_projection_1nn(pca_train, train_y, pca_test, test_y)
        
        rand_train = data["train_x"] @ proj_rand
        rand_test = data["test_x"] @ proj_rand
        rand_acc = evaluate_projection_1nn(rand_train, train_y, rand_test, test_y)
        
        control_acc = max(pca_acc, rand_acc)
        
        # Supervised Upper Bound at width w
        supervised_acc = train_supervised_probe(data["train_x"], train_y, data["test_x"], test_y, w, seed=42)
        
        # Initialize and train QPL local autoencoder at width w
        qpl = HybridQPL(input_dim=768, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=1.0).to(DEVICE)
        qpl.initialize_basis(w)
        
        # Unsupervised learning
        train_qpl(qpl, data["train_x"], train_y, variant="local_autoencoder", epochs=20, seed=42)
        
        # Evaluate QPL on test set
        qpl_eval = run_evaluation(qpl, test_x, test_y, train_x, train_y, variant="local_autoencoder")
        qpl_acc = qpl_eval["acc"]
        
        # Dimension-specific recovered gain G_d
        denom = supervised_acc - control_acc
        g_d = (qpl_acc - control_acc) / denom if denom > 1e-8 else float("nan")
        
        print(f"{w:5d} | {control_acc*100:20.2f}% | {supervised_acc*100:16.2f}% | {qpl_acc*100:16.2f}% | {g_d*100:13.2f}%")
        
        phase_a_results[w] = {
            "control_acc": control_acc,
            "supervised_acc": supervised_acc,
            "qpl_acc": qpl_acc,
            "g_d": g_d
        }
        
        # Track best width using validation accuracy
        qpl_val = run_evaluation(qpl, val_x, val_y, train_x, train_y, variant="local_autoencoder")
        if qpl_val["acc"] > best_val_acc:
            best_val_acc = qpl_val["acc"]
            best_width = w
            
    print(f"\n-> Best width selected by validation performance: {best_width}D")
    
    # ---------------------------------------------------------------------------
    # Phase B: Dynamics Sweep (at best_width)
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"  PHASE B: DYNAMICS SWEEP (at {best_width}D)")
    print("="*80)
    print("Rho (k) | Temperature | Val Accuracy | Test Accuracy | Fallback Rate | Winner Flips")
    print("-" * 85)
    
    # Generate test parameters
    rho_values = [0.05, 0.10, 0.20]
    temps = [0.5, 1.0, 2.0]
    
    best_dynamics = (3, 1.0)
    best_dyn_val_acc = -1.0
    
    for rho in rho_values:
        k = max(1, int(rho * best_width))
        for t in temps:
            qpl = HybridQPL(input_dim=768, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=t).to(DEVICE)
            qpl.initialize_basis(best_width)
            
            # Train Full QPL with soft competition and anti-hebbian updates
            train_qpl(qpl, data["train_x"], train_y, variant="full_qpl", k_wta=k, epochs=25, seed=42)
            
            val_eval = run_evaluation(qpl, val_x, val_y, train_x, train_y, variant="full_qpl", k_wta=k)
            test_eval = run_evaluation(qpl, test_x, test_y, train_x, train_y, variant="full_qpl", k_wta=k)
            
            print(f"{rho:7.2f} ({k:d}) | {t:11.1f} | {val_eval['acc']*100:11.2f}% | {test_eval['acc']*100:12.2f}% | {val_eval['fallback_rate']*100:12.4f}% | {val_eval['median_steps']:.1f}")
            
            if val_eval["acc"] > best_dyn_val_acc:
                best_dyn_val_acc = val_eval["acc"]
                best_dynamics = (k, t)
                
    best_k, best_t = best_dynamics
    print(f"\n-> Best dynamics selected: k={best_k}, temperature={best_t}")
    
    # ---------------------------------------------------------------------------
    # Stage 2 Ablation Matrix
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print("  STAGE 2 ABLATION MATRIX")
    print("="*80)
    
    variants = [
        ("Orthogonal Static", "orthogonal_static", None),
        ("Local Autoencoder", "local_autoencoder", None),
        ("Local + Soft Competition", "local_soft_competition", None),
        ("Local + Anti-Hebbian", "local_anti_hebbian", None),
        ("Full QPL (no homeo)", "full_qpl", best_k),
        ("Full QPL (with homeo)", "full_qpl", best_k)
    ]
    
    for name, variant, k_val in variants:
        qpl = HybridQPL(input_dim=768, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=best_t).to(DEVICE)
        qpl.initialize_basis(best_width)
        
        # Train variant
        homeo_lr = 1e-3 if "with homeo" in name else 0.0
        lrs = {"V": 1e-2, "W": 1e-2, "L": 1e-2, "b": 1e-2, "homeo": homeo_lr}
        
        train_qpl(qpl, data["train_x"], train_y, variant=variant, k_wta=k_val, epochs=25, lrs=lrs, seed=42)
        eval_metrics = run_evaluation(qpl, test_x, test_y, train_x, train_y, variant=variant, k_wta=k_val)
        
        print(f"Variant: {name:<25} | Test Acc: {eval_metrics['acc']*100:.2f}% | Fallback Rate: {eval_metrics['fallback_rate']*100:.3f}% | Gini: {eval_metrics['gini']:.3f} | Dead Units: {eval_metrics['dead_units']*100:.1f}%")

if __name__ == "__main__":
    main()
