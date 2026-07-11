"""
run_corrected_stage1_probe.py — Corrected Stage 1 Recoverability Probe
======================================================================
Groups the 100 facts by their unique prompt templates (34 unique slots)
to resolve the label conflict in the experimental design and evaluate
the true representational recoverability of frozen GPT-2.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agnis_continual_v2 import build_hybrid
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


# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------
def cache_raw_representations():
    print("[Cache] Loading model foundation for representation caching...")
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    with open("agnis_scaling_dataset.json", "r") as f:
        blocks = json.load(f)
        
    all_facts = [f for b in blocks for f in b]
    
    # Corrected slot mapping: group by unique f["probe"]
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    probe_to_class = {p: idx for idx, p in enumerate(unique_probes)}
    print(f"[Cache] Mapped {len(all_facts)} facts to {len(unique_probes)} unique slots.")
    
    train_queries = []
    train_labels = []
    val_queries = []
    val_labels = []
    test_queries = []
    test_labels = []
    
    print("[Cache] Processing query templates...")
    for idx_f, f in enumerate(all_facts):
        fid = f["id"]
        label = probe_to_class[f["probe"]]
        
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
        
        # 3. Test Templates
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            item_ids = tokenizer.encode(item, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                _, item_h = gpt2_forward(hybrid, item_ids)
                item_q_raw = item_h[0, -min(2, item_h.shape[1]):, :].mean(dim=0)
            test_queries.append(item_q_raw.cpu())
            test_labels.append(label)
            
    return {
        "train_x": torch.stack(train_queries), "train_y": torch.tensor(train_labels),
        "val_x": torch.stack(val_queries), "val_y": torch.tensor(val_labels),
        "test_x": torch.stack(test_queries), "test_y": torch.tensor(test_labels)
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_1nn_accuracy(model_fn, x_data, y_data, ref_x_data, ref_y_data):
    if hasattr(model_fn, "eval"):
        model_fn.eval()
    with torch.no_grad():
        z_queries = model_fn(x_data.to(DEVICE))
        z_refs = model_fn(ref_x_data.to(DEVICE))
        
    sims = torch.matmul(z_queries, z_refs.T)
    correct = 0
    for idx in range(len(y_data)):
        q_label = y_data[idx].item()
        pred_idx = sims[idx].argmax().item()
        if ref_y_data[pred_idx].item() == q_label:
            correct += 1
    return correct / len(y_data)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    data = cache_raw_representations()
    ref_x = data["train_x"]
    ref_y = data["train_y"]
    
    # Raw Cosine Accuracy
    raw_acc = evaluate_1nn_accuracy(lambda x: F.normalize(x, dim=-1), data["test_x"], data["test_y"], ref_x, ref_y)
    print(f"\n[Baseline] Raw Cosine 1-NN Accuracy on 34 slots: {raw_acc*100:.2f}%")
    
    # Train Supervised Residual MLP Probe over 5 seeds
    seeds = [1000 * (i + 1) for i in range(5)]
    probe_accs = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        probe = ResidualMetricProbe(input_dim=768, hidden_dim=512, output_dim=128).to(DEVICE)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(150):
            probe.train()
            z_train = probe(data["train_x"].to(DEVICE))
            loss = supervised_contrastive_loss(z_train, data["train_y"].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        test_acc = evaluate_1nn_accuracy(probe, data["test_x"], data["test_y"], ref_x, ref_y)
        train_acc = evaluate_1nn_accuracy(probe, data["train_x"], data["train_y"], ref_x, ref_y)
        probe_accs.append(test_acc)
        print(f"  Seed {seed} -> Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%")
        
    mean = np.mean(probe_accs)
    print(f"\n[Result] Corrected Residual MLP Test Accuracy: {mean*100:.2f}%")


if __name__ == "__main__":
    main()
