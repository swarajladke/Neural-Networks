"""
run_multi_model_corrected_probe.py — Multi-Model Corrected Stage 1 Probe
======================================================================
Evaluates true representational recoverability and paraphrase generalization
across GPT-2, Qwen-0.5B, and SmolLM-360M using the 34 unique slot classes.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Data Extraction
# ---------------------------------------------------------------------------
def extract_pooled_representations(model_name):
    print(f"\n[Extract] Loading {model_name} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(DEVICE)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    hidden_dim = model.config.hidden_size
    
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
            
    for idx_f, f in enumerate(all_facts):
        fid = f["id"]
        label = probe_to_class[f["probe"]]
        
        # 1. Train Templates
        for idx_t in range(3):
            prompt = get_prompt_only(f, idx_t)
            ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(ids)
                # Mean-pooling over sequence
                rep = outputs.hidden_states[-1][0].mean(dim=0).cpu().float()
            train_queries.append(rep)
            train_labels.append(label)
            
        # 2. Validation Templates
        dev_item = f["train_paraphrases"][-1]
        item_ids = tokenizer.encode(dev_item, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(item_ids)
            rep = outputs.hidden_states[-1][0].mean(dim=0).cpu().float()
        val_queries.append(rep)
        val_labels.append(label)
        
        # 3. Test Templates
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            item_ids = tokenizer.encode(item, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(item_ids)
                rep = outputs.hidden_states[-1][0].mean(dim=0).cpu().float()
            test_queries.append(rep)
            test_labels.append(label)
            
    return {
        "train_x": torch.stack(train_queries), "train_y": torch.tensor(train_labels),
        "val_x": torch.stack(val_queries), "val_y": torch.tensor(val_labels),
        "test_x": torch.stack(test_queries), "test_y": torch.tensor(test_labels),
        "hidden_dim": hidden_dim
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


def compute_student_t_interval(scores):
    n = len(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1) if n > 1 else 0.0
    sem = std_dev / np.sqrt(n) if n > 0 else 0.0
    t_val = 2.776445  # t_0.975 for df=4
    h = sem * t_val
    return mean, mean - h, mean + h


# ---------------------------------------------------------------------------
# Main Sweep
# ---------------------------------------------------------------------------
def main():
    models = ["HuggingFaceTB/SmolLM2-360M"]
    results = {}
    
    for model_name in models:
        try:
            data = extract_pooled_representations(model_name)
        except Exception as e:
            print(f"[Error] Failed to process {model_name}: {e}")
            continue
            
        ref_x = data["train_x"]
        ref_y = data["train_y"]
        
        # Raw Cosine Accuracy
        raw_acc = evaluate_1nn_accuracy(lambda x: F.normalize(x, dim=-1), data["test_x"], data["test_y"], ref_x, ref_y)
        print(f"  -> Raw Cosine 1-NN Accuracy: {raw_acc*100:.2f}%")
        
        seeds = [1000 * (i + 1) for i in range(5)]
        probe_accs = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            probe = ResidualMetricProbe(input_dim=data["hidden_dim"], hidden_dim=512, output_dim=128).to(DEVICE)
            opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
            
            for epoch in range(150):
                probe.train()
                z_train = probe(data["train_x"].to(DEVICE))
                loss = supervised_contrastive_loss(z_train, data["train_y"].to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            test_acc = evaluate_1nn_accuracy(probe, data["test_x"], data["test_y"], ref_x, ref_y)
            probe_accs.append(test_acc)
            
        mean, lcb, ucb = compute_student_t_interval(probe_accs)
        results[model_name] = {
            "raw_acc": raw_acc,
            "mean_acc": mean,
            "lcb": lcb
        }
        print(f"  -> Probe Test Acc (Mean): {mean*100:.2f}% | 95% LCB: {lcb*100:.2f}%")
        
    print("\n" + "="*80)
    print("  CORRECTED 34-SLOT MODEL COMPARISON")
    print("="*80)
    print(f"  {'Model Name':<30} | {'Raw 1-NN':<10} | {'Probe Mean':<10} | {'95% LCB':<10}")
    print("  " + "-" * 56)
    for model_name in results:
        res = results[model_name]
        print(f"  {model_name:<30} | {res['raw_acc']*100:>8.1f}% | {res['mean_acc']*100:>8.1f}% | {res['lcb']*100:>8.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
