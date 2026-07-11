"""
run_multi_model_recoverability_probe.py — Multi-Model Recoverability Probe
==========================================================================
Evaluates multiple frozen base LLMs (GPT-2, Qwen-0.5B, SmolLM-360M) and
extraction pooling methods (Last-Token, Mean, Multi-Layer, Attention)
on the 100-fact dataset under strict Stage 1 validation criteria.
"""
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Models for Probe
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
# Dynamic Extraction Pooling
# ---------------------------------------------------------------------------
def extract_pooled_representations(model_name, pooling_type):
    """
    Loads model and tokenizer, extracts hidden states for all fact prompts,
    and pools them according to pooling_type.
    """
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
    fact_ids = [f["id"] for f in all_facts]
    fid_to_idx = {fid: idx for idx, fid in enumerate(fact_ids)}
    
    train_queries = []
    train_labels = []
    val_queries = []
    val_labels = []
    test_queries = []
    test_labels = []
    
    # Simple templates loading functions
    def get_prompt_only(f, idx):
        if idx == 0:
            return f["probe"]
        elif idx == 1:
            prefix = f["qa"].split(f["statement"])[0]
            return prefix + f["probe"]
        else:
            return f["cloze"].split("_____")[0].strip()

    print(f"[Extract] Extracting representations using '{pooling_type}' pooling...")
    
    # Cache all prompts to process in batch
    for idx_f, f in enumerate(all_facts):
        fid = f["id"]
        label = fid_to_idx[fid]
        
        # 1. Train Templates (stmt, QA, cloze prompts)
        for idx_t in range(3):
            prompt = get_prompt_only(f, idx_t)
            ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(ids)
                # hidden_states is tuple: (layers + 1) of shape (1, sequence_length, hidden_dim)
                layers = outputs.hidden_states
                
                if pooling_type == "last_token":
                    # Last layer, last token
                    rep = layers[-1][0, -1, :]
                elif pooling_type == "mean":
                    # Last layer, average over sequence length
                    rep = layers[-1][0].mean(dim=0)
                elif pooling_type == "middle_layer":
                    # Middle layer, last token
                    mid_idx = len(layers) // 2
                    rep = layers[mid_idx][0, -1, :]
                elif pooling_type == "multi_layer":
                    # Average over last 4 layers, mean over sequence
                    stacked = torch.stack([layers[i][0] for i in [-1, -2, -3, -4]])  # (4, seq_len, D)
                    rep = stacked.mean(dim=0).mean(dim=0)  # average layers and seq length
                else:
                    raise ValueError(f"Unknown pooling type: {pooling_type}")
                    
            train_queries.append(rep.cpu())
            train_labels.append(label)
            
        # 2. Validation Templates
        dev_item = f["train_paraphrases"][-1]
        item_ids = tokenizer.encode(dev_item, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(item_ids)
            layers = outputs.hidden_states
            if pooling_type == "last_token":
                rep = layers[-1][0, -1, :]
            elif pooling_type == "mean":
                rep = layers[-1][0].mean(dim=0)
            elif pooling_type == "middle_layer":
                mid_idx = len(layers) // 2
                rep = layers[mid_idx][0, -1, :]
            elif pooling_type == "multi_layer":
                stacked = torch.stack([layers[i][0] for i in [-1, -2, -3, -4]])
                rep = stacked.mean(dim=0).mean(dim=0)
        val_queries.append(rep.cpu())
        val_labels.append(label)
        
        # 3. Test Templates
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            item_ids = tokenizer.encode(item, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(item_ids)
                layers = outputs.hidden_states
                if pooling_type == "last_token":
                    rep = layers[-1][0, -1, :]
                elif pooling_type == "mean":
                    rep = layers[-1][0].mean(dim=0)
                elif pooling_type == "middle_layer":
                    mid_idx = len(layers) // 2
                    rep = layers[mid_idx][0, -1, :]
                elif pooling_type == "multi_layer":
                    stacked = torch.stack([layers[i][0] for i in [-1, -2, -3, -4]])
                    rep = stacked.mean(dim=0).mean(dim=0)
            test_queries.append(rep.cpu())
            test_labels.append(label)
            
    return {
        "train_x": torch.stack(train_queries), "train_y": torch.tensor(train_labels),
        "val_x": torch.stack(val_queries), "val_y": torch.tensor(val_labels),
        "test_x": torch.stack(test_queries), "test_y": torch.tensor(test_labels),
        "hidden_dim": hidden_dim
    }


# ---------------------------------------------------------------------------
# Evaluation Helpers
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
    t_val = 2.776445  # t_0.975 for df=4 (n=5 seeds)
    h = sem * t_val
    return mean, mean - h, mean + h


# ---------------------------------------------------------------------------
# Main Sweep Pipeline
# ---------------------------------------------------------------------------
def run_comparison_sweeps():
    parser = argparse.ArgumentParser(description="Multi-Model Stage 1 Probe")
    parser.add_argument("--models", type=str, nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M"],
                        help="List of HF model names to evaluate")
    parser.add_argument("--poolings", type=str, nargs="+", default=["last_token", "mean", "middle_layer", "multi_layer"],
                        help="List of pooling types to evaluate")
    args = parser.parse_args()
    
    results = {}
    
    for model_name in args.models:
        results[model_name] = {}
        for pool_type in args.poolings:
            try:
                data = extract_pooled_representations(model_name, pool_type)
            except Exception as e:
                print(f"[Error] Failed to extract from {model_name} with {pool_type}: {e}")
                continue
                
            ref_x = data["train_x"]
            ref_y = data["train_y"]
            
            # Baseline Raw 1-NN Accuracy
            raw_acc = evaluate_1nn_accuracy(lambda x: F.normalize(x, dim=-1), data["test_x"], data["test_y"], ref_x, ref_y)
            print(f"  -> Raw Cosine 1-NN Accuracy: {raw_acc*100:.2f}%")
            
            # Train Probe over 5 seeds
            seeds = [1000 * (i + 1) for i in range(5)]
            probe_accs = []
            
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                probe = ResidualMetricProbe(input_dim=data["hidden_dim"], hidden_dim=512, output_dim=128).to(DEVICE)
                opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
                
                for epoch in range(120):
                    probe.train()
                    z_train = probe(data["train_x"].to(DEVICE))
                    loss = supervised_contrastive_loss(z_train, data["train_y"].to(DEVICE))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                test_acc = evaluate_1nn_accuracy(probe, data["test_x"], data["test_y"], ref_x, ref_y)
                probe_accs.append(test_acc)
                
            mean, lcb, ucb = compute_student_t_interval(probe_accs)
            results[model_name][pool_type] = {
                "raw_acc": raw_acc,
                "mean_acc": mean,
                "lcb": lcb,
                "ucb": ucb
            }
            print(f"  -> Probe Test Acc (Mean): {mean*100:.2f}% | 95% LCB: {lcb*100:.2f}%")
            
    # Print comparison summary table
    print("\n" + "="*80)
    print("  HEAD-TO-HEAD MODEL POOLING COMPARISON")
    print("="*80)
    for model_name in results:
        print(f"\nModel: {model_name}")
        print(f"  {'Pooling Type':<16} | {'Raw 1-NN':<10} | {'Probe Mean':<10} | {'95% LCB':<10}")
        print("  " + "─" * 52)
        for pool_type in results[model_name]:
            res = results[model_name][pool_type]
            print(f"  {pool_type:<16} | {res['raw_acc']*100:>8.1f}% | {res['mean_acc']*100:>8.1f}% | {res['lcb']*100:>8.1f}%")
    print("="*80)


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    run_comparison_sweeps()
