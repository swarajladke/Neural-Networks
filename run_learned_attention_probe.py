"""
run_learned_attention_probe.py — Stage 1 Learned Attention Recoverability Probe
=============================================================================
Extracts full token-level sequence hidden states from the last layer of
frozen LLMs and trains a learned attention-pooling projection module
to ignore syntax template tokens and isolate fact-specific identifiers.
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
class AttentionPooledResidualMetricProbe(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, 1, bias=False)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_padded, mask):
        # x_padded: (B, T, D)
        # mask: (B, T) boolean
        attn_logits = self.query_proj(x_padded).squeeze(-1)  # (B, T)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, T)
        
        # Weighted sum: (B, D)
        x_pooled = torch.bmm(attn_weights.unsqueeze(1), x_padded).squeeze(1)
        
        z = self.skip(x_pooled) + self.body(x_pooled)
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
# Extract and Pad Sequences
# ---------------------------------------------------------------------------
def extract_unpooled_sequences(model_name):
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
    
    train_seqs = []
    train_labels = []
    val_seqs = []
    val_labels = []
    test_seqs = []
    test_labels = []
    
    def get_prompt_only(f, idx):
        if idx == 0:
            return f["probe"]
        elif idx == 1:
            prefix = f["qa"].split(f["statement"])[0]
            return prefix + f["probe"]
        else:
            return f["cloze"].split("_____")[0].strip()

    print("[Extract] Extracting unpooled token hidden states...")
    
    for idx_f, f in enumerate(all_facts):
        fid = f["id"]
        label = fid_to_idx[fid]
        
        # 1. Train Templates
        for idx_t in range(3):
            prompt = get_prompt_only(f, idx_t)
            ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(ids)
                # Hidden state of last layer: (1, seq_len, D)
                rep = outputs.hidden_states[-1][0].cpu().float()
            train_seqs.append(rep)
            train_labels.append(label)
            
        # 2. Validation Templates
        dev_item = f["train_paraphrases"][-1]
        item_ids = tokenizer.encode(dev_item, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(item_ids)
            rep = outputs.hidden_states[-1][0].cpu().float()
        val_seqs.append(rep)
        val_labels.append(label)
        
        # 3. Test Templates
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            item_ids = tokenizer.encode(item, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(item_ids)
                rep = outputs.hidden_states[-1][0].cpu().float()
            test_seqs.append(rep)
            test_labels.append(label)
            
    # Pad sequences to max length
    def pad_tensors(tensor_list):
        max_len = max([t.shape[0] for t in tensor_list])
        padded_tensors = []
        masks = []
        for t in tensor_list:
            len_t = t.shape[0]
            pad_size = max_len - len_t
            if pad_size > 0:
                padded = F.pad(t, (0, 0, 0, pad_size), "constant", 0.0)
            else:
                padded = t
            padded_tensors.append(padded)
            # True for active positions, False for padded
            mask = torch.cat([torch.ones(len_t, dtype=torch.bool), torch.zeros(pad_size, dtype=torch.bool)])
            masks.append(mask)
        return torch.stack(padded_tensors), torch.stack(masks)

    train_x, train_mask = pad_tensors(train_seqs)
    val_x, val_mask = pad_tensors(val_seqs)
    test_x, test_mask = pad_tensors(test_seqs)
    
    return {
        "train_x": train_x, "train_mask": train_mask, "train_y": torch.tensor(train_labels),
        "val_x": val_x, "val_mask": val_mask, "val_y": torch.tensor(val_labels),
        "test_x": test_x, "test_mask": test_mask, "test_y": torch.tensor(test_labels),
        "hidden_dim": hidden_dim
    }


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------
def evaluate_attention_1nn(model_fn, x_padded, mask, y_data, ref_x_padded, ref_mask, ref_y_data):
    model_fn.eval()
    with torch.no_grad():
        z_queries = model_fn(x_padded.to(DEVICE), mask.to(DEVICE))
        z_refs = model_fn(ref_x_padded.to(DEVICE), ref_mask.to(DEVICE))
        
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
    t_val = 2.776445  # df=4 for 5 seeds
    h = sem * t_val
    return mean, mean - h, mean + h


# ---------------------------------------------------------------------------
# Main Sweep Pipeline
# ---------------------------------------------------------------------------
def run_attention_sweeps():
    models = ["gpt2", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M"]
    results = {}
    
    for model_name in models:
        try:
            data = extract_unpooled_sequences(model_name)
        except Exception as e:
            print(f"[Error] Failed to extract from {model_name}: {e}")
            continue
            
        ref_x = data["train_x"]
        ref_mask = data["train_mask"]
        ref_y = data["train_y"]
        
        # Train Probe over 5 seeds
        seeds = [1000 * (i + 1) for i in range(5)]
        probe_accs = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            probe = AttentionPooledResidualMetricProbe(input_dim=data["hidden_dim"], hidden_dim=512, output_dim=128).to(DEVICE)
            opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
            
            for epoch in range(150):
                probe.train()
                z_train = probe(data["train_x"].to(DEVICE), data["train_mask"].to(DEVICE))
                loss = supervised_contrastive_loss(z_train, data["train_y"].to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            test_acc = evaluate_attention_1nn(probe, data["test_x"], data["test_mask"], data["test_y"], ref_x, ref_mask, ref_y)
            probe_accs.append(test_acc)
            
        mean, lcb, ucb = compute_student_t_interval(probe_accs)
        results[model_name] = {
            "mean_acc": mean,
            "lcb": lcb,
            "ucb": ucb
        }
        print(f"  -> {model_name} Learned Attention Test Acc (Mean): {mean*100:.2f}% | 95% LCB: {lcb*100:.2f}%")
        
    print("\n" + "="*80)
    print("  LEARNED ATTENTION POOLING PROBE SUMMARY")
    print("="*80)
    print(f"  {'Model Name':<30} | {'Probe Mean':<10} | {'95% LCB':<10}")
    print("  " + "-" * 56)
    for model_name in results:
        res = results[model_name]
        print(f"  {model_name:<30} | {res['mean_acc']*100:>8.1f}% | {res['lcb']*100:>8.1f}%")
    print("="*80)


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    run_attention_sweeps()
