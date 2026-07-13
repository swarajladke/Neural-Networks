"""
run_relation_verifier_training.py — Phase B.1 and B.2: Lightweight Semantic-Relation Verifier.
============================================================================================
Trains a lightweight relation verifier using query-candidate feature interaction
[q, k, |q - k|, q * k] to reject semantic hard negatives (relational collisions, entity swaps).
Target: Reduce semantic hard-negative FPR to <= 5% at 95% TPR.
"""

import os
import json
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from student_encoder import StudentEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_100_PATH = "smollm2_embeddings_100slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
INPUT_DIM = 960

# ---------------------------------------------------------------------------
# Lightweight Relation Verifier Model
# ---------------------------------------------------------------------------
class RelationVerifier(nn.Module):
    def __init__(self, input_dim=960):
        super().__init__()
        # We combine a bilinear scoring path and a concatenated MLP path
        self.bilinear = nn.Bilinear(input_dim, input_dim, 1)
        # Input size: 4 * input_dim + 2 (cat q, k, |q-k|, q*k, cos_sim, dist)
        self.fc1 = nn.Linear(input_dim * 4 + 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, q, k):
        # q: (B, input_dim), k: (B, input_dim)
        diff = torch.abs(q - k)
        mult = q * k
        cos_sim = torch.sum(q * k, dim=-1, keepdim=True)
        dist = torch.norm(q - k, p=2, dim=-1, keepdim=True)
        
        # Concat path
        x_concat = torch.cat([q, k, diff, mult, cos_sim, dist], dim=-1)
        x_mlp = F.relu(self.bn1(self.fc1(x_concat)))
        x_mlp = self.dropout(x_mlp)
        x_mlp = F.relu(self.bn2(self.fc2(x_mlp)))
        x_mlp = self.dropout(x_mlp)
        
        # Bilinear path
        x_bil = self.bilinear(q, k).squeeze(-1)
        
        # Combine paths
        out = self.fc3(x_mlp).squeeze(-1) + x_bil
        return torch.sigmoid(out)

# ---------------------------------------------------------------------------
# In-Domain Semantic Hard Negatives Builder for Pairs
# ---------------------------------------------------------------------------
def build_verifier_dataset(tokenizer, model, all_facts, cache_data, unique_probes):
    print("\n[Data] Synthesizing verifier dataset (Positive & Hard Negative pairs)...")
    
    probe_to_class = {p: idx for idx, p in enumerate(unique_probes)}
    
    # 1. Load train/test representations from cache
    train_x = cache_data["train_x"].to(DEVICE) # (300, 960)
    test_x = cache_data["test_x"].to(DEVICE)   # (400, 960)
    
    positive_pairs = []
    semantic_neg_pairs = []
    
    # Generate Positive Pairs: (paraphrase query test_x, reference train_x) for same fact
    for f_idx, fact in enumerate(all_facts):
        # references of this fact: f_idx*3, f_idx*3 + 1, f_idx*3 + 2
        refs = train_x[f_idx*3 : (f_idx+1)*3]
        # test queries of this fact: f_idx*4, f_idx*4 + 1, ..., f_idx*4 + 3
        queries = test_x[f_idx*4 : (f_idx+1)*4]
        
        for q in queries:
            for r in refs:
                positive_pairs.append((q, r, f_idx))
                
    # Generate Semantic Hard Negatives (Relational collisions / entity swaps)
    # Pair test queries of fact A with train references of fact B that shares the same entity but different relation
    for f_idx_a, fact_a in enumerate(all_facts):
        entity_a = fact_a["location"] if "location" in fact_a else fact_a.get("compound", fact_a.get("planet"))
        for f_idx_b, fact_b in enumerate(all_facts):
            if f_idx_a == f_idx_b:
                continue
            entity_b = fact_b["location"] if "location" in fact_b else fact_b.get("compound", fact_b.get("planet"))
            
            # If they share the same entity (or share category, e.g. chemistry vs chemistry)
            if entity_a == entity_b or fact_a["category"] == fact_b["category"]:
                # Relational collision: Query A paired with Reference B
                queries_a = test_x[f_idx_a*4 : (f_idx_a+1)*4]
                refs_b = train_x[f_idx_b*3 : (f_idx_b+1)*3]
                for q in queries_a:
                    for r in refs_b:
                        semantic_neg_pairs.append((q, r, f_idx_b))
                        
    # Shuffle and trim negatives to balance positives
    random.shuffle(semantic_neg_pairs)
    semantic_neg_pairs = semantic_neg_pairs[:len(positive_pairs) * 3]
    
    print(f"  - Positive pairs generated        : {len(positive_pairs)}")
    print(f"  - Semantic Hard-Negative pairs    : {len(semantic_neg_pairs)}")
    
    # 2. General controls from instruction_corpus.txt
    general_sentences = []
    if os.path.exists("instruction_corpus.txt"):
        with open("instruction_corpus.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 20 and not line.startswith("#"):
                    general_sentences.append(line)
                    if len(general_sentences) == 200:
                        break
    if len(general_sentences) < 100:
        general_sentences = ["Standard English sentence representing a general out-of-domain control example."] * 200
        
    general_vectors = []
    batch_size = 32
    for i in range(0, len(general_sentences), batch_size):
        batch_text = general_sentences[i : i + batch_size]
        enc = tokenizer(batch_text, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            hidden = outputs.hidden_states[-1]
            mask = enc.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1)
            for j in range(len(batch_text)):
                general_vectors.append(pooled[j])
    general_x = torch.stack(general_vectors).to(DEVICE)
    
    # Create General negative pairs: Pair general sentences with all train references
    general_neg_pairs = []
    for g_vec in general_x:
        for r_idx in range(len(train_x)):
            general_neg_pairs.append((g_vec, train_x[r_idx], r_idx // 3))
    random.shuffle(general_neg_pairs)
    general_neg_pairs = general_neg_pairs[:len(positive_pairs)]
    print(f"  - General Control pairs generated : {len(general_neg_pairs)}")
    
    return positive_pairs, semantic_neg_pairs, general_neg_pairs

# ---------------------------------------------------------------------------
# ECE calculation (Expected Calibration Error)
# ---------------------------------------------------------------------------
def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

# ---------------------------------------------------------------------------
# Training & Verification Gate Check
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Required dataset file {DATASET_PATH} not found.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        raise RuntimeError("FAIL-CLOSED: Failed to load real SmolLM2 model for verification generation") from e
        
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
        
    from run_student_continual_benchmarks import ensure_100_fact_embeddings
    cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    all_facts = [fact for b in blocks for fact in b]
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    
    # 1. Build pairs
    pos_pairs, sem_negs, gen_negs = build_verifier_dataset(tokenizer, model, all_facts, cache_data, unique_probes)
    
    # 2. Split into train & test (70/30 disjoint facts to verify generalization to unseen facts)
    # Train set uses pairs from first 70 facts, Test set uses pairs from last 30 facts
    train_pos = [p for p in pos_pairs if p[2] < 70]
    test_pos = [p for p in pos_pairs if p[2] >= 70]
    
    train_sem = [n for n in sem_negs if n[2] < 70]
    test_sem = [n for n in sem_negs if n[2] >= 70]
    
    train_gen = [n for n in gen_negs if n[2] < 70]
    test_gen = [n for n in gen_negs if n[2] >= 70]
    
    print(f"\n[Split] Split Train/Test:")
    print(f"  - Train: Positives: {len(train_pos)} | Semantic Negatives: {len(train_sem)} | General Negatives: {len(train_gen)}")
    print(f"  - Test : Positives: {len(test_pos)} | Semantic Negatives: {len(test_sem)} | General Negatives: {len(test_gen)}")
    
    # 3. Train Verifier
    verifier = RelationVerifier(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCELoss(reduction='none')
    
    # Build train tensor dataset
    train_q = torch.stack([p[0] for p in train_pos] + [n[0] for n in train_sem] + [n[0] for n in train_gen])
    train_k = torch.stack([p[1] for p in train_pos] + [n[1] for n in train_sem] + [n[1] for n in train_gen])
    train_y = torch.tensor([1.0] * len(train_pos) + [0.0] * len(train_sem) + [0.0] * len(train_gen), device=DEVICE)
    
    print("\n[Training] Training lightweight verifier MLP (60 epochs with class weighting)...")
    N = len(train_y)
    batch_size = 64
    for epoch in range(60):
        verifier.train()
        indices = list(range(N))
        random.shuffle(indices)
        epoch_loss = 0.0
        for idx in range(0, N, batch_size):
            b_idx = indices[idx : idx + batch_size]
            q_b = train_q[b_idx]
            k_b = train_k[b_idx]
            y_b = train_y[b_idx]
            
            pred = verifier(q_b, k_b)
            loss_elementwise = criterion(pred, y_b)
            weight = torch.ones_like(y_b)
            weight[y_b == 1.0] = 4.0
            loss = (loss_elementwise * weight).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_idx)
            
    # 4. Evaluation
    verifier.eval()
    with torch.no_grad():
        test_pos_q = torch.stack([p[0] for p in test_pos])
        test_pos_k = torch.stack([p[1] for p in test_pos])
        pred_pos = verifier(test_pos_q, test_pos_k).cpu().numpy()
        
        test_sem_q = torch.stack([n[0] for n in test_sem])
        test_sem_k = torch.stack([n[1] for n in test_sem])
        pred_sem = verifier(test_sem_q, test_sem_k).cpu().numpy()
        
        test_gen_q = torch.stack([n[0] for n in test_gen])
        test_gen_k = torch.stack([n[1] for n in test_gen])
        pred_gen = verifier(test_gen_q, test_gen_k).cpu().numpy()
        
    # Calculate TPR thresholds and corresponding FPRs
    # Select threshold where TPR is exactly 95% on test set
    tpr_95_thresh = np.percentile(pred_pos, 5)  # 5th percentile means 95% are above
    tpr_90_thresh = np.percentile(pred_pos, 10)
    tpr_99_thresh = np.percentile(pred_pos, 1)
    
    # Calculate FPRs
    fpr_sem_95 = (pred_sem >= tpr_95_thresh).mean()
    fpr_gen_95 = (pred_gen >= tpr_95_thresh).mean()
    
    fpr_sem_90 = (pred_sem >= tpr_90_thresh).mean()
    fpr_gen_90 = (pred_gen >= tpr_90_thresh).mean()
    
    fpr_sem_99 = (pred_sem >= tpr_99_thresh).mean()
    fpr_gen_99 = (pred_gen >= tpr_99_thresh).mean()
    
    # AUROC and AUPRC calculation
    all_preds = np.concatenate([pred_pos, pred_sem, pred_gen])
    all_labels = np.concatenate([np.ones_like(pred_pos), np.zeros_like(pred_sem), np.zeros_like(pred_gen)])
    
    # ECE (Expected Calibration Error)
    ece_val = compute_ece(all_preds, all_labels)
    
    # Coverage vs Selective Accuracy: Accuracy at varying verification confidence thresholds
    cov_50_mask = all_preds >= 0.50
    selective_acc_50 = (all_labels[cov_50_mask] == 1.0).mean() if cov_50_mask.any() else 1.0
    coverage_50 = cov_50_mask.mean()
    
    cov_80_mask = all_preds >= 0.80
    selective_acc_80 = (all_labels[cov_80_mask] == 1.0).mean() if cov_80_mask.any() else 1.0
    coverage_80 = cov_80_mask.mean()
    
    print("\n" + "="*80)
    print("  PHASE B.2: RELATION VERIFIER SPECIFICITY GATE REPORT")
    print("="*80)
    print(f"  - 95% TPR Verification Threshold               : {tpr_95_thresh:.4f}")
    print(f"  - General Controls FPR at 95% TPR               : {fpr_gen_95*100:.2f}%")
    print(f"  - Semantic Hard-Negatives FPR at 95% TPR        : {fpr_sem_95*100:.2f}% (Target <= 5.0%)")
    
    status = "PASSED" if fpr_sem_95 <= 0.05 else "FAILED"
    print(f"  - Specificity Gate Status                       : {status}")
    print("-"*80)
    
    print(f"  - FPR at 90% TPR                               : Semantic: {fpr_sem_90*100:.2f}% | General: {fpr_gen_90*100:.2f}%")
    print(f"  - FPR at 99% TPR                               : Semantic: {fpr_sem_99*100:.2f}% | General: {fpr_gen_99*100:.2f}%")
    print(f"  - Expected Calibration Error (ECE)             : {ece_val:.4f}")
    print(f"  - Coverage vs Selective Accuracy:")
    print(f"    - Threshold >= 0.50                          : Coverage: {coverage_50*100:.1f}% | Selective Accuracy: {selective_acc_50*100:.2f}%")
    print(f"    - Threshold >= 0.80                          : Coverage: {coverage_80*100:.1f}% | Selective Accuracy: {selective_acc_80*100:.2f}%")
    print("="*80)
    
    # Save verifier weights
    torch.save(verifier.state_dict(), "best_relation_verifier.pt")
    print("[Training] Saved relation verifier weights to best_relation_verifier.pt.")

if __name__ == "__main__":
    main()
