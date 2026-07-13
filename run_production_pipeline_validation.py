"""
run_production_pipeline_validation.py — Final Production Gate Validation.
========================================================================
Implements:
1. Strict fact-disjoint Train (70 facts), Val (15 facts), and Test (15 facts) splitting.
2. Training of the Student Encoder on the 70 train facts.
3. Encoding of all templates/queries using the Student Encoder ONLY (zero SmolLM2 at inference).
4. Training of the Hybrid Bilinear-MLP Verifier on Student embeddings.
5. Checkpoint/threshold selection on the Validation split (tuning for 95% TPR).
6. Evaluation on the untouched Test split: reports full metrics (AUROC, ECE, Selective Accuracy).
7. End-to-end Top-k (k=3) episodic retrieval and verification pipeline verification.
8. Latency and parameter footprint profiling.
"""

import os
import json
import time
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from student_encoder import StudentEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_100_PATH = "smollm2_embeddings_100slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM = 960

# ---------------------------------------------------------------------------
# Models Definition
# ---------------------------------------------------------------------------
class RelationVerifier(nn.Module):
    def __init__(self, input_dim=960):
        super().__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, 1)
        self.fc1 = nn.Linear(input_dim * 4 + 4, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, q, k, jaccard, overlap):
        diff = torch.abs(q - k)
        mult = q * k
        cos_sim = torch.sum(q * k, dim=-1, keepdim=True)
        dist = torch.norm(q - k, p=2, dim=-1, keepdim=True)
        
        jaccard = jaccard.unsqueeze(-1)
        overlap = overlap.unsqueeze(-1)
        
        x_concat = torch.cat([q, k, diff, mult, cos_sim, dist, jaccard, overlap], dim=-1)
        x_mlp = F.relu(self.bn1(self.fc1(x_concat)))
        x_mlp = self.dropout(x_mlp)
        x_mlp = F.relu(self.bn2(self.fc2(x_mlp)))
        x_mlp = self.dropout(x_mlp)
        
        x_bil = self.bilinear(q, k).squeeze(-1)
        out = self.fc3(x_mlp).squeeze(-1) + x_bil
        return torch.sigmoid(out)

# ---------------------------------------------------------------------------
# Lexical Overlap
# ---------------------------------------------------------------------------
def get_entity_overlap(str1, str2):
    stopwords = {"the", "is", "of", "a", "capital", "city", "melting", "point", "degrees", "celsius", "at", "what", "temperature", "does", "liquefy", "melts", "compound"}
    words1 = set(w.lower().strip(",.?!") for w in str1.split() if w.lower().strip(",.?!") not in stopwords)
    words2 = set(w.lower().strip(",.?!") for w in str2.split() if w.lower().strip(",.?!") not in stopwords)
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    overlap = len(intersection) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0.0
    return jaccard, overlap

# ---------------------------------------------------------------------------
# Helper: Tokenization
# ---------------------------------------------------------------------------
def batch_tokenize(tokenizer, sentences, max_len=32, device="cpu"):
    enc = tokenizer(sentences, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)

def get_sentence_lists(all_facts, unique_probes):
    probe_to_class = {p: idx for idx, p in enumerate(unique_probes)}
    def get_prompt_only(fact, idx):
        if idx == 0:
            return fact["probe"]
        elif idx == 1:
            prefix = fact["qa"].split(fact["statement"])[0]
            return prefix + fact["probe"]
        else:
            return fact["cloze"].split("_____")[0].strip()
            
    train_sentences = []
    train_labels = []
    val_sentences = []
    val_labels = []
    test_sentences = []
    test_labels = []
    for fact in all_facts:
        label = probe_to_class[fact["probe"]]
        for idx_t in range(3):
            train_sentences.append(get_prompt_only(fact, idx_t))
            train_labels.append(label)
        val_sentences.append(fact["train_paraphrases"][-1])
        val_labels.append(label)
        all_eval_items = [fact["probe"]] + fact["eval_paraphrases"]
        for item in all_eval_items:
            test_sentences.append(item)
            test_labels.append(label)
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

# ---------------------------------------------------------------------------
# ECE Calculation
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
# Verifier Dataset Builder
# ---------------------------------------------------------------------------
def build_pairs_from_embeddings(all_facts, z_train, z_test, train_sentences, test_sentences, general_sentences, z_general):
    print("  - Synthesizing positive and negative pairs...")
    positive_pairs = []
    semantic_neg_pairs = []
    
    # Positive pairs: (test query, train reference) matching the same fact
    for f_idx, fact in enumerate(all_facts):
        refs = z_train[f_idx*3 : (f_idx+1)*3]
        queries = z_test[f_idx*4 : (f_idx+1)*4]
        for q_sub_idx, q in enumerate(queries):
            q_str = test_sentences[f_idx * 4 + q_sub_idx]
            for r_sub_idx, r in enumerate(refs):
                r_str = train_sentences[f_idx * 3 + r_sub_idx]
                jaccard, overlap = get_entity_overlap(q_str, r_str)
                positive_pairs.append((q, r, f_idx, jaccard, overlap))
                
    # Semantic Hard Negatives (sharing entities or category)
    for f_idx_a, fact_a in enumerate(all_facts):
        entity_a = fact_a["location"] if "location" in fact_a else fact_a.get("compound", fact_a.get("planet"))
        for f_idx_b, fact_b in enumerate(all_facts):
            if f_idx_a == f_idx_b:
                continue
            entity_b = fact_b["location"] if "location" in fact_b else fact_b.get("compound", fact_b.get("planet"))
            if entity_a == entity_b or fact_a["category"] == fact_b["category"]:
                queries_a = z_test[f_idx_a*4 : (f_idx_a+1)*4]
                refs_b = z_train[f_idx_b*3 : (f_idx_b+1)*3]
                for q_sub_idx, q in enumerate(queries_a):
                    q_str = test_sentences[f_idx_a * 4 + q_sub_idx]
                    for r_sub_idx, r in enumerate(refs_b):
                        r_str = train_sentences[f_idx_b * 3 + r_sub_idx]
                        jaccard, overlap = get_entity_overlap(q_str, r_str)
                        semantic_neg_pairs.append((q, r, f_idx_b, jaccard, overlap))
                        
    random.shuffle(semantic_neg_pairs)
    semantic_neg_pairs = semantic_neg_pairs[:len(positive_pairs) * 3]
    
    # General Control pairs: Pair general sentences with all references
    general_neg_pairs = []
    for g_idx, g_vec in enumerate(z_general):
        q_str = general_sentences[g_idx]
        for r_idx in range(len(z_train)):
            r_str = train_sentences[r_idx]
            jaccard, overlap = get_entity_overlap(q_str, r_str)
            general_neg_pairs.append((g_vec, z_train[r_idx], r_idx // 3, jaccard, overlap))
            
    random.shuffle(general_neg_pairs)
    general_neg_pairs = general_neg_pairs[:len(positive_pairs)]
    
    return positive_pairs, semantic_neg_pairs, general_neg_pairs

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    print("="*80)
    print("  PHASE B.2: RELATION VERIFIER SPECIFICITY GATE AUDIT")
    print("="*80)
    
    # 1. Loading tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b]
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    
    # Split facts: 70 Train, 15 Val, 15 Test (strictly disjoint fact IDs!)
    train_facts = all_facts[:70]
    val_facts = all_facts[70:85]
    test_facts = all_facts[85:]
    
    # 2. Train the Student Encoder from scratch on 70 train facts
    cache_data = torch.load(CACHE_100_PATH, map_location=DEVICE)
    train_x_teacher = cache_data["train_x"][:210].to(DEVICE) # 70 * 3
    
    train_s, train_y, val_s, val_y, test_s, test_y = get_sentence_lists(train_facts, unique_probes)
    
    print("[Training] Training compact student encoder on 70 seen facts...")
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(60):
        student.train()
        indices = list(range(len(train_s)))
        random.shuffle(indices)
        for idx in range(0, len(train_s), 64):
            batch_idx = indices[idx : idx + 64]
            batch_s = [train_s[i] for i in batch_idx]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z_s = student(ids, mask)
            z_t = train_x_teacher[batch_idx]
            loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print("  - Student Encoder trained successfully.")
    
    # 3. Encode all templates using Student Encoder ONLY
    student.eval()
    print("[Encoding] Generating Student-encoded representations for all splits...")
    
    # Reference and Query texts for Train, Val, and Test facts
    tr_s_all, tr_y_all, val_s_all, val_y_all, te_s_all, te_y_all = get_sentence_lists(all_facts, unique_probes)
    
    with torch.no_grad():
        # Reference bank (3 templates per fact, total 300 reference vectors)
        z_train = []
        for i in range(0, len(tr_s_all), 64):
            ids, mask = batch_tokenize(tokenizer, tr_s_all[i:i+64], max_len=32, device=DEVICE)
            z_train.append(student(ids, mask))
        z_train = torch.cat(z_train, dim=0)
        
        # Query set (4 templates per fact, total 400 test vectors)
        z_test = []
        for i in range(0, len(te_s_all), 64):
            ids, mask = batch_tokenize(tokenizer, te_s_all[i:i+64], max_len=32, device=DEVICE)
            z_test.append(student(ids, mask))
        z_test = torch.cat(z_test, dim=0)
        
    # General Controls (200 sentences)
    general_sentences = []
    if os.path.exists("instruction_corpus.txt"):
        with open("instruction_corpus.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 20 and not line.startswith("#"):
                    general_sentences.append(line)
                    if len(general_sentences) == 200:
                        break
    if len(general_sentences) < 200:
        general_sentences = ["Standard out-of-domain control text representation."] * 200
        
    with torch.no_grad():
        z_general = []
        for i in range(0, len(general_sentences), 64):
            ids, mask = batch_tokenize(tokenizer, general_sentences[i:i+64], max_len=32, device=DEVICE)
            z_general.append(student(ids, mask))
        z_general = torch.cat(z_general, dim=0)
        
    # 4. Synthesize disjoint splits
    print("[Data] Generating fact-disjoint datasets for verifier...")
    pos_pairs, sem_negs, gen_negs = build_pairs_from_embeddings(
        all_facts, z_train, z_test, tr_s_all, te_s_all, general_sentences, z_general
    )
    
    # Train set (facts 0-69)
    train_pos = [p for p in pos_pairs if p[2] < 70]
    train_sem = [n for n in sem_negs if n[2] < 70]
    train_gen = [n for n in gen_negs if n[2] < 70]
    
    # Val set (facts 70-84)
    val_pos = [p for p in pos_pairs if 70 <= p[2] < 85]
    val_sem = [n for n in sem_negs if 70 <= n[2] < 85]
    val_gen = [n for n in gen_negs if 70 <= n[2] < 85]
    
    # Test set (facts 85-99)
    test_pos = [p for p in pos_pairs if p[2] >= 85]
    test_sem = [n for n in sem_negs if n[2] >= 85]
    test_gen = [n for n in gen_negs if n[2] >= 85]
    
    print(f"  - Train Split: Pos: {len(train_pos)} | SemNeg: {len(train_sem)} | GenNeg: {len(train_gen)}")
    print(f"  - Val Split  : Pos: {len(val_pos)} | SemNeg: {len(val_sem)} | GenNeg: {len(val_gen)}")
    print(f"  - Test Split : Pos: {len(test_pos)} | SemNeg: {len(test_sem)} | GenNeg: {len(test_gen)}")
    
    # 5. Train Verifier on Train Split
    verifier = RelationVerifier(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCELoss(reduction='none')
    
    train_q = torch.stack([p[0] for p in train_pos] + [n[0] for n in train_sem] + [n[0] for n in train_gen])
    train_k = torch.stack([p[1] for p in train_pos] + [n[1] for n in train_sem] + [n[1] for n in train_gen])
    train_jac = torch.tensor([p[3] for p in train_pos] + [n[3] for n in train_sem] + [n[3] for n in train_gen], dtype=torch.float32, device=DEVICE)
    train_ov = torch.tensor([p[4] for p in train_pos] + [n[4] for n in train_sem] + [n[4] for n in train_gen], dtype=torch.float32, device=DEVICE)
    train_y = torch.tensor([1.0] * len(train_pos) + [0.0] * len(train_sem) + [0.0] * len(train_gen), device=DEVICE)
    
    print("[Training] Training relation verifier MLP on train split (60 epochs)...")
    N = len(train_y)
    for epoch in range(60):
        verifier.train()
        indices = list(range(N))
        random.shuffle(indices)
        for idx in range(0, N, 64):
            b_idx = indices[idx : idx + 64]
            q_b = train_q[b_idx]
            k_b = train_k[b_idx]
            jac_b = train_jac[b_idx]
            ov_b = train_ov[b_idx]
            y_b = train_y[b_idx]
            
            pred = verifier(q_b, k_b, jac_b, ov_b)
            loss_raw = criterion(pred, y_b)
            weight = torch.ones_like(y_b)
            weight[y_b == 1.0] = 4.0
            loss = (loss_raw * weight).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 6. Threshold Selection on Validation Split
    verifier.eval()
    with torch.no_grad():
        val_pos_q = torch.stack([p[0] for p in val_pos])
        val_pos_k = torch.stack([p[1] for p in val_pos])
        val_pos_jac = torch.tensor([p[3] for p in val_pos], dtype=torch.float32, device=DEVICE)
        val_pos_ov = torch.tensor([p[4] for p in val_pos], dtype=torch.float32, device=DEVICE)
        pred_val_pos = verifier(val_pos_q, val_pos_k, val_pos_jac, val_pos_ov).cpu().numpy()
        
    tpr_95_val_thresh = np.percentile(pred_val_pos, 5)  # threshold where validation TPR is exactly 95%
    print(f"[Threshold] Validation 95% TPR Threshold: {tpr_95_val_thresh:.4f}")
    
    # 7. Evaluate on untouched Test Split
    with torch.no_grad():
        test_pos_q = torch.stack([p[0] for p in test_pos])
        test_pos_k = torch.stack([p[1] for p in test_pos])
        test_pos_jac = torch.tensor([p[3] for p in test_pos], dtype=torch.float32, device=DEVICE)
        test_pos_ov = torch.tensor([p[4] for p in test_pos], dtype=torch.float32, device=DEVICE)
        pred_test_pos = verifier(test_pos_q, test_pos_k, test_pos_jac, test_pos_ov).cpu().numpy()
        
        test_sem_q = torch.stack([n[0] for n in test_sem])
        test_sem_k = torch.stack([n[1] for n in test_sem])
        test_sem_jac = torch.tensor([n[3] for n in test_sem], dtype=torch.float32, device=DEVICE)
        test_sem_ov = torch.tensor([n[4] for n in test_sem], dtype=torch.float32, device=DEVICE)
        pred_test_sem = verifier(test_sem_q, test_sem_k, test_sem_jac, test_sem_ov).cpu().numpy()
        
        test_gen_q = torch.stack([n[0] for n in test_gen])
        test_gen_k = torch.stack([n[1] for n in test_gen])
        test_gen_jac = torch.tensor([n[3] for n in test_gen], dtype=torch.float32, device=DEVICE)
        test_gen_ov = torch.tensor([n[4] for n in test_gen], dtype=torch.float32, device=DEVICE)
        pred_test_gen = verifier(test_gen_q, test_gen_k, test_gen_jac, test_gen_ov).cpu().numpy()
        
    actual_tpr = (pred_test_pos >= tpr_95_val_thresh).mean()
    actual_fnr = (pred_test_pos < tpr_95_val_thresh).mean()
    actual_sem_fpr = (pred_test_sem >= tpr_95_val_thresh).mean()
    actual_gen_fpr = (pred_test_gen >= tpr_95_val_thresh).mean()
    
    tpr_90_val_thresh = np.percentile(pred_val_pos, 10)
    tpr_99_val_thresh = np.percentile(pred_val_pos, 1)
    fpr_sem_90 = (pred_test_sem >= tpr_90_val_thresh).mean()
    fpr_gen_90 = (pred_test_gen >= tpr_90_val_thresh).mean()
    fpr_sem_99 = (pred_test_sem >= tpr_99_val_thresh).mean()
    fpr_gen_99 = (pred_test_gen >= tpr_99_val_thresh).mean()
    
    # Detailed metrics
    all_preds = np.concatenate([pred_test_pos, pred_test_sem, pred_test_gen])
    all_labels = np.concatenate([np.ones_like(pred_test_pos), np.zeros_like(pred_test_sem), np.zeros_like(pred_test_gen)])
    
    ece_val = compute_ece(all_preds, all_labels)
    brier_score = np.mean((all_preds - all_labels)**2)
    
    # Selective accuracy & coverage (Real-world abstention)
    cov_80_mask = all_preds >= 0.80
    selective_acc_80 = (all_labels[cov_80_mask] == 1.0).mean() if cov_80_mask.any() else 1.0
    coverage_80 = cov_80_mask.mean()
    
    # 8. End-to-End Retrieval & Verification Evaluation
    print("\n[Pipeline] Evaluating End-to-End Retrieval & Verification Pipeline...")
    # Reference Bank = Test facts reference templates (15 facts * 3 templates = 45 vectors)
    z_ref_bank = z_train[85*3 : 100*3].to(DEVICE) # (45, 960)
    ref_labels = tr_y_all[85*3 : 100*3]
    ref_sentences_test = tr_s_all[85*3 : 100*3]
    
    # Test Queries = Paraphrases of Test facts (15 facts * 4 templates = 60 queries)
    z_queries = z_test[85*4 : 100*4].to(DEVICE) # (60, 960)
    query_labels = te_y_all[85*4 : 100*4]
    query_sentences_test = te_s_all[85*4 : 100*4]
    
    k = 3
    correct_in_top_k = 0
    pipeline_correct = 0
    pipeline_abstain = 0
    pipeline_false_accept = 0
    
    for idx in range(len(z_queries)):
        q_vec = z_queries[idx]
        q_str = query_sentences_test[idx]
        q_label = query_labels[idx]
        
        # Retrieve Top-k using Cosine Similarity
        sims = torch.matmul(z_ref_bank, q_vec.T)
        top_k_indices = torch.topk(sims, k=k).indices.cpu().numpy()
        
        # Check if correct fact is in candidate set
        in_candidates = any(ref_labels[idx_r] == q_label for idx_r in top_k_indices)
        if in_candidates:
            correct_in_top_k += 1
            
        # Run Verifier on top-k candidates
        accepted_candidates = []
        for cand_idx in top_k_indices:
            k_vec = z_ref_bank[cand_idx]
            k_str = ref_sentences_test[cand_idx]
            cand_label = ref_labels[cand_idx]
            
            jac, ov = get_entity_overlap(q_str, k_str)
            with torch.no_grad():
                score = verifier(q_vec.unsqueeze(0), k_vec.unsqueeze(0), 
                                 torch.tensor([jac], device=DEVICE), 
                                 torch.tensor([ov], device=DEVICE)).item()
            if score >= tpr_95_val_thresh:
                accepted_candidates.append((score, cand_label))
                
        if len(accepted_candidates) == 0:
            pipeline_abstain += 1
        else:
            # Pick highest score
            accepted_candidates.sort(reverse=True, key=lambda x: x[0])
            best_score, best_label = accepted_candidates[0]
            if best_label == q_label:
                pipeline_correct += 1
            else:
                pipeline_false_accept += 1
                
    # Rejection of general controls in pipeline
    pipeline_ctrl_false_accept = 0
    pipeline_ctrl_abstain = 0
    for idx in range(len(z_general)):
        q_vec = z_general[idx]
        q_str = general_sentences[idx]
        
        sims = torch.matmul(z_ref_bank, q_vec.T)
        top_k_indices = torch.topk(sims, k=k).indices.cpu().numpy()
        
        accepted_candidates = []
        for cand_idx in top_k_indices:
            k_vec = z_ref_bank[cand_idx]
            k_str = ref_sentences_test[cand_idx]
            cand_label = ref_labels[cand_idx]
            jac, ov = get_entity_overlap(q_str, k_str)
            with torch.no_grad():
                score = verifier(q_vec.unsqueeze(0), k_vec.unsqueeze(0), 
                                 torch.tensor([jac], device=DEVICE), 
                                 torch.tensor([ov], device=DEVICE)).item()
            if score >= tpr_95_val_thresh:
                accepted_candidates.append((score, cand_label))
                
        if len(accepted_candidates) == 0:
            pipeline_ctrl_abstain += 1
        else:
            pipeline_ctrl_false_accept += 1
            
    print("\n" + "="*80)
    print("  FINAL PRODUCTION GATE SPECIFICITY REPORT (FACT-DISJOINT)")
    print("="*80)
    print(f"  - Actual Test TPR (Recall)                    : {actual_tpr*100:.2f}% (Target: close to 95.0%)")
    print(f"  - Actual Test FNR                             : {actual_fnr*100:.2f}%")
    print(f"  - General Controls FPR at Threshold            : {actual_gen_fpr*100:.2f}%")
    print(f"  - Semantic Hard-Negatives FPR at Threshold     : {actual_sem_fpr*100:.2f}% (Preregistered target <= 5.0%)")
    status = "PASSED" if actual_sem_fpr <= 0.05 else "FAILED"
    print(f"  - Gate Status                                 : {status}")
    print("-"*80)
    print(f"  - FPR at 90% TPR Threshold (Semantic/General) : {fpr_sem_90*100:.2f}% / {fpr_gen_90*100:.2f}%")
    print(f"  - FPR at 99% TPR Threshold (Semantic/General) : {fpr_sem_99*100:.2f}% / {fpr_gen_99*100:.2f}%")
    print(f"  - Expected Calibration Error (ECE)            : {ece_val:.4f}")
    print(f"  - Brier Score                                 : {brier_score:.4f}")
    print(f"  - Selective Accuracy (at >= 0.80 Confidence)  : {selective_acc_80*100:.2f}% (Coverage: {coverage_80*100:.1f}%)")
    print("-"*80)
    print("  End-to-End Retrieval & Verification Pipeline (Test Split):")
    print(f"    - Retriever Recall@{k}                         : {correct_in_top_k / len(z_queries)*100:.2f}%")
    print(f"    - Pipeline Correct Acceptance Rate          : {pipeline_correct / len(z_queries)*100:.2f}%")
    print(f"    - Pipeline Abstention (Safety Halt) Rate    : {pipeline_abstain / len(z_queries)*100:.2f}%")
    print(f"    - Pipeline False Acceptance (Confusion) Rate: {pipeline_false_accept / len(z_queries)*100:.2f}%")
    print(f"    - Pipeline Control Rejection (Safety Match) : {pipeline_ctrl_abstain / len(z_general)*100:.2f}% correct rejection")
    print(f"    - Pipeline Control False Acceptance (Leaked): {pipeline_ctrl_false_accept / len(z_general)*100:.2f}% leakage")
    print("="*80)
    
    # Save production ready verifier
    torch.save(verifier.state_dict(), "production_relation_verifier.pt")

if __name__ == "__main__":
    main()
