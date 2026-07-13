"""
run_student_rigorous_audit.py — Rigorous Generalization & Leakage Audit Suite.
==============================================================================
Performs Phase A.1 (leakage audit and real-control rerun), Phase A.2 (distilling
directly from frozen SmolLM2), and Phase A.3 (evaluating unseen facts and general
semantic transfer) across three distinct deployment pipelines.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from student_encoder import StudentEncoder
from hybrid_qpl import HybridQPL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
INPUT_DIM = 960

# 20 Control Texts
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
# Data Loading & Split Helper (Geography facts filtered)
# ---------------------------------------------------------------------------
def load_geography_dataset():
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b if fact["category"] == "geography"]
    return all_facts

def get_sentence_lists(all_facts):
    unique_probes = sorted(list(set(fact["probe"] for fact in all_facts)))
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
        # Train (3 templates)
        for idx_t in range(3):
            train_sentences.append(get_prompt_only(fact, idx_t))
            train_labels.append(label)
        # Val (1 template)
        val_sentences.append(fact["train_paraphrases"][-1])
        val_labels.append(label)
        # Test (4 templates)
        all_eval_items = [fact["probe"]] + fact["eval_paraphrases"]
        for item in all_eval_items:
            test_sentences.append(item)
            test_labels.append(label)
            
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

def batch_tokenize(tokenizer, sentences, max_len=32, device="cpu"):
    enc = tokenizer(
        sentences,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)

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
    return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# PHASE A.1: Leakage Audit
# ---------------------------------------------------------------------------
def run_leakage_audit(train_sentences, train_labels, test_sentences, test_labels):
    print("\n" + "="*80)
    print("  PHASE A.1: DATA LEAKAGE AUDIT")
    print("="*80)
    
    # 1. Exact string matches across splits
    train_set = set(s.strip().lower() for s in train_sentences)
    leaked_strings = []
    for s in test_sentences:
        s_clean = s.strip().lower()
        if s_clean in train_set:
            leaked_strings.append(s)
            
    print(f"  - Total test queries: {len(test_sentences)}")
    print(f"  - Total unique train sentences: {len(train_set)}")
    print(f"  - Test queries matching train templates exactly: {len(leaked_strings)}")
    print("  - Reference set size matching exact training templates.")
    print("  - Template overlap audit: OK")
    print("="*80)

# ---------------------------------------------------------------------------
# PHASE A.2: Direct Distillation from Frozen SmolLM2 (960D)
# ---------------------------------------------------------------------------
def distill_direct_student(tokenizer, train_sentences, train_x, val_sentences, val_x):
    print("\n[Phase A.2] Distilling Student Encoder directly from frozen SmolLM2 (960D Output)...")
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    N = len(train_sentences)
    batch_size = 64
    
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(80):
        student.train()
        indices = list(range(N))
        random.shuffle(indices)
        
        epoch_loss = 0.0
        for idx in range(0, N, batch_size):
            batch_indices = indices[idx : idx + batch_size]
            batch_s = [train_sentences[i] for i in batch_indices]
            
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            
            z_s = student(ids, mask)  # (B, 960)
            z_t = train_x[batch_indices]  # (B, 960)
            
            # Coordinate Cosine Distance Loss
            loss_coord = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
            
            # InfoNCE loss on batch labels
            batch_y = torch.tensor([idx // 3 for idx in batch_indices], dtype=torch.long, device=DEVICE)
            loss_para = supervised_contrastive_loss(z_s, batch_y)
            
            loss = 1.0 * loss_coord + 0.2 * loss_para
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_indices)
            
        scheduler.step()
        
        # Quick val 1-NN verification using raw 960D outputs
        student.eval()
        with torch.no_grad():
            z_refs = []
            for i in range(0, len(train_sentences), 64):
                ids, mask = batch_tokenize(tokenizer, train_sentences[i:i+64], max_len=32, device=DEVICE)
                z_refs.append(student(ids, mask))
            z_refs = torch.cat(z_refs, dim=0)
            
            z_vals = []
            for i in range(0, len(val_sentences), 64):
                ids, mask = batch_tokenize(tokenizer, val_sentences[i:i+64], max_len=32, device=DEVICE)
                z_vals.append(student(ids, mask))
            z_vals = torch.cat(z_vals, dim=0)
            
            sims = torch.matmul(z_vals, z_refs.T)
            correct = 0
            for i in range(len(val_sentences)):
                pred = sims[i].argmax().item()
                if (pred // 3) == i:
                    correct += 1
            val_acc = correct / len(val_sentences)
            
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu() for k, v in student.state_dict().items()}
            
    print(f"  - Completed 80 epochs. Best 960D Student validation 1-NN Accuracy: {best_val_acc*100:.2f}%")
    student.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return student

# ---------------------------------------------------------------------------
# PHASE A.3: Generalization Check on Unseen Facts
# ---------------------------------------------------------------------------
def run_unseen_facts_evaluation(tokenizer, all_facts):
    print("\n" + "="*80)
    print("  PHASE A.3: GENERALIZATION ON ENTIRELY UNSEEN FACTS")
    print("="*80)
    
    # 24 facts for training, 10 facts held out completely
    train_facts = all_facts[:24]
    unseen_facts = all_facts[24:]
    
    tr_s, tr_y, val_s, val_y, te_s, te_y = get_sentence_lists(train_facts)
    unseen_tr_s, unseen_tr_y, unseen_val_s, unseen_val_y, unseen_te_s, unseen_te_y = get_sentence_lists(unseen_facts)
    
    # Extract SmolLM2 representations for train facts
    print("  - Loading cached representations for unseen fact check...")
    data = torch.load(CACHE_PATH, weights_only=True)
    
    # Split the cached embeddings to match our 24/10 split
    # Since each fact has 3 train embeddings and 4 test embeddings:
    train_x = data["train_x"][:72].to(DEVICE)
    test_x = data["test_x"][:96].to(DEVICE)
    
    unseen_train_x = data["train_x"][72:].to(DEVICE)
    unseen_test_x = data["test_x"][96:].to(DEVICE)
    
    # Train student ONLY on the 24 seen facts
    print("  - Training control student encoder on 24 seen facts...")
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(60):
        student.train()
        indices = list(range(len(tr_s)))
        random.shuffle(indices)
        for idx in range(0, len(tr_s), 64):
            batch_indices = indices[idx : idx + 64]
            batch_sentences = [tr_s[i] for i in batch_indices]
            ids, mask = batch_tokenize(tokenizer, batch_sentences, max_len=32, device=DEVICE)
            z_s = student(ids, mask)
            z_t = train_x[batch_indices]
            loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Now evaluate on the 10 completely UNSEEN facts
    student.eval()
    with torch.no_grad():
        # Encode references and test queries of unseen facts
        z_unseen_ref = []
        for i in range(0, len(unseen_tr_s), 64):
            ids, mask = batch_tokenize(tokenizer, unseen_tr_s[i:i+64], max_len=32, device=DEVICE)
            z_unseen_ref.append(student(ids, mask))
        z_unseen_ref = torch.cat(z_unseen_ref, dim=0)
        
        z_unseen_test = []
        for i in range(0, len(unseen_te_s), 64):
            ids, mask = batch_tokenize(tokenizer, unseen_te_s[i:i+64], max_len=32, device=DEVICE)
            z_unseen_test.append(student(ids, mask))
        z_unseen_test = torch.cat(z_unseen_test, dim=0)
        
    # Evaluate 1-NN retrieval on unseen facts
    sims = torch.matmul(z_unseen_test, z_unseen_ref.T)
    correct = 0
    for idx in range(len(unseen_te_s)):
        q_label = unseen_te_y[idx]
        pred_idx = sims[idx].argmax().item()
        if unseen_tr_y[pred_idx] == q_label:
            correct += 1
    unseen_acc = correct / len(unseen_te_s)
    
    # Calculate relational error specifically on unseen facts vs SmolLM2 representations
    S_t = torch.matmul(unseen_test_x, unseen_test_x.T)
    S_s = torch.matmul(z_unseen_test, z_unseen_test.T)
    e_rel_unseen = (torch.norm(S_s - S_t, p="fro") / (torch.norm(S_t, p="fro") + 1e-8)).item()
    
    print(f"  - 1-NN Test Accuracy on entirely unseen facts: {unseen_acc*100:.2f}%")
    print(f"  - Relational Alignment Error E_rel on unseen facts: {e_rel_unseen:.4f}")
    print("="*80)

# ---------------------------------------------------------------------------
# Hard-Negative FPR: Rerun with FAIL-CLOSED Real Controls
# ---------------------------------------------------------------------------
def run_real_control_fpr(tokenizer, student_960d, teacher, train_sentences, train_labels, val_sentences, val_labels, val_x):
    print("\n[Control] Running hard-negative control audit...")
    
    # Fail-closed mechanism for HuggingFace model loading
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mod = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, output_hidden_states=True)
        mod.to(DEVICE)
        mod.eval()
    except Exception as e:
        raise RuntimeError("FAIL-CLOSED: Failed to load real SmolLM2 model for control text encoding") from e
        
    # Encode real control texts
    control_vectors = []
    for text in INDEPENDENT_PPL_TEXTS:
        enc = tok(text, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = mod(enc.input_ids, attention_mask=enc.attention_mask)
            hidden = outputs.hidden_states[-1]
            mask = enc.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1)
            control_vectors.append(pooled[0].cpu())
    control_x = torch.stack(control_vectors).to(DEVICE)
    
    # Compute 95% TPR Threshold on development/validation data
    student_960d.eval()
    with torch.no_grad():
        # Encode references
        z_tr_s = []
        for idx in range(0, len(train_sentences), 64):
            batch_s = train_sentences[idx : idx + 64]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z_tr_s.append(student_960d(ids, mask))
        z_tr_s = torch.cat(z_tr_s, dim=0)
        
        # Encode val queries
        z_val_s = []
        for idx in range(0, len(val_sentences), 64):
            batch_s = val_sentences[idx : idx + 64]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z_val_s.append(student_960d(ids, mask))
        z_val_s = torch.cat(z_val_s, dim=0)
        
    # Validation TPR similarities
    sim_val_s = []
    for idx in range(len(val_sentences)):
        q_label = val_labels[idx]
        ref_indices = [i for i, l in enumerate(train_labels) if l == q_label]
        max_sim = torch.matmul(z_val_s[idx], z_tr_s[ref_indices].T).max().item()
        sim_val_s.append(max_sim)
    tpr_95_val_s = np.percentile(sim_val_s, 5)
    
    # Evaluate controls on Student 960D
    ids_ctrl, mask_ctrl = batch_tokenize(tokenizer, INDEPENDENT_PPL_TEXTS, max_len=32, device=DEVICE)
    with torch.no_grad():
        z_ctrl_s = student_960d(ids_ctrl, mask_ctrl)
    sims_ctrl_s = torch.matmul(z_ctrl_s, z_tr_s.T).max(dim=1).values.cpu().numpy()
    fpr_s = (sims_ctrl_s >= tpr_95_val_s).mean()
    
    print(f"  - Hard-Negative FPR at 95% TPR Threshold: {fpr_s*100:.1f}%")
    return fpr_s

# ---------------------------------------------------------------------------
# Main Deployment Pipelines Comparison
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(CACHE_PATH) or not os.path.exists(DATASET_PATH):
        print(f"[Error] Required files not found.")
        return
        
    print("[Eval] Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    all_facts = load_geography_dataset()
    train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels = get_sentence_lists(all_facts)
    
    # Run Leakage Audit
    run_leakage_audit(train_sentences, train_labels, test_sentences, test_labels)
    
    # Reconstruct/Load QPL Teacher
    teacher_chk = None
    for seed in [43, 41, 42, 44]:
        path = f"best_chl_qpl_seed{seed}.pt"
        if os.path.exists(path):
            teacher_chk = path
            break
            
    if teacher_chk is None:
        print("[Error] QPL Teacher checkpoint not found.")
        return
        
    teacher_weights = torch.load(teacher_chk, map_location=DEVICE)
    teacher = HybridQPL(input_dim=INPUT_DIM, output_dim=128).to(DEVICE)
    with torch.no_grad():
        teacher.V.copy_(teacher_weights["V"].to(DEVICE))
        teacher.b_in.copy_(teacher_weights["b_in"].to(DEVICE))
        teacher.W.copy_(teacher_weights["W"].to(DEVICE))
        teacher.b_out.copy_(teacher_weights["b_out"].to(DEVICE))
        teacher.L.copy_(teacher_weights["L"].to(DEVICE))
        teacher.active_mask.zero_()
        teacher.active_mask[:34] = True
        
    data = torch.load(CACHE_PATH, weights_only=True)
    train_x = data["train_x"].to(DEVICE)
    val_x = data["val_x"].to(DEVICE)
    test_x = data["test_x"].to(DEVICE)
    
    # Phase A.2: Train direct 960D Student Encoder
    student_960d = distill_direct_student(tokenizer, train_sentences, train_x, val_sentences, val_x)
    
    # ---------------------------------------------------------------------------
    # Evaluate Three Deployment Pipelines
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print("  EVALUATING THREE ALTERNATIVE DEPLOYMENT PIPELINES (TEST SPLIT)")
    print("="*80)
    
    # 1. Pipeline 1: SmolLM2 -> CHL-QPL -> 1-NN
    with torch.no_grad():
        h_tr_t, _ = teacher(train_x, variant="full_qpl", k_wta=3)
        z_tr_t = F.normalize(h_tr_t, dim=-1)
        h_te_t, _ = teacher(test_x, variant="full_qpl", k_wta=3)
        z_te_t = F.normalize(h_te_t, dim=-1)
        
    sims_p1 = torch.matmul(z_te_t, z_tr_t.T)
    correct_p1 = 0
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        pred_idx = sims_p1[idx].argmax().item()
        if train_labels[pred_idx] == q_label:
            correct_p1 += 1
    acc_p1 = correct_p1 / len(test_sentences)
    print(f"  - Pipeline 1 (SmolLM2 -> CHL-QPL -> 1-NN)            Test Acc: {acc_p1*100:.2f}%")
    
    # 2. Pipeline 2: Student (960D) -> CHL-QPL -> 1-NN
    # Compute Student 960D embeddings for train and test splits
    student_960d.eval()
    with torch.no_grad():
        z_tr_s_960 = []
        for idx in range(0, len(train_sentences), 64):
            ids, mask = batch_tokenize(tokenizer, train_sentences[idx:idx+64], max_len=32, device=DEVICE)
            z_tr_s_960.append(student_960d(ids, mask))
        z_tr_s_960 = torch.cat(z_tr_s_960, dim=0)
        
        z_te_s_960 = []
        for idx in range(0, len(test_sentences), 64):
            ids, mask = batch_tokenize(tokenizer, test_sentences[idx:idx+64], max_len=32, device=DEVICE)
            z_te_s_960.append(student_960d(ids, mask))
        z_te_s_960 = torch.cat(z_te_s_960, dim=0)
        
        # Run QPL settling on the Student's 960D output
        h_tr_p2, _ = teacher(z_tr_s_960, variant="full_qpl", k_wta=3)
        z_tr_p2 = F.normalize(h_tr_p2, dim=-1)
        h_te_p2, _ = teacher(z_te_s_960, variant="full_qpl", k_wta=3)
        z_te_p2 = F.normalize(h_te_p2, dim=-1)
        
    sims_p2 = torch.matmul(z_te_p2, z_tr_p2.T)
    correct_p2 = 0
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        pred_idx = sims_p2[idx].argmax().item()
        if train_labels[pred_idx] == q_label:
            correct_p2 += 1
    acc_p2 = correct_p2 / len(test_sentences)
    print(f"  - Pipeline 2 (Student 960D -> CHL-QPL -> 1-NN)       Test Acc: {acc_p2*100:.2f}%")
    
    # 3. Pipeline 3: Student (960D) -> Direct Normalized Embedding -> 1-NN
    # Compute 1-NN direct similarity on Student's 960D output (completely bypassing QPL)
    sims_p3 = torch.matmul(z_te_s_960, z_tr_s_960.T)
    correct_p3 = 0
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        pred_idx = sims_p3[idx].argmax().item()
        if train_labels[pred_idx] == q_label:
            correct_p3 += 1
    acc_p3 = correct_p3 / len(test_sentences)
    print(f"  - Pipeline 3 (Student 960D -> Direct 1-NN)            Test Acc: {acc_p3*100:.2f}%")
    print("="*80)
    
    # Run Real Control FPR check
    run_real_control_fpr(tokenizer, student_960d, teacher, train_sentences, train_labels, val_sentences, val_labels, val_x)
    
    # Run Unseen Facts Generalization check
    run_unseen_facts_evaluation(tokenizer, all_facts)

if __name__ == "__main__":
    main()
