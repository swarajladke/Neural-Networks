"""
train_student_encoder.py — Phase A Student Encoder Distillation Training.
==========================================================================
Loads the pre-trained SmolLM2 tokenizer and training dataset. Loads the best CHL-trained
QPL teacher checkpoint, generates teacher target representations, and distills them
into the sub-10M parameter StudentEncoder using a multi-objective loss function.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from student_encoder import StudentEncoder
from hybrid_qpl import HybridQPL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
INPUT_DIM = 960
OUTPUT_DIM = 128

# ---------------------------------------------------------------------------
# Reconstruct Sentence Lists and Labels
# ---------------------------------------------------------------------------
def get_sentence_lists():
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b if fact["category"] == "geography"]
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
        # Train
        for idx_t in range(3):
            train_sentences.append(get_prompt_only(fact, idx_t))
            train_labels.append(label)
        # Val
        val_sentences.append(fact["train_paraphrases"][-1])
        val_labels.append(label)
        # Test
        all_eval_items = [fact["probe"]] + fact["eval_paraphrases"]
        for item in all_eval_items:
            test_sentences.append(item)
            test_labels.append(label)
            
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

# ---------------------------------------------------------------------------
# Tokenization Helper
# ---------------------------------------------------------------------------
def batch_tokenize(tokenizer, sentences, max_len=32, device="cpu"):
    enc = tokenizer(
        sentences,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)

# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (InfoNCE)
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
# Validation Evaluator
# ---------------------------------------------------------------------------
def evaluate_student(student, tokenizer, train_sentences, train_labels, val_sentences, val_labels, batch_size=64):
    student.eval()
    with torch.no_grad():
        # Encode references
        z_refs = []
        for idx in range(0, len(train_sentences), batch_size):
            batch_s = train_sentences[idx : idx + batch_size]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z = student(ids, mask)
            z_refs.append(z.cpu())
        z_refs = torch.cat(z_refs, dim=0)
        
        # Encode validation queries
        z_queries = []
        for idx in range(0, len(val_sentences), batch_size):
            batch_s = val_sentences[idx : idx + batch_size]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z = student(ids, mask)
            z_queries.append(z.cpu())
        z_queries = torch.cat(z_queries, dim=0)
        
    sims = torch.matmul(z_queries, z_refs.T)
    correct = 0
    for idx in range(len(val_sentences)):
        q_label = val_labels[idx]
        pred_idx = sims[idx].argmax().item()
        if train_labels[pred_idx] == q_label:
            correct += 1
    return correct / len(val_sentences)

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(CACHE_PATH) or not os.path.exists(DATASET_PATH):
        print(f"[Error] Required files not found. Ensure '{CACHE_PATH}' and '{DATASET_PATH}' exist.")
        return
        
    print("[Distill] Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_sentences, train_labels, val_sentences, val_labels, _, _ = get_sentence_lists()
    
    # Find and Load Teacher Checkpoint
    teacher_chk = None
    for seed in [43, 41, 42, 44]:
        path = f"best_chl_qpl_seed{seed}.pt"
        if os.path.exists(path):
            teacher_chk = path
            break
            
    if teacher_chk is None:
        print("[Error] No trained teacher checkpoint found. Please run run_qpl_stage4_final_test.py first.")
        return
        
    print(f"[Distill] Loading QPL Teacher Checkpoint: {teacher_chk}")
    teacher_weights = torch.load(teacher_chk, map_only=True if torch.cuda.is_available() else False)
    
    # Reconstruct teacher QPL representation targets
    teacher_qpl = HybridQPL(input_dim=INPUT_DIM, output_dim=128).to(DEVICE)
    with torch.no_grad():
        teacher_qpl.V.copy_(teacher_weights["V"].to(DEVICE))
        teacher_qpl.b_in.copy_(teacher_weights["b_in"].to(DEVICE))
        teacher_qpl.W.copy_(teacher_weights["W"].to(DEVICE))
        teacher_qpl.b_out.copy_(teacher_weights["b_out"].to(DEVICE))
        teacher_qpl.L.copy_(teacher_weights["L"].to(DEVICE))
        teacher_qpl.active_mask.zero_()
        teacher_qpl.active_mask[:34] = True
        
    data = torch.load(CACHE_PATH, weights_only=True)
    train_x = data["train_x"].to(DEVICE)
    val_x = data["val_x"].to(DEVICE)
    
    print("[Distill] Generating teacher target coordinates...")
    with torch.no_grad():
        h_train, _ = teacher_qpl(train_x, variant="full_qpl", k_wta=3)
        z_train_teacher = F.normalize(h_train, dim=-1)
        
        h_val, _ = teacher_qpl(val_x, variant="full_qpl", k_wta=3)
        z_val_teacher = F.normalize(h_val, dim=-1)
        
    # Initialize Student Encoder
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    
    best_val_acc = 0.0
    num_epochs = 120
    batch_size = 64
    N = len(train_sentences)
    
    print(f"[Distill] Training Student Encoder for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        student.train()
        indices = list(range(N))
        random.shuffle(indices)
        
        epoch_loss = 0.0
        for idx in range(0, N, batch_size):
            batch_indices = indices[idx : idx + batch_size]
            batch_s = [train_sentences[i] for i in batch_indices]
            batch_y = torch.tensor([train_labels[i] for i in batch_indices], dtype=torch.long, device=DEVICE)
            
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            
            z_s = student(ids, mask)
            z_t = z_train_teacher[batch_indices]
            
            # Loss 1: Coordinate Alignment (Cosine Distance)
            loss_coord = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
            
            # Loss 2: Relational Preservation (Scale-Invariant Frobenius Error)
            S_s = torch.matmul(z_s, z_s.T)
            S_t = torch.matmul(z_t, z_t.T)
            loss_rel = torch.norm(S_s - S_t, p="fro") / (torch.norm(S_t, p="fro") + 1e-8)
            
            # Loss 3: Paraphrase Contrastive Loss (InfoNCE)
            loss_para = supervised_contrastive_loss(z_s, batch_y)
            
            # Joint Objective
            loss = 1.0 * loss_coord + 0.5 * loss_rel + 0.1 * loss_para
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_indices)
            
        scheduler.step()
        
        # Evaluate validation performance
        val_acc = evaluate_student(student, tokenizer, train_sentences, train_labels, val_sentences, val_labels, batch_size=batch_size)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), "best_student_encoder.pt")
            
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:03d} | Avg Loss: {epoch_loss/N:.4f} | Val 1-NN Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")
            
    print(f"\n[Distill] Training Complete. Best Student Validation Accuracy: {best_val_acc*100:.2f}%")
    print("Saved best student weights to 'best_student_encoder.pt'.")

if __name__ == "__main__":
    main()
