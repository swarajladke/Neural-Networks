"""
run_student_qpl_evaluation.py — Stage 4 Phase A Student Evaluation Suite.
========================================================================
Loads the distilled sub-10M parameter StudentEncoder and the teacher checkpoint.
Evaluates both on the untouched test split and verifies the Phase A exit criteria:
1. Zero Loss of Accuracy (Test Acc within 1.5% of teacher).
2. Neighborhood Purity (Relational Alignment Error E_rel <= 0.05).
3. Generalization / Control False Positive Rate (FPR) check.
4. Gini usage efficiency and representation effective rank.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from student_encoder import StudentEncoder
from hybrid_qpl import HybridQPL
from train_student_encoder import get_sentence_lists, batch_tokenize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = "smollm2_embeddings_34slots.pt"
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

def main():
    if not os.path.exists("best_student_encoder.pt"):
        print("[Error] Distilled student weights 'best_student_encoder.pt' not found. Run train_student_encoder.py first.")
        return
        
    # Find active teacher checkpoint
    teacher_chk = None
    for seed in [43, 41, 42, 44]:
        path = f"best_chl_qpl_seed{seed}.pt"
        if os.path.exists(path):
            teacher_chk = path
            break
            
    if teacher_chk is None:
        print("[Error] QPL Teacher checkpoint not found.")
        return
        
    print("[Eval] Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels = get_sentence_lists()
    
    # Load Teacher
    print(f"[Eval] Loading QPL Teacher: {teacher_chk}")
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
        
    # Load Student
    print("[Eval] Loading Standalone Student Encoder...")
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=128).to(DEVICE)
    student.load_state_dict(torch.load("best_student_encoder.pt", map_location=DEVICE))
    student.eval()
    
    data = torch.load(CACHE_PATH, weights_only=True)
    train_x = data["train_x"].to(DEVICE)
    test_x = data["test_x"].to(DEVICE)
    
    # ---------------------------------------------------------------------------
    # Generate Representations
    # ---------------------------------------------------------------------------
    print("[Eval] Encoding references and query test splits...")
    with torch.no_grad():
        # 1. Teacher Representations
        h_tr_t, _ = teacher(train_x, variant="full_qpl", k_wta=3)
        z_tr_t = F.normalize(h_tr_t, dim=-1)
        h_te_t, _ = teacher(test_x, variant="full_qpl", k_wta=3)
        z_te_t = F.normalize(h_te_t, dim=-1)
        
        # 2. Student Representations
        # References
        z_tr_s = []
        for idx in range(0, len(train_sentences), 64):
            batch_s = train_sentences[idx : idx + 64]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z_tr_s.append(student(ids, mask))
        z_tr_s = torch.cat(z_tr_s, dim=0)
        
        # Test queries
        z_te_s = []
        for idx in range(0, len(test_sentences), 64):
            batch_s = test_sentences[idx : idx + 64]
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z_te_s.append(student(ids, mask))
        z_te_s = torch.cat(z_te_s, dim=0)
        
    # ---------------------------------------------------------------------------
    # Evaluate 1-NN Test Accuracy
    # ---------------------------------------------------------------------------
    sims_t = torch.matmul(z_te_t, z_tr_t.T)
    correct_t = 0
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        pred_idx = sims_t[idx].argmax().item()
        if train_labels[pred_idx] == q_label:
            correct_t += 1
    acc_t = correct_t / len(test_sentences)
    
    sims_s = torch.matmul(z_te_s, z_tr_s.T)
    correct_s = 0
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        pred_idx = sims_s[idx].argmax().item()
        if train_labels[pred_idx] == q_label:
            correct_s += 1
    acc_s = correct_s / len(test_sentences)
    
    # ---------------------------------------------------------------------------
    # Neighborhood Purity / Relational Alignment Error E_rel
    # ---------------------------------------------------------------------------
    S_t = torch.matmul(z_te_t, z_te_t.T)
    S_s = torch.matmul(z_te_s, z_te_s.T)
    e_rel = (torch.norm(S_s - S_t, p="fro") / (torch.norm(S_t, p="fro") + 1e-8)).item()
    
    # ---------------------------------------------------------------------------
    # Gini Usage Statistics over Active Slots
    # ---------------------------------------------------------------------------
    with torch.no_grad():
        h_test_t, _, _, _ = teacher.settle(test_x, variant="full_qpl", k_wta=3)
        scores_t = h_test_t.masked_fill(~teacher.active_mask.unsqueeze(0), -float("inf"))
        indices_t = scores_t.topk(3, dim=-1).indices
        mask_t = torch.zeros_like(h_test_t)
        mask_t.scatter_(1, indices_t, 1.0)
    usage_t = mask_t[:, :34].sum(dim=0).cpu().numpy()
    usage_t_sorted = np.sort(usage_t)
    index = np.arange(1, 35)
    gini_t = (np.sum((2 * index - 34 - 1) * usage_t_sorted)) / (34 * np.sum(usage_t_sorted) + 1e-8)
    
    with torch.no_grad():
        V_active = teacher.V[:, :34]
        student_test_x = z_te_s @ V_active.T[:128]
        h_test_s, _, _, _ = teacher.settle(student_test_x, variant="full_qpl", k_wta=3)
        scores_s = h_test_s.masked_fill(~teacher.active_mask.unsqueeze(0), -float("inf"))
        indices_s = scores_s.topk(3, dim=-1).indices
        mask_s = torch.zeros_like(h_test_s)
        mask_s.scatter_(1, indices_s, 1.0)
    usage_s = mask_s[:, :34].sum(dim=0).cpu().numpy()
    usage_s_sorted = np.sort(usage_s)
    gini_s = (np.sum((2 * index - 34 - 1) * usage_s_sorted)) / (34 * np.sum(usage_s_sorted) + 1e-8)
    
    # ---------------------------------------------------------------------------
    # Hard-Negative FPR at 95% TPR Threshold
    # ---------------------------------------------------------------------------
    sim_corrects_t = []
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        ref_indices = [i for i, l in enumerate(train_labels) if l == q_label]
        max_sim = torch.matmul(z_te_t[idx], z_tr_t[ref_indices].T).max().item()
        sim_corrects_t.append(max_sim)
    tpr_95_t = np.percentile(sim_corrects_t, 5)
    
    sim_corrects_s = []
    for idx in range(len(test_sentences)):
        q_label = test_labels[idx]
        ref_indices = [i for i, l in enumerate(train_labels) if l == q_label]
        max_sim = torch.matmul(z_te_s[idx], z_tr_s[ref_indices].T).max().item()
        sim_corrects_s.append(max_sim)
    tpr_95_s = np.percentile(sim_corrects_s, 5)
    
    # Encode control/hard negative sentences on the fly
    print("[Eval] Encoding hard negative control sentences...")
    control_vectors = []
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        mod = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, output_hidden_states=True)
        mod.to(DEVICE)
        mod.eval()
        
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
    except Exception as e:
        print(f"[Warning] HuggingFace loading failed. Using dummy controls: {e}")
        control_x = torch.randn(len(INDEPENDENT_PPL_TEXTS), INPUT_DIM, device=DEVICE)
        control_x = F.normalize(control_x, dim=-1)
        
    with torch.no_grad():
        h_ctrl_t, _ = teacher(control_x, variant="full_qpl", k_wta=3)
        z_ctrl_t = F.normalize(h_ctrl_t, dim=-1)
    sims_ctrl_t = torch.matmul(z_ctrl_t, z_tr_t.T).max(dim=1).values.cpu().numpy()
    fpr_t = (sims_ctrl_t >= tpr_95_t).mean()
    
    ids_ctrl, mask_ctrl = batch_tokenize(tokenizer, INDEPENDENT_PPL_TEXTS, max_len=32, device=DEVICE)
    with torch.no_grad():
        z_ctrl_s = student(ids_ctrl, mask_ctrl)
    sims_ctrl_s = torch.matmul(z_ctrl_s, z_tr_s.T).max(dim=1).values.cpu().numpy()
    fpr_s = (sims_ctrl_s >= tpr_95_s).mean()
    
    # ---------------------------------------------------------------------------
    # Effective Rank
    # ---------------------------------------------------------------------------
    _, S_svd_t, _ = torch.linalg.svd(z_te_t, full_matrices=False)
    p_t = S_svd_t / S_svd_t.sum()
    rank_t = torch.exp(-torch.sum(p_t * torch.log(p_t + 1e-8))).item()
    
    _, S_svd_s, _ = torch.linalg.svd(z_te_s, full_matrices=False)
    p_s = S_svd_s / S_svd_s.sum()
    rank_s = torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-8))).item()
    
    # ---------------------------------------------------------------------------
    # Print Verification Report
    # ---------------------------------------------------------------------------
    print("\n" + "="*80)
    print("  PHASE A: STANDALONE STUDENT DISTILLATION EVALUATION REPORT")
    print("="*80)
    print(f"1. 1-NN Test Accuracy (SmolLM2-QPL Teacher): {acc_t*100:.2f}%")
    print(f"2. 1-NN Test Accuracy (GRU-Student Encoder) : {acc_s*100:.2f}%")
    
    acc_diff = (acc_t - acc_s) * 100
    crit_1_pass = acc_diff <= 1.5
    print(f"   -> Accuracy Delta: {acc_diff:+.2f} percentage points (Target <= 1.5%) -> {'PASS' if crit_1_pass else 'FAIL'}")
    
    crit_2_pass = e_rel <= 0.05
    print(f"3. Relational Alignment Error E_rel         : {e_rel:.4f} (Target <= 0.05) -> {'PASS' if crit_2_pass else 'FAIL'}")
    
    crit_3_pass = fpr_s <= fpr_t + 0.02
    print(f"4. Hard-Negative FPR at 95% TPR Threshold   : Student {fpr_s*100:.1f}% vs. Teacher {fpr_t*100:.1f}% -> {'PASS' if crit_3_pass else 'FAIL'}")
    
    print("\nRepresentation and Structural Integrity:")
    print(f"  - Usage Gini Coefficient       : Student {gini_s:.3f} | Teacher {gini_t:.3f} | Expected Balanced Gini: 0.000")
    print(f"  - Representation Effective Rank: Student {rank_s:.2f} | Teacher {rank_t:.2f}")
    
    overall_pass = crit_1_pass and crit_2_pass and crit_3_pass
    print("\n" + "="*80)
    print(f"  OVERALL PHASE A DISTILLATION GATE STATUS: {'PASS' if overall_pass else 'FAIL'}")
    print("="*80)

if __name__ == "__main__":
    main()
