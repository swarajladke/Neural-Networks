import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_100_PATH = "smollm2_embeddings_100slots.pt" if os.path.exists("smollm2_embeddings_100slots.pt") else ("../smollm2_embeddings_100slots.pt" if os.path.exists("../smollm2_embeddings_100slots.pt") else "smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"

def find_offline_model_path():
    for path in ["../local_smollm2", "local_smollm2"]:
        if os.path.exists(os.path.join(path, "model.safetensors")):
            return path
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files):
                if "smollm" in root.lower():
                    return root
    return "HuggingFaceTB/SmolLM2-360M"

MODEL_ID = find_offline_model_path()
INPUT_DIM = 960

# ---------------------------------------------------------------------------
# Models Definition
# ---------------------------------------------------------------------------
class StudentEncoder(nn.Module):
    def __init__(self, vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, mask):
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)
        mask_expanded = mask.unsqueeze(-1)
        pooled = (outputs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
        out = self.fc(pooled)
        return F.normalize(out, dim=-1)

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
# Helpers
# ---------------------------------------------------------------------------
def batch_tokenize(tokenizer, sentences, max_len=32, device="cpu"):
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    enc = tokenizer(sentences, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)

def get_entity_overlap(q_str, k_str):
    q_words = set(q_str.lower().split())
    k_words = set(k_str.lower().split())
    intersection = q_words & k_words
    union = q_words | k_words
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    overlap = len(intersection) / len(q_words) if len(q_words) > 0 else 0.0
    return jaccard, overlap

def get_sentence_lists(all_facts):
    def get_prompt_only(f, idx):
        if idx == 0:
            return f["probe"]
        elif idx == 1:
            prefix = f["qa"].split(f["statement"])[0]
            return prefix + f["probe"]
        else:
            return f["cloze"].split("_____")[0].strip()

    train_s, test_s = [], []
    for f in all_facts:
        for idx in range(3):
            train_s.append(get_prompt_only(f, idx))
        test_s.append(f["probe"])
        test_s.extend(f["eval_paraphrases"][:3]) # 4 total test templates
    return train_s, test_s

# ---------------------------------------------------------------------------
def train_verifier_on_fly(cache_data):
    print("[Verifier] Training relation verifier on the fly for CL evaluation...")
    verifier = RelationVerifier(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 70 facts for training (facts 0 to 69)
    train_x = cache_data["train_x"][:210].to(DEVICE) # 70 facts x 3 refs = 210
    
    pos_q, pos_k = [], []
    neg_q, neg_k = [], []
    
    for f in range(70):
        for i in range(3):
            for j in range(3):
                if i != j:
                    pos_q.append(train_x[f*3 + i])
                    pos_k.append(train_x[f*3 + j])
        for _ in range(6):
            f_neg = random.choice([x for x in range(70) if x != f])
            pos_idx = random.randint(0, 2)
            neg_idx = random.randint(0, 2)
            neg_q.append(train_x[f*3 + pos_idx])
            neg_k.append(train_x[f_neg*3 + neg_idx])
            
    pos_q = torch.stack(pos_q)
    pos_k = torch.stack(pos_k)
    neg_q = torch.stack(neg_q)
    neg_k = torch.stack(neg_k)
    
    q_all = torch.cat([pos_q, neg_q], dim=0)
    k_all = torch.cat([pos_k, neg_k], dim=0)
    labels = torch.cat([torch.ones(pos_q.shape[0]), torch.zeros(neg_q.shape[0])], dim=0).to(DEVICE)
    
    jaccard = torch.ones(q_all.shape[0], device=DEVICE) * 0.5
    overlap = torch.ones(q_all.shape[0], device=DEVICE) * 0.5
    
    verifier.train()
    for epoch in range(15):
        scores = verifier(q_all, k_all, jaccard, overlap)
        loss = F.binary_cross_entropy(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    verifier.eval()
    return verifier

# ---------------------------------------------------------------------------
# Continual Learning Simulation
# ---------------------------------------------------------------------------
def run_continual_experiment(tokenizer, all_facts, cache_data, condition="agnis_replay", shuffles=5, seeds=3, lr=3e-4, lambda_ewc=0.1, lambda_anchor=0.1):
    results = []
    trajectories = []
    
    # Stratify into 10 blocks of 10 facts
    blocks = [all_facts[i*10 : (i+1)*10] for i in range(10)]
    
    block_train_s = []
    block_test_s = []
    block_train_x = []
    block_test_x = []
    for b_idx in range(10):
        tr_s, te_s = get_sentence_lists(blocks[b_idx])
        block_train_s.append(tr_s)
        block_test_s.append(te_s)
        block_train_x.append(cache_data["train_x"][b_idx*30 : (b_idx+1)*30].to(DEVICE))
        block_test_x.append(cache_data["test_x"][b_idx*40 : (b_idx+1)*40].to(DEVICE))

    # Pre-load verifier
    verifier = RelationVerifier(input_dim=INPUT_DIM).to(DEVICE)
    if os.path.exists("production_relation_verifier.pt"):
        print("[Verifier] Loading pre-trained verifier checkpoint...")
        verifier.load_state_dict(torch.load("production_relation_verifier.pt", map_location=DEVICE))
    else:
        verifier = train_verifier_on_fly(cache_data)
    verifier.eval()

    # Generate 5 fixed block shuffles
    random.seed(42)
    order_list = []
    for _ in range(shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)

    for shuffle_idx, order in enumerate(order_list):
        for seed in range(101, 101 + seeds):
            # Seed initialization
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

            # R[step, block] matrix (10x10) to evaluate block j's recall at step t
            R = np.zeros((10, 10))
            
            # Diagnostic drift dictionaries
            emb_drift_history = []
            student_drift_history = []
            verifier_drift_history = []
            ranking_drift_history = []
            
            # Enforce pre-training phase on first 5 blocks
            base_blocks = order[:5]
            base_s = [s for b in base_blocks for s in block_train_s[b]]
            base_x = torch.cat([block_train_x[b] for b in base_blocks], dim=0)
            
            # Train base representation to convergence
            student.train()
            N_base = len(base_s)
            for epoch in range(60):
                indices = list(range(N_base))
                random.shuffle(indices)
                for idx in range(0, N_base, 32):
                    b_idx = indices[idx : idx + 32]
                    b_s = [base_s[k] for k in b_idx]
                    ids, mask = batch_tokenize(tokenizer, b_s, device=DEVICE)
                    z_s = student(ids, mask)
                    z_t = base_x[b_idx]
                    loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            base_state = {k: v.clone() for k, v in student.state_dict().items()}
            
            # Save anchor representations of base blocks to compute anchor drift penalty
            student.eval()
            with torch.no_grad():
                anchor_ids, anchor_mask = batch_tokenize(tokenizer, base_s, device=DEVICE)
                anchor_embeddings = student(anchor_ids, anchor_mask)

            # Record baseline for base blocks at step 4
            # We construct a reference database of ONLY the base blocks (strict historical isolation)
            base_refs = []
            base_ref_labels = []
            for b in base_blocks:
                base_refs.append(block_train_s[b])
                base_ref_labels.extend([i for i in range(b*10, (b+1)*10) for _ in range(3)])
            
            with torch.no_grad():
                # Encode references
                z_refs_list = []
                for refs in base_refs:
                    for k in range(0, len(refs), 32):
                        ids, mask = batch_tokenize(tokenizer, refs[k:k+32], device=DEVICE)
                        z_refs_list.append(student(ids, mask))
                z_refs_accumulated = torch.cat(z_refs_list, dim=0)
                
                # Evaluate recall on each base block
                for b in base_blocks:
                    test_s = block_test_s[b]
                    test_labels = [idx for idx in range(b*10, (b+1)*10) for _ in range(4)]
                    
                    z_queries = []
                    for k in range(0, len(test_s), 32):
                        ids, mask = batch_tokenize(tokenizer, test_s[k:k+32], device=DEVICE)
                        z_queries.append(student(ids, mask))
                    z_queries = torch.cat(z_queries, dim=0)
                    
                    recall_count = 0
                    for q_idx, q_vec in enumerate(z_queries):
                        sims = torch.matmul(z_refs_accumulated, q_vec.unsqueeze(0).T).squeeze(-1)
                        best_idx = torch.argmax(sims).item()
                        pred_label = base_ref_labels[best_idx]
                        if pred_label == test_labels[q_idx]:
                            recall_count += 1
                    R[4, b] = recall_count / len(z_queries)

            # Anchor dict of embeddings for drift metrics
            anchor_ref_embeddings = {}
            for b in base_blocks:
                with torch.no_grad():
                    ref_s = block_train_s[b]
                    z_refs = []
                    for k in range(0, len(ref_s), 32):
                        ids, mask = batch_tokenize(tokenizer, ref_s[k:k+32], device=DEVICE)
                        z_refs.append(student(ids, mask))
                    anchor_ref_embeddings[b] = torch.cat(z_refs, dim=0)

            # Incremental sequential steps: 5 to 9
            for step in range(5, 10):
                curr_block = order[step]
                
                # Enforce strict historical isolation: memory index has blocks 0 to step-1
                # Build baseline before learning block step (with block step's references inserted)
                student.eval()
                seen_blocks_before = order[:step + 1]
                seen_refs_before = []
                seen_labels_before = []
                for b in seen_blocks_before:
                    seen_refs_before.append(block_train_s[b])
                    seen_labels_before.extend([i for i in range(b*10, (b+1)*10) for _ in range(3)])
                
                with torch.no_grad():
                    z_refs_before_list = []
                    for refs in seen_refs_before:
                        for k in range(0, len(refs), 32):
                            ids, mask = batch_tokenize(tokenizer, refs[k:k+32], device=DEVICE)
                            z_refs_before_list.append(student(ids, mask))
                    z_refs_before = torch.cat(z_refs_before_list, dim=0)
                    
                    # R_before for curr_block
                    test_s = block_test_s[curr_block]
                    test_labels = [idx for idx in range(curr_block*10, (curr_block+1)*10) for _ in range(4)]
                    z_queries = []
                    for k in range(0, len(test_s), 32):
                        ids, mask = batch_tokenize(tokenizer, test_s[k:k+32], device=DEVICE)
                        z_queries.append(student(ids, mask))
                    z_queries = torch.cat(z_queries, dim=0)
                    
                    recall_count = 0
                    for q_idx, q_vec in enumerate(z_queries):
                        sims = torch.matmul(z_refs_before, q_vec.unsqueeze(0).T).squeeze(-1)
                        best_idx = torch.argmax(sims).item()
                        pred_label = seen_labels_before[best_idx]
                        if pred_label == test_labels[q_idx]:
                            recall_count += 1
                    r_before = recall_count / len(z_queries)
                
                # Set up optimizer for this block's updates
                if condition == "agnis_replay":
                    inc_optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
                else:
                    inc_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

                # Train updates
                if condition == "frozen_encoder_writable_memory":
                    # Frozen encoder does not adapt its parameters
                    pass
                elif condition == "naive_sequential":
                    student.train()
                    curr_s = block_train_s[curr_block]
                    curr_x = block_train_x[curr_block]
                    for epoch in range(10):
                        indices = list(range(30))
                        random.shuffle(indices)
                        for idx in range(0, 30, 16):
                            b_idx = indices[idx : idx + 16]
                            b_s = [curr_s[i] for i in b_idx]
                            ids, mask = batch_tokenize(tokenizer, b_s, device=DEVICE)
                            z_s = student(ids, mask)
                            z_t = curr_x[b_idx]
                            loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                            inc_optimizer.zero_grad()
                            loss.backward()
                            inc_optimizer.step()
                elif condition == "agnis_replay":
                    student.train()
                    curr_s = block_train_s[curr_block]
                    curr_x = block_train_x[curr_block]
                    for epoch in range(10):
                        indices = list(range(30))
                        random.shuffle(indices)
                        for idx in range(0, 30, 16):
                            b_idx = indices[idx : idx + 16]
                            b_s = [curr_s[i] for i in b_idx]
                            ids, mask = batch_tokenize(tokenizer, b_s, device=DEVICE)
                            z_s = student(ids, mask)
                            z_t = curr_x[b_idx]
                            
                            # 1. New-fact distillation loss (un-diluted, matching naive sequential)
                            loss_distill = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                            
                            # 2. Stability coordinate preservation penalty (L2 anchor constraint)
                            cur_anchor_embeddings = student(anchor_ids, anchor_mask)
                            loss_anchor = F.mse_loss(cur_anchor_embeddings, anchor_embeddings)
                            
                            # 3. L2 parameter EWC stability regularization
                            loss_ewc = 0.0
                            for name, param in student.named_parameters():
                                if param.requires_grad:
                                    loss_ewc += torch.sum((param - base_state[name]) ** 2)
                                    
                            loss = loss_distill + lambda_anchor * loss_anchor + lambda_ewc * loss_ewc
                            
                            inc_optimizer.zero_grad()
                            loss.backward()
                            inc_optimizer.step()
                elif condition == "offline":
                    student.train()
                    all_tr_s = [s for b in order[:step+1] for s in block_train_s[b]]
                    all_tr_x = torch.cat([block_train_x[b] for b in order[:step+1]], dim=0)
                    N_total = len(all_tr_s)
                    for epoch in range(15):
                        indices = list(range(N_total))
                        random.shuffle(indices)
                        for idx in range(0, N_total, 32):
                            b_idx = indices[idx : idx + 32]
                            b_s = [all_tr_s[i] for i in b_idx]
                            ids, mask = batch_tokenize(tokenizer, b_s, device=DEVICE)
                            z_s = student(ids, mask)
                            z_t = all_tr_x[b_idx]
                            loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                            inc_optimizer.zero_grad()
                            loss.backward()
                            inc_optimizer.step()

                # Evaluate recall on all seen blocks under the updated student
                student.eval()
                seen_blocks_after = order[:step + 1]
                seen_refs_after = []
                seen_labels_after = []
                for b in seen_blocks_after:
                    seen_refs_after.append(block_train_s[b])
                    seen_labels_after.extend([i for i in range(b*10, (b+1)*10) for _ in range(3)])
                
                with torch.no_grad():
                    z_refs_after_list = []
                    for refs in seen_refs_after:
                        for k in range(0, len(refs), 32):
                            ids, mask = batch_tokenize(tokenizer, refs[k:k+32], device=DEVICE)
                            z_refs_after_list.append(student(ids, mask))
                    z_refs_after = torch.cat(z_refs_after_list, dim=0)
                    
                    for b in seen_blocks_after:
                        b_test_s_eval = block_test_s[b]
                        b_test_labels_eval = [idx for idx in range(b*10, (b+1)*10) for _ in range(4)]
                        
                        z_queries_eval = []
                        for k in range(0, len(b_test_s_eval), 32):
                            ids, mask = batch_tokenize(tokenizer, b_test_s_eval[k:k+32], device=DEVICE)
                            z_queries_eval.append(student(ids, mask))
                        z_queries_eval = torch.cat(z_queries_eval, dim=0)
                        
                        recall_count = 0
                        for q_idx, q_vec in enumerate(z_queries_eval):
                            sims = torch.matmul(z_refs_after, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_idx = torch.argmax(sims).item()
                            pred_label = seen_labels_after[best_idx]
                            if pred_label == b_test_labels_eval[q_idx]:
                                recall_count += 1
                        R[step, b] = recall_count / len(z_queries_eval)

                # Monitor Independent Drift Sources for Base Blocks
                drift_emb_list = []
                drift_stud_list = []
                drift_verifier_list = []
                drift_ranking_list = []
                
                with torch.no_grad():
                    for b in base_blocks:
                        ref_s = block_train_s[b]
                        z_refs_curr = []
                        for k in range(0, len(ref_s), 32):
                            ids, mask = batch_tokenize(tokenizer, ref_s[k:k+32], device=DEVICE)
                            z_refs_curr.append(student(ids, mask))
                        z_refs_curr = torch.cat(z_refs_curr, dim=0)
                        
                        # 1. Reference embedding drift (drift of stored prototypes)
                        z_refs_anchor = anchor_ref_embeddings[b]
                        drift_emb = (1.0 - (z_refs_curr * z_refs_anchor).sum(dim=-1)).mean().item()
                        drift_emb_list.append(drift_emb)
                        
                        # 2. Student output coordinate drift (anchored base sentences)
                        cur_anchor_block = student(anchor_ids, anchor_mask)
                        drift_stud = F.mse_loss(cur_anchor_block, anchor_embeddings).item()
                        drift_stud_list.append(drift_stud)
                        
                        # 3. Verifier score drift (relative change in verifier prediction confidence)
                        test_s = block_test_s[b]
                        for q_idx, q_str in enumerate(test_s):
                            ids, mask = batch_tokenize(tokenizer, [q_str], device=DEVICE)
                            q_s = student(ids, mask)[0]
                            k_vec = z_refs_curr[q_idx // 4]
                            k_str = ref_s[q_idx // 4]
                            
                            jac, ov = get_entity_overlap(q_str, k_str)
                            score = verifier(q_s.unsqueeze(0), k_vec.unsqueeze(0),
                                             torch.tensor([jac], device=DEVICE),
                                             torch.tensor([ov], device=DEVICE)).item()
                            drift_verifier_list.append(score)
                            
                        # 4. Retrieval ranking change (overlap fraction of the top candidate)
                        z_queries_b = []
                        for k in range(0, len(test_s), 32):
                            ids, mask = batch_tokenize(tokenizer, test_s[k:k+32], device=DEVICE)
                            z_queries_b.append(student(ids, mask))
                        z_queries_b = torch.cat(z_queries_b, dim=0)
                        
                        ranking_match = 0
                        for q_idx, q_vec in enumerate(z_queries_b):
                            sims_anchor = torch.matmul(z_refs_anchor, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_anchor = torch.argmax(sims_anchor).item()
                            
                            sims_curr = torch.matmul(z_refs_curr, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_curr = torch.argmax(sims_curr).item()
                            
                            if best_anchor == best_curr:
                                ranking_match += 1
                        drift_ranking_list.append(ranking_match / len(z_queries_b))

                emb_drift_history.append(np.mean(drift_emb_list))
                student_drift_history.append(np.mean(drift_stud_list))
                verifier_drift_history.append(np.mean(drift_verifier_list))
                ranking_drift_history.append(np.mean(drift_ranking_list))
                
                # R_after for plasticity calculation
                r_after = R[step, curr_block]
                
                # Plasticity: R_after - R_before
                plasticity_gain = r_after - r_before
                results.append({
                    "condition": condition,
                    "shuffle": shuffle_idx,
                    "seed": seed,
                    "step": step,
                    "curr_block": curr_block,
                    "plasticity_gain": plasticity_gain
                })

            # Calculate metrics at the final step N=9 (end of sequence)
            # BWT: R[9, b] - R[first_seen_step, b]
            bwt_vals = []
            signed_bwt_per_block = {}
            for j in range(10):
                first_seen_step = order.index(j)
                if first_seen_step < 9:
                    start_step = max(4, first_seen_step)
                    bwt_val = R[9, j] - R[start_step, j]
                    bwt_vals.append(bwt_val)
                    signed_bwt_per_block[f"block_{j}"] = bwt_val
            mean_bwt = np.mean(bwt_vals) if len(bwt_vals) > 0 else 0.0

            # Forgetting: max_t R[t, b] - R[9, b]
            forgetting_vals = []
            forgetting_per_block = {}
            for j in range(10):
                first_seen_step = order.index(j)
                if first_seen_step < 9:
                    start_step = max(4, first_seen_step)
                    max_seen = np.max(R[start_step:9, j])
                    fgt_val = max_seen - R[9, j]
                    forgetting_vals.append(fgt_val)
                    forgetting_per_block[f"block_{j}"] = fgt_val
            mean_forgetting = np.mean(forgetting_vals) if len(forgetting_vals) > 0 else 0.0
            worst_forgetting = np.max(forgetting_vals) if len(forgetting_vals) > 0 else 0.0

            # Collect results for this seed/shuffle run
            results[-1].update({
                "mean_bwt": mean_bwt,
                "mean_forgetting": mean_forgetting,
                "worst_forgetting": worst_forgetting,
                "forgetting_per_block": forgetting_per_block,
                "signed_bwt_per_block": signed_bwt_per_block,
                "drift_emb": np.mean(emb_drift_history),
                "drift_student": np.mean(student_drift_history),
                "drift_verifier_score": np.mean(verifier_drift_history),
                "drift_ranking_overlap": np.mean(ranking_drift_history)
            })
            
            # Log the complete trajectory for diagnostic reporting
            trajectories.append({
                "condition": condition,
                "shuffle": shuffle_idx,
                "seed": seed,
                "order": order,
                "R_matrix": R.tolist()
            })
            
    # Save per-block trajectories to file
    with open(f"trajectories_{condition}.json", "w") as f:
        json.dump(trajectories, f, indent=2)
        
    return results

def main():
    print("="*80)
    print("  AGNIS PHASE C.2.1: DIAGNOSTIC ROBUSTNESS & LEAKAGE AUDIT")
    print("="*80)
    
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[Data] Scaling dataset not found at {DATASET_PATH}. Reconstructing automatically...")
        os.system("python generate_scaling_dataset.py")
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Scaling dataset not found at {DATASET_PATH}")
            
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b]
    print(f"[Data] Loaded {len(blocks)} blocks containing {len(all_facts)} total facts.")
    
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load cache
    if not os.path.exists(CACHE_100_PATH):
        print(f"[Cache] Embeddings cache {CACHE_100_PATH} not found. Reconstructing automatically...")
        from transformers import AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            raise RuntimeError("FAIL-CLOSED: Failed to load SmolLM2 model to generate embeddings cache") from e
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    else:
        cache_data = torch.load(CACHE_100_PATH, weights_only=True)
    
import sys
import glob

def print_audit_report(all_summary, all_runs_results, sweep_configs, conditions):
    # Print validation report
    print("\n" + "="*80)
    print("  FINAL CONTINUAL-LEARNING METRICS COMPILATION REPORT")
    print("="*80)
    print("  Condition                                | Plasticity  | Forgetting | Worst-Block | BWT        | Emb Drift | Output Drift | Verifier Score | Ranking Overlap")
    print("  -------------------------------------------------------------------------------------------------------------------------------------------------------------")
    for cond in conditions:
        if cond in all_summary:
            s = all_summary[cond]
            print(f"  {cond:40s} | {s['plasticity_gain']*100:10.2f}% | {s['forgetting']*100:9.2f}% | {s['worst_forgetting']*100:10.2f}% | {s['bwt']*100:9.2f}% | {s['drift_emb']:9.6f} | {s['drift_student']:12.6f} | {s['drift_verifier_score']:14.4f} | {s['drift_ranking_overlap']*100:14.2f}%")
    print("="*80)

    # Print Signed BWT Breakdown
    print("\n" + "-"*80)
    print("  SIGNED PER-BLOCK BACKWARD TRANSFER (BWT) BREAKDOWN")
    print("-"*80)
    for cond in conditions:
        if cond in all_summary:
            s = all_summary[cond]
            bwt_str = ", ".join([f"{k}: {v*100:+.1f}%" for k, v in s["signed_bwt"].items()])
            print(f"  * {cond:40s} -> {bwt_str}")
    print("="*80)

    # Exit Criteria Checks
    if "naive_sequential" not in all_summary or "frozen_encoder_writable_memory" not in all_summary:
        print("\n[Notice] Controls (frozen / naive_sequential) missing in current evaluation. Run --merge after completing all chunks to execute full paired audit.")
        return

    print("\n" + "-"*80)
    print("  PASS/FAIL AUDIT AGAINST DECLARED CONTINUAL-LEARNING PASS CRITERIA")
    print("-"*80)
    
    naive_sum = all_summary["naive_sequential"]
    frozen_sum = all_summary["frozen_encoder_writable_memory"]
    frozen_leak_pass = (abs(frozen_sum["plasticity_gain"]) <= 0.001)
    
    print(f"  - Frozen Learning Gain is ~0.00% (No leakage): {'PASSED' if frozen_leak_pass else 'FAILED'} (Observed: {frozen_sum['plasticity_gain']*100:.2f}%)")
    
    for name, _, _, _ in sweep_configs:
        if name not in all_summary or name not in all_runs_results:
            continue
        s = all_summary[name]
        
        # Calculate paired excess forgetting at the run/seed/block level
        res_cond = [r for r in all_runs_results[name] if "forgetting_per_block" in r]
        res_frozen = [r for r in all_runs_results["frozen_encoder_writable_memory"] if "forgetting_per_block" in r]
        
        excess_mean_runs = []
        excess_worst_runs = []
        
        for run_idx in range(len(res_cond)):
            fgt_cond = res_cond[run_idx]["forgetting_per_block"]
            fgt_frozen = res_frozen[run_idx]["forgetting_per_block"]
            
            excess_blocks = []
            for blk in fgt_cond:
                excess_blocks.append(fgt_cond[blk] - fgt_frozen[blk])
                
            excess_mean_runs.append(np.mean(excess_blocks))
            excess_worst_runs.append(np.max(excess_blocks))
            
        mean_excess_forgetting = np.mean(excess_mean_runs) if len(excess_mean_runs) > 0 else 0.0
        worst_excess_forgetting = np.max(excess_worst_runs) if len(excess_worst_runs) > 0 else 0.0
        
        forgetting_pass = (mean_excess_forgetting <= 0.02)
        worst_forgetting_pass = (worst_excess_forgetting <= 0.05)
        plasticity_pass = (s["plasticity_gain"] >= 0.95 * naive_sum["plasticity_gain"])
        ranking_pass = (s["drift_ranking_overlap"] >= 0.95)
        
        status = "PASSED (Certified CL Robust)" if (forgetting_pass and worst_forgetting_pass and plasticity_pass and ranking_pass) else "FAILED"
        print(f"\n  * Configuration: {name}")
        print(f"    - Paired Excess Forgetting <= 2.0%      : {'PASSED' if forgetting_pass else 'FAILED'} (Observed: {mean_excess_forgetting*100:.2f}%)")
        print(f"    - Worst Paired Excess Forgetting <= 5.0%: {'PASSED' if worst_forgetting_pass else 'FAILED'} (Observed: {worst_excess_forgetting*100:.2f}%)")
        print(f"    - Plasticity Gain within 95% of naive (Target >= {0.95*naive_sum['plasticity_gain']*100:.2f}%): {'PASSED' if plasticity_pass else 'FAILED'} (Observed: {s['plasticity_gain']*100:.2f}%)")
        print(f"    - Ranking Overlap >= 95.0%              : {'PASSED' if ranking_pass else 'FAILED'} (Observed: {s['drift_ranking_overlap']*100:.2f}%)")
        print(f"    - Overall Certification Status         : {status}")
    print("="*80)


def main():
    sweep_configs = [
        # Milli-scale sweeps at standard learning rate 1e-3
        ("agnis_replay_ewc0.001_anc0.001", 1e-3, 0.001, 0.001),
        ("agnis_replay_ewc0.002_anc0.002", 1e-3, 0.002, 0.002),
        ("agnis_replay_ewc0.005_anc0.002", 1e-3, 0.005, 0.002),
        ("agnis_replay_ewc0.005_anc0.005", 1e-3, 0.005, 0.005),
        ("agnis_replay_ewc0.01_anc0.01", 1e-3, 0.01, 0.01),
        ("agnis_replay_ewc0.02_anc0.01", 1e-3, 0.02, 0.01),
        ("agnis_replay_ewc0.02_anc0.02", 1e-3, 0.02, 0.02),
        # Higher learning rates to overcome EWC anchoring tension
        ("agnis_replay_lr1.5e-3_ewc0.005", 1.5e-3, 0.005, 0.005),
        ("agnis_replay_lr2e-3_ewc0.01", 2e-3, 0.01, 0.01)
    ]
    all_twelve = ["frozen_encoder_writable_memory", "naive_sequential"] + [c[0] for c in sweep_configs] + ["offline"]

    # Check for --merge command
    if len(sys.argv) > 1 and "--merge" in sys.argv:
        print("\n" + "="*80)
        print("  MERGING CONTINUAL LEARNING CHUNK RESULTS AND TRAJECTORIES")
        print("="*80)
        all_summary = {}
        all_runs_results = {}
        
        chunk_summary_files = sorted(glob.glob("continual_learning_results_*.json"))
        chunk_summary_files = [f for f in chunk_summary_files if f != "continual_learning_results.json"]
        chunk_runs_files = sorted(glob.glob("trajectories_*.json"))
        
        print(f"[Merge] Found {len(chunk_summary_files)} summary chunk files and {len(chunk_runs_files)} trajectory files.")
        for sf in chunk_summary_files:
            with open(sf, "r") as f:
                all_summary.update(json.load(f))
                
        for rf in chunk_runs_files:
            with open(rf, "r") as f:
                all_runs_results.update(json.load(f))
                
        conditions = [c for c in all_twelve if c in all_summary]
        
        with open("continual_learning_results.json", "w") as f:
            json.dump(all_summary, f, indent=2)
        with open("trajectories_all.json", "w") as f:
            json.dump(all_runs_results, f, indent=2)
            
        print_audit_report(all_summary, all_runs_results, sweep_configs, conditions)
        return

    # Parse dataset
    if not os.path.exists(DATASET_PATH):
        print(f"[Dataset] {DATASET_PATH} not found. Generating dataset automatically...")
        from generate_scaling_dataset import build_fact_dataset
        blocks = build_fact_dataset()
        with open(DATASET_PATH, "w") as f:
            json.dump(blocks, f, indent=2)
    else:
        with open(DATASET_PATH, "r") as f:
            blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b]
    print(f"[Data] Loaded {len(blocks)} blocks containing {len(all_facts)} total facts.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if not os.path.exists(CACHE_100_PATH):
        print(f"[Cache] Embeddings cache {CACHE_100_PATH} not found. Reconstructing automatically...")
        from transformers import AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            raise RuntimeError("FAIL-CLOSED: Failed to load SmolLM2 model to generate embeddings cache") from e
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    else:
        cache_data = torch.load(CACHE_100_PATH, weights_only=True)

    # Determine conditions to run
    req_cond_str = os.environ.get("CONDITIONS") or os.environ.get("AGNIS_CONDITIONS")
    chunk_tag = os.environ.get("CHUNK_NAME") or os.environ.get("CHUNK")
    
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--conditions="):
            req_cond_str = arg.split("=", 1)[1]
        elif arg == "--conditions" and i + 1 < len(sys.argv):
            req_cond_str = sys.argv[i + 1]
        elif arg.startswith("--chunk="):
            chunk_tag = arg.split("=", 1)[1]
        elif arg == "--chunk" and i + 1 < len(sys.argv):
            chunk_tag = sys.argv[i + 1]
            
    if req_cond_str:
        req_list = [c.strip() for c in req_cond_str.split(",") if c.strip()]
        conditions = [c for c in all_twelve if c in req_list]
    else:
        conditions = all_twelve

    is_subset = len(conditions) < len(all_twelve)
    if is_subset or chunk_tag:
        tag = chunk_tag if chunk_tag else "_".join([c[:10] for c in conditions])
        if len(tag) > 40:
            tag = f"chunk_{hash(tag) & 0xffffffff:08x}"
        summary_out_path = f"continual_learning_results_{tag}.json"
        runs_out_path = f"trajectories_{tag}.json"
    else:
        summary_out_path = "continual_learning_results.json"
        runs_out_path = "trajectories_all.json"

    all_summary = {}
    all_runs_results = {}

    for cond in conditions:
        print(f"\n[Running] Evaluating condition: {cond}...")
        t0 = time.time()
        
        cfg = [c for c in sweep_configs if c[0] == cond]
        if len(cfg) > 0:
            name, lr_val, l_ewc, l_anchor = cfg[0]
            res = run_continual_experiment(tokenizer, all_facts, cache_data, 
                                           condition="agnis_replay", shuffles=5, seeds=3,
                                           lr=lr_val, lambda_ewc=l_ewc, lambda_anchor=l_anchor)
        else:
            res = run_continual_experiment(tokenizer, all_facts, cache_data, condition=cond, shuffles=5, seeds=3)
            
        t_el = time.time() - t0
        all_runs_results[cond] = res
        
        pls = [r["plasticity_gain"] for r in res]
        fgts = [r["mean_forgetting"] for r in res if "mean_forgetting" in r]
        w_fgts = [r["worst_forgetting"] for r in res if "worst_forgetting" in r]
        bwts = [r["mean_bwt"] for r in res if "mean_bwt" in r]
        
        drifts_emb = [r["drift_emb"] for r in res if "drift_emb" in r]
        drifts_student = [r["drift_student"] for r in res if "drift_student" in r]
        drifts_verifier = [r["drift_verifier_score"] for r in res if "drift_verifier_score" in r]
        drifts_ranking = [r["drift_ranking_overlap"] for r in res if "drift_ranking_overlap" in r]
        
        signed_bwt_dicts = [r["signed_bwt_per_block"] for r in res if "signed_bwt_per_block" in r]
        
        flat_signed_bwts = {}
        for block_key in [f"block_{i}" for i in range(10)]:
            vals = [d[block_key] for d in signed_bwt_dicts if block_key in d]
            flat_signed_bwts[block_key] = np.mean(vals) if len(vals) > 0 else 0.0
        
        all_summary[cond] = {
            "plasticity_gain": np.mean(pls), "plasticity_gain_std": np.std(pls),
            "forgetting": np.mean(fgts), "forgetting_std": np.std(fgts),
            "worst_forgetting": np.mean(w_fgts), "worst_forgetting_std": np.std(w_fgts),
            "bwt": np.mean(bwts), "bwt_std": np.std(bwts),
            "drift_emb": np.mean(drifts_emb),
            "drift_student": np.mean(drifts_student),
            "drift_verifier_score": np.mean(drifts_verifier),
            "drift_ranking_overlap": np.mean(drifts_ranking),
            "signed_bwt": flat_signed_bwts,
            "latency": t_el / (5 * 3)
        }
        print(f"  - Completed in {t_el:.2f}s.")
        print(f"  - Plasticity Gain: {all_summary[cond]['plasticity_gain']*100:.2f}% | Forgetting: {all_summary[cond]['forgetting']*100:.2f}%")

    with open(summary_out_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    with open(runs_out_path, "w") as f:
        json.dump(all_runs_results, f, indent=2)

    print_audit_report(all_summary, all_runs_results, sweep_configs, conditions)

if __name__ == "__main__":
    main()
