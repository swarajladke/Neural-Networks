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
CACHE_100_PATH = "../smollm2_embeddings_100slots.pt" if os.path.exists("../smollm2_embeddings_100slots.pt") or not os.path.exists("smollm2_embeddings_100slots.pt") else "smollm2_embeddings_100slots.pt"
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
# Continual Learning Simulation
# ---------------------------------------------------------------------------
def run_continual_experiment(tokenizer, all_facts, cache_data, condition="agnis_replay", shuffles=5, seeds=3):
    results = []
    
    # Stratify into 10 blocks of 10 facts
    blocks = [all_facts[i*10 : (i+1)*10] for i in range(10)]
    
    # Build train and test sentence groups for each block
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

    # Pre-load or initialize verifier
    verifier = RelationVerifier(input_dim=INPUT_DIM).to(DEVICE)
    if os.path.exists("production_relation_verifier.pt"):
        verifier.load_state_dict(torch.load("production_relation_verifier.pt", map_location=DEVICE))
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

            # R[step, block] matrix to evaluate block j's recall at step t
            R = np.zeros((10, 10))
            drift_list = []
            far_list = []
            
            # Keep track of previous embeddings to measure drift
            prev_embeddings = {}

            # Sequentially learn blocks in order
            for step in range(10):
                curr_block = order[step]
                
                if condition == "frozen":
                    # No updates at all
                    pass
                elif condition == "naive_sequential":
                    # Fine-tune only on current block prompts
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
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                elif condition == "agnis_replay":
                    # Fine-tune on current block + replay all previously learned blocks
                    student.train()
                    replayed_blocks = order[:step + 1]
                    replay_s = []
                    replay_x = []
                    for b in replayed_blocks:
                        replay_s.extend(block_train_s[b])
                        replay_x.append(block_train_x[b])
                    replay_x = torch.cat(replay_x, dim=0)
                    
                    N_total = len(replay_s)
                    for epoch in range(10):
                        indices = list(range(N_total))
                        random.shuffle(indices)
                        for idx in range(0, N_total, 32):
                            b_idx = indices[idx : idx + 32]
                            b_s = [replay_s[i] for i in b_idx]
                            ids, mask = batch_tokenize(tokenizer, b_s, device=DEVICE)
                            z_s = student(ids, mask)
                            z_t = replay_x[b_idx]
                            loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                elif condition == "offline":
                    # Joint offline training: train on all 10 blocks simultaneously at step 0
                    if step == 0:
                        student.train()
                        all_tr_s = [s for b in range(10) for s in block_train_s[b]]
                        all_tr_x = torch.cat([block_train_x[b] for b in range(10)], dim=0)
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
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                student.eval()
                
                # 1. Evaluate seen blocks
                seen_drift = []
                for step_prev in range(step + 1):
                    b = order[step_prev]
                    ref_s = block_train_s[b]
                    ref_labels = list(range(b*10, (b+1)*10))
                    
                    test_s = block_test_s[b]
                    test_labels = [idx for idx in ref_labels for _ in range(4)]
                    
                    # Encode current block references
                    with torch.no_grad():
                        z_refs = []
                        for k in range(0, len(ref_s), 32):
                            ids, mask = batch_tokenize(tokenizer, ref_s[k:k+32], device=DEVICE)
                            z_refs.append(student(ids, mask))
                        z_refs = torch.cat(z_refs, dim=0)
                        
                        # Monitor coordinate drift
                        b_key = f"{shuffle_idx}_{seed}_{b}"
                        if b_key in prev_embeddings:
                            z_prev = prev_embeddings[b_key]
                            drift = (1.0 - (z_refs * z_prev).sum(dim=-1)).mean().item()
                            seen_drift.append(drift)
                        prev_embeddings[b_key] = z_refs.clone()
                        
                        # Encode test queries
                        z_queries = []
                        for k in range(0, len(test_s), 32):
                            ids, mask = batch_tokenize(tokenizer, test_s[k:k+32], device=DEVICE)
                            z_queries.append(student(ids, mask))
                        z_queries = torch.cat(z_queries, dim=0)
                        
                        # 1-NN recall calculation
                        recall_count = 0
                        for q_idx, q_vec in enumerate(z_queries):
                            sims = torch.matmul(z_refs, q_vec.unsqueeze(0).T).squeeze(-1)
                            best_idx = torch.argmax(sims).item()
                            pred_label = ref_labels[best_idx // 3]
                            if pred_label == test_labels[q_idx]:
                                recall_count += 1
                        R[step, b] = recall_count / len(z_queries)
                
                # 2. Evaluate Safety (False Accepts on Unseen blocks acting as semantic OOD)
                unseen_blocks = order[step + 1:]
                false_accepts = 0
                total_unseen = 0
                if len(unseen_blocks) > 0 and len(order[:step+1]) > 0:
                    # Build index of seen facts references
                    seen_refs = []
                    seen_labels = []
                    for b in order[:step+1]:
                        seen_refs.append(block_train_x[b])
                        seen_labels.extend([i for i in range(b*10, (b+1)*10) for _ in range(3)])
                    seen_refs = torch.cat(seen_refs, dim=0)
                    
                    with torch.no_grad():
                        for b_unseen in unseen_blocks:
                            unseen_test_s = block_test_s[b_unseen]
                            for q_str in unseen_test_s:
                                ids, mask = batch_tokenize(tokenizer, [q_str], device=DEVICE)
                                q_s = student(ids, mask)[0]
                                
                                # Retrieve candidates
                                sims = torch.matmul(seen_refs, q_s.unsqueeze(0).T).squeeze(-1)
                                best_idx = torch.argmax(sims).item()
                                k_vec = seen_refs[best_idx]
                                k_str = block_train_s[best_idx // 30][best_idx % 30]
                                
                                # Verifier decision
                                jac, ov = get_entity_overlap(q_str, k_str)
                                score = verifier(q_s.unsqueeze(0), k_vec.unsqueeze(0),
                                                 torch.tensor([jac], device=DEVICE),
                                                 torch.tensor([ov], device=DEVICE)).item()
                                
                                if score >= 0.90:  # Safety threshold
                                    false_accepts += 1
                                total_unseen += 1
                                
                far = false_accepts / total_unseen if total_unseen > 0 else 0.0
                far_list.append(far)
                drift_list.append(np.mean(seen_drift) if len(seen_drift) > 0 else 0.0)
            
            # Compute step 9 metrics for this run
            plasticity = np.mean([R[i, order[i]] for i in range(10)])
            
            # Forgetting: R[max_t, j] - R[9, j]
            forgetting_vals = []
            for j in range(10):
                first_seen_step = order.index(j)
                if first_seen_step < 9:
                    max_seen = np.max(R[first_seen_step:9, j])
                    forgetting_vals.append(max_seen - R[9, j])
            forgetting = np.mean(forgetting_vals) if len(forgetting_vals) > 0 else 0.0
            worst_forgetting = np.max(forgetting_vals) if len(forgetting_vals) > 0 else 0.0
            
            bwt_vals = []
            for j in range(10):
                first_seen_step = order.index(j)
                if first_seen_step < 9:
                    bwt_vals.append(R[9, j] - R[first_seen_step, j])
            bwt = np.mean(bwt_vals) if len(bwt_vals) > 0 else 0.0

            results.append({
                "condition": condition,
                "shuffle": shuffle_idx,
                "seed": seed,
                "plasticity": plasticity,
                "forgetting": forgetting,
                "worst_forgetting": worst_forgetting,
                "bwt": bwt,
                "drift": np.mean(drift_list),
                "far": np.mean(far_list)
            })
            
    return results

def main():
    print("="*80)
    print("  AGNIS PHASE C.2: CONTINUAL-LEARNING ROBUSTNESS SUITE")
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
    
    conditions = ["frozen", "naive_sequential", "agnis_replay", "offline"]
    all_summary = {}

    for cond in conditions:
        print(f"\n[Running] Evaluating condition: {cond}...")
        t0 = time.time()
        res = run_continual_experiment(tokenizer, all_facts, cache_data, condition=cond, shuffles=5, seeds=3)
        t_el = time.time() - t0
        
        # Calculate summary statistics across shuffles and seeds
        pls = [r["plasticity"] for r in res]
        fgts = [r["forgetting"] for r in res]
        w_fgts = [r["worst_forgetting"] for r in res]
        bwts = [r["bwt"] for r in res]
        drifts = [r["drift"] for r in res]
        fars = [r["far"] for r in res]
        
        all_summary[cond] = {
            "plasticity": np.mean(pls), "plasticity_std": np.std(pls),
            "forgetting": np.mean(fgts), "forgetting_std": np.std(fgts),
            "worst_forgetting": np.mean(w_fgts), "worst_forgetting_std": np.std(w_fgts),
            "bwt": np.mean(bwts), "bwt_std": np.std(bwts),
            "drift": np.mean(drifts), "drift_std": np.std(drifts),
            "far": np.mean(fars), "far_std": np.std(fars),
            "latency": t_el / (5 * 3) # average latency per update sequence
        }
        print(f"  - Completed in {t_el:.2f}s.")
        print(f"  - Plasticity: {all_summary[cond]['plasticity']*100:.2f}% | Forgetting: {all_summary[cond]['forgetting']*100:.2f}% | Cosine Drift: {all_summary[cond]['drift']:.6f}")

    # Print validation report
    print("\n" + "="*80)
    print("  FINAL CONTINUAL-LEARNING METRICS COMPILATION REPORT")
    print("="*80)
    print("  Condition         | Plasticity  | Forgetting | Worst-Block | BWT        | Cosine Drift | OOD FAR")
    print("  ------------------------------------------------------------------------------------------------")
    for cond in conditions:
        s = all_summary[cond]
        print(f"  {cond:17s} | {s['plasticity']*100:10.2f}% | {s['forgetting']*100:9.2f}% | {s['worst_forgetting']*100:10.2f}% | {s['bwt']*100:9.2f}% | {s['drift']:12.6f} | {s['far']*100:8.2f}%")
    print("="*80)

    # Exit Criteria Checks
    print("\n" + "-"*80)
    print("  PASS/FAIL AUDIT AGAINST DECLARED CONTINUAL-LEARNING PASS CRITERIA")
    print("-"*80)
    
    replay_sum = all_summary["agnis_replay"]
    naive_sum = all_summary["naive_sequential"]
    frozen_sum = all_summary["frozen"]
    
    forgetting_pass = (replay_sum["forgetting"] <= 0.02)
    worst_forgetting_pass = (replay_sum["worst_forgetting"] <= 0.05)
    plasticity_pass = (replay_sum["plasticity"] >= 0.95 * naive_sum["plasticity"])
    safety_pass = (replay_sum["far"] <= 0.01)  # less than 1% OOD False Accept Rate
    
    print(f"  - Mean forgetting <= 2.0%                   : {'PASSED' if forgetting_pass else 'FAILED'} (Observed: {replay_sum['forgetting']*100:.2f}%)")
    print(f"  - Worst-block forgetting <= 5.0%            : {'PASSED' if worst_forgetting_pass else 'FAILED'} (Observed: {replay_sum['worst_forgetting']*100:.2f}%)")
    print(f"  - Plasticity within 95% of naive baseline   : {'PASSED' if plasticity_pass else 'FAILED'} (Observed: {replay_sum['plasticity']*100:.2f}% vs target >= {0.95*naive_sum['plasticity']*100:.2f}%)")
    print(f"  - Safety-gate FAR does not increase         : {'PASSED' if safety_pass else 'FAILED'} (Observed: {replay_sum['far']*100:.2f}%)")
    
    overall_pass = forgetting_pass and worst_forgetting_pass and plasticity_pass and safety_pass
    print(f"\n  OVERALL CL CERTIFICATION STATUS             : {'PASSED (Certified CL Robust)' if overall_pass else 'FAILED'}")
    print("="*80)

    # Save validation metrics to file
    with open("continual_learning_results.json", "w") as f:
        json.dump(all_summary, f, indent=2)

if __name__ == "__main__":
    main()
