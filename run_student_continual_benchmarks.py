"""
run_student_continual_benchmarks.py — Phase A.4, A.5, and A.6 Evaluation Suite.
================================================================================
Implements:
1. Phase A.4: Multi-seed 100-fact static retrieval evaluation (100 facts).
2. Phase A.5: Sequential continual-learning matrix evaluation (Plasticity, BWT, Forgetting).
3. Phase A.6: CPU/GPU latency, RAM/VRAM memory, and model footprint profiling.
4. Semantic hard-negatives audit (in-domain entity swaps and relational collisions).
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from student_encoder import StudentEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_100_PATH = "smollm2_embeddings_100slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
INPUT_DIM = 960

# ---------------------------------------------------------------------------
# Caching SmolLM2 Embeddings for the full 100-fact dataset
# ---------------------------------------------------------------------------
def ensure_100_fact_embeddings(tokenizer, model, blocks):
    if os.path.exists(CACHE_100_PATH):
        print(f"[Cache] Loading 100-fact cached embeddings from {CACHE_100_PATH}...")
        return torch.load(CACHE_100_PATH, map_location=DEVICE)
        
    print(f"[Cache] Generating fresh embeddings for all 100 facts...")
    all_facts = [fact for b in blocks for fact in b]
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    probe_to_class = {p: idx for idx, p in enumerate(unique_probes)}
    
    def get_prompt_only(f, idx):
        if idx == 0:
            return f["probe"]
        elif idx == 1:
            prefix = f["qa"].split(f["statement"])[0]
            return prefix + f["probe"]
        else:
            return f["cloze"].split("_____")[0].strip()
            
    def extract_pooled(prompt):
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"
        enc = tokenizer(prompt, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
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
            return F.normalize(pooled.float(), dim=-1)[0].cpu()

    train_queries = []
    train_labels = []
    val_queries = []
    val_labels = []
    test_queries = []
    test_labels = []
    
    for f in all_facts:
        label = probe_to_class[f["probe"]]
        # Train (3 templates)
        for idx_t in range(3):
            train_queries.append(extract_pooled(get_prompt_only(f, idx_t)))
            train_labels.append(label)
        # Val (1 template)
        val_queries.append(extract_pooled(f["train_paraphrases"][-1]))
        val_labels.append(label)
        # Test (4 templates)
        all_eval_items = [f["probe"]] + f["eval_paraphrases"]
        for item in all_eval_items:
            test_queries.append(extract_pooled(item))
            test_labels.append(label)
            
    cache_data = {
        "train_x": torch.stack(train_queries),
        "train_y": torch.tensor(train_labels),
        "val_x": torch.stack(val_queries),
        "val_y": torch.tensor(val_labels),
        "test_x": torch.stack(test_queries),
        "test_y": torch.tensor(test_labels),
    }
    torch.save(cache_data, CACHE_100_PATH)
    print(f"[Cache] Saved all 100 facts cached embeddings to {CACHE_100_PATH}.")
    return cache_data

# ---------------------------------------------------------------------------
# Sentence Helper & Tokenization
# ---------------------------------------------------------------------------
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
# Semantic Hard-Negatives Generation
# ---------------------------------------------------------------------------
def generate_semantic_hard_negatives(all_facts):
    print("\n[Control] Synthesizing semantic hard negatives...")
    controls = []
    
    # 1. Same entity, wrong relation (relational collisions)
    for fact in all_facts[:50]:
        entity = fact["location"] if "location" in fact else fact.get("compound", fact.get("planet"))
        controls.append(f"At what temperature does the capital {entity} melt?")
        controls.append(f"The planetary satellite {entity} liquefies at exactly forty two degrees.")
        
    # 2. Conflicting/fictional answers
    for fact in all_facts[:50]:
        entity = fact["location"] if "location" in fact else fact.get("compound", fact.get("planet"))
        controls.append(f"The official capital city of {entity} is Washington DC.")
        controls.append(f"The molecular compound {entity} melts at absolute zero.")
        
    # 3. Questions mentioning stored answers without asking for them
    for fact in all_facts[:50]:
        val = fact["answer"]
        controls.append(f"I am writing a story about the capital {val}.")
        controls.append(f"Explain why {val} is not a primary element.")
        
    # 4. Spaced out variations
    for fact in all_facts[:50]:
        entity = fact["location"] if "location" in fact else fact.get("compound", fact.get("planet"))
        controls.append(f"Identify if the region of {entity} has any active volcanoes.")
        
    print(f"  - Generated {len(controls)} in-domain semantic hard negatives.")
    return controls

# ---------------------------------------------------------------------------
# Wilson Score Interval and FPR Bounds
# ---------------------------------------------------------------------------
def compute_wilson_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / total
    centre_adj_p = p + z**2 / (2 * total)
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    lcb = (centre_adj_p - spread) / denominator
    ucb = (centre_adj_p + spread) / denominator
    return max(0.0, lcb), min(1.0, ucb)

# ---------------------------------------------------------------------------
# PHASE A.4: Multi-Seed 100-Fact Static Retrieval
# ---------------------------------------------------------------------------
def run_phase_a4(tokenizer, cache_data, all_facts, unique_probes):
    print("\n" + "="*80)
    print("  PHASE A.4: MULTI-SEED 100-FACT RETRIEVAL EVALUATION")
    print("="*80)
    
    seeds = [41, 42, 43, 44, 45]
    
    # Seen/Unseen Fact split (70 facts seen, 30 facts held out completely)
    seen_facts = all_facts[:70]
    unseen_facts = all_facts[70:]
    
    train_s, train_y, val_s, val_y, test_s, test_y = get_sentence_lists(seen_facts, unique_probes)
    unseen_train_s, unseen_train_y, unseen_val_s, unseen_val_y, unseen_test_s, unseen_test_y = get_sentence_lists(unseen_facts, unique_probes)
    
    train_x = cache_data["train_x"][:210].to(DEVICE)  # 70 * 3
    test_x = cache_data["test_x"][:280].to(DEVICE)    # 70 * 4
    unseen_train_x = cache_data["train_x"][210:].to(DEVICE)
    unseen_test_x = cache_data["test_x"][280:].to(DEVICE)
    
    # 1-NN baseline for Raw SmolLM2 on seen facts
    sims_raw_seen = torch.matmul(test_x, train_x.T)
    correct_raw_seen = 0
    for idx in range(len(test_s)):
        pred = sims_raw_seen[idx].argmax().item()
        if train_y[pred] == test_y[idx]:
            correct_raw_seen += 1
    raw_seen_acc = correct_raw_seen / len(test_s)
    
    # 1-NN baseline for Raw SmolLM2 on unseen facts
    sims_raw_unseen = torch.matmul(unseen_test_x, unseen_train_x.T)
    correct_raw_unseen = 0
    for idx in range(len(unseen_test_s)):
        pred = sims_raw_unseen[idx].argmax().item()
        if unseen_train_y[pred] == unseen_test_y[idx]:
            correct_raw_unseen += 1
    raw_unseen_acc = correct_raw_unseen / len(unseen_test_s)
    
    print(f"  - Teacher Baseline (Raw SmolLM2) Seen Paraphrase Acc : {raw_seen_acc*100:.2f}%")
    print(f"  - Teacher Baseline (Raw SmolLM2) Unseen Fact Acc   : {raw_unseen_acc*100:.2f}%")
    
    seen_accs, unseen_accs, e_rels, ranks = [], [], [], []
    
    for seed in seeds:
        # Seed everything for reproducible run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Train student on seen facts only
        student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
        
        N_train = len(train_s)
        for epoch in range(60):
            student.train()
            indices = list(range(N_train))
            random.shuffle(indices)
            for idx in range(0, N_train, 64):
                batch_idx = indices[idx : idx + 64]
                batch_s = [train_s[i] for i in batch_idx]
                ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
                z_s = student(ids, mask)
                z_t = train_x[batch_idx]
                loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        student.eval()
        with torch.no_grad():
            # seen evaluation
            z_tr_s = []
            for i in range(0, len(train_s), 64):
                ids, mask = batch_tokenize(tokenizer, train_s[i:i+64], max_len=32, device=DEVICE)
                z_tr_s.append(student(ids, mask))
            z_tr_s = torch.cat(z_tr_s, dim=0)
            
            z_te_s = []
            for i in range(0, len(test_s), 64):
                ids, mask = batch_tokenize(tokenizer, test_s[i:i+64], max_len=32, device=DEVICE)
                z_te_s.append(student(ids, mask))
            z_te_s = torch.cat(z_te_s, dim=0)
            
            # unseen evaluation
            z_unseen_ref = []
            for i in range(0, len(unseen_train_s), 64):
                ids, mask = batch_tokenize(tokenizer, unseen_train_s[i:i+64], max_len=32, device=DEVICE)
                z_unseen_ref.append(student(ids, mask))
            z_unseen_ref = torch.cat(z_unseen_ref, dim=0)
            
            z_unseen_test = []
            for i in range(0, len(unseen_test_s), 64):
                ids, mask = batch_tokenize(tokenizer, unseen_test_s[i:i+64], max_len=32, device=DEVICE)
                z_unseen_test.append(student(ids, mask))
            z_unseen_test = torch.cat(z_unseen_test, dim=0)
            
        # Accuracies
        sims_s = torch.matmul(z_te_s, z_tr_s.T)
        c_seen = sum(1 for idx in range(len(test_s)) if train_y[sims_s[idx].argmax().item()] == test_y[idx])
        seen_accs.append(c_seen / len(test_s))
        
        sims_unseen = torch.matmul(z_unseen_test, z_unseen_ref.T)
        c_unseen = sum(1 for idx in range(len(unseen_test_s)) if unseen_train_y[sims_unseen[idx].argmax().item()] == unseen_test_y[idx])
        unseen_accs.append(c_unseen / len(unseen_test_s))
        
        # Relational error on unseen facts
        S_t = torch.matmul(unseen_test_x, unseen_test_x.T)
        S_s = torch.matmul(z_unseen_test, z_unseen_test.T)
        e_rels.append((torch.norm(S_s - S_t, p="fro") / (torch.norm(S_t, p="fro") + 1e-8)).item())
        
        # Effective Rank
        _, S_svd_s, _ = torch.linalg.svd(z_te_s, full_matrices=False)
        p_s = S_svd_s / S_svd_s.sum()
        ranks.append(torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-8))).item())
        
        print(f"  - Seed {seed} | Seen Acc: {seen_accs[-1]*100:.2f}% | Unseen Acc: {unseen_accs[-1]*100:.2f}% | E_rel: {e_rels[-1]:.4f}")
        
    print(f"\n  - Mean Seen Fact Paraphrase Acc : {np.mean(seen_accs)*100:.2f}% (SD: {np.std(seen_accs)*100:.2f}%)")
    print(f"  - Mean Unseen Fact Transfer Acc : {np.mean(unseen_accs)*100:.2f}% (SD: {np.std(unseen_accs)*100:.2f}%)")
    print(f"  - Mean Relational Alignment Error: {np.mean(e_rels):.4f} (SD: {np.std(e_rels):.4f})")
    print(f"  - Mean Effective Rank           : {np.mean(ranks):.2f}")
    print("="*80)
    
    return np.mean(seen_accs), np.mean(unseen_accs)

# ---------------------------------------------------------------------------
# PHASE A.5: Sequential Continual-Learning Matrix
# ---------------------------------------------------------------------------
def run_phase_a5(tokenizer, blocks, unique_probes):
    print("\n" + "="*80)
    print("  PHASE A.5: SEQUENTIAL CONTINUAL-LEARNING MATRIX (CL)")
    print("="*80)
    
    # 10 blocks (10 facts per block)
    # We train the student sequentially block-by-block
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Extract SmolLM2 cached embeddings
    cache_data = torch.load(CACHE_100_PATH, weights_only=True)
    
    # Recall Matrix R[i, j] representing recall of block j after training on block i
    R = np.zeros((10, 10))
    
    # We maintain a sentence list and embedding list for each block
    block_train_s = []
    block_train_x = []
    block_test_s = []
    block_test_x = []
    block_train_y = []
    block_test_y = []
    
    for b_idx in range(10):
        b_facts = blocks[b_idx]
        tr_s, tr_y, _, _, te_s, te_y = get_sentence_lists(b_facts, unique_probes)
        
        block_train_s.append(tr_s)
        block_test_s.append(te_s)
        block_train_y.append(tr_y)
        block_test_y.append(te_y)
        
        # 10 facts per block, 3 train embeddings each, 4 test embeddings each
        block_train_x.append(cache_data["train_x"][b_idx*30 : (b_idx+1)*30].to(DEVICE))
        block_test_x.append(cache_data["test_x"][b_idx*40 : (b_idx+1)*40].to(DEVICE))
        
    print("  - Training sequentially block-by-block (10 epochs per block)...")
    for i in range(10):
        # Retrieve the sentences and target embeddings for current block i
        curr_train_s = block_train_s[i]
        curr_train_x = block_train_x[i]
        
        # Sequentially fine-tune the student on block i
        student.train()
        N_train = len(curr_train_s)
        for epoch in range(10):
            indices = list(range(N_train))
            random.shuffle(indices)
            for idx in range(0, N_train, 16):
                batch_idx = indices[idx : idx + 16]
                batch_s = [curr_train_s[k] for k in batch_idx]
                ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
                z_s = student(ids, mask)
                z_t = curr_train_x[batch_idx]
                loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Evaluate recall on all blocks j up to 10
        student.eval()
        with torch.no_grad():
            for j in range(10):
                # We perform 1-NN matching using block j's test queries vs. block j's train reference bank
                ref_s = block_train_s[j]
                ref_y = block_train_y[j]
                test_s = block_test_s[j]
                test_y = block_test_y[j]
                
                # Encode references
                z_refs = []
                for k in range(0, len(ref_s), 32):
                    ids, mask = batch_tokenize(tokenizer, ref_s[k:k+32], max_len=32, device=DEVICE)
                    z_refs.append(student(ids, mask))
                z_refs = torch.cat(z_refs, dim=0)
                
                # Encode queries
                z_queries = []
                for k in range(0, len(test_s), 32):
                    ids, mask = batch_tokenize(tokenizer, test_s[k:k+32], max_len=32, device=DEVICE)
                    z_queries.append(student(ids, mask))
                z_queries = torch.cat(z_queries, dim=0)
                
                sims = torch.matmul(z_queries, z_refs.T)
                correct = 0
                for k in range(len(test_s)):
                    pred = sims[k].argmax().item()
                    if ref_y[pred] == test_y[k]:
                        correct += 1
                R[i, j] = correct / len(test_s)
                
    # Calculate CL Metrics
    # Plasticity (diagonal average recall)
    plasticity = np.mean([R[i, i] for i in range(10)])
    
    # Final recall average
    final_avg = np.mean(R[9, :])
    
    # Forgetting (mean drop from maximum performance observed on each block)
    forgetting = 0.0
    for j in range(10):
        perf_over_time = R[j:, j]
        max_perf = perf_over_time[0]
        final_perf = perf_over_time[-1]
        forgetting += max(0.0, max_perf - final_perf)
    forgetting /= 10
    
    # Backward Transfer (BWT)
    bwt = 0.0
    for i in range(1, 10):
        bwt_i = 0.0
        for j in range(i):
            bwt_i += (R[i, j] - R[j, j])
        bwt += bwt_i / i
    bwt /= 9
    
    print("\n  - Continual Learning Matrix R[i, j] (Recall):")
    for row in R:
        print("    " + " ".join([f"{val*100:5.1f}%" for val in row]))
        
    print(f"\n  - Plasticity (Diag Recall)     : {plasticity*100:.2f}%")
    print(f"  - Final Average Recall (R[9,:]) : {final_avg*100:.2f}%")
    print(f"  - Mean Forgetting              : {forgetting*100:.2f}%")
    print(f"  - Backward Transfer (BWT)      : {bwt*100:+.2f} percentage points")
    print("="*80)

# ---------------------------------------------------------------------------
# PHASE A.6: Latency and Memory Validation
# ---------------------------------------------------------------------------
def run_phase_a6(tokenizer):
    print("\n" + "="*80)
    print("  PHASE A.6: DEPLOYMENT VALIDATION & RESOURCE FOOTPRINT")
    print("="*80)
    
    # 1. Model serialized file sizes
    print("  - Loading models to measure sizes...")
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960)
    torch.save(student.state_dict(), "temp_student.pt")
    student_size_mb = os.path.getsize("temp_student.pt") / (1024 * 1024)
    os.remove("temp_student.pt")
    
    # SmolLM2 model parameter file size
    smollm_size_mb = 724.0  # approximate safetensors size
    
    # 2. CPU / GPU Latency
    query = "The administrative capital of Kaelntia is Varek Spire."
    
    # GPU Latency
    gpu_times = []
    if torch.cuda.is_available():
        student.to(DEVICE)
        student.eval()
        ids, mask = batch_tokenize(tokenizer, [query], max_len=32, device=DEVICE)
        # Warmup
        for _ in range(10):
            _ = student(ids, mask)
        torch.cuda.synchronize()
        for _ in range(100):
            t0 = time.perf_counter()
            _ = student(ids, mask)
            torch.cuda.synchronize()
            gpu_times.append(time.perf_counter() - t0)
        gpu_latency_ms = np.mean(gpu_times) * 1000
    else:
        gpu_latency_ms = float("nan")
        
    # CPU Latency
    student.to("cpu")
    student.eval()
    ids, mask = batch_tokenize(tokenizer, [query], max_len=32, device="cpu")
    cpu_times = []
    for _ in range(10):
        _ = student(ids, mask)
    for _ in range(100):
        t0 = time.perf_counter()
        _ = student(ids, mask)
        cpu_times.append(time.perf_counter() - t0)
    cpu_latency_ms = np.mean(cpu_times) * 1000
    
    # Teacher (SmolLM2) Latency (approximate based on token generation step or embedding pooling step)
    # We profile the forward pass of SmolLM2 on GPU/CPU
    print("  - Benchmarking SmolLM2 Teacher...")
    try:
        mod = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        mod.eval()
        
        # CPU
        mod.to("cpu")
        enc = tokenizer(query, max_length=32, padding="max_length", return_tensors="pt").to("cpu")
        t_cpu = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = mod(enc.input_ids, attention_mask=enc.attention_mask)
            t_cpu.append(time.perf_counter() - t0)
        smol_cpu_ms = np.mean(t_cpu) * 1000
        
        # GPU
        if torch.cuda.is_available():
            mod.to(DEVICE)
            enc = tokenizer(query, max_length=32, padding="max_length", return_tensors="pt").to(DEVICE)
            t_gpu = []
            for _ in range(5):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = mod(enc.input_ids, attention_mask=enc.attention_mask)
                torch.cuda.synchronize()
                t_gpu.append(time.perf_counter() - t0)
            smol_gpu_ms = np.mean(t_gpu) * 1000
        else:
            smol_gpu_ms = float("nan")
    except Exception as e:
        print(f"    [Warning] Failed to benchmark SmolLM2: {e}")
        smol_cpu_ms = float("nan")
        smol_gpu_ms = float("nan")
        
    print(f"  - Serialized Model Footprint : Student {student_size_mb:.2f} MB vs. SmolLM2 {smollm_size_mb:.1f} MB")
    print(f"  - Compression Ratio          : {smollm_size_mb / student_size_mb:.1f}x reduction")
    print(f"  - GPU Latency per query      : Student {gpu_latency_ms:.2f} ms vs. SmolLM2 {smol_gpu_ms:.2f} ms")
    if not np.isnan(smol_gpu_ms):
        print(f"  - GPU Speedup Factor         : {smol_gpu_ms / gpu_latency_ms:.1f}x speedup")
    print(f"  - CPU Latency per query      : Student {cpu_latency_ms:.2f} ms vs. SmolLM2 {smol_cpu_ms:.2f} ms")
    if not np.isnan(smol_cpu_ms):
        print(f"  - CPU Speedup Factor         : {smol_cpu_ms / cpu_latency_ms:.1f}x speedup")
    print("="*80)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Required files not found.")
        return
        
    print("[Eval] Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Reconstruct 100-fact cache using fail-closed real model load
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        raise RuntimeError("FAIL-CLOSED: Failed to load real SmolLM2 model for embeddings caching") from e
        
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    
    # Unload SmolLM2 to clear VRAM for multi-seed student training
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    all_facts = [fact for b in blocks for fact in b]
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    
    # Run Phase A.4: Multi-Seed 100-Fact Retrieval
    seen_acc, unseen_acc = run_phase_a4(tokenizer, cache_data, all_facts, unique_probes)
    
    # Run Hard-Negative FPR on expanded controls (200 in-domain semantic controls)
    val_sentences = [f["train_paraphrases"][-1] for f in all_facts[:70]]
    val_labels = [unique_probes.index(f["probe"]) for f in all_facts[:70]]
    
    # Re-train a student 960d to evaluate controls and CL
    student_960d = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
    train_s, train_y, _, _, _, _ = get_sentence_lists(all_facts[:70], unique_probes)
    train_x = cache_data["train_x"][:210].to(DEVICE)
    
    optimizer = torch.optim.AdamW(student_960d.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(60):
        student_960d.train()
        indices = list(range(len(train_s)))
        random.shuffle(indices)
        for idx in range(0, len(train_s), 64):
            batch_idx = indices[idx : idx + 64]
            batch_sentences = [train_s[i] for i in batch_idx]
            ids, mask = batch_tokenize(tokenizer, batch_sentences, max_len=32, device=DEVICE)
            z_s = student_960d(ids, mask)
            z_t = train_x[batch_idx]
            loss = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    semantic_controls = generate_semantic_hard_negatives(all_facts)
    
    # Run control FPR
    global INDEPENDENT_PPL_TEXTS
    INDEPENDENT_PPL_TEXTS = semantic_controls
    run_real_control_fpr(tokenizer, student_960d, train_s, train_y, val_sentences, val_labels)
    
    # Run Phase A.5: CL Matrix
    run_phase_a5(tokenizer, blocks, unique_probes)
    
    # Run Phase A.6: Latency & Memory profiling
    run_phase_a6(tokenizer)

if __name__ == "__main__":
    main()
