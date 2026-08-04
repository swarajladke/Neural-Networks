"""
audit_smollm2_failures.py — Audit of SmolLM2-360M 1-NN Retrieval Failures
===========================================================================
Analyzes the 110 failing test queries out of 400 test queries for raw SmolLM2-360M embeddings.
Checks how many failing test queries have their nearest incorrect neighbour inside a near-duplicate
fact pair (cosine similarity > 0.95 between the true fact's reference embedding and the neighbour fact's reference embedding).
"""

import os
import torch
import torch.nn.functional as F

def audit_failures():
    cache_path = "smollm2_embeddings_100slots.pt"
    if not os.path.exists(cache_path):
        print(f"[Cache] Embeddings cache {cache_path} not found. Reconstructing automatically...")
        import json
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
        with open("agnis_scaling_dataset.json", "r") as f:
            blocks = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        model.eval()
        data = ensure_100_fact_embeddings(tokenizer, model, blocks)
    else:
        data = torch.load(cache_path, map_location="cpu")
    train_x = data["train_x"]  # 300 x 960 (3 per fact)
    train_y = data["train_y"]  # 300 (fact indices 0..99)
    test_x = data["test_x"]    # 400 x 960 (4 per fact)
    test_y = data["test_y"]    # 400 (fact indices 0..99)

    # Compute fact centroid reference embeddings (100 x 960)
    fact_centroids = torch.zeros(100, train_x.shape[1])
    for fact_idx in range(100):
        mask = (train_y == fact_idx)
        fact_centroids[fact_idx] = F.normalize(train_x[mask].mean(dim=0), dim=-1)

    # Inter-fact cosine similarity matrix (100 x 100)
    fact_sim_matrix = torch.matmul(fact_centroids, fact_centroids.T)

    total_queries = len(test_x)
    correct_count = 0
    failing_queries = []

    for i in range(total_queries):
        q = test_x[i]
        sims = torch.matmul(train_x, q.unsqueeze(-1)).squeeze(-1)
        best_idx = torch.argmax(sims).item()
        pred_fact = train_y[best_idx].item()
        true_fact = test_y[i].item()

        if pred_fact == true_fact:
            correct_count += 1
        else:
            # Distance/similarity between true fact and incorrectly predicted fact
            inter_fact_sim = fact_sim_matrix[true_fact, pred_fact].item()
            failing_queries.append({
                "query_idx": i,
                "true_fact": true_fact,
                "pred_fact": pred_fact,
                "inter_fact_sim": inter_fact_sim,
                "is_near_duplicate": inter_fact_sim > 0.95
            })

    num_failed = len(failing_queries)
    num_near_duplicates = sum(1 for f in failing_queries if f["is_near_duplicate"])
    pct_near_duplicates = (num_near_duplicates / num_failed * 100.0) if num_failed > 0 else 0.0

    print("="*80)
    print(f"  SmolLM2-360M RAW 1-NN RETRIEVAL FAILURE AUDIT")
    print("="*80)
    print(f"  Total Test Queries                     : {total_queries}")
    print(f"  Correct Queries                        : {correct_count} ({correct_count/total_queries*100:.2f}%)")
    print(f"  Failing Queries                        : {num_failed} ({num_failed/total_queries*100:.2f}%)")
    print(f"  Failures in Near-Duplicate Pairs (>0.95): {num_near_duplicates} out of {num_failed} ({pct_near_duplicates:.2f}%)")
    print("="*80)

if __name__ == "__main__":
    audit_failures()
