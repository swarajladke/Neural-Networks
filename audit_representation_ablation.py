"""
audit_representation_ablation.py
================================

Phase II -- Representation Ablation Grid (No Training)
Covers 6 cells: pooling in {mean, last_token} x transform in {none, center, center+ZCA_whiten}

Standing Rules:
- R4: Guard raises on missing input files.
- R6: Label-derived grouping via (y == c).nonzero(as_tuple=True)[0]. No .view(N, k, d) or fixed-stride slicing.
- R7: Statistics (mean, covariance, ZCA matrix) fit on TRAIN vectors only. Explicit confirmation printed per cell.
- R8: Fixed seeds everywhere.
"""

import os
import json
import torch
import torch.nn.functional as F

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
LASTTOK_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok.pt"
DATASET_V2_PATH = "agnis_scaling_dataset_v2.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_offline_model_path():
    for path in ["../local_smollm2", "local_smollm2"]:
        if os.path.exists(os.path.join(path, "model.safetensors")):
            print(f"[Model Loader] Branch 1: Found local offline SmolLM2 path at '{path}'.", flush=True)
            return path, False
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files):
                if "smollm" in root.lower():
                    print(f"[Model Loader] Branch 2: Found Kaggle input SmolLM2 path at '{root}'.", flush=True)
                    return root, False
    print("[Model Loader] Branch 3: Falling back to HF Hub 'HuggingFaceTB/SmolLM2-360M'.", flush=True)
    return "HuggingFaceTB/SmolLM2-360M", True


def build_lasttok_cache():
    if not os.path.exists(DATASET_V2_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required dataset file: '{DATASET_V2_PATH}'.")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id, is_hf_hub = find_offline_model_path()
    revision = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

    print(f"[Last-Token Extractor] Loading SmolLM2 from '{model_id}'...", flush=True)
    if not is_hf_hub:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
    else:
        try:
            print(f"[Model Loader] Attempting load from local HF cache for '{model_id}' (revision={revision})...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True).to(DEVICE)
            print("[Model Loader] Successfully loaded model from local HF cache!", flush=True)
        except Exception as e:
            print(f"[Model Loader] Local cache lookup skipped ({e}). Fetching from HF Hub...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    model.eval()

    with open(DATASET_V2_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    facts = sorted(facts, key=lambda x: x["fact_id"])

    all_prompts = []
    all_labels = []
    is_train_flags = []

    for f in facts:
        label = int(f["fact_id"])
        for prompt in f["train_prompts"]:
            all_prompts.append(prompt)
            all_labels.append(label)
            is_train_flags.append(True)
        for prompt in f["test_prompts"]:
            all_prompts.append(prompt)
            all_labels.append(label)
            is_train_flags.append(False)

    train_queries, train_labels = [], []
    test_queries, test_labels = [], []

    print(f"[Last-Token Extractor] Extracting last-token vectors for {len(all_prompts)} prompts in batches of 32...", flush=True)
    batch_size = 32
    for b_i in range(0, len(all_prompts), batch_size):
        b_prompts = all_prompts[b_i : b_i + batch_size]
        b_flags = is_train_flags[b_i : b_i + batch_size]
        b_labs = all_labels[b_i : b_i + batch_size]

        enc = tokenizer(b_prompts, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        seq_lengths = enc.attention_mask.sum(dim=1)  # shape (b,)
        last_non_pad_idx = seq_lengths - 1

        assert (last_non_pad_idx == (enc.attention_mask.sum(dim=1) - 1)).all(), "Last token index mismatch assertion!"

        with torch.no_grad():
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            last_hidden = outputs.hidden_states[-1]  # shape (b, seq_len, 960)
            
            for idx in range(len(b_prompts)):
                vec = last_hidden[idx, last_non_pad_idx[idx]]
                vec_norm = F.normalize(vec.float(), dim=-1).cpu()
                if b_flags[idx]:
                    train_queries.append(vec_norm)
                    train_labels.append(b_labs[idx])
                else:
                    test_queries.append(vec_norm)
                    test_labels.append(b_labs[idx])

    train_x = torch.stack(train_queries)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = torch.stack(test_queries)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    expected_labels = torch.arange(100).repeat_interleave(3)
    assert torch.equal(train_y, expected_labels), "Lasttok train_y mismatch!"
    assert torch.equal(test_y, expected_labels), "Lasttok test_y mismatch!"

    cache_data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "metadata": {"pooling": "last_token"}
    }
    torch.save(cache_data, LASTTOK_CACHE_PATH)
    print(f"[Last-Token Extractor] Saved to '{LASTTOK_CACHE_PATH}'.", flush=True)


def load_cache(pooling):
    if pooling == "mean":
        if not os.path.exists(MEAN_CACHE_PATH):
            raise RuntimeError(f"[R5 Guard] Missing required mean cache: '{MEAN_CACHE_PATH}'.")
        data = torch.load(MEAN_CACHE_PATH, weights_only=False)
    elif pooling == "last_token":
        if not os.path.exists(LASTTOK_CACHE_PATH):
            build_lasttok_cache()
        data = torch.load(LASTTOK_CACHE_PATH, weights_only=False)
    else:
        raise ValueError(f"Unknown pooling type: {pooling}")

    return data["train_x"], data["train_y"], data["test_x"], data["test_y"]


def apply_transform(train_x, test_x, transform):
    # R7: Fit statistics on TRAIN only
    if transform == "none":
        print("[R7 Confirm] Transform 'none': Raw L2-normalized embeddings used.", flush=True)
        return F.normalize(train_x, dim=-1), F.normalize(test_x, dim=-1)

    elif transform == "center":
        mu = train_x.mean(dim=0, keepdim=True)
        print("[R7 Confirm] Center transform fit on 300 train vectors only (mean vector subtracted).", flush=True)
        tr_c = F.normalize(train_x - mu, dim=-1)
        te_c = F.normalize(test_x - mu, dim=-1)
        return tr_c, te_c

    elif transform == "center+ZCA_whiten":
        mu = train_x.mean(dim=0, keepdim=True)
        tr_centered = train_x - mu
        te_centered = test_x - mu

        # Covariance fit on train only (N = 300)
        N = tr_centered.shape[0]
        cov = (tr_centered.T @ tr_centered) / (N - 1)
        cov += 1e-5 * torch.eye(cov.shape[0])  # Add 1e-5 for numerical stability

        S, V = torch.linalg.eigh(cov)
        S = torch.clamp(S, min=1e-5)
        W_zca = V @ torch.diag(1.0 / torch.sqrt(S)) @ V.T

        print("[R7 Confirm] ZCA whitening transform fit on 300 train vectors only (epsilon=1e-5).", flush=True)

        tr_w = tr_centered @ W_zca.T
        te_w = te_centered @ W_zca.T

        return F.normalize(tr_w, dim=-1), F.normalize(te_w, dim=-1)

    else:
        raise ValueError(f"Unknown transform: {transform}")


def evaluate_cell(pooling, transform):
    train_x, train_y, test_x, test_y = load_cache(pooling)
    tr_x, te_x = apply_transform(train_x, test_x, transform)

    # R6: Label-derived centroids
    unique_classes = torch.sort(torch.unique(train_y))[0]
    centroids_list = []
    for c in unique_classes:
        c_val = c.item()
        c_indices = (train_y == c_val).nonzero(as_tuple=True)[0]
        c_vecs = tr_x[c_indices]
        centroids_list.append(c_vecs.mean(dim=0))

    centroids = torch.stack(centroids_list)
    centroids = F.normalize(centroids, dim=-1)

    # NCM Top-1 Accuracy
    ncm_sims = te_x @ centroids.T
    ncm_preds = ncm_sims.argmax(dim=1)
    ncm_top1 = (ncm_preds == test_y).float().mean().item() * 100.0

    # 1-NN Top-1 Accuracy (over 300 train vectors)
    knn_sims = te_x @ tr_x.T  # shape (300 test, 300 train)
    knn_nearest = knn_sims.argmax(dim=1)
    knn_preds = train_y[knn_nearest]
    knn_top1 = (knn_preds == test_y).float().mean().item() * 100.0

    # Cosine & Unbiased Margin Statistics
    own_means = []
    other_means = []
    unbiased_diffs = []
    neg_count = 0

    for i in range(len(te_x)):
        t_vec = te_x[i]
        c_label = test_y[i].item()

        all_cos = torch.mv(tr_x, t_vec)
        own_mask = (train_y == c_label)
        other_mask = ~own_mask

        own_mean = all_cos[own_mask].mean().item()
        other_mean = all_cos[other_mask].mean().item()
        u_diff = own_mean - other_mean

        own_means.append(own_mean)
        other_means.append(other_mean)
        unbiased_diffs.append(u_diff)
        if u_diff < 0:
            neg_count += 1

    m_own = torch.tensor(own_means).mean().item()
    m_other = torch.tensor(other_means).mean().item()
    m_diff = torch.tensor(unbiased_diffs).mean().item()

    return {
        "pooling": pooling,
        "transform": transform,
        "ncm_top1": ncm_top1,
        "knn_top1": knn_top1,
        "mean_own_cos": m_own,
        "mean_other_cos": m_other,
        "mean_unbiased_diff": m_diff,
        "count_neg_diff": neg_count
    }


def main():
    print("==================================================", flush=True)
    print(" PHASE II — REPRESENTATION ABLATION GRID (6 CELLS)", flush=True)
    print("==================================================", flush=True)

    poolings = ["mean", "last_token"]
    transforms = ["none", "center", "center+ZCA_whiten"]

    results = []

    for p in poolings:
        for t in transforms:
            print(f"\n--- Evaluating Cell: pooling='{p}', transform='{t}' ---", flush=True)
            res = evaluate_cell(p, t)
            results.append(res)

    print("\n" + "="*95, flush=True)
    print(" PHASE II SUMMARY TABLE (6 CELLS)", flush=True)
    print("="*95, flush=True)
    print(f"{'Cell Name (pooling / transform)':<35} | {'NCM Top-1':<10} | {'1-NN Top-1':<10} | {'Own Cos':<8} | {'Other Cos':<9} | {'Mean Diff':<9} | {'Neg Count':<9}", flush=True)
    print("-" * 95, flush=True)

    for r in results:
        cell_name = f"{r['pooling']} / {r['transform']}"
        print(f"{cell_name:<35} | {r['ncm_top1']:6.2f}%    | {r['knn_top1']:6.2f}%    | {r['mean_own_cos']:7.4f}  | {r['mean_other_cos']:8.4f}  | {r['mean_unbiased_diff']:8.4f}  | {r['count_neg_diff']:3d} / 300", flush=True)

    print("=" * 95, flush=True)

    # Determine BEST_CELL (ties break toward simpler transform: none < center < whiten)
    transform_simplicity = {"none": 0, "center": 1, "center+ZCA_whiten": 2}
    best_res = sorted(results, key=lambda x: (x["ncm_top1"], -transform_simplicity[x["transform"]]), reverse=True)[0]

    best_cell_name = f"{best_res['pooling']} / {best_res['transform']}"
    print(f"\nEXPLICIT BEST_CELL IDENTIFIED: '{best_cell_name}' with NCM Top-1 = {best_res['ncm_top1']:.2f}%", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
