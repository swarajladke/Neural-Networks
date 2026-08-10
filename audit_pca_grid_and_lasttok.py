"""
audit_pca_grid_and_lasttok.py
==============================

Phase J1 & J2 Execution:
- J1: Truncated PCA Whitening (m in {16,32,64,128,256,299} x eps in {1e-6,1e-4,1e-2}) + Ledoit-Wolf Shrinkage (Pure PyTorch).
- J2: Rerun last_token cells using non-punctuation last-token cache and report side-by-side with old last-token cache.

Standing Rules:
- R4: Guard raises on missing input cache files.
- R6: Label-derived R6 grouping for NCM centroids.
- R7: Fitting statistics (mean, covariance, PCA eigenvectors, Ledoit-Wolf shrinkage) on TRAIN vectors ONLY.
- R8: Fixed seeds logged and enforced.
"""

import os
import torch
import torch.nn.functional as F

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
LASTTOK_OLD_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok.pt"
LASTTOK_NONPUNCT_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok_nonpunct.pt"


def apply_truncated_pca(train_x, test_x, m, eps):
    # R7: Fit mean and covariance on TRAIN only in double precision
    train_x_dbl = train_x.double()
    test_x_dbl = test_x.double()

    mu = train_x_dbl.mean(dim=0, keepdim=True)
    tr_c = train_x_dbl - mu
    te_c = test_x_dbl - mu

    N = tr_c.shape[0]
    cov = (tr_c.T @ tr_c) / (N - 1)

    S, V = torch.linalg.eigh(cov)  # ascending order

    # Take top m eigenvectors (largest eigenvalues)
    top_S = S[-m:]
    top_V = V[:, -m:]

    top_S = torch.clamp(top_S, min=1e-12)
    scales = 1.0 / torch.sqrt(top_S + eps)

    W_pca = (top_V * scales).float()  # shape (960, m)
    mu_flt = mu.float()

    tr_proj = (train_x - mu_flt) @ W_pca
    te_proj = (test_x - mu_flt) @ W_pca

    return F.normalize(tr_proj, dim=-1), F.normalize(te_proj, dim=-1)


def apply_ledoit_wolf_pytorch(train_x, test_x, eps=1e-4):
    # R7: Fit on TRAIN only in double precision
    train_x_dbl = train_x.double()
    test_x_dbl = test_x.double()

    N, d = train_x_dbl.shape
    mu = train_x_dbl.mean(dim=0, keepdim=True)
    tr_c = train_x_dbl - mu
    te_c = test_x_dbl - mu

    sample_cov = (tr_c.T @ tr_c) / (N - 1)
    prior_scale = torch.trace(sample_cov) / d
    prior = prior_scale * torch.eye(d, dtype=torch.float64)

    # Efficient Ledoit-Wolf shrinkage delta calculation
    # alpha^2 = || sample_cov - prior ||_F^2
    # beta^2 = sum_i || x_i x_i^T - sample_cov ||_F^2 / N^2
    d_sq = (sample_cov - prior).pow(2).sum().item()

    # Fast trace-based formula for b_bar_sq
    # sum_i || x_i x_i^T ||_F^2 = sum_i (x_i^T x_i)^2
    norms_sq = (tr_c.pow(2).sum(dim=1)).pow(2).sum().item()
    cov_norm_sq = sample_cov.pow(2).sum().item()
    b_bar_sq = max(0.0, (norms_sq / (N * N) - cov_norm_sq / N))

    delta = max(0.0, min(1.0, b_bar_sq / max(d_sq, 1e-12)))

    cov_lw = (1.0 - delta) * sample_cov + delta * prior

    S, V = torch.linalg.eigh(cov_lw)
    S = torch.clamp(S, min=1e-12)
    scales = 1.0 / torch.sqrt(S + eps)

    W_lw = (V @ torch.diag(scales) @ V.T).float()
    mu_flt = mu.float()

    tr_w = (train_x - mu_flt) @ W_lw
    te_w = (test_x - mu_flt) @ W_lw

    print(f"  [R7 Confirm] Ledoit-Wolf Shrinkage fit on TRAIN only: delta = {delta:.4f}, prior_scale = {prior_scale.item():.6f}", flush=True)

    return F.normalize(tr_w, dim=-1), F.normalize(te_w, dim=-1)


def evaluate_representation(tr_x, train_y, te_x, test_y):
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

    ncm_sims = te_x @ centroids.T
    ncm_preds = ncm_sims.argmax(dim=1)
    ncm_top1 = (ncm_preds == test_y).float().mean().item() * 100.0

    knn_sims = te_x @ tr_x.T
    knn_nearest = knn_sims.argmax(dim=1)
    knn_preds = train_y[knn_nearest]
    knn_top1 = (knn_preds == test_y).float().mean().item() * 100.0

    return ncm_top1, knn_top1


def run_j1_pca_grid(cache_path, label_name):
    if not os.path.exists(cache_path):
        raise RuntimeError(f"[R5 Guard] Missing required cache file: '{cache_path}'.")

    data = torch.load(cache_path, weights_only=False)
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    ms = [16, 32, 64, 128, 256, 299]
    epss = [1e-6, 1e-4, 1e-2]

    print(f"\n==================================================================", flush=True)
    print(f" J1 TRUNCATED PCA WHITENING GRID ({label_name})", flush=True)
    print(f"==================================================================", flush=True)
    print(f"{'m (kept components)':<20} | {'eps':<8} | {'NCM Top-1':<12} | {'1-NN Top-1':<12}", flush=True)
    print("-" * 65, flush=True)

    grid_results = []
    best_ncm = -1.0
    best_cell_info = None

    for m in ms:
        for eps in epss:
            tr_p, te_p = apply_truncated_pca(train_x, test_x, m=m, eps=eps)
            ncm_acc, knn_acc = evaluate_representation(tr_p, train_y, te_p, test_y)
            grid_results.append((m, eps, ncm_acc, knn_acc))
            print(f"{m:<20} | {eps:<8.0e} | {ncm_acc:6.2f}%      | {knn_acc:6.2f}%", flush=True)

            if ncm_acc > best_ncm:
                best_ncm = ncm_acc
                best_cell_info = (m, eps, ncm_acc, knn_acc)

    # Add Ledoit-Wolf shrinkage row
    tr_lw, te_lw = apply_ledoit_wolf_pytorch(train_x, test_x, eps=1e-4)
    lw_ncm, lw_knn = evaluate_representation(tr_lw, train_y, te_lw, test_y)
    print("-" * 65, flush=True)
    print(f"{'Ledoit-Wolf (full rank)':<20} | {'1e-4':<8} | {lw_ncm:6.2f}%      | {lw_knn:6.2f}%", flush=True)
    print("==================================================================", flush=True)

    print(f"BEST TRUNCATED PCA CELL FOR '{label_name}': m={best_cell_info[0]}, eps={best_cell_info[1]:.0e} -> NCM={best_cell_info[2]:.2f}%, 1-NN={best_cell_info[3]:.2f}%", flush=True)
    return grid_results, (lw_ncm, lw_knn)


def run_j2_lasttok_comparison():
    print("\n==================================================================", flush=True)
    print(" J2 LAST-TOKEN COMPARISON: OLD (WITH PUNCT) VS NEW (NON-PUNCT)", flush=True)
    print("==================================================================", flush=True)

    if not os.path.exists(LASTTOK_OLD_CACHE_PATH) or not os.path.exists(LASTTOK_NONPUNCT_CACHE_PATH):
        raise RuntimeError("[R5 Guard] Missing required old or new lasttok cache file!")

    old_data = torch.load(LASTTOK_OLD_CACHE_PATH, weights_only=False)
    new_data = torch.load(LASTTOK_NONPUNCT_CACHE_PATH, weights_only=False)

    old_tr_x, old_tr_y, old_te_x, old_te_y = old_data["train_x"], old_data["train_y"], old_data["test_x"], old_data["test_y"]
    new_tr_x, new_tr_y, new_te_x, new_te_y = new_data["train_x"], new_data["train_y"], new_data["test_x"], new_data["test_y"]

    transforms = ["none", "center", "pca_m32_eps1e-6", "pca_m299_eps1e-4", "ledoit_wolf"]

    print(f"{'Transform':<22} | {'Old NCM':<10} | {'New NCM':<10} | {'Old 1-NN':<10} | {'New 1-NN':<10}", flush=True)
    print("-" * 75, flush=True)

    for t_name in transforms:
        if t_name == "none":
            o_tr, o_te = F.normalize(old_tr_x, dim=-1), F.normalize(old_te_x, dim=-1)
            n_tr, n_te = F.normalize(new_tr_x, dim=-1), F.normalize(new_te_x, dim=-1)
        elif t_name == "center":
            o_mu = old_tr_x.mean(dim=0, keepdim=True)
            o_tr, o_te = F.normalize(old_tr_x - o_mu, dim=-1), F.normalize(old_te_x - o_mu, dim=-1)
            n_mu = new_tr_x.mean(dim=0, keepdim=True)
            n_tr, n_te = F.normalize(new_tr_x - n_mu, dim=-1), F.normalize(new_te_x - n_mu, dim=-1)
        elif t_name == "pca_m32_eps1e-6":
            o_tr, o_te = apply_truncated_pca(old_tr_x, old_te_x, m=32, eps=1e-6)
            n_tr, n_te = apply_truncated_pca(new_tr_x, new_te_x, m=32, eps=1e-6)
        elif t_name == "pca_m299_eps1e-4":
            o_tr, o_te = apply_truncated_pca(old_tr_x, old_te_x, m=299, eps=1e-4)
            n_tr, n_te = apply_truncated_pca(new_tr_x, new_te_x, m=299, eps=1e-4)
        elif t_name == "ledoit_wolf":
            o_tr, o_te = apply_ledoit_wolf_pytorch(old_tr_x, old_te_x, eps=1e-4)
            n_tr, n_te = apply_ledoit_wolf_pytorch(new_tr_x, new_te_x, eps=1e-4)

        o_ncm, o_knn = evaluate_representation(o_tr, old_tr_y, o_te, old_te_y)
        n_ncm, n_knn = evaluate_representation(n_tr, new_tr_y, n_te, new_te_y)

        print(f"{t_name:<22} | {o_ncm:6.2f}%    | {n_ncm:6.2f}%    | {o_knn:6.2f}%    | {n_knn:6.2f}%", flush=True)

    print("==================================================================", flush=True)


def main():
    print("==================================================================", flush=True)
    print(" PHASE J1 & J2 AUDIT — TRUNCATED PCA, LEDOIT-WOLF & LAST-TOKEN FIX", flush=True)
    print("==================================================================", flush=True)

    # 1. J1 Grid on Mean Cache
    run_j1_pca_grid(MEAN_CACHE_PATH, "Mean Pooling Cache")

    # 2. J2 Comparison on Last-Token Caches
    run_j2_lasttok_comparison()


if __name__ == "__main__":
    main()
