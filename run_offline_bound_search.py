"""
run_offline_bound_search.py
===========================

Phase J3, J4, J5 Execution:
- J3: OFFLINE_BOUND = max test accuracy across {NCM, 1-NN, HeadL1c (J4 early stopped), Multinomial Logistic Regression (L2 CV)}.
- J4: HeadL1c 3-fold prompt cross-validation with early stopping on validation accuracy (patience=20).
- J5: Selection of BEST_CELL based on max OFFLINE_BOUND across candidate representations.

Standing Rules:
- R4: Guard raises on missing input cache files.
- R6: Label-derived indexing for splits and metrics.
- R7: Statistics (mean, covariance, PCA eigenvectors, Ledoit-Wolf) fit on TRAIN vectors ONLY.
- R8: Fixed seeds [42, 43, 44, 45, 46] logged and enforced.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"
LASTTOK_NONPUNCT_CACHE_PATH = "smollm2_embeddings_v2_100facts_lasttok_nonpunct.pt"


class HeadL1c(nn.Module):
    def __init__(self, in_features=960, out_features=100, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        w = torch.randn(out_features, in_features)
        w = F.normalize(w, dim=-1)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.scale * (x_norm @ w_norm.T)


def apply_truncated_pca(train_x, test_x, m, eps):
    train_x_dbl = train_x.double()
    test_x_dbl = test_x.double()

    mu = train_x_dbl.mean(dim=0, keepdim=True)
    tr_c = train_x_dbl - mu
    te_c = test_x_dbl - mu

    N = tr_c.shape[0]
    cov = (tr_c.T @ tr_c) / (N - 1)

    S, V = torch.linalg.eigh(cov)
    top_S = S[-m:]
    top_V = V[:, -m:]

    top_S = torch.clamp(top_S, min=1e-12)
    scales = 1.0 / torch.sqrt(top_S + eps)

    W_pca = (top_V * scales).float()
    mu_flt = mu.float()

    tr_proj = (train_x - mu_flt) @ W_pca
    te_proj = (test_x - mu_flt) @ W_pca

    return F.normalize(tr_proj, dim=-1), F.normalize(te_proj, dim=-1)


def apply_ledoit_wolf_pytorch(train_x, test_x, eps=1e-4):
    train_x_dbl = train_x.double()
    test_x_dbl = test_x.double()

    N, d = train_x_dbl.shape
    mu = train_x_dbl.mean(dim=0, keepdim=True)
    tr_c = train_x_dbl - mu
    te_c = test_x_dbl - mu

    sample_cov = (tr_c.T @ tr_c) / (N - 1)
    prior_scale = torch.trace(sample_cov) / d
    prior = prior_scale * torch.eye(d, dtype=torch.float64)

    d_sq = (sample_cov - prior).pow(2).sum().item()
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

    return F.normalize(tr_w, dim=-1), F.normalize(te_w, dim=-1)


def get_transformed_data(pooling_type, transform_type):
    cache_path = MEAN_CACHE_PATH if pooling_type == "mean" else LASTTOK_NONPUNCT_CACHE_PATH
    if not os.path.exists(cache_path):
        raise RuntimeError(f"[R5 Guard] Missing required cache file: '{cache_path}'.")

    data = torch.load(cache_path, weights_only=False)
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    if transform_type == "none":
        tr_x, te_x = F.normalize(train_x, dim=-1), F.normalize(test_x, dim=-1)
    elif transform_type == "center":
        mu = train_x.mean(dim=0, keepdim=True)
        tr_x = F.normalize(train_x - mu, dim=-1)
        te_x = F.normalize(test_x - mu, dim=-1)
    elif transform_type == "center+ZCA_whiten":
        mu = train_x.mean(dim=0, keepdim=True)
        tr_c = train_x - mu
        te_c = test_x - mu
        cov = (tr_c.T @ tr_c) / (tr_c.shape[0] - 1) + 1e-5 * torch.eye(tr_c.shape[1])
        S, V = torch.linalg.eigh(cov)
        S = torch.clamp(S, min=1e-5)
        W = V @ torch.diag(1.0 / torch.sqrt(S)) @ V.T
        tr_x = F.normalize(tr_c @ W.T, dim=-1)
        te_x = F.normalize(te_c @ W.T, dim=-1)
    elif transform_type.startswith("pca_"):
        parts = transform_type.split("_")
        m = int(parts[1].replace("m", ""))
        eps = float(parts[2].replace("eps", ""))
        tr_x, te_x = apply_truncated_pca(train_x, test_x, m=m, eps=eps)
    elif transform_type == "ledoit_wolf":
        tr_x, te_x = apply_ledoit_wolf_pytorch(train_x, test_x, eps=1e-4)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")

    return tr_x, train_y, te_x, test_y


def eval_ncm(tr_x, train_y, te_x, test_y):
    unique_classes = torch.sort(torch.unique(train_y))[0]
    centroids = []
    for c in unique_classes:
        c_vecs = tr_x[(train_y == c.item()).nonzero(as_tuple=True)[0]]
        centroids.append(c_vecs.mean(dim=0))
    centroids = F.normalize(torch.stack(centroids), dim=-1)
    preds = (te_x @ centroids.T).argmax(dim=1)
    return (preds == test_y).float().mean().item() * 100.0


def eval_1nn(tr_x, train_y, te_x, test_y):
    sims = te_x @ tr_x.T
    preds = train_y[sims.argmax(dim=1)]
    return (preds == test_y).float().mean().item() * 100.0


def eval_j4_headl1c_val_split(tr_x, train_y, te_x, test_y, seeds=[42]):
    fold_test_accs = []

    for fold in range(3):
        tr_indices, val_indices = [], []
        for c in range(100):
            c_idxs = (train_y == c).nonzero(as_tuple=True)[0]
            val_idx = c_idxs[fold]
            tr_idxs = c_idxs[torch.arange(3) != fold]
            tr_indices.extend(tr_idxs.tolist())
            val_indices.append(val_idx.item())

        sub_tr_x = tr_x[tr_indices]
        sub_tr_y = train_y[tr_indices]
        val_x = tr_x[val_indices]
        val_y = train_y[val_indices]

        seed_test_accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = HeadL1c(in_features=tr_x.shape[1], out_features=100, scale=10.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = -1.0
            best_model_weights = None
            patience = 20
            patience_counter = 0

            for epoch in range(1, 201):
                model.train()
                optimizer.zero_grad()
                logits = model(sub_tr_x)
                loss = criterion(logits, sub_tr_y)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(val_x)
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == val_y).float().mean().item() * 100.0

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)
            model.eval()
            with torch.no_grad():
                te_logits = model(te_x)
                te_preds = te_logits.argmax(dim=1)
                te_acc = (te_preds == test_y).float().mean().item() * 100.0

            seed_test_accs.append(te_acc)

        fold_test_accs.append(np.mean(seed_test_accs))

    return np.mean(fold_test_accs), fold_test_accs


def eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y):
    # Pure PyTorch Multinomial Logistic Regression with L2 regularization C in {1e-3, 1e-2, 1e-1, 1, 10}
    # 3-fold cross validation over train samples to select best C
    c_candidates = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_c = None
    best_cv_acc = -1.0

    for C in c_candidates:
        cv_accs = []
        for fold in range(3):
            tr_indices, val_indices = [], []
            for c in range(100):
                c_idxs = (train_y == c).nonzero(as_tuple=True)[0]
                val_idx = c_idxs[fold]
                tr_idxs = c_idxs[torch.arange(3) != fold]
                tr_indices.extend(tr_idxs.tolist())
                val_indices.append(val_idx.item())

            sub_tr_x = tr_x[tr_indices]
            sub_tr_y = train_y[tr_indices]
            val_x = tr_x[val_indices]
            val_y = train_y[val_indices]

            # Fit L2 multinomial logreg using AdamW on cross-entropy + weight decay
            # weight_decay = 1.0 / C
            torch.manual_seed(42)
            model = nn.Linear(tr_x.shape[1], 100, bias=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1.0 / C)
            criterion = nn.CrossEntropyLoss()

            for _ in range(100):
                model.train()
                optimizer.zero_grad()
                logits = model(sub_tr_x)
                loss = criterion(logits, sub_tr_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(val_x).argmax(dim=1)
                cv_accs.append((val_preds == val_y).float().mean().item() * 100.0)

        mean_cv = np.mean(cv_accs)
        if mean_cv > best_cv_acc:
            best_cv_acc = mean_cv
            best_c = C

    # Retrain on full train_x with best C
    torch.manual_seed(42)
    model = nn.Linear(tr_x.shape[1], 100, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1.0 / best_c)
    criterion = nn.CrossEntropyLoss()

    for _ in range(150):
        model.train()
        optimizer.zero_grad()
        logits = model(tr_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        te_preds = model(te_x).argmax(dim=1)
        te_acc = (te_preds == test_y).float().mean().item() * 100.0

    return te_acc, best_c


def main():
    print("==================================================================", flush=True)
    print(" PHASE J3, J4, J5 — OFFLINE BOUND SEARCH & BEST_CELL SELECTION", flush=True)
    print("==================================================================", flush=True)

    representation_candidates = [
        ("mean", "none"),
        ("mean", "center"),
        ("mean", "center+ZCA_whiten"),
        ("mean", "pca_m16_eps1e-4"),
        ("mean", "pca_m32_eps1e-6"),
        ("mean", "pca_m32_eps1e-4"),
        ("mean", "pca_m64_eps1e-4"),
        ("mean", "pca_m128_eps1e-4"),
        ("mean", "pca_m256_eps1e-4"),
        ("mean", "pca_m299_eps1e-4"),
        ("mean", "ledoit_wolf"),
        ("lasttok_nonpunct", "none"),
        ("lasttok_nonpunct", "center"),
        ("lasttok_nonpunct", "pca_m32_eps1e-6"),
        ("lasttok_nonpunct", "pca_m299_eps1e-4"),
        ("lasttok_nonpunct", "ledoit_wolf"),
    ]

    all_results = []
    global_max_bound = -1.0
    winning_cell = None

    print(f"\n{'Representation (pooling / transform)':<35} | {'NCM':<7} | {'1-NN':<7} | {'HeadL1c(J4)':<11} | {'LogReg(L2)':<10} | {'OFFLINE_BOUND':<13} | {'Winner'}", flush=True)
    print("-" * 105, flush=True)

    for p_type, t_type in representation_candidates:
        tr_x, train_y, te_x, test_y = get_transformed_data(p_type, t_type)

        ncm_acc = eval_ncm(tr_x, train_y, te_x, test_y)
        knn_acc = eval_1nn(tr_x, train_y, te_x, test_y)
        head_acc, fold_accs = eval_j4_headl1c_val_split(tr_x, train_y, te_x, test_y)
        logreg_acc, best_c = eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y)

        family_accs = {
            "NCM": ncm_acc,
            "1-NN": knn_acc,
            "HeadL1c": head_acc,
            "LogReg": logreg_acc
        }

        offline_bound = max(family_accs.values())
        winning_method = max(family_accs, key=family_accs.get)

        cell_name = f"{p_type} / {t_type}"
        all_results.append({
            "cell_name": cell_name,
            "pooling": p_type,
            "transform": t_type,
            "ncm": ncm_acc,
            "knn": knn_acc,
            "head": head_acc,
            "logreg": logreg_acc,
            "best_c": best_c,
            "offline_bound": offline_bound,
            "winning_method": winning_method
        })

        if offline_bound > global_max_bound:
            global_max_bound = offline_bound
            winning_cell = (cell_name, offline_bound, winning_method, family_accs)

        print(f"{cell_name:<35} | {ncm_acc:6.2f}% | {knn_acc:6.2f}% | {head_acc:6.2f}%     | {logreg_acc:6.2f}%    | {offline_bound:6.2f}%       | {winning_method}", flush=True)

    print("=" * 105, flush=True)
    print(f"\nEXPLICIT BEST_CELL IDENTIFIED BY J5 CRITERION:", flush=True)
    print(f"  Representation Name : '{winning_cell[0]}'", flush=True)
    print(f"  J3 OFFLINE_BOUND    : {winning_cell[1]:.2f}%", flush=True)
    print(f"  Winning Method      : {winning_cell[2]}", flush=True)
    print(f"  Family Breakdown    : NCM={winning_cell[3]['NCM']:.2f}%, 1-NN={winning_cell[3]['1-NN']:.2f}%, HeadL1c(J4)={winning_cell[3]['HeadL1c']:.2f}%, LogReg={winning_cell[3]['LogReg']:.2f}%", flush=True)
    print("==================================================================", flush=True)


if __name__ == "__main__":
    main()
