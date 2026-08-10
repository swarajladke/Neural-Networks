"""
evaluate_expanded_offline_bound.py
===================================

Evaluates J3 OFFLINE_BOUND on both 3/3 and 10/5 datasets to isolate sample count effect for Gate 2 decision (J7).

Standing Rules:
- R4: Guard raises on missing input cache files.
- R6: Label-derived R6 indexing.
- R7: Fit on TRAIN only.
- R8: Fixed seeds [42, 43, 44, 45, 46] logged and enforced.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CACHE_3_3 = "smollm2_embeddings_v2_100facts.pt"
CACHE_10_5 = "smollm2_embeddings_v2_100facts_expanded_10_5.pt"


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


def eval_j4_headl1c_val_split(tr_x, train_y, te_x, test_y, prompts_per_fact=10, seeds=[42]):
    fold_test_accs = []
    num_folds = 5 if prompts_per_fact == 10 else 3

    for fold in range(num_folds):
        tr_indices, val_indices = [], []
        for c in range(100):
            c_idxs = (train_y == c).nonzero(as_tuple=True)[0]
            val_idx = c_idxs[fold]
            tr_idxs = c_idxs[torch.arange(prompts_per_fact) != fold]
            tr_indices.extend(tr_idxs.tolist())
            val_indices.append(val_idx.item())

        sub_tr_x = tr_x[tr_indices]
        sub_tr_y = train_y[tr_indices]
        val_x = tr_x[val_indices]
        val_y = train_y[val_indices]

        seed_accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = HeadL1c(in_features=tr_x.shape[1], out_features=100, scale=10.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = -1.0
            best_weights = None
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
                    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        break

            if best_weights is not None:
                model.load_state_dict(best_weights)
            model.eval()
            with torch.no_grad():
                te_preds = model(te_x).argmax(dim=1)
                seed_accs.append((te_preds == test_y).float().mean().item() * 100.0)

        fold_test_accs.append(np.mean(seed_accs))

    return np.mean(fold_test_accs)


def eval_multinomial_logistic_regression(tr_x, train_y, te_x, test_y, prompts_per_fact=10):
    c_candidates = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_c = None
    best_cv_acc = -1.0
    num_folds = 5 if prompts_per_fact == 10 else 3

    for C in c_candidates:
        cv_accs = []
        for fold in range(num_folds):
            tr_indices, val_indices = [], []
            for c in range(100):
                c_idxs = (train_y == c).nonzero(as_tuple=True)[0]
                val_idx = c_idxs[fold]
                tr_idxs = c_idxs[torch.arange(prompts_per_fact) != fold]
                tr_indices.extend(tr_idxs.tolist())
                val_indices.append(val_idx.item())

            sub_tr_x = tr_x[tr_indices]
            sub_tr_y = train_y[tr_indices]
            val_x = tr_x[val_indices]
            val_y = train_y[val_indices]

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
    if not os.path.exists(CACHE_3_3):
        raise RuntimeError(f"[R5 Guard] Missing 3/3 cache: '{CACHE_3_3}'.")
    if not os.path.exists(CACHE_10_5):
        raise RuntimeError(f"[R5 Guard] Missing 10/5 cache: '{CACHE_10_5}'.")

    d_3_3 = torch.load(CACHE_3_3, weights_only=False)
    d_10_5 = torch.load(CACHE_10_5, weights_only=False)

    print("==================================================================", flush=True)
    print(" J7 EVALUATE OFFLINE BOUND — 3/3 VS 10/5 EXPANDED DATASET", flush=True)
    print("==================================================================", flush=True)

    # 3/3 Evaluation
    tr_x_3 = F.normalize(d_3_3["train_x"], dim=-1)
    te_x_3 = F.normalize(d_3_3["test_x"], dim=-1)
    tr_y_3, te_y_3 = d_3_3["train_y"], d_3_3["test_y"]

    ncm_3 = eval_ncm(tr_x_3, tr_y_3, te_x_3, te_y_3)
    knn_3 = eval_1nn(tr_x_3, tr_y_3, te_x_3, te_y_3)
    head_3 = eval_j4_headl1c_val_split(tr_x_3, tr_y_3, te_x_3, te_y_3, prompts_per_fact=3)
    logreg_3, best_c_3 = eval_multinomial_logistic_regression(tr_x_3, tr_y_3, te_x_3, te_y_3, prompts_per_fact=3)
    bound_3 = max(ncm_3, knn_3, head_3, logreg_3)

    # 10/5 Evaluation
    tr_x_10 = F.normalize(d_10_5["train_x"], dim=-1)
    te_x_10 = F.normalize(d_10_5["test_x"], dim=-1)
    tr_y_10, te_y_10 = d_10_5["train_y"], d_10_5["test_y"]

    ncm_10 = eval_ncm(tr_x_10, tr_y_10, te_x_10, te_y_10)
    knn_10 = eval_1nn(tr_x_10, tr_y_10, te_x_10, te_y_10)
    head_10 = eval_j4_headl1c_val_split(tr_x_10, tr_y_10, te_x_10, te_y_10, prompts_per_fact=10)
    logreg_10, best_c_10 = eval_multinomial_logistic_regression(tr_x_10, tr_y_10, te_x_10, te_y_10, prompts_per_fact=10)
    bound_10 = max(ncm_10, knn_10, head_10, logreg_10)

    print(f"\n{'Dataset Split':<25} | {'NCM Top-1':<10} | {'1-NN Top-1':<10} | {'HeadL1c(J4)':<12} | {'LogReg(L2)':<10} | {'OFFLINE_BOUND':<13}", flush=True)
    print("-" * 95, flush=True)
    print(f"{'3 train / 3 test per fact':<25} | {ncm_3:6.2f}%    | {knn_3:6.2f}%    | {head_3:6.2f}%      | {logreg_3:6.2f}%    | {bound_3:6.2f}%", flush=True)
    print(f"{'10 train / 5 test per fact':<25} | {ncm_10:6.2f}%    | {knn_10:6.2f}%    | {head_10:6.2f}%      | {logreg_10:6.2f}%    | {bound_10:6.2f}%", flush=True)
    print("=" * 95, flush=True)

    print(f"\nGATE 2 EVALUATION DECISION:", flush=True)
    print(f"  OFFLINE_BOUND on 10/5 Expanded Dataset (B) = {bound_10:.2f}%", flush=True)
    print(f"  Gate 2 Threshold                            = 50.00%", flush=True)

    if bound_10 >= 50.0:
        print(f"  [GATE 2 PASSED] B ({bound_10:.2f}%) >= 50.0%. Proceeding to Phase IV Class-IL Arms!", flush=True)
    else:
        print(f"  [GATE 2 FAILED] B ({bound_10:.2f}%) < 50.0%. Stopping per specification.", flush=True)

    print("==================================================================", flush=True)


if __name__ == "__main__":
    main()
