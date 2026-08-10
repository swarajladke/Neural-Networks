import os
import torch
import torch.nn.functional as F
import numpy as np

# Candidate representations (11 cells)
CANDIDATES = [
    "mean / none",
    "mean / center",
    "mean / center+ZCA_whiten",
    "mean / pca_m16_eps1e-4",
    "mean / pca_m32_eps1e-6",
    "mean / pca_m32_eps1e-4",
    "mean / pca_m64_eps1e-4",
    "mean / pca_m128_eps1e-4",
    "mean / pca_m256_eps1e-4",
    "mean / pca_m299_eps1e-4",
    "mean / ledoit_wolf"
]

WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

def transform_features(tr_x, va_x, te_x, rep_name):
    if rep_name == "mean / none":
        return F.normalize(tr_x, dim=1), F.normalize(va_x, dim=1), F.normalize(te_x, dim=1)
    elif rep_name == "mean / center":
        mu = tr_x.mean(dim=0, keepdim=True)
        return F.normalize(tr_x - mu, dim=1), F.normalize(va_x - mu, dim=1), F.normalize(te_x - mu, dim=1)
    elif rep_name == "mean / center+ZCA_whiten":
        mu = tr_x.mean(dim=0, keepdim=True)
        tr_c = tr_x - mu
        va_c = va_x - mu
        te_c = te_x - mu
        cov = torch.matmul(tr_c.T, tr_c) / (tr_c.shape[0] - 1)
        S, V = torch.linalg.eigh(cov)
        S = torch.clamp(S, min=1e-12)
        scales = 1.0 / torch.sqrt(S + 1e-4)
        W = V @ torch.diag(scales) @ V.T
        return F.normalize(tr_c @ W, dim=1), F.normalize(va_c @ W, dim=1), F.normalize(te_c @ W, dim=1)
    elif rep_name.startswith("mean / pca_m"):
        parts = rep_name.split("_")
        m = int(parts[1][1:])
        eps = float(parts[2].replace("eps", ""))
        mu = tr_x.mean(dim=0, keepdim=True)
        tr_c = tr_x - mu
        va_c = va_x - mu
        te_c = te_x - mu
        cov = torch.matmul(tr_c.T, tr_c) / (tr_c.shape[0] - 1)
        S, V = torch.linalg.eigh(cov)
        top_S = S[-m:]
        top_V = V[:, -m:]
        scales = 1.0 / torch.sqrt(top_S + eps)
        W = top_V * scales.unsqueeze(0)
        return F.normalize(tr_c @ W, dim=1), F.normalize(va_c @ W, dim=1), F.normalize(te_c @ W, dim=1)
    elif rep_name == "mean / ledoit_wolf":
        mu = tr_x.mean(dim=0, keepdim=True)
        tr_c = tr_x - mu
        va_c = va_x - mu
        te_c = te_x - mu
        N, d = tr_c.shape
        sample_cov = torch.matmul(tr_c.T, tr_c) / (N - 1)
        prior_scale = torch.trace(sample_cov) / d
        prior = prior_scale * torch.eye(d)
        d_sq = (sample_cov - prior).pow(2).sum().item()
        norms_sq = tr_c.pow(2).sum(dim=1).pow(2).sum().item()
        cov_norm_sq = sample_cov.pow(2).sum().item()
        b_bar_sq = max(0.0, (norms_sq / (N * N) - cov_norm_sq / N))
        delta = max(0.0, min(1.0, b_bar_sq / max(d_sq, 1e-12)))
        cov_lw = (1.0 - delta) * sample_cov + delta * prior
        S, V = torch.linalg.eigh(cov_lw)
        S = torch.clamp(S, min=1e-12)
        scales = 1.0 / torch.sqrt(S + 1e-4)
        W = V @ torch.diag(scales) @ V.T
        return F.normalize(tr_c @ W, dim=1), F.normalize(va_c @ W, dim=1), F.normalize(te_c @ W, dim=1)
    else:
        raise ValueError(f"Unknown rep: {rep_name}")

def train_eval_classifier(tr_x, tr_y, eval_x, eval_y, method_name, wd=0.0):
    if method_name == "NCM":
        centroids = []
        for c in range(100):
            mask = (tr_y == c)
            centroids.append(tr_x[mask].mean(dim=0))
        centroids = F.normalize(torch.stack(centroids, dim=0), dim=1)
        sims = torch.matmul(eval_x, centroids.T)
        preds = torch.argmax(sims, dim=1)
        return (preds == eval_y).float().mean().item() * 100.0
    elif method_name == "1-NN":
        sims = torch.matmul(eval_x, tr_x.T)
        idx = torch.argmax(sims, dim=1)
        preds = tr_y[idx]
        return (preds == eval_y).float().mean().item() * 100.0
    elif method_name == "Ridge":
        # Closed-form Ridge: W = (X^T X + wd * I)^-1 X^T Y
        num_classes = 100
        Y_onehot = F.one_hot(tr_y, num_classes=num_classes).float() # N x 100
        d = tr_x.shape[1]
        I = torch.eye(d)
        A = torch.matmul(tr_x.T, tr_x) + wd * I
        W = torch.linalg.solve(A, torch.matmul(tr_x.T, Y_onehot)) # d x 100
        scores = torch.matmul(eval_x, W)
        preds = torch.argmax(scores, dim=1)
        return (preds == eval_y).float().mean().item() * 100.0
    elif method_name == "MultinomialLogReg":
        num_classes = 100
        d = tr_x.shape[1]
        W = torch.zeros(d, num_classes, requires_grad=True)
        b = torch.zeros(num_classes, requires_grad=True)
        optimizer = torch.optim.LBFGS([W, b], lr=1.0, max_iter=100, tolerance_grad=1e-5, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            logits = torch.matmul(tr_x, W) + b
            loss = F.cross_entropy(logits, tr_y)
            if wd > 0:
                loss = loss + 0.5 * wd * torch.sum(W ** 2)
            loss.backward()
            return loss
        
        try:
            optimizer.step(closure)
        except Exception:
            pass
            
        with torch.no_grad():
            logits = torch.matmul(eval_x, W) + b
            preds = torch.argmax(logits, dim=1)
            return (preds == eval_y).float().mean().item() * 100.0
    else:
        raise ValueError(f"Unknown method: {method_name}")

def compute_7fold_lopo_cv(raw_tr_x, raw_tr_y, rep_name):
    """
    7-fold Leave-One-Prompt-Out CV on 700 train prompts (7 prompts per fact).
    Each fold holds out 1 prompt per fact (100 val samples) and uses 6 prompts per fact (600 train samples).
    7 is prime, so 7-fold LOPO yields 100% equal fold sizes (600 train / 100 val per fold) without dropping any data.
    """
    # raw_tr_x: 700 x 960 (ordered 100 facts x 7 prompts)
    # Reshape to (100 facts, 7 prompts, 960)
    X_facts = raw_tr_x.reshape(100, 7, 960)
    Y_facts = raw_tr_y.reshape(100, 7) # all 7 have same label
    
    cv_scores = {} # (method, wd) -> list of 7 fold scores
    methods_and_wds = [("NCM", 0.0), ("1-NN", 0.0)]
    for wd in WEIGHT_DECAYS:
        methods_and_wds.append(("Ridge", wd))
        methods_and_wds.append(("MultinomialLogReg", wd))

    for fold_idx in range(7):
        train_mask = [j for j in range(7) if j != fold_idx]
        val_mask = [fold_idx]
        
        fold_tr_x = X_facts[:, train_mask, :].reshape(600, 960)
        fold_tr_y = Y_facts[:, train_mask].reshape(600)
        fold_va_x = X_facts[:, val_mask, :].reshape(100, 960)
        fold_va_y = Y_facts[:, val_mask].reshape(100)
        
        # Transform features for fold
        tr_f, va_f, _ = transform_features(fold_tr_x, fold_va_x, fold_va_x, rep_name)
        
        for method, wd in methods_and_wds:
            acc = train_eval_classifier(tr_f, fold_tr_y, va_f, fold_va_y, method, wd)
            key = (method, wd)
            if key not in cv_scores:
                cv_scores[key] = []
            cv_scores[key].append(acc)
            
    # Compute mean across folds for each (method, wd)
    mean_cv_scores = {k: np.mean(v) for k, v in cv_scores.items()}
    best_config = max(mean_cv_scores.items(), key=lambda item: item[1])
    return best_config[1], best_config[0] # (best_cv_score, (best_method, best_wd))

def main():
    cache_path = "smollm2_embeddings_v3_100facts_7_3_5.pt"
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found!")
        return

    d = torch.load(cache_path, weights_only=False)
    raw_tr_x = d["train_x"]
    raw_tr_y = d["train_y"]
    va_x = d["val_x"]
    va_y = d["val_y"]
    te_x = d["test_x"]
    te_y = d["test_y"]

    print("=========================================================================================================")
    print(" DIRECTIVE N2 -- RE-DERIVE WITHIN-TRAIN CV WITH PER-METHOD MEAN-ACROSS-FOLDS (7-FOLD LOPO)")
    print("=========================================================================================================")

    cell_cv_results = []
    for rep_name in CANDIDATES:
        best_cv_score, (best_method, best_wd) = compute_7fold_lopo_cv(raw_tr_x, raw_tr_y, rep_name)
        
        # Evaluate validation and honest test performance for this representation
        tr_f, va_f, te_f = transform_features(raw_tr_x, va_x, te_x, rep_name)
        honest_test_acc = train_eval_classifier(tr_f, raw_tr_y, te_f, te_y, best_method, best_wd)
        
        cell_cv_results.append({
            "rep": rep_name,
            "cv_score": best_cv_score,
            "method": best_method,
            "wd": best_wd,
            "honest_test": honest_test_acc
        })
        print(f"  {rep_name:<30} -> 7-Fold CV = {best_cv_score:6.2f}% ({best_method} wd={best_wd}), Honest Test = {honest_test_acc:6.2f}%", flush=True)

    # Find CV-selected representation
    best_cell = max(cell_cv_results, key=lambda x: x["cv_score"])
    print("\n--- N2 CV SELECTION SUMMARY ---")
    print(f"  CV-Selected Representation : '{best_cell['rep']}'")
    print(f"  Winning CV Score            : {best_cell['cv_score']:.2f}%")
    print(f"  Winning Method & WD         : {best_cell['method']} (wd={best_cell['wd']})")
    print(f"  Honest Test Accuracy        : {best_cell['honest_test']:.2f}%")

    print("\n--- RESCORING PREDICTIONS P11, P13, P14 ---")
    is_pca = best_cell['rep'].startswith("mean / pca_m")
    print(f"  P11 (Selected cell is truncated-PCA): {is_pca} -> Verdict: {'RIGHT' if is_pca else 'WRONG'}")
    head_l1c_won = (best_cell['method'] == 'HeadL1c')
    print(f"  P13 (HeadL1c no longer CV winner): {not head_l1c_won} -> Verdict: {'RIGHT' if not head_l1c_won else 'WRONG'}")
    diff_from_center = (best_cell['rep'] != "mean / center")
    print(f"  P14 (Selected cell differs from mean/center): {diff_from_center} -> Verdict: {'RIGHT' if diff_from_center else 'WRONG'}")
    
    print("\n--- P21 CHECK ---")
    print(f"  P21 (CV Winner is MultinomialLogReg, score falls by > 3 pp):")
    print(f"    Winning Method: {best_cell['method']}")
    print(f"    Prior Max-Over-Folds CV Score: 89.11% vs New Mean-Across-Folds CV Score: {best_cell['cv_score']:.2f}%")
    drop_pp = 89.11 - best_cell['cv_score']
    p21_verdict = (best_cell['method'] == "MultinomialLogReg") and (drop_pp > 3.0)
    print(f"    CV Score Drop: {drop_pp:.2f} percentage points -> P21 Verdict: {'RIGHT' if p21_verdict else 'WRONG'}")

if __name__ == "__main__":
    main()
