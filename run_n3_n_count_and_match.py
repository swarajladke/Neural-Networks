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
        num_classes = 100
        Y_onehot = F.one_hot(tr_y, num_classes=num_classes).float()
        d = tr_x.shape[1]
        I = torch.eye(d)
        A = torch.matmul(tr_x.T, tr_x) + wd * I
        W = torch.linalg.solve(A, torch.matmul(tr_x.T, Y_onehot))
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

def main():
    cache_path = "smollm2_embeddings_v3_100facts_7_3_5.pt"
    d = torch.load(cache_path, weights_only=False)
    tr_x, tr_y = d["train_x"], d["train_y"]
    va_x, va_y = d["val_x"], d["val_y"]
    te_x, te_y = d["test_x"], d["test_y"]

    # Compute candidate configurations per cell:
    # 1 (NCM) + 1 (1-NN) + 7 (Ridge) + 7 (LogReg) = 16 configs per cell
    per_cell_configs = 1 + 1 + len(WEIGHT_DECAYS) + len(WEIGHT_DECAYS) # 16
    N_m1 = len(CANDIDATES) * per_cell_configs # 11 x 16 = 176
    N_m6 = 12 * per_cell_configs              # 12 x 16 = 192
    N_total = N_m1 + N_m6                     # 368

    print("=========================================================================================================")
    print(f" DIRECTIVE N3 -- COMPUTED TEST EVALUATION COUNTS AND CONFIG MATCH REPORT (N_m1={N_m1}, N_total={N_total})")
    print("=========================================================================================================")
    print(f"  True Per-Cell Candidate Configs Count : {per_cell_configs} test evaluations")
    print(f"  M1 Disjoint-Split Test Evaluations    : N_m1 = {N_m1} test evaluations (11 cells x {per_cell_configs})")
    print(f"  M6 Dimension Sweep Test Evaluations   : N_m6 = {N_m6} test evaluations (12 cells x {per_cell_configs})")
    print(f"  Project Total Test Evaluations        : N_total = {N_total} test evaluations")

    print("\n" + f"{'Representation':<28} | {'Val Selected Config':<28} | {'Test Selected Config':<28} | {'Config Differ':<13} | {'WD Differ':<10} | {'Method Differ':<13}")
    print("-" * 135)

    wd_differ_count = 0

    for rep_name in CANDIDATES:
        tr_f, va_f, te_f = transform_features(tr_x, va_x, te_x, rep_name)
        
        # Build grid of candidate configs
        grid_configs = [("NCM", 0.0), ("1-NN", 0.0)]
        for wd in WEIGHT_DECAYS:
            grid_configs.append(("Ridge", wd))
            grid_configs.append(("MultinomialLogReg", wd))
            
        # Validation selection
        val_res = []
        for m_name, wd in grid_configs:
            acc = train_eval_classifier(tr_f, tr_y, va_f, va_y, m_name, wd)
            val_res.append((acc, (m_name, wd)))
        val_acc, (val_method, val_wd) = max(val_res, key=lambda x: x[0])
        
        # Test selection (max ceiling)
        test_res = []
        for m_name, wd in grid_configs:
            acc = train_eval_classifier(tr_f, tr_y, te_f, te_y, m_name, wd)
            test_res.append((acc, (m_name, wd)))
        test_acc, (test_method, test_wd) = max(test_res, key=lambda x: x[0])
        
        config_differ = (val_method, val_wd) != (test_method, test_wd)
        method_differ = (val_method != test_method)
        
        if val_method in ["NCM", "1-NN"]:
            wd_differ_str = "N/A"
        else:
            wd_differ = (val_wd != test_wd)
            wd_differ_str = str(wd_differ)
            if wd_differ:
                wd_differ_count += 1
                
        val_cfg_str = f"{val_method} (wd={val_wd})" if val_method not in ["NCM", "1-NN"] else val_method
        test_cfg_str = f"{test_method} (wd={test_wd})" if test_method not in ["NCM", "1-NN"] else test_method

        print(f"{rep_name:<28} | {val_cfg_str:<28} | {test_cfg_str:<28} | {str(config_differ):<13} | {wd_differ_str:<10} | {str(method_differ):<13}")

    print("=" * 135)
    print(f"\n  P16 Scorecard Verification (Verbatim: Val WD differs from Test WD on >= 5 cells):")
    print(f"    Cells where WD Differed (excluding N/A): {wd_differ_count} of 11 cells")
    print(f"    Verdict: {'RIGHT' if wd_differ_count >= 5 else 'WRONG'}")

    print(f"\n  P22 Check (N_m1 strictly > 11):")
    print(f"    Computed N_m1 = {N_m1} > 11 -> P22 Verdict: {'RIGHT' if N_m1 > 11 else 'WRONG'}")

if __name__ == "__main__":
    main()
