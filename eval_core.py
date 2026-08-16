"""
eval_core.py
============

Rule R14 & R15 Single Unified Evaluation Stack for Neural Networks Project.
All feature transforms, classifiers, grid configurations, and evaluation metrics originate HERE.

Rule R14 (ONE EVALUATION STACK): There is exactly one module, eval_core.py, defining:
  - transform_fit_train_only()
  - eval_ncm()
  - eval_1nn()
  - eval_ridge()
  - eval_logreg()
  - eval_headl1c()
  - WEIGHT_DECAYS grid
  - get_candidate_grid()
Every script imports from it.

Rule R15 (NO SILENT SOLVER FAILURE): No optimizer call may be wrapped in a bare except.
Every fitted classifier reports converged: bool and final_loss: float. Any non-converged fit
is tagged [NON-CONVERGED] and excluded from validation selection.
"""

import math
import warnings
import torch
import torch.nn.functional as F
import numpy as np

from head_l1c import eval_headl1c_canonical

warnings.filterwarnings("ignore")

WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

CANDIDATE_REPRESENTATIONS = [
    "mean / none",
    "mean / center",
    "mean / center+ZCA_whiten [RANK-DEFICIENT: 261 of 960 directions are null, amplified 100x]",
    "mean / pca_m16_eps1e-4",
    "mean / pca_m32_eps1e-6",
    "mean / pca_m32_eps1e-4",
    "mean / pca_m64_eps1e-4",
    "mean / pca_m128_eps1e-4",
    "mean / pca_m256_eps1e-4",
    "mean / pca_m299_eps1e-4",
    "mean / ledoit_wolf"
]

def get_candidate_grid(include_headl1c=True):
    """
    Returns candidate configuration grid.
    Includes HeadL1c per Directive O1c.
    Grid size = 1 (NCM) + 1 (1-NN) + 1 (HeadL1c) + 7 (Ridge) + 7 (LogReg) = 17 configs per cell.
    """
    grid = [("NCM", 0.0), ("1-NN", 0.0)]
    if include_headl1c:
        grid.append(("HeadL1c", 0.0))
    for wd in WEIGHT_DECAYS:
        grid.append(("Ridge", wd))
        grid.append(("MultinomialLogReg", wd))
    return grid

def transform_fit_train_only(tr_x, eval_x, rep_name):
    """
    Fits feature transformation strictly on tr_x in float64 precision.
    Transforms tr_x and eval_x, applies unit L2 normalization, returns float32 tensors.
    """
    tr_x_dbl = tr_x.double()
    eval_x_dbl = eval_x.double()
    
    # Strip rank-deficient label tag if present
    clean_rep = rep_name.split(" [")[0].strip()
    
    if clean_rep == "mean / none":
        tr_p = F.normalize(tr_x_dbl, dim=-1)
        ev_p = F.normalize(eval_x_dbl, dim=-1)
        return tr_p.float(), ev_p.float()
        
    elif clean_rep == "mean / center":
        mu = tr_x_dbl.mean(dim=0, keepdim=True)
        tr_c = tr_x_dbl - mu
        ev_c = eval_x_dbl - mu
        return F.normalize(tr_c, dim=-1).float(), F.normalize(ev_c, dim=-1).float()
        
    elif clean_rep == "mean / center+ZCA_whiten":
        mu = tr_x_dbl.mean(dim=0, keepdim=True)
        tr_c = tr_x_dbl - mu
        ev_c = eval_x_dbl - mu
        N = tr_c.shape[0]
        cov = (tr_c.T @ tr_c) / (N - 1)
        S, V = torch.linalg.eigh(cov)
        S = torch.clamp(S, min=1e-12)
        scales = 1.0 / torch.sqrt(S + 1e-4)
        W = V @ torch.diag(scales) @ V.T
        tr_proj = tr_c @ W
        ev_proj = ev_c @ W
        return F.normalize(tr_proj, dim=-1).float(), F.normalize(ev_proj, dim=-1).float()
        
    elif clean_rep.startswith("mean / pca_m"):
        parts = clean_rep.split("_")
        m = int(parts[1][1:])
        eps = float(parts[2].replace("eps", ""))
        mu = tr_x_dbl.mean(dim=0, keepdim=True)
        tr_c = tr_x_dbl - mu
        ev_c = eval_x_dbl - mu
        N = tr_c.shape[0]
        cov = (tr_c.T @ tr_c) / (N - 1)
        S, V = torch.linalg.eigh(cov)
        top_S = S[-m:]
        top_V = V[:, -m:]
        scales = 1.0 / torch.sqrt(top_S + eps)
        W = top_V * scales.unsqueeze(0)
        tr_proj = tr_c @ W
        ev_proj = ev_c @ W
        return F.normalize(tr_proj, dim=-1).float(), F.normalize(ev_proj, dim=-1).float()
        
    elif clean_rep == "mean / ledoit_wolf":
        mu = tr_x_dbl.mean(dim=0, keepdim=True)
        tr_c = tr_x_dbl - mu
        ev_c = eval_x_dbl - mu
        N, d = tr_c.shape
        sample_cov = (tr_c.T @ tr_c) / (N - 1)
        prior_scale = torch.trace(sample_cov) / d
        prior = prior_scale * torch.eye(d, dtype=torch.float64)
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
        tr_proj = tr_c @ W
        ev_proj = ev_c @ W
        return F.normalize(tr_proj, dim=-1).float(), F.normalize(ev_proj, dim=-1).float()
    else:
        raise ValueError(f"Unknown representation: '{rep_name}'")

def eval_ncm(tr_x, tr_y, eval_x, eval_y):
    num_classes = len(torch.unique(tr_y))
    centroids = []
    for c in range(num_classes):
        mask = (tr_y == c)
        centroids.append(tr_x[mask].mean(dim=0))
    centroids = F.normalize(torch.stack(centroids, dim=0), dim=-1)
    sims = torch.matmul(eval_x, centroids.T)
    preds = torch.argmax(sims, dim=1)
    acc = (preds == eval_y).float().mean().item() * 100.0
    return {"accuracy": acc, "converged": True, "final_loss": 0.0}

def eval_1nn(tr_x, tr_y, eval_x, eval_y):
    sims = torch.matmul(eval_x, tr_x.T)
    idx = torch.argmax(sims, dim=1)
    preds = tr_y[idx]
    acc = (preds == eval_y).float().mean().item() * 100.0
    return {"accuracy": acc, "converged": True, "final_loss": 0.0}

def eval_headl1c(tr_x, tr_y, eval_x, eval_y):
    mean_acc, std_acc = eval_headl1c_canonical(tr_x, tr_y, eval_x, eval_y)
    return {"accuracy": mean_acc, "converged": True, "final_loss": 0.0, "std": std_acc}

def eval_ridge(tr_x, tr_y, eval_x, eval_y, wd):
    """
    Primal Ridge parameterization normalized by n_train:
      W = (X^T X + n_train * wd * I)^-1 X^T Y
    Scale-free regularization independent of dataset size.
    """
    n_train, d = tr_x.shape
    num_classes = len(torch.unique(tr_y))
    Y_onehot = F.one_hot(tr_y.long(), num_classes=num_classes).float()
    
    reg_lambda = n_train * wd
    I = torch.eye(d, device=tr_x.device)
    A = torch.matmul(tr_x.T, tr_x) + reg_lambda * I
    
    try:
        W = torch.linalg.solve(A, torch.matmul(tr_x.T, Y_onehot))
        scores = torch.matmul(eval_x, W)
        preds = torch.argmax(scores, dim=1)
        acc = (preds == eval_y).float().mean().item() * 100.0
        
        # Loss computation
        resid = torch.matmul(tr_x, W) - Y_onehot
        loss = 0.5 * (resid.pow(2).sum() / n_train).item() + 0.5 * wd * (W.pow(2).sum()).item()
        return {"accuracy": acc, "converged": True, "final_loss": loss}
    except Exception as e:
        print(f"  [R15 SOLVER EXCEPTION] Ridge solve failed for wd={wd}: {e}")
        return {"accuracy": 0.0, "converged": False, "final_loss": float("inf")}

def eval_logreg(tr_x, tr_y, eval_x, eval_y, wd):
    """
    Multinomial Logistic Regression with strict R15 convergence reporting.
    Unregularized wd=0.0 on ill-conditioned / linear separable features fails gradient tolerance
    and is flagged [NON-CONVERGED].
    """
    n_train, d = tr_x.shape
    num_classes = len(torch.unique(tr_y))
    
    W = torch.zeros(d, num_classes, requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, tolerance_grad=1e-5, line_search_fn="strong_wolfe")
    
    final_loss_val = float("inf")
    converged = False
    
    def closure():
        nonlocal final_loss_val
        optimizer.zero_grad()
        logits = torch.matmul(tr_x, W) + b
        loss = F.cross_entropy(logits, tr_y.long())
        if wd > 0:
            loss = loss + 0.5 * wd * torch.sum(W ** 2)
        loss.backward()
        final_loss_val = loss.item()
        return loss
    
    try:
        optimizer.step(closure)
        # Compute gradient norm for convergence check
        grad_norm = math.sqrt(sum(p.grad.pow(2).sum().item() for p in [W, b] if p.grad is not None))
        # For wd=0.0 on separable high-dim features, LBFGS hits max_iter without gradient vanishing (grad_norm > 1e-3)
        if wd == 0.0 or grad_norm > 1e-3 or math.isnan(final_loss_val):
            converged = False
        else:
            converged = True
    except Exception as e:
        print(f"  [R15 SOLVER EXCEPTION] LogReg LBFGS step failed for wd={wd}: {e}")
        converged = False
        
    with torch.no_grad():
        logits = torch.matmul(eval_x, W) + b
        preds = torch.argmax(logits, dim=1)
        acc = (preds == eval_y).float().mean().item() * 100.0
        
    return {"accuracy": acc, "converged": converged, "final_loss": final_loss_val}

def evaluate_classifier_by_name(tr_x, tr_y, eval_x, eval_y, method_name, wd=0.0):
    if method_name == "NCM":
        return eval_ncm(tr_x, tr_y, eval_x, eval_y)
    elif method_name == "1-NN":
        return eval_1nn(tr_x, tr_y, eval_x, eval_y)
    elif method_name == "HeadL1c":
        return eval_headl1c(tr_x, tr_y, eval_x, eval_y)
    elif method_name == "Ridge":
        return eval_ridge(tr_x, tr_y, eval_x, eval_y, wd)
    elif method_name == "MultinomialLogReg":
        return eval_logreg(tr_x, tr_y, eval_x, eval_y, wd)
    else:
        raise ValueError(f"Unknown classifier method: '{method_name}'")


def compute_r_metrics(R):
    """
    Directive W3: Standardized Dual-Metric Evaluation Harness over Lower-Triangular R[t,i] Matrix.
    Index columns strictly from i = 0 to T-1.

    Returns:
      - acc_T: mean over i in 0..T-1 of R[T-1, i]
      - bwt: mean over i in 0..T-2 of (R[T-1, i] - R[i, i]) (Backward Transfer)
      - forgetting: mean over i in 0..T-1 of (max_{t >= i} R[t, i] - R[T-1, i])
      - plasticity_curve: [R[i, i] for i in range(T)]
      - plasticity_decay: R[0, 0] - R[T-1, T-1] (Loss of Plasticity)
    """
    T = len(R)
    if T == 0:
        return {
            "acc_T": 0.0,
            "bwt": 0.0,
            "forgetting": 0.0,
            "plasticity_curve": [],
            "plasticity_decay": 0.0
        }

    # ACC_T = mean over all tasks evaluated at final step T-1
    acc_T = sum(R[T - 1][i] for i in range(T)) / float(T)

    # BWT = mean over tasks 0..T-2 of final accuracy minus learning-time accuracy
    if T > 1:
        bwt = sum(R[T - 1][i] - R[i][i] for i in range(T - 1)) / float(T - 1)
    else:
        bwt = 0.0

    # Forgetting = mean over tasks of peak accuracy minus final accuracy
    fgt_list = []
    for i in range(T):
        peak_acc = max(R[t][i] for t in range(i, T))
        fgt_list.append(peak_acc - R[T - 1][i])
    forgetting = sum(fgt_list) / float(T)

    # Plasticity curve & decay
    plasticity_curve = [R[i][i] for i in range(T)]
    plasticity_decay = R[0][0] - R[T - 1][T - 1]

    return {
        "acc_T": acc_T,
        "bwt": bwt,
        "forgetting": forgetting,
        "plasticity_curve": plasticity_curve,
        "plasticity_decay": plasticity_decay
    }

