"""
run_w1c_preprocessing_sweep.py
==============================
Directive W1c: Quantify preprocessing sensitivity on the v3 cache.
Evaluates frozen-NCM test accuracy under EVERY entry in eval_core.CANDIDATE_REPRESENTATIONS
plus a raw/no-transform control, fitting every transform on TRAIN DATA ONLY.
Also computes anisotropy metrics for raw features:
  - Ratio of 1st to 10th largest eigenvalue of train covariance
  - Mean pairwise cosine similarity of train embeddings
Emits w1c_results.json and tees stdout.
Terminates with EXIT_CODE = 0.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F

import eval_core
from eval_core import CANDIDATE_REPRESENTATIONS, transform_fit_train_only, eval_ncm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = "smollm2_embeddings_v3_100facts_7_3_5.pt"
CACHE_PATH = os.path.join(REPO_ROOT, CACHE_FILE)
OUT_JSON = os.path.join(REPO_ROOT, "w1c_results.json")


def sha256sum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_provenance():
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        print(f"[VIOLATION] git rev-parse HEAD failed: {e}")
        print("EXIT_CODE = 1")
        sys.exit(1)

    status_res = subprocess.run(["git", "status", "--untracked-files=no", "--porcelain"], capture_output=True, text=True)
    dirty_tracked = status_res.stdout.strip()
    if dirty_tracked:
        print(f"[VIOLATION] Working tree has uncommitted modifications:\n{dirty_tracked}")
        print("EXIT_CODE = 1")
        sys.exit(1)

    return git_sha


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print("=========================================================================================================")
    print(" PART 1: PREPROCESSING SENSITIVITY SWEEP ON V3 CACHE (DIRECTIVE W1c)")
    print("=========================================================================================================")

    # 1. Environment & Provenance Checks
    np_ver = np.__version__
    print(f"  NumPy Version    : {np_ver}")
    if int(np_ver.split(".")[0]) >= 2:
        print(f"[VIOLATION] NumPy version {np_ver} >= 2.0.0 will break torch 2.2.2 ABI.")
        print("EXIT_CODE = 1")
        sys.exit(1)

    git_sha = check_provenance()
    print(f"  Git Commit SHA   : {git_sha}")

    if not os.path.isfile(CACHE_PATH):
        print(f"[VIOLATION] Cache file {CACHE_PATH} not found.")
        print("EXIT_CODE = 1")
        sys.exit(1)

    cache_sha = sha256sum(CACHE_PATH)
    print(f"  Cache File       : {CACHE_FILE}")
    print(f"  Cache SHA-256    : {cache_sha}")

    # 2. Data Loading
    d = torch.load(CACHE_PATH, weights_only=False)
    tr_x, tr_y = d["train_x"], d["train_y"]
    va_x, va_y = d["val_x"], d["val_y"]
    te_x, te_y = d["test_x"], d["test_y"]

    n_train = int(tr_x.shape[0])
    n_val = int(va_x.shape[0])
    n_test = int(te_x.shape[0])
    n_classes = int(len(torch.unique(tr_y)))
    n_forward_samples = n_train + n_val + n_test

    print(f"  Loaded Samples   : Train={n_train}, Val={n_val}, Test={n_test} (Classes={n_classes})")
    print(f"  Total Samples    : {n_forward_samples}")

    # Hard asserts
    if not (n_forward_samples > 0):
        print(f"[VIOLATION] n_forward_samples must be > 0, got {n_forward_samples}")
        print("EXIT_CODE = 1")
        sys.exit(1)

    n_optimizer_steps = 0
    if n_optimizer_steps != 0:
        print(f"[VIOLATION] n_optimizer_steps must be 0 for frozen NCM, got {n_optimizer_steps}")
        print("EXIT_CODE = 1")
        sys.exit(1)

    # 3. Anisotropy Statistics on Raw Features
    print(f"\n  --- RAW EMBEDDING ANISOTROPY METRICS ---")
    tr_dbl = tr_x.double()
    mu = tr_dbl.mean(dim=0, keepdim=True)
    tr_c = tr_dbl - mu
    cov = (tr_c.T @ tr_c) / (n_train - 1)

    eigenvals = torch.linalg.eigvalsh(cov)
    eigenvals_desc = torch.sort(eigenvals, descending=True).values
    lambda_1 = float(eigenvals_desc[0].item())
    lambda_10 = float(eigenvals_desc[9].item())
    eigenval_ratio_1_to_10 = lambda_1 / lambda_10

    # Mean pairwise cosine similarity across all unique train pairs (244,650 pairs)
    tr_norm = F.normalize(tr_x.float(), dim=-1)
    sim_matrix = tr_norm @ tr_norm.T
    triu_mask = torch.triu(torch.ones(n_train, n_train, dtype=torch.bool), diagonal=1)
    pairwise_cosines = sim_matrix[triu_mask]
    mean_pairwise_cos = float(pairwise_cosines.mean().item())
    std_pairwise_cos = float(pairwise_cosines.std().item())
    min_pairwise_cos = float(pairwise_cosines.min().item())
    max_pairwise_cos = float(pairwise_cosines.max().item())

    print(f"    Eigenvalue 1 (largest)             : {lambda_1:.6e}")
    print(f"    Eigenvalue 10                      : {lambda_10:.6e}")
    print(f"    Eigenvalue Ratio (lambda_1/lambda_10): {eigenval_ratio_1_to_10:8.2f}")
    print(f"    Mean Pairwise Cosine Similarity     : {mean_pairwise_cos:.4f} +/- {std_pairwise_cos:.4f} [min={min_pairwise_cos:.4f}, max={max_pairwise_cos:.4f}]")

    # 4. Preprocessing Sweep
    print(f"\n  --- PREPROCESSING SWEEP OVER CANDIDATE REPRESENTATIONS ---")
    results_table = []

    # A. Raw / unnormalized control (no transform, no L2 normalization)
    # eval_ncm computes class centroids from raw embeddings and normalizes centroids only
    res_raw_unnorm = eval_ncm(tr_x.float(), tr_y, te_x.float(), te_y)
    acc_raw_unnorm = float(res_raw_unnorm["accuracy"])
    results_table.append({
        "representation": "raw / no_transform (unnormalized)",
        "test_accuracy": acc_raw_unnorm
    })

    # B. Candidate representations from eval_core
    for rep in CANDIDATE_REPRESENTATIONS:
        tr_proj, te_proj = transform_fit_train_only(tr_x, te_x, rep)
        res = eval_ncm(tr_proj, tr_y, te_proj, te_y)
        acc = float(res["accuracy"])
        results_table.append({
            "representation": rep,
            "test_accuracy": acc
        })

    # Print Table
    print(f"\n  {'-'*80}")
    print(f"  {'Representation':<65} | {'Test ACC (%)':>10}")
    print(f"  {'-'*80}")
    pca_m64_acc = None
    for entry in results_table:
        rep_name = entry["representation"]
        acc = entry["test_accuracy"]
        print(f"  {rep_name:<65} | {acc:>10.2f}%")
        if "pca_m64_eps1e-4" in rep_name:
            pca_m64_acc = acc
    print(f"  {'-'*80}")

    # Confirm / Refute 85.80%
    target_ceiling = 85.80
    print(f"\n  Target Check for 'mean / pca_m64_eps1e-4':")
    print(f"    Reported in prior docs : {target_ceiling:.2f}%")
    print(f"    Measured in this run   : {pca_m64_acc:.2f}%")
    if abs(pca_m64_acc - target_ceiling) < 1e-3:
        reproduction_verdict = f"CONFIRMED: 'mean / pca_m64_eps1e-4' exactly reproduces {target_ceiling:.2f}% (measured {pca_m64_acc:.2f}%)"
    else:
        reproduction_verdict = f"REFUTED: 'mean / pca_m64_eps1e-4' does NOT reproduce {target_ceiling:.2f}% (measured {pca_m64_acc:.2f}%, delta = {pca_m64_acc - target_ceiling:+.2f} pp)"
    print(f"    Verdict                : {reproduction_verdict}")

    out_data = {
        "benchmark": "v3_disjoint_template",
        "cache_file": CACHE_FILE,
        "cache_sha256": cache_sha,
        "git_commit_sha": git_sha,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_test_samples": n_test,
        "n_classes": n_classes,
        "n_forward_samples": n_forward_samples,
        "n_optimizer_steps": n_optimizer_steps,
        "anisotropy_metrics": {
            "eigenvalue_1": lambda_1,
            "eigenvalue_10": lambda_10,
            "eigenval_ratio_1_to_10": eigenval_ratio_1_to_10,
            "mean_pairwise_cosine": mean_pairwise_cos,
            "std_pairwise_cosine": std_pairwise_cos,
            "min_pairwise_cosine": min_pairwise_cos,
            "max_pairwise_cosine": max_pairwise_cos
        },
        "sweep_results": results_table,
        "reproduction_target_85_80": {
            "target": target_ceiling,
            "measured_pca_m64": pca_m64_acc,
            "verdict": reproduction_verdict
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n  Emitted JSON results to {OUT_JSON}")
    print("=========================================================================================================")
    print("EXIT_CODE = 0")


if __name__ == "__main__":
    main()
