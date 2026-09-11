"""
run_w1b_frozen_ncm_measured.py
==============================
Measures honest frozen-NCM baseline on smollm2_embeddings_v3_100facts_7_3_5.pt
as one single evaluation per seed over SEEDS = [42, 43, 44, 45, 46].
Emits w1b_results.json and tees stdout.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import torch
import torch.nn.functional as F

from eval_core import eval_ncm

SEEDS = [42, 43, 44, 45, 46]
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = "smollm2_embeddings_v3_100facts_7_3_5.pt"


def sha256sum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main():
    print("=========================================================================================================")
    print(" STEP 1 -- HONEST FROZEN-NCM MEASUREMENT ON V3 CACHE")
    print("=========================================================================================================")

    cache_path = os.path.join(REPO_ROOT, CACHE_FILE)
    if not os.path.isfile(cache_path):
        print(f"ERROR: Cache file {cache_path} not found.")
        sys.exit(1)

    file_sha256 = sha256sum(cache_path)
    git_sha = get_git_commit()

    d = torch.load(cache_path, weights_only=False)
    tr_x, tr_y = d["train_x"], d["train_y"]
    va_x, va_y = d["val_x"], d["val_y"]
    te_x, te_y = d["test_x"], d["test_y"]

    n_train = int(tr_x.shape[0])
    n_val = int(va_x.shape[0])
    n_test = int(te_x.shape[0])
    n_classes = int(len(torch.unique(tr_y)))

    print(f"  Cache File       : {CACHE_FILE}")
    print(f"  Cache SHA-256    : {file_sha256}")
    print(f"  Git Commit SHA   : {git_sha}")
    print(f"  Loaded Samples   : Train={n_train}, Val={n_val}, Test={n_test} (Classes={n_classes})")

    # Assert n_optimizer_steps is 0 for NCM
    n_optimizer_steps = 0
    assert n_optimizer_steps == 0, "NCM must have zero optimizer steps"

    seed_accuracies = []
    print(f"\n  Evaluating frozen NCM on normalized features across SEEDS = {SEEDS}...")

    for seed in SEEDS:
        torch.manual_seed(seed)
        # NCM normalizes features and computes cosine/dot product with class centroids
        # In eval_core, eval_ncm takes normalized features or normalizes centroids internally.
        # Following standard practice in this repo: unit L2 normalize feature vectors
        tr_x_norm = F.normalize(tr_x.float(), dim=-1)
        te_x_norm = F.normalize(te_x.float(), dim=-1)

        res = eval_ncm(tr_x_norm, tr_y, te_x_norm, te_y)
        acc = float(res["accuracy"])
        seed_accuracies.append(acc)
        print(f"    Seed {seed:2d} -> Frozen NCM Test Accuracy = {acc:5.2f}%")

    mean_acc = sum(seed_accuracies) / float(len(seed_accuracies))
    std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in seed_accuracies) / float(len(seed_accuracies) - 1))

    print(f"\n  -------------------------------------------------------------------------------------------------------")
    print(f"  Honest Frozen-NCM Test Accuracy : {mean_acc:5.2f}% +/- {std_acc:4.2f}% (n={len(SEEDS)} seeds)")
    print(f"  n_optimizer_steps                : {n_optimizer_steps}")
    print(f"  -------------------------------------------------------------------------------------------------------")

    # Comparison to published numbers
    measured_joint_offline = 93.00
    gap_vs_honest_ncm = measured_joint_offline - mean_acc
    print(f"\n  Comparison against 93.00% Joint Offline:")
    print(f"    93.00% - Measured Frozen NCM ({mean_acc:5.2f}%) = {gap_vs_honest_ncm:+6.2f} percentage points")
    print(f"    Comparison to 85.80% (Optimistic Ceiling)    : delta = {mean_acc - 85.80:+6.2f} pp")
    print(f"    Comparison to 82.60% (Honest Test Acc)       : delta = {mean_acc - 82.60:+6.2f} pp")

    results_json = {
        "benchmark": "v3_disjoint_template",
        "cache_file": CACHE_FILE,
        "cache_sha256": file_sha256,
        "git_commit_sha": git_sha,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_test_samples": n_test,
        "n_classes": n_classes,
        "n_optimizer_steps": n_optimizer_steps,
        "seeds": SEEDS,
        "seed_accuracies": seed_accuracies,
        "frozen_ncm_mean": mean_acc,
        "frozen_ncm_std": std_acc,
        "measured_joint_offline_reference": measured_joint_offline,
        "adaptation_gap_corrected": gap_vs_honest_ncm
    }

    out_json_path = os.path.join(REPO_ROOT, "w1b_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"  Emitted JSON results to {out_json_path}")
    print("=========================================================================================================")
    print("EXIT_CODE = 0")


if __name__ == "__main__":
    main()
