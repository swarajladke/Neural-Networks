"""
run_joint_offline_probe.py
===========================

Phase III -- Joint Offline Upper Bound Probe
Evaluates HeadL1c on all 300 train vectors / 100 classes simultaneously for 200 epochs across seeds [42, 43, 44, 45, 46].

Configurations Evaluated:
1. BEST_CELL: 'mean / center+ZCA_whiten'
2. Baseline Cell: 'mean / none'

Standing Rules:
- R4: Guard raises on missing input file.
- R6: Label-derived grouping for metrics/evaluation.
- R7: Transforms fit on TRAIN data only. Explicit confirmation printed.
- R8: Seeds [42, 43, 44, 45, 46] fixed and logged.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN_CACHE_PATH = "smollm2_embeddings_v2_100facts.pt"


class HeadL1c(nn.Module):
    def __init__(self, in_features=960, out_features=100, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        # Normalize initialization
        w = torch.randn(out_features, in_features)
        w = F.normalize(w, dim=-1)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        # x shape (batch, 960), weight shape (100, 960)
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cos_sim = x_norm @ w_norm.T  # shape (batch, 100)
        return self.scale * cos_sim


def apply_transform(train_x, test_x, transform_name):
    if transform_name == "none":
        print("  [R7 Confirm] Transform 'none': Raw L2-normalized vectors used.", flush=True)
        return F.normalize(train_x, dim=-1), F.normalize(test_x, dim=-1)
    elif transform_name == "center+ZCA_whiten":
        mu = train_x.mean(dim=0, keepdim=True)
        tr_centered = train_x - mu
        te_centered = test_x - mu

        # Covariance fit on train only
        N = tr_centered.shape[0]
        cov = (tr_centered.T @ tr_centered) / (N - 1)
        cov += 1e-5 * torch.eye(cov.shape[0])

        S, V = torch.linalg.eigh(cov)
        S = torch.clamp(S, min=1e-5)
        W_zca = V @ torch.diag(1.0 / torch.sqrt(S)) @ V.T

        print("  [R7 Confirm] ZCA whitening transform fit on 300 train vectors only (epsilon=1e-5).", flush=True)

        tr_w = tr_centered @ W_zca.T
        te_w = te_centered @ W_zca.T

        return F.normalize(tr_w, dim=-1), F.normalize(te_w, dim=-1)
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


def run_single_probe(train_x, train_y, test_x, test_y, seed, epochs=200):
    torch.manual_seed(seed)
    model = HeadL1c(in_features=960, out_features=100, scale=10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs_to_100_train = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        # Track train accuracy
        with torch.no_grad():
            train_preds = logits.argmax(dim=1)
            train_acc = (train_preds == train_y).float().mean().item() * 100.0
            if train_acc == 100.0 and epochs_to_100_train is None:
                epochs_to_100_train = epoch

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        final_tr_logits = model(train_x)
        final_tr_preds = final_tr_logits.argmax(dim=1)
        final_tr_acc = (final_tr_preds == train_y).float().mean().item() * 100.0

        final_te_logits = model(test_x)
        final_te_preds = final_te_logits.argmax(dim=1)
        final_te_acc = (final_te_preds == test_y).float().mean().item() * 100.0

    return {
        "seed": seed,
        "train_acc": final_tr_acc,
        "test_acc": final_te_acc,
        "epochs_to_100_train": epochs_to_100_train if epochs_to_100_train is not None else "Never reached 100%"
    }


def evaluate_configuration(config_name, transform_name):
    if not os.path.exists(MEAN_CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required cache file: '{MEAN_CACHE_PATH}'.")

    data = torch.load(MEAN_CACHE_PATH, weights_only=False)
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    print(f"\n==================================================", flush=True)
    print(f" EVALUATING OFFLINE PROBE: {config_name}", flush=True)
    print(f"==================================================", flush=True)

    tr_x, te_x = apply_transform(train_x, test_x, transform_name)
    seeds = [42, 43, 44, 45, 46]
    run_results = []

    print(f"{'Seed':<8} | {'Final Train Acc':<16} | {'Final Test Acc':<16} | {'Epochs to 100% Train'}", flush=True)
    print("-" * 65, flush=True)

    for seed in seeds:
        res = run_single_probe(tr_x, train_y, te_x, test_y, seed=seed, epochs=200)
        run_results.append(res)
        print(f"{res['seed']:<8} | {res['train_acc']:6.2f}%          | {res['test_acc']:6.2f}%          | {res['epochs_to_100_train']}", flush=True)

    test_accs = torch.tensor([r["test_acc"] for r in run_results])
    train_accs = torch.tensor([r["train_acc"] for r in run_results])

    mean_test = test_accs.mean().item()
    std_test = test_accs.std().item()
    mean_train = train_accs.mean().item()

    print("-" * 65, flush=True)
    print(f"SUMMARY FOR '{config_name}':", flush=True)
    print(f"  Mean Train Accuracy: {mean_train:.2f}%", flush=True)
    print(f"  Mean Test Accuracy : {mean_test:.2f}% +/- {std_test:.2f}%", flush=True)
    print("==================================================", flush=True)

    return mean_test, std_test


def main():
    print("==================================================", flush=True)
    print(" PHASE III — JOINT OFFLINE UPPER BOUND PROBE", flush=True)
    print("==================================================", flush=True)

    # 1. BEST_CELL: mean / center+ZCA_whiten
    best_mean, best_std = evaluate_configuration("BEST_CELL (mean / center+ZCA_whiten)", "center+ZCA_whiten")

    # 2. Baseline Cell: mean / none
    base_mean, base_std = evaluate_configuration("Baseline Cell (mean / none)", "none")

    print("\n==================================================", flush=True)
    print(" PHASE III FINAL COMPARISON SUMMARY", flush=True)
    print("==================================================", flush=True)
    print(f"BEST_CELL (mean / center+ZCA_whiten) : {best_mean:.2f}% +/- {best_std:.2f}%", flush=True)
    print(f"Baseline Cell (mean / none)          : {base_mean:.2f}% +/- {base_std:.2f}%", flush=True)

    # Reference Line for RESULTS.md
    ref_line = f"Joint offline upper bound (100 classes, 3 train / 3 test per class, BEST_CELL mean / center+ZCA_whiten): {best_mean:.2f}% +/- {best_std:.2f}% test accuracy over 5 seeds. No Class-IL result on this dataset may exceed this value. Any reported figure above it is invalid by construction."
    print(f"\n[REFERENCE LINE FOR RESULTS.MD]:\n{ref_line}", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
