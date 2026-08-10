"""
run_gate1_diagnostic.py
========================

Gate 1 Diagnostic (J = 34.80% < 40%):
Rebuilds the joint offline probe at reduced class counts by subsampling classes:
100, 50, 25, 10 classes (seeded, same 5 seeds: 42, 43, 44, 45, 46).
Evaluates HeadL1c on BEST_CELL ('mean / center+ZCA_whiten') representations.
Reports mean and std test accuracy (J) at each class count.

Standing Rules:
- R4: Guard raises on missing input file.
- R6: Label-derived indexing for class subsampling and evaluation.
- R7: Transforms fit on TRAIN vectors only for the selected class subset.
- R8: Fixed seeds [42, 43, 44, 45, 46] logged and enforced.
"""

import os
import random
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
        w = torch.randn(out_features, in_features)
        w = F.normalize(w, dim=-1)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.scale * (x_norm @ w_norm.T)


def apply_best_cell_transform(train_x, test_x):
    # R7: Fit statistics on TRAIN only
    mu = train_x.mean(dim=0, keepdim=True)
    tr_centered = train_x - mu
    te_centered = test_x - mu

    N = tr_centered.shape[0]
    cov = (tr_centered.T @ tr_centered) / (N - 1)
    cov += 1e-5 * torch.eye(cov.shape[0])

    S, V = torch.linalg.eigh(cov)
    S = torch.clamp(S, min=1e-5)
    W_zca = V @ torch.diag(1.0 / torch.sqrt(S)) @ V.T

    tr_w = tr_centered @ W_zca.T
    te_w = te_centered @ W_zca.T

    return F.normalize(tr_w, dim=-1), F.normalize(te_w, dim=-1)


def run_probe_for_subset(train_x, train_y, test_x, test_y, num_classes, seeds=[42, 43, 44, 45, 46]):
    # Select seeded class subset (using seed 42 for class selection permutation)
    rng = random.Random(42)
    all_classes = sorted(list(set(train_y.tolist())))
    selected_classes = sorted(rng.sample(all_classes, num_classes))

    # Map original class IDs to 0..num_classes-1
    class_map = {orig_c: new_c for new_c, orig_c in enumerate(selected_classes)}

    # R6: Label-derived indexing to select train/test vectors for selected classes
    tr_mask = torch.tensor([y.item() in class_map for y in train_y])
    te_mask = torch.tensor([y.item() in class_map for y in test_y])

    tr_sub_x = train_x[tr_mask]
    te_sub_x = test_x[te_mask]

    tr_sub_y = torch.tensor([class_map[y.item()] for y in train_y[tr_mask]], dtype=torch.long)
    te_sub_y = torch.tensor([class_map[y.item()] for y in test_y[te_mask]], dtype=torch.long)

    # R7: Apply BEST_CELL transform fit on current train subset only
    tr_transformed, te_transformed = apply_best_cell_transform(tr_sub_x, te_sub_x)

    test_accs = []
    epochs_to_100 = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = HeadL1c(in_features=960, out_features=num_classes, scale=10.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        ep_100 = None
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            logits = model(tr_transformed)
            loss = criterion(logits, tr_sub_y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tr_preds = logits.argmax(dim=1)
                tr_acc = (tr_preds == tr_sub_y).float().mean().item() * 100.0
                if tr_acc == 100.0 and ep_100 is None:
                    ep_100 = epoch

        model.eval()
        with torch.no_grad():
            te_logits = model(te_transformed)
            te_preds = te_logits.argmax(dim=1)
            te_acc = (te_preds == te_sub_y).float().mean().item() * 100.0

        test_accs.append(te_acc)
        epochs_to_100.append(ep_100 if ep_100 is not None else "Never")

    test_accs_t = torch.tensor(test_accs)
    return test_accs_t.mean().item(), test_accs_t.std().item(), test_accs, epochs_to_100


def main():
    if not os.path.exists(MEAN_CACHE_PATH):
        raise RuntimeError(f"[R5 Guard] Missing required cache file: '{MEAN_CACHE_PATH}'.")

    data = torch.load(MEAN_CACHE_PATH, weights_only=False)
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    print("==================================================", flush=True)
    print(" GATE 1 DIAGNOSTIC — OFFLINE PROBE VS CLASS COUNT", flush=True)
    print(" (Triggered because J = 34.80% < 40.00%)", flush=True)
    print("==================================================", flush=True)

    class_counts = [100, 50, 25, 10]
    diag_results = {}

    for k in class_counts:
        print(f"\n--- Running Subsampled Probe for {k} Classes ---", flush=True)
        mean_j, std_j, raw_accs, ep_100s = run_probe_for_subset(train_x, train_y, test_x, test_y, k)
        diag_results[k] = (mean_j, std_j)

        print(f"Subsampled Class Count : {k} classes", flush=True)
        print(f"Per-Seed Test Accs (42..46): {[f'{a:.2f}%' for a in raw_accs]}", flush=True)
        print(f"Epochs to 100% Train   : {ep_100s}", flush=True)
        print(f"J ({k} classes)          : {mean_j:.2f}% +/- {std_j:.2f}%", flush=True)

    print("\n" + "="*60, flush=True)
    print(" GATE 1 DIAGNOSTIC SUMMARY TABLE", flush=True)
    print("="*60, flush=True)
    print(f"{'Class Count (k)':<18} | {'Offline Joint Accuracy J (Mean +/- Std)'}", flush=True)
    print("-" * 60, flush=True)
    for k in class_counts:
        m, s = diag_results[k]
        print(f"{k:<18} | {m:6.2f}% +/- {s:.2f}%", flush=True)
    print("="*60, flush=True)

    # Diagnosis Statement
    j_10 = diag_results[10][0]
    j_100 = diag_results[100][0]

    print("\n--- DIAGNOSTIC FINDINGS & CONCLUSION ---", flush=True)
    print(f"1. At k=10 classes, joint offline test accuracy reaches {j_10:.2f}%.", flush=True)
    print(f"2. At k=100 classes, joint offline test accuracy drops to {j_100:.2f}%.", flush=True)
    if j_10 >= 80.0:
        print("Conclusion: The primary bottleneck is CLASS COUNT / CAPACITY. The underlying representation holds high discriminative signal for small class subsets, but performance scales down sharply as 100-way logit competition increases.", flush=True)
    else:
        print("Conclusion: The primary bottleneck is REPRESENTATION DISCRIMINABILITY. Accuracy remains low even at small class counts.", flush=True)

    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
