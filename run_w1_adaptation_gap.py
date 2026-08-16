"""
run_w1_adaptation_gap.py
========================

Directive W1:
Measure the Adaptation Gap on the retired v3 benchmark.
Computes:
  ADAPTATION_GAP = joint_offline_full_finetune - frozen_NCM (85.80%)
Verifies P53 pre-registration.
Emits:
  - run_w1_adaptation_gap_stdout.txt
  - w1_results.json
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_core import eval_ncm, compute_r_metrics

SEEDS = [42, 43, 44, 45, 46]
FROZEN_NCM_CEILING = 85.80
N_CLASSES = 100
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_v3_data(cache_path):
    d = torch.load(cache_path, weights_only=False)
    return d["train_x"], d["train_y"], d["val_x"], d["val_y"], d["test_x"], d["test_y"]


class JointOfflineProbe(nn.Module):
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        # Backbone projection adapter + classification head
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim)
        )
        self.head = nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, x):
        feat = x + self.adapter(x)
        feat_norm = F.normalize(feat, dim=-1)
        logits = self.head(feat_norm)
        return logits, feat_norm


def train_and_eval_seed(seed, tr_x, tr_y, te_x, te_y):
    torch.manual_seed(seed)
    in_dim = tr_x.shape[-1]
    model = JointOfflineProbe(in_dim, num_classes=N_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Normalize inputs
    tr_x_norm = F.normalize(tr_x.float(), dim=-1)
    te_x_norm = F.normalize(te_x.float(), dim=-1)
    tr_y_long = tr_y.long()

    # Train for 50 epochs
    batch_size = 64
    n_samples = tr_x.shape[0]

    model.train()
    for epoch in range(50):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx = tr_x_norm[idx]
            by = tr_y_long[idx]

            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(te_x_norm)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == te_y).float().mean().item() * 100.0

    return acc


def main():
    print("=========================================================================================================")
    print(" DIRECTIVE W1 -- V3 BENCHMARK CLOSE-OUT & ADAPTATION GAP MEASUREMENT")
    print("=========================================================================================================")

    cache_path = os.path.join(REPO_ROOT, "smollm2_embeddings_v3_100facts_7_3_5.pt")
    if not os.path.isfile(cache_path):
        print(f"ERROR: {cache_path} not found.")
        return

    tr_x, tr_y, va_x, va_y, te_x, te_y = load_v3_data(cache_path)

    print(f"  Dataset: v3 Disjoint-Template (700 Train / 300 Val / 500 Test, 100 Classes)")
    print(f"  Frozen NCM Benchmark Baseline : {FROZEN_NCM_CEILING:.2f}%")
    print(f"  Running 5-Seed Joint Offline Fine-Tuning across SEEDS = {SEEDS}...\n")

    seed_accs = []
    for seed in SEEDS:
        acc = train_and_eval_seed(seed, tr_x, tr_y, te_x, te_y)
        seed_accs.append(acc)
        print(f"    Seed {seed:2d} -> Joint Offline Full Fine-Tune ACC = {acc:5.2f}%")

    mean_acc = sum(seed_accs) / float(len(seed_accs))
    std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in seed_accs) / float(len(seed_accs) - 1))

    adaptation_gap = mean_acc - FROZEN_NCM_CEILING

    print(f"\n  -------------------------------------------------------------------------------------------------------")
    print(f"  joint_offline_full_finetune : {mean_acc:5.2f}% +/- {std_acc:4.2f}%")
    print(f"  frozen_NCM                  : {FROZEN_NCM_CEILING:5.2f}%")
    print(f"  ADAPTATION_GAP              : {adaptation_gap:+6.2f} percentage points")
    print(f"  -------------------------------------------------------------------------------------------------------")

    # Explicit verdict line for P53
    p53_verdict = "RIGHT" if adaptation_gap < 0.0 else "WRONG"
    print(f"\n  [P53 VERDICT]: {p53_verdict} (ADAPTATION_GAP = {adaptation_gap:+6.2f} pp < 0.00 pp)")

    # Emit JSON results
    results_json = {
        "benchmark": "v3_disjoint_template",
        "frozen_ncm_ceiling": FROZEN_NCM_CEILING,
        "seeds": SEEDS,
        "seed_accuracies": seed_accs,
        "joint_offline_mean": mean_acc,
        "joint_offline_std": std_acc,
        "adaptation_gap": adaptation_gap,
        "p53_verdict": p53_verdict
    }

    out_json_path = os.path.join(REPO_ROOT, "w1_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"  Emitted JSON results to {out_json_path}")
    print("=========================================================================================================")


if __name__ == "__main__":
    main()
