"""
run_w7_determinism.py
=====================
Directive W7 -- Item W7-0: Pin Reproducibility

Investigates and resolves the divergence of identical naive fine-tuning baselines:
  - Run A (W5): Naive Task 0 ACC, seed 42 = 32.80%
  - Run B (W6): Naive Task 0 ACC, seed 42 = 40.80%
  - W3 Baseline: Naive Task 0 ACC, seed 42 = 31.90%

Tests:
  1. Standard configuration run 3 times inside single process + 1 time in fresh subprocess.
  2. Full determinism configuration run 3 times inside single process + 1 time in fresh subprocess.
  3. Batch sample indices, class order, split sizes, and weight decay protocol comparison.
"""

import os
import sys
import copy
import time
import math
import random
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Benchmark Hyperparameters
SEED = 42
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
LR_BASE = 0.005
WEIGHT_DECAY = 5e-4

CANONICAL_BLOCKS = [
    [42, 41, 91, 9, 65, 50, 1, 70, 15, 78],  # Task 0
    [73, 10, 55, 56, 72, 45, 48, 92, 76, 37],  # Task 1
]


def set_seed(seed):
    """Seed all standard RNGs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def enable_full_determinism(seed=42):
    """Enable strict determinism across PyTorch, cuDNN, and cuBLAS."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  [Notice] torch.use_deterministic_algorithms: {e}")
    np.random.seed(seed)
    random.seed(seed)


def partition_indices(targets, n_train=400, n_val=100, seed=42):
    g = torch.Generator().manual_seed(seed)
    targets_t = torch.tensor(targets)
    train_indices = []
    val_indices = []
    for c in range(100):
        c_idxs = (targets_t == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(c_idxs), generator=g)
        shuffled = c_idxs[perm]
        train_indices.extend(shuffled[:n_train].tolist())
        val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
    return train_indices, val_indices


class ResNet18Primary(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.extract_features(x)
        logits = self.fc(h)
        return logits, h

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def get_data(data_dir="./data", deterministic=False, seed=42):
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomCrop(112, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_norm
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        imagenet_norm
    ])

    ds_tr = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    ds_te = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_eval)

    train_idxs, val_idxs = partition_indices(ds_tr.targets, n_train=400, n_val=100, seed=42)

    t0_classes = CANONICAL_BLOCKS[0]
    t1_classes = CANONICAL_BLOCKS[1]

    t0_tr = [i for i in train_idxs if ds_tr.targets[i] in t0_classes]
    t1_tr = [i for i in train_idxs if ds_tr.targets[i] in t1_classes]
    t0_te = [i for i, y in enumerate(ds_te.targets) if y in t0_classes]
    t1_te = [i for i, y in enumerate(ds_te.targets) if y in t1_classes]

    gen0 = torch.Generator().manual_seed(seed) if deterministic else None
    gen1 = torch.Generator().manual_seed(seed + 1) if deterministic else None

    t0_loader = DataLoader(
        Subset(ds_tr, t0_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=gen0,
        worker_init_fn=seed_worker if deterministic else None
    )
    t1_loader = DataLoader(
        Subset(ds_tr, t1_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=gen1,
        worker_init_fn=seed_worker if deterministic else None
    )
    t0_test = DataLoader(Subset(ds_te, t0_te), batch_size=BATCH_SIZE, shuffle=False)
    t1_test = DataLoader(Subset(ds_te, t1_te), batch_size=BATCH_SIZE, shuffle=False)

    return t0_loader, t1_loader, t0_test, t1_test, train_idxs, val_idxs, ds_tr, ds_te, t0_tr, t1_tr


def evaluate_task(model, loader, device):
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            preds = logits.argmax(dim=-1)
            cor += (preds == by).sum().item()
            tot += by.size(0)
    return (cor / tot) * 100.0 if tot > 0 else 0.0


def compute_param_checksum(model):
    """Compute exact float64 sum of all parameters."""
    total = 0.0
    for p in model.parameters():
        total += float(p.data.double().sum().item())
    return total


def run_naive_two_task(deterministic=False, seed=42, weight_decay=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if deterministic:
        enable_full_determinism(seed)
    else:
        set_seed(seed)

    t0_loader, t1_loader, t0_test, t1_test, train_idxs, val_idxs, ds_tr, ds_te, t0_tr, t1_tr = get_data(
        deterministic=deterministic, seed=seed
    )

    if deterministic:
        enable_full_determinism(seed)
    else:
        set_seed(seed)

    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    # Train Task 0
    sched0 = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    for ep in range(EPOCHS_PER_TASK):
        model.train()
        for bx, by in t0_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
        sched0.step()

    acc_t0_after_t0 = evaluate_task(model, t0_test, device)
    chk_t0 = compute_param_checksum(model)

    # Train Task 1
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    for ep in range(EPOCHS_PER_TASK):
        model.train()
        for bx, by in t1_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
        sched1.step()

    acc_t0_after_t1 = evaluate_task(model, t0_test, device)
    acc_t1_after_t1 = evaluate_task(model, t1_test, device)
    chk_t1 = compute_param_checksum(model)

    return {
        "acc_t0_after_t0": acc_t0_after_t0,
        "acc_t0_after_t1": acc_t0_after_t1,
        "acc_t1_after_t1": acc_t1_after_t1,
        "checksum_t0": chk_t0,
        "checksum_t1": chk_t1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["full", "single_fresh"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()

    if args.mode == "single_fresh":
        res = run_naive_two_task(deterministic=args.deterministic, seed=42, weight_decay=args.weight_decay)
        print(f"FRESH_RUN_RESULT: T0_init={res['acc_t0_after_t0']:.2f}% | T0_final={res['acc_t0_after_t1']:.2f}% | T1_final={res['acc_t1_after_t1']:.2f}% | Chk_T0={res['checksum_t0']:.8f} | Chk_T1={res['checksum_t1']:.8f}")
        return

    print("=" * 110)
    print(" DIRECTIVE W7 -- ITEM W7-0: REPRODUCIBILITY PINNING & RECONCILIATION")
    print("==========================================================================================")
    print("  Seed               : 42")
    print("  Tasks Evaluated    : Tasks 0 & 1 (Canonical Blocks)")
    print("  Discrepancy Audit  : W5 reported 32.80% vs W6 reported 40.80% (Delta = 8.00 pp)")
    print("==========================================================================================")

    # Print Data Construction Diagnostics
    t0_loader, t1_loader, t0_test, t1_test, train_idxs, val_idxs, ds_tr, ds_te, t0_tr, t1_tr = get_data(deterministic=False, seed=42)
    print("\n  [Data Construction Audit]")
    print(f"    Class Order Task 0   : {CANONICAL_BLOCKS[0]}")
    print(f"    Class Order Task 1   : {CANONICAL_BLOCKS[1]}")
    print(f"    Total Partition Sizes: Train = {len(train_idxs)}, Val = {len(val_idxs)}, Test = {len(ds_te)}")
    print(f"    Task 0 Split Sizes   : Train = {len(t0_tr)}, Test = {len(t0_test.dataset)}")
    print(f"    Task 1 Split Sizes   : Train = {len(t1_tr)}, Test = {len(t1_test.dataset)}")

    # Sample indices of first batch
    t0_first_batch_indices = [t0_tr[i] for i in range(10)]
    t1_first_batch_indices = [t1_tr[i] for i in range(10)]
    print(f"    Task 0 First 10 Sample Indices: {t0_first_batch_indices}")
    print(f"    Task 1 First 10 Sample Indices: {t1_first_batch_indices}")

    # 1. Standard Configuration (Old W6 Setup)
    print("\n" + "-" * 110)
    print(" (1) STANDARD CONFIGURATION (torch.manual_seed, cuDNN deterministic, no CuBLAS env / no explicit DataLoader generator)")
    print("-" * 110)
    print(f"  {'Run':<18} | {'Task 0 Init ACC':<18} | {'Task 0 Final ACC':<18} | {'Task 1 Final ACC':<18} | {'Param Checksum (T0)':<22} | {'Param Checksum (T1)'}")
    print("  " + "-" * 110)

    std_results = []
    for i in range(3):
        res = run_naive_two_task(deterministic=False, seed=42, weight_decay=5e-4)
        std_results.append(res)
        print(f"  Single-Proc #{i+1:<4} | {res['acc_t0_after_t0']:5.2f}%             | {res['acc_t0_after_t1']:5.2f}%             | {res['acc_t1_after_t1']:5.2f}%             | {res['checksum_t0']:<22.8f} | {res['checksum_t1']:.8f}")

    # Fresh Process for Standard
    proc_std = subprocess.run([sys.executable, __file__, "--mode=single_fresh"], capture_output=True, text=True)
    fresh_std_line = [l for l in proc_std.stdout.split("\n") if "FRESH_RUN_RESULT:" in l]
    print(f"  Fresh Subprocess   : {fresh_std_line[0] if fresh_std_line else 'Failed'}")

    # 2. Full Determinism Configuration (New Setup)
    print("\n" + "-" * 110)
    print(" (2) FULL DETERMINISM CONFIGURATION (cuDNN deterministic, benchmark=False, use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG, explicit DataLoader generator)")
    print("-" * 110)
    print(f"  {'Run':<18} | {'Task 0 Init ACC':<18} | {'Task 0 Final ACC':<18} | {'Task 1 Final ACC':<18} | {'Param Checksum (T0)':<22} | {'Param Checksum (T1)'}")
    print("  " + "-" * 110)

    det_results = []
    for i in range(3):
        res = run_naive_two_task(deterministic=True, seed=42, weight_decay=5e-4)
        det_results.append(res)
        print(f"  Single-Proc #{i+1:<4} | {res['acc_t0_after_t0']:5.2f}%             | {res['acc_t0_after_t1']:5.2f}%             | {res['acc_t1_after_t1']:5.2f}%             | {res['checksum_t0']:<22.8f} | {res['checksum_t1']:.8f}")

    # Fresh Process for Deterministic
    proc_det = subprocess.run([sys.executable, __file__, "--mode=single_fresh", "--deterministic"], capture_output=True, text=True)
    fresh_det_line = [l for l in proc_det.stdout.split("\n") if "FRESH_RUN_RESULT:" in l]
    print(f"  Fresh Subprocess   : {fresh_det_line[0] if fresh_det_line else 'Failed'}")

    # 3. Protocol Difference Isolation: Weight Decay Test
    print("\n" + "-" * 110)
    print(" (3) PROTOCOL DIFFERENCE ISOLATION: WEIGHT DECAY 1e-4 (W5) vs 5e-4 (W3/W6)")
    print("-" * 110)
    res_wd1e4 = run_naive_two_task(deterministic=True, seed=42, weight_decay=1e-4)
    res_wd5e4 = run_naive_two_task(deterministic=True, seed=42, weight_decay=5e-4)
    print(f"  Weight Decay = 1e-4 (W5)   -> Task 0 Final: {res_wd1e4['acc_t0_after_t1']:5.2f}% | Task 1 Final: {res_wd1e4['acc_t1_after_t1']:5.2f}%")
    print(f"  Weight Decay = 5e-4 (W3/W6)-> Task 0 Final: {res_wd5e4['acc_t0_after_t1']:5.2f}% | Task 1 Final: {res_wd5e4['acc_t1_after_t1']:5.2f}%")
    print(f"  Weight Decay Impact        : {res_wd5e4['acc_t0_after_t1'] - res_wd1e4['acc_t0_after_t1']:+.2f} pp Task 0 Final ACC")

    print("\n" + "=" * 110)
    print(" [W7-0 VERDICT]")
    chk_match_std = (std_results[0]['checksum_t1'] == std_results[1]['checksum_t1'] == std_results[2]['checksum_t1'])
    chk_match_det = (det_results[0]['checksum_t1'] == det_results[1]['checksum_t1'] == det_results[2]['checksum_t1'])
    print(f"  Standard Multi-Run Intra-Process Parameter Exact Match: {chk_match_std}")
    print(f"  Full Determinism Intra-Process Parameter Exact Match  : {chk_match_det}")
    print("==========================================================================================")
    print("EXIT_CODE = 0")
    print("==========================================================================================")


if __name__ == "__main__":
    main()
