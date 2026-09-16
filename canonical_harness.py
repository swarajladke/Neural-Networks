#!/usr/bin/env python3
"""
canonical_harness.py
====================
The single authoritative canonical harness and configuration module for
Split-CIFAR-100 continual learning experiments on ResNet-18.

Every script in the benchmark must import and invoke setup_canonical_environment()
and use the standard data and model construction utilities defined here.
"""

import os
import sys
import random
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ---------------------------------------------------------------------
# CANONICAL BENCHMARK CONSTANTS
# ---------------------------------------------------------------------
NUM_TASKS = 10
CLASSES_PER_TASK = 10
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
LR_BASE = 0.005
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
ETA_MIN = 1e-4

# Certified Canonical Split-CIFAR-100 Class Ordering (from class_order_split_cifar100.json, Seed 42)
CANONICAL_BLOCKS = [
    [42, 41, 91, 9, 65, 50, 1, 70, 15, 78],    # Task 0
    [73, 10, 55, 56, 72, 45, 48, 92, 76, 37],  # Task 1
    [30, 21, 32, 96, 80, 49, 83, 26, 87, 33],  # Task 2
    [8, 47, 59, 63, 74, 44, 98, 94, 60, 28],   # Task 3
    [64, 18, 51, 88, 77, 85, 99, 93, 27, 40],  # Task 4
    [54, 53, 23, 62, 29, 61, 81, 79, 71, 95],  # Task 5
    [34, 86, 2, 75, 17, 36, 12, 19, 3, 5],     # Task 6
    [89, 43, 67, 24, 66, 31, 14, 25, 4, 39],   # Task 7
    [6, 82, 69, 7, 20, 97, 13, 84, 11, 22],    # Task 8
    [16, 52, 68, 57, 38, 0, 58, 35, 90, 46]    # Task 9
]


def setup_canonical_environment(seed=42):
    """
    Establish single canonical configuration across PyTorch, cuDNN, and cuBLAS.
    Prints all configuration settings at top of execution.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  [WARN] use_deterministic_algorithms failed: {e}")

    np.random.seed(seed)
    random.seed(seed)

    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        pass

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("===================================================================================================")
    print(" AUTHORITATIVE CANONICAL BENCHMARK CONFIGURATION")
    print("===================================================================================================")
    print(f"  Git Commit SHA                : {git_sha}")
    print(f"  Execution Device              : {device_name}")
    print(f"  Benchmark Seed                : {seed}")
    print(f"  cuDNN Deterministic           : {torch.backends.cudnn.deterministic}")
    print(f"  cuDNN Benchmark               : {torch.backends.cudnn.benchmark}")
    print(f"  CUBLAS_WORKSPACE_CONFIG       : {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT_SET')}")
    print(f"  Deterministic Algorithms      : True (warn_only=True)")
    print(f"  Standard Learning Rate        : {LR_BASE} (CosineAnnealingLR, eta_min={ETA_MIN})")
    print(f"  Standard Weight Decay         : {WEIGHT_DECAY}")
    print(f"  Standard Momentum             : {MOMENTUM}")
    print(f"  Standard Epochs per Task      : {EPOCHS_PER_TASK}")
    print(f"  Batch Size                    : {BATCH_SIZE}")
    print("===================================================================================================\n")


def seed_worker(worker_id):
    """Seed individual DataLoader workers deterministically."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def partition_indices(targets, n_train=400, n_val=100, seed=42):
    """Canonical 400-train / 100-val per class partition using explicit Generator."""
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
    """Authoritative ResNet-18 architecture with ImageNet-1k V1 initialization."""
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


def compute_param_checksum(model):
    """Compute exact float64 parameter checksum across all model parameters."""
    total = 0.0
    for p in model.parameters():
        total += float(p.data.double().sum().item())
    return total


def get_canonical_dataloaders(data_dir="./data", seed=42, num_tasks=2):
    """
    Construct canonical DataLoaders with seeded generators and proper transforms.
    """
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
    ds_ev = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_eval)
    ds_te = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_eval)

    train_idxs, val_idxs = partition_indices(ds_tr.targets, n_train=400, n_val=100, seed=seed)

    task_train_loaders = []
    task_train_eval_loaders = []
    task_val_loaders = []
    task_test_loaders = []

    for t_idx in range(num_tasks):
        classes = CANONICAL_BLOCKS[t_idx]
        t_tr = [i for i in train_idxs if ds_tr.targets[i] in classes]
        t_val = [i for i in val_idxs if ds_tr.targets[i] in classes]
        t_te = [i for i, y in enumerate(ds_te.targets) if y in classes]

        gen = torch.Generator().manual_seed(seed + t_idx)

        tr_ld = DataLoader(
            Subset(ds_tr, t_tr),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=gen,
            worker_init_fn=seed_worker
        )
        tr_ev_ld = DataLoader(
            Subset(ds_ev, t_tr),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        val_ld = DataLoader(
            Subset(ds_ev, t_val),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        te_ld = DataLoader(
            Subset(ds_te, t_te),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        task_train_loaders.append(tr_ld)
        task_train_eval_loaders.append(tr_ev_ld)
        task_val_loaders.append(val_ld)
        task_test_loaders.append(te_ld)

    return {
        "train": task_train_loaders,
        "train_eval": task_train_eval_loaders,
        "val": task_val_loaders,
        "test": task_test_loaders,
        "train_idxs": train_idxs,
        "val_idxs": val_idxs,
        "ds_tr": ds_tr,
        "ds_ev": ds_ev,
        "ds_te": ds_te
    }


def evaluate_task(model, loader, device, allowed_classes=None):
    """
    Authoritative evaluation function.
    - If allowed_classes is None: computes 100-way class-IL accuracy and class prediction distribution.
    - If allowed_classes is provided: computes task-aware masked accuracy over allowed classes.
    """
    model.eval()
    cor, tot = 0, 0
    pred_counts = {}
    
    mask = None
    if allowed_classes is not None:
        allowed_t = torch.tensor(allowed_classes, device=device)

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            if allowed_classes is not None:
                m = torch.full_like(logits, float("-inf"))
                m[:, allowed_t] = 0.0
                logits = logits + m
            
            preds = logits.argmax(dim=-1)
            cor += (preds == by).sum().item()
            tot += by.size(0)

            for p in preds.cpu().tolist():
                pred_counts[p] = pred_counts.get(p, 0) + 1

    acc = (cor / tot) * 100.0 if tot > 0 else 0.0
    return acc, pred_counts
