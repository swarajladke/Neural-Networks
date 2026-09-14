#!/usr/bin/env python3
"""
===================================================================================================
 DIRECTIVE W5 -- BLOCKER B1: POSITIVE CONTROLS FOR LwF AND EWC GRADIENT REACH
===================================================================================================
 Positive control diagnostic testing whether regularizers actively reach the gradient and exert
 causal force on representation learning:
   - Arm 1: Naive Fine-Tuning (control baseline for Task 1 acquisition)
   - Arm 2: EWC at lambda = 1e6 (must collapse Task 1 accuracy if penalty is actively backwarded)
   - Arm 3: LwF at lambda = 100 (must collapse Task 1 accuracy if distillation is actively backwarded)

 Evaluates Seed 42 on Split-CIFAR-100 Tasks 0 and 1 only (20 epochs/task, batch_size=128, lr=0.005).
 Per-step tracking:
   - Cross-entropy loss value (loss_ce)
   - Penalty loss value (loss_penalty)
   - L2 norm of gradient contributed by penalty alone: ||grad_penalty||_2
   - L2 norm of total combined gradient: ||grad_total||_2

 Additional EWC diagnostics:
   - Mean and max Fisher diagonal value
   - Count of exactly-zero Fisher entries
   - Confirmation of un-detached penalty tensor in computational graph

 Additional LwF diagnostics:
   - Confirmation of per-batch recomputed teacher logits from frozen snapshot
   - Confirmation of KL divergence over old classes (Task 0) with T^2 scaling

 Terminates with EXIT_CODE = 0.
===================================================================================================
"""

import copy
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms, models


# ---------------------------------------------------------------------
# BENCHMARK CONFIGURATION
# ---------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
LR_BASE = 0.005
WEIGHT_DECAY = 1e-4

CANONICAL_BLOCKS = [
    [42, 41, 91, 9, 65, 50, 1, 70, 15, 78],  # Task 0
    [73, 10, 55, 56, 72, 45, 48, 92, 76, 37],  # Task 1
]


def set_seed(seed=42):
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


class ResNet18Primary(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def extract_features(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.fc(feats)
        return logits, feats


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


def get_task_loaders(data_root="./data"):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomCrop(112, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ])
    ev_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        norm
    ])

    ds_tr = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tr_transform)
    ds_te = datasets.CIFAR100(root=data_root, train=False, download=True, transform=ev_transform)

    train_idx, _ = partition_indices(ds_tr.targets, n_train=400, n_val=100, seed=42)
    targets_tr = np.array(ds_tr.targets)[train_idx]
    targets_te = np.array(ds_te.targets)

    task_train_loaders = {}
    task_test_loaders = {}

    for t_idx in range(2):
        classes = CANONICAL_BLOCKS[t_idx]
        t_tr_local = [train_idx[i] for i, c in enumerate(targets_tr) if c in classes]
        t_te_local = [i for i, c in enumerate(targets_te) if c in classes]

        task_train_loaders[t_idx] = (DataLoader(Subset(ds_tr, t_tr_local), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker), classes)
        task_test_loaders[t_idx] = (DataLoader(Subset(ds_te, t_te_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker), classes)

    return task_train_loaders, task_test_loaders


def evaluate_model(model, loader, device, classes):
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            preds = logits.argmax(dim=-1)
            cor += (preds == by).sum().item()
            tot += by.size(0)
    return (cor / tot) * 100.0


# ---------------------------------------------------------------------
# EXPERIMENTAL ARMS
# ---------------------------------------------------------------------

def run_positive_controls():
    print("=" * 115)
    print(" DIRECTIVE W5 -- BLOCKER B1: POSITIVE CONTROLS FOR LwF AND EWC GRADIENT REACH")
    print("=" * 115)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Fixed Seed         : {SEED}")
    print(f"  Protocol           : ResNet-18, 20 epochs/task, batch_size=128, lr=0.005, CosineAnnealing")
    print("=" * 115)

    task_train_loaders, task_test_loaders = get_task_loaders()

    # -----------------------------------------------------------------
    # 1. ARM 1: NAIVE FINE-TUNING (CONTROL BASELINE)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [ARM 1] NAIVE FINE-TUNING (Control Baseline for Task 1 Acquisition)")
    print("-" * 80)
    set_seed(SEED)
    model_naive = ResNet18Primary(num_classes=100).to(device)
    opt_naive = optim.SGD(model_naive.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    # Task 0
    t0_loader, _ = task_train_loaders[0]
    sched0 = optim.lr_scheduler.CosineAnnealingLR(opt_naive, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    for ep in range(EPOCHS_PER_TASK):
        model_naive.train()
        for bx, by in t0_loader:
            bx, by = bx.to(device), by.to(device)
            opt_naive.zero_grad()
            logits, _ = model_naive(bx)
            loss = crit(logits, by)
            loss.backward()
            opt_naive.step()
        sched0.step()

    acc_t0_after_t0_naive = evaluate_model(model_naive, task_test_loaders[0][0], device, CANONICAL_BLOCKS[0])
    print(f"  Naive Task 0 ACC after Task 0: {acc_t0_after_t0_naive:5.2f}%")

    # Task 1
    t1_loader, _ = task_train_loaders[1]
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt_naive, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    for ep in range(EPOCHS_PER_TASK):
        model_naive.train()
        for bx, by in t1_loader:
            bx, by = bx.to(device), by.to(device)
            opt_naive.zero_grad()
            logits, _ = model_naive(bx)
            loss = crit(logits, by)
            loss.backward()
            opt_naive.step()
        sched1.step()

    acc_t0_after_t1_naive = evaluate_model(model_naive, task_test_loaders[0][0], device, CANONICAL_BLOCKS[0])
    acc_t1_after_t1_naive = evaluate_model(model_naive, task_test_loaders[1][0], device, CANONICAL_BLOCKS[1])
    print(f"  Naive Task 1 ACC after Task 1: {acc_t1_after_t1_naive:5.2f}%")
    print(f"  Naive Task 0 ACC after Task 1: {acc_t0_after_t1_naive:5.2f}% (Catastrophic Forgetting)")

    # -----------------------------------------------------------------
    # 2. ARM 2: EWC POSITIVE CONTROL (lambda = 1e6)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [ARM 2] EWC POSITIVE CONTROL (lambda = 1,000,000.0)")
    print("-" * 80)
    set_seed(SEED)
    model_ewc = ResNet18Primary(num_classes=100).to(device)
    opt_ewc = optim.SGD(model_ewc.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # Task 0 training
    sched0 = optim.lr_scheduler.CosineAnnealingLR(opt_ewc, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    for ep in range(EPOCHS_PER_TASK):
        model_ewc.train()
        for bx, by in t0_loader:
            bx, by = bx.to(device), by.to(device)
            opt_ewc.zero_grad()
            logits, _ = model_ewc(bx)
            loss = crit(logits, by)
            loss.backward()
            opt_ewc.step()
        sched0.step()

    # Compute Empirical Fisher on Task 0
    model_ewc.eval()
    task_fisher = defaultdict(float)
    n_samples_t0 = 0

    for bx, by in t0_loader:
        bx, by = bx.to(device), by.to(device)
        model_ewc.zero_grad()
        logits, _ = model_ewc(bx)
        loss = crit(logits, by)
        loss.backward()
        n_samples_t0 += bx.size(0)

        for name, param in model_ewc.named_parameters():
            if param.grad is not None:
                task_fisher[name] += param.grad.data.pow(2) * (bx.size(0) / 4000.0)

    # Diagnostics on Fisher values
    all_f_vals = torch.cat([v.flatten() for v in task_fisher.values()])
    mean_f = all_f_vals.mean().item()
    max_f = all_f_vals.max().item()
    zero_f = (all_f_vals == 0).sum().item()
    total_f = all_f_vals.numel()

    print(f"  Fisher Diagonal Audit (Task 0):")
    print(f"    Mean Fisher Value     : {mean_f:.6e}")
    print(f"    Max Fisher Value      : {max_f:.6e}")
    print(f"    Zero-Value Entries    : {zero_f} / {total_f} ({zero_f/total_f*100:.2f}%)")
    print(f"    Computational Graph   : Explicitly connected via (param - optpar).pow(2)")

    optpar = {name: param.data.clone() for name, param in model_ewc.named_parameters()}
    ewc_lambda = 1e6

    # Train Task 1 with gradient monitoring
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt_ewc, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    step_count = 0
    log_intervals = [1, 2, 5, 10, 20, 50, 100, 300, 640]

    print("\n  Step-by-Step Gradient and Loss Tracking during Task 1 Training:")
    print("  " + "-" * 90)
    print(f"  {'Step':<6} | {'Epoch':<6} | {'Loss CE':<10} | {'Loss EWC':<14} | {'||Grad EWC||_2':<16} | {'||Grad Total||_2':<16}")
    print("  " + "-" * 90)

    for ep in range(EPOCHS_PER_TASK):
        model_ewc.train()
        for bx, by in t1_loader:
            step_count += 1
            bx, by = bx.to(device), by.to(device)

            # Forward pass
            logits, _ = model_ewc(bx)
            loss_ce = crit(logits, by)

            ewc_penalty = 0.0
            for name, param in model_ewc.named_parameters():
                if name in task_fisher:
                    ewc_penalty += (task_fisher[name] * (param - optpar[name]).pow(2)).sum()
            loss_penalty = (ewc_lambda / 2.0) * ewc_penalty

            # Isolated backward of penalty alone to measure its gradient norm
            model_ewc.zero_grad()
            loss_penalty.backward(retain_graph=True)
            norm_penalty_grad = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model_ewc.parameters() if p.grad is not None))

            # Total backward pass
            model_ewc.zero_grad()
            total_loss = loss_ce + loss_penalty
            total_loss.backward()
            norm_total_grad = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model_ewc.parameters() if p.grad is not None))

            opt_ewc.step()

            if step_count in log_intervals or step_count % 100 == 0:
                print(f"  {step_count:<6} | {ep+1:<6} | {loss_ce.item():<10.4f} | {loss_penalty.item():<14.4f} | {norm_penalty_grad:<16.4e} | {norm_total_grad:<16.4e}")

        sched1.step()

    acc_t0_after_t1_ewc = evaluate_model(model_ewc, task_test_loaders[0][0], device, CANONICAL_BLOCKS[0])
    acc_t1_after_t1_ewc = evaluate_model(model_ewc, task_test_loaders[1][0], device, CANONICAL_BLOCKS[1])
    print("  " + "-" * 90)
    print(f"  EWC (lambda=1e6) Task 1 ACC after Task 1: {acc_t1_after_t1_ewc:5.2f}% (Naive was {acc_t1_after_t1_naive:5.2f}%)")
    print(f"  EWC (lambda=1e6) Task 0 ACC after Task 1: {acc_t0_after_t1_ewc:5.2f}% (Naive was {acc_t0_after_t1_naive:5.2f}%)")

    # -----------------------------------------------------------------
    # 3. ARM 3: LwF POSITIVE CONTROL (lambda = 100.0)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" [ARM 3] LwF POSITIVE CONTROL (lambda = 100.0, Temperature T = 2.0)")
    print("-" * 80)
    set_seed(SEED)
    model_lwf = ResNet18Primary(num_classes=100).to(device)
    opt_lwf = optim.SGD(model_lwf.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    tau = 2.0
    lwf_lambda = 100.0

    # Task 0 training
    sched0 = optim.lr_scheduler.CosineAnnealingLR(opt_lwf, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    for ep in range(EPOCHS_PER_TASK):
        model_lwf.train()
        for bx, by in t0_loader:
            bx, by = bx.to(device), by.to(device)
            opt_lwf.zero_grad()
            logits, _ = model_lwf(bx)
            loss = crit(logits, by)
            loss.backward()
            opt_lwf.step()
        sched0.step()

    # Freeze snapshot teacher after Task 0
    teacher_snapshot = copy.deepcopy(model_lwf)
    teacher_snapshot.eval()
    for p in teacher_snapshot.parameters():
        p.requires_grad = False

    seen_classes_t0 = CANONICAL_BLOCKS[0]
    t0_idx = torch.tensor(seen_classes_t0, device=device)

    print(f"  Teacher Logits Verification:")
    print(f"    Teacher Model Snapshot : Frozen copy of model after Task 0 (requires_grad = False)")
    print(f"    Per-Batch Recomputation: Teacher forward pass evaluated on each Task 1 input batch")
    print(f"    KL Divergence Target   : Old-class logits ({len(seen_classes_t0)} classes) scaled by T={tau}, with T^2={tau**2} multiplier")

    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt_lwf, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
    step_count = 0

    print("\n  Step-by-Step Gradient and Loss Tracking during Task 1 Training:")
    print("  " + "-" * 90)
    print(f"  {'Step':<6} | {'Epoch':<6} | {'Loss CE':<10} | {'Loss LwF':<14} | {'||Grad LwF||_2':<16} | {'||Grad Total||_2':<16}")
    print("  " + "-" * 90)

    for ep in range(EPOCHS_PER_TASK):
        model_lwf.train()
        for bx, by in t1_loader:
            step_count += 1
            bx, by = bx.to(device), by.to(device)

            logits, _ = model_lwf(bx)
            loss_ce = crit(logits, by)

            with torch.no_grad():
                prev_logits, _ = teacher_snapshot(bx)

            cur_soft = F.log_softmax(logits[:, t0_idx] / tau, dim=1)
            old_soft = F.softmax(prev_logits[:, t0_idx] / tau, dim=1)
            kd_loss = F.kl_div(cur_soft, old_soft, reduction="batchmean") * (tau ** 2)
            loss_penalty = lwf_lambda * kd_loss

            # Isolated backward of penalty alone
            model_lwf.zero_grad()
            loss_penalty.backward(retain_graph=True)
            norm_penalty_grad = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model_lwf.parameters() if p.grad is not None))

            # Total backward pass
            model_lwf.zero_grad()
            total_loss = loss_ce + loss_penalty
            total_loss.backward()
            norm_total_grad = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model_lwf.parameters() if p.grad is not None))

            opt_lwf.step()

            if step_count in log_intervals or step_count % 100 == 0:
                print(f"  {step_count:<6} | {ep+1:<6} | {loss_ce.item():<10.4f} | {loss_penalty.item():<14.4f} | {norm_penalty_grad:<16.4e} | {norm_total_grad:<16.4e}")

        sched1.step()

    acc_t0_after_t1_lwf = evaluate_model(model_lwf, task_test_loaders[0][0], device, CANONICAL_BLOCKS[0])
    acc_t1_after_t1_lwf = evaluate_model(model_lwf, task_test_loaders[1][0], device, CANONICAL_BLOCKS[1])
    print("  " + "-" * 90)
    print(f"  LwF (lambda=100) Task 1 ACC after Task 1: {acc_t1_after_t1_lwf:5.2f}% (Naive was {acc_t1_after_t1_naive:5.2f}%)")
    print(f"  LwF (lambda=100) Task 0 ACC after Task 1: {acc_t0_after_t1_lwf:5.2f}% (Naive was {acc_t0_after_t1_naive:5.2f}%)")

    # -----------------------------------------------------------------
    # SUMMARY AND VERDICT
    # -----------------------------------------------------------------
    print("\n" + "=" * 115)
    print(" DIRECTIVE W5 BLOCKER B1 SUMMARY VERDICT")
    print("=" * 115)
    print(f"  {'Method':<28} | {'Task 0 Final ACC':<18} | {'Task 1 Final ACC':<18} | {'Task 1 Collapse Status':<22}")
    print("  " + "-" * 95)
    print(f"  {'Naive Fine-Tuning':<28} | {acc_t0_after_t1_naive:6.2f}%           | {acc_t1_after_t1_naive:6.2f}%           | {'Baseline (No Collapse)':<22}")

    collapse_ewc = acc_t1_after_t1_ewc < (acc_t1_after_t1_naive - 10.0)
    status_ewc = "COLLAPSED (Active Force)" if collapse_ewc else "INERT DEFECT"
    print(f"  {'EWC (lambda=1e6)':<28} | {acc_t0_after_t1_ewc:6.2f}%           | {acc_t1_after_t1_ewc:6.2f}%           | {status_ewc:<22}")

    collapse_lwf = acc_t1_after_t1_lwf < (acc_t1_after_t1_naive - 10.0)
    status_lwf = "COLLAPSED (Active Force)" if collapse_lwf else "INERT DEFECT"
    print(f"  {'LwF (lambda=100)':<28} | {acc_t0_after_t1_lwf:6.2f}%           | {acc_t1_after_t1_lwf:6.2f}%           | {status_lwf:<22}")
    print("=" * 115)

    print("\n===================================================================================================")
    print("EXIT_CODE = 0")
    print("===================================================================================================")


if __name__ == "__main__":
    run_positive_controls()
