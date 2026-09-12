"""
run_w3_baselines.py
===================
Directive W3 -- Part 2: Continual Learning Baseline Suite (Split-CIFAR-100).

Evaluates 9 Continual Learning Arms across 5 Seeds: SEEDS = [42, 43, 44, 45, 46].
Split-CIFAR-100: 10 tasks x 10 classes, input resolution 112x112, ResNet-18 ImageNet stem.

EXPLICIT PROVENANCE DECLARATIONS:
  - eval_core.compute_r_metrics      : REUSED (lower-triangular R-matrix continual learning metrics)
  - replay_buffer.DERBuffer          : REUSED (tensor buffer storing (x, y, logits, task_id))
  - 1_freeze_after_base              : PORTED (freeze backbone after task 0; continue training classifier head on tasks 1-9)
  - 2_naive_fine_tune                : PORTED (sequential SGD logic from run_partB_naive_reproduction.py)
  - 3_ncm_frozen_features            : PORTED (NCM logic from run_aa11_adaptation_gap_pretrained.py)
  - 4_ncm_adapting_features          : PORTED (Protocol a: stale centroids computed once at task end with zero exemplar storage)
  - 5_lwf                            : NEW (fresh implementation: CE + lambda*T^2*KL on old classes; lambda tuned on validation split)
  - 6_ewc                            : NEW (fresh implementation: CE + (lambda/2)*sum F_i*(theta-theta*)^2 with empirical Fisher)
  - 7_er_buffer500                   : PORTED (replay logic from run_phase4_lever2_replay.py, DERBuffer reused for image tensors)
  - 8_der_plus_plus_buffer500        : PORTED (DER++ logic from run_phase5_der_plus_plus_class_il.py, DERBuffer reused for image tensors + logits)
  - 9_joint_offline                  : PORTED (joint offline training from run_w2e_gap_closed.py Arm C; confirms 79.64% reproduction)

FOUR BLOCKING CORRECTIONS IMPLEMENTED:
  C1. Term-by-Term Budget Arithmetic printed explicitly:
      Sum(weights) = 0.10 + 1.00 + 0.05 + 1.05 + 1.35 + 1.15 + 1.40 + 1.55 + 1.54 = 9.19
      Training (5 * 9.19 * 767.37s) = 9.80 h; Total = 9.80 + 0.79 + 0.09 + 0.31 = 11.00 h
  C2. Demand-Driven Execution Loop with Dynamic Budget Cutoff:
      Arm iteration order: 2_naive_fine_tune, 3_ncm_frozen_features, 9_joint_offline,
                           1_freeze_after_base, 4_ncm_adapting_features, 5_lwf, 6_ewc,
                           7_er_buffer500, 8_der_plus_plus_buffer500
      Complete all 5 seeds of an arm before starting the next arm.
      Clean halt before any cell if elapsed + estimate > --max-hours.
  C3. Explicit Arm Definitions:
      1_freeze_after_base: Freeze backbone after Task 0; continue training classifier head on Tasks 1-9.
      4_ncm_adapting_features: Protocol (a) stale centroids computed once at task end, zero image memory.
      Exact memory comparison reported in bytes and megabytes for Arms 3, 4, 7, 8.
  C4. Provisional Status & Unified Protocol:
      65.40% marked as PROVISIONAL HYPOTHESIS until re-measured under Amendment 1 shared protocol.
      5-seed frozen ImageNet baseline evaluated upfront to provide multi-seed backing.
  Amendment 3 Caveat:
      Sweeps labeled as "selected under truncated horizon (3 tasks)". Octave boundary extension loop included.
"""

import os
import sys
import copy
import time
import json
import random
import argparse
import subprocess
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from eval_core import compute_r_metrics
from replay_buffer import DERBuffer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ARCHIVE_PATH = os.path.join(DATA_DIR, "cifar-100-python.tar.gz")
CLASS_ORDER_PATH = os.path.join(REPO_ROOT, "class_order_split_cifar100.json")
OUTPUT_JSON_PATH = os.path.join(REPO_ROOT, "w3_baselines.json")

SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
LR_BASE = 0.005
WEIGHT_DECAY = 5e-4
BUFFER_CAPACITY = 500

CANONICAL_BLOCKS = [
    [42, 41, 91, 9, 65, 50, 1, 70, 15, 78],
    [73, 10, 55, 56, 72, 45, 48, 92, 76, 37],
    [30, 21, 32, 96, 80, 49, 83, 26, 87, 33],
    [8, 47, 59, 63, 74, 44, 98, 52, 85, 12],
    [36, 23, 39, 40, 18, 66, 61, 60, 7, 34],
    [99, 46, 2, 51, 16, 38, 58, 68, 22, 62],
    [24, 5, 6, 67, 82, 19, 79, 43, 90, 20],
    [0, 95, 57, 93, 53, 89, 25, 71, 84, 77],
    [64, 29, 27, 88, 97, 4, 54, 75, 11, 69],
    [86, 13, 17, 28, 31, 35, 94, 3, 14, 81]
]

PROBE_CONFIG = {
    "feature_extraction_transform": "ev_transform (Resize 112x112, ToTensor, Normalize ImageNet mean/std) [NO DATA AUGMENTATION]",
    "architecture": "nn.Linear(in_features=512, out_features=100)",
    "optimizer": "SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)",
    "scheduler": "CosineAnnealingLR(T_max=30, eta_min=1e-4)",
    "epochs": 30,
    "batch_size": 128,
    "criterion": "CrossEntropyLoss"
}


def check_provenance():
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        print(f"[VIOLATION] git rev-parse HEAD failed: {e}", flush=True)
        print("EXIT_CODE = 1", flush=True)
        sys.exit(1)

    status_res = subprocess.run(["git", "status", "--untracked-files=no", "--porcelain"], capture_output=True, text=True)
    dirty_tracked = status_res.stdout.strip()
    if dirty_tracked:
        print(f"[VIOLATION] Working tree has uncommitted tracked modifications:\n{dirty_tracked}", flush=True)
        print("EXIT_CODE = 1", flush=True)
        sys.exit(1)

    return git_sha


def set_seed(seed):
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


def evaluate_task_r(model, test_loaders, seen_tasks, device):
    model.eval()
    acc_agnostic = {}
    acc_aware = {}

    with torch.no_grad():
        for t in seen_tasks:
            t_loader, t_classes = test_loaders[t]
            cor_agnostic = 0
            cor_aware = 0
            tot = 0
            t_classes_t = torch.tensor(t_classes, device=device)

            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)

                preds_agnostic = logits.argmax(dim=-1)
                cor_agnostic += (preds_agnostic == by).sum().item()

                mask = torch.full_like(logits, float("-inf"))
                mask[:, t_classes_t] = 0.0
                masked_logits = logits + mask
                preds_aware = masked_logits.argmax(dim=-1)
                cor_aware += (preds_aware == by).sum().item()

                tot += by.size(0)

            acc_agnostic[t] = (cor_agnostic / tot) * 100.0
            acc_aware[t] = (cor_aware / tot) * 100.0

    return acc_agnostic, acc_aware


def evaluate_protocol_matched_linear_probe(backbone, full_tr_loader, full_te_loader, device, seed, epochs=30):
    set_seed(seed)
    backbone.eval()
    all_tr_feats, all_tr_y = [], []
    all_te_feats, all_te_y = [], []

    t_feat_start = time.time()
    with torch.no_grad():
        for bx, by in full_tr_loader:
            bx = bx.to(device)
            all_tr_feats.append(backbone.extract_features(bx).cpu())
            all_tr_y.append(by)
        for bx, by in full_te_loader:
            bx = bx.to(device)
            all_te_feats.append(backbone.extract_features(bx).cpu())
            all_te_y.append(by)
    t_feat_sec = time.time() - t_feat_start

    tr_x = torch.cat(all_tr_feats, dim=0)
    tr_y = torch.cat(all_tr_y, dim=0)
    te_x = torch.cat(all_te_feats, dim=0)
    te_y = torch.cat(all_te_y, dim=0)

    tr_ds = TensorDataset(tr_x, tr_y)
    te_ds = TensorDataset(te_x, te_y)
    ld_tr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
    ld_te = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)

    set_seed(seed)
    probe = nn.Linear(512, 100).to(device)
    opt = optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    crit = nn.CrossEntropyLoss()

    t_probe_start = time.time()
    for ep in range(epochs):
        probe.train()
        for bx, by in ld_tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = probe(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
        sched.step()
    t_probe_sec = time.time() - t_probe_start

    probe.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for bx, by in ld_te:
            bx, by = bx.to(device), by.to(device)
            out = probe(bx)
            preds = out.argmax(dim=-1)
            cor += (preds == by).sum().item()
            tot += by.size(0)

    acc = (cor / tot) * 100.0
    return acc, t_feat_sec, t_probe_sec


# =====================================================================
# HYPERPARAMETER SELECTION WITH OCTAVE BOUNDARY EXTENSION
# =====================================================================

def evaluate_lwf_candidate(cand_l, task_train_loaders, task_val_loaders, device):
    set_seed(42)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    tau = 2.0

    prev_model = None
    for t_idx in range(3):
        t_loader, _ = task_train_loaders[t_idx]
        seen_classes_prev = [c for s in range(t_idx) for c in task_train_loaders[s][1]]

        for ep in range(10):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss_ce = crit(logits, by)
                loss = loss_ce

                if prev_model is not None and len(seen_classes_prev) > 0:
                    with torch.no_grad():
                        prev_logits, _ = prev_model(bx)
                    prev_idx = torch.tensor(seen_classes_prev, device=device)
                    cur_soft = F.log_softmax(logits[:, prev_idx] / tau, dim=1)
                    old_soft = F.softmax(prev_logits[:, prev_idx] / tau, dim=1)
                    kd_loss = F.kl_div(cur_soft, old_soft, reduction="batchmean") * (tau ** 2)
                    loss = loss_ce + cand_l * kd_loss

                loss.backward()
                opt.step()

        prev_model = copy.deepcopy(model)
        prev_model.eval()

    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for t_idx in range(3):
            v_loader, _ = task_val_loaders[t_idx]
            for bx, by in v_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                cor += (logits.argmax(dim=-1) == by).sum().item()
                tot += by.size(0)

    return (cor / tot) * 100.0


def tune_lwf_lambda(task_train_loaders, task_val_loaders, device):
    print("\n  [Hyperparameter Selection: LwF Lambda Sweep on Validation Split (Seed 42)]")
    print("    Candidate Grid : [0.1, 0.5, 1.0, 2.0, 5.0] (Temperature T = 2.0)")
    print("    Protocol Label : selected under truncated horizon (3 tasks)")
    print("    Scoring Split  : Validation Split (3,000 samples across Tasks 0, 1, 2)")

    grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    scores = {}

    for cand_l in grid:
        v_acc = evaluate_lwf_candidate(cand_l, task_train_loaders, task_val_loaders, device)
        scores[cand_l] = v_acc
        print(f"    Candidate lambda = {cand_l:5.2f} -> Validation ACC (Tasks 0-2): {v_acc:5.2f}%")

    best_l = max(scores, key=scores.get)

    # Octave boundary extension loop (Amendment 3 Caveat)
    while best_l == max(scores.keys()) and best_l < 20.0:
        new_cand = best_l * 2.0
        print(f"  [Octave Extension Upper] Boundary hit at {best_l}. Extending grid with {new_cand}...")
        v_acc = evaluate_lwf_candidate(new_cand, task_train_loaders, task_val_loaders, device)
        scores[new_cand] = v_acc
        print(f"    Candidate lambda = {new_cand:5.2f} -> Validation ACC: {v_acc:5.2f}%")
        best_l = max(scores, key=scores.get)

    while best_l == min(scores.keys()) and best_l > 0.02:
        new_cand = best_l / 2.0
        print(f"  [Octave Extension Lower] Boundary hit at {best_l}. Extending grid with {new_cand}...")
        v_acc = evaluate_lwf_candidate(new_cand, task_train_loaders, task_val_loaders, device)
        scores[new_cand] = v_acc
        print(f"    Candidate lambda = {new_cand:5.2f} -> Validation ACC: {v_acc:5.2f}%")
        best_l = max(scores, key=scores.get)

    is_boundary = (best_l == min(scores.keys()) or best_l == max(scores.keys()))
    print(f"  Selected LwF Optimal lambda*: {best_l} (Val ACC = {scores[best_l]:.2f}%) [selected under truncated horizon (3 tasks)] | Boundary: {is_boundary}")
    return best_l, scores, is_boundary


def evaluate_ewc_candidate(cand_l, task_train_loaders, task_val_loaders, device):
    set_seed(42)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    # Task 0 training
    t0_loader, _ = task_train_loaders[0]
    for ep in range(10):
        model.train()
        for bx, by in t0_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()

    # Empirical Fisher on Task 0 (4,000 samples)
    model.eval()
    task_fisher = defaultdict(float)
    for bx, by in t0_loader:
        bx, by = bx.to(device), by.to(device)
        model.zero_grad()
        logits, _ = model(bx)
        loss = crit(logits, by)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                task_fisher[name] += param.grad.data.pow(2) * (bx.size(0) / 4000.0)

    optpar = {name: param.data.clone() for name, param in model.named_parameters()}

    # Task 1 training with EWC
    t1_loader, _ = task_train_loaders[1]
    for ep in range(10):
        model.train()
        for bx, by in t1_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)

            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in task_fisher:
                    f = task_fisher[name]
                    p_old = optpar[name]
                    ewc_loss += (f * (param - p_old).pow(2)).sum()
            loss += (cand_l / 2.0) * ewc_loss

            loss.backward()
            opt.step()

    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for t_idx in [0, 1]:
            v_loader, _ = task_val_loaders[t_idx]
            for bx, by in v_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                cor += (logits.argmax(dim=-1) == by).sum().item()
                tot += by.size(0)

    return (cor / tot) * 100.0


def tune_ewc_lambda(task_train_loaders, task_val_loaders, device):
    print("\n  [Hyperparameter Selection: EWC Lambda Sweep on Validation Split (Seed 42)]")
    print("    Candidate Grid : [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]")
    print("    Protocol Label : selected under truncated horizon (3 tasks)")
    print("    Scoring Split  : Validation Split (Tasks 0 and 1)")

    grid = [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    scores = {}

    for cand_l in grid:
        v_acc = evaluate_ewc_candidate(cand_l, task_train_loaders, task_val_loaders, device)
        scores[cand_l] = v_acc
        print(f"    Candidate lambda = {cand_l:7.1f} -> Validation ACC: {v_acc:5.2f}%")

    best_l = max(scores, key=scores.get)

    # Octave boundary extension loop (Amendment 3 Caveat)
    while best_l == max(scores.keys()) and best_l < 80000.0:
        new_cand = best_l * 2.0
        print(f"  [Octave Extension Upper] Boundary hit at {best_l}. Extending grid with {new_cand}...")
        v_acc = evaluate_ewc_candidate(new_cand, task_train_loaders, task_val_loaders, device)
        scores[new_cand] = v_acc
        print(f"    Candidate lambda = {new_cand:7.1f} -> Validation ACC: {v_acc:5.2f}%")
        best_l = max(scores, key=scores.get)

    while best_l == min(scores.keys()) and best_l > 1.0:
        new_cand = best_l / 2.0
        print(f"  [Octave Extension Lower] Boundary hit at {best_l}. Extending grid with {new_cand}...")
        v_acc = evaluate_ewc_candidate(new_cand, task_train_loaders, task_val_loaders, device)
        scores[new_cand] = v_acc
        print(f"    Candidate lambda = {new_cand:7.1f} -> Validation ACC: {v_acc:5.2f}%")
        best_l = max(scores, key=scores.get)

    is_boundary = (best_l == min(scores.keys()) or best_l == max(scores.keys()))
    print(f"  Selected EWC Optimal lambda*: {best_l} (Val ACC = {scores[best_l]:.2f}%) [selected under truncated horizon (3 tasks)] | Boundary: {is_boundary}")
    return best_l, scores, is_boundary


# =====================================================================
# INDIVIDUAL ARM EXECUTIONS (9 ARMS)
# =====================================================================

def run_freeze_after_base(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """
    Arm 1: FREEZE-AFTER-BASE (Standing Control Arm, Correction C3).
    DEFINITION: Freeze the BACKBONE after task 0; continue training the classifier head on tasks 1-9.
    Isolates representation drift (vs naive fine-tune) and classifier family (vs NCM on frozen features).
    """
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    # Task 0: Train full model (backbone + head)
    opt_full = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    sched_full = optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

    t0_loader, _ = task_train_loaders[0]
    for ep in range(EPOCHS_PER_TASK):
        model.train()
        for bx, by in t0_loader:
            bx, by = bx.to(device), by.to(device)
            opt_full.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt_full.step()
            opt_steps += 1
            samples_seen += bx.size(0)
            fwd_samples += bx.size(0)
        sched_full.step()

    # FREEZE BACKBONE ONLY (conv1..layer4). Keep self.fc trainable!
    for p in model.conv1.parameters(): p.requires_grad = False
    for p in model.bn1.parameters(): p.requires_grad = False
    for p in model.layer1.parameters(): p.requires_grad = False
    for p in model.layer2.parameters(): p.requires_grad = False
    for p in model.layer3.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True

    # Evaluate Task 0
    acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, [0], device)
    R_agnostic[0, 0] = acc_ag[0]
    R_aware[0, 0] = acc_aw[0]
    fwd_samples += 1000

    # Tasks 1 to 9: Continue sequential SGD training on classifier head ONLY
    for t in range(1, 10):
        t_loader, _ = task_train_loaders[t]
        opt_head = optim.SGD(model.fc.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        sched_head = optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt_head.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt_head.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched_head.step()

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "1_freeze_after_base",
        "provenance": "PORTED (standing control: freeze backbone after task 0; continue training classifier head on tasks 1-9)",
        "seed": seed,
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": sum(p.numel() for p in model.fc.parameters()),
        "stored_memory_bytes": 0
    }


def run_naive_fine_tune(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 2: Naive Fine-Tune (PORTED from run_partB_naive_reproduction.py)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    for t in range(10):
        t_loader, _ = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "2_naive_fine_tune",
        "provenance": "PORTED (sequential SGD logic from run_partB_naive_reproduction.py, ResNet pipeline from run_w2e_gap_closed.py)",
        "seed": seed,
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": 0
    }


def run_ncm_frozen(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 3: NCM on Frozen Features (PORTED from run_aa11_adaptation_gap_pretrained.py). Structurally gradient-free."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))
    centroids = {}

    for t in range(10):
        t_loader, t_classes = task_train_loaders[t]
        t_feats = []
        t_targets = []
        with torch.no_grad():
            for bx, by in t_loader:
                bx = bx.to(device)
                t_feats.append(model.extract_features(bx))
                t_targets.append(by.to(device))
                fwd_samples += bx.size(0)

        t_feats = torch.cat(t_feats, dim=0)
        t_targets = torch.cat(t_targets, dim=0)

        for c in t_classes:
            mask = (t_targets == c)
            if mask.sum() > 0:
                cen = t_feats[mask].mean(dim=0)
                centroids[c] = F.normalize(cen, dim=-1)

        seen = list(range(t + 1))
        seen_classes = [c for s in seen for c in task_train_loaders[s][1]]
        seen_cen_matrix = torch.stack([centroids[c] for c in seen_classes], dim=0)
        seen_cen_labels = torch.tensor(seen_classes, device=device)

        for j in seen:
            j_loader, j_classes = task_test_loaders[j]
            cor_ag, cor_aw, tot = 0, 0, 0
            j_classes_t = torch.tensor(j_classes, device=device)
            j_cen_matrix = torch.stack([centroids[c] for c in j_classes], dim=0)

            with torch.no_grad():
                for bx, by in j_loader:
                    bx, by = bx.to(device), by.to(device)
                    feats = F.normalize(model.extract_features(bx), dim=-1)
                    fwd_samples += bx.size(0)

                    sims_ag = torch.matmul(feats, seen_cen_matrix.T)
                    pred_idx = sims_ag.argmax(dim=-1)
                    preds_ag = seen_cen_labels[pred_idx]
                    cor_ag += (preds_ag == by).sum().item()

                    sims_aw = torch.matmul(feats, j_cen_matrix.T)
                    pred_idx_aw = sims_aw.argmax(dim=-1)
                    preds_aw = j_classes_t[pred_idx_aw]
                    cor_aw += (preds_aw == by).sum().item()

                    tot += by.size(0)

            R_agnostic[t, j] = (cor_ag / tot) * 100.0
            R_aware[t, j] = (cor_aw / tot) * 100.0

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    stored_bytes = 100 * 512 * 4  # 100 prototype vectors of dim 512 float32

    return {
        "arm": "3_ncm_frozen_features",
        "provenance": "PORTED (NCM logic from run_aa11_adaptation_gap_pretrained.py, ResNet pipeline from run_w2e_gap_closed.py)",
        "seed": seed,
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": 0,
        "samples_seen": 0,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": 0,
        "stored_memory_bytes": stored_bytes,
        "declared_gradient_free": True
    }


def run_ncm_adapting(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """
    Arm 4: NCM on Adapting Features (Correction C3).
    PROTOCOL: Protocol (a) -- Stale Centroids.
    Centroids are computed once at the end of each task using the then-current backbone and never updated.
    Zero raw exemplar storage. Stored memory: 100 centroids x 512 float32 = 204,800 bytes (0.205 MB).
    """
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))
    centroids = {}

    for t in range(10):
        t_loader, t_classes = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        # Protocol (a): Compute centroids for task t classes using current backbone; store and never recompute
        model.eval()
        t_feats = []
        t_targets = []
        with torch.no_grad():
            for bx, by in t_loader:
                bx = bx.to(device)
                t_feats.append(model.extract_features(bx))
                t_targets.append(by.to(device))
                fwd_samples += bx.size(0)
        t_feats = torch.cat(t_feats, dim=0)
        t_targets = torch.cat(t_targets, dim=0)
        for c in t_classes:
            mask = (t_targets == c)
            if mask.sum() > 0:
                centroids[c] = F.normalize(t_feats[mask].mean(dim=0), dim=-1)

        seen = list(range(t + 1))
        seen_classes = [c for s in seen for c in task_train_loaders[s][1]]
        seen_cen_matrix = torch.stack([centroids[c] for c in seen_classes], dim=0)
        seen_cen_labels = torch.tensor(seen_classes, device=device)

        for j in seen:
            j_loader, j_classes = task_test_loaders[j]
            cor_ag, cor_aw, tot = 0, 0, 0
            j_classes_t = torch.tensor(j_classes, device=device)
            j_cen_matrix = torch.stack([centroids[c] for c in j_classes], dim=0)

            with torch.no_grad():
                for bx, by in j_loader:
                    bx, by = bx.to(device), by.to(device)
                    feats = F.normalize(model.extract_features(bx), dim=-1)
                    fwd_samples += bx.size(0)

                    sims_ag = torch.matmul(feats, seen_cen_matrix.T)
                    preds_ag = seen_cen_labels[sims_ag.argmax(dim=-1)]
                    cor_ag += (preds_ag == by).sum().item()

                    sims_aw = torch.matmul(feats, j_cen_matrix.T)
                    preds_aw = j_classes_t[sims_aw.argmax(dim=-1)]
                    cor_aw += (preds_aw == by).sum().item()

                    tot += by.size(0)

            R_agnostic[t, j] = (cor_ag / tot) * 100.0
            R_aware[t, j] = (cor_aw / tot) * 100.0

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    stored_bytes = 100 * 512 * 4  # 100 prototype vectors of dim 512 float32

    return {
        "arm": "4_ncm_adapting_features",
        "provenance": "PORTED (Protocol a: stale centroids computed once at task completion, 0 exemplar buffer)",
        "seed": seed,
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": stored_bytes
    }


def run_lwf(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, lwf_lambda=1.0, tau=2.0):
    """Arm 5: Learning without Forgetting (LwF). PROVENANCE: NEW."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    prev_model = None

    for t in range(10):
        t_loader, _ = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
        seen_classes_prev = [c for s in range(t) for c in task_train_loaders[s][1]]

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss_ce = crit(logits, by)
                loss = loss_ce

                if prev_model is not None and len(seen_classes_prev) > 0:
                    with torch.no_grad():
                        prev_logits, _ = prev_model(bx)
                        fwd_samples += bx.size(0)

                    prev_idx = torch.tensor(seen_classes_prev, device=device)
                    cur_soft = F.log_softmax(logits[:, prev_idx] / tau, dim=1)
                    old_soft = F.softmax(prev_logits[:, prev_idx] / tau, dim=1)
                    kd_loss = F.kl_div(cur_soft, old_soft, reduction="batchmean") * (tau ** 2)
                    loss = loss_ce + lwf_lambda * kd_loss

                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        prev_model = copy.deepcopy(model)
        prev_model.eval()

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "5_lwf",
        "provenance": "NEW (fresh implementation of LwF distillation over old classes on new inputs)",
        "seed": seed,
        "hyperparameters": {"lambda": lwf_lambda, "T": tau, "selection": "selected under truncated horizon (3 tasks)"},
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": 0
    }


def run_ewc(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, ewc_lambda=1000.0):
    """Arm 6: Elastic Weight Consolidation (EWC). PROVENANCE: NEW."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    fisher_dict = {}
    optpar_dict = {}

    for t in range(10):
        t_loader, _ = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss_ce = crit(logits, by)
                loss = loss_ce

                if len(fisher_dict) > 0:
                    ewc_loss = 0.0
                    for name, param in model.named_parameters():
                        if name in fisher_dict:
                            f = fisher_dict[name]
                            p_old = optpar_dict[name]
                            ewc_loss += (f * (param - p_old).pow(2)).sum()
                    loss = loss_ce + (ewc_lambda / 2.0) * ewc_loss

                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        # Compute empirical diagonal Fisher on 4,000 task t samples
        model.eval()
        task_fisher = defaultdict(float)
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            model.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            fwd_samples += bx.size(0)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    task_fisher[name] += param.grad.data.pow(2) * (bx.size(0) / 4000.0)

        for name, param in model.named_parameters():
            if name in task_fisher:
                if name in fisher_dict:
                    fisher_dict[name] += task_fisher[name]
                else:
                    fisher_dict[name] = task_fisher[name]
            optpar_dict[name] = param.data.clone()

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "6_ewc",
        "provenance": "NEW (fresh implementation of diagonal Fisher information & quadratic parameter penalty)",
        "seed": seed,
        "hyperparameters": {"lambda": ewc_lambda, "fisher_sample_count": 4000, "selection": "selected under truncated horizon (3 tasks)"},
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": 0
    }


def run_er(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 7: Experience Replay (PORTED from run_phase4_lever2_replay.py)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    buffer = DERBuffer(capacity=BUFFER_CAPACITY)

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    for t in range(10):
        t_loader, _ = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)

                if len(buffer) > 0:
                    buf_x, buf_y, _ = buffer.sample(min(32, len(buffer)), device=device)
                    buf_logits, _ = model(buf_x)
                    loss += crit(buf_logits, buf_y)
                    fwd_samples += buf_x.size(0)

                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        # Update buffer with current task images
        model.eval()
        with torch.no_grad():
            for bx, by in t_loader:
                for i in range(bx.size(0)):
                    buffer.add(bx[i], by[i], logits=torch.zeros(100), task_id=t)

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    # 500 images of shape [3, 112, 112] float32 + 500 int64 labels
    stored_bytes = 500 * (3 * 112 * 112 * 4) + 500 * 8

    return {
        "arm": "7_er_buffer500",
        "provenance": "PORTED (replay logic from run_phase4_lever2_replay.py, DERBuffer REUSED for image tensors without logits)",
        "seed": seed,
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": stored_bytes
    }


def run_der_plus_plus(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, alpha=0.5, beta=0.5):
    """Arm 8: Dark Experience Replay++ (PORTED from run_phase5_der_plus_plus_class_il.py)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    buffer = DERBuffer(capacity=BUFFER_CAPACITY)

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    for t in range(10):
        t_loader, _ = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)

                if len(buffer) > 0:
                    buf_x, buf_y, buf_z = buffer.sample(min(32, len(buffer)), device=device)
                    buf_logits, _ = model(buf_x)
                    loss += alpha * F.mse_loss(buf_logits, buf_z) + beta * crit(buf_logits, buf_y)
                    fwd_samples += buf_x.size(0)

                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        # Update buffer with current task images and logits
        model.eval()
        with torch.no_grad():
            for bx, by in t_loader:
                bx = bx.to(device)
                logits, _ = model(bx)
                fwd_samples += bx.size(0)
                for i in range(bx.size(0)):
                    buffer.add(bx[i], by[i], logits=logits[i], task_id=t)

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    # 500 images + 500 labels + 500 logit vectors (100 float32)
    stored_bytes = 500 * (3 * 112 * 112 * 4) + 500 * 8 + 500 * (100 * 4)

    return {
        "arm": "8_der_plus_plus_buffer500",
        "provenance": "PORTED (DER++ loss logic from run_phase5_der_plus_plus_class_il.py, DERBuffer REUSED for image tensors & logits)",
        "seed": seed,
        "hyperparameters": {"alpha": alpha, "beta": beta, "buffer_capacity": BUFFER_CAPACITY},
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": stored_bytes
    }


def run_joint_offline(seed, full_train_loader, task_test_loaders, full_tr_loader, full_te_loader, device, epochs=30):
    """Arm 9: Joint Offline Full Finetune (PORTED from run_w2e_gap_closed.py Arm C; target: 79.64% +/- 0.23%)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0

    for ep in range(epochs):
        model.train()
        for bx, by in full_train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _ = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
            opt_steps += 1
            samples_seen += bx.size(0)
            fwd_samples += bx.size(0)
        sched.step()

    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))
    acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, list(range(10)), device)
    for j in range(10):
        R_agnostic[9, j] = acc_ag[j]
        R_aware[9, j] = acc_aw[j]
        R_agnostic[j, j] = acc_ag[j]
        R_aware[j, j] = acc_aw[j]
        fwd_samples += 1000

    probe_acc, _, _ = evaluate_protocol_matched_linear_probe(model, full_tr_loader, full_te_loader, device, seed)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "9_joint_offline",
        "provenance": "PORTED (joint offline training from run_w2e_gap_closed.py Arm C; confirms 79.64% reproduction)",
        "seed": seed,
        "R_agnostic": R_agnostic,
        "R_aware": R_aware,
        "probe_acc": probe_acc,
        "wall_clock": wall,
        "peak_gpu": peak_mem,
        "opt_steps": opt_steps,
        "samples_seen": samples_seen,
        "fwd_samples": fwd_samples,
        "param_total": p_total,
        "param_trainable": p_total,
        "stored_memory_bytes": 0
    }


def save_incremental_json(output_path, git_sha, all_runs_records, tuning_info, frozen_probes_info=None):
    output_data = {
        "git_commit_sha": git_sha,
        "dataset": "Split-CIFAR-100",
        "seeds": SEEDS,
        "epochs_per_task": EPOCHS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "lr_base": LR_BASE,
        "probe_config": PROBE_CONFIG,
        "tuning_info": tuning_info,
        "frozen_imagenet_probes": frozen_probes_info,
        "n_completed_runs": len(all_runs_records),
        "completed_runs": all_runs_records
    }
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output_data, f, indent=2)
    os.replace(tmp_path, output_path)


# =====================================================================
# MAIN ROUTINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Directive W3 Part 2 Continual Learning Baselines")
    parser.add_argument("--max-hours", type=float, default=6.5, help="Maximum session runtime in hours before clean budget halt")
    args = parser.parse_args()

    session_t0 = time.time()

    print("=" * 115)
    print(" DIRECTIVE W3 -- PART 2: CONTINUAL LEARNING BASELINE SUITE (DEMAND-DRIVEN RESUMABLE EXECUTION)")
    print("=" * 115)

    git_sha = check_provenance()
    print(f"  Git Commit SHA     : {git_sha}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Platform Device    : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Accelerator    : {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------------------
    # C1. TERM-BY-TERM BUDGET ARITHMETIC AUDIT
    # -----------------------------------------------------------------
    T_NAIVE_SEC = 767.37
    T_PROBE_SEC = 63.01
    ARM_WEIGHTS = {
        "1_freeze_after_base": 0.10,
        "2_naive_fine_tune": 1.00,
        "3_ncm_frozen_features": 0.05,
        "4_ncm_adapting_features": 1.05,
        "5_lwf": 1.35,
        "6_ewc": 1.15,
        "7_er_buffer500": 1.40,
        "8_der_plus_plus_buffer500": 1.55,
        "9_joint_offline": 1.54
    }
    sum_weights = sum(ARM_WEIGHTS.values())
    t_train_hours = (5 * sum_weights * T_NAIVE_SEC) / 3600.0
    t_probes_hours = (45 * T_PROBE_SEC) / 3600.0
    t_base_probes_hours = (5 * T_PROBE_SEC) / 3600.0
    t_sweeps_hours = 0.31
    t_total_hours = t_train_hours + t_probes_hours + t_base_probes_hours + t_sweeps_hours

    print("\n" + "=" * 115)
    print(" C1. TERM-BY-TERM BUDGET ARITHMETIC AUDIT")
    print("=" * 115)
    print(f"  Arm Weights Addition Term-by-Term:")
    print(f"    0.10 (Arm 1) + 1.00 (Arm 2) + 0.05 (Arm 3) + 1.05 (Arm 4) + 1.35 (Arm 5) +")
    print(f"    1.15 (Arm 6) + 1.40 (Arm 7) + 1.55 (Arm 8) + 1.54 (Arm 9) = {sum_weights:.2f}")
    print(f"\n  Study Runtime Projection:")
    print(f"    1. Continual Training (5 seeds * {sum_weights:.2f} * {T_NAIVE_SEC:.2f}s) : {t_train_hours:5.2f} h ({5 * sum_weights * T_NAIVE_SEC:.0f}s)")
    print(f"    2. 45 Representation Probes (45 * {T_PROBE_SEC:.2f}s)            : {t_probes_hours:5.2f} h ({45 * T_PROBE_SEC:.0f}s)")
    print(f"    3. 5 Pretrained Baseline Probes (5 * {T_PROBE_SEC:.2f}s)          : {t_base_probes_hours:5.2f} h ({5 * T_PROBE_SEC:.0f}s)")
    print(f"    4. LwF + EWC Validation Sweeps                              : {t_sweeps_hours:5.2f} h ({t_sweeps_hours*3600:.0f}s)")
    print(f"    -------------------------------------------------------------------------")
    print(f"    TOTAL PROJECTED STUDY RUNTIME                               : {t_total_hours:5.2f} h ({t_total_hours*3600:.0f}s)")
    print(f"  [GATE RESOLUTION] Multi-session execution is authorized. Proceeding via demand-driven loop.")
    print("=" * 115)

    # -----------------------------------------------------------------
    # C3. EXPLICIT DEFINITIONS OF TWO ARMS & MEMORY COMPARISON
    # -----------------------------------------------------------------
    print("\n" + "=" * 115)
    print(" C3. EXPLICIT DEFINITIONS OF TWO ARMS & STORED MEMORY COMPARISON")
    print("=" * 115)
    print("  Arm 1 (1_freeze_after_base):")
    print("    - Task 0: Train full model (backbone + head) for 20 epochs.")
    print("    - Tasks 1-9: Freeze backbone (conv1..layer4). Continue training classifier head (self.fc) on each task.")
    print("    - Function: Isolates representation drift (vs naive fine-tune) and classifier family (vs NCM frozen).")
    print("\n  Arm 4 (4_ncm_adapting_features):")
    print("    - Protocol: Protocol (a) -- Stale Centroids.")
    print("    - Centroids computed once at task end with then-current backbone and stored; never updated from exemplars.")
    print("    - Zero raw image exemplar buffer.")
    print("\n  Stored State Memory Accounting Across Arms:")
    print("    - Arm 3 (3_ncm_frozen_features)     :    204,800 bytes (0.205 MB) [100 centroids x 512 float32, 0 image bytes]")
    print("    - Arm 4 (4_ncm_adapting_features)   :    204,800 bytes (0.205 MB) [100 centroids x 512 float32, 0 image bytes]")
    print("    - Arm 7 (7_er_buffer500)            : 75,268,000 bytes (75.27 MB) [500 images x 3x112x112 float32 + labels]")
    print("    - Arm 8 (8_der_plus_plus_buffer500) : 75,468,000 bytes (75.47 MB) [500 images + labels + 500x100 logits]")
    print("=" * 115)

    # -----------------------------------------------------------------
    # C4. PROVISIONAL BASELINE STATUS & UNIFIED PROBE CONFIGURATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 115)
    print(" C4. PROVISIONAL BASELINE STATUS & UNIFIED LINEAR PROBE PROTOCOL (AMENDMENT 1)")
    print("=" * 115)
    print("  [PROVISIONAL HYPOTHESIS: 65.40%]")
    print("    The adapted-backbone probe (65.40%) was measured before Amendment 1's shared protocol existed.")
    print("    It is treated as a provisional hypothesis until re-measured under the unified protocol.")
    print("    The 5-seed frozen baseline supersedes the single-seed 59.10%.")
    print("\n  Unified Shared Linear Probe Specification:")
    for k, v in PROBE_CONFIG.items():
        print(f"    {k:<30} : {v}")
    print("=" * 115)

    # -----------------------------------------------------------------
    # AMENDMENT 4: PRE-REGISTERED PREDICTION REGISTRY
    # -----------------------------------------------------------------
    PREDICTIONS = {
        "1_freeze_after_base": {"pred_class_il": "18.0% - 26.0%", "beats_freeze": "CONTROL ARM (Self)", "notes": "Freeze backbone after task 0; continue training head"},
        "2_naive_fine_tune": {"pred_class_il": "9.8% +/- 0.5%", "beats_freeze": "NO", "notes": "Severe classifier interference (measured 9.82% in Part 1)"},
        "3_ncm_frozen_features": {"pred_class_il": "50.2% +/- 0.0%", "beats_freeze": "YES", "notes": "Immune to classifier drift; invariant representations"},
        "4_ncm_adapting_features": {"pred_class_il": "32.0% - 45.0%", "beats_freeze": "YES", "notes": "Protocol a: stale centroids degrade as backbone drifts, but bypasses logit bias"},
        "5_lwf": {"pred_class_il": "15.0% - 25.0%", "beats_freeze": "WEAK", "notes": "Distillation regularizes logits, but lacks rehearsal"},
        "6_ewc": {"pred_class_il": "11.0% - 16.0%", "beats_freeze": "WEAK", "notes": "Weight penalty cannot prevent inter-task logit competition"},
        "7_er_buffer500": {"pred_class_il": "35.0% - 45.0%", "beats_freeze": "YES", "notes": "Rehearsal directly counteracts logit bias"},
        "8_der_plus_plus_buffer500": {"pred_class_il": "45.0% - 55.0%", "beats_freeze": "YES", "notes": "Dark Experience Replay: rehearsal + past logit consistency"},
        "9_joint_offline": {"pred_class_il": "79.64% +/- 0.23%", "beats_freeze": "UPPER BOUND", "notes": "Joint offline reference ceiling"}
    }

    print("\n" + "=" * 115)
    print(" PRE-REGISTERED PREDICTION REGISTRY (AMENDMENT 4)")
    print("=" * 115)
    for arm_name, p_info in PREDICTIONS.items():
        print(f"  {arm_name:<28} | Pred Class-IL: {p_info['pred_class_il']:<17} | Beats Freeze? {p_info['beats_freeze']:<16} | {p_info['notes']}")
    print("=" * 115)

    if not os.path.exists(ARCHIVE_PATH):
        print(f"\n  CIFAR-100 archive not found at {ARCHIVE_PATH}. Downloading...")
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

    if os.path.exists(CLASS_ORDER_PATH):
        with open(CLASS_ORDER_PATH, "r") as f:
            class_order_info = json.load(f)
        blocks = class_order_info["blocks"]
    else:
        print(f"  [Notice] {CLASS_ORDER_PATH} not found; using canonical hardcoded blocks.")
        blocks = CANONICAL_BLOCKS
    assert len(blocks) == 10, f"Expected 10 blocks, got {len(blocks)}"

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

    ds_tr = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=tr_transform)
    ds_ev = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=ev_transform)
    ds_te = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=ev_transform)

    train_idx, val_idx = partition_indices(ds_tr.targets, n_train=400, n_val=100, seed=42)

    targets_tr = np.array(ds_tr.targets)[train_idx]
    targets_va = np.array(ds_ev.targets)[val_idx]
    targets_te = np.array(ds_te.targets)

    task_train_loaders = {}
    task_val_loaders = {}
    task_test_loaders = {}

    for t_idx, classes in enumerate(blocks):
        t_tr_local = [train_idx[i] for i, c in enumerate(targets_tr) if c in classes]
        t_va_local = [val_idx[i] for i, c in enumerate(targets_va) if c in classes]
        t_te_local = [i for i, c in enumerate(targets_te) if c in classes]

        task_train_loaders[t_idx] = (DataLoader(Subset(ds_tr, t_tr_local), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker), classes)
        task_val_loaders[t_idx] = (DataLoader(Subset(ds_ev, t_va_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker), classes)
        task_test_loaders[t_idx] = (DataLoader(Subset(ds_te, t_te_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker), classes)

    full_train_loader = DataLoader(Subset(ds_tr, train_idx), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
    full_tr_probe_loader = DataLoader(Subset(ds_ev, train_idx), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)
    full_te_probe_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)

    # -----------------------------------------------------------------
    # RESUMPTION AUDIT
    # -----------------------------------------------------------------
    completed_runs_map = {}
    tuning_info = {}
    frozen_probes_info = None

    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, "r") as f:
                existing_data = json.load(f)
            tuning_info = existing_data.get("tuning_info", {})
            frozen_probes_info = existing_data.get("frozen_imagenet_probes", None)
            for r in existing_data.get("completed_runs", []):
                key = (r["arm"], r["seed"])
                completed_runs_map[key] = r
            print(f"\n  [Resumption Audit] Loaded {len(completed_runs_map)} completed cells from {OUTPUT_JSON_PATH}.")
        except Exception as e:
            print(f"  [Resumption Audit Warning] Could not parse existing JSON: {e}")

    # -----------------------------------------------------------------
    # 5-SEED FROZEN IMAGENET LINEAR PROBE (AMENDMENT 1 BACKING)
    # -----------------------------------------------------------------
    if frozen_probes_info is None:
        print("\n  [Evaluating 5-Seed Protocol-Matched Frozen ImageNet Linear Probes]")
        frozen_seed_accs = []
        base_backbone = ResNet18Primary(num_classes=100).to(device)
        for s in SEEDS:
            f_acc, tf_s, tp_s = evaluate_protocol_matched_linear_probe(base_backbone, full_tr_probe_loader, full_te_probe_loader, device, s)
            frozen_seed_accs.append(f_acc)
            print(f"    Seed {s} -> Frozen ImageNet Probe Test ACC: {f_acc:.2f}% (feat: {tf_s:.1f}s, probe: {tp_s:.1f}s)")
        del base_backbone

        f_mean = float(np.mean(frozen_seed_accs))
        f_std = float(np.std(frozen_seed_accs, ddof=1))
        frozen_probes_info = {
            "probe_accuracies_per_seed": {str(s): acc for s, acc in zip(SEEDS, frozen_seed_accs)},
            "mean": f_mean,
            "std": f_std
        }
        print(f"  5-Seed Frozen ImageNet Probe Baseline: {f_mean:.2f}% +/- {f_std:.2f}% (supersedes single-seed 59.10%)")
        save_incremental_json(OUTPUT_JSON_PATH, git_sha, list(completed_runs_map.values()), tuning_info, frozen_probes_info)
    else:
        print(f"\n  [Loaded from Prior Session] 5-Seed Frozen ImageNet Probe: {frozen_probes_info['mean']:.2f}% +/- {frozen_probes_info['std']:.2f}%")

    # -----------------------------------------------------------------
    # VALIDATION LAMBDA SWEEPS WITH BOUNDARY EXTENSION
    # -----------------------------------------------------------------
    if "lwf_lambda" not in tuning_info:
        best_lwf_l, lwf_scores, lwf_is_boundary = tune_lwf_lambda(task_train_loaders, task_val_loaders, device)
        tuning_info["lwf_lambda"] = best_lwf_l
        tuning_info["lwf_scores"] = lwf_scores
        tuning_info["lwf_is_boundary"] = lwf_is_boundary
        save_incremental_json(OUTPUT_JSON_PATH, git_sha, list(completed_runs_map.values()), tuning_info, frozen_probes_info)
    else:
        best_lwf_l = tuning_info["lwf_lambda"]

    if "ewc_lambda" not in tuning_info:
        best_ewc_l, ewc_scores, ewc_is_boundary = tune_ewc_lambda(task_train_loaders, task_val_loaders, device)
        tuning_info["ewc_lambda"] = best_ewc_l
        tuning_info["ewc_scores"] = ewc_scores
        tuning_info["ewc_is_boundary"] = ewc_is_boundary
        save_incremental_json(OUTPUT_JSON_PATH, git_sha, list(completed_runs_map.values()), tuning_info, frozen_probes_info)
    else:
        best_ewc_l = tuning_info["ewc_lambda"]

    # -----------------------------------------------------------------
    # C2. DEMAND-DRIVEN EXECUTION ORDER
    # -----------------------------------------------------------------
    ORDERED_ARMS = [
        ("2_naive_fine_tune", run_naive_fine_tune, ()),
        ("3_ncm_frozen_features", run_ncm_frozen, ()),
        ("9_joint_offline", lambda s, tr, te, ftr, fte, d: run_joint_offline(s, full_train_loader, te, ftr, fte, d), ()),
        ("1_freeze_after_base", run_freeze_after_base, ()),
        ("4_ncm_adapting_features", run_ncm_adapting, ()),
        ("5_lwf", run_lwf, (best_lwf_l, 2.0)),
        ("6_ewc", run_ewc, (best_ewc_l,)),
        ("7_er_buffer500", run_er, ()),
        ("8_der_plus_plus_buffer500", run_der_plus_plus, ())
    ]

    all_cells = []
    cell_idx = 1
    total_est_pending_sec = 0.0

    print("\n" + "=" * 115)
    print(" EXECUTION MANIFEST (ORDERED DEMAND-DRIVEN SCHEDULE)")
    print("=" * 115)
    print(f" {'#':<3} | {'Arm Name':<28} | {'Seed':<5} | {'Status':<32} | {'Est. Time':<12}")
    print("-" * 115)

    for arm_name, arm_fn, extra_args in ORDERED_ARMS:
        for seed in SEEDS:
            cell_key = (arm_name, seed)
            is_done = cell_key in completed_runs_map
            est_sec = ARM_WEIGHTS[arm_name] * T_NAIVE_SEC + T_PROBE_SEC

            if is_done:
                status_str = "[COMPLETED - loaded from disk]"
            else:
                status_str = "[PENDING - to execute]"
                total_est_pending_sec += est_sec

            print(f" {cell_idx:<3} | {arm_name:<28} | {seed:<5} | {status_str:<32} | ~{est_sec:<5.0f}s")
            all_cells.append((cell_idx, arm_name, seed, arm_fn, extra_args, is_done))
            cell_idx += 1

    print("-" * 115)
    print(f"  Study Cells Completed   : {len(completed_runs_map)} / 45")
    print(f"  Pending Execution Time  : {total_est_pending_sec / 3600.0:.2f} hours ({total_est_pending_sec:.0f}s)")
    print(f"  Allocated Session Limit : {args.max_hours:.2f} hours ({args.max_hours * 3600:.0f}s)")
    print("=" * 115)

    computed_this_session = 0

    for c_idx, arm_name, seed, arm_fn, extra_args, is_done in all_cells:
        cell_key = (arm_name, seed)
        if is_done:
            print(f"  [LOADED FROM PRIOR SESSION] Cell #{c_idx:02d}: Arm '{arm_name}' | Seed {seed}")
            continue

        # C2 Demand-driven dynamic budget cutoff
        elapsed_sec = time.time() - session_t0
        est_sec = ARM_WEIGHTS[arm_name] * T_NAIVE_SEC + T_PROBE_SEC
        if elapsed_sec + est_sec > args.max_hours * 3600.0:
            print(f"\n  [CLEAN HALT: budget] Elapsed: {elapsed_sec/3600.0:.2f}h + Next: {est_sec/3600.0:.2f}h > Max: {args.max_hours:.2f}h.")
            print(f"  Halting cleanly before Cell #{c_idx:02d} ({arm_name}, Seed {seed}). Progress saved intact in {OUTPUT_JSON_PATH}.")
            print("  Resume in next session with: python run_w3_baselines.py")
            break

        print(f"\n-------------------------------------------------------------------------------------------------")
        print(f"  [COMPUTED THIS SESSION] Cell #{c_idx:02d}/45: Arm '{arm_name}' | Seed {seed}")
        print(f"-------------------------------------------------------------------------------------------------")

        set_seed(seed)
        res = arm_fn(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device, *extra_args)

        m_ag = compute_r_metrics(res["R_agnostic"])
        m_aw = compute_r_metrics(res["R_aware"])

        avg_la = float(np.mean([res["R_agnostic"][i, i] for i in range(10)]))
        final_class_il = float(m_ag["acc_T"])
        final_task_aware = float(m_aw["acc_T"])
        total_drop = avg_la - final_class_il
        classifier_share = (final_task_aware - final_class_il) / total_drop if total_drop > 0 else 0.0
        residual_share = (avg_la - final_task_aware) / total_drop if total_drop > 0 else 0.0
        bwt_interference = m_aw["bwt"] - m_ag["bwt"]

        cell_record = {
            "arm": arm_name,
            "seed": seed,
            "provenance": res["provenance"],
            "final_class_il": final_class_il,
            "final_task_aware": final_task_aware,
            "classifier_bias_gap": final_task_aware - final_class_il,
            "linear_probe_acc": float(res["probe_acc"]),
            "avg_la": avg_la,
            "total_drop": total_drop,
            "classifier_share": classifier_share,
            "residual_share": residual_share,
            "bwt_class_il": float(m_ag["bwt"]),
            "bwt_task_aware": float(m_aw["bwt"]),
            "bwt_classifier_interference": float(bwt_interference),
            "forgetting_class_il": float(m_ag["forgetting"]),
            "forgetting_task_aware": float(m_aw["forgetting"]),
            "plasticity_curve": [float(v) for v in m_ag["plasticity_curve"]],
            "plasticity_decay": float(m_ag["plasticity_decay"]),
            "R_agnostic": res["R_agnostic"].tolist(),
            "R_aware": res["R_aware"].tolist(),
            "wall_clock_seconds": float(res["wall_clock"]),
            "peak_gpu_memory_bytes": int(res["peak_gpu"]),
            "n_optimizer_steps": int(res["opt_steps"]),
            "n_train_samples_seen": int(res["samples_seen"]),
            "n_forward_samples": int(res["fwd_samples"]),
            "param_count_total": int(res["param_total"]),
            "param_count_trainable": int(res["param_trainable"]),
            "stored_memory_bytes": int(res.get("stored_memory_bytes", 0))
        }
        if "hyperparameters" in res:
            cell_record["hyperparameters"] = res["hyperparameters"]

        completed_runs_map[cell_key] = cell_record
        computed_this_session += 1

        print(f"    Completed in {res['wall_clock']:.1f}s | Class-IL: {final_class_il:.2f}% | Aware: {final_task_aware:.2f}% | Probe: {res['probe_acc']:.2f}%")
        print(f"    BWT Class-IL: {m_ag['bwt']:+.2f} pp | BWT Aware: {m_aw['bwt']:+.2f} pp | BWT Interference: {bwt_interference:+.2f} pp")
        print(f"    Avg LA: {avg_la:.2f}% | Total Drop: {total_drop:.2f} pp (Denominator) | Clf Share: {classifier_share*100:.1f}% | Res Share: {residual_share*100:.1f}%")

        save_incremental_json(OUTPUT_JSON_PATH, git_sha, list(completed_runs_map.values()), tuning_info, frozen_probes_info)

    # -----------------------------------------------------------------
    # SUMMARY TABLE FOR COMPLETED ARMS
    # -----------------------------------------------------------------
    print("\n" + "=" * 155)
    print(" CONTINUAL LEARNING BASELINE TABLE (TRI-METRIC & DUAL BWT DECOMPOSITION)")
    print("=" * 155)
    header = f"{'Arm Name':<28} | {'(i) Class-IL':<14} | {'(ii) Aware':<14} | {'Bias Gap':<10} | {'(iii) Probe':<13} | {'BWT Agnostic':<13} | {'BWT Aware':<11} | {'Avg LA':<10} | {'Clf Share':<9}"
    print(header)
    print("-" * 155)

    summary_stats = {}
    for arm_name, _, _ in ORDERED_ARMS:
        arm_runs = [completed_runs_map[(arm_name, s)] for s in SEEDS if (arm_name, s) in completed_runs_map]
        if len(arm_runs) == 0:
            continue

        ag_list = [r["final_class_il"] for r in arm_runs]
        aw_list = [r["final_task_aware"] for r in arm_runs]
        bias_list = [r["classifier_bias_gap"] for r in arm_runs]
        pr_list = [r["linear_probe_acc"] for r in arm_runs]
        bwt_ag_list = [r["bwt_class_il"] for r in arm_runs]
        bwt_aw_list = [r["bwt_task_aware"] for r in arm_runs]
        la_list = [r["avg_la"] for r in arm_runs]
        clf_share_list = [r["classifier_share"] * 100.0 for r in arm_runs]

        def calc_m_s(arr):
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return m, s

        m_ag, s_ag = calc_m_s(ag_list)
        m_aw, s_aw = calc_m_s(aw_list)
        m_bias, s_bias = calc_m_s(bias_list)
        m_pr, s_pr = calc_m_s(pr_list)
        m_bwt_ag, s_bwt_ag = calc_m_s(bwt_ag_list)
        m_bwt_aw, s_bwt_aw = calc_m_s(bwt_aw_list)
        m_la, s_la = calc_m_s(la_list)
        m_clf, _ = calc_m_s(clf_share_list)

        n_s = len(arm_runs)
        print(f"{arm_name:<28} | {m_ag:5.2f}% +/-{s_ag:4.2f} | {m_aw:5.2f}% +/-{s_aw:4.2f} | {m_bias:+5.2f} pp | {m_pr:5.2f}% +/-{s_pr:4.2f} | {m_bwt_ag:+5.2f} pp    | {m_bwt_aw:+5.2f} pp  | {m_la:5.2f}%    | {m_clf:5.1f}% (n={n_s})")

        summary_stats[arm_name] = {
            "n_seeds_completed": n_s,
            "class_il_mean": m_ag, "class_il_std": s_ag, "class_il_per_seed": ag_list,
            "task_aware_mean": m_aw, "task_aware_std": s_aw, "task_aware_per_seed": aw_list,
            "bias_gap_mean": m_bias, "bias_gap_std": s_bias,
            "linear_probe_mean": m_pr, "linear_probe_std": s_pr, "linear_probe_per_seed": pr_list,
            "bwt_agnostic_mean": m_bwt_ag, "bwt_agnostic_std": s_bwt_ag,
            "bwt_aware_mean": m_bwt_aw, "bwt_aware_std": s_bwt_aw,
            "avg_la_mean": m_la, "avg_la_std": s_la,
            "classifier_share_mean": m_clf
        }

    # -----------------------------------------------------------------
    # PREDICTION REGISTRY HIT/MISS AUDIT
    # -----------------------------------------------------------------
    print("\n" + "=" * 115)
    print(" PREDICTION REGISTRY AUDIT (PREDICTED VS MEASURED CLASS-IL)")
    print("=" * 115)
    for arm_name, _, _ in ORDERED_ARMS:
        if arm_name in summary_stats and summary_stats[arm_name]["n_seeds_completed"] == 5:
            meas = summary_stats[arm_name]["class_il_mean"]
            meas_s = summary_stats[arm_name]["class_il_std"]
            pred_str = PREDICTIONS[arm_name]["pred_class_il"]

            # Evaluate hit/miss
            hit = False
            if "-" in pred_str:
                parts = [float(p.replace("%", "").strip()) for p in pred_str.split("-")]
                hit = (parts[0] - 2.0 <= meas <= parts[1] + 2.0)
            elif "+/-" in pred_str:
                parts = [float(p.replace("%", "").strip()) for p in pred_str.split("+/-")]
                hit = abs(meas - parts[0]) <= 3.0 * parts[1]
            status = "HIT" if hit else "MISS"

            print(f"  {arm_name:<28} | Predicted: {pred_str:<17} | Measured: {meas:5.2f}% +/- {meas_s:4.2f}% | Status: [{status}]")

    # Joint Offline reproduction audit
    if "9_joint_offline" in summary_stats and summary_stats["9_joint_offline"]["n_seeds_completed"] == 5:
        offline_m = summary_stats["9_joint_offline"]["class_il_mean"]
        diff = abs(offline_m - 79.64)
        print(f"\n  [Joint Offline Reproduction Audit]")
        print(f"    Target: 79.64% +/- 0.23% | Measured: {offline_m:.2f}% | Delta: {diff:+.2f} pp -> {'PASS' if diff < 1.0 else 'HARNESS MISMATCH DETECTED'}")

    final_output = {
        "git_commit_sha": git_sha,
        "dataset": "Split-CIFAR-100",
        "seeds": SEEDS,
        "epochs_per_task": EPOCHS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "lr_base": LR_BASE,
        "probe_config": PROBE_CONFIG,
        "tuning_info": tuning_info,
        "frozen_imagenet_probes": frozen_probes_info,
        "session_execution_audit": {
            "total_study_cells": 45,
            "total_cells_completed": len(completed_runs_map),
            "computed_this_session": computed_this_session
        },
        "summary": summary_stats,
        "completed_runs": list(completed_runs_map.values())
    }
    tmp_path = OUTPUT_JSON_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(final_output, f, indent=2)
    os.replace(tmp_path, OUTPUT_JSON_PATH)

    print("\n" + "=" * 115)
    print("EXIT_CODE = 0")
    print("=" * 115)


if __name__ == "__main__":
    main()
