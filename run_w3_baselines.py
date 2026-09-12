"""
run_w3_baselines.py
===================
Directive W3 -- Part 2: The Baseline Table (Split-CIFAR-100 Continual Learning Benchmark).

Evaluates 9 Continual Learning Arms across 5 Seeds: SEEDS = [42, 43, 44, 45, 46].
Split-CIFAR-100: 10 tasks x 10 classes, input resolution 112x112, ResNet-18 ImageNet stem.

ARMS (9):
  1. FREEZE-AFTER-BASE   (Standing Rule 1 permanent control arm)
  2. naive fine-tune     (Sequential SGD)
  3. NCM on frozen features (Zero-gradient metric baseline)
  4. NCM on adapting features (Centroids on adapting backbone)
  5. LwF                 (Learning without Forgetting, logit distillation)
  6. EWC                 (Elastic Weight Consolidation, diagonal Fisher penalty)
  7. ER (buffer 500)     (Experience Replay with reservoir buffer)
  8. DER++ (buffer 500)  (Dark Experience Replay++ with logit matching)
  9. joint offline       (Upper bound, re-measuring w2e setup to confirm 79.64% reproduction)

TRI-METRIC DECOMPOSED EVALUATION FOR EVERY ARM:
  (i)   Task-Agnostic (Class-IL) accuracy over all 100 classes
  (ii)  Task-Aware accuracy with task-ID gating
  (iii) Linear probe over all 100 classes trained on frozen final representation

CONTINUAL LEARNING METRICS VIA eval_core.compute_r_metrics:
  ACC_T, BWT, Forgetting, Plasticity Curve, Plasticity Decay, Average Learning Accuracy (Avg LA).
"""

import os
import sys
import copy
import time
import json
import random
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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ARCHIVE_PATH = os.path.join(DATA_DIR, "cifar-100-python.tar.gz")
CLASS_ORDER_PATH = os.path.join(REPO_ROOT, "class_order_split_cifar100.json")
OUTPUT_JSON_PATH = os.path.join(REPO_ROOT, "w3_baselines.json")

SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20  # 20 epochs x 32 steps = 640 steps per task; 6,400 steps over 10 tasks
LR_BASE = 0.005
WEIGHT_DECAY = 5e-4
BUFFER_CAPACITY = 500


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


class ReservoirBuffer:
    """Reservoir Replay Buffer for ER and DER++."""
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = []  # list of (x, y, logits, task_id)
        self.total_seen = 0

    def add(self, x, y, logits=None, task_id=0):
        self.total_seen += 1
        entry = (x.detach().cpu(), int(y.item()), logits.detach().cpu() if logits is not None else None, task_id)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            if random.random() < self.capacity / self.total_seen:
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = entry

    def sample(self, n, device):
        if not self.buffer:
            return None, None, None
        n = min(n, len(self.buffer))
        batch = random.sample(self.buffer, n)
        bx = torch.stack([b[0] for b in batch]).to(device)
        by = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        if batch[0][2] is not None:
            bz = torch.stack([b[2] for b in batch]).to(device)
        else:
            bz = None
        return bx, by, bz

    def __len__(self):
        return len(self.buffer)


def evaluate_task_r(model, test_loaders, seen_tasks, device):
    """
    Evaluates both Task-Agnostic (Class-IL) and Task-Aware accuracies for all seen tasks.
    """
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


def train_frozen_linear_probe(backbone, full_tr_loader, full_te_loader, device, epochs=30):
    """(iii) Linear probe on frozen final representation over all 100 classes."""
    backbone.eval()
    all_tr_feats, all_tr_y = [], []
    all_te_feats, all_te_y = [], []

    with torch.no_grad():
        for bx, by in full_tr_loader:
            bx = bx.to(device)
            all_tr_feats.append(backbone.extract_features(bx).cpu())
            all_tr_y.append(by)
        for bx, by in full_te_loader:
            bx = bx.to(device)
            all_te_feats.append(backbone.extract_features(bx).cpu())
            all_te_y.append(by)

    tr_x = torch.cat(all_tr_feats, dim=0)
    tr_y = torch.cat(all_tr_y, dim=0)
    te_x = torch.cat(all_te_feats, dim=0)
    te_y = torch.cat(all_te_y, dim=0)

    tr_ds = TensorDataset(tr_x, tr_y)
    te_ds = TensorDataset(te_x, te_y)
    ld_tr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    ld_te = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    probe = nn.Linear(512, 100).to(device)
    opt = optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    crit = nn.CrossEntropyLoss()

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

    probe.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for bx, by in ld_te:
            bx, by = bx.to(device), by.to(device)
            out = probe(bx)
            preds = out.argmax(dim=-1)
            cor += (preds == by).sum().item()
            tot += by.size(0)

    return (cor / tot) * 100.0


# =====================================================================
# INDIVIDUAL ARM EXECUTIONS
# =====================================================================

def run_freeze_after_base(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 1: FREEZE-AFTER-BASE (Standing Rule 1)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    # Base Phase: Task 0
    t0_loader, _ = task_train_loaders[0]
    for ep in range(EPOCHS_PER_TASK):
        model.train()
        for bx, by in t0_loader:
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

    # Freeze completely after Task 0
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Task boundary evaluations
    for t in range(10):
        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "freeze_after_base",
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
        "param_trainable": 0  # frozen after base
    }


def run_naive_fine_tune(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 2: Naive Fine-Tune."""
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "naive_fine_tune",
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
        "final_model": model  # can be passed to NCM on adapting features
    }


def run_ncm_frozen(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 3: NCM on Frozen Features."""
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

    # Stored centroids: dict from class_id to 512-dim tensor
    centroids = {}

    for t in range(10):
        t_loader, t_classes = task_train_loaders[t]
        # Extract features for task t
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

        # Evaluate on all seen tasks
        seen = list(range(t + 1))
        seen_classes = [c for s in seen for c in task_train_loaders[s][1]]
        seen_cen_matrix = torch.stack([centroids[c] for c in seen_classes], dim=0)  # (N_seen, 512)
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

                    # (i) Task-agnostic: nearest centroid over all seen classes
                    sims_ag = torch.matmul(feats, seen_cen_matrix.T)
                    pred_idx = sims_ag.argmax(dim=-1)
                    preds_ag = seen_cen_labels[pred_idx]
                    cor_ag += (preds_ag == by).sum().item()

                    # (ii) Task-aware: nearest centroid within task j
                    sims_aw = torch.matmul(feats, j_cen_matrix.T)
                    pred_idx_aw = sims_aw.argmax(dim=-1)
                    preds_aw = j_classes_t[pred_idx_aw]
                    cor_aw += (preds_aw == by).sum().item()

                    tot += by.size(0)

            R_agnostic[t, j] = (cor_ag / tot) * 100.0
            R_aware[t, j] = (cor_aw / tot) * 100.0

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "ncm_frozen_features",
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
        "param_trainable": 0
    }


def run_ncm_adapting(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 4: NCM on Adapting Features."""
    # Run sequential fine-tuning and calculate centroids on the newly adapted features at each step
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

        # Compute NCM on the current adapting model for all seen tasks
        seen = list(range(t + 1))
        model.eval()
        centroids = {}
        for s in seen:
            s_loader, s_classes = task_train_loaders[s]
            s_feats = []
            s_targets = []
            with torch.no_grad():
                for bx, by in s_loader:
                    bx = bx.to(device)
                    s_feats.append(model.extract_features(bx))
                    s_targets.append(by.to(device))
                    fwd_samples += bx.size(0)
            s_feats = torch.cat(s_feats, dim=0)
            s_targets = torch.cat(s_targets, dim=0)
            for c in s_classes:
                mask = (s_targets == c)
                if mask.sum() > 0:
                    centroids[c] = F.normalize(s_feats[mask].mean(dim=0), dim=-1)

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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "ncm_adapting_features",
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
        "param_trainable": p_total
    }


def run_lwf(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, lambda_distill=1.0, tau=2.0):
    """Arm 5: Learning without Forgetting (LwF)."""
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
        t_loader, t_classes = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        seen_classes_prev = [c for s in range(t) for c in task_train_loaders[s][1]]

        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)

                if prev_model is not None and len(seen_classes_prev) > 0:
                    with torch.no_grad():
                        prev_logits, _ = prev_model(bx)
                        fwd_samples += bx.size(0)

                    prev_idx = torch.tensor(seen_classes_prev, device=device)
                    cur_soft = F.log_softmax(logits[:, prev_idx] / tau, dim=1)
                    old_soft = F.softmax(prev_logits[:, prev_idx] / tau, dim=1)
                    kd_loss = F.kl_div(cur_soft, old_soft, reduction="batchmean") * (tau ** 2)
                    loss += lambda_distill * kd_loss

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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "lwf",
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
        "param_trainable": p_total
    }


def run_ewc(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, lambda_ewc=1000.0):
    """Arm 6: Elastic Weight Consolidation (EWC)."""
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
                loss = crit(logits, by)

                if len(fisher_dict) > 0:
                    ewc_loss = 0.0
                    for name, param in model.named_parameters():
                        if name in fisher_dict:
                            f = fisher_dict[name]
                            p_old = optpar_dict[name]
                            ewc_loss += (f * (param - p_old).pow(2)).sum()
                    loss += (lambda_ewc / 2.0) * ewc_loss

                loss.backward()
                opt.step()
                opt_steps += 1
                samples_seen += bx.size(0)
                fwd_samples += bx.size(0)
            sched.step()

        # Compute Fisher diagonal on task t
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "ewc",
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
        "param_trainable": p_total
    }


def run_er(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 7: Experience Replay (ER, buffer capacity 500)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    buffer = ReservoirBuffer(capacity=BUFFER_CAPACITY)

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
                    buf_x, buf_y, _ = buffer.sample(min(32, len(buffer)), device)
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
                    buffer.add(bx[i], by[i], logits=None, task_id=t)

        seen = list(range(t + 1))
        acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_ag[j]
            R_aware[t, j] = acc_aw[j]
            fwd_samples += 1000

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "er_buffer500",
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
        "param_trainable": p_total
    }


def run_der_plus_plus(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, alpha=0.5, beta=0.5):
    """Arm 8: Dark Experience Replay++ (DER++, buffer capacity 500)."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    buffer = ReservoirBuffer(capacity=BUFFER_CAPACITY)

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
                    buf_x, buf_y, buf_z = buffer.sample(min(32, len(buffer)), device)
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "der_plus_plus_buffer500",
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
        "param_trainable": p_total
    }


def run_joint_offline(seed, full_train_loader, task_test_loaders, full_tr_loader, full_te_loader, device, epochs=30):
    """Arm 9: Joint Offline Full Finetune (Upper Bound, replicating w2e Arm C)."""
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

    # Evaluated at step 9 across all tasks
    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))
    acc_ag, acc_aw = evaluate_task_r(model, task_test_loaders, list(range(10)), device)
    for j in range(10):
        R_agnostic[9, j] = acc_ag[j]
        R_aware[9, j] = acc_aw[j]
        fwd_samples += 1000

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "joint_offline",
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
        "param_trainable": p_total
    }


def main():
    print("=" * 95)
    print(" DIRECTIVE W3 -- PART 2: CONTINUAL LEARNING BASELINE TABLE (9 ARMS x 5 SEEDS)")
    print("=" * 95)

    git_sha = check_provenance()
    print(f"  Git Commit SHA     : {git_sha}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Platform Device    : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Accelerator    : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(ARCHIVE_PATH):
        print(f"  CIFAR-100 archive not found at {ARCHIVE_PATH}. Downloading...")
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

    with open(CLASS_ORDER_PATH, "r") as f:
        class_order_info = json.load(f)
    blocks = class_order_info["blocks"]
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
    task_test_loaders = {}

    for t_idx, classes in enumerate(blocks):
        t_tr_local = [train_idx[i] for i, c in enumerate(targets_tr) if c in classes]
        t_te_local = [i for i, c in enumerate(targets_te) if c in classes]

        task_train_loaders[t_idx] = (DataLoader(Subset(ds_tr, t_tr_local), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker), classes)
        task_test_loaders[t_idx] = (DataLoader(Subset(ds_te, t_te_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker), classes)

    full_train_loader = DataLoader(Subset(ds_tr, train_idx), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
    full_tr_probe_loader = DataLoader(Subset(ds_ev, train_idx), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)
    full_te_probe_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)

    # Arm Dispatch
    ARM_NAMES = [
        "1_freeze_after_base",
        "2_naive_fine_tune",
        "3_ncm_frozen_features",
        "4_ncm_adapting_features",
        "5_lwf",
        "6_ewc",
        "7_er_buffer500",
        "8_der_plus_plus_buffer500",
        "9_joint_offline"
    ]

    all_results = defaultdict(list)

    for seed in SEEDS:
        print(f"\n=========================================================================================================")
        print(f" EXECUTING SEED {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"=========================================================================================================")

        # Arm 1: FREEZE-AFTER-BASE
        print(f"\n  [Seed {seed}] Running Arm 1: FREEZE-AFTER-BASE...")
        res1 = run_freeze_after_base(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["1_freeze_after_base"].append(res1)
        r_met1 = compute_r_metrics(res1["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met1['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res1['R_aware'][9, :]):5.2f}% | Probe: {res1['probe_acc']:5.2f}% | BWT: {r_met1['bwt']:+5.2f} pp")

        # Arm 2: Naive Fine-Tune
        print(f"\n  [Seed {seed}] Running Arm 2: Naive Fine-Tune...")
        res2 = run_naive_fine_tune(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["2_naive_fine_tune"].append(res2)
        r_met2 = compute_r_metrics(res2["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met2['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res2['R_aware'][9, :]):5.2f}% | Probe: {res2['probe_acc']:5.2f}% | BWT: {r_met2['bwt']:+5.2f} pp")

        # Arm 3: NCM on Frozen Features
        print(f"\n  [Seed {seed}] Running Arm 3: NCM on Frozen Features...")
        res3 = run_ncm_frozen(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["3_ncm_frozen_features"].append(res3)
        r_met3 = compute_r_metrics(res3["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met3['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res3['R_aware'][9, :]):5.2f}% | Probe: {res3['probe_acc']:5.2f}% | BWT: {r_met3['bwt']:+5.2f} pp")

        # Arm 4: NCM on Adapting Features
        print(f"\n  [Seed {seed}] Running Arm 4: NCM on Adapting Features...")
        res4 = run_ncm_adapting(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["4_ncm_adapting_features"].append(res4)
        r_met4 = compute_r_metrics(res4["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met4['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res4['R_aware'][9, :]):5.2f}% | Probe: {res4['probe_acc']:5.2f}% | BWT: {r_met4['bwt']:+5.2f} pp")

        # Arm 5: LwF
        print(f"\n  [Seed {seed}] Running Arm 5: LwF...")
        res5 = run_lwf(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["5_lwf"].append(res5)
        r_met5 = compute_r_metrics(res5["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met5['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res5['R_aware'][9, :]):5.2f}% | Probe: {res5['probe_acc']:5.2f}% | BWT: {r_met5['bwt']:+5.2f} pp")

        # Arm 6: EWC
        print(f"\n  [Seed {seed}] Running Arm 6: EWC...")
        res6 = run_ewc(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["6_ewc"].append(res6)
        r_met6 = compute_r_metrics(res6["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met6['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res6['R_aware'][9, :]):5.2f}% | Probe: {res6['probe_acc']:5.2f}% | BWT: {r_met6['bwt']:+5.2f} pp")

        # Arm 7: ER (buffer 500)
        print(f"\n  [Seed {seed}] Running Arm 7: ER (buffer 500)...")
        res7 = run_er(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["7_er_buffer500"].append(res7)
        r_met7 = compute_r_metrics(res7["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met7['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res7['R_aware'][9, :]):5.2f}% | Probe: {res7['probe_acc']:5.2f}% | BWT: {r_met7['bwt']:+5.2f} pp")

        # Arm 8: DER++ (buffer 500)
        print(f"\n  [Seed {seed}] Running Arm 8: DER++ (buffer 500)...")
        res8 = run_der_plus_plus(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["8_der_plus_plus_buffer500"].append(res8)
        r_met8 = compute_r_metrics(res8["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met8['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res8['R_aware'][9, :]):5.2f}% | Probe: {res8['probe_acc']:5.2f}% | BWT: {r_met8['bwt']:+5.2f} pp")

        # Arm 9: Joint Offline (reproducing w2e Arm C setup)
        print(f"\n  [Seed {seed}] Running Arm 9: Joint Offline...")
        res9 = run_joint_offline(seed, full_train_loader, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device)
        all_results["9_joint_offline"].append(res9)
        r_met9 = compute_r_metrics(res9["R_agnostic"])
        print(f"    -> Class-IL ACC_T: {r_met9['acc_T']:5.2f}% | Aware ACC_T: {np.mean(res9['R_aware'][9, :]):5.2f}% | Probe: {res9['probe_acc']:5.2f}%")

    # =====================================================================
    # COMPUTE SUMMARY STATISTICS & EMIT BASELINE TABLE
    # =====================================================================
    print("\n" + "=" * 125)
    print(" CONTINUAL LEARNING BASELINE TABLE (SPLIT-CIFAR-100, N = 5 SEEDS)")
    print("=" * 125)
    header = f"{'Arm Name':<28} | {'(i) Class-IL ACC_T':<18} | {'(ii) Aware ACC_T':<18} | {'Bias Gap (ii-i)':<15} | {'(iii) Probe ACC':<16} | {'BWT (pp)':<14} | {'Forgetting':<12}"
    print(header)
    print("-" * 125)

    summary_json = {}

    for arm in ARM_NAMES:
        arm_runs = all_results[arm]
        acc_ag_list = []
        acc_aw_list = []
        bias_list = []
        probe_list = []
        bwt_list = []
        fgt_list = []
        la_list = []

        for r in arm_runs:
            r_met = compute_r_metrics(r["R_agnostic"])
            ag = r_met["acc_T"]
            aw = float(np.mean(r["R_aware"][9, :]))
            acc_ag_list.append(ag)
            acc_aw_list.append(aw)
            bias_list.append(aw - ag)
            probe_list.append(r["probe_acc"])
            bwt_list.append(r_met["bwt"])
            fgt_list.append(r_met["forgetting"])
            la_list.append(float(np.mean([r["R_agnostic"][i, i] for i in range(10)])))

        def m_std(vals):
            m = np.mean(vals)
            s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            return m, s

        m_ag, s_ag = m_std(acc_ag_list)
        m_aw, s_aw = m_std(acc_aw_list)
        m_bias, s_bias = m_std(bias_list)
        m_pr, s_pr = m_std(probe_list)
        m_bwt, s_bwt = m_std(bwt_list)
        m_fgt, s_fgt = m_std(fgt_list)
        m_la, s_la = m_std(la_list)

        print(f"{arm:<28} | {m_ag:5.2f}% +/- {s_ag:4.2f}% | {m_aw:5.2f}% +/- {s_aw:4.2f}% | {m_bias:+5.2f} +/- {s_bias:4.2f} | {m_pr:5.2f}% +/- {s_pr:4.2f}% | {m_bwt:+5.2f} +/- {s_bwt:4.2f} | {m_fgt:5.2f}% +/- {s_fgt:4.2f}%")

        summary_json[arm] = {
            "class_il_acc_T_mean": float(m_ag),
            "class_il_acc_T_std": float(s_ag),
            "class_il_per_seed": acc_ag_list,
            "task_aware_acc_T_mean": float(m_aw),
            "task_aware_acc_T_std": float(s_aw),
            "task_aware_per_seed": acc_aw_list,
            "classifier_bias_gap_mean": float(m_bias),
            "classifier_bias_gap_std": float(s_bias),
            "linear_probe_mean": float(m_pr),
            "linear_probe_std": float(s_pr),
            "linear_probe_per_seed": probe_list,
            "bwt_mean": float(m_bwt),
            "bwt_std": float(s_bwt),
            "forgetting_mean": float(m_fgt),
            "forgetting_std": float(s_fgt),
            "avg_la_mean": float(m_la),
            "avg_la_std": float(s_la),
            "runs": [
                {
                    "seed": r["seed"],
                    "R_agnostic": r["R_agnostic"].tolist(),
                    "R_aware": r["R_aware"].tolist(),
                    "linear_probe_acc": float(r["probe_acc"]),
                    "wall_clock_seconds": float(r["wall_clock"]),
                    "peak_gpu_memory_bytes": int(r["peak_gpu"]),
                    "n_optimizer_steps": int(r["opt_steps"]),
                    "n_train_samples_seen": int(r["samples_seen"]),
                    "n_forward_samples": int(r["fwd_samples"]),
                    "param_count_total": int(r["param_total"]),
                    "param_count_trainable": int(r["param_trainable"])
                }
                for r in arm_runs
            ]
        }

    # Audit checks
    offline_reproduced = abs(summary_json["9_joint_offline"]["class_il_acc_T_mean"] - 79.64) < 2.0
    print("\n  Audit Checks:")
    print(f"    Joint Offline reproduces w2e baseline (79.64% +/- 0.23%): {'PASS' if offline_reproduced else 'WARNING'} (Observed: {summary_json['9_joint_offline']['class_il_acc_T_mean']:.2f}%)")

    # Write output JSON
    final_output = {
        "git_commit_sha": git_sha,
        "dataset": "Split-CIFAR-100",
        "seeds": SEEDS,
        "epochs_per_task": EPOCHS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "lr_base": LR_BASE,
        "summary": summary_json
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"  Emitted results to {OUTPUT_JSON_PATH}")

    print("\n" + "=" * 95)
    print("EXIT_CODE = 0")
    print("=" * 95)


if __name__ == "__main__":
    main()
