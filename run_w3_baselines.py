"""
run_w3_baselines.py
===================
Directive W3 -- Part 2: The Baseline Table (Split-CIFAR-100 Continual Learning Benchmark).

Evaluates 9 Continual Learning Arms across 5 Seeds: SEEDS = [42, 43, 44, 45, 46].
Split-CIFAR-100: 10 tasks x 10 classes, input resolution 112x112, ResNet-18 ImageNet stem.

EXPLICIT PROVENANCE DECLARATIONS:
  - eval_core.compute_r_metrics      : REUSED (lower-triangular R-matrix continual learning metrics)
  - replay_buffer.DERBuffer          : REUSED (tensor buffer storing (x, y, logits, task_id))
  - 1_freeze_after_base              : PORTED (standing control logic from run_phase4_lever2_replay.py, ResNet pipeline ported from run_w2e_gap_closed.py)
  - 2_naive_fine_tune                : PORTED (sequential SGD logic from run_partB_naive_reproduction.py, ResNet pipeline ported from run_w2e_gap_closed.py)
  - 3_ncm_frozen_features            : PORTED (NCM logic from run_aa11_adaptation_gap_pretrained.py, ResNet pipeline ported from run_w2e_gap_closed.py)
  - 4_ncm_adapting_features          : PORTED (NCM centroids on adapting backbone, ported from eval_core.py & run_w2e_gap_closed.py)
  - 5_lwf                            : NEW (fresh implementation: CE + lambda*T^2*KL on old classes; tuned on validation split)
  - 6_ewc                            : NEW (fresh implementation: CE + (lambda/2)*sum F_i*(theta-theta*)^2 with empirical Fisher; tuned on validation split)
  - 7_er_buffer500                   : PORTED (replay logic from run_phase4_lever2_replay.py, DERBuffer reused for visual tensors without logits)
  - 8_der_plus_plus_buffer500        : PORTED (DER++ logic from run_phase5_der_plus_plus_class_il.py, DERBuffer reused for visual tensors + logits)
  - 9_joint_offline                  : PORTED (joint offline training from run_w2e_gap_closed.py Arm C; confirms 79.64% reproduction)

TRI-METRIC DECOMPOSED EVALUATION FOR EVERY ARM:
  (i)   Task-Agnostic (Class-IL) accuracy over all 100 classes
  (ii)  Task-Aware accuracy with task-ID gating
  (iii) Linear probe over all 100 classes trained on frozen final representation

RESUMABLE / INCREMENTAL LOGGING:
  Saves w3_baselines.json after every (arm, seed) execution. On restart, skips all completed runs.
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


def train_frozen_linear_probe(backbone, full_tr_loader, full_te_loader, device, epochs=30):
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
# HYPERPARAMETER TUNING FOR NEW ARMS ON VALIDATION SPLIT (SEED 42)
# =====================================================================

def tune_lwf_lambda(task_train_loaders, task_val_loaders, device):
    """
    NEW Arm: LwF
    Tunes lambda on validation split over candidate grid: [0.1, 0.5, 1.0, 2.0, 5.0].
    Evaluates Tasks 0 and 1.
    """
    print("\n  [Hyperparameter Tuning: LwF on Validation Split (Seed 42)]")
    grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    scores = {}

    for cand_l in grid:
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

        prev_model = copy.deepcopy(model)
        prev_model.eval()

        # Task 1 training with LwF
        t1_loader, _ = task_train_loaders[1]
        seen_classes_prev = task_train_loaders[0][1]
        prev_idx = torch.tensor(seen_classes_prev, device=device)
        tau = 2.0

        for ep in range(10):
            model.train()
            for bx, by in t1_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)

                with torch.no_grad():
                    prev_logits, _ = prev_model(bx)

                cur_soft = F.log_softmax(logits[:, prev_idx] / tau, dim=1)
                old_soft = F.softmax(prev_logits[:, prev_idx] / tau, dim=1)
                kd_loss = F.kl_div(cur_soft, old_soft, reduction="batchmean") * (tau ** 2)
                loss += cand_l * kd_loss

                loss.backward()
                opt.step()

        # Validation evaluation on seen tasks (0 and 1)
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for t_idx in [0, 1]:
                v_loader, _ = task_val_loaders[t_idx]
                for bx, by in v_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits, _ = model(bx)
                    preds = logits.argmax(dim=-1)
                    cor += (preds == by).sum().item()
                    tot += by.size(0)
        v_acc = (cor / tot) * 100.0
        scores[cand_l] = v_acc
        print(f"    Candidate lambda = {cand_l:5.2f} -> Validation ACC: {v_acc:5.2f}%")

    best_l = max(scores, key=scores.get)
    is_boundary = (best_l == grid[0] or best_l == grid[-1])
    print(f"  Selected LwF Optimal lambda*: {best_l} (Val ACC = {scores[best_l]:.2f}%) | Position: {'BOUNDARY' if is_boundary else 'INTERIOR'} of {grid}")
    return best_l, scores, is_boundary


def tune_ewc_lambda(task_train_loaders, task_val_loaders, device):
    """
    NEW Arm: EWC
    Tunes lambda on validation split over candidate grid: [100.0, 500.0, 1000.0, 5000.0, 10000.0].
    Evaluates Tasks 0 and 1.
    """
    print("\n  [Hyperparameter Tuning: EWC on Validation Split (Seed 42)]")
    grid = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
    scores = {}

    for cand_l in grid:
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

        # Compute Fisher diagonal on Task 0 (using 4,000 samples)
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

        # Validation evaluation on seen tasks (0 and 1)
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for t_idx in [0, 1]:
                v_loader, _ = task_val_loaders[t_idx]
                for bx, by in v_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits, _ = model(bx)
                    preds = logits.argmax(dim=-1)
                    cor += (preds == by).sum().item()
                    tot += by.size(0)
        v_acc = (cor / tot) * 100.0
        scores[cand_l] = v_acc
        print(f"    Candidate lambda = {cand_l:7.1f} -> Validation ACC: {v_acc:5.2f}%")

    best_l = max(scores, key=scores.get)
    is_boundary = (best_l == grid[0] or best_l == grid[-1])
    print(f"  Selected EWC Optimal lambda*: {best_l} (Val ACC = {scores[best_l]:.2f}%) | Position: {'BOUNDARY' if is_boundary else 'INTERIOR'} of {grid}")
    return best_l, scores, is_boundary


# =====================================================================
# INDIVIDUAL ARM EXECUTIONS
# =====================================================================

def run_freeze_after_base(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 1: FREEZE-AFTER-BASE (PORTED from run_phase4_lever2_replay.py, ResNet pipeline ported from run_w2e_gap_closed.py)."""
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

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

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
        "arm": "1_freeze_after_base",
        "provenance": "PORTED (standing control logic from run_phase4_lever2_replay.py, ResNet pipeline from run_w2e_gap_closed.py)",
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
        "param_trainable": 0
    }


def run_naive_fine_tune(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 2: Naive Fine-Tune (PORTED from run_partB_naive_reproduction.py, ResNet pipeline ported from run_w2e_gap_closed.py)."""
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
        "param_trainable": p_total
    }


def run_ncm_frozen(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 3: NCM on Frozen Features (PORTED from run_aa11_adaptation_gap_pretrained.py, ResNet pipeline from run_w2e_gap_closed.py)."""
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

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
        "param_trainable": 0
    }


def run_ncm_adapting(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device):
    """Arm 4: NCM on Adapting Features (PORTED from eval_core.py:eval_ncm and run_w2e_gap_closed.py)."""
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
        "arm": "4_ncm_adapting_features",
        "provenance": "PORTED (adapting backbone from sequential SGD, centroid computation from eval_core.py:eval_ncm)",
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


def run_lwf(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, lwf_lambda=1.0, tau=2.0):
    """
    Arm 5: Learning without Forgetting (LwF).
    PROVENANCE: NEW.
    Objective: L = CE(new data) + lambda * T^2 * KL(softmax(prev_logits/T) || softmax(cur_logits/T)) over old classes.
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "5_lwf",
        "provenance": "NEW (fresh implementation of LwF distillation over old classes on new inputs)",
        "seed": seed,
        "hyperparameters": {"lambda": lwf_lambda, "T": tau},
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


def run_ewc(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, ewc_lambda=1000.0):
    """
    Arm 6: Elastic Weight Consolidation (EWC).
    PROVENANCE: NEW.
    Objective: L = CE + (lambda/2) * sum_i F_i (theta_i - theta_i*)^2.
    Fisher estimate uses all 4,000 samples per completed task.
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "6_ewc",
        "provenance": "NEW (fresh implementation of diagonal Fisher information & quadratic parameter penalty)",
        "seed": seed,
        "hyperparameters": {"lambda": ewc_lambda, "fisher_sample_count": 4000},
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
    """Arm 7: Experience Replay (PORTED from run_phase4_lever2_replay.py, DERBuffer reused for visual tensors without logits)."""
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

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
        "param_trainable": p_total
    }


def run_der_plus_plus(seed, task_train_loaders, task_test_loaders, full_tr_loader, full_te_loader, device, alpha=0.5, beta=0.5):
    """Arm 8: Dark Experience Replay++ (PORTED from run_phase5_der_plus_plus_class_il.py, DERBuffer REUSED)."""
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

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

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
        "param_trainable": p_total
    }


def run_joint_offline(seed, full_train_loader, task_test_loaders, full_tr_loader, full_te_loader, device, epochs=30):
    """Arm 9: Joint Offline Full Finetune (PORTED from run_w2e_gap_closed.py Arm C; confirms 79.64% reproduction)."""
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
        fwd_samples += 1000

    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device)
    fwd_samples += 50000

    wall = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    p_total = sum(p.numel() for p in model.parameters())

    return {
        "arm": "9_joint_offline",
        "provenance": "PORTED (joint offline training from run_w2e_gap_closed.py Arm C)",
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


# =====================================================================
# PERSISTENCE & RESUMABILITY HELPER
# =====================================================================

def save_incremental_json(output_path, git_sha, all_runs_records, tuning_info):
    output_data = {
        "git_commit_sha": git_sha,
        "dataset": "Split-CIFAR-100",
        "provenance_summary": {
            "eval_core.compute_r_metrics": "REUSED",
            "replay_buffer.DERBuffer": "REUSED",
            "1_freeze_after_base": "PORTED",
            "2_naive_fine_tune": "PORTED",
            "3_ncm_frozen_features": "PORTED",
            "4_ncm_adapting_features": "PORTED",
            "5_lwf": "NEW",
            "6_ewc": "NEW",
            "7_er_buffer500": "PORTED",
            "8_der_plus_plus_buffer500": "PORTED",
            "9_joint_offline": "PORTED"
        },
        "tuning_info": tuning_info,
        "completed_runs": [
            {
                "arm": r["arm"],
                "seed": r["seed"],
                "provenance": r["provenance"],
                "R_agnostic": r["R_agnostic"].tolist() if isinstance(r["R_agnostic"], np.ndarray) else r["R_agnostic"],
                "R_aware": r["R_aware"].tolist() if isinstance(r["R_aware"], np.ndarray) else r["R_aware"],
                "linear_probe_acc": float(r["probe_acc"]),
                "wall_clock_seconds": float(r["wall_clock"]),
                "peak_gpu_memory_bytes": int(r["peak_gpu"]),
                "n_optimizer_steps": int(r["opt_steps"]),
                "n_train_samples_seen": int(r["samples_seen"]),
                "n_forward_samples": int(r["fwd_samples"]),
                "param_count_total": int(r["param_total"]),
                "param_count_trainable": int(r["param_trainable"])
            }
            for r in all_runs_records
        ]
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    print("=" * 105)
    print(" DIRECTIVE W3 -- PART 2: CONTINUAL LEARNING BASELINE TABLE (9 ARMS x 5 SEEDS)")
    print("=" * 105)

    git_sha = check_provenance()
    print(f"  Git Commit SHA     : {git_sha}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Platform Device    : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Accelerator    : {torch.cuda.get_device_name(0)}")

    print("\n  PROVENANCE DECLARATIONS:")
    print("    - eval_core.compute_r_metrics : REUSED (lower-triangular R-matrix continual learning metrics)")
    print("    - replay_buffer.DERBuffer     : REUSED (tensor buffer for DER++ and ER)")
    print("    - 1_freeze_after_base         : PORTED (standing control logic from run_phase4_lever2_replay.py)")
    print("    - 2_naive_fine_tune           : PORTED (sequential SGD logic from run_partB_naive_reproduction.py)")
    print("    - 3_ncm_frozen_features       : PORTED (NCM logic from run_aa11_adaptation_gap_pretrained.py)")
    print("    - 4_ncm_adapting_features     : PORTED (adapting backbone + NCM centroids from eval_core.py)")
    print("    - 5_lwf                       : NEW    (fresh implementation: CE + lambda*T^2*KL on old classes)")
    print("    - 6_ewc                       : NEW    (fresh implementation: CE + (lambda/2)*sum F_i*(theta-theta*)^2)")
    print("    - 7_er_buffer500              : PORTED (replay logic from run_phase4_lever2_replay.py, DERBuffer REUSED)")
    print("    - 8_der_plus_plus_buffer500   : PORTED (DER++ loss logic from run_phase5_der_plus_plus_class_il.py)")
    print("    - 9_joint_offline             : PORTED (joint offline training from run_w2e_gap_closed.py Arm C)")

    print("\n  HEADL1c FRAMING AUDIT:")
    print("    In Phase 4, HeadL1c evaluated on static 960-d embedding cache smollm2_embeddings_100slots.pt.")
    print("    Because representation drift was physically impossible by construction (frozen transformer),")
    print("    the reported BWT of -42.09% was 100% CLASSIFIER DRIFT by construction, not an ambiguous mixture.")

    if not os.path.exists(ARCHIVE_PATH):
        print(f"  CIFAR-100 archive not found at {ARCHIVE_PATH}. Downloading...")
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

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

    # -------------------------------------------------------------
    # RESUMPTION AUDIT
    # -------------------------------------------------------------
    completed_runs_map = {}
    tuning_info = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, "r") as f:
                existing_data = json.load(f)
            tuning_info = existing_data.get("tuning_info", {})
            for r in existing_data.get("completed_runs", []):
                key = (r["arm"], r["seed"])
                r["R_agnostic"] = np.array(r["R_agnostic"])
                r["R_aware"] = np.array(r["R_aware"])
                r["probe_acc"] = r["linear_probe_acc"]
                r["wall_clock"] = r["wall_clock_seconds"]
                r["peak_gpu"] = r["peak_gpu_memory_bytes"]
                r["opt_steps"] = r["n_optimizer_steps"]
                r["samples_seen"] = r["n_train_samples_seen"]
                r["fwd_samples"] = r["n_forward_samples"]
                r["param_total"] = r["param_count_total"]
                r["param_trainable"] = r["param_count_trainable"]
                completed_runs_map[key] = r
            print(f"\n  [Resumption Audit] Loaded {len(completed_runs_map)} completed (arm, seed) runs from {OUTPUT_JSON_PATH}.")
        except Exception as e:
            print(f"  [Resumption Audit Warning] Could not parse existing JSON: {e}")

    # Tune hyperparameters for NEW arms if not already recorded
    if "lwf_lambda" not in tuning_info:
        best_lwf_l, lwf_scores, lwf_is_boundary = tune_lwf_lambda(task_train_loaders, task_val_loaders, device)
        tuning_info["lwf_lambda"] = best_lwf_l
        tuning_info["lwf_scores"] = lwf_scores
        tuning_info["lwf_is_boundary"] = lwf_is_boundary
    else:
        best_lwf_l = tuning_info["lwf_lambda"]

    if "ewc_lambda" not in tuning_info:
        best_ewc_l, ewc_scores, ewc_is_boundary = tune_ewc_lambda(task_train_loaders, task_val_loaders, device)
        tuning_info["ewc_lambda"] = best_ewc_l
        tuning_info["ewc_scores"] = ewc_scores
        tuning_info["ewc_is_boundary"] = ewc_is_boundary
    else:
        best_ewc_l = tuning_info["ewc_lambda"]

    ARM_DISPATCH = [
        ("1_freeze_after_base", run_freeze_after_base, ()),
        ("2_naive_fine_tune", run_naive_fine_tune, ()),
        ("3_ncm_frozen_features", run_ncm_frozen, ()),
        ("4_ncm_adapting_features", run_ncm_adapting, ()),
        ("5_lwf", run_lwf, (best_lwf_l, 2.0)),
        ("6_ewc", run_ewc, (best_ewc_l,)),
        ("7_er_buffer500", run_er, ()),
        ("8_der_plus_plus_buffer500", run_der_plus_plus, ()),
        ("9_joint_offline", lambda s, tr, te, ftr, fte, d: run_joint_offline(s, full_train_loader, te, ftr, fte, d), ())
    ]

    total_cells = len(ARM_DISPATCH) * len(SEEDS)
    computed_this_session = 0

    for arm_name, arm_fn, extra_args in ARM_DISPATCH:
        for seed in SEEDS:
            cell_key = (arm_name, seed)
            if cell_key in completed_runs_map:
                print(f"  [Cache Hit] Arm '{arm_name}' on Seed {seed} already completed. Loaded from disk.")
                continue

            print(f"\n-------------------------------------------------------------------------------------------------")
            print(f"  RUNNING: Arm '{arm_name}' | Seed {seed} ({len(completed_runs_map)+1}/{total_cells})")
            print(f"-------------------------------------------------------------------------------------------------")

            res = arm_fn(seed, task_train_loaders, task_test_loaders, full_tr_probe_loader, full_te_probe_loader, device, *extra_args)
            completed_runs_map[cell_key] = res
            computed_this_session += 1

            r_met = compute_r_metrics(res["R_agnostic"])
            print(f"    Completed in {res['wall_clock']:.1f}s | Class-IL ACC: {r_met['acc_T']:.2f}% | Aware ACC: {np.mean(res['R_aware'][9, :]):.2f}% | Probe: {res['probe_acc']:.2f}% | BWT: {r_met['bwt']:+.2f} pp")

            # Flush immediately to disk after every single (arm, seed)
            save_incremental_json(OUTPUT_JSON_PATH, git_sha, list(completed_runs_map.values()), tuning_info)

    # -------------------------------------------------------------
    # STATISTICAL SUMMARY & FINAL TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 135)
    print(" CONTINUAL LEARNING BASELINE TABLE (SPLIT-CIFAR-100, 9 ARMS x 5 SEEDS)")
    print("=" * 135)
    header = f"{'Arm Name':<28} | {'(i) Class-IL ACC_T':<18} | {'(ii) Aware ACC_T':<18} | {'Bias Gap (ii-i)':<15} | {'(iii) Probe ACC':<16} | {'BWT (pp)':<14} | {'Forgetting':<12}"
    print(header)
    print("-" * 135)

    summary_stats = {}
    for arm_name, _, _ in ARM_DISPATCH:
        arm_runs = [completed_runs_map[(arm_name, s)] for s in SEEDS if (arm_name, s) in completed_runs_map]
        assert len(arm_runs) == 5, f"Expected 5 seeds for {arm_name}, got {len(arm_runs)}"

        ag_list, aw_list, bias_list, pr_list, bwt_list, fgt_list, la_list = [], [], [], [], [], [], []

        for r in arm_runs:
            r_met = compute_r_metrics(r["R_agnostic"])
            ag = r_met["acc_T"]
            aw = float(np.mean(r["R_aware"][9, :]))
            ag_list.append(ag)
            aw_list.append(aw)
            bias_list.append(aw - ag)
            pr_list.append(r["probe_acc"])
            bwt_list.append(r_met["bwt"])
            fgt_list.append(r_met["forgetting"])
            la_list.append(float(np.mean([r["R_agnostic"][i, i] for i in range(10)])))

        def calc_m_s(arr):
            return float(np.mean(arr)), float(np.std(arr, ddof=1))

        m_ag, s_ag = calc_m_s(ag_list)
        m_aw, s_aw = calc_m_s(aw_list)
        m_bias, s_bias = calc_m_s(bias_list)
        m_pr, s_pr = calc_m_s(pr_list)
        m_bwt, s_bwt = calc_m_s(bwt_list)
        m_fgt, s_fgt = calc_m_s(fgt_list)
        m_la, s_la = calc_m_s(la_list)

        print(f"{arm_name:<28} | {m_ag:5.2f}% +/- {s_ag:4.2f}% | {m_aw:5.2f}% +/- {s_aw:4.2f}% | {m_bias:+5.2f} +/- {s_bias:4.2f} | {m_pr:5.2f}% +/- {s_pr:4.2f}% | {m_bwt:+5.2f} +/- {s_bwt:4.2f} | {m_fgt:5.2f}% +/- {s_fgt:4.2f}%")

        summary_stats[arm_name] = {
            "class_il_mean": m_ag, "class_il_std": s_ag, "class_il_per_seed": ag_list,
            "task_aware_mean": m_aw, "task_aware_std": s_aw, "task_aware_per_seed": aw_list,
            "bias_gap_mean": m_bias, "bias_gap_std": s_bias,
            "linear_probe_mean": m_pr, "linear_probe_std": s_pr, "linear_probe_per_seed": pr_list,
            "bwt_mean": m_bwt, "bwt_std": s_bwt,
            "forgetting_mean": m_fgt, "forgetting_std": s_fgt,
            "avg_la_mean": m_la, "avg_la_std": s_la
        }

    # Final JSON structure update
    final_output = {
        "git_commit_sha": git_sha,
        "dataset": "Split-CIFAR-100",
        "seeds": SEEDS,
        "epochs_per_task": EPOCHS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "lr_base": LR_BASE,
        "provenance_summary": {
            "eval_core.compute_r_metrics": "REUSED",
            "replay_buffer.DERBuffer": "REUSED",
            "1_freeze_after_base": "PORTED",
            "2_naive_fine_tune": "PORTED",
            "3_ncm_frozen_features": "PORTED",
            "4_ncm_adapting_features": "PORTED",
            "5_lwf": "NEW",
            "6_ewc": "NEW",
            "7_er_buffer500": "PORTED",
            "8_der_plus_plus_buffer500": "PORTED",
            "9_joint_offline": "PORTED"
        },
        "session_execution_audit": {
            "total_cells": total_cells,
            "loaded_from_disk": len(completed_runs_map) - computed_this_session,
            "computed_this_session": computed_this_session
        },
        "tuning_info": tuning_info,
        "summary": summary_stats,
        "completed_runs": [
            {
                "arm": r["arm"],
                "seed": r["seed"],
                "provenance": r["provenance"],
                "R_agnostic": r["R_agnostic"].tolist() if isinstance(r["R_agnostic"], np.ndarray) else r["R_agnostic"],
                "R_aware": r["R_aware"].tolist() if isinstance(r["R_aware"], np.ndarray) else r["R_aware"],
                "linear_probe_acc": float(r["probe_acc"]),
                "wall_clock_seconds": float(r["wall_clock"]),
                "peak_gpu_memory_bytes": int(r["peak_gpu"]),
                "n_optimizer_steps": int(r["opt_steps"]),
                "n_train_samples_seen": int(r["samples_seen"]),
                "n_forward_samples": int(r["fwd_samples"]),
                "param_count_total": int(r["param_total"]),
                "param_count_trainable": int(r["param_trainable"])
            }
            for r in completed_runs_map.values()
        ]
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n  Final baseline results successfully saved to {OUTPUT_JSON_PATH}")

    # Reproduction check for Joint Offline
    offline_m = summary_stats["9_joint_offline"]["class_il_mean"]
    print("\n  Reproduction Audit:")
    print(f"    Joint Offline Upper Bound (w2e target: 79.64% +/- 0.23%): Measured {offline_m:.2f}% -> {'PASS' if abs(offline_m - 79.64) < 2.0 else 'WARNING'}")

    print("\n" + "=" * 105)
    print("EXIT_CODE = 0")
    print("=" * 105)


if __name__ == "__main__":
    main()
