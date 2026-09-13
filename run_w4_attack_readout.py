#!/usr/bin/env python3
"""
===================================================================================================
DIRECTIVE W4 (TASK 5 RE-SCOPED) -- EXEMPLAR-FREE ATTACK ON THE CLASSIFIER READOUT
===================================================================================================
Strictly protocol-matched to Directive W3 (ResNet-18, 20 epochs/task, batch_size=128, lr=0.005).
Audits the +23.73 pp headroom between stale class centroids (41.98%) and the jointly-fitted
linear probe ceiling (65.71%) on the identical naive-adapted ResNet-18 backbone:
  - M1: Whitened / Shared-Covariance NCM (SLDA-equivalent; Hayes & Kanan, CVPR 2020)
  - M2: Semantic Drift Compensation (SDC; Yu et al., CVPR 2020)
  - Control Random Trigger M1: Permuted covariance structure
  - Control Random Trigger M2: Random spherical drift vectors of matched norm
  - Parameter-Matched Baseline: Naive fine-tune linear head (9.53%)
  - Direct Predecessor: Stale class centroids (41.98%)
  - Permanent Standing Control Arm: FREEZE-AFTER-BASE (8.67%)
  - Upper-Bound Ceiling: Joint linear probe (65.71%)

Evaluated across SEEDS = [42, 43, 44, 45, 46] on Split-CIFAR-100 (10 tasks x 10 classes).
Hyperparameters tuned strictly on the held-out validation split under a truncated 3-task horizon.
===================================================================================================
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
from torchvision import datasets, transforms, models

# ---------------------------------------------------------------------
# BENCHMARK AND HARNESS CONFIGURATION (MATCHED TO W3)
# ---------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
NUM_CLASSES = 100
CLASSES_PER_TASK = 10
NUM_TASKS = 10
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
LR_BASE = 0.005
WEIGHT_DECAY = 1e-4

CEILING_PROBE_ACC = 65.71
PREDECESSOR_ACC = 41.98
HEADROOM_DENOMINATOR = CEILING_PROBE_ACC - PREDECESSOR_ACC  # 23.73 pp
NAIVE_BWT = -88.06
OFFLINE_BWT = 0.00
RETENTION_DENOMINATOR = OFFLINE_BWT - NAIVE_BWT  # 88.06 pp

OUTPUT_JSON_PATH = "w4_attack_readout.json"


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


# ---------------------------------------------------------------------
# DATASET SPLIT-CIFAR-100 (10 TASKS x 10 CLASSES)
# ---------------------------------------------------------------------
def get_cifar100_loaders(data_root="./data"):
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

    try:
        ds_tr = datasets.CIFAR100(root=data_root, train=True, download=False, transform=tr_transform)
        ds_ev = datasets.CIFAR100(root=data_root, train=True, download=False, transform=ev_transform)
        ds_te = datasets.CIFAR100(root=data_root, train=False, download=False, transform=ev_transform)
    except Exception:
        print("  CIFAR-100 archive not found locally. Downloading to:", data_root)
        ds_tr = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tr_transform)
        ds_ev = datasets.CIFAR100(root=data_root, train=True, download=True, transform=ev_transform)
        ds_te = datasets.CIFAR100(root=data_root, train=False, download=True, transform=ev_transform)

    train_idx, val_idx = partition_indices(ds_tr.targets, n_train=400, n_val=100, seed=42)

    targets_tr = np.array(ds_tr.targets)[train_idx]
    targets_va = np.array(ds_ev.targets)[val_idx]
    targets_te = np.array(ds_te.targets)

    blocks = [list(range(t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK)) for t in range(NUM_TASKS)]

    task_train_loaders = {}
    task_train_eval_loaders = {}
    task_val_loaders = {}
    task_test_loaders = {}

    for t_idx, classes in enumerate(blocks):
        t_tr_local = [train_idx[i] for i, c in enumerate(targets_tr) if c in classes]
        t_va_local = [val_idx[i] for i, c in enumerate(targets_va) if c in classes]
        t_te_local = [i for i, c in enumerate(targets_te) if c in classes]

        task_train_loaders[t_idx] = (
            DataLoader(Subset(ds_tr, t_tr_local), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker),
            classes
        )
        task_train_eval_loaders[t_idx] = (
            DataLoader(Subset(ds_ev, t_tr_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker),
            classes
        )
        task_val_loaders[t_idx] = (
            DataLoader(Subset(ds_ev, t_va_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker),
            classes
        )
        task_test_loaders[t_idx] = (
            DataLoader(Subset(ds_te, t_te_local), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker),
            classes
        )

    full_tr_probe_loader = DataLoader(Subset(ds_ev, train_idx), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)
    full_te_probe_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)

    return (task_train_loaders, task_train_eval_loaders, task_val_loaders,
            task_test_loaders, full_tr_probe_loader, full_te_probe_loader)


# ---------------------------------------------------------------------
# MODEL ARCHITECTURE (RESNET-18 PRIMARY)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# CONTINUAL LEARNING EVALUATION METRICS
# ---------------------------------------------------------------------
def compute_r_metrics(R):
    T = R.shape[0]
    acc_T = float(np.mean(R[T - 1, :T]))
    bwt = float(np.mean([R[T - 1, i] - R[i, i] for i in range(T - 1)])) if T > 1 else 0.0
    return {"acc_T": acc_T, "bwt": bwt}


def evaluate_protocol_matched_linear_probe(backbone, full_tr_loader, full_te_loader, device, seed=42, epochs=30):
    """Protocol-matched 30-epoch linear probe on extracted features (exactly matching W3)."""
    set_seed(seed)
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
    ld_tr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker)
    ld_te = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)

    set_seed(seed)
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

    acc = (cor / tot) * 100.0
    return float(acc)


# ---------------------------------------------------------------------
# M1 & M2 VALIDATION HYPERPARAMETER SELECTION (TRUNCATED 3-TASK HORIZON)
# ---------------------------------------------------------------------
def tune_m1_slda_validation(task_train_loaders, task_val_loaders, device):
    """
    Tune M1 (SLDA) on Seed 42 across Tasks 0, 1, 2 on validation set.
    Hyperparameters: shrinkage eps in [1e-4, 1e-3, 1e-2, 1e-1, 1.0], normalize in [True, False].
    """
    print("\n  [Validation Hyperparameter Sweep: M1 Whitened / Shared-Covariance NCM (SLDA)]")
    print("    Protocol Label : selected under truncated horizon (3 tasks)")
    print("    Scoring Split  : Validation Split (3,000 samples across Tasks 0, 1, 2)")
    print("    Candidates     : eps in [0.0001, 0.001, 0.01, 0.1, 1.0], normalize in [True, False]")

    set_seed(42)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    # Train 3 tasks on train split
    task_features_raw = {}
    task_features_norm = {}
    task_targets = {}

    for t in range(3):
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
            sched.step()

        # Extract features for task t
        model.eval()
        t_f_list, t_y_list = [], []
        with torch.no_grad():
            for bx, by in t_loader:
                bx = bx.to(device)
                t_f_list.append(model.extract_features(bx))
                t_y_list.append(by.to(device))
        feats_raw = torch.cat(t_f_list, dim=0)
        feats_norm = F.normalize(feats_raw, dim=-1)
        targets = torch.cat(t_y_list, dim=0)

        task_features_raw[t] = feats_raw
        task_features_norm[t] = feats_norm
        task_targets[t] = targets

    # Val set feature extraction
    val_feats_raw, val_feats_norm, val_targets = [], [], []
    with torch.no_grad():
        for t in range(3):
            v_loader, _ = task_val_loaders[t]
            for bx, by in v_loader:
                bx = bx.to(device)
                f = model.extract_features(bx)
                val_feats_raw.append(f)
                val_feats_norm.append(F.normalize(f, dim=-1))
                val_targets.append(by.to(device))
    X_val_raw = torch.cat(val_feats_raw, dim=0)
    X_val_norm = torch.cat(val_feats_norm, dim=0)
    y_val = torch.cat(val_targets, dim=0)

    grid_eps = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    grid_norm = [True, False]
    scores = {}
    best_score = -1.0
    best_cfg = None

    for norm in grid_norm:
        feat_dict = task_features_norm if norm else task_features_raw
        X_eval = X_val_norm if norm else X_val_raw

        # Compute centroids and running covariance
        centroids = {}
        cov = torch.zeros((512, 512), device=device)
        total_samples = 0

        for t in range(3):
            t_feats = feat_dict[t]
            t_y = task_targets[t]
            classes = torch.unique(t_y).tolist()
            for c in classes:
                mask = (t_y == c)
                c_feats = t_feats[mask]
                c_mean = c_feats.mean(dim=0)
                centroids[c] = c_mean
                diff = c_feats - c_mean
                cov += torch.matmul(diff.T, diff)
                total_samples += c_feats.size(0)

        cov /= max(total_samples - 1, 1)

        for eps in grid_eps:
            reg_cov = (1.0 - eps) * cov + eps * torch.eye(512, device=device)
            try:
                inv_cov = torch.linalg.inv(reg_cov)
            except Exception:
                inv_cov = torch.linalg.pinv(reg_cov)

            seen_classes = sorted(list(centroids.keys()))
            cen_matrix = torch.stack([centroids[c] for c in seen_classes], dim=0)
            labels_tensor = torch.tensor(seen_classes, device=device)

            W = torch.matmul(inv_cov, cen_matrix.T)
            b = -0.5 * (cen_matrix * torch.matmul(cen_matrix, inv_cov)).sum(dim=1)

            logits_val = torch.matmul(X_eval, W) + b
            preds = labels_tensor[logits_val.argmax(dim=1)]
            val_acc = float((preds == y_val).float().mean().item() * 100.0)

            key_str = f"eps={eps}_norm={norm}"
            scores[key_str] = val_acc
            print(f"    Candidate: eps={eps:<6} | normalize={str(norm):<5} -> Validation ACC: {val_acc:5.2f}%")

            if val_acc > best_score:
                best_score = val_acc
                best_cfg = (eps, norm)

    is_boundary = (best_cfg[0] in [grid_eps[0], grid_eps[-1]])
    print(f"  Selected M1 (SLDA) Optimal Config: eps={best_cfg[0]}, norm={best_cfg[1]} (Val ACC: {best_score:.2f}%) | Boundary: {is_boundary}")
    return best_cfg, scores, is_boundary


def tune_m2_sdc_validation(task_train_loaders, task_val_loaders, device):
    """
    Tune M2 (SDC; Yu et al., CVPR 2020) on Seed 42 across Tasks 0, 1, 2 on validation set.
    Hyperparameters: bandwidth sigma in [0.25, 0.5, 1.0, 2.0, 5.0], renormalize in [True, False].
    """
    print("\n  [Validation Hyperparameter Sweep: M2 Semantic Drift Compensation (SDC)]")
    print("    Protocol Label : selected under truncated horizon (3 tasks)")
    print("    Scoring Split  : Validation Split (3,000 samples across Tasks 0, 1, 2)")
    print("    Candidates     : sigma in [0.25, 0.5, 1.0, 2.0, 5.0], renormalize in [True, False]")

    set_seed(42)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    grid_sigma = [0.25, 0.5, 1.0, 2.0, 5.0]
    grid_renorm = [True, False]

    # Pre-train and store checkpoints for tasks 0, 1, 2
    checkpoints = {}
    for t in range(3):
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
            sched.step()
        checkpoints[t] = copy.deepcopy(model.state_dict())

    scores = {}
    best_score = -1.0
    best_cfg = None

    for renorm in grid_renorm:
        for sigma in grid_sigma:
            centroids = {}
            for t in range(3):
                model.load_state_dict(checkpoints[t])
                model.eval()

                t_loader, t_classes = task_train_loaders[t]
                t_f_new, t_targets = [], []
                with torch.no_grad():
                    for bx, by in t_loader:
                        bx = bx.to(device)
                        t_f_new.append(model.extract_features(bx))
                        t_targets.append(by.to(device))
                t_f_new = torch.cat(t_f_new, dim=0)
                t_targets = torch.cat(t_targets, dim=0)

                cur_mu_new = {}
                for c in t_classes:
                    mask = (t_targets == c)
                    m = t_f_new[mask].mean(dim=0)
                    cur_mu_new[c] = F.normalize(m, dim=-1) if renorm else m

                if t > 0:
                    prev_model = ResNet18Primary(num_classes=100).to(device)
                    prev_model.load_state_dict(checkpoints[t - 1])
                    prev_model.eval()

                    t_f_old = []
                    with torch.no_grad():
                        for bx, _ in t_loader:
                            bx = bx.to(device)
                            t_f_old.append(prev_model.extract_features(bx))
                    t_f_old = torch.cat(t_f_old, dim=0)

                    cur_mu_old = {}
                    for c in t_classes:
                        mask = (t_targets == c)
                        m_old = t_f_old[mask].mean(dim=0)
                        cur_mu_old[c] = F.normalize(m_old, dim=-1) if renorm else m_old

                    cur_drifts = {c: (cur_mu_new[c] - cur_mu_old[c]) for c in t_classes}

                    for past_c in list(centroids.keys()):
                        past_mu = centroids[past_c]
                        dists = torch.tensor([torch.norm(past_mu - cur_mu_old[k])**2 for k in t_classes], device=device)
                        weights = F.softmax(-dists / (2.0 * (sigma ** 2)), dim=0)
                        drift_vec = sum(weights[i] * cur_drifts[k] for i, k in enumerate(t_classes))

                        updated_mu = past_mu + drift_vec
                        if renorm:
                            updated_mu = F.normalize(updated_mu, dim=-1)
                        centroids[past_c] = updated_mu

                for c in t_classes:
                    centroids[c] = cur_mu_new[c]

            model.load_state_dict(checkpoints[2])
            model.eval()

            val_feats, val_targets = [], []
            with torch.no_grad():
                for t in range(3):
                    v_loader, _ = task_val_loaders[t]
                    for bx, by in v_loader:
                        bx = bx.to(device)
                        f = model.extract_features(bx)
                        val_feats.append(F.normalize(f, dim=-1) if renorm else f)
                        val_targets.append(by.to(device))
            X_val = torch.cat(val_feats, dim=0)
            y_val = torch.cat(val_targets, dim=0)

            seen_classes = sorted(list(centroids.keys()))
            cen_matrix = torch.stack([centroids[c] for c in seen_classes], dim=0)
            labels_tensor = torch.tensor(seen_classes, device=device)

            if renorm:
                sims = torch.matmul(X_val, cen_matrix.T)
                preds = labels_tensor[sims.argmax(dim=1)]
            else:
                dists = torch.cdist(X_val, cen_matrix)
                preds = labels_tensor[dists.argmin(dim=1)]

            val_acc = float((preds == y_val).float().mean().item() * 100.0)
            key_str = f"sigma={sigma}_renorm={renorm}"
            scores[key_str] = val_acc
            print(f"    Candidate: sigma={sigma:<4} | renormalize={str(renorm):<5} -> Validation ACC: {val_acc:5.2f}%")

            if val_acc > best_score:
                best_score = val_acc
                best_cfg = (sigma, renorm)

    is_boundary = (best_cfg[0] in [grid_sigma[0], grid_sigma[-1]])
    print(f"  Selected M2 (SDC) Optimal Config: sigma={best_cfg[0]}, renorm={best_cfg[1]} (Val ACC: {best_score:.2f}%) | Boundary: {is_boundary}")
    return best_cfg, scores, is_boundary


# ---------------------------------------------------------------------
# STANDING CONTROL ARM: 1_FREEZE_AFTER_BASE
# ---------------------------------------------------------------------
def run_freeze_after_base(seed, task_train_loaders, task_test_loaders, device):
    """Permanent standing control arm per Rule 1."""
    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    crit = nn.CrossEntropyLoss()

    R_agnostic = np.zeros((10, 10))

    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    t0_loader, _ = task_train_loaders[0]
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

    # Freeze backbone
    for param in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in param.parameters():
            p.requires_grad = False

    # Evaluate task 0
    model.eval()
    t0_test, _ = task_test_loaders[0]
    cor, tot = 0, 0
    with torch.no_grad():
        for bx, by in t0_test:
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            cor += (logits.argmax(dim=-1) == by).sum().item()
            tot += by.size(0)
    R_agnostic[0, 0] = (cor / tot) * 100.0

    # Train tasks 1-9 (head only)
    opt_head = optim.SGD(model.fc.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    for t in range(1, 10):
        t_loader, _ = task_train_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=EPOCHS_PER_TASK, eta_min=1e-4)
        for ep in range(EPOCHS_PER_TASK):
            model.train()
            model.eval()  # Keep BN frozen in eval
            for bx, by in t_loader:
                bx, by = bx.to(device), by.to(device)
                opt_head.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt_head.step()
            sched.step()

        # Evaluate seen tasks
        model.eval()
        seen = list(range(t + 1))
        for j in seen:
            j_loader, _ = task_test_loaders[j]
            cor_ag, tot_j = 0, 0
            with torch.no_grad():
                for bx, by in j_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits, _ = model(bx)
                    preds_ag = logits.argmax(dim=-1)
                    cor_ag += (preds_ag == by).sum().item()
                    tot_j += by.size(0)
            R_agnostic[t, j] = (cor_ag / tot_j) * 100.0

    return compute_r_metrics(R_agnostic)


# ---------------------------------------------------------------------
# SHARED BACKBONE ADAPTATION & COMPREHENSIVE READOUT EVALUATION
# ---------------------------------------------------------------------
def run_attack_readout_cell(seed, m1_cfg, m2_cfg, loaders, device):
    """
    Executes one seed cell:
    1. Trains backbone sequentially on Tasks 0..9 (identical to Arm 2 & Arm 4).
    2. Concurrently evaluates readouts at each task:
       - Arm 2: Sequential linear head
       - Arm 4: Stale class centroids
       - M1: SLDA whitened / shared-covariance NCM
       - Control Random M1: Permuted covariance structure
       - M2: SDC semantic drift compensation
       - Control Random M2: Random spherical drift vectors of matched norm
    3. Evaluates 30-epoch joint linear probe ceiling.
    """
    (task_train_loaders, task_train_eval_loaders, task_val_loaders,
     task_test_loaders, full_tr_probe_loader, full_te_probe_loader) = loaders

    m1_eps, m1_norm = m1_cfg
    m2_sigma, m2_renorm = m2_cfg

    set_seed(seed)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    R_linear = np.zeros((10, 10))
    R_stale = np.zeros((10, 10))
    R_m1_slda = np.zeros((10, 10))
    R_m1_ctrl = np.zeros((10, 10))
    R_m2_sdc = np.zeros((10, 10))
    R_m2_ctrl = np.zeros((10, 10))

    # Stored state for readouts
    centroids_stale = {}
    centroids_m2_sdc = {}
    centroids_m2_ctrl = {}

    # M1 SLDA running covariance
    m1_centroids = {}
    m1_cov = torch.zeros((512, 512), device=device)
    m1_total_samples = 0

    t0 = time.time()
    prev_model = None

    for t in range(10):
        t_tr_loader, t_classes = task_train_loaders[t]
        t_ev_loader, _ = task_train_eval_loaders[t]
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_TASK, eta_min=1e-4)

        # 1. Train backbone + head on task t
        for ep in range(EPOCHS_PER_TASK):
            model.train()
            for bx, by in t_tr_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits, _ = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt.step()
            sched.step()

        # 2. Extract unaugmented prototype features for current task t
        model.eval()
        t_feats_raw, t_targets = [], []
        with torch.no_grad():
            for bx, by in t_ev_loader:
                bx = bx.to(device)
                t_feats_raw.append(model.extract_features(bx))
                t_targets.append(by.to(device))
        t_feats_raw = torch.cat(t_feats_raw, dim=0)
        t_targets = torch.cat(t_targets, dim=0)

        # Stale centroids (Arm 4)
        for c in t_classes:
            mask = (t_targets == c)
            centroids_stale[c] = F.normalize(t_feats_raw[mask].mean(dim=0), dim=-1)

        # M1 (SLDA) update
        m1_feat_input = F.normalize(t_feats_raw, dim=-1) if m1_norm else t_feats_raw
        for c in t_classes:
            mask = (t_targets == c)
            c_feats = m1_feat_input[mask]
            c_mean = c_feats.mean(dim=0)
            m1_centroids[c] = c_mean
            diff = c_feats - c_mean
            m1_cov += torch.matmul(diff.T, diff)
            m1_total_samples += c_feats.size(0)

        # M2 (SDC) drift update
        cur_mu_new = {}
        for c in t_classes:
            mask = (t_targets == c)
            m_new = t_feats_raw[mask].mean(dim=0)
            cur_mu_new[c] = F.normalize(m_new, dim=-1) if m2_renorm else m_new

        if t > 0 and prev_model is not None:
            prev_model.eval()
            t_feats_prev = []
            with torch.no_grad():
                for bx, _ in t_ev_loader:
                    bx = bx.to(device)
                    t_feats_prev.append(prev_model.extract_features(bx))
            t_feats_prev = torch.cat(t_feats_prev, dim=0)

            cur_mu_old = {}
            for c in t_classes:
                mask = (t_targets == c)
                m_old = t_feats_prev[mask].mean(dim=0)
                cur_mu_old[c] = F.normalize(m_old, dim=-1) if m2_renorm else m_old

            cur_drifts = {c: (cur_mu_new[c] - cur_mu_old[c]) for c in t_classes}

            for past_c in list(centroids_m2_sdc.keys()):
                past_mu = centroids_m2_sdc[past_c]
                dists = torch.tensor([torch.norm(past_mu - cur_mu_old[k])**2 for k in t_classes], device=device)
                weights = F.softmax(-dists / (2.0 * (m2_sigma ** 2)), dim=0)
                drift_vec = sum(weights[i] * cur_drifts[k] for i, k in enumerate(t_classes))

                up_sdc = past_mu + drift_vec
                if m2_renorm:
                    up_sdc = F.normalize(up_sdc, dim=-1)
                centroids_m2_sdc[past_c] = up_sdc

                # Control M2: Random spherical drift vector of identical norm
                drift_norm = torch.norm(drift_vec).item()
                rand_dir = torch.randn(512, device=device)
                rand_dir = F.normalize(rand_dir, dim=-1) * drift_norm
                up_ctrl = centroids_m2_ctrl[past_c] + rand_dir
                if m2_renorm:
                    up_ctrl = F.normalize(up_ctrl, dim=-1)
                centroids_m2_ctrl[past_c] = up_ctrl

        for c in t_classes:
            centroids_m2_sdc[c] = cur_mu_new[c]
            centroids_m2_ctrl[c] = cur_mu_new[c]

        prev_model = copy.deepcopy(model)
        prev_model.eval()

        # Invert regularized covariance for M1 SLDA
        m1_norm_cov = m1_cov / max(m1_total_samples - 1, 1)
        m1_reg_cov = (1.0 - m1_eps) * m1_norm_cov + m1_eps * torch.eye(512, device=device)
        try:
            m1_inv_cov = torch.linalg.inv(m1_reg_cov)
        except Exception:
            m1_inv_cov = torch.linalg.pinv(m1_reg_cov)

        p_idx = torch.randperm(512, device=device)
        m1_inv_cov_ctrl = m1_inv_cov[p_idx, :][:, p_idx]

        # 3. Evaluate all readouts on seen tasks
        seen = list(range(t + 1))
        seen_classes = [c for s in seen for c in task_train_loaders[s][1]]
        seen_labels = torch.tensor(seen_classes, device=device)

        cen_mat_stale = torch.stack([centroids_stale[c] for c in seen_classes], dim=0)

        m1_cen_mat = torch.stack([m1_centroids[c] for c in seen_classes], dim=0)
        W_m1 = torch.matmul(m1_inv_cov, m1_cen_mat.T)
        b_m1 = -0.5 * (m1_cen_mat * torch.matmul(m1_cen_mat, m1_inv_cov)).sum(dim=1)

        W_m1_ctrl = torch.matmul(m1_inv_cov_ctrl, m1_cen_mat.T)
        b_m1_ctrl = -0.5 * (m1_cen_mat * torch.matmul(m1_cen_mat, m1_inv_cov_ctrl)).sum(dim=1)

        cen_mat_sdc = torch.stack([centroids_m2_sdc[c] for c in seen_classes], dim=0)
        cen_mat_ctrl_sdc = torch.stack([centroids_m2_ctrl[c] for c in seen_classes], dim=0)

        for j in seen:
            j_loader, _ = task_test_loaders[j]
            cor_lin, cor_stale = 0, 0
            cor_m1, cor_m1_ctrl = 0, 0
            cor_m2, cor_m2_ctrl = 0, 0
            tot_j = 0

            with torch.no_grad():
                for bx, by in j_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits, raw_f = model(bx)
                    norm_f = F.normalize(raw_f, dim=-1)

                    # 1. Linear head (global argmax over 100 classes per W3 standard)
                    preds_lin = logits.argmax(dim=-1)
                    cor_lin += (preds_lin == by).sum().item()

                    # 2. Stale centroids
                    sims_stale = torch.matmul(norm_f, cen_mat_stale.T)
                    cor_stale += (seen_labels[sims_stale.argmax(dim=-1)] == by).sum().item()

                    # 3. M1 SLDA
                    m1_input = norm_f if m1_norm else raw_f
                    log_m1 = torch.matmul(m1_input, W_m1) + b_m1
                    cor_m1 += (seen_labels[log_m1.argmax(dim=-1)] == by).sum().item()

                    # 4. Control M1
                    log_m1_ctrl = torch.matmul(m1_input, W_m1_ctrl) + b_m1_ctrl
                    cor_m1_ctrl += (seen_labels[log_m1_ctrl.argmax(dim=-1)] == by).sum().item()

                    # 5. M2 SDC
                    if m2_renorm:
                        sims_sdc = torch.matmul(norm_f, cen_mat_sdc.T)
                        preds_m2 = seen_labels[sims_sdc.argmax(dim=-1)]
                        sims_m2_ctrl = torch.matmul(norm_f, cen_mat_ctrl_sdc.T)
                        preds_m2_ctrl = seen_labels[sims_m2_ctrl.argmax(dim=-1)]
                    else:
                        preds_m2 = seen_labels[torch.cdist(raw_f, cen_mat_sdc).argmin(dim=-1)]
                        preds_m2_ctrl = seen_labels[torch.cdist(raw_f, cen_mat_ctrl_sdc).argmin(dim=-1)]

                    cor_m2 += (preds_m2 == by).sum().item()
                    cor_m2_ctrl += (preds_m2_ctrl == by).sum().item()

                    tot_j += by.size(0)

            R_linear[t, j] = (cor_lin / tot_j) * 100.0
            R_stale[t, j] = (cor_stale / tot_j) * 100.0
            R_m1_slda[t, j] = (cor_m1 / tot_j) * 100.0
            R_m1_ctrl[t, j] = (cor_m1_ctrl / tot_j) * 100.0
            R_m2_sdc[t, j] = (cor_m2 / tot_j) * 100.0
            R_m2_ctrl[t, j] = (cor_m2_ctrl / tot_j) * 100.0

    # 4. Evaluate Protocol-Matched Linear Probe Ceiling on adapting backbone
    probe_acc = evaluate_protocol_matched_linear_probe(
        model, full_tr_probe_loader, full_te_probe_loader, device, seed=seed, epochs=30
    )

    # 5. Standing Control Arm
    m_freeze = run_freeze_after_base(seed, task_train_loaders, task_test_loaders, device)

    wall = time.time() - t0

    return {
        "seed": seed,
        "wall_clock": wall,
        "probe_ceiling": probe_acc,
        "freeze_after_base": m_freeze,
        "linear_head": compute_r_metrics(R_linear),
        "stale_centroids": compute_r_metrics(R_stale),
        "m1_slda": compute_r_metrics(R_m1_slda),
        "control_m1": compute_r_metrics(R_m1_ctrl),
        "m2_sdc": compute_r_metrics(R_m2_sdc),
        "control_m2": compute_r_metrics(R_m2_ctrl),
        "R_linear": R_linear.tolist(),
        "R_stale": R_stale.tolist(),
        "R_m1_slda": R_m1_slda.tolist(),
        "R_m2_sdc": R_m2_sdc.tolist()
    }


# ---------------------------------------------------------------------
# MAIN RESUMABLE EXECUTION LOOP & AUDITED REPORTING
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Directive W4 Task 5 Re-Scoped: Exemplar-Free Attack on Readout")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CIFAR-100")
    parser.add_argument("--max_hours", type=float, default=6.5, help="Session timeout in hours")
    args = parser.parse_args()

    session_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        pass

    print("=" * 115)
    print(" DIRECTIVE W4 -- TASK 5 RE-SCOPED: ATTACK THE READOUT (EXEMPLAR-FREE HEADROOM AUDIT)")
    print("===================================================================================")
    print(f"  Git Commit SHA     : {git_sha}")
    print(f"  Platform Device    : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Accelerator    : {torch.cuda.get_device_name(0)}")
    print(f"  Evaluation Seeds   : {SEEDS} (n={len(SEEDS)})")
    print(f"  Available Headroom : +{HEADROOM_DENOMINATOR:.2f} pp (Ceiling {CEILING_PROBE_ACC:.2f}% - Predecessor {PREDECESSOR_ACC:.2f}%)")
    print("=" * 115)

    loaders = get_cifar100_loaders(args.data_dir)
    (task_train_loaders, task_train_eval_loaders, task_val_loaders,
     task_test_loaders, full_tr_probe_loader, full_te_probe_loader) = loaders

    # 1. Resumption Audit
    completed_cells = {}
    tuning_info = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, "r") as f:
                saved = json.load(f)
                tuning_info = saved.get("tuning_info", {})
                for run in saved.get("completed_runs", []):
                    completed_cells[run["seed"]] = run
            print(f"  [Resumption Audit] Loaded {len(completed_cells)} / {len(SEEDS)} completed seeds from {OUTPUT_JSON_PATH}.")
        except Exception as e:
            print(f"  [Resumption Warning] Failed to load existing JSON: {e}")

    # 2. Hyperparameter Sweeps (Seed 42 on Validation Split, Truncated 3-Task Horizon)
    if "m1_cfg" not in tuning_info:
        m1_cfg, m1_scores, m1_boundary = tune_m1_slda_validation(task_train_loaders, task_val_loaders, device)
        tuning_info["m1_cfg"] = m1_cfg
        tuning_info["m1_scores"] = m1_scores
        tuning_info["m1_boundary"] = m1_boundary
    else:
        m1_cfg = tuning_info["m1_cfg"]
        print(f"\n  [Loaded from Prior Session] M1 (SLDA) Optimal Config: eps={m1_cfg[0]}, norm={m1_cfg[1]}")

    if "m2_cfg" not in tuning_info:
        m2_cfg, m2_scores, m2_boundary = tune_m2_sdc_validation(task_train_loaders, task_val_loaders, device)
        tuning_info["m2_cfg"] = m2_cfg
        tuning_info["m2_scores"] = m2_scores
        tuning_info["m2_boundary"] = m2_boundary
    else:
        m2_cfg = tuning_info["m2_cfg"]
        print(f"  [Loaded from Prior Session] M2 (SDC) Optimal Config: sigma={m2_cfg[0]}, renorm={m2_cfg[1]}")

    # 3. Demand-Driven Resumable Execution Loop over Seeds
    for seed in SEEDS:
        if seed in completed_cells:
            print(f"  [LOADED FROM DISK] Seed {seed} already completed.")
            continue

        elapsed = time.time() - session_t0
        if elapsed + 1200.0 > args.max_hours * 3600.0:
            print(f"\n  [CLEAN HALT] Elapsed {elapsed/3600.0:.2f}h. Halting cleanly before Seed {seed}.")
            break

        print(f"\n-------------------------------------------------------------------------------------------------")
        print(f"  [COMPUTING SEED {seed}] ResNet-18 Adaptation & Multi-Readout Evaluation")
        print(f"-------------------------------------------------------------------------------------------------")

        res = run_attack_readout_cell(seed, m1_cfg, m2_cfg, loaders, device)
        completed_cells[seed] = res

        print(f"    Completed in {res['wall_clock']:.1f}s")
        print(f"    Linear Head       : Class-IL = {res['linear_head']['acc_T']:.2f}% | BWT = {res['linear_head']['bwt']:+.2f} pp")
        print(f"    Stale Centroids   : Class-IL = {res['stale_centroids']['acc_T']:.2f}% | BWT = {res['stale_centroids']['bwt']:+.2f} pp")
        print(f"    M1 (SLDA Whitened): Class-IL = {res['m1_slda']['acc_T']:.2f}% | BWT = {res['m1_slda']['bwt']:+.2f} pp")
        print(f"    Control M1 (Rand) : Class-IL = {res['control_m1']['acc_T']:.2f}% | BWT = {res['control_m1']['bwt']:+.2f} pp")
        print(f"    M2 (SDC Drift)    : Class-IL = {res['m2_sdc']['acc_T']:.2f}% | BWT = {res['m2_sdc']['bwt']:+.2f} pp")
        print(f"    Control M2 (Rand) : Class-IL = {res['control_m2']['acc_T']:.2f}% | BWT = {res['control_m2']['bwt']:+.2f} pp")
        print(f"    Linear Probe Ceil : ACC = {res['probe_ceiling']:.2f}%")

        out_data = {
            "git_commit_sha": git_sha,
            "tuning_info": tuning_info,
            "completed_runs": list(completed_cells.values())
        }
        with open(OUTPUT_JSON_PATH + ".tmp", "w") as f:
            json.dump(out_data, f, indent=2)
        os.replace(OUTPUT_JSON_PATH + ".tmp", OUTPUT_JSON_PATH)

    # 4. Final Audited Summary Tables
    if len(completed_cells) == len(SEEDS):
        runs = [completed_cells[s] for s in SEEDS]

        def get_stats(extractor):
            vals = [extractor(r) for r in runs]
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return m, s, vals

        arms = [
            ("1_freeze_after_base (Control)", lambda r: r["freeze_after_base"]["acc_T"], lambda r: r["freeze_after_base"]["bwt"]),
            ("2_naive_fine_tune (Linear)", lambda r: r["linear_head"]["acc_T"], lambda r: r["linear_head"]["bwt"]),
            ("4_ncm_adapting (Stale Centroids)", lambda r: r["stale_centroids"]["acc_T"], lambda r: r["stale_centroids"]["bwt"]),
            ("control_random_trigger_M1", lambda r: r["control_m1"]["acc_T"], lambda r: r["control_m1"]["bwt"]),
            ("M1_slda_whitened (SLDA)", lambda r: r["m1_slda"]["acc_T"], lambda r: r["m1_slda"]["bwt"]),
            ("control_random_trigger_M2", lambda r: r["control_m2"]["acc_T"], lambda r: r["control_m2"]["bwt"]),
            ("M2_sdc_drift_compensated", lambda r: r["m2_sdc"]["acc_T"], lambda r: r["m2_sdc"]["bwt"]),
            ("joint_linear_probe (Ceiling)", lambda r: r["probe_ceiling"], lambda r: 0.0)
        ]

        print("\n" + "=" * 145)
        print(" DIRECTIVE W4 TASK 5 AUDITED RESULTS TABLE (EXEMPLAR-FREE HEADROOM ATTACK)")
        print("=" * 145)
        print(f"{'Method / Arm Name':<34} | {'Class-IL ACC':<16} | {'BWT Agnostic':<14} | {'Ret Gap Closed':<15} | {'% Headroom Closed':<20}")
        print("-" * 145)

        summary_results = {}
        for name, fn_acc, fn_bwt in arms:
            m_acc, s_acc, vals_acc = get_stats(fn_acc)
            m_bwt, s_bwt, _ = get_stats(fn_bwt)

            ret_gap = ((m_bwt - NAIVE_BWT) / RETENTION_DENOMINATOR) * 100.0
            headroom_closed = ((m_acc - PREDECESSOR_ACC) / HEADROOM_DENOMINATOR) * 100.0

            ret_str = f"{ret_gap:+6.2f}%" if "Probe" not in name else "N/A"
            head_str = f"{headroom_closed:+6.2f}% (/{HEADROOM_DENOMINATOR:.2f}pp)"

            print(f"{name:<34} | {m_acc:5.2f}% +/- {s_acc:4.2f}% | {m_bwt:+6.2f} pp    | {ret_str:<15} | {head_str:<20}")

            summary_results[name] = {
                "class_il_mean": m_acc, "class_il_std": s_acc, "per_seed": vals_acc,
                "bwt_mean": m_bwt, "bwt_std": s_bwt,
                "retention_gap_closed": ret_gap,
                "headroom_closed": headroom_closed
            }

        print("=" * 145)
        print("\n  [Headroom Closure Significance Test vs Direct Predecessor (41.98% +/- 1.27%)]")
        for m_name in ["M1_slda_whitened (SLDA)", "M2_sdc_drift_compensated"]:
            res_m = summary_results[m_name]["class_il_mean"]
            res_s = summary_results[m_name]["class_il_std"]
            delta = res_m - PREDECESSOR_ACC
            beats_1sigma = (res_m > PREDECESSOR_ACC + 1.27)
            status = "BEATS 1-SIGMA" if beats_1sigma else "WITHIN 1-SIGMA NOISE / NEGATIVE"
            print(f"    {m_name:<30}: {res_m:5.2f}% +/- {res_s:4.2f}% | Delta: {delta:+5.2f} pp | Status: [{status}]")

        out_data = {
            "git_commit_sha": git_sha,
            "tuning_info": tuning_info,
            "headroom_denominator_pp": HEADROOM_DENOMINATOR,
            "retention_denominator_pp": RETENTION_DENOMINATOR,
            "summary": summary_results,
            "completed_runs": list(completed_cells.values())
        }
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(out_data, f, indent=2)

    print("\n" + "=" * 115)
    print("EXIT_CODE = 0")
    print("=" * 115)


if __name__ == "__main__":
    main()
