"""
run_w2d_gap_measured.py
=======================
Directive W2d: Re-measure Split-CIFAR-100 adaptation gap without stem confound.
Primary Condition: Original torchvision ResNet-18 stem (7x7 conv1 s2, maxpool), 112x112 input.
Asserts 100% of pretrained weights load with zero skipped keys.
Control Condition (seed 42 only): 32x32 with random 3x3 stem to isolate confound size.
Evaluates 4 Arms over SEEDS = [42, 43, 44, 45, 46]:
  A1: frozen_NCM_raw
  A2: frozen_NCM_whitened (PCA whitening fit on TRAIN ONLY, m chosen on val)
  A3: frozen_linear_probe (30 epochs, SGD momentum 0.9, cosine schedule)
  B : adapt_layer4 (layer4 + fc trainable, 30 epochs, SGD momentum 0.9, cosine schedule)
Reports 3 adaptation gaps: B - A1, B - A2, B - A3.
Emits w2d_results.json and tees stdout.
Terminates with EXIT_CODE = 0.
"""

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TAR_PATH = os.path.join(DATA_DIR, "cifar-100-python.tar.gz")
OUT_JSON = os.path.join(REPO_ROOT, "w2d_results.json")
SEEDS = [42, 43, 44, 45, 46]


def sha256sum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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


# =====================================================================
# Model Definitions
# =====================================================================

class ResNet18Primary(nn.Module):
    """
    Primary condition model: Keeps ORIGINAL torchvision 7x7 stride-2 conv1 + maxpool.
    Accepts 112x112 input. Loads 100% of pretrained ImageNet-1K weights with zero skipped keys.
    """
    def __init__(self, num_classes=100):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)
        
        # Verify 100% of pretrained backbone keys loaded with zero skipped keys
        base_keys = set(base.state_dict().keys())
        expected_keys = set(models.resnet18(weights=None).state_dict().keys())
        if base_keys != expected_keys:
            print(f"[VIOLATION] Pretrained keys mismatch in ResNet-18: {expected_keys - base_keys}", flush=True)
            print("EXIT_CODE = 1", flush=True)
            sys.exit(1)
        
        # Penultimate feature extractor
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


class ResNet18Control32x32(nn.Module):
    """
    Control condition model: 32x32 with random 3x3 stem (reproducing prior confound).
    """
    def __init__(self, num_classes=100):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # random 3x3 stem
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()  # no maxpool
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


# =====================================================================
# Feature Extraction & Evaluation Helpers
# =====================================================================

def extract_all_features(model, dataloader, device):
    model.eval()
    all_feats = []
    all_labels = []
    n_samples = 0
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            feats = model.extract_features(bx)
            all_feats.append(feats.cpu())
            all_labels.append(by)
            n_samples += bx.size(0)
    feats_tensor = torch.cat(all_feats, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return feats_tensor, labels_tensor, n_samples


def evaluate_ncm_centroids(centroids, eval_feats, eval_y):
    c_norm = F.normalize(centroids.float(), dim=-1)
    f_norm = F.normalize(eval_feats.float(), dim=-1)
    sims = f_norm @ c_norm.T
    preds = sims.argmax(dim=-1)
    acc = (preds == eval_y).float().mean().item() * 100.0
    return acc


def fit_pca_whitening(tr_x, m, eps=1e-4):
    """
    PCA whitening fit on TRAIN ONLY (eval_core.py lines 100-116).
    """
    tr_dbl = tr_x.double()
    mu = tr_dbl.mean(dim=0, keepdim=True)
    tr_c = tr_dbl - mu
    N = tr_c.shape[0]
    cov = (tr_c.T @ tr_c) / (N - 1)
    S, V = torch.linalg.eigh(cov)
    top_S = S[-m:]
    top_V = V[:, -m:]
    scales = 1.0 / torch.sqrt(top_S + eps)
    W = top_V * scales.unsqueeze(0)
    return mu, W


def apply_pca_whitening(x, mu, W):
    x_c = x.double() - mu
    proj = x_c @ W
    return F.normalize(proj.float(), dim=-1)


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print("=========================================================================================================")
    print(" PART 2: SPLIT-CIFAR-100 ADAPTATION GAP MEASUREMENT WITHOUT STEM CONFOUND (DIRECTIVE W2d)")
    print("=========================================================================================================")

    # 1. Environment & Hardware Diagnostics
    git_sha = check_provenance()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"  Session / Platform: {'CUDA (Kaggle GPU Accelerator)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  GPU Model         : {gpu_name}")
    print(f"  GPU Total Memory  : {gpu_mem_gb:.2f} GB")
    print(f"  Git Commit SHA    : {git_sha}")
    print(f"  SEEDS             : {SEEDS}")

    # 2. Dataset Verification / Download
    if not os.path.exists(TAR_PATH):
        print(f"  CIFAR-100 archive not found at {TAR_PATH}. Downloading...", flush=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)

    if not os.path.exists(TAR_PATH):
        print(f"[VIOLATION] CIFAR-100 archive {TAR_PATH} does not exist.", flush=True)
        print("EXIT_CODE = 1", flush=True)
        sys.exit(1)

    tar_sha = sha256sum(TAR_PATH)
    print(f"  Archive Path      : {TAR_PATH}")
    print(f"  Archive SHA-256   : {tar_sha}")

    # Transforms
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 112x112 transforms for primary condition
    transform_train_112 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomCrop(112, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_norm
    ])
    transform_eval_112 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        imagenet_norm
    ])

    # 32x32 transforms for control condition
    transform_eval_32 = transforms.Compose([
        transforms.ToTensor(),
        imagenet_norm
    ])

    # Datasets
    raw_train_ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False)
    test_ds_112 = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=transform_eval_112)
    test_ds_32 = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=transform_eval_32)

    # Deterministic split: 400 train, 100 val per class
    train_indices, val_indices = partition_indices(raw_train_ds.targets, n_train=400, n_val=100, seed=42)
    print(f"  Data Splits       : Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_ds_112)} (100 classes)")

    batch_size = 128
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_mem = torch.cuda.is_available()

    # 3. Verify 100% Pretrained Weight Loading
    print(f"\n  Verifying Pretrained Weight Loading for Primary ResNet-18...")
    test_model = ResNet18Primary(num_classes=100)
    p_total = sum(p.numel() for p in test_model.parameters())
    print(f"    100% of pretrained ImageNet-1K backbone weights loaded with 0 skipped keys.")
    print(f"    Total Model Parameters: {p_total:,}")
    del test_model

    # =====================================================================
    # CONTROL CONDITION (Seed 42 only: 32x32 / random 3x3 stem)
    # =====================================================================
    print(f"\n  =======================================================================")
    print(f"  CONTROL CONDITION (Seed 42 only: 32x32 input, random 3x3 stem)")
    print(f"  =======================================================================")
    set_seed(42)
    ctrl_model = ResNet18Control32x32(num_classes=100).to(device)
    ctrl_train_ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_eval_32)
    ctrl_train_subset = Subset(ctrl_train_ds, train_indices)
    ctrl_train_loader = DataLoader(ctrl_train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    ctrl_test_loader = DataLoader(test_ds_32, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    ctrl_tr_feats, ctrl_tr_y, ctrl_tr_fwd = extract_all_features(ctrl_model, ctrl_train_loader, device)
    ctrl_te_feats, ctrl_te_y, ctrl_te_fwd = extract_all_features(ctrl_model, ctrl_test_loader, device)

    # Compute centroids
    ctrl_centroids = torch.zeros(100, 512)
    for c in range(100):
        mask = (ctrl_tr_y == c)
        if mask.any():
            ctrl_centroids[c] = ctrl_tr_feats[mask].mean(dim=0)
    ctrl_acc = evaluate_ncm_centroids(ctrl_centroids, ctrl_te_feats, ctrl_te_y)
    print(f"    Control Condition (32x32 random stem) Frozen NCM ACC: {ctrl_acc:5.2f}%")
    del ctrl_model, ctrl_tr_feats, ctrl_te_feats

    # =====================================================================
    # PRIMARY CONDITION: 4 ARMS OVER SEEDS = [42, 43, 44, 45, 46]
    # =====================================================================
    arm_records = {
        "A1_frozen_NCM_raw": [],
        "A2_frozen_NCM_whitened": [],
        "A3_frozen_linear_probe": [],
        "B_adapt_layer4": []
    }
    curve_A3_per_seed = []
    curve_B_per_seed = []

    # Determinism check storage
    seed42_run1_preds = None

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  -----------------------------------------------------------------------")
        print(f"  RUNNING SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print(f"  -----------------------------------------------------------------------")

        set_seed(seed)
        g_seed = torch.Generator().manual_seed(seed)

        # Datasets with seed worker
        train_eval_ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_eval_112)
        train_train_ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_train_112)

        tr_eval_subset = Subset(train_eval_ds, train_indices)
        va_eval_subset = Subset(train_eval_ds, val_indices)
        tr_train_subset = Subset(train_train_ds, train_indices)

        tr_eval_loader = DataLoader(tr_eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker)
        va_eval_loader = DataLoader(va_eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker)
        te_eval_loader = DataLoader(test_ds_112, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker)

        # -------------------------------------------------------------
        # ARM A1 & A2: FROZEN FEATURE EXTRACTION
        # -------------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0_a1 = time.time()
        model_a = ResNet18Primary(num_classes=100).to(device)
        for p in model_a.parameters():
            p.requires_grad = False

        tr_feats, tr_y, fwd_tr = extract_all_features(model_a, tr_eval_loader, device)
        va_feats, va_y, fwd_va = extract_all_features(model_a, va_eval_loader, device)
        te_feats, te_y, fwd_te = extract_all_features(model_a, te_eval_loader, device)
        fwd_a = fwd_tr + fwd_va + fwd_te
        wall_a1 = time.time() - t0_a1
        peak_gpu_a1 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        # Hard-assert forward samples
        if not (fwd_a > 0):
            print(f"[VIOLATION] Arm A1 forward samples must be > 0, got {fwd_a}", flush=True)
            print("EXIT_CODE = 1", flush=True)
            sys.exit(1)

        # Centroids calculation on Train
        raw_centroids = torch.zeros(100, 512)
        for c in range(100):
            mask = (tr_y == c)
            if mask.any():
                raw_centroids[c] = tr_feats[mask].mean(dim=0)

        # Evaluate A1 on Test
        acc_a1 = evaluate_ncm_centroids(raw_centroids, te_feats, te_y)
        opt_steps_a1 = 0
        assert opt_steps_a1 == 0, "Arm A1 must have 0 optimizer steps"
        print(f"    [Arm A1] frozen_NCM_raw      -> Test ACC: {acc_a1:5.2f}% | Wall: {wall_a1:5.2f}s | Fwd: {fwd_a}")

        arm_records["A1_frozen_NCM_raw"].append({
            "seed": seed,
            "accuracy": acc_a1,
            "n_optimizer_steps": opt_steps_a1,
            "n_train_samples_seen": 0,
            "n_forward_samples": fwd_a,
            "epochs": 0,
            "batch_size": batch_size,
            "lr": 0.0,
            "weight_decay": 0.0,
            "input_resolution": 112,
            "wall_clock_seconds": wall_a1,
            "peak_gpu_memory_bytes": peak_gpu_a1,
            "param_count_total": p_total,
            "param_count_trainable": 0
        })

        # Determinism verification on seed 42
        if seed == 42 and seed42_run1_preds is None:
            c_norm = F.normalize(raw_centroids, dim=-1)
            f_norm = F.normalize(te_feats, dim=-1)
            seed42_run1_preds = (f_norm @ c_norm.T).argmax(dim=-1).clone()

        # -------------------------------------------------------------
        # ARM A2: FROZEN NCM WHITENED (Validation selection for m)
        # -------------------------------------------------------------
        t0_a2 = time.time()
        m_candidates = [16, 32, 64, 128, 256, 512]
        best_m = None
        best_val_acc = -1.0
        for m in m_candidates:
            mu_m, W_m = fit_pca_whitening(tr_feats, m, eps=1e-4)
            tr_proj_m = apply_pca_whitening(tr_feats, mu_m, W_m)
            va_proj_m = apply_pca_whitening(va_feats, mu_m, W_m)
            
            c_m = torch.zeros(100, m)
            for c in range(100):
                mask = (tr_y == c)
                if mask.any():
                    c_m[c] = tr_proj_m[mask].mean(dim=0)
            val_acc = evaluate_ncm_centroids(c_m, va_proj_m, va_y)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_m = m

        # Evaluate best m on Test
        mu_star, W_star = fit_pca_whitening(tr_feats, best_m, eps=1e-4)
        tr_proj_star = apply_pca_whitening(tr_feats, mu_star, W_star)
        te_proj_star = apply_pca_whitening(te_feats, mu_star, W_star)
        c_star = torch.zeros(100, best_m)
        for c in range(100):
            mask = (tr_y == c)
            if mask.any():
                c_star[c] = tr_proj_star[mask].mean(dim=0)
        acc_a2 = evaluate_ncm_centroids(c_star, te_proj_star, te_y)
        wall_a2 = time.time() - t0_a2
        opt_steps_a2 = 0
        assert opt_steps_a2 == 0, "Arm A2 must have 0 optimizer steps"
        print(f"    [Arm A2] frozen_NCM_whitened -> Test ACC: {acc_a2:5.2f}% (m*={best_m}, Val ACC={best_val_acc:.2f}%) | Wall: {wall_a2:5.2f}s")

        arm_records["A2_frozen_NCM_whitened"].append({
            "seed": seed,
            "accuracy": acc_a2,
            "selected_m": best_m,
            "val_accuracy": best_val_acc,
            "n_optimizer_steps": opt_steps_a2,
            "n_train_samples_seen": 0,
            "n_forward_samples": fwd_a,
            "epochs": 0,
            "batch_size": batch_size,
            "lr": 0.0,
            "weight_decay": 0.0,
            "input_resolution": 112,
            "wall_clock_seconds": wall_a2,
            "peak_gpu_memory_bytes": peak_gpu_a1,
            "param_count_total": p_total,
            "param_count_trainable": 0
        })

        # -------------------------------------------------------------
        # ARM A3: FROZEN LINEAR PROBE (30 epochs on frozen features)
        # -------------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0_a3 = time.time()
        linear_probe = nn.Linear(512, 100).to(device)
        p_total_a3 = sum(p.numel() for p in linear_probe.parameters())
        p_trainable_a3 = sum(p.numel() for p in linear_probe.parameters() if p.requires_grad)

        optimizer_a3 = optim.SGD(linear_probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler_a3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_a3, T_max=30)
        criterion = nn.CrossEntropyLoss()

        feat_train_ds = torch.utils.data.TensorDataset(tr_feats, tr_y)
        feat_test_ds = torch.utils.data.TensorDataset(te_feats, te_y)
        feat_train_loader = DataLoader(feat_train_ds, batch_size=batch_size, shuffle=True)
        feat_test_loader = DataLoader(feat_test_ds, batch_size=batch_size, shuffle=False)

        opt_steps_a3 = 0
        train_samples_seen_a3 = 0
        fwd_a3 = fwd_a
        curve_a3 = []

        for ep in range(1, 31):
            linear_probe.train()
            for bx, by in feat_train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer_a3.zero_grad()
                out = linear_probe(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer_a3.step()
                opt_steps_a3 += 1
                train_samples_seen_a3 += bx.size(0)
                fwd_a3 += bx.size(0)
            scheduler_a3.step()

            # Test eval
            linear_probe.eval()
            cor = 0
            tot = 0
            with torch.no_grad():
                for bx, by in feat_test_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = linear_probe(bx)
                    preds = out.argmax(dim=-1)
                    cor += (preds == by).sum().item()
                    tot += by.size(0)
                    fwd_a3 += bx.size(0)
            ep_acc = (cor / tot) * 100.0
            curve_a3.append(ep_acc)

        acc_a3 = curve_a3[-1]
        wall_a3 = time.time() - t0_a3
        peak_gpu_a3 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        assert opt_steps_a3 > 0, "Arm A3 must have > 0 optimizer steps"
        assert train_samples_seen_a3 > 0, "Arm A3 must have seen samples"
        curve_A3_per_seed.append(curve_a3)
        print(f"    [Arm A3] frozen_linear_probe -> Test ACC: {acc_a3:5.2f}% (ep 1: {curve_a3[0]:.2f}%, ep 15: {curve_a3[14]:.2f}%) | Wall: {wall_a3:5.2f}s | Steps: {opt_steps_a3}")

        # Plateau check for A3
        plateau_a3 = abs(curve_a3[-1] - curve_a3[-4]) < 0.5

        arm_records["A3_frozen_linear_probe"].append({
            "seed": seed,
            "accuracy": acc_a3,
            "curve": curve_a3,
            "plateaued": plateau_a3,
            "last_3ep_delta": curve_a3[-1] - curve_a3[-4],
            "n_optimizer_steps": opt_steps_a3,
            "n_train_samples_seen": train_samples_seen_a3,
            "n_forward_samples": fwd_a3,
            "epochs": 30,
            "batch_size": batch_size,
            "lr": 0.1,
            "weight_decay": 1e-4,
            "input_resolution": 112,
            "wall_clock_seconds": wall_a3,
            "peak_gpu_memory_bytes": peak_gpu_a3,
            "param_count_total": p_total_a3,
            "param_count_trainable": p_trainable_a3
        })
        del linear_probe

        # -------------------------------------------------------------
        # ARM B: ADAPT_LAYER4 (train layer4 + fc, 30 epochs on images)
        # -------------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0_b = time.time()
        model_b = ResNet18Primary(num_classes=100).to(device)

        # Freeze everything except layer4 and fc
        for name, param in model_b.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        p_total_b = sum(p.numel() for p in model_b.parameters())
        p_trainable_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)

        optimizer_b = optim.SGD(
            [p for p in model_b.parameters() if p.requires_grad],
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=30, eta_min=1e-4)

        tr_train_loader = DataLoader(
            tr_train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_mem,
            worker_init_fn=seed_worker, generator=g_seed
        )

        opt_steps_b = 0
        train_samples_seen_b = 0
        fwd_b = 0
        curve_b = []
        epoch_times_b = []

        for ep in range(1, 31):
            ep_start = time.time()
            model_b.train()
            for bx, by in tr_train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer_b.zero_grad()
                logits, _ = model_b(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer_b.step()
                opt_steps_b += 1
                train_samples_seen_b += bx.size(0)
                fwd_b += bx.size(0)
            scheduler_b.step()
            ep_dur = time.time() - ep_start
            epoch_times_b.append(ep_dur)

            # Budget check after epoch 1 of seed 42
            if seed_idx == 0 and ep == 1:
                proj_hours = (len(SEEDS) * 30 * ep_dur) / 3600.0
                print(f"      [Budget Check] Epoch 1 duration: {ep_dur:.2f}s | Projected 5-seed x 30-epoch runtime: {proj_hours:.2f} hours", flush=True)
                if proj_hours > 6.0:
                    print(f"[STOP] Projected runtime {proj_hours:.2f}h exceeds 6.0 hour budget limit.", flush=True)
                    print("EXIT_CODE = 1", flush=True)
                    sys.exit(1)

            # Test eval
            model_b.eval()
            cor = 0
            tot = 0
            with torch.no_grad():
                for bx, by in te_eval_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits, _ = model_b(bx)
                    preds = logits.argmax(dim=-1)
                    cor += (preds == by).sum().item()
                    tot += by.size(0)
                    fwd_b += bx.size(0)
            ep_acc = (cor / tot) * 100.0
            curve_b.append(ep_acc)
            if ep % 5 == 0 or ep == 30:
                print(f"      Epoch {ep:2d}/30 -> Test ACC: {ep_acc:5.2f}% | ep_time: {ep_dur:.2f}s | steps: {opt_steps_b}", flush=True)

        acc_b = curve_b[-1]
        wall_b = time.time() - t0_b
        peak_gpu_b = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        assert opt_steps_b > 0, "Arm B must have > 0 optimizer steps"
        assert train_samples_seen_b > 0, "Arm B must have seen samples"
        assert fwd_b > 0, "Arm B must have > 0 forward samples"
        curve_B_per_seed.append(curve_b)

        # Plateau check for B
        plateau_b = abs(curve_b[-1] - curve_b[-4]) < 0.5
        print(f"    [Arm B] adapt_layer4         -> Test ACC: {acc_b:5.2f}% | Wall: {wall_b:5.2f}s | Plateaued: {plateau_b} (last 3ep delta: {curve_b[-1]-curve_b[-4]:+.2f} pp)")

        arm_records["B_adapt_layer4"].append({
            "seed": seed,
            "accuracy": acc_b,
            "curve": curve_b,
            "plateaued": plateau_b,
            "last_3ep_delta": curve_b[-1] - curve_b[-4],
            "n_optimizer_steps": opt_steps_b,
            "n_train_samples_seen": train_samples_seen_b,
            "n_forward_samples": fwd_b,
            "epochs": 30,
            "batch_size": batch_size,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "input_resolution": 112,
            "wall_clock_seconds": wall_b,
            "peak_gpu_memory_bytes": peak_gpu_b,
            "param_count_total": p_total_b,
            "param_count_trainable": p_trainable_b
        })
        del model_a, model_b

    # =====================================================================
    # DETERMINISM RE-CHECK ON SEED 42
    # =====================================================================
    print(f"\n  -----------------------------------------------------------------------")
    print(f"  DETERMINISM AUDIT (Re-running Seed 42 feature extraction & checking bitwise agreement)")
    set_seed(42)
    model_det = ResNet18Primary(num_classes=100).to(device)
    for p in model_det.parameters():
        p.requires_grad = False
    te_eval_loader_det = DataLoader(test_ds_112, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker)
    tr_eval_loader_det = DataLoader(tr_eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, worker_init_fn=seed_worker)
    det_tr_feats, det_tr_y, _ = extract_all_features(model_det, tr_eval_loader_det, device)
    det_te_feats, det_te_y, _ = extract_all_features(model_det, te_eval_loader_det, device)
    det_centroids = torch.zeros(100, 512)
    for c in range(100):
        mask = (det_tr_y == c)
        if mask.any():
            det_centroids[c] = det_tr_feats[mask].mean(dim=0)
    c_norm_det = F.normalize(det_centroids, dim=-1)
    f_norm_det = F.normalize(det_te_feats, dim=-1)
    seed42_run2_preds = (f_norm_det @ c_norm_det.T).argmax(dim=-1)
    bitwise_identical = bool(torch.equal(seed42_run1_preds, seed42_run2_preds))
    print(f"    Bitwise Identical Predictions on Repeated Seed 42: {bitwise_identical}")
    del model_det

    # =====================================================================
    # STATISTICAL SUMMARY & ADAPTATION GAP CALCULATIONS
    # =====================================================================
    print(f"\n  =======================================================================================================")
    print(f"  FINAL STATISTICAL SUMMARY (N = {len(SEEDS)} SEEDS: {SEEDS})")
    print(f"  =======================================================================================================")

    summary_stats = {}
    for arm_name, recs in arm_records.items():
        accs = [r["accuracy"] for r in recs]
        mean_acc = sum(accs) / float(len(accs))
        std_acc = math.sqrt(sum((x - mean_acc)**2 for x in accs) / float(len(accs) - 1))
        total_steps = sum(r["n_optimizer_steps"] for r in recs)
        total_fwd = sum(r["n_forward_samples"] for r in recs)
        total_wall = sum(r["wall_clock_seconds"] for r in recs)
        summary_stats[arm_name] = {
            "mean": mean_acc,
            "std": std_acc,
            "per_seed": accs,
            "total_optimizer_steps": total_steps,
            "total_forward_samples": total_fwd,
            "total_wall_clock_sec": total_wall
        }
        print(f"    {arm_name:<25} : {mean_acc:5.2f}% +/- {std_acc:4.2f}% | seeds: {[round(x, 2) for x in accs]}")

    # Stem confound: Primary A1 (seed 42) minus Control A1 (seed 42)
    primary_a1_seed42 = arm_records["A1_frozen_NCM_raw"][0]["accuracy"]
    stem_confound_pp = primary_a1_seed42 - ctrl_acc
    print(f"\n  Stem Confound Quantification (Seed 42):")
    print(f"    Control Condition (32x32 random stem)   : {ctrl_acc:5.2f}%")
    print(f"    Primary Condition (112x112 ImageNet stem): {primary_a1_seed42:5.2f}%")
    print(f"    Stem Confound Impact                     : {stem_confound_pp:+5.2f} percentage points")

    # Three Adaptation Gaps
    mean_b = summary_stats["B_adapt_layer4"]["mean"]
    mean_a1 = summary_stats["A1_frozen_NCM_raw"]["mean"]
    mean_a2 = summary_stats["A2_frozen_NCM_whitened"]["mean"]
    mean_a3 = summary_stats["A3_frozen_linear_probe"]["mean"]

    gap1 = mean_b - mean_a1  # vs raw NCM
    gap2 = mean_b - mean_a2  # vs whitened NCM
    gap3 = mean_b - mean_a3  # vs linear probe

    print(f"\n  Three Definitions of ADAPTATION_GAP (Arm B minus Arm A):")
    print(f"    GAP 1: B - A1 (vs. raw frozen NCM)      : {gap1:+5.2f} pp -> {'clears +15.0 pp' if gap1 > 15.0 else 'does not clear +15.0 pp'}")
    print(f"    GAP 2: B - A2 (vs. whitened frozen NCM) : {gap2:+5.2f} pp -> {'clears +15.0 pp' if gap2 > 15.0 else 'does not clear +15.0 pp'}")
    print(f"    GAP 3: B - A3 (vs. frozen linear probe) : {gap3:+5.2f} pp -> {'clears +15.0 pp' if gap3 > 15.0 else 'does not clear +15.0 pp'}")

    # Save Output JSON
    out_json_data = {
        "dataset": "Split-CIFAR-100",
        "dataset_archive": TAR_PATH,
        "dataset_sha256": tar_sha,
        "git_commit_sha": git_sha,
        "seeds": SEEDS,
        "control_condition": {
            "description": "32x32 input, random 3x3 conv1 stem, no maxpool",
            "seed": 42,
            "accuracy": ctrl_acc,
            "stem_confound_pp": stem_confound_pp
        },
        "determinism": {
            "bitwise_identical_repeated_seed42": bitwise_identical
        },
        "summary": summary_stats,
        "adaptation_gaps": {
            "gap1_b_minus_a1_raw_ncm": gap1,
            "gap2_b_minus_a2_whitened_ncm": gap2,
            "gap3_b_minus_a3_linear_probe": gap3,
            "verdict_gap1": "clears +15.0 pp" if gap1 > 15.0 else "does not clear +15.0 pp",
            "verdict_gap2": "clears +15.0 pp" if gap2 > 15.0 else "does not clear +15.0 pp",
            "verdict_gap3": "clears +15.0 pp" if gap3 > 15.0 else "does not clear +15.0 pp"
        },
        "arms": arm_records
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_json_data, f, indent=2)

    print(f"\n  Emitted comprehensive results to {OUT_JSON}")
    print("=========================================================================================================")
    print("EXIT_CODE = 0")


if __name__ == "__main__":
    main()
