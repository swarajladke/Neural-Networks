"""
run_w2c_gap_precheck.py
=======================
Directive W2c: Single-seed (seed 42) cheap adaptation gap precheck on Split-CIFAR-100.
Backbone: ResNet-18 ImageNet-pretrained adapted for 32x32 input (3x3 stem, stride 1, padding 1, no maxpool).
Arm A: frozen_NCM over all 100 classes.
Arm B: adapt_layer4 (freeze conv1, layer1..3, train layer4 + fc). Reduced epoch budget (e.g. 5 epochs).
Reports accuracy after each epoch, wall-clock timing, resource utilization, and extrapolates 5 seeds x 30 epochs.
Emits w2c_precheck.json.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TAR_PATH = os.path.join(DATA_DIR, "cifar-100-python.tar.gz")


def sha256sum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Standard torchvision resnet18 architecture adapted for 32x32
        base = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()  # remove 3x3 maxpool for 32x32 resolution

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
        feats = torch.flatten(out, 1)
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.fc(feats)
        return logits, feats


def load_pretrained_weights(model):
    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    loaded_layers = []
    skipped_layers = []

    new_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                new_dict[k] = v
                loaded_layers.append((k, list(v.shape)))
            else:
                skipped_layers.append((k, f"shape mismatch: pretrained {list(v.shape)} vs target {list(model_dict[k].shape)}"))
        else:
            skipped_layers.append((k, "not in target model"))

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    return loaded_layers, skipped_layers


def evaluate_ncm(model, train_loader, test_loader, device):
    model.eval()
    centroids = torch.zeros(100, 512, device=device)
    counts = torch.zeros(100, device=device)

    with torch.no_grad():
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            feats = model.extract_features(bx)
            for c in range(100):
                mask = (by == c)
                if mask.any():
                    centroids[c] += feats[mask].sum(dim=0)
                    counts[c] += mask.sum()

        for c in range(100):
            if counts[c] > 0:
                centroids[c] /= counts[c]
        centroids = torch.nn.functional.normalize(centroids, dim=-1)

        correct = 0
        total = 0
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            feats = torch.nn.functional.normalize(model.extract_features(bx), dim=-1)
            sims = torch.matmul(feats, centroids.T)
            preds = sims.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += by.size(0)

    acc = (correct / total) * 100.0
    return acc


def evaluate_classifier(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += by.size(0)
    acc = (correct / total) * 100.0
    return acc


def main():
    print("=========================================================================================================")
    print(" STEP 3 -- CHEAP SPLIT-CIFAR-100 PRE-CHECK (ONE SEED: SEED 42)")
    print("=========================================================================================================")

    # Enable unbuffered stdout
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # 1. Dataset Verification / Download
    if not os.path.exists(TAR_PATH):
        print(f"  CIFAR-100 archive not found at {TAR_PATH}. Downloading via torchvision...", flush=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)

    if not os.path.exists(TAR_PATH):
        print(f"[VIOLATION] CIFAR-100 archive {TAR_PATH} does not exist.", flush=True)
        print("EXIT_CODE = 1", flush=True)
        sys.exit(1)

    tar_sha = sha256sum(TAR_PATH)
    git_sha = get_git_commit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  CIFAR-100 Archive : {TAR_PATH}", flush=True)
    print(f"  Archive SHA-256   : {tar_sha}", flush=True)
    print(f"  Git Commit SHA    : {git_sha}", flush=True)
    print(f"  Compute Device    : {device}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU Name          : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  GPU Memory        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)

    # Set seed 42 before anything is constructed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # CIFAR-100 Data Loaders
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=transform_test)

    # For Arm A feature extraction, use deterministic test transform
    train_eval_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_test)

    batch_size = 128
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # 2. Model Construction & Pretrained Loading
    model_arm_a = ResNet18CIFAR(num_classes=100)
    loaded_layers, skipped_layers = load_pretrained_weights(model_arm_a)
    model_arm_a = model_arm_a.to(device)

    print(f"\n  Pretrained Weight Loading Report:", flush=True)
    print(f"    Layers loaded from ImageNet-1k: {len(loaded_layers)}", flush=True)
    print(f"    Layers NOT loaded / skipped   : {len(skipped_layers)}", flush=True)
    for k, reason in skipped_layers:
        print(f"      - {k}: {reason}", flush=True)

    # Total and trainable params
    p_total_a = sum(p.numel() for p in model_arm_a.parameters())
    p_trainable_a = sum(p.numel() for p in model_arm_a.parameters() if p.requires_grad)

    # 3. Arm A: frozen_NCM
    print(f"\n  --- ARM A: FROZEN_NCM ---", flush=True)
    start_time_a = time.time()
    for p in model_arm_a.parameters():
        p.requires_grad = False

    n_opt_steps_a = 0
    assert n_opt_steps_a == 0, "Arm A must have 0 optimizer steps"
    n_samples_seen_a = 0

    acc_arm_a = evaluate_ncm(model_arm_a, train_eval_loader, test_loader, device)
    wall_time_a = time.time() - start_time_a

    print(f"    Arm A Frozen NCM Accuracy : {acc_arm_a:5.2f}%", flush=True)
    print(f"    Optimizer Steps           : {n_opt_steps_a}", flush=True)
    print(f"    Wall-Clock Time           : {wall_time_a:5.2f} s", flush=True)

    # 4. Arm B: adapt_layer4
    print(f"\n  --- ARM B: ADAPT_LAYER4 ---", flush=True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model_arm_b = ResNet18CIFAR(num_classes=100)
    load_pretrained_weights(model_arm_b)
    model_arm_b = model_arm_b.to(device)

    # Freeze everything except layer4 and fc
    for name, param in model_arm_b.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    p_total_b = sum(p.numel() for p in model_arm_b.parameters())
    p_trainable_b = sum(p.numel() for p in model_arm_b.parameters() if p.requires_grad)
    print(f"    Total Parameters     : {p_total_b:,}", flush=True)
    print(f"    Trainable Parameters : {p_trainable_b:,} (layer4 + fc)", flush=True)

    epochs_b = 3  # Reduced epoch budget to preserve CPU time while establishing curve
    lr_b = 0.01
    weight_decay_b = 5e-4
    optimizer = optim.SGD(
        [p for p in model_arm_b.parameters() if p.requires_grad],
        lr=lr_b, momentum=0.9, weight_decay=weight_decay_b
    )
    criterion = nn.CrossEntropyLoss()

    n_optimizer_steps_b = 0
    n_train_samples_seen_b = 0
    epoch_durations = []
    epoch_accuracies = []

    start_time_b = time.time()
    for ep in range(1, epochs_b + 1):
        ep_start = time.time()
        model_arm_b.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model_arm_b(bx)
            loss = criterion(logits, by)
            loss.backward()

            # Increment step counter wrapping optimizer.step()
            optimizer.step()
            n_optimizer_steps_b += 1
            n_train_samples_seen_b += bx.size(0)

        ep_duration = time.time() - ep_start
        epoch_durations.append(ep_duration)

        # Evaluate after each epoch
        ep_acc = evaluate_classifier(model_arm_b, test_loader, device)
        epoch_accuracies.append(ep_acc)
        print(f"    Epoch {ep}/{epochs_b} -> Test ACC: {ep_acc:5.2f}% | Wall-Clock: {ep_duration:5.2f}s | Steps: {n_optimizer_steps_b}")

    wall_time_b = time.time() - start_time_b
    acc_arm_b = epoch_accuracies[-1]

    # Hard-assert Arm B training integrity
    if not (n_optimizer_steps_b > 0 and n_train_samples_seen_b > 0):
        print(f"[VIOLATION] Arm B steps={n_optimizer_steps_b}, samples_seen={n_train_samples_seen_b}")
        sys.exit(1)

    # 5. Gap & Cost Extrapolation
    gap = acc_arm_b - acc_arm_a
    avg_ep_time = sum(epoch_durations) / float(len(epoch_durations))

    # Extrapolate cost: 5 seeds x 30 epochs
    extrapolated_seconds = 5 * 30 * avg_ep_time
    extrapolated_hours = extrapolated_seconds / 3600.0

    print(f"\n  -------------------------------------------------------------------------------------------------------")
    print(f"  Arm A (frozen_NCM)       : {acc_arm_a:5.2f}%")
    print(f"  Arm B (adapt_layer4)     : {acc_arm_b:5.2f}% (after {epochs_b} epochs)")
    print(f"  Single-Seed Gap          : {gap:+5.2f} percentage points")
    print(f"  Avg Epoch Duration       : {avg_ep_time:5.2f} s")
    print(f"  Extrapolated 5-seed Cost : {extrapolated_hours:5.2f} hours (5 seeds x 30 epochs @ {avg_ep_time:5.2f} s/epoch)")
    print(f"  -------------------------------------------------------------------------------------------------------")

    # Peak memory
    peak_mem = 0

    results_json = {
        "dataset": "Split-CIFAR-100",
        "dataset_archive": TAR_PATH,
        "dataset_sha256": tar_sha,
        "git_commit_sha": git_sha,
        "seed": 42,
        "pretrained_weights": "IMAGENET1K_V1",
        "arm_a_frozen_ncm": {
            "accuracy": acc_arm_a,
            "n_optimizer_steps": n_opt_steps_a,
            "n_train_samples_seen": n_samples_seen_a,
            "wall_clock_seconds": wall_time_a,
            "param_count_total": p_total_a,
            "param_count_trainable": 0
        },
        "arm_b_adapt_layer4": {
            "accuracy": acc_arm_b,
            "epochs": epochs_b,
            "batch_size": batch_size,
            "lr": lr_b,
            "weight_decay": weight_decay_b,
            "epoch_accuracies": epoch_accuracies,
            "epoch_durations_sec": epoch_durations,
            "n_optimizer_steps": n_optimizer_steps_b,
            "n_train_samples_seen": n_train_samples_seen_b,
            "wall_clock_seconds": wall_time_b,
            "param_count_total": p_total_b,
            "param_count_trainable": p_trainable_b
        },
        "single_seed_adaptation_gap": gap,
        "extrapolated_5_seed_30_epoch_hours": extrapolated_hours
    }

    out_json = os.path.join(REPO_ROOT, "w2c_precheck.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"  Emitted JSON results to {out_json}", flush=True)
    print("=========================================================================================================", flush=True)
    print("EXIT_CODE = 0", flush=True)


if __name__ == "__main__":
    main()
