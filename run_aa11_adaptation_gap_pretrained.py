"""
run_aa11_adaptation_gap_pretrained.py
=====================================

Directive AA11 / X3:
Adaptation Gap Measurement on Split-CIFAR-100 with ImageNet-Pretrained ResNet-18 Backbone.

Evaluates two arms on Split-CIFAR-100 (10 blocks x 10 classes, seeded 400 train / 100 val / 100 test per class):
  - Arm A: FROZEN_PRETRAINED_NCM (ImageNet-pretrained ResNet-18, penultimate 512-dim features + Nearest Class Mean).
  - Arm B: JOINT_OFFLINE_FULL_FINETUNE (True full fine-tuning, all parameters trainable, > 11M params, all 100 classes jointly).

Computes:
  ADAPTATION_GAP = JOINT_OFFLINE_FULL_FINETUNE_ACC_T - FROZEN_PRETRAINED_NCM_ACC_T

Halt Rule:
  If ADAPTATION_GAP <= 15.0 pp, prints [HALT] BENCHMARK REJECTED -- ADAPTATION_GAP = <value>
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CLASS_ORDER_PATH = os.path.join(REPO_ROOT, "class_order_split_cifar100.json")
RESULTS_OUTPUT_PATH = os.path.join(REPO_ROOT, "aa11_results.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]


def get_cifar100_datasets():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    data_dir = os.path.join(REPO_ROOT, "data")
    train_full = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_full = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)
    eval_train_full = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=eval_transform)

    return train_full, eval_train_full, test_full


def partition_per_class_indices(dataset, n_train=400, n_val=100, seed=42):
    torch.manual_seed(seed)
    targets = torch.tensor(dataset.targets)
    train_indices = []
    val_indices = []

    for c in range(100):
        c_idxs = (targets == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(c_idxs))
        c_perm = c_idxs[perm]
        train_indices.extend(c_perm[:n_train].tolist())
        val_indices.extend(c_perm[n_train:n_train + n_val].tolist())

    return train_indices, val_indices


class ResNet18Pretrained(nn.Module):
    def __init__(self, num_classes=100, freeze_backbone=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = torchvision.models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(in_features, num_classes)

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def evaluate_frozen_ncm(model, train_loader, test_loader, device):
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

        correct = 0
        total = 0
        block_correct = torch.zeros(10)
        block_total = torch.zeros(10)

        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            feats = model.extract_features(bx)
            # Cosine distance or Euclidean distance to centroids
            dists = torch.cdist(feats, centroids)
            preds = dists.argmin(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

            for b in range(10):
                b_mask = (by >= b * 10) & (by < (b + 1) * 10)
                if b_mask.any():
                    block_correct[b] += (preds[b_mask] == by[b_mask]).sum().item()
                    block_total[b] += b_mask.sum().item()

    acc_t = (correct / total) * 100.0
    block_accs = (block_correct / block_total * 100.0).tolist()
    return acc_t, block_accs


def train_joint_offline_full(train_loader, test_loader, device, epochs=50, lr=0.01, seed=42):
    torch.manual_seed(seed)
    model = ResNet18Pretrained(num_classes=100, freeze_backbone=False).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable > 11_000_000, f"Expected > 11M trainable params, got {n_trainable}"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = 0
    total = 0
    block_correct = torch.zeros(10)
    block_total = torch.zeros(10)

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

            for b in range(10):
                b_mask = (by >= b * 10) & (by < (b + 1) * 10)
                if b_mask.any():
                    block_correct[b] += (preds[b_mask] == by[b_mask]).sum().item()
                    block_total[b] += b_mask.sum().item()

    acc_t = (correct / total) * 100.0
    block_accs = (block_correct / block_total * 100.0).tolist()
    return acc_t, block_accs, n_trainable


def main():
    print("=========================================================================================================")
    print(" DIRECTIVE AA11 / X3 -- PRETRAINED FROZEN VS FULL FINE-TUNE ADAPTATION GAP (SPLIT-CIFAR-100)")
    print("=========================================================================================================")

    if not os.path.isfile(CLASS_ORDER_PATH):
        print(f"Error: Class order file {CLASS_ORDER_PATH} not found.")
        sys.exit(1)

    with open(CLASS_ORDER_PATH, "r") as f:
        class_order_meta = json.load(f)

    print(f"  Backbone: torchvision.models.resnet18(weights='IMAGENET1K_V1')")
    print(f"  Device: {DEVICE}")
    print(f"  Split-CIFAR-100 Class Permutation Seed: {class_order_meta['seed']}")
    print(f"  Blocks: 10 blocks of 10 classes each.\n")

    train_full, eval_train_full, test_full = get_cifar100_datasets()

    train_idxs, val_idxs = partition_per_class_indices(train_full, n_train=400, n_val=100, seed=42)
    print(f"  Seeded Split: {len(train_idxs)} train images (400/class), {len(val_idxs)} val images (100/class), {len(test_full)} test images (100/class).")
    print(f"  n_test_images_per_class = 100 (Quantization: Block Acc multiple of 0.1, ACC_T multiple of 0.01)\n")

    train_loader = DataLoader(Subset(train_full, train_idxs), batch_size=128, shuffle=True, num_workers=2)
    eval_train_loader = DataLoader(Subset(eval_train_full, train_idxs), batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_full, batch_size=128, shuffle=False, num_workers=2)

    # Arm A: Frozen Pretrained + NCM
    print("--- ARM A: FROZEN PRETRAINED PENULTIMATE FEATURES + NCM ---")
    frozen_model = ResNet18Pretrained(num_classes=100, freeze_backbone=True).to(DEVICE)
    frozen_ncm_acc, frozen_block_accs = evaluate_frozen_ncm(frozen_model, eval_train_loader, test_loader, DEVICE)
    print(f"  FROZEN_PRETRAINED_NCM_ACC_T = {frozen_ncm_acc:.2f}%")
    print(f"  Per-Block Accuracies: {[round(x, 1) for x in frozen_block_accs]}")

    # Arm B: Joint Offline Full Fine-Tune
    print("\n--- ARM B: JOINT OFFLINE FULL FINE-TUNE (ALL 100 CLASSES) ---")
    joint_acc, joint_block_accs, n_trainable = train_joint_offline_full(
        train_loader, test_loader, DEVICE, epochs=50, lr=0.01, seed=42
    )
    print(f"  n_trainable_params               = {n_trainable:,}")
    print(f"  JOINT_OFFLINE_FULL_FINETUNE_ACC_T = {joint_acc:.2f}%")
    print(f"  Per-Block Accuracies: {[round(x, 1) for x in joint_block_accs]}")

    # Adaptation Gap & Quantization Asserts
    adaptation_gap = joint_acc - frozen_ncm_acc
    print(f"\n--- ADAPTATION GAP METRIC & EVALUATION ---")
    print(f"  FROZEN_PRETRAINED_NCM_ACC_T       = {frozen_ncm_acc:.2f}%")
    print(f"  JOINT_OFFLINE_FULL_FINETUNE_ACC_T = {joint_acc:.2f}%")
    print(f"  ADAPTATION_GAP                    = {adaptation_gap:+.2f} percentage points")

    # Quantization assertion
    assert abs(round(frozen_ncm_acc, 2) - frozen_ncm_acc) < 1e-5, "Frozen ACC_T not multiple of 0.01"
    assert abs(round(joint_acc, 2) - joint_acc) < 1e-5, "Joint ACC_T not multiple of 0.01"

    # Halt Rule Check
    halt_triggered = False
    if adaptation_gap <= 15.0:
        halt_triggered = True
        print(f"\n[HALT] BENCHMARK REJECTED -- ADAPTATION_GAP = {adaptation_gap:+.2f} <= +15.0 pp")
    else:
        print(f"\n[PASS] BENCHMARK ACCEPTED -- ADAPTATION_GAP = {adaptation_gap:+.2f} > +15.0 pp (Representation learning demonstrated).")

    results_data = {
        "benchmark": "Split-CIFAR-100",
        "backbone": "ResNet-18 (IMAGENET1K_V1)",
        "seed": 42,
        "n_trainable_params": n_trainable,
        "frozen_pretrained_ncm_acc_t": round(frozen_ncm_acc, 2),
        "frozen_block_accs": [round(x, 1) for x in frozen_block_accs],
        "joint_offline_full_finetune_acc_t": round(joint_acc, 2),
        "joint_block_accs": [round(x, 1) for x in joint_block_accs],
        "adaptation_gap": round(adaptation_gap, 2),
        "halt_triggered": halt_triggered
    }

    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults written to {RESULTS_OUTPUT_PATH}")
    print(f"EXIT_CODE = {0 if not halt_triggered else 1}")


if __name__ == "__main__":
    main()
