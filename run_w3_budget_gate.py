"""
run_w3_budget_gate.py
=====================
Directive W3 -- Part 1: Budget Gate Verification (Amended).

Runs ONE arm (Naive Fine-Tune) on ONE seed (Seed 42) across all 10 tasks of Split-CIFAR-100 at 112x112.
Measures wall-clock time for continual training AND linear probe fitting.
Projects the total time for the full 9 arms x 5 seeds study INCLUDING:
  - All 9 continual learning arms x 5 seeds
  - 45 frozen linear probes (tri-metric component iii)
  - LwF lambda hyperparameter validation sweep
  - EWC lambda hyperparameter validation sweep

If the projected total exceeds 7.0 hours, it HALTS and reports the projection.
"""

import os
import sys
import time
import json
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ARCHIVE_PATH = os.path.join(DATA_DIR, "cifar-100-python.tar.gz")
CLASS_ORDER_PATH = os.path.join(REPO_ROOT, "class_order_split_cifar100.json")
OUTPUT_JSON_PATH = os.path.join(REPO_ROOT, "w3_budget_gate_results.json")

BATCH_SIZE = 128
EPOCHS_PER_TASK = 20  # 20 epochs x 32 steps = 640 steps/task; 10 tasks = 6,400 steps
LR = 0.005
WEIGHT_DECAY = 5e-4


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


def train_frozen_linear_probe(backbone, train_loader, test_loader, device, epochs=30):
    backbone.eval()
    all_tr_feats, all_tr_y = [], []
    all_te_feats, all_te_y = [], []

    with torch.no_grad():
        for bx, by in train_loader:
            bx = bx.to(device)
            all_tr_feats.append(backbone.extract_features(bx).cpu())
            all_tr_y.append(by)
        for bx, by in test_loader:
            bx = bx.to(device)
            all_te_feats.append(backbone.extract_features(bx).cpu())
            all_te_y.append(by)

    tr_x = torch.cat(all_tr_feats, dim=0)
    tr_y = torch.cat(all_tr_y, dim=0)
    te_x = torch.cat(all_te_feats, dim=0)
    te_y = torch.cat(all_te_y, dim=0)

    tr_ds = torch.utils.data.TensorDataset(tr_x, tr_y)
    te_ds = torch.utils.data.TensorDataset(te_x, te_y)
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


def main():
    print("=" * 95)
    print(" DIRECTIVE W3 -- PART 1: AMENDED BUDGET GATE VERIFICATION")
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

    full_tr_loader = DataLoader(Subset(ds_ev, train_idx), batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)
    full_te_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker)

    # -------------------------------------------------------------
    # EXECUTE NAIVE FINE-TUNE ON SEED 42 ACROSS ALL 10 TASKS
    # -------------------------------------------------------------
    set_seed(42)
    model = ResNet18Primary(num_classes=100).to(device)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    R_agnostic = np.zeros((10, 10))
    R_aware = np.zeros((10, 10))

    opt_steps = 0
    samples_seen = 0
    fwd_samples = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0_training = time.time()

    print(f"\n  [1/2] Running Naive Fine-Tune over 10 tasks (Epochs/task: {EPOCHS_PER_TASK}, LR: {LR})...")

    for t in range(10):
        t_start = time.time()
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
        acc_agnostic, acc_aware = evaluate_task_r(model, task_test_loaders, seen, device)
        for j in seen:
            R_agnostic[t, j] = acc_agnostic[j]
            R_aware[t, j] = acc_aware[j]
            fwd_samples += 1000

        t_elapsed = time.time() - t_start
        mean_seen_agnostic = np.mean([R_agnostic[t, j] for j in seen])
        mean_seen_aware = np.mean([R_aware[t, j] for j in seen])
        print(f"    Task {t}/9 complete ({t_elapsed:5.1f}s) | Agnostic ACC: {mean_seen_agnostic:5.2f}% | Aware ACC: {mean_seen_aware:5.2f}%")

    t_training_naive = time.time() - t0_training

    # (iii) Linear probe on final representation
    print("\n  [2/2] Measuring Linear Probe extraction & fitting time...")
    t0_probe = time.time()
    probe_acc = train_frozen_linear_probe(model, full_tr_loader, full_te_loader, device, epochs=30)
    t_probe_measured = time.time() - t0_probe
    fwd_samples += 50000

    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    final_acc_agnostic = np.mean(R_agnostic[9, :])
    final_acc_aware = np.mean(R_aware[9, :])
    bias_gap = final_acc_aware - final_acc_agnostic

    print(f"\n  --- NAIVE FINE-TUNE (SEED 42) BENCHMARK MEASUREMENTS ---")
    print(f"    Measured Continual Training Duration : {t_training_naive:5.2f}s ({t_training_naive/60.0:.2f} mins)")
    print(f"    Measured Linear Probe Fitting Duration : {t_probe_measured:5.2f}s ({t_probe_measured/60.0:.2f} mins)")
    print(f"    (i)   Task-Agnostic (Class-IL) ACC_T   : {final_acc_agnostic:5.2f}%")
    print(f"    (ii)  Task-Aware ACC_T                 : {final_acc_aware:5.2f}%")
    print(f"    Classifier Bias Gap (ii - i)           : {bias_gap:+5.2f} pp")
    print(f"    (iii) Frozen Feature Linear Probe      : {probe_acc:5.2f}%")
    print(f"    Peak GPU Memory                        : {peak_gpu / (1024**2):.2f} MB")

    # -------------------------------------------------------------
    # AMENDED COMPREHENSIVE BUDGET PROJECTION
    # -------------------------------------------------------------
    # 1. Continual learning training across 9 arms x 5 seeds:
    # Relative weights based on computational complexity:
    #   1. FREEZE-AFTER-BASE : ~0.15 (1 task only)
    #   2. Naive Fine-Tune   :  1.00
    #   3. NCM Frozen        : ~0.10 (zero opt steps)
    #   4. NCM Adapting      : ~0.10 (zero opt steps)
    #   5. LwF               : ~1.35 (forward snapshot + KL distillation)
    #   6. EWC               : ~1.25 (empirical Fisher accumulation)
    #   7. ER (buffer 500)   : ~1.40 (replay buffer sampling)
    #   8. DER++ (buffer 500): ~1.55 (replay sampling + MSE logit matching)
    #   9. Joint Offline     : ~1.00 (single 30-epoch joint run)
    ARM_WEIGHTS = {
        "freeze_after_base": 0.15,
        "naive_fine_tune": 1.00,
        "ncm_frozen": 0.10,
        "ncm_adapting": 0.10,
        "lwf": 1.35,
        "ewc": 1.25,
        "er": 1.40,
        "der_plus_plus": 1.55,
        "joint_offline": 1.00
    }
    sum_arm_weights = sum(ARM_WEIGHTS.values())  # ~ 7.90
    proj_training_sec = t_training_naive * sum_arm_weights * 5  # 5 seeds

    # 2. Tri-metric linear probe cost across 9 arms x 5 seeds = 45 probes:
    proj_probe_sec = t_probe_measured * 45

    # 3. Hyperparameter validation sweeps for NEW arms (Seed 42):
    # LwF lambda grid (5 candidates) on Task 0..1 validation split:
    proj_lwf_sweep_sec = t_training_naive * 0.25 * 5  # ~ 1.25 x t_naive
    # EWC lambda grid (5 candidates) on Task 0..1 validation split:
    proj_ewc_sweep_sec = t_training_naive * 0.25 * 5  # ~ 1.25 x t_naive

    total_projected_sec = proj_training_sec + proj_probe_sec + proj_lwf_sweep_sec + proj_ewc_sweep_sec
    total_projected_hr = total_projected_sec / 3600.0

    print(f"\n=========================================================================================================")
    print(" AMENDED COMPREHENSIVE BUDGET PROJECTION")
    print("=========================================================================================================")
    print(f"  1. 9 Arms x 5 Seeds Training Projection       : {proj_training_sec/3600.0:5.2f} hours ({proj_training_sec:.0f}s)")
    print(f"  2. 45 Frozen Linear Probes Projection          : {proj_probe_sec/3600.0:5.2f} hours ({proj_probe_sec:.0f}s)")
    print(f"  3. LwF Validation Lambda Sweep Projection      : {proj_lwf_sweep_sec/3600.0:5.2f} hours ({proj_lwf_sweep_sec:.0f}s)")
    print(f"  4. EWC Validation Lambda Sweep Projection      : {proj_ewc_sweep_sec/3600.0:5.2f} hours ({proj_ewc_sweep_sec:.0f}s)")
    print(f"  ---------------------------------------------------------------------------------------")
    print(f"  TOTAL PROJECTED STUDY RUNTIME                  : {total_projected_hr:5.2f} hours ({total_projected_sec:.0f}s)")
    print(f"  Budget Gate Limit                              :  7.00 hours (25,200s)")

    budget_passed = (total_projected_hr <= 7.00)
    verdict = "PASSED" if budget_passed else "EXCEEDED"
    print(f"  Gate Status                                    : [BUDGET GATE: {verdict}]")

    if not budget_passed:
        print(f"\n  [HALT] Total projected runtime ({total_projected_hr:.2f}h) exceeds the 7.00-hour limit.")
        print("  Terminating now. Per directive: A partial run is invalid, not partial credit.")

    # Save output JSON
    results = {
        "git_commit_sha": git_sha,
        "seed": 42,
        "arm": "naive_fine_tune",
        "epochs_per_task": EPOCHS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "measured_continual_training_sec": t_training_naive,
        "measured_linear_probe_sec": t_probe_measured,
        "peak_gpu_memory_bytes": peak_gpu,
        "n_optimizer_steps": opt_steps,
        "n_train_samples_seen": samples_seen,
        "n_forward_samples": fwd_samples,
        "task_agnostic_acc_T": float(final_acc_agnostic),
        "task_aware_acc_T": float(final_acc_aware),
        "classifier_bias_gap": float(bias_gap),
        "frozen_linear_probe_acc": float(probe_acc),
        "R_matrix_agnostic": R_agnostic.tolist(),
        "R_matrix_aware": R_aware.tolist(),
        "budget_projections": {
            "training_hours": float(proj_training_sec / 3600.0),
            "linear_probe_hours": float(proj_probe_sec / 3600.0),
            "lwf_sweep_hours": float(proj_lwf_sweep_sec / 3600.0),
            "ewc_sweep_hours": float(proj_ewc_sweep_sec / 3600.0),
            "total_projected_hours": float(total_projected_hr),
            "budget_limit_hours": 7.0,
            "gate_status": verdict,
            "proceed": bool(budget_passed)
        }
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results to {OUTPUT_JSON_PATH}")

    print("=" * 95)
    print("EXIT_CODE = 0")
    print("=" * 95)


if __name__ == "__main__":
    main()
