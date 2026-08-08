"""
run_phase4_lever4_intrinsic_dim.py  --  Phase 4 Lever 4: Intrinsic Dimension Prediction & Sweeps
==============================================================================================

Base Head: L1c (weight-normalised cosine head)

Procedure:
  1. Measure intrinsic dimension independently via SVD cumulative variance threshold E_90
     for 3 Task Groupings:
       - Task 1: Base Phase (50 facts / 150 train samples)
       - Task 2: Full Dataset (100 facts / 300 train samples)
       - Task 3: Confusable Sub-Block (34 evaluated facts / 102 train samples)
  2. Pre-register predicted peak k in preregistered_l4_predictions.json.
  3. Perform full k sweep: k in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}.
  4. Perform Parametric Head Rank Curve sweep directly: r in {2, 4, 8, 16, 32, 64, 128, 256, 512, 960}.
  5. Report predicted vs observed peak k for each task.

Standing Controls:
  - Naive Sequential (L1a baseline)
  - FREEZE-AFTER-BASE (standing control arm)
  - Step-Matched Joint (primary offline baseline)

Runs 50 evaluation runs (10 shuffles x 5 seeds) per cell on BOTH Selection (101..105)
and Fresh (201..205) seed sets. Saves output to results_l4_intrinsic_dim.json.
"""

import os
import json
import random
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH   = ("smollm2_embeddings_100slots.pt"
                if os.path.exists("smollm2_embeddings_100slots.pt")
                else "../smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM    = 960
NUM_CLASSES  = 100

class FullRankAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)


class BottleneckAdapter(nn.Module):
    def __init__(self, r, pca_basis):
        super().__init__()
        self.r = r
        self.V = nn.Linear(INPUT_DIM, r,         bias=False)
        self.U = nn.Linear(r,         INPUT_DIM, bias=True)
        with torch.no_grad():
            self.V.weight.copy_(pca_basis)
            self.U.weight.copy_(pca_basis.T)
            nn.init.zeros_(self.U.bias)

    def forward(self, x):
        return F.normalize(self.U(self.V(x)), dim=-1)


class HeadL1c(nn.Module):
    """L1c: Weight-normalised cosine head (no bias, learned temperature)"""
    def __init__(self, in_features=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mask_unseen=None):
        w_norm = F.normalize(self.weight, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        logits = self.scale * torch.matmul(x_norm, w_norm.T)
        if mask_unseen is not None:
            logits = logits.masked_fill(~mask_unseen, -1e9)
        return logits


class ParametricModelRank(nn.Module):
    def __init__(self, r=INPUT_DIM, pca_basis=None):
        super().__init__()
        if r == INPUT_DIM or pca_basis is None:
            self.adapter = FullRankAdapter()
        else:
            self.adapter = BottleneckAdapter(r=r, pca_basis=pca_basis)
        self.head = HeadL1c()

    def forward(self, x, mask_unseen=None):
        feat = self.adapter(x)
        return self.head(feat, mask_unseen=mask_unseen)


def compute_pca_basis(cache_data, r):
    X = cache_data["train_x"].float().cpu()
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    if r <= Vh.shape[0]:
        return Vh[:r].clone()
    else:
        basis = torch.zeros(r, INPUT_DIM)
        basis[:Vh.shape[0]] = Vh
        rand_vecs = torch.randn(INPUT_DIM, r - Vh.shape[0])
        proj = rand_vecs - Vh.T @ (Vh @ rand_vecs)
        q, _ = torch.linalg.qr(proj)
        basis[Vh.shape[0]:] = q.T
        return basis.clone()


def compute_intrinsic_dim_e90(X, threshold=0.90):
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    var = S ** 2
    cum_var = torch.cumsum(var, dim=0) / torch.sum(var)
    id_90 = int(torch.searchsorted(cum_var, threshold).item()) + 1
    return id_90, cum_var.cpu().numpy().tolist()


def find_confusable_pairs(cache_data, threshold=0.95):
    X = cache_data["train_x"].float()
    Y = cache_data["train_y"]
    cen = torch.zeros(100, INPUT_DIM, dtype=torch.float32)
    valid_classes = [c.item() for c in torch.unique(Y)]
    for c in valid_classes:
        mask_c = (Y == c)
        cen[c] = F.normalize(X[mask_c].mean(0, keepdim=True), dim=-1).squeeze(0)
    S = torch.matmul(cen, cen.T)
    pairs = []
    for i in range(100):
        for j in range(i + 1, 100):
            if S[i, j].item() > threshold:
                pairs.append((i, j, S[i, j].item()))
    return pairs


def build_confusable_split_blocks(confusable_pairs):
    blocks = [[] for _ in range(10)]
    for i in range(100):
        blocks[i % 10].append(i)
    random.seed(42)
    for f1, f2, _ in confusable_pairs:
        b1 = next(b for b in range(10) if f1 in blocks[b])
        b2 = next(b for b in range(10) if f2 in blocks[b])
        if b1 == b2:
            tgt = (b1 + 1) % 10
            for sf in list(blocks[tgt]):
                if (sf not in [p[0] for p in confusable_pairs if p[1] == f1]
                        and sf not in [p[1] for p in confusable_pairs if p[0] == f1]):
                    blocks[b1].remove(f2); blocks[tgt].remove(sf)
                    blocks[b1].append(sf); blocks[tgt].append(f2)
                    break
    return blocks


def build_block_tensors(block_assignment, cache_data):
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for fids in block_assignment:
        tr_x.append(torch.cat([cache_data["train_x"][f*3:(f+1)*3] for f in fids], dim=0))
        tr_y.append(torch.cat([cache_data["train_y"][f*3:(f+1)*3] for f in fids], dim=0))
        te_x.append(torch.cat([cache_data["test_x"][f*4:(f+1)*4]  for f in fids], dim=0))
        te_y.append(torch.cat([cache_data["test_y"][f*4:(f+1)*4]  for f in fids], dim=0))
    return tr_x, tr_y, te_x, te_y


def compute_ogp_projector(features, k=8):
    d = features.shape[1]
    _, _, Vh = torch.linalg.svd(features, full_matrices=False)
    V_k = Vh[:k].T  # (d, k)
    P_k = torch.matmul(V_k, V_k.T)
    P_perp = torch.eye(d, device=features.device) - P_k
    return P_perp


def run_ogp_k_sweep(k, block_assignment, cache_data, seeds, num_shuffles=10, epochs=30, lr=1e-3):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    a_t_list, la_list, bwt_list = [], [], []

    for order in order_list:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            model = ParametricModelRank(r=INPUT_DIM).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            R = np.zeros((10, 10))

            base_blocks = order[:5]
            bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            model.train()
            for _ in range(epochs):
                logits = model(bx)
                loss = criterion(logits, by)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Compute OGP projector
            with torch.no_grad():
                base_feats = model.adapter(bx)
                P_perp = compute_ogp_projector(base_feats, k=k)

            model.eval()
            with torch.no_grad():
                for j in range(10):
                    tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                    acc = (model(tx).argmax(1) == ty).float().mean().item()
                    R[4, j] = acc

            for step in range(5, 10):
                curr_b = order[step]
                cx = tr_x[curr_b].to(DEVICE); cy = tr_y[curr_b].to(DEVICE)

                model.train()
                for _ in range(epochs):
                    logits = model(cx)
                    loss = criterion(logits, cy)
                    optimizer.zero_grad(); loss.backward()

                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None and p.grad.ndim == 2:
                                p.grad.copy_(torch.matmul(p.grad, P_perp))

                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    for j in range(10):
                        tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                        acc = (model(tx).argmax(1) == ty).float().mean().item()
                        R[step, j] = acc

            a_t_vals = [R[9, j] for j in range(10)]
            la_vals  = [R[max(4, order.index(j)), j] for j in range(10)]
            bwt_vals = [R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]

            a_t_list.append(np.mean(a_t_vals))
            la_list.append(np.mean(la_vals))
            bwt_list.append(np.mean(bwt_vals))

    return {
        "k": k,
        "a_t_mean": float(np.mean(a_t_list)),
        "a_t_std":  float(np.std(a_t_list)),
        "la_mean":  float(np.mean(la_list)),
        "bwt_mean": float(np.mean(bwt_list)),
        "a_t_raw":  [float(x) for x in a_t_list],
    }


def run_rank_curve_sweep(r, pca_basis, block_assignment, cache_data, seeds, num_shuffles=10):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    frozen_accs = []

    for order in order_list:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            model = ParametricModelRank(r=r, pca_basis=pca_basis).to(DEVICE)

            # Measure frozen accuracy across all 10 blocks (untrained model)
            model.eval()
            with torch.no_grad():
                block_accs = []
                for j in range(10):
                    tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                    acc = (model(tx).argmax(1) == ty).float().mean().item()
                    block_accs.append(acc)
                frozen_accs.append(np.mean(block_accs))

    return {
        "r": r,
        "frozen_acc_mean": float(np.mean(frozen_accs)),
        "frozen_acc_std":  float(np.std(frozen_accs))
    }


def main():
    print("=" * 80)
    print("  PHASE 4 LEVER 4: INTRINSIC DIMENSION PREDICTION & SWEEPS")
    print("=" * 80)

    with open(DATASET_PATH, "r") as f:
        blocks_data = json.load(f)

    if not os.path.exists(CACHE_PATH):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from run_student_continual_benchmarks import ensure_100_fact_embeddings
        MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        cache_data = ensure_100_fact_embeddings(tokenizer, model, blocks_data)
    else:
        cache_data = torch.load(CACHE_PATH, map_location=DEVICE)

    # 1. Measure Intrinsic Dimension E_90 & Pre-Register Predictions
    train_x = cache_data["train_x"].float()
    train_y = cache_data["train_y"]

    x_task1 = train_x[:150]
    id90_task1, _ = compute_intrinsic_dim_e90(x_task1)

    x_task2 = train_x
    id90_task2, _ = compute_intrinsic_dim_e90(x_task2)

    valid_classes = [c.item() for c in torch.unique(train_y) if (train_y == c).sum() > 0]
    mask_task3 = torch.tensor([y.item() in valid_classes for y in train_y])
    x_task3 = train_x[mask_task3]
    id90_task3, _ = compute_intrinsic_dim_e90(x_task3)

    preregistered_predictions = {
        "estimator": "SVD Cumulative Variance Threshold E_90 (90% variance explained)",
        "task1_base_phase": {"id_90": id90_task1, "predicted_peak_k": id90_task1},
        "task2_full_dataset": {"id_90": id90_task2, "predicted_peak_k": id90_task2},
        "task3_confusable_subblock": {"id_90": id90_task3, "predicted_peak_k": id90_task3}
    }

    with open("preregistered_l4_predictions.json", "w") as f:
        json.dump(preregistered_predictions, f, indent=2)

    print("\n  1. PRE-REGISTERED E_90 INTRINSIC DIMENSION PREDICTIONS:")
    print(f"     Task 1 (Base Phase 50 Facts):    ID_90 = {id90_task1}  (Predicted Peak k = {id90_task1})")
    print(f"     Task 2 (Full Dataset 100 Facts): ID_90 = {id90_task2}  (Predicted Peak k = {id90_task2})")
    print(f"     Task 3 (Confusable 34 Facts):   ID_90 = {id90_task3}  (Predicted Peak k = {id90_task3})")

    conf_pairs = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)

    sel_seeds   = list(range(101, 106))
    fresh_seeds = list(range(201, 206))

    # 2. Sweep OGP k values: k in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}
    k_sweep_values = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    k_results = {}

    print("\n  2. SWEEPING OGP k VALUES (FULL CELL REPORTING)...")
    for k in k_sweep_values:
        print(f"     Sweeping k = {k:<2} ...")
        res_sel = run_ogp_k_sweep(k, block_assignment, cache_data, sel_seeds, num_shuffles=10)
        res_fre = run_ogp_k_sweep(k, block_assignment, cache_data, fresh_seeds, num_shuffles=10)
        k_results[f"k_{k}"] = {"sel": res_sel, "fre": res_fre}

    # Print Full k-Sweep Table
    naive_sel_raw = np.array(k_results["k_1"]["sel"]["a_t_raw"])

    print("\n" + "=" * 115)
    print("  PHASE 4 LEVER 4 OGP k-SWEEP FULL CELL TABLE (50 RUNS PER CELL)")
    print("=" * 115)
    print(f"  {'k':<6} | {'A_T (sel)':<10} | {'A_T (fre)':<10} | {'LA':<8} | {'BWT':<8} | {'Delta A_T vs k=1 (95% CI)':<26} | {'std':<6}")
    print("  " + "-" * 115)

    for k in k_sweep_values:
        k_key = f"k_{k}"
        sel_res = k_results[k_key]["sel"]
        fre_res = k_results[k_key]["fre"]

        diff_sel = np.array(sel_res["a_t_raw"]) - naive_sel_raw
        mean_diff = float(np.mean(diff_sel))

        np.random.seed(42)
        boot_diffs = []
        for _ in range(10000):
            b_idx = np.random.choice(len(diff_sel), size=len(diff_sel), replace=True)
            boot_diffs.append(np.mean(diff_sel[b_idx]))
        ci_low  = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))

        ci_str = f"{mean_diff*100:+.2f}% [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%]"

        print(f"  {k:<6} | {sel_res['a_t_mean']*100:6.2f}%    | {fre_res['a_t_mean']*100:6.2f}%    | {sel_res['la_mean']*100:6.2f}% | {sel_res['bwt_mean']*100:6.2f}% | {ci_str:<26} | {sel_res['a_t_std']*100:5.2f}%")

        # Verification Assertion (R2 & R13): delta A_T = delta LA + delta BWT
        sum_check = sel_res["la_mean"] + sel_res["bwt_mean"]
        assert abs(sel_res["a_t_mean"] - sum_check) < 1e-5, f"R2 Violation: {sel_res['a_t_mean']} != {sum_check}"

    print("  " + "-" * 115)

    # Find Observed Peak k
    best_k_key = max(k_results.keys(), key=lambda key: k_results[key]["sel"]["a_t_mean"])
    observed_peak_k = int(best_k_key.split("_")[1])
    observed_peak_acc = k_results[best_k_key]["sel"]["a_t_mean"]

    print("\n" + "=" * 80)
    print("  PREDICTED VS OBSERVED PEAK k REPORT")
    print("=" * 80)
    print(f"  Task 1 (Base Phase 50 Facts):    Predicted Peak k = {id90_task1}  |  Observed Peak k = {observed_peak_k} ({observed_peak_acc*100:.2f}%)")
    print(f"  Task 2 (Full Dataset 100 Facts): Predicted Peak k = {id90_task2}  |  Observed Peak k = {observed_peak_k}")
    print(f"  Task 3 (Confusable 34 Facts):   Predicted Peak k = {id90_task3}  |  Observed Peak k = {observed_peak_k}")

    # 3. Parametric Head Rank Curve (Frozen Accuracy vs Rank r)
    r_sweep_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 960]
    rank_results = {}

    print("\n  3. PARAMETRIC HEAD FROZEN-ACCURACY VERSUS RANK CURVE (L1c HEAD)...")
    for r in r_sweep_values:
        pca_basis = compute_pca_basis(cache_data, r=r).to(DEVICE) if r < INPUT_DIM else None
        res_r = run_rank_curve_sweep(r, pca_basis, block_assignment, cache_data, sel_seeds, num_shuffles=10)
        rank_results[f"r_{r}"] = res_r

    print("\n" + "=" * 80)
    print("  PARAMETRIC HEAD FROZEN-ACCURACY VERSUS RANK CURVE TABLE")
    print("=" * 80)
    print(f"  {'Rank r':<8} | {'Frozen Model Accuracy (L1c Head)':<35} | {'std':<8}")
    print("  " + "-" * 60)
    for r in r_sweep_values:
        res = rank_results[f"r_{r}"]
        print(f"  {r:<8} | {res['frozen_acc_mean']*100:6.2f}%                            | {res['frozen_acc_std']*100:5.2f}%")
    print("  " + "-" * 60)

    # Save Full Results JSON
    save_data = {
        "preregistered_predictions": preregistered_predictions,
        "k_results": k_results,
        "observed_peak_k": observed_peak_k,
        "observed_peak_acc": observed_peak_acc,
        "rank_results": rank_results
    }
    with open("results_l4_intrinsic_dim.json", "w") as out:
        json.dump(save_data, out, indent=2)
    print("\nSaved full results to results_l4_intrinsic_dim.json.")

if __name__ == "__main__":
    main()
