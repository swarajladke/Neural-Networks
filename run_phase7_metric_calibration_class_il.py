"""
run_phase7_metric_calibration_class_il.py  --  Phase 7 Local Metric Calibration & Temperature Scaling
=====================================================================================================

Optimizes logit scale alignment between Level 1 (f = 0 frozen base) and Level 2 (f = fast non-parametric cache)
under STRICT CLASS-IL EVALUATION.

Goal: Close the gap between Phase 6 final accuracy (A_T = 64.95%) and the acquisition ceiling (LA = 76.50%).

Arms Evaluated (50 Runs per Condition: 10 Shuffles x 5 Seeds per Seed Set):
  1. naive_l1c: Naive sequential training baseline.
  2. freeze_after_base: Zero parameter updates after base phase (Standing Rule 1 Control Floor).
  3. phase6_dual_continuum: Uncalibrated fixed-scale Level 1 (f=0) + Level 2 (f=fast) (64.95% baseline).
  4. phase7_temp_calibrated: Temperature-scaled Level 2 cache logits (tau scaling).
  5. phase7_margin_calibrated: Margin-adjusted Level 2 cache logits (gamma offset).
  6. phase7_full_metric_calibrated: Joint temperature scaling + margin offset over Level 2 cache.

Standing Rules Enforced:
  - R1: FREEZE-AFTER-BASE included as standing control floor.
  - R2: Decomposed retention and acquisition gaps reported.
  - R21: No hardcoded result literals; exact R21 runtime guards.
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH   = "smollm2_embeddings_100slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM    = 960
NUM_CLASSES  = 100


class HeadL1c(nn.Module):
    """Weight-normalized cosine head without bias (scale=10.0)."""
    def __init__(self, in_features=960, out_features=100, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


def find_confusable_pairs(cache_data, threshold=0.95):
    X = cache_data["train_x"].float()
    y = cache_data["train_y"]
    cen = torch.zeros(100, INPUT_DIM, dtype=torch.float32)
    for i in range(100):
        mask = (y == i)
        if mask.sum() > 0:
            cen[i] = F.normalize(X[mask].mean(0, keepdim=True), dim=-1).squeeze(0)
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
        b_tr_x, b_tr_y, b_te_x, b_te_y = [], [], [], []
        for f in fids:
            tr_idx = (cache_data["train_y"] == f).nonzero(as_tuple=True)[0]
            te_idx = (cache_data["test_y"]  == f).nonzero(as_tuple=True)[0]

            assert len(tr_idx) > 0, f"Task 5.1 Guard Failed: Fact {f} has 0 train samples in cache."
            assert len(te_idx) > 0, f"Task 5.1 Guard Failed: Fact {f} has 0 test samples in cache."
            assert (cache_data["train_y"][tr_idx] == f).all(), f"Task 5.1 Guard Failed: train_y label mismatch for fact {f}"
            assert (cache_data["test_y"][te_idx]  == f).all(), f"Task 5.1 Guard Failed: test_y label mismatch for fact {f}"

            b_tr_x.append(cache_data["train_x"][tr_idx])
            b_tr_y.append(cache_data["train_y"][tr_idx])
            b_te_x.append(cache_data["test_x"][te_idx])
            b_te_y.append(cache_data["test_y"][te_idx])

        tr_x.append(torch.cat(b_tr_x, dim=0))
        tr_y.append(torch.cat(b_tr_y, dim=0))
        te_x.append(torch.cat(b_te_x, dim=0))
        te_y.append(torch.cat(b_te_y, dim=0))
    return tr_x, tr_y, te_x, te_y


def eval_class_il_calibrated(model, te_x, te_y, R, step_pos, level2_cache=None, tau=1.0, gamma=0.0):
    """Evaluate Class-IL accuracy with optional Temperature (tau) and Margin (gamma) calibration, returning flip_count and sample_traces."""
    model.eval()
    flip_count = 0
    sample_traces = []

    with torch.no_grad():
        for j in range(10):
            tx = te_x[j].to(DEVICE)
            ty = te_y[j].to(DEVICE)

            # Uncalibrated baseline logits
            logits_uncal = model(tx)
            logits_cal   = model(tx)

            if level2_cache is not None and len(level2_cache) > 0:
                tx_norm = F.normalize(tx, dim=1)
                cache_classes = list(level2_cache.keys())
                cache_centroids = torch.stack([level2_cache[c] for c in cache_classes]).to(DEVICE)
                cache_centroids_norm = F.normalize(cache_centroids, dim=1)

                sims_uncal = 10.0 * torch.matmul(tx_norm, cache_centroids_norm.T)
                sims_cal   = (10.0 / tau) * torch.matmul(tx_norm, cache_centroids_norm.T) - gamma

                for idx, c in enumerate(cache_classes):
                    logits_uncal[:, c] = torch.max(logits_uncal[:, c], sims_uncal[:, idx])
                    logits_cal[:, c]   = torch.max(logits_cal[:, c], sims_cal[:, idx])

            preds_uncal = logits_uncal.argmax(dim=1)
            preds_cal   = logits_cal.argmax(dim=1)

            diff_mask = (preds_uncal != preds_cal)
            flip_count += int(diff_mask.sum().item())

            # Collect 10 sample traces for Task 0.3 verification
            if step_pos == 9 and j == 0 and len(sample_traces) < 10 and level2_cache is not None and len(level2_cache) > 0:
                for row_idx in range(min(10, len(tx))):
                    top2_uncal_vals, top2_uncal_idx = torch.topk(logits_uncal[row_idx], 2)
                    top2_cal_vals,   top2_cal_idx   = torch.topk(logits_cal[row_idx], 2)

                    trace = {
                        "row": row_idx,
                        "target": int(ty[row_idx].item()),
                        "uncal_top1_cls": int(top2_uncal_idx[0].item()),
                        "uncal_top1_val": float(top2_uncal_vals[0].item()),
                        "uncal_top2_cls": int(top2_uncal_idx[1].item()),
                        "uncal_top2_val": float(top2_uncal_vals[1].item()),
                        "cal_top1_cls":   int(top2_cal_idx[0].item()),
                        "cal_top1_val":   float(top2_cal_vals[0].item()),
                        "cal_top2_cls":   int(top2_cal_idx[1].item()),
                        "cal_top2_val":   float(top2_cal_vals[1].item()),
                        "source_top1":    "cache" if int(top2_cal_idx[0].item()) in level2_cache else "head"
                    }
                    sample_traces.append(trace)

            R[step_pos, j] = (preds_cal == ty).float().mean().item()

    return flip_count, sample_traces


def run_phase7_arm(arm_name, block_assignment, cache_data, seeds, num_shuffles=10, epochs_per_block=30, lr=1e-3):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)

    a_t_records, la_list, bwt_list = [], [], []
    total_flip_count = 0
    collected_traces = []

    for shuffle_idx, order in enumerate(order_list):
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = HeadL1c(in_features=960, out_features=100, scale=10.0).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            R = np.zeros((10, 10))
            level2_cache = {}

            # Base phase (Blocks 0..4) — Level 1 (f = 0 after step 4)
            base_blocks = order[:5]
            bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            model.train()
            for ep in range(epochs_per_block * 5):
                logits = model(bx)
                loss = criterion(logits, by)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Set calibration parameters based on arm type
            tau   = 0.85 if "temp" in arm_name or "full" in arm_name else 1.0
            gamma = 0.50 if "margin" in arm_name or "full" in arm_name else 0.0

            flips, traces = eval_class_il_calibrated(model, te_x, te_y, R, 4, level2_cache if "continuum" in arm_name or "calibrated" in arm_name else None, tau, gamma)
            total_flip_count += flips
            if traces and not collected_traces:
                collected_traces = traces

            # Incremental phase (Steps 5..9)
            for step_pos in range(5, 10):
                curr_b = order[step_pos]
                cx = tr_x[curr_b].to(DEVICE)
                cy = tr_y[curr_b].to(DEVICE)

                if arm_name == "freeze_after_base":
                    flips, _ = eval_class_il_calibrated(model, te_x, te_y, R, step_pos)
                    total_flip_count += flips
                    continue

                if "continuum" in arm_name or "calibrated" in arm_name:
                    # LEVEL 1 IS FROZEN (f = 0)
                    with torch.no_grad():
                        for c in cy.unique():
                            mask = (cy == c)
                            centroid = cx[mask].mean(dim=0)
                            level2_cache[int(c.item())] = centroid
                    flips, traces = eval_class_il_calibrated(model, te_x, te_y, R, step_pos, level2_cache, tau, gamma)
                    total_flip_count += flips
                    if traces and not collected_traces:
                        collected_traces = traces
                    continue

                # Naive arm
                for ep in range(epochs_per_block):
                    model.train()
                    logits_curr = model(cx)
                    loss_curr = criterion(logits_curr, cy)
                    optimizer.zero_grad(); loss_curr.backward(); optimizer.step()

                flips, _ = eval_class_il_calibrated(model, te_x, te_y, R, step_pos)
                total_flip_count += flips

            # R21 Guards
            for t in range(4, 10):
                if np.all(R[t, :] == 0.0):
                    raise RuntimeError(f"R21 Failure: R row {t} never written in {arm_name}. Halting.")

            # Audit log for named run (shuffle_idx == 0, seed == 101)
            if shuffle_idx == 0 and seed == 101:
                print(f"\n--- [Audit Log: {arm_name} | shuffle=0, seed=101] ---")
                print(f"Realized Block Order: {order}")
                print("10x10 R Matrix (Evaluation Accuracy per Step/Block):")
                for r_idx in range(10):
                    row_str = " ".join([f"{R[r_idx, c_idx]:.4f}" for c_idx in range(10)])
                    print(f"  Step {r_idx}: [{row_str}]")

            a_t = float(np.mean(R[9, :]))
            la  = float(np.mean([R[max(4, order.index(j)), j] for j in range(10)]))
            bwt = float(np.mean([R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]))

            n_j = [len(te_y[b]) for b in range(10)]
            r_mat_list = R.tolist()
            a_t_records.append({
                "shuffle": shuffle_idx,
                "seed": seed,
                "a_t": a_t,
                "la": la,
                "bwt": bwt,
                "order": order,
                "per_block_test_counts": n_j,
                "r_matrix": r_mat_list
            })
            la_list.append(la)
            bwt_list.append(bwt)

    a_t_vals = [r["a_t"] for r in a_t_records]
    return {
        "arm_name": arm_name,
        "a_t_mean": float(np.mean(a_t_vals)),
        "a_t_std":  float(np.std(a_t_vals)),
        "la_mean":  float(np.mean(la_list)),
        "bwt_mean": float(np.mean(bwt_list)),
        "total_prediction_flips": total_flip_count,
        "numeric_sample_traces": collected_traces,
        "a_t_raw":  a_t_records,
    }


def main():
    print("=" * 110)
    print("  PHASE 7 LOCAL METRIC CALIBRATION & INSTRUMENTATION VERIFICATION")
    print("=" * 110)

    # Print PyTorch & Hardware Information
    print(f"PyTorch Version: {torch.__version__}")
    cuda_ver = torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'
    print(f"PyTorch CUDA Version: {cuda_ver}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device_name}")

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

    conf_pairs = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)

    sel_seeds   = list(range(101, 106))
    fresh_seeds = list(range(201, 206))

    arms = [
        "naive_l1c",
        "freeze_after_base",
        "phase6_dual_continuum",
        "phase7_temp_calibrated",
        "phase7_margin_calibrated",
        "phase7_full_metric_calibrated"
    ]
    results = {}

    for arm in arms:
        print(f"\n  --> Running {arm} (50 Selection Runs + 50 Fresh Runs)...")
        t0 = time.perf_counter()
        res_sel = run_phase7_arm(arm, block_assignment, cache_data, sel_seeds, num_shuffles=10)
        res_fre = run_phase7_arm(arm, block_assignment, cache_data, fresh_seeds, num_shuffles=10)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"      Wall-Clock Execution Time: {elapsed:.2f} seconds")
        results[arm] = {"sel": res_sel, "fre": res_fre}

    print("\n" + "=" * 110)
    print("  PHASE 7 CLASS-IL METRIC CALIBRATION & FLIP COUNTER VERIFICATION")
    print("=" * 110)
    print(f"  {'Arm Name':<30} | {'A_T (sel)':<10} | {'A_T (fre)':<10} | {'Prediction Flips':<18}")
    print("  " + "-" * 110)
    for arm in arms:
        res_sel = results[arm]["sel"]
        res_fre = results[arm]["fre"]
        total_flips = res_sel["total_prediction_flips"] + res_fre["total_prediction_flips"]
        print(f"  {arm:<30} | {res_sel['a_t_mean']*100:6.2f}%    | {res_fre['a_t_mean']*100:6.2f}%    | {total_flips:<18}")
    print("  " + "-" * 110)

    save_path = "results_phase7_metric_calibration.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Phase 7 verified results to {save_path}.")

    from validate_results_artifact import validate_results_json
    valid = validate_results_json(save_path)
    if not valid:
        raise RuntimeError(f"Mechanical Validation FAILED on {save_path}. Halting.")


if __name__ == "__main__":
    main()
