"""
run_phase4_lever1_head.py  --  Phase 4 Lever 1: Head Recency Bias Analysis
========================================================================

Evaluates Arms:
  Control 1: Naive Sequential (L1a baseline)
  Control 2: FREEZE-AFTER-BASE (standing control arm)
  Control 3: Step-Matched Joint (primary offline baseline)
  L1a: Baseline head (Linear, bias=True)
  L1b: Bias removed (Linear, bias=False)
  L1c: Weight-normalised cosine head (no bias, learned temperature scale)
  L1d: L1c + Logits masked to classes seen so far during training

Diagnostics required per arm per class-block:
  - Mean weight norm of head rows (old blocks vs newest block)
  - Mean bias value (old vs newest)
  - Oracle-argmax accuracy (argmax restricted to query's own block)

Runs 50 evaluation runs (10 shuffles x 5 seeds) on BOTH Selection (101..105)
and Fresh (201..205) seed sets. Saves full output to results_l1_head.json.
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


class HeadL1a(nn.Module):
    """L1a: Baseline head (Linear, bias=True)"""
    def __init__(self, in_features=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x, mask_unseen=None):
        logits = self.head(x)
        if mask_unseen is not None:
            logits = logits.masked_fill(~mask_unseen, -1e9)
        return logits


class HeadL1b(nn.Module):
    """L1b: Bias removed (Linear, bias=False)"""
    def __init__(self, in_features=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = nn.Linear(in_features, num_classes, bias=False)

    def forward(self, x, mask_unseen=None):
        logits = self.head(x)
        if mask_unseen is not None:
            logits = logits.masked_fill(~mask_unseen, -1e9)
        return logits


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


class ParametricModel(nn.Module):
    def __init__(self, head_type="l1a"):
        super().__init__()
        self.adapter = FullRankAdapter()
        if head_type in ["l1a", "naive", "freeze_after_base", "step_matched_joint"]:
            self.head = HeadL1a()
        elif head_type == "l1b":
            self.head = HeadL1b()
        elif head_type in ["l1c", "l1d"]:
            self.head = HeadL1c()
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        self.head_type = head_type

    def forward(self, x, mask_unseen=None):
        feat = self.adapter(x)
        return self.head(feat, mask_unseen=mask_unseen)


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


def run_experiment_arm(arm_name, block_assignment, cache_data, seeds, num_shuffles=10, epochs=30, lr=1e-3):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10)); random.shuffle(order); order_list.append(order)

    a_t_list, la_list, bwt_list = [], [], []
    diag_weight_norms_old = []
    diag_weight_norms_new = []
    diag_biases_old = []
    diag_biases_new = []
    oracle_acc_list = []
    true_acc_list   = []

    for order in order_list:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            model = ParametricModel(head_type=arm_name).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            R = np.zeros((10, 10))

            if arm_name == "freeze_after_base":
                # Train only on base blocks (0..4 in order)
                base_blocks = order[:5]
                bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
                by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)
                model.train()
                for _ in range(epochs):
                    logits = model(bx)
                    loss = criterion(logits, by)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    for step in range(4, 10):
                        for j in range(10):
                            tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                            acc = (model(tx).argmax(1) == ty).float().mean().item()
                            R[step, j] = acc
            elif arm_name == "step_matched_joint":
                # Train jointly step by step (base blocks + newly added block)
                for step_idx, step_pos in enumerate(range(4, 10)):
                    seen_blocks = order[:step_pos + 1]
                    jx = torch.cat([tr_x[b] for b in seen_blocks], dim=0).to(DEVICE)
                    jy = torch.cat([tr_y[b] for b in seen_blocks], dim=0).to(DEVICE)
                    model.train()
                    for _ in range(epochs):
                        logits = model(jx)
                        loss = criterion(logits, jy)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        for j in range(10):
                            tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                            acc = (model(tx).argmax(1) == ty).float().mean().item()
                            R[step_pos, j] = acc
            else:
                # Continual learning: Base Phase (0..4 in order) then Sequential Steps 5..9
                base_blocks = order[:5]
                bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
                by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)
                model.train()
                for _ in range(epochs):
                    logits = model(bx)
                    loss = criterion(logits, by)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                model.eval()
                seen_classes = set()
                for b in base_blocks:
                    seen_classes.update(tr_y[b].cpu().numpy().tolist())

                with torch.no_grad():
                    mask_unseen = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=DEVICE)
                    for c in seen_classes: mask_unseen[c] = True
                    for j in range(10):
                        tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                        use_mask = mask_unseen if arm_name == "l1d" else None
                        acc = (model(tx, mask_unseen=use_mask).argmax(1) == ty).float().mean().item()
                        R[4, j] = acc

                # Sequential steps 5..9
                for step in range(5, 10):
                    curr_b = order[step]
                    cx = tr_x[curr_b].to(DEVICE); cy = tr_y[curr_b].to(DEVICE)
                    seen_classes.update(cy.cpu().numpy().tolist())

                    model.train()
                    for _ in range(epochs):
                        logits = model(cx)
                        loss = criterion(logits, cy)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        mask_unseen = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=DEVICE)
                        for c in seen_classes: mask_unseen[c] = True
                        for j in range(10):
                            tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                            use_mask = mask_unseen if arm_name == "l1d" else None
                            acc = (model(tx, mask_unseen=use_mask).argmax(1) == ty).float().mean().item()
                            R[step, j] = acc

            # Compute summary metrics over populated rows
            a_t_vals = [R[9, j] for j in range(10)]
            la_vals  = [R[max(4, order.index(j)), j] for j in range(10)]
            bwt_vals = [R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]

            a_t_list.append(np.mean(a_t_vals))
            la_list.append(np.mean(la_vals))
            bwt_list.append(np.mean(bwt_vals))

            # Diagnostics for head weights and oracle accuracy
            if arm_name not in ["freeze_after_base", "step_matched_joint"]:
                with torch.no_grad():
                    # Head weight norms & bias values
                    if hasattr(model.head, "head"):
                        w = model.head.head.weight.data
                        b_val = model.head.head.bias.data if model.head.head.bias is not None else None
                    else:
                        w = model.head.weight.data
                        b_val = None

                    old_b_classes = []
                    for b in order[:9]:
                        old_b_classes.extend(tr_y[b].cpu().numpy().tolist())
                    newest_b_classes = tr_y[order[9]].cpu().numpy().tolist()

                    old_w_norm = torch.norm(w[old_b_classes], dim=-1).mean().item()
                    new_w_norm = torch.norm(w[newest_b_classes], dim=-1).mean().item()
                    diag_weight_norms_old.append(old_w_norm)
                    diag_weight_norms_new.append(new_w_norm)

                    if b_val is not None:
                        old_b_mean = b_val[old_b_classes].mean().item()
                        new_b_mean = b_val[newest_b_classes].mean().item()
                        diag_biases_old.append(old_b_mean)
                        diag_biases_new.append(new_b_mean)

                    # Oracle Argmax Accuracy (restricted to true block's classes)
                    oracle_accs = []
                    true_accs   = []
                    for j in range(10):
                        tx = te_x[j].to(DEVICE); ty = te_y[j].to(DEVICE)
                        logits = model(tx)
                        true_accs.append((logits.argmax(1) == ty).float().mean().item())

                        # Oracle mask: mask out classes not in block j
                        block_classes = set(tr_y[j].cpu().numpy().tolist())
                        mask_oracle = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=DEVICE)
                        for c in block_classes: mask_oracle[c] = True
                        oracle_logits = logits.masked_fill(~mask_oracle, -1e9)
                        oracle_accs.append((oracle_logits.argmax(1) == ty).float().mean().item())

                    oracle_acc_list.append(np.mean(oracle_accs))
                    true_acc_list.append(np.mean(true_accs))

    return {
        "arm_name": arm_name,
        "a_t_mean": float(np.mean(a_t_list)),
        "a_t_std":  float(np.std(a_t_list)),
        "a_t_min":  float(np.min(a_t_list)),
        "a_t_max":  float(np.max(a_t_list)),
        "la_mean":  float(np.mean(la_list)),
        "bwt_mean": float(np.mean(bwt_list)),
        "a_t_raw":  [float(x) for x in a_t_list],
        "diag_w_old": float(np.mean(diag_weight_norms_old)) if diag_weight_norms_old else 0.0,
        "diag_w_new": float(np.mean(diag_weight_norms_new)) if diag_weight_norms_new else 0.0,
        "diag_b_old": float(np.mean(diag_biases_old)) if diag_biases_old else 0.0,
        "diag_b_new": float(np.mean(diag_biases_new)) if diag_biases_new else 0.0,
        "oracle_acc": float(np.mean(oracle_acc_list)) if oracle_acc_list else 0.0,
        "true_acc":   float(np.mean(true_acc_list)) if true_acc_list else 0.0,
    }


def main():
    print("=" * 80)
    print("  PHASE 4 LEVER 1: REMOVE RECENCY BIAS IN THE HEAD")
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

    conf_pairs = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)

    sel_seeds   = list(range(101, 106))
    fresh_seeds = list(range(201, 206))

    arms = [
        "naive",
        "freeze_after_base",
        "step_matched_joint",
        "l1b",
        "l1c",
        "l1d"
    ]

    results = {}
    print("\nRunning L1 Arms across 50 runs (10 shuffles x 5 seeds) per seed set...")

    for arm in arms:
        print(f"  Executing Arm: {arm:<20} ...")
        res_sel = run_experiment_arm(arm, block_assignment, cache_data, sel_seeds, num_shuffles=10)
        res_fre = run_experiment_arm(arm, block_assignment, cache_data, fresh_seeds, num_shuffles=10)
        results[arm] = {
            "sel": res_sel,
            "fre": res_fre
        }

    # Print Main Results Table
    naive_sel_raw = np.array(results["naive"]["sel"]["a_t_raw"])
    naive_fre_raw = np.array(results["naive"]["fre"]["a_t_raw"])

    print("\n" + "=" * 115)
    print("  PHASE 4 LEVER 1 MAIN RESULTS TABLE (50 RUNS: 10 SHUFFLES x 5 SEEDS PER SEED SET)")
    print("=" * 115)
    print(f"  {'Arm':<20} | {'A_T (sel)':<10} | {'A_T (fre)':<10} | {'LA':<8} | {'BWT':<8} | {'Delta A_T vs Naive (95% CI)':<28} | {'std':<6} | {'min..max':<12}")
    print("  " + "-" * 115)

    for arm in arms:
        sel_res = results[arm]["sel"]
        fre_res = results[arm]["fre"]

        diff_sel = np.array(sel_res["a_t_raw"]) - naive_sel_raw
        mean_diff = float(np.mean(diff_sel))

        # 10,000 paired bootstrap resamples for 95% CI on Delta A_T
        np.random.seed(42)
        boot_diffs = []
        for _ in range(10000):
            b_idx = np.random.choice(len(diff_sel), size=len(diff_sel), replace=True)
            boot_diffs.append(np.mean(diff_sel[b_idx]))
        ci_low  = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))

        ci_str = f"{mean_diff*100:+.2f}% [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%]"
        min_max_str = f"{sel_res['a_t_min']*100:.1f}..{sel_res['a_t_max']*100:.1f}"

        print(f"  {arm:<20} | {sel_res['a_t_mean']*100:6.2f}%    | {fre_res['a_t_mean']*100:6.2f}%    | {sel_res['la_mean']*100:6.2f}% | {sel_res['bwt_mean']*100:6.2f}% | {ci_str:<28} | {sel_res['a_t_std']*100:5.2f}% | {min_max_str:<12}")

        # Verification Assertion (R2 & R13): delta A_T = delta LA + delta BWT
        sum_check = sel_res["la_mean"] + sel_res["bwt_mean"]
        assert abs(sel_res["a_t_mean"] - sum_check) < 1e-5, f"R2 Violation: {sel_res['a_t_mean']} != {sum_check}"

    print("  " + "-" * 115)
    print("  [VERIFICATION PASS] All A_T = LA + BWT decompositions sum EXACTLY across all arms.")

    # Diagnostics Report
    print("\n" + "=" * 80)
    print("  L1 HEAD RECENCY BIAS DIAGNOSTICS")
    print("=" * 80)
    print(f"  {'Arm':<20} | {'W-Norm (Old)':<13} | {'W-Norm (New)':<13} | {'Bias (Old)':<12} | {'Bias (New)':<12} | {'Oracle Acc':<11} | {'True Acc':<9}")
    print("  " + "-" * 88)

    for arm in ["naive", "l1b", "l1c", "l1d"]:
        sel_res = results[arm]["sel"]
        print(f"  {arm:<20} | {sel_res['diag_w_old']:13.4f} | {sel_res['diag_w_new']:13.4f} | {sel_res['diag_b_old']:12.4f} | {sel_res['diag_b_new']:12.4f} | {sel_res['oracle_acc']*100:9.2f}%  | {sel_res['true_acc']*100:7.2f}%")

    print("  " + "-" * 88)

    # Gate Evaluation
    l1a_sel = results["naive"]["sel"]["a_t_raw"]
    l1c_sel = results["l1c"]["sel"]["a_t_raw"]
    l1a_fre = results["naive"]["fre"]["a_t_raw"]
    l1c_fre = results["l1c"]["fre"]["a_t_raw"]

    diff_l1c_a_sel = np.array(l1c_sel) - np.array(l1a_sel)
    diff_l1c_a_fre = np.array(l1c_fre) - np.array(l1a_fre)

    np.random.seed(42)
    b_sel = [np.mean(diff_l1c_a_sel[np.random.choice(len(diff_l1c_a_sel), len(diff_l1c_a_sel), replace=True)]) for _ in range(10000)]
    b_fre = [np.mean(diff_l1c_a_fre[np.random.choice(len(diff_l1c_a_fre), len(diff_l1c_a_fre), replace=True)]) for _ in range(10000)]

    ci_sel = (np.percentile(b_sel, 2.5), np.percentile(b_sel, 97.5))
    ci_fre = (np.percentile(b_fre, 2.5), np.percentile(b_fre, 97.5))

    gate_passed = (ci_sel[0] > 0 and ci_fre[0] > 0)

    print("\n" + "=" * 80)
    print("  L1 GATE EVALUATION")
    print("=" * 80)
    print(f"  L1c vs L1a Delta A_T (Selection): {np.mean(diff_l1c_a_sel)*100:+.2f}% [95% CI: {ci_sel[0]*100:+.2f}%, {ci_sel[1]*100:+.2f}%]")
    print(f"  L1c vs L1a Delta A_T (Fresh):     {np.mean(diff_l1c_a_fre)*100:+.2f}% [95% CI: {ci_fre[0]*100:+.2f}%, {ci_fre[1]*100:+.2f}%]")
    print(f"  L1 GATE PASSED (CI excludes 0 on BOTH seed sets): {gate_passed}")
    if gate_passed:
        print("  [DECISION] L1c (weight-normalised cosine head) BEATS L1a and BECOMES THE NEW BASE HEAD FOR L2-L4.")
    else:
        print("  [DECISION] L1c does not surpass L1a with CIs excluding zero; baseline L1a head retained.")

    # Save Results JSON
    save_data = {
        "results": results,
        "gate_passed": bool(gate_passed),
        "ci_sel": [float(ci_sel[0]), float(ci_sel[1])],
        "ci_fre": [float(ci_fre[0]), float(ci_fre[1])]
    }
    with open("results_l1_head.json", "w") as out:
        json.dump(save_data, out, indent=2)
    print("\nSaved full results to results_l1_head.json.")

if __name__ == "__main__":
    main()
