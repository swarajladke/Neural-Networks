"""
run_phase5_der_plus_plus_class_il.py  --  Phase 5 Class-IL Dark Experience Replay (DER++) Benchmark
==================================================================================================

Evaluates Dark Experience Replay (DER++) against standard Replay and Naive baselines under
STRICT CLASS-IL EVALUATION (no task-ID gating, all 100 classes evaluated simultaneously).

Arms Evaluated (50 Runs per Condition: 10 Shuffles x 5 Seeds per Seed Set):
  1. naive_l1c: Naive sequential training with L1c normalized head.
  2. freeze_after_base: Zero parameter updates after base phase (Standing Rule 1).
  3. replay_m5_ce: Standard Experience Replay (m=5 exemplars/class, CE loss).
  4. der_m5: Dark Experience Replay (m=5 exemplars/class, MSE logit matching alpha=0.5).
  5. der_plus_plus_m5: DER++ (m=5 exemplars/class, MSE logit matching alpha=0.5 + CE replay beta=0.5).

Standing Rules Enforced:
  - R1: FREEZE-AFTER-BASE included as standing control.
  - R2: Decomposed retention and acquisition gaps reported.
  - R21: No hardcoded result literals; exact R21 runtime guards.
"""

import os
import json
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


def eval_class_il_r_matrix(model, te_x, te_y, R, step_pos):
    """Evaluate Class-IL accuracy over ALL 100 classes across 10 test blocks."""
    model.eval()
    with torch.no_grad():
        for j in range(10):
            tx = te_x[j].to(DEVICE)
            ty = te_y[j].to(DEVICE)
            logits = model(tx)
            preds = logits.argmax(dim=1)
            R[step_pos, j] = (preds == ty).float().mean().item()


def run_phase5_arm(arm_name, block_assignment, cache_data, seeds, num_shuffles=10, epochs_per_block=30, lr=1e-3, m_per_class=5):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)

    a_t_list, la_list, bwt_list = [], [], []

    from replay_buffer import DERBuffer

    for order in order_list:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = HeadL1c(in_features=960, out_features=100, scale=10.0).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            R = np.zeros((10, 10))
            buffer = DERBuffer(capacity=100)

            # Base phase (Blocks 0..4)
            base_blocks = order[:5]
            bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            model.train()
            for ep in range(epochs_per_block * 5):
                logits = model(bx)
                loss = criterion(logits, by)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Populate buffer with base phase exemplars
            model.eval()
            with torch.no_grad():
                base_logits = model(bx)
                for i in range(len(bx)):
                    buffer.add(bx[i], by[i], base_logits[i], task_id=0)

            eval_class_il_r_matrix(model, te_x, te_y, R, 4)

            # Incremental phase (Steps 5..9)
            for step_pos in range(5, 10):
                if arm_name == "freeze_after_base":
                    eval_class_il_r_matrix(model, te_x, te_y, R, step_pos)
                    continue

                curr_b = order[step_pos]
                cx = tr_x[curr_b].to(DEVICE)
                cy = tr_y[curr_b].to(DEVICE)

                for ep in range(epochs_per_block):
                    model.train()
                    logits_curr = model(cx)
                    loss_curr = criterion(logits_curr, cy)

                    loss_total = loss_curr

                    if arm_name == "replay_m5_ce" and len(buffer) > 0:
                        xb, yb, _ = buffer.sample(min(16, len(buffer)), device=DEVICE)
                        loss_total += criterion(model(xb), yb)
                    elif arm_name == "der_m5" and len(buffer) > 0:
                        xb, _, zb = buffer.sample(min(16, len(buffer)), device=DEVICE)
                        loss_total += 0.5 * F.mse_loss(model(xb), zb)
                    elif arm_name == "der_plus_plus_m5" and len(buffer) > 0:
                        xb, yb, zb = buffer.sample(min(16, len(buffer)), device=DEVICE)
                        logits_buf = model(xb)
                        loss_total += 0.5 * F.mse_loss(logits_buf, zb) + 0.5 * criterion(logits_buf, yb)

                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

                # Add current block exemplars to buffer
                model.eval()
                with torch.no_grad():
                    clogits = model(cx)
                    for i in range(len(cx)):
                        buffer.add(cx[i], cy[i], clogits[i], task_id=step_pos)

                eval_class_il_r_matrix(model, te_x, te_y, R, step_pos)

            # R21 Guards
            for t in range(4, 10):
                if np.all(R[t, :] == 0.0):
                    raise RuntimeError(f"R21 Failure: R row {t} never written in {arm_name}. Halting.")
            if np.allclose(R[4:10, :].std(axis=0), 0.0) and arm_name != "freeze_after_base":
                raise RuntimeError(f"R21 Failure: All R rows identical in {arm_name}. Halting.")

            a_t = float(np.mean(R[9, :]))
            la  = float(np.mean([R[max(4, order.index(j)), j] for j in range(10)]))
            bwt = float(np.mean([R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]))

            n_j = [len(te_y[b]) for b in range(10)]
            r_mat_list = R.tolist()
            a_t_list.append({
                "shuffle": len(a_t_list) // len(seeds),
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

    a_t_vals = [r["a_t"] for r in a_t_list]
    return {
        "arm_name": arm_name,
        "a_t_mean": float(np.mean(a_t_vals)),
        "a_t_std":  float(np.std(a_t_vals)),
        "la_mean":  float(np.mean(la_list)),
        "bwt_mean": float(np.mean(bwt_list)),
        "a_t_raw":  a_t_list,
    }


def main():
    print("=" * 110)
    print("  PHASE 5 CLASS-IL DARK EXPERIENCE REPLAY (DER++) BENCHMARK")
    print("=" * 110)

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

    arms = ["naive_l1c", "freeze_after_base", "replay_m5_ce", "der_m5", "der_plus_plus_m5"]
    results = {}

    for arm in arms:
        print(f"  --> Running {arm} (50 Selection Runs + 50 Fresh Runs)...")
        res_sel = run_phase5_arm(arm, block_assignment, cache_data, sel_seeds, num_shuffles=10)
        res_fre = run_phase5_arm(arm, block_assignment, cache_data, fresh_seeds, num_shuffles=10)
        results[arm] = {"sel": res_sel, "fre": res_fre}

    print("\n" + "=" * 110)
    print("  PHASE 5 CLASS-IL DER++ RESULTS TABLE")
    print("=" * 110)
    print(f"  {'Arm Name':<22} | {'A_T (sel)':<10} | {'A_T (fre)':<10} | {'LA':<8} | {'BWT':<8}")
    print("  " + "-" * 110)
    for arm in arms:
        res_sel = results[arm]["sel"]
        res_fre = results[arm]["fre"]
        print(f"  {arm:<22} | {res_sel['a_t_mean']*100:6.2f}%    | {res_fre['a_t_mean']*100:6.2f}%    | {res_sel['la_mean']*100:6.2f}% | {res_sel['bwt_mean']*100:+6.2f}%")
    print("  " + "-" * 110)

    save_path = "results_phase5_der_plus_plus.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Phase 5 results to {save_path}.")


if __name__ == "__main__":
    main()
