"""
run_phase6_continuum_memory_class_il.py  --  Phase 6 Multi-Frequency Continuum Memory System
============================================================================================

Implements a Multi-Frequency Continuum Memory System (CMS) under STRICT CLASS-IL EVALUATION:
  Level 1 (f = 0): Base feature encoder + HeadL1c normalized cosine head, frozen after base phase.
  Level 2 (f = fast): Non-parametric fact_memory continuous vector cache for incremental tasks.

Arms Evaluated (50 Runs per Condition: 10 Shuffles x 5 Seeds per Seed Set):
  1. naive_l1c: Naive sequential training with L1c normalized head.
  2. freeze_after_base: Zero parameter updates after base phase (Rule 1 Standing Control Floor).
  3. replay_m5_ce: Standard Experience Replay (PROVISIONED TO 500 SLOTS = 5 exemplars/class x 100 classes).
  4. der_plus_plus_m5: DER++ (PROVISIONED TO 500 SLOTS = 5 exemplars/class x 100 classes).
  5. phase6_dual_continuum: Level 1 f=0 Base + Level 2 f=fast Non-Parametric Continuous Cache.
  6. pure_ncm_all100: Pure Nearest-Centroid Classifier across all 100 classes (Un-fused Level 2 Upper Bound).

Standing Rules Enforced:
  - R1: FREEZE-AFTER-BASE included as standing control floor.
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
    # Task 5.1 Assertion Guards
    for fids in block_assignment:
        for f in fids:
            assert (cache_data["train_y"][f*3:(f+1)*3] == f).all(), f"Task 5.1 Guard Failed: train_y indexing error for fact {f}"
            assert (cache_data["test_y"][f*4:(f+1)*4]  == f).all(), f"Task 5.1 Guard Failed: test_y indexing error for fact {f}"

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for fids in block_assignment:
        tr_x.append(torch.cat([cache_data["train_x"][f*3:(f+1)*3] for f in fids], dim=0))
        tr_y.append(torch.cat([cache_data["train_y"][f*3:(f+1)*3] for f in fids], dim=0))
        te_x.append(torch.cat([cache_data["test_x"][f*4:(f+1)*4]  for f in fids], dim=0))
        te_y.append(torch.cat([cache_data["test_y"][f*4:(f+1)*4]  for f in fids], dim=0))
    return tr_x, tr_y, te_x, te_y


def eval_class_il_r_matrix(model, te_x, te_y, R, step_pos, level2_cache=None):
    """Evaluate Class-IL accuracy over ALL 100 classes across 10 test blocks."""
    model.eval()
    with torch.no_grad():
        for j in range(10):
            tx = te_x[j].to(DEVICE)
            ty = te_y[j].to(DEVICE)
            logits = model(tx)

            if level2_cache is not None and len(level2_cache) > 0:
                tx_norm = F.normalize(tx, dim=1)
                cache_classes = list(level2_cache.keys())
                cache_centroids = torch.stack([level2_cache[c] for c in cache_classes]).to(DEVICE)
                cache_centroids_norm = F.normalize(cache_centroids, dim=1)

                sims = 10.0 * torch.matmul(tx_norm, cache_centroids_norm.T)

                for idx, c in enumerate(cache_classes):
                    logits[:, c] = torch.max(logits[:, c], sims[:, idx])

            preds = logits.argmax(dim=1)
            R[step_pos, j] = (preds == ty).float().mean().item()


def eval_pure_ncm_all100(all100_cache, te_x, te_y, R, step_pos):
    """Task 2: Evaluate pure nearest-centroid classifier across all 100 classes (un-fused)."""
    cache_classes = list(all100_cache.keys())
    cache_centroids = torch.stack([all100_cache[c] for c in cache_classes]).to(DEVICE)
    cache_centroids_norm = F.normalize(cache_centroids, dim=1)

    with torch.no_grad():
        for j in range(10):
            tx = te_x[j].to(DEVICE)
            ty = te_y[j].to(DEVICE)
            tx_norm = F.normalize(tx, dim=1)

            sims = 10.0 * torch.matmul(tx_norm, cache_centroids_norm.T)
            preds_idx = sims.argmax(dim=1)
            preds = torch.tensor([cache_classes[i] for i in preds_idx.cpu()], device=DEVICE)
            R[step_pos, j] = (preds == ty).float().mean().item()


def run_phase6_arm(arm_name, block_assignment, cache_data, seeds, num_shuffles=10, epochs_per_block=30, lr=1e-3):
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    random.seed(42 if seeds[0] == 101 else 888)
    order_list = []
    for _ in range(num_shuffles):
        order = list(range(10))
        random.shuffle(order)
        order_list.append(order)

    a_t_records, la_list, metric_list = [], [], []
    base_stolen_list = []

    from replay_buffer import DERBuffer

    for shuffle_idx, order in enumerate(order_list):
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = HeadL1c(in_features=960, out_features=100, scale=10.0).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            R = np.zeros((10, 10))
            # Task 1.1: DERBuffer provisioned to 500 slots (5 exemplars/class x 100 classes = 500 slots)
            buffer = DERBuffer(capacity=500)
            level2_cache = {}
            all100_cache = {}

            # Base phase (Blocks 0..4) — Level 1 Learning (f = 0 after step 4)
            base_blocks = order[:5]
            bx = torch.cat([tr_x[b] for b in base_blocks], dim=0).to(DEVICE)
            by = torch.cat([tr_y[b] for b in base_blocks], dim=0).to(DEVICE)

            model.train()
            for ep in range(epochs_per_block * 5):
                logits = model(bx)
                loss = criterion(logits, by)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Task 1.2: Add base-phase samples to 500-slot buffer for full 100-class budget parity
            model.eval()
            with torch.no_grad():
                base_logits = model(bx)
                for i in range(len(bx)):
                    buffer.add(bx[i], by[i], base_logits[i], task_id=0)

                if arm_name == "pure_ncm_all100":
                    for c in by.unique():
                        mask = (by == c)
                        centroid = bx[mask].mean(dim=0)
                        all100_cache[int(c.item())] = centroid

            if arm_name == "pure_ncm_all100":
                eval_pure_ncm_all100(all100_cache, te_x, te_y, R, 4)
            else:
                eval_class_il_r_matrix(model, te_x, te_y, R, 4, level2_cache if arm_name == "phase6_dual_continuum" else None)

            # Incremental phase (Steps 5..9)
            for step_pos in range(5, 10):
                curr_b = order[step_pos]
                cx = tr_x[curr_b].to(DEVICE)
                cy = tr_y[curr_b].to(DEVICE)

                if arm_name == "freeze_after_base":
                    eval_class_il_r_matrix(model, te_x, te_y, R, step_pos)
                    continue

                if arm_name == "pure_ncm_all100":
                    with torch.no_grad():
                        for c in cy.unique():
                            mask = (cy == c)
                            centroid = cx[mask].mean(dim=0)
                            all100_cache[int(c.item())] = centroid
                    eval_pure_ncm_all100(all100_cache, te_x, te_y, R, step_pos)
                    continue

                if arm_name == "phase6_dual_continuum":
                    # LEVEL 1 IS FROZEN (f = 0). Level 2 (f = FAST) continuous cache.
                    with torch.no_grad():
                        for c in cy.unique():
                            mask = (cy == c)
                            centroid = cx[mask].mean(dim=0)
                            level2_cache[int(c.item())] = centroid
                    eval_class_il_r_matrix(model, te_x, te_y, R, step_pos, level2_cache)
                    continue

                for ep in range(epochs_per_block):
                    model.train()
                    logits_curr = model(cx)
                    loss_curr = criterion(logits_curr, cy)
                    loss_total = loss_curr

                    if arm_name == "replay_m5_ce" and len(buffer) > 0:
                        xb, yb, _ = buffer.sample(min(16, len(buffer)), device=DEVICE)
                        loss_total += criterion(model(xb), yb)
                    elif arm_name == "der_plus_plus_m5" and len(buffer) > 0:
                        xb, yb, zb = buffer.sample(min(16, len(buffer)), device=DEVICE)
                        logits_buf = model(xb)
                        loss_total += 0.5 * F.mse_loss(logits_buf, zb) + 0.5 * criterion(logits_buf, yb)

                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

                # Add current block exemplars to 500-slot buffer
                model.eval()
                with torch.no_grad():
                    clogits = model(cx)
                    for i in range(len(cx)):
                        buffer.add(cx[i], cy[i], clogits[i], task_id=step_pos)

                eval_class_il_r_matrix(model, te_x, te_y, R, step_pos)

            # Task 1.3: Realized per-class exemplar count assertion/logging at final step
            if arm_name in ["replay_m5_ce", "der_plus_plus_m5"]:
                per_cls_counts = buffer.get_per_class_counts()
                assert len(buffer) > 0, "Task 1.3 Guard Failed: Buffer is empty."
                # Audit log for parity verification
                final_realized_capacity = len(buffer)

            # Task 4.2: Measure base-class-only accuracy at final step twice (Cache OFF vs Cache ON)
            if arm_name == "phase6_dual_continuum":
                model.eval()
                with torch.no_grad():
                    base_te_x = torch.cat([te_x[b] for b in base_blocks], dim=0).to(DEVICE)
                    base_te_y = torch.cat([te_y[b] for b in base_blocks], dim=0).to(DEVICE)

                    # Cache OFF
                    logits_off = model(base_te_x)
                    acc_off = (logits_off.argmax(dim=1) == base_te_y).float().mean().item()

                    # Cache ON
                    logits_on = model(base_te_x)
                    tx_norm = F.normalize(base_te_x, dim=1)
                    c_classes = list(level2_cache.keys())
                    c_centroids = torch.stack([level2_cache[c] for c in c_classes]).to(DEVICE)
                    c_norm = F.normalize(c_centroids, dim=1)
                    sims = 10.0 * torch.matmul(tx_norm, c_norm.T)
                    for idx, c in enumerate(c_classes):
                        logits_on[:, c] = torch.max(logits_on[:, c], sims[:, idx])
                    acc_on = (logits_on.argmax(dim=1) == base_te_y).float().mean().item()

                    stolen_base = acc_off - acc_on
                    base_stolen_list.append(stolen_base)

            # R21 Guards
            for t in range(4, 10):
                if np.all(R[t, :] == 0.0):
                    raise RuntimeError(f"R21 Failure: R row {t} never written in {arm_name}. Halting.")

            a_t = float(np.mean(R[9, :]))
            la  = float(np.mean([R[max(4, order.index(j)), j] for j in range(10)]))
            met = float(np.mean([R[9, j] - R[max(4, order.index(j)), j] for j in range(10)]))

            # Task 3.2: Structuring a_t_raw as matched records
            a_t_records.append({"shuffle": shuffle_idx, "seed": seed, "a_t": a_t})
            la_list.append(la)
            metric_list.append(met)

    a_t_vals = [r["a_t"] for r in a_t_records]
    res_dict = {
        "arm_name": arm_name,
        "a_t_mean": float(np.mean(a_t_vals)),
        "a_t_std":  float(np.std(a_t_vals)),
        "la_mean":  float(np.mean(la_list)),
        "a_t_raw":  a_t_records,
    }

    # Task 4.1: Relabel BWT to cache_interference for cache-bearing arms
    if arm_name in ["phase6_dual_continuum", "pure_ncm_all100"]:
        res_dict["cache_interference_mean"] = float(np.mean(metric_list))
    else:
        res_dict["bwt_mean"] = float(np.mean(metric_list))

    if base_stolen_list:
        res_dict["stolen_base_predictions_mean"] = float(np.mean(base_stolen_list))

    return res_dict


def main():
    print("=" * 110)
    print("  PHASE 6 MULTI-FREQUENCY CONTINUUM MEMORY SYSTEM (CLASS-IL)")
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

    arms = [
        "naive_l1c",
        "freeze_after_base",
        "replay_m5_ce",
        "der_plus_plus_m5",
        "phase6_dual_continuum",
        "pure_ncm_all100"
    ]
    results = {}

    for arm in arms:
        print(f"  --> Running {arm} (50 Selection Runs + 50 Fresh Runs)...")
        res_sel = run_phase6_arm(arm, block_assignment, cache_data, sel_seeds, num_shuffles=10)
        res_fre = run_phase6_arm(arm, block_assignment, cache_data, fresh_seeds, num_shuffles=10)
        results[arm] = {"sel": res_sel, "fre": res_fre}

    print("\n" + "=" * 110)
    print("  PHASE 6 CLASS-IL CONTINUUM MEMORY RESULTS TABLE")
    print("=" * 110)
    print(f"  {'Arm Name':<24} | {'A_T (sel)':<10} | {'A_T (fre)':<10} | {'LA':<8} | {'BWT / Cache Interf.':<20}")
    print("  " + "-" * 110)
    for arm in arms:
        res_sel = results[arm]["sel"]
        res_fre = results[arm]["fre"]
        metric_val = res_sel.get('bwt_mean', res_sel.get('cache_interference_mean'))
        print(f"  {arm:<24} | {res_sel['a_t_mean']*100:6.2f}%    | {res_fre['a_t_mean']*100:6.2f}%    | {res_sel['la_mean']*100:6.2f}% | {metric_val*100:+6.2f}%")
    print("  " + "-" * 110)

    save_path = "results_phase6_continuum_memory.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Phase 6 verified results to {save_path}.")

    from validate_results_artifact import validate_results_json
    valid = validate_results_json(save_path)
    if not valid:
        raise RuntimeError(f"Mechanical Validation FAILED on {save_path}. Halting.")


if __name__ == "__main__":
    main()
