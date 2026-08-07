"""
run_graded_ceiling_test.py  --  Graded Continuous Predictor Test for Retrieval Ceiling
=====================================================================================

Predictor: x_q = raw cosine similarity of test query q to its nearest incorrect class centroid.
Outcome:   y_q = binary retrieval failure indicator after 50-fact base adaptation.

Reports:
  1. Mean raw cosine predictor x_q for Failed vs Passed queries (Trained vs Untrained).
  2. Logistic Regression coefficient beta_1.
  3. Odds Ratio per 0.01 cosine change (OR_0.01 = exp(0.01 * beta_1)), 95% CI, and p-value.
"""

import os
import json
import random
import math
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH   = ("smollm2_embeddings_100slots.pt"
                if os.path.exists("smollm2_embeddings_100slots.pt")
                else "../smollm2_embeddings_100slots.pt")
DATASET_PATH = "agnis_scaling_dataset.json"
INPUT_DIM    = 960

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


def compute_pca_basis(cache_data, r):
    X = cache_data["train_x"].float().cpu()
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:r].clone()


def supervised_contrastive_loss(z, y, tau=0.05):
    sim  = torch.matmul(z, z.T) / tau
    N    = z.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    pos  = (y.unsqueeze(0) == y.unsqueeze(1)) & mask
    lm, _ = torch.max(sim * mask.float(), dim=1, keepdim=True)
    logits = sim - lm.detach()
    exp_l  = torch.exp(logits) * mask.float()
    lp     = logits - torch.log(exp_l.sum(1, keepdim=True).clamp_min(1e-12))
    mlp    = (pos.float() * lp).sum(1) / pos.float().sum(1).clamp_min(1.0)
    return -mlp.mean()


def find_confusable_pairs(cache_data, threshold=0.95):
    X = cache_data["train_x"].float()
    cen = torch.zeros(100, INPUT_DIM, dtype=torch.float32)
    for i in range(100):
        samples = X[i*3:(i+1)*3]
        cen[i] = F.normalize(samples.mean(0, keepdim=True), dim=-1).squeeze(0)
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


def run_logistic_regression(X_pred, Y_fail):
    """Fit Logit(Y_fail) = beta_0 + beta_1 * X_pred and return OR per 0.01 cosine."""
    X_const = sm.add_constant(X_pred)
    model = sm.Logit(Y_fail, X_const)
    res = model.fit(disp=False)

    beta_1 = res.params[1]
    se_beta_1 = res.bse[1]
    p_value = res.pvalues[1]

    # OR per 0.01 cosine unit
    or_001 = np.exp(0.01 * beta_1)
    ci_low_001 = np.exp(0.01 * (beta_1 - 1.96 * se_beta_1))
    ci_high_001 = np.exp(0.01 * (beta_1 + 1.96 * se_beta_1))

    return beta_1, or_001, (ci_low_001, ci_high_001), p_value


def main():
    print("=" * 80)
    print("  GRADED CONTINUOUS PREDICTOR TEST FOR RETRIEVAL CEILING")
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

    pca_basis_r32 = compute_pca_basis(cache_data, r=32).to(DEVICE)
    conf_pairs = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    # Compute raw centroids
    cen_raw = torch.zeros(100, INPUT_DIM, device=DEVICE)
    for i in range(100):
        samples = cache_data["train_x"][i*3:(i+1)*3].float().to(DEVICE)
        cen_raw[i] = F.normalize(samples.mean(0, keepdim=True), dim=-1).squeeze(0)

    # Train adapter on 50 base facts
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    adapter = BottleneckAdapter(r=32, pca_basis=pca_basis_r32).to(DEVICE)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=1e-4)

    joint_train_x_base = torch.cat([tr_x[b] for b in range(5)], dim=0).to(DEVICE)
    joint_train_y_base = torch.cat([tr_y[b] for b in range(5)], dim=0).to(DEVICE)

    adapter.train()
    for _ in range(100):
        proj = adapter(joint_train_x_base)
        loss = supervised_contrastive_loss(proj, joint_train_y_base)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    adapter.eval()
    all_train_x = torch.cat([tr_x[b] for b in range(10)], dim=0).to(DEVICE)
    all_train_y = torch.cat([tr_y[b] for b in range(10)], dim=0).to(DEVICE)

    with torch.no_grad():
        z_refs = adapter(all_train_x)

        predictors_tr, failures_tr = [], []
        predictors_un, failures_un = [], []

        for b in range(10):
            test_x_b = te_x[b].to(DEVICE)
            test_y_b = te_y[b].to(DEVICE)

            # Raw unadapted test query embeddings
            raw_q = F.normalize(test_x_b, dim=-1)
            raw_sims = torch.matmul(raw_q, cen_raw.T)

            # Adapted predictions
            z_queries = adapter(test_x_b)
            ad_sims   = torch.matmul(z_queries, z_refs.T)

            for q_idx in range(len(test_y_b)):
                correct_class = test_y_b[q_idx].item()

                # Predictor: raw cosine to nearest incorrect centroid
                raw_sim_vec = raw_sims[q_idx].clone()
                raw_sim_vec[correct_class] = -999.0
                raw_nearest_incorrect_cos = raw_sim_vec.max().item()

                # Outcome: failed after adaptation
                pred_idx = torch.argmax(ad_sims[q_idx]).item()
                pred_class = all_train_y[pred_idx].item()
                is_failed = int(pred_class != correct_class)

                if b < 5:
                    predictors_tr.append(raw_nearest_incorrect_cos)
                    failures_tr.append(is_failed)
                else:
                    predictors_un.append(raw_nearest_incorrect_cos)
                    failures_un.append(is_failed)

    predictors_tr = np.array(predictors_tr)
    failures_tr   = np.array(failures_tr)
    predictors_un = np.array(predictors_un)
    failures_un   = np.array(failures_un)

    predictors_all = np.concatenate([predictors_tr, predictors_un])
    failures_all   = np.concatenate([failures_tr, failures_un])

    print("\n  1. MEAN RAW COSINE PREDICTOR (x_q) FOR FAILED VS PASSED QUERIES:")
    print(f"     Trained Subset (200 q):   Failed mean x_q = {np.mean(predictors_tr[failures_tr==1]):.4f} ± {np.std(predictors_tr[failures_tr==1]):.4f} | Passed mean x_q = {np.mean(predictors_tr[failures_tr==0]):.4f} ± {np.std(predictors_tr[failures_tr==0]):.4f}")
    print(f"     Untrained Subset (200 q): Failed mean x_q = {np.mean(predictors_un[failures_un==1]):.4f} ± {np.std(predictors_un[failures_un==1]):.4f} | Passed mean x_q = {np.mean(predictors_un[failures_un==0]):.4f} ± {np.std(predictors_un[failures_un==0]):.4f}")
    print(f"     Combined Population(400 q):Failed mean x_q = {np.mean(predictors_all[failures_all==1]):.4f} ± {np.std(predictors_all[failures_all==1]):.4f} | Passed mean x_q = {np.mean(predictors_all[failures_all==0]):.4f} ± {np.std(predictors_all[failures_all==0]):.4f}")

    print("\n  2. LOGISTIC REGRESSION ANALYSIS (Predicting Post-Adaptation Failure):")

    beta_tr, or_tr, ci_tr, p_tr = run_logistic_regression(predictors_tr, failures_tr)
    print(f"     Trained Subset:   beta_1 = {beta_tr:+.4f}, OR per +0.01 cos = {or_tr:.4f} (95% CI: [{ci_tr[0]:.4f}, {ci_tr[1]:.4f}]), p = {p_tr:.4e}")

    beta_un, or_un, ci_un, p_un = run_logistic_regression(predictors_un, failures_un)
    print(f"     Untrained Subset: beta_1 = {beta_un:+.4f}, OR per +0.01 cos = {or_un:.4f} (95% CI: [{ci_un[0]:.4f}, {ci_un[1]:.4f}]), p = {p_un:.4e}")

    beta_all, or_all, ci_all, p_all = run_logistic_regression(predictors_all, failures_all)
    print(f"     Combined Population: beta_1 = {beta_all:+.4f}, OR per +0.01 cos = {or_all:.4f} (95% CI: [{ci_all[0]:.4f}, {ci_all[1]:.4f}]), p = {p_all:.4e}")

    print("\n  3. DECISION & CEILING CAUSE RESOLUTION:")
    if p_all < 0.05:
        print(f"     [RESOLVED] Raw nearest-incorrect cosine similarity is a STATISTICALLY SIGNIFICANT predictor of failure.")
        print(f"     [EFFECT SIZE] Each +0.01 increase in raw nearest-incorrect cosine increases post-adaptation failure odds by {or_all:.4f}x (95% CI: [{ci_all[0]:.4f}, {ci_all[1]:.4f}], p = {p_all:.4e}).")
        print(f"     [ACTION] Document raw cosine proximity as a statistically significant predictor of retrieval failure.")
    else:
        print(f"     [UNRESOLVED] Predictor is not statistically significant (p = {p_all:.4e}). Ceiling cause remains unresolved.")

    print("=" * 80)


if __name__ == "__main__":
    main()
