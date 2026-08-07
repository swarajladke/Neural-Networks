"""
run_graded_ceiling_reanalysis.py  --  Graded Ceiling Test Statistical Reanalysis
================================================================================

Fits 4 Logistic Regression Models predicting post-adaptation retrieval failure Y in {0, 1}:
  M1: Y ~ x_q           (raw nearest-incorrect centroid cosine)
  M2: Y ~ c_q           (raw correct centroid cosine)
  M3: Y ~ x_q + c_q     (joint model; tests whether x_q or c_q dominates)
  M4: Y ~ m_q           (raw margin m_q = c_q - x_q)

For each model:
  - Naive & Cluster-Robust (by fact ID, 100 clusters) Standard Errors, 95% CIs, p-values.
  - Odds Ratio (OR) per +0.01 predictor change with Naive and Clustered 95% CIs.
  - McFadden's Pseudo-R^2.
  - ROC AUC score & Optimal Threshold Confusion Matrix.
  - Correlation corr(x_q, c_q) across all 400 test queries.
"""

import os
import json
import random
import math
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
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


def fit_and_report_model(X_df, Y, groups, model_name, var_names):
    """
    Fits Logit model with naive and cluster-robust standard errors.
    Reports beta, naive 95% CI, clustered 95% CI, OR per +0.01, McFadden R^2, and AUC.
    """
    X_const = sm.add_constant(X_df)
    
    # 1. Naive Model
    model_naive = sm.Logit(Y, X_const)
    res_naive   = model_naive.fit(disp=False)
    
    # 2. Cluster-Robust Model (grouped by fact ID)
    res_clust   = res_naive.get_robustcov_results(cov_type='cluster', groups=groups)

    # McFadden Pseudo-R^2
    mcfadden_r2 = float(res_naive.prsquared)

    # Predicted probabilities and ROC AUC
    preds_prob = res_naive.predict(X_const)
    auc_score  = float(roc_auc_score(Y, preds_prob))

    # Optimal Threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(Y, preds_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    preds_binary = (preds_prob >= best_thresh).astype(int)
    cm = confusion_matrix(Y, preds_binary)

    print(f"\n  [{model_name}]")
    print(f"    McFadden Pseudo-R^2: {mcfadden_r2:.4f}  |  ROC AUC: {auc_score:.4f}  |  Optimal Thresh: {best_thresh:.4f}")
    print(f"    Confusion Matrix (Thresh = {best_thresh:.4f}): TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    for idx, var in enumerate(var_names, start=1):
        b_val      = res_naive.params[idx]
        
        # Naive CI & p-val
        se_n       = res_naive.bse[idx]
        p_n        = res_naive.pvalues[idx]
        ci_n_low   = b_val - 1.96 * se_n
        ci_n_high  = b_val + 1.96 * se_n
        or_n       = np.exp(0.01 * b_val)
        or_n_low   = np.exp(0.01 * ci_n_low)
        or_n_high  = np.exp(0.01 * ci_n_high)

        # Clustered CI & p-val
        se_c       = res_clust.bse[idx]
        p_c        = res_clust.pvalues[idx]
        ci_c_low   = b_val - 1.96 * se_c
        ci_c_high  = b_val + 1.96 * se_c
        or_c_low   = np.exp(0.01 * ci_c_low)
        or_c_high  = np.exp(0.01 * ci_c_high)

        print(f"    Variable '{var}':")
        print(f"      beta = {b_val:+.4f}")
        print(f"      Naive SE = {se_n:.4f} | Naive p = {p_n:.4e} | Naive OR_0.01 = {or_n:.4f} (95% CI: [{or_n_low:.4f}, {or_n_high:.4f}])")
        print(f"      Clustered SE = {se_c:.4f} | Clustered p = {p_c:.4e} | Clustered OR_0.01 = {or_n:.4f} (95% CI: [{or_c_low:.4f}, {or_c_high:.4f}])")

    return {
        "mcfadden_r2": mcfadden_r2,
        "auc": auc_score,
        "best_thresh": best_thresh,
        "cm": cm.tolist(),
        "res_naive": res_naive,
        "res_clust": res_clust
    }


def main():
    print("=" * 80)
    print("  GRADED CEILING TEST STATISTICAL REANALYSIS (M1, M2, M3, M4)")
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

    # Raw centroids (100 facts)
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

    x_q_list, c_q_list, m_q_list, y_list, fact_id_list = [], [], [], [], []

    with torch.no_grad():
        z_refs = adapter(all_train_x)

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

                # Raw c_q: cosine to correct class centroid
                c_q_val = raw_sims[q_idx, correct_class].item()

                # Raw x_q: cosine to nearest incorrect class centroid
                raw_sim_vec = raw_sims[q_idx].clone()
                raw_sim_vec[correct_class] = -999.0
                x_q_val = raw_sim_vec.max().item()

                # Raw margin m_q = c_q - x_q
                m_q_val = c_q_val - x_q_val

                # Outcome: failed after adaptation
                pred_idx = torch.argmax(ad_sims[q_idx]).item()
                pred_class = all_train_y[pred_idx].item()
                is_failed = int(pred_class != correct_class)

                x_q_list.append(x_q_val)
                c_q_list.append(c_q_val)
                m_q_list.append(m_q_val)
                y_list.append(is_failed)
                fact_id_list.append(correct_class)

    X_q = np.array(x_q_list)
    C_q = np.array(c_q_list)
    M_q = np.array(m_q_list)
    Y   = np.array(y_list)
    Groups = np.array(fact_id_list)

    total_failures = int(np.sum(Y))
    print(f"\n  AUDIT CHECK: Total Failures Post-Adaptation = {total_failures} / {len(Y)} queries ({(1-total_failures/len(Y))*100:.2f}% Accuracy)")

    # Compute correlation between x_q and c_q over all 400 queries
    corr_x_c, p_corr = stats.pearsonr(X_q, C_q)
    print(f"\n  CORRELATION BETWEEN RAW PREDICTORS OVER ALL 400 QUERIES:")
    print(f"    corr(x_q, c_q) = {corr_x_c:+.4f} (p = {p_corr:.4e})")

    # Fit Models M1, M2, M3, M4
    m1_res = fit_and_report_model(X_q, Y, Groups, "M1: Y ~ x_q (Nearest Incorrect)", ["x_q"])
    m2_res = fit_and_report_model(C_q, Y, Groups, "M2: Y ~ c_q (Correct Centroid)", ["c_q"])
    
    X_m3 = np.column_stack([X_q, C_q])
    m3_res = fit_and_report_model(X_m3, Y, Groups, "M3: Y ~ x_q + c_q (Joint Model)", ["x_q", "c_q"])
    
    m4_res = fit_and_report_model(M_q, Y, Groups, "M4: Y ~ m_q (Raw Margin m_q = c_q - x_q)", ["m_q"])

    # Decision Rule Evaluation
    print("\n" + "=" * 80)
    print("  DECISION RULE EVALUATION & CEILING CAUSE RESOLUTION")
    print("=" * 80)

    p_m3_x = m3_res["res_clust"].pvalues[1]
    p_m3_c = m3_res["res_clust"].pvalues[2]
    beta_m3_x = m3_res["res_clust"].params[1]
    beta_m3_c = m3_res["res_clust"].params[2]

    if m4_res["auc"] >= max(m1_res["auc"], m2_res["auc"], m3_res["auc"]):
        print(f"  [DECISION: MARGIN LIMIT] Model M4 (Raw Margin m_q) achieves peak ROC AUC = {m4_res['auc']:.4f} (McFadden R^2 = {m4_res['mcfadden_r2']:.4f}).")
        print("  [CONCLUSION] The retrieval accuracy ceiling is a MARGIN LIMIT. Initial margin pre-adaptation dictates post-adaptation success.")
    elif p_m3_c < 0.05 and p_m3_x >= 0.05:
        print(f"  [DECISION: ENCODER ALIGNMENT LIMIT] c_q dominates in M3 (p_c = {p_m3_c:.4e}) while x_q loses significance (p_x = {p_m3_x:.4e}).")
        print("  [CONCLUSION] The ceiling is an ENCODER ALIGNMENT limit. Raw confusability is refuted.")
    elif beta_m3_x < 0 and p_m3_x < 0.05:
        print(f"  [DECISION: OPEN FINDING] x_q retains a statistically significant negative coefficient (beta = {beta_m3_x:.4f}, p = {p_m3_x:.4e}) after controlling for c_q.")
        print("  [CONCLUSION] Counterintuitive geometry interaction: report as an OPEN FINDING.")

    print("=" * 80)


if __name__ == "__main__":
    main()
