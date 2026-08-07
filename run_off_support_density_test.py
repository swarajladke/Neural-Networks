"""
run_off_support_density_test.py  --  Off-Support & Local-Density Test (N1 - N6)
==============================================================================

Predictors (all from RAW unadapted embeddings):
  d_q    = max cos(q, r) over the 50 BASE-TRAINED reference vectors (support proximity)
  mean_q = mean cos(q, z_y) over all 99 incorrect centroids (global offset / scale)
  gap_q  = x_q - mean_q (local peakedness)
  k5_q   = mean of top-5 cos(q, z_y) for y != correct (neighbourhood density)

Fits Models N1..N6 with Fact-Clustered SEs:
  N1: Y ~ mean_q
  N2: Y ~ gap_q
  N3: Y ~ d_q
  N4: Y ~ x_q + mean_q
  N5: Y ~ x_q + d_q
  N6: Y ~ k5_q

Additional Audit & Diagnostics:
  - Pearson correlations: corr(x_q, mean_q), corr(x_q, d_q)
  - Fact-level cluster bootstrap 95% CIs on M1 and M3 ROC AUCs (10,000 resamples of facts).
  - Number of DISTINCT FACTS among failures.
  - Mean +/- std of c_q for failed vs passed queries.
  - Threshold 0.0777 Precision (27.2%) & Specificity (77.5%).
"""

import os
import json
import random
import math
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
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
    mlp    = (pos.float() * lp).sum(1) / pos.float().sum(1].clamp_min(1.0)
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


def fit_model(X_df, Y, groups, model_name, var_names):
    """Fits Logit model with cluster-robust SEs and returns summary metrics."""
    X_const = sm.add_constant(X_df)
    model = sm.Logit(Y, X_const)
    res_naive = model.fit(disp=False)
    res_clust = model.fit(cov_type='cluster', cov_kwds={'groups': groups}, disp=False)

    mcfadden_r2 = float(res_naive.prsquared)
    preds_prob  = res_naive.predict(X_const)
    auc_score   = float(roc_auc_score(Y, preds_prob))

    print(f"\n  [{model_name}]")
    print(f"    McFadden R^2 = {mcfadden_r2:.4f}  |  ROC AUC = {auc_score:.4f}")
    for idx, var in enumerate(var_names, start=1):
        b_val = res_naive.params[idx]
        se_c  = res_clust.bse[idx]
        p_c   = res_clust.pvalues[idx]
        print(f"    Variable '{var}': beta = {b_val:+.4f}, Clustered SE = {se_c:.4f}, Clustered p = {p_c:.4e}")

    return {
        "mcfadden_r2": mcfadden_r2,
        "auc": auc_score,
        "res_naive": res_naive,
        "res_clust": res_clust,
        "preds_prob": preds_prob
    }


def fact_bootstrap_auc_ci(X_pred, Y, groups, num_bootstrap=10000):
    """Compute fact-clustered bootstrap 95% CI on ROC AUC by resampling 100 facts."""
    np.random.seed(42)
    unique_facts = np.unique(groups)
    N_facts = len(unique_facts)
    boot_aucs = []

    # Map fact -> array of indices for that fact
    fact_indices = {f: np.where(groups == f)[0] for f in unique_facts}

    for _ in range(num_bootstrap):
        sampled_facts = np.random.choice(unique_facts, size=N_facts, replace=True)
        boot_idx = np.concatenate([fact_indices[f] for f in sampled_facts])

        b_Y = Y[boot_idx]
        if len(np.unique(b_Y)) < 2:
            continue

        b_X = X_pred[boot_idx]
        X_const = sm.add_constant(b_X)
        try:
            model = sm.Logit(b_Y, X_const)
            res = model.fit(disp=False)
            probs = res.predict(X_const)
            boot_aucs.append(roc_auc_score(b_Y, probs))
        except Exception:
            continue

    boot_aucs = np.array(boot_aucs)
    ci_low  = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))
    return ci_low, ci_high


def main():
    print("=" * 80)
    print("  OFF-SUPPORT / LOCAL-DENSITY TEST & FACT-CLUSTERED BOOTSTRAP AUC CIs")
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
    block_assignment = build_confusable_split_blocks(confusable_pairs)
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    # Raw centroids (100 facts)
    cen_raw = torch.zeros(100, INPUT_DIM, device=DEVICE)
    for i in range(100):
        samples = cache_data["train_x"][i*3:(i+1)*3].float().to(DEVICE)
        cen_raw[i] = F.normalize(samples.mean(0, keepdim=True), dim=-1).squeeze(0)

    # Train adapter on 50 base facts (blocks 0..4)
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

    # Raw base-trained reference vectors for d_q calculation (50 vectors)
    raw_base_refs = F.normalize(joint_train_x_base, dim=-1)

    x_q_l, c_q_l, mean_q_l, gap_q_l, k5_q_l, d_q_l = [], [], [], [], [], []
    y_l, fact_l = [], []

    with torch.no_grad():
        z_refs = adapter(all_train_x)

        for b in range(10):
            test_x_b = te_x[b].to(DEVICE)
            test_y_b = te_y[b].to(DEVICE)

            # Raw test queries
            raw_q = F.normalize(test_x_b, dim=-1)
            raw_sims = torch.matmul(raw_q, cen_raw.T)  # (40, 100)
            raw_base_sims = torch.matmul(raw_q, raw_base_refs.T) # (40, 150)

            # Adapted predictions
            z_queries = adapter(test_x_b)
            ad_sims   = torch.matmul(z_queries, z_refs.T)

            for q_idx in range(len(test_y_b)):
                correct_class = test_y_b[q_idx].item()

                # c_q: raw cosine to correct centroid
                c_val = raw_sims[q_idx, correct_class].item()

                # x_q: raw cosine to nearest incorrect centroid
                sim_vec = raw_sims[q_idx].clone()
                sim_vec[correct_class] = -999.0
                x_val = sim_vec.max().item()

                # mean_q: mean cosine over all 99 incorrect centroids
                incorrect_indices = [i for i in range(100) if i != correct_class]
                mean_val = raw_sims[q_idx, incorrect_indices].mean().item()

                # gap_q: local peakedness x_q - mean_q
                gap_val = x_val - mean_val

                # k5_q: mean of top 5 incorrect centroid cosines
                top5_val = torch.topk(raw_sims[q_idx, incorrect_indices], k=5).values.mean().item()

                # d_q: max cosine to base-trained reference vectors
                d_val = raw_base_sims[q_idx].max().item()

                # Post-adaptation failure (evaluated against 300 reference vectors)
                pred_idx = torch.argmax(ad_sims[q_idx]).item()
                pred_class = all_train_y[pred_idx].item()
                is_failed = int(pred_class != correct_class)

                x_q_l.append(x_val)
                c_q_l.append(c_val)
                mean_q_l.append(mean_val)
                gap_q_l.append(gap_val)
                k5_q_l.append(top5_val)
                d_q_l.append(d_val)
                y_l.append(is_failed)
                fact_l.append(correct_class)

    X_q = np.array(x_q_l)
    C_q = np.array(c_q_l)
    Mean_q = np.array(mean_q_l)
    Gap_q = np.array(gap_q_l)
    K5_q = np.array(k5_q_l)
    D_q = np.array(d_q_l)
    Y = np.array(y_l)
    Groups = np.array(fact_l)

    # 1. Failure Audit Details
    failed_mask = (Y == 1)
    failed_facts = np.unique(Groups[failed_mask])
    print(f"\n  1. FAILURE AUDIT DETAILS:")
    print(f"     Total Failures: {int(np.sum(Y))} / {len(Y)} queries ({(1 - np.mean(Y))*100:.2f}% Accuracy)")
    print(f"     Distinct Facts Among Failures: {len(failed_facts)} distinct facts out of {len(np.unique(Groups))} total facts")

    # 2. c_q Range Restriction Check
    print(f"\n  2. c_q RANGE RESTRICTION CHECK:")
    print(f"     Failed Queries c_q: {np.mean(C_q[failed_mask]):.4f} ± {np.std(C_q[failed_mask]):.4f} (Min={np.min(C_q[failed_mask]):.4f}, Max={np.max(C_q[failed_mask]):.4f})")
    print(f"     Passed Queries c_q: {np.mean(C_q[~failed_mask]):.4f} ± {np.std(C_q[~failed_mask]):.4f} (Min={np.min(C_q[~failed_mask]):.4f}, Max={np.max(C_q[~failed_mask]):.4f})")

    # 3. Predictor Correlations
    corr_x_mean, p_x_mean = stats.pearsonr(X_q, Mean_q)
    corr_x_d, p_x_d       = stats.pearsonr(X_q, D_q)
    print(f"\n  3. PREDICTOR PEARSON CORRELATIONS (400 QUERIES):")
    print(f"     corr(x_q, mean_q) = {corr_x_mean:+.4f} (p = {p_x_mean:.4e})")
    print(f"     corr(x_q, d_q)    = {corr_x_d:+.4f} (p = {p_x_d:.4e})")

    # 4. Models N1 - N6
    fit_model(Mean_q, Y, Groups, "N1: Y ~ mean_q (Global Offset)", ["mean_q"])
    fit_model(Gap_q, Y, Groups, "N2: Y ~ gap_q (Local Peakedness)", ["gap_q"])
    fit_model(D_q, Y, Groups, "N3: Y ~ d_q (Support Proximity)", ["d_q"])

    X_n4 = np.column_stack([X_q, Mean_q])
    fit_model(X_n4, Y, Groups, "N4: Y ~ x_q + mean_q (Nearest Incorrect + Global Offset)", ["x_q", "mean_q"])

    X_n5 = np.column_stack([X_q, D_q])
    fit_model(X_n5, Y, Groups, "N5: Y ~ x_q + d_q (Nearest Incorrect + Support Proximity)", ["x_q", "d_q"])

    fit_model(K5_q, Y, Groups, "N6: Y ~ k5_q (Top-5 Neighbourhood Density)", ["k5_q"])

    # 5. Fact-Clustered Bootstrap 95% CIs for M1 and M3 AUCs
    print("\n  5. FACT-CLUSTERED BOOTSTRAP 95% CIs ON M1 AND M3 AUCs (10,000 Resamples of 100 Facts):")
    m1_ci_low, m1_ci_high = fact_bootstrap_auc_ci(X_q, Y, Groups, num_bootstrap=10000)
    print(f"     M1 (Y ~ x_q) AUC Clustered Bootstrap 95% CI: [{m1_ci_low:.4f}, {m1_ci_high:.4f}]")

    X_m3 = np.column_stack([X_q, C_q])
    m3_ci_low, m3_ci_high = fact_bootstrap_auc_ci(X_m3, Y, Groups, num_bootstrap=10000)
    print(f"     M3 (Y ~ x_q + c_q) AUC Clustered Bootstrap 95% CI: [{m3_ci_low:.4f}, {m3_ci_high:.4f}]")

    # 6. Threshold 0.0777 Classification Metrics (M1)
    X_const_m1 = sm.add_constant(X_q)
    m1_logit = sm.Logit(Y, X_const_m1).fit(disp=False)
    probs_m1 = m1_logit.predict(X_const_m1)
    preds_0777 = (probs_m1 >= 0.0777).astype(int)
    cm_0777 = confusion_matrix(Y, preds_0777)
    tn, fp, fn, tp = cm_0777.ravel()

    precision   = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    sensitivity = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0

    print("\n  6. THRESHOLD 0.0777 CLASSIFICATION METRICS (M1 Model):")
    print(f"     Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"     Precision:   {precision:.2f}%")
    print(f"     Sensitivity: {sensitivity:.2f}% (FN = {fn})")
    print(f"     Specificity: {specificity:.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
