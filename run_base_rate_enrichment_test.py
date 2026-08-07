"""
run_base_rate_enrichment_test.py  --  Base-Rate Enrichment & Fisher's Exact Test
=================================================================================

Calculates:
  1. Fraction of 100 facts involved in at least one raw >0.95 confusable pair.
  2. Fraction of 400 test queries whose target fact is in at least one raw >0.95 confusable pair.
  3. 2x2 Contingency Tables ({Failed, Passed} x {Confusable Target, Non-Confusable Target})
     for Trained (200 q), Untrained (200 q), and Combined (400 q).
  4. Fisher's exact test, Odds Ratio (OR), 95% Confidence Intervals, and p-values.
"""

import os
import json
import random
import math
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
    conf_facts = set()
    for i in range(100):
        for j in range(i + 1, 100):
            if S[i, j].item() > threshold:
                pairs.append((i, j, S[i, j].item()))
                conf_facts.add(i)
                conf_facts.add(j)
    return pairs, conf_facts


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


def compute_fisher_exact_and_ci(table):
    """
    table: [[a, b], [c, d]]
      a: Failed & Confusable Target
      b: Failed & Non-Confusable Target
      c: Passed & Confusable Target
      d: Passed & Non-Confusable Target
    """
    res = stats.fisher_exact(table)
    odds_ratio = res.statistic
    p_value = res.pvalue

    # Compute 95% Woolf CI for Odds Ratio
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
    if a > 0 and b > 0 and c > 0 and d > 0:
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_low = np.exp(log_or - 1.96 * se_log_or)
        ci_high = np.exp(log_or + 1.96 * se_log_or)
    else:
        ci_low, ci_high = 0.0, float('inf')

    return odds_ratio, (ci_low, ci_high), p_value


def main():
    print("=" * 80)
    print("  SECTION 10 BASE-RATE ENRICHMENT TEST & FISHER'S EXACT TEST")
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
    conf_pairs, conf_facts = find_confusable_pairs(cache_data, threshold=0.95)
    block_assignment = build_confusable_split_blocks(conf_pairs)
    tr_x, tr_y, te_x, te_y = build_block_tensors(block_assignment, cache_data)

    # 1. Fact Base Rate
    num_conf_facts = len(conf_facts)
    fact_base_rate = num_conf_facts / 100.0
    print(f"\n  1. FACT POPULATION BASE RATE:")
    print(f"     Confusable Facts (in >0.95 pair): {num_conf_facts} / 100 ({fact_base_rate*100:.1f}%)")

    # 2. Test Query Base Rate (400 test queries)
    # Each fact has 4 test queries
    query_confusable_mask = []
    for b in range(10):
        for f in block_assignment[b]:
            is_conf = (f in conf_facts)
            query_confusable_mask.extend([is_conf] * 4) # 4 test queries per fact
    query_confusable_mask = np.array(query_confusable_mask)
    query_base_rate = np.mean(query_confusable_mask)
    print(f"     Test Query Population Base Rate: {np.sum(query_confusable_mask)} / 400 ({query_base_rate*100:.1f}%)")

    # 3. Adapted Evaluation (50 base facts)
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
        trained_results = []
        untrained_results = []

        for b in range(10):
            test_x_b = te_x[b].to(DEVICE)
            test_y_b = te_y[b].to(DEVICE)
            z_queries = adapter(test_x_b)
            sims = torch.matmul(z_queries, z_refs.T)

            for q_idx in range(len(test_y_b)):
                correct_class = test_y_b[q_idx].item()
                pred_idx = torch.argmax(sims[q_idx]).item()
                pred_class = all_train_y[pred_idx].item()
                passed = (pred_class == correct_class)
                is_conf = (correct_class in conf_facts)

                entry = {"passed": passed, "is_confusable": is_conf, "fact": correct_class}
                if b < 5:
                    trained_results.append(entry)
                else:
                    untrained_results.append(entry)

    # Build 2x2 Tables
    def analyze_subset(results, name):
        failed_conf   = sum(1 for r in results if not r["passed"] and r["is_confusable"])
        failed_nconf  = sum(1 for r in results if not r["passed"] and not r["is_confusable"])
        passed_conf   = sum(1 for r in results if r["passed"] and r["is_confusable"])
        passed_nconf  = sum(1 for r in results if r["passed"] and not r["is_confusable"])

        table = [[failed_conf, failed_nconf], [passed_conf, passed_nconf]]
        or_val, (ci_low, ci_high), p_val = compute_fisher_exact_and_ci(table)

        total_failed = failed_conf + failed_nconf
        conf_fail_rate = (failed_conf / total_failed * 100.0) if total_failed > 0 else 0.0

        print(f"\n  2x2 CONTINGENCY TABLE & FISHER EXACT TEST -- {name.upper()}:")
        print(f"     Table [[Failed & Conf, Failed & NonConf], [Passed & Conf, Passed & NonConf]]:")
        print(f"       {table}")
        print(f"     Failures involving Confusable Target: {failed_conf} / {total_failed} ({conf_fail_rate:.1f}%)")
        print(f"     Population Base Rate for Subset:     {sum(1 for r in results if r['is_confusable'])} / {len(results)} ({sum(1 for r in results if r['is_confusable'])/len(results)*100:.1f}%)")
        print(f"     Fisher's Exact Test Odds Ratio (OR): {or_val:.4f}  (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
        print(f"     p-value:                            {p_val:.4e}  ({'STATISTICALLY SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")

        return {
            "table": table,
            "failed_conf_pct": conf_fail_rate,
            "or": or_val,
            "ci": (ci_low, ci_high),
            "p_val": p_val
        }

    tr_res  = analyze_subset(trained_results, "Trained Subset (200 q)")
    un_res  = analyze_subset(untrained_results, "Untrained Subset (200 q)")
    comb_res = analyze_subset(trained_results + untrained_results, "Combined Population (400 q)")

    print("\n  3. VERDICT & ENRICHMENT DECISION:")
    if comb_res["p_val"] >= 0.05 or abs(comb_res["failed_conf_pct"] - query_base_rate*100) < 5.0:
        print("     [DECISION] Query failure rate on confusable facts matches population base rate (~68%).")
        print("     [ACTION] WITHDRAW 'primary binding constraint' claim. Mark ceiling cause as UNRESOLVED.")
    else:
        print(f"     [DECISION] Statistically significant enrichment observed (OR = {comb_res['or']:.2f}, p = {comb_res['p_val']:.4e}).")
        print(f"     [ACTION] KEEP binding constraint claim and quote Odds Ratio {comb_res['or']:.2f} (95% CI: [{comb_res['ci'][0]:.2f}, {comb_res['ci'][1]:.2f}]).")

    print("=" * 80)


if __name__ == "__main__":
    main()
