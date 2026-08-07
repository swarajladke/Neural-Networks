# Orthogonal Gradient Projection for Continual Retrieval Adaptation
## Empirical Results Report

---

## 1. Benchmark Definition

**Task**: 1-nearest-neighbour retrieval over a stored reference set.

**Corpus**: 100 synthetic facts, each with 3 training reference sentences and 4 test queries (300 train, 400 test).

**Block Structure**: 10 blocks of 10 facts each, `CONFUSABLE-SPLIT` protocol. Confusable pairs (cosine similarity > 0.95 between fact centroids) are placed in *different* blocks to maximise inter-block interference. Block ordering shuffled uniformly at random per run.

**Training Protocol per Block**: `epochs = 30`, `lr = 1e-3`, `weight_decay = 1e-4`. Loss: Supervised Contrastive Loss, temperature `tau = 0.05`. Optimiser: AdamW.

**Adapter Architecture**: Linear projection W in R^{960x960}, bias b in R^{960}, output L2-normalised. Total trainable parameters: **922,560**. Initialised to identity.

**Encoder**: SmolLM2-360M (HuggingFaceTB/SmolLM2-360M). **Frozen throughout.** All embeddings are mean-pooled last-hidden-layer representations (dim = 960).

**Evaluation Metric**: Final accuracy A_T = mean per-block test accuracy at step 9. A_T resolution: 0.25% (400 test queries total). Per-block accuracy resolution: 2.5% (40 queries per block).

**Run Configuration**: 50 runs per condition = 10 shuffles x 5 seeds. Selection Seeds {101..105}; Fresh Replication Seeds {211..215}. Fully disjoint.

---

## 2. Frozen Encoder Floor & Plasticity Precondition

**Frozen Adapter (Identity, no training)**: A_T(frozen) = **72.50% +/- 0.00%** (live assertion; every run is identical by construction).

**Plasticity Precondition -- offline (joint retraining) vs frozen**:
- Selection: A_T(offline) - A_T(frozen) = 94.98% - 72.50% = **+22.48%**
- Fresh: A_T(offline) - A_T(frozen) = 94.45% - 72.50% = **+21.95%**

The adapter learns substantially (~+22 points). Mechanisms are evaluated by how much of A_T(offline) - A_T(method) they close. Frozen establishes only that the adapter can learn.

**Naive Sequential Fine-tuning** (train each block in sequence, no memory):

| Metric | Selection (50 runs) | Fresh (50 runs) |
|:---|:---:|:---:|
| A_T | 89.62% +/- 2.11% (86.75..92.25%) | 90.48% +/- 2.62% (86.75..93.75%) |
| LA (Learning Accuracy) | 90.67% +/- 1.64% | 90.43% +/- 1.74% |
| BWT (= A_T - LA) | **-1.05%** | **+0.05%** |
| Observed Forgetting | 3.42% +/- 1.25% | 2.65% +/- 1.83% |

**Offline Joint Upper Bound**:

| Metric | Selection (50 runs) | Fresh (50 runs) |
|:---|:---:|:---:|
| A_T | 94.98% +/- 0.74% (94.00..95.75%) [CONFIRMED] | 94.45% +/- 1.31% (92.00..96.50%) |
| LA | 93.05% +/- 1.02% | 92.70% +/- 1.26% |
| BWT | **+1.93%** | **+1.75%** |
| Observed Forgetting | 0.65% +/- 0.32% | 0.72% +/- 0.71% |

**Continual Learning Gap**:
- Selection: 94.98% - 89.62% = **+5.35%**
- Fresh: 94.45% - 90.48% = **+3.97%**

> **Cross-set heterogeneity**: The fresh seed set shows a *smaller* CL gap (3.97% vs 5.35%) yet produces *larger* OGP gains. Naive BWT shifts from -1.05 (selection) to +0.05 (fresh). The particular draw of block orderings dominates the numeric outcome more than rank-budget tuning does within [4, 128].

---

## 3. OGP Rank Sweep -- Per-Seed-Set Results (50 Runs per Condition per Seed Set)

**Method**: After training on base blocks 0-4, before each sequential block step t in [5, 9]:
1. Accumulate past training inputs M in R^{N_past x 960}.
2. SVD: right singular vectors V_k in R^{960 x k}.
3. Gradient projection applied to grad_W only: grad_W <- grad_W * (I - V_k V_k^T). The 960-dim bias b is unprojected (see Limitation 6).

At step 9, N_past = 270 (9 blocks x 30 refs). k protected directions; 960 - k free.

**Metric definitions**:
- A_T: final mean per-block test accuracy.
- LA: mean accuracy at each block's first-seen step (populated-row guard: base phase blocks use row 4 of R).
- BWT = A_T - LA (exact identity).
- Observed Forgetting = mean_j[max_{t>=t_j} R[t,j] - R[9,j]] (robust to unpopulated rows; does NOT satisfy A_T = LA - Fgt).
- Paired 95% CI: 10,000-sample bootstrap on within-run differences vs naive.

All values below are per-seed-set; the `[sel | fre]` format shows Selection Seeds 101-105 and Fresh Seeds 211-215 separately. No single pooled estimate is reported because the two seed sets differ in naive BWT by 1.10 pp and in OGP gain by up to 0.47 pp, making an unweighted mean potentially misleading.

| k | A_T [sel\|fre] | LA [sel\|fre] | BWT [sel\|fre] | Obs.Fgt [sel\|fre] | CI sel | CI fre | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive | 89.62\|90.48 | 90.67\|90.43 | -1.05\|+0.05 | 3.42\|2.65 | +0.00 [+0.00,+0.00] | +0.00 [+0.00,+0.00] | Baseline |
| 4 | 91.60\|91.65 | 91.18\|90.77 | +0.42\|+0.88 | 2.53\|2.08 | +1.98 [+1.51,+2.43] | +1.17 [+0.78,+1.58] | Both sig. |
| 6 | 90.10\|91.20 | 90.65\|90.75 | -0.55\|+0.45 | 3.35\|2.47 | +0.47 [-0.07,+1.03] | +0.72 [+0.25,+1.20] | Null\|Sig -- DISAGREE |
| 8 | 89.65\|91.42 | 90.58\|90.75 | -0.93\|+0.67 | 3.57\|2.30 | +0.02 [-0.53,+0.59] | +0.95 [+0.54,+1.37] | Null\|Sig -- DISAGREE |
| 10 | 90.40\|92.60 | 91.07\|91.35 | -0.67\|+1.25 | 3.52\|1.67 | +0.77 [+0.45,+1.07] | +2.12 [+1.61,+2.66] | Both sig. |
| 12 | 91.38\|92.83 | 91.40\|91.45 | -0.02\|+1.38 | 3.10\|1.95 | +1.75 [+1.33,+2.19] | +2.35 [+1.88,+2.83] | Both sig. |
| 16 | 91.67\|93.15 | 91.08\|91.53 | +0.59\|+1.62 | 2.62\|1.35 | +2.05 [+1.65,+2.42] | +2.67 [+2.12,+3.31] | Both sig. |
| **24 (headline)** | **91.88\|93.20** | **91.40\|91.63** | **+0.48\|+1.57** | **2.02\|1.27** | **+2.25 [+1.79,+2.70]** | **+2.72 [+2.31,+3.16]** | **Both sig.** |
| **32 (anchor)** | **91.60\|92.55** | **91.32\|91.38** | **+0.28\|+1.17** | **2.25\|1.40** | **+1.97 [+1.55,+2.41]** | **+2.07 [+1.58,+2.56]** | **Both sig.** |
| 64 | 90.33\|91.75 | 91.15\|91.40 | -0.82\|+0.35 | 2.62\|2.45 | +0.70 [+0.33,+1.06] | +1.27 [+0.60,+1.99] | Both sig. |
| 128 | 89.70\|91.78 | 91.00\|91.33 | -1.30\|+0.45 | 3.02\|2.12 | +0.07 [-0.26,+0.40] | +1.30 [+0.68,+1.95] | Null\|Sig -- DISAGREE |
| Offline | 94.98\|94.45 | 93.05\|92.70 | +1.93\|+1.75 | 0.65\|0.72 | +5.35 [+4.75,+5.95] | +3.97 [+3.28,+4.66] | Upper Bound |

**Headline (k=24)**: Highest A_T on both seed sets. No resolvable single optimum in k in [4, 32] -- all CIs overlap -- but k=24 gives the highest point estimate on both.

**Robustness Anchor (k=32)**: Smallest between-set spread in paired gain: +1.97 vs +2.07, range 0.10 pp. Primary reproducibility anchor.

**Cross-set disagreements (k=6, 8, 128)**: k=128 additionally measured significantly worse in the preliminary 15-run sweep (-0.75, CI [-1.18,-0.35]), null on selection-50 (+0.07, [-0.26,+0.40]), and significantly better on fresh-50 (+1.30, [+0.68,+1.95]) -- three mutually disjoint CIs for the same condition on the same original seed family. **15-run paired CIs were over-confident. No 15-run result in this project should be cited without 50-run confirmation.**

---

## 4. Control Arms (k=32, 50 Runs per Condition per Seed Set)

| Arm | Description | A_T [sel\|fre] | BWT [sel\|fre] | CI sel | CI fre | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| TOP-32 (OGP) | Top-32 right singular vectors of accumulated past inputs M | 91.60\|92.55 | +0.28\|+1.17 | +1.97 [+1.55,+2.41] | +2.07 [+1.58,+2.56] | Sig. both |
| RANDOM-32 | Random rank-32 orthonormal subspace, resampled per run | 89.66\|90.37 | -0.88\|+0.00 | +0.03 [-0.17,+0.24] | -0.11 [-0.25,+0.03] | True null both |
| BOTTOM-32 | Bottom-32 right singular vectors of past inputs M | 89.70\|90.48 | -0.98\|+0.08 | +0.07 [+0.01,+0.15] | -0.00 [-0.09,+0.09] | Negligible (sel CI width 0.14 vs std 2.20) |
| CURRENT-32 | Top-32 singular vectors of current block only | 87.78\|86.35 | -1.29\|-2.15 | -1.85 [-2.48,-1.25] | -4.13 [-5.29,-2.95] | Sig. worse both |

**Mechanistic conclusion**: RANDOM-32 is null on both seed sets (OGP is not a generic rank constraint). BOTTOM-32 has a negligible net A_T effect despite confirmed weight displacement (max|W_bot - W_naive| = 0.055; R matrices differ by up to 5.0 pp per block, but the aggregate A_T change is negligible at 0.25% quantisation resolution). CURRENT-32 significantly degrades A_T and BWT on both seed sets. Only TOP-32 (top principal directions of the accumulated past-input matrix) produces the gain. The mechanism is confirmed as described.

---

## 5. Exact Decomposition (delta_A_T = delta_LA + delta_BWT)

A_T = LA + BWT is an exact identity (BWT = A_T - LA by definition). Observed Forgetting (max-based) is reported separately and does NOT satisfy this identity.

**At k=24 vs naive:**

| Seed Set | delta_A_T | delta_LA (Acquisition share) | delta_BWT (Retention share) | Closure |
|:---|:---:|:---:|:---:|:---:|
| Selection (naive BWT=-1.05, k24 BWT=+0.48) | +2.26% | +0.73% (32.3%) | +1.53% (67.7%) | 0.73+1.53=2.26 EXACT |
| Fresh (naive BWT=+0.05, k24 BWT=+1.57) | +2.72% | +1.20% (44.1%) | +1.52% (55.9%) | 1.20+1.52=2.72 EXACT |

OGP improves both retention (55-68% of total gain) and acquisition (32-44%). The acquisition improvement likely arises because projecting out the dominant past-activation subspace acts as a structured regulariser, constraining the optimiser to directions less prone to conflicting gradients across blocks.

---

## 6. Prior Art & Defensible Contributions

**Prior Art:**
- Farajtabar et al., "Orthogonal Gradient Descent for Continual Learning," AISTATS 2020.
- Saha, Garg & Roy, "Gradient Projection Memory for Continual Learning," ICLR 2021.

Both project task gradients into the complement of a stored basis built from past inputs. This work applies the same mechanism and does NOT claim novelty of the core algorithm.

**Defensible Contributions:**

1. **Domain application**: Transfer to a *frozen*-pretrained-encoder linear retrieval adapter under Supervised Contrastive Loss, rather than a learned classifier with cross-entropy.

2. **Rank-budget sweep with 50-run resolution**: Systematic evaluation of k in {4,6,8,10,12,16,24,32,64,128} establishing the stability-plasticity boundary. Gains are significant on both seed sets for k in {4,10,12,16,24,32,64}. No single optimum is resolvable within [4,32] at 50-run resolution.

3. **Three-way mechanistic control**: RANDOM-32 (generic rank constraint), BOTTOM-32 (non-principal past directions), CURRENT-32 (current-only subspace). All three are null or significantly negative while TOP-32 is significantly positive on both seed sets -- a pattern not standard in GPM papers.

4. **Inference about coordinate-aligned masking**: RANDOM-32 is null on both seed sets, indicating that a subspace unaligned with past activation geometry confers no protection. This predicts that coordinate-aligned weight masking (AGNIS's R_mask "synaptic shielding", i.e. GPM in the standard basis) would likewise be inert. This is consistent with R_mask never having produced a measurable effect -- however, R_mask was never validly executed (the mask term was absent from the weight update in run_control_battery.py and Control E could not run due to the task_idx oracle requirement). The inertness of R_mask remains an inference, not a measurement.

---

## 7. Headline Summary

OGP with k in [16, 32] improves final accuracy by **+2.0 to +2.7 percentage points** over naive sequential fine-tuning (paired 95% CIs excluding zero on two independent seed sets of 50 runs each), and raises BWT from -1.05/+0.05 (naive) to +0.28/+1.17 (k=32) and +0.48/+1.57 (k=24). At k=24, OGP raises BWT by **+1.52 to +1.53 points** and reduces observed forgetting by **1.38 to 1.40 points** (both CIs exclude zero on both seed sets).

k=24 shows the highest mean A_T on both seed sets and is the headline result. k=32 is the robustness anchor (between-set gain spread 0.10 pp). Gap recovery at k=24 is 42% (selection, gap 5.35%) to 69% (fresh, gap 3.97%); at k=32 it is 37% to 52%. These percentages vary because the naive-offline gap differs between seed sets. The absolute gain (+2.0 to +2.7 points) is the more stable quantity and should be the primary citation.

---

## 8. Limitations

1. **4-to-5 point ceiling**: The measurable CL gap is 3.97-5.35 points. At k=24, OGP recovers 42% on the selection set and 69% on the fresh set; at k=32, 37% and 52%. The remaining gap is not addressed.

2. **Cross-set verdict disagreements at k=6, 8, 128**: Three rank values produce opposite verdicts across independent 50-run seed sets. Block-ordering variance accounts for more outcome variance than rank-budget tuning does.

3. **15-run results are over-confident**: The 15-run preliminary sweep produced CIs that disagreed with 50-run results on the same seed family (k=128: -0.75 [-1.18,-0.35] at 15 runs vs +0.07 [-0.26,+0.40] at 50 runs). No 15-run result from this project should be cited without 50-run confirmation.

4. **Single-task retrieval benchmark**: All 10 blocks are the same task type (1-NN retrieval from a fixed corpus). The mechanism has not been evaluated under multi-task classification. Positive transfer across similar tasks may mask forgetting that would appear under genuinely dissimilar task distributions.

5. **Synthetic corpus with engineered interference**: Confusable-pair placement was constructed to maximise inter-block interference. Results may not generalise to naturally occurring corpora.

6. **Bias vector unprojected**: The gradient projection (I - V_k V_k^T) is applied to grad_W only. The 960-dim bias b is updated without projection, so each gradient step shifts outputs by delta_b in all directions including the protected subspace. The empirical effect of this gap was not measured in this suite. Either extend the projection to include b, or treat the k-direction protection as approximate.

---

## 9. Phase 2 Calibrated Forgetting Benchmark Results (50 Runs per Condition per Seed Set)

**Calibrated Configuration**: `BottleneckAdapter` $r=32$ (uncentred PCA init), `epochs = 100`, `lr = 1e-2`, `weight_decay = 1e-4`.

Under this configuration, naive sequential fine-tuning exhibits severe catastrophic forgetting (BWT $= -12.35\%$ / $-9.15\%$, CL gap $= +19.72\text{ pp}$ / $+16.62\text{ pp}$), while offline joint training reaches $94.20\%$ / $94.15\%$ $A_T$.

### Disclosure D1: Phase 1 Axis Progression & Forgetting Attribution

| Cell | Config | Naive A_T | Naive LA | Naive BWT | Offline A_T | CL Gap | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `A_r32` | $r=32, \text{ep}=30, \text{lr}=1\text{e-}3$ | 85.08% | 81.25% | **+3.83%** | 93.25% | +8.17% | Mild |
| `B_r32_ep100` | $r=32, \text{ep}=100, \text{lr}=1\text{e-}3$ | 89.92% | 88.83% | **+1.08%** | 95.25% | +5.33% | Mild |
| `C_r32_ep100_lr1e-02` | $r=32, \text{ep}=100, \text{lr}=1\text{e-}2$ | 75.08% | 87.58% | **-12.50%** | 95.17% | +20.08% | **Target Met** |

**Attribution**: Catastrophic forgetting is directly attributable to the high learning rate ($\text{lr}=10^{-2}$) acting within the constrained bottleneck ($r=32$). Neither $r=32$ alone at $\text{lr}=10^{-3}$ nor $\text{lr}=10^{-2}$ at full rank ($r=960$) produces BWT $< -10\%$.

### Disclosure D2: Parameterisation Projection Specification
In the `BottleneckAdapter` ($W = U V$), gradient projection is applied to `grad_V` (shape $32 \times 960$), multiplying by $\text{proj\_mat} = I_{960} - P P^\top$ (shape $960 \times 960$). Because $V \in \mathbb{R}^{32 \times 960}$, projecting along 960-d space with rank $k \le 32$ restricts the 960-d input space without annihilating $V$.

### Leading Results & S1/S2 Corrections:
- **Gap Recovery (S2)**: At $k=32$, OGP recovers **14.7%** of the CL gap on Selection ($2.90 / 19.72$) and **13.2%** on Fresh ($2.20 / 16.62$), down from $37\%\text{--}52\%$ in the mild regime.
- **Retention Share Direction (S1)**: Under real catastrophic forgetting, OGP's gain becomes **MORE acquisition-driven** ($\Delta LA = +2.63\text{ pp} / +1.45\text{ pp}$ vs $\Delta BWT = +0.27\text{ pp} / +0.75\text{ pp}$; retention share fell to **9.5% – 34.1%**, down from $55.9\%\text{--}67.7\%$ in the mild regime). This weakens the pure memory-protection interpretation.

### Selection Seeds 101..105 (50 Runs per Condition):

| Condition | $A_T$ (Min..Max) | $LA$ | BWT ($A_T - LA$) | Obs. Fgt | $\Delta A_T$ vs Naive (95% CI) | $\Delta BWT$ vs Naive (95% CI) | Retention Share | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive** | 74.47% ± 5.48% (64.75..81.00%) | 86.83% | -12.35% | 13.77% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | 0.0% | Baseline |
| **Offline** | 94.20% ± 1.36% (92.25..96.50%) | 92.88% | +1.32% | 1.18% | +19.72% [+18.15%, +21.25%] | +13.67% [+12.15%, +15.20%] | 69.3% | Upper Bound |
| **OGP (k=2)** | 77.10% ± 4.82% (66.50..84.25%) | 88.10% | -11.00% | 12.72% | +2.63% [+1.04%, +4.17%] | +1.35% [+0.06%, +2.62%] | 51.4% | 🎉 **SUCCESS** |
| **OGP (k=4)** | 75.60% ± 3.94% (69.75..82.50%) | 88.85% | -13.25% | 14.20% | +1.13% [-0.65%, +2.83%] | -0.90% [-2.46%, +0.60%] | -80.0% | True Null |
| **OGP (k=8)** | 74.75% ± 5.36% (67.75..85.50%) | 88.92% | -14.18% | 15.28% | +0.27% [-1.23%, +1.79%] | -1.82% [-3.22%, -0.52%] | -663.6% | True Null |
| **OGP (k=12)** | 77.95% ± 5.43% (68.00..85.00%) | 89.15% | -11.20% | 12.70% | +3.48% [+1.62%, +5.32%] | +1.15% [-0.57%, +2.84%] | 33.1% | 🎉 **SUCCESS** |
| **OGP (k=16)** | 76.90% ± 7.81% (65.25..90.75%) | 89.25% | -12.35% | 13.82% | +2.43% [+0.41%, +4.34%] | +0.00% [-1.80%, +1.63%] | 0.0% | 🎉 **SUCCESS** |
| **OGP (k=24)** | 80.05% ± 4.63% (73.75..86.50%) | 89.82% | -9.78% | 11.32% | +5.58% [+3.98%, +7.06%] | +2.58% [+1.13%, +3.90%] | 46.2% | 🎉 **SUCCESS** |
| **OGP (k=32)** | 77.38% ± 7.96% (57.50..88.00%) | 89.45% | -12.08% | 13.15% | +2.90% [+0.02%, +5.47%] | +0.27% [-2.38%, +2.54%] | 9.5% | 🎉 **SUCCESS** |
| **RANDOM-32** | 73.36% ± 6.39% (59.50..86.00%) | 86.59% | -13.23% | 14.83% | -1.12% [-2.34%, +0.05%] | -0.88% [-1.95%, +0.20%] | 79.0% | True Null |
| **BOTTOM-32** | 73.90% ± 5.05% (64.75..81.50%) | 86.92% | -13.02% | 14.35% | -0.57% [-1.06%, -0.10%] | -0.67% [-1.10%, -0.27%] | 117.4% | ❌ **SIG WORSE** |
| **CURRENT-32**| 83.80% ± 4.95% (75.50..90.25%) | 89.60% | -5.80% | 7.27% | +9.33% [+8.11%, +10.46%] | +6.55% [+5.38%, +7.65%] | 70.2% | Pending Controls C1-C4 |

### Fresh Replication Seeds 211..215 (50 Runs per Condition):

| Condition | $A_T$ (Min..Max) | $LA$ | BWT ($A_T - LA$) | Obs. Fgt | $\Delta A_T$ vs Naive (95% CI) | $\Delta BWT$ vs Naive (95% CI) | Retention Share | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive** | 77.53% ± 4.60% (71.00..85.00%) | 86.67% | -9.15% | 10.82% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | 0.0% | Baseline |
| **Offline** | 94.15% ± 1.66% (91.50..97.00%) | 91.55% | +2.60% | 1.07% | +16.62% [+15.00%, +18.25%] | +11.75% [+10.15%, +13.35%] | 70.7% | Upper Bound |
| **OGP (k=2)** | 74.58% ± 5.54% (64.25..86.50%) | 86.92% | -12.35% | 14.12% | -2.95% [-4.25%, -1.70%] | -3.20% [-4.50%, -1.88%] | 108.5% | ❌ **SIG WORSE** |
| **OGP (k=4)** | 77.53% ± 5.80% (62.50..84.25%) | 88.60% | -11.07% | 12.28% | -0.00% [-2.01%, +1.94%] | -1.92% [-3.67%, -0.21%] | 0.0% | True Null |
| **OGP (k=8)** | 75.35% ± 7.10% (63.25..86.00%) | 87.70% | -12.35% | 13.67% | -2.18% [-4.22%, +0.00%] | -3.20% [-5.11%, -1.19%] | 147.1% | True Null |
| **OGP (k=12)** | 75.65% ± 5.00% (65.75..85.25%) | 88.10% | -12.45% | 13.68% | -1.88% [-3.68%, +0.01%] | -3.30% [-4.89%, -1.62%] | 176.0% | True Null |
| **OGP (k=16)** | 73.50% ± 6.88% (63.50..84.50%) | 87.40% | -13.90% | 15.07% | -4.03% [-5.78%, -2.20%] | -4.75% [-6.52%, -2.92%] | 118.0% | ❌ **SIG WORSE** |
| **OGP (k=24)** | 78.77% ± 4.94% (69.50..87.50%) | 88.15% | -9.37% | 11.05% | +1.25% [-0.30%, +2.74%] | -0.22% [-1.63%, +1.14%] | -18.0% | True Null |
| **OGP (k=32)** | 79.73% ± 6.19% (71.50..91.00%) | 88.12% | -8.40% | 10.12% | +2.20% [+0.25%, +4.29%] | +0.75% [-1.00%, +2.52%] | 34.1% | 🎉 **SUCCESS** |
| **RANDOM-32** | 77.27% ± 4.88% (66.50..86.75%) | 86.67% | -9.40% | 11.12% | -0.26% [-1.49%, +0.97%] | -0.25% [-1.38%, +0.87%] | 100.0% | True Null |
| **BOTTOM-32** | 76.60% ± 5.00% (68.25..85.75%) | 86.42% | -9.82% | 11.72% | -0.92% [-1.84%, -0.04%] | -0.67% [-1.48%, +0.09%] | ❌ **SIG WORSE** |
| **CURRENT-32**| 82.25% ± 3.61% (77.25..90.50%) | 86.82% | -4.58% | 6.62% | +4.72% [+3.00%, +6.46%] | +4.57% [+3.21%, +6.10%] | 96.8% | Pending Controls C1-C4 |

### S3 Note on CURRENT-32
CURRENT-32 shows elevated performance (+9.33% / +4.72% A_T). Decisive controls C1-C4 (learning rate sweep, freeze-after-base, gradient-norm ratio logging, and gradient-clip control) are pending to determine whether this reflects step-size damping or genuine regularisation. No mechanistic claims are made for CURRENT-32 until C1-C4 complete.

---

*All metrics computed with populated-row guard on R matrix (base phase row 4 evaluated after joint training). Decomposition uses exact BWT = A_T - LA identity. All CIs: 10,000-sample paired bootstrap. Repository: github.com/swarajladke/Neural-Networks, HEAD commit e99a051.*

