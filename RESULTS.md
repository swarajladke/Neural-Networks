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
| **CURRENT-32**| 83.80% ± 4.95% (75.50..90.25%) | 89.60% | -5.80% | 7.27% | +9.33% [+8.11%, +10.46%] | +6.55% [+5.38%, +7.65%] | 70.2% | Refuted (Off-Switch Artifact) |

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
| **CURRENT-32**| 82.25% ± 3.61% (77.25..90.50%) | 86.82% | -4.58% | 6.62% | +4.72% [+3.00%, +6.46%] | +4.57% [+3.21%, +6.10%] | 96.8% | Refuted (Off-Switch Artifact) |

### Decisive Controls C1–C4 Summary

1. **Control C3 (Gradient Norm Ratio)**:
   - `TOP-32`: Mean ratio $||\text{grad\_proj}|| / ||\text{grad\_raw}|| = \mathbf{0.116224}$ ($11.62\%$ gradient retained).
   - `CURRENT-32`: Mean ratio $||\text{grad\_proj}|| / ||\text{grad\_raw}|| = \mathbf{0.000005}$ ($99.9995\%$ gradient annihilated; projected norm $= 0.000000$).
   - **Conclusion**: `CURRENT-32` is a literal **gradient off-switch**. Because the adapter has rank $32$ and `CURRENT-32` projects orthogonal to the 32 principal directions of the current batch, $P P^\top \approx I_{32}$, annihilating virtually all gradient updates during sequential steps 5..9.

2. **Control C2 (Freeze-After-Base Null Hypothesis)**:
   - `FREEZE-AFTER-BASE` (zero adaptation for blocks 5..9): Selection $A_T = \mathbf{88.95\%}$, Fresh $A_T = \mathbf{86.68\%}$.
   - **Conclusion**: Zero adaptation significantly **outperforms `CURRENT-32`** ($83.80\%$ / $82.25\%$). `CURRENT-32` is an imperfect off-switch that allowed minor leakage of destructive updates, performing worse than doing no updates at all.

3. **Control C1 (Naive Fine-Tuning at Lower Learning Rates)**:
   - `naive_lr1e-3` ($\text{lr}=10^{-3}$, $\text{ep}=100$): Selection $A_T = \mathbf{85.67\%}$, Fresh $A_T = \mathbf{85.32\%}$.
   - **Conclusion**: Lowering the learning rate to $10^{-3}$ **outperforms `CURRENT-32`** on both seed sets without any subspace projection.

4. **Control C4 (Gradient Clip Control)**:
   - `GRADIENT-CLIP-C4` (naive $\text{lr}=10^{-2}$ clipped to near-zero norm): Selection $A_T = \mathbf{88.95\%}$, Fresh $A_T = \mathbf{86.65\%}$. Replicates `FREEZE-AFTER-BASE` exactly.

**Final Verdict**: `CURRENT-32`'s elevated score was a trivial gradient annihilation artifact. It is refuted as a selective regular## 10. Phase 3: Parametric Memory Benchmark & C2 Breakdown Results

## 10. Phase 3: Parametric Memory Benchmark & C2 Breakdown Results

### 10.1 Raw Array Verification & C2 Step-9 Reconciliation (VERIFIED)
- **Raw File Checksum**: Dumped to `c2_raw_arrays.json` (SHA-256: `533bfdae6847efa704614de9df41f67b6c92a76591010489e5872019234857bc`).
- **Unweighted Mean Reconciliation**: Computed directly from `c2_raw_arrays.json` by `dump_c2_raw_data.py`:
  - Base-Trained Blocks (`order[0:5]`): **91.05% ± 1.72%** (Selection) / **90.20% ± 2.49%** (Fresh)
  - Never-Trained Blocks (`order[5:10]`): **86.85% ± 3.51%** (Selection) / **83.15% ± 4.55%** (Fresh)
  - Unweighted Mean: **88.95%** (Selection) / **86.68%** (Fresh) — matches measured overall C2 $A_T$ ($88.95\% \pm 2.41\%$ / $86.67\% \pm 3.44\%$) down to $0.01\%$.

#### Empirically Verified Replacement Tests, Scale-Invariant Margin Test & Failure Audit (`run_section10_final_verification.py`)
1. **Retraction of $W$-Decomposition**:
   - In $d=960$ dimensions, a rank-32 matrix $W$ has at most 32 non-zero eigenvalues, whereas $a I$ requires rank 960. The maximum theoretical variance explainable by $a I + b \cdot \mathbf{1}\mathbf{1}^\top$ for rank 32 in 960-d is $\le 3.33\%$. The $0.86\%$ measurement was uninformative because the null hypothesis was structurally unreachable under low rank.
2. **Scale-Invariant Margin Test ($m = \cos(q, z_{\text{correct}}) - \max_{y \neq \text{correct}} \cos(q, z_y)$)**:
   - *Raw Baseline (400 Test Queries)*: Trained $m = \mathbf{+0.0065 \pm 0.0121}$ (Error $= 28.50\%$, Acc $= 71.50\%$); Untrained $m = \mathbf{+0.0059 \pm 0.0100}$ (Error $= 26.50\%$, Acc $= 73.50\%$).
   - *Adapted Map (50 Base Facts)*: Trained $m = \mathbf{+0.2671 \pm 0.1977}$ (Error $= \mathbf{8.50\%}$, Acc $= \mathbf{91.50\%}$); Untrained $m = \mathbf{+0.2185 \pm 0.2126}$ (Error $= \mathbf{15.50\%}$, Acc $= \mathbf{84.50\%}$).
   - *Exact Alignment*: The fraction $m < 0$ equals the observed retrieval error rate **EXACTLY** ($8.50\%$ trained, $15.50\%$ untrained). Margin analysis and retrieval harness are in **100% mathematical agreement**.
3. **Retrieval Ceiling Graded Continuous Predictor Test & Fact Map Audit (`audit_fact_map_and_c_q_bug.py`)**:
   - *Data Change Correction & Retraction Notice (ITEM 3)*:
     - **RETRACTED**: Prior M1 ROC AUC $0.8954$ is **RETRACTED** $\rightarrow$ New canonical M1 ROC AUC is **0.7994** (300-sample 1-NN outcome, 48 failures) / **0.8217** (100-centroid 1-NN outcome, 41 failures).
     - **RETRACTED**: Prior McFadden $R^2 = 0.2485$ is **RETRACTED** $\rightarrow$ New canonical McFadden $R^2$ is **0.1055** (300-sample outcome) / **0.1204** (100-centroid outcome).
     - **RETRACTED**: Prior $\text{corr}(x_q, d_q) = 0.8729$ is **RETRACTED** $\rightarrow$ New canonical $\text{corr}(x_q, d_q)$ is **+0.7068**.
     - *Cause*: Prior run assigned class centroids by arbitrary block slice index `i*3:(i+1)*3` instead of matching `train_y == c` class labels. Correcting class label matching properly pairs test queries with their true class centroids ($\text{mean}(c_q) = 0.9747$, $\text{mean}(x_q) = 0.9699$).
   - *Defect Documentation (ITEM 4)*:
     - *Location*: `run_off_support_density_test.py` lines 81 & 221, `run_graded_ceiling_reanalysis.py` lines 81 & 221.
     - *Defect Code*: `samples = X[i*3:(i+1)*3]` assuming sequential class ordering.
     - *Fix Code*: `mask_c = (train_y == c)` grouping by class label tensor. Zero unindexed slicing patterns remain across all scripts.
   - *Baseline Floor Resolution (ITEM 5)*:
     - **72.50% (290/400)** is DECLARED CANONICAL throughout the paper for raw unadapted 1-NN retrieval using class centroids computed directly from all available training samples per class ($cen_c = \text{mean}(train\_x[train\_y==c])$).
   - *Fact Map Audit & Cluster Size Resolution (ITEM 1)*:
     - Across the 400 test queries in this evaluation split (blocks 0..9 with confusable assignment), test queries span **34 distinct facts** with a queries-per-fact distribution of: **min = 4, median = 12.0, max = 24**. All cluster-robust standard errors and bootstraps are refitted on **34 distinct fact clusters**.
   - *Per-Fact Failure Tables (ITEM 2)*:
     - **48-Failure Outcome (300 Train Samples 1-NN)**: 12 distinct facts contain failures (Fact 1: 4/16, Fact 4: 4/16, Fact 12: 6/12, Fact 13: 6/12, Fact 15: 6/12, Fact 18: 8/16, Fact 20: 3/12, Fact 25: 3/12, Fact 28: 1/4, Fact 30: 1/4, Fact 31: 3/12, Fact 32: 3/12; Sum $= 48/400$).
     - **41-Failure Outcome (100 Centroids 1-NN)**: 11 distinct facts contain failures (Fact 1: 4/16, Fact 4: 4/16, Fact 12: 6/12, Fact 13: 6/12, Fact 15: 6/12, Fact 18: 4/16, Fact 20: 3/12, Fact 25: 3/12, Fact 28: 1/4, Fact 30: 1/4, Fact 31: 3/12; Sum $= 41/400$).
   - *Query-Level vs. Fact-Level Predictor Analysis (ITEM C)*:
     - **Intraclass Correlation Coefficients (ICC(1,1))**: $x_q$ $\text{ICC} = \mathbf{0.0488}$, $k5_q$ $\text{ICC} = \mathbf{0.0204}$, $d_q$ $\text{ICC} = \mathbf{0.0680}$. Low ICCs ($< 0.07$) prove predictors vary query-by-query, confirming they are **QUERY-LEVEL PREDICTORS**.
     - **Leave-One-FACT-Out Cross-Validated (LOFO-CV) AUCs**: M1 ($Y \sim x_q$) LOFO-CV $\text{AUC} = \mathbf{0.7498}$ (vs $0.7994$ standard AUC); N6 ($Y \sim k5_q$) LOFO-CV $\text{AUC} = \mathbf{0.6929}$ (vs $0.7472$ standard AUC). High LOFO-CV AUCs confirm strong out-of-fact generalization.
   - *Side-by-Side Outcome Comparison (ITEM D)*:

| Predictor Name | 48-Fail Outcome (300 Train Samples 1-NN) | | | 41-Fail Outcome (100 Centroids 1-NN) | | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| | McFadden $R^2$ | ROC AUC | Clustered $p$-val | McFadden $R^2$ | ROC AUC | Clustered $p$-val |
| **$x_q$ (Nearest Incorrect)** | $0.1055$ | $0.7994$ | $4.68 \times 10^{-4}$ | $0.1204$ | $0.8217$ | $2.14 \times 10^{-4}$ |
| **$\text{mean}_q$ (Global Offset)** | $0.0355$ | $0.6335$ | $0.0162$ | $0.0178$ | $0.5837$ | $0.0536$ |
| **$\text{gap}_q$ (Local Peakedness)** | $0.0899$ | $0.7064$ | $3.15 \times 10^{-5}$ | $0.0670$ | $0.6808$ | $1.25 \times 10^{-4}$ |
| **$d_q$ (Support Proximity)** | **0.1739** | **0.8242** | **$6.14 \times 10^{-4}$** | **0.1898** | **0.8385** | **$9.61 \times 10^{-4}$** |
| **$k5_q$ (Top-5 Density)** | $0.0821$ | $0.7472$ | $2.89 \times 10^{-4}$ | $0.1039$ | $0.7795$ | $3.17 \times 10^{-5}$ |

   - *Canonical Definition of $d_q$ & Synthesis (ITEM 6)*:
     - **Canonical Definition**: $d_q = \max_{r \in \text{BaseTrain}} \cos(q, r)$ where $\text{BaseTrain}$ is the set of **150 raw training embeddings** (3 samples per fact $\times$ 50 base-phase facts).
     - **Collinearity & VIF**: Pearson $r(x_q, d_q) = +0.7068$, yielding Variance Inflation Factor $\text{VIF} = \mathbf{1.9983} \approx 2.00$ for Model N5 ($x_q + d_q$).
     - **Synthesis**: $d_q$ (Support Proximity) achieves the **highest McFadden $R^2$ ($0.1739 / 0.1898$)** and **highest ROC AUC ($0.8242 / 0.8385$)** across BOTH outcomes. Support proximity to base-trained reference vectors is the **single strongest predictor** of post-adaptation retrieval performance, driving the $+14\text{ pp}$ generic metric transfer result. Documented as an **OPEN FINDING**.
4. **Fixed Evaluation Set Base-Size Curve (Evaluated on Fixed Blocks 5–9, 50 Facts)**:

| Base Phase Size | Base Blocks Trained | Fixed Evaluation Set Accuracy (Blocks 5–9, 50 Facts) | Block-Selection Std Across Seeds | Difference vs 70.50% Frozen Floor |
|:---|:---:|:---:|:---:|:---:|
| **Frozen Floor** | 0 blocks (0 facts) | **70.50%** | $\pm 0.00\%$ | Baseline Floor |
| **10 facts** | 1 block | **60.00%** | $\pm 4.07\%$ | **-10.50 pp (BELOW FLOOR)** |
| **20 facts** | 2 blocks | **67.20%** | $\pm 6.87\%$ | **-3.30 pp (BELOW FLOOR)** |
| **30 facts** | 3 blocks | **79.00%** | $\pm 5.15\%$ | **+8.50 pp (ABOVE FLOOR)** |
| **40 facts** | 4 blocks | **84.70%** | $\pm 0.87\%$ | **+14.20 pp (ABOVE FLOOR)** |
| **50 facts** | 5 blocks | **84.50%** | $\pm 0.00\%$ | **+14.00 pp (ABOVE FLOOR)** |

   - *Correction*: Adaptation degrades unseen-fact retrieval below the 70.50% floor at $B=1$ (60.00%) and $B=2$ (67.20%); crosses the floor between $B=2$ and $B=3$ (79.00%); and saturates by $B=4$ (84.70%) with no further gain at $B=5$ (84.50%). The progression is smooth and saturating, not a phase transition. The $B=1..4$ error bars reflect **block-selection variance** (since base blocks were drawn randomly per seed); adapter training for a fixed set of blocks is **deterministic**.
5. **Seed Wiring, Step-9 Max Weight Diff & Axis D Evidence Status ($n=1$)**:
   - At Step 9 in the 50-run benchmark (where shuffle order varies per seed), $\max_{i, j} |W_{\text{seed101}}[i, j] - W_{\text{seed102}}[i, j]| = \mathbf{2.5985}$, proving sequence shuffling creates distinct weight matrices across runs.
   - `BottleneckAdapter` is deterministically initialized from PCA basis (`pca_basis`). When fixed blocks `[0, 1, 2, 3, 4]` are trained without shuffle variation, $\max |W_{\text{seed101}} - W_{\text{seed102}}| = 0.000000$.
   - **Axis D Evidence Status ($n=1$)**: Because the single-pair probe executed without block order shuffle variation, its five seeds were identical deterministic runs. The single-pair probe's $0.00\text{ pp}$ interference finding rests on $n=1$. (Axis D remains dropped).

---

### 10.2 Baseline Framing & Sensitivity Analysis: Step-Matched Joint Upper Bound
- **Baseline Definition & Renaming**: The joint training baseline that trains 30 epochs per added block step-by-step is designated as the **Step-Matched Joint Upper Bound** (formerly termed incremental joint). It matches the exact step-by-step information availability and epoch budget of the sequential benchmark.
- **Unconstrained Asymptotic Ceiling (True Joint)**: Single-pass joint training on all 100 classes for 300 epochs yields an asymptotic ceiling $A_T = \mathbf{97.23\%}$ (Selection) / $\mathbf{97.15\%}$ (Fresh). Its $LA$ ($96.78\%$) is evaluated on a model that has already been trained on all 100 classes for 300 epochs, making it an unconstrained ceiling rather than a step-matched baseline.
- **Sensitivity Analysis of Decomposed Gaps**:
  - *Against Step-Matched Joint Upper Bound (Primary Baseline)*: Retention Gap Available $= \mathbf{+59.90\text{ pp}}$ ($+34.76 - [-25.14]$), Acquisition Gap Available $= \mathbf{+6.77\text{ pp}}$ ($62.46 - 55.69$). Ratio $\approx \mathbf{9:1}$ (Retention dominates).
  - *Against Unconstrained Asymptotic Ceiling (True Joint)*: Retention Gap Available $= \mathbf{+25.59\text{ pp}}$ ($+0.45 - [-25.14]$), Acquisition Gap Available $= \mathbf{+41.08\text{ pp}}$ ($96.78 - 55.70$). Ratio $\approx \mathbf{1:1.6}$ (Acquisition dominates).
  - *Justification*: The Step-Matched Joint Upper Bound is the correct baseline for continual learning gap decomposition because it enforces step-by-step information availability and equal epoch budgets per step.

---

### 10.3 Full-Rank ($r=960$) Results — Selection Seeds 101..105 (50 Runs per Cell)

### 10.2 Full-Rank ($r=960$) Results — Selection Seeds 101..105 (50 Runs per Cell)

| Condition | $A_T$ Mean ± Std (Min..Max) | $LA$ | BWT | $\Delta A_T$ vs Naive (95% CI) | $\Delta BWT$ vs Naive (95% CI) | Ret. Gap (True / Incr) | Acq. Gap (True / Incr) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline** | 30.56% ± 7.93% (19.75..46.00%) | 55.69% | -25.14% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | +0.0% / +0.0% | +0.0% / +0.0% |
| **Offline Upper Bound**| 97.23% ± 0.99% (95.00..99.50%) | 96.78% | +0.45% | +66.67% [+64.47%, +68.76%] | +25.59% [+23.15%, +28.00%] | +100.0% / +100.0% | +100.0% / +100.0% |
| **FREEZE-AFTER-BASE**| 27.40% ± 3.68% (19.75..38.00%) | 27.40% | +0.00% | -3.16% [-5.41%, -1.07%] | +25.14% [+22.67%, +27.45%] | +98.2% / +42.0% | -68.9% / -417.9% |
| **OGP_k2** | 38.45% ± 7.49% (26.75..50.50%) | 55.68% | -17.24% | +7.89% [+5.97%, +9.94%] | +7.90% [+5.88%, +10.02%] | +30.9% / +13.2% | -0.0% / -0.1% |
| **OGP_k4** | 43.77% ± 5.46% (34.00..53.50%) | 55.16% | -11.39% | +13.22% [+10.83%, +15.60%] | +13.75% [+11.43%, +16.00%] | +53.7% / +23.0% | -1.3% / -7.9% |
| **OGP_k8 (Optimum)**| **44.60% ± 4.62% (35.50..51.50%)** | **53.82%** | **-9.22%** | **+14.05% [+11.31%, +16.69%]** | **+15.92% [+13.27%, +18.39%]** | **+62.2% / +26.6%** | **-4.6% / -27.6%** |
| **OGP_k12** | 42.71% ± 5.24% (29.00..51.75%) | 53.38% | -10.67% | +12.16% [+9.65%, +14.51%] | +14.47% [+12.01%, +16.76%] | +56.5% / +24.2% | -5.6% / -34.2% |
| **OGP_k16** | 38.65% ± 5.59% (29.25..50.00%) | 52.32% | -13.67% | +8.09% [+5.25%, +10.89%] | +11.47% [+8.95%, +13.90%] | +44.8% / +19.1% | -8.2% / -49.9% |
| **OGP_k24** | 29.52% ± 6.91% (20.00..46.00%) | 48.86% | -19.34% | -1.04% [-4.09%, +1.96%] | +5.80% [+3.12%, +8.41%] | +22.7% / +9.7% | -16.6% / -100.9% |
| **OGP_k32** | 23.05% ± 7.36% (14.00..39.25%) | 45.94% | -22.90% | -7.51% [-10.30%, -4.71%] | +2.24% [-0.04%, +4.47%] | +8.8% / +3.7% | -23.7% / -144.0% |
| **RANDOM-k (all k)** | 30.22%..30.62% (19.25..46.00%) | ~55.4% | ~-25.0% | ~0.00% [CIs include 0] | ~0.10% [CIs include 0] | ~+0.4% / +0.2% | ~-0.5% / -3.0% |
| **BOTTOM-k (all k)** | 30.53%..30.56% (19.75..46.00%) | ~55.7% | ~-25.1% | ~0.00% [CIs include 0] | ~0.00% [CIs include 0] | 0.0% / 0.0% | 0.0% / 0.0% |

#### Full-Rank ($r=960$) Results — Fresh Replication Seeds 211..215 (50 Runs per Cell):

| Condition | $A_T$ Mean ± Std (Min..Max) | $LA$ | BWT | $\Delta A_T$ vs Naive (95% CI) | $\Delta BWT$ vs Naive (95% CI) | Ret. Gap (True / Incr) | Acq. Gap (True / Incr) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline** | 27.78% ± 7.58% (16.50..41.25%) | 53.98% | -26.20% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | +0.0% / +0.0% | +0.0% / +0.0% |
| **Offline Upper Bound**| 97.15% ± 1.58% (93.50..99.75%) | 96.58% | +0.57% | +69.37% [+67.42%, +71.29%] | +26.77% [+24.30%, +29.10%] | +100.0% / +100.0% | +100.0% / +100.0% |
| **FREEZE-AFTER-BASE**| 22.92% ± 3.36% (16.00..30.00%) | 22.92% | +0.00% | -4.86% [-7.29%, -2.42%] | +26.20% [+24.39%, +27.99%] | +97.9% / +41.0% | -72.9% / -569.9% |
| **OGP_k2** | 39.82% ± 7.40% (26.75..53.50%) | 53.37% | -13.55% | +12.05% [+9.41%, +14.71%] | +12.66% [+10.06%, +15.30%] | +47.3% / +19.8% | -1.4% / -11.2% |
| **OGP_k4** | 41.51% ± 7.25% (28.75..53.00%) | 53.47% | -11.96% | +13.73% [+11.03%, +16.49%] | +14.24% [+11.81%, +16.71%] | +53.2% / +22.3% | -1.2% / -9.4% |
| **OGP_k8 (Optimum)**| **42.83% ± 6.22% (31.50..56.75%)** | **53.05%** | **-10.22%** | **+15.06% [+12.59%, +17.60%]** | **+15.99% [+13.77%, +18.25%]** | **+59.7% / +25.0%** | **-2.2% / -17.1%** |
| **OGP_k12** | 40.57% ± 6.90% (26.25..51.50%) | 51.73% | -11.16% | +12.80% [+10.54%, +15.06%] | +15.05% [+12.98%, +17.13%] | +56.2% / +23.5% | -5.3% / -41.3% |
| **OGP_k16** | 37.10% ± 7.75% (18.75..48.50%) | 50.90% | -13.80% | +9.33% [+6.67%, +11.99%] | +12.40% [+9.88%, +14.90%] | +46.3% / +19.4% | -7.2% / -56.4% |
| **OGP_k24** | 29.58% ± 6.54% (19.00..42.50%) | 48.22% | -18.65% | +1.80% [-0.58%, +4.29%] | +7.56% [+5.48%, +9.74%] | +28.2% / +11.8% | -13.5% / -105.6% |
| **OGP_k32** | 28.00% ± 6.79% (18.00..41.25%) | 46.68% | -18.68% | +0.23% [-2.16%, +2.68%] | +7.53% [+5.57%, +9.58%] | +28.1% / +11.8% | -17.1% / -133.9% |
| **RANDOM-k (all k)** | 26.47%..27.69% (13.00..42.25%) | ~53.4% | ~-26.3% | ~0.00% [CIs include 0] | ~0.00% [CIs include 0] | ~0.0% / +0.0% | ~-0.5% / -2.0% |
| **BOTTOM-k (all k)** | 27.62%..27.78% (13.00..41.25%) | ~53.9% | ~-26.2% | ~0.00% [CIs include 0] | ~0.00% [CIs include 0] | 0.0% / 0.0% | 0.0% / 0.0% |

> **Cross-Set Disagreement Flag ($k=32$)**: At $k=32$, Selection seeds produce significant degradation ($\Delta A_T = -7.51\% [-10.30\%, -4.71\%]$), whereas Fresh seeds produce a true null ($\Delta A_T = +0.23\% [-2.16\%, +2.68\%]$).

---

### 10.3 Refutation of Intrinsic-Rank Scaling at Bottleneck Capacity ($r=32$)

> **REFUTATION**: The prediction that optimal $k$ scales with adapter rank within a task is **NOT supported**. At $r=32$, no $k \in \{2, 4, 8\}$ produces a significant gain; OGP requires $r \gg k$ and fails entirely at bottleneck capacity. This is a limitation of the mechanism, not a scaling law.

#### Bottleneck ($r=32$) Results — Selection Seeds 101..105 (50 Runs per Cell):

| Condition | $A_T$ Mean ± Std (Min..Max) | $LA$ | BWT | $\Delta A_T$ vs Naive (95% CI) | $\Delta BWT$ vs Naive (95% CI) | Ret. Gap Closed | Acq. Gap Closed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline** | 21.00% ± 5.62% (13.50..34.00%) | 40.93% | -19.93% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | +0.0% | +0.0% |
| **Offline Upper Bound**| 91.47% ± 1.56% (88.75..95.25%) | 91.20% | +0.27% | +70.47% [+68.85%, +71.95%] | +20.20% [+18.15%, +22.10%] | +100.0% | +100.0% |
| **FREEZE-AFTER-BASE**| 16.24% ± 3.74% (10.00..25.50%) | 16.24% | +0.00% | -4.77% [-6.56%, -3.09%] | +19.93% [+17.84%, +21.87%] | +98.7% | -49.1% |
| **OGP_k2** | 20.00% ± 3.51% (13.50..30.25%) | 40.00% | -20.00% | -1.00% [-1.91%, -0.20%] | -0.07% [-0.98%, +0.74%] | -0.3% | -1.8% |
| **OGP_k4** | 21.04% ± 3.83% (14.25..29.50%) | 40.38% | -19.34% | +0.04% [-1.01%, +0.97%] | +0.59% [-0.49%, +1.55%] | +2.9% | -1.1% |
| **OGP_k8** | 20.79% ± 3.55% (14.25..29.25%) | 39.95% | -19.16% | -0.21% [-1.53%, +0.91%] | +0.77% [-0.50%, +1.88%] | +3.8% | -1.9% |
| **OGP_k12** | 18.84% ± 3.82% (11.75..27.50%) | 39.19% | -20.36% | -2.16% [-3.82%, -0.73%] | -0.42% [-1.91%, +0.84%] | -2.1% | -3.5% |
| **OGP_k16** | 17.15% ± 3.85% (11.00..24.75%) | 38.48% | -21.33% | -3.85% [-5.67%, -2.33%] | -1.40% [-3.05%, -0.03%] | -6.9% | -4.9% |
| **OGP_k24** | 15.63% ± 3.42% (11.00..23.00%) | 37.33% | -21.70% | -5.37% [-7.13%, -3.92%] | -1.77% [-3.36%, -0.43%] | -8.8% | -7.2% |
| **OGP_k32** | 14.57% ± 2.96% (11.00..21.00%) | 36.80% | -22.24% | -6.44% [-8.10%, -5.06%] | -2.31% [-3.84%, -1.02%] | -11.4% | -8.2% |
| **RANDOM-k (all k)** | 20.33%..20.89% (13.50..34.00%) | ~40.6% | ~-20.1% | ~-0.30% [CIs include 0] | ~-0.15% [CIs include 0] | -0.7% | -0.7% |
| **BOTTOM-k (all k)** | 20.97%..21.03% (13.50..34.00%) | ~40.9% | ~-19.9% | ~0.00% [CIs include 0] | ~0.00% [CIs include 0] | 0.0% | 0.0% |

#### Bottleneck ($r=32$) Results — Fresh Replication Seeds 211..215 (50 Runs per Cell):

| Condition | $A_T$ Mean ± Std (Min..Max) | $LA$ | BWT | $\Delta A_T$ vs Naive (95% CI) | $\Delta BWT$ vs Naive (95% CI) | Ret. Gap Closed | Acq. Gap Closed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline** | 19.60% ± 6.38% (10.00..33.75%) | 39.77% | -20.17% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | +0.0% | +0.0% |
| **Offline Upper Bound**| 89.47% ± 2.90% (84.25..96.25%) | 89.15% | +0.32% | +69.87% [+68.39%, +71.34%] | +20.49% [+18.40%, +22.40%] | +100.0% | +100.0% |
| **FREEZE-AFTER-BASE**| 15.66% ± 2.87% (10.25..24.75%) | 15.66% | +0.00% | -3.94% [-5.97%, -1.98%] | +20.17% [+18.69%, +21.66%] | +98.4% | -48.8% |
| **OGP_k2** | 18.88% ± 5.19% (10.00..32.00%) | 39.18% | -20.30% | -0.72% [-1.61%, +0.11%] | -0.13% [-0.93%, +0.60%] | -0.6% | -1.2% |
| **OGP_k4** | 19.48% ± 5.26% (11.50..32.00%) | 39.85% | -20.37% | -0.12% [-1.06%, +0.76%] | -0.20% [-1.06%, +0.62%] | -1.0% | +0.2% |
| **OGP_k8** | 19.76% ± 4.92% (11.50..32.00%) | 39.71% | -19.95% | +0.16% [-0.80%, +1.08%] | +0.22% [-0.65%, +1.06%] | +1.1% | -0.1% |
| **OGP_k12** | 17.57% ± 5.10% (10.00..30.25%) | 38.77% | -21.20% | -2.03% [-3.35%, -0.80%] | -1.03% [-2.18%, +0.05%] | -5.0% | -2.0% |
| **OGP_k16** | 16.96% ± 4.01% (10.00..27.75%) | 38.31% | -21.35% | -2.64% [-3.84%, -1.48%] | -1.18% [-2.16%, -0.23%] | -5.8% | -3.0% |
| **OGP_k24** | 15.00% ± 3.66% (10.00..22.75%) | 37.32% | -22.33% | -4.61% [-6.22%, -3.04%] | -2.16% [-3.41%, -0.92%] | -10.5% | -5.0% |
| **OGP_k32** | 14.55% ± 3.28% (10.00..23.00%) | 37.09% | -22.54% | -5.06% [-6.66%, -3.52%] | -2.37% [-3.63%, -1.12%] | -11.6% | -5.4% |
| **RANDOM-k (all k)** | 19.29%..19.50% (10.00..34.50%) | ~39.6% | ~-20.1% | ~-0.20% [CIs include 0] | ~0.00% [CIs include 0] | ~0.0% | ~-0.3% |
| **BOTTOM-k (all k)** | 19.56%..19.60% (10.00..33.75%) | ~39.8% | ~-20.2% | ~0.00% [CIs include 0] | ~0.00% [CIs include 0] | 0.0% | 0.0% |

---

### 10.4 Reframed Theoretical Conclusions

1. **Decomposed Gap Analysis & Genuine Stability-Plasticity Tradeoff**:
   - At $k=8$ ($r=960$), OGP closes **62.2%** (Selection) / **59.7%** (Fresh) of the available true retention gap ($\Delta BWT = +15.92\text{ pp} / +15.99\text{ pp}$ out of $+25.59 / +26.77\text{ pp}$), while closing **-4.6%** / **-2.2%** of the acquisition gap ($LA$ falls $1.87\text{ pp} / 0.93\text{ pp}$ from naive).
   - Retention share exceeds $100\%$ ($\Delta BWT / \Delta A_T = 113.4\%$). This represents the **first genuine stability-plasticity tradeoff measured in this project**, appearing only once memory became parametric.
2. **Absolute Standing & Deficit Identification**: $44.60\%$ $A_T$ vs $97.23\%$ offline upper bound closes **21.1%** of the total gap. Naive acquisition failure ($LA = 55.69\%$ vs offline $LA = 96.78\%$) remains the primary deficit.
3. **Task Subspace & Degradation at $k \ge 24$ (Conjecture)**:
   - *Note*: The $\le 64$-d task subspace figure is a conjecture carried over from the retrieval regime's PCA capacity curve, not a direct measurement on the 100-class classification head. Under this conjecture, protecting $k \ge 24$ principal directions projects out past input directions from the low-dimensional task subspace, starving new block acquisition.

---

*All metrics computed with populated-row guard on R matrix (base phase row 4 evaluated after joint training). Decomposition uses exact BWT = A_T - LA identity. All CIs: 10,000-sample paired bootstrap. Repository: github.com/swarajladke/Neural-Networks, HEAD commit 02c88eb.*
bootstrap. Repository: github.com/swarajladke/Neural-Networks, HEAD commit 9530645.*



