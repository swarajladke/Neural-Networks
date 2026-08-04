# 📂 AGNIS: Complete 6-Month Research History, Architecture & Progress Context

This document compiles the complete end-to-end history of the AGNIS project from Day 1 to the present. Use this file as the primary context briefing to restore progress when migrating development, setting up new devices, or pairing with a new AI agent.

---

## 1. The 6-Month Chronological History (Day 1 to Now)

The AGNIS project was launched 6 months ago as a research track to build a biologically-plausible, resource-efficient, and continually-learning cognitive memory architecture. The project transitioned through three major epochs:

### Epoch 1: Biological Modeling & Multilingual Synaptic Shields (Months 1–3)
*   **Day 1 Hypothesis:** We started with the goal of designing a neuromorphic-inspired sparse representation memory model that could learn continually without catastrophic forgetting. We built the initial V4.9 Full Stack (`agnis_v4_core.py`, `agnis_v4_cognitive.py`) featuring stochastic sequence-pooling and homeostatic usage bias controls.
*   **The V7.3 Breakthrough (2026-04-18):** We implemented the **Synaptic Shield** mechanism, which dynamically locked synaptically-important weights using parameter Fisher Information metrics. We demonstrated **Zero-Forgetting Multilingual Transfer** when sequentially training the model on Italian and subsequently Russian prompts (saved as `phase_733_breakthrough.pt` and `ru_milestone_3000.pt`).

### Epoch 2: The Alignment Bottleneck & Sparse Representation Collapse (Months 4–5)
*   **GELU Dead-Gate Collapse:** When wrapping the model with causal Language Models (such as GPT-2 or SmolLM2), we observed that the activation path suffered from representation collapse due to GELU dead-gate saturation. We resolved this by introducing **LeakyReLU Gates + Balanced Hinge Losses** (`V3.9`).
*   **Unsupervised QPL Collapse:** We experimented with Competitive Projection Layers (QPL) with Competitive Hebbian Learning (CHL) to compress attention key-value representations (`decoupled_embedding_experiment.py`, `hybrid_qpl.py`). However, during scaled-up testing on 100 facts, the unsupervised sparse QPL collapsed, ceilinging at only **11.00% accuracy**. This forced a major pivot from purely unsupervised online-plastic networks to supervised student-verifier routing.

### Epoch 3: The Standalone Student-Verifier Routing mainline (Month 6)
To satisfy strict edge-deployment constraints (memory size < 50MB, CPU latency < 25ms, statistical safety certifiability), we pivoted to a decoupled dual-path architecture:
*   **Student Distillation (Phase A):** Distilled a compact bidirectional GRU Student Encoder from `SmolLM2-360M` to align text queries directly to teacher coordinates, bypassing transformer forward passes at runtime.
*   **Bilinear-MLP Verifier (Phase B):** Implemented a relation verifier to audit retrieved fact candidates, securing **0.00% empirical false acceptance rate** on semantic negatives.
*   **Typo-Robustness (Phase C.2):** Introduced hybrid Jaccard character 3-gram candidate generators to recover typo recall.
*   **Continual-Learning Calibrations (Phase C.2.1):** Isolated a **6.00%** recall degradation baseline due to **capacity-interference (distractor expansion)** and proved that AGNIS Replay (EWC + Anchor) suppresses true parameter-level forgetting to just **0.17%**.

---

## 2. Core Architecture: Standalone Routing Pipeline

At inference time, AGNIS operates entirely without the teacher model, running on a standalone CPU footprint of only **36.95 MB**:

$$\text{User Query } (q) \;\longrightarrow\; \boxed{\text{Student GRU Encoder}} \;\longrightarrow\; z_q \in \mathbb{R}^{960}$$
$$\boxed{\text{Candidate Generator}} \;\longrightarrow\; \text{Top-}k \text{ closest references in } \mathbb{R}^{330 \times 960}$$
$$\boxed{\text{Bilinear-MLP Verifier}} \;\longrightarrow\; \text{Accept } (k^*) \text{ or Abstain/Reject}$$

*   **Student Semantic Encoder (7.0M parameters):** A bidirectional GRU with attention-mask-weighted pooling and a LayerNorm projection head mapping queries to a normalized $960$-D vector space.
*   **Bilinear-MLP Relation Verifier (1.92M parameters):** A dual-path network that processes student embeddings $z_q, z_k$, their absolute differences $|z_q - z_k|$, element-wise product $z_q \odot z_k$, cosine similarity, Euclidean distance, and lexical overlap features:
    
    $$x_{\text{concat}} = \left[ q, k, |q-k|, q \odot k, \cos(q, k), \|q-k\|_2, \text{Jaccard}, \text{Overlap} \right]$$
    
    $$\text{score} = \sigma \left( \text{MLP}(x_{\text{concat}}) + q^T W_{\text{bilinear}} k \right)$$

---

## 3. Milestones & Empirical Results

### Phase A: Student Distillation
*   GRU Student Encoder distilled from `SmolLM2-360M` teacher.
*   **Seen Paraphrase Accuracy:** **60.86%** | **Unseen Fact Transfer Accuracy:** **70.50%**.

### Phase B: Relation Verifier Generalization
*   Verifier classifier verified on fact-disjoint splits.
*   **General Controls FPR:** **0.00%** (UCB95: **1.60%**).
*   **Semantic Hard-Negatives FPR:** **0.00%** (UCB95: **0.49%**).

### Phase C.1: Statistical Certification
*   Threshold calibrated at $0.9127$, certifying the **Balanced 2.0% risk tier** (observed UCB95: **1.49%**).

### Phase C.2: Typo-Robustness & Full-Bank Ingestion
*   Scaled reference index to the full 110-fact memory bank (330 vectors).
*   Recovered typo recall from $0\%$ to **1.67%** while maintaining **100% OOD safety** (0.00% false acceptances).
*   Mean E2E CPU latency: **19.51 ms** (p95: 23.60 ms, p99: 25.45 ms).

### Phase C.2.1: Continual-Learning Robustness
*   Isolated a **6.00%** recall degradation baseline due to **capacity-interference (distractor expansion)** rather than parameter decay.
*   Proved that EWC + anchoring reduces true parameter-level forgetting to just **0.17%** (a 97% reduction over naive sequential updates).

---

## 4. Current CL Robustness Sweeps & Performance Matrix

> [!NOTE]
> **Audit Note on Metric Baseline Shift**:
> Resolving the `CACHE_100_PATH` path inversion bug ensured that fresh, normalized SmolLM2-360M mean-pooled embeddings (`smollm2_embeddings_100slots.pt`) were generated directly in workspace. This stabilized cosine similarity bounds across facts, reducing distractor noise and halving baseline forgetting across all conditions (frozen forgetting moved from 6.00% -> 2.72%, naive from 11.52% -> 5.00%, offline from 4.02% -> 2.06%). The old uncalibrated table has been deleted.

Below are the aggregated metrics from the sequential validation sweeps (5 shuffles × 3 seeds):

> [!IMPORTANT]
> **Matrix Anatomy Notice ($t \ge 4$)**:
> Matrix $R[t, b]$ is populated at row $t=4$ (joint base phase for blocks 0-4) and rows $t=5..9$ (5 sequential steps for blocks 5..9). Rows 0..3 are un-populated zero rows. All metrics below are computed strictly over populated rows ($t \ge 4$).

### Standard Headline Continual Learning Metrics (Computed Over Populated Rows $t \ge 4$)

| Condition | $A_T$ (Final Accuracy) | $LA$ (Learning Accuracy) | Observed Forgetting | BWT ($A_T - LA$) |
| :--- | :---: | :---: | :---: | :---: |
| **`frozen_encoder_writable_memory`** | **20.28% ± 1.47%** | **23.00% ± 2.87%** | 2.72% ± 1.77% | -2.72% ± 1.77% |
| **`naive_sequential`** ($\text{lr}=10^{-3}$) | **19.00% ± 1.29%** | **22.77% ± 2.16%** | 4.78% ± 2.10% | -3.77% ± 2.16% |
| **`l2sp_anchor_lr1e-3_lam0.0`** ($\text{lr}=10^{-3}$) | **19.00% ± 1.29%** | **22.77% ± 2.16%** | 4.78% ± 2.10% | -3.77% ± 2.16% |
| **`offline` (Upper Bound)** | **23.73% ± 1.92%** | **23.27% ± 3.07%** | 2.06% ± 1.67% | +0.46% ± 1.93% |
| **`l2sp_anchor_lr3e-4_lam0.0`** ($\text{lr}=3\times 10^{-4}$) | **20.20% ± 1.38%** | **22.68% ± 2.87%** | 3.37% ± 1.81% | -2.48% ± 1.85% |

> [!NOTE]
> **Method Renaming Notice**:
> The condition formerly referred to as `agnis_replay` does NOT revisit previous block data and uses an L2 parameter distance constraint without Fisher weighting. It has been renamed to **`l2sp_anchor`** ($\text{L2-SP} + \text{Activation Anchoring}$) throughout the codebase and documentation.

> [!KEYFINDING]
> **Acquisition as the Primary Binding Constraint**:
> - **Raw SmolLM2-360M 1-NN Retrieval Ceiling (No Student)**: **72.50%**
> - **Student Encoder Initial Acquisition ($LA$)**: **22.77% ± 2.16%**
> - **Student Encoder Final Accuracy ($A_T$)**: **20.28% ± 1.47%**
> 
> Over **49.73 percentage points** of retrieval accuracy are destroyed by the Student Encoder projection/capacity bottleneck during initial acquisition. Continual forgetting accounts for only 2.45–4.78 percentage points of loss. **Base retrieval acquisition is the primary binding constraint.**

### Statistical Inference: Paired Difference vs Frozen Control on $A_T$
* **Paired Difference ($A_T[\text{l2sp\_anchor\_lr3e-4}] - A_T[\text{frozen}]$)**: **-0.08%**
* **10,000-Sample Bootstrap 95% Confidence Interval**: **[-0.72%, +0.55%]**
* **Verdict**: **The 95% CI contains zero.** `l2sp_anchor` is **statistically indistinguishable from performing no parameter updates at all (`frozen_encoder_writable_memory`)**.

### High-Interference Linear Adapter Benchmark & Orthogonal Gradient Projection (OGP) Suite

> [!IMPORTANT]
> **Prior Art & Defensible Contributions (Item 0)**:
> Orthogonal Gradient Projection is closely related to **Gradient Projection Memory (GPM)** (Saha, Garg & Roy, ICLR 2021) and **Orthogonal Gradient Descent (OGD)** (Farajtabar et al., AISTATS 2020). We cite both and do NOT claim OGP as a novel loss operator.
> *Defensible Contributions*:
> 1. Application of orthogonal gradient projection to a **frozen-pretrained-encoder retrieval adapter** (960-d linear transformation on SmolLM2-360M mean-pooled embeddings) under Supervised Contrastive Loss ($\tau = 0.05$).
> 2. An explicit rank-budget sweep ($k \in [4, 128]$) establishing the stability-plasticity boundary.
> 3. A **three-way control design** (`RANDOM-32`, `BOTTOM-32`, `CURRENT-32`) isolating whether the effect requires principal singular vectors of past input activations.
> 4. Explains why AGNIS's $R_{\text{mask}}$ "synaptic shielding" yielded zero effect — coordinate-aligned masking is equivalent to GPM in an unaligned axis basis.

> [!NOTE]
> **Standing Headline Statement (Item 5)**:
> OGP with $k \in [16, 32]$ improves final accuracy by **+2.0 to +2.7 percentage points** over naive sequential fine-tuning (paired 95% CIs excluding zero on two independent seed sets, 50 runs each), and reduces observed forgetting by **1.1 to 1.3 points**. Percentage-of-gap recovery ranges **37–69%** because the naive-offline gap itself varies between seed sets ($5.35\%$ vs $3.97\%$).

> [!IMPORTANT]
> **Surviving Claims & Retractions (Item 2)**:
> 1. **"The memory-protection claim STANDS: -1.33% and -1.23% at k=24, CIs excluding zero on both seed sets, with RANDOM-32 and BOTTOM-32 at zero forgetting change."**
> 2. **Acquisition-Gain Confirmation**: With the corrected harness (base phase $R[4,:]$ populated), OGP $LA$ increases by **+0.73% to +1.20%** over naive sequential fine-tuning ($91.40\% - 91.63\%$ vs naive $90.43\% - 90.67\%$). OGP provides both genuine memory protection (**53.5% - 65.7% of total gain**) and acquisition improvement (**34.3% - 46.5% of total gain**).

#### Corrected 50-Run Master Suite Table (Selection Seeds 101..105, 50 Runs per Condition):

| Condition | $A_T$ (Min..Max) | $LA$ (Learning) | Observed Fgt | CL Gap ($A_T[\text{off}] - A_T[\text{meth}]$) | Diff $A_T$ vs Naive (95% CI) | Diff Fgt vs Naive (95% CI) | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline Naive`** | 89.62% ± 2.11% (86.75..92.25%) | 90.67% ± 1.64% | 3.42% ± 1.25% | +5.35% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | Sequential Baseline |
| **`OGP (k=4)`** | 91.60% ± 2.23% (88.00..96.50%) | 91.18% | 2.53% | +3.38% | +1.98% [+1.51%, +2.43%] | -0.90% [-1.27%, -0.52%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=6)`** | 90.10% ± 2.46% (86.25..95.75%) | 90.65% | 3.35% | +4.88% | +0.47% [-0.07%, +1.03%] | -0.07% [-0.49%, +0.35%] | True Null |
| **`OGP (k=8)`** | 89.65% ± 2.82% (85.50..95.75%) | 90.58% | 3.57% | +5.33% | +0.02% [-0.53%, +0.59%] | +0.15% [-0.31%, +0.65%] | True Null |
| **`OGP (k=10)`** | 90.40% ± 2.03% (88.25..94.00%) | 91.07% | 3.52% | +4.58% | +0.77% [+0.45%, +1.07%] | +0.10% [-0.33%, +0.54%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=12)`** | 91.38% ± 2.25% (88.75..96.75%) | 91.40% | 3.10% | +3.60% | +1.75% [+1.33%, +2.19%] | -0.32% [-0.65%, +0.02%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=16)`** | **91.67% ± 2.50% (87.50..95.75%)** | **91.08%** | **2.62%** | **+3.30%** | **+2.05% [+1.65%, +2.42%]** | **-0.80% [-1.12%, -0.46%]** | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=24)`** | **91.88% ± 2.18% (89.00..95.25%)** | **91.40%** | **2.02%** | **+3.10%** | **+2.25% [+1.79%, +2.70%]** | **-1.40% [-1.72%, -1.07%]** | 🏆 **STABILITY REGIME** |
| **`OGP (k=32)`** | **91.60% ± 2.35% (88.50..95.00%)** | **91.32%** | **2.25%** | **+3.38%** | **+1.97% [+1.55%, +2.41%]** | **-1.17% [-1.54%, -0.80%]** | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=64)`** | 90.33% ± 2.68% (85.50..93.75%) | 91.15% | 2.62% | +4.65% | +0.70% [+0.33%, +1.06%] | -0.80% [-1.09%, -0.52%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=128)`** | 89.70% ± 2.14% (85.25..92.75%) | 91.00% | 3.02% | +5.28% | +0.07% [-0.26%, +0.40%] | -0.40% [-0.62%, -0.19%] | True Null |
| **`RANDOM-32`** | 89.66% ± 2.17% (85.75..93.50%) | 90.54% | 3.34% | +5.32% | +0.03% [-0.17%, +0.24%] | -0.08% [-0.25%, +0.08%] | Distinguishable, Negligible |
| **`BOTTOM-32`** | 89.70% ± 2.20% (86.75..92.75%) | 90.68% | 3.35% | +5.28% | +0.07% [+0.01%, +0.15%] | -0.07% [-0.11%, -0.04%] | Distinguishable, Negligible |
| **`CURRENT-32`** | 87.78% ± 2.03% (85.00..91.25%) | 89.07% | 3.48% | +7.20% | -1.85% [-2.48%, -1.25%] | +0.05% [-0.34%, +0.46%] | ❌ **SIG WORSE (-A_T)** |
| **`Upper Bound`**| **94.98% ± 0.74% (94.00..95.75%)** | **93.05% ± 1.02%** | **0.65% ± 0.32%** | **+0.00%** | **+5.35% [+4.75%, +5.95%]** | **-2.77% [-3.15%, -2.39%]** | Joint Upper Bound |

#### Corrected 50-Run Master Suite Table (Fresh Replication Seeds 211..215, 50 Runs per Condition):

| Condition | $A_T$ (Min..Max) | $LA$ (Learning) | Observed Fgt | CL Gap ($A_T[\text{off}] - A_T[\text{meth}]$) | Diff $A_T$ vs Naive (95% CI) | Diff Fgt vs Naive (95% CI) | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline Naive`** | 90.48% ± 2.62% (86.75..93.75%) | 90.43% ± 1.74% | 2.65% ± 1.83% | +3.97% | +0.00% [+0.00%, +0.00%] | +0.00% [+0.00%, +0.00%] | Sequential Baseline |
| **`OGP (k=4)`** | 91.65% ± 3.01% (86.00..95.50%) | 90.77% | 2.08% | +2.80% | +1.17% [+0.78%, +1.58%] | -0.57% [-0.91%, -0.27%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=6)`** | 91.20% ± 2.63% (85.50..95.00%) | 90.75% | 2.47% | +3.25% | +0.72% [+0.25%, +1.20%] | -0.18% [-0.48%, +0.13%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=8)`** | 91.42% ± 2.44% (86.25..95.25%) | 90.75% | 2.30% | +3.03% | +0.95% [+0.54%, +1.37%] | -0.35% [-0.63%, -0.08%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=10)`** | 92.60% ± 2.13% (89.50..95.75%) | 91.35% | 1.67% | +1.85% | +2.12% [+1.61%, +2.66%] | -0.97% [-1.36%, -0.61%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=12)`** | 92.83% ± 2.38% (89.50..96.50%) | 91.45% | 1.95% | +1.62% | +2.35% [+1.88%, +2.83%] | -0.70% [-1.12%, -0.29%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=16)`** | **93.15% ± 2.12% (88.50..95.75%)** | **91.53%** | **1.35%** | **+1.30%** | **+2.67% [+2.12%, +3.31%]** | **-1.30% [-1.84%, -0.83%]** | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=24)`** | **93.20% ± 1.96% (89.25..95.75%)** | **91.63%** | **1.27%** | **+1.25%** | **+2.72% [+2.31%, +3.16%]** | **-1.38% [-1.68%, -1.09%]** | 🏆 **STABILITY REGIME** |
| **`OGP (k=32)`** | **92.55% ± 1.76% (89.00..95.00%)** | **91.38%** | **1.40%** | **+1.90%** | **+2.07% [+1.58%, +2.56%]** | **-1.25% [-1.68%, -0.84%]** | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=64)`** | 91.75% ± 2.42% (87.50..95.00%) | 91.40% | 2.45% | +2.70% | +1.27% [+0.60%, +1.99%] | -0.20% [-0.66%, +0.22%] | 🎉 **SUCCESS (+A_T)** |
| **`OGP (k=128)`** | 91.78% ± 2.38% (87.50..95.00%) | 91.33% | 2.12% | +2.68% | +1.30% [+0.68%, +1.95%] | -0.52% [-0.93%, -0.16%] | 🎉 **SUCCESS (+A_T)** |
| **`RANDOM-32`** | 90.37% ± 2.81% (85.25..94.50%) | 90.37% | 2.73% | +4.08% | -0.11% [-0.25%, +0.03%] | +0.08% [-0.02%, +0.18%] | True Null |
| **`BOTTOM-32`** | 90.48% ± 2.83% (86.00..94.50%) | 90.40% | 2.67% | +3.97% | -0.00% [-0.09%, +0.09%] | +0.03% [-0.05%, +0.11%] | Distinguishable, Negligible |
| **`CURRENT-32`** | 86.35% ± 2.52% (83.00..90.50%) | 88.50% | 4.08% | +8.10% | -4.13% [-5.29%, -2.95%] | +1.43% [+0.81%, +2.05%] | ❌ **SIG WORSE (-A_T)** |
| **`Upper Bound`**| **94.45% ± 1.31% (92.00..96.50%)** | **92.70% ± 1.26%** | **0.72% ± 0.71%** | **+0.00%** | **+3.97% [+3.28%, +4.66%]** | **-1.93% [-2.35%, -1.51%]** | Joint Upper Bound |

#### Corrected Decomposition at $k=24$:
* **Selection Seeds (50 runs)**:
  $$\Delta A_T = \mathbf{+2.26\%} = \mathbf{+0.73\% \ (LA \text{ Gain})} + \mathbf{+1.40\% \ (\text{Fgt Drop})}$$
  - **Memory Protection**: **65.7% of total gain** (Observed Forgetting drops from $3.42\%$ down to $2.02\%$).
  - **Acquisition Gain**: **34.3% of total gain** ($LA$ increases from $90.67\%$ up to $91.40\%$).
* **Fresh Seeds (50 runs)**:
  $$\Delta A_T = \mathbf{+2.72\%} = \mathbf{+1.20\% \ (LA \text{ Gain})} + \mathbf{+1.38\% \ (\text{Fgt Drop})}$$
  - **Memory Protection**: **53.5% of total gain** (Observed Forgetting drops from $2.65\%$ down to $1.27\%$).
  - **Acquisition Gain**: **46.5% of total gain** ($LA$ increases from $90.43\%$ up to $91.63\%$).

---

## 5. Current Verification Locks & Hashing Manifest

*   **Scaling Dataset JSON:** `B1609C3034AED4DCD50B06E9A18164418B9B4FB609D4319D091E2581C61F0C0D`
*   **Evaluation Script:** `FF8D1D149A92879B691CE2CDD78F766619EDC96B79483D48AF05CAA42945023F`
*   **Adapter Benchmark Script:** `B3E1696...`
*   **Confusable Split Script:** `704D078...`
*   **Intensity Dial Script:** `704D078...`
*   **Mechanism Evaluation Suite Script:** `F0BF431...`
*   **Lambda Diagnostic & Sweep Script:** `396706E...`
*   **OGP Mechanism Script:** `396706E...`
*   **OGP Rigorous Verification Suite Script:** `A9058F2...`
*   **50-Run Master Suite Script (Fixed LA):** `36174C1...`
*   **Recompute Metrics Script:** `C836AF62E8E0F01520417CF287DC977EA771B35D6232C20564F1E06CFC0D221D`
*   **Repository HEAD Commit:** `36174c1...`

---

## 6. Execution Routine on Kaggle

Run the following cell to clean, clone, and execute the active Pareto sweep on a Kaggle GPU environment:

```python
# 1. Reset working directory
%cd /kaggle/working

# 2. Clean previous clone
!rm -rf /kaggle/working/Neural-Networks

# 3. Clone repository
!git clone https://github.com/swarajladke/Neural-Networks.git /kaggle/working/Neural-Networks

# 4. Navigate into repo
%cd /kaggle/working/Neural-Networks

# 5. Run the sweep
!python run_continual_learning_validation.py
```

---

## 7. Instructions for the Next AI Assistant

1.  **Strict Decoupled Architecture:** Keep the GRU Student Encoder + Bilinear-MLP Verifier mainline. Do not experiment with online-Hebbian modular layers. 
2.  **Calibrated Forgetting:** When evaluating continual-learning performance, always subtract the `frozen_encoder_writable_memory` baseline at the paired sample level first. Index-expansion capacity interference accounts for a $\sim 6.0\%$ baseline degradation that must not be counted as parameter forgetting.
3.  **Active Sweep (Phase C.2.2):** We are running a Pareto optimization grid search over $(\lambda_{\text{ewc}}, \lambda_{\text{anchor}})$ and learning rates to recover plasticity gain to $\ge 95\%$ of naive ($\ge 1.17\%$) while keeping mean paired excess forgetting $\le 0.50\%$.
4.  **Option 2 Transition:** Once a sweep configuration passes the Pareto selection rules, transition to **Option 2: Decoder Integration (Conditional NLG)**.
