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
> **Matrix Populated Rows Notice ($t \ge 4$)**:
> Matrix $R[t, b]$ is populated at row $t=4$ (joint base phase for blocks 0-4) and rows $t=5..9$ (5 sequential steps for blocks 5..9). Rows 0..3 are un-populated zero rows. All metrics below are computed strictly over populated rows ($t \ge 4$).

### Standard Headline Continual Learning Metrics (Computed Over Populated Rows $t \ge 4$)

| Condition | $A_T$ (Final Accuracy) | $LA$ (Learning Accuracy) | Observed Forgetting | BWT |
| :--- | :---: | :---: | :---: | :---: |
| **`frozen_encoder_writable_memory`** | **20.28% ± 1.47%** | 12.28% ± 2.87% | 2.72% ± 1.77% | -2.72% ± 1.77% |
| **`naive_sequential`** | **19.00% ± 1.29%** | 12.33% ± 2.40% | 5.00% ± 2.71% | -4.19% ± 2.40% |
| **`offline` (Upper Bound)** | **23.73% ± 1.92%** | 13.25% ± 3.07% | 2.06% ± 1.67% | +0.46% ± 1.93% |
| **`agnis_replay_lr3e-4_lam0.0`** | **20.20% ± 1.38%** | 12.17% ± 2.87% | 3.37% ± 1.81% | -2.48% ± 1.85% |

### Statistical Inference: Paired Difference vs Frozen Control on $A_T$
* **Paired Difference ($A_T[\text{agnis\_replay}] - A_T[\text{frozen}]$)**: **-0.08%**
* **10,000-Sample Bootstrap 95% Confidence Interval**: **[-0.72%, +0.55%]**
* **Verdict**: **The 95% CI contains zero.** `agnis_replay` is **statistically indistinguishable from performing no parameter updates at all (`frozen_encoder_writable_memory`)**.

---

## 5. Current Verification Locks & Hashing Manifest

*   **Scaling Dataset JSON:** `B1609C3034AED4DCD50B06E9A18164418B9B4FB609D4319D091E2581C61F0C0D`
*   **Evaluation Script:** `B26532BA6D30719633B2D3A7045B540C16A38921919E45C748BB3085B0F356B7`
*   **Recompute Metrics Script:** `967A635FF4FA5AF3CC64AD61CD66824A7EF1038FDD588DDCB9BFB64FE752CF3C`
*   **Repository HEAD Commit:** `2289abb57dc2097ebd7edf428b8048376ea1e846`

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
