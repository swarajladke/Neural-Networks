# 📂 AGNIS: Project Context, Architecture & Migration Briefing

This document serves as the single source of truth for the AGNIS project. Use this file to restore context when migrating development, validation, or pairing with a new AI agent on another machine.

---

## 1. Project Vision & Core Architecture

AGNIS (**A**daptive **G**ated **N**on-linear **I**nference **S**ystem) is a standalone, lightweight edge-routing memory system designed to route raw text queries directly to local fact databases on low-compute edge devices (CPUs).

### Target Deployment Pipeline:
$$\text{User Query } (q) \;\longrightarrow\; \boxed{\text{Student GRU Encoder}} \;\longrightarrow\; z_q \in \mathbb{R}^{960}$$
$$\boxed{\text{Candidate Generator}} \;\longrightarrow\; \text{Top-}k \text{ closest references in } \mathbb{R}^{330 \times 960}$$
$$\boxed{\text{Bilinear-MLP Verifier}} \;\longrightarrow\; \text{Accept } (k^*) \text{ or Abstain/Reject}$$

*   **Student Encoder (7.0M parameters):** A bidirectional GRU mapping text to a normalized $960$-dimensional vector space.
*   **Relation Verifier (1.92M parameters):** A dual-path Bilinear-MLP network that evaluates cosine similarity, Euclidean distance, text Jaccard overlap, and character 3-gram candidate overlaps to decide whether to route or reject.

### Why not LoRA or Standard RAG?
1.  **Low Footprint:** Runs entirely on CPU with a standalone disk size of **36.95 MB** (no heavy local transformer models needed at runtime).
2.  **Statistically Certified Safety:** Gated by one-sided Clopper-Pearson bounds on disjoint splits, ensuring false-routing rates remain strictly below user-defined risk budgets ($\le 2.0\%$ target).
3.  **Low Latency:** E2E query encoding and verifier routing runs in **19.51 ms** mean CPU latency.

---

## 2. Key Files in Repository

*   `run_production_pipeline_validation.py`: The primary validation script. Performs stratified split manifests, trains the student encoder, calibrates thresholds on validation splits, runs the E2E adaptive routing pipeline, and outputs Clopper-Pearson statistical certification reports.
*   `run_continual_learning_validation.py`: Simulated sequential learning benchmark (5 shuffles × 3 seeds) that evaluates EWC parameter regularization and coordinate anchoring against a frozen-memory control baseline.
*   `generate_scaling_dataset.py`: Reconstructs the 100-fact scaling dataset divided into 10 stratified blocks of 10 facts (containing cloze prompts, question-answer pairs, and paraphrases).
*   `run_relation_verifier_training.py`: Standalone script to train the Bilinear-MLP relation verifier model.

---

## 3. Milestones & Implementation Progress

### Phase A: Student Distillation (Complete)
*   Distilled the student GRU encoder from a frozen `SmolLM2-360M` teacher.
*   Achieved **60.86%** seen paraphrase and **70.50%** unseen fact transfer accuracy, matching teacher relational geometry.

### Phase B: relation Verifier (Complete)
*   Bilinear-MLP classifier verified on fact-disjoint splits.
*   Empirical False Positive Rate (FPR) of **0.00%** (UCB95 CP: **0.49%** for semantic distractors and **1.60%** for general controls), passing the project's safety gate.

### Phase C.1: Statistical Certification (Complete)
*   Calibrated validation threshold at $0.9127$, certifying the **Balanced 2.0% risk tier** (observed UCB95: **1.49%**).

### Phase C.2: Typo-Robustness & Full-Bank Ingestion (Complete)
*   Scaled reference index to include the full 110-fact memory bank (330 vectors).
*   Integrated a hybrid lexical character 3-gram generator to recover typo recall from $0\%$ to **1.67%** while maintaining **100% OOD safety** (0.00% false acceptances).

### Phase C.2.1: Continual-Learning Robustness (Complete)
*   Isolated a **6.00%** recall degradation baseline due to **capacity-interference (distractor expansion)** rather than parameter decay.
*   Proved that EWC + anchoring suppresses true parameter-level forgetting to just **0.17%** (a 97% reduction over naive sequential updates).

### Phase C.2.2: Stability-Plasticity Pareto Sweep (Active)
*   Running an 11-configuration grid search over learning rates and $(\lambda_{\text{ewc}}, \lambda_{\text{anchor}})$ regularizers to optimize the stability-plasticity trade-off and recover plasticity gain to $\ge 95\%$ of naive ($\ge 1.17\%$) while keeping paired excess forgetting $\le 0.50\%$.

---

## 4. Current Verification Locks & Commit Hashes

*   **Policy Config JSON:** `5936834ef973905cc775e4151d2918bfe7a2bd454fcd9df84713450b02fe2a1d`
*   **Split Manifest v2 JSON:** `84954b3b6881f5296f5164f7edc9a18c6ef8c4d39b63973857e3b876b05790cb`
*   **Verifier Checkpoint:** `ac7eec1e022d534c102a0a178f604a6f6dcd8a1d7222eb83f6d9aee3b5555e1c`
*   **Student weights Checkpoint:** `d10ae2edf7c46192fd5a2bc0c794b445099645d76da02c0586b17047195d36f6`
*   **Scaling Dataset JSON:** `5ce9b57f25f38c051d6ea77ce823e92a8d12996c94667287688867ac4ec76b75`
*   **Evaluation Script:** `2edbee21deb63581953125b3f4cfcc4a75a1a96d45766f178928cb4a33d3c385`
*   **Repository HEAD Commit:** `54527e0`

---

## 5. Execution Routine on Kaggle

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

## 6. Instructions for the Next AI Assistant

1.  **Retired Path (QPL CHL collapse):** Do not attempt to re-introduce the sparse unsupervised QPL lateral-learning or CHL projection layers. The sparse projection collapsed representation capacity to $11\%$ on 100 facts. The standalone Student GRU + Bilinear-MLP Verifier is the certified and selected mainline.
2.  **Calibrated Forgetting:** When evaluating continual-learning performance, always subtract the `frozen_encoder_writable_memory` baseline at the paired sample level first. Index-expansion capacity interference accounts for a $\sim 6.0\%$ baseline degradation that must not be counted as parameter forgetting.
3.  **Next Target (Option 2):** Once the Pareto sweep determines the optimal EWC/anchoring parameters that recover plasticity gain to $\ge 95\%$ of naive, transition to **Option 2: Decoder Integration (Conditional NLG)**.
