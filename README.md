# Neural Networks Research Project: Continual Learning & Sequential Model Editing

[![Protocol: AGENTS.md](https://img.shields.io/badge/Protocol-AGENTS.md-blue.svg)](./AGENTS.md)
[![Verification: Pre--flight Test Suite](https://img.shields.io/badge/Pre--flight%20Tests-155%20Passed-brightgreen.svg)](./tests/test_metrics.py)
[![Environment: Remote Kaggle T4](https://img.shields.io/badge/Execution-Kaggle%20Tesla%20T4-orange.svg)](#-execution--reproducibility-pipeline)

## 🔬 Research Overview

This repository is an **empirical research engineering project** focused on the scientific understanding and mitigation of catastrophic forgetting in neural representations. Our work spans:

1. **Sequential Model Editing & Parametric Memory in Autoregressive LLMs** (Directives S0-1 through S0-14): Investigating why localized, closed-form parametric updates (e.g., ROME-style rank-1 updates and null-space projections) fail over long sequence horizons ($N = 200$ sequential fact edits across multiple seeds in GPT-2), auditing key-collision mechanisms, and testing algorithmic fixes such as slot consolidation and template protection.
2. **Continual Representation Learning Benchmarks** (Directives W & AA): Evaluating feature drift, backward transfer, and the adaptation gap between frozen pretrained representations and joint fine-tuning in class-incremental image classification (Split-CIFAR-100).

---

## ⚖️ The Prime Directive

All work in this repository is strictly bound by the research protocol defined in [`AGENTS.md`](./AGENTS.md):

> **A run that cannot be recomputed from pinned inputs is not a result. A number that cannot name the artifact it came from is not a measurement. A gate that can be edited to pass is not a gate.**

Every rule exists to eliminate confirmation bias, silent test degradation, definitional drift, and metric fabrication.

---

## 🎯 Active Research Tracks

### Track 1: Sequential Model Editing in Transformers (Directive S0 Series)

Parametric knowledge editing seeks to update specific factual associations inside autoregressive language models (e.g., layer 1 `c_proj` MLP weights) without degrading unedited factual knowledge or general language modeling capacity.

```
Sequential Edits (t=1..200) ──► Key Extraction (k_t) ──► Target Value Optimization (v*)
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       ▼                                             ▼
                          [Arm C1: Strict Protection]                  [Arm C2: Slot Consolidation]
                          Null-space projector audits                  Collision group G (cos >= 0.999)
                          ALL prior keys; unprojectable                Joint margin targeting (v*)
                          updates marked unwritable.                   Rank-1 write + non-colliding null space
```

#### Key Directives & Findings:
* **Null-Space Projection ($P_0$ & Sequential Projector $P_t$)**: Directives S0-11 and S0-12 tested sequential updates constrained to the uncentered key covariance null space $P_0$ and the orthogonal complement of previously edited keys in float64.
* **Full-Prompt Template Protection (S0-13)**: Tested whether protecting non-subject prompt tokens via SVD-truncated bases prevents forgetting. The pre-registered primary hypothesis was falsified (full-prompt protection reduced first-50 retention).
* **Key-Collision Audit & Slot Consolidation (S0-14)**:
  * Disclosed and audited a critical defect in earlier trackers where keys with cosine similarity $\ge 0.999$ were excluded from projection constraints, leaving them unprotected.
  * Implemented **Stage A Key-Collision Audit**: Evaluating $2 \times 2$ contingency tables (`excluded_later` $\times$ `lost_at_end`) via exact two-sided Fisher's exact test and collision anatomy decomposition.
  * Evaluated **Arm C2 (Slot Consolidation)**: Formed collision groups $G$ for colliding keys, optimized joint target values $v^*$ with margin targeting, and enforced exact null-space protection on all non-colliding prior keys via SVD-decoupled bases.

### Track 2: Continual Representation Learning & Adaptation Gap (Directive AA Series)

* Evaluates continual feature adaptation under sequential class-incremental learning (Split-CIFAR-100, ResNet-18).
* Formally separates representation acquisition from catastrophic forgetting by benchmarking against permanent standing control arms:
  $$\text{Adaptation Gap} = \text{Accuracy}_{\text{Joint Offline Fine-Tune}} - \text{Accuracy}_{\text{Frozen Pretrained NCM}}$$
* Implements the **Adaptation Gap Criterion**: If representation learning closes $\le +15.0$ percentage points over fixed features, representation learning contributes insufficiently and the task partition is rejected.

---

## 🛡️ Research Protocol & Verification Safeguards

Every script, commit, and report is enforced through mechanical code constraints:

1. **Pre-Flight Metric & Guard Test Suite** ([`tests/test_metrics.py`](./tests/test_metrics.py)):
   * **155 automated unit tests** must execute and pass before any machine learning model or checkpoint is loaded into memory.
   * Asserts exact mathematical properties: projector idempotence ($P^2 = P$), symmetry ($P^T = P$), rank-1 Conv1D updates with bias, exact Student's $t$ distributions, exact Wilcoxon signed-rank tests (64-permutation enumeration), and Fisher's exact tests.
2. **AST Startup Literal Scanner** ([`AGENTS.md` Appendix C.2](./AGENTS.md)):
   * Statically parses Python Abstract Syntax Trees (AST) across all runner and metric modules.
   * Prohibits unlisted decimal numbers, percentage literals, or hardcoded empirical values in source code and f-strings.
3. **Bitwise Bit-Reproduction Gates**:
   * **Gate G0**: Bitwise reproduction of historical unconstrained SGD controls ($669\text{ steps} \to 200/200\text{ immediate} \to 8/200\text{ terminal}$).
   * **Gate G0b**: SHA-256 verification of pinned covariance and null-space projectors (`cov_l1.pt`, `v_null_l1.pt`).
   * **Gate G0c**: Exact reproduction of historical baseline seeds before downstream experimental branches can execute.
4. **Denominated Metrics & Raw Outcomes** ([`AGENTS.md` §1.1, §3.7](./AGENTS.md)):
   * No bare percentages or scalars: every metric returns an explicit `Measurement(numerator, denominator, mode, input_set)`.
   * All experimental runs serialize the raw per-unit boolean outcome vectors into results artifacts (`experiments/results/*.json`) to enable independent re-analysis.
5. **Pre-Registered Paired Statistical Inference**:
   * All primary comparisons are pre-registered and evaluated using paired seed-level tests: exact Student's $t$ ($\text{df}=5$), exact Wilcoxon signed-rank, and Newcombe hybrid score confidence intervals.

---

## 📂 Repository Structure

```text
├── AGENTS.md                  # Governing Research Protocol & Policy Layer (R1-R21)
├── README.md                  # Project Overview & Architecture Guide
├── b1_facts.json              # Pinned fact dataset (200 facts per seed)
├── experiments/
│   ├── run_s0_14.py           # Directive S0-14 Pipeline Runner (Gates -> Stage A -> Stage B -> L3)
│   ├── s0_14_tracker.py       # Trackers: Strict (C1), Slot Consolidation (C2), Template (C3)
│   ├── s0_14_audit.py         # Stage A key-collision audit, Fisher exact test, collision anatomy
│   ├── s0_13_protection.py   # S0-13 SVD-truncated basis protection & margin targeting
│   ├── s0_13_localization.py # S0-13 L2a drift, L2b output patching, and weight reconstruction
│   ├── s0_11_constraints.py  # Historical null-space projectors and covariance estimation
│   ├── b1_inject.py           # Core transformer hook injection & greedy decoding routines
│   ├── data.py                # Seeded fact sampling & pinned WikiText-2 evaluation slices
│   ├── metrics.py             # Verified metric definitions & Measurement constructors
│   ├── stats.py               # Closed-form and exact paired statistical inference routines
│   └── results/               # Machine-generated results JSON artifacts (with raw outcome vectors)
├── reports/                   # Formal directive reports and empirical finding audits
│   ├── S0-10.md               # S0-10 Diagnostic & Conv1D Bias Repair Report
│   ├── S0-11.md               # S0-11 Null Space Projection & Stage C Report
│   ├── S0-12.md               # S0-12 Dual-Subspace Capacity & Modal Collapse Audit
│   └── S0-13.md               # S0-13 Full-Prompt SVD Protection Falsification Report
├── tests/
│   └── test_metrics.py        # 155 pre-flight metric stubs, AST literal scanner, math proofs
└── tools/
    ├── make_report.py         # Automated report generator interpolating from JSON results
    └── verify_all_numbers.py  # Number verification guard checking documentation against stdout
```

---

## 💻 Execution & Reproducibility Pipeline

In accordance with project rules, **heavy training, sweeps, and model-editing pipelines are executed on remote GPU environments (Kaggle Tesla T4)** under a clean, pinned git checkout. Local development is strictly restricted to authoring, AST verification, unit testing, report formatting, and git management.

### 1. Local Pre-Flight Verification
Run the pre-flight test suite and AST literal scanner locally (fast, CPU-only, zero accelerator needed):

```bash
python tests/test_metrics.py
```

### 2. Remote GPU Execution (Kaggle Notebook)
To reproduce experiments on Kaggle (Tesla T4 GPU), run a clean checkout in a notebook cell:

```bash
%%bash
rm -rf Neural-Networks
git clone https://github.com/swarajladke/Neural-Networks.git
cd Neural-Networks

# 1. Run Pre-flight Metric and Audit Unit Tests
python tests/test_metrics.py

# 2. Run Directive Pipeline (e.g., S0-14 Audit & Slot Consolidation)
python experiments/run_s0_14.py --device cuda --ceiling 25000.0
```

### 3. Generating Verified Reports
Reports are never hand-authored. They are automatically compiled from machine-generated results artifacts:

```bash
python tools/make_report.py S0-14
python tools/make_report.py S0-14 --verify
```

---

## 📜 Standing Rules of the Laboratory

Key research constraints enforced by [`AGENTS.md`](./AGENTS.md):
* **No Manual Result Fabrication (§3.1)**: No measured number may be typed into code, print statements, or markdown prose; all numbers must be interpolated directly from results JSON artifacts.
* **Immutable Thresholds (§2.1, §2.2)**: Thresholds are read from content-addressed artifacts and verified by hash. A gate cannot be edited or widened to pass.
* **Permanent Control Arm (§13 Rule 1)**: `FREEZE-AFTER-BASE` (zero updates) must be present in every evaluation table to confirm continual learning outperforms doing nothing.
* **Decomposed Reporting (§13 Rule 2)**: Acquisition and retention must be reported as separate gaps, preventing retention gains from masking acquisition failures.
* **Negative Findings as Primary Knowledge (§2.4)**: A failing gate or falsified hypothesis is a successful measurement that protects subsequent research from pursuing flawed assumptions.
