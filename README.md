# Continual Learning & Representation Learning Benchmark Suite

## 🚀 Overview

This repository develops, evaluates, and audits Continual Learning (CL) mechanisms under strict **Class-Incremental Learning (Class-IL)** evaluation, formal pre-registration, and verifiable provenance.

The research suite focuses on measuring representation adaptation, catastrophic forgetting, and backward transfer across parametric and non-parametric continual learning architectures.

---

## 🔬 Active Research Benchmark (Directives W & AA)

### **Split-CIFAR-100 Adaptation Gap Benchmark**
Evaluates continual representation learning using a **ResNet-18 (`IMAGENET1K_V1`)** backbone partitioned into 10 sequential tasks (10 classes per task, seeded 400 train / 100 val / 100 test per class):

- **Arm A (Frozen Pretrained Baseline + NCM)**: Fixed penultimate features with parameter-free Nearest Class Mean inference.
- **Arm B (Joint Offline Full Fine-Tune)**: Full gradient optimization of all backbone parameters across all 100 classes jointly.
- **Adaptation Gap Criterion**:
  $$\text{ADAPTATION\_GAP} = \text{JOINT\_OFFLINE\_FULL\_FINETUNE\_ACC\_T} - \text{FROZEN\_PRETRAINED\_NCM\_ACC\_T}$$
  *Halt Rule*: If $\text{ADAPTATION\_GAP} \le +15.0\text{ percentage points}$, representation learning contributes insufficiently and the task setup is rejected.

#### Running the Benchmark (Kaggle / Remote):
```bash
python run_aa11_adaptation_gap_pretrained.py
```
*Seeded class split metadata*: [`class_order_split_cifar100.json`](./class_order_split_cifar100.json)

---

## 🛡️ Verification, Provenance, & Audit Guards

The codebase enforces strict mechanical verification of all experimental assertions against committed machine-generated logs:

1. **Universal Number Verification Guard** ([`verify_all_numbers.py`](./verify_all_numbers.py)):
   * Audits every numeric literal in [`walkthrough.md`](./walkthrough.md) and [`RESULTS.md`](./RESULTS.md).
   * Verifies whole-token matches against git-tracked `*_stdout.txt` logs.
   * Generates a per-literal audit map: [`number_verification_map.tsv`](./number_verification_map.tsv).
   * Enforces classification rules (`THRESHOLD`, `DERIVED`, `RETRACTED`) defined in [`number_classification.json`](./number_classification.json).

2. **Strict Sourced Citation Guard** ([`run_p7_strict_citation_audit.py`](./run_p7_strict_citation_audit.py)):
   * Asserts 1-to-1 pre-registration integrity against [`predictions_phase_I_to_V.md`](./predictions_phase_I_to_V.md).
   * Validates literal presence, file suffix integrity, and partitioned status counts (`PASS`, `FAIL`, `EXEMPT`, `NOT_MEASURED`).

3. **Script Execution Status Generator** ([`build_execution_status.py`](./build_execution_status.py)):
   * Scans all repository runners, verifying presence, byte counts, and commit SHAs for stdout logs.

---

## 📋 Standing Rules for Continual Learning (Rules R1–R21)

All continual learning evaluations are governed by the standing rules in [`AGENTS.md`](./AGENTS.md), including:
* **Rule R1 (Permanent Control Arm)**: `FREEZE-AFTER-BASE` must be included in every continual-learning table.
* **Rule R2 (Decomposed Gap Reporting)**: Retention and acquisition gaps must be reported as separate metrics.
* **Rule R3 (Correction Flags)**: Any quantity that changes value between reports must be flagged in the formal correction registry.
* **Rule R16 (No Structurally Constant Metric)**: Metrics that are structurally constant by construction are forbidden.
* **Rule R17 (Seed Before Construction)**: Seed initialization must precede module construction across `SEEDS = [42, 43, 44, 45, 46]`.
* **Rule R18 (One Classifier Family Per Comparison)**: Comparisons across arms must hold classifier families fixed.
* **Rule R19 (Paste-Only Documentation)**: Artifact and execution tables must be verbatim pastes of committed `*_stdout.txt` logs.
* **Rule R20 (Paste-Only Counts)**: Audit counts and tallies must appear solely as verbatim log pastes.
* **Rule R21 (Exit-Code Integrity)**: Any script encountering a violation must terminate with a non-zero exit code (`EXIT_CODE = <n>`).

---

## 🏛️ Historical Archive & Legacy Checkpoint Status

* **Pre-Audit Historical Archive**: Earlier exploratory sprint results and custom cognitive architectures (Phases 1–7, SmolLM2 synthetic prompt-classification benchmarks) have been archived into [`RESULTS_ARCHIVE.md`](./RESULTS_ARCHIVE.md) per Directive AA8 (Option A).
* **Orphaned Binary Checkpoints**: Historical checkpoints retained in `checkpoints/` (`ru_milestone_*.pt`, `italian_baseline_v72.pt`, `phase_733_breakthrough.pt`) are preserved for non-destructive record purposes only.
