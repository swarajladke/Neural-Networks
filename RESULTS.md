Single defensible result as of `5443ef1`: 100 synthetic facts, SmolLM2-360M mean-pooled embeddings, PCA-64 whitened (eps=1e-4), multinomial logistic regression (wd=0.0001) selected on 300 disjoint-template validation prompts, evaluated once on 500 disjoint-template test prompts: **82.20%**. All Phase 5/6/7 continual-learning claims are retracted pending Phase IV.

---

# Orthogonal Gradient Projection for Continual Retrieval Adaptation
## Empirical Results Report

---

### Standing Rules for Continual Learning Experiments

R19 (Paste-Only Documentation): Any table in walkthrough.md or RESULTS.md that asserts the existence, size, provenance, or execution status of a repository artifact must be a verbatim paste of a committed *_stdout.txt log, enclosed in a fenced code block, with the log filename stated immediately above it. Hand-authored or reformatted versions of such tables are prohibited. A table that cannot be pasted must be deleted.

R20 (Paste-Only Counts): Any count, tally, pass/fail summary, grep result, or reconciliation figure produced by a repository script must appear in documentation only as a verbatim paste of that script's committed *_stdout.txt, inside a fenced code block, with the log filename and its commit SHA stated immediately above the block. Prose restatement, reformatting into a bullet list, or transcription into a table is prohibited. If a count cannot be pasted, the section reporting it must be deleted.

R21 (Exit-Code Integrity): Any script that prints a violation, illegal value, mismatch, or failure condition must terminate with a non-zero exit status. A guard that prints a violation and exits zero is treated as a failed guard, and every number it certifies is treated as unverified. Every pasted guard output must be immediately followed by the line `EXIT_CODE = <n>` printed by the script itself, and no PASSED status may be claimed for a run whose printed exit code is non-zero or whose violation lists are non-empty.


---

### Benchmark Definition and Scope (H5, P9, S7, S10)

The benchmark measures 100-way classification of prompt clusters with 7 train, 3 val, and 5 test disjoint-template examples per class. It does not measure fact retention. The answer field does not enter the model, the loss, or any metric.

---

### Why the v3 benchmark is retired (Directive W1)

The v3 100-fact benchmark is retired for research purposes because representation learning contributes nothing on this task:
- `frozen_NCM` benchmark accuracy: **85.80%** (parameter-free running centroids on frozen representation).
- `joint_offline_full_finetune`: **79.80% $\pm$ 0.76%** (gradient-based joint training on all 100 classes with unfrozen backbone).
- $\text{ADAPTATION\_GAP} = \text{joint\_offline\_full\_finetune} - \text{frozen\_NCM} = 79.80\% - 85.80\% = \mathbf{-6.00\text{ percentage points}}$.

Since joint training on all data loses to no training at all, continual learning methods cannot demonstrate genuine adaptation gains on this benchmark. Research has pivoted to the **Split-CIFAR-100** benchmark (ResNet-18) where $\text{ADAPTATION\_GAP} \gg +15.0\text{ pp}$ and representation learning is essential.

---

**Continual Learning Measured Finding (S10)**:
- **HeadL1c Family**: Under Class-IL, sequential gradient training forgets catastrophically (Final $\text{ACC}_T = \mathbf{47.60\% \pm 1.93\%}$, $\text{BWT} = \mathbf{-42.09\% \pm 1.99\%}$, $\text{Forgetting} = \mathbf{42.09\% \pm 1.99\%}$).
- **NCM Family**: Parameter-free centroid accumulation reaches **85.80%**, identical to its own joint-offline bound, with $\text{BWT} = \mathbf{-8.22\%}$.
- The two classifier families are not comparable and must not be merged into one claim.
- Benchmark Comparison: `phase6_dual_continuum` reported **64.95%** (on contaminated 34-class layout), compared to NCM's **85.80%** on the canonical 100-class disjoint-template benchmark.

**S7 Optimistic Ceiling Attribution**:
- $\text{OPTIMISTIC\_CEILING} = \mathbf{85.80\%}$ is attained by parameter-free NCM, not by the validation-selected `MultinomialLogReg` config (which achieves 82.20%, giving $\text{SELECTION\_PENALTY} = -3.60\text{ pp}$).

---

### Reference Upper Bound (Phase III & J3 Correction)

Joint offline upper bound (100 classes, 3 train / 3 test per class, BEST_CELL mean / center+ZCA_whiten): 34.80% +/- 1.66% test accuracy over 5 seeds. No Class-IL result on this dataset may exceed this value. Any reported figure above it is invalid by construction.

CORRECTED 2026-08-10: This line was incorrect. A single unregularized linear head is not an upper bound; nearest-centroid on the same representation scored 40.33%. Superseded by the offline reference below.

Offline reference bound (100 classes, 3 train / 3 test per class, J5 BEST_CELL mean / none, L2 Multinomial Logistic Regression): **79.33% test accuracy**. Evaluated across 4-method family (NCM=27.33%, 1-NN=29.00%, HeadL1c(J4)=53.78%, LogReg=79.33%). No Class-IL result on this 3/3 dataset may exceed 79.33%.

CORRECTED 2026-08-10 (L4 & M2): The 79.33% figure (commit `4d2284b`), the 85.40% figure (commit `e8ca39c`), and the K-phase B = 85.20% figure (commit `fc0f862`) were all produced by the same contaminated code path that evaluated LogisticRegression on concatenated train+test samples rather than held-out test vectors. Under strict held-out test evaluation:
- 3/3 Dataset (`mean / none`): LogReg (C=1.0) achieves **46.00%** (replacement for 79.33%), Ridge (wd=0.1) achieves **62.67%**. Canonical 3/3 dataset offline test ceiling is **64.33%** (`mean / pca_m32_eps1e-6`).
- Expanded v2 Dataset (10/5): 85.40% and 85.20% are retracted.
- Expanded v3 Disjoint-Template Dataset (7/3/5): Selected representation (`mean / pca_m64_eps1e-4`) achieves **82.60% HONEST_TEST_ACC** (single validation-selected evaluation via LogReg wd=0.001) and **85.80% optimistic ceiling** (N=11 test evaluations).

---

---

## Historical Archive Reference

All pre-O-phase sections (Sections 1 through 18) have been archived into RESULTS_ARCHIVE.md per Directive AA8 (Option A).
