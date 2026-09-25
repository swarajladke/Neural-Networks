# CONTEXT — Continual Learning Research Program

**Author:** Swaraj Ladke  
**Repository:** `swarajladke/Neural-Networks`  
**Last Updated:** September 25, 2026 (Reflecting Directive S0-6 completion, commit `733fa24`)

---

## 1. Mission and Research Goal

The objective of this research program is to build a neural network architecture and training algorithm that **learns continuously without catastrophic forgetting**.

**Precise Specification:** *A neural model comparable in structure and utility to modern autoregressive and feedforward models, with one fundamental architectural difference: it ingests new knowledge incrementally over an indefinite operational horizon without requiring fine-tuning, replay buffers, or full retraining, preserving prior knowledge above negative control floors.*

The focus is physical learning mechanics, representation geometry, and causal isolation—not benchmark gaming.

---

## 2. Infrastructure & Compute Constraints

- **Local Machine:** Windows, Python 3.11, PyTorch 2.2.2+cpu (strictly no GPU execution locally).
  - *Standing Rule:* No local execution of training, testing, parameter sweeps, or model verifications. The local workspace is strictly for code authoring, AST scanning, unit stubs, report formatting, git commits, and git pushes.
- **Remote Accelerator:** Kaggle Notebooks (Tesla T4 GPU, 14.56 GB VRAM, Python 3.12, PyTorch 2.10.0+cu128, HuggingFace Transformers 5.0.0).
  - *Session Ceiling:* Kaggle hard session limit is ~6.5 hours (23,400 s).
  - *Compute Budget Ceiling:* Every experiment must budget and project its wall-clock beforehand, with execution halted if projected time exceeds 70% of the session ceiling (**16,380.0 s**).

---

## 3. Standing Scientific Protocol (Enforced by AGENTS.md)

1. **No Measured Value May Be Typed in Source or Text:** Every reported number must be interpolated by format string at runtime from a committed results JSON carrying its producing commit SHA. AST literal scanners (`NUMERIC = re.compile(r"\d+\.\d+|\d+\s*%|%\s*\d+")`) descend into print statements and f-string literals to abort on typed measurements.
2. **Provenance Guard 2.0:** The `Measurement` primitive cannot be instantiated via public constructor; it is constructible strictly via `Measurement.from_outcomes(outcomes, metric, arm, scope, input_set, mode)` with a module-private sentinel and a closed `POPULATION_REGISTRY`.
3. **No Zero-Step Results:** A run is valid only if all seeds exit with code 0 and record nonzero optimizer steps and nonzero samples seen. A script that outputs a results table without computing gradients or loading data is fabrication.
4. **Gate 0 (Positive Control):** Before measuring any new condition, the experiment must re-run and bit-reproduce known prior cells within stated tolerance. (e.g., S0-6 required exact reproduction of S0-5 pinned reference values: [669, 664, 656] steps, 1,989 total, 600/600 efficacy, 33/600 retention across seeds 0–2).
5. **Efficacy Gate & Dual Retention Reporting:** No retention or quality metric may be reported from any condition whose pooled immediate injection efficacy is below 90.00%. If an intervention exhausts steps and fails the gate, dual retention reporting is mandatory:
   - **3a Conditional Retention:** Evaluated strictly on the subset of facts that successfully injected.
   - **3b Matched-Subset Comparison:** Evaluated on the unconstrained control arm over the exact identical fact subset.
6. **Denominator Sum Assertions:** Denominators must equal the exact population the claim addresses. Any combined denominator must be computed as an expanded sum, printed (`20 + 20 + 20 + 20 = 80`), and asserted against the expected population.
7. **Negative Control Floors:** Every retention claim must be reported against its own negative control floor, never against zero. The worst individual control must always be printed beside pooled figures.
8. **Single-Artifact Reporting (AGENTS.md Section 14):** Reports are generated strictly by `python tools/make_report.py <directive>`, validated against strict formatting bans (no LaTeX `$`, no raw HTML, no mermaid, no hyperlinks), and verified byte-for-byte (`--verify`).
9. **Structural Limit:** All primary experiment scripts must remain strictly under 600 lines (`experiments/b1_inject.py` is at 598 lines).

---

## 4. The Two Tracks

### Track A — Image Continual Learning (Split-CIFAR-100, ResNet-18)
Established empirical findings across 5 seeds:
- **Joint Offline (Upper Ceiling):** 79.62% ± 0.21 class-IL; linear probe 79.30%.
- **Naive Fine-Tuning:** 9.53% ± 0.21 class-IL, 83.57% task-aware, probe 65.71%, BWT −88.06 pp.
- **Nearest Class Mean (NCM) on Frozen Features:** 47.12% class-IL (canonical single-pass 50.24%), probe 59.19% ± 0.09.
- **NCM on Adapting Features:** 41.98%, probe 65.63%.
- **ADAPTATION_GAP:** Validated at **+20.43 pp**.
- **Central Finding:** Forgetting is overwhelmingly **classifier bias, not representation loss**. The gap between class-IL (9.53%) and task-aware (83.57%) accuracy is +74 pp, and a linear probe on the catastrophically-forgotten backbone reaches 65.71%—exceeding the frozen backbone. The representation degrades minimally; the readout layer collapses.
- **Withdrawn Methods (Do Not Cite):** `5_lwf`, `6_ewc` (inert penalties), `7_er_buffer500`, `8_der_plus_plus_buffer500` (missing records), `1_freeze_after_base` (BatchNorm leak).

---

### Track B — Sequential Knowledge Injection into GPT-2 Small (Active Focus)
- **Model:** GPT-2 Small (124,439,808 parameters; tied `lm_head.weight` and `transformer.wte.weight`, 50,257 × 768).
- **Target Parameter Block:** `lm_head.weight` (row-wise projection across 50,257 rows, $d=768$).
- **Datasets:**
  - Pinned 1,000 synthetic facts (`b1_facts.json`, SHA-256 `285638ad…`), 4 relations × 250 facts (`born_city`, `profession`, `plays_instrument`, `capital_of_country`). Multi-token object target fraction: 97.0%.
  - Pinned WikiText-2 capability slice (1,000 sequences, SHA-256 `3fd93350…`, pre-edit baseline perplexity **36.03**).
  - Pinned control probe set (200 template prompts, SHA-256 `8f4ffa6b…`).

#### Core Discoveries on Track B (Directives S0-2 to S0-6):
1. **The Readout Shortcut:** Under unconstrained SGD, 95.7–95.8% of the single-edit gradient norm concentrates in the readout layer (`wte` + `ln_f`). Derivation: $\sqrt{1 - 63.02^2 / 215.10^2} = 95.7\%$.
2. **Localization Closure:** Resetting the ~15–17 K target token-embedding rows (~0.012% of network parameters) eliminates ~92% of capability damage and 100% of retention.
3. **Collinear Overwriting:** Mean pairwise cosine between sequential edit gradient directions is $0.9888 \pm 0.0072$ (within-relation $\approx 0.9997$; cross-relation $0.9857$). Sequential edits collapse onto nearly identical directions, causing immediate destructive interference.
4. **Causal Subspace Orthogonal Projection (Arm B):** Dynamically building an orthonormal basis $Q_t \in \mathbb{R}^{d \times r}$ ($r=1$) spanning previous update vectors and projecting new updates orthogonal to $Q_t$ ($P_\perp = I - Q_t Q_t^T$) strictly satisfies the Pythagorean identity ($\|P \Delta\|^2 + \|P_\perp \Delta\|^2 = \|\Delta\|^2$) across all 1,200 sequential edits.
5. **Perplexity Dissociation (S0-5):** Arm B decouples injection efficacy from general capability destruction, preventing runaway perplexity explosion.
6. **Negative Control Floors (S0-6, $N=1200$ across 6 seeds, pooled $N=4800$):**
   - `never_edited`: 6/1200 (0.50% [0.23%, 1.09%])
   - `random_direction_magnitude_matched`: 1/1200 (0.08% [0.01%, 0.47%])
   - `wrong_target` (**worst individual control**): 58/1200 (4.83% [3.76%, 6.20%])
   - `pre_edit_baseline`: 1/1200 (0.08% [0.01%, 0.47%])
   - **Pooled Floor:** 66/4800 (1.38%)

---

## 5. Current Benchmark: Directive S0-6 Empirical Results (Commit `733fa24`)

Directive S0-6 tested the hypothesis that the 20-edit retention horizon (where edits 1–180 retain at $\le 4.44\%$ while edits 181–200 retain at $\approx 20\%$) is an artifact of the greedy zero-margin stopping rule ($\text{margin} = p(y^*) - \max_{j \ne y^*} p(j) \ge 0$).

S0-6 executed a decoupled factor design sweeping margin $\delta \in \{0.0, 1.0, 3.0, 6.0\}$ in Arm A (`r0_unconstrained`) across 6 seeds (`SEEDS = [0, 1, 2, 3, 4, 5]`, $N=1200$ facts per condition, `max_steps = 100`), while evaluating Arm B (`r1_causal_perstep`) and Arm F (`r1_magnitude_only`) at $\delta = 0.0$ ($N=1200$).

### 5.1 Primary Deliverables Panel ($N=1200$ across 6 seeds)
| Condition | Pooled Imm Efficacy | Terminal Retention | Gen (3xN) | Locality KL | WikiText-2 PPL | Monotone Horizon $k$ | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `r0_unconstrained_d0.0` | 1200/1200 (100.00%) | 78/1200 (6.50%) | 203/3600 (5.64%) | 3.0666 | 60.71 | $k = 120$ | PASSED |
| `r0_unconstrained_d1.0` | 1200/1200 (100.00%) | 81/1200 (6.75%) | 207/3600 (5.75%) | 3.4705 | 71.52 | **$k = 140$** | PASSED |
| `r0_unconstrained_d3.0` | 1200/1200 (100.00%) | 75/1200 (6.25%) | 193/3600 (5.36%) | 4.3519 | 103.26 | $k = 70$ | PASSED |
| `r0_unconstrained_d6.0` | 871/1200 (72.58%) | 76/1200 (6.33%) | 185/3600 (5.14%) | 6.2062 | 225.06 | $k = 70$ | **FAILED (<90%)** |
| `r1_causal_perstep_d0.0`| 1187/1200 (98.92%) | 85/1200 (7.08%) | 190/3600 (5.28%) | 2.9537 | **50.84** | **$k = 150$** | PASSED |
| `r1_magnitude_only_d0.0`| 1197/1200 (99.75%) | 89/1200 (7.42%) | 216/3600 (6.00%) | 3.0053 | 56.01 | $k = 140$ | PASSED |

### 5.2 Key Scientific Conclusions
1. **Margin Expansion & Collapse:**
   - Shifting from greedy zero-margin ($\delta=0.0$) to $\delta=1.0$ expands the monotone retention horizon from $k=120$ to **$k=140$ edits** (+20 edits) with minor perplexity growth ($60.71 \to 71.52$).
   - Higher margins ($\delta=3.0, 6.0$) cause the horizon to **collapse back to $k=70$ edits**. Over-optimization degrades base network features, exploding WikiText-2 PPL to $103.26$ and $225.06$.
   - At $\delta=6.0$, 329 edits exhausted 100 optimizer steps, causing immediate efficacy to fall to **72.58%** (failing Gate 4). Dual retention reporting:
     - 3a Conditional Retention (successful edits): 55/871 (6.31% [4.88%, 8.13%])
     - 3b Matched-Subset Comparison (Arm A $\delta=0$ on same facts): 61/871 (7.00% [5.49%, 8.89%])
2. **Causal Projection Establishes the Longest Horizon:**
   - **Arm B (`r1_causal_perstep` at $\delta=0.0$)** achieved the longest monotone retention horizon observed to date: **$k=150$ edits** (remainder retention $14/300 = 4.67\%$, matching the control floor).
   - Arm B preserved model capability best (**WikiText-2 PPL 50.84** vs Arm A's 60.71; paired $t = -4.18, df = 5, p < 0.01$).
3. **Geometry vs Magnitude:**
   - Magnitude control alone (Arm F) achieves $k=140$ edits and PPL 56.01.
   - Causal projection (Arm B) provides an additional $+10$ edits of horizon ($k=150$) and superior perplexity preservation ($50.84$), confirming that geometric orthogonalization provides distinct protective value beyond step-size damping.
4. **Accounting & Protocol Integrity:**
   - Pre-flight test suite: 49/49 passed; AST literal scanner: 0 violations.
   - Positive controls passed on seeds 0–2 for all three baseline arms.
   - Total optimizer steps: 112,667 steps, closing to line-item attribution with **$\Delta = 0$**.
   - Wall-clock budget: Projected 15,651.0 s, Actual 15,932.9 s ($\le 16,380.0$ s ceiling).

---

## 6. Where the Research Goes Next (Upcoming Directives)

1. **Subspace Rank Scaling ($r > 1$):**
   All experiments to date utilized rank $r=1$ projection. Does increasing subspace rank ($r \in \{2, 4, 8\}$) protect older memory trajectories beyond $k=150$, or does rank accumulation exhaust the 768-dimensional row space and impede new acquisition?
2. **Joint Factorization (Margin + Causal Projection):**
   S0-6 evaluated Arm B strictly at $\delta=0.0$. Does combining causal projection with the optimal stopping margin ($\delta=1.0$) break through the $k=150$ horizon without language degradation?
3. **Layer Distribution & Internal Memory Consolidation:**
   With 95.8% of gradient norm landing in the readout layer under unconstrained SGD, can targeted orthogonalization or projection into intermediate MLP / Key-Value layers provide an associative store that protects early-sequence facts (edits 1–50)?

---

## 7. Active Codebase Organization

The repository has been restructured cleanly:
- `experiments/b1_inject.py`: Primary injection experiment engine (strictly maintained $< 600$ lines).
- `experiments/metrics.py`: Mathematical metrics, Wilson score confidence intervals, and Provenance Guard 2.0.
- `experiments/data.py`: Synthetic facts generation, deterministic seed sampling, and `CausalSubspaceManager`.
- `tests/test_metrics.py`: 49 pre-flight unit tests and AST literal scanner.
- `tools/make_report.py`: AGENTS.md §14 single-artifact report generator and byte-verifier.
- `reports/`: Validated markdown reports (`S0-2.md`, `S0-3.md`, `S0-4.md`, `S0-5.md`, `S0-6.md`).
- `experiments/results/`: Machine-readable results JSON artifacts (`s0_5.json`, `s0_6.json`).
- `s0_6_stdout.txt`: Verbatim execution stdout log.
- **Do not touch AGNIS files** (`agnis*.py`, quarantined legacy attempt).
