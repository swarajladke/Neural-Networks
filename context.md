# CONTEXT — Continual Learning Research Program

**Author:** Swaraj Ladke  
**Repository:** `swarajladke/Neural-Networks`  
**Last Updated:** September 25, 2026 (Reflecting Directive S0-7a completion, commit `e1286f1`)

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
4. **Gate 0 (Positive Control):** Before measuring any new condition, the experiment must re-run and bit-reproduce known prior cells within stated tolerance.
5. **Efficacy Gate & Dual Retention Reporting:** No retention or quality metric may be reported from any condition whose pooled immediate injection efficacy is below 90.00%. If an intervention exhausts steps and fails the gate, dual retention reporting is mandatory:
   - **3a Conditional Retention:** Evaluated strictly on the subset of facts that successfully injected.
   - **3b Matched-Subset Comparison:** Evaluated on the unconstrained control arm over the exact identical fact subset.
6. **Denominator Sum Assertions:** Denominators must equal the exact population the claim addresses. Any combined denominator must be computed as an expanded sum, printed (`20 + 20 + 20 + 20 = 80`), and asserted against the expected population.
7. **Negative Control Floors:** Every retention claim must be reported against its own negative control floor, never against zero. The worst individual control must always be printed beside pooled figures.
8. **Single-Artifact Reporting (AGENTS.md Section 14):** Reports are generated strictly by `python tools/make_report.py <directive>`, validated against strict formatting bans (no LaTeX `$`, no raw HTML, no mermaid, no hyperlinks), and verified byte-for-byte (`--verify`).
9. **Structural Limit:** All primary experiment scripts must remain strictly under 600 lines (`experiments/b1_inject.py` is at 597 lines).
10. **Raw Per-Unit Outcome Serialization (Rule 3.7):** Every count-based result must serialize raw per-unit boolean outcome vectors keyed by seed and within-sequence index into its results JSON.

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

## 5. Prior Benchmark: Directive S0-6 Empirical Results (Commit `733fa24`)

S0-6 executed a decoupled factor design sweeping margin $\delta \in \{0.0, 1.0, 3.0, 6.0\}$ in Arm A (`r0_unconstrained`) across 6 seeds (`SEEDS = [0, 1, 2, 3, 4, 5]`, $N=1200$ facts per condition, `max_steps = 100`), while evaluating Arm B (`r1_causal_perstep`) and Arm F (`r1_magnitude_only`) at $\delta = 0.0$ ($N=1200$).

### 5.1 Primary Deliverables Panel ($N=1200$ across 6 seeds)
| Condition | Pooled Imm Efficacy | Terminal Retention | Gen (3xN) | Locality KL | WikiText-2 PPL | Trailing Separation Depth $k$ | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `r0_unconstrained_d0.0` | 1200/1200 (100.00%) | 78/1200 (6.50%) | 203/3600 (5.64%) | 3.0666 | 60.71 | $k = 120$ | PASSED |
| `r0_unconstrained_d1.0` | 1200/1200 (100.00%) | 81/1200 (6.75%) | 207/3600 (5.75%) | 3.4705 | 71.52 | **$k = 140$** | PASSED |
| `r0_unconstrained_d3.0` | 1200/1200 (100.00%) | 75/1200 (6.25%) | 193/3600 (5.36%) | 4.3519 | 103.26 | $k = 70$ | PASSED |
| `r0_unconstrained_d6.0` | 871/1200 (72.58%) | 76/1200 (6.33%) | 185/3600 (5.14%) | 6.2062 | 225.06 | $k = 70$ | **FAILED (<90%)** |
| `r1_causal_perstep_d0.0`| 1187/1200 (98.92%) | 85/1200 (7.08%) | 190/3600 (5.28%) | 2.9537 | **50.84** | **$k = 150$** | PASSED |
| `r1_magnitude_only_d0.0`| 1197/1200 (99.75%) | 89/1200 (7.42%) | 216/3600 (6.00%) | 3.0053 | 56.01 | $k = 140$ | PASSED |

---

## 6. Current Benchmark: Directive S0-7a Audit and Recalibration (Commit `e1286f1`)

Directive S0-7a conducted a rigorous mathematical and statistical audit of the retention horizon estimator $k$, its fragility, permutation null distribution, exact paired inference, and the evaluation ceiling.

### 6.1 Key Empirical Findings & Recalibration Matrix
1. **Mathematical Anatomy of the Horizon Statistic:**
   - The estimator `compute_monotone_retention_horizon` does **not** count surviving facts or measure durable retention across earlier edits.
   - It measures the **depth of a trailing recency window** $[200-k, 200)$ whose pooled Wilson lower bound strictly exceeds the control floor upper bound ([0.0376, 0.0620] from `wrong_target`: 58/1200) continuously up to the first failing step.
   - It stops at the first failure and discards any later re-separating steps. In `r0_unconstrained_d0.0`, $k=130$ fails ($61/780 = 7.82\%$, Wilson lo $0.0614 < 0.0620$), but $k=140$ re-separates ($66/840 = 7.86\%$, Wilson lo $0.0622 > 0.0620$).
2. **Extreme Boundary Fragility:**
   - The reported horizon $k=150$ in Arm B (`r1_causal_perstep_d0.0`) separates from the control floor by only $+0.0010$ (Wilson interval: [0.0630, 0.0983] vs floor hi $0.0620$).
   - **Flip Margin to Destroy $k=150$:** Exactly **2 matching facts** flipped to non-matches inside the trailing window of 900 observations collapses the horizon to $k=100$.
   - **Flip Margin to Extend to $k=160$:** Only **4 facts** outside the window need to flip to match to extend the horizon.
3. **Null Distribution and Pre-Registered Decision Rule:**
   - 10,000 Monte Carlo permutations per condition (within-seed and pooled) evaluated the probability of observing $k \ge 150$ under the null hypothesis of uniform, order-independent retention.
   - Null 95th percentile across all conditions is $k=0$; null 99th percentile for Arm B is $k=30$.
   - One-sided p-value for Arm B: $p = 0.0008$.
   - Pre-registered decision rule: **PASSED**. Binding verdict: **VALID — RETENTION HORIZON SURVIVED NULL CALIBRATION**. The recency-gradient separation observed in Arm B is not an artifact of random sampling.
   - **Critical Scope of Permutation Test (Directive S0-7b §1.1):** The permutation test validates ONLY the existence of a recency gradient, not horizon magnitude or any ranking between arms. All five tested conditions passed the null calibration, and `r0_unconstrained_d0.0` passed more strongly ($p = 0.0001$) than `r1_causal_perstep_d0.0` ($p = 0.0008$). No between-arm comparison was performed in S0-6 or S0-7a.
4. **Correction Notice on Multi-Token Object Target Disclosure (Directive S0-7b §3 & Amendment 1 §A):**
   - The historical multi-token target object fraction disclosed across reports S0-2 through S0-6 was computed on bare object strings without leading space (`tokenizer.encode(f["object"].strip())`).
   - In GPT-2 byte-pair encoding, bare strings lack the space character byte, causing word-initial entity words to be split into multiple tokens. In actual training and evaluation (`b1_inject.py` lines 72 and 93), prompt text is concatenated with an intervening space (`f"{edit_prompt} {object}"`), realizing the leading-space convention. Under this leading-space convention, object words tokenize with leading space, yielding a substantially lower multi-token fraction.
   - Crucially, zero objects in `b1_facts.json` exceed the `max_new_tokens` ceiling of 5 (mean length = 1.36, max = 3). Greedy decoding with `max_new_tokens=5` did not truncate any targets.
5. **Withdrawal of S0-6 Conclusions 2 and 3 (Directive S0-7b §8.3):**
   - **S0-6 Conclusion 2 (Causal Projection Achieves Longest Horizon): WITHDRAWN.** Withdrawn on the grounds that between-arm horizon differences are smaller than measured flip margins (flipping 2 facts inside the window destroys the horizon), and no inferential between-arm comparison was performed.
   - **S0-6 Conclusion 3 (Geometry Adds Value Beyond Magnitude): WITHDRAWN.** Withdrawn on the grounds that between-arm differences fall within noise and flip margins. Any between-arm slope difference demonstrated in Stage H3 of S0-7b represents a new empirical finding and does not retroactively rehabilitate S0-6 claims.
6. **Statistical Machinery Corrections:**
   - **Proper Two-Proportion Test (D1):** Newcombe hybrid score intervals for the difference between the trailing window proportion and the negative control floor (`wrong_target`: 58/1200) strictly exclude zero for all conditions (Arm B diff: $+0.0306$, 95% CI $[+0.0096, +0.0528]$).
   - **Exact Paired Inference (D2):** Exact Student's $t$ with dynamic degrees of freedom ($df = \text{len}(\text{diffs}) - 1$) and Wilcoxon signed-rank ($n=6, 2^6=64$) tests audited across 18 comparisons:
     - Arm B vs Arm A Perplexity Claim: **SUPPORTED** under Student's $t$ ($t = -4.18, df = 5, \text{exact } p = 0.0087 < 0.01$). Under Wilcoxon signed-rank, $W=0.0 \implies \text{exact } p = 2/64 = 0.03125$, which represents the exact theoretical floor for $n=6$.
7. **Serialization Gap Identified & Protocol Updated:**
   - Analyses B4 (Seed jackknife), B5 (Per-seed $k$), and C1 (Control-arm $k$) were not computable from S0-6 artifacts because raw per-unit outcome vectors were not serialized.
   - Added Rule 3.7 to `AGENTS.md` mandating raw per-unit boolean outcome vector serialization in all future results JSON artifacts.

| Condition | Observed $k$ | Wilson Interval at $k$ | Signed Gap | Flips to Destroy | Flips to Extend +10 | Null P95 | Null P99 | Permutation $p$ | Two-Prop Newcombe 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `r1_causal_perstep_d0.0` | **150** | [0.0630, 0.0983] | +0.0010 | **2** | 4 | 0 | 30 | **0.0008** | [+0.0096, +0.0528] |
| `r0_unconstrained_d0.0`  | 120 | [0.0641, 0.1043] | +0.0021 | 2 | 1 | 0 | 20 | 0.0001 | [+0.0111, +0.0584] |
| `r0_unconstrained_d1.0`  | 140 | [0.0633, 0.1001] | +0.0013 | 2 | 3 | 0 | 20 | 0.0003 | [+0.0100, +0.0544] |
| `r0_unconstrained_d3.0`  | 70  | [0.0707, 0.1271] | +0.0087 | 5 | 1 | 0 | 10 | 0.0003 | [+0.0188, +0.0805] |
| `r0_unconstrained_d6.0`  | 70  | [0.0707, 0.1271] | +0.0087 | 5 | 1 | 0 | 20 | 0.0007 | [+0.0188, +0.0805] |

---

## 7. Active Directive: Directive S0-7b Execution

Directive S0-7b addresses all unresolved issues from S0-7a under the strict execution order:
**Stage J -> Gate 0 -> Remainder of Stage F -> Stage G -> Stage H -> Stage I**

1. **Stage J (Tokenization Audit):** Complete disclosure audit measuring object lengths and multi-token fractions under bare vs leading-space conventions at runtime without hardcoded literal percentages.
2. **Gate 0 (Bit-Reproduction Positive Control):** Seed 0 of `r0_unconstrained_d0.0` re-run first, verifying exact match of optimizer steps (669), immediate efficacy (200/200), and terminal retention (8/200) with early abort protocol.
3. **Stage F (Re-Emission):** Re-runs 4 intervention arms and 4 negative controls across 6 seeds ($N=1200$) emitting full Rule 3.7 raw boolean outcome vectors.
4. **Stage G (Uncomputable Analyses):** G0 full-population reproduction, G1 B4 seed jackknife ($N=1000$), G2 B5 per-seed horizons ($N=200$), G3 C1 control horizons, and G4 completion of B2, B3, and C3 ($20 \times 6 = 120$ asserted tests).
5. **Stage H (Estimator Repair & Between-Arm Test):** H1 maximal separating depth beside first-crossing $k$ (auditing S0-6 Conclusion 1); H2 position-resolved retention curves and first-50-edit retention vs floor; H3 paired seed-level logistic position slope test with dynamic degrees of freedom and exact Wilcoxon floor; H4 selection-corrected fixed-window ($k=100$) Newcombe test.
6. **Stage I (Weight-Tying Confound):** Unties `lm_head.weight` from `transformer.wte.weight` with verified independent clone; runs untied cells on seeds 0..2 while reusing Stage F tied cells; evaluates Difference-in-Differences for WikiText-2 perplexity.

---

## 8. Active Codebase Organization

- `experiments/b1_inject.py`: Primary injection experiment engine (strictly maintained $< 600$ lines, currently 559 lines).
- `experiments/stats.py`: Standalone statistical engine (incomplete beta, exact Student $t$, exact Wilcoxon signed-rank, Newcombe intervals, Newton-Raphson logistic regression, seed cluster bootstrap; zero SciPy dependency).
- `experiments/stage_j.py`: Stage J tokenization disclosure audit engine (bare vs leading-space conventions on 1,000 facts).
- `experiments/re_emission.py`: Empirical budget projection, Gate 0 early abort runner, and Stage F re-emission engine with Rule 3.7 raw vector serialization.
- `experiments/s0_7b_audit.py`: Stage G and Stage H analysis engine (G0, B4 jackknife, B5 per-seed, C1 control horizons, G4 completion, H1 maximal depth, H2 position curves, H3 paired slope tests, H4 selection correction).
- `experiments/weight_tying.py`: Stage I weight-tying confound runner with verified independent clones and WikiText-2 PPL diff-in-diffs.
- `experiments/run_s0_7b.py`: Master orchestrator for Directive S0-7b.
- `experiments/horizon_audit.py`: S0-7a horizon audit runner (maintained $< 600$ lines).
- `experiments/run_s0_7a.py`: Master orchestrator for Directive S0-7a.
- `experiments/metrics.py`: Mathematical metrics, Wilson score confidence intervals, and Provenance Guard 2.0 with closed `POPULATION_REGISTRY`.
- `experiments/data.py`: Synthetic facts generation, deterministic seed sampling, and `CausalSubspaceManager`.
- `tests/test_metrics.py`: Pre-flight unit test suite (Tests 1.1-1.6, 2.1-2.4, 3.1-3.18) and AST literal scanner.
- `tools/make_report.py`: AGENTS.md §14 single-artifact report generator and byte-verifier supporting S0-2 through S0-7b.
- `reports/`: Validated markdown reports (`S0-2.md`, `S0-3.md`, `S0-4.md`, `S0-5.md`, `S0-6.md`, `S0-7a.md`, `S0-7b.md`).
- `experiments/results/`: Machine-readable results JSON artifacts (`s0_5.json`, `s0_6.json`, `s0_7a.json`, `s0_7b.json`).
- `s0_6_stdout.txt`, `s0_7a_stdout.txt`, `s0_7b_stdout.txt`: Verbatim execution stdout logs.
- **Do not touch AGNIS files** (`agnis*.py`, quarantined legacy attempt).

