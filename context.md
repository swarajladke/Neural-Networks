# CONTEXT — Continual Learning Research Program

**Author:** Swaraj Ladke  
**Repository:** `swarajladke/Neural-Networks`  
**Last Updated:** October 1, 2026 (Reflecting Directive S0-10 completion, commit `99d3335`)

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

## 7. Directive S0-7b Benchmark Findings (Commit `f53a244`)

Directive S0-7b executed full-population re-emission on GPU ($N=1200$, 6 seeds, 24,398 optimizer steps, wall-clock 11,537.4 s) under exact Gate 0 bit-reproduction, serializing complete Rule 3.7 boolean outcome vectors.

### 7.1 Empirical Outcomes & Status of Historical Claims
1. **No Early Retention Above Negative Control Floor:**
   - First-50-edit retention across all four arms overlapped the negative control floor (`wrong_target`: 58/1200, 4.83% [3.76%, 6.20%]). For Arm B (`r1_causal_perstep_d0.0`), first-50 retention was 14/300 (4.67% [2.80%, 7.68%]), statistically indistinguishable from background noise.
2. **Formal Retirement of the Trailing-Window Horizon Statistic (AGENTS.md §1.7):**
   - The trailing-window separation-depth statistic $k$ is formally retired. Per-seed estimates carried standard deviations exceeding their means, jackknife spans overlapped across all arms, the metric increases with sample size at fixed underlying retention, and boundary flip margins are 2 facts. It may NOT be used as a primary or secondary endpoint in any future directive.
3. **S0-6 Conclusion 1 (Margin Expands Horizon): WITHDRAWN.**
   - Under the maximal separating depth estimator $k_{\text{max}}$, unconstrained injection ($\delta=0.0$) attains $k_{\text{max}}=140$, matching margin-scaled injection ($\delta=1.0$) at $k_{\text{max}}=140$. The previously reported 20-edit advantage was an artifact of the first-crossing estimator stopping upon transient dips below threshold.
4. **S0-6 Conclusions 2 & 3: WITHDRAWN UNCONDITIONALLY.**
   - Between-arm differences are smaller than empirical flip margins, and between-arm logistic position slope testing showed no statistically significant difference between Arm B and Arm F ($t(5)=1.54, p=0.18$) or Arm A ($t(5)=-1.15, p=0.30$).
5. **Surviving Findings:**
   - **The Recency Gradient:** Recent-half retention (edits 100..199, $N=600$) strictly separates from the control floor in Arm B (54/600 = 9.00% vs 58/1200 = 4.83%, Newcombe 95% CI $[+1.72\%, +6.94\%]$).
   - **Capability Preservation under Causal Projection:** Arm B preserves language modeling capability (WikiText-2 perplexity 50.84 vs 60.71 in Arm A, $t(5)=-4.18, p=0.0087$).
   - **Weight-Tying Independence (Stage I):** Breaking the embedding tie leaves Arm A's perplexity damage intact (DiD $\Delta\Delta \text{PPL} = -0.8002 \pm 0.3023$, $t(2)=-4.5845, p=0.0444$), placing capability damage firmly on the output projection side.
   - **Multi-Token Disclosure Correction:** Bare-string tokenization historically yielded 97.0% multi-token targets; greedy decoding with the leading-space convention actually realizes 30.8% multi-token targets, with zero exceeding the 5-token budget.

---

## 8. Directive S0-8 Empirical Findings (Commit `4f293e9` / `3522df6`)

Directive S0-8 relocated the sequential rank-1 write target off the readout and into the feed-forward value projection `transformer.h.L.mlp.c_proj.weight` ($L \in \{1, 6, 10\}$) under a completely frozen readout (asserted bitwise-zero parameter delta across all seeds on `lm_head.weight`, `transformer.wte.weight`, and `transformer.ln_f`).

### Established State as of S0-8:
1. **Readout Editing Null:** Readout editing (`lm_head.weight`, tied to `wte`) produced no early-sequence retention above the `wrong_target` floor, and no paraphrase generalization above floor.
2. **MLP Write Inefficacy:** S0-8 moved a rank-1 SGD write to `transformer.h.L.mlp.c_proj.weight`, $L \in \{1, 6, 10\}$, readout frozen and asserted. Immediate efficacy was 20/1200 (1.67%), 7/1200 (0.58%), and 1/1200 (0.08%) for $L = 1, 6, 10$, and every edit exhausted `max_steps=100`. **The site was not writable under that optimizer and budget.** No retention or generalization conclusion is drawn from S0-8, because retention from arms below the 90.00% efficacy gate is not reportable (AGENTS.md §11.5).
3. **No Positive Control at New Write Site:** S0-8 had no positive control at the new write site. Whether the failure is a physical property of the site or an algorithmic defect in the write path is unknown.
4. **Compute Overrun:** S0-8 took 16,637.48 s against a 7,820 s projection and exceeded the compute ceiling (16,380.0 s).

---

## 9. Directive S0-9 Empirical Findings (Completed)

**Objective:** Positive control first. Establish whether any write procedure at `mlp.c_proj` at some layer can achieve at least 90.00% immediate efficacy on the pinned facts (single edit, fresh model each time) before any sequential or retention experiment.

### Established State as of S0-9:
1. **Gate 0 Exact Bit-Reproduction:**
   - Seed 0 fact sequence sampled using `sample_200_facts(facts_1000, seed=0)`.
   - Re-confirmed exact historical baseline: 669 optimizer steps, 200/200 immediate efficacy, 8/200 terminal retention. Status: PASSED.
2. **Stage P (Write-Path Verification):**
   - All 20 facts at Layer 6 verified operable on fresh model states with nonzero finite gradients ($||\nabla W|| \in [5.087, 10.820]$) and strictly positive log-probability gains under unconstrained gradient step ($\Delta \text{LogP} \in [+0.2586, +1.0743]$). Status: PASSED.
3. **Stage W (Writability Sweep across 100 Facts):**
   - **W1 SGD at $\text{lr}=3.0\times 10^{-5}$ (S0-8 baseline):** 0/100 (0.00%) immediate efficacy across all layers ($L \in \{1, 3, 6, 9, 11\}$). Reproduces S0-8 write-inefficacy finding.
   - **W1 SGD at $\text{lr}=3.0\times 10^{-4}$ (10x baseline):** 0/100 (0.00%) immediate efficacy across all layers.
   - **W1 SGD at $\text{lr}=3.0\times 10^{-3}$ (100x baseline):**
     - $L=1$: 80/100 (80.00%) [71.12%, 86.66%], steps = 57.7, PPL = 33.90, LocKL = 0.0195
     - $L=3$: 79/100 (79.00%) [70.02%, 85.83%], steps = 58.5, PPL = 33.94, LocKL = 0.0215
     - $L=6$: 75/100 (75.00%) [65.70%, 82.45%], steps = 60.6, PPL = 33.92, LocKL = 0.0220
     - $L=9$: 22/100 (22.00%) [15.00%, 31.07%], steps = 93.7, PPL = 33.81, LocKL = 0.0072
     - $L=11$: 25/100 (25.00%) [17.55%, 34.30%], steps = 93.6, PPL = 33.89, LocKL = 0.0059
   - **W2 Closed-Form Key-Value Update ($\lambda=0.5, \text{lr}_v=0.1, 20\text{ steps}$):** 0/100 (0.00%) immediate efficacy across all swept layers.
4. **Feasibility Gate ($\ge 90.00\%$ Immediate Efficacy):**
   - **Result:** FAILED across all arms and layers. Maximum observed efficacy was 80.00% ($L=1$ at $\text{lr}=3\times 10^{-3}$).
5. **Negative Control Evaluation (wrong_target & wrong_target_paraphrase):**
   - `wrong_target` (100 canonical prompts): 0/100 (0.00%) [0.00%, 3.70%].
   - `wrong_target_paraphrase` (300 paraphrase prompts): 1/300 (0.33%) [0.06%, 1.86%].
6. **Pre-Flight Test Suite:** 127 tests run, 127 passed, 0 failures. AST literal scanner: 0 unlisted violations.

---

---

## 10. Established State as of Directive S0-10 (Commit `99d3335`)

1. **Gate 0 Exact Bit-Reproduction:** Seed 0 of `r0_unconstrained_d0.0` reproduced identically at 669 steps, 200/200 immediate efficacy, 8/200 terminal retention.
2. **Q1 — MLP Writability Feasibility Gate:**
   - Multiple write procedures at `transformer.h.L.mlp.c_proj.weight` surpassed the 90.00% immediate efficacy feasibility gate across layers 1, 3, and 6.
   - At L1, full-gradient matrix SGD reached 98.00% (lr 3e-3, cap 300) and 100.00% (lr 1e-2 and 3e-2). Repaired closed-form write reached 97.00% (mean 13.3 steps).
   - Pre-registered selection rule (minimum Locality KL, breaking ties with delta PPL) selected **`W2_Repaired_L1`** (`LocKL = 0.0482`, `Delta PPL = +0.12`).
3. **Q2 — Closed-Form Write Diagnostic (Stage D):**
   - In S0-9, closed-form writing achieved 0% efficacy because `Conv1D` bias was omitted from the rank-1 update math ($k(W+\Delta) \ne v^*$ in the presence of $b$, residual error 1.046).
   - $v^*$ optimization required 100 optimization steps with $L_2$ regularization $\lambda = 0.0$ to achieve 95.0% immediate efficacy on 20 facts.
4. **Q3 — Sequential Retention at Writable Site (Stage S):**
   - Conducted across 6 independent seeds x 200 sequential edits with bitwise-zero frozen readout verified on all seeds.
   - **Primary Endpoint (First-50 Terminal Retention, Pooled N=300):** 6/300 (2.00%) vs `wrong_target` floor 58/1200 (4.83%). Difference: -2.83%, Newcombe 95% Hybrid Score CI: [-4.57%, -0.30%]. Verdict: **BELOW** floor.
   - **Secondary Endpoint (First-50 Paraphrase Generalization, Pooled N=900):** 10/900 (1.11%) vs `wrong_target_paraphrase` floor 1/300 (0.33%). Difference: +0.78%, Newcombe 95% Hybrid Score CI: [-0.83%, +1.74%]. Verdict: **AT** floor (spanning zero).
   - **Perplexity Degradation:** Sequential rank-1 MLP value updates caused severe global representation drift, with full-slice perplexity exploding to 623 – 12,331 (vs unedited baseline 33.87).
   - **Conclusion:** While mid-layer MLP value projections are writable with high immediate efficacy and low single-edit perturbation, **unconstrained sequential rank-1 updates do not sustain retention above matched negative controls and suffer severe cumulative drift**.

---

## 11. Established State as of Directive S0-11 (Commit `07cb7ee` / Canonical Run)

Directive S0-11 evaluated constrained sequential knowledge injection at the writable MLP site (`c_proj.weight`) across Layers 1 and 6 over 6 seeds $\times$ 200 sequential edits, incorporating Amendment 1 directives for Stage N diagnostic and Stage N2 corrected closed-form null projection.

### 11.1 Stage 0 & Gate 0 Historical Baseline Re-Confirmation
- Seed 0 of `r0_unconstrained_d0.0` reproduced identically bit-for-bit: 669 steps, 200/200 immediate efficacy, 8/200 terminal retention. Status: **PASSED**.
- Stage S of S0-10 was reconciled under AGENTS.md §11.5 as non-reportable due to model collapse (PPL 623–12,331 vs baseline 36.03) and efficacy gate failure (82.00% < 90.00%).

### 11.2 Stage C: Key Covariance & Null Space Geometry
- Collected 100 sequences (51,200 tokens) from WikiText-2 train split (asserted disjoint from capability slice).
- Layer 1: Condition number $1.31 \times 10^8$ ($\lambda_{\max}=11.2, \lambda_{\min}=8.54 \times 10^{-8}$); null dimension 2,235 / 3,072 (relative threshold $10^{-3}$, retained energy 9.35%).
- Layer 6: Condition number $1.65 \times 10^5$ ($\lambda_{\max}=19.1, \lambda_{\min}=1.16 \times 10^{-4}$); null dimension 2,235 / 3,072 (relative threshold $10^{-3}$, retained energy 16.74%).

### 11.3 Invalidated Arm: Original A-null Specification & Stage N Diagnostic Post-Mortem
- **Status:** Original `A-null_L1` arm is classified **INVALID — IMPLEMENTATION DEFECT UNDER INVESTIGATION**. Seed 0 and Seed 1 figures from the halted run are retained for diagnosis only and may not appear in any scientific verdict or finding.
- **Specification Errors Acknowledged:** (1) The directive specified $\Delta W = \Delta W_{\text{cov}} P_0$, which fails to preserve the key-to-value constraint for the edited fact and amplifies null components via ridge-regularized $C^{-1}$; (2) Null tolerance was specified as an unnormalized absolute bound on $\|k_j \Delta W\|$.
- **Stage N Float64 Diagnostic (Seed 0, L1, First 50 Edits):**
  - Projector orientation was verified correct (mean $\|P_0 k_{\text{pres}}\| / \|k_{\text{pres}}\| = 0.0000$, confirming true null-space projection).
  - Primary defect confirmed: **(i) Norm amplification from $C^{-1}$ in the null space**. Average update norm amplified $6.25\times$ (peak $83.09\times$ at edit 45) relative to A-cov, driving full PPL to 2,350 by edit 50.
  - Prior float32 lattice derivation formally rejected.

### 11.4 Stage N2: Corrected Closed-Form Null-Space Projector
- Formulated closed-form orthogonal projection: $\Delta = \frac{(P k) r^T}{k^T P k}$ in float64 using Gram-Schmidt basis tracking $P = P_0 \cap (\text{span}(k_0, \dots, k_{t-1}))^\perp$.
- Relative residual gate $\|k \Delta - r\| / \|r\| \le 10^{-8}$ passed bit-for-bit (observed residuals $\approx 8 \times 10^{-17}$).
- All evaluated arms survived E0 capability ceiling (PPL $\le 72.06$) and passed E1 immediate efficacy gate ($\ge 90.00\%$).

### 11.5 Sequential Retention & Generalization Panel ($N=300$ retention, $N=900$ generalization across 6 seeds)
| Arm | Pooled Imm Efficacy (E1) | Terminal PPL (E0) | First-50 Retention (E2) | Diff vs Matched Floor (Newcombe 95% CI) | Paraphrase Gen (E3) | Diff vs Matched Floor (Newcombe 95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `A-cov_L1` | 1136/1200 (94.67%) | 36.46 – 36.77 | 34/300 (11.33%) | +11.33 pp [+7.98, +15.42] (ABOVE) | 35/900 (3.89%) | +2.78 pp [+1.36, +4.33] (ABOVE) |
| `A-null_L1_corr` | 1184/1200 (98.67%) | 39.42 – 44.35 | **139/300 (46.33%)** | **+46.33 pp [+40.63, +51.99] (ABOVE)** | 89/900 (9.89%) | +8.78 pp [+6.77, +10.96] (ABOVE) |
| `A-null_L1_loose` | 1175/1200 (97.92%) | 45.30 – 51.71 | 101/300 (33.67%) | +33.67 pp [+28.40, +39.19] (ABOVE) | 63/900 (7.00%) | +5.89 pp [+4.14, +7.81] (ABOVE) |
| `A-cov_L6` | 1198/1200 (99.83%) | 36.38 – 36.50 | 48/300 (16.00%) | +16.00 pp [+12.08, +20.57] (ABOVE) | 95/900 (10.56%) | +9.44 pp [+7.38, +11.68] (ABOVE) |
| `A-null_L6_corr` | 1192/1200 (99.33%) | 38.47 – 40.93 | **131/300 (43.67%)** | **+43.67 pp [+38.03, +49.32] (ABOVE)** | 93/900 (10.33%) | +9.22 pp [+7.18, +11.44] (ABOVE) |

- **Core Finding:** Corrected null-space projection (`A-null_L1_corr`) provides a **$4.1\times$ improvement** in sequential retention over covariance-weighted updates alone (46.33% vs 11.33% at L1) and a **$2.7\times$ improvement** at L6 (43.67% vs 16.00%), while holding perplexity tightly controlled (39.4–44.4 vs baseline 36.03).
- **Dynamic Pruning:** `A-sgd_L6` was cleanly pruned at the end of the budget ceiling without exceeding session limits.

---

## 12. Established State as of Directive S0-12 (Commit `17d81e3` / Canonical Run)

Directive S0-12 (incorporating Amendment 1) verified the first positive sequential retention result from S0-11, diagnosed cross-process divergence (Stage D), evaluated sequential negative controls including sham sequence controls, localized remaining loss via activation patching (Stage L), and evaluated full-prompt protection (Stage F).

### 12.1 Stage D: Cross-Process Divergence Diagnosis (Amendment 1 §B)
- **Seeds Evaluated:** Seeds 1, 3, and 5.
- **Diagnostics Completed:**
  - Cached-state hashes: Layer 1 Covariance $C$ (`cc6ecaf7...` vs `70bbda39...`) and Projector $P_0$ (`03f6561d...` vs `bb1412bb...`) diverged bitwise from S0-11.
  - Same-process repeat (Seed 3): Identical vectors, identical step counts, identical final weight SHA (`4ceba085d42d39f5...`).
  - Fresh-process repeat (Seed 3): Independent processes match in-process bit-for-bit.
  - Order replication: Execution following A-cov_L1 yielded 198/200 immediate efficacy and 14/50 retention (diverged from S0-11 196/200 and 13/50).
- **Classification:** `(iii) changed cached inputs`. The slight edit-count difference on Seed 3 (+2 immediate, +1 retention) is attributed to subtle numerical divergence in cached covariance inputs across runs; the execution harness itself is 100% deterministic.
- **Gate V1 Verdict:** Recorded as **FAILED** per Amendment 1 §A; S0-11 `A-null_L1_corr` numbers are superseded, and S0-12 Stage V is established as the measurement of record.

### 12.2 Sequential Negative Control Floors & Comparator Rule
- **Candidate Floors:**
  - `never_edited`: Canonical 0/300 (0.00%), Paraphrase 8/900 (0.89%)
  - `pre_edit_baseline`: Canonical 0/300 (0.00%), Paraphrase 8/900 (0.89%)
  - `sham_sequence` ($v^* = v_0$, Seed 0 scaled): Canonical 0/300 (0.00%), Paraphrase 6/900 (0.67%)
- **Selected Primary Floor:** Canonical **0/300 (0.00%)**, Paraphrase **8/900 (0.89%)** ($\max$ of candidate arms).
- **Secondary Floor (`wrong_target`):** Canonical 7/300, Paraphrase 13/900 for `A-null_L1_corr`; Canonical 19/300, Paraphrase 30/900 for `A-cov_L6`.

### 12.3 Stage V: Verified Positive Retention & Generalization
- **`A-null_L1_corr` (Measurement of Record across 6 Seeds $\times$ 200 Edits):**
  - **Immediate Efficacy:** 1186/1200 (98.83%).
  - **Primary Endpoint E2 (First-50 Terminal Retention):** **141/300 (47.00%)** vs primary floor **0/300 (0.00%)** $\rightarrow$ **Diff +47.00 pp [+41.28 pp, +52.65 pp]**, **VERDICT: ABOVE** ($p < 10^{-15}$).
  - **Secondary Endpoint E3 (Paraphrase Generalization):** **88/900 (9.78%)** vs primary floor **8/900 (0.89%)** $\rightarrow$ **Diff +8.89 pp [+6.92 pp, +11.05 pp]**, **VERDICT: ABOVE**.
  - **Object Recurrence Separation:** Recurring objects retained at **126/276 (45.65%)**, non-recurring objects retained at **15/24 (62.50%)**, establishing that retention is driven by true internal model modification rather than late-sequence lexical re-injection.
  - **Modal Collapse Audit:** Dominant prediction is `'Brasilia, where the'` at 2.33% ($\ll 50\%$), confirming healthy diversity.
- **`A-cov_L6`:**
  - **Immediate Efficacy:** 1198/1200 (99.83%).
  - **Primary Endpoint E2:** **50/300 (16.67%)** vs primary floor **0/300 (0.00%)** $\rightarrow$ **Diff +16.67 pp [+12.67 pp, +21.30 pp]**, **VERDICT: ABOVE**.
  - **Secondary Endpoint E3:** **96/900 (10.67%)** vs primary floor **8/900 (0.89%)** $\rightarrow$ **Diff +9.78 pp [+7.74 pp, +12.01 pp]**, **VERDICT: ABOVE**.
  - **Modal Collapse Audit:** Dominant prediction is `'Ottawa, Ottawa, and'` at 3.00%, confirming healthy diversity.

### 12.4 Stage L: Activation Patching Loss Localization & Stage F Gate
- **Lost Facts Analyzed:** 29 facts lost at sequence end on Seed 0.
- **Patching Recovery Outcomes:**
  - Condition (a) Subject last-token only: 0/29 (0.00%)
  - Condition (b) Non-subject prompt positions only: 0/29 (0.00%)
  - Condition (c) All prompt positions: 0/29 (0.00%)
  - Mean Residual Subject Key Drift: $0.0000 \times 10^0$.
- **Finding:** Patching early activations does not restore lost outputs once downstream representation weights drift.
- **Stage F Gate:** Non-subject implication rate was $0.00\% \le 10.0\%$, cleanly skipping Stage F (full-prompt protection) per pre-registered protocol rule.

---

## 13. Active Codebase Organization

- `experiments/run_s0_12.py`: Master orchestrator for Directive S0-12 (Stage D, sequential controls, Stage V verification, Stage L patching, Stage F gate).
- `experiments/s0_12_diagnosis.py`: Directive S0-12 Stage D divergence diagnostic engine.
- `experiments/s0_12_localization.py`: Directive S0-12 Stage L activation patching and FullPrompt tracker engine.
- `experiments/s0_11_constraints.py`: Directive S0-11 engine (covariance estimation, null-space tracking, corrected closed-form updates, sequential execution).
- `experiments/s0_11_diagnostic.py`: Directive S0-11 Stage N float64 diagnostic engine.
- `experiments/run_s0_11.py`: Master orchestrator for Directive S0-11.
- `experiments/s0_10_repair.py`: Directive S0-10 engine (restore verification, diagnostic, repaired closed-form, fullgrad SGD, capability/locality, sequential execution).
- `experiments/run_s0_10.py`: Master orchestrator for Directive S0-10.
- `experiments/s0_9_writability.py`: Directive S0-9 engine.
- `experiments/run_s0_9.py`: Master orchestrator for Directive S0-9.
- `experiments/s0_8_relocate.py`: Directive S0-8 engine.
- `experiments/run_s0_8.py`: Master orchestrator for Directive S0-8.
- `experiments/b1_inject.py`: Historical injection experiment engine (< 600 lines).
- `experiments/stats.py`: Statistical engine (MDE, Newcombe intervals, exact Student t, exact Wilcoxon with combinatorial floor).
- `experiments/metrics.py`: Mathematical metrics, Wilson intervals, and Provenance Guard 2.0 with closed `POPULATION_REGISTRY`.
- `experiments/data.py`: Synthetic facts generation, deterministic seed sampling, and perplexity evaluation.
- `tests/test_metrics.py`: Pre-flight unit test suite (139 tests) and AST literal scanner.
- `tools/make_report.py`: Single-artifact report generator and byte-verifier supporting S0-2 through S0-12.
- `reports/`: Validated markdown reports (`S0-2.md` through `S0-12.md`).
- `experiments/results/`: Machine-readable results JSON artifacts (`s0_5.json` through `s0_12.json`).
- `s0_12_stdout.txt`: Verbatim execution stdout log for S0-12 (10,304.36 s, Exit Code 0).
- `s0_11_stdout.txt`: Verbatim execution stdout log for S0-11.
- `s0_10_stdout.txt`: Verbatim execution stdout log for S0-10.
- **Do not touch AGNIS files** (`agnis*.py`, quarantined legacy attempt).




