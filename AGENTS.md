# AGENTS.md -- Standing Rules for Continual Learning Experiments

1. **Permanent Control Arm**:
   `FREEZE-AFTER-BASE` (zero parameter updates after the base training phase) MUST be included as a permanent standing control arm in EVERY continual-learning evaluation and table. Any mechanism that does not outperform doing nothing (`FREEZE-AFTER-BASE`) has not demonstrated continual learning.

2. **Decomposed Gap Reporting**:
   Always report the retention and acquisition gaps closed as separate percentages of their own available gaps:
   - $\text{Retention Gap Closed} = \Delta \text{BWT} / (\text{Offline BWT} - \text{Naive BWT})$
   - $\text{Acquisition Gap Closed} = \Delta \text{LA} / (\text{Offline LA} - \text{Naive LA})$
   A single "% of total gap" metric conceals whether a mechanism actually mitigates forgetting or alters task acquisition.

3. **Correction & Value Change Flags**:
   Any reported quantity that changes value between reports must be explicitly flagged as a **CORRECTION** detailing the prior value, the new value, and the exact cause (whether code diff, hyperparameter change, or definitional change). Never present a definitional change as a physical measurement.

4. **Base-Rate Enrichment & Significance Testing**:
   Any "X of Y failures involve Z" claim must be accompanied by the base rate of Z in the population and a statistical significance test (e.g., Fisher's exact test with Odds Ratio and 95% Confidence Interval). A raw proportion without a base rate is not evidence of a causal constraint. When the population base rate approaches 0% or 100%, binary contingency tests are undefined or uninformative, and a graded continuous predictor (e.g., logistic regression on continuous raw similarity) must be used instead.

5. **Matched Evaluation Contexts (Rule R11)**:
   A prediction may only be scored against measurements taken on the same dataset and the same evaluation protocol. The dataset name must be explicitly stated in every cell of the Empirical Measurement column.

6. **No Structurally Constant Metric (Rule R16)**:
   Before printing any derived metric, prove it can take at least two values. Any expression whose numerator and denominator are forced equal by construction, or whose inputs are identically zero by construction, is forbidden. Every derived metric must be accompanied by a printed line stating what input change would alter it. If no such change exists, delete the metric.

7. **Seed Before Construction (Rule R17)**:
   `torch.manual_seed(seed)` must execute before any module instantiation or random draw. Every stochastic arm runs over `SEEDS = [42,43,44,45,46]` and reports mean ± std. No single-draw number may appear in any table.

8. **One Classifier Family Per Comparison (Rule R18)**:
   Any table comparing arms, and any prediction of the form "arm A vs arm B," must hold the classifier fixed. If arms use different classifiers, split the table by classifier and report the cross-classifier difference separately, labelled as such.

9. **Paste-Only Documentation (Rule R19)**:
   Any table in walkthrough.md or RESULTS.md that asserts the existence, size, provenance, or execution status of a repository artifact must be a verbatim paste of a committed *_stdout.txt log, enclosed in a fenced code block, with the log filename stated immediately above it. Hand-authored or reformatted versions of such tables are prohibited. A table that cannot be pasted must be deleted.




