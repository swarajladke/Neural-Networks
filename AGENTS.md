# AGENTS.md — Research Engineering Protocol

**Scope:** every script, commit, and report in this repository that produces, transforms, or
cites a measured number.

**Read this before writing code, not after.** If a rule here conflicts with a task
instruction, stop and say so in prose. Do not resolve the conflict silently.

---

## 0. The prime directive

> **A run that cannot be recomputed from pinned inputs is not a result. A number that cannot
> name the artifact it came from is not a measurement. A gate that can be edited to pass is
> not a gate.**

Everything below is a mechanism for enforcing one of those three sentences.

**Mechanism over promise.** A rule an agent must remember is a rule that will be broken. A
rule encoded as a type constraint, an assertion, or a build-time scanner cannot be broken
without a crash. Whenever you have the choice, write the mechanism.

**Every rule in this file exists because it was violated and cost a run.** Appendix A lists
the specific incident behind each one. Read it — rules without causes get rationalized away.

---

## 1. Metrics contract

**1.1 No metric may return a bare number.** Every count-based metric returns a `Measurement`
(Appendix C.1) carrying numerator, denominator, the identity of the input set, and the
execution mode. A function that returns `100.0` is forbidden; it must return `(20, 20)` with
context. *(Enforces: Section 13, Rule 2, Rule 6 [R16])*

**1.2 The denominator must equal the population the claim is about.** If a claim is about
twenty facts, the denominator is twenty. A metric that evaluates one item and reports a
percentage is a defect, even when the percentage is arithmetically correct for that one item.
*(Enforces: Section 13, Rule 2, Rule 5 [R11])*

**1.3 Assert the denominator.** Where a denominator is known in advance, assert it. Where it
is a sum of parts, compute it as that sum, print the sum expanded (`20 + 20 + 20 + 20 = 80`),
and assert equality. Never write a denominator as a literal. *(Enforces: Section 13, Rule 6 [R16])*

**1.4 Impossible values halt the run.** `numerator > denominator`, a negative count, a rate
above 100%, or a denominator of zero must raise immediately and name the offending metric.
Never print an impossible value and continue. *(Enforces: Section 13, Rule 11 [R21])*

**1.5 Every metric needs a docstring stating what it counts and what would make it zero.** If
two metrics are similar, the docstring must state what distinguishes them. An undocumented
distinction between two metrics is an unmeasured distinction. *(Enforces: Section 13, Rule 6 [R16])*

**1.6 Metrics live in one module and import nothing from the experiment.** No model loading,
no I/O, no printing of results. A metric must be testable with a hand-built stub in
milliseconds and no accelerator.

---

## 2. Thresholds and gates

**2.1 A threshold is immutable.** Thresholds, tolerances, and reference values are read at
runtime from a content-addressed artifact — a git blob whose hash is verified in code, or a
file whose hash is verified in code. Verification failure halts the run. *(Enforces: Section 13, Rule 3, Rule 11 [R21])*

**2.2 Never change a threshold in the same change as observing a value.** If a gate fails, the
permitted actions are: fix the code, fix the data, or report the failure. Changing the
threshold is not among them. A commit that edits a threshold and reports a pass is invalid on
its face, regardless of its message. *(Enforces: Section 13, Rule 3)*

**2.3 The word "calibrate" is banned from commit messages** touching any threshold, tolerance,
reference value, or gate. If you believe a threshold is genuinely wrong, say so in prose, let
the gate fail, and stop.

**2.4 A failing gate is a valid and useful outcome.** It is the instrument working. Your job
is to report it, not to remove it. Reporting a failure costs nothing; defeating a safeguard
costs every run that follows. *(Enforces: Section 13, Rule 11 [R21])*

**2.5 Tolerance rules are expressed as code, not stored as intervals.** Store the reference
value; compute the interval from a rule written in source, and print the rule's text beside
the interval it produced. A widening then appears as a source diff instead of a data edit.
*(Enforces: Section 13, Rule 6 [R16])*

**2.6 A failed gate blocks everything downstream of it.** If a positive control fails, no
subsequent measurement may be printed, and no verdict may be published. Exit nonzero at the
failure point. *(Enforces: Section 13, Rule 11 [R21])*

**2.7 Cite gates and rules by their actual identifiers.** Before citing a rule number, open
the file and confirm it exists. Do not cite a rule you have not read.

---

## 3. Provenance

**3.1 Every reported number is interpolated at runtime from a results object.** No measured
value may be typed into source, into a print string, into a report, or into a comment that a
reader could mistake for a measurement. *(Enforces: Section 13, Rule 9 [R19], Rule 10 [R20])*

**3.2 That includes the literal segments of f-strings.** `print(f"  modal was 'oslo'")` is a
typed measurement. The build-time scanner in Appendix C.2 descends into f-strings for exactly
this reason. *(Enforces: Section 13, Rule 9 [R19], Rule 10 [R20])*

**3.3 Every results artifact records, at minimum:** the producing commit SHA, a content hash
for every input dataset, the resolved version and hash of every model or checkpoint, the
environment fingerprint, total optimizer steps, and total samples seen. **The run fails if the
step count or the sample count is zero.** *(Enforces: Section 13, Rule 3, Rule 11 [R21])*

**3.4 A number with no source does not exist.** Before using a value as a reference, locate it
in a committed artifact. If `git grep` finds it only in the file that consumes it, it has no
provenance and must not be used. Say so and stop. *(Enforces: Section 13, Rule 5 [R11], Rule 9 [R19])*

**3.5 Every result carries the identity of the input set that produced it.** Two runs over
different subsets are different cells and may never be compared as though they were one. Print
the input-set name, its size, and its hash beside every result. *(Enforces: Section 13, Rule 5 [R11])*

**3.6 Results are immutable once recorded.** A superseded or withdrawn result is marked
withdrawn in place, with a reason and a superseding artifact. It is never silently left in a
live file, and never quietly overwritten. *(Enforces: Section 13, Rule 3)*

---

## 4. No carry-over, no fabrication

**4.1 A quantity not measured in this run must not be printed in this run.** Not as "carried
over", not as "expected", not as context, not greyed out. If a report needs it, the report
cites the earlier artifact by hash. The current run's stdout contains only what the current
run computed. *(Enforces: Section 13, Rule 9 [R19], Rule 10 [R20])*

**4.2 Never state an expected result in source.** A comment or print reading "expected to be
around 2.4" pre-commits the run to its conclusion and will be read as corroboration.
Predictions belong in the directive or the prose, never in the script.

**4.3 Verdicts are pure functions of the results object.** Build the verdict string by format
string from computed values, at the end, after the values exist. Never write a headline before
a run and never leave one in place after. *(Enforces: Section 13, Rule 11 [R21])*

**4.4 A run must touch its data and take its steps.** Assert that samples seen and optimizer
steps are nonzero. Assert the dataset hash. A script that produces a results table without
loading data or computing a gradient is fabrication, whatever its exit code. *(Enforces: Section 13, Rule 11 [R21])*

**4.5 Exit code 0 is not evidence of validity.** It means nothing crashed. Validity comes from
recomputability: pinned inputs, recorded hashes, and a test suite that fails when a behaviour
changes. *(Enforces: Section 13, Rule 11 [R21])*

---

## 5. Determinism and environment

**5.1 Pin everything that can move.** Model revisions by immutable commit hash, never by a
branch name such as `main`. Record the resolved snapshot path and the weight file hash.

**5.2 Print the environment fingerprint on every run:** framework versions, CUDA and cuDNN
versions, accelerator name, kernel and backend flags, deterministic-algorithm flags, seeds,
and a fresh-load parameter checksum.

**5.3 Declare the execution mode of every measurement,** including training-versus-inference
mode where the model contains dropout, normalization, or any stochastic layer. A quantity
measured with dropout active and the same quantity measured with it inactive are **different
estimands**. Label both. Never compare them, and never compare either to an unlabelled
historical value. *(Enforces: Section 13, Rule 5 [R11])*

**5.4 Mode consistency is asserted, not assumed.** Thread the expected mode through as an
argument and assert the module's flag matches it. Do not hardcode an assertion that a flag is
always one value; diagnostics legitimately need both.

**5.5 Repeat counts follow the noise.** A deterministic configuration is repeated to *prove*
determinism, asserted identical, and its zero spread is reported as determinism — never as
low variance. A stochastic configuration is repeated to *estimate* a distribution, and its
per-repeat values are all printed. *(Enforces: Section 13, Rule 7 [R17])*

**5.6 Never derive a tolerance from a zero spread.** A zero-width band is unfalsifiable and
will fail on any environment change, which invites exactly the threshold editing that §2
forbids.

---

## 6. Data pinning

**6.1 Experimental data is loaded from a hashed file on disk, never generated at runtime.**
Assert the hash before use. *(Enforces: Section 13, Rule 5 [R11])*

**6.2 If a generator exists, prove it agrees with the pinned file.** Regenerate at startup and
assert field-by-field equality across every record and every key. A generator that has drifted
while the pinned file is loaded is a live landmine: it will be used the day someone regenerates.

**6.3 Hash the bytes, and say which hash you mean.** A hash of the file bytes and a hash of a
canonical re-serialization are different values. Record which one you are reporting. Two
"unchanged hashes" computed by different functions have told us nothing.

**6.4 Assert the shape assumptions your metrics rely on.** If a normalizer keeps only the first
whitespace-delimited token, assert no target contains whitespace. Check for tabs and newlines,
not just spaces. Every silent truncation is a mis-scored result.

**6.5 Control and probe sets are pinned and hashed like everything else.** A control floor
computed from unpinned data is not a floor. *(Enforces: Section 13, Rule 4)*

---

## 7. Structural limits

**7.1 One experiment, one file, under 600 lines.** Beyond that, no reviewer and no agent holds
it in working memory, and defects survive for months. The evidence is in this repository.

**7.2 Separate the four concerns into separate modules:** metric definitions, experiment
procedure, reporting, and tests. Metrics must not import the experiment. *(Enforces: Section 1.6)*

**7.3 Do not grow a script that has failed an audit.** Build the replacement beside it, port
only components that are individually proven, and leave the original untouched as a record.

**7.4 Retire code by moving it out of the experiment path,** not by leaving it importable. A
function that fabricates results, takes no gradient step, or loads no data must be unreachable
from any experiment entry point.

**7.5 Reuse proven components verbatim.** When porting a component that has passed an explicit
test suite, copy it unchanged and comment the commit it came from. Rewriting a proven function
re-opens a closed question.

---

## 8. Tests before compute

**8.1 The test suite runs before any model loads, and a failure exits nonzero.** No accelerator
time is spent on an unverified instrument. *(Enforces: Section 13, Rule 11 [R21])*

**8.2 Test behaviours, not outcomes.** Any rule the experiment depends on — a stopping
condition, a matching rule, a normalization, a selection rule — needs a test against
hand-constructed inputs with the answer known by hand. Inferring that a rule works because the
output looked plausible is how a silently changed stopping condition produced a full run of
zeros that every other check passed.

**8.3 Write a regression test for every defect you fix,** encoding the specific wrong
behaviour. A fixed defect without a test is a defect scheduled to recur.

**8.4 Test the guards themselves.** Feed the impossible-value guard an impossible value and
assert it raises. Feed the literal scanner a violating line and assert it fails. An untested
guard is decoration.

**8.5 Zero tests run is a failure,** not a pass. Print the count run and the count passed.

**8.6 Project the compute budget before requesting it.** Print projected wall-clock per stage
against the session limit, and abort if the projection exceeds the ceiling.

---

## 9. Reporting contract

**9.1 stdout is measurement. Prose is changelog.** stdout contains fingerprints, tests,
per-repeat values, and computed verdicts. Prose outside stdout states what you changed, which
file and line, and why — nothing else. *(Enforces: Section 13, Rule 9 [R19], Rule 10 [R20])*

**9.2 No executive summary, no key-outcomes list, and no interpretation** unless the directive
explicitly asks for it. A summary written alongside the tables is where headlines drift free
of their evidence.

**9.3 Print per-repeat and per-seed values, always.** Summary statistics are printed in
addition to the raw values, never instead of them. A mean whose inputs are hidden cannot be
checked. *(Enforces: Section 13, Rule 7 [R17])*

**9.4 Print every gate, including the ones that fail,** with observed value, reference value,
reference source hash, the tolerance rule text, the interval, the signed deviation, and the
outcome. Print all of them even when an early one fails. *(Enforces: Section 13, Rule 11 [R21])*

**9.5 A claim is reported against its own control floor,** never against zero, and never
against a pooled floor alone. Always print the single worst individual control beside any
pooled figure, because pooling hides the one control that is beating the experiment. *(Enforces: Section 13, Rule 4)*

**9.6 Reconcile any quantity that appears twice.** If one configuration is evaluated in two
places, compute it once and reuse it, or print both with their input-set identities and
reconcile the difference explicitly. Two unreconciled values for one cell invalidate both. *(Enforces: Section 13, Rule 5 [R11])*

**9.7 State the limits of what was measured.** If a gate was skipped, a control was not run, or
a sample was small, say so in the same breath as the result.

---

## 10. Failure protocol

When a check, gate, assertion, or test fails, do these in order, and nothing else:

1. **Stop.** Do not proceed to downstream stages.
2. **Print the full diagnostic**: observed value, reference, source hash, rule, interval,
   deviation.
3. **Exit nonzero.** *(Enforces: Section 13, Rule 11 [R21])*
4. **Report the failure in prose, plainly, without softening.**
5. **Diagnose before you touch anything.** Ask what real defect would produce exactly this
   deviation. Check execution mode, input-set identity, data hashes, and version drift *before*
   concluding the check is wrong.
6. **If you conclude the check itself is wrong, say so and stop.** Do not act on that
   conclusion. A human decides.

**Forbidden responses to a failure:** widening an interval; substituting the observed value as
the new reference; switching to a different input set; relaxing a comparison; re-seeding until
it passes; adding an entry to an allow-list; disabling the check; downgrading it to a warning;
or proceeding while printing the failure.

**A failing check is information you were about to lose.** Every one of those responses
destroys it.

---

## 11. Ablation and intervention validity

**11.1 Prove an ablation changed something.** Before interpreting its result, assert the
parameters it targets have a nonzero delta from the pre-intervention state. Resetting
parameters that never moved is a no-op, and its "result" is entailed by the setup.
*(Enforces: Section 13, Rule 6 [R16])*

**11.2 Report the count of distinct states an ablation produced.** If seven conditions yield
two distinct outcomes, say so. Conditions that are identical by construction carry no
information and must not be presented as a partition. *(Enforces: Section 13, Rule 6 [R16])*

**11.3 Distinguish entailed from measured.** If a result follows necessarily from the
configuration — because a component was frozen, or a term was zero — label it as a sanity
check. It is not a finding. *(Enforces: Section 13, Rule 1, Rule 6 [R16])*

**11.4 State non-additivity when partitions overlap.** If components sum past 100%, print the
sum and say the partition is non-additive. Do not present overlapping parts as a decomposition.

**11.5 An intervention that failed measures nothing.** No quality, retention, or downstream
number may be reported from a condition whose intervention did not take effect at the stated
rate. Check efficacy first, as a properly denominated rate over the full population, and gate
on it. *(Enforces: Section 1.2, Section 13, Rule 2)*

---

## 12. Pre-commit checklist

Paste this completed into every report. An unchecked box is a blocker, not a note. Every item must be marked with exactly one of three states, and nothing else:
  `[x]`  performed and evidenced in this report
  `[ ]`  required but not done — this is a blocker, state why
  `[N/A — not performed this run]`  the run did not include this activity
An item marked `[x]` must have its evidence locatable in this report. Ticking an item whose evidence is absent is a protocol violation in itself, and is treated as more serious than leaving it unticked.

```
[ ] Report generated by tools/make_report.py, not hand-authored
[ ] Report regeneration verified: regenerated output is byte-identical to the committed file
[ ] Tests ran before any model load; N run, N passed, zero failures
[ ] Every count-based metric returned an explicit numerator/denominator pair
[ ] Every denominator asserted or printed as an expanded sum
[ ] No numerator exceeds its denominator anywhere in output
[ ] No threshold, tolerance, or reference value edited in this change
[ ] All reference values read at runtime from a hash-verified artifact
[ ] AST literal scanner passed; allow-list printed with per-entry justification
[ ] No measured value typed in source, including inside f-string literal segments
[ ] No quantity printed that this run did not compute
[ ] No expected result stated anywhere in source
[ ] Input hashes asserted: dataset, controls, capability slice
[ ] Generator regenerated and asserted field-by-field equal to the pinned file
[ ] Model pinned by immutable revision; weight hash recorded
[ ] Environment fingerprint printed
[ ] Execution mode declared for every measurement
[ ] Per-repeat and per-seed values printed, not only summaries
[ ] Optimizer steps > 0 and samples seen > 0, asserted
[ ] Every gate printed with observed, reference, source hash, rule, interval, deviation
[ ] Worst individual control printed beside every pooled floor
[ ] Every ablation shown to have a nonzero parameter delta
[ ] Any quantity appearing twice computed once, or reconciled explicitly
[ ] Verdict strings generated from the results object by format string
[ ] Exit code recorded; failing gates reported, not removed
```

---

## 13. Domain Policy — Preserved Standing Rules for Continual Learning Experiments

*All eleven original rules from `AGENTS.md` (v1, 2026-08-16) are preserved below verbatim with dual numbering intact. They form the research-policy layer. The mechanisms in Sections 1–12 enforce this layer.*

1. **Permanent Control Arm**:
   `FREEZE-AFTER-BASE` (zero parameter updates after the base training phase) MUST be included as a permanent standing control arm in EVERY continual-learning evaluation and table. Any mechanism that does not outperform doing nothing (`FREEZE-AFTER-BASE`) has not demonstrated continual learning.
   *(Cross-reference: Enforced by Section 11.3, Section 11.5; Conflict documented in Section 16.1)*

2. **Decomposed Gap Reporting**:
   Always report the retention and acquisition gaps closed as separate percentages of their own available gaps:
   - $\text{Retention Gap Closed} = \Delta \text{BWT} / (\text{Offline BWT} - \text{Naive BWT})$
   - $\text{Acquisition Gap Closed} = \Delta \text{LA} / (\text{Offline LA} - \text{Naive LA})$
   A single "% of total gap" metric conceals whether a mechanism actually mitigates forgetting or alters task acquisition.
   *(Cross-reference: Enforced by Section 1.1, Section 1.2, Section 11.5)*

3. **Correction & Value Change Flags**:
   Any reported quantity that changes value between reports must be explicitly flagged as a **CORRECTION** detailing the prior value, the new value, and the exact cause (whether code diff, hyperparameter change, or definitional change). Never present a definitional change as a physical measurement.
   *(Cross-reference: Enforced by Section 2.1, Section 2.2, Section 3.3, Section 3.6)*

4. **Base-Rate Enrichment & Significance Testing**:
   Any "X of Y failures involve Z" claim must be accompanied by the base rate of Z in the population and a statistical significance test (e.g., Fisher's exact test with Odds Ratio and 95% Confidence Interval). A raw proportion without a base rate is not evidence of a causal constraint. When the population base rate approaches 0% or 100%, binary contingency tests are undefined or uninformative, and a graded continuous predictor (e.g., logistic regression on continuous raw similarity) must be used instead.
   *(Cross-reference: Enforced by Section 6.5, Section 9.5)*

5. **Matched Evaluation Contexts (Rule R11)**:
   A prediction may only be scored against measurements taken on the same dataset and the same evaluation protocol. The dataset name must be explicitly stated in every cell of the Empirical Measurement column.
   *(Cross-reference: Enforced by Section 1.2, Section 3.4, Section 3.5, Section 5.3, Section 6.1, Section 9.6)*

6. **No Structurally Constant Metric (Rule R16)**:
   Before printing any derived metric, prove it can take at least two values. Any expression whose numerator and denominator are forced equal by construction, or whose inputs are identically zero by construction, is forbidden. Every derived metric must be accompanied by a printed line stating what input change would alter it. If no such change exists, delete the metric.
   *(Cross-reference: Enforced by Section 1.1, Section 1.3, Section 1.5, Section 2.5, Section 11.1, Section 11.2, Section 11.3)*

7. **Seed Before Construction (Rule R17)**:
   `torch.manual_seed(seed)` must execute before any module instantiation or random draw. Every stochastic arm runs over `SEEDS = [42,43,44,45,46]` and reports mean ± std. No single-draw number may appear in any table.
   *(Cross-reference: Enforced by Section 5.5, Section 9.3; Conflict documented in Section 16.2)*

8. **One Classifier Family Per Comparison (Rule R18)**:
   Any table comparing arms, and any prediction of the form "arm A vs arm B," must hold the classifier fixed. If arms use different classifiers, split the table by classifier and report the cross-classifier difference separately, labelled as such.
   *(Cross-reference: Enforced by Section 3.5, Section 9.6)*

9. **Paste-Only Documentation (Rule R19)**:
   Any table in walkthrough.md or RESULTS.md that asserts the existence, size, provenance, or execution status of a repository artifact must be a verbatim paste of a committed *_stdout.txt log, enclosed in a fenced code block, with the log filename stated immediately above it. Hand-authored or reformatted versions of such tables are prohibited. A table that cannot be pasted must be deleted.
   *(Cross-reference: Enforced by Section 3.1, Section 3.2, Section 4.1, Section 9.1, verify_all_numbers.py)*

10. **Paste-Only Counts (Rule R20)**:
    Any count, tally, pass/fail summary, grep result, or reconciliation figure produced by a repository script must appear in documentation only as a verbatim paste of that script's committed `*_stdout.txt`, inside a fenced code block, with the log filename and its commit SHA stated immediately above the block. Prose restatement, reformatting into a bullet list, or transcription into a table is prohibited. If a count cannot be pasted, the section reporting it must be deleted.
    *(Cross-reference: Enforced by Section 3.1, Section 3.2, Section 4.1, Section 9.1, run_p7_strict_citation_audit.py)*

11. **Exit-Code Integrity (Rule R21)**:
    Any script that prints a violation, illegal value, mismatch, or failure condition must terminate with a non-zero exit status. A guard that prints a violation and exits zero is treated as a failed guard, and every number it certifies is treated as unverified. Every pasted guard output must be immediately followed by the line `EXIT_CODE = <n>` printed by the script itself, and no PASSED status may be claimed for a run whose printed exit code is non-zero or whose violation lists are non-empty.
    *(Cross-reference: Enforced by Section 1.4, Section 2.1, Section 2.4, Section 2.6, Section 3.3, Section 4.3, Section 4.4, Section 4.5, Section 8.1, Section 9.4, Section 10.3)*

---

## 14. Single-Artifact Reporting

**14.1 One run produces exactly one report file at `reports/<DIRECTIVE_ID>.md`.** It is self-contained: a reader with no access to the repository, the chat history, or any other file can audit it completely.

**14.2 Section order is fixed.** Do not add, remove, or reorder sections. Where a section does not apply, include it with the single line `[N/A — not performed this run]`.
1. Run header
2. What changed
3. Input fingerprints
4. Environment fingerprint
5. Test suite result
6. Measurements
7. Comparisons and observations
8. Pre-commit checklist
9. Complete stdout
10. Artifacts written

**14.3 Contents of each section:**
- **1. Run header**: directive identifier, producing commit SHA, platform, wall-clock seconds, exit code. Five lines, plain text, no table.
- **2. What changed**: prose only: which file, which lines, and why. No measurements in this section at all. No summary of results. No conclusions. If the results JSON lacks required keys, list each gap explicitly as `[MISSING FROM ARTIFACT — key <name> absent from results JSON]`.
- **3. Input fingerprints**: every input the run consumed, with its SHA-256, its record count, and whether the hash was asserted. Include the model revision and weight hash.
- **4. Environment fingerprint**: framework versions, accelerator, CUDA and cuDNN, kernel and backend flags, deterministic-algorithm flags, seeds, fresh-load parameter checksum.
- **5. Test suite result**: tests run, tests passed, failures, and whether the suite ran before any model was loaded. Zero tests run is reported as a failure.
- **6. Measurements**: plain pipe tables. Every count-based value appears as `numerator/denominator (percentage)`, never as a bare percentage. Every table states the input-set name and the execution mode in its caption. Per-repeat rows come first; summary statistics come after, and every summary must be reducible from the rows printed above it.
- **7. Comparisons and observations**: any comparison against a historical or reference value, with the reference source hash. Labelled as observation. No pass, fail, reproduced, matched, or certified language unless a gate was formally defined and evaluated.
- **8. Pre-commit checklist**: the Section 12 checklist verbatim, every item marked with exactly one of three states (`[x]`, `[ ]`, `[N/A — not performed this run]`). An item marked `[x]` must have its evidence locatable in this report.
- **9. Complete stdout**: the entire committed log inside a 5-backtick block (` ````` `), unedited except as permitted by §14.5. State the log filename and its commit SHA on the line immediately above the fence.
- **10. Artifacts written**: every file the run created or modified, with its path and SHA-256.

**14.4 Formatting prohibitions.** The report is plain Markdown intended to survive being copied as text. The generator must not emit any of the following, and must fail if asked to:
- diagrams of any kind, including mermaid
- LaTeX or math delimiters, including inline math wrapped in backticks
- raw HTML, including collapsible sections and styled callouts
- emoji, and admonition syntax such as bracketed IMPORTANT or NOTE markers
- hyperlinks of any kind. Commit SHAs, file paths, and hashes appear as plain text only. Never emit a local filesystem path as a link.
- images, badges, or embedded media
- an executive summary, a key-outcomes list, a headline, a verdict, or any interpretation of what the numbers mean. Sections 1 through 10 are the whole report.

**14.5 Log reduction is permitted in exactly one narrow form, or not at all.** Progress bars and model-loader banners may be removed if and only if the report states the exact regular expression used and the count of lines removed, immediately above the stdout fence. Nothing else may be filtered, summarized, reordered, or reformatted. If in doubt, include everything.

**14.6 Nesting discipline.** The stdout block uses 5 backticks (` ````` `). Any code shown inside sections 1 through 8 uses 3 backticks (` ``` `). A fence must always be longer than the longest backtick run it contains. After generating the report, the generator must verify that every fence it opened is closed and that no inner backtick run equals or exceeds its enclosing fence length, failing if not.

**14.7 Chat delivery.** When the report is handed to a reviewer, the entire file is pasted, with nothing added, removed, summarized, or rephrased, and no accompanying commentary beyond a single line naming the directive and the commit SHA. The file is the message.

**14.8 Research progress tracking (`context.md`).** After every successful run, report generation, and verification, update `context.md` in the repository root to record current research progress, newly established empirical findings, benchmark tables, producing commit SHA, and next planned directives. This ensures any subsequent agent or collaborator has an immediate, up-to-date briefing of the cumulative state of the research program.

---

## 15. Compliance Matrix (Rules 1 to 11)

| Rule # | Rule Name | Dual Identifier | Current Enforcement Status | Enforcement Implementation & Gaps |
|:---|:---|:---|:---|:---|
| **Rule 1** | Permanent Control Arm (`FREEZE-AFTER-BASE`) | — | **Conflict / Unenforced** | Mandates `FREEZE-AFTER-BASE` in every table, but the arm was withdrawn due to BatchNorm leakage. No CI script checks table inclusion. (Conflict 16.1). |
| **Rule 2** | Decomposed Gap Reporting | — | **Prose only** | Mathematical definitions given in prose; no automated parser enforces dual-gap reporting across experiment tables. |
| **Rule 3** | Correction & Value Change Flags | — | **Prose only** | Tracked manually in ledgers/markdown; no automated diff-scanner flags undocumented inter-report numerical shifts. |
| **Rule 4** | Base-Rate Enrichment & Significance Testing | — | **Prose only** | Causal tests (Fisher's exact test, Wilson score) are implemented ad-hoc; no CI linter enforces base rates on causal claims. |
| **Rule 5** | Matched Evaluation Contexts | Rule R11 | **Executable mechanism (partial)** | Pinned dataset hashes (`b1_facts.json`) and distinct input set checks are enforced at script startup; cross-table context matching remains prose-audited. |
| **Rule 6** | No Structurally Constant Metric | Rule R16 | **Prose only** | Conceptual guideline; partially operationalized by ablation delta check (§11.1), but no general AST check for metric non-constancy exists. |
| **Rule 7** | Seed Before Construction | Rule R17 | **Conflict / Partial Executable** | `torch.manual_seed()` is enforced at startup; but 5-seed requirement `[42,43,44,45,46]` conflicts with recent 3-seed / single-draw practice (Conflict 16.2). |
| **Rule 8** | One Classifier Family Per Comparison | Rule R18 | **Prose only** | Followed during table construction; no code linter enforces classifier consistency. |
| **Rule 9** | Paste-Only Documentation | Rule R19 | **Executable mechanism** | Programmatically verified by `verify_all_numbers.py` (matching numbers against committed `*_stdout.txt` logs). |
| **Rule 10** | Paste-Only Counts | Rule R20 | **Executable mechanism** | Programmatically verified by `run_p7_strict_citation_audit.py` and `verify_all_numbers.py`. |
| **Rule 11** | Exit-Code Integrity | Rule R21 | **Executable mechanism** | Enforced by audit scripts, test suites, and git execution checks (`EXIT_CODE = <n>` assertion and nonzero exit on violation). |

---

## 16. Outstanding Rule Conflicts (Awaiting Human Ruling)

Per Directive P-1 Step 3, these two conflicts are identified in the record and must not be resolved unilaterally by an agent.

### 16.1 Conflict A: Rule 1 vs. Withdrawn Control Arm
- **The Rule**: Rule 1 states: "`FREEZE-AFTER-BASE` (zero parameter updates after the base training phase) MUST be included as a permanent standing control arm in EVERY continual-learning evaluation and table."
- **The Conflict**: The `FREEZE-AFTER-BASE` control arm was formally withdrawn in earlier project phases due to a detected BatchNorm leakage defect. Mandating a permanent control arm that is itself invalidated and withdrawn leaves every subsequent evaluation and table structurally non-compliant with Rule 1.
- **Action Required**: A human ruling must determine whether Rule 1 is to be amended, replaced with a repaired leak-free control arm, or formally retired.

### 16.2 Conflict B: Rule 7 (R17) vs. Experimental Seed Practice
- **The Rule**: Rule 7 (R17) states: "`torch.manual_seed(seed)` must execute before any module instantiation or random draw. Every stochastic arm runs over `SEEDS = [42,43,44,45,46]` and reports mean ± std. No single-draw number may appear in any table."
- **The Conflict**: Recent experimental runs (e.g. B1-1D, B1-1E, B1-1G-REV) evaluated multi-ordering stability across only three seeds (`[42, 43, 44]`) or ran single-seed diagnostic probes (e.g., seed 42), printing single-draw numbers in reference and diagnostic tables.
- **Action Required**: A human ruling must specify which is to change: the rule (e.g., adjusting required seed arrays per experiment tier) or the practice (strictly requiring all 5 seeds `[42..46]` and banning all single-draw numbers). The rule is not amended here.

---

## Appendix A — Incident catalogue

Each rule above was written after a specific failure. These are the failures.

| # | Incident | Rule |
|---|---|---|
| 1 | A metric evaluated **one** item and reported 100%; it was read as a rate over twenty for five consecutive reports, and the gate depending on it was never actually evaluated | §1.1, §1.2 |
| 2 | A reference file was edited across **four commits**, each substituting an observed value or widening an interval, until the positive control passed. Commit messages read "CALIBRATE" | §2.1–§2.5 |
| 3 | Two reference values used as gate tolerances existed **nowhere in the repository** except the file consuming them | §3.4 |
| 4 | A gradient-share figure was printed with **neither** input measured in that run; it matched a voided run's stored value to four significant figures | §4.1 |
| 5 | Measurements were typed inside **f-string** literal segments, so a scanner checking only plain constants reported zero hits | §3.2, §C.2 |
| 6 | Source printed "expected around ~2.4" before the run; the run then reported 2.39 and it read as confirmation | §4.2 |
| 7 | A pooled denominator was hardcoded to 120 while its numerator summed four groups of 20; the floor every claim was judged against was wrong by half | §1.3 |
| 8 | A control rate of **450%** was printed and execution continued | §1.4 |
| 9 | A positive control ran on a different 20-item subset than the experiment; the two were compared as one cell. The control's subset had only 16 distinct targets, so a collapsed model scored 15% for free | §3.5, §1.2 |
| 10 | A seven-condition ablation collapsed to **two** distinct states because the parameters being reset had been frozen and never moved. "Prediction held" was a tautology | §11.1–§11.3 |
| 11 | A stopping rule was changed silently; every cell reported 0% efficacy; the null was published as a refutation. Every other check passed | §8.2 |
| 12 | A data generator had drifted on 250 of 1,000 records, masked because the experiment loaded a pinned file. Two bugs, not one: a missing field override and a transposed tuple unpack | §6.2 |
| 13 | A model was loaded from a floating branch name, so "immutable reference" depended on an upstream tag | §5.1 |
| 14 | Four scripts generated accuracy matrices from hardcoded constants, took no gradient step, and loaded no dataset. 27 fabricated numbers were certified as measured by the existing audit apparatus | §4.4, §4.5 |
| 15 | A reference cell was assembled from two different sources: retention from one subset, perplexity from an unrelated ablation condition. No measured cell ever had both | §3.5, §9.6 |
| 16 | The same configuration was evaluated twice in one run, reporting 11/20 and 5/20, unreconciled | §9.6 |
| 17 | Gates 3 and 6 read FAIL and downstream verdicts were published anyway, under a summary declaring the opposite | §2.6, §4.3 |
| 18 | A gate was satisfied by a value carried from an earlier commit and labelled "not measured in this run" | §4.1 |
| 19 | Only summary statistics were printed; the mean and the determinism claim were both unverifiable | §9.3 |
| 20 | A report certified compliance with rules R19–R21 of a file containing 11 rules | §2.7 |
| 21 | A quantity measured with dropout active was compared to the same quantity measured with dropout inactive, and a 21% difference was diagnosed as nondeterminism | §5.3 |
| 22 | A withdrawn headline remained in the live results artifact after its own withdrawal registry declared it retired | §3.6 |

---

## Appendix B — Why the previous protocol failed

The prior protocol had eleven rules and two audit scripts. They verified that numbers in
Markdown matched numbers in stdout — **transcription fidelity**. They could not detect a script
that never touched its dataset, and they treated a clean exit as evidence of validity. This was
demonstrated: 27 deliberately fabricated numbers passed every check and were certified as
measured.

The lesson is not "add more rules." It is that **the auditable surface was the wrong one.**
Transcription is the last and least likely place for a research defect. The defects live in
metric definitions, gate thresholds, input identity, and execution mode — upstream of anything
a transcription audit can see.

So this protocol audits four things instead:

1. **Can it be recomputed?** Pinned inputs, recorded hashes, pinned revisions.
2. **Is the metric what it claims?** Explicit denominators, tested against hand-built stubs.
3. **Could the gate have been moved?** Content-addressed thresholds, rules as code.
4. **Did the code touch the data?** Nonzero steps, nonzero samples, asserted hashes.

---

## Appendix C — Reference implementations

Copy these. Do not reimplement them.

### C.1 A type that makes a bare percentage unrepresentable

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Measurement:
    """A count-based measurement that cannot be reported without its denominator.

    Enforces the metrics contract: every rate carries the population it was computed
    over, the identity of the input set, and the execution mode. Impossible values
    raise at construction rather than being printed.
    """
    name: str
    numerator: float
    denominator: float
    input_set: str           # e.g. "distinct20_seed42"
    mode: str                # e.g. "eval_no_dropout"

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError(f"{self.name}: denominator must be positive, got {self.denominator}")
        if self.numerator < 0:
            raise ValueError(f"{self.name}: negative numerator {self.numerator}")
        if self.numerator > self.denominator:
            raise ValueError(
                f"{self.name}: numerator {self.numerator} exceeds denominator "
                f"{self.denominator} -- impossible value, halting"
            )

    @property
    def pct(self) -> float:
        return 100.0 * self.numerator / self.denominator

    def __str__(self) -> str:
        return (f"{self.name}: {self.numerator:g}/{self.denominator:g} "
                f"({self.pct:.2f}%) [set={self.input_set}, mode={self.mode}]")


def pooled(name: str, parts: list[Measurement]) -> Measurement:
    """Pool measurements, printing the denominator as an expanded sum.

    Never accept a hardcoded pooled denominator: it must equal the sum of parts.
    """
    if not parts:
        raise ValueError(f"{name}: cannot pool zero parts")
    num = sum(p.numerator for p in parts)
    den = sum(p.denominator for p in parts)
    print(f"  {name}: "
          f"{' + '.join(f'{p.numerator:g}' for p in parts)} = {num:g}  over  "
          f"{' + '.join(f'{p.denominator:g}' for p in parts)} = {den:g}")
    worst = max(parts, key=lambda p: p.pct)
    print(f"  {name}: worst individual part -> {worst}")
    return Measurement(name, num, den, parts[0].input_set, parts[0].mode)
```

### C.2 A literal scanner that descends into f-strings

```python
import ast
import re
import sys

NUMERIC = re.compile(r"\d+\.\d+|\d+\s*%|%\s*\d+")

# Formatting constants only. Never put a domain quantity here.
# Each entry needs a one-line justification printed at runtime.
ALLOW_LIST = {
    "=" * 115: "table rule",
    "-" * 115: "table rule",
}


def _format_spec_node_ids(call: ast.Call) -> set[int]:
    """Ids of nodes inside f-string format specs, e.g. the '>6.2f' in {x:>6.2f}.

    These legitimately contain decimals and are not measurements.
    """
    ids: set[int] = set()
    for sub in ast.walk(call):
        if isinstance(sub, ast.FormattedValue) and sub.format_spec is not None:
            for n in ast.walk(sub.format_spec):
                ids.add(id(n))
    return ids


def scan_for_typed_literals(path: str) -> list[tuple[int, str]]:
    """Report numeric literals reaching a print call, including inside f-strings.

    ast.walk over the Call descends into JoinedStr.values, so an f-string's literal
    segments are inspected. A scanner that checks only bare Constant arguments will
    miss print(f"modal was 'oslo' at 8.7%") entirely -- which is the exact defect
    this exists to catch.
    """
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        skip = _format_spec_node_ids(node)
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and id(sub) not in skip):
                text = sub.value
                if NUMERIC.search(text) and text not in ALLOW_LIST:
                    hits.append((getattr(sub, "lineno", -1), text))
    return hits


def enforce_no_typed_literals(path: str) -> None:
    print("  [Literal scanner] allow-list:")
    for entry, why in ALLOW_LIST.items():
        print(f"    {entry[:24]!r}: {why}")
    hits = scan_for_typed_literals(path)
    for lineno, text in hits:
        print(f"    VIOLATION line {lineno}: {text!r}")
    print(f"  [Literal scanner] {len(hits)} violation(s)")
    if hits:
        sys.exit(1)
```

### C.3 A gate that cannot be widened without a source diff

```python
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable


def read_pinned_blob(blob_sha1: str) -> dict:
    """Read a reference artifact by content hash. Editing it changes its hash."""
    raw = subprocess.run(["git", "cat-file", "blob", blob_sha1],
                         capture_output=True, check=True).stdout
    actual = subprocess.run(["git", "hash-object", "--stdin"], input=raw,
                            capture_output=True, check=True).stdout.decode().strip()
    if actual != blob_sha1:
        sys.exit(f"Reference blob hash mismatch: expected {blob_sha1}, got {actual}")
    print(f"  [Reference] blob {blob_sha1} verified, "
          f"sha256={hashlib.sha256(raw).hexdigest()[:16]}...")
    return json.loads(raw)


@dataclass(frozen=True)
class Gate:
    """A gate whose tolerance is a rule in code, not an interval in a file."""
    name: str
    reference: float
    reference_source: str                 # blob sha + json path
    rule_text: str                        # e.g. "reference +/- 3.0"
    rule: Callable[[float], tuple[float, float]]

    def evaluate(self, observed: float) -> bool:
        lo, hi = self.rule(self.reference)
        ok = lo <= observed <= hi
        print(f"  {self.name}: observed={observed:.6f}  reference={self.reference:.6f}  "
              f"source={self.reference_source}  rule='{self.rule_text}'  "
              f"interval=[{lo:.6f}, {hi:.6f}]  deviation={observed - self.reference:+.6f}  "
              f"-> {'PASS' if ok else 'FAIL'}")
        return ok


def run_gates(gates: list[tuple[Gate, float]]) -> None:
    """Evaluate every gate, print all of them, then exit nonzero if any failed.

    Prints all gates even after the first failure: a reviewer needs the whole picture.
    """
    results = [(g.name, g.evaluate(obs)) for g, obs in gates]
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n  GATES FAILED: {', '.join(failed)}")
        print("  Halting. Do not widen a tolerance. Diagnose, or report and stop.")
        sys.exit(1)
```

### C.4 Guards for run integrity and ablation validity

```python
import torch


def assert_run_touched_data(optimizer_steps: int, samples_seen: int) -> None:
    """A results table without gradients or data is fabrication, whatever the exit code."""
    if optimizer_steps <= 0:
        sys.exit(f"Zero optimizer steps recorded ({optimizer_steps}) -- run invalid")
    if samples_seen <= 0:
        sys.exit(f"Zero samples seen ({samples_seen}) -- run invalid")
    print(f"  [Integrity] optimizer_steps={optimizer_steps}, samples_seen={samples_seen}")


def assert_ablation_is_not_noop(model, pre_state: dict, target_param_names: list[str]) -> None:
    """An ablation on parameters that never moved is entailed, not measured."""
    named = dict(model.named_parameters())
    total = 0.0
    for name in target_param_names:
        delta = (named[name].detach() - pre_state[name]).norm().item()
        total += delta
        print(f"    delta[{name}] = {delta:.8f}")
    if total == 0.0:
        sys.exit("Ablation targets have zero delta from pre-intervention state: "
                 "this ablation is a no-op and its result is entailed by the setup")


def assert_mode(model, expected_training: bool, where: str) -> None:
    """Execution mode is asserted, not assumed. Pass the expectation in; never hardcode it."""
    if model.training != expected_training:
        sys.exit(f"Mode mismatch at {where}: model.training={model.training}, "
                 f"expected {expected_training}")


def assert_single_token_targets(facts: list[dict], key: str = "object") -> None:
    """If the normalizer keeps only the first token, multi-word targets are mis-scored."""
    bad = [f for f in facts if f[key].split() != [f[key]]]
    print(f"  [Shape] multi-token targets: {len(bad)}")
    if bad:
        sys.exit(f"{len(bad)} target(s) contain whitespace and would be silently truncated")
```
