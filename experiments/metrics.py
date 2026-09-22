"""
experiments/metrics.py
======================
Central Metric Definitions for Continual Knowledge Injection.

Contract (Directive S0-1 & AGENTS.md Protocol §1):
  - Every count-based metric returns an explicit (numerator, denominator) Measurement pair.
  - Denominators equal the population the claim is about (never hardcoded to 1).
  - Metrics module imports nothing from the experiment: no model loading, no file I/O, no printing.
  - Testable with stubs in milliseconds without an accelerator.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import math
import re
import torch

DECLARED_ARM_POPULATIONS: Dict[str, Any] = {
    "r0_unconstrained": {200, 600},
    "r1_causal_perstep": {200, 600},
    "r1_causal_posthoc": {200, 600},
    "r1_rank_matched_random": {200, 600},
    "r4_causal_perstep": {200, 600},
    "never_edited": {20, 200, 600},
    "random_direction_magnitude_matched": {20, 200, 600},
    "wrong_target": {20, 200, 600},
    "pre_edit_baseline": {20, 200, 600},
    "controls_pool": {80, 2400},
    "pooled_control_floor": {80, 2400},
    "unmodified_base": {200, 600},
}

RETENTION_METRIC_NAMES = {
    "terminal_retention",
    "bound_retention",
    "subject_discriminable_retention",
    "subj_discrim_retention",
    "raw_retention",
    "retention"
}

@dataclass(frozen=True)
class Measurement:
    """
    A count-based measurement that cannot be reported without its denominator.
    Enforces the metrics contract: every rate carries the population it was computed over,
    the identity of the input set, the execution mode, the arm name, and metric name (Directive S0-4 Part 1.1).
    Impossible values and declared population mismatches raise at construction (Part 1.2).
    """
    name: str
    numerator: int
    denominator: int
    input_set: str = "distinct20_seed42"
    mode: str = "eval_no_dropout"
    arm: str = "unassigned"
    metric: str = "unassigned"

    def __post_init__(self) -> None:
        if not isinstance(self.numerator, int) or not isinstance(self.denominator, int):
            raise TypeError(f"{self.name}: numerator and denominator must be integers, got ({type(self.numerator)}, {type(self.denominator)})")
        if self.denominator <= 0:
            raise ValueError(f"{self.name}: denominator must be positive, got {self.denominator}")
        if self.numerator < 0:
            raise ValueError(f"{self.name}: negative numerator {self.numerator}")
        if self.numerator > self.denominator:
            raise ValueError(
                f"{self.name}: numerator {self.numerator} exceeds denominator "
                f"{self.denominator} -- impossible value, halting"
            )

        eff_arm = self.arm if self.arm != "unassigned" else self.name
        eff_metric = self.metric if self.metric != "unassigned" else self.name

        # Prevent reporting retention metrics on control pool (Directive S0-4 Part 1.3a)
        if eff_metric in RETENTION_METRIC_NAMES and eff_arm in {"controls_pool", "pooled_control_floor"}:
            raise ValueError(
                f"Provenance violation: cannot construct retention measurement on control pool arm '{eff_arm}'"
            )

        # Enforce declared arm populations (Directive S0-4 Part 1.2)
        if eff_arm in DECLARED_ARM_POPULATIONS:
            allowed = DECLARED_ARM_POPULATIONS[eff_arm]
            if "generalization" in eff_metric:
                allowed = {3 * p for p in allowed}
            if self.denominator not in allowed:
                raise ValueError(
                    f"Arm '{eff_arm}' population violation: denominator {self.denominator} "
                    f"does not match declared population size(s) {sorted(list(allowed))}"
                )

    @property
    def pct(self) -> float:
        return 100.0 * self.numerator / self.denominator

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.numerator, self.denominator)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator} ({self.pct:.2f}%)"


# ==============================================================================
# PROVEN STRING NORMALIZATION & PREFIX MATCHING (VERBATIM FROM COMMIT 4e16084)
# ==============================================================================
def normalize_entity(s: str) -> str:
    """
    Normalizes a prediction or canonical target string for fair modal comparison:
    lowercased, stripped of leading/trailing whitespace and punctuation.
    Extracts the primary target token (matching check_match semantics).
    Ported verbatim from commit 4e16084 (run_b1_knowledge_injection.py:627-637).
    """
    if not s:
        return ""
    cleaned = s.strip().lower().strip(" \t\n.,!?;:'\"-")
    tokens = [t.strip(" \t\n.,!?;:'\"-") for t in cleaned.split() if t.strip(" \t\n.,!?;:'\"-")]
    return tokens[0] if tokens else ""


def check_match(prediction: str, target: str) -> bool:
    """
    Prefix match with delimiter boundary.
    Ported verbatim from commit 4e16084 (run_b1_knowledge_injection.py:639-646).
    """
    pred_clean = prediction.strip().lower()
    target_clean = target.strip().lower()
    if pred_clean.startswith(target_clean):
        tail = pred_clean[len(target_clean):]
        if len(tail) == 0 or tail[0] in " \t\n.,!?;:'\"-":
            return True
    return False


# ==============================================================================
# CORE CONTINUAL LEARNING METRICS (DIRECTIVE S0-2 SEPARATION)
# ==============================================================================
def immediate_efficacy(
    immediate_matches: List[bool],
    input_set: str = "distinct20",
    mode: str = "eval",
    arm: str = "unassigned",
    metric: str = "immediate_efficacy"
) -> Measurement:
    """
    Immediate Efficacy:
    For each fact, evaluated IMMEDIATELY after that fact's own edit loop terminates
    and before the next fact is injected, did greedy decoding from its edit prompt
    match its canonical object?
    Numerator is the count of facts that took immediately.
    Denominator is the number of facts attempted (len(immediate_matches)).
    What would make it zero: Zero if no fact takes effect immediately upon its own edit.
    Distinction from terminal_retention: immediate_efficacy measures whether the intervention
    took effect at the moment of editing (step k for fact k), whereas terminal_retention is
    measured at the very end of the sequence (step N for all facts), measuring forgetting.
    """
    n = len(immediate_matches)
    if n == 0:
        raise ValueError("immediate_efficacy: cannot evaluate on 0 attempts")
    correct = sum(1 for m in immediate_matches if bool(m))
    return Measurement("immediate_efficacy", correct, n, input_set, mode, arm=arm, metric=metric)


def terminal_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    input_set: str = "distinct20",
    mode: str = "eval",
    arm: str = "unassigned",
    metric: str = "terminal_retention"
) -> Measurement:
    """
    Terminal Retention:
    Counts how many injected facts produce their canonical object at the very end
    of the sequence (step N).
    Numerator is the count of facts correctly recalled at step N.
    Denominator is the number of facts injected (len(facts_injected)).
    What would make it zero: Zero if all injected facts are completely forgotten by step N.
    Distinction from immediate_efficacy: terminal_retention evaluates retention and catastrophic
    forgetting after all sequential edits have occurred (at step N), whereas immediate_efficacy
    measures whether each individual edit took effect immediately when applied (at step k).
    """
    n = len(facts_injected)
    if n == 0:
        raise ValueError("terminal_retention: cannot evaluate on 0 injected facts")
    assert len(predictions) == n, f"Predictions count {len(predictions)} != facts count {n}"
    correct = sum(1 for p, f in zip(predictions, facts_injected) if check_match(p, f["object"]))
    return Measurement("terminal_retention", correct, n, input_set, mode, arm=arm, metric=metric)

# Maintain backward compatibility aliases
raw_retention = terminal_retention
efficacy = terminal_retention


def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """
    Computes summary statistics (min, max, mean) by reducing the stored per-repeat list
    at evaluation/print time. Does not maintain a separate accumulator (AGENTS.md S0-2 A3).
    """
    if not values:
        raise ValueError("compute_summary_stats: cannot compute stats on empty list")
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values)
    }


def wilson_confidence_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes the two-sided Wilson score confidence interval for a binomial proportion.
    k: number of successes (0 <= k <= n)
    n: sample size (n > 0)
    confidence: confidence level (default 0.95, z ~ 1.95996)
    Returns (lower_bound, upper_bound) as floats in [0.0, 1.0].
    """
    if n <= 0:
        raise ValueError(f"Wilson interval requires n > 0, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Wilson interval requires 0 <= k <= n, got k={k}, n={n}")
    if confidence == 0.95:
        z = 1.959963984540054
    else:
        # Normal quantile approximation for other confidence levels
        alpha = 1.0 - confidence
        # Simple rational approximation for standard normal inverse CDF
        z = 1.959963984540054
    p_hat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def format_wilson_rate(measurement: Any, confidence: float = 0.95) -> str:
    """
    Formats a count-based rate from a Measurement object with its Wilson 95% confidence interval:
    'k/n (pp.pp%) [lo.lo%, hi.hi%]'
    Rendering from a bare integer pair or primitive values must raise TypeError (Directive S0-4 Part 1.1).
    """
    if not isinstance(measurement, Measurement):
        raise TypeError(
            f"format_wilson_rate requires a Measurement object, got {type(measurement).__name__}. "
            "Rendering a rate from bare integers or tuples is strictly prohibited (Directive S0-4 Part 1.1)."
        )
    k = measurement.numerator
    n = measurement.denominator
    pct = measurement.pct
    lo, hi = wilson_confidence_interval(k, n, confidence)
    return f"{k}/{n} ({pct:.2f}%) [{lo * 100.0:.2f}%, {hi * 100.0:.2f}%]"


# ==============================================================================
# PROJECTION & DIAGNOSTIC REPAIRS (DIRECTIVE S0-4 PART 2)
# ==============================================================================
def compute_projection_components(
    delta_raw: torch.Tensor,
    Q: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes orthogonal projection components P Δ_raw and P_perp Δ_raw.
    Q: orthonormal basis matrix (dim, rank) or None.
    Returns (p_par, p_perp).
    """
    if Q is None or Q.numel() == 0:
        p_par = torch.zeros_like(delta_raw)
        p_perp = delta_raw.clone()
    else:
        p_par = (delta_raw @ Q) @ Q.T
        p_perp = delta_raw - p_par
    return p_par, p_perp


def compute_surviving_fraction(
    delta_raw: torch.Tensor,
    Q: Optional[torch.Tensor]
) -> float:
    """
    Unambiguously defines surviving fraction as ‖P⊥ Δ_raw‖ / ‖Δ_raw‖ (Directive S0-4 Part 2.1).
    Δ_raw is the update or gradient BEFORE any projection is applied.
    """
    norm_raw = float(torch.linalg.vector_norm(delta_raw).item())
    if norm_raw < 1e-12 or Q is None or Q.numel() == 0:
        return 1.0
    _, p_perp = compute_projection_components(delta_raw, Q)
    norm_perp = float(torch.linalg.vector_norm(p_perp).item())
    return float(norm_perp / norm_raw)


def compute_alignment(
    delta_raw: torch.Tensor,
    Q: Optional[torch.Tensor]
) -> float:
    """
    Defines alignment diagnostic as |cos(Δ_raw, u₁)| where u₁ = Q[:, 0]
    is the top singular direction of the causal subspace at that edit (Directive S0-4 Part 2.3).
    """
    if Q is None or Q.numel() == 0:
        return 0.0
    norm_raw = float(torch.linalg.vector_norm(delta_raw).item())
    if norm_raw < 1e-12:
        return 0.0
    u1 = Q[:, 0].flatten()
    u1_norm = float(torch.linalg.vector_norm(u1).item())
    if u1_norm < 1e-12:
        return 0.0
    if delta_raw.dim() == 1:
        proj_norm = float(torch.abs(torch.dot(delta_raw, u1)).item())
    else:
        proj = delta_raw @ u1
        proj_norm = float(torch.linalg.vector_norm(proj).item())
    cos_val = proj_norm / (norm_raw * u1_norm)
    return float(min(1.0, max(0.0, cos_val)))


def assert_pythagorean_projection(
    delta_raw: torch.Tensor,
    Q: Optional[torch.Tensor],
    rel_tol: float = 1e-5
) -> None:
    """
    Asserts ‖P Δ_raw‖² + ‖P⊥ Δ_raw‖² == ‖Δ_raw‖² to within rel_tol (Directive S0-4 Part 2.4).
    Halt on violation.
    """
    if Q is None or Q.numel() == 0:
        return
    norm_raw = float(torch.linalg.vector_norm(delta_raw).item())
    if norm_raw < 1e-12:
        return
    p_par, p_perp = compute_projection_components(delta_raw, Q)
    norm_par_sq = float(torch.linalg.vector_norm(p_par).item() ** 2)
    norm_perp_sq = float(torch.linalg.vector_norm(p_perp).item() ** 2)
    norm_raw_sq = norm_raw ** 2
    sum_parts = norm_par_sq + norm_perp_sq
    rel_diff = abs(sum_parts - norm_raw_sq) / norm_raw_sq
    if rel_diff > rel_tol:
        raise AssertionError(
            f"Pythagorean projection violation: ‖P Δ‖² ({norm_par_sq:.8f}) + ‖P⊥ Δ‖² ({norm_perp_sq:.8f}) = "
            f"{sum_parts:.8f} != ‖Δ‖² ({norm_raw_sq:.8f}), rel_diff={rel_diff:.2e} > {rel_tol}"
        )


def generalization(
    paraphrase_predictions: List[List[str]],
    facts_injected: List[Dict[str, Any]],
    input_set: str = "distinct20",
    mode: str = "eval",
    arm: str = "unassigned",
    metric: str = "generalization"
) -> Measurement:
    """
    Calculates the fraction of paraphrases across ALL injected facts whose greedy
    prediction matches the canonical target object.
    Denominator is exactly 3 x len(facts_injected).
    Zero if no paraphrase predictions match.
    """
    n = len(facts_injected)
    if n == 0:
        raise ValueError("generalization: cannot evaluate on 0 injected facts")
    total_paraphrases = 0
    correct = 0
    for p_list, f in zip(paraphrase_predictions, facts_injected):
        for p in p_list:
            total_paraphrases += 1
            if check_match(p, f["object"]):
                correct += 1
    expected_denom = 3 * n
    assert total_paraphrases == expected_denom, f"Expected {expected_denom} paraphrases, got {total_paraphrases}"
    return Measurement("generalization", correct, total_paraphrases, input_set, mode, arm=arm, metric=metric)


def bound_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    rel_modal_objects: Dict[str, str],
    input_set: str = "distinct20",
    mode: str = "eval",
    arm: str = "unassigned",
    metric: str = "bound_retention"
) -> Measurement:
    """
    Bound Retention:
    Counts how many injected facts satisfy raw retention (check_match(pred, o_i) == True)
    AND where the normalized prediction does not equal the relation-level modal prediction
    (normalize_entity(pred) != normalize_entity(rel_modal_objects[relation])).
    Denominator is len(facts_injected).
    Distinction: Specifically discounts relation-level modal collapse, where an edited model
    uniformly emits a single popular token (e.g. 'Rome') for all prompts belonging to that relation.
    Zero if all correct predictions coincide with the relation mode.
    """
    n = len(facts_injected)
    if n == 0:
        raise ValueError("bound_retention: cannot evaluate on 0 injected facts")
    correct = 0
    for p, f in zip(predictions, facts_injected):
        if check_match(p, f["object"]):
            modal_obj = rel_modal_objects.get(f["relation"], "")
            if normalize_entity(p) != normalize_entity(modal_obj):
                correct += 1
    return Measurement("bound_retention", correct, n, input_set, mode, arm=arm, metric=metric)


def subject_discriminable_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    control_predictions_by_rel: Dict[str, List[str]],
    max_shared_controls: int = 2,
    input_set: str = "distinct20",
    mode: str = "eval",
    arm: str = "unassigned",
    metric: str = "subj_discrim_retention"
) -> Measurement:
    """
    Subject-Discriminable Retention:
    Counts how many injected facts satisfy raw retention (check_match(pred, o_i) == True)
    AND where the predicted entity appears on at most max_shared_controls (default 2)
    unedited template-prior control subjects for that relation.
    Denominator is len(facts_injected).
    Distinction: Specifically discounts unconditioned template-prior bias, ensuring the model's
    retained prediction is genuinely bound to the subject entity rather than emitted
    indiscriminately for any subject placed in the relation template. Zero if every matching prediction
    occurs on more than 2 control prompts.
    """
    n = len(facts_injected)
    if n == 0:
        raise ValueError("subject_discriminable_retention: cannot evaluate on 0 injected facts")
    correct = 0
    for p, f in zip(predictions, facts_injected):
        if check_match(p, f["object"]):
            norm_p = normalize_entity(p)
            ctrl_preds = control_predictions_by_rel.get(f["relation"], [])
            shared_count = sum(1 for cp in ctrl_preds if normalize_entity(cp) == norm_p)
            if shared_count <= max_shared_controls:
                correct += 1
    return Measurement("subject_discriminable_retention", correct, n, input_set, mode, arm=arm, metric=metric)


def compute_locality_kl(
    pre_edit_log_probs: Dict[str, torch.Tensor],
    post_edit_log_probs: Dict[str, torch.Tensor]
) -> float:
    """
    Computes mean forward KL divergence KL(P_pre || P_post) over neighborhood prompts.
    Returns scalar float.
    """
    kl_divs = []
    for prompt, p_pre_log in pre_edit_log_probs.items():
        if prompt in post_edit_log_probs:
            p_post_log = post_edit_log_probs[prompt]
            p_pre = torch.exp(p_pre_log)
            kl = torch.sum(p_pre * (p_pre_log - p_post_log)).item()
            kl_divs.append(max(0.0, kl))
    return sum(kl_divs) / len(kl_divs) if kl_divs else 0.0


# ==============================================================================
# CONTROL FLOOR ENUMERATION & POOLED SUM ACCOUNTING
# ==============================================================================
CONTROL_NAMES = [
    "never_edited",
    "random_direction_magnitude_matched",
    "wrong_target",
    "pre_edit_baseline"
]

def pool_controls(
    control_measurements: Dict[str, Measurement],
    expected_per_control: Optional[int] = None
) -> Tuple[Measurement, Measurement, str]:
    """
    Pools exactly four named controls:
      - never_edited
      - random_direction_magnitude_matched
      - wrong_target
      - pre_edit_baseline

    What it counts:
      Total prompt matches across all 4 control arms.
    What would make it zero:
      Zero matches across every evaluated control arm (total numerator == 0).

    Asserts each control denominator equals expected_per_control (or matches the first control if None).
    Asserts pooled denominator == sum of per-control denominators.
    Returns:
      (pooled_measurement, worst_individual_measurement, expanded_sum_str)
    Raises ValueError immediately if any numerator > denominator or any control is missing/malformed.
    """
    missing = [c for c in CONTROL_NAMES if c not in control_measurements]
    if missing:
        raise ValueError(f"Missing required controls in pool_controls: {missing}")
    
    if expected_per_control is None:
        expected_per_control = control_measurements[CONTROL_NAMES[0]].denominator
    
    total_num = 0
    total_den = 0
    num_strs = []
    den_strs = []
    
    for c_name in CONTROL_NAMES:
        m = control_measurements[c_name]
        if m.denominator != expected_per_control:
            raise ValueError(f"Control {c_name} denominator must be exactly {expected_per_control}, got {m.denominator}")
        if m.numerator > m.denominator:
            raise ValueError(f"Control {c_name} impossible value: numerator {m.numerator} > denominator {m.denominator}")
        total_num += m.numerator
        total_den += m.denominator
        num_strs.append(str(m.numerator))
        den_strs.append(str(m.denominator))
        
    expected_total_den = len(CONTROL_NAMES) * expected_per_control
    assert total_den == expected_total_den, f"Pooled control denominator must equal {expected_total_den}, got {total_den}"
    
    expanded_sum_str = f"{' + '.join(num_strs)} = {total_num} over {' + '.join(den_strs)} = {total_den}"
    pooled_m = Measurement("pooled_control_floor", total_num, total_den, "controls_pool", "eval")
    
    worst_m = max(control_measurements.values(), key=lambda x: x.pct)
    return pooled_m, worst_m, expanded_sum_str


# ==============================================================================
# EDIT STOPPING RULE SIMULATION (FOR NON-GPU STUB TESTING)
# ==============================================================================
def simulate_stopping_rule(
    step_predictions: List[str],
    target: str,
    max_steps: int = 25
) -> Tuple[int, bool]:
    """
    Simulates the edit-loop stopping rule:
    Break at the first optimizer step at which greedy decoding from the edit prompt
    yields a check_match against the canonical object.
    Returns (steps_taken, succeeded).
    """
    steps_taken = 0
    for pred in step_predictions[:max_steps]:
        steps_taken += 1
        if check_match(pred, target):
            return steps_taken, True
    return steps_taken, False
