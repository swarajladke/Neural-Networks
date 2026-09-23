"""
experiments/metrics.py
======================
Central Metric Definitions for Continual Knowledge Injection.

Contract (Directive S0-5 & AGENTS.md Protocol §1):
  - Every count-based metric returns an explicit (numerator, denominator) Measurement pair.
  - Measurement is constructible strictly via Measurement.from_outcomes(...) with a module-private sentinel.
  - Closed registry scope -> legal_denominator with fixed test denominators.
  - Denominators equal the population the claim is about (never hardcoded to 1).
  - Metrics module imports nothing from the experiment: no model loading, no file I/O, no printing.
  - Testable with stubs in milliseconds without an accelerator.
"""

from typing import Tuple, List, Dict, Any, Optional, Sequence
import math
import torch

_FACTORY_SENTINEL = object()

POPULATION_REGISTRY: Dict[str, int] = {
    "per_seed": 200,
    "pooled": 1200,
    "pooled_600": 600,
    "generalization_per_seed": 600,
    "generalization_pooled": 3600,
    "generalization_pooled_1800": 1800,
    "recency_bin": 60,
    "revert_bin": 120,
    "controls_pooled": 4800,
    "controls_pooled_2400": 2400,
    "control_arm_pooled": 1200,
    "control_arm_pooled_600": 600,
    "fixture_1": 1,
    "fixture_3": 3,
    "fixture_5": 5,
    "fixture_12": 12,
    "fixture_20": 20,
    "fixture_60": 60,
    "fixture_80": 80,
    "fixture_600": 600,
    "fixture_1200": 1200,
}

RETENTION_METRIC_NAMES = {
    "terminal_retention",
    "bound_retention",
    "subject_discriminable_retention",
    "subj_discrim_retention",
    "raw_retention",
    "retention"
}


class Measurement:
    """
    A count-based measurement that cannot be reported without its denominator.
    Enforces the metrics contract: every rate carries the population it was computed over,
    the identity of the input set, the execution mode, the arm name, and metric name (Directive S0-5 Part 1).
    Constructible strictly via Measurement.from_outcomes(...) using a private sentinel.
    Direct public constructor calls raise TypeError immediately.
    """
    __slots__ = ("name", "numerator", "denominator", "input_set", "mode", "arm", "metric")
    name: str
    numerator: int
    denominator: int
    input_set: str
    mode: str
    arm: str
    metric: str

    def __init__(
        self,
        name: str,
        numerator: int,
        denominator: int,
        input_set: str,
        mode: str,
        arm: str = "unassigned",
        metric: str = "unassigned",
        _sentinel: Any = None
    ) -> None:
        if _sentinel is not _FACTORY_SENTINEL:
            raise TypeError(
                "Measurement cannot be constructed directly. Use Measurement.from_outcomes(...) "
                "to guarantee provenance (Directive S0-5 Part 1)."
            )
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise TypeError(f"{name}: numerator and denominator must be integers, got ({type(numerator)}, {type(denominator)})")
        if denominator <= 0:
            raise ValueError(f"{name}: denominator must be positive, got {denominator}")
        if numerator < 0:
            raise ValueError(f"{name}: negative numerator {numerator}")
        if numerator > denominator:
            raise ValueError(
                f"{name}: numerator {numerator} exceeds denominator "
                f"{denominator} -- impossible value, halting"
            )
        if not mode or not isinstance(mode, str) or mode.strip() == "":
            raise TypeError("Execution mode must be explicitly declared as a non-empty string (e.g. 'eval_no_dropout')")
        if not input_set or not isinstance(input_set, str) or input_set.strip() == "":
            raise TypeError("input_set must be explicitly declared as a non-empty string")

        eff_arm = arm if arm != "unassigned" else name
        eff_metric = metric if metric != "unassigned" else name

        # Prevent reporting retention metrics on control pool (Directive S0-4 Part 1.3a / S0-5)
        if eff_metric in RETENTION_METRIC_NAMES and eff_arm in {"controls_pool", "pooled_control_floor"}:
            raise ValueError(
                f"Provenance violation: cannot construct retention measurement on control pool arm '{eff_arm}'"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "input_set", input_set)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "arm", eff_arm)
        object.__setattr__(self, "metric", eff_metric)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Measurement instances are immutable")

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[bool],
        metric: str,
        arm: str,
        scope: str,
        input_set: str,
        mode: str
    ) -> "Measurement":
        """
        Public factory enforcing Provenance Guard 2.0 (Directive S0-5 Part 1):
        Constructs a Measurement from an explicit boolean outcome sequence, verifying
        the population against the closed POPULATION_REGISTRY.
        """
        if not isinstance(outcomes, (list, tuple)):
            raise TypeError(f"outcomes must be a list or tuple of bools, got {type(outcomes)}")
        if scope not in POPULATION_REGISTRY:
            raise ValueError(
                f"Unknown or unauthorized scope '{scope}'. Must be registered in POPULATION_REGISTRY: "
                f"{sorted(list(POPULATION_REGISTRY.keys()))}"
            )
        expected_denom = POPULATION_REGISTRY[scope]
        actual_denom = len(outcomes)
        if actual_denom != expected_denom:
            raise ValueError(
                f"Scope '{scope}' requires denominator {expected_denom}, got {actual_denom}"
            )
        k = sum(1 for x in outcomes if bool(x))
        return cls(
            name=metric,
            numerator=k,
            denominator=actual_denom,
            input_set=input_set,
            mode=mode,
            arm=arm,
            metric=metric,
            _sentinel=_FACTORY_SENTINEL
        )

    @property
    def pct(self) -> float:
        return 100.0 * self.numerator / self.denominator

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.numerator, self.denominator)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator} ({self.pct:.2f}%)"

    def __repr__(self) -> str:
        return f"Measurement(name='{self.name}', numerator={self.numerator}, denominator={self.denominator}, input_set='{self.input_set}', mode='{self.mode}', arm='{self.arm}', metric='{self.metric}')"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Measurement):
            return False
        return (
            self.name == other.name and
            self.numerator == other.numerator and
            self.denominator == other.denominator and
            self.input_set == other.input_set and
            self.mode == other.mode and
            self.arm == other.arm and
            self.metric == other.metric
        )

    def __hash__(self) -> int:
        return hash((self.name, self.numerator, self.denominator, self.input_set, self.mode, self.arm, self.metric))


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
# CORE CONTINUAL LEARNING METRICS (DIRECTIVE S0-5 PROVENANCE-GUARDED)
# ==============================================================================
def immediate_efficacy(
    immediate_matches: List[bool],
    input_set: str,
    mode: str,
    arm: str = "unassigned",
    metric: str = "immediate_efficacy",
    scope: str = "per_seed"
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
    return Measurement.from_outcomes(
        immediate_matches,
        metric=metric,
        arm=arm,
        scope=scope,
        input_set=input_set,
        mode=mode
    )


def terminal_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    input_set: str,
    mode: str,
    arm: str = "unassigned",
    metric: str = "terminal_retention",
    scope: str = "per_seed"
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
    matches = [check_match(p, f["object"]) for p, f in zip(predictions, facts_injected)]
    return Measurement.from_outcomes(
        matches,
        metric=metric,
        arm=arm,
        scope=scope,
        input_set=input_set,
        mode=mode
    )

# Maintain backward compatibility aliases
raw_retention = terminal_retention
efficacy = terminal_retention


def generalization(
    paraphrase_predictions: List[List[str]],
    facts_injected: List[Dict[str, Any]],
    input_set: str,
    mode: str,
    arm: str = "unassigned",
    metric: str = "generalization",
    scope: str = "generalization_per_seed"
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
    matches = []
    for p_list, f in zip(paraphrase_predictions, facts_injected):
        for p in p_list:
            matches.append(check_match(p, f["object"]))
    return Measurement.from_outcomes(
        matches,
        metric=metric,
        arm=arm,
        scope=scope,
        input_set=input_set,
        mode=mode
    )


def bound_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    rel_modal_objects: Dict[str, str],
    input_set: str,
    mode: str,
    arm: str = "unassigned",
    metric: str = "bound_retention",
    scope: str = "per_seed"
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
    assert len(predictions) == n, f"Predictions count {len(predictions)} != facts count {n}"
    matches = []
    for p, f in zip(predictions, facts_injected):
        is_ret = check_match(p, f["object"])
        modal_obj = rel_modal_objects.get(f["relation"], "")
        not_modal = (normalize_entity(p) != normalize_entity(modal_obj))
        matches.append(is_ret and not_modal)
    return Measurement.from_outcomes(
        matches,
        metric=metric,
        arm=arm,
        scope=scope,
        input_set=input_set,
        mode=mode
    )


def subject_discriminable_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    control_predictions_by_rel: Dict[str, List[str]],
    max_shared_controls: int = 2,
    input_set: str = "",
    mode: str = "",
    arm: str = "unassigned",
    metric: str = "subj_discrim_retention",
    scope: str = "per_seed"
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
    assert len(predictions) == n, f"Predictions count {len(predictions)} != facts count {n}"
    matches = []
    for p, f in zip(predictions, facts_injected):
        is_ret = check_match(p, f["object"])
        norm_p = normalize_entity(p)
        ctrl_preds = control_predictions_by_rel.get(f["relation"], [])
        shared_count = sum(1 for cp in ctrl_preds if normalize_entity(cp) == norm_p)
        not_prior = (shared_count <= max_shared_controls)
        matches.append(is_ret and not_prior)
    return Measurement.from_outcomes(
        matches,
        metric=metric,
        arm=arm,
        scope=scope,
        input_set=input_set,
        mode=mode
    )


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
# PROJECTION & DIAGNOSTIC REPAIRS (DIRECTIVE S0-5 PART 1)
# ==============================================================================
def assert_orthonormality(Q: Optional[torch.Tensor], tol: float = 1e-6) -> None:
    """
    Asserts ‖Q^T Q - I‖_max < tol for orthonormal basis matrix Q (Directive S0-5 Part 1.2).
    Halts if violated.
    """
    if Q is None or Q.numel() == 0:
        return
    rank = Q.shape[1]
    I = torch.eye(rank, device=Q.device, dtype=Q.dtype)
    diff = float(torch.max(torch.abs(Q.T @ Q - I)).item())
    if diff >= tol:
        raise AssertionError(
            f"Subspace basis Q fails orthonormality: ‖Q^T Q - I‖_max = {diff:.2e} >= {tol}"
        )


def compute_projection_components(
    delta_raw: torch.Tensor,
    Q: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes orthogonal projection components P Δ_raw and P_perp Δ_raw.
    Q: orthonormal basis matrix (dim, rank) or None.
    Supports 1D vector (dim,) and 2D matrix (num_rows, dim).
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
    Defines surviving fraction as ‖P⊥ Δ_raw‖ / ‖Δ_raw‖ (Directive S0-4 Part 2.1 / S0-5 Part 1).
    Supports 1D vector (dim,) and 2D matrix (num_rows, dim).
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
    Defines alignment diagnostic as true cosine with the top singular direction u₁ = Q[:, 0]:
      - For 1D vector: |cos(δ_raw, u₁)| = |δ_raw · u₁| / (‖δ_raw‖ ‖u₁‖)
      - For 2D matrix: ‖Δ_raw u₁‖ / (‖Δ_raw‖_F ‖u₁‖)
    (Directive S0-4 Part 2.3 / S0-5 Part 1.4).
    """
    if Q is None or Q.numel() == 0:
        return 0.0
    norm_raw = float(torch.linalg.vector_norm(delta_raw).item())
    if norm_raw < 1e-12:
        return 0.0
    u1 = Q[:, 0]
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
    pooled_outcomes: List[bool] = []
    
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
        pooled_outcomes.extend([True] * m.numerator + [False] * (m.denominator - m.numerator))
        
    expected_total_den = len(CONTROL_NAMES) * expected_per_control
    assert total_den == expected_total_den, f"Pooled control denominator must equal {expected_total_den}, got {total_den}"
    
    expanded_sum_str = f"{' + '.join(num_strs)} = {total_num} over {' + '.join(den_strs)} = {total_den}"
    if total_den == 4800:
        scope = "controls_pooled"
    elif total_den == 2400:
        scope = "controls_pooled_2400"
    else:
        scope = "fixture_80"
    pooled_m = Measurement.from_outcomes(
        pooled_outcomes,
        metric="pooled_control_floor",
        arm="pooled_control_floor",
        scope=scope,
        input_set="controls_pool",
        mode="eval_no_dropout"
    )
    
    worst_m = max(control_measurements.values(), key=lambda x: x.pct)
    return pooled_m, worst_m, expanded_sum_str


# ==============================================================================
# PAIRED STATISTICAL INFERENCE (DIRECTIVE S0-6 SECTION 2.4)
# ==============================================================================
def compute_paired_stats(x1: Sequence[float], x2: Sequence[float]) -> Dict[str, Any]:
    """
    Computes paired difference statistics across matching seeds:
    - difference vector d = x1 - x2
    - mean difference d_bar
    - sample standard deviation s_d (ddof=1)
    - paired t-statistic: t = d_bar / (s_d / sqrt(n)), df = n - 1
    - Wilcoxon signed-rank test statistic W
    """
    assert len(x1) == len(x2), f"Length mismatch: {len(x1)} != {len(x2)}"
    n = len(x1)
    assert n >= 2, "Paired analysis requires at least 2 pairs"
    diffs = [float(a - b) for a, b in zip(x1, x2)]
    mean_d = sum(diffs) / float(n)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / float(n - 1)
    std_d = math.sqrt(var_d)
    
    se_d = std_d / math.sqrt(n)
    t_stat = (mean_d / se_d) if se_d > 1e-12 else 0.0
    df = n - 1
    
    nz_diffs = [d for d in diffs if abs(d) > 1e-9]
    if len(nz_diffs) == 0:
        w_stat = 0.0
    else:
        abs_diffs = [(abs(d), i, 1 if d > 0 else -1) for i, d in enumerate(nz_diffs)]
        abs_diffs.sort(key=lambda x: x[0])
        ranks = [0.0] * len(abs_diffs)
        i = 0
        while i < len(abs_diffs):
            j = i
            while j < len(abs_diffs) and abs(abs_diffs[j][0] - abs_diffs[i][0]) < 1e-9:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j
        w_plus = sum(ranks[k] for k in range(len(abs_diffs)) if abs_diffs[k][2] > 0)
        w_minus = sum(ranks[k] for k in range(len(abs_diffs)) if abs_diffs[k][2] < 0)
        w_stat = min(w_plus, w_minus)
    
    return {
        "diffs": diffs,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "t_stat": t_stat,
        "df": df,
        "wilcoxon_stat": w_stat
    }


# ==============================================================================
# MONOTONE RETENTION HORIZON SEARCH (DIRECTIVE S0-6 SECTION 2.3 & 4.1)
# ==============================================================================
def compute_monotone_retention_horizon(
    terminal_matches_by_seed: Dict[int, List[bool]],
    floor_interval: Tuple[float, float],
    step_size: int = 10,
    total_edits: int = 200
) -> Dict[str, Any]:
    """
    Finds largest k in {10, 20, ..., total_edits} such that retention over
    edits (total_edits - k) ... total_edits is separable from the negative control
    floor by non-overlapping 95% Wilson intervals for EVERY k' <= k (first failure point).
    """
    seeds = sorted(terminal_matches_by_seed.keys())
    num_seeds = len(seeds)
    floor_lo, floor_hi = floor_interval
    
    step_verdicts = []
    largest_k = 0
    monotone_broken = False
    
    for k in range(step_size, total_edits + 1, step_size):
        start_idx = total_edits - k
        outcomes_k = [terminal_matches_by_seed[s][i] for s in seeds for i in range(start_idx, total_edits)]
        num_k = sum(1 for x in outcomes_k if x)
        den_k = len(outcomes_k)
        w_lo, w_hi = wilson_confidence_interval(num_k, den_k)
        separates = (w_lo > floor_hi)
        step_verdicts.append({
            "k": k, "numerator": num_k, "denominator": den_k,
            "rate": (num_k / den_k) if den_k > 0 else 0.0,
            "wilson_lo": w_lo, "wilson_hi": w_hi, "separates": separates
        })
        if not monotone_broken:
            if separates:
                largest_k = k
            else:
                monotone_broken = True
                
    # Compute remainder retention over edits 1 ... (total_edits - largest_k)
    rem_k = total_edits - largest_k
    rem_data = None
    if rem_k > 0:
        rem_outcomes = [terminal_matches_by_seed[s][i] for s in seeds for i in range(0, rem_k)]
        r_num = sum(1 for x in rem_outcomes if x)
        r_den = len(rem_outcomes)
        r_lo, r_hi = wilson_confidence_interval(r_num, r_den)
        rem_data = {
            "k": rem_k, "numerator": r_num, "denominator": r_den,
            "rate": (r_num / r_den) if r_den > 0 else 0.0,
            "wilson_lo": r_lo, "wilson_hi": r_hi
        }
        
    return {
        "horizon_k": largest_k,
        "step_verdicts": step_verdicts,
        "remainder": rem_data
    }


# ==============================================================================
# PRINCIPLED REVERSION PATTERN CLASSIFIER (DIRECTIVE S0-6 PART 3.1 & 4.2)
# ==============================================================================
def classify_reversion_pattern(bin_intervals: Sequence[Tuple[float, float]], bin_rates: Sequence[float]) -> str:
    """
    Principled classifier for reversion dose-response:
    - If all 95% Wilson intervals share a common intersection (max(lo) <= min(hi)),
      all bins are statistically indistinguishable from a constant line -> FLAT.
    - Otherwise, if rates are monotonically decreasing with surviving fraction -> GRADED.
    - Otherwise -> THRESHOLD-LIKE.
    """
    assert len(bin_intervals) == len(bin_rates) and len(bin_intervals) > 0
    max_lo = max(inv[0] for inv in bin_intervals)
    min_hi = min(inv[1] for inv in bin_intervals)
    
    if max_lo <= min_hi:
        return "FLAT — NO DOSE RESPONSE DETECTED"
        
    # Check monotonicity
    is_monotonic = all(bin_rates[i] >= bin_rates[i+1] for i in range(len(bin_rates)-1))
    if is_monotonic and bin_intervals[0][0] > bin_intervals[-1][1]:
        return "GRADED: Revert rate decreases progressively with increasing surviving fraction."
        
    return "THRESHOLD-LIKE: Reversion occurs sharply across surviving fraction boundary."


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

