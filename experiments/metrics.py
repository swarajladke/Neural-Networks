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

@dataclass(frozen=True)
class Measurement:
    """
    A count-based measurement that cannot be reported without its denominator.
    Enforces the metrics contract: every rate carries the population it was computed over,
    the identity of the input set, and the execution mode.
    Impossible values raise at construction rather than being printed.
    """
    name: str
    numerator: int
    denominator: int
    input_set: str = "distinct20_seed42"
    mode: str = "eval_no_dropout"

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
# CORE CONTINUAL LEARNING METRICS
# ==============================================================================
def efficacy(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    input_set: str = "distinct20",
    mode: str = "eval"
) -> Measurement:
    """
    Calculates the fraction of ALL injected facts whose canonical object is currently
    the greedy continuation of that fact's edit prompt.
    Denominator is the count of facts injected so far (len(facts_injected)), never 1.
    Zero if no predictions match their canonical targets.
    """
    n = len(facts_injected)
    if n == 0:
        raise ValueError("efficacy: cannot evaluate on 0 injected facts")
    assert len(predictions) == n, f"Predictions count {len(predictions)} != facts count {n}"
    correct = sum(1 for p, f in zip(predictions, facts_injected) if check_match(p, f["object"]))
    return Measurement("efficacy", correct, n, input_set, mode)


def generalization(
    paraphrase_predictions: List[List[str]],
    facts_injected: List[Dict[str, Any]],
    input_set: str = "distinct20",
    mode: str = "eval"
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
    return Measurement("generalization", correct, total_paraphrases, input_set, mode)


def raw_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    input_set: str = "distinct20",
    mode: str = "eval"
) -> Measurement:
    """
    Raw Retention:
    Counts how many injected facts (s_i, r_i, o_i) have check_match(pred_i, o_i) == True
    when prompted with the fact's edit prompt.
    Denominator is len(facts_injected).
    Distinction: Evaluates raw surface recall of canonical objects across the injected sequence,
    without correcting for modal collapse or template-level bias. Zero if no target is recalled.
    """
    n = len(facts_injected)
    if n == 0:
        raise ValueError("raw_retention: cannot evaluate on 0 injected facts")
    correct = sum(1 for p, f in zip(predictions, facts_injected) if check_match(p, f["object"]))
    return Measurement("raw_retention", correct, n, input_set, mode)


def bound_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    rel_modal_objects: Dict[str, str],
    input_set: str = "distinct20",
    mode: str = "eval"
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
    return Measurement("bound_retention", correct, n, input_set, mode)


def subject_discriminable_retention(
    predictions: List[str],
    facts_injected: List[Dict[str, Any]],
    control_predictions_by_rel: Dict[str, List[str]],
    max_shared_controls: int = 2,
    input_set: str = "distinct20",
    mode: str = "eval"
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
    return Measurement("subject_discriminable_retention", correct, n, input_set, mode)


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
    control_measurements: Dict[str, Measurement]
) -> Tuple[Measurement, Measurement, str]:
    """
    Pools exactly four named controls:
      - never_edited
      - random_direction_magnitude_matched
      - wrong_target
      - pre_edit_baseline
    Each evaluates on exactly 20 facts.
    Asserts pooled denominator == 80.
    Returns:
      (pooled_measurement, worst_individual_measurement, expanded_sum_str)
    Raises ValueError immediately if any numerator > denominator or any control is missing/malformed.
    """
    missing = [c for c in CONTROL_NAMES if c not in control_measurements]
    if missing:
        raise ValueError(f"Missing required controls in pool_controls: {missing}")
    
    total_num = 0
    total_den = 0
    num_strs = []
    den_strs = []
    
    for c_name in CONTROL_NAMES:
        m = control_measurements[c_name]
        if m.denominator != 20:
            raise ValueError(f"Control {c_name} denominator must be exactly 20, got {m.denominator}")
        if m.numerator > m.denominator:
            raise ValueError(f"Control {c_name} impossible value: numerator {m.numerator} > denominator {m.denominator}")
        total_num += m.numerator
        total_den += m.denominator
        num_strs.append(str(m.numerator))
        den_strs.append(str(m.denominator))
        
    assert total_den == 80, f"Pooled control denominator must equal 80, got {total_den}"
    
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
