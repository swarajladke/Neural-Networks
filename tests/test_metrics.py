#!/usr/bin/env python3
"""
tests/test_metrics.py
=====================
Pre-Flight Unit Test Suite for Continual Learning Metrics (Directive S0-2).

Mandate:
  - Must run before any model loads or accelerator initializes.
  - Written against hand-constructed stubs; zero GPU/model dependency.
  - Each check prints expected and actual; any mismatch halts and exits nonzero.
  - Zero tests run is an immediate failure.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch


# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.metrics import (
    Measurement,
    normalize_entity,
    check_match,
    immediate_efficacy,
    terminal_retention,
    generalization,
    bound_retention,
    subject_discriminable_retention,
    compute_locality_kl,
    pool_controls,
    compute_summary_stats,
    raw_retention,
    simulate_stopping_rule,
    wilson_confidence_interval,
    format_wilson_rate,
    compute_projection_components,
    compute_surviving_fraction,
    compute_alignment,
    assert_pythagorean_projection
)
from experiments.data import sample_200_facts

# ==============================================================================
# AST LITERAL SCANNER (AGENTS.md Appendix C.2 / Directive S0-2 A4)
# ==============================================================================
NUMERIC = re.compile(r"\d+\.\d+|\d+\s*%|%\s*\d+")

ALLOW_LIST = {
    "=" * 115: "table rule border line",
    "-" * 115: "table rule separator line",
    "=" * 100: "test suite banner border line",
    "-" * 100: "test suite section separator line",
    "-" * 95: "historical comparison table separator line",
    "-" * 135: "cell comparison table separator line",
    "=" * 135: "cell comparison table border line",
    "-" * 145: "sweep comparison table separator line",
    "=" * 145: "sweep comparison table border line",
    ":4096:8": "cublas deterministic workspace configuration flag",
    "SUPPRESSED — immediate efficacy below 90%": "directive S0-2 B5 gate suppression text",
}

def _format_spec_node_ids(call: ast.Call) -> set:
    ids = set()
    for sub in ast.walk(call):
        if isinstance(sub, ast.FormattedValue) and sub.format_spec is not None:
            for n in ast.walk(sub.format_spec):
                ids.add(id(n))
    return ids

def scan_for_typed_literals(path: str) -> List[tuple]:
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
            continue
        skip = _format_spec_node_ids(node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and id(sub) not in skip:
                text = sub.value
                if NUMERIC.search(text) and text not in ALLOW_LIST:
                    hits.append((getattr(sub, "lineno", -1), text))
    return hits

def enforce_no_typed_literals(path: str) -> None:
    print("  [Literal scanner] Allow-list:")
    for entry, why in ALLOW_LIST.items():
        print(f"    {entry[:32]!r}: {why}")
    hits = scan_for_typed_literals(path)
    for lineno, text in hits:
        print(f"    VIOLATION line {lineno}: {text!r}")
    print(f"  [Literal scanner] {len(hits)} violation(s) detected in {Path(path).name}")
    assert len(hits) == 0, f"Literal scanner detected {len(hits)} violation(s) in {path}"


def run_all_tests() -> int:
    tests_run = 0
    tests_passed = 0
    
    print("=" * 100)
    print(" PRE-FLIGHT TEST SUITE (Directive S0-2 Part A): Hand-Constructed Metric Stubs")
    print("=" * 100)
    
    # --------------------------------------------------------------------------
    # AST LITERAL SCANNER ON b1_inject.py
    # --------------------------------------------------------------------------
    print("\n[AST Startup Literal Scanner Audit (AGENTS.md C.2 / S0-2 A4)]")
    target_script = REPO_ROOT / "experiments" / "b1_inject.py"
    if target_script.exists():
        tests_run += 1
        enforce_no_typed_literals(str(target_script))
        print("  AST Literal Scanner: PASSED (0 unlisted decimal/percent literals).")
        tests_passed += 1

    # --------------------------------------------------------------------------
    # A2.1 IMMEDIATE EFFICACY VS TERMINAL RETENTION DISAGREEMENT FIXTURE
    # --------------------------------------------------------------------------
    print("\n[A2.1 Immediate Efficacy vs Terminal Retention Disagreement Fixture]")
    # 20 facts injected sequentially:
    # All 20 took immediately upon their own edit: immediate_matches = [True] * 20
    # Later edits overwrite earlier ones, leaving only 4 matching at step 20
    facts_20 = [{"object": f"target_{i}"} for i in range(20)]
    imm_matches_20 = [True] * 20
    preds_step20 = [f"target_{i}" if i >= 16 else "overwritten_target" for i in range(20)]
    
    m_imm = immediate_efficacy(imm_matches_20, input_set="distinct20", mode="eval")
    m_term = terminal_retention(preds_step20, facts_20, input_set="distinct20", mode="eval")
    tests_run += 1
    exp_imm = (20, 20)
    exp_term = (4, 20)
    print(f"  Immediate Efficacy (all took)     : Expected {exp_imm}, Actual {m_imm.pair} -> {m_imm}")
    print(f"  Terminal Retention (4 survived)   : Expected {exp_term}, Actual {m_term.pair} -> {m_term}")
    assert m_imm.pair == exp_imm, f"Immediate efficacy mismatch: {m_imm.pair} != {exp_imm}"
    assert m_term.pair == exp_term, f"Terminal retention mismatch: {m_term.pair} != {exp_term}"
    assert m_imm.pair != m_term.pair, "Defect: immediate_efficacy and terminal_retention are identical!"
    tests_passed += 1

    # A2.2 IMMEDIATE EFFICACY FAILURE ON EXHAUSTED MAX_STEPS
    print("\n[A2.2 Immediate Efficacy Failure on Exhausted Max Steps]")
    # Fact 0 took (matched at step 3) -> True
    # Fact 1 exhausted 25 steps without matching -> False
    # Fact 2 took (matched at step 25) -> True
    imm_matches_3 = [True, False, True]
    m_imm_3 = immediate_efficacy(imm_matches_3, input_set="test3", mode="eval")
    tests_run += 1
    exp_imm_3 = (2, 3)
    print(f"  Immediate Efficacy (1 exhausted)  : Expected {exp_imm_3}, Actual {m_imm_3.pair} -> {m_imm_3}")
    assert m_imm_3.pair == exp_imm_3, f"Mismatch: expected {exp_imm_3}, got {m_imm_3.pair}"
    tests_passed += 1

    # A2.3 DENOMINATOR EQUALITY ASSERTION ACROSS N in {1, 5, 12, 20}
    print("\n[A2.3 Denominator Equality Across N in {1, 5, 12, 20}]")
    tests_run += 1
    for k in [1, 5, 12, 20]:
        k_matches = [True] * k
        k_facts = [{"object": f"target_{i}"} for i in range(k)]
        k_preds = [f"target_{i}" for i in range(k)]
        res_imm = immediate_efficacy(k_matches, input_set=f"test_{k}")
        res_term = terminal_retention(k_preds, k_facts, input_set=f"test_{k}")
        assert res_imm.denominator == k, f"Immediate efficacy denominator {res_imm.denominator} != {k}"
        assert res_term.denominator == k, f"Terminal retention denominator {res_term.denominator} != {k}"
    print("  Denominator Assertion             : Strictly equals N for both metrics across N in [1, 5, 12, 20]")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # A3. SUMMARY STATISTIC DEFECT REGRESSION TEST
    # --------------------------------------------------------------------------
    print("\n[A3. Summary Statistic Reduction Test (S0-2 A3)]")
    # Pinned list from S0-1 incident: 0, 1, 0, 0, 2, 1, 0, 1, 1, 1 (sum=7, mean=0.70)
    incident_values = [0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    stats = compute_summary_stats(incident_values)
    tests_run += 1
    exp_min, exp_max, exp_mean = 0.0, 2.0, 0.70
    print(f"  Summary stats on incident list    : min={stats['min']}, max={stats['max']}, mean={stats['mean']:.4f}")
    assert stats["min"] == exp_min, f"Min mismatch: expected {exp_min}, got {stats['min']}"
    assert stats["max"] == exp_max, f"Max mismatch: expected {exp_max}, got {stats['max']}"
    assert abs(stats["mean"] - exp_mean) < 1e-6, f"Mean mismatch: expected {exp_mean}, got {stats['mean']}"
    tests_passed += 1

    # --------------------------------------------------------------------------
    # 3.2 GENERALIZATION TEST
    # --------------------------------------------------------------------------
    print("\n[3.2 Generalization Metric Unit Tests]")
    para_preds_40_60 = [[f"target_{i}", f"target_{i}", "wrong_paraphrase"] for i in range(20)]
    m_gen = generalization(para_preds_40_60, facts_20)
    tests_run += 1
    exp_gen = (40, 60)
    print(f"  Test 3.2 (20 facts, 2/3 correct)  : Expected {exp_gen}, Actual {m_gen.pair} -> {m_gen}")
    assert m_gen.pair == exp_gen, f"Mismatch: expected {exp_gen}, got {m_gen.pair}"
    tests_passed += 1
    
    # --------------------------------------------------------------------------
    # 3.3 THREE RETENTION METRICS (HAND-CONSTRUCTED FIXTURE)
    # --------------------------------------------------------------------------
    print("\n[3.3 Retention Metrics Unit Tests on Hand-Counted Fixture]")
    fixture_facts = [
        {"fact_id": 0, "relation": "born_city", "object": "Paris"},
        {"fact_id": 1, "relation": "born_city", "object": "Berlin"},
        {"fact_id": 2, "relation": "born_city", "object": "Rome"},
        {"fact_id": 3, "relation": "instrument", "object": "piano"},
        {"fact_id": 4, "relation": "instrument", "object": "flute"}
    ]
    fixture_preds = ["Paris", "Berlin", "Paris", "piano", "drums"]
    fixture_rel_modals = {"born_city": "Paris", "instrument": "violin"}
    fixture_ctrl_preds = {
        "born_city": ["Paris", "Berlin", "Berlin", "Berlin", "Madrid"],
        "instrument": ["violin", "violin", "guitar"]
    }
    
    m_raw = raw_retention(fixture_preds, fixture_facts)
    tests_run += 1
    exp_raw = (3, 5)
    print(f"  Test 3.3a (Terminal/Raw Retention): Expected {exp_raw}, Actual {m_raw.pair} -> {m_raw}")
    assert m_raw.pair == exp_raw, f"Raw retention mismatch: expected {exp_raw}, got {m_raw.pair}"
    tests_passed += 1
    
    m_bound = bound_retention(fixture_preds, fixture_facts, fixture_rel_modals)
    tests_run += 1
    exp_bound = (2, 5)
    print(f"  Test 3.3b (Bound Retention)      : Expected {exp_bound}, Actual {m_bound.pair} -> {m_bound}")
    assert m_bound.pair == exp_bound, f"Bound retention mismatch: expected {exp_bound}, got {m_bound.pair}"
    tests_passed += 1
    
    m_disc = subject_discriminable_retention(fixture_preds, fixture_facts, fixture_ctrl_preds, max_shared_controls=2)
    tests_run += 1
    exp_disc = (2, 5)
    print(f"  Test 3.3c (Subj-Discrim Ret)     : Expected {exp_disc}, Actual {m_disc.pair} -> {m_disc}")
    assert m_disc.pair == exp_disc, f"Subj-discrim retention mismatch: expected {exp_disc}, got {m_disc.pair}"
    tests_passed += 1
    
    # --------------------------------------------------------------------------
    # 3.4 STOPPING RULE UNIT TESTS
    # --------------------------------------------------------------------------
    print("\n[3.4 Stopping Rule Simulation Tests]")
    target_token = "Rome"
    steps_a = ["Paris", "Berlin", "Rome", "Rome", "Rome"]
    steps_taken_a, succ_a = simulate_stopping_rule(steps_a, target_token, max_steps=25)
    tests_run += 1
    print(f"  Test 3.4a (Terminates on match)   : Expected (3, True), Actual ({steps_taken_a}, {succ_a})")
    assert (steps_taken_a, succ_a) == (3, True)
    tests_passed += 1
    
    steps_b = ["Rome", "Rome", "Rome"]
    steps_taken_b, succ_b = simulate_stopping_rule(steps_b, target_token, max_steps=25)
    tests_run += 1
    print(f"  Test 3.4b (Terminates at step 1)  : Expected (1, True), Actual ({steps_taken_b}, {succ_b})")
    assert (steps_taken_b, succ_b) == (1, True)
    tests_passed += 1
    
    steps_c = ["Paris"] * 30
    steps_taken_c, succ_c = simulate_stopping_rule(steps_c, target_token, max_steps=15)
    tests_run += 1
    print(f"  Test 3.4c (Runs to max_steps)     : Expected (15, False), Actual ({steps_taken_c}, {succ_c})")
    assert (steps_taken_c, succ_c) == (15, False)
    tests_passed += 1
    
    steps_d = ["token_loss_5.0", "token_loss_3.0", "token_loss_1.0", "token_loss_0.5"]
    steps_taken_d, succ_d = simulate_stopping_rule(steps_d, target_token, max_steps=4)
    tests_run += 1
    print(f"  Test 3.4d (Falling loss no match) : Expected (4, False), Actual ({steps_taken_d}, {succ_d})")
    assert (steps_taken_d, succ_d) == (4, False)
    tests_passed += 1
    
    # --------------------------------------------------------------------------
    # 3.5 CHECK_MATCH & NORMALIZE_ENTITY: 12-CASE PROVEN EQUIVALENCE SUITE
    # --------------------------------------------------------------------------
    print("\n[3.5 Proven 12-Case Match & Normalization Equivalence Suite]")
    test_12_suite = [
        (" Canberra on 23 April 1946", "Canberra", True, "canberra"),
        (" oboe.", "oboe", True, "oboe"),
        (" Oslo, Oslo Oslo Oslo", "Oslo", True, "oslo"),
        (" photographer and astronomer. She", "photographer", True, "photographer"),
        (" accordion. He plays", "accordion", True, "accordion"),
        (" Rome.", "Rome", True, "rome"),
        (" Rome", "Rome", True, "rome"),
        (" Osloman", "Oslo", False, "osloman"),
        (" Romeo", "Rome", False, "romeo"),
        (" Cairo", "Canberra", False, "cairo"),
        (" New Yorker", "New York", False, "new"),
        (" New York, USA", "New York", True, "new")
    ]
    for idx, (t_pred, t_tgt, exp_m, exp_n) in enumerate(test_12_suite):
        tests_run += 1
        act_m = check_match(t_pred, t_tgt)
        act_n = normalize_entity(t_pred)
        assert act_m == exp_m and act_n == exp_n
        tests_passed += 1
    print(f"  Test 3.5 (12-Case Equivalence)    : All 12 proven test cases PASSED identically.")
    
    # --------------------------------------------------------------------------
    # 3.6 CONTROL ACCOUNTING & IMPOSSIBLE-VALUE GUARD
    # --------------------------------------------------------------------------
    print("\n[3.6 Control Accounting & Impossible-Value Guard Tests]")
    valid_controls = {
        "never_edited": Measurement("never_edited", 0, 20),
        "random_direction_magnitude_matched": Measurement("random_direction_magnitude_matched", 1, 20),
        "wrong_target": Measurement("wrong_target", 0, 20),
        "pre_edit_baseline": Measurement("pre_edit_baseline", 2, 20)
    }
    pooled_m, worst_m, exp_sum_str = pool_controls(valid_controls)
    tests_run += 1
    print(f"  Test 3.6a (Valid Control Pooling) : Expanded Sum -> {exp_sum_str}")
    print(f"                                      Pooled Floor -> {pooled_m}, Worst -> {worst_m.name}: {worst_m}")
    assert pooled_m.pair == (3, 80)
    assert worst_m.name == "pre_edit_baseline" and worst_m.pair == (2, 20)
    tests_passed += 1
    
    tests_run += 1
    impossible_caught = False
    try:
        invalid_measurement = Measurement("faulty_control", 21, 20)
    except ValueError as e:
        impossible_caught = True
        print(f"  Test 3.6b (Numerator > Denominator): Correctly halted with ValueError: {e}")
    assert impossible_caught, "FAIL: Impossible value (num > den) failed to raise ValueError!"
    tests_passed += 1
    
    # --------------------------------------------------------------------------
    # PART 1: PROVENANCE GUARD TESTS (DIRECTIVE S0-4 PART 1.3 & 1.4)
    # --------------------------------------------------------------------------
    print("\n[PART 1: Provenance Guard Tests (Directive S0-4 Part 1)]")
    
    # 1.3a Control pool numerator with arm population denominator raises
    tests_run += 1
    caught_1_3a = False
    try:
        Measurement("terminal_retention", 33, 600, arm="controls_pool", metric="terminal_retention")
    except ValueError as e:
        caught_1_3a = True
        print(f"  Test 1.3a (Control-pool retention raises)   : Caught expected ValueError: {e}")
    assert caught_1_3a, "FAIL: Constructing retention on controls_pool with arm population denominator failed to raise!"
    tests_passed += 1

    # 1.3b Numerator 21 of denominator 20 raises
    tests_run += 1
    caught_1_3b = False
    try:
        Measurement("faulty_control", 21, 20)
    except ValueError as e:
        caught_1_3b = True
        print(f"  Test 1.3b (num 21 > den 20 raises)         : Caught expected ValueError: {e}")
    assert caught_1_3b, "FAIL: Impossible value (num > den) failed to raise ValueError!"
    tests_passed += 1

    # 1.3c Legitimate measurement round-trips through renderer unchanged
    tests_run += 1
    m_valid = Measurement("terminal_retention", 33, 600, arm="r0_unconstrained", metric="terminal_retention")
    rendered = format_wilson_rate(m_valid)
    print(f"  Test 1.3c (Renderer round-trip)            : {rendered}")
    assert rendered.startswith("33/600 (5.50%) [")
    # Rendering bare integers raises TypeError
    caught_type_err = False
    try:
        format_wilson_rate((33, 600))  # type: ignore
    except TypeError:
        caught_type_err = True
    assert caught_type_err, "FAIL: format_wilson_rate on bare tuple failed to raise TypeError!"
    tests_passed += 1

    # 1.4 Grep harness and report generator for loose-variable rate formatting
    print("\n[1.4 Loose-Variable Rate Formatting Grep Audit]")
    tests_run += 1
    loose_violations = 0
    harness_p = REPO_ROOT / "experiments" / "b1_inject.py"
    report_p = REPO_ROOT / "tools" / "make_report.py"
    for target_p in [harness_p, report_p]:
        if not target_p.exists(): continue
        t_tree = ast.parse(target_p.read_text(encoding="utf-8"), filename=str(target_p))
        for t_node in ast.walk(t_tree):
            if isinstance(t_node, ast.Call):
                f_name = ""
                if isinstance(t_node.func, ast.Name): f_name = t_node.func.id
                elif isinstance(t_node.func, ast.Attribute): f_name = t_node.func.attr
                if f_name == "format_wilson_rate":
                    if len(t_node.args) >= 2 and not (isinstance(t_node.args[1], ast.Constant) and isinstance(t_node.args[1].value, float)):
                        loose_violations += 1
                        print(f"    VIOLATION in {target_p.name}:{t_node.lineno}: format_wilson_rate called with loose variables")
    print(f"  Test 1.4 (Loose-variable rate formatting)  : {loose_violations} violation(s) detected.")
    assert loose_violations == 0, f"FAIL: {loose_violations} loose-variable rate formatting site(s) found!"
    tests_passed += 1

    # --------------------------------------------------------------------------
    # PART 2: SURVIVING-FRACTION & ALIGNMENT UNIT TESTS (DIRECTIVE S0-4 PART 2)
    # --------------------------------------------------------------------------
    print("\n[PART 2: Surviving-Fraction & Alignment Unit Tests (Directive S0-4 Part 2)]")
    
    # 2.2a Delta lying entirely inside subspace -> 0.0000
    Q_1d = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).T
    delta_in = torch.tensor([2.0, 0.0, 0.0, 0.0])
    sf_in = compute_surviving_fraction(delta_in, Q_1d)
    tests_run += 1
    print(f"  Test 2.2a (Delta inside subspace)          : Expected 0.0000, Actual {sf_in:.4f}")
    assert abs(sf_in - 0.0000) < 1e-4
    tests_passed += 1

    # 2.2b Delta entirely orthogonal to subspace -> 1.0000
    delta_orth = torch.tensor([0.0, 3.0, 0.0, 0.0])
    sf_orth = compute_surviving_fraction(delta_orth, Q_1d)
    tests_run += 1
    print(f"  Test 2.2b (Delta orthogonal to subspace)   : Expected 1.0000, Actual {sf_orth:.4f}")
    assert abs(sf_orth - 1.0000) < 1e-4
    tests_passed += 1

    # 2.2c Delta at 45 degrees to rank-1 subspace -> 0.7071
    delta_45 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    sf_45 = compute_surviving_fraction(delta_45, Q_1d)
    tests_run += 1
    print(f"  Test 2.2c (Delta at 45 deg to rank-1)      : Expected 0.7071, Actual {sf_45:.4f}")
    assert abs(sf_45 - 0.7071) < 1e-3
    tests_passed += 1

    # 2.2d Delta with 3 of 4 unit components inside rank-3 subspace -> 0.5000
    Q_3d = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).T
    delta_3of4 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sf_3of4 = compute_surviving_fraction(delta_3of4, Q_3d)
    tests_run += 1
    print(f"  Test 2.2d (3 of 4 unit in rank-3)          : Expected 0.5000, Actual {sf_3of4:.4f}")
    assert abs(sf_3of4 - 0.5000) < 1e-4
    tests_passed += 1

    # 2.3 Alignment diagnostic on synthetic vectors with hand-computed cosines (1.0, 0.0, 0.7071)
    tests_run += 1
    u1_mock = Q_1d
    align_1 = compute_alignment(torch.tensor([2.0, 0.0, 0.0, 0.0]), u1_mock)
    align_0 = compute_alignment(torch.tensor([0.0, 3.0, 0.0, 0.0]), u1_mock)
    align_45 = compute_alignment(torch.tensor([1.0, 1.0, 0.0, 0.0]), u1_mock)
    print(f"  Test 2.3 (Alignment cosines)               : Expected (1.0000, 0.0000, 0.7071), Actual ({align_1:.4f}, {align_0:.4f}, {align_45:.4f})")
    assert abs(align_1 - 1.0000) < 1e-4
    assert abs(align_0 - 0.0000) < 1e-4
    assert abs(align_45 - 0.7071) < 1e-3
    tests_passed += 1

    # 2.4 Pythagorean projection identity check
    tests_run += 1
    assert_pythagorean_projection(delta_45, Q_1d)
    assert_pythagorean_projection(delta_3of4, Q_3d)
    print("  Test 2.4 (Pythagorean projection identity) : PASSED on synthetic test cases")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # 3.7 WILSON CONFIDENCE INTERVAL UNIT TESTS
    # --------------------------------------------------------------------------
    print("\n[3.7 Wilson Score Confidence Interval Unit Tests]")
    lo_4, hi_4 = wilson_confidence_interval(4, 20)
    tests_run += 1
    print(f"  Test 3.7a (Wilson k=4, n=20)       : lo={lo_4:.4f}, hi={hi_4:.4f}")
    assert 0.080 <= lo_4 <= 0.082 and 0.415 <= hi_4 <= 0.417
    tests_passed += 1

    lo_0, hi_0 = wilson_confidence_interval(0, 20)
    tests_run += 1
    print(f"  Test 3.7b (Wilson k=0, n=20)       : lo={lo_0:.4f}, hi={hi_0:.4f}")
    assert lo_0 == 0.0 and 0.160 <= hi_0 <= 0.162
    tests_passed += 1

    # --------------------------------------------------------------------------
    # 3.8 STATISTICAL POWER SAMPLING UNIT TESTS
    # --------------------------------------------------------------------------
    print("\n[3.8 N=200 Sequence Sampling Unit Tests]")
    mock_facts = [{"fact_id": i, "subject": f"Subj_{i}", "relation": "born_city", "object": f"City_{i}"} for i in range(1000)]
    seq_0, hash_0 = sample_200_facts(mock_facts, seed=0)
    seq_1, hash_1 = sample_200_facts(mock_facts, seed=1)
    tests_run += 1
    print(f"  Test 3.8a (N=200 seed=0 hash)      : {hash_0[:16]}... (200 facts)")
    assert len(seq_0) == 200 and len(set(f["fact_id"] for f in seq_0)) == 200
    assert len(seq_1) == 200 and hash_0 != hash_1
    tests_passed += 1

    # --------------------------------------------------------------------------
    # 3.9 CAUSAL SUBSPACE ORTHOGONAL PROJECTION TESTS
    # --------------------------------------------------------------------------
    print("\n[3.9 Causal Subspace Orthogonal Projection Tests]")
    torch.manual_seed(42)
    dim = 64
    rank = 4
    basis = torch.randn(dim, rank)
    Q, _ = torch.linalg.qr(basis)
    grad = torch.randn(10, dim)
    proj_comp = (grad @ Q) @ Q.T
    grad_proj = grad - proj_comp
    overlap = torch.norm(grad_proj @ Q).item()
    tests_run += 1
    print(f"  Test 3.9a (Orthogonality norm)     : {overlap:.8f} (Expected near zero)")
    assert overlap < 1e-5
    tests_passed += 1

    # --------------------------------------------------------------------------
    # 3.10 TEST SUITE SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f" PRE-FLIGHT TEST SUMMARY: {tests_run} tests run, {tests_passed} tests passed, 0 failures.")
    print("=" * 100)
    assert tests_run > 0, "Test suite failure: zero tests run"
    assert tests_run == tests_passed, f"Test suite failure: {tests_run - tests_passed} tests failed"
    return 0

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
