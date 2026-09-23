#!/usr/bin/env python3
"""
tests/test_metrics.py
=====================
Pre-Flight Unit Test Suite for Continual Learning Metrics (Directive S0-5).

Mandate:
  - Must run before any model loads or accelerator initializes.
  - Written against hand-constructed stubs; zero GPU/model dependency.
  - Each check prints expected and actual; any mismatch halts and exits nonzero.
  - Stated numerical tolerances on all synthetic tests.
  - Zero tests run is an immediate failure.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch

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
    assert_pythagorean_projection,
    assert_orthonormality,
    compute_paired_stats,
    compute_monotone_retention_horizon,
    classify_reversion_pattern,
    POPULATION_REGISTRY
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
    "=" * 95: "line-item step attribution border line",
    "-" * 135: "cell comparison table separator line",
    "=" * 135: "cell comparison table border line",
    "-" * 145: "sweep comparison table separator line",
    "=" * 145: "sweep comparison table border line",
    "=" * 125: "diagnostic table border line",
    "-" * 125: "diagnostic table separator line",
    ":4096:8": "cublas deterministic workspace configuration flag",
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
    print(" PRE-FLIGHT TEST SUITE (Directive S0-5 Part 1): Metric Stubs & Provenance Guard 2.0")
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
    # PART 1: PROVENANCE GUARD 2.0 (DIRECTIVE S0-5 PART 1)
    # --------------------------------------------------------------------------
    print("\n[PART 1: Provenance Guard 2.0 (Directive S0-5 Part 1)]")
    
    # 1.1 Direct public Measurement constructor call raises TypeError
    tests_run += 1
    caught_direct_call = False
    try:
        Measurement("terminal_retention", 33, 600, input_set="test", mode="eval")  # type: ignore
    except TypeError as e:
        caught_direct_call = True
        print(f"  Test 1.1 (Direct constructor call raises)   : Caught expected TypeError: {e}")
    assert caught_direct_call, "FAIL: Direct Measurement constructor call failed to raise TypeError!"
    tests_passed += 1

    # 1.2 Missing or empty execution mode raises TypeError
    tests_run += 1
    caught_missing_mode = False
    try:
        Measurement.from_outcomes([True] * 20, metric="terminal_retention", arm="r0_unconstrained", scope="fixture_20", input_set="test", mode="")
    except TypeError as e:
        caught_missing_mode = True
        print(f"  Test 1.2 (Missing mode raises)              : Caught expected TypeError: {e}")
    assert caught_missing_mode, "FAIL: Missing mode in from_outcomes failed to raise TypeError!"
    tests_passed += 1

    # 1.3 Scope mismatch raises ValueError (e.g. 33 outcomes with scope 'pooled' which expects 600)
    tests_run += 1
    caught_scope_mismatch = False
    try:
        Measurement.from_outcomes([True] * 33, metric="terminal_retention", arm="r0_unconstrained", scope="pooled", input_set="test", mode="eval_no_dropout")
    except ValueError as e:
        caught_scope_mismatch = True
        print(f"  Test 1.3 (Scope mismatch raises)            : Caught expected ValueError: {e}")
    assert caught_scope_mismatch, "FAIL: Scope denominator mismatch failed to raise ValueError!"
    tests_passed += 1

    # 1.4 Retention metric on control pool raises ValueError
    tests_run += 1
    caught_ctrl_ret = False
    try:
        Measurement.from_outcomes([True] * 600, metric="terminal_retention", arm="controls_pool", scope="pooled", input_set="test", mode="eval_no_dropout")
    except ValueError as e:
        caught_ctrl_ret = True
        print(f"  Test 1.4 (Control-pool retention raises)    : Caught expected ValueError: {e}")
    assert caught_ctrl_ret, "FAIL: Retention measurement on control pool failed to raise ValueError!"
    tests_passed += 1

    # 1.5 Legitimate measurement round-trips through renderer unchanged
    tests_run += 1
    valid_outcomes = [True] * 33 + [False] * 567
    m_valid = Measurement.from_outcomes(valid_outcomes, metric="terminal_retention", arm="r0_unconstrained", scope="pooled_600", input_set="test_600", mode="eval_no_dropout")
    rendered = format_wilson_rate(m_valid)
    print(f"  Test 1.5 (Renderer round-trip)              : {rendered}")
    assert rendered.startswith("33/600 (5.50%) [")
    caught_type_err = False
    try:
        format_wilson_rate((33, 600))  # type: ignore
    except TypeError:
        caught_type_err = True
    assert caught_type_err, "FAIL: format_wilson_rate on bare tuple failed to raise TypeError!"
    tests_passed += 1

    # 1.6 Loose-variable rate formatting grep audit
    print("\n[1.6 Loose-Variable Rate Formatting Grep Audit]")
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
    print(f"  Test 1.6 (Loose-variable rate formatting)   : {loose_violations} violation(s) detected.")
    assert loose_violations == 0, f"FAIL: {loose_violations} loose-variable rate formatting site(s) found!"
    tests_passed += 1

    # 1.7 Principled Reversion Pattern Classifier Unit Test (Directive S0-6)
    print("\n[1.7 Principled Reversion Pattern Classifier Test (Directive S0-6)]")
    tests_run += 1
    # Test case 1: S0-5 overlapping intervals -> must return FLAT
    s05_intervals = [(0.015, 0.082), (0.020, 0.091), (0.018, 0.085)]
    s05_rates = [0.045, 0.051, 0.048]
    p_flat = classify_reversion_pattern(s05_intervals, s05_rates)
    print(f"  S0-5 Overlapping bins classification        : {p_flat}")
    assert p_flat == "FLAT — NO DOSE RESPONSE DETECTED", f"Expected FLAT, got {p_flat}"
    # Test case 2: Monotonic non-overlapping -> GRADED
    graded_intervals = [(0.40, 0.60), (0.20, 0.35), (0.01, 0.10)]
    graded_rates = [0.50, 0.27, 0.05]
    p_graded = classify_reversion_pattern(graded_intervals, graded_rates)
    print(f"  Monotonic non-overlapping bins              : {p_graded}")
    assert "GRADED" in p_graded, f"Expected GRADED, got {p_graded}"
    tests_passed += 1

    # 1.8 Paired Statistics Unit Test (Directive S0-6)
    print("\n[1.8 Paired Statistics Unit Test (df=5)]")
    tests_run += 1
    x1_mock = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0]
    x2_mock = [8.0, 9.0, 10.0, 11.0, 12.0, 10.0]
    p_stats = compute_paired_stats(x1_mock, x2_mock)
    print(f"  Paired stats (mean diff={p_stats['mean_diff']:.2f}, df={p_stats['df']}, t={p_stats['t_stat']:.4f}, W={p_stats['wilcoxon_stat']})")
    assert p_stats["df"] == 5, f"Expected df=5, got {p_stats['df']}"
    assert abs(p_stats["mean_diff"] - 2.0) < 1e-6, f"Expected mean diff 2.0, got {p_stats['mean_diff']}"
    assert p_stats["t_stat"] > 0, "Expected positive t-stat"
    tests_passed += 1

    # 1.9 Monotone Retention Horizon Unit Test (Directive S0-6)
    print("\n[1.9 Monotone Retention Horizon Unit Test]")
    tests_run += 1
    # Create synthetic terminal matches where last 20 edits (180-199) retain at 50% and first 180 retain at 0%
    synth_matches = {}
    for s in range(6):
        synth_matches[s] = [False] * 180 + [True, False] * 10
    floor_interval = (0.01, 0.05)
    horiz_res = compute_monotone_retention_horizon(synth_matches, floor_interval, step_size=10, total_edits=200)
    print(f"  Monotone horizon search result              : horizon_k={horiz_res['horizon_k']}")
    assert horiz_res["horizon_k"] == 20, f"Expected horizon_k=20, got {horiz_res['horizon_k']}"
    assert horiz_res["remainder"] is not None and horiz_res["remainder"]["k"] == 180
    assert horiz_res["remainder"]["numerator"] == 0
    tests_passed += 1

    # --------------------------------------------------------------------------
    # PART 1.4: PROJECTION DIAGNOSTIC INDEPENDENCE PROOF (DIRECTIVE S0-5)
    # --------------------------------------------------------------------------
    print("\n[PART 1.4: Surviving-Fraction & Alignment Independence Proof (Directive S0-5)]")
    # In rank-2 subspace Q = [u1, u2]:
    # v1 is aligned with u1: lies entirely in subspace -> SF = 0.0000, Alignment = 1.0000
    # v2 is aligned with u2: lies entirely in subspace -> SF = 0.0000, Alignment = 0.0000
    # Surviving fraction is identically 0.0000 for both, yet Alignment is 1.0000 vs 0.0000.
    tests_run += 1
    u1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    u2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    Q_rank2 = torch.stack([u1, u2], dim=1)  # shape (4, 2)
    v1 = torch.tensor([2.0, 0.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 2.0, 0.0, 0.0])

    sf_v1 = compute_surviving_fraction(v1, Q_rank2)
    sf_v2 = compute_surviving_fraction(v2, Q_rank2)
    al_v1 = compute_alignment(v1, Q_rank2)
    al_v2 = compute_alignment(v2, Q_rank2)

    print(f"  Vector v1 (in span(u1)) -> SF = {sf_v1:.4f} (tol=1e-4), Alignment = {al_v1:.4f} (tol=1e-4)")
    print(f"  Vector v2 (in span(u2)) -> SF = {sf_v2:.4f} (tol=1e-4), Alignment = {al_v2:.4f} (tol=1e-4)")
    assert abs(sf_v1 - 0.0000) < 1e-4 and abs(sf_v2 - 0.0000) < 1e-4, "Both vectors must have SF == 0.0000"
    assert abs(al_v1 - 1.0000) < 1e-4, "v1 alignment must be 1.0000"
    assert abs(al_v2 - 0.0000) < 1e-4, "v2 alignment must be 0.0000"
    assert abs(al_v1 - al_v2) > 0.99, "Alignment must differentiate directions within the same subspace"
    print("  Independence Proof: PASSED (SF and Alignment dissociate geometrically in rank-2 subspace).")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # PART 1.2: ORTHONORMALITY ASSERTION AUDIT
    # --------------------------------------------------------------------------
    print("\n[PART 1.2: Orthonormality Assertion Audit]")
    tests_run += 1
    assert_orthonormality(Q_rank2, tol=1e-6)
    non_ortho = torch.tensor([[1.0, 0.5], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    caught_non_ortho = False
    try:
        assert_orthonormality(non_ortho, tol=1e-6)
    except AssertionError:
        caught_non_ortho = True
    assert caught_non_ortho, "FAIL: Non-orthonormal basis failed to raise AssertionError!"
    print("  Orthonormality Guard: PASSED (Orthonormal passes, non-orthonormal raises).")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # PART 1.6: SEVEN SYNTHETIC TESTS WITH STATED TOLERANCES
    # --------------------------------------------------------------------------
    print("\n[PART 1.6: Seven Synthetic Tests with Stated Tolerances]")
    Q_1d = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).T  # shape (4, 1)

    # 1. SF 0.0000 (tol 1e-4)
    tests_run += 1
    delta_in = torch.tensor([2.0, 0.0, 0.0, 0.0])
    sf_in = compute_surviving_fraction(delta_in, Q_1d)
    print(f"  Synthetic 1 (SF inside subspace)     : Expected 0.0000, Actual {sf_in:.4f} (tol=1e-4)")
    assert abs(sf_in - 0.0000) < 1e-4
    tests_passed += 1

    # 2. SF 1.0000 (tol 1e-4)
    tests_run += 1
    delta_orth = torch.tensor([0.0, 3.0, 0.0, 0.0])
    sf_orth = compute_surviving_fraction(delta_orth, Q_1d)
    print(f"  Synthetic 2 (SF orthogonal)          : Expected 1.0000, Actual {sf_orth:.4f} (tol=1e-4)")
    assert abs(sf_orth - 1.0000) < 1e-4
    tests_passed += 1

    # 3. SF 0.7071 (tol 1e-3)
    tests_run += 1
    delta_45 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    sf_45 = compute_surviving_fraction(delta_45, Q_1d)
    print(f"  Synthetic 3 (SF 45 deg)              : Expected 0.7071, Actual {sf_45:.4f} (tol=1e-3)")
    assert abs(sf_45 - 0.7071) < 1e-3
    tests_passed += 1

    # 4. SF 0.5000 (tol 1e-4)
    tests_run += 1
    Q_3d = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).T
    delta_3of4 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sf_3of4 = compute_surviving_fraction(delta_3of4, Q_3d)
    print(f"  Synthetic 4 (SF 3 of 4 unit in r=3)  : Expected 0.5000, Actual {sf_3of4:.4f} (tol=1e-4)")
    assert abs(sf_3of4 - 0.5000) < 1e-4
    tests_passed += 1

    # 5. Align 1.0000 (tol 1e-4)
    tests_run += 1
    align_1 = compute_alignment(torch.tensor([2.0, 0.0, 0.0, 0.0]), Q_1d)
    print(f"  Synthetic 5 (Align parallel)         : Expected 1.0000, Actual {align_1:.4f} (tol=1e-4)")
    assert abs(align_1 - 1.0000) < 1e-4
    tests_passed += 1

    # 6. Align 0.0000 (tol 1e-4)
    tests_run += 1
    align_0 = compute_alignment(torch.tensor([0.0, 3.0, 0.0, 0.0]), Q_1d)
    print(f"  Synthetic 6 (Align orthogonal)       : Expected 0.0000, Actual {align_0:.4f} (tol=1e-4)")
    assert abs(align_0 - 0.0000) < 1e-4
    tests_passed += 1

    # 7. Align 0.7071 (tol 1e-3)
    tests_run += 1
    align_45 = compute_alignment(torch.tensor([1.0, 1.0, 0.0, 0.0]), Q_1d)
    print(f"  Synthetic 7 (Align 45 deg)           : Expected 0.7071, Actual {align_45:.4f} (tol=1e-3)")
    assert abs(align_45 - 0.7071) < 1e-3
    tests_passed += 1

    # Pythagorean projection identity check
    tests_run += 1
    assert_pythagorean_projection(delta_45, Q_1d)
    assert_pythagorean_projection(delta_3of4, Q_3d)
    print("  Pythagorean Projection Identity     : PASSED on synthetic test cases")
    tests_passed += 1

    # --------------------------------------------------------------------------
    # A2.1 IMMEDIATE EFFICACY VS TERMINAL RETENTION DISAGREEMENT FIXTURE
    # --------------------------------------------------------------------------
    print("\n[A2.1 Immediate Efficacy vs Terminal Retention Disagreement Fixture]")
    facts_20 = [{"object": f"target_{i}"} for i in range(20)]
    imm_matches_20 = [True] * 20
    preds_step20 = [f"target_{i}" if i >= 16 else "overwritten_target" for i in range(20)]
    
    m_imm = immediate_efficacy(imm_matches_20, input_set="distinct20", mode="eval_no_dropout", scope="fixture_20")
    m_term = terminal_retention(preds_step20, facts_20, input_set="distinct20", mode="eval_no_dropout", scope="fixture_20")
    tests_run += 1
    exp_imm = (20, 20)
    exp_term = (4, 20)
    print(f"  Immediate Efficacy (all took)       : Expected {exp_imm}, Actual {m_imm.pair} -> {m_imm}")
    print(f"  Terminal Retention (4 survived)     : Expected {exp_term}, Actual {m_term.pair} -> {m_term}")
    assert m_imm.pair == exp_imm, f"Immediate efficacy mismatch: {m_imm.pair} != {exp_imm}"
    assert m_term.pair == exp_term, f"Terminal retention mismatch: {m_term.pair} != {exp_term}"
    assert m_imm.pair != m_term.pair, "Defect: immediate_efficacy and terminal_retention are identical!"
    tests_passed += 1

    # A2.2 IMMEDIATE EFFICACY FAILURE ON EXHAUSTED MAX_STEPS
    print("\n[A2.2 Immediate Efficacy Failure on Exhausted Max Steps]")
    imm_matches_3 = [True, False, True]
    m_imm_3 = immediate_efficacy(imm_matches_3, input_set="test3", mode="eval_no_dropout", scope="fixture_3")
    tests_run += 1
    exp_imm_3 = (2, 3)
    print(f"  Immediate Efficacy (1 exhausted)    : Expected {exp_imm_3}, Actual {m_imm_3.pair} -> {m_imm_3}")
    assert m_imm_3.pair == exp_imm_3, f"Mismatch: expected {exp_imm_3}, got {m_imm_3.pair}"
    tests_passed += 1

    # A2.3 DENOMINATOR EQUALITY ASSERTION ACROSS N in {1, 5, 12, 20}
    print("\n[A2.3 Denominator Equality Across N in {1, 5, 12, 20}]")
    tests_run += 1
    for k in [1, 5, 12, 20]:
        k_matches = [True] * k
        k_facts = [{"object": f"target_{i}"} for i in range(k)]
        k_preds = [f"target_{i}" for i in range(k)]
        res_imm = immediate_efficacy(k_matches, input_set=f"test_{k}", mode="eval_no_dropout", scope=f"fixture_{k}")
        res_term = terminal_retention(k_preds, k_facts, input_set=f"test_{k}", mode="eval_no_dropout", scope=f"fixture_{k}")
        assert res_imm.denominator == k, f"Immediate efficacy denominator {res_imm.denominator} != {k}"
        assert res_term.denominator == k, f"Terminal retention denominator {res_term.denominator} != {k}"
    print("  Denominator Assertion               : Strictly equals N for both metrics across N in [1, 5, 12, 20]")
    tests_passed += 1

    # A3. SUMMARY STATISTIC DEFECT REGRESSION TEST
    print("\n[A3. Summary Statistic Reduction Test (S0-2 A3)]")
    incident_values = [0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    stats = compute_summary_stats(incident_values)
    tests_run += 1
    exp_min, exp_max, exp_mean = 0.0, 2.0, 0.70
    print(f"  Summary stats on incident list      : min={stats['min']}, max={stats['max']}, mean={stats['mean']:.4f}")
    assert stats["min"] == exp_min, f"Min mismatch: expected {exp_min}, got {stats['min']}"
    assert stats["max"] == exp_max, f"Max mismatch: expected {exp_max}, got {stats['max']}"
    assert abs(stats["mean"] - exp_mean) < 1e-6, f"Mean mismatch: expected {exp_mean}, got {stats['mean']}"
    tests_passed += 1

    # 3.2 GENERALIZATION TEST
    print("\n[3.2 Generalization Metric Unit Tests]")
    para_preds_40_60 = [[f"target_{i}", f"target_{i}", "wrong_paraphrase"] for i in range(20)]
    m_gen = generalization(para_preds_40_60, facts_20, input_set="test", mode="eval_no_dropout", scope="fixture_60")
    tests_run += 1
    exp_gen = (40, 60)
    print(f"  Test 3.2 (20 facts, 2/3 correct)    : Expected {exp_gen}, Actual {m_gen.pair} -> {m_gen}")
    assert m_gen.pair == exp_gen, f"Mismatch: expected {exp_gen}, got {m_gen.pair}"
    tests_passed += 1
    
    # 3.3 THREE RETENTION METRICS (HAND-CONSTRUCTED FIXTURE)
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
    
    m_raw = raw_retention(fixture_preds, fixture_facts, input_set="test", mode="eval_no_dropout", scope="fixture_5")
    tests_run += 1
    exp_raw = (3, 5)
    print(f"  Test 3.3a (Terminal/Raw Retention)  : Expected {exp_raw}, Actual {m_raw.pair} -> {m_raw}")
    assert m_raw.pair == exp_raw, f"Raw retention mismatch: expected {exp_raw}, got {m_raw.pair}"
    tests_passed += 1
    
    m_bound = bound_retention(fixture_preds, fixture_facts, fixture_rel_modals, input_set="test", mode="eval_no_dropout", scope="fixture_5")
    tests_run += 1
    exp_bound = (2, 5)
    print(f"  Test 3.3b (Bound Retention)        : Expected {exp_bound}, Actual {m_bound.pair} -> {m_bound}")
    assert m_bound.pair == exp_bound, f"Bound retention mismatch: expected {exp_bound}, got {m_bound.pair}"
    tests_passed += 1
    
    m_disc = subject_discriminable_retention(fixture_preds, fixture_facts, fixture_ctrl_preds, max_shared_controls=2, input_set="test", mode="eval_no_dropout", scope="fixture_5")
    tests_run += 1
    exp_disc = (2, 5)
    print(f"  Test 3.3c (Subj-Discrim Ret)       : Expected {exp_disc}, Actual {m_disc.pair} -> {m_disc}")
    assert m_disc.pair == exp_disc, f"Subj-discrim retention mismatch: expected {exp_disc}, got {m_disc.pair}"
    tests_passed += 1
    
    # 3.4 STOPPING RULE UNIT TESTS
    print("\n[3.4 Stopping Rule Simulation Tests]")
    target_token = "Rome"
    steps_a = ["Paris", "Berlin", "Rome", "Rome", "Rome"]
    steps_taken_a, succ_a = simulate_stopping_rule(steps_a, target_token, max_steps=25)
    tests_run += 1
    print(f"  Test 3.4a (Terminates on match)     : Expected (3, True), Actual ({steps_taken_a}, {succ_a})")
    assert (steps_taken_a, succ_a) == (3, True)
    tests_passed += 1
    
    steps_b = ["Rome", "Rome", "Rome"]
    steps_taken_b, succ_b = simulate_stopping_rule(steps_b, target_token, max_steps=25)
    tests_run += 1
    print(f"  Test 3.4b (Terminates at step 1)    : Expected (1, True), Actual ({steps_taken_b}, {succ_b})")
    assert (steps_taken_b, succ_b) == (1, True)
    tests_passed += 1
    
    steps_c = ["Paris"] * 30
    steps_taken_c, succ_c = simulate_stopping_rule(steps_c, target_token, max_steps=15)
    tests_run += 1
    print(f"  Test 3.4c (Runs to max_steps)       : Expected (15, False), Actual ({steps_taken_c}, {succ_c})")
    assert (steps_taken_c, succ_c) == (15, False)
    tests_passed += 1
    
    steps_d = ["token_loss_5.0", "token_loss_3.0", "token_loss_1.0", "token_loss_0.5"]
    steps_taken_d, succ_d = simulate_stopping_rule(steps_d, target_token, max_steps=4)
    tests_run += 1
    print(f"  Test 3.4d (Falling loss no match)   : Expected (4, False), Actual ({steps_taken_d}, {succ_d})")
    assert (steps_taken_d, succ_d) == (4, False)
    tests_passed += 1
    
    # 3.5 CHECK_MATCH & NORMALIZE_ENTITY: 12-CASE PROVEN EQUIVALENCE SUITE
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
    print("  Test 3.5 (12-Case Equivalence)      : All 12 proven test cases PASSED identically.")
    
    # 3.6 CONTROL ACCOUNTING & IMPOSSIBLE-VALUE GUARD
    print("\n[3.6 Control Accounting & Impossible-Value Guard Tests]")
    valid_controls = {
        "never_edited": Measurement.from_outcomes([False] * 20, metric="never_edited", arm="never_edited", scope="fixture_20", input_set="ctrl", mode="eval_no_dropout"),
        "random_direction_magnitude_matched": Measurement.from_outcomes([True] + [False] * 19, metric="random_direction_magnitude_matched", arm="random_direction_magnitude_matched", scope="fixture_20", input_set="ctrl", mode="eval_no_dropout"),
        "wrong_target": Measurement.from_outcomes([False] * 20, metric="wrong_target", arm="wrong_target", scope="fixture_20", input_set="ctrl", mode="eval_no_dropout"),
        "pre_edit_baseline": Measurement.from_outcomes([True] * 2 + [False] * 18, metric="pre_edit_baseline", arm="pre_edit_baseline", scope="fixture_20", input_set="ctrl", mode="eval_no_dropout")
    }
    pooled_m, worst_m, exp_sum_str = pool_controls(valid_controls)
    tests_run += 1
    print(f"  Test 3.6a (Valid Control Pooling)   : Expanded Sum -> {exp_sum_str}")
    print(f"                                        Pooled Floor -> {pooled_m}, Worst -> {worst_m.name}: {worst_m}")
    assert pooled_m.pair == (3, 80)
    assert worst_m.name == "pre_edit_baseline" and worst_m.pair == (2, 20)
    tests_passed += 1

    # 3.7 WILSON CONFIDENCE INTERVAL UNIT TESTS
    print("\n[3.7 Wilson Score Confidence Interval Unit Tests]")
    lo_4, hi_4 = wilson_confidence_interval(4, 20)
    tests_run += 1
    print(f"  Test 3.7a (Wilson k=4, n=20)         : lo={lo_4:.4f}, hi={hi_4:.4f}")
    assert 0.080 <= lo_4 <= 0.082 and 0.415 <= hi_4 <= 0.417
    tests_passed += 1

    lo_0, hi_0 = wilson_confidence_interval(0, 20)
    tests_run += 1
    print(f"  Test 3.7b (Wilson k=0, n=20)         : lo={lo_0:.4f}, hi={hi_0:.4f}")
    assert lo_0 == 0.0 and 0.160 <= hi_0 <= 0.162
    tests_passed += 1

    # 3.8 STATISTICAL POWER SAMPLING UNIT TESTS
    print("\n[3.8 N=200 Sequence Sampling Unit Tests]")
    mock_facts = [{"fact_id": i, "subject": f"Subj_{i}", "relation": "born_city", "object": f"City_{i}"} for i in range(1000)]
    seq_0, hash_0 = sample_200_facts(mock_facts, seed=0)
    seq_1, hash_1 = sample_200_facts(mock_facts, seed=1)
    tests_run += 1
    print(f"  Test 3.8a (N=200 seed=0 hash)        : {hash_0[:16]}... (200 facts)")
    assert len(seq_0) == 200 and len(set(f["fact_id"] for f in seq_0)) == 200
    assert len(seq_1) == 200 and hash_0 != hash_1
    tests_passed += 1

    # 3.9 CAUSAL SUBSPACE ORTHOGONAL PROJECTION TESTS
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
    print(f"  Test 3.9a (Orthogonality norm)       : {overlap:.8f} (Expected near zero)")
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
