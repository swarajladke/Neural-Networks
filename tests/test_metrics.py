#!/usr/bin/env python3
"""
tests/test_metrics.py
=====================
Pre-Flight Unit Test Suite for Continual Learning Metrics (Directive S0-1 Part 3).

Mandate:
  - Must run before any model loads or accelerator initializes.
  - Written against hand-constructed stubs; zero GPU/model dependency.
  - Each check prints expected and actual; any mismatch halts and exits nonzero.
  - Zero tests run is an immediate failure.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.metrics import (
    Measurement,
    normalize_entity,
    check_match,
    efficacy,
    generalization,
    raw_retention,
    bound_retention,
    subject_discriminable_retention,
    pool_controls,
    simulate_stopping_rule,
    CONTROL_NAMES
)

def run_all_tests() -> int:
    tests_run = 0
    tests_passed = 0
    
    print("=" * 100)
    print(" PRE-FLIGHT TEST SUITE (Directive S0-1 Part 3): Hand-Constructed Metric Stubs")
    print("=" * 100)
    
    # --------------------------------------------------------------------------
    # 3.1 EFFICACY TESTS
    # --------------------------------------------------------------------------
    print("\n[3.1 Efficacy Metric Unit & Regression Tests]")
    
    # Case A: 20 facts, exactly 13 succeeding
    facts_20 = [{"object": f"target_{i}"} for i in range(20)]
    preds_13_of_20 = [f"target_{i}" if i < 13 else "wrong_token" for i in range(20)]
    m_eff_13 = efficacy(preds_13_of_20, facts_20)
    tests_run += 1
    exp_13 = (13, 20)
    print(f"  Test 3.1a (20 facts, 13 match)   : Expected {exp_13}, Actual {m_eff_13.pair} -> {m_eff_13}")
    assert m_eff_13.pair == exp_13, f"Mismatch: expected {exp_13}, got {m_eff_13.pair}"
    tests_passed += 1
    
    # Case B: 20 facts, all 20 succeeding
    preds_20_of_20 = [f"target_{i}" for i in range(20)]
    m_eff_20 = efficacy(preds_20_of_20, facts_20)
    tests_run += 1
    exp_20 = (20, 20)
    print(f"  Test 3.1b (20 facts, 20 match)   : Expected {exp_20}, Actual {m_eff_20.pair} -> {m_eff_20}")
    assert m_eff_20.pair == exp_20, f"Mismatch: expected {exp_20}, got {m_eff_20.pair}"
    tests_passed += 1
    
    # Case C: 1 fact injected and succeeding -> must return (1, 1), NEVER bare float 100.0
    facts_1 = [{"object": "target_0"}]
    preds_1 = ["target_0"]
    m_eff_1 = efficacy(preds_1, facts_1)
    tests_run += 1
    exp_1 = (1, 1)
    print(f"  Test 3.1c (1 fact, 1 match)      : Expected {exp_1}, Actual {m_eff_1.pair} (Type: {type(m_eff_1).__name__})")
    assert m_eff_1.pair == exp_1, f"Mismatch: expected {exp_1}, got {m_eff_1.pair}"
    assert not isinstance(m_eff_1, float), f"Defect recurrence: efficacy returned bare float {m_eff_1}"
    tests_passed += 1
    
    # Case D: Regression test: denominator MUST equal count of injected facts
    tests_run += 1
    for k in [1, 5, 12, 20]:
        sub_facts = [{"object": f"target_{i}"} for i in range(k)]
        sub_preds = [f"target_{i}" for i in range(k)]
        m_k = efficacy(sub_preds, sub_facts)
        assert m_k.denominator == k, f"Regression failure: denominator {m_k.denominator} != injected count {k}"
    print("  Test 3.1d (Regression Guard)     : Denominator strictly equals injected facts count N for N in [1, 5, 12, 20]")
    tests_passed += 1
    
    # --------------------------------------------------------------------------
    # 3.2 GENERALIZATION TEST
    # --------------------------------------------------------------------------
    print("\n[3.2 Generalization Metric Unit Tests]")
    # 20 facts, exactly 2 of 3 paraphrases correct per fact -> (40, 60)
    para_preds_40_60 = [
        [f"target_{i}", f"target_{i}", "wrong_paraphrase"]
        for i in range(20)
    ]
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
    # Fixture: 5 facts across 2 relations
    # Fact 0: rel="born_city", obj="Paris", pred="Paris", rel_modal="Paris", shared_ctrl=1
    #         -> raw match: True, bound: False (is modal), subj_disc: True (shared<=2)
    # Fact 1: rel="born_city", obj="Berlin", pred="Berlin", rel_modal="Paris", shared_ctrl=3
    #         -> raw match: True, bound: True (not modal), subj_disc: False (shared>2)
    # Fact 2: rel="born_city", obj="Rome", pred="Paris", rel_modal="Paris", shared_ctrl=1
    #         -> raw match: False, bound: False, subj_disc: False
    # Fact 3: rel="instrument", obj="piano", pred="piano", rel_modal="violin", shared_ctrl=0
    #         -> raw match: True, bound: True (not modal), subj_disc: True (shared<=2)
    # Fact 4: rel="instrument", obj="flute", pred="drums", rel_modal="violin", shared_ctrl=0
    #         -> raw match: False, bound: False, subj_disc: False
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
        "born_city": ["Paris", "Berlin", "Berlin", "Berlin", "Madrid"],  # 'Paris' appears 1x, 'Berlin' appears 3x
        "instrument": ["violin", "violin", "guitar"]                      # 'piano' appears 0x
    }
    
    # Hand-counted expected results:
    # Raw retention: facts 0, 1, 3 match -> 3 / 5
    m_raw = raw_retention(fixture_preds, fixture_facts)
    tests_run += 1
    exp_raw = (3, 5)
    print(f"  Test 3.3a (Raw Retention)        : Expected {exp_raw}, Actual {m_raw.pair} -> {m_raw}")
    assert m_raw.pair == exp_raw, f"Raw retention mismatch: expected {exp_raw}, got {m_raw.pair}"
    tests_passed += 1
    
    # Bound retention: facts 1, 3 match and are non-modal -> 2 / 5
    m_bound = bound_retention(fixture_preds, fixture_facts, fixture_rel_modals)
    tests_run += 1
    exp_bound = (2, 5)
    print(f"  Test 3.3b (Bound Retention)      : Expected {exp_bound}, Actual {m_bound.pair} -> {m_bound}")
    assert m_bound.pair == exp_bound, f"Bound retention mismatch: expected {exp_bound}, got {m_bound.pair}"
    tests_passed += 1
    
    # Subject-discriminable retention: facts 0, 3 match and shared_ctrl <= 2 -> 2 / 5
    m_disc = subject_discriminable_retention(fixture_preds, fixture_facts, fixture_ctrl_preds, max_shared_controls=2)
    tests_run += 1
    exp_disc = (2, 5)
    print(f"  Test 3.3c (Subj-Discrim Ret)     : Expected {exp_disc}, Actual {m_disc.pair} -> {m_disc}")
    assert m_disc.pair == exp_disc, f"Subj-discrim retention mismatch: expected {exp_disc}, got {m_disc.pair}"
    tests_passed += 1
    
    # --------------------------------------------------------------------------
    # 3.4 STOPPING RULE UNIT TESTS (SYNTHETIC LOGITS / PREDICTIONS)
    # --------------------------------------------------------------------------
    print("\n[3.4 Stopping Rule Simulation Tests]")
    target_token = "Rome"
    
    # Case A: Target becomes argmax at step 3 -> terminates at step 3 with True
    steps_a = ["Paris", "Berlin", "Rome", "Rome", "Rome"]
    steps_taken_a, succ_a = simulate_stopping_rule(steps_a, target_token, max_steps=25)
    tests_run += 1
    print(f"  Test 3.4a (Terminates on match)   : Expected (3, True), Actual ({steps_taken_a}, {succ_a})")
    assert (steps_taken_a, succ_a) == (3, True), f"Mismatch: expected (3, True), got ({steps_taken_a}, {succ_a})"
    tests_passed += 1
    
    # Case B: Target is already argmax at step 1 -> terminates immediately at step 1
    steps_b = ["Rome", "Rome", "Rome"]
    steps_taken_b, succ_b = simulate_stopping_rule(steps_b, target_token, max_steps=25)
    tests_run += 1
    print(f"  Test 3.4b (Terminates at step 1)  : Expected (1, True), Actual ({steps_taken_b}, {succ_b})")
    assert (steps_taken_b, succ_b) == (1, True), f"Mismatch: expected (1, True), got ({steps_taken_b}, {succ_b})"
    tests_passed += 1
    
    # Case C: Target never becomes argmax -> runs to max_steps and reports False
    steps_c = ["Paris"] * 30
    steps_taken_c, succ_c = simulate_stopping_rule(steps_c, target_token, max_steps=15)
    tests_run += 1
    print(f"  Test 3.4c (Runs to max_steps)     : Expected (15, False), Actual ({steps_taken_c}, {succ_c})")
    assert (steps_taken_c, succ_c) == (15, False), f"Mismatch: expected (15, False), got ({steps_taken_c}, {succ_c})"
    tests_passed += 1
    
    # Case D: Falling loss while target is NOT argmax -> must NOT terminate early
    # (Simulated by 10 non-matching predictions with decreasing losses)
    steps_d = ["token_loss_5.0", "token_loss_3.0", "token_loss_1.0", "token_loss_0.5"]
    steps_taken_d, succ_d = simulate_stopping_rule(steps_d, target_token, max_steps=4)
    tests_run += 1
    print(f"  Test 3.4d (Falling loss no match) : Expected (4, False), Actual ({steps_taken_d}, {succ_d})")
    assert (steps_taken_d, succ_d) == (4, False), f"Mismatch: expected (4, False), got ({steps_taken_d}, {succ_d})"
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
        assert act_m == exp_m, f"Case {idx+1} match mismatch: expected {exp_m}, got {act_m}"
        assert act_n == exp_n, f"Case {idx+1} normalize mismatch: expected {exp_n}, got {act_n}"
        tests_passed += 1
    print(f"  Test 3.5 (12-Case Equivalence)    : All 12 proven test cases PASSED identically.")
    
    # --------------------------------------------------------------------------
    # 3.6 CONTROL ACCOUNTING & IMPOSSIBLE-VALUE GUARD
    # --------------------------------------------------------------------------
    print("\n[3.6 Control Accounting & Impossible-Value Guard Tests]")
    # Case A: Valid controls
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
    assert pooled_m.pair == (3, 80), f"Expected (3, 80), got {pooled_m.pair}"
    assert worst_m.name == "pre_edit_baseline" and worst_m.pair == (2, 20)
    tests_passed += 1
    
    # Case B: Impossible value numerator > denominator (e.g. 21/20, simulating the 450% incident)
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
    # 3.7 TEST SUITE SUMMARY
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
