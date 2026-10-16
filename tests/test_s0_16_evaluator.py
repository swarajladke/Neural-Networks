#!/usr/bin/env python3
"""
tests/test_s0_16_evaluator.py -- Pre-Flight Unit Test Suite for Directive S0-16 Evaluator

Mandate (Directive S0-16 Section 7 & AGENTS.md Section 8):
  - Validates evaluator without model runs or accelerator initializes.
  - Fixtures:
    1. 50 facts x 3 paraphrases = 150 evaluated examples
    2. 200 facts x 3 paraphrases = 600 evaluated examples
    3. Delayed successes (I=0, T=1) correctly handled
    4. Zero immediate successes (I=0 for all)
    5. Mixed acquired / forgotten / delayed outcomes (full 2x2 table)
    6. Arm population mismatches raise ValueError
    7. Paired statistical test with unequal magnitudes and exact p-value enumeration
    8. Tensor storage accounting and orthonormality residual verification
"""

import math
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.s0_16_evaluator import (
    EvaluationRecord,
    compute_transition_table,
    evaluate_paraphrase_records,
    compute_exact_paired_sign_flip_test,
    audit_tensor_basis_orthonormality,
    audit_memory_breakdown
)


def make_record(f_id: int, seed: int, arm: str, p_type: str, p_id: str, is_match: bool) -> EvaluationRecord:
    return EvaluationRecord(
        fact_id=f_id,
        seed=seed,
        arm=arm,
        checkpoint="post_edit_200",
        prompt_type=p_type,
        prompt_id=p_id,
        prompt_text=f"Prompt for fact {f_id}",
        target_text="target",
        prediction_text="target" if is_match else "wrong",
        is_match=is_match
    )


def test_paraphrase_fixtures_150_and_600():
    # Fixture 1: 50 facts x 3 paraphrases = 150 examples
    records_150 = []
    for f in range(50):
        for p in range(3):
            # 1 match per fact -> 50 / 150 matches
            is_m = (p == 0)
            records_150.append(make_record(f, 0, "c3", "paraphrase", f"p_{f}_{p}", is_m))

    res_150 = evaluate_paraphrase_records(records_150)
    assert res_150["denominator"] == 150, f"Expected denominator 150, got {res_150['denominator']}"
    assert res_150["numerator"] == 50, f"Expected numerator 50, got {res_150['numerator']}"
    assert abs(res_150["rate"] - (50.0 / 150.0)) < 1e-12

    # Fixture 2: 200 facts x 3 paraphrases = 600 examples
    records_600 = []
    for f in range(200):
        for p in range(3):
            # Matches exactly 146 total examples
            is_m = (len(records_600) < 146)
            records_600.append(make_record(f, 0, "c3", "paraphrase", f"p_{f}_{p}", is_m))

    res_600 = evaluate_paraphrase_records(records_600)
    assert res_600["denominator"] == 600, f"Expected denominator 600, got {res_600['denominator']}"
    assert res_600["numerator"] == 146, f"Expected numerator 146, got {res_600['numerator']}"
    assert abs(res_600["rate"] - (146.0 / 600.0)) < 1e-12
    print("  [Pass] Paraphrase fixtures: 150 and 600 example evaluations asserted.")


def test_transition_delayed_successes():
    # 20 facts: 10 acquired & retained, 5 acquired & forgotten, 3 delayed successes, 2 unsuccessful both
    imm_recs = []
    term_recs = []
    for f in range(20):
        if f < 10:
            imm_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", True))
            term_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", True))
        elif f < 15:
            imm_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", True))
            term_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", False))
        elif f < 18:
            imm_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", False))
            term_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", True))
        else:
            imm_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", False))
            term_recs.append(make_record(f, 0, "test", "canonical", f"c_{f}", False))

    tbl = compute_transition_table(imm_recs, term_recs)
    c = tbl["counts"]
    assert c["a_acquired_and_retained"] == 10
    assert c["b_acquired_then_forgotten"] == 5
    assert c["c_delayed_success"] == 3
    assert c["d_unsuccessful_both"] == 2
    assert tbl["n_total"] == 20
    assert tbl["immediate_efficacy"]["numerator"] == 15
    assert tbl["terminal_accuracy"]["numerator"] == 13
    assert tbl["conditional_retention"]["numerator"] == 10
    assert tbl["conditional_retention"]["denominator"] == 15
    assert abs(tbl["conditional_retention"]["rate"] - (10.0 / 15.0)) < 1e-12
    print("  [Pass] Transition table: delayed successes and exact conditional retention asserted.")


def test_transition_zero_immediate_successes():
    # All 50 facts fail immediate acquisition (e.g. Sham baseline)
    imm_recs = [make_record(f, 0, "sham", "canonical", f"c_{f}", False) for f in range(50)]
    term_recs = [make_record(f, 0, "sham", "canonical", f"c_{f}", False) for f in range(50)]

    tbl = compute_transition_table(imm_recs, term_recs)
    assert tbl["immediate_efficacy"]["numerator"] == 0
    assert tbl["immediate_efficacy"]["denominator"] == 50
    assert tbl["terminal_accuracy"]["numerator"] == 0
    assert tbl["conditional_retention"]["numerator"] == 0
    assert tbl["conditional_retention"]["denominator"] == 0
    assert tbl["conditional_retention"]["rate"] == 0.0
    print("  [Pass] Transition table: zero immediate successes handled without zero division.")


def test_transition_population_mismatch_guard():
    imm_recs = [make_record(f, 0, "arm1", "canonical", f"c_{f}", True) for f in range(10)]
    term_recs = [make_record(f + 5, 0, "arm1", "canonical", f"c_{f+5}", True) for f in range(10)]

    mismatch_caught = False
    try:
        compute_transition_table(imm_recs, term_recs)
    except ValueError as e:
        mismatch_caught = True
        assert "distinct fact_id sets" in str(e)

    assert mismatch_caught, "Expected ValueError on disjoint fact ID populations"
    print("  [Pass] Transition table: population mismatch guard verified.")


def test_exact_paired_sign_flip_test():
    # 8 paired differences with unequal magnitudes
    diffs = [0.08, 0.12, 0.05, -0.02, 0.15, 0.10, 0.07, 0.04]
    res = compute_exact_paired_sign_flip_test(diffs)

    assert res["n_pairs"] == 8
    assert res["total_permutations"] == 256
    assert res["resolution_floor"] == (2.0 / 256.0)
    assert res["observed_mean_diff"] == sum(diffs) / 8.0
    assert 0.0 <= res["exact_p_value"] <= 1.0

    # Unanimous positive differences: all 8 positive
    diffs_unanimous = [0.10] * 8
    res_unan = compute_exact_paired_sign_flip_test(diffs_unanimous)
    # Only 2 permutations (all +1 and all -1) match or exceed the observed mean
    assert res_unan["extreme_count"] == 2
    assert abs(res_unan["exact_p_value"] - (2.0 / 256.0)) < 1e-12
    print("  [Pass] Exact sign-flip test: 256-permutation enumeration and resolution asserted.")


def test_tensor_basis_orthonormality_and_memory():
    # 1. Valid orthonormal basis: 2235 ambient x 50 columns
    m, k = 2235, 50
    rand_mat = torch.randn(m, k, dtype=torch.float64)
    q_valid, _ = torch.linalg.qr(rand_mat)
    audit_q = audit_tensor_basis_orthonormality(q_valid)
    assert audit_q["is_orthonormal"], f"Expected orthonormal basis, got error {audit_q['residual_frob']}"
    assert audit_q["residual_frob"] <= 1e-10

    # 2. Defective basis: columns > rows (impossible to be orthonormal)
    defective = torch.randn(10, 20, dtype=torch.float64)
    audit_def = audit_tensor_basis_orthonormality(defective)
    assert not audit_def["is_orthonormal"]
    assert "DEFECT" in audit_def["diagnosis"]

    # 3. Memory audit breakdown
    v_null = torch.zeros((3072, 2235), dtype=torch.float64)
    q_prompt = torch.zeros((2235, 100), dtype=torch.float64)
    stored_keys = [torch.zeros(3072, dtype=torch.float64) for _ in range(50)]
    stored_vstars = [torch.zeros(768, dtype=torch.float64) for _ in range(50)]

    mem = audit_memory_breakdown(v_null, q_prompt, stored_keys, stored_vstars, num_edits=50)
    expected_v_null_bytes = 3072 * 2235 * 8
    expected_v_null_mib = expected_v_null_bytes / (1024.0 ** 2)
    assert mem["v_null"]["bytes"] == expected_v_null_bytes, f"Bytes mismatch: {mem['v_null']['bytes']} != {expected_v_null_bytes}"
    assert abs(float(mem["v_null"]["mib"]) - expected_v_null_mib) < 1e-4, f"MiB mismatch: {mem['v_null']['mib']} != {expected_v_null_mib}"
    assert mem["distinct_subject_slots"] == 50
    assert mem["total_state_bytes"] > mem["total_dynamic_bytes"]
    print("  [Pass] Basis orthonormality and memory breakdown accounting asserted.")


def main():
    print("=" * 80)
    print(" DIRECTIVE S0-16: PRE-FLIGHT EVALUATOR UNIT TEST SUITE")
    print("=" * 80)
    test_paraphrase_fixtures_150_and_600()
    test_transition_delayed_successes()
    test_transition_zero_immediate_successes()
    test_transition_population_mismatch_guard()
    test_exact_paired_sign_flip_test()
    test_tensor_basis_orthonormality_and_memory()
    print("=" * 80)
    print(" ALL 6 TEST SUITES PASSED (0 FAILURES).")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
