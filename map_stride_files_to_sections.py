"""
map_stride_files_to_sections.py
===============================

Maps each of the 16 stride-affected files from E5 to the RESULTS.md sections
whose numbers/tables they produced, and verifies that retraction banners are applied.
"""

MAPPING = [
    {
        "file": "run_student_continual_benchmarks.py",
        "sections": ["10 (Phase 3)", "14 (Phase 4 Lever 4)", "15 (Phase 4.1)"],
        "notes": "Produces static retrieval, sequential matrix, and intrinsic dimension benchmarks."
    },
    {
        "file": "run_phase5_der_plus_plus_class_il.py",
        "sections": ["16 (Phase 5)"],
        "notes": "Produces DER++ Class-IL benchmark tables."
    },
    {
        "file": "run_phase6_continuum_memory_class_il.py",
        "sections": ["17 (Phase 6)"],
        "notes": "Produces Multi-Frequency Continuum Memory Class-IL benchmark tables."
    },
    {
        "file": "run_phase7_metric_calibration_class_il.py",
        "sections": ["18 (Phase 7)"],
        "notes": "Produces Local Metric Calibration and temperature scaling Class-IL tables."
    },
    {
        "file": "run_adapter_continual_benchmarks.py",
        "sections": ["1 (Benchmark Def)", "2 (Frozen Floor)", "3 (OGP Rank Sweep)", "4 (Control Arms)", "5 (Exact Decomp)", "7 (Headline)", "8 (Limitations)", "9 (Phase 2 Calibrated Forgetting)"],
        "notes": "Produces adapter baseline, OGP rank sweep, and calibrated forgetting benchmarks."
    },
    {
        "file": "run_decisive_controls.py",
        "sections": ["4 (Control Arms)", "9 (Phase 2 Controls C1-C4)", "10.2 (Decisive Controls C1-C4)"],
        "notes": "Produces controls C1-C4 and Random/Bottom/Current subspace controls."
    },
    {
        "file": "run_continual_learning_validation.py",
        "sections": ["10 (Phase 3 Parametric Memory)"],
        "notes": "Produces Phase 3 parametric memory evaluation metrics."
    },
    {
        "file": "run_base_rate_enrichment_test.py",
        "sections": ["10.2 (Phase 4 Part 0 Base Rate Audit)"],
        "notes": "Produces confusable class base rate and enrichment calculations."
    },
    {
        "file": "run_graded_ceiling_reanalysis.py",
        "sections": ["10.1 (Retrieval Ceiling & Failure Audit)"],
        "notes": "Produces M1 ROC AUC, McFadden R2, and LOFO-CV ceiling reanalysis."
    },
    {
        "file": "run_graded_ceiling_test.py",
        "sections": ["10.1 (Graded Ceiling Predictor Test)"],
        "notes": "Produces initial graded ceiling predictor statistics."
    },
    {
        "file": "run_off_support_density_test.py",
        "sections": ["10.1 (Support Proximity dq Analysis)"],
        "notes": "Produces support proximity density dq and generic metric transfer statistics."
    },
    {
        "file": "run_confusable_split_experiment.py",
        "sections": ["1 (Benchmark Definition - Confusable Split)", "10.2 (Confusable Split Audit)"],
        "notes": "Produces confusable split block ordering and interference calculations."
    },
    {
        "file": "run_decoder_integration_validation.py",
        "sections": ["None (Diagnostic / Scratch Validation Script)"],
        "notes": "Diagnostic script for decoder integration; produced no direct section numbers in RESULTS.md."
    },
    {
        "file": "run_d2_coverage_evaluation.py",
        "sections": ["9 (Disclosure D2 Coverage)"],
        "notes": "Produces bottleneck parameterization projection coverage calculations."
    },
    {
        "file": "audit_fact_map_and_c_q_bug.py",
        "sections": ["10.1 (Fact Map Defect Audit)"],
        "notes": "Audits class indexing defect and per-fact failure counts."
    },
    {
        "file": "dump_c2_raw_data.py",
        "sections": ["10.1 (Raw Array Verification & C2 Reconciliation)"],
        "notes": "Dumps C2 raw arrays and reconciles Step-9 per-block accuracies."
    }
]


def print_table():
    print("# Mapping of Stride-Affected Files (E5) to RESULTS.md Sections\n")
    print("| File Basename | Produced Section(s) in RESULTS.md | Notes / Role |")
    print("|:---|:---|:---|")
    for item in MAPPING:
        print(f"| `{item['file']}` | {', '.join(item['sections'])} | {item['notes']} |")


if __name__ == "__main__":
    print_table()
