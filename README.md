# Continual Learning & Multi-Frequency Continuum Memory System

## 🚀 Overview

This repository evaluates Continual Learning (CL) mechanisms on a 100-class benchmark using **SmolLM2-360M** embeddings under strict **Class-Incremental Learning (Class-IL)** evaluation.

Key Mechanisms:
- **Level 1 (f = 0)**: Frozen base feature representation (eliminates representation corruption).
- **Level 2 (f = fast)**: Non-parametric `fact_memory` continuous vector cache acquiring new classes with zero backprop parameter updates.
- **Pure NCM Upper Bound (`pure_ncm_all100`)**: Un-fused Nearest-Centroid Classifier across all 100 classes achieving **71.50% Class-IL accuracy** ($+14.01\text{ pp}$ over `FREEZE-AFTER-BASE`).

---

## 🧪 Running the Benchmarks

### **Phase 6: Multi-Frequency Continuum Memory System**
Runs the 100-run Class-IL benchmark across 6 arms:
```bash
python run_phase6_continuum_memory_class_il.py
```

### **Phase 7: Local Metric Calibration & Temperature Scaling**
Runs logit scale alignment and prediction flip instrumentation:
```bash
python run_phase7_metric_calibration_class_il.py
```

### **Task 3 Matched Paired Bootstrap CIs**
Computes 10,000-sample matched paired bootstrap CIs on `(shuffle, seed)` keys:
```bash
python compute_phase6_paired_bootstrap.py
```

---

## ⚠️ Provenance & Legacy Checkpoint Status Notice

Commit `6e2876a` cleaned up obsolete legacy sprint scripts (`v10`..`v23` series, prompt injection tests, and old marathons).

### Orphaned Checkpoint Binaries (No Provenance at HEAD)
The following binary checkpoint files in `checkpoints/` are retained for historical record but have **no reproducible source script at HEAD** (producing scripts removed in commit `6e2876a`):
- `checkpoints/ru_milestone_500.pt` .. `checkpoints/ru_milestone_3000.pt`
- `checkpoints/italian_baseline_v72.pt`
- `checkpoints/phase_733_breakthrough.pt`

**Choice**: Retained in `checkpoints/` with explicit non-provenance documentation above.
