# State of the Record: Continual Learning Benchmark

**Date**: 2026-09-16  
**Repository**: `Neural-Networks` (Split-CIFAR-100, ResNet-18, 10 Tasks of 10 Classes, `SEEDS = [42,43,44,45,46]`)  
**Status Schema**: `STANDING`, `PROVISIONAL`, `WITHDRAWN`, or `NO ARTIFACT`

---

## 1. Continual Learning Nine-Arm Baselines

| Arm | Name | Class-IL (Mean ± Std) | Status | Producing Script | Committed Log | Commit SHA |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **Arm 1** | `1_freeze_after_base` | $8.67\% \pm 0.04\%$ | `WITHDRAWN` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 1** | `1_freeze_after_base (W4)` | $9.41\% \pm 0.16\%$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |
| **Arm 2** | `2_naive_fine_tune` | $9.53\% \pm 0.21\%$ | `STANDING` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 3** | `3_ncm_frozen_features` | $47.12\% \pm 0.08\%$ | `STANDING` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 4** | `4_ncm_adapting_features` | $41.98\% \pm 1.27\%$ | `PROVISIONAL` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 5** | `5_lwf` | $10.17\% \pm 0.24\%$ | `WITHDRAWN` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 6** | `6_ewc` | $10.32\% \pm 0.28\%$ | `WITHDRAWN` | None (JSON missing) | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 7** | `7_er_buffer500` | $36.94\% \pm 0.37\%$ | `WITHDRAWN` | None (JSON missing) | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 8** | `8_der_plus_plus_buffer500` | $41.16\% \pm 0.97\%$ | `WITHDRAWN` | None (JSON missing) | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Arm 9** | `9_joint_offline` | $79.62\% \pm 0.21\%$ | `STANDING` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |

*Notes*:
- Arm 1 W3 is withdrawn because `model.train()` allowed BatchNorm running statistics to adapt on subsequent tasks.
- Arm 1 W4 is withdrawn because it was executed on non-canonical sequential class ordering `range(0, 10)`.
- Arm 4 is provisional due to augmented training views in prototype extraction.
- Arms 5, 6, 7, 8 are withdrawn on provenance grounds (missing or incomplete in `w3_baselines.json`).

---

## 2. Adaptation Gap & Representation Bounds

| Metric / Result | Value | Status | Producing Script | Committed Log | Commit SHA |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Adaptation Gap (Joint vs Frozen Probe)** | $+20.43\text{ pp}$ | `STANDING` | `run_w2e_gap_closed.py` | `run_w2e_gap_closed_stdout.txt` | `6c107be` |
| **Frozen ImageNet Linear Probe** | $59.19\% \pm 0.09\%$ | `STANDING` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |
| **Full Offline Linear Probe (Joint Backbone)** | $79.30\% \pm 0.11\%$ | `STANDING` | `run_w3_baselines.py` | `run_w3_baselines_stdout.txt` | `1e4d3e6` |

---

## 3. Readout Attack & Headroom Evaluation (Task 5 Horizon)

| Result | Value | Status | Producing Script | Committed Log | Commit SHA |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Predecessor Readout Ceiling** | $65.58\% \pm 0.41\%$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |
| **Available Headroom** | $22.32\text{ pp}$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |
| **Method M1 (SLDA)** | $43.43\% \pm 0.59\%$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |
| **Method M2 (SDC, Boundary Sweep)** | $44.33\% \pm 0.58\%$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |
| **Random-Trigger Control 1** | $43.27\% \pm 0.58\%$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |
| **Random-Trigger Control 2** | $19.79\% \pm 0.31\%$ | `WITHDRAWN` | `run_w4_attack_readout.py` | `run_w4_attack_readout_stdout.txt` | `351ab7f` |

*Reason for Withdrawal*: `run_w4_attack_readout.py` line 132 partitioned tasks via sequential class order (`range(t * 10, (t + 1) * 10)`) rather than the canonical class order.

---

## 4. Diagnostic & Exploratory Sweeps

| Result | Value | Status | Producing Script | Committed Log | Commit SHA |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **SDC Downward Grid Optimal ($\sigma=0.25$, renorm=True)** | $67.63\%$ | `STANDING` | `run_w7_diagnostics.py` | `run_w7_suite_stdout.txt` | `940c32f` |
| **SDC $\Delta=0$ Reference (renorm=True)** | $62.90\%$ | `STANDING` | `run_w7_diagnostics.py` | `run_w7_suite_stdout.txt` | `940c32f` |
| **SDC Uniform Limit ($\sigma=\infty$, renorm=True)** | $63.93\%$ | `STANDING` | `run_w7_diagnostics.py` | `run_w7_suite_stdout.txt` | `940c32f` |
| **SDC Hard 1-NN Assignment Limit** | $64.63\%$ | `STANDING` | `run_w7_diagnostics.py` | `run_w7_suite_stdout.txt` | `940c32f` |
| **W5 Positive Controls (+23.2 pp EWC, 56.00% / 83.10%)** | $+23.2\text{ pp}$ | `WITHDRAWN` | `run_w5_positive_controls.py` | `run_w5_positive_controls_stdout.txt` | `66e68ba` |
