# Pre-Registered Predictions (Phases I to V & J1-J7 Updates)

Date: 2026-08-10

The following pre-registered predictions are recorded prior to executing Phase II, Phase III, Phase IV, and Phase V.

1. **P1**: Last-token pooling will outperform mean pooling on NCM top-1.
2. **P2**: Centering will improve NCM top-1 over no transform for both poolings.
3. **P3**: Joint offline test accuracy will exceed NCM top-1 of the same configuration.
4. **P4**: Joint offline test accuracy will be below 64.95% (the previously reported `phase6_dual_continuum` figure).
5. **P5**: Every Class-IL arm will score below joint offline.

---

### Additional Pre-Registered Predictions (J1-J7 Re-Registration)

6. **P6**: Truncated PCA-whitening will beat the current broken ZCA's 40.33% NCM.
7. **P7**: The last-token cell will improve substantially once punctuation is excluded, but will still trail mean pooling.
8. **P8**: Moving from 3 train prompts to 10 will raise OFFLINE_BOUND by more than 10 percentage points.
9. **P9**: Corrected Gate 1 diagnostic accuracy will be monotonically non-increasing in k.

---

### Additional Pre-Registered Predictions (K5 & K6 Re-Registration)

10. **P10**: CV-selected test accuracy will be lower than the max-over-cells value by more than 3 percentage points.
11. **P11**: The CV procedure will select a truncated-PCA representation, not mean/none.
12. **P12**: Plain Linear will beat HeadL1c on every representation in the J3 table.

---
*Scorecard and verification will be recorded after completing all phases.*

