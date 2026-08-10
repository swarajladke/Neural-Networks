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
10. **P10**: CV-selected test accuracy will be lower than the max-over-cells value by more than 3 percentage points.
11. **P11**: The CV procedure will select a truncated-PCA representation, not mean/none.
12. **P12**: Plain Linear will beat HeadL1c on every representation in the J3 table.
13. **P13**: After the CV bug is fixed, HeadL1c will no longer be the CV-winning method.
14. **P14**: After the fix, the CV-selected representation will differ from mean/center.
15. **P15**: Correcting the CV bug will reduce the reported CV score of the winning cell by more than 5 percentage points.

---

### Additional Pre-Registered Predictions (M1-M7 Re-Registration)

16. **P16**: Validation-selected weight decay will differ from test-selected weight decay on at least 5 of the 11 cells.
17. **P17**: HONEST_TEST_ACC for the selected representation will be at least 2 pp below the reported 85.60%.
18. **P18**: r_before will exceed +0.80, i.e. the within-train CV was already predictive on v3 and the disjoint-template split is not the reason selection improved.
19. **P19**: Train-val centroid cosine will exceed train-test centroid cosine.

---

### Additional Pre-Registered Predictions (N1-N9 Re-Registration)

20. **P20**: Recomputed 3/3 NCM test accuracy on mean/pca_m32_eps1e-4 will equal that of mean/pca_m32_eps1e-6 to within 0.01 pp, restoring P10 to WRONG.
21. **P21**: With per-method mean-across-folds scoring, the v3 CV winner will be MultinomialLogReg rather than NCM, and the winning CV score will fall by more than 3 pp relative to the max-over-methods-per-fold value.
22. **P22**: The printed N_test_evals will be strictly greater than 11.
23. **P23**: At eps=1e-2, m=128 disjoint validation accuracy will exceed the eps=1e-4, m=128 value (58.67%) by more than 15 pp.

---

### Additional Pre-Registered Predictions (O1-O8 Re-Registration)

24. **P24**: Under the unified stack, the validation-selected config for mean/pca_m64_eps1e-4 will be a LogReg with wd > 0, not wd = 0.0, and the wd=0.0 fit will be flagged NON-CONVERGED.
25. **P25**: HONEST_TEST_ACC under the unified stack will fall within 2.0 pp of 82.60%.
26. **P26**: The max-abs elementwise difference between the pca_m32_eps1e-6 and pca_m32_eps1e-4 transformed test matrices will be strictly greater than zero on BOTH caches, meaning the identical v3 metrics are a coincidence of argmax ties rather than identical representations.
27. **P27**: In Phase IV, freeze_after_base will exceed naive_l1c final average accuracy by more than 20 pp, and ncm_incremental will land within 5 pp of the joint_offline NCM value.

---
*Scorecard and verification will be recorded after completing all phases.*
