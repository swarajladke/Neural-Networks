# AGENTS.md -- Standing Rules for Continual Learning Experiments

1. **Permanent Control Arm**:
   `FREEZE-AFTER-BASE` (zero parameter updates after the base training phase) MUST be included as a permanent standing control arm in EVERY continual-learning evaluation and table. Any mechanism that does not outperform doing nothing (`FREEZE-AFTER-BASE`) has not demonstrated continual learning.
