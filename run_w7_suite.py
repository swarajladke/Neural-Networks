"""
run_w7_suite.py
===============
Directive W7: Master Execution Suite

Sequentially executes:
  - W7-0: Reproducibility Pinning & Protocol Reconciliation (run_w7_determinism.py)
  - W7-1: LwF Distillation Rewrite & Diagnostics (run_w7_diagnostics.py)
  - W7-2: EWC Stable Binding Regime Sweep (run_w7_diagnostics.py)
  - W7-3: SDC Boundary Extension & Sweep Reconciliation (run_w7_diagnostics.py)
  - W7-4: Resource Counter & Live Buffer Memory Audit (run_w7_diagnostics.py)

Ensures every section prints literal untruncated measurements and terminates with EXIT_CODE = 0.
"""

import os
import sys
import subprocess
import time


def main():
    suite_t0 = time.time()
    print("=" * 115)
    print(" DIRECTIVE W7 -- MASTER SUITE (W7-0, W7-1, W7-2, W7-3, W7-4)")
    print("===================================================================================")
    print("  Host Execution: Kaggle (Tesla T4)")
    print("  Protocol      : ResNet-18, Split-CIFAR-100, 20 epochs/task, lr=0.005, weight_decay=5e-4")
    print("===================================================================================\n")

    # Step 1: Execute W7-0 (Reproducibility Pinning)
    print("\n" + "#" * 115)
    print(" [SECTION 1 / 2] EXECUTING ITEM W7-0 (REPRODUCIBILITY PINNING)")
    print("#" * 115 + "\n")
    p0 = subprocess.run([sys.executable, "run_w7_determinism.py"], text=True)
    if p0.returncode != 0:
        print(f"\n[ERROR] run_w7_determinism.py exited with non-zero code: {p0.returncode}")
        sys.exit(p0.returncode)

    # Step 2: Execute W7-1, W7-2, W7-3, W7-4
    print("\n" + "#" * 115)
    print(" [SECTION 2 / 2] EXECUTING ITEMS W7-1, W7-2, W7-3, W7-4 (DIAGNOSTICS & REPAIRS)")
    print("#" * 115 + "\n")
    p1 = subprocess.run([sys.executable, "run_w7_diagnostics.py"], text=True)
    if p1.returncode != 0:
        print(f"\n[ERROR] run_w7_diagnostics.py exited with non-zero code: {p1.returncode}")
        sys.exit(p1.returncode)

    total_elapsed = time.time() - suite_t0
    print("\n" + "=" * 115)
    print(f" DIRECTIVE W7 MASTER SUITE COMPLETE (Total Wall Clock: {total_elapsed:.1f}s)")
    print(" EXIT_CODE = 0")
    print("===================================================================================")


if __name__ == "__main__":
    main()
