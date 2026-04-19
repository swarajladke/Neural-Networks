from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "v5_2_hippocampal_test.py",
    "v6_recurrent_test.py",
    "v6_vectorization_benchmark.py",
    "v6_thermal_test.py",
    "v7_synaptic_shield_smoke_test.py",
]


def main() -> int:
    print("==================================================")
    print(" AGNIS MAINLINE REGRESSION SUITE")
    print("==================================================")

    failures: list[str] = []
    for script in SCRIPTS:
        print(f"\n>>> Running {script}")
        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        if result.returncode != 0:
            failures.append(script)

    print("\n==================================================")
    if failures:
        print(" REGRESSION FAILURES")
        print("==================================================")
        for script in failures:
            print(f"- {script}")
        return 1

    print(" REGRESSION SUITE PASSED")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
