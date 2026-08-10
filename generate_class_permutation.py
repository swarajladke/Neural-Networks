"""
generate_class_permutation.py
==============================

Generates file-backed class permutation for reproducible Gate 1 diagnostic (L3).
Uses random.Random(42).shuffle(list(range(100))).
"""

import json
import random

rng = random.Random(42)
perm = list(range(100))
rng.shuffle(perm)

assert len(perm) == 100
assert set(perm) == set(range(100))

out_path = "class_permutation_seed42.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(perm, f, indent=2)

print(f"Saved {out_path}: {perm[:10]}...")
