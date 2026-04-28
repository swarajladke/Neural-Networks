"""
v13_abstrax_dream.py — The First Dream Cycle
=============================================
V3 Phase 32: Cross-Domain Affinity Analysis

Loads the trained Octa-Marathon checkpoint and runs the AbstraX Engine
to discover structural overlaps between the 8 language manifolds.

This is the first step toward Conceptual Intelligence:
  - If affinity is HIGH between two languages, they independently
    discovered similar weight patterns → candidates for Dream Neuron folding.
  - If affinity is LOW, the slivers are truly independent databases
    → shared meta-neurons must be introduced and re-trained.

Usage:
    python v13_abstrax_dream.py
"""

import torch
import json
import os
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import AbstraXEngine

def run_dream_cycle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "agnis_marathon_final.pt"
    baselines_path = "agnis_marathon_baselines.json"

    # --- Step 1: Load the trained checkpoint ---
    print("=" * 60)
    print("  AGNIS V3 — AbstraX Dream Cycle")
    print("  Cross-Domain Affinity Analysis")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[ERROR] Please ensure the Octa-Marathon checkpoint is in the project directory.")
        print("[HINT]  Download it from Google Drive if needed.")
        return

    # Load baselines to determine language order
    if os.path.exists(baselines_path):
        with open(baselines_path, "r") as f:
            baselines = json.load(f)
        print(f"\n[Load] Baselines found for {len(baselines)} languages: {list(baselines.keys())}")
    else:
        print("[WARN] No baselines file found, using default 8-language order.")
        baselines = None

    # --- Step 2: Reconstruct the hierarchy dimensions ---
    # Load the raw checkpoint to detect the actual architecture size
    raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint is a list of tuples, one per layer.
    # Each tuple's first element is V, which has shape [input_dim, output_dim]
    first_layer_V = raw_state[0][0]  # V tensor of layer 0
    total_width = first_layer_V.shape[1]

    # Detect the language layout
    langs = ["en", "de", "ru", "es", "it", "mr", "ro", "fr"]
    n_per_lang = 256
    base_dim = n_per_lang

    # Auto-detect: check if total_width matches expected
    detected_n_langs = total_width // n_per_lang
    if detected_n_langs != len(langs):
        print(f"[WARN] Expected {len(langs)} languages but detected {detected_n_langs} from width {total_width}")
        langs = langs[:detected_n_langs]

    lang_ranges = {}
    for i, code in enumerate(langs):
        start = i * n_per_lang
        end = start + n_per_lang
        lang_ranges[code] = (start, end)

    print(f"[Load] Architecture width: {total_width} neurons")
    print(f"[Load] Language ranges: {lang_ranges}")

    # --- Step 3: Build the hierarchy and load weights ---
    # Reconstruct the layer dimensions from the checkpoint
    layer_dims = []
    for layer_state in raw_state:
        V_tensor = layer_state[0]
        layer_dims.append(V_tensor.shape[0])  # input_dim
    # Add the output_dim of the last layer
    last_V = raw_state[-1][0]
    layer_dims.append(last_V.shape[1])

    print(f"[Load] Layer dimensions: {layer_dims}")

    hierarchy = PredictiveHierarchy(layer_dims, device=device)
    hierarchy.load_checkpoint(checkpoint_path)
    print(f"[Load] Checkpoint loaded successfully.\n")

    # --- Step 4: Run the AbstraX Engine ---
    abstrax = AbstraXEngine(hierarchy, lang_ranges)

    # Analyze each layer
    for layer_idx in range(len(hierarchy.layers)):
        print(f"\n{'─'*60}")
        print(f"  LAYER {layer_idx} ANALYSIS")
        print(f"{'─'*60}")

        result = abstrax.compute_pairwise_affinity(layer_idx=layer_idx)
        pairs = abstrax.print_affinity_report(
            result,
            title=f"Layer {layer_idx} Cross-Domain Affinity"
        )

        # Check for Dream Neuron candidates at multiple thresholds
        print(f"\n  Fold Candidate Scan:")
        for threshold in [0.8, 0.6, 0.4, 0.2]:
            candidates = abstrax.identify_fold_candidates(result, threshold=threshold)
            if candidates:
                break  # Found candidates, stop lowering threshold

        # Per-component breakdown for the most interesting pair
        if pairs:
            top_score, top_a, top_b = pairs[0]
            print(f"\n  Detailed Component Breakdown: {top_a} ↔ {top_b}")
            print(f"  {'─'*40}")
            for comp in ['V', 'W', 'R', 'R_gate', 'b_in']:
                i = abstrax.lang_codes.index(top_a)
                j = abstrax.lang_codes.index(top_b)
                val = result['per_component'][comp][i, j].item()
                bar = "█" * int(max(0, val) * 20)
                print(f"    {comp:>8}: {val:>8.4f}  {bar}")

    # --- Step 5: Summary and Next Steps ---
    print(f"\n{'='*60}")
    print(f"  DREAM CYCLE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  The AbstraX Engine has analyzed the structural relationships")
    print(f"  between all {len(langs)} language manifolds across {len(hierarchy.layers)} layers.")
    print(f"\n  INTERPRETATION GUIDE:")
    print(f"  • Affinity > 0.7  →  Strong shared structure (FOLDABLE)")
    print(f"  • Affinity 0.3-0.7 →  Moderate overlap (potential for meta-neurons)")
    print(f"  • Affinity < 0.3  →  Independent structure (fully isolated)")
    print(f"\n  High affinity proves the network independently discovered")
    print(f"  shared patterns. Low affinity means we need to introduce")
    print(f"  shared Dream Neurons and re-train to induce convergence.")


if __name__ == "__main__":
    run_dream_cycle()
