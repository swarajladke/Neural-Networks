import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent
import time
import os
from experiment_utils import metric_to_float

def run_deep_pillar_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- AGNIS Phase 8.0: Deep Scaffolding Stability ---")
    print(f"Target Depth: 100 Layers | Width: 32 Neurons")
    print(f"Device: {device}")

    # 1. Initialize the Vertical Pillar
    # [32] -> [32] x 99 -> [32]
    layer_dims = [32] * 101 # 100 layers total
    hierarchy = PredictiveHierarchy(layer_dims, device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)

    # 2. Define Signal Induction Task
    # Goal: Propagate a specific 32-bit pattern through 100 layers.
    # If the local predictive update holds, surprise should decay consistently across layers.
    source_signal = torch.randn(1, 32, device=device).sign() # High-contrast binary signal
    
    batches = 100
    print(f"Starting Signal Induction (Depth=100)...")

    start_time = time.time()
    
    # Trackers
    global_surprise_history = []
    layer_wise_surprise_history = [] # To visualize "Diffusion"

    try:
        for b in range(batches):
            # 1. Observation (Settling 100 layers)
            # High settling steps (200) to ensure deep convergence
            _, global_surprise = agent.observe_and_learn(source_signal, source_signal, max_steps=180)
            global_surprise_value = metric_to_float(global_surprise)
            global_surprise_history.append(global_surprise_value)

            # 2. Extract Layer-wise Surprise (Internal Probe)
            # surprise = |target - prediction|
            with torch.no_grad():
                layer_surprises = [layer.error.abs().mean().item() for layer in hierarchy.layers]
                layer_wise_surprise_history.append(layer_surprises)

            if (b + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Batch {b+1:03d} | Global Surprise: {global_surprise_value:.4f} | Avg Layer Surprise: {sum(layer_surprises)/len(layer_surprises):.4f} | Time: {elapsed:.1f}s")

        # 3. Stability Verification
        print("\n--- Stability Analysis ---")
        l1_surprise = layer_wise_surprise_history[-1][0]
        l100_surprise = layer_wise_surprise_history[-1][-1]
        
        print(f"Layer 01 Surprise: {l1_surprise:.4f}")
        print(f"Layer 100 Surprise: {l100_surprise:.4f}")
        
        diffusion_ratio = l100_surprise / (l1_surprise + 1e-8)
        print(f"Diffusion Ratio (L100/L1): {diffusion_ratio:.4f}")
        
        if diffusion_ratio < 2.0:
            print(">>> SIGNAL INDUCTION STABLE: Local learning scales to deep hierarchies. <<<")
        else:
            print(">>> WARNING: Surprise exploding at depth. Recalibrating Scaffold. <<<")

        # Save results for visualization
        torch.save({
            'global_history': global_surprise_history,
            'diffusion_history': layer_wise_surprise_history,
            'config': {'depth': 100, 'width': 32}
        }, "v9_scaling_results.pt")

    except Exception as e:
        print(f"CRITICAL FAILURE during deep scaling: {e}")

if __name__ == "__main__":
    run_deep_pillar_benchmark()
