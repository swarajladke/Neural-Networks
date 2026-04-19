import torch
import time
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent
from experiment_utils import metric_to_float

def run_hippocampal_test():
    print("==================================================")
    print(" V5.2: Hippocampal Episodic Memory Test")
    print("==================================================")

    device = "cpu"
    torch.manual_seed(42)
    # Small hierarchy for fast testing
    hierarchy = PredictiveHierarchy([8, 16, 8], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    # Seed a low expected-surprise baseline so the first novel sample creates
    # a true dopamine burst and natural hippocampal storage.
    agent.neuromodulator.predicted_surprise = 0.05

    # Define a distinct pattern
    x_pattern = torch.randn(1, 8)
    y_target = torch.randn(1, 8)

    print("\n[Step 1] Initial Exposure (Cold Start)")
    # This should trigger high surprise and storage
    start_time = time.time()
    w, s_initial = agent.observe_and_learn(x_pattern, y_target, max_steps=150)
    end_time = time.time()
    s_initial = metric_to_float(s_initial)
    
    print(f"-> Initial Surprise: {s_initial:.6f}")
    print(f"-> Settling Time: {(end_time - start_time)*1000:.2f}ms")
    print(f"-> Memories Stored: {len(agent.hippocampus.memory)}")

    if len(agent.hippocampus.memory) == 0:
        raise AssertionError("Epiphany was not stored naturally. Check neuromodulation/storage thresholds.")

    print("\n[Step 2] Second Exposure (One-Shot Recall)")
    # This should trigger the fast-path injection
    start_time = time.time()
    w, s_recall = agent.observe_and_learn(x_pattern, y_target, max_steps=150)
    end_time = time.time()
    s_recall = metric_to_float(s_recall)

    print(f"-> Recall Surprise: {s_recall:.6f}")
    print(f"-> Recall Time: {(end_time - start_time)*1000:.2f}ms")
    
    improvement = s_initial / (s_recall + 1e-9)
    print(f"\n[Results]")
    if s_recall < s_initial * 0.25:
        print(f"[PASS] One-shot recall success! Surprise reduced by {improvement:.1f}x via hippocampal injection.")
    else:
        raise AssertionError(f"Recall surprise did not drop significantly. Current: {s_recall:.6f}")

    if (end_time - start_time) < 0.1: # Heuristic for fast path
         print("[PASS] Fast-path injection bypassed iterative settling.")
    else:
         raise AssertionError("Fast-path injection did not bypass iterative settling.")

if __name__ == "__main__":
    run_hippocampal_test()
