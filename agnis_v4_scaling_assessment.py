import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, Experience
import random
import matplotlib.pyplot as plt
from experiment_utils import metric_to_float

def stress_test_stability():
    device = "cpu"
    hierarchy = PredictiveHierarchy([4, 4, 4, 1], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    
    tasks = 50 
    neurons_drift = []
    accuracies = []
    surprise_history = []
    
    print("Starting AGNIS V4.9 Stability Stress Test...")
    
    for t in range(tasks):
        # Generate a random task (Input -> Output mapping)
        # This simulates environmental noise and shifting manifolds
        x = torch.randn(1, 4)
        y = torch.randn(1, 1).sign() # Binary goal
        
        # 1. Observation
        weight, surprise = agent.observe_and_learn(x, y, task_id=t)
        surprise_history.append(metric_to_float(surprise))
        
        # 2. Replay & Potential Expansion
        # We manually trigger dream_replay more frequently to speed up the test
        agent.dream_replay(batch_size=8)
        
        current_neurons = hierarchy.layers[0].output_dim
        neurons_drift.append(current_neurons)
        
        if (t+1) % 10 == 0:
            print(f"Task {t+1}/{tasks} | L0 Neurons: {current_neurons} | Last Surprise: {metric_to_float(surprise):.4f}")

    # Final Assessment Checks
    # 1. Structural Drift check: Are V_mask/W_mask still valid?
    mask_integrity = True
    for col in hierarchy.layers:
        if col.V_mask.shape != col.V.shape: mask_integrity = False
        
    print("\n--- Assessment Results ---")
    print(f"Total Neurons Recruited: {hierarchy.layers[0].output_dim - 1}")
    print(f"Mask Integrity: {'OK' if mask_integrity else 'FAILED'}")
    print(f"Gradient Shielding Check: {torch.sum(hierarchy.layers[0].V_mask == 0).item()} nodes shielded.")

if __name__ == "__main__":
    stress_test_stability()
