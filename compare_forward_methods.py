"""
DIAGNOSTIC: Comparing Forward Methods
Runs standard and tensorized forward passes on identical AGNIS V2 state.
"""

import torch
import numpy as np
from enhanced_agnis_v2 import EnhancedAGNISV2
from complex_tasks import create_parity_task

def run_comparison():
    print("--- ARCHITECTURAL DRIFT AUDIT ---")
    
    # 1. Setup
    torch.manual_seed(42)
    agnis = EnhancedAGNISV2(input_dim=10, output_dim=3)
    
    # 2. Mock a task and some weights
    task_id = 0
    agnis.begin_task(task_id)
    
    # Force some data into task_usage to avoid 0.5 defaults
    for nid in agnis.neurons:
        agnis.neurons[nid].task_usage[task_id] = 1.0
        agnis.neurons[nid].memory = torch.randn(agnis.neuron_dim)

    test_x = torch.randn(10)
    
    print("\n[Input Grounding]")
    print(f"X: {test_x.mean().item():.4f}")

    # 3. Execution Standard
    agnis.infer_task(test_x) # Sets active_neurons
    
    # Reset activations for clean test
    for n in agnis.neurons.values():
        n.activation = torch.zeros(agnis.neuron_dim)
        
    y_std = agnis.forward(test_x, num_steps=2)
    
    # 4. Execution Tensorized
    # Reset activations AGAIN for identical start
    for n in agnis.neurons.values():
        n.activation = torch.zeros(agnis.neuron_dim)
        
    y_vec = agnis.forward_tensorized(test_x, num_steps=2, active_only=True)

    # 5. Delta Analysis
    print("\n[Output Matrix]")
    print(f"Standard:   {y_std.detach().numpy()}")
    print(f"Tensorized: {y_vec.detach().numpy()}")
    
    diff = torch.abs(y_std - y_vec).mean().item()
    print(f"\nAbsolute Mean Error: {diff:.8f}")
    
    if diff < 1e-6:
        print(">>> RESULT: MATHEMATICAL PARITY (V2.1 STANDARD)")
    else:
        print(">>> RESULT: STRUCTURAL DRIFT DETECTED")
        
        # Check if it's due to active_only
        print("\nVerifying active_only vs full_graph...")
        y_vec_full = agnis.forward_tensorized(test_x, num_steps=2, active_only=False)
        diff_full = torch.abs(y_std - y_vec_full).mean().item()
        print(f"Full Graph Tensorized Error: {diff_full:.8f}")

if __name__ == "__main__":
    run_comparison()
