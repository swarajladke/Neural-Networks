"""
DUAL-TASK STORM VERIFICATION
Testing AGNIS V2 with simultaneous high-entropy tasks (Parity & Reversal)
"""

import torch
import random
import numpy as np
from enhanced_agnis_v2 import EnhancedAGNISV2, train_agnis_v2
from complex_tasks import create_parity_task, create_reversal_task

def run_dual_verification():
    # 1. Initialization
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    agnis = EnhancedAGNISV2(
        input_dim=10,
        output_dim=3,
        initial_hidden=40,
        neuron_dim=12,
        max_new_neurons_per_task=20,
        replay_capacity_per_task=100,
        replay_every=10,  # Frequent replay for high-entropy stabilization
        self_rewriter_enabled=True,
        self_audit_enabled=True
    )

    # 2. Generate Interleaved Task Data
    print("Generating High-Entropy Task Data...")
    parity_data = create_parity_task(100, 10)
    reversal_data = create_reversal_task(100, 10)

    task_sequence = [
        ("Parity-N", parity_data),
        ("Reversal", reversal_data)
    ]

    # 3. Execute Simultaneous Storm Training
    # self_directed=True will interleave tasks and use task inference
    print("\nStarting Dual-Task Storm (Simultaneous Training)...")
    train_agnis_v2(
        agnis, 
        task_sequence, 
        epochs_per_task=25, 
        known_boundaries=False, # Model must autonomously infer task boundaries
        self_directed=True,
        self_directed_cycles=150,
        samples_per_epoch=5
    )

    # 4. Final Retention Stats
    stats = agnis.get_stats()
    print("\n" + "="*70)
    print("POST-STORM ARCHITECTURAL STATUS")
    print("="*70)
    print(f"Total Neurons: {stats['current_neurons']}")
    print(f"Total Connections: {stats['current_connections']}")
    print(f"Schemas Discovered: {stats['schemas']}")
    print(f"Self-Stability: {stats['self_stability']:.4f}")
    
    # 5. Experimental Verification: Tensorized Forward Pass
    print("\nVerifying Tensorized Message Passing Prototype...")
    test_x, test_y = parity_data[0]
    
    # Ensure current_task_id is set for gating
    agnis.infer_task(test_x)
    
    y_std = agnis.forward(test_x)
    y_tensor = agnis.forward_tensorized(test_x)
    
    print(f"Standard Forward Output:   {y_std.detach().numpy()}")
    print(f"Tensorized Forward Output: {y_tensor.detach().numpy()}")
    
    # Check parity prediction (Target 0: Even, 1: Odd)
    parity_pred = torch.argmax(y_tensor).item()
    target_parity = torch.argmax(test_y).item()
    print(f"Target Parity Index: {target_parity} | Predicted: {parity_pred}")

if __name__ == "__main__":
    run_dual_verification()
