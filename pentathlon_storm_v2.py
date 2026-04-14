"""
PENTATHLON STORM V2: Simultaneous 5-Task Verification
Testing AGNIS V2 with 5 high-entropy domains in an interleaved data stream.
"""

from __future__ import annotations
import torch
import random
import numpy as np
from enhanced_agnis_v2 import EnhancedAGNISV2, train_agnis_v2
from complex_tasks import (
    create_parity_task, 
    create_reversal_task, 
    create_associative_task, 
    create_algorithmic_task, 
    create_structural_task
)

def run_pentathlon_storm():
    # 1. Initialization
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("--- ARCHITECTING PENTATHLON STORM ---")
    agnis = EnhancedAGNISV2(
        input_dim=10,
        output_dim=3,
        initial_hidden=80,
        neuron_dim=12,
        max_new_neurons_per_task=30,
        replay_capacity_per_task=200,
        replay_every=20,
        physical_intuition_enabled=True,
        self_rewriter_enabled=True,
        self_audit_enabled=True,
        mastery_mode_enabled=True,
        mastery_loss_threshold=0.15,
        mastery_lr_multiplier=10.0,
        mastery_corr_ema_scale=0.6,
        mastery_forward_steps=3,
        hebb_active_threshold=0.01
    )

    # 2. Generate 5 Symbolic Datasets
    print("Generating Datasets for 5 Domains...")
    task_sets = [
        ("Parity-N", create_parity_task(100, 10)),
        ("Reversal", create_reversal_task(100, 10)),
        ("Associative", create_associative_task(100, 10)),
        ("Algorithmic", create_algorithmic_task(100, 10)),
        ("Structural", create_structural_task(100, 10))
    ]

    # 3. Simultaneous Storm Execution
    # self_directed=True: Tasks are shuffled and boundaries are inferred.
    print("\nStarting Phase 35: Simultaneous Pentathlon Storm...")
    train_agnis_v2(
        agnis, 
        task_sets, 
        epochs_per_task=5, 
        known_boundaries=False, 
        self_directed=True,
        self_directed_cycles=20, 
        samples_per_epoch=10
    )

    # 4. Final Diagnostics
    stats = agnis.get_stats()

    # 5. Final Cognitive Diagnostics
    print("\n" + "="*70)
    print("PHASE 35 RECOVERY & RETENTION REPORT")
    print("="*70)
    print(f"Total Cognitive Neurons: {stats['current_neurons']}")
    print(f"Total Inter-Synapses:    {stats['current_connections']}")
    print(f"Schemas Clustered:       {stats['schemas']}")
    print(f"Global Stability:        {stats['self_stability']:.4f}")

    # 6. Comparative Inference Audit
    print("\nExecuting Inference Comparison (Standard vs Tensorized)...")
    test_x, test_y = task_sets[0][1][0] # Test on Parity Task
    agnis.infer_task(test_x)
    
    # Run Standard Pass
    # Reset activations for clean baseline
    for n in agnis.neurons.values():
        n.activation = torch.zeros(agnis.neuron_dim)
    y_std = agnis.forward(test_x, num_steps=2)
    
    # Run Tensorized Pass
    for n in agnis.neurons.values():
        n.activation = torch.zeros(agnis.neuron_dim)
    y_vec = agnis.forward_tensorized(test_x, num_steps=2, active_only=True)
    
    pred_std = torch.argmax(y_std).item()
    pred_vec = torch.argmax(y_vec).item()
    target_idx = torch.argmax(test_y).item()
    
    print(f"Standard Prediction:   {pred_std} | Values: {y_std.detach().numpy().flatten()}")
    print(f"Tensorized Prediction: {pred_vec} | Values: {y_vec.detach().numpy().flatten()}")
    print(f"Target Index:          {target_idx}")
    
    # 7. Verdict Analysis
    parity_drift = torch.abs(y_std - y_vec).mean().item()
    print(f"\n[Verdict Analysis]")
    print(f"Architectural Parity Drift: {parity_drift:.8f}")
    
    if parity_drift < 1e-4:
        print(">>> STATUS: ARCHITECTURAL PARITY (V2.1 VERIFIED)")
    else:
        print(">>> STATUS: STRUCTURAL DRIFT DETECTED (PARITY ERROR)")

    if pred_std == target_idx:
        print(">>> STATUS: MODEL PREDICTION SUCCESS")
    else:
        print(">>> STATUS: MODEL PREDICTION FAILURE (Requires More Training)")

if __name__ == "__main__":
    run_pentathlon_storm()
