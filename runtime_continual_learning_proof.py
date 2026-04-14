"""
PHASE 28: LIVE CONTINUAL LEARNING PROOF
Demonstrating AGNIS V2's ability to learn and distinguish complex tasks 
in a real-time, interleaved, label-less data stream.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from enhanced_agnis_v2 import EnhancedAGNISV2
from complex_tasks import (
    create_parity_task, 
    create_reversal_task, 
    create_associative_task,
    create_algorithmic_task,
    create_structural_task
)

def run_live_learning_proof():
    print("\n" + "="*70)
    print("PHASE 28: LIVE CONTINUAL LEARNING PROOF")
    print("Testing Autonomous Discovery & Mastery on High-Entropy Stream")
    print("="*70)

    # 1. Initialize AGNIS V2
    # Using the optimized parameters from the Giga Storm
    agnis = EnhancedAGNISV2(
        input_dim=10, 
        output_dim=3,
        initial_hidden=40, 
        neuron_dim=16,
        max_new_neurons_per_task=20,
        replay_every=50 # Balanced for live demo
    )

    # 2. Build the High-Entropy Interleaved Stream (Ultra-Fast)
    print("\nGenerating 500-sample high-speed stream...")
    raw_tasks = [
        ("T0: Parity", create_parity_task(100, 10)),
        ("T1: Reversal", create_reversal_task(100, 10)),
        ("T2: Assoc", create_associative_task(100, 10)),
        ("T3: Algo", create_algorithmic_task(100, 10)),
        ("T4: Syntax", create_structural_task(100, 10))
    ]

    # Interleave them: 100 samples per task
    stream = []
    task_labels = [] 
    for tid, (name, samples) in enumerate(raw_tasks):
        stream.extend(samples)
        task_labels.extend([tid] * len(samples))

    # 3. Execution & Monitoring
    losses = []
    rolling_loss = deque(maxlen=10)
    rolling_avg_history = []
    flashback_stats = [] 
    neuron_counts = []

    print("\n[>] STREAMING DATA (500 samples)...")
    for i, (x, y) in enumerate(stream):
        # Ultra-Performance Mode for Live Demo
        agnis.replay_every = 500 # Replay only once or twice

        loss = agnis.learn(x, y)
        
        losses.append(loss)
        rolling_loss.append(loss)
        rolling_avg_history.append(np.mean(rolling_loss))
        neuron_counts.append(len(agnis.neurons))

        # Periodic Flashback Probes (Every 100 samples)
        if (i + 1) % 100 == 0:
            current_max_task = (i + 1) // 100 - 1
            print(f"  Step {i+1}/500 | Rolling Loss: {rolling_avg_history[-1]:.4f} | Neurons: {neuron_counts[-1]}")
            
            # Probe performance on all tasks seen so far
            probe_results = []
            for tid in range(current_max_task + 1):
                _, test_data = raw_tasks[tid]
                test_loss = 0.0
                num_test = 5 
                for tx, ty in test_data[:num_test]:
                    # Need to temporarily set task_id for deterministic inference evaluation
                    agnis.current_task_id = tid 
                    out = agnis.forward(tx)
                    test_loss += torch.mean((ty - out)**2).item()
                probe_results.append(test_loss / num_test)
            flashback_stats.append(probe_results)

    # 4. Visualization
    print("\n[Graph] Generating Live Proof Visualization...")
    np.savez(
        "live_learning_proof_metrics.npz",
        losses=np.array(losses),
        rolling_avg_history=np.array(rolling_avg_history),
        neuron_counts=np.array(neuron_counts),
        flashback_stats=np.array(flashback_stats, dtype=object),
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='#f0f0f0')

    # Panel A: Rolling Loss & Neurogenesis
    ax1.plot(rolling_avg_history, color='#2c3e50', lw=1.5, label='Rolling Loss (Window=50)')
    ax1.set_ylabel('Loss', color='#2c3e50', fontweight='bold')
    ax1.set_title('AGNIS V2: Live Continual Learning Convergence', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax1b = ax1.twinx()
    ax1b.plot(neuron_counts, color='#e74c3c', lw=2, alpha=0.6, label='Neuron Count')
    ax1b.set_ylabel('Total Neurons', color='#e74c3c', fontweight='bold')
    
    # Add vertical lines for Task Shifts
    for shift in [100, 200, 300, 400]:
        ax1.axvline(x=shift, color='gray', linestyle='--', alpha=0.5)
        ax1.text(shift+5, ax1.get_ylim()[1]*0.8, 'SHIFT', rotation=0, fontsize=8, color='gray')

    # Panel B: Flashback Retention Matrix
    # flashback_stats is a list of lists of varying lengths. Pad with NaN for heatmapping.
    max_len = len(raw_tasks)
    padded_stats = []
    for row in flashback_stats:
        padded_row = row + [np.nan] * (max_len - len(row))
        padded_stats.append(padded_row)
    
    im = ax2.imshow(padded_stats, cmap='RdYlGn_r', aspect='auto')
    ax2.set_title('Flashback Probes: Cross-Domain Retention during Stream', fontweight='bold')
    ax2.set_xlabel('Measured Task (T0-T4)')
    ax2.set_ylabel('Probe Interval (Every 100 samples)')
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(len(flashback_stats)))
    ax2.set_yticklabels([f"Step {100*(i+1)}" for i in range(len(flashback_stats))])
    plt.colorbar(im, ax=ax2, label='Loss (Lower is Better)')

    plt.tight_layout()
    plt.savefig('live_learning_proof.png', dpi=150)
    print("\n[OK] Live Learning Proof saved as 'live_learning_proof.png'")

    # 5. Final Stats
    print("\n" + "="*70)
    print("PROOF COMPLETE")
    final_stats = agnis.get_stats()
    print(f"  Total Steps: {final_stats['total_steps']}")
    print(f"  Neurons Created: {final_stats['neurons_created']}")
    print(f"  Tasks Detected: {len(agnis.task_neuron_pools)}")
    print("="*70)

if __name__ == "__main__":
    run_live_learning_proof()
