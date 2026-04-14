"""
PHASE 31: 50-DOMAIN SIMULTANEOUS HARD MASTERY
LEARNING 50 UNIQUE HIGH-ENTROPY TASKS IN AN INTERLEAVED STREAM.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from enhanced_agnis_v2 import EnhancedAGNISV2
from complex_tasks import (
    create_parity_task, 
    create_reversal_task, 
    create_associative_task,
    create_algorithmic_task,
    create_structural_task
)

def generate_50_hard_tasks(samples_per_task: int = 20) -> List[Tuple[str, List[Tuple[torch.Tensor, torch.Tensor]]]]:
    tasks = []
    # 10 Variations of Parity (varying lengths)
    for i in range(10):
        seq_len = 8 + i
        tasks.append((f"Parity_L{seq_len}", create_parity_task(samples_per_task, seq_len)))
    
    # 10 Variations of Reversal (varying lengths)
    for i in range(10):
        seq_len = 8 + i
        tasks.append((f"Reversal_L{seq_len}", create_reversal_task(samples_per_task, seq_len)))
    
    # 10 Variations of Associative (varying dimensions)
    for i in range(10):
        dim = 10 + i
        tasks.append((f"Assoc_D{dim}", create_associative_task(samples_per_task, dim)))
    
    # 10 Variations of Algorithmic (varying dimensions)
    for i in range(10):
        dim = 10 + i
        tasks.append((f"Algo_D{dim}", create_algorithmic_task(samples_per_task, dim)))
    
    # 10 Variations of Structural (varying dimensions)
    for i in range(10):
        dim = 10 + i
        tasks.append((f"Struct_D{dim}", create_structural_task(samples_per_task, dim)))
        
    random.shuffle(tasks)
    return tasks[:50]

def run_simultaneous_hard_test():
    print("\n" + "="*70)
    print("PHASE 31: 50-DOMAIN SIMULTANEOUS HARD MASTERY")
    print("Testing Simultaneous Learning on High-Entropy Parallel Stream")
    print("="*70)

    # 1. Setup
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)

    samples_per_task = 50
    tasks = generate_50_hard_tasks(samples_per_task)
    
    agnis = EnhancedAGNISV2(
        input_dim=20, # Max dimension across all tasks
        output_dim=3,
        initial_hidden=80,
        neuron_dim=16,
        max_new_neurons_per_task=20,
        gating_enabled=True,
        shared_gate_ratio=0.15,
        oracle_enabled=True,
        oracle_window=15,
        replay_every=25
    )

    # Padding function for varying input sizes
    def pad_input(x: torch.Tensor, target_dim: int = 20) -> torch.Tensor:
        if x.shape[0] >= target_dim:
            return x[:target_dim]
        padding = torch.zeros(target_dim - x.shape[0])
        return torch.cat([x, padding])

    # 2. Build Interleaved Parallel Stream
    print(f"Building parallel stream of 50 hard domains...")
    stream = []
    # Interleave: 1 sample from each task, repeating for samples_per_task
    # This ensures "simultaneous" discovery and mastery demand.
    for s_idx in range(samples_per_task):
        for t_idx, (name, data) in enumerate(tasks):
            x, y = data[s_idx]
            stream.append((t_idx, pad_input(x), y))

    # 3. Execution
    print(f"[>] LEARNING {len(stream)} INTERLEAVED SAMPLES...")
    losses = []
    rolling_loss = []
    
    # Audit periodically
    audit_interval = 250 # Every 5 complete cycles
    audit_retention = []

    for i, (tid, x, y) in enumerate(stream):
        # The model must infer the task boundary and domain on every step
        loss = agnis.learn(x, y) # Let it infer task_id
        losses.append(loss)
        rolling_loss.append(loss)
        if len(rolling_loss) > 50: rolling_loss.pop(0)

        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{len(stream)} | Rolling Loss: {np.mean(rolling_loss):.4f} | Neurons: {len(agnis.neurons)}")

    # 4. Final Audit: Retention across all 50 tasks
    print("\n[Audit] Final Simultaneous Retention Sweep...")
    final_retention = []
    for tid in range(50):
        test_data = tasks[tid][1][-5:] # Last 5 samples
        test_loss = 0.0
        # Set context manually for clean audit
        agnis.begin_task(tid)
        for tx, ty in test_data:
            out = agnis.forward(pad_input(tx))
            test_loss += torch.mean((ty - out)**2).item()
        final_retention.append(test_loss / len(test_data))
        if tid % 10 == 0:
            print(f"    Task {tid} ({tasks[tid][0]}): {final_retention[-1]:.4f}")

    # 5. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(final_retention, marker='o', color='#2ecc71', label='Loss per Domain')
    plt.axhline(y=np.mean(final_retention), color='#e74c3c', linestyle='--', label=f'Avg: {np.mean(final_retention):.4f}')
    plt.title("Phase 31: Final Retention across 50 Simultaneous Domains", fontweight='bold')
    plt.xlabel("Domain Index")
    plt.ylabel("Loss (Lower is Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phase_31_simultaneous_retention.png', dpi=150)
    print("\n[OK] Results saved as 'phase_31_simultaneous_retention.png'")

    print("\n" + "="*70)
    print("SIMULTANEOUS MASTERY COMPLETE")
    stats = agnis.get_stats()
    print(f"  Final Neuron Count: {stats['current_neurons']}")
    print(f"  Total Steps: {stats['total_steps']}")
    print("="*70)

if __name__ == "__main__":
    run_simultaneous_hard_test()
