"""
PHASE 30: 50-DOMAIN SIMULTANEOUS MASTERY CHALLENGE
Testing AGNIS V2 with Prototype Replay and Neural Gating.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from enhanced_agnis_v2 import EnhancedAGNISV2

def create_task(num_samples: int, input_dim: int, output_dim: int):
    x = torch.randn(num_samples, input_dim)
    w = torch.randn(input_dim, output_dim)
    y = torch.matmul(x, w)
    # Convert to classification targets (one-hot or indices)
    y = torch.argmax(y, dim=1)
    # One-hot encoding
    y_one_hot = torch.zeros(num_samples, output_dim)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    return list(zip(x, y_one_hot))

def run_50_domain_test():
    print("\n" + "="*70)
    print("PHASE 30: 50-DOMAIN SIMULTANEOUS MASTERY CHALLENGE")
    print("Engine: AGNIS V2 (Gated & Oracle-Enhanced)")
    print("="*70)

    # 1. Setup
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    input_dim = 10
    output_dim = 3
    num_tasks = 50
    epochs_per_task = 15

    agnis = EnhancedAGNISV2(
        input_dim=input_dim,
        output_dim=output_dim,
        initial_hidden=60,
        neuron_dim=16,
        max_new_neurons_per_task=15,
        replay_capacity_per_task=50,
        replay_every=20,
        gating_enabled=True,
        shared_gate_ratio=0.2,
        freeze_base_after_tasks=5,
        oracle_enabled=True,
        prototype_only_after=5 # Fast prototyping for this test
    )

    # 2. Generate Data
    print(f"Generating {num_tasks} unique tasks...")
    task_sequence = []
    for i in range(num_tasks):
        dataset = create_task(100, input_dim, output_dim)
        task_sequence.append((f"Task_{i}", dataset))

    # 3. Training Loop
    retention_matrix = []
    
    for t_idx, (name, dataset) in enumerate(task_sequence):
        print(f"\n[Task {t_idx+1}/{num_tasks}] Learning: {name}")
        agnis.begin_task(t_idx)
        
        for epoch in range(epochs_per_task):
            total_loss = 0.0
            random.shuffle(dataset)
            for x, y in dataset:
                loss = agnis.learn(x, y, task_id=t_idx)
                total_loss += loss
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_task} | Avg Loss: {total_loss/len(dataset):.4f}")

        # Finalize and Lock
        # (Assuming consolidate exists or we just move on)
        # agnis.consolidate_task(t_idx, strength=0.3)

        # 4. Evaluation (Selective for speed)
        # Check Task 0 and every 10th task to monitor drift
        if (t_idx + 1) % 10 == 0 or t_idx == num_tasks - 1:
            print(f"\n  [Audit] Testing Retention after {t_idx+1} tasks:")
            retention_row = []
            for test_idx in range(t_idx + 1):
                # Use only first 10 samples for fast probing
                test_loss = 0.0
                test_data = task_sequence[test_idx][1][:10]
                # Manually set task id for eval
                agnis.current_task_id = test_idx
                if agnis.gating_enabled:
                    agnis.active_neurons = agnis.task_gate_masks[test_idx]
                
                for tx, ty in test_data:
                    out = agnis.forward(tx)
                    test_loss += torch.mean((ty - out)**2).item()
                avg_loss = test_loss / len(test_data)
                retention_row.append(avg_loss)
                if test_idx % 10 == 0 or test_idx == t_idx:
                    print(f"    Retention {task_sequence[test_idx][0]}: {avg_loss:.4f}")
            
            while len(retention_row) < num_tasks:
                retention_row.append(np.nan)
            retention_matrix.append(retention_row)

    # 5. Final Visualization
    print("\n[Graph] Generating 50-Domain Retention Heatmap...")
    plt.figure(figsize=(14, 10))
    ret_array = np.array(retention_matrix)
    sns.heatmap(
        ret_array,
        annot=False,
        cmap="RdYlGn_r",
        xticklabels=False,
        yticklabels=[f"After T{10*(i+1)}" if (10*(i+1)) <= num_tasks else f"After T{num_tasks}" for i in range(len(retention_matrix))],
        cbar_kws={"label": "Loss (Stability)"}
    )
    plt.title("Phase 30: 50-Domain Simultaneous Mastery Challenge", fontweight='bold')
    plt.xlabel("Domain Index (T0 - T49)")
    plt.ylabel("Training Milestone")
    plt.tight_layout()
    plt.savefig('giga_storm_v2_50_task_results.png', dpi=150)
    print("\n[OK] Results saved as 'giga_storm_v2_50_task_results.png'")

    stats = agnis.get_stats()
    print("\n" + "="*70)
    print("CHALLENGE COMPLETE")
    print(f"  Final Neuron Count: {stats['current_neurons']}")
    print(f"  Total Steps: {stats['total_steps']}")
    print("="*70)

if __name__ == "__main__":
    run_50_domain_test()
