"""
PHASE 31 VERIFICATION: 50-DOMAIN RECALL TEST
Query the trained AGNIS V2 model on samples from all 50 domains.
"""

import torch
import numpy as np
import random
from enhanced_agnis_v2 import EnhancedAGNISV2
from complex_tasks import (
    create_parity_task, 
    create_reversal_task, 
    create_associative_task,
    create_algorithmic_task,
    create_structural_task
)

def generate_50_hard_tasks(samples_per_task: int = 5):
    """Generate test samples for all 50 domains."""
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
        
    return tasks

def pad_input(x: torch.Tensor, target_dim: int = 20) -> torch.Tensor:
    if x.shape[0] >= target_dim:
        return x[:target_dim]
    padding = torch.zeros(target_dim - x.shape[0])
    return torch.cat([x, padding])

def run_recall_test():
    print("\n" + "="*70)
    print("PHASE 31 VERIFICATION: 50-DOMAIN RECALL TEST")
    print("Testing if the model accurately recalls knowledge from all 50 domains")
    print("="*70)

    # Set seeds for reproducibility
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)

    # Initialize the model with same config as training
    agnis = EnhancedAGNISV2(
        input_dim=20,
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

    # Note: The model needs to be retrained or loaded from a checkpoint
    # For this demo, we'll do a quick train and then test
    print("\n[1/3] Quick training on all 50 domains...")
    
    train_tasks = generate_50_hard_tasks(samples_per_task=20)
    
    # Quick interleaved training
    stream = []
    for s_idx in range(20):
        for t_idx, (name, data) in enumerate(train_tasks):
            x, y = data[s_idx]
            stream.append((t_idx, pad_input(x), y))
    
    for i, (tid, x, y) in enumerate(stream):
        agnis.learn(x, y)
        if (i + 1) % 200 == 0:
            print(f"  Training step {i+1}/{len(stream)}")
    
    print("\n[2/3] Generating fresh test samples...")
    test_tasks = generate_50_hard_tasks(samples_per_task=3)
    
    print("\n[3/3] RECALL TEST RESULTS:")
    print("-" * 70)
    
    total_correct = 0
    total_samples = 0
    
    for t_idx, (name, data) in enumerate(test_tasks):
        agnis.begin_task(t_idx)
        task_correct = 0
        task_total = len(data)
        
        for x, y in data:
            output = agnis.forward(pad_input(x))
            
            # For classification tasks, check if the argmax matches
            pred_class = torch.argmax(output).item()
            true_class = torch.argmax(y).item()
            
            if pred_class == true_class:
                task_correct += 1
                total_correct += 1
            total_samples += 1
        
        accuracy = (task_correct / task_total) * 100
        status = "✓" if accuracy >= 66.7 else "✗"
        
        if t_idx % 5 == 0:  # Print every 5th domain for brevity
            print(f"  {status} Domain {t_idx:2d} ({name:15s}): {task_correct}/{task_total} correct ({accuracy:.1f}%)")
    
    print("-" * 70)
    overall_accuracy = (total_correct / total_samples) * 100
    print(f"\n📊 OVERALL RECALL ACCURACY: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy >= 50:
        print("🏆 VERDICT: Model demonstrates significant recall across 50 domains!")
    else:
        print("⚠️ VERDICT: Model needs more training for reliable recall.")
    
    print("="*70)

if __name__ == "__main__":
    run_recall_test()
