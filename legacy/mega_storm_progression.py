"""
PHASE 26: THE MEGA DATA STORM
Automated progression: 10 -> 20 -> 30 simultaneous complex tasks.
"""

import torch
import random
import numpy as np
import os
import time
from enhanced_agnis_v2 import EnhancedAGNISV2, train_agnis_v2
from complex_tasks import (
    create_parity_task, 
    create_reversal_task, 
    create_associative_task,
    create_algorithmic_task,
    create_structural_task
)

def generate_mega_dataset(num_tasks: int, samples_per_task: int = 30):
    """Generate a mix of complex tasks with varying parameters"""
    mega_tasks = []
    
    for i in range(num_tasks):
        task_type = i % 5
        seed = i * 100
        random.seed(seed)
        torch.manual_seed(seed)
        
        if task_type == 0:
            name = f"Parity_L{8+(i//5)}"
            dataset = create_parity_task(samples_per_task, 8+(i//5))
        elif task_type == 1:
            name = f"Reversal_S{10+(i%3)}"
            dataset = create_reversal_task(samples_per_task, 10+(i%3))
        elif task_type == 2:
            name = f"Associative_V{i}"
            dataset = create_associative_task(samples_per_task, 10)
        elif task_type == 3:
            name = f"AlgoShift_{i}"
            dataset = create_algorithmic_task(samples_per_task, 10)
        else:
            name = f"Syntax_{i}"
            dataset = create_structural_task(samples_per_task, 10)
            
        mega_tasks.append((name, dataset))
    
    return mega_tasks

def run_mega_storm(num_tasks: int, agnis=None):
    print(f"\n\n{'#'*80}")
    print(f"### STAGE: {num_tasks} SIMULTANEOUS TASKS")
    print(f"{'#'*80}")
    
    # Optimized for speed
    tasks = generate_mega_dataset(num_tasks, samples_per_task=20)
    
    if agnis is None:
        agnis = EnhancedAGNISV2(
            input_dim=10, 
            output_dim=3,
            initial_hidden=min(40 + num_tasks*2, 120), 
            neuron_dim=16,
            max_new_neurons_per_task=15,
            replay_every=15
        )
    
    # Interleave everything for simultaneous learning
    full_train_stream = []
    for tid, (name, dataset) in enumerate(tasks):
        for x, y in dataset:
            full_train_stream.append((x, y, tid))
            
    random.shuffle(full_train_stream)
    
    epochs = 25 # High speed for progression
    for epoch in range(epochs):
        random.shuffle(full_train_stream)
        epoch_loss = 0.0
        for x, y, _ in full_train_stream:
            # Autonomous inference mode
            loss = agnis.learn(x, y, task_id=None)
            epoch_loss += loss
        
        if (epoch + 1) % 5 == 0:
            print(f"  [{num_tasks} Tasks] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(full_train_stream):.4f}")
            
    # Evaluation
    success_count = 0
    total_loss = 0.0
    for name, dataset in tasks:
        task_loss = 0.0
        for x, y in dataset[:10]:
            out = agnis.forward(x)
            task_loss += ((out - y)**2).mean().item()
        avg_l = task_loss / 10
        total_loss += avg_l
        if avg_l < 0.5: success_count += 1
        
    avg_total_loss = total_loss / num_tasks
    print(f"\nSTAGE RESULTS:")
    print(f"  Average Loss: {avg_total_loss:.4f}")
    print(f"  Tasks Mastered: {success_count}/{num_tasks}")
    print(f"  Neurons: {len(agnis.neurons)}")
    
    return agnis, avg_total_loss < 0.6 # Success threshold

if __name__ == "__main__":
    results_log = "mega_storm_results.txt"
    with open(results_log, "w") as f:
        f.write("AGNIS MEGA STORM LOG\n")
        f.write("="*30 + "\n")
        
    current_agnis = None
    stages = [10, 20, 30]
    
    for s in stages:
        current_agnis, success = run_mega_storm(s, current_agnis)
        
        with open(results_log, "a") as f:
            f.write(f"Stage {s} Tasks: {'SUCCESS' if success else 'FAILURE'}\n")
            f.write(f"  Neurons: {len(current_agnis.neurons)}\n")
            f.write(f"  Steps: {current_agnis.stats['total_steps']}\n\n")
            
        if not success:
            print(f"Stopping at Stage {s} due to high entropy collapse.")
            break
            
    print("\nMEGA STORM COMPLETE. Check mega_storm_results.txt")
