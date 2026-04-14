"""
PHASE 27: THE GIGA DATA STORM
Automated trial: 50 simultaneous complex reasoning tasks.
"""

import torch
import random
import numpy as np
import time
from enhanced_agnis_v2 import EnhancedAGNISV2
from complex_tasks import (
    create_parity_task, 
    create_reversal_task, 
    create_associative_task,
    create_algorithmic_task,
    create_structural_task
)

def generate_giga_dataset(num_tasks: int, samples_per_task: int = 15):
    """Generate a diverse mix of 50 complex tasks"""
    mega_tasks = []
    
    for i in range(num_tasks):
        task_type = i % 5
        seed = i * 222
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Varying lengths for symbolic complexity
        length = 8 + (i // 10)
        
        if task_type == 0:
            name = f"Parity_L{length}"
            dataset = create_parity_task(samples_per_task, length)
        elif task_type == 1:
            name = f"Reversal_S{length}"
            dataset = create_reversal_task(samples_per_task, length)
        elif task_type == 2:
            name = f"Associative_{i}"
            dataset = create_associative_task(samples_per_task, 10)
        elif task_type == 3:
            name = f"AlgoShift_{i}"
            dataset = create_algorithmic_task(samples_per_task, 10)
        else:
            name = f"Syntax_{i}"
            dataset = create_structural_task(samples_per_task, 10)
            
        mega_tasks.append((name, dataset))
    
    return mega_tasks

def run_giga_storm():
    num_tasks = 50
    print(f"\n\n{'#'*80}")
    print(f"### PHASE 27: GIGA DATA STORM ({num_tasks} SIMULTANEOUS TASKS)")
    print(f"{'#'*80}")
    
    start_time = time.time()
    tasks = generate_giga_dataset(num_tasks)
    
    agnis = EnhancedAGNISV2(
        input_dim=10, 
        output_dim=3,
        initial_hidden=80, 
        neuron_dim=12,
        max_new_neurons_per_task=10, # Lean growth for speed
        replay_every=200 # Heavy thinning for Giga Storm speed
    )
    
    # Interleave everything for simultaneous learning
    full_train_stream = []
    for tid, (name, dataset) in enumerate(tasks):
        for x, y in dataset:
            full_train_stream.append((x, y, tid))
            
    random.shuffle(full_train_stream)
    
    # 5 Epochs of the Data Storm
    epochs = 8
    print(f"\n🚀 LAUNCHING THE 50-TASK STORM ({len(full_train_stream)} steps per epoch)...")
    
    for epoch in range(epochs):
        random.shuffle(full_train_stream)
        epoch_loss = 0.0
        for x, y, _ in full_train_stream:
            # Fully autonomous inference across 50 domains
            loss = agnis.learn(x, y, task_id=None)
            epoch_loss += loss
        
        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(full_train_stream):.4f} - Time: {elapsed:.1f}m")
        
        if elapsed > 55: # Safety brake for the 1-hour window
            print("Reached 1-hour threshold. Finalizing...")
            break
            
    # Evaluation
    print("\n📊 GIGA STORM ACCURACY AUDIT:")
    total_loss = 0.0
    mastered = 0
    for name, dataset in tasks:
        task_loss = 0.0
        for x, y in dataset[:5]:
            out = agnis.forward(x)
            task_loss += ((out - y)**2).mean().item()
        avg_l = task_loss / 5
        total_loss += avg_l
        if avg_l < 0.6: mastered += 1
        
    avg_total_loss = total_loss / num_tasks
    print(f"\nFINAL GIGA RESULTS:")
    print(f"  Total Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"  Average Loss: {avg_total_loss:.4f}")
    print(f"  Tasks Mastered: {mastered}/{num_tasks}")
    
    stats = agnis.get_stats()
    print(f"  Total Neurons: {stats['current_neurons']}")
    print(f"  Total Connections: {stats['current_connections']}")
    print(f"  Tasks Inferred: {stats['tasks_learned']}")
    
    # Log to a special Giga Storm file
    with open("giga_storm_results.txt", "w") as f:
        f.write(f"AGNIS GIGA STORM RESULTS\n")
        f.write(f"Tasks: {num_tasks}\n")
        f.write(f"Mastered: {mastered}/{num_tasks}\n")
        f.write(f"Avg Loss: {avg_total_loss:.4f}\n")
        f.write(f"Neurons: {stats['current_neurons']}\n")
        f.write(f"Time: {(time.time() - start_time)/60:.1f}m\n")

if __name__ == "__main__":
    run_giga_storm()
