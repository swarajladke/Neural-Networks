import torch
import random
import time
import numpy as np
from typing import List, Tuple, Dict
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, Experience
from complex_tasks import (
    create_parity_task,
    create_reversal_task,
    create_associative_task,
    create_algorithmic_task,
    create_structural_task
)

class DomainManager:
    """Generates 200 systematic variations of core symbolic tasks."""
    def __init__(self, input_dim: int = 12, samples_per_task: int = 50):
        self.input_dim = input_dim
        self.samples_per_task = samples_per_task
        self.task_names = []
        self.datasets = []
        
        # 5 Types, 40 Variations each = 200 domains
        types = [
            ("Parity", create_parity_task),
            ("Reversal", create_reversal_task),
            ("Associative", create_associative_task),
            ("Algorithmic", create_algorithmic_task),
            ("Structural", create_structural_task)
        ]
        
        for v in range(40):
            for t_name, gen_fn in types:
                # Vary complexity L/D based on version and type
                # L/D ranges from 6 to 15 (rolling cycle)
                l_val = 6 + (v % 10)
                full_name = f"{t_name}_L{l_val}_V{v}" if t_name in ["Parity", "Reversal"] else f"{t_name}_D{l_val}_V{v}"
                
                # Deterministic seed for this version
                torch.manual_seed(v * 100)
                random.seed(v * 100)
                
                raw_data = gen_fn(num_samples=samples_per_task, seq_len=l_val) if t_name in ["Parity", "Reversal"] else gen_fn(num_samples=samples_per_task, dim=l_val)
                
                # Pad inputs to fixed input_dim
                padded_data = []
                for x, y in raw_data:
                    x_fixed = torch.zeros(self.input_dim)
                    x_fixed[:min(len(x), self.input_dim)] = x[:self.input_dim]
                    padded_data.append((x_fixed.unsqueeze(0), y.unsqueeze(0)))
                
                self.task_names.append(full_name)
                self.datasets.append(padded_data)

def evaluate_retention(agent: CognitivePredictiveAgent, datasets: List, task_indices: List[int]) -> float:
    """Evaluates mean MSE across a set of task indices."""
    total_mse = 0.0
    with torch.no_grad():
        for idx in task_indices:
            data = datasets[idx]
            task_mse = 0.0
            # Sample 5 probes per task
            probes = random.sample(data, min(len(data), 5))
            for x, y in probes:
                pred = agent.hierarchy.predict_label(x, max_steps=150)
                task_mse += torch.nn.functional.mse_loss(pred[:, :3], y[:, :3]).item()
            total_mse += (task_mse / len(probes))
    return total_mse / len(task_indices)

def run_100_domain_benchmark():
    print("="*60)
    print(" AGNIS V4.9: 100-DOMAIN MARATHON BENCHMARK")
    print("="*60)
    
    device = "cpu"
    INPUT_DIM = 12
    OUTPUT_DIM = 3
    STEPS_PER_TASK = 5 # Balanced for quality and speed
    
    manager = DomainManager(input_dim=INPUT_DIM, samples_per_task=50)
    manager.task_names = manager.task_names[:100]
    manager.datasets = manager.datasets[:100]
    
    hierarchy = PredictiveHierarchy([INPUT_DIM, 16, 16, OUTPUT_DIM], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    
    plot_data = [] # For the retention vs domain index chart
    results = []
    start_time = time.time()
    
    # Sequential Stream: Task-by-Task
    for t_idx, task_name in enumerate(manager.task_names):
        task_data = manager.datasets[t_idx]
        task_mse_accum = 0.0
        
        # Train for STEPS_PER_TASK on this domain
        indices = list(range(len(task_data)))
        random.shuffle(indices)
        
        for i in range(STEPS_PER_TASK):
            sample_idx = indices[i % len(indices)]
            x, y = task_data[sample_idx]
            agent.observe_and_learn(x, y, task_id=t_idx)
            
        # Consolidation Dream after each task
        agent.dream_replay(batch_size=min(16, len(agent.buffer)))
        
        # Periodic Evaluation of the current task retention
        with torch.no_grad():
            probes = random.sample(task_data, 5)
            current_mse = 0.0
            for px, py in probes:
                pred = agent.hierarchy.predict_label(px)
                current_mse += torch.nn.functional.mse_loss(pred[:, :3], py[:, :3]).item()
            plot_data.append(current_mse / 5.0)

        # Periodic Reporting
        if (t_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Task {t_idx+1}/100 | {task_name:25s} | N: {hierarchy.layers[0].output_dim} | {elapsed:.1f}s")
            
    # Final Result
    import json
    with open('v4_9_benchmark_plot_data_100.json', 'w') as f:
        json.dump(plot_data, f)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"BENCHMARK 100 COMPLETE in {total_time/60:.2f} minutes")
    print(f"Data saved to v4_9_benchmark_plot_data_100.json")
    print("="*60)

if __name__ == "__main__":
    run_100_domain_benchmark()
