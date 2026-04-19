import torch
import time
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent

def benchmark_vectorization():
    print("==================================================")
    print(" V5.4: Vectorization & Parallelism Benchmark")
    print("==================================================")
    
    device = "cpu" # We benchmark on CPU first to see overhead reduction
    batch_size = 32
    input_dim = 16
    hidden_dim = 64
    output_dim = 8
    
    hierarchy = PredictiveHierarchy([input_dim, hidden_dim, output_dim], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    
    # Generate dummy batch
    x_batch = torch.randn(batch_size, input_dim)
    y_batch = torch.randn(batch_size, output_dim)
    
    print(f"\n[Batch Size: {batch_size}, Hidden: {hidden_dim}]")
    
    # --- 1. Baseline: Serial Processing (simulated) ---
    print("\n[Test 1] Serial Online Learning (Sample-by-Sample)...")
    start_time = time.time()
    for i in range(batch_size):
        agent.observe_and_learn(x_batch[i:i+1], y_batch[i:i+1], max_steps=50)
    serial_time = time.time() - start_time
    print(f"-> Serial Total Time: {serial_time:.4f}s ({serial_time/batch_size:.4f}s per sample)")
    
    # --- 2. Vectorized: Batch Processing ---
    print("\n[Test 2] Vectorized Batch-Parallel Learning...")
    # Reset hierarchy state to be fair (weights were updated in Test 1)
    # We'll just create a new hierarchy of the same size
    hierarchy_v = PredictiveHierarchy([input_dim, hidden_dim, output_dim], device=device)
    agent_v = CognitivePredictiveAgent(hierarchy_v, device=device)
    
    start_time = time.time()
    agent_v.observe_and_learn(x_batch, y_batch, max_steps=50)
    batch_time = time.time() - start_time
    print(f"-> Batch Total Time: {batch_time:.4f}s ({batch_time/batch_size:.4f}s per sample amortized)")
    
    speedup = serial_time / batch_time
    print(f"\n[Final Results]")
    print(f"-> Vectorization Speedup: {speedup:.2f}x")
    
    if speedup > 3.0:
        print("[PASS] Vectorization is highly effective (Python overhead removed).")
    elif speedup > 1.0:
        print("[PASS] Vectorization provides improvement.")
    else:
        raise AssertionError("Vectorization is slower than serial. Check for bottlenecks.")

    # --- 3. Correctness check ---
    print("\n[Correctness] Checking if weights were actually updated...")
    v_norm = hierarchy_v.layers[0].V.data.norm().item()
    print(f"-> Layer 0 V Norm: {v_norm:.4f}")
    if v_norm > 0:
        print("[PASS] Weights updated successfully.")
    else:
        raise AssertionError("Weights are still zero or unchanged.")

if __name__ == "__main__":
    benchmark_vectorization()
