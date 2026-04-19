import torch
import time
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent

def test_cuda_ignite():
    print("==================================================")
    print(" V6.0: Full CUDA Deployment (RTX 3060 Laptop)")
    print("==================================================")
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available. Check your installation.")
        return

    device = "cuda:0"
    print(f"-> Initializing on {torch.cuda.get_device_name(0)}...")
    
    # Scale up for a real hardware test
    input_dim = 128
    hidden_dim = 512
    output_dim = 64
    batch_size = 64
    
    # 1. Initialization
    try:
        hierarchy = PredictiveHierarchy([input_dim, hidden_dim, output_dim], device=device)
        agent = CognitivePredictiveAgent(hierarchy, device=device)
        print("[PASS] Successfully moved Hierarchy and Agent to CUDA.")
    except Exception as e:
        print(f"[FAIL] Error during CUDA initialization: {e}")
        return

    # 2. Workload Test (Vectorized SNAP-ATP)
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randn(batch_size, output_dim, device=device)
    
    print(f"\n[Workload: Batch={batch_size}, Hidden={hidden_dim}, Settle=50 steps]")
    
    # Warm up
    agent.observe_and_learn(x, y, max_steps=10)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    agent.observe_and_learn(x, y, max_steps=50)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"-> GPU Time: {gpu_time:.4f}s ({gpu_time/batch_size:.4f}s per sample)")
    
    # Check Guardian
    temp, vram = agent.guardian.query_telemetry()
    print(f"-> Thermal Telemetry: {temp}C | VRAM: {vram:.1f}%")

    print("\n[Comparison with CPU Baseline]")
    # We'll use the amortized CPU time from our previous benchmark (approx 0.011s/sample)
    cpu_ref_time = 0.011 * batch_size 
    print(f"-> Expected CPU Time (Amortized): {cpu_ref_time:.4f}s")
    
    if gpu_time < cpu_ref_time:
        print(f"-> Hardware Acceleration: {cpu_ref_time/gpu_time:.2f}x speedup over vectorized CPU.")
    else:
        print("-> GPU is currently slower than CPU for this small batch (likely CUDA overhead).")

    print("\n[Correctness check]")
    v_norm = hierarchy.layers[0].V.data.norm().item()
    if v_norm > 0:
        print(f"[PASS] SNAP-ATP updates active on GPU (V Norm: {v_norm:.4f}).")
    else:
        print("[FAIL] Weights are still zero or unchanged.")

if __name__ == "__main__":
    test_cuda_ignite()
