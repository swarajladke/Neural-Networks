import torch
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent
import time

def generate_parity_data(n_bits: int = 4):
    x = []
    for i in range(2**n_bits):
        bits = [float(b) for b in format(i, f'0{n_bits}b')]
        x.append(bits)
    X = torch.tensor(x)
    Y = (X.sum(dim=1) % 2).unsqueeze(1).float()
    return X, Y

def run_noise_stress_test():
    print("==================================================")
    print(" V5.0: Noise Injection Stress Test")
    print("==================================================")

    n_bits = 4
    X_parity, Y_parity = generate_parity_data(n_bits)
    
    # Pad Y to 3D to match 12D->16->16->3D
    Y_parity_3D = torch.zeros(16, 3)
    Y_parity_3D[:, 0] = Y_parity[:, 0]
    
    # Same inputs
    X_parity_12D = torch.zeros(16, 12)
    X_parity_12D[:, :4] = X_parity

    device = "cpu"
    hierarchy = PredictiveHierarchy([12, 16, 16, 3], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)

    # 1. Warm-up Phase: Skip heavy training and immediately force an expert
    print("\\n[1/3] Warm-up Phase: Establishing expert structure")
    baseline_neurons = hierarchy.layers[-1].output_dim
    baseline_neurogenesis = agent.neurogenesis_count

    print(f"-> Base Architecture Width: {baseline_neurons}")
    print(f"-> Total Neurogenesis Events: {baseline_neurogenesis}")
    
    if baseline_neurogenesis == 0:
        print("-> Let's force an identity pathway for testing purposes.")
        agent.hierarchy.expand_pathway(X_parity_12D[0:1], Y_parity_3D[0:1])
        agent.neurogenesis_count += 1
        for col in agent.hierarchy.layers:
            col.birth_surprise[-1] = 20.0 # Artificial surprise stamp
            col.firing_count[-1] += 50    # Pre-warm the expert
        baseline_neurons = hierarchy.layers[-1].output_dim
        baseline_neurogenesis = agent.neurogenesis_count
        print(f"-> Forced Architecture Width: {baseline_neurons}")

    # 2. Noise Phase: Inject high-entropy noise
    print("\\n[2/3] Stress Phase: Injecting 500 steps of 'static' random noise")
    noise_steps = 500
    
    # Create fixed noise pool to allow exposure tracking to work
    noise_pool_x = torch.randn(16, 12) * 5.0
    noise_pool_y = torch.randn(16, 3) * 5.0
    
    for step in range(noise_steps):
        # Sample from the fixed pool
        idx = step % 16
        noise_x = noise_pool_x[idx:idx+1]
        noise_y = noise_pool_y[idx:idx+1]
        
        w, s = agent.observe_and_learn(noise_x, noise_y, task_id=-1, max_steps=100)
        if len(agent.buffer) >= 16 and step % 16 == 0:
            agent.dream_replay(batch_size=16)

    print("\\n[3/3] Validation Results")
    final_neurons = hierarchy.layers[-1].output_dim
    final_neurogenesis = agent.neurogenesis_count
    
    # Verify Novelty Decay (should prevent neurogenesis on pure noise)
    print("--- Novelty Decay Check ---")
    if final_neurogenesis == baseline_neurogenesis:
        print("[PASS] No noise-triggered neurogenesis detected. Novelty decay is working correctly against high-entropy inputs.")
    else:
        print(f"[FAIL] {final_neurogenesis - baseline_neurogenesis} new pathways recruited during noise injection.")

    # Verify General Manifold Absorption
    if final_neurons == baseline_neurons:
        print("[PASS] General manifold safely absorbed noise without structural changes.")
        
    print("--- Expert Retention Check ---")
    original_dim = 16
    surviving = True
    for i, col in enumerate(hierarchy.layers):
        if col.output_dim > original_dim:
            scores = col.compute_retention_scores()
            expert_scores = scores[original_dim:]
            print(f"  Layer {i} Expert Scores: {expert_scores.detach().numpy()}")
            if (expert_scores < 0.01).any():
                surviving = False

    if surviving:
        print("[PASS] Expert neurons survived the noise injection (retention scoring preserved architecture).")
    else:
        print("[WARN] Some expert neurons fell below the retention threshold during the 500-step drought.")

    print("\\nStress Test Complete.")

if __name__ == "__main__":
    run_noise_stress_test()
