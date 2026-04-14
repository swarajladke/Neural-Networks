import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent
import random

def generate_parity_data(n_bits: int = 4):
    x = []
    for i in range(2**n_bits):
        bits = [float(b) for b in format(i, f'0{n_bits}b')]
        x.append(bits)
    X = torch.tensor(x)
    # Bipolar labels: -1 and 1
    Y = (2.0 * (X.sum(dim=1) % 2) - 1.0).unsqueeze(1).float()
    return X, Y

def run_phase_d():
    n_bits = 4
    X, Y = generate_parity_data(n_bits)
    device = "cpu"

    # Start with Tabula Rasa (Bottleneck 1) to prove autonomous growth
    hierarchy = PredictiveHierarchy([n_bits, 1, 1, 1], device=device)
    for col in hierarchy.layers:
        col.eta_V = 0.05
        col.eta_W = 0.05
        col.weight_clamp = 3.0

    agent = CognitivePredictiveAgent(hierarchy, device=device)

    epochs = 40
    settle_steps = 150

    print("==================================================")
    print(" Phase D: Autonomous Neurogenesis (The Holy Grail)")
    print(" [Bipolar Identity Slivers - One-Shot Logic]    ")
    print("==================================================")

    for epoch in range(epochs + 1):
        indices = list(range(X.shape[0]))
        random.shuffle(indices)

        for idx in indices:
            test_x = X[idx].unsqueeze(0)
            true_y = Y[idx].unsqueeze(0)
            
            agent.observe_and_learn(
                test_x, true_y, task_id=0, max_steps=settle_steps,
                recognition_weight=1.0, beta_push=5.0
            )
            
        agent.dream_replay(batch_size=16, max_steps=settle_steps, recognition_weight=1.0, beta_push=5.0)

        correct = 0
        total_mse = 0.0
        with torch.no_grad():
            for i in range(16):
                test_x = X[i].unsqueeze(0)
                true_y = Y[i].item() # -1 or 1
                agent.hierarchy.reset_states(batch_size=1)
                pred_tensor = agent.hierarchy.predict_label(test_x, max_steps=150)
                pred_y = pred_tensor[0, 0].item()

                # Bipolar decision: > 0 is 1.0, else -1.0
                pred_class = 1.0 if pred_y > 0.0 else -1.0
                if pred_class == true_y:
                    correct += 1
                total_mse += (pred_y - true_y) ** 2

        acc = correct / 16.0 * 100
        total_mse /= 16.0
        
        v1_shape = agent.hierarchy.layers[0].V.shape
        print(f"Epoch {epoch:<4} | Acc: {acc:>4.1f}% | MSE: {total_mse:.4f} | L0 Neurons: {v1_shape[1]}")
        
        if acc >= 100.0 and total_mse < 0.1:
            print(f">>> 100% ACCURACY REACHED! Parity Solved perfectly at epoch {epoch}! <<<")
            break

if __name__ == "__main__":
    run_phase_d()
