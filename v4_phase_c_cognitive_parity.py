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
    Y = (X.sum(dim=1) % 2).unsqueeze(1).float()
    return X, Y

def run_phase_c():
    n_bits = 4
    X, Y = generate_parity_data(n_bits)
    device = "cpu"

    hierarchy = PredictiveHierarchy([n_bits, 16, 16, 16], device=device)
    for col in hierarchy.layers:
        col.eta_V = 0.05
        col.eta_W = 0.05
        col.weight_clamp = 3.0

    agent = CognitivePredictiveAgent(hierarchy, device=device)

    epochs = 1500
    settle_steps = 150

    print("==================================================")
    print(" Phase C: Cognitive Parity Validation (v4 Power Move)")
    print(" [Salience-scaled online learning + Dream Replay]")
    print("==================================================")

    for epoch in range(epochs + 1):
        indices = list(range(X.shape[0]))
        random.shuffle(indices)

        epoch_salience = []
        for idx in indices:
            test_x = X[idx].unsqueeze(0)
            true_y = Y[idx].unsqueeze(0)
            
            w, s = agent.observe_and_learn(
                test_x, true_y, task_id=0, max_steps=settle_steps,
                recognition_weight=1.0, beta_push=5.0
            )
            epoch_salience.append(w)
            
        # Replay dream to consolidate buffered sparse features
        agent.dream_replay(batch_size=16, max_steps=settle_steps, recognition_weight=1.0, beta_push=5.0)

        if epoch % 5 == 0:
            correct = 0
            total_mse = 0.0
            with torch.no_grad():
                for i in range(16):
                    test_x = X[i].unsqueeze(0)
                    true_y = Y[i].item()
                    agent.hierarchy.reset_states(batch_size=1)
                    pred_tensor = agent.hierarchy.predict_label(test_x, max_steps=150)
                    pred_y = pred_tensor[0, 0].item()

                    pred_class = 1.0 if pred_y > 0.5 else 0.0
                    if pred_class == true_y:
                        correct += 1
                    total_mse += (pred_y - true_y) ** 2

            acc = correct / 16.0 * 100
            total_mse /= 16.0
            
            avg_w = sum(epoch_salience)/len(epoch_salience)
            v1_norm = agent.hierarchy.layers[0].V.norm().item()
            v2_norm = agent.hierarchy.layers[1].V.norm().item()
            wtop_norm = agent.hierarchy.layers[-1].W.norm().item()
            
            print(f"Epoch {epoch:<4} | Acc: {acc:>5.1}% | MSE: {total_mse:.4f} | Salience: {avg_w:.2f}")
            print(f"         | V1: {v1_norm:.2f} V2: {v2_norm:.2f} | W-top: {wtop_norm:.2f}")
            print(f"         | Buffer size: {len(agent.buffer)}")

if __name__ == "__main__":
    run_phase_c()
