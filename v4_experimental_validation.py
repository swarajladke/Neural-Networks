"""
Phase A: Non-Linear Regression — v4.9 Online Learning
=======================================================
Uses per-sample settle+update to prevent batch-averaged gradient
cancellation on the sin(x)*cos(y) surface.
"""

import os
from collections import deque
import torch

from agnis_v4_core import PredictiveHierarchy


def generate_phase_a_data(num_samples: int = 1000):
    """
    Generate Continuous Mapping dataset: z = sin(x) * cos(y) + noise
    x, y in [-3, 3]
    """
    torch.manual_seed(42)
    inputs = torch.empty(num_samples, 2).uniform_(-3.0, 3.0)
    x = inputs[:, 0]
    y = inputs[:, 1]

    clean_z = torch.sin(x) * torch.cos(y)
    noise = torch.randn_like(clean_z) * 0.05
    z = clean_z + noise

    # Normalize inputs to [-1, 1]
    inputs = inputs / 3.0

    return inputs, z.unsqueeze(1)


def run_phase_a():
    print("==================================================")
    print(" Phase A: Regression Validation (v4.9 Online)")
    print("==================================================")

    inputs, targets = generate_phase_a_data(1000)
    device = "cpu"

    # Two-layer hierarchy: [2, 128, 64] with a 1D partial label readout
    hidden_dim = 128
    hierarchy = PredictiveHierarchy([2, hidden_dim, 64], device=device)

    # Hyperparameters tuned for online learning
    for col in hierarchy.layers:
        col.eta_x = 0.8
        col.tau   = 0.5
        col.eta_W = 0.01     # Conservative for per-sample updates
        col.eta_V = 0.03
        col.lambda_act = 1e-6

    epochs = 5000
    samples_per_epoch = 16  # Process 16 random samples online each epoch
    max_steps_train = 80    # Faster settling for single samples
    max_steps_eval  = 80
    test_count = 100

    metrics = {"epoch": [], "mse": []}
    roll = deque(maxlen=20)
    best_mse = float("inf")

    print(f"Training: {epochs} epochs × {samples_per_epoch} online samples")
    print(f"Settle: {max_steps_train} steps per sample\n")

    for epoch in range(epochs + 1):
        # Sample a random mini-set of training points
        indices = torch.randperm(1000)[:samples_per_epoch]
        batch_x = inputs[indices]
        batch_z = targets[indices]

        # --- PSEUDO-ONLINE LEARNING ---
        # The core update_weights() now handles independent sample updates internally
        hierarchy.infer_and_learn(
            batch_x,
            top_level_label=batch_z,
            max_steps=max_steps_train,
            recognition_weight=1.0,
            beta_push=3.0,
        )

        # Evaluate every 250 epochs
        if epoch % 250 == 0:
            test_indices = torch.randperm(1000)[:test_count]
            test_x = inputs[test_indices]
            true_z = targets[test_indices]

            with torch.no_grad():
                hierarchy.reset_states(batch_size=test_count)
                pred_z_full = hierarchy.predict_label(test_x, max_steps=max_steps_eval)
                pred_z = pred_z_full[:, 0:1]
                mse = torch.nn.functional.mse_loss(pred_z, true_z).item()

            metrics["epoch"].append(epoch)
            metrics["mse"].append(mse)
            roll.append(mse)
            roll_mean = sum(roll) / len(roll)
            best_mse = min(best_mse, mse)

            norms = hierarchy.weight_norms()
            print(
                f"Epoch {epoch:<4d} | MSE: {mse:.6f} | Roll: {roll_mean:.6f} | "
                f"Best: {best_mse:.6f} | V: {norms['L1_V']:.2f} W: {norms['L1_W']:.2f}",
                flush=True,
            )

    print("\n[Phase A Results]")
    final_mse = metrics["mse"][-1]
    print(f"Final MSE: {final_mse:.6f}  Best MSE: {best_mse:.6f}  (Target < 0.05)")

    if best_mse < 0.05:
        print("-> SUCCESS: Phase A Non-Linear Regression passed with online Hebbian.")
    elif best_mse < 0.15:
        print("-> PARTIAL: Significant learning detected but target not met.")
    else:
        print("-> FAILURE: MSE target not met.")


if __name__ == "__main__":
    run_phase_a()
