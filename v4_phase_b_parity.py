"""
Phase B: 4-Bit Parity Validation — v4.9 Online Learning
=========================================================
Uses per-sample settle+update (Option C) to prevent batch-averaged
outer product cancellation that killed v4.8's ability to learn XOR.
"""

import torch
from agnis_v4_core import PredictiveHierarchy


def generate_parity_data(n_bits: int = 4):
    """Generate all 2^n combinations and their parity."""
    x = []
    for i in range(2**n_bits):
        bits = [float(b) for b in format(i, f'0{n_bits}b')]
        x.append(bits)
    X = torch.tensor(x)
    Y = (X.sum(dim=1) % 2).unsqueeze(1).float()
    return X, Y


def run_phase_b_online():
    print("==================================================")
    print(" Phase B: Parity Validation (v4.9 Online Learning)")
    print(" [Per-Sample Settle + 150 Steps + 5.0x Beta Push]")
    print("==================================================")

    n_bits = 4
    X, Y = generate_parity_data(n_bits)
    device = "cpu"

    # Architecture: [4, 16, 16, 16] - top layer is 16D, label is 1D readout
    hierarchy = PredictiveHierarchy([n_bits, 16, 16, 16], device=device)

    # Hyperparameters tuned for online learning
    for col in hierarchy.layers:
        col.eta_x = 0.9
        col.tau   = 0.5
        col.eta_W = 0.01   # Scaled down: 16 updates/epoch vs 1 in batch mode
        col.eta_V = 0.03   # Scaled down from 0.05 for same reason

    epochs = 5000
    curriculum_epochs = 1500
    settle_steps = 150      # Single samples settle much faster than batches

    print(f"Training on {2**n_bits}-bit patterns (ONLINE per-sample)...")

    best_acc = 0.0
    for epoch in range(epochs + 1):
        # Recognition Curriculum: 0.0 -> 1.0
        rec_weight = min(1.0, epoch / curriculum_epochs)

        # Shuffle sample order each epoch (prevents order bias)
        indices = torch.randperm(16)
        batch_x = X[indices]
        batch_y = Y[indices]

        # --- PSEUDO-ONLINE LEARNING ---
        # State settles across the batch in parallel, but updates are unrolled 
        # sample-by-sample internally in update_weights()
        hierarchy.infer_and_learn(
            batch_x, top_level_label=batch_y,
            max_steps=settle_steps,
            recognition_weight=rec_weight,
            beta_push=5.0
        )

        if epoch % 250 == 0:
            with torch.no_grad():
                correct = 0
                total_mse = 0
                for i in range(16):
                    test_x = X[i].unsqueeze(0)
                    true_y = Y[i].item()

                    hierarchy.reset_states(batch_size=1)
                    pred_y_tensor = hierarchy.predict_label(test_x, max_steps=200)
                    pred_y = pred_y_tensor[0, 0].item()

                    pred_class = 1.0 if pred_y > 0.5 else 0.0
                    if pred_class == true_y:
                        correct += 1
                    total_mse += (pred_y - true_y) ** 2

                acc = (correct / 16) * 100
                avg_mse = total_mse / 16
                best_acc = max(best_acc, acc)

                norms = hierarchy.weight_norms()

                print(f"Epoch {epoch:<4d} | RecW: {rec_weight:.2f} | "
                      f"Acc: {acc:>5.1f}% (best: {best_acc:.1f}%) | "
                      f"MSE: {avg_mse:.4f}")
                print(f"         | V1: {norms['L1_V']:.2f} V2: {norms['L2_V']:.2f} | "
                      f"W-top: {norms['top_W']:.2f}", flush=True)

                if acc == 100.0 and epoch > curriculum_epochs:
                    print(f"\n[SUCCESS] 100% Accuracy at epoch {epoch}!")
                    break

    print(f"\n[Phase B v4.9 Online Final Results]")
    print(f"Final Accuracy: {acc:.1f}%")
    print(f"Best Accuracy:  {best_acc:.1f}%")

    if best_acc > 95.0:
        print("-> SUCCESS: Predictive coding solved parity with online Hebbian learning.")
    elif best_acc > 60.0:
        print("-> PARTIAL: Learning signal detected but not converged.")
    else:
        print("-> FAILURE: No significant learning occurred.")

if __name__ == "__main__":
    run_phase_b_online()
