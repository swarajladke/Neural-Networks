import torch
from agnis_v4_core import PredictiveHierarchy

def test_direct_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f" AGNIS DIRECT PARITY TEST ({device.upper()})")
    print(f"==================================================")

    # Generate all 16 patterns
    patterns = []
    targets = []
    for i in range(16):
        bits = [float(b) for b in format(i, "04b")]
        patterns.append(bits)
        targets.append([float(sum(bits) % 2)])
    
    x_batch = torch.tensor(patterns, dtype=torch.float32, device=device)
    y_batch = torch.tensor(targets, dtype=torch.float32, device=device)

    # Architecture: 4 -> 16 -> 8 -> 1
    hierarchy = PredictiveHierarchy([4, 16, 8, 1], device=device)
    
    # Fast learning for direct mapping
    for col in hierarchy.layers:
        col.eta_V = 0.2
        col.eta_W = 0.1

    epochs = 100
    for epoch in range(1, epochs + 1):
        hierarchy.reset_states(batch_size=16)
        
        # Supervised Training: Input bits AND Label simultaneously
        hierarchy.infer_and_learn(
            x_batch, 
            top_level_label=y_batch, 
            max_steps=100, 
            recognition_weight=1.0, 
            beta_push=5.0
        )

        if epoch % 10 == 0:
            with torch.no_grad():
                hierarchy.reset_states(batch_size=16)
                preds = hierarchy.predict_label(x_batch, max_steps=100)
                pred_bits = (torch.sigmoid(preds[:, :1]) > 0.5).float()
                acc = (pred_bits == y_batch).sum().item() / 16.0
                print(f" Epoch {epoch:3d} | Direct Acc: {acc:.3f}")
                if acc == 1.0:
                    print(f"\n>>> DIRECT PARITY SOLVED AT EPOCH {epoch} <<<")
                    break

if __name__ == "__main__":
    test_direct_parity()
