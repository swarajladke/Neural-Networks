import random
import torch
from agnis_v4_core import PredictiveHierarchy

def test_shallow_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f" AGNIS SHALLOW PARITY TEST ({device.upper()})")
    print(f"==================================================")

    # Architecture: 4 -> 32 -> 1
    hierarchy = PredictiveHierarchy([4, 32, 1], device=device)
    
    # Generate all 16 patterns
    samples = []
    for i in range(16):
        bits = torch.tensor([[float(b) for b in format(i, "04b")]], dtype=torch.float32, device=device)
        target = torch.tensor([[float(sum([float(b) for b in format(i, "04b")]) % 2)]], dtype=torch.float32, device=device)
        samples.append((bits, target))
    
    # Standard stable learning
    for col in hierarchy.layers:
        col.eta_V = 0.05
        col.eta_W = 0.05

    epochs = 500
    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        for x, y in samples:
            hierarchy.reset_states(batch_size=1)
            # Supervised Training (Online)
            hierarchy.infer_and_learn(
                x, 
                top_level_label=y, 
                max_steps=100, 
                recognition_weight=1.0, 
                beta_push=5.0
            )

        if epoch % 50 == 0:
            with torch.no_grad():
                correct = 0
                for x, y in samples:
                    hierarchy.reset_states(batch_size=1)
                    preds = hierarchy.predict_label(x, max_steps=100)
                    pred_bits = (torch.sigmoid(preds[:, :1]) > 0.5).float()
                    if pred_bits == y:
                        correct += 1
                acc = correct / 16.0
                print(f" Epoch {epoch:3d} | Online Acc: {acc:.3f}")
                if acc == 1.0:
                    print(f"\n>>> SHALLOW PARITY SOLVED AT EPOCH {epoch} <<<")
                    break

if __name__ == "__main__":
    test_shallow_parity()
