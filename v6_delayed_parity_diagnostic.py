import torch
import time
from agnis_v4_core import PredictiveHierarchy

def run_diagnostic():
    device = "cuda"
    print(f"--- NUCLEAR GPU SPRINT (RTX 3060) ---")
    
    # 1. Prepare Batch (Entire truth table)
    patterns = []
    targets = []
    for i in range(16):
        bits = [float(b) for b in format(i, "04b")]
        patterns.append(bits)
        targets.append([float(sum(bits) % 2)])
    
    x_bits = torch.tensor(patterns, dtype=torch.float32, device=device)
    target_batch = torch.tensor(targets, dtype=torch.float32, device=device)
    blank = torch.zeros_like(x_bits)

    # 2. Build Hierarchy
    hierarchy = PredictiveHierarchy([4, 64, 1], device=device) # Wider for XOR basis
    for col in hierarchy.layers: 
        col.eta_V = 0.5  # Fast recognition
        col.eta_R = 0.01 # Slow stable recurrence

    start_time = time.time()
    for epoch in range(1, 251): # Longer training for specialization
        hierarchy.reset_states(batch_size=16)
        
        # COHERENT SETTLEMENT (Depth over Breadth)
        hierarchy.infer_and_learn(x_bits, max_steps=100) # Deep encoding
        hierarchy.infer_and_learn(blank, max_steps=20, recognition_weight=0.0)
        hierarchy.infer_and_learn(blank, top_level_label=target_batch, max_steps=150, recognition_weight=0.0, beta_push=10.0, warm_start=True)

        if epoch % 10 == 0:
            with torch.no_grad():
                hierarchy.reset_states(batch_size=16)
                hierarchy.predict_label(x_bits, max_steps=100, update_temporal=True)
                hierarchy.infer_and_learn(blank, max_steps=20, recognition_weight=0.0, beta_push=0.0)
                preds = hierarchy.predict_label(blank, max_steps=150, update_temporal=False, recognition_weight=0.0)
                acc = ((torch.sigmoid(preds[:, :1]) > 0.5).float() == target_batch).sum().item() / 16.0
                print(f" Epoch {epoch:3d} | Acc: {acc:.3f}")

    print(f"DONE. Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_diagnostic()
