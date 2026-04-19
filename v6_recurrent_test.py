import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent
from experiment_utils import metric_to_float

def test_temporal_recurrence():
    print("==================================================")
    print(" V5.3: Native Recurrence Validation Test")
    print("==================================================")
    
    device = "cpu"
    torch.manual_seed(7)
    # Small hierarchy: 4 input -> 16 hidden -> 4 output
    # Inputs: One-hot vectors for 'A' (1,0,0,0), 'B' (0,1,0,0), 'C' (0,0,1,0)
    hierarchy = PredictiveHierarchy([4, 16, 4], device=device)
    agent = CognitivePredictiveAgent(hierarchy, device=device)
    
    # Vocabulary
    A = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    B = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    C = torch.tensor([[0.0, 0.1, 1.0, 0.0]]) # C
    
    # sequence: A -> B, B -> C, C -> A
    sequence = [(A, B), (B, C), (C, A)]
    
    print("\n[Phase 1] Training on token-by-token sequence...")
    epochs = 100
    first_epoch_surprise = 0
    last_epoch_surprise = 0
    
    for epoch in range(epochs):
        total_surprise = 0
        for x, y in sequence:
            _, surprise = agent.observe_and_learn(x, y, max_steps=100, warm_start=True)
            total_surprise += metric_to_float(surprise)
        
        avg_surprise = total_surprise / len(sequence)
        if epoch == 0: first_epoch_surprise = avg_surprise
        last_epoch_surprise = avg_surprise
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{100} - Avg Surprise: {avg_surprise:.6f}")

    print("\n[Phase 2] Testing Zero-Window Prediction (Temporal Recall)...")
    # Feed 'A', expect high surprise (cold start), but it should prepare the state for 'B'
    with torch.no_grad():
        print("Feeding 'A' (Prime sequence)...")
        agent.hierarchy.predict_label(A, max_steps=150, update_temporal=True)
        
        print("Feeding 'B' (Expect internal 'C')...")
        pred_C = agent.hierarchy.predict_label(B, max_steps=1, update_temporal=True)
        
        recall_surprise = torch.nn.functional.mse_loss(pred_C, C).item()
        print(f"-> Prediction Surprise for 'C' (1-step settling): {recall_surprise:.6f}")
        
        agent.hierarchy.reset_states()
        cold_C = agent.hierarchy.predict_label(B, max_steps=1, update_temporal=False)
        cold_surprise = torch.nn.functional.mse_loss(cold_C, C).item()
        print(f"-> Cold Surprise (No recurrence): {cold_surprise:.6f}")

    print("\n[Final Results]")
    print(f"-> Surprise Trend: {first_epoch_surprise:.4f} -> {last_epoch_surprise:.4f}")
    
    if last_epoch_surprise < first_epoch_surprise * 0.5:
        print("[PASS] Stability Fix Verified: Surprise trended down over training.")
    else:
        raise AssertionError("Stability Fix Failed: Surprise did not decrease significantly.")

    if recall_surprise < cold_surprise:
        improvement = (cold_surprise - recall_surprise) / cold_surprise * 100
        print(f"[PASS] Recurrence demonstrated: Hidden state improved prediction by {improvement:.1f}%")
    else:
        raise AssertionError("Recurrence effect still weak or obstructive.")

if __name__ == "__main__":
    test_temporal_recurrence()
