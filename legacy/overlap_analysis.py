import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from agi_slm import AGISmallLanguageModel, load_agi_data

def analyze_overlap():
    print("🔬 EXPERT OVERLAP ANALYSIS")
    
    # Same tasks as before
    task_a = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune
    """ * 10
    
    task_b = """
    SELECT * FROM users WHERE id = 123;
    INSERT INTO logs (event_id, timestamp) VALUES (999, '2023-01-01');
    DELETE FROM sessions WHERE session_token IS NULL;
    """ * 10

    chars = list(set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:;!?'\"()[]{}<>=-_\\/@#%&*+\n"))
    model = AGISmallLanguageModel(chars)
    model.eval()
    
    # We need to hook into the MoE engine to track usage
    # The MoE is deep inside: model.brain.internal_state_engine.moe_layer
    # Let's verify the path first via inspection or just try to access usage stats
    
    print("Resetting expert usage stats...")
    moe = model.brain.internal_state_engine
    moe.expert_usage = torch.zeros(moe.num_experts)
    
    # 1. Run Task A (Task ID 0)
    print("Running Task A (Context 0)...")
    model.brain.current_state.task_id = 0
    tokens_a = model.encode(task_a)
    with torch.no_grad():
        model(tokens_a)
        
    usage_a = moe.expert_usage.clone().numpy()
    # Normalize
    if usage_a.sum() > 0: usage_a = usage_a / usage_a.sum()
    
    # Reset
    moe.expert_usage = torch.zeros(moe.num_experts)
    
    # 2. Run Task B (Task ID 1)
    print("Running Task B (Context 1)...")
    model.brain.current_state.task_id = 1
    tokens_b = model.encode(task_b)
    with torch.no_grad():
        model(tokens_b)
        
    usage_b = moe.expert_usage.clone().numpy()
    if usage_b.sum() > 0: usage_b = usage_b / usage_b.sum()
    
    # 3. Calculate Overlap
    # Cosine similarity or simple intersection
    overlap = np.minimum(usage_a, usage_b).sum()
    
    print("-" * 40)
    print(f"Task A Top Experts: {np.argsort(usage_a)[-3:]}")
    print(f"Task B Top Experts: {np.argsort(usage_b)[-3:]}")
    print(f"Expert Distribution Overlap: {overlap:.4f} (0.0 = Isolated, 1.0 = Identical)")
    
    if overlap > 0.3:
        print("❌ WARNING: Significant Expert Overlap detected.")
        print("   This explains the Partial Forgetting.")
    else:
        print("✅ Low Overlap. Forgetting might be due to shared weights elsewhere.")

    # Plot
    plt.figure(figsize=(10, 6))
    x = range(len(usage_a))
    plt.bar(x, usage_a, alpha=0.5, label='Shakespeare Experts', color='blue')
    plt.bar(x, usage_b, alpha=0.5, label='SQL Experts', color='red')
    plt.xlabel("Expert ID")
    plt.ylabel("Normalized Activation Frequency")
    plt.title(f"Expert Specialization Conflict (Overlap: {overlap:.2f})")
    plt.legend()
    plt.savefig("expert_overlap.png")
    print("📉 Saved expert_overlap.png")

if __name__ == "__main__":
    analyze_overlap()
