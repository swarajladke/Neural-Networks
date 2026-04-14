"""
PHASE 21: The Continual Learning Pentathlon (5-Task Stress Test)

This script tests the AGI system's ability to learn 5 different domains
sequentially without "forgetting" any of them.

Domains:
1. Shakespeare (Poetic/formal)
2. SQL Scripting (Structural/technical)
3. Python Coding (Algorithmic/procedural)
4. Medical Science (Technical terminology)
5. Logic & Mathematics (Formal reasoning)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from agi_slm import AGISmallLanguageModel

def get_pentathlon_data():
    # 1. Shakespeare
    task1 = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date:
    Sometime too hot the eye of heaven shines,
    And often is his gold complexion dimm'd;
    And every fair from fair sometime declines,
    By chance or nature's changing course untrimm'd;
    """ * 10
    
    # 2. SQL
    task2 = """
    SELECT u.name, p.title 
    FROM users u 
    JOIN posts p ON u.id = p.user_id 
    WHERE u.active = 1 
    ORDER BY p.created_at DESC;
    INSERT INTO audit_log (action, user_id) VALUES ('LOGIN', 45);
    """ * 15
    
    # 3. Python
    task3 = """
    def quicksort(arr):
        if len(arr) <= 1: return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    print(quicksort([3,6,8,10,1,2,1]))
    """ * 15
    
    # 4. Medical (Biology)
    task4 = """
    The cardiovascular system consists of the heart, veins, and arteries.
    Oxygenated blood is pumped from the left ventricle into the aorta.
    The alveoli in the lungs facilitate gas exchange via diffusion.
    Neurotransmitters like dopamine and serotonin regulate synaptic firing.
    Mitochondria are the primary site of ATP synthesis within eukaryotes.
    """ * 20
    
    # 5. Logic & Math
    task5 = """
    If All P are Q, and All Q are R, then All P are R.
    Let x = 5. Find y if y = 2x + 10. (y = 20)
    For every integer n, if n is even, then n^2 is even.
    The sequence 1, 1, 2, 3, 5, 8 follows the Fibonacci rule: f(n) = f(n-1) + f(n-2).
    A implies B; A is true; Therefore, B is true (Modus Ponens).
    """ * 20
    
    return [task1.strip(), task2.strip(), task3.strip(), task4.strip(), task5.strip()]

def run_pentathlon():
    print("🔥 THE CONTINUAL LEARNING PENTATHLON (5-Task Challenge)")
    print("Goal: Maintain Stability and Plasticity across 5 Domains.")
    
    tasks = get_pentathlon_data()
    all_text = "".join(tasks)
    chars = sorted(list(set(all_text + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:;!?'\"()[]{}<>=-_\\/@#%&*+\n")))
    
    model = AGISmallLanguageModel(chars, d_model=256, replay_capacity=1000)
    
    LR = 0.001
    TOKENS_PER_TASK = 300
    
    # Trackers
    retention_matrix = np.zeros((5, 5)) # Row: Current Task, Col: Task being measured
    loss_history = {i: [] for i in range(5)}
    
    criterion = nn.CrossEntropyLoss()
    
    def measure_task_loss(task_idx):
        text = tasks[task_idx][:150] # Test on first 150 chars
        with torch.no_grad():
            model.eval()
            tokens = model.encode(text)
            x = tokens[:-1]
            y = tokens[1:]
            logits = model(x)
            loss = criterion(logits.view(-1, len(chars)), y.view(-1))
            return loss.item()

    # Establish Baselines
    print("\n--- Establishing Baselines ---")
    baselines = [measure_task_loss(i) for i in range(5)]
    for i, b in enumerate(baselines):
        print(f"Task {i+1} Baseline: {b:.4f}")

    # Training Loop
    for t_idx, task_text in enumerate(tasks):
        print(f"\n🚀 LEARNING TASK {t_idx+1}...")
        tokens = model.encode(task_text)
        
        for i in range(min(len(tokens)-1, TOKENS_PER_TASK)):
            inp_char = chars[tokens[i].item()]
            tgt_char = chars[tokens[i+1].item()]
            
            # Use Task ID for MoE routing
            loss = model.runtime_learn(inp_char, tgt_char, learning_rate=LR, task_id=t_idx)
            
            if i % 100 == 0:
                print(f"  Step {i:3d}: Loss {loss:.4f}")
        
        # Audit all tasks learned so far (and future baselines)
        print(f"📊 Audit after Task {t_idx+1}:")
        for audit_idx in range(5):
            current_loss = measure_task_loss(audit_idx)
            retention_matrix[t_idx, audit_idx] = current_loss
            status = "Learned" if audit_idx <= t_idx else "Baseline"
            print(f"  - Task {audit_idx+1} ({status}): {current_loss:.4f}")

    # Final Summary
    print("\n" + "="*60)
    print("🏆 PENTATHLON COMPLETE")
    print("="*60)
    
    # Calculate Final Retention for all tasks relative to their "best" learned state
    print("\n📈 Knowledge Stability Audit:")
    for i in range(5):
        best_loss = retention_matrix[i, i] # Loss immediately after learning
        final_loss = retention_matrix[4, i] # Loss after Task 5
        base_loss = baselines[i]
        
        # Retention Score: (Baseline - Final) / (Baseline - Best)
        denom = (base_loss - best_loss)
        if abs(denom) < 1e-6: denom = 1e-6
        score = (base_loss - final_loss) / denom
        
        print(f"Task {i+1} Retention: {score:.2f} (Best: {best_loss:.2f}, Final: {final_loss:.2f})")

    # Visualize Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(retention_matrix, cmap='viridis_r')
    plt.colorbar(label='Loss (Lower is Better)')
    plt.xticks(range(5), [f"T{i+1}" for i in range(5)])
    plt.yticks(range(5), [f"After T{i+1}" for i in range(5)])
    plt.title("Continual Learning Pentathlon: Knowledge Retention Matrix")
    plt.xlabel("Measured Task")
    plt.ylabel("Training Progress")
    
    # Annotate values
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{retention_matrix[i,j]:.2f}", 
                     ha="center", va="center", color="white" if retention_matrix[i,j] > 4 else "black")

    plt.savefig("pentathlon_results.png")
    print("\n📊 Saved 'pentathlon_results.png'")

if __name__ == "__main__":
    run_pentathlon()
