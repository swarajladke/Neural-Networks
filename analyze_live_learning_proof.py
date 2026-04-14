
import numpy as np
import os

def analyze():
    print("="*60)
    print("AGNIS V2: Live Continual Learning Analysis")
    print("="*60)
    
    metrics_file = "live_learning_proof_metrics.npz"
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found.")
        return

    data = np.load(metrics_file, allow_pickle=True)
    losses = data['losses']
    rolling_loss = data['rolling_avg_history']
    neuron_counts = data['neuron_counts']
    flashback_raw = data['flashback_stats']

    print(f"Total Steps: {len(losses)}")
    print(f"Final Neuron Count: {neuron_counts[-1]}")
    print(f"Peak Neuron Count: {np.max(neuron_counts)}")
    print(f"Final Rolling Loss: {rolling_loss[-1]:.4f}")
    
    print("\n[Flashback Retention Metrics]")
    # formatted as step -> task_losses
    # flashback_raw is object array of lists
    for i, row in enumerate(flashback_raw):
        step = (i + 1) * 100
        print(f"Step {step}: {['T'+str(j)+': '+str(round(val, 4)) for j, val in enumerate(row)]}")
    
    print("="*60)

if __name__ == "__main__":
    analyze()
