import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_v4_benchmark_chart():
    try:
        with open('v4_9_benchmark_plot_data_100.json', 'r') as f:
            plot_data = json.load(f)
    except FileNotFoundError:
        print("Error: v4_9_benchmark_plot_data_100.json not found.")
        return

    # Clean the data (ensure it's valid floats)
    plot_data = [float(x) for x in plot_data]
    mean_val = np.mean(plot_data)
    
    # Create the high-fidelity plot in the requested style
    plt.figure(figsize=(16, 8))
    
    # Matching the 'phase_150' style
    plt.plot(range(len(plot_data)), plot_data, marker='o', markersize=4, linestyle='-', 
             color='#27ae60', alpha=0.8, linewidth=1.5, label=f'MSE per Domain')
    
    # Horizontal mean line
    plt.axhline(y=mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Avg: {mean_val:.4f}')
    
    # Formatting
    plt.title('Phase 100: Final Retention across 100 Domains (AGNIS V4.9)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Domain Index', fontsize=12)
    plt.ylabel('Loss (Lower is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    # Save with high DPI for premium look
    save_path = 'phase_200_simultaneous_retention_v4.png'
    plt.savefig(save_path, dpi=200)
    print(f"Proper diagram saved to {save_path}")

if __name__ == "__main__":
    generate_v4_benchmark_chart()
