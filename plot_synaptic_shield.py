import matplotlib.pyplot as plt
import re
import numpy as np

def generate_synaptic_shield_chart(log_path, output_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse baseline
    baseline_match = re.search(r"Baseline Italian Surprise:\s*([\d.]+)", content)
    baseline_it = float(baseline_match.group(1)) if baseline_match else 1.0

    # Parse batches
    batches = []
    ru_surprises = []
    it_retention = []

    # Pattern for Batch blocks
    # ## Batch 500
    # - **RU Surprise:** 1.0342 | **IT Retention:** 0.9827
    pattern = r"## Batch (\d+)\n- \*\*RU Surprise:\*\* ([\d.]+) \| \*\*IT Retention:\*\* ([\d.]+)"
    matches = re.findall(pattern, content)

    for b, ru, it in matches:
        batches.append(int(b))
        ru_surprises.append(float(ru))
        it_retention.append(float(it))

    if not batches:
        print("No data found in log.")
        return

    # Create Plot
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Russian Surprise Curve
    color_ru = '#00f2ff' # Cyan
    ax1.set_xlabel('Training Batch (Russian Discovery Phase)', fontsize=12, color='white')
    ax1.set_ylabel('Russian Discovery Surprise', color=color_ru, fontsize=12)
    ax1.plot(batches, ru_surprises, color=color_ru, linewidth=3, marker='o', label='Russian Discovery')
    ax1.tick_params(axis='y', labelcolor=color_ru)
    ax1.grid(True, alpha=0.1)

    # Italian Retention Curve (Second Y Axis)
    ax2 = ax1.twinx()
    color_it = '#ff0055' # Neon Pink
    ax2.set_ylabel('Italian Retention Probe (Relative to Baseline)', color=color_it, fontsize=12)
    
    # Normalize IT Retention to percentage of original baseline accuracy
    # Lower surprise = better, so 1.0 / (it_surprise / baseline)
    retention_pct = [ (baseline_it / val) * 100 for val in it_retention]
    
    ax2.plot(batches, retention_pct, color=color_it, linewidth=4, marker='s', label='Italian Retention (Shielded)')
    ax2.axhline(y=100, color='white', linestyle='--', alpha=0.5, label='Original Mastery (Baseline)')
    
    ax2.set_ylim(80, 120) # Focus on the stability
    ax2.tick_params(axis='y', labelcolor=color_it)

    # Title & Aesthetics
    plt.title('AGNIS Phase 7.3: The Synaptic Shield Protocol\nZero-Forgetting Proof (Bilingual Manifold Isolation)', fontsize=16, pad=20, color='white', fontweight='bold')
    
    # Unified Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, facecolor='#1a1a1a')

    # Add Text Annotations
    plt.text(batches[-1], retention_pct[-1]-2, f"Final: {retention_pct[-1]:.1f}%", color=color_it, fontweight='bold', ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    generate_synaptic_shield_chart('russian_discovery_log_v73.md', 'synaptic_shield_breakthrough.png')
