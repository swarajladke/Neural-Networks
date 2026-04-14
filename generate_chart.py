"""Generate the V2 vs V3 benchmark chart from saved JSON results."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(r'c:\Users\Helios\Desktop\Neural Networks\v2_vs_v3_results.json', 'r') as f:
    data = json.load(f)

v2 = data['v2']
v3 = data['v3']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AGNIS V2 vs V3: Head-to-Head Benchmark (5 min each)', fontsize=16, fontweight='bold')

# 1. Loss over time
ax = axes[0, 0]
ax.plot(v2['loss_history_sample'], label='V2', color='#2c3e50', alpha=0.8, lw=1.2)
ax.plot(v3['loss_history_sample'], label='V3', color='#e74c3c', alpha=0.8, lw=1.2)
ax.set_title('Training Loss (Sampled)', fontweight='bold')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Retention timeline
ax = axes[0, 1]
v2_tl = [(s['elapsed_s'], s['mean_loss']) for s in v2['retention_timeline']]
v3_tl = [(s['elapsed_s'], s['mean_loss']) for s in v3['retention_timeline']]
if v2_tl:
    ax.plot(*zip(*v2_tl), 'o-', label='V2', color='#2c3e50', lw=2)
if v3_tl:
    ax.plot(*zip(*v3_tl), 's-', label='V3', color='#e74c3c', lw=2)
ax.set_title('Mean Retention Over Time', fontweight='bold')
ax.set_xlabel('Elapsed (s)')
ax.set_ylabel('Mean Loss (lower=better)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Per-task bars
ax = axes[1, 0]
tasks = list(v2['final_retention'].keys())
v2v = [v2['final_retention'][t] for t in tasks]
v3v = [v3['final_retention'].get(t, 0) for t in tasks]
x = np.arange(len(tasks))
w = 0.35
ax.bar(x - w/2, v2v, w, label='V2', color='#2c3e50', alpha=0.8)
ax.bar(x + w/2, v3v, w, label='V3', color='#e74c3c', alpha=0.8)
ax.set_title('Per-Task Final Retention', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=25, ha='right')
ax.set_ylabel('Loss (lower=better)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Summary
ax = axes[1, 1]
ax.axis('off')
lines = [
    f"{'Metric':<25s}  {'V2':>10s}  {'V3':>10s}",
    "-" * 50,
    f"{'Total Steps':<25s}  {v2['total_steps']:>10,d}  {v3['total_steps']:>10,d}",
    f"{'Steps/sec':<25s}  {v2['steps_per_second']:>10.1f}  {v3['steps_per_second']:>10.1f}",
    f"{'Final Neurons':<25s}  {v2['final_neurons']:>10,d}  {v3['final_neurons']:>10,d}",
    f"{'Connections':<25s}  {v2['final_connections']:>10,d}  {v3['final_connections']:>10,d}",
    f"{'Loss (first 100)':<25s}  {v2['mean_loss_first_100']:>10.4f}  {v3['mean_loss_first_100']:>10.4f}",
    f"{'Loss (last 100)':<25s}  {v2['mean_loss_last_100']:>10.4f}  {v3['mean_loss_last_100']:>10.4f}",
    f"{'Improvement':<25s}  {v2['loss_improvement_pct']:>9.1f}%  {v3['loss_improvement_pct']:>9.1f}%",
    f"{'Mean Retention':<25s}  {v2['final_mean_retention']:>10.4f}  {v3['final_mean_retention']:>10.4f}",
    "",
    "WINNER: AGNIS V3 (3-1)"
]
txt = "\n".join(lines)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('Summary', fontweight='bold')

plt.tight_layout()
plt.savefig(r'c:\Users\Helios\Desktop\Neural Networks\v2_vs_v3_benchmark.png', dpi=150)
print('Plot saved to v2_vs_v3_benchmark.png')
