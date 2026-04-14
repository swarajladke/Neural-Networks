"""
AGNIS V2 vs V3: Head-to-Head 5-Minute Benchmark
=================================================
Runs both engines on identical tasks for 5 minutes each,
then compares: retention, learning speed, neuron growth, and loss.
"""

import time
import sys
import os
import json
import random
from collections import deque
from typing import List, Tuple, Dict

import torch
import numpy as np

# ── Task generators (same as complex_tasks.py) ──────────────────────────
from complex_tasks import (
    create_parity_task,
    create_reversal_task,
    create_associative_task,
    create_algorithmic_task,
    create_structural_task,
)

TASK_GENERATORS = [
    ("Parity",       create_parity_task),
    ("Reversal",     create_reversal_task),
    ("Associative",  create_associative_task),
    ("Algorithmic",  create_algorithmic_task),
    ("Structural",   create_structural_task),
]

INPUT_DIM = 10
OUTPUT_DIM = 3
SEED = 42
RUN_SECONDS = 300  # 5 minutes per engine


# ── Helpers ──────────────────────────────────────────────────────────────
def generate_shared_datasets(num_samples: int = 200) -> List[Tuple[str, int, List]]:
    """Generate the same datasets for both engines."""
    datasets = []
    for task_id, (name, gen_fn) in enumerate(TASK_GENERATORS):
        data = gen_fn(num_samples, INPUT_DIM)
        datasets.append((name, task_id, data))
    return datasets


def evaluate_retention(model, datasets, num_probes: int = 20) -> Dict[str, float]:
    """Evaluate retention across all tasks."""
    results = {}
    for name, task_id, data in datasets:
        model.current_task_id = task_id
        if hasattr(model, "gating_enabled") and model.gating_enabled:
            if task_id in model.task_gate_masks:
                model.active_neurons = model.task_gate_masks[task_id]

        total_loss = 0.0
        count = min(num_probes, len(data))
        for x, y in data[:count]:
            out = model.forward(x)
            loss = torch.mean((y - out) ** 2).item()
            total_loss += loss
        results[name] = total_loss / max(1, count)
    return results


def run_engine(engine_label: str, model, datasets, run_seconds: int) -> Dict:
    """
    Train a model on interleaved tasks for `run_seconds` seconds.
    Returns detailed metrics.
    """
    print(f"\n{'='*70}")
    print(f"  ENGINE: {engine_label}")
    print(f"  Duration: {run_seconds}s ({run_seconds//60}m {run_seconds%60}s)")
    print(f"{'='*70}")

    # Flatten dataset into a big interleaved stream
    stream = []
    for name, task_id, data in datasets:
        for x, y in data:
            stream.append((task_id, x, y))

    # Metrics tracking
    loss_history = []
    rolling_loss = deque(maxlen=50)
    neuron_counts = []
    connection_counts = []
    retention_snapshots = []
    steps_completed = 0
    tasks_started = set()

    # Pre-register all tasks
    for _, task_id, _ in datasets:
        model.begin_task(task_id)
        tasks_started.add(task_id)

    print(f"  Tasks registered: {len(tasks_started)}")
    stats_init = model.get_stats()
    print(f"  Initial neurons: {stats_init.get('current_neurons', 'N/A')}")
    print(f"  Initial connections: {stats_init.get('current_connections', 'N/A')}")
    print(f"\n  [>] Training started...")

    start_time = time.time()
    last_print = start_time
    last_snapshot = start_time

    while True:
        elapsed = time.time() - start_time
        if elapsed >= run_seconds:
            break

        # Pick a random sample from the interleaved stream
        task_id, x, y = random.choice(stream)
        loss = model.learn(x, y, task_id=task_id)

        steps_completed += 1
        loss_history.append(loss)
        rolling_loss.append(loss)
        neuron_counts.append(len(model.neurons))

        conn_count = sum(len(n.outgoing) for n in model.neurons.values())
        connection_counts.append(conn_count)

        # Print progress every 30s
        now = time.time()
        if now - last_print >= 30:
            avg = sum(rolling_loss) / max(1, len(rolling_loss))
            pct = elapsed / run_seconds * 100
            print(f"  [{pct:5.1f}%] Step {steps_completed:>6d} | "
                  f"Rolling Loss: {avg:.4f} | "
                  f"Neurons: {len(model.neurons)} | "
                  f"Elapsed: {elapsed:.0f}s")
            last_print = now

        # Retention snapshot every 60s
        if now - last_snapshot >= 60:
            retention = evaluate_retention(model, datasets)
            retention_snapshots.append({
                "elapsed_s": round(elapsed, 1),
                "step": steps_completed,
                "retention": retention,
                "mean_loss": round(sum(retention.values()) / max(1, len(retention)), 6),
            })
            last_snapshot = now

    # Final snapshot
    elapsed_final = time.time() - start_time
    final_retention = evaluate_retention(model, datasets)
    retention_snapshots.append({
        "elapsed_s": round(elapsed_final, 1),
        "step": steps_completed,
        "retention": final_retention,
        "mean_loss": round(sum(final_retention.values()) / max(1, len(final_retention)), 6),
    })

    final_stats = model.get_stats()

    # Compute summary
    first_100 = loss_history[:100] if len(loss_history) >= 100 else loss_history
    last_100 = loss_history[-100:] if len(loss_history) >= 100 else loss_history

    result = {
        "engine": engine_label,
        "total_steps": steps_completed,
        "total_time_s": round(elapsed_final, 2),
        "steps_per_second": round(steps_completed / max(1, elapsed_final), 2),
        "initial_neurons": stats_init.get("current_neurons", 0),
        "final_neurons": final_stats.get("current_neurons", len(model.neurons)),
        "neurons_created": final_stats.get("neurons_created", 0),
        "final_connections": sum(len(n.outgoing) for n in model.neurons.values()),
        "mean_loss_first_100": round(sum(first_100) / max(1, len(first_100)), 6),
        "mean_loss_last_100": round(sum(last_100) / max(1, len(last_100)), 6),
        "loss_improvement_pct": 0.0,
        "final_retention": {k: round(v, 6) for k, v in final_retention.items()},
        "final_mean_retention": round(sum(final_retention.values()) / max(1, len(final_retention)), 6),
        "retention_timeline": retention_snapshots,
        "loss_history_sample": [round(l, 4) for l in loss_history[::max(1, len(loss_history)//200)]],
    }

    first_avg = result["mean_loss_first_100"]
    last_avg = result["mean_loss_last_100"]
    if first_avg > 0:
        result["loss_improvement_pct"] = round((first_avg - last_avg) / first_avg * 100, 2)

    print(f"\n  -- {engine_label} RESULTS --")
    print(f"  Total steps:        {result['total_steps']:,}")
    print(f"  Steps/second:       {result['steps_per_second']:.1f}")
    print(f"  Neurons:            {result['initial_neurons']} -> {result['final_neurons']}")
    print(f"  Connections:        {result['final_connections']:,}")
    print(f"  Loss (first 100):   {result['mean_loss_first_100']:.4f}")
    print(f"  Loss (last 100):    {result['mean_loss_last_100']:.4f}")
    print(f"  Improvement:        {result['loss_improvement_pct']:.1f}%")
    print(f"  Final Retention:")
    for task_name, loss_val in final_retention.items():
        print(f"    {task_name:15s}: {loss_val:.4f}")
    print(f"  Mean Retention:     {result['final_mean_retention']:.4f}")

    return result


def print_comparison(v2_result: Dict, v3_result: Dict):
    """Print a side-by-side comparison table."""
    print(f"\n\n{'='*70}")
    print(f"  AGNIS V2 vs V3: SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    def row(label, v2_val, v3_val, fmt="{}", better="lower"):
        v2_s = fmt.format(v2_val)
        v3_s = fmt.format(v3_val)
        if better == "lower":
            winner = "V2" if v2_val < v3_val else ("V3" if v3_val < v2_val else "TIE")
        elif better == "higher":
            winner = "V2" if v2_val > v3_val else ("V3" if v3_val > v2_val else "TIE")
        else:
            winner = ""
        print(f"  {label:30s} | {v2_s:>12s} | {v3_s:>12s} | {winner:>3s}")

    print(f"  {'Metric':30s} | {'AGNIS V2':>12s} | {'AGNIS V3':>12s} | {'Win':>3s}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*3}-")

    row("Total Steps", v2_result["total_steps"], v3_result["total_steps"], "{:,}", "higher")
    row("Steps/Second", v2_result["steps_per_second"], v3_result["steps_per_second"], "{:.1f}", "higher")
    row("Final Neurons", v2_result["final_neurons"], v3_result["final_neurons"], "{:,}", "")
    row("Final Connections", v2_result["final_connections"], v3_result["final_connections"], "{:,}", "")
    row("Neurons Created", v2_result["neurons_created"], v3_result["neurons_created"], "{:,}", "")
    row("Loss (first 100)", v2_result["mean_loss_first_100"], v3_result["mean_loss_first_100"], "{:.4f}", "lower")
    row("Loss (last 100)", v2_result["mean_loss_last_100"], v3_result["mean_loss_last_100"], "{:.4f}", "lower")
    row("Loss Improvement pct", v2_result["loss_improvement_pct"], v3_result["loss_improvement_pct"], "{:.1f}%", "higher")
    row("Mean Retention", v2_result["final_mean_retention"], v3_result["final_mean_retention"], "{:.4f}", "lower")

    print(f"\n  Per-Task Retention:")
    print(f"  {'Task':15s} | {'V2':>10s} | {'V3':>10s} | {'Win':>3s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*3}-")
    for task_name in v2_result["final_retention"]:
        v2_l = v2_result["final_retention"][task_name]
        v3_l = v3_result["final_retention"].get(task_name, float("nan"))
        winner = "V2" if v2_l < v3_l else ("V3" if v3_l < v2_l else "TIE")
        print(f"  {task_name:15s} | {v2_l:10.4f} | {v3_l:10.4f} | {winner:>3s}")

    # Determine overall winner
    v2_wins = 0
    v3_wins = 0
    # Lower loss = better
    if v2_result["mean_loss_last_100"] < v3_result["mean_loss_last_100"]: v2_wins += 1
    else: v3_wins += 1
    if v2_result["final_mean_retention"] < v3_result["final_mean_retention"]: v2_wins += 1
    else: v3_wins += 1
    # Higher throughput = better
    if v2_result["steps_per_second"] > v3_result["steps_per_second"]: v2_wins += 1
    else: v3_wins += 1
    # Higher improvement = better
    if v2_result["loss_improvement_pct"] > v3_result["loss_improvement_pct"]: v2_wins += 1
    else: v3_wins += 1

    print(f"")
    print(f"  +---------------------------------------------------+")
    if v2_wins > v3_wins:
        print(f"  |  OVERALL WINNER: AGNIS V2  ({v2_wins}-{v3_wins})                  |")
    elif v3_wins > v2_wins:
        print(f"  |  OVERALL WINNER: AGNIS V3  ({v3_wins}-{v2_wins})                  |")
    else:
        print(f"  |  RESULT: TIE  ({v2_wins}-{v3_wins})                              |")
    print(f"  +---------------------------------------------------+")


def main():
    print("=" * 70)
    print("  AGNIS V2 vs V3: HEAD-TO-HEAD BENCHMARK")
    print(f"  Duration: {RUN_SECONDS}s per engine ({RUN_SECONDS//60}m)")
    print(f"  Tasks: {len(TASK_GENERATORS)} ({', '.join(n for n, _ in TASK_GENERATORS)})")
    print(f"  Seed: {SEED}")
    print("=" * 70)

    # Fix seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Generate shared datasets
    datasets = generate_shared_datasets(num_samples=200)

    # ── Run V2 ───────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n[1/2] Importing AGNIS V2 from enhanced_agnis_v2.py...")
    import enhanced_agnis_v2 as v2_module
    v2_model = v2_module.EnhancedAGNISV2(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        initial_hidden=40,
        neuron_dim=16,
        max_new_neurons_per_task=20,
        replay_capacity_per_task=100,
        replay_every=50,
        gating_enabled=True,
        shared_gate_ratio=0.25,
        freeze_base_after_tasks=3,
        oracle_enabled=True,
    )

    v2_result = run_engine("AGNIS V2", v2_model, datasets, RUN_SECONDS)

    # ── Run V3 ───────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n\n[2/2] Importing AGNIS V3 from agnis_v3.py...")
    import agnis_v3 as v3_module
    v3_model = v3_module.EnhancedAGNISV2(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        initial_hidden=40,
        neuron_dim=16,
        max_new_neurons_per_task=20,
        replay_capacity_per_task=100,
        replay_every=50,
        gating_enabled=True,
        shared_gate_ratio=0.25,
        freeze_base_after_tasks=3,
        oracle_enabled=True,
        # V3-specific features
        meta_controller_enabled=True,
        goal_generator_enabled=True,
        semantic_schema_enabled=True,
        self_audit_enabled=True,
        self_rewriter_enabled=True,
        episodic_memory_enabled=True,
        few_shot_enabled=True,
        social_reasoning_enabled=True,
        physical_intuition_enabled=True,
        value_alignment_enabled=True,
        long_term_planning_enabled=True,
        reward_composer_enabled=True,
        intrinsic_enabled=True,
        world_model_enabled=True,
    )

    v3_result = run_engine("AGNIS V3", v3_model, datasets, RUN_SECONDS)

    # ── Comparison ───────────────────────────────────────────────────────
    print_comparison(v2_result, v3_result)

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "v2_vs_v3_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"v2": v2_result, "v3": v3_result}, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Generate retention plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("AGNIS V2 vs V3: Head-to-Head Benchmark", fontsize=16, fontweight="bold")

        # 1. Loss over time (sampled)
        ax = axes[0, 0]
        v2_loss = v2_result["loss_history_sample"]
        v3_loss = v3_result["loss_history_sample"]
        ax.plot(v2_loss, label="V2", color="#2c3e50", alpha=0.8, lw=1.2)
        ax.plot(v3_loss, label="V3", color="#e74c3c", alpha=0.8, lw=1.2)
        ax.set_title("Training Loss (Sampled)", fontweight="bold")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Retention timeline
        ax = axes[0, 1]
        v2_timeline = [(s["elapsed_s"], s["mean_loss"]) for s in v2_result["retention_timeline"]]
        v3_timeline = [(s["elapsed_s"], s["mean_loss"]) for s in v3_result["retention_timeline"]]
        if v2_timeline:
            ax.plot(*zip(*v2_timeline), "o-", label="V2", color="#2c3e50", lw=2)
        if v3_timeline:
            ax.plot(*zip(*v3_timeline), "s-", label="V3", color="#e74c3c", lw=2)
        ax.set_title("Mean Retention Over Time", fontweight="bold")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Mean Loss (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Per-task final retention bar chart
        ax = axes[1, 0]
        task_names = list(v2_result["final_retention"].keys())
        v2_vals = [v2_result["final_retention"][t] for t in task_names]
        v3_vals = [v3_result["final_retention"].get(t, 0) for t in task_names]
        x_pos = np.arange(len(task_names))
        width = 0.35
        ax.bar(x_pos - width/2, v2_vals, width, label="V2", color="#2c3e50", alpha=0.8)
        ax.bar(x_pos + width/2, v3_vals, width, label="V3", color="#e74c3c", alpha=0.8)
        ax.set_title("Per-Task Final Retention", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_names, rotation=25, ha="right")
        ax.set_ylabel("Loss (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Summary stats comparison
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = (
            f"{'Metric':<25s}  {'V2':>10s}  {'V3':>10s}\n"
            f"{'-'*50}\n"
            f"{'Total Steps':<25s}  {v2_result['total_steps']:>10,d}  {v3_result['total_steps']:>10,d}\n"
            f"{'Steps/sec':<25s}  {v2_result['steps_per_second']:>10.1f}  {v3_result['steps_per_second']:>10.1f}\n"
            f"{'Final Neurons':<25s}  {v2_result['final_neurons']:>10,d}  {v3_result['final_neurons']:>10,d}\n"
            f"{'Final Connections':<25s}  {v2_result['final_connections']:>10,d}  {v3_result['final_connections']:>10,d}\n"
            f"{'Loss (first 100)':<25s}  {v2_result['mean_loss_first_100']:>10.4f}  {v3_result['mean_loss_first_100']:>10.4f}\n"
            f"{'Loss (last 100)':<25s}  {v2_result['mean_loss_last_100']:>10.4f}  {v3_result['mean_loss_last_100']:>10.4f}\n"
            f"{'Improvement pct':<25s}  {v2_result['loss_improvement_pct']:>9.1f}%  {v3_result['loss_improvement_pct']:>9.1f}%\n"
            f"{'Mean Retention':<25s}  {v2_result['final_mean_retention']:>10.4f}  {v3_result['final_mean_retention']:>10.4f}\n"
        )
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title("Summary", fontweight="bold")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "v2_vs_v3_benchmark.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to: {plot_path}")
    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")

    print("\n  BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
