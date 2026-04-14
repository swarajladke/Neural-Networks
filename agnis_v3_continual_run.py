"""
Run AGNIS v3 with continual-learning oriented settings and report retention/forgetting.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch

from agnis_v3 import EnhancedAGNISV2, train_agnis_v2


def create_simple_task(
    num_samples: int = 100,
    input_dim: int = 10,
    output_dim: int = 3,
    task_type: str = "classification"
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_samples):
        x = torch.randn(input_dim)
        if task_type == "classification":
            total = x.sum().item()
            if total < -1:
                y = torch.tensor([1.0, 0.0, 0.0])
            elif total > 1:
                y = torch.tensor([0.0, 0.0, 1.0])
            else:
                y = torch.tensor([0.0, 1.0, 0.0])
        else:
            y = torch.sin(x[:output_dim])
        dataset.append((x, y))
    return dataset


def summarize_retention(retention_matrix: List[List[float]]) -> None:
    if not retention_matrix:
        print("No retention data.")
        return
    mat = np.array(retention_matrix, dtype=np.float32)
    # Lower loss is better. Compute forgetting as final loss - best historical loss per task.
    best = np.nanmin(mat, axis=0)
    final = mat[-1]
    forgetting = np.maximum(0.0, final - best)
    print("\nRetention summary (lower is better):")
    for i, (b, f, g) in enumerate(zip(best, final, forgetting)):
        print(f"  Task {i}: best={b:.4f} final={f:.4f} forgetting={g:.4f}")
    print(f"Avg forgetting: {float(np.mean(forgetting)):.4f}")


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    agnis = EnhancedAGNISV2(
        input_dim=10,
        output_dim=3,
        initial_hidden=40,
        neuron_dim=12,
        max_new_neurons_per_task=20,
        replay_capacity_per_task=200,
        replay_every=20,
        # Continual learning / stability knobs
        use_tensorized_forward=True,
        tensor_graph_cache_enabled=True,
        output_head_enabled=True,
        output_head_lr=0.05,
        self_supervised_enabled=True,
        self_supervised_noise=0.05,
        plasticity_decay=0.01,
        plasticity_recovery=0.02,
        plasticity_min=0.2,
        autonomous_replay_boost=0.5,
        intrinsic_enabled=True,
        oracle_enabled=True,
        world_model_enabled=True,
        few_shot_enabled=True,
    )

    # Optional: bias toward faster learning early
    agnis.enable_learning_first(True)

    tasks = [
        ("Task A: Negative Numbers", create_simple_task(60, 10, 3, "classification")),
        ("Task B: Positive Numbers", create_simple_task(60, 10, 3, "classification")),
        ("Task C: Sine Wave", create_simple_task(60, 10, 3, "regression")),
        ("Task D: Mixed Classification", create_simple_task(60, 10, 3, "classification")),
        ("Task E: Small Regression", create_simple_task(60, 10, 3, "regression")),
    ]

    retention = train_agnis_v2(
        agnis,
        tasks,
        epochs_per_task=10,
        known_boundaries=True,
        self_directed=False,
        samples_per_epoch=20,
        consolidate_every=10,
        schema_recompute_every=10,
    )

    summarize_retention(retention)


if __name__ == "__main__":
    main()
