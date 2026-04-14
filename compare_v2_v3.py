"""
Compare AGNIS v2 vs v3 on the same continual-learning task sequence.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch

import enhanced_agnis_v2 as v2
import agnis_v3 as v3


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


def compute_forgetting(retention_matrix: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mat = np.array(retention_matrix, dtype=np.float32)
    best = np.nanmin(mat, axis=0)
    final = mat[-1]
    forgetting = np.maximum(0.0, final - best)
    avg_forgetting = float(np.mean(forgetting))
    return best, final, forgetting, avg_forgetting


def run_v2(tasks):
    agnis = v2.EnhancedAGNISV2(
        input_dim=10,
        output_dim=3,
        initial_hidden=40,
        neuron_dim=12,
        max_new_neurons_per_task=20,
        replay_capacity_per_task=200,
        replay_every=20,
    )
    retention = v2.train_agnis_v2(
        agnis,
        tasks,
        epochs_per_task=3,
        known_boundaries=True,
        self_directed=False,
        samples_per_epoch=8,
        consolidate_every=10,
        schema_recompute_every=10,
    )
    return retention


def run_v3(tasks):
    agnis = v3.EnhancedAGNISV2(
        input_dim=10,
        output_dim=3,
        initial_hidden=40,
        neuron_dim=12,
        max_new_neurons_per_task=18,
        replay_capacity_per_task=300,
        replay_every=10,
        # v3 features
        use_tensorized_forward=True,
        tensor_graph_cache_enabled=True,
        output_head_enabled=True,
        output_head_lr=0.04,
        self_supervised_enabled=True,
        self_supervised_noise=0.03,
        plasticity_decay=0.006,
        plasticity_recovery=0.03,
        plasticity_min=0.25,
        autonomous_replay_boost=1.0,
        intrinsic_enabled=True,
        oracle_enabled=True,
        world_model_enabled=True,
        few_shot_enabled=True,
        oracle_replay_burst=5,
        freeze_base_after_tasks=2,
        hebb_noise_scale=0.03,
        hebb_corr_ema=0.9,
        hebb_corr_clip=0.9,
        mastery_mode_enabled=True,
        mastery_loss_threshold=0.12,
    )
    agnis.enable_learning_first(True)
    retention = v3.train_agnis_v2(
        agnis,
        tasks,
        epochs_per_task=3,
        known_boundaries=True,
        self_directed=False,
        samples_per_epoch=8,
        consolidate_every=5,
        schema_recompute_every=5,
    )
    return retention


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    tasks = [
        ("Task A: Negative Numbers", create_simple_task(50, 10, 3, "classification")),
        ("Task B: Positive Numbers", create_simple_task(50, 10, 3, "classification")),
        ("Task C: Sine Wave", create_simple_task(50, 10, 3, "regression")),
    ]

    print("Running v2...")
    retention_v2 = run_v2(tasks)
    print("Running v3 (tuned)...")
    retention_v3 = run_v3(tasks)

    b2, f2, g2, avg2 = compute_forgetting(retention_v2)
    b3, f3, g3, avg3 = compute_forgetting(retention_v3)

    print("\n=== Retention / Forgetting Summary ===")
    for i in range(len(tasks)):
        print(
            f"Task {i}: v2 best={b2[i]:.4f} final={f2[i]:.4f} forget={g2[i]:.4f} | "
            f"v3 best={b3[i]:.4f} final={f3[i]:.4f} forget={g3[i]:.4f}"
        )
    print(f"\nAvg forgetting: v2={avg2:.4f} | v3={avg3:.4f}")


if __name__ == "__main__":
    main()
