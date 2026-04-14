"""
PHASE 24: HIGH-ENTROPY STRESS BENCH
Defining complex symbolic and algorithmic tasks to stress the AGNIS V2 architecture.
"""

import torch
import random
import numpy as np
from typing import List, Tuple

def create_parity_task(num_samples: int = 100, seq_len: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Task 0: Parity-N (N-bit XOR)
    Mapping to 3D target for cross-task consistency.
    """
    dataset = []
    for _ in range(num_samples):
        x = torch.randint(0, 2, (seq_len,)).float()
        parity = x.sum() % 2
        # Target: [Even, Odd, Padding]
        y = torch.tensor([1.0, 0.0, 0.0] if parity == 0 else [0.0, 1.0, 0.0])
        dataset.append((x, y))
    return dataset

def create_reversal_task(num_samples: int = 100, seq_len: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Task 1: Symbolic Sequence Reversal
    Requires precise temporal indexing and recall.
    """
    dataset = []
    for _ in range(num_samples):
        x = torch.randn(seq_len)
        y = torch.flip(x, dims=[0])[:3] # Target first 3 elements of reversed seq
        dataset.append((x, y))
    return dataset

def create_associative_task(num_samples: int = 100, dim: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Task 2: Associative Key-Value Memory
    Mapping random high-dim anchors to specific "meaning" vectors.
    """
    # Create fixed anchors
    anchors = [torch.randn(dim) for _ in range(3)]
    targets = [torch.eye(3)[i] for i in range(3)]
    
    dataset = []
    for _ in range(num_samples):
        idx = random.randint(0, 2)
        # Add slight noise to anchor
        x = anchors[idx] + torch.randn(dim) * 0.1
        y = targets[idx]
        dataset.append((x, y))
    return dataset

def create_algorithmic_task(num_samples: int = 100, dim: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Task 3: Algorithmic Logic (Bit-Shift / Increment)
    Mapping inputs to their transformation.
    """
    dataset = []
    for _ in range(num_samples):
        x = torch.randint(0, 2, (dim,)).float()
        # Binary increment logic simulation
        y = torch.roll(x, shifts=1) # Simple bit shift for scaling
        y = y[:3] # Keep output small
        dataset.append((x, y))
    return dataset

def create_structural_task(num_samples: int = 100, dim: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Task 4: Structural Syntax / Patterns
    Recognizing patterns like [A, B, A, B]
    """
    dataset = []
    for _ in range(num_samples):
        pattern_type = random.randint(0, 1)
        if pattern_type == 0:
            # ABAB
            base_a = torch.randn(1)
            base_b = torch.randn(1)
            x = torch.cat([base_a, base_b, base_a, base_b, torch.randn(dim-4)])
            y = torch.tensor([1.0, 0.0, 0.0])
        else:
            # AAAA
            base_a = torch.randn(1)
            x = torch.cat([base_a, base_a, base_a, base_a, torch.randn(dim-4)])
            y = torch.tensor([0.0, 1.0, 0.0])
        dataset.append((x, y))
    return dataset
