"""
agnis_validation_harness.py — Continual Learning Metrics and Validation (V4.5)
=============================================================================
Computes sequential block matrices (R_{i,j}) and standardized continual learning
metrics (Plasticity, Forgetting, BWT, FWT, Behavioral NLL/KL drift).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


class CLMetricsEvaluator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        
        # Performance matrices (shape: (T + 1, T))
        # row i is stage i (0 = baseline, 1..T are sequential stages)
        # col j is block j (0..T-1)
        self.R_exact = np.zeros((num_blocks + 1, num_blocks))
        self.R_paraphrase = np.zeros((num_blocks + 1, num_blocks))
        self.R_gate_tpr = np.zeros((num_blocks + 1, num_blocks))
        self.R_gate_fpr = np.zeros((num_blocks + 1, num_blocks))

    def update_metrics(
        self,
        stage: int,
        block_idx: int,
        exact_recall: float,
        paraphrase_recall: float,
        gate_tpr: float,
        gate_fpr: float,
    ) -> None:
        """Record evaluation performance of stage i on block j."""
        self.R_exact[stage, block_idx] = exact_recall
        self.R_paraphrase[stage, block_idx] = paraphrase_recall
        self.R_gate_tpr[stage, block_idx] = gate_tpr
        self.R_gate_fpr[stage, block_idx] = gate_fpr

    def compute_cl_summary(self, matrix_type: str = "exact") -> dict[str, float]:
        """Compute Plasticity, Average Recall, Forgetting, BWT, FWT for a given matrix."""
        R = self.R_exact if matrix_type == "exact" else self.R_paraphrase
        T = self.num_blocks
        
        # 1. Plasticity: P_i = R_{i, i} (using 1-indexed stage for block i)
        plasticity = [R[i, i - 1] for i in range(1, T + 1)]
        
        # 2. Final Average Recall: A_T = mean(R_{T, j})
        final_recall = float(np.mean(R[T, :]))
        
        # 3. Average Forgetting: F_{T, j} = max_{k=j..T-1} R_{k, j} - R_{T, j}
        forgetting = []
        for j in range(T - 1): # only for historical blocks
            # k maps to stage indices: j+1 .. T-1 (which are rows j+1 .. T-1)
            scores = [R[k, j] for k in range(j + 1, T)]
            max_hist = max(scores) if scores else R[j + 1, j]
            forgetting.append(max_hist - R[T, j])
        avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0
        
        # 4. Backward Transfer: BWT = mean(R_{T, j} - R_{j, j})
        bwt = []
        for j in range(T - 1):
            # stage j+1 corresponds to block j (0-indexed)
            bwt.append(R[T, j] - R[j + 1, j])
        avg_bwt = float(np.mean(bwt)) if bwt else 0.0
        
        # 5. Forward Transfer: FWT = mean(R_{j-1, j} - R_{0, j})
        fwt = []
        for j in range(1, T):
            # j is block j, prior stage is j (row j), base stage is 0 (row 0)
            fwt.append(R[j, j] - R[0, j])
        avg_fwt = float(np.mean(fwt)) if fwt else 0.0
        
        return {
            "plasticity": plasticity,
            "final_recall": final_recall,
            "forgetting": avg_forgetting,
            "bwt": avg_bwt,
            "fwt": avg_fwt,
        }

    def print_matrices(self) -> None:
        """Print matrices in lower-triangular aligned format."""
        T = self.num_blocks
        
        def print_matrix(name: str, mat: np.ndarray):
            print(f"\n--- Matrix R_{name} (stages 0..{T} vs blocks 1..{T}) ---")
            header = "Stage | " + " ".join(f"Blk {j+1}" for j in range(T))
            print(header)
            print("-" * len(header))
            for i in range(T + 1):
                row_str = f"  {i:2d}  | " + " ".join(f"{mat[i, j]*100:5.1f}%" for j in range(T))
                print(row_str)
                
        print_matrix("Exact", self.R_exact)
        print_matrix("Paraphrase", self.R_paraphrase)
        print_matrix("Gate TPR", self.R_gate_tpr)
        print_matrix("Gate FPR", self.R_gate_fpr)


def compute_behavioral_divergence(
    base_probs: torch.Tensor,
    hybrid_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Compute asymmetric KL divergence D_KL(P_base || P_hybrid) averaged over non-padded tokens.
    
    base_probs   : (N, V) probability distribution
    hybrid_probs : (N, V) probability distribution
    """
    # Clip to avoid log(0)
    eps = 1e-12
    p = torch.clamp(base_probs, eps, 1.0)
    q = torch.clamp(hybrid_probs, eps, 1.0)
    
    kl = p * (torch.log(p) - torch.log(q)) # (N, V)
    kl_sum = kl.sum(dim=-1) # (N,)
    
    if mask is not None:
        kl_sum = kl_sum * mask.float()
        return float((kl_sum.sum() / mask.float().sum()).item())
    
    return float(kl_sum.mean().item())
