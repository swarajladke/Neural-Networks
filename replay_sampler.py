"""
replay_sampler.py — Compressed Prototype Replay with Teacher Distillation (V4.5)
=============================================================================
Stores coordinate prototypes of consolidated facts to enable functional replay
during sequential consolidation. Uses Medoid prototypes (the real key closest to
the arithmetic mean) to stay on the observed hidden manifold.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F


class ReplaySampler:
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
        self.prototypes: dict[str, torch.Tensor] = {} # fact_id -> prototype tensor (cpu, shape (E,))
        self.variances: dict[str, float] = {} # fact_id -> average deviation (1 - cos)

    def update_fact(self, fact_id: str, keys: torch.Tensor) -> None:
        """Store the medoid prototype (the actual key nearest to the mean) and compute variance."""
        if keys.shape[0] == 0:
            return
        
        # keys: shape (L, E)
        with torch.no_grad():
            mean_key = keys.mean(dim=0, keepdim=True) # (1, E)
            norm_keys = F.normalize(keys, dim=-1)
            norm_mean = F.normalize(mean_key, dim=-1)
            
            # Find medoid: max cosine similarity to the mean key
            sims = norm_keys @ norm_mean.T # (L, 1)
            medoid_idx = sims.argmax().item()
            medoid = norm_keys[medoid_idx].cpu()
            self.prototypes[fact_id] = medoid
            
            # Estimate natural radius (mean of 1 - cos(key, medoid))
            cos_sims = norm_keys @ medoid.to(keys.device).unsqueeze(1) # (L, 1)
            deviations = 1.0 - cos_sims
            self.variances[fact_id] = float(deviations.mean().item())

    def sample_historical(self, count: int, device: torch.device, sigma: float = 0.01) -> torch.Tensor:
        """Sample with replacement from prototypes, perturb, and re-normalize."""
        fact_ids = list(self.prototypes.keys())
        if not fact_ids:
            return torch.empty(0, self.embed_dim, device=device)
            
        idxs = torch.randint(0, len(fact_ids), (count,))
        sampled_keys = []
        for idx in idxs:
            fid = fact_ids[idx]
            proto = self.prototypes[fid].to(device)
            
            # Perturb in direction orthogonal to proto, or simple perturbation + re-normalization
            perturbed = proto + torch.randn_like(proto) * sigma
            perturbed = F.normalize(perturbed, dim=-1)
            sampled_keys.append(perturbed)
            
        return torch.stack(sampled_keys)

    def payload_bytes(self) -> int:
        """Return raw float32 tensor storage size of the prototypes."""
        return sum(p.numel() * 4 for p in self.prototypes.values())

    def serialized_bytes(self, path: str = "temp_sampler.pt") -> int:
        """Return serialized file size in bytes including keys/metadata."""
        state = {
            "prototypes": self.prototypes,
            "variances": self.variances,
            "embed_dim": self.embed_dim,
        }
        torch.save(state, path)
        size = os.path.getsize(path)
        try:
            os.remove(path)
        except OSError:
            pass
        return size
