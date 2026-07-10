"""
replay_sampler.py — Compressed Prototype Replay with Teacher Distillation (V4.5)
=============================================================================
Stores coordinate prototypes of consolidated facts. Supports three sampling
strategies:

  1. Gaussian:  perturb medoid prototypes with small isotropic noise
  2. Tangent:   spherical linear interpolation (SLERP) between prototype pairs
  3. Dirichlet: normalized convex combination of all prototypes for a fact

All sampled coordinates are L2-normalized to stay on the unit hypersphere,
matching the MLP input distribution.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SLERP — spherical linear interpolation on the unit hypersphere
# ---------------------------------------------------------------------------

def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Spherical linear interpolation between two L2-normalized vectors.
    Falls back to normalized linear interpolation when vectors are nearly identical.

    a, b : (..., E) normalized tensors
    t    : scalar in [0, 1]
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)           # angle between vectors
    sin_theta = torch.sin(theta)

    # Spherical path
    spherical = (
        torch.sin((1 - t) * theta) / sin_theta * a
        + torch.sin(t * theta) / sin_theta * b
    )
    # Linear fallback for near-identical vectors
    linear = F.normalize((1 - t) * a + t * b, dim=-1)

    use_linear = sin_theta.abs() < 1e-4
    result = torch.where(use_linear, linear, spherical)
    return F.normalize(result, dim=-1)


# ---------------------------------------------------------------------------
# ReplaySampler
# ---------------------------------------------------------------------------

class ReplaySampler:
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
        # fact_variant_id -> prototype tensor (cpu, L2-normalized, shape (E,))
        self.prototypes: dict[str, torch.Tensor] = {}
        # fact_variant_id -> average angular deviation (1 - cos)
        self.variances: dict[str, float] = {}
        # fact base id -> list of variant keys (e.g. ["F01_stmt", "F01_qa", "F01_cloze"])
        self.fact_variants: dict[str, list[str]] = {}
        
        # Caching pools to accelerate training (avoid rebuilding pools at every step)
        self._slerp_pool: torch.Tensor | None = None
        self._dirichlet_pool: torch.Tensor | None = None
        self._mixed_pool: torch.Tensor | None = None
        self._all_protos_tensor: torch.Tensor | None = None

    def update_fact(self, fact_id: str, keys: torch.Tensor) -> None:
        """Store the medoid prototype (the actual key nearest to the mean)."""
        if keys.shape[0] == 0:
            return

        with torch.no_grad():
            norm_keys = F.normalize(keys, dim=-1)
            mean_key = norm_keys.mean(dim=0, keepdim=True)
            norm_mean = F.normalize(mean_key, dim=-1)

            # Medoid: key with max cosine sim to the mean
            sims = norm_keys @ norm_mean.T
            medoid_idx = sims.argmax().item()
            medoid = norm_keys[medoid_idx].cpu()
            self.prototypes[fact_id] = medoid

            # Angular variance
            cos_sims = norm_keys @ medoid.to(keys.device).unsqueeze(1)
            self.variances[fact_id] = float((1.0 - cos_sims).mean().item())

        # Track which base fact this variant belongs to
        base_id = fact_id.rsplit("_", 1)[0] if "_" in fact_id else fact_id
        if base_id not in self.fact_variants:
            self.fact_variants[base_id] = []
        if fact_id not in self.fact_variants[base_id]:
            self.fact_variants[base_id].append(fact_id)
            
        # Invalidate cached pools
        self._slerp_pool = None
        self._dirichlet_pool = None
        self._mixed_pool = None
        self._all_protos_tensor = None

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def sample_gaussian(self, count: int, device: torch.device, sigma: float = 0.003) -> torch.Tensor:
        """
        Strategy A: Gaussian perturbation around medoid prototypes.
        Fully vectorized to run in microseconds with zero Python loops.
        """
        fact_ids = list(self.prototypes.keys())
        if not fact_ids:
            return torch.empty(0, self.embed_dim, device=device)

        if self._all_protos_tensor is None:
            self._all_protos_tensor = torch.stack([self.prototypes[fid] for fid in fact_ids]).cpu()

        idxs = torch.randint(0, self._all_protos_tensor.shape[0], (count,))
        base = self._all_protos_tensor[idxs].to(device)
        perturbed = base + torch.randn_like(base) * sigma
        return F.normalize(perturbed, dim=-1)

    # Alias for backward compatibility
    def sample_historical(self, count: int, device: torch.device, sigma: float = 0.003) -> torch.Tensor:
        return self.sample_gaussian(count, device, sigma)

    def sample_tangent_slerp(
        self,
        count: int,
        device: torch.device,
        t_values: tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 0.85),
        teacher=None,
        replay_gate_min: float = 0.3,
        margin_min: float = 0.0,
        sigma_noise: float = 0.002,
    ) -> torch.Tensor:
        """
        Strategy B / E: Pairwise SLERP between prototypes of the same fact.
        Caches the pool dynamically on CPU to run at GPU speeds.
        """
        if self._slerp_pool is None:
            candidates = []
            for base_id, variants in self.fact_variants.items():
                if len(variants) < 2:
                    continue
                protos = [self.prototypes[v] for v in variants if v in self.prototypes]
                if len(protos) < 2:
                    continue

                # All pairwise combinations
                for i in range(len(protos)):
                    for j in range(i + 1, len(protos)):
                        for t in t_values:
                            interp = slerp(protos[i], protos[j], t)
                            if sigma_noise > 0:
                                interp = F.normalize(interp + torch.randn_like(interp) * sigma_noise, dim=-1)
                            candidates.append(interp.cpu())

            if not candidates:
                self._slerp_pool = self.sample_gaussian(max(64, count), torch.device("cpu")).cpu()
            else:
                pool = torch.stack(candidates)
                if teacher is not None:
                    pool = _filter_by_teacher(pool.to(device), teacher, replay_gate_min, margin_min).cpu()
                if pool.shape[0] == 0:
                    self._slerp_pool = self.sample_gaussian(max(64, count), torch.device("cpu")).cpu()
                else:
                    self._slerp_pool = pool

        pool = self._slerp_pool
        idxs = torch.randint(0, pool.shape[0], (count,))
        return pool[idxs].to(device)

    def sample_dirichlet(
        self,
        count: int,
        device: torch.device,
        gamma: float = 1.0,
        sigma_noise: float = 0.002,
        teacher=None,
        replay_gate_min: float = 0.3,
        margin_min: float = 0.0,
    ) -> torch.Tensor:
        """
        Strategy: Dirichlet-weighted convex combination of a fact's prototypes.
        """
        if self._dirichlet_pool is None:
            if not self.fact_variants:
                self._dirichlet_pool = self.sample_gaussian(max(64, count), torch.device("cpu")).cpu()
            else:
                base_ids = list(self.fact_variants.keys())
                pool_size = max(5000, count * 2)
                samples_per_fact = max(1, pool_size // len(base_ids))
                candidates = []

                for base_id in base_ids:
                    variants = [v for v in self.fact_variants[base_id] if v in self.prototypes]
                    if not variants:
                        continue
                    protos = torch.stack([self.prototypes[v] for v in variants])  # (K, E)
                    K = protos.shape[0]

                    for _ in range(samples_per_fact):
                        alpha = torch.distributions.Dirichlet(
                            torch.full((K,), gamma)
                        ).sample()
                        combo = (alpha.unsqueeze(-1) * protos).sum(dim=0)
                        if sigma_noise > 0:
                            combo = combo + torch.randn_like(combo) * sigma_noise
                        candidates.append(F.normalize(combo, dim=-1).cpu())

                if not candidates:
                    self._dirichlet_pool = self.sample_gaussian(max(64, count), torch.device("cpu")).cpu()
                else:
                    pool = torch.stack(candidates)
                    if teacher is not None:
                        pool = _filter_by_teacher(pool.to(device), teacher, replay_gate_min, margin_min).cpu()
                    if pool.shape[0] == 0:
                        self._dirichlet_pool = self.sample_gaussian(max(64, count), torch.device("cpu")).cpu()
                    else:
                        self._dirichlet_pool = pool

        pool = self._dirichlet_pool
        idxs = torch.randint(0, pool.shape[0], (count,))
        return pool[idxs].to(device)

    def sample_mixed(
        self,
        count: int,
        device: torch.device,
        slerp_frac: float = 0.5,
        teacher=None,
        replay_gate_min: float = 0.3,
        margin_min: float = 0.0,
    ) -> torch.Tensor:
        """50/50 mixture of SLERP tangent samples and Dirichlet interior samples."""
        if self._mixed_pool is None:
            n_slerp = max(2500, count)
            n_dirichlet = max(2500, count)
            s1 = self.sample_tangent_slerp(n_slerp, torch.device("cpu"), teacher=teacher,
                                            replay_gate_min=replay_gate_min, margin_min=margin_min)
            s2 = self.sample_dirichlet(n_dirichlet, torch.device("cpu"), teacher=teacher,
                                        replay_gate_min=replay_gate_min, margin_min=margin_min)
            self._mixed_pool = torch.cat([s1, s2], dim=0).cpu()

        pool = self._mixed_pool
        idxs = torch.randint(0, pool.shape[0], (count,))
        return pool[idxs].to(device)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def payload_bytes(self) -> int:
        """Raw float32 tensor storage of all prototypes."""
        return sum(p.numel() * 4 for p in self.prototypes.values())

    def bytes_per_fact(self) -> float:
        """Average payload bytes per base fact."""
        n_facts = len(self.fact_variants) if self.fact_variants else max(1, len(self.prototypes))
        return self.payload_bytes() / n_facts

    def serialized_bytes(self, path: str = "temp_sampler.pt") -> int:
        """Serialized file size including all metadata."""
        state = {
            "prototypes": self.prototypes,
            "variances": self.variances,
            "fact_variants": self.fact_variants,
            "embed_dim": self.embed_dim,
        }
        torch.save(state, path)
        size = os.path.getsize(path)
        try:
            os.remove(path)
        except OSError:
            pass
        return size


# ---------------------------------------------------------------------------
# Teacher-filtering helper
# ---------------------------------------------------------------------------

def _filter_by_teacher(
    pool: torch.Tensor,
    teacher,
    replay_gate_min: float,
    margin_min: float,
) -> torch.Tensor:
    """
    Filter replay candidates through immutable teacher.
    Retains only samples where:
      - gate probability >= replay_gate_min
      - top-1 predicted token has a logit margin >= margin_min above the runner-up
    """
    if pool.shape[0] == 0:
        return pool

    kept = []
    with torch.no_grad():
        for z in pool:
            logits, sim = teacher(z.unsqueeze(0))
            gate_prob = torch.sigmoid(
                teacher.gate_sharpness * (sim[0, 0] - teacher.gate_threshold)
            ).item() if hasattr(teacher, "gate_sharpness") else 1.0

            if gate_prob < replay_gate_min:
                continue

            top2 = logits[0].topk(2).values
            margin = (top2[0] - top2[1]).item()
            if margin < margin_min:
                continue

            kept.append(z)

    return torch.stack(kept) if kept else pool[:0]
