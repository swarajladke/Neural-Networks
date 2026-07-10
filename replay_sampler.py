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

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def sample_gaussian(self, count: int, device: torch.device, sigma: float = 0.003) -> torch.Tensor:
        """
        Strategy A: Gaussian perturbation around medoid prototypes.
        Perturb each sampled prototype by σ in random direction, re-normalize.
        """
        fact_ids = list(self.prototypes.keys())
        if not fact_ids:
            return torch.empty(0, self.embed_dim, device=device)

        idxs = torch.randint(0, len(fact_ids), (count,))
        samples = []
        for idx in idxs:
            proto = self.prototypes[fact_ids[idx]].to(device)
            perturbed = proto + torch.randn_like(proto) * sigma
            samples.append(F.normalize(perturbed, dim=-1))
        return torch.stack(samples)

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
        Optionally filters generated coordinates through an immutable teacher
        to avoid distilling ambiguous off-manifold behavior.

        For each fact with ≥2 variants, SLERP between every pair at multiple t values.
        Adds small Gaussian noise and re-normalizes after interpolation.

        teacher (optional): frozen MLP to filter invalid samples.
        """
        candidates = []

        for base_id, variants in self.fact_variants.items():
            if len(variants) < 2:
                continue
            protos = [self.prototypes[v].to(device) for v in variants if v in self.prototypes]
            if len(protos) < 2:
                continue

            # All pairwise combinations
            for i in range(len(protos)):
                for j in range(i + 1, len(protos)):
                    for t in t_values:
                        interp = slerp(protos[i], protos[j], t)
                        if sigma_noise > 0:
                            interp = F.normalize(interp + torch.randn_like(interp) * sigma_noise, dim=-1)
                        candidates.append(interp)

        if not candidates:
            return self.sample_gaussian(count, device)

        pool = torch.stack(candidates)  # (N_candidates, E)

        # Teacher filtering (optional)
        if teacher is not None:
            pool = _filter_by_teacher(pool, teacher, replay_gate_min, margin_min)

        if pool.shape[0] == 0:
            return self.sample_gaussian(count, device)

        # Sample with replacement from filtered pool
        idxs = torch.randint(0, pool.shape[0], (count,))
        return pool[idxs]

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
        Covers the interior of the prototype simplex, not just edges.

        z = normalize( sum_k alpha_k * z_k )  where alpha ~ Dirichlet(gamma)
        """
        if not self.fact_variants:
            return self.sample_gaussian(count, device)

        base_ids = list(self.fact_variants.keys())
        samples_per_fact = max(1, count // len(base_ids))
        candidates = []

        for base_id in base_ids:
            variants = [v for v in self.fact_variants[base_id] if v in self.prototypes]
            if not variants:
                continue
            protos = torch.stack([self.prototypes[v] for v in variants]).to(device)  # (K, E)
            K = protos.shape[0]

            for _ in range(samples_per_fact):
                # Sample Dirichlet weights
                alpha = torch.distributions.Dirichlet(
                    torch.full((K,), gamma, device=device)
                ).sample()  # (K,)
                combo = (alpha.unsqueeze(-1) * protos).sum(dim=0)  # (E,)
                if sigma_noise > 0:
                    combo = combo + torch.randn_like(combo) * sigma_noise
                candidates.append(F.normalize(combo, dim=-1))

        if not candidates:
            return self.sample_gaussian(count, device)

        pool = torch.stack(candidates)

        if teacher is not None:
            pool = _filter_by_teacher(pool, teacher, replay_gate_min, margin_min)

        if pool.shape[0] == 0:
            return self.sample_gaussian(count, device)

        idxs = torch.randint(0, pool.shape[0], (count,))
        return pool[idxs]

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
        n_slerp = int(count * slerp_frac)
        n_dirichlet = count - n_slerp
        s1 = self.sample_tangent_slerp(n_slerp, device, teacher=teacher,
                                        replay_gate_min=replay_gate_min, margin_min=margin_min)
        s2 = self.sample_dirichlet(n_dirichlet, device, teacher=teacher,
                                    replay_gate_min=replay_gate_min, margin_min=margin_min)
        if s1.shape[0] == 0:
            return s2
        if s2.shape[0] == 0:
            return s1
        return torch.cat([s1, s2], dim=0)

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
