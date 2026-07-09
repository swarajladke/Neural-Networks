"""
fact_memory.py — V4.2 Episodic Key-Value Fact Memory + Fuzzy Query Projection
==============================================================================
The hippocampal fast path of a complementary learning system.

The V4.0 memory probe proved the smooth Hebbian recurrent state collapses to
a single attractor at write time (mean off-diagonal boundary cosine 0.979;
answer ranks 4k-47k of 50257). Correlated outer-product updates without
pattern separation amplify whatever direction is already dominant — more
passes or a higher beta_push deepens the collapse instead of fixing it.
What is missing is sparse, non-interfering, one-shot episodic storage.
This module provides exactly that:

  write : key   = frozen GPT-2 final hidden state at position t
                  (a content-addressable context code we get for free)
          value = the token id observed at position t+1
  read  : anisotropy-corrected cosine top-k attention over keys
          -> next-token probability distribution
  gate  : confidence gate on the best-match similarity. Exact context
          prefixes match at ~1.0; unrelated text falls far below the
          threshold, so the memory is silent off-fact (retention and
          PPL untouched).

V4.2 — Fuzzy Context Retrieval:
  V4.1c retrieval was lookup-exact: keys come from exact prompt prefixes, so
  a rephrased question falls below the gate. V4.2 adds a learned residual
  query projection applied to queries (never keys) before the anisotropy
  correction. The residual form q' = q + MLP(q) with a zero-initialized
  output layer is EXACTLY the identity at init (a plain 768->256->768
  bottleneck cannot represent identity: rank <= 256), so an untrained
  projection provably preserves the V4.1c results. Training happens in
  agnis_continual_v4_2.py via InfoNCE computed in the read-time similarity
  space exposed by read_space()/to_read_space().
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQueryProjection(nn.Module):
    """q' = q + MLP(q) with a zero-initialized output layer.

    Guarantees exact identity at init, so adding it to an already-calibrated
    memory changes nothing until it is trained. MLP: Linear(E, H) ->
    LeakyReLU(0.1) -> Linear(H, E) with fc_out weights and bias zeroed.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, negative_slope: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc_out(self.act(self.fc_in(x)))


class EpisodicFactMemory(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        vocab_size: int = 50257,
        top_k: int = 8,
        read_temp: float = 0.03,
        gate_threshold: float = 0.95,
        gate_sharpness: float = 80.0,
        lam_max: float = 0.95,
        npc_project: int = 5,
        proj_hidden: int = 256,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.read_temp = read_temp
        self.gate_threshold = gate_threshold
        self.gate_sharpness = gate_sharpness
        self.lam_max = lam_max
        self.npc_project = npc_project
        dev = torch.device(device)
        self.register_buffer("keys_raw", torch.empty(0, embed_dim, device=dev))
        self.register_buffer("values", torch.empty(0, dtype=torch.long, device=dev))
        # V4.2: learned fuzzy query alignment (identity until trained).
        self.query_proj = ResidualQueryProjection(embed_dim, proj_hidden).to(dev)

    def __len__(self) -> int:
        return self.keys_raw.shape[0]

    @torch.no_grad()
    def write(self, hidden_states: torch.Tensor, next_token_ids: torch.Tensor) -> None:
        """Store (context -> next token) pairs.

        hidden_states : (T, E) GPT-2 final hidden states for positions 0..T-1
        next_token_ids: (T,)  the token observed after each position
        """
        dev = self.keys_raw.device
        self.keys_raw = torch.cat([self.keys_raw, hidden_states.to(dev)], dim=0)
        self.values = torch.cat([self.values, next_token_ids.to(dev).long()], dim=0)

    def read_space(self):
        """Return (mu, V_sub): the anisotropy correction of the stored keys.

        mu    : (1, E) mean stored key (centering).
        V_sub : (E, npc) top principal components to project out, or None
                when projection-out is inactive.

        The V4.2 contrastive trainer MUST use this exact space so the learned
        geometry matches what the confidence gate sees at read time.
        """
        with torch.no_grad():
            mu = self.keys_raw.mean(dim=0, keepdim=True)
            centered = self.keys_raw - mu
            V_sub = None
            if self.npc_project > 0 and len(self) > self.npc_project:
                _, _, V = torch.svd(centered)
                V_sub = V[:, : self.npc_project]
        return mu, V_sub

    def to_read_space(self, x: torch.Tensor, mu: torch.Tensor, V_sub: torch.Tensor | None) -> torch.Tensor:
        """Center by mu, project out V_sub, L2-normalize. Differentiable
        (the contrastive trainer backprops through it into query_proj);
        read() wraps it in no_grad."""
        x = x - mu
        if V_sub is not None:
            x = x - x @ V_sub @ V_sub.T
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def read(self, queries: torch.Tensor):
        """queries: (..., E) -> (p_mem (..., V), lam (..., 1), max_sim (..., 1)).

        V4.2: queries first pass through the learned query projection, then
        both keys and queries are centered by the mean stored key and the top
        principal components are projected out before cosine similarity.
        GPT-2 hidden states are strongly anisotropic (unrelated contexts can
        exceed 0.9 raw cosine); the correction removes the shared component
        so the confidence gate separates cleanly, while exact context
        prefixes still score ~1.0.
        """
        lead = queries.shape[:-1]
        q_raw = queries.reshape(-1, self.embed_dim).to(self.keys_raw.device)
        n = q_raw.shape[0]
        if len(self) == 0:
            zeros_v = torch.zeros(n, self.vocab_size, device=q_raw.device)
            zeros_1 = torch.zeros(n, 1, device=q_raw.device)
            return (
                zeros_v.reshape(*lead, self.vocab_size),
                zeros_1.reshape(*lead, 1),
                zeros_1.reshape(*lead, 1),
            )

        q_raw = self.query_proj(q_raw)          # V4.2 fuzzy alignment (queries only)
        mu, V_sub = self.read_space()
        k = self.to_read_space(self.keys_raw, mu, V_sub)
        q = self.to_read_space(q_raw, mu, V_sub)

        sims = q @ k.T                                        # (N, M)
        kk = min(self.top_k, len(self))
        top_sims, top_idx = sims.topk(kk, dim=-1)             # (N, kk)
        w = F.softmax(top_sims / self.read_temp, dim=-1)      # (N, kk)
        p_mem = torch.zeros(n, self.vocab_size, device=q_raw.device, dtype=w.dtype)
        p_mem.scatter_add_(1, self.values[top_idx], w)
        max_sim = top_sims[:, :1]                             # (N, 1)
        lam = self.lam_max * torch.sigmoid(
            self.gate_sharpness * (max_sim - self.gate_threshold)
        )
        return (
            p_mem.reshape(*lead, self.vocab_size),
            lam.reshape(*lead, 1),
            max_sim.reshape(*lead, 1),
        )
