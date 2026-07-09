"""
fact_memory.py — V4.1 Episodic Key-Value Fact Memory
=====================================================
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

No gradients, no training loop: writing is one forward pass per fact.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicFactMemory(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        vocab_size: int = 50257,
        top_k: int = 8,
        read_temp: float = 0.03,
        gate_threshold: float = 0.90,
        gate_sharpness: float = 60.0,
        lam_max: float = 0.95,
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
        dev = torch.device(device)
        self.register_buffer("keys_raw", torch.empty(0, embed_dim, device=dev))
        self.register_buffer("values", torch.empty(0, dtype=torch.long, device=dev))

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

    @torch.no_grad()
    def read(self, queries: torch.Tensor):
        """queries: (..., E) -> (p_mem (..., V), lam (..., 1), max_sim (..., 1)).

        Both keys and queries are centered by the mean stored key before
        cosine similarity. GPT-2 hidden states are strongly anisotropic
        (unrelated contexts can exceed 0.9 raw cosine); centering removes
        the shared component so the confidence gate separates cleanly,
        while exact context prefixes still score ~1.0.
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

        mu = self.keys_raw.mean(dim=0, keepdim=True)
        k = F.normalize(self.keys_raw - mu, dim=-1)
        q = F.normalize(q_raw - mu, dim=-1)
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
