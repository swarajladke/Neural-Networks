"""
fact_memory.py — V4.2b Episodic Key-Value Fact Memory + Fuzzy Query Projection
===============================================================================
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

V4.2b — Average Gating:
  Adds optional causal average pooling (pool_len) over the last pool_len
  positions of both keys and queries, reducing sensitivity to the exact
  tail token of a paraphrase (e.g. "Exactly" vs "exactly"). Default
  pool_len=1 is a strict no-op, so the V4.1 pipeline and previously saved
  checkpoints keep their original behavior. Write-side pooling must be
  applied to the FULL hidden sequence before slicing answer positions —
  see pool_sequence() docstring.
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


class JointSlowMemoryMLP(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=50257, hidden_dim=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        self.logits_head = nn.Linear(hidden_dim, vocab_size)
        self.gate_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h = self.shared(x)
        logits = self.logits_head(h)
        gate = self.gate_head(h)
        return logits, gate


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
        pool_len: int = 1,
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
        self.pool_len = pool_len
        # V4.3: (mu, V_sub) cache — recomputing a full SVD per read() call is
        # prohibitive at scale. Invalidated on every write().
        self._space_cache = None
        dev = torch.device(device)
        self.register_buffer("keys_raw", torch.empty(0, embed_dim, device=dev))
        self.register_buffer("values", torch.empty(0, dtype=torch.long, device=dev))
        # V4.2: learned fuzzy query alignment (identity until trained).
        self.query_proj = ResidualQueryProjection(embed_dim, proj_hidden).to(dev)
        # Horizon A: Consolidated Memory MLP
        self.slow_mlp = None

    def __len__(self) -> int:
        return self.keys_raw.shape[0]

    @torch.no_grad()
    def write(self, hidden_states: torch.Tensor, next_token_ids: torch.Tensor) -> None:
        """Store (context -> next token) pairs.

        hidden_states : (T, E) GPT-2 final hidden states for positions 0..T-1
        next_token_ids: (T,)  the token observed after each position

        V4.2b: when pool_len > 1, the caller must pass hidden states that were
        pooled with pool_sequence() over the FULL sequence BEFORE slicing, so
        boundary keys include their prompt-side neighbor exactly like
        read-time queries at the same position do.
        """
        dev = self.keys_raw.device
        self.keys_raw = torch.cat([self.keys_raw, hidden_states.to(dev)], dim=0)
        self.values = torch.cat([self.values, next_token_ids.to(dev).long()], dim=0)
        self._space_cache = None  # V4.3: key set changed -> recompute space

    def pool_sequence(self, h: torch.Tensor) -> torch.Tensor:
        """Causal average pooling along the time dim (dim -2).

        h_pool[t] = mean(h[max(0, t - pool_len + 1) .. t]) — each position
        averages itself with up to pool_len - 1 predecessors, reducing
        sensitivity to the exact tail token. Position 0 is unchanged.

        V4.2b CRITICAL: at write time this must be applied to the FULL hidden
        sequence BEFORE slicing answer positions. Pooling only the stored
        slice would leave the boundary key without its prompt-side neighbor
        while read-time queries at the boundary DO average theirs — a
        systematic key/query mismatch at exactly the position that gates
        first-token retrieval. pool_len=1 is a strict no-op.
        """
        if self.pool_len <= 1 or h.dim() < 2 or h.shape[-2] < 2:
            return h
        pooled = h.clone()
        for t in range(1, h.shape[-2]):
            s = max(0, t - self.pool_len + 1)
            pooled[..., t, :] = h[..., s : t + 1, :].mean(dim=-2)
        return pooled

    def read_space(self):
        """Return (mu, V_sub): the anisotropy correction of the stored keys.

        mu    : (1, E) mean stored key (centering).
        V_sub : (E, npc) top principal components to project out, or None
                when projection-out is inactive.

        The V4.2 contrastive trainer MUST use this exact space so the learned
        geometry matches what the confidence gate sees at read time.

        V4.3: the result is cached until the next write(). Above 4096 keys a
        randomized low-rank PCA (torch.pca_lowrank) replaces the full SVD;
        only the top npc_project components are needed. If buffers are
        swapped manually (e.g. load_state_dict), reset _space_cache = None.
        """
        if self._space_cache is not None:
            return self._space_cache
        with torch.no_grad():
            mu = self.keys_raw.mean(dim=0, keepdim=True)
            centered = self.keys_raw - mu
            V_sub = None
            if self.npc_project > 0 and len(self) > self.npc_project:
                if len(self) > 4096:
                    _, _, V = torch.pca_lowrank(
                        centered,
                        q=min(self.npc_project + 6, centered.shape[1]),
                        center=False,
                        niter=4,
                    )
                else:
                    _, _, V = torch.svd(centered)
                V_sub = V[:, : self.npc_project]
        self._space_cache = (mu, V_sub)
        return self._space_cache

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

        V4.2b: when pool_len > 1 and queries carry a time dim (dim -2), each
        position is causally average-pooled with its predecessors before the
        query projection, mirroring the pooled keys.

        V4.2: queries pass through the learned query projection, then both
        keys and queries are centered by the mean stored key and the top
        principal components are projected out before cosine similarity.
        GPT-2 hidden states are strongly anisotropic (unrelated contexts can
        exceed 0.9 raw cosine); the correction removes the shared component
        so the confidence gate separates cleanly, while exact context
        prefixes still score ~1.0.
        """
        queries = self.pool_sequence(queries)   # V4.2b tail smoothing (no-op if pool_len=1)
        lead = queries.shape[:-1]
        q_raw = queries.reshape(-1, self.embed_dim)
        n = q_raw.shape[0]

        if self.slow_mlp is not None:
            # Horizon A: Consolidated Memory Mode (cortex-only)
            dev = next(self.slow_mlp.parameters()).device
            q_raw = q_raw.to(dev)
            q_raw = self.query_proj(q_raw)
            mu, V_sub = self.read_space()
            q = self.to_read_space(q_raw, mu, V_sub)
            logits_mem, sim_val = self.slow_mlp(q)
            p_mem = F.softmax(logits_mem / self.read_temp, dim=-1)
            lam = self.lam_max * torch.sigmoid(
                self.gate_sharpness * (sim_val - self.gate_threshold)
            )
            return (
                p_mem.reshape(*lead, self.vocab_size),
                lam.reshape(*lead, 1),
                sim_val.reshape(*lead, 1),
            )

        q_raw = q_raw.to(self.keys_raw.device)
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

    def consolidate(self, q_fact: torch.Tensor, pos_idx: torch.Tensor, q_ctrl: torch.Tensor, epochs: int = 200) -> None:
        """Consolidate the episodic memory database into a joint MLP (logits + gate).
        Once trained, the database is deleted and inference runs cortical-only."""
        dev = self.keys_raw.device
        mu, V_sub = self.read_space()
        k_read = self.to_read_space(self.keys_raw, mu, V_sub).detach()
        
        # Build inputs and targets
        train_inputs = []
        target_tokens = []
        target_sims = []
        
        # Compute exact similarity targets using the episodic path
        with torch.no_grad():
            q_fact_read = self.to_read_space(self.query_proj(q_fact), mu, V_sub)
            sims_fact = q_fact_read @ k_read.T
            max_sims_fact = sims_fact.max(dim=-1).values
            
            q_ctrl_read = self.to_read_space(self.query_proj(q_ctrl), mu, V_sub)
            sims_ctrl = q_ctrl_read @ k_read.T
            max_sims_ctrl = sims_ctrl.max(dim=-1).values
            
        for i in range(q_fact_read.shape[0]):
            train_inputs.append(q_fact_read[i])
            target_tokens.append(self.values[pos_idx[i]].item())
            target_sims.append(max_sims_fact[i].item())
            
        for i in range(q_ctrl_read.shape[0]):
            train_inputs.append(q_ctrl_read[i])
            target_tokens.append(0) # dummy
            target_sims.append(max_sims_ctrl[i].item())
            
        train_inputs = torch.stack(train_inputs).to(dev)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long, device=dev)
        target_sims = torch.tensor(target_sims, dtype=torch.float, device=dev).unsqueeze(-1)
        
        # Add raw keys
        self_sims = (k_read * k_read).sum(dim=-1, keepdim=True)
        train_inputs = torch.cat([train_inputs, k_read], dim=0)
        target_tokens = torch.cat([target_tokens, self.values], dim=0)
        target_sims = torch.cat([target_sims, self_sims], dim=0)
        
        # Initialize and train MLP
        self.slow_mlp = JointSlowMemoryMLP(vocab_size=self.vocab_size).to(dev)
        optimizer = torch.optim.AdamW(self.slow_mlp.parameters(), lr=1e-3, weight_decay=0.01)
        self.slow_mlp.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, sim_pred = self.slow_mlp(train_inputs)
            fact_mask = (target_sims > 0.5).squeeze(-1)
            loss_ce = F.cross_entropy(logits[fact_mask], target_tokens[fact_mask])
            loss_gate = F.mse_loss(sim_pred, target_sims)
            loss = loss_ce + 10.0 * loss_gate
            loss.backward()
            optimizer.step()
        self.slow_mlp.eval()
        
        # Evict episodic list keys and values to free memory!
        self.keys_raw = torch.empty(0, self.embed_dim, device=dev)
        self.values = torch.empty(0, dtype=torch.long, device=dev)

