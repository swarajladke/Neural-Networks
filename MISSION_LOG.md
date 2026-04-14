## Hebbian Update Optimizations for AGNIS V2

### Summary
This note focuses on vectorizing Hebbian updates in `enhanced_agnis_v2.py` to reduce Python-loop overhead while preventing weight drift, and outlines practical data-structure changes that keep the local-learning flavor intact.

### Proposed Architectural Optimizations
1. **Batch Hebbian updates by target neuron**
   - In `learn()`, build a batched tensor of incoming activations for each target neuron with an error signal, then compute correlations in a single matmul.
   - Use a packed layout: `pre_acts` shape `[K, D]`, `post_act` shape `[D]` -> correlation via `(pre_acts * post_act).mean(dim=1)`.
   - Apply updates to the subset of connections for that target in a single loop over that neuron instead of loop-over-all-neurons.

2. **Sparse “active edge” cache**
   - Maintain per-step a list of active connections (edges where both pre and post activations are above a small threshold).
   - Hebbian updates run only on this active edge list, lowering work and reducing noisy drift.
   - Cache can be built from `incoming` adjacency without global scans.

3. **Vectorized delta update for per-task adapters**
   - For connections with a task delta: stack A and B for a group of edges with same `(dim, rank)` and update via `A += noise * corr` and `B += noise * corr` using broadcasted tensors.
   - Use a shared noise tensor per batch to reduce randomness variance (reduces drift).

4. **Correlation clipping and EMA**
   - Replace raw correlation with `corr = clamp(ema_corr, -c, c)` where `ema_corr = 0.9 * ema_corr + 0.1 * corr`.
   - This prevents single-step spikes from destabilizing weights.

5. **Noise schedule tied to consolidation**
   - Scale the update noise by `(1 - consolidation)` and by a global annealing factor that decays over steps.
   - Keeps learning early, stabilizes later without freezing completely.

6. **Move from per-edge random noise to low-rank perturbation**
   - Instead of `randn_like(A)` per edge, sample one low-rank noise basis per batch and scale by correlation.
   - Dramatically reduces high-frequency drift while keeping stochastic exploration.

### Minimal Code-Change Sketch (Conceptual)
1. Collect `targets_with_error`.
2. For each target:
   - Build `pre_acts` tensor from incoming sources.
   - Compute `corrs` vector.
   - Apply vectorized updates to `A`/`B` for edges that have task deltas; apply base update for the rest.

### Expected Impact
1. 2-5x speed-up in Hebbian update path (less Python overhead).
2. Lower drift due to correlation EMA and shared noise basis.
3. Better scaling for Phase 29+ due to active-edge filtering.

[CODEX -> ANTIGRAVITY]
