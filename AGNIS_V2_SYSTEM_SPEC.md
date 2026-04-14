# AGNIS V2 System Spec (Enhanced)

This document describes the full architecture, data flow, and diagnostics for the current `enhanced_agnis_v2.py` implementation.

**Scope**
- Continual learning with task inference.
- Dynamic neuron growth with per‑task low‑rank deltas.
- Replay with self‑distillation and surprise prioritization.
- Salience and intrinsic motivation.
- Schema clustering, schema‑aware gating, and adaptive interference control.
- World‑model imagination and curriculum scheduling.
- Diagnostics and export artifacts.

**System Diagram**
```text
Inputs -> TaskInference -> TaskID
                 |             |
                 |             v
                 |      Task Capacity + Gates
                 |             |
                 v             v
            Forward Pass -> Outputs
                 |             |
                 |             v
                 |         Loss + Error
                 |             |
                 v             v
WorldModel <--- Learn Loop <- Salience + Intrinsic
     |                 |            |
     |                 v            v
     |         Replay + Surprise  Self‑Model
     |                 |
     v                 v
Imagination         Schema Engine
     |                 |
     v                 v
Counterfactuals   Gate Tighten/Relax
```

**Core Modules**
- `TaskInference`  
  Mean‑stat embedding, prototype matching, and task ID creation.
- `EnhancedNeuronV2` / `EnhancedConnectionV2`  
  Local memory, Hebbian traces, low‑rank per‑task deltas, consolidation.
- `ReplayBuffer` + `SurpriseBuffer`  
  Self‑distillation replay and high‑salience capture.
- `SalienceEngine`  
  Surprise + progress → per‑step learning weight.
- `IntrinsicMotivationEngine`  
  Novelty, uncertainty, progress, boredom → intrinsic weight.
- `AbstraxEngine`  
  Schema clustering and task folding based on prototype similarity.
- `WorldModelLite`  
  Lightweight internal model for imagination and counterfactuals.
- `SelfModelGraph`  
  Identity anchors and trait tracking (stability, plasticity, curiosity).

**Learning Loop (Simplified)**
1. Infer task ID from input embedding.
2. Ensure task‑specific capacity exists and activate gate mask.
3. Forward pass through gated network.
4. Compute error and loss.
5. Update world model.
6. Compute salience and intrinsic weights.
7. Apply local error propagation and Hebbian updates.
8. Store sample in replay and surprise buffers.
9. Periodically replay and consolidate.
10. Schema‑level diagnostics and gate adjustments.

**Replay Policy**
- Mix of:
  - Surprise samples (priority).
  - Replay buffer samples (long‑term retention).
  - Imagined samples (world model perturbations).
- Schema‑aware weighting:
  - Older tasks up‑weighted.
  - Larger schemas get a mild bonus.
  - Under‑replayed schemas get balancing weight.
  - Dominant schemas get dampening.

**Schema Dynamics**
- Tasks clustered into schemas via cosine similarity.
- Gate masks include shared neurons plus schema‑neighbor neurons.
- Interference detection tightens gates for unstable schemas.
- Stability detection relaxes gates to re‑enable transfer.

**Curriculum (Self‑Directed Mode)**
- Task selection uses novelty, uncertainty, competence, boredom, schema size, and recency.
- Enables internal, adaptive ordering of tasks.

**Diagnostics and Artifacts**
- `enhanced_agnis_v2_retention.png` (task‑level)
- `agnis_v2_schema_retention.png` (schema‑level)
- `agnis_v2_schema_transfer.png` (within vs cross schema)
- `agnis_v2_schema_replay.png` (replay balance)
- `agnis_v2_schema_gate_sharing.png` (gate overlap)
- `agnis_v2_schema_stability_events.png` (tighten/relax counts)
- `agnis_v2_schema_salience.png`
- `agnis_v2_schema_curiosity.png`
- `agnis_v2_schema_boredom.png`
- `agnis_v2_schema_progress.png`
- `agnis_v2_schema_uncertainty_trend.png`
- `agnis_v2_schema_competition.png`
- `agnis_v2_schema_dampening.png`
- `agnis_v2_schema_metrics.csv`
- `agnis_v2_run_summary.json`

**Key Configuration Knobs**
- `shared_gate_ratio`  
  Shared neuron exposure for transfer.
- `schema_gate_ratio`  
  Cross‑schema neuron sharing.
- `schema_replay_balance_strength`  
  Replay balancing based on schema coverage.
- `schema_competition_balance_strength`  
  Dampening of dominant schemas.
- `world_model_enabled` / `world_model_lr`  
  Internal simulation and imagination strength.
- `intrinsic_enabled`  
  Curiosity‑driven weighting.

**Recommended Usage**
- Standard continual training: `self_directed=False`.
- Self‑directed curriculum: `self_directed=True` and set cycle count.
- Run diagnostics after training to inspect schema health and transfer.

**Notes**
- Task embeddings are mean‑based; stronger embeddings improve schema quality.
- World model is linear by design for stability and speed.
- Diagnostics are extensive and can be trimmed if runtime cost is high.
