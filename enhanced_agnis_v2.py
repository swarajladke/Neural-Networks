"""
AGNIS ENHANCED V2: Continual Learning Architecture

Additions over enhanced_agnis.py:
1. Task inference module (no hard task boundaries required)
2. Selective task deltas (create per-task capacity only when needed)
3. Replay buffer with self-distillation
4. Resource budgeting for neurogenesis
"""

from __future__ import annotations
import torch
import numpy as np
import random
import heapq
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

@dataclass
class Experience:
    task_id: int
    x: torch.Tensor
    y: torch.Tensor
    teacher: torch.Tensor
    surprise: float
    salience: float
    step: int


class SalienceEngine:
    """
    Computes salience weights from surprise and learning progress.
    """

    def __init__(
        self,
        ema: float = 0.95,
        min_weight: float = 0.2,
        max_weight: float = 2.5,
        surprise_scale: float = 1.0,
        progress_scale: float = 1.0
    ):
        self.ema = ema
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.surprise_scale = surprise_scale
        self.progress_scale = progress_scale
        self.loss_ema: Optional[float] = None

    def compute(self, loss: float, y: torch.Tensor, output: torch.Tensor) -> Tuple[float, float, float]:
        surprise = torch.abs(y - output).mean().item()
        if self.loss_ema is None:
            progress = 0.0
            self.loss_ema = loss
        else:
            progress = max(0.0, self.loss_ema - loss)
            self.loss_ema = self.ema * self.loss_ema + (1 - self.ema) * loss

        salience_raw = self.surprise_scale * surprise + self.progress_scale * progress
        weight = 0.5 + salience_raw
        weight = max(self.min_weight, min(self.max_weight, weight))
        return weight, surprise, progress

class IntrinsicMotivationEngine:
    """
    Tracks novelty, progress, and boredom to modulate learning pressure.
    """

    def __init__(
        self,
        ema: float = 0.95,
        novelty_scale: float = 0.6,
        progress_scale: float = 0.6,
        uncertainty_scale: float = 0.4,
        boredom_scale: float = 0.6,
        boredom_window: int = 50,
        min_weight: float = 0.4,
        max_weight: float = 2.5
    ):
        self.ema = ema
        self.novelty_scale = novelty_scale
        self.progress_scale = progress_scale
        self.uncertainty_scale = uncertainty_scale
        self.boredom_scale = boredom_scale
        self.boredom_window = max(10, boredom_window)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._state: Dict[int, Dict[str, float]] = defaultdict(lambda: {
            "loss_ema": None,
            "novelty_ema": 0.0,
            "boredom_counter": 0.0,
            "uncertainty_ema": 0.0
        })
        self.last_metrics = {
            "novelty": 0.0,
            "progress": 0.0,
            "boredom": 0.0,
            "uncertainty": 0.0
        }

    def compute(self, task_id: int, loss: float, novelty: float, uncertainty: float = 0.0) -> float:
        state = self._state[task_id]
        prev_loss = state["loss_ema"]
        if prev_loss is None:
            progress = 0.0
            state["loss_ema"] = loss
        else:
            progress = max(0.0, prev_loss - loss)
            state["loss_ema"] = self.ema * prev_loss + (1 - self.ema) * loss

        state["novelty_ema"] = self.ema * state["novelty_ema"] + (1 - self.ema) * novelty
        state["uncertainty_ema"] = self.ema * state["uncertainty_ema"] + (1 - self.ema) * uncertainty

        low_novelty = state["novelty_ema"] < 0.2
        low_progress = progress < 0.02
        if low_novelty and low_progress:
            state["boredom_counter"] = min(self.boredom_window, state["boredom_counter"] + 1)
        else:
            state["boredom_counter"] = max(0.0, state["boredom_counter"] - 1)

        boredom = state["boredom_counter"] / self.boredom_window
        weight = (
            1.0
            + self.novelty_scale * state["novelty_ema"]
            + self.progress_scale * progress
            + self.uncertainty_scale * uncertainty
            - self.boredom_scale * boredom
        )
        weight = max(self.min_weight, min(self.max_weight, weight))

        self.last_metrics = {
            "novelty": float(state["novelty_ema"]),
            "progress": float(progress),
            "boredom": float(boredom),
            "uncertainty": float(uncertainty)
        }
        return weight

class SurpriseBuffer:
    """
    Priority buffer for high-surprise/high-salience experiences.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._heap: List[Tuple[float, int, Experience]] = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, exp: Experience):
        score = exp.surprise * exp.salience
        item = (score, self._counter, exp)
        self._counter += 1

        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, item)
            return

        if self._heap and score > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def sample(self, k: int) -> List[Experience]:
        if not self._heap or k <= 0:
            return []
        items = [it[2] for it in self._heap]
        weights = [max(1e-6, it[0]) for it in self._heap]
        k = min(k, len(items))
        return random.choices(items, weights=weights, k=k)

class SelfModelGraph:
    """
    Lightweight self-model for identity anchors and competence estimates.
    """

    def __init__(self, dim: int = 8, ema: float = 0.98):
        self.dim = dim
        self.ema = ema
        self.identity_vector = torch.zeros(dim)
        self.prev_identity = torch.zeros(dim)
        self.self_pred_error = 0.0
        self.competence: Dict[int, float] = {}
        self.traits = {
            "stability": 0.5,
            "plasticity": 0.5,
            "curiosity": 0.5
        }
        self.loss_ema: Optional[float] = None

    def update(self, loss: float, surprise: float, active_ratio: float, task_id: int, total_tasks: int):
        features = torch.tensor([
            loss,
            surprise,
            active_ratio,
            float(total_tasks),
            float(task_id),
            self.traits["stability"],
            self.traits["plasticity"],
            self.traits["curiosity"]
        ], dtype=torch.float32)
        if self.dim > features.numel():
            pad = torch.zeros(self.dim - features.numel())
            features = torch.cat([features, pad], dim=0)
        elif self.dim < features.numel():
            features = features[: self.dim]

        self.prev_identity = self.identity_vector.clone()
        self.identity_vector = self.ema * self.identity_vector + (1 - self.ema) * features
        self.self_pred_error = float(((self.identity_vector - self.prev_identity) ** 2).mean().item())

        if self.loss_ema is None:
            progress = 0.0
            self.loss_ema = loss
        else:
            progress = self.loss_ema - loss
            self.loss_ema = self.ema * self.loss_ema + (1 - self.ema) * loss

        stability = self.traits["stability"]
        stability += 0.02 * max(0.0, progress) - 0.01 * surprise
        stability = max(0.0, min(1.0, stability))
        self.traits["stability"] = stability
        self.traits["plasticity"] = max(0.0, min(1.0, 1.0 - stability))
        curiosity = self.traits["curiosity"] + 0.01 * surprise - 0.01 * max(0.0, progress)
        self.traits["curiosity"] = max(0.0, min(1.0, curiosity))

        if task_id is not None:
            prev = self.competence.get(task_id, loss)
            self.competence[task_id] = self.ema * prev + (1 - self.ema) * loss


@dataclass
class EpisodicEvent:
    step: int
    task_id: int
    x: torch.Tensor
    y: torch.Tensor
    output: torch.Tensor
    surprise: float
    salience: float
    affect: Tuple[float, float]  # (valence, arousal)
    emb: torch.Tensor


class EpisodicMemory:
    """
    Event memory with time + affect tags for rapid binding and retrieval.
    """

    def __init__(self, maxlen: int = 2000):
        self.maxlen = max(100, maxlen)
        self._events: deque = deque(maxlen=self.maxlen)

    def add_event(self, event: EpisodicEvent):
        self._events.append(event)

    def sample(self, k: int) -> List[EpisodicEvent]:
        if not self._events or k <= 0:
            return []
        k = min(k, len(self._events))
        return random.sample(list(self._events), k=k)

    def sample_similar(self, query_emb: torch.Tensor, k: int) -> List[EpisodicEvent]:
        if not self._events or k <= 0:
            return []
        query = query_emb / (query_emb.norm() + 1e-8)
        scored = []
        for ev in self._events:
            emb = ev.emb / (ev.emb.norm() + 1e-8)
            sim = float((query * emb).sum().item())
            scored.append((sim, ev))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ev for _sim, ev in scored[: min(k, len(scored))]]

    def sample_similar_affect(
        self,
        query_emb: torch.Tensor,
        affect: Tuple[float, float],
        k: int,
        affect_weight: float = 0.25
    ) -> List[EpisodicEvent]:
        if not self._events or k <= 0:
            return []
        query = query_emb / (query_emb.norm() + 1e-8)
        scored = []
        for ev in self._events:
            emb = ev.emb / (ev.emb.norm() + 1e-8)
            sim = float((query * emb).sum().item())
            dist = abs(ev.affect[0] - affect[0]) + abs(ev.affect[1] - affect[1])
            aff_sim = 1.0 / (1.0 + dist)
            scored.append((sim + affect_weight * aff_sim, ev))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ev for _sim, ev in scored[: min(k, len(scored))]]

    def __len__(self) -> int:
        return len(self._events)


class GoalGenerator:
    """
    Generates self-directed goals from intrinsic metrics and self-state.
    """

    def __init__(self):
        self.last_goal = None

    def update(self, intrinsic_metrics: Dict[str, float], self_state: torch.Tensor):
        # Placeholder: integrate drives and intrinsic metrics into goal proposals.
        self.last_goal = {
            "novelty": intrinsic_metrics.get("novelty", 0.0),
            "uncertainty": intrinsic_metrics.get("uncertainty", 0.0),
            "boredom": intrinsic_metrics.get("boredom", 0.0)
        }

    def propose(self):
        return self.last_goal


class MetaController:
    """
    Meta-cognitive controller for dynamic learning modulation.
    """

    def compute_modulators(
        self,
        loss: float,
        uncertainty: float,
        novelty: float,
        boredom: float,
        self_pred_error: float = 0.0
    ) -> Dict[str, float]:
        # Simple heuristic: upweight learning when uncertainty/novelty are high,
        # downweight when boredom is high.
        error_scale = 1.0 + 0.5 * uncertainty + 0.3 * novelty - 0.4 * boredom + 0.2 * self_pred_error
        replay_scale = 1.0 + 0.6 * uncertainty + 0.2 * novelty + 0.3 * boredom
        error_scale = max(0.5, min(1.8, error_scale))
        replay_scale = max(0.5, min(2.0, replay_scale))
        return {
            "error_scale": error_scale,
            "replay_scale": replay_scale,
            "lr_scale": error_scale
        }


class ProceduralSkillLibrary:
    """
    Stores skill abstractions and habit policies.
    """

    def __init__(self):
        self.skills: Dict[int, Dict] = {}

    def update_skill(self, task_id: int, stats: Dict):
        prev = self.skills.get(task_id, {})
        ema = prev.get("loss_ema", stats.get("intrinsic_progress", 0.0))
        loss_val = stats.get("intrinsic_progress", 0.0)
        loss_ema = 0.9 * ema + 0.1 * loss_val
        habit = loss_ema < 0.05
        self.skills[task_id] = {
            "loss_ema": loss_ema,
            "habit": habit,
            "last_stats": stats
        }


class SemanticSchemaLearner:
    """
    Slow-learning semantic schema builder (placeholder hooks).
    """

    def __init__(self, sim_threshold: float = 0.85):
        self.sim_threshold = sim_threshold
        self.schema_embeddings: Dict[int, torch.Tensor] = {}
        self.schema_graph: Dict[int, List[int]] = {}

    def _cos_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return float((a_norm * b_norm).sum().item())

    def update(self, schema_clusters: Dict[int, List[int]], task_prototypes: Dict[int, torch.Tensor]):
        # Build schema embeddings as mean of task prototypes
        self.schema_embeddings = {}
        for sid, tasks in schema_clusters.items():
            protos = [task_prototypes[tid] for tid in tasks if tid in task_prototypes]
            if not protos:
                continue
            self.schema_embeddings[sid] = torch.stack(protos, dim=0).mean(dim=0)

        # Build schema similarity graph
        self.schema_graph = {sid: [] for sid in self.schema_embeddings.keys()}
        ids = list(self.schema_embeddings.keys())
        for i, sid in enumerate(ids):
            for j in range(i + 1, len(ids)):
                oid = ids[j]
                sim = self._cos_sim(self.schema_embeddings[sid], self.schema_embeddings[oid])
                if sim >= self.sim_threshold:
                    self.schema_graph[sid].append(oid)
                    self.schema_graph[oid].append(sid)


class PredictiveProcessing:
    """
    General linear predictor used for unsupervised reconstruction and physics-like consistency.
    """

    def __init__(self, input_dim: int, lr: float = 1e-3, name: str = "predictive"):
        self.W = torch.randn(input_dim, input_dim) * 0.01
        self.b = torch.zeros(input_dim)
        self.lr = lr
        self.last_loss = 0.0
        self.name = name

    def update(self, x: torch.Tensor) -> float:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pred = x @ self.W + self.b
        err = pred - x
        loss = (err ** 2).mean().item()
        grad_W = x.t() @ (err / max(1, x.shape[0]))
        grad_b = err.mean(dim=0)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        self.last_loss = loss
        return loss


class UnsupervisedPredictor(PredictiveProcessing):
    """
    Backwards-compatible wrapper for label-free reconstruction.
    """

    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__(input_dim=input_dim, lr=lr, name="unsupervised")


class FewShotMemory:
    """
    Fast binding memory for few-shot retrieval.
    """

    def __init__(self, max_per_task: int = 50):
        self.max_per_task = max(5, max_per_task)
        self.storage: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_per_task))

    def add(self, task_id: int, x: torch.Tensor, y: torch.Tensor):
        self.storage[task_id].append((x.detach().clone(), y.detach().clone()))

    def predict(self, task_id: int, x: torch.Tensor, k: int = 3) -> Optional[torch.Tensor]:
        if task_id not in self.storage or not self.storage[task_id]:
            return None
        items = list(self.storage[task_id])
        if x.dim() == 1:
            x = x.unsqueeze(0)
        xq = x.mean(dim=0)
        scored = []
        for xi, yi in items:
            xi_vec = xi.mean(dim=0) if xi.dim() > 1 else xi
            dist = torch.norm(xq - xi_vec).item()
            scored.append((dist, yi))
        scored.sort(key=lambda t: t[0])
        top = [y for _d, y in scored[: max(1, min(k, len(scored)))]]
        return torch.stack(top, dim=0).mean(dim=0)


class RewardComposer:
    """
    Composes a multi-component reward beyond scalar loss.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "surprise": 0.3,
            "novelty": 0.2,
            "uncertainty": 0.2,
            "progress": 0.2,
            "boredom": -0.3
        }

    def compute(self, metrics: Dict[str, float]) -> float:
        reward = 0.0
        for k, w in self.weights.items():
            reward += w * metrics.get(k, 0.0)
        return reward


class PlannerLite:
    """
    Lightweight counterfactual evaluator using the world model.
    """

    def __init__(self, world_model: WorldModelLite):
        self.world_model = world_model

    def counterfactual_score(self, x: torch.Tensor, task_id: int, samples: int = 4) -> float:
        return self.world_model.counterfactual_variance(x, task_id, num=samples, noise=0.05)


class SocialReasoner:
    """
    Minimal theory-of-mind placeholder: tracks belief-like state.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.last_belief = torch.zeros(dim)

    def infer_belief(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            x = x.mean(dim=0)
        if x.numel() >= self.dim:
            belief = x[: self.dim].detach().clone()
        else:
            pad = torch.zeros(self.dim - x.numel())
            belief = torch.cat([x.detach().clone(), pad], dim=0)
        self.last_belief = belief
        return belief


class PhysicalIntuitionModel(PredictiveProcessing):
    """
    Backwards-compatible wrapper for physics-style predictive processing.
    """

    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__(input_dim=input_dim, lr=lr, name="physics")


class ValueAlignmentModule:
    """
    Simple value constraint adapter for reward shaping.
    """

    def __init__(self, penalty_weight: float = 0.2):
        self.penalty_weight = penalty_weight

    def adjust(self, reward: float, affect: Tuple[float, float]) -> float:
        # Penalize high arousal + negative valence as a proxy for misalignment risk.
        valence, arousal = affect
        penalty = max(0.0, -valence) * arousal * self.penalty_weight
        return reward - penalty


class ActionPerceptionLoop:
    """
    Minimal action-perception loop state tracker.
    """

    def __init__(self, dim: int = 8):
        self.state = torch.zeros(dim)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            x = x.mean(dim=0)
        if x.numel() >= self.state.numel():
            self.state = x[: self.state.numel()].detach().clone()
        else:
            pad = torch.zeros(self.state.numel() - x.numel())
            self.state = torch.cat([x.detach().clone(), pad], dim=0)
        return self.state


class SelfAudit:
    """
    Detects instability and suggests replay boosts.
    """

    def __init__(self, window: int = 20, spike_ratio: float = 1.5):
        self.window = max(5, window)
        self.spike_ratio = max(1.1, spike_ratio)
        self.history = deque(maxlen=self.window)

    def update(self, loss: float) -> float:
        self.history.append(loss)
        if len(self.history) < self.window:
            return 0.0
        avg = sum(self.history) / len(self.history)
        if avg > 0 and loss > self.spike_ratio * avg:
            return 0.3
        return 0.0


class ValueAlignmentLearner:
    """
    Learns a lightweight penalty model from affect signals.
    """

    def __init__(self, ema: float = 0.98, base_penalty: float = 0.2):
        self.ema = ema
        self.base_penalty = base_penalty
        self.neg_valence_ema = 0.0
        self.arousal_ema = 0.0
        self.risk_ema = 0.0

    def update(self, affect: Tuple[float, float], reward: float) -> float:
        valence, arousal = affect
        neg_valence = max(0.0, -valence)
        risk = neg_valence * max(0.0, arousal)

        self.neg_valence_ema = self.ema * self.neg_valence_ema + (1 - self.ema) * neg_valence
        self.arousal_ema = self.ema * self.arousal_ema + (1 - self.ema) * max(0.0, arousal)
        self.risk_ema = self.ema * self.risk_ema + (1 - self.ema) * risk

        penalty_weight = self.base_penalty * (1.0 + 2.0 * self.risk_ema)
        penalty = penalty_weight * risk
        return reward - penalty


class LongTermPlanner:
    """
    Maintains a rolling plan over tasks using competence and intrinsic signals.
    """

    def __init__(self, horizon: int = 50, refresh_every: int = 25, bias_strength: float = 0.35):
        self.horizon = max(5, horizon)
        self.refresh_every = max(5, refresh_every)
        self.bias_strength = max(0.1, min(1.0, bias_strength))
        self._steps = 0
        self.target_task: Optional[int] = None
        self.remaining = 0

    def update(
        self,
        task_id: int,
        loss: float,
        competence: Dict[int, float],
        intrinsic_state: Dict[int, Dict[str, float]],
        goal: Optional[Dict[str, float]] = None
    ):
        self._steps += 1
        if self.remaining > 0:
            self.remaining -= 1

        if self._steps % self.refresh_every != 0 and self.remaining > 0:
            return

        if not competence:
            return

        scored = []
        for tid, comp in competence.items():
            state = intrinsic_state.get(tid)
            novelty = state["novelty_ema"] if state else 0.2
            uncertainty = state["uncertainty_ema"] if state else 0.2
            boredom = (state["boredom_counter"] / max(1, 50)) if state else 0.0
            goal_bias = 0.0
            if goal:
                goal_bias = 0.2 * goal.get("novelty", 0.0) * novelty
                goal_bias += 0.2 * goal.get("uncertainty", 0.0) * uncertainty
                goal_bias -= 0.2 * goal.get("boredom", 0.0) * boredom
            score = (1.0 / (1.0 + comp)) + 0.3 * uncertainty + 0.2 * novelty - 0.2 * boredom + goal_bias
            scored.append((score, tid))
        scored.sort(reverse=True)
        if scored:
            self.target_task = scored[0][1]
            self.remaining = self.horizon

    def bias(self, task_id: int) -> float:
        if self.target_task is None or self.remaining <= 0:
            return 1.0
        if task_id == self.target_task:
            return 1.0 + self.bias_strength
        return 1.0

    def current_plan(self) -> Optional[Dict]:
        if self.target_task is None or self.remaining <= 0:
            return None
        return {"target_task": self.target_task, "remaining": self.remaining}


class SelfRewriter:
    """
    Safe self-adjustment of internal hyperparameters.
    """

    def __init__(self, interval: int = 200):
        self.interval = max(50, interval)

    def step(self, agnis: "EnhancedAGNISV2", traits: Dict[str, float], loss: float, audit_signal: float = 0.0):
        if agnis.stats["total_steps"] % self.interval != 0:
            return

        stability = traits.get("stability", 0.5)
        plasticity = traits.get("plasticity", 0.5)
        curiosity = traits.get("curiosity", 0.5)

        agnis.hebb_noise_scale = max(0.01, min(0.2, agnis.hebb_noise_scale * (1.0 + 0.2 * plasticity - 0.1 * stability)))
        agnis.hebb_corr_ema = max(0.6, min(0.98, agnis.hebb_corr_ema * (1.0 + 0.1 * stability - 0.05 * plasticity)))

        agnis.salience_engine.min_weight = max(0.1, min(0.6, agnis.salience_engine.min_weight * (1.0 + 0.1 * stability)))
        agnis.salience_engine.max_weight = max(1.5, min(3.5, agnis.salience_engine.max_weight * (1.0 + 0.1 * curiosity)))

        agnis.intrinsic_engine.novelty_scale = max(0.2, min(1.2, agnis.intrinsic_engine.novelty_scale * (1.0 + 0.2 * curiosity)))
        agnis.intrinsic_engine.boredom_scale = max(0.2, min(1.2, agnis.intrinsic_engine.boredom_scale * (1.0 + 0.2 * stability)))

        if loss > 0.5:
            agnis.replay_every = max(50, int(agnis.replay_every * 0.9))
        else:
            agnis.replay_every = min(400, int(agnis.replay_every * 1.05))

        if audit_signal > 0.0:
            agnis.replay_every = max(20, int(agnis.replay_every * (1.0 - 0.15 * audit_signal)))
            agnis.salience_engine.ema = max(0.85, agnis.salience_engine.ema - 0.01 * (1.0 + audit_signal))
            agnis.intrinsic_engine.ema = max(0.85, agnis.intrinsic_engine.ema - 0.01 * (1.0 + audit_signal))
        else:
            agnis.salience_engine.ema = min(0.99, agnis.salience_engine.ema + 0.002)
            agnis.intrinsic_engine.ema = min(0.99, agnis.intrinsic_engine.ema + 0.002)
class AbstraxEngine:
    """
    Cross-domain affinity analysis and task folding into schema clusters.
    """

    def __init__(self, sim_threshold: float = 0.85):
        self.sim_threshold = sim_threshold
        self.schema_clusters: Dict[int, List[int]] = {}
        self.task_schema: Dict[int, int] = {}
        self.schema_prototypes: Dict[int, torch.Tensor] = {}
        self._next_schema_id = 0

    def _cos_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return float((a_norm * b_norm).sum().item())

    def recompute(self, prototypes: Dict[int, torch.Tensor]):
        self.schema_clusters = {}
        self.task_schema = {}
        self.schema_prototypes = {}
        self._next_schema_id = 0

        if not prototypes:
            return

        for tid in sorted(prototypes.keys()):
            proto = prototypes[tid].detach().clone()
            assigned = False
            best_schema = None
            best_sim = -1.0

            for sid, sproto in self.schema_prototypes.items():
                sim = self._cos_sim(proto, sproto)
                if sim > best_sim:
                    best_sim = sim
                    best_schema = sid

            if best_schema is not None and best_sim >= self.sim_threshold:
                self.schema_clusters[best_schema].append(tid)
                # Update schema prototype as running mean
                members = self.schema_clusters[best_schema]
                count = len(members)
                self.schema_prototypes[best_schema] = (self.schema_prototypes[best_schema] * (count - 1) + proto) / count
                self.task_schema[tid] = best_schema
                assigned = True

            if not assigned:
                sid = self._next_schema_id
                self._next_schema_id += 1
                self.schema_clusters[sid] = [tid]
                self.schema_prototypes[sid] = proto
                self.task_schema[tid] = sid

    def schema_size(self, task_id: int) -> int:
        sid = self.task_schema.get(task_id)
        if sid is None:
            return 1
        return len(self.schema_clusters.get(sid, []))

class WorldModelLite:
    """
    Lightweight linear world model for counterfactual rollouts.
    """

    def __init__(self, input_dim: int, output_dim: int, task_embed_dim: int = 8, lr: float = 1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_embed_dim = task_embed_dim
        self.lr = lr
        self.W = torch.randn(input_dim + task_embed_dim, output_dim) * 0.01
        self.b = torch.zeros(output_dim)
        self.updates = 0
        self.last_loss = 0.0
        self._task_embeds: Dict[int, torch.Tensor] = {}

    def _task_embed(self, task_id: int) -> torch.Tensor:
        if task_id in self._task_embeds:
            return self._task_embeds[task_id]
        rng = random.Random(2024 + int(task_id))
        vec = torch.tensor([rng.uniform(-1.0, 1.0) for _ in range(self.task_embed_dim)], dtype=torch.float32)
        self._task_embeds[task_id] = vec
        return vec

    def _augment(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        emb = self._task_embed(task_id).unsqueeze(0).expand(x.shape[0], -1)
        return torch.cat([x, emb], dim=1)

    def predict(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x_aug = self._augment(x, task_id)
        return x_aug @ self.W + self.b

    def update(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
        x_aug = self._augment(x, task_id)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        pred = x_aug @ self.W + self.b
        error = pred - y
        loss = (error ** 2).mean().item()
        grad_W = x_aug.t() @ (error / max(1, x_aug.shape[0]))
        grad_b = error.mean(dim=0)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        self.updates += 1
        self.last_loss = loss
        return loss

    def counterfactual_variance(self, x: torch.Tensor, task_id: int, num: int = 6, noise: float = 0.05) -> float:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        preds = []
        for _ in range(max(2, num)):
            jitter = torch.randn_like(x) * noise
            preds.append(self.predict(x + jitter, task_id))
        stacked = torch.stack(preds, dim=0)
        return float(stacked.var(dim=0).mean().item())

class TaskInference:
    """
    Lightweight task inference via input statistics embeddings.
    Maintains prototype vectors per task and assigns the closest.
    """

    def __init__(self, input_dim: int, embed_dim: int = 32, new_task_threshold: float = 1.5):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.new_task_threshold = new_task_threshold
        self.prototypes: Dict[int, torch.Tensor] = {}
        self.next_task_id = 0
        self.last_distance = 0.0
        self.last_is_new = False

        # Random projection matrix to keep it lightweight
        self.proj = torch.randn(input_dim * 2, embed_dim) * 0.1

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Pad or truncate to self.input_dim
        if x.shape[1] < self.input_dim:
            padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1])
            x = torch.cat([x, padding], dim=1)
        elif x.shape[1] > self.input_dim:
            x = x[:, :self.input_dim]

        # Fast Mean-only Embedding (Standard Deviation is too expensive for scaling)
        mean = x.mean(dim=0)
        feats = torch.cat([mean, mean], dim=0) # Duplicate for dimension consistency
        return feats @ self.proj

    def infer(self, x: torch.Tensor) -> int:
        emb = self._embed(x)
        self.last_is_new = False
        self.last_distance = 0.0
        if not self.prototypes:
            tid = self.next_task_id
            self.prototypes[tid] = emb.clone()
            self.next_task_id += 1
            self.last_is_new = True
            return tid

        # Find nearest prototype
        best_tid = None
        best_dist = float("inf")
        for tid, proto in self.prototypes.items():
            dist = torch.norm(emb - proto).item()
            if dist < best_dist:
                best_dist = dist
                best_tid = tid
        self.last_distance = best_dist

        # Create new task if far
        if best_dist > self.new_task_threshold:
            tid = self.next_task_id
            self.prototypes[tid] = emb.clone()
            self.next_task_id += 1
            self.last_is_new = True
            return tid

        return best_tid

    def update(self, x: torch.Tensor, task_id: int, momentum: float = 0.9):
        emb = self._embed(x)
        if task_id not in self.prototypes:
            self.prototypes[task_id] = emb.clone()
            self.next_task_id = max(self.next_task_id, task_id + 1)
            return
        self.prototypes[task_id] = momentum * self.prototypes[task_id] + (1 - momentum) * emb

class ReplayBuffer:
    """
    Per-task replay buffer storing (x, y, teacher_output).
    """

    def __init__(self, capacity_per_task: int = 200):
        self.capacity_per_task = capacity_per_task
        self.storage: Dict[int, deque] = defaultdict(lambda: deque(maxlen=capacity_per_task))
        self.prototypes: Dict[int, Dict[str, torch.Tensor]] = {}

    def add(self, task_id: int, x: torch.Tensor, y: torch.Tensor, teacher: torch.Tensor):
        self.storage[task_id].append((x.detach().clone(), y.detach().clone(), teacher.detach().clone()))
        self._update_prototype(task_id, x, y, teacher)

    def _update_prototype(self, task_id: int, x: torch.Tensor, y: torch.Tensor, teacher: torch.Tensor):
        x = x.detach().clone()
        y = y.detach().clone()
        teacher = teacher.detach().clone()
        if task_id not in self.prototypes:
            self.prototypes[task_id] = {
                "count": torch.tensor(1.0),
                "x_mean": x,
                "y_mean": y,
                "t_mean": teacher
            }
            return

        stats = self.prototypes[task_id]
        count = stats["count"] + 1.0
        stats["x_mean"] = stats["x_mean"] + (x - stats["x_mean"]) / count
        stats["y_mean"] = stats["y_mean"] + (y - stats["y_mean"]) / count
        stats["t_mean"] = stats["t_mean"] + (teacher - stats["t_mean"]) / count
        stats["count"] = count

    def sample(
        self,
        batch_size: int,
        task_ids: Optional[List[int]] = None,
        task_weights: Optional[List[float]] = None,
        force_prototype_tasks: Optional[List[int]] = None
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        items = []
        available_tasks = task_ids if task_ids is not None else list(self.storage.keys())
        if not available_tasks:
            return items

        for _ in range(batch_size):
            if task_weights is not None and len(task_weights) == len(available_tasks):
                tid = random.choices(available_tasks, weights=task_weights, k=1)[0]
            else:
                tid = random.choice(available_tasks)

            if force_prototype_tasks and tid in force_prototype_tasks and tid in self.prototypes:
                stats = self.prototypes[tid]
                items.append((tid, stats["x_mean"], stats["y_mean"], stats["t_mean"]))
                continue

            if len(self.storage[tid]) == 0:
                if tid in self.prototypes:
                    stats = self.prototypes[tid]
                    items.append((tid, stats["x_mean"], stats["y_mean"], stats["t_mean"]))
                continue

            x, y, t = random.choice(list(self.storage[tid]))
            items.append((tid, x, y, t))
        return items

class EnhancedNeuronV2:
    def __init__(self, neuron_id: int, dim: int = 32, task_id: Optional[int] = None):
        self.id = neuron_id
        self.dim = dim
        self.task_id = task_id

        self.activation = torch.zeros(dim)
        self.memory = torch.zeros(dim)

        self.incoming: Dict[int, "EnhancedConnectionV2"] = {}
        self.outgoing: Dict[int, "EnhancedConnectionV2"] = {}

        self.age = 0
        self.total_activation = 0.0
        self.activation_history = []
        self.task_usage = defaultdict(float)

        self.learning_rate = 0.02
        self.plasticity = 1.0
        self.consolidation = 0.0
        self.neuron_type = "general"

    def activate(self, input_signals: Dict[int, torch.Tensor], current_task_id: Optional[int] = None):
        total_input = torch.zeros(self.dim)
        
        # Optimized connection loop
        for source_id, connection in self.incoming.items():
            if source_id in input_signals:
                total_input += connection.forward(input_signals[source_id], current_task_id)

        if current_task_id is not None and current_task_id in self.task_usage:
            memory_gate = self.task_usage[current_task_id] / (sum(self.task_usage.values()) + 1e-6)
        else:
            memory_gate = 0.5
        total_input += memory_gate * self.memory

        self.activation = torch.tanh(total_input)

        self.age += 1
        activation_mag = self.activation.abs().mean().item()
        self.total_activation += activation_mag
        self.activation_history.append(activation_mag)
        if current_task_id is not None:
            self.task_usage[current_task_id] += activation_mag
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
        return self.activation

    def update_memory(self, error_signal: torch.Tensor):
        effective_lr = self.learning_rate * self.plasticity * (1 - self.consolidation)
        learning_signal = effective_lr * error_signal
        self.memory = 0.95 * self.memory + 0.05 * learning_signal
        self.plasticity *= 0.9999

    def consolidate(self, strength: float = 0.1):
        self.consolidation = min(1.0, self.consolidation + strength)

class EnhancedConnectionV2:
    def __init__(self, source_id: int, target_id: int, dim: int = 32, rank: int = 8):
        self.source_id = source_id
        self.target_id = target_id
        self.dim = dim
        self.rank = rank

        self.W_base = torch.randn(dim, dim) * 0.01
        self.task_deltas: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.strength = 1.0
        self.age = 0
        self.usage_per_task = defaultdict(int)
        self.consolidation = 0.0
        self.hebbian_trace = 0.0

    def ensure_task_delta(self, task_id: int):
        if task_id not in self.task_deltas:
            A = torch.randn(self.dim, self.rank) * 0.01
            B = torch.randn(self.rank, self.dim) * 0.01
            self.task_deltas[task_id] = (A, B)

    def forward(self, signal: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        # Ultra-Fast Forward Pass
        if task_id is not None and task_id in self.task_deltas:
            A, B = self.task_deltas[task_id]
            output = signal @ (self.W_base + A @ B)
        else:
            output = signal @ self.W_base
            
        output *= self.strength
        self.age += 1
        if task_id is not None:
            self.usage_per_task[task_id] += 1
        return output

    def hebbian_update_with_corr(
        self,
        corr: float,
        task_id: Optional[int] = None,
        noise_basis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        corr_ema: float = 0.9,
        corr_clip: float = 0.5,
        noise_scale: float = 0.05,
        allow_base_update: bool = True,
        lr_multiplier: float = 1.0
    ):
        lr = 0.001 * lr_multiplier * (1 - self.consolidation)
        if lr < 1e-6:
            return

        # Smooth and clip correlation to reduce drift
        self.hebbian_trace = corr_ema * self.hebbian_trace + (1 - corr_ema) * corr
        corr_val = max(-corr_clip, min(corr_clip, self.hebbian_trace))
        if abs(corr_val) < 1e-6:
            return

        if noise_basis is None:
            noise_A = torch.randn(self.dim, self.rank) * 0.01
            noise_B = torch.randn(self.rank, self.dim) * 0.01
        else:
            noise_A, noise_B = noise_basis

        if task_id is not None and task_id in self.task_deltas:
            A, B = self.task_deltas[task_id]
            A += noise_A * (corr_val * lr * noise_scale)
            B += noise_B * (corr_val * lr * noise_scale)
        elif allow_base_update and self.consolidation < 0.5:
            self.W_base += (noise_A @ noise_B) * (corr_val * lr * noise_scale)

        self.strength = max(0.1, min(2.0, self.strength + 0.001 * corr_val))

    def consolidate(self, strength: float = 0.1):
        self.consolidation = min(1.0, self.consolidation + strength)

class EnhancedAGNISV2:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_hidden: int = 40,
        neuron_dim: int = 12,
        max_new_neurons_per_task: int = 20,
        replay_capacity_per_task: int = 200,
        replay_every: int = 200,
        hebb_active_threshold: float = 0.01,
        hebb_corr_clip: float = 0.8,
        hebb_corr_ema: float = 0.8,
        hebb_noise_scale: float = 0.05,
        gating_enabled: bool = True,
        shared_gate_ratio: float = 0.25,
        freeze_base_after_tasks: int = 3,
        oracle_enabled: bool = True,
        oracle_window: int = 20,
        oracle_spike_ratio: float = 1.5,
        oracle_replay_burst: int = 3,
        prototype_only_after: int = 100,
        world_model_enabled: bool = True,
        world_model_lr: float = 1e-3,
        intrinsic_enabled: bool = True,
        mastery_mode_enabled: bool = False,
        mastery_loss_threshold: float = 0.15,
        mastery_lr_multiplier: float = 10.0,
        mastery_corr_ema_scale: float = 0.6,
        mastery_forward_steps: int = 3,
        schema_gate_ratio: float = 0.25,
        schema_replay_balance_strength: float = 0.3,
        schema_competition_balance_strength: float = 0.2,
        episodic_memory_enabled: bool = True,
        episodic_memory_size: int = 2000,
        meta_controller_enabled: bool = True,
        goal_generator_enabled: bool = True,
        semantic_schema_enabled: bool = True,
        skill_library_enabled: bool = True,
        self_pred_loss_weight: float = 0.1,
        unsupervised_enabled: bool = True,
        unsupervised_lr: float = 1e-3,
        physical_intuition_lr: float = 1e-3,
        few_shot_enabled: bool = True,
        few_shot_alpha: float = 0.2,
        few_shot_k: int = 3,
        reward_composer_enabled: bool = True,
        social_reasoning_enabled: bool = True,
        physical_intuition_enabled: bool = True,
        value_alignment_enabled: bool = True,
        value_alignment_learning_enabled: bool = True,
        action_perception_enabled: bool = True,
        long_term_planning_enabled: bool = True,
        long_term_planner_horizon: int = 50,
        long_term_planner_refresh: int = 25,
        self_audit_enabled: bool = True,
        self_rewriter_enabled: bool = True,
        self_rewriter_interval: int = 200
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_dim = neuron_dim

        self.neurons: Dict[int, EnhancedNeuronV2] = {}
        self.next_id = 0

        self.input_neurons: List[int] = []
        self.output_neurons: List[int] = []
        self.hidden_neurons: List[int] = []

        self.current_task_id: Optional[int] = None
        self.task_neuron_pools: Dict[int, List[int]] = {}
        self.shared_neurons: List[int] = []
        self.task_gate_masks: Dict[int, set] = {}
        self.active_neurons: set = set()

        self.stats = {
            "neurons_created": 0,
            "neurons_removed": 0,
            "connections_created": 0,
            "connections_removed": 0,
            "total_steps": 0
        }

        self.growth_enabled = True
        self.pruning_enabled = False

        self.max_new_neurons_per_task = max_new_neurons_per_task
        self.replay_every = replay_every
        self.replay_buffer = ReplayBuffer(capacity_per_task=replay_capacity_per_task)
        self.task_inference = TaskInference(input_dim=input_dim)
        self.hebb_active_threshold = hebb_active_threshold
        self.hebb_corr_clip = hebb_corr_clip
        self.hebb_corr_ema = hebb_corr_ema
        self.hebb_noise_scale = hebb_noise_scale
        self.gating_enabled = gating_enabled
        self.shared_gate_ratio = max(0.0, min(1.0, shared_gate_ratio))
        self.freeze_base_after_tasks = max(0, freeze_base_after_tasks)
        self.oracle_enabled = oracle_enabled
        self.oracle_window = max(5, oracle_window)
        self.oracle_spike_ratio = max(1.1, oracle_spike_ratio)
        self.oracle_replay_burst = max(1, oracle_replay_burst)
        self.prototype_only_after = max(0, prototype_only_after)
        self.loss_history = deque(maxlen=self.oracle_window)
        self.salience_engine = SalienceEngine()
        self.intrinsic_enabled = intrinsic_enabled
        self.intrinsic_engine = IntrinsicMotivationEngine()
        self.surprise_buffer = SurpriseBuffer(max_size=500)
        self.self_model = SelfModelGraph(dim=8)
        self.abstrax = AbstraxEngine(sim_threshold=0.85)
        self.world_model_enabled = world_model_enabled
        self.schema_gate_ratio = max(0.0, min(1.0, schema_gate_ratio))
        self.schema_replay_balance_strength = max(0.0, min(1.0, schema_replay_balance_strength))
        self.schema_competition_balance_strength = max(0.0, min(1.0, schema_competition_balance_strength))
        self.mastery_mode_enabled = mastery_mode_enabled
        self.mastery_loss_threshold = max(0.0, mastery_loss_threshold)
        self.mastery_lr_multiplier = max(1.0, mastery_lr_multiplier)
        self.mastery_corr_ema_scale = max(0.1, min(1.0, mastery_corr_ema_scale))
        self.mastery_forward_steps = max(1, mastery_forward_steps)
        self.world_model = WorldModelLite(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            task_embed_dim=8,
            lr=world_model_lr
        )
        self.curriculum_age: Dict[int, int] = defaultdict(int)
        self.replay_schema_counts: Dict[int, int] = defaultdict(int)
        self.replay_total_samples = 0
        self.schema_loss_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.oracle_window))
        self.schema_salience_sum: Dict[int, float] = defaultdict(float)
        self.schema_salience_count: Dict[int, int] = defaultdict(int)
        self.last_query_emb: Optional[torch.Tensor] = None
        self.episodic_memory_enabled = episodic_memory_enabled
        self.episodic_memory = EpisodicMemory(maxlen=episodic_memory_size)
        self.meta_controller_enabled = meta_controller_enabled
        self.meta_controller = MetaController()
        self.goal_generator_enabled = goal_generator_enabled
        self.goal_generator = GoalGenerator()
        self.semantic_schema_enabled = semantic_schema_enabled
        self.semantic_schema = SemanticSchemaLearner()
        self.skill_library_enabled = skill_library_enabled
        self.skill_library = ProceduralSkillLibrary()
        self.self_pred_loss_weight = max(0.0, self_pred_loss_weight)
        self.last_affect: Optional[Tuple[float, float]] = None
        self.unsupervised_enabled = unsupervised_enabled
        self.unsupervised_predictor = PredictiveProcessing(self.input_dim, lr=unsupervised_lr, name="unsupervised")
        self.few_shot_enabled = few_shot_enabled
        self.few_shot_alpha = max(0.0, min(1.0, few_shot_alpha))
        self.few_shot_k = max(1, few_shot_k)
        self.few_shot_memory = FewShotMemory()
        self.reward_composer_enabled = reward_composer_enabled
        self.reward_composer = RewardComposer()
        self.planner = PlannerLite(self.world_model)
        self.social_reasoning_enabled = social_reasoning_enabled
        self.social_reasoner = SocialReasoner()
        self.physical_intuition_enabled = physical_intuition_enabled
        self.physical_intuition = PredictiveProcessing(self.input_dim, lr=physical_intuition_lr, name="physics")
        self.value_alignment_enabled = value_alignment_enabled
        self.value_alignment = ValueAlignmentModule()
        self.value_alignment_learning_enabled = value_alignment_learning_enabled
        self.value_alignment_learner = ValueAlignmentLearner()
        self.action_perception_enabled = action_perception_enabled
        self.action_loop = ActionPerceptionLoop()
        self.long_term_planning_enabled = long_term_planning_enabled
        self.long_term_plan: Optional[Dict] = None
        self.long_term_planner = LongTermPlanner(
            horizon=long_term_planner_horizon,
            refresh_every=long_term_planner_refresh
        )
        self.self_audit_enabled = self_audit_enabled
        self.self_audit = SelfAudit(window=self.oracle_window, spike_ratio=self.oracle_spike_ratio)
        self.last_audit_signal = 0.0
        self.self_rewriter_enabled = self_rewriter_enabled
        self.self_rewriter = SelfRewriter(interval=self_rewriter_interval)

        self._initialize_network(initial_hidden)

    def _initialize_network(self, initial_hidden: int):
        for _ in range(self.input_dim):
            nid = self._add_neuron(neuron_type="input")
            self.input_neurons.append(nid)

        for _ in range(initial_hidden):
            nid = self._add_neuron(neuron_type="hidden")
            self.hidden_neurons.append(nid)
            self.shared_neurons.append(nid)

        for _ in range(self.output_dim):
            nid = self._add_neuron(neuron_type="output")
            self.output_neurons.append(nid)

        self._initialize_connections()

    def _initialize_connections(self):
        for inp_id in self.input_neurons:
            num_targets = min(20, len(self.hidden_neurons))
            targets = random.sample(self.hidden_neurons, num_targets)
            for target_id in targets:
                self._add_connection(inp_id, target_id)

        for h_id in self.hidden_neurons:
            if random.random() < 0.5:
                num_targets = min(5, len(self.hidden_neurons))
                targets = random.sample(self.hidden_neurons, num_targets)
                for target_id in targets:
                    if target_id != h_id:
                        self._add_connection(h_id, target_id)

        for out_id in self.output_neurons:
            for h_id in self.hidden_neurons:
                self._add_connection(h_id, out_id)

    def _add_neuron(self, neuron_type: str = "hidden", task_id: Optional[int] = None) -> int:
        nid = self.next_id
        self.next_id += 1
        neuron = EnhancedNeuronV2(nid, dim=self.neuron_dim, task_id=task_id)
        neuron.neuron_type = neuron_type
        self.neurons[nid] = neuron
        self.stats["neurons_created"] += 1
        return nid

    def _add_connection(self, source_id: int, target_id: int) -> bool:
        if source_id not in self.neurons or target_id not in self.neurons:
            return False
        if source_id == target_id:
            return False
        if target_id in self.neurons[source_id].outgoing:
            return False
        connection = EnhancedConnectionV2(source_id, target_id, dim=self.neuron_dim, rank=8)
        self.neurons[source_id].outgoing[target_id] = connection
        self.neurons[target_id].incoming[source_id] = connection
        self.stats["connections_created"] += 1
        return True

    def _ensure_task_capacity(self, task_id: int):
        if task_id in self.task_neuron_pools:
            return
        self.task_neuron_pools[task_id] = []
        for _ in range(self.max_new_neurons_per_task):
            nid = self._add_neuron(neuron_type="hidden", task_id=task_id)
            self.hidden_neurons.append(nid)
            self.task_neuron_pools[task_id].append(nid)

            for shared_nid in random.sample(self.shared_neurons, min(10, len(self.shared_neurons))):
                self._add_connection(shared_nid, nid)
            for out_id in self.output_neurons:
                self._add_connection(nid, out_id)

        # Selectively enable deltas only for connections involving new pool
        for nid in self.task_neuron_pools[task_id]:
            for source_id, conn in self.neurons[nid].incoming.items():
                conn.ensure_task_delta(task_id)
            for target_id, conn in self.neurons[nid].outgoing.items():
                conn.ensure_task_delta(task_id)

        self._build_task_gate(task_id)

    def _build_task_gate(self, task_id: int):
        rng = random.Random(1337 + task_id)
        gate = set(self.input_neurons + self.output_neurons)
        if task_id in self.task_neuron_pools:
            gate.update(self.task_neuron_pools[task_id])
        shared_count = int(len(self.shared_neurons) * self.shared_gate_ratio)
        if self.shared_neurons and shared_count > 0:
            gate.update(rng.sample(self.shared_neurons, min(shared_count, len(self.shared_neurons))))
        if self.schema_gate_ratio > 0.0:
            sid = self.abstrax.task_schema.get(task_id)
            if sid is not None:
                schema_tasks = [tid for tid in self.abstrax.schema_clusters.get(sid, []) if tid != task_id]
                schema_candidates: List[int] = []
                for tid in schema_tasks:
                    schema_candidates.extend(self.task_neuron_pools.get(tid, []))
                if schema_candidates:
                    add_count = int(len(schema_candidates) * self.schema_gate_ratio)
                    add_count = max(1, min(len(schema_candidates), add_count))
                    gate.update(rng.sample(schema_candidates, add_count))
        self.task_gate_masks[task_id] = gate

    def _set_active_task(self, task_id: int):
        self.current_task_id = task_id
        if task_id not in self.task_gate_masks:
            self._build_task_gate(task_id)
        self.active_neurons = self.task_gate_masks.get(task_id, set())


    def begin_task(self, task_id: int):
        self._ensure_task_capacity(task_id)
        self._set_active_task(task_id)

    def infer_task(self, x: torch.Tensor) -> int:
        task_id = self.task_inference.infer(x)
        self._ensure_task_capacity(task_id)
        self._set_active_task(task_id)
        return task_id

    def forward(self, x: torch.Tensor, num_steps: int = 2) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        for i, nid in enumerate(self.input_neurons):
            if i < x.shape[1]:
                self.neurons[nid].activation = x[:, i].mean().repeat(self.neuron_dim)

        for _ in range(num_steps):
            activations = {nid: neuron.activation for nid, neuron in self.neurons.items()}
            for nid in self.hidden_neurons + self.output_neurons:
                if self.gating_enabled and nid not in self.active_neurons:
                    # Silence inactive neurons to reduce interference
                    self.neurons[nid].activation = torch.zeros(self.neuron_dim)
                    continue
                neuron = self.neurons[nid]
                input_signals = {sid: activations[sid] for sid in neuron.incoming.keys()}
                neuron.activate(input_signals, self.current_task_id)

        outputs = torch.stack([self.neurons[nid].activation.mean() for nid in self.output_neurons])
        if batch_size > 1:
            outputs = outputs.unsqueeze(0).expand(batch_size, -1)
        return outputs

    def build_tensor_graph(self, active_only: bool = True) -> Dict[str, torch.Tensor]:
        """
        Build a tensorized edge list for experimental fast message passing.
        """
        if active_only and self.gating_enabled:
            node_ids = sorted(self.active_neurons)
        else:
            node_ids = sorted(self.neurons.keys())
        if not node_ids:
            return {"node_ids": [], "edge_index": torch.empty((2, 0), dtype=torch.long), "edge_weight": torch.empty((0, self.neuron_dim, self.neuron_dim))}

        index = {nid: i for i, nid in enumerate(node_ids)}
        edge_src: List[int] = []
        edge_tgt: List[int] = []
        edge_weight: List[torch.Tensor] = []

        for tgt_id in node_ids:
            neuron = self.neurons[tgt_id]
            for src_id, conn in neuron.incoming.items():
                if src_id not in index:
                    continue
                weight = conn.W_base
                if self.current_task_id is not None and self.current_task_id in conn.task_deltas:
                    A, B = conn.task_deltas[self.current_task_id]
                    weight = weight + A @ B
                edge_src.append(index[src_id])
                edge_tgt.append(index[tgt_id])
                edge_weight.append(weight * conn.strength)

        if not edge_src:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight_tensor = torch.empty((0, self.neuron_dim, self.neuron_dim))
        else:
            edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
            edge_weight_tensor = torch.stack(edge_weight, dim=0)

        return {
            "node_ids": node_ids,
            "edge_index": edge_index,
            "edge_weight": edge_weight_tensor
        }

    def forward_tensorized(self, x: torch.Tensor, num_steps: int = 2, active_only: bool = True) -> torch.Tensor:
        """
        Experimental tensorized forward pass for V3-scale speed experiments.
        Optimized for device consistency and O(1) message passing.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        model_device = self.neurons[self.input_neurons[0]].activation.device
        device = model_device
        if x.device != device:
            x = x.to(device)

        # Initialize input activations
        for i, nid in enumerate(self.input_neurons):
            if i < x.shape[1]:
                # Reduce batch to mean for V2-style collective activation
                self.neurons[nid].activation = x[:, i].mean().repeat(self.neuron_dim).to(device)

        graph = self.build_tensor_graph(active_only=active_only)
        node_ids = graph["node_ids"]
        if not node_ids:
            outputs = torch.zeros(self.output_dim, device=device)
            if batch_size > 1:
                outputs = outputs.unsqueeze(0).expand(batch_size, -1)
            return outputs

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device)

        # Pre-cache constant node features
        node_act = torch.stack([self.neurons[nid].activation for nid in node_ids], dim=0).to(device)
        memory = torch.stack([self.neurons[nid].memory for nid in node_ids], dim=0).to(device)
        update_mask = torch.tensor([self.neurons[nid].neuron_type != "input" for nid in node_ids], dtype=torch.bool, device=device)

        # Pre-calculate memory gates to avoid per-step overhead
        memory_gate_list = []
        for nid in node_ids:
            neuron = self.neurons[nid]
            total_u = sum(neuron.task_usage.values()) + 1e-6
            gate = neuron.task_usage.get(self.current_task_id, 0.0) / total_u if self.current_task_id is not None else 0.5
            memory_gate_list.append(gate)
        memory_gate_t = torch.tensor(memory_gate_list, dtype=node_act.dtype, device=device).unsqueeze(1)

        for _ in range(num_steps):
            if edge_index.numel() == 0:
                break
            
            # Message aggregation via einsum and index_add
            src_act = node_act[edge_index[0]]
            msg = torch.einsum("ed,edk->ek", src_act, edge_weight)
            
            agg = torch.zeros_like(node_act)
            agg.index_add_(0, edge_index[1], msg)
            
            # Apply memory integration
            agg = agg + memory_gate_t * memory

            # Selective update (non-inputs only)
            updated = node_act.clone()
            updated[update_mask] = torch.tanh(agg[update_mask])
            node_act = updated

        # Flush activations back to neurons
        for idx, nid in enumerate(node_ids):
            if self.neurons[nid].neuron_type != "input":
                self.neurons[nid].activation = node_act[idx].detach()

        outputs = torch.stack([self.neurons[nid].activation.mean() for nid in self.output_neurons]).to(device)
        if batch_size > 1:
            outputs = outputs.unsqueeze(0).expand(batch_size, -1)
        return outputs

    def learn(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None, error_propagation_steps: int = 1):
        if task_id is None:
            task_id = self.infer_task(x)
        self._set_active_task(task_id)
        self.task_inference.update(x, task_id)

        forward_steps = self.mastery_forward_steps if self.mastery_mode_enabled else 2
        output = self.forward(x, num_steps=forward_steps)

        if y.dim() == 1:
            y = y.unsqueeze(0)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        if self.action_perception_enabled:
            _ = self.action_loop.step(x.detach())
        if self.social_reasoning_enabled:
            _ = self.social_reasoner.infer_belief(x.detach())
        if self.physical_intuition_enabled:
            phys_loss = self.physical_intuition.update(x.detach())
        else:
            phys_loss = 0.0

        if self.few_shot_enabled:
            fs_pred = self.few_shot_memory.predict(task_id, x, k=self.few_shot_k)
            if fs_pred is not None:
                fs_pred = fs_pred.unsqueeze(0) if fs_pred.dim() == 1 else fs_pred
                y = (1.0 - self.few_shot_alpha) * y + self.few_shot_alpha * fs_pred

        error = y - output
        loss = (error ** 2).mean()

        if self.world_model_enabled:
            self.world_model.update(x.detach(), y.detach(), task_id)

        salience_weight, surprise, _progress = self.salience_engine.compute(loss.item(), y, output)
        novelty = 0.0
        if self.task_inference.new_task_threshold > 0:
            novelty = self.task_inference.last_distance / self.task_inference.new_task_threshold
        novelty = max(0.0, min(2.0, novelty))
        # cache query embedding for episodic retrieval
        self.last_query_emb = self.task_inference._embed(x.detach())
        if self.intrinsic_enabled:
            uncertainty = 0.0
            if self.world_model_enabled:
                uncertainty = self.planner.counterfactual_score(x.detach(), task_id)
            intrinsic_weight = self.intrinsic_engine.compute(
                task_id=task_id,
                loss=float(loss.item()),
                novelty=novelty,
                uncertainty=uncertainty
            )
            salience_weight = max(0.2, min(2.5, salience_weight * intrinsic_weight))
            if self.goal_generator_enabled:
                self.goal_generator.update(self.intrinsic_engine.last_metrics, self.self_model.identity_vector)
        goal = self.goal_generator.propose() if self.goal_generator_enabled else None
        active_ratio = len(self.active_neurons) / max(1, len(self.neurons))
        affect = (-float(loss.item()), float(surprise))
        self.last_affect = affect
        sid = self.abstrax.task_schema.get(task_id, task_id)
        self.schema_salience_sum[sid] += salience_weight
        self.schema_salience_count[sid] += 1
        if self.episodic_memory_enabled:
            self.episodic_memory.add_event(EpisodicEvent(
                step=self.stats["total_steps"],
                task_id=task_id,
                x=x.detach().clone(),
                y=y.detach().clone(),
                output=output.detach().clone(),
                surprise=surprise,
                salience=salience_weight,
                affect=affect,
                emb=self.last_query_emb.detach().clone()
            ))
        self.self_model.update(
            loss=float(loss.item()),
            surprise=surprise,
            active_ratio=active_ratio,
            task_id=task_id,
            total_tasks=len(self.task_neuron_pools)
        )
        if self.long_term_planning_enabled:
            self.long_term_planner.update(
                task_id=task_id,
                loss=float(loss.item()),
                competence=self.self_model.competence,
                intrinsic_state=self.intrinsic_engine._state,
                goal=goal
            )
        if self.unsupervised_enabled:
            unsup_loss = self.unsupervised_predictor.update(x.detach())
            salience_weight *= (1.0 + 0.05 * unsup_loss)
        if self.meta_controller_enabled:
            mods = self.meta_controller.compute_modulators(
                loss=float(loss.item()),
                uncertainty=uncertainty if self.intrinsic_enabled else 0.0,
                novelty=novelty,
                boredom=self.intrinsic_engine.last_metrics["boredom"] if self.intrinsic_enabled else 0.0,
                self_pred_error=self.self_model.self_pred_error
            )
        else:
            mods = {"error_scale": 1.0, "replay_scale": 1.0, "lr_scale": 1.0}
        if self.reward_composer_enabled:
            reward = self.reward_composer.compute({
                "surprise": surprise,
                "novelty": novelty,
                "uncertainty": uncertainty if self.intrinsic_enabled else 0.0,
                "progress": self.intrinsic_engine.last_metrics["progress"] if self.intrinsic_enabled else 0.0,
                "boredom": self.intrinsic_engine.last_metrics["boredom"] if self.intrinsic_enabled else 0.0
            })
            if self.value_alignment_enabled:
                reward = self.value_alignment.adjust(reward, affect)
            if self.value_alignment_learning_enabled:
                reward = self.value_alignment_learner.update(affect, reward)
            salience_weight *= max(0.5, min(1.5, 1.0 + reward))
        if self.long_term_planning_enabled:
            plan = self.long_term_planner.current_plan()
            if goal is not None or plan is not None:
                self.long_term_plan = {
                    "goal": goal,
                    "plan": plan,
                    "self_state": self.self_model.identity_vector.detach().clone()
                }
        if self.self_audit_enabled:
            replay_boost = self.self_audit.update(float(loss.item()))
            mods["replay_scale"] = mods["replay_scale"] * (1.0 + replay_boost)
            self.last_audit_signal = replay_boost
        else:
            self.last_audit_signal = 0.0
        if self.self_rewriter_enabled:
            self.self_rewriter.step(self, self.self_model.traits, float(loss.item()), audit_signal=self.last_audit_signal)
        if phys_loss > 0:
            salience_weight *= (1.0 + 0.05 * phys_loss)

        comp = self.self_model.competence.get(task_id, float(loss.item()))
        mastered = comp <= self.mastery_loss_threshold
        lr_boost = self.mastery_lr_multiplier if (self.mastery_mode_enabled and not mastered) else 1.0
        corr_ema = self.hebb_corr_ema
        if self.mastery_mode_enabled and not mastered:
            corr_ema = max(0.5, min(0.98, self.hebb_corr_ema * self.mastery_corr_ema_scale))

        error_signals = {}
        for i, nid in enumerate(self.output_neurons):
            if i < error.shape[1]:
                error_signals[nid] = error[:, i].mean().repeat(self.neuron_dim)

        self._propagate_error(
            error_signals=error_signals,
            task_id=task_id,
            steps=error_propagation_steps,
            error_scale=0.3 * salience_weight * mods["error_scale"] * lr_boost * (1.0 + self.self_pred_loss_weight * self.self_model.self_pred_error),
            usage_threshold=5,
            corr_ema=corr_ema,
            lr_multiplier=lr_boost
        )

        # Store for replay (self-distillation)
        self.replay_buffer.add(task_id, x, y, output.detach().mean(dim=0))
        self.surprise_buffer.add(Experience(
            task_id=task_id,
            x=x.detach().clone(),
            y=y.detach().clone(),
            teacher=output.detach().mean(dim=0),
            surprise=surprise,
            salience=salience_weight,
            step=self.stats["total_steps"]
        ))
        if self.few_shot_enabled:
            self.few_shot_memory.add(task_id, x, y)

        # Replay step
        if self.stats["total_steps"] % self.replay_every == 0:
            self._replay_step(replay_scale=mods["replay_scale"])

        if self.oracle_enabled:
            self._oracle_check(float(loss.item()))

        if self.oracle_enabled:
            self._schema_oracle_check(task_id, float(loss.item()))

        self.stats["total_steps"] += 1
        return loss.item()

    def _replay_step(self, batch_size: int = 10, distill_weight: float = 0.7, replay_scale: float = 1.0):
        if not self.replay_buffer.storage and len(self.surprise_buffer) == 0:
            return

        effective_batch = max(1, int(batch_size * replay_scale))
        samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        surprise_k = int(effective_batch * 0.3)
        if len(self.surprise_buffer) > 0 and surprise_k > 0:
            for exp in self.surprise_buffer.sample(surprise_k):
                bias = self._goal_replay_bias(exp.task_id)
                samples.append((exp.task_id, exp.x, exp.y, exp.teacher, exp.salience * bias))

        if self.episodic_memory_enabled and len(self.episodic_memory) > 0:
            episodic_k = int(effective_batch * 0.2)
            if self.last_query_emb is not None:
                if self.last_affect is not None:
                    evs = self.episodic_memory.sample_similar_affect(self.last_query_emb, self.last_affect, episodic_k)
                else:
                    evs = self.episodic_memory.sample_similar(self.last_query_emb, episodic_k)
            else:
                evs = self.episodic_memory.sample(episodic_k)
            for ev in evs:
                teacher = ev.output.mean(dim=0) if ev.output.dim() > 1 else ev.output
                bias = self._goal_replay_bias(ev.task_id)
                samples.append((ev.task_id, ev.x, ev.y, teacher, ev.salience * bias))

        remaining = max(0, effective_batch - len(samples))
        if self.replay_buffer.storage and remaining > 0:
            task_ids = list(self.replay_buffer.storage.keys())
            max_tid = max(task_ids)
            # Weight older tasks higher to counter forgetting
            task_weights = []
            num_schemas = max(1, len(self.abstrax.schema_clusters))
            mean_per_schema = self.replay_total_samples / num_schemas if self.replay_total_samples > 0 else 0.0
            competition = compute_schema_competition(
                schema_replay_counts=self.replay_schema_counts,
                schema_salience_sum=self.schema_salience_sum,
                schema_salience_count=self.schema_salience_count
            )
            mean_comp = float(np.mean(list(competition.values()))) if competition else 0.0
            for tid in task_ids:
                age_weight = 1.0 + (max_tid - tid) * 0.15
                schema_bonus = 1.0 + max(0, self.abstrax.schema_size(tid) - 1) * 0.1
                balance = 1.0
                if mean_per_schema > 0 and self.schema_replay_balance_strength > 0:
                    sid = self.abstrax.task_schema.get(tid, tid)
                    schema_count = self.replay_schema_counts.get(sid, 0)
                    delta = (mean_per_schema - schema_count) / mean_per_schema
                    balance = 1.0 + self.schema_replay_balance_strength * delta
                    balance = max(0.5, min(1.5, balance))
                if self.schema_competition_balance_strength > 0 and competition:
                    sid = self.abstrax.task_schema.get(tid, tid)
                    comp = competition.get(sid, mean_comp)
                    comp_delta = mean_comp - comp
                    comp_balance = 1.0 + self.schema_competition_balance_strength * comp_delta
                    comp_balance = max(0.5, min(1.5, comp_balance))
                else:
                    comp_balance = 1.0
                goal_bias = self._goal_replay_bias(tid)
                task_weights.append(age_weight * schema_bonus * balance * comp_balance * goal_bias)
            force_proto = [tid for tid in task_ids if tid <= max_tid - self.prototype_only_after]

            replay_samples = self.replay_buffer.sample(
                batch_size=remaining,
                task_ids=task_ids,
                task_weights=task_weights,
                force_prototype_tasks=force_proto
            )
            for tid, x, y, teacher in replay_samples:
                bias = self._goal_replay_bias(tid)
                samples.append((tid, x, y, teacher, 1.0 * bias))

        if not samples:
            return

        if self.world_model_enabled and samples:
            imagination_k = min(int(effective_batch * 0.2), len(samples))
            if imagination_k > 0:
                base_samples = random.sample(samples, imagination_k)
                imagined = []
                for tid, x, _y, _teacher, _sal in base_samples:
                    noise = torch.randn_like(x) * 0.05
                    x_perturbed = x + noise
                    y_hat = self.world_model.predict(x_perturbed, tid).detach()
                    imagined.append((tid, x_perturbed, y_hat, y_hat.mean(dim=0), 0.6))
                samples[:imagination_k] = imagined

        for tid, _x, _y, _t, _sal in samples:
            sid = self.abstrax.task_schema.get(tid, tid)
            self.replay_schema_counts[sid] += 1
            self.replay_total_samples += 1

        for tid, x, y, teacher, salience_weight in samples:
            self._set_active_task(tid)
            replay_steps = self.mastery_forward_steps if self.mastery_mode_enabled else 2
            output = self.forward(x, num_steps=replay_steps)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            if teacher.dim() == 1:
                teacher = teacher.unsqueeze(0)
            if y.dim() == 1:
                y = y.unsqueeze(0)

            distill_loss = ((output - teacher) ** 2).mean()
            task_loss = ((output - y) ** 2).mean()
            effective_distill = max(0.2, min(0.95, distill_weight * salience_weight))
            _loss = effective_distill * distill_loss + (1 - effective_distill) * task_loss

            error = y - output
            error_signals = {}
            for i, nid in enumerate(self.output_neurons):
                if i < error.shape[1]:
                    error_signals[nid] = error[:, i].mean().repeat(self.neuron_dim)
            self._propagate_error(
                error_signals=error_signals,
                task_id=tid,
                steps=2,
                error_scale=0.2 * salience_weight,
                usage_threshold=2,
                corr_ema=self.hebb_corr_ema
            )

    def _oracle_check(self, loss_value: float):
        self.loss_history.append(loss_value)
        if len(self.loss_history) < self.oracle_window:
            return
        avg = sum(self.loss_history) / len(self.loss_history)
        if avg <= 0:
            return
        if loss_value > self.oracle_spike_ratio * avg:
            for _ in range(self.oracle_replay_burst):
                self._replay_step()

    def _schema_oracle_check(self, task_id: int, loss_value: float):
        sid = self.abstrax.task_schema.get(task_id, task_id)
        history = self.schema_loss_history[sid]
        history.append(loss_value)
        if len(history) < self.oracle_window:
            return
        avg = sum(history) / len(history)
        if avg <= 0:
            return
        if loss_value > self.oracle_spike_ratio * avg:
            # Focus replay on tasks from this schema
            schema_tasks = self.abstrax.schema_clusters.get(sid, [])
            if not schema_tasks:
                return
            for _ in range(self.oracle_replay_burst):
                samples = self.replay_buffer.sample(
                    batch_size=8,
                    task_ids=schema_tasks
                )
                if not samples:
                    continue
                for tid, x, y, teacher in samples:
                    self._set_active_task(tid)
                    output = self.forward(x)
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    if teacher.dim() == 1:
                        teacher = teacher.unsqueeze(0)
                    if y.dim() == 1:
                        y = y.unsqueeze(0)
                    distill_loss = ((output - teacher) ** 2).mean()
                    task_loss = ((output - y) ** 2).mean()
                    _loss = 0.7 * distill_loss + 0.3 * task_loss
                    error = y - output
                    error_signals = {}
                    for i, nid in enumerate(self.output_neurons):
                        if i < error.shape[1]:
                            error_signals[nid] = error[:, i].mean().repeat(self.neuron_dim)
                    self._propagate_error(
                        error_signals=error_signals,
                        task_id=tid,
                        steps=2,
                        error_scale=0.2,
                        usage_threshold=2,
                        corr_ema=self.hebb_corr_ema
                    )

    def _propagate_error(
        self,
        error_signals: Dict[int, torch.Tensor],
        task_id: int,
        steps: int,
        error_scale: float,
        usage_threshold: int,
        corr_ema: Optional[float] = None,
        lr_multiplier: float = 1.0
    ):
        use_corr_ema = self.hebb_corr_ema if corr_ema is None else corr_ema
        for _ in range(steps):
            new_errors = {}
            for nid, neuron in self.neurons.items():
                if self.gating_enabled and nid not in self.active_neurons:
                    continue
                if nid not in error_signals:
                    continue
                error_sig = error_signals[nid]
                neuron.update_memory(error_sig)

                if not neuron.incoming:
                    continue
                source_ids = list(neuron.incoming.keys())
                if self.gating_enabled:
                    source_ids = [sid for sid in source_ids if sid in self.active_neurons]
                    if not source_ids:
                        continue
                pre_acts = torch.stack([self.neurons[sid].activation for sid in source_ids], dim=0)
                post_act = neuron.activation
                corrs = (pre_acts * post_act).mean(dim=1)

                noise_A = torch.randn(self.neuron_dim, 8) * 0.01
                noise_B = torch.randn(8, self.neuron_dim) * 0.01

                allow_base_update = len(self.task_neuron_pools) < self.freeze_base_after_tasks
                for idx, source_id in enumerate(source_ids):
                    corr_val = corrs[idx].item()
                    if abs(corr_val) < self.hebb_active_threshold:
                        continue
                    connection = neuron.incoming[source_id]
                    if connection.usage_per_task[task_id] > usage_threshold:
                        connection.ensure_task_delta(task_id)
                    connection.hebbian_update_with_corr(
                        corr=corr_val,
                        task_id=task_id,
                        noise_basis=(noise_A, noise_B),
                        corr_ema=use_corr_ema,
                        corr_clip=self.hebb_corr_clip,
                        noise_scale=self.hebb_noise_scale,
                        allow_base_update=allow_base_update,
                        lr_multiplier=lr_multiplier
                    )
                    if source_id not in new_errors:
                        new_errors[source_id] = torch.zeros(self.neuron_dim)
                    new_errors[source_id] += error_sig * connection.strength * error_scale
            error_signals = new_errors

    def consolidate_task(self, task_id: int, strength: float = 0.2):
        schema_size = self.abstrax.schema_size(task_id)
        schema_boost = 1.0 + min(0.6, max(0.0, (schema_size - 1) * 0.1))
        adj_strength = max(0.05, min(0.6, strength * schema_boost))
        if task_id in self.task_neuron_pools:
            for nid in self.task_neuron_pools[task_id]:
                if nid in self.neurons:
                    self.neurons[nid].consolidate(adj_strength)
        for neuron in self.neurons.values():
            for connection in neuron.outgoing.values():
                if task_id in connection.usage_per_task:
                    usage = connection.usage_per_task[task_id]
                    if usage > 10:
                        connection.consolidate(adj_strength)
        if self.skill_library_enabled:
            self.skill_library.update_skill(task_id, self.get_stats())

    def get_stats(self) -> Dict:
        num_connections = sum(len(n.outgoing) for n in self.neurons.values())
        schema_sizes = list(self.abstrax.schema_clusters.values())
        avg_schema = 0.0
        if schema_sizes:
            avg_schema = sum(len(s) for s in schema_sizes) / len(schema_sizes)
        return {
            **self.stats,
            "current_neurons": len(self.neurons),
            "current_hidden": len(self.hidden_neurons),
            "current_connections": num_connections,
            "avg_degree": num_connections / max(len(self.neurons), 1),
            "tasks_learned": len(self.task_neuron_pools),
            "schemas": len(self.abstrax.schema_clusters),
            "avg_schema_size": round(avg_schema, 3),
            "world_model_updates": self.world_model.updates if self.world_model_enabled else 0,
            "world_model_loss": round(self.world_model.last_loss, 6) if self.world_model_enabled else 0.0,
            "replay_total_samples": self.replay_total_samples,
            "intrinsic_novelty": round(self.intrinsic_engine.last_metrics["novelty"], 4),
            "intrinsic_progress": round(self.intrinsic_engine.last_metrics["progress"], 4),
            "intrinsic_boredom": round(self.intrinsic_engine.last_metrics["boredom"], 4),
            "intrinsic_uncertainty": round(self.intrinsic_engine.last_metrics["uncertainty"], 4),
            "self_stability": round(self.self_model.traits["stability"], 4),
            "self_plasticity": round(self.self_model.traits["plasticity"], 4),
            "self_curiosity": round(self.self_model.traits["curiosity"], 4)
        }

    def sleep_consolidate(self, steps: int = 2, decay: float = 0.0005):
        """
        Offline consolidation: low-rate replay + mild synaptic renormalization.
        """
        for _ in range(max(1, steps)):
            self._replay_step(batch_size=12, distill_weight=0.8)

        if decay <= 0:
            return
        for neuron in self.neurons.values():
            for conn in neuron.outgoing.values():
                if conn.age > 200 and sum(conn.usage_per_task.values()) == 0:
                    conn.W_base *= (1.0 - decay)
                    conn.strength = max(0.1, conn.strength * (1.0 - decay))

    def recompute_schemas(self):
        """
        Cluster task prototypes into schema groups for abstraction folding.
        """
        self.abstrax.recompute(self.task_inference.prototypes)
        for tid in list(self.task_neuron_pools.keys()):
            self._build_task_gate(tid)
        if self.semantic_schema_enabled:
            self.semantic_schema.update(self.abstrax.schema_clusters, self.task_inference.prototypes)

    def select_task_for_curriculum(self, num_tasks: int, temperature: float = 1.0) -> int:
        """
        Select a task index based on intrinsic state, competence, schema size, and recency.
        """
        if num_tasks <= 1:
            return 0

        goal = self.goal_generator.propose() if self.goal_generator_enabled else None
        goal_novelty = goal.get("novelty", 0.0) if goal else 0.0
        goal_uncertainty = goal.get("uncertainty", 0.0) if goal else 0.0
        goal_boredom = goal.get("boredom", 0.0) if goal else 0.0

        scores = []
        for tid in range(num_tasks):
            state = self.intrinsic_engine._state.get(tid)
            novelty = state["novelty_ema"] if state else 0.2
            boredom = (state["boredom_counter"] / self.intrinsic_engine.boredom_window) if state else 0.0
            uncertainty = state["uncertainty_ema"] if state else 0.3
            competence = self.self_model.competence.get(tid)
            if competence is None:
                competence_score = 1.0
            else:
                competence_score = 1.0 / (1.0 + competence)

            schema_bonus = 1.0 + max(0, self.abstrax.schema_size(tid) - 1) * 0.05
            age_bonus = 1.0 + 0.05 * self.curriculum_age.get(tid, 0)

            score = (
                0.5 * competence_score
                + 0.3 * novelty
                + 0.2 * uncertainty
                - 0.3 * boredom
            )
            goal_bias = 0.15 * goal_novelty * novelty + 0.15 * goal_uncertainty * uncertainty - 0.15 * goal_boredom * boredom
            score = score + goal_bias
            score = max(0.05, score) * schema_bonus * age_bonus
            if self.long_term_planning_enabled:
                score *= self.long_term_planner.bias(tid)
            scores.append(score)

        if temperature <= 0:
            chosen = int(np.argmax(scores))
        else:
            scaled = [s / max(1e-6, temperature) for s in scores]
            chosen = random.choices(range(num_tasks), weights=scaled, k=1)[0]

        for tid in range(num_tasks):
            self.curriculum_age[tid] += 1
        self.curriculum_age[chosen] = 0
        return chosen

    def _goal_replay_bias(self, task_id: int) -> float:
        if not self.goal_generator_enabled:
            return 1.0
        goal = self.goal_generator.propose()
        if not goal:
            return 1.0
        state = self.intrinsic_engine._state.get(task_id)
        if not state:
            return 1.0
        novelty = state["novelty_ema"]
        uncertainty = state["uncertainty_ema"]
        boredom = state["boredom_counter"] / self.intrinsic_engine.boredom_window
        bias = 1.0
        bias += 0.3 * goal.get("novelty", 0.0) * novelty
        bias += 0.3 * goal.get("uncertainty", 0.0) * uncertainty
        bias -= 0.3 * goal.get("boredom", 0.0) * boredom
        if self.long_term_planning_enabled:
            bias *= self.long_term_planner.bias(task_id)
        return max(0.7, min(1.5, bias))

def train_agnis_v2(
    agnis: EnhancedAGNISV2,
    task_sequence: List[Tuple[str, List]],
    epochs_per_task: int = 30,
    known_boundaries: bool = True,
    self_directed: bool = False,
    self_directed_cycles: Optional[int] = None,
    samples_per_epoch: int = 20,
    curriculum_temperature: float = 1.0,
    consolidate_every: int = 10,
    schema_recompute_every: int = 10
):
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING WITH ENHANCED AGNIS V2")
    print("=" * 70)

    retention_matrix = []

    schema_steps: List[int] = []
    schema_counts: List[int] = []
    schema_avg_sizes: List[float] = []

    if not self_directed:
        for task_idx, (task_name, dataset) in enumerate(task_sequence):
            print(f"\n{'-' * 70}")
            print(f"TASK {task_idx}: {task_name}")
            print(f"{'-' * 70}")

            if known_boundaries:
                agnis.begin_task(task_idx)

            for epoch in range(epochs_per_task):
                epoch_loss = 0.0
                for x, y in dataset:
                    loss = agnis.learn(x, y, task_id=task_idx if known_boundaries else None)
                    epoch_loss += loss
                avg_loss = epoch_loss / len(dataset)
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs_per_task} - Loss: {avg_loss:.4f}")

            agnis.consolidate_task(task_idx, strength=0.3)
            agnis.sleep_consolidate(steps=2, decay=0.0005)
            agnis.recompute_schemas()
            schema_steps.append(task_idx)
            if agnis.abstrax.schema_clusters:
                avg_schema = sum(len(s) for s in agnis.abstrax.schema_clusters.values()) / len(agnis.abstrax.schema_clusters)
            else:
                avg_schema = 0.0
            schema_counts.append(len(agnis.abstrax.schema_clusters))
            schema_avg_sizes.append(avg_schema)

            print("\n  Retention Test:")
            retention_row = []
            for test_idx, (test_name, test_dataset) in enumerate(task_sequence[: task_idx + 1]):
                if known_boundaries:
                    agnis.current_task_id = test_idx
                test_loss = 0.0
                for x, y in test_dataset[:20]:
                    output = agnis.forward(x)
                    if y.dim() == 1:
                        y = y.unsqueeze(0)
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    test_loss += ((y - output) ** 2).mean().item()
                avg_test_loss = test_loss / min(20, len(test_dataset))
                retention_row.append(avg_test_loss)
                print(f"    {test_name}: {avg_test_loss:.4f}")

            while len(retention_row) < len(task_sequence):
                retention_row.append(np.nan)
            retention_matrix.append(retention_row)
    else:
        total_cycles = self_directed_cycles or (len(task_sequence) * epochs_per_task)
        print(f"\n{'-' * 70}")
        print(f"SELF-DIRECTED CURRICULUM: {total_cycles} cycles")
        print(f"{'-' * 70}")

        curriculum_tasks: List[int] = []
        curriculum_losses: List[float] = []
        curriculum_novelty: List[float] = []
        curriculum_uncertainty: List[float] = []
        curriculum_boredom: List[float] = []

        for cycle in range(total_cycles):
            task_idx = agnis.select_task_for_curriculum(len(task_sequence), temperature=curriculum_temperature)
            task_name, dataset = task_sequence[task_idx]
            if known_boundaries:
                agnis.begin_task(task_idx)

            batch = random.sample(dataset, min(samples_per_epoch, len(dataset)))
            epoch_loss = 0.0
            for x, y in batch:
                loss = agnis.learn(x, y, task_id=task_idx if known_boundaries else None)
                epoch_loss += loss
            avg_loss = epoch_loss / max(1, len(batch))
            curriculum_tasks.append(task_idx)
            curriculum_losses.append(avg_loss)
            curriculum_novelty.append(agnis.intrinsic_engine.last_metrics["novelty"])
            curriculum_uncertainty.append(agnis.intrinsic_engine.last_metrics["uncertainty"])
            curriculum_boredom.append(agnis.intrinsic_engine.last_metrics["boredom"])

            if (cycle + 1) % max(1, total_cycles // 10) == 0:
                print(f"  Cycle {cycle+1}/{total_cycles} - Task {task_idx} - Loss: {avg_loss:.4f}")

            if consolidate_every > 0 and (cycle + 1) % consolidate_every == 0:
                agnis.consolidate_task(task_idx, strength=0.2)
                agnis.sleep_consolidate(steps=1, decay=0.0003)

            if schema_recompute_every > 0 and (cycle + 1) % schema_recompute_every == 0:
                agnis.recompute_schemas()
                schema_steps.append(cycle + 1)
                if agnis.abstrax.schema_clusters:
                    avg_schema = sum(len(s) for s in agnis.abstrax.schema_clusters.values()) / len(agnis.abstrax.schema_clusters)
                else:
                    avg_schema = 0.0
                schema_counts.append(len(agnis.abstrax.schema_clusters))
                schema_avg_sizes.append(avg_schema)

        print("\n  Final Retention Test:")
        retention_row = []
        for test_idx, (test_name, test_dataset) in enumerate(task_sequence):
            if known_boundaries:
                agnis.current_task_id = test_idx
            test_loss = 0.0
            for x, y in test_dataset[:20]:
                output = agnis.forward(x)
                if y.dim() == 1:
                    y = y.unsqueeze(0)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                test_loss += ((y - output) ** 2).mean().item()
            avg_test_loss = test_loss / min(20, len(test_dataset))
            retention_row.append(avg_test_loss)
            print(f"    {test_name}: {avg_test_loss:.4f}")

        while len(retention_row) < len(task_sequence):
            retention_row.append(np.nan)
        retention_matrix.append(retention_row)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    stats = agnis.get_stats()
    print(f"Final neurons: {stats['current_neurons']}")
    print(f"Final connections: {stats['current_connections']}")
    return retention_matrix


def compute_schema_competition(
    schema_replay_counts: Dict[int, int],
    schema_salience_sum: Dict[int, float],
    schema_salience_count: Dict[int, int]
) -> Dict[int, float]:
    schema_ids = sorted(set(schema_replay_counts.keys()).union(schema_salience_sum.keys()))
    if not schema_ids:
        return {}

    competition = {}
    max_replay = max(schema_replay_counts.values()) if schema_replay_counts else 1
    for sid in schema_ids:
        replay_score = schema_replay_counts.get(sid, 0) / max(1, max_replay)
        salience_score = schema_salience_sum.get(sid, 0.0) / max(1, schema_salience_count.get(sid, 0))
        competition[sid] = 0.6 * replay_score + 0.4 * salience_score
    return competition

if __name__ == "__main__":
    from agnis import create_simple_task

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    agnis = EnhancedAGNISV2(
        input_dim=10,
        output_dim=3,
        initial_hidden=40,
        neuron_dim=12,
        max_new_neurons_per_task=15,
        replay_capacity_per_task=100,
        replay_every=20
    )

    tasks = [
        (f"Task {i}", create_simple_task(20, 10, 3, "classification"))
        for i in range(5)
    ]

    train_agnis_v2(agnis, tasks, epochs_per_task=10, known_boundaries=True)
