import torch
import random
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional
from agnis_v4_core import PredictiveHierarchy

@dataclass
class Experience:
    task_id: int
    x: torch.Tensor
    y: torch.Tensor
    surprise: float
    salience: float
    step: int

class SalienceEngine:
    """Computes salience weights from surprise and learning progress."""
    def __init__(
        self,
        ema: float = 0.95,
        min_weight: float = 0.2,
        max_weight: float = 5.0,  # Allow highly salient experiences to boost learning 5x
        surprise_scale: float = 2.0,
        progress_scale: float = 1.0
    ):
        self.ema = ema
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.surprise_scale = surprise_scale
        self.progress_scale = progress_scale
        self.loss_ema: Optional[float] = None

    def compute(self, loss: float) -> Tuple[float, float, float]:
        surprise = loss
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

class SurpriseBuffer:
    """Priority buffer for high-surprise/high-salience experiences."""
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

class CognitivePredictiveAgent:
    """
    Wraps the V4 PredictiveHierarchy with V2/V3 Cognitive Ecosystems:
    Salience, Surprise Buffering, and Offline Replay Dreaming.
    """
    def __init__(self, hierarchy: PredictiveHierarchy, device: str = "cpu"):
        self.hierarchy = hierarchy
        self.device = device
        self.salience_engine = SalienceEngine()
        self.buffer = SurpriseBuffer()
        self.step_count = 0
        
        # Save base learning rates
        self.base_lrs = []
        for col in self.hierarchy.layers:
            self.base_lrs.append((col.eta_V, col.eta_W))

    def observe_and_learn(self, x: torch.Tensor, y: torch.Tensor, task_id: int = 0, 
                          max_steps: int = 150, recognition_weight: float = 1.0, beta_push: float = 5.0):
        """Processes a single sample online."""
        # 1. Do a dry-run inference to compute surprise (MSE without label guidance)
        self.hierarchy.reset_states(batch_size=1)
        with torch.no_grad():
            pred_y_tensor = self.hierarchy.predict_label(x, max_steps=max_steps)
            if pred_y_tensor.shape[1] > y.shape[1]:
                d = y.shape[1]
                pred_y_tensor = pred_y_tensor[:, :d]
            surprise_loss = torch.nn.functional.mse_loss(pred_y_tensor, y).item()
            
        # 2. Compute Salience
        weight, surprise, progress = self.salience_engine.compute(surprise_loss)

        # 3. Add to Episodic Buffer if surprise is notable
        if surprise > 0.05:
            exp = Experience(task_id, x, y, surprise, weight, self.step_count)
            self.buffer.add(exp)

        # 4. Scale Learning Rates dynamically for this sample based on Salience
        self._apply_salience_weight(weight)

        # 5. Run standard online learning
        self.hierarchy.infer_and_learn_online(
            x, top_level_label=y,
            max_steps=max_steps,
            recognition_weight=recognition_weight,
            beta_push=beta_push
        )
        
        # 6. Restore base learning rates
        self._apply_salience_weight(1.0, restore=True)
        self.step_count += 1
        
        return weight, surprise

    def dream_replay(self, batch_size: int = 16, max_steps: int = 150, recognition_weight: float = 1.0, beta_push: float = 5.0):
        """Extracts highly salient memories and dreams about them to bootstrap the sparse dictionary."""
        if len(self.buffer) < batch_size:
            return 0.0

        experiences = self.buffer.sample(batch_size)
        x_batch = torch.cat([e.x for e in experiences], dim=0)
        y_batch = torch.cat([e.y for e in experiences], dim=0)
        
        # Boost learning rates for dream consolidation
        self._apply_salience_weight(2.0)
        
        # Check for Neurogenesis Trigger
        avg_surprise = sum(e.surprise for e in experiences) / len(experiences)
        avg_salience = sum(e.salience for e in experiences) / len(experiences)
        
        if avg_surprise > 0.25 and avg_salience > 1.1:
            # Persistent anomaly! Trigger identity sliver neurogenesis
            # Pick the sample with highest individual surprise in the batch
            sample_surprises = [torch.nn.functional.mse_loss(self.hierarchy.predict_label(x_batch[i:i+1]), y_batch[i:i+1]).item() for i in range(len(experiences))]
            best_idx = int(torch.argmax(torch.tensor(sample_surprises)))
            
            print(f">>> RECRUITING IDENTITY SLIVER PATHWAY for Sample {best_idx} (Surprise: {sample_surprises[best_idx]:.4f}) <<<")
            self.hierarchy.expand_pathway(x_batch[best_idx:best_idx+1], y_batch[best_idx:best_idx+1])
            
            # --- NEURAL CONSOLIDATION ---
            # Immediately 'burn-in' the new pathway with a high-intensity focused dream
            print(f">>> CONSOLIDATING PATHWAY {self.hierarchy.layers[0].output_dim-1} <<<")
            self._apply_salience_weight(5.0) # Maximum learning push
            self.hierarchy.infer_and_learn_online(
                x_batch[best_idx:best_idx+1], 
                top_level_label=y_batch[best_idx:best_idx+1],
                max_steps=250, # More steps to settle the new structure
                recognition_weight=1.5,
                beta_push=10.0 # Force alignment
            )
            self._apply_salience_weight(1.0, restore=True)
            
            self.buffer._heap.clear()
            self.buffer._counter = 0
            return len(experiences)
        
        self.hierarchy.infer_and_learn(
            x_batch, top_level_label=y_batch,
            max_steps=max_steps,
            recognition_weight=recognition_weight,
            beta_push=beta_push
        )
        
        self._apply_salience_weight(1.0, restore=True)
        return len(experiences)

    def _apply_salience_weight(self, weight: float, restore: bool = False):
        if restore:
            for i, col in enumerate(self.hierarchy.layers):
                col.eta_V, col.eta_W = self.base_lrs[i]
        else:
            for i, col in enumerate(self.hierarchy.layers):
                base_v, base_w = self.base_lrs[i]
                col.eta_V = base_v * weight
                col.eta_W = base_w * weight
