"""
Experience Replay Buffer for Continual Learning

Stores samples from past tasks and allows mixed training on old + new data.
This prevents catastrophic forgetting by periodically "dreaming about the past".
"""

import torch
import random
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class ReplayBuffer:
    """
    Experience Replay Buffer for Continual Learning.
    
    Stores (input, target, task_id) tuples and supports:
    - Random sampling across all tasks
    - Task-specific sampling for balanced replay
    - Reservoir sampling to maintain diversity
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: List[Tuple[str, str, int]] = []  # (input_str, target_str, task_id)
        self.task_indices: Dict[int, List[int]] = defaultdict(list)
        self.position = 0  # For reservoir sampling
        self.total_seen = 0  # Total samples seen (for reservoir sampling probability)
        
    def add(self, input_str: str, target_str: str, task_id: int = 0):
        """
        Add a sample to the buffer using reservoir sampling.
        This ensures uniform sampling from the entire stream, not just recent samples.
        """
        self.total_seen += 1
        
        if len(self.buffer) < self.capacity:
            # Buffer not full, just append
            idx = len(self.buffer)
            self.buffer.append((input_str, target_str, task_id))
            self.task_indices[task_id].append(idx)
        else:
            # Reservoir sampling: replace random element with probability capacity/total_seen
            if random.random() < self.capacity / self.total_seen:
                idx = random.randint(0, self.capacity - 1)
                
                # Remove old index from task tracking
                old_task_id = self.buffer[idx][2]
                if idx in self.task_indices[old_task_id]:
                    self.task_indices[old_task_id].remove(idx)
                
                # Add new sample
                self.buffer[idx] = (input_str, target_str, task_id)
                self.task_indices[task_id].append(idx)
    
    def sample(self, n: int) -> List[Tuple[str, str, int]]:
        """
        Sample n random samples from the buffer (uniform across all tasks).
        """
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(n, len(self.buffer)))
    
    def sample_by_task(self, n_per_task: int) -> List[Tuple[str, str, int]]:
        """
        Sample n samples from EACH task (balanced replay).
        """
        samples = []
        for task_id, indices in self.task_indices.items():
            if indices:
                valid_indices = [i for i in indices if i < len(self.buffer)]
                sampled_indices = random.sample(valid_indices, min(n_per_task, len(valid_indices)))
                for idx in sampled_indices:
                    samples.append(self.buffer[idx])
        return samples
    
    def get_task_count(self) -> Dict[int, int]:
        """
        Get count of samples per task.
        """
        return {task_id: len(indices) for task_id, indices in self.task_indices.items()}
    
    def __len__(self):
        return len(self.buffer)
    
    def __repr__(self):
        task_counts = self.get_task_count()
        return f"ReplayBuffer(size={len(self.buffer)}, tasks={task_counts})"


class GEMOptimizer:
    """
    Gradient Episodic Memory (GEM) Optimizer.
    
    Modifies gradients to not increase loss on old tasks.
    When the new gradient would hurt old task performance,
    it projects the gradient to be orthogonal to the old task gradient.
    
    Reference: Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning" (2017)
    """
    
    def __init__(self, model: torch.nn.Module, margin: float = 0.5):
        """
        Args:
            model: The neural network model
            margin: Minimum allowed dot product (0 = orthogonal, negative = interfering)
        """
        self.model = model
        self.margin = margin
        self.task_gradients: Dict[int, List[torch.Tensor]] = {}
        
    def store_task_gradient(self, task_id: int, gradients: List[torch.Tensor]):
        """
        Store reference gradients for a completed task.
        Call this after finishing training on a task.
        """
        # Clone and detach gradients
        self.task_gradients[task_id] = [g.clone().detach() if g is not None else None 
                                        for g in gradients]
    
    def project_gradient(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project gradients to not interfere with old task gradients.
        
        For each old task:
            If dot(grad_new, grad_old) < margin:
                grad_new -= (dot / ||grad_old||^2) * grad_old
        """
        if not self.task_gradients:
            return gradients  # No old tasks to protect
        
        projected = []
        for i, grad in enumerate(gradients):
            if grad is None:
                projected.append(None)
                continue
            
            grad_new = grad.clone()
            
            for task_id, old_grads in self.task_gradients.items():
                if i >= len(old_grads) or old_grads[i] is None:
                    continue
                    
                grad_old = old_grads[i]
                
                # Compute dot product
                dot = (grad_new * grad_old).sum()
                
                if dot < self.margin:
                    # Gradient would hurt old task, project it away
                    grad_old_norm_sq = (grad_old ** 2).sum() + 1e-8
                    grad_new = grad_new - ((dot - self.margin) / grad_old_norm_sq) * grad_old
            
            projected.append(grad_new)
        
        return projected
    
    def apply_projected_gradients(self, projected_gradients: List[torch.Tensor]):
        """
        Apply projected gradients to model parameters.
        """
        for param, proj_grad in zip(self.model.parameters(), projected_gradients):
            if proj_grad is not None and param.grad is not None:
                param.grad = proj_grad


def mixed_replay_step(model, 
                      new_data: Tuple[str, str],
                      replay_buffer: ReplayBuffer,
                      task_id: int,
                      replay_ratio: float = 0.3,
                      replay_samples: int = 4) -> Tuple[float, float]:
    """
    Perform a single training step with mixed new + replayed data.
    
    Args:
        model: The model with runtime_learn method
        new_data: (input_str, target_str) for new task
        replay_buffer: Buffer containing old samples
        task_id: Current task ID
        replay_ratio: Probability of including replay in this step
        replay_samples: Number of old samples to replay
    
    Returns:
        (new_loss, replay_loss)
    """
    new_input, new_target = new_data
    
    # Always learn new data
    new_loss = model.runtime_learn(new_input, new_target, task_id=task_id)
    
    # Maybe replay old data
    replay_loss = 0.0
    if random.random() < replay_ratio and len(replay_buffer) > 0:
        old_samples = replay_buffer.sample(replay_samples)
        for old_input, old_target, old_task_id in old_samples:
            loss = model.runtime_learn(old_input, old_target, task_id=old_task_id)
            replay_loss += loss
        replay_loss /= max(len(old_samples), 1)
    
    # Store new sample in buffer
    replay_buffer.add(new_input, new_target, task_id)
    
    return new_loss, replay_loss
