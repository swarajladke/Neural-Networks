"""
Synaptic Intelligence (SI) for Online Continual Learning

This module implements importance-weighted plasticity that allows a neural network
to learn continuously without freezing weights or requiring task boundaries.

Key Concepts:
- Omega (ω): Per-parameter importance scores, updated online
- Delta (δ): Accumulated parameter changes since last consolidation
- Penalty: λ * Σ ω_i * (θ_i - θ*_i)²

Reference: Zenke et al., "Continual Learning Through Synaptic Intelligence" (2017)
"""

import torch
import torch.nn as nn
from typing import Dict

class SynapticIntelligence:
    """
    Online Continual Learning via Synaptic Intelligence.
    
    Unlike EWC (which requires task boundaries), SI updates importance
    scores incrementally during training, enabling truly continuous learning.
    """
    
    def __init__(self, model: nn.Module, lambda_si: float = 100.0, epsilon: float = 0.1):
        """
        Args:
            model: The neural network to protect
            lambda_si: Regularization strength (higher = more protection)
            epsilon: Small constant for numerical stability
        """
        self.model = model
        self.lambda_si = lambda_si
        self.epsilon = epsilon
        
        # Per-parameter importance scores (ω)
        self.omega: Dict[str, torch.Tensor] = {}
        
        # Anchor weights (θ*) - the "reference" values
        self.anchor: Dict[str, torch.Tensor] = {}
        
        # Accumulated gradients for importance estimation
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.importance_accumulator: Dict[str, torch.Tensor] = {}
        
        self._initialize()
    
    def _initialize(self):
        """Initialize tracking tensors for all parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param.data)
                self.anchor[name] = param.data.clone()
                self.prev_params[name] = param.data.clone()
                self.importance_accumulator[name] = torch.zeros_like(param.data)
    
    def update_importance(self):
        """
        Update importance scores based on parameter changes and gradients.
        Call this AFTER each optimizer.step().
        
        The key insight: if a parameter changed AND reduced the loss,
        it was "important" for that learning step.
        
        Importance += (gradient * parameter_change) / (parameter_change² + ε)
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.prev_params:
                # How much did this parameter change?
                delta = param.data - self.prev_params[name]
                
                # If it has a gradient, estimate its contribution to loss reduction
                if param.grad is not None:
                    # Positive contribution = gradient aligned with change
                    contribution = -param.grad.data * delta
                    
                    # Accumulate importance (normalized by change magnitude)
                    self.importance_accumulator[name] += contribution.clamp(min=0)
                
                # Update previous params for next step
                self.prev_params[name] = param.data.clone()
    
    def consolidate(self):
        """
        Consolidate learned knowledge by updating omega and anchor.
        Call this periodically (e.g., every N steps or when switching contexts).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.omega:
                # Update importance scores
                delta_squared = (param.data - self.anchor[name]).pow(2) + self.epsilon
                new_omega = self.importance_accumulator[name] / delta_squared
                
                # Running average of importance (prevents explosion)
                self.omega[name] = 0.9 * self.omega[name] + 0.1 * new_omega
                
                # Update anchor to current weights
                self.anchor[name] = param.data.clone()
                
                # Reset accumulator
                self.importance_accumulator[name].zero_()
    
    def penalty(self) -> torch.Tensor:
        """
        Compute the SI regularization penalty.
        Add this to your loss: total_loss = task_loss + si.penalty()
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.omega:
                # Penalize deviations from anchor, weighted by importance
                loss += (self.omega[name] * (param - self.anchor[name]).pow(2)).sum()
        
        return self.lambda_si * loss
    
    def get_importance_stats(self) -> Dict[str, float]:
        """Get summary statistics of importance scores for debugging"""
        stats = {}
        for name in self.omega:
            omega = self.omega[name]
            stats[name] = {
                'mean': omega.mean().item(),
                'max': omega.max().item(),
                'nonzero_ratio': (omega > 0.01).float().mean().item()
            }
        return stats


class ContinualLearner:
    """
    High-level wrapper that combines a model with Synaptic Intelligence.
    """
    
    def __init__(self, model: nn.Module, lambda_si: float = 100.0, consolidate_every: int = 100):
        self.model = model
        self.si = SynapticIntelligence(model, lambda_si=lambda_si)
        self.consolidate_every = consolidate_every
        self.step_count = 0
    
    def train_step(self, loss_fn, optimizer, *args, **kwargs):
        """
        Perform a single training step with SI protection.
        
        Args:
            loss_fn: Callable that returns the task loss
            optimizer: The optimizer
            *args, **kwargs: Passed to loss_fn
        """
        optimizer.zero_grad()
        
        # Compute task loss + SI penalty
        task_loss = loss_fn(*args, **kwargs)
        si_penalty = self.si.penalty()
        total_loss = task_loss + si_penalty
        
        total_loss.backward()
        optimizer.step()
        
        # Update importance scores after each step
        self.si.update_importance()
        
        # Periodically consolidate
        self.step_count += 1
        if self.step_count % self.consolidate_every == 0:
            self.si.consolidate()
        
        return {
            'task_loss': task_loss.item(),
            'si_penalty': si_penalty.item(),
            'total_loss': total_loss.item()
        }
