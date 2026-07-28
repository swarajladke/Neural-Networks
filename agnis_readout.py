"""
agnis_readout.py — Local, Backprop-Free Dynamic Delta-Softmax Readout with Isolated Masking
=============================================================================================
Implements a 1-layer local delta-rule readout over hierarchy representations.
Performs row-isolated column-centering so prior task readout rows remain perfectly centered
and frozen, preventing cross-layer logit drift during continual learning.

Update rule: m = beta * m + (1-beta) * dW; W += eta * m * W_mask
Teaching signal: target_h = top_h + kappa * (y - p) @ W_top.T
"""

import math
import torch
import torch.nn.functional as F

class DeltaSoftmaxReadout:
    """
    Local, backprop-free softmax readout over hierarchy state representations.
    Combines orthogonal sensory features with temporal context states for dynamic prediction.
    Supports W_mask to freeze prior task readout weights during continual learning.
    """
    def __init__(self, d_sensory: int, d_hierarchy: int, vocab_size: int, device: torch.device, eta: float = 0.5, kappa: float = 1.0, beta: float = 0.9):
        self.d_sensory = d_sensory
        self.d_hierarchy = d_hierarchy
        self.d_total = d_sensory + d_hierarchy
        self.vocab_size = vocab_size
        self.device = device
        self.eta = eta
        self.kappa = kappa
        self.beta = beta
        
        # Initialize W (d_total x vocab_size), W_mask, and momentum buffer m_W
        self.W = torch.randn(self.d_total, vocab_size, device=device) * 0.1
        self.W -= self.W.mean(dim=-1, keepdim=True)
        self.W_mask = torch.ones(self.d_total, 1, device=device)
        self.m_W = torch.zeros_like(self.W)

    def freeze_prior_rows(self):
        """Freezes all existing readout rows so prior task readout weights are immutable."""
        # Ensure pre-existing rows are centered before freezing
        self.W -= self.W.mean(dim=-1, keepdim=True)
        self.W_mask.zero_()
        print(f"    [Readout Shield] Frozen prior {self.d_total} readout rows.")

    def expand_capacity(self, n_new: int, freeze_prior: bool = True):
        """Expands readout weight matrix W and unmasks only the newly recruited rows."""
        if freeze_prior:
            self.freeze_prior_rows()
            
        new_W = torch.randn(self.d_total + n_new, self.vocab_size, device=self.device) * 0.1
        new_W[:self.d_total, :] = self.W.data
        new_W[self.d_total:, :] -= new_W[self.d_total:, :].mean(dim=-1, keepdim=True)
        self.W = new_W
        
        new_W_mask = torch.zeros(self.d_total + n_new, 1, device=self.device)
        new_W_mask[:self.d_total, :] = self.W_mask.data
        new_W_mask[self.d_total:, :] = 1.0  # Only newly recruited rows are trainable
        self.W_mask = new_W_mask
        
        self.m_W = torch.zeros_like(self.W)
        
        self.d_hierarchy += n_new
        self.d_total += n_new
        print(f"    [Readout] Expanded capacity: W shape {self.W.shape} (Trainable rows = {n_new})")

    def combine_and_normalize(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor) -> torch.Tensor:
        s_norm = F.normalize(sensory_input, dim=-1, eps=1e-8)
        h_norm = F.normalize(h_hierarchy, dim=-1, eps=1e-8)
        return torch.cat([s_norm, h_norm], dim=-1)

    def logits(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor) -> torch.Tensor:
        feat = self.combine_and_normalize(sensory_input, h_hierarchy)
        raw_logits = feat @ self.W
        return raw_logits - raw_logits.mean(dim=-1, keepdim=True)

    def log_probs(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(sensory_input, h_hierarchy), dim=-1)

    def update(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        feat = self.combine_and_normalize(sensory_input, h_hierarchy)
        p = torch.softmax(self.logits(sensory_input, h_hierarchy), dim=-1)
        err = y_onehot - p
        
        dW = (feat.t() @ err) / feat.shape[0]
        dW_masked = dW * self.W_mask
        
        # Momentum update for smooth convergence
        self.m_W = self.beta * self.m_W + (1.0 - self.beta) * dW_masked
        self.W += self.eta * self.m_W
        
        # Isolated row-wise centering: center ONLY the active trainable rows
        active_indices = (self.W_mask.squeeze(-1) > 0.0)
        if active_indices.any():
            self.W[active_indices, :] -= self.W[active_indices, :].mean(dim=-1, keepdim=True)
            
        return err

    def teaching_signal(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, top_h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.logits(sensory_input, h_hierarchy), dim=-1)
        err = y_onehot - p
        d_top = top_h.shape[1]
        W_top = self.W[-d_top:, :]
        return top_h + self.kappa * (err @ W_top.t())


def get_hierarchy_state(hierarchy) -> torch.Tensor:
    """Concatenates all layer states [layer_0.x, layer_1.x]."""
    states = [col.x for col in hierarchy.layers]
    return torch.cat(states, dim=-1)


def one_hot(ids, vocab_size: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(len(ids), vocab_size, device=device)
    x[torch.arange(len(ids)), torch.as_tensor(ids, device=device)] = 1.0
    return x
