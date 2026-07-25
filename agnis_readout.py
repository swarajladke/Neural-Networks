"""
agnis_readout.py — Local, Backprop-Free Dynamic Delta-Softmax Readout
======================================================================
Implements a 1-layer local delta-rule readout over the hierarchy representations.
Concatenates all layer states h_concat = [layer_0.x, layer_1.x] to combine
sensory input features with temporal context representations.

Update rule: dW = eta * h_norm^T (y - p)
Teaching signal: target_h = h + kappa * (y - p) @ W.T
"""

import math
import torch
import torch.nn.functional as F

class DeltaSoftmaxReadout:
    """
    Local, backprop-free softmax readout over hierarchy state representations.
    Combines sensory input states and temporal context states for dynamic prediction.
    No gradients are backpropagated through the AGNIS core.
    """
    def __init__(self, d_hidden: int, vocab_size: int, device: torch.device, eta: float = 0.2, kappa: float = 1.0):
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.device = device
        self.eta = eta
        self.kappa = kappa
        
        # Initialize W to produce dynamic logit scale
        self.W = torch.randn(d_hidden, vocab_size, device=device) * 0.1

    def normalize_h(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(h, dim=-1, eps=1e-8)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.normalize_h(h)
        return h_norm @ self.W

    def log_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(h), dim=-1)

    def update(self, h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h_norm = self.normalize_h(h)
        p = torch.softmax(self.logits(h), dim=-1)
        err = y_onehot - p
        
        # Pure dynamic weight matrix update
        self.W += self.eta * (h_norm.t() @ err) / h_norm.shape[0]
        return err

    def teaching_signal(self, h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        Target for the hierarchy's top layer: h nudged along the readout error.
        One matmul back through W -- single layer, strictly local.
        """
        h_norm = self.normalize_h(h)
        p = torch.softmax(self.logits(h), dim=-1)
        err = y_onehot - p
        return h + self.kappa * (err @ self.W.t()[:, -h.shape[1]:])


def get_hierarchy_state(hierarchy) -> torch.Tensor:
    """Concatenates all layer states [layer_0.x, layer_1.x] into a single feature vector."""
    states = [col.x for col in hierarchy.layers]
    return torch.cat(states, dim=-1)


def one_hot(ids, vocab_size: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(len(ids), vocab_size, device=device)
    x[torch.arange(len(ids)), torch.as_tensor(ids, device=device)] = 1.0
    return x
