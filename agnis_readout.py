"""
agnis_readout.py — Local, Backprop-Free Normalized Delta-Softmax Readout
=========================================================================
Implements a 1-layer local delta-rule readout over the top PredictiveHierarchy state.
Uses L2-normalized top hidden states h_norm to prevent magnitude explosion during
iterative settlement.

Update rule: dW = eta * h_norm^T (y - p)
Teaching signal: target_h = h + kappa * (y - p) @ W.T
"""

import math
import torch
import torch.nn.functional as F

class DeltaSoftmaxReadout:
    """
    Local, backprop-free softmax readout over the top hierarchy state.
    Uses L2-normalized hidden states h_norm for numerical stability.
    No gradients are backpropagated through the AGNIS core.
    """
    def __init__(self, d_hidden: int, vocab_size: int, device: torch.device, eta: float = 0.01, kappa: float = 0.1):
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.device = device
        self.eta = eta
        self.kappa = kappa
        
        # Small Gaussian initialization to break symmetry
        self.W = torch.randn(d_hidden, vocab_size, device=device) * 0.01
        self.b = torch.zeros(vocab_size, device=device)

    def normalize_h(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(h, dim=-1, eps=1e-8)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.normalize_h(h)
        return h_norm @ self.W + self.b

    def log_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(h), dim=-1)

    def update(self, h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h_norm = self.normalize_h(h)
        p = torch.softmax(self.logits(h), dim=-1)
        err = y_onehot - p
        
        self.W += self.eta * (h_norm.t() @ err) / h_norm.shape[0]
        self.b += self.eta * err.mean(dim=0)
        return err

    def teaching_signal(self, h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        Target for the hierarchy's top layer: h nudged along the readout error.
        One matmul back through W -- single layer, strictly local.
        Pass this to infer_and_learn(top_level_label=...).
        """
        h_norm = self.normalize_h(h)
        p = torch.softmax(self.logits(h), dim=-1)
        err = y_onehot - p
        return h + self.kappa * (err @ self.W.t())


def one_hot(ids, vocab_size: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(len(ids), vocab_size, device=device)
    x[torch.arange(len(ids)), torch.as_tensor(ids, device=device)] = 1.0
    return x
