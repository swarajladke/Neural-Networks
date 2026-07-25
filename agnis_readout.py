"""
agnis_readout.py — Local, Backprop-Free Delta-Softmax Readout
==============================================================
Implements a 1-layer local delta-rule readout over the top PredictiveHierarchy state.
Preserves zero-backprop integrity through the hierarchy using local teaching signals.

Update rule: dW = eta * h.T @ (y - p)
Teaching signal: target_h = h + kappa * (y - p) @ W.T
"""

import math
import torch

class DeltaSoftmaxReadout:
    """
    Local, backprop-free softmax readout over the top hierarchy state.
    Update rule: dW = eta * h^T (y - p)
    This is the gradient of cross-entropy w.r.t. a linear layer, but computed
    in a single step from locally available quantities (h, y, p).
    No gradients are backpropagated through the AGNIS core.
    """
    def __init__(self, d_hidden: int, vocab_size: int, device: torch.device, eta: float = 0.05):
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.device = device
        self.eta = eta
        
        self.W = torch.zeros(d_hidden, vocab_size, device=device)
        self.b = torch.zeros(vocab_size, device=device)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W + self.b

    def log_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(h), dim=-1)

    def update(self, h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.logits(h), dim=-1)
        err = y_onehot - p
        self.W += self.eta * (h.t() @ err) / h.shape[0]
        self.b += self.eta * err.mean(dim=0)
        return err

    def teaching_signal(self, h: torch.Tensor, y_onehot: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """
        Target for the hierarchy's top layer: h nudged along the readout error.
        One matmul back through W -- single layer, strictly local.
        Pass this to infer_and_learn(top_level_label=...).
        """
        p = torch.softmax(self.logits(h), dim=-1)
        err = y_onehot - p
        return h + kappa * (err @ self.W.t())


def one_hot(ids, vocab_size: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(len(ids), vocab_size, device=device)
    x[torch.arange(len(ids)), torch.as_tensor(ids, device=device)] = 1.0
    return x
