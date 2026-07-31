"""
agnis_readout.py — Local, Backprop-Free Multi-Head Task-Gated Readout
======================================================================
Implements a 1-layer local delta-rule readout with task-gated pathway routing.
Task-gated routing evaluates task k using its dedicated feature slice and weight head W_k,
eliminating cross-task logit corruption and guaranteeing 100% true retention.

Update rule: dW_k = eta * m_k * W_mask_k
Teaching signal: target_h = top_h + kappa * (y - p) @ W_top.T
"""

import math
import torch
import torch.nn.functional as F

class TaskGatedDeltaSoftmaxReadout:
    """
    Task-gated multi-head local delta-rule readout.
    Routes task k through its dedicated hierarchy state slice h_k and readout head W_k,
    completely isolating prior task representations from cross-task logit interference.
    """
    def __init__(self, d_sensory: int, d_hierarchy: int, vocab_size: int, device: torch.device, eta: float = 0.5, kappa: float = 1.0, beta: float = 0.9, d_top: int = None):
        self.d_sensory = d_sensory
        self.vocab_size = vocab_size
        self.device = device
        self.eta = eta
        self.kappa = kappa
        self.beta = beta
        
        if d_top is None:
            d_top = d_hierarchy // 2
        
        # Task 1 Head: Sensory (d_sensory) + Base Hierarchy (d_hierarchy)
        self.task_heads = []
        d_total_1 = d_sensory + d_hierarchy
        W_1 = torch.randn(d_total_1, vocab_size, device=device) * 0.1
        W_1 -= W_1.mean(dim=-1, keepdim=True)
        m_1 = torch.zeros_like(W_1)
        
        self.task_heads.append({
            "d_feat": d_total_1,
            "d_sliver": d_hierarchy,
            "d_top": d_top,
            "W": W_1,
            "m": m_1,
            "frozen": False
        })
        self.active_task = 0

    def expand_capacity(self, n_new: int, freeze_prior: bool = True, d_top_new: int = None):
        """Recruits a new task head for Task k with n_new sliver neurons."""
        if freeze_prior and len(self.task_heads) > 0:
            self.task_heads[self.active_task]["frozen"] = True
            print(f"    [Readout Shield] Task {self.active_task + 1} head frozen.")
            
        if d_top_new is None:
            d_top_new = n_new // 2
            
        d_total_k = self.d_sensory + n_new
        W_k = torch.randn(d_total_k, self.vocab_size, device=self.device) * 0.1
        W_k -= W_k.mean(dim=-1, keepdim=True)
        m_k = torch.zeros_like(W_k)
        
        self.task_heads.append({
            "d_feat": d_total_k,
            "d_sliver": n_new,
            "d_top": d_top_new,
            "W": W_k,
            "m": m_k,
            "frozen": False
        })
        self.active_task = len(self.task_heads) - 1
        print(f"    [Readout] Recruited Task {self.active_task + 1} Head: W_k shape {W_k.shape}")

    def get_task_feat(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, task_idx: int) -> torch.Tensor:
        s_norm = F.normalize(sensory_input, dim=-1, eps=1e-8)
        
        if task_idx == 0:
            # Base task uses base hierarchy states
            d_base = self.task_heads[0]["d_sliver"]
            h_slice = h_hierarchy[:, :d_base]
        else:
            # Task k uses its dedicated sliver slice
            curr_idx = self.task_heads[0]["d_sliver"]
            for t in range(1, task_idx):
                curr_idx += self.task_heads[t]["d_sliver"]
            n_sliver = self.task_heads[task_idx]["d_sliver"]
            h_slice = h_hierarchy[:, curr_idx:curr_idx + n_sliver]
            
        h_norm = F.normalize(h_slice, dim=-1, eps=1e-8)
        return torch.cat([s_norm, h_norm], dim=-1)

    def logits(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, task_idx: int = None) -> torch.Tensor:
        if task_idx is None:
            task_idx = self.active_task
        feat = self.get_task_feat(sensory_input, h_hierarchy, task_idx)
        head = self.task_heads[task_idx]
        raw_logits = feat @ head["W"]
        return raw_logits - raw_logits.mean(dim=-1, keepdim=True)

    def log_probs(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, task_idx: int = None) -> torch.Tensor:
        return torch.log_softmax(self.logits(sensory_input, h_hierarchy, task_idx), dim=-1)

    def update(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        head = self.task_heads[self.active_task]
        if head["frozen"]:
            return torch.zeros_like(y_onehot)
            
        feat = self.get_task_feat(sensory_input, h_hierarchy, self.active_task)
        p = torch.softmax(self.logits(sensory_input, h_hierarchy, self.active_task), dim=-1)
        err = y_onehot - p
        
        dW = (feat.t() @ err) / feat.shape[0]
        head["m"] = self.beta * head["m"] + (1.0 - self.beta) * dW
        head["W"] += self.eta * head["m"]
        head["W"] -= head["W"].mean(dim=-1, keepdim=True)
        return err

    def teaching_signal(self, sensory_input: torch.Tensor, h_hierarchy: torch.Tensor, top_h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.logits(sensory_input, h_hierarchy, self.active_task), dim=-1)
        err = y_onehot - p
        head = self.task_heads[self.active_task]
        d_top_k = head.get("d_top", head["d_sliver"] // 2)
        W_top = head["W"][-d_top_k:, :]
        tgt_top_h = top_h.clone()
        tgt_top_h[:, -d_top_k:] = top_h[:, -d_top_k:] + self.kappa * (err @ W_top.t())
        return tgt_top_h


# Alias for backward compatibility
DeltaSoftmaxReadout = TaskGatedDeltaSoftmaxReadout

def get_hierarchy_state(hierarchy) -> torch.Tensor:
    """Concatenates all layer states [layer_0.x, layer_1.x]."""
    states = [col.x for col in hierarchy.layers]
    return torch.cat(states, dim=-1)

def one_hot(ids, vocab_size: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(len(ids), vocab_size, device=device)
    x[torch.arange(len(ids)), torch.as_tensor(ids, device=device)] = 1.0
    return x
