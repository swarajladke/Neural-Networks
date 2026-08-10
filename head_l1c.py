"""
head_l1c.py
===========

Single canonical definition of HeadL1c imported by all scripts (L2).

HeadL1c: Cosine-similarity linear classifier with normalized weights and fixed temperature scale=10.0.

Canonical Hyperparameters:
- lr = 0.01
- max_epochs = 50
- scale = 10.0
- weight_decay = 1e-4
- zero early stopping on evaluation sets inside CV / diagnostics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CANONICAL_LR = 0.01
CANONICAL_EPOCHS = 50
CANONICAL_SCALE = 10.0
CANONICAL_WEIGHT_DECAY = 1e-4
SEEDS = [42, 43, 44, 45, 46]


def py_mean(vals):
    return float(sum(vals)) / float(len(vals))


def py_std(vals):
    m = py_mean(vals)
    return float((sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5)


class HeadL1c(nn.Module):
    def __init__(self, in_features=960, out_features=100, scale=CANONICAL_SCALE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        w = torch.randn(out_features, in_features)
        w = F.normalize(w, dim=-1)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.scale * (x_norm @ w_norm.T)


def eval_headl1c_canonical(tr_x, train_y, eval_x, eval_y, seeds=SEEDS, lr=CANONICAL_LR, epochs=CANONICAL_EPOCHS, scale=CANONICAL_SCALE):
    """
    Evaluates HeadL1c across 5 seeds using fixed epoch budget with NO early stopping on evaluation set (L1, L2).
    """
    accs = []
    d = tr_x.shape[1]
    out_classes = int(torch.max(train_y).item()) + 1

    for seed in seeds:
        torch.manual_seed(seed)
        model = HeadL1c(in_features=d, out_features=out_classes, scale=scale)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CANONICAL_WEIGHT_DECAY)

        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(tr_x)
            loss = F.cross_entropy(logits, train_y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(eval_x).argmax(dim=1)
            accs.append((preds == eval_y).float().mean().item() * 100.0)

    return py_mean(accs), py_std(accs)
