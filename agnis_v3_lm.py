"""
AGNIS V3: Small Language Model Prototype (Continual Runtime Loop)

Design goals:
- Small transformer LM (causal) with byte-level tokenizer.
- Streaming, continual training loop (no epochs).
- Lightweight replay buffer to reduce forgetting.
- Simple stability/plasticity modulation using loss EMA.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: List[int]) -> str:
        return bytes(ids).decode("utf-8", errors="ignore")


@dataclass
class LMConfig:
    vocab_size: int = 256
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 2048, 2048)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def maybe_grow_depth(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    max_layers: int,
    device: torch.device
) -> bool:
    if model.cfg.n_layer >= max_layers:
        return False
    new_block = Block(model.cfg.n_embd, model.cfg.n_head, model.cfg.dropout).to(device)
    model.blocks.append(new_block)
    model.cfg.n_layer += 1
    # Add new params to optimizer with same hyperparams
    base = optimizer.param_groups[0].copy()
    base.pop("params", None)
    optimizer.add_param_group({"params": new_block.parameters(), **base})
    model.train()
    return True


class PriorityReplayBuffer:
    def __init__(self, max_batches: int = 200):
        self.max_batches = max_batches
        self.buffer: Deque[Tuple[torch.Tensor, torch.Tensor, float]] = deque(maxlen=max_batches)

    def add(self, x: torch.Tensor, y: torch.Tensor, score: float):
        self.buffer.append((x.detach().cpu().clone(), y.detach().cpu().clone(), float(score)))

    def sample(self, k: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.buffer:
            return []
        k = min(k, len(self.buffer))
        items = list(self.buffer)
        weights = [max(1e-6, s) for _x, _y, s in items]
        chosen = random.choices(items, weights=weights, k=k)
        return [(x, y) for x, y, _s in chosen]


class AGNISController:
    """
    Lightweight controller to modulate replay and learning rate from drift signals.
    """

    def __init__(
        self,
        replay_ratio: float = 0.2,
        replay_min: float = 0.05,
        replay_max: float = 0.5,
        ema: float = 0.95
    ):
        self.replay_ratio = replay_ratio
        self.replay_min = replay_min
        self.replay_max = replay_max
        self.ema = ema
        self.loss_ema = None

    def update(self, loss_val: float) -> float:
        if self.loss_ema is None:
            self.loss_ema = loss_val
        else:
            self.loss_ema = self.ema * self.loss_ema + (1 - self.ema) * loss_val
        if loss_val > self.loss_ema * 1.05:
            self.replay_ratio = min(self.replay_max, self.replay_ratio + 0.05)
        else:
            self.replay_ratio = max(self.replay_min, self.replay_ratio - 0.02)
        return self.replay_ratio


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def stream_batches(data: List[int], block_size: int, batch_size: int, device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    data_len = len(data)
    while True:
        ix = torch.randint(0, data_len - block_size - 1, (batch_size,))
        x = torch.stack([torch.tensor(data[i:i + block_size]) for i in ix]).to(device)
        y = torch.stack([torch.tensor(data[i + 1:i + block_size + 1]) for i in ix]).to(device)
        yield x, y


def split_domains(
    data: List[int],
    num_domains: int,
    eval_fraction: float = 0.1
) -> List[Dict[str, List[int]]]:
    if num_domains <= 0:
        raise ValueError("num_domains must be >= 1")
    if not (0.0 < eval_fraction < 0.5):
        raise ValueError("eval_fraction must be between 0 and 0.5")

    total = len(data)
    chunk = max(1, total // num_domains)
    domains = []
    for i in range(num_domains):
        start = i * chunk
        end = total if i == num_domains - 1 else (i + 1) * chunk
        slice_ids = data[start:end]
        if len(slice_ids) < 2:
            continue
        eval_len = max(1, int(len(slice_ids) * eval_fraction))
        train_ids = slice_ids[:-eval_len]
        eval_ids = slice_ids[-eval_len:]
        domains.append({"id": i, "train": train_ids, "eval": eval_ids})
    return domains


def evaluate_model(
    model: TransformerLM,
    eval_sets: List[List[int]],
    block_size: int,
    batch_size: int,
    device: torch.device,
    batches: int = 5
) -> List[float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for ids in eval_sets:
            if len(ids) < block_size + 2:
                losses.append(float("nan"))
                continue
            stream = stream_batches(ids, block_size, batch_size, device)
            batch_losses = []
            for _ in range(batches):
                x, y = next(stream)
                _logits, loss = model(x, y)
                batch_losses.append(float(loss.item()))
            losses.append(sum(batch_losses) / max(1, len(batch_losses)))
    model.train()
    return losses


def generate_text(
    model: TransformerLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device
) -> str:
    if max_new_tokens <= 0:
        return prompt
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    block_size = model.cfg.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    filt = torch.full_like(logits, float("-inf"))
                    filt.scatter_(1, ix, v)
                    logits = filt
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    out = tokenizer.decode(idx[0].tolist())
    model.train()
    return out


def safe_print(text: str):
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def save_retention_reports(
    eval_history: List[List[float]],
    csv_path: str,
    png_path: str
):
    if not eval_history:
        return
    rows = len(eval_history)
    cols = max(len(r) for r in eval_history)
    matrix = np.full((rows, cols), np.nan)
    for i, row in enumerate(eval_history):
        for j, val in enumerate(row):
            matrix[i, j] = val

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["after_domain"] + [f"domain_{i+1}" for i in range(cols)]
            writer.writerow(header)
            for i in range(rows):
                formatted = [f"{v:.6f}" if not np.isnan(v) else "" for v in matrix[i]]
                writer.writerow([i + 1] + formatted)

    if png_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available; skipping retention PNG.")
            return
        fig_w = max(6, cols * 0.6)
        fig_h = max(4, rows * 0.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xlabel("Measured Domain")
        ax.set_ylabel("After Domain")
        ax.set_xticks(range(cols))
        ax.set_xticklabels([f"D{i+1}" for i in range(cols)], rotation=45, ha="right")
        ax.set_yticks(range(rows))
        ax.set_yticklabels([f"After D{i+1}" for i in range(rows)])
        fig.colorbar(im, ax=ax, label="Loss (Lower is Better)")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)


def train_lm(
    data_path: str,
    steps: int = 500,
    batch_size: int = 8,
    lr: float = 3e-4,
    replay_ratio: float = 0.2,
    replay_max: int = 200,
    cfg: LMConfig | None = None,
    use_controller: bool = True
):
    cfg = cfg or LMConfig()
    tokenizer = ByteTokenizer()

    text = load_text(data_path)
    data = tokenizer.encode(text)
    if len(data) < cfg.block_size + 2:
        raise ValueError("Dataset too small for block size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    replay = PriorityReplayBuffer(max_batches=replay_max)
    controller = AGNISController(replay_ratio=replay_ratio)
    stream = stream_batches(data, cfg.block_size, batch_size, device)

    for step in range(1, steps + 1):
        x, y = next(stream)
        logits, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Replay (salience-weighted by loss)
        lval = loss.item()
        replay.add(x, y, score=lval)
        extra = int(replay_ratio * batch_size)
        if extra > 0:
            for rx, ry in replay.sample(1):
                rx = rx.to(device)
                ry = ry.to(device)
                _, rloss = model(rx, ry)
                opt.zero_grad()
                rloss.backward()
                opt.step()

        # Controller modulation
        if use_controller:
            replay_ratio = controller.update(lval)
            for group in opt.param_groups:
                group["lr"] = lr * (0.7 + 0.6 * replay_ratio)

        if step % 50 == 0:
            print(f"Step {step}/{steps} | Loss: {lval:.4f} | Replay: {replay_ratio:.2f} | Device: {device.type}")
    return model, tokenizer, device


def train_lm_continual(
    data_path: str | None,
    domains: int = 5,
    steps_per_domain: int = 200,
    batch_size: int = 8,
    lr: float = 3e-4,
    replay_ratio: float = 0.2,
    replay_max: int = 200,
    eval_fraction: float = 0.1,
    eval_batches: int = 5,
    cfg: LMConfig | None = None,
    use_controller: bool = True,
    save_retention: bool = True,
    retention_csv: str = "agnis_v3_lm_retention.csv",
    retention_png: str = "agnis_v3_lm_retention.png",
    shuffle_domains: bool = False,
    seed: int = 42,
    domain_files: List[str] | None = None,
    auto_grow: bool = False,
    grow_max_layers: int = 12,
    grow_patience: int = 2,
    grow_min_delta: float = 0.01
):
    cfg = cfg or LMConfig()
    tokenizer = ByteTokenizer()

    domain_sets = []
    if domain_files:
        for idx, path in enumerate(domain_files):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Domain file not found: {path}")
            text = load_text(path)
            data = tokenizer.encode(text)
            if len(data) < cfg.block_size + 2:
                raise ValueError(f"Domain file too small for block size: {path}")
            chunks = split_domains(data, 1, eval_fraction=eval_fraction)
            if not chunks:
                raise ValueError(f"Failed to split domain file: {path}")
            # Each file is a single domain
            entry = chunks[0]
            entry["id"] = idx
            entry["path"] = path
            domain_sets.append(entry)
    else:
        if data_path is None:
            raise ValueError("data_path is required when domain_files is not set.")
        text = load_text(data_path)
        data = tokenizer.encode(text)
        if len(data) < cfg.block_size + 2:
            raise ValueError("Dataset too small for block size.")
        domain_sets = split_domains(data, domains, eval_fraction=eval_fraction)
        if not domain_sets:
            raise ValueError("No domains were created. Reduce block size or domains.")

    rng = random.Random(seed)
    domains_by_id = {d["id"]: d for d in domain_sets}
    ordered_ids = [d["id"] for d in domain_sets]
    if shuffle_domains:
        rng.shuffle(domain_sets)
        order = [d["id"] for d in domain_sets]
        print(f"Shuffled domain order: {order}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    replay = PriorityReplayBuffer(max_batches=replay_max)
    controller = AGNISController(replay_ratio=replay_ratio)

    eval_history: List[List[float]] = []
    seen_ids: set[int] = set()
    best_mean = None
    plateau_count = 0

    for d_idx, domain in enumerate(domain_sets):
        seen_ids.add(domain["id"])
        train_ids = domain["train"]
        eval_ids = domain["eval"]
        if len(train_ids) < cfg.block_size + 2:
            print(f"Domain {d_idx}: skipped (too small after split)")
            continue

        stream = stream_batches(train_ids, cfg.block_size, batch_size, device)
        label = f"Domain {d_idx+1}/{len(domain_sets)}"
        if "path" in domain:
            label += f" | {os.path.basename(domain['path'])}"
        print(f"\n{label} | Train tokens: {len(train_ids)} | Eval tokens: {len(eval_ids)}")

        for step in range(1, steps_per_domain + 1):
            x, y = next(stream)
            _logits, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            lval = loss.item()
            replay.add(x, y, score=lval)
            extra = int(replay_ratio * batch_size)
            if extra > 0:
                for rx, ry in replay.sample(1):
                    rx = rx.to(device)
                    ry = ry.to(device)
                    _rlogits, rloss = model(rx, ry)
                    opt.zero_grad()
                    rloss.backward()
                    opt.step()

            if use_controller:
                replay_ratio = controller.update(lval)
                for group in opt.param_groups:
                    group["lr"] = lr * (0.7 + 0.6 * replay_ratio)

            if step % max(1, steps_per_domain // 4) == 0:
                print(f"  Step {step}/{steps_per_domain} | Loss: {lval:.4f} | Replay: {replay_ratio:.2f}")

        # Evaluate retention across all domains (unseen marked as NaN)
        eval_sets = []
        for did in ordered_ids:
            if did in seen_ids:
                eval_sets.append(domains_by_id[did]["eval"])
            else:
                eval_sets.append([])
        losses = evaluate_model(
            model=model,
            eval_sets=eval_sets,
            block_size=cfg.block_size,
            batch_size=batch_size,
            device=device,
            batches=eval_batches
        )
        eval_history.append(losses)
        mean_loss = sum([v for v in losses if v == v]) / max(1, len([v for v in losses if v == v]))
        print(f"  Retention after Domain {d_idx+1}: mean loss = {mean_loss:.4f}")
        print(f"  Per-domain: {', '.join([f'{v:.4f}' if v == v else 'nan' for v in losses])}")

        if auto_grow:
            if best_mean is None or (best_mean - mean_loss) > grow_min_delta:
                best_mean = mean_loss
                plateau_count = 0
            else:
                plateau_count += 1
            if plateau_count >= max(1, grow_patience):
                grew = maybe_grow_depth(
                    model=model,
                    optimizer=opt,
                    max_layers=grow_max_layers,
                    device=device
                )
                if grew:
                    print(f"  [AutoGrow] Added block. New depth: {model.cfg.n_layer}")
                else:
                    print("  [AutoGrow] Max depth reached; no growth.")
                plateau_count = 0

    if save_retention:
        save_retention_reports(eval_history, retention_csv, retention_png)
        print(f"\nSaved retention CSV: {retention_csv}")
        print(f"Saved retention PNG: {retention_png}")

    return eval_history, model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="agi_training_data.txt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--embd", type=int, default=256)
    parser.add_argument("--replay_ratio", type=float, default=0.2)
    parser.add_argument("--replay_max", type=int, default=200)
    parser.add_argument("--continual", action="store_true", help="Run sequential domain continual-learning test.")
    parser.add_argument("--domains", type=int, default=5)
    parser.add_argument("--steps_per_domain", type=int, default=200)
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--eval_batches", type=int, default=5)
    parser.add_argument("--retention_csv", type=str, default="agnis_v3_lm_retention.csv")
    parser.add_argument("--retention_png", type=str, default="agnis_v3_lm_retention.png")
    parser.add_argument("--no_save_retention", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Generate text after training.")
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--gen_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--shuffle_domains", action="store_true", help="Shuffle domain order for baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain_files", type=str, default="", help="Comma-separated domain text files for continual mode.")
    parser.add_argument("--auto_grow", action="store_true", help="Auto-grow model depth based on retention plateau.")
    parser.add_argument("--grow_max_layers", type=int, default=12)
    parser.add_argument("--grow_patience", type=int, default=2)
    parser.add_argument("--grow_min_delta", type=float, default=0.01)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    cfg = LMConfig(
        vocab_size=256,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        dropout=0.1
    )
    domain_files = [p.strip() for p in args.domain_files.split(",") if p.strip()]
    if args.continual:
        _history, model, tokenizer, device = train_lm_continual(
            data_path=args.data if not domain_files else None,
            domains=args.domains,
            steps_per_domain=args.steps_per_domain,
            batch_size=args.batch_size,
            lr=args.lr,
            replay_ratio=args.replay_ratio,
            replay_max=args.replay_max,
            eval_fraction=args.eval_fraction,
            eval_batches=args.eval_batches,
            cfg=cfg,
            save_retention=not args.no_save_retention,
            retention_csv=args.retention_csv,
            retention_png=args.retention_png,
            shuffle_domains=args.shuffle_domains,
            seed=args.seed,
            domain_files=domain_files if domain_files else None,
            auto_grow=args.auto_grow,
            grow_max_layers=args.grow_max_layers,
            grow_patience=args.grow_patience,
            grow_min_delta=args.grow_min_delta
        )
    else:
        model, tokenizer, device = train_lm(
            data_path=args.data,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            replay_ratio=args.replay_ratio,
            replay_max=args.replay_max,
            cfg=cfg
        )

    if args.generate:
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.gen_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print("\n=== GENERATED ===")
        safe_print(text)


if __name__ == "__main__":
    main()
