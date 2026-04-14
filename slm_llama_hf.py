"""
Minimal LLaMA-style SLM trainer + HF exporter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import sentencepiece as spm
from tokenizers import Tokenizer as HFTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)


class BPETokenizer:
    def __init__(self, tokenizer_json: str):
        self.tokenizer = HFTokenizer.from_file(tokenizer_json)
        self.vocab_size = self.tokenizer.get_vocab_size()
        vocab = self.tokenizer.get_vocab()
        self.pad_id = vocab.get("<pad>", 0)
        self.bos_id = vocab.get("<s>", 1)
        self.eos_id = vocab.get("</s>", 2)
        self.unk_id = vocab.get("<unk>", 3)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

@dataclass
class LlamaConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    n_kv_head: int = 8
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def intermediate_size(self) -> int:
        # LLaMA uses ~2.6x hidden size
        return int(2.6 * self.n_embd)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, D]
    B, T, H, D = x.shape
    x = x.view(B, T, H, D // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    cos = cos[:T].unsqueeze(0).unsqueeze(2).to(x.device)
    sin = sin[:T].unsqueeze(0).unsqueeze(2).to(x.device)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).reshape(B, T, H, D)
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, (cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)))
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def _ensure_rope(self):
        if self.rope_cos.numel() == 0:
            cos, sin = precompute_rope(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        self._ensure_rope()
        qkv = self.qkv(x)
        q, k, v = torch.split(
            qkv,
            [self.cfg.n_head * self.cfg.head_dim, self.cfg.n_kv_head * self.cfg.head_dim, self.cfg.n_kv_head * self.cfg.head_dim],
            dim=2,
        )
        q = q.view(B, T, self.cfg.n_head, self.cfg.head_dim)
        k = k.view(B, T, self.cfg.n_kv_head, self.cfg.head_dim)
        v = v.view(B, T, self.cfg.n_kv_head, self.cfg.head_dim)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Expand KV heads to match Q heads if needed
        if self.cfg.n_kv_head != self.cfg.n_head:
            repeat = self.cfg.n_head // self.cfg.n_kv_head
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)
        mask = self.mask[:T, :T].to(att.device)
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        hidden = cfg.intermediate_size
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LlamaLM(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


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


def save_hf(model: LlamaLM, tokenizer_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state = export_hf_state_dict(model)
    torch.save(state, os.path.join(out_dir, "pytorch_model.bin"))
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": model.cfg.vocab_size,
        "hidden_size": model.cfg.n_embd,
        "intermediate_size": model.cfg.intermediate_size,
        "num_hidden_layers": model.cfg.n_layer,
        "num_attention_heads": model.cfg.n_head,
        "num_key_value_heads": model.cfg.n_kv_head,
        "rms_norm_eps": model.cfg.rms_norm_eps,
        "max_position_embeddings": model.cfg.max_seq_len,
        "rope_theta": model.cfg.rope_theta,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "unk_token_id": 3,
        "tie_word_embeddings": False,
        "hidden_act": "silu"
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    if tokenizer_path:
        if tokenizer_path.endswith(".model"):
            with open(tokenizer_path, "rb") as src, open(os.path.join(out_dir, "tokenizer.model"), "wb") as dst:
                dst.write(src.read())
        elif tokenizer_path.endswith(".json"):
            with open(tokenizer_path, "rb") as src, open(os.path.join(out_dir, "tokenizer.json"), "wb") as dst:
                dst.write(src.read())
            tokenizer_config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "model_max_length": model.cfg.max_seq_len,
            }
            with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, indent=2)
            special_tokens_map = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
            with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
                json.dump(special_tokens_map, f, indent=2)


def export_hf_state_dict(model: LlamaLM) -> dict:
    sd = {}
    sd["model.embed_tokens.weight"] = model.tok_emb.weight.detach().cpu()
    sd["model.norm.weight"] = model.norm.weight.detach().cpu()
    sd["lm_head.weight"] = model.head.weight.detach().cpu()

    n_head = model.cfg.n_head
    n_kv = model.cfg.n_kv_head
    hd = model.cfg.head_dim
    q_size = n_head * hd
    k_size = n_kv * hd
    v_size = n_kv * hd

    for i, block in enumerate(model.blocks):
        prefix = f"model.layers.{i}."
        sd[prefix + "input_layernorm.weight"] = block.ln1.weight.detach().cpu()
        sd[prefix + "post_attention_layernorm.weight"] = block.ln2.weight.detach().cpu()

        qkv = block.attn.qkv.weight.detach().cpu()
        q_w = qkv[:q_size, :]
        k_w = qkv[q_size:q_size + k_size, :]
        v_w = qkv[q_size + k_size:q_size + k_size + v_size, :]
        sd[prefix + "self_attn.q_proj.weight"] = q_w
        sd[prefix + "self_attn.k_proj.weight"] = k_w
        sd[prefix + "self_attn.v_proj.weight"] = v_w
        sd[prefix + "self_attn.o_proj.weight"] = block.attn.proj.weight.detach().cpu()

        sd[prefix + "mlp.gate_proj.weight"] = block.mlp.w1.weight.detach().cpu()
        sd[prefix + "mlp.up_proj.weight"] = block.mlp.w3.weight.detach().cpu()
        sd[prefix + "mlp.down_proj.weight"] = block.mlp.w2.weight.detach().cpu()

    return sd


def train(
    data_path: str,
    tokenizer_path: str,
    tokenizer_type: str,
    steps: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    lr: float,
    save_hf_dir: str | None,
    load_state: str | None = None
):
    if tokenizer_type == "spm":
        tokenizer = SPTokenizer(tokenizer_path)
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer(tokenizer_path)
    else:
        raise ValueError("tokenizer_type must be 'spm' or 'bpe'")
    text = load_text(data_path)
    data = tokenizer.encode(text)
    if len(data) < block_size + 2:
        raise ValueError("Dataset too small for block size.")

    cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_kv_head=n_head
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaLM(cfg).to(device)
    if load_state:
        if not os.path.exists(load_state):
            raise FileNotFoundError(f"State dict not found: {load_state}")
        state = torch.load(load_state, map_location="cpu")
        model.load_state_dict(state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    stream = stream_batches(data, block_size, batch_size, device)

    if steps > 0:
        for step in range(1, steps + 1):
            x, y = next(stream)
            _logits, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | Device: {device.type}")

    if save_hf_dir:
        save_hf(model, tokenizer_path, save_hf_dir)
        print(f"Saved HF model to {save_hf_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="multilang_corpus.txt")
    parser.add_argument("--tokenizer_path", type=str, default="slm_spm.model")
    parser.add_argument("--tokenizer_type", type=str, default="spm")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--embd", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_hf", type=str, default="slm_llama_hf")
    parser.add_argument("--load_state", type=str, default="")
    args = parser.parse_args()

    train(
        data_path=args.data,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        lr=args.lr,
        save_hf_dir=args.save_hf,
        load_state=args.load_state if args.load_state else None
    )


if __name__ == "__main__":
    main()
