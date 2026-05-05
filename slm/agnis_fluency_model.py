"""
slm/agnis_fluency_model.py
==========================
Fluency-oriented AGNIS language model.

Design goals:
  - keep AGNIS as the frozen predictive core
  - use a stronger language readout path than nearest-neighbor decoding
  - make training and inference use the exact same fusion logic

Architecture:
  token ids
    -> embedding
    -> AGNIS core prediction
    -> fusion features [emb, core, emb*core, emb-core]
    -> projection MLP
    -> LM head
    -> vocab logits
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agnis_v4_core import PredictiveHierarchy

try:
    from tokenizers import Tokenizer as HFTokenizer
    _HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    _HF_TOKENIZERS_AVAILABLE = False


DEFAULT_EMBED_DIM = 110
DEFAULT_VOCAB_SIZE = 4096
DEFAULT_CONTEXT_SIZE = 64
DEFAULT_HF_TOKENIZER = "slm_bpe_tokenizer_en.json"
FALLBACK_HF_TOKENIZER = "slm_bpe_tokenizer.json"


class AGNISFluencyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        fusion_hidden_dim: int | None = None,
        dropout: float = 0.15,
        device: str = "cpu",
        tie_weights: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.tie_weights = tie_weights
        self.dropout = dropout
        self._tokenizer = None

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, 1024, embed_dim], device=str(self.device)
        )

        hidden_dim = fusion_hidden_dim or (embed_dim * 4)
        self.fusion_norm = nn.LayerNorm(embed_dim * 4).to(self.device)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ).to(self.device)
        self.out_norm = nn.LayerNorm(embed_dim).to(self.device)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(self.device)
        self.tie_output_weights()

    def tie_output_weights(self) -> None:
        if self.tie_weights and self.lm_head.weight.shape == self.embedding.weight.shape:
            self.lm_head.weight = self.embedding.weight

    def load_core_checkpoint(self, path: str, tokenizer_path: str | None = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Core checkpoint not found: {path}")

        print(f"[Loading Core] {path}")
        self.hierarchy.load_checkpoint(path)

        detected_idim = self.hierarchy.layers[0].input_dim
        if detected_idim != self.embed_dim:
            self.embed_dim = detected_idim
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim).to(self.device)

            hidden_dim = self.proj[0].out_features
            self.fusion_norm = nn.LayerNorm(self.embed_dim * 4).to(self.device)
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.embed_dim),
                nn.Dropout(self.dropout),
            ).to(self.device)
            self.out_norm = nn.LayerNorm(self.embed_dim).to(self.device)
            self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False).to(self.device)
            self.tie_output_weights()

        self._load_tokenizer(tokenizer_path)

    def load_fluency_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fluency checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.embedding.load_state_dict(state["embedding"])
        self.fusion_norm.load_state_dict(state["fusion_norm"])
        self.proj.load_state_dict(state["proj"])
        self.out_norm.load_state_dict(state["out_norm"])
        self.lm_head.load_state_dict(state["lm_head"])
        self.tie_output_weights()

    def save_fluency_checkpoint(self, path: str) -> None:
        payload = {
            "model": {
                "embedding": self.embedding.state_dict(),
                "fusion_norm": self.fusion_norm.state_dict(),
                "proj": self.proj.state_dict(),
                "out_norm": self.out_norm.state_dict(),
                "lm_head": self.lm_head.state_dict(),
            },
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "context_size": self.context_size,
                "tie_weights": self.tie_weights,
                "dropout": self.dropout,
            },
        }
        torch.save(payload, path)

    def _load_tokenizer(self, path: str | None = None) -> None:
        if not _HF_TOKENIZERS_AVAILABLE:
            return
        
        candidates = [path] if path else [DEFAULT_HF_TOKENIZER, FALLBACK_HF_TOKENIZER]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                self._tokenizer = HFTokenizer.from_file(candidate)
                vocab_size = self._tokenizer.get_vocab_size()
                if vocab_size != self.vocab_size:
                    self.vocab_size = vocab_size
                    self.embedding = nn.Embedding(vocab_size, self.embed_dim).to(self.device)
                    self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False).to(self.device)
                    self.tie_output_weights()
                print(f"[BPE] Loaded HF tokenizer from {candidate} (vocab: {self.vocab_size})")
                return

    def freeze_core(self) -> None:
        for param in self.hierarchy.parameters():
            param.requires_grad_(False)

    def reset_states(self, batch_size: int = 1) -> None:
        self.hierarchy.reset_states(batch_size=batch_size)

    def _normalize_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(token_ids), dim=-1)

    def _predict_core(self, emb: torch.Tensor, update_temporal: bool = True, max_steps: int = 1) -> torch.Tensor:
        with torch.no_grad():
            core = self.hierarchy.predict_label(
                emb,
                max_steps=max_steps,
                update_temporal=update_temporal,
            )
        if core.shape[1] > self.embed_dim:
            core = core[:, : self.embed_dim]
        elif core.shape[1] < self.embed_dim:
            pad = torch.zeros(core.shape[0], self.embed_dim - core.shape[1], device=core.device)
            core = torch.cat([core, pad], dim=1)
        return F.normalize(core, dim=-1)

    def fuse(self, emb: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([emb, core, emb * core, emb - core], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.proj(fused)
        fused = self.out_norm(fused + emb)
        return fused

    def step_logits(self, token_ids: torch.Tensor, update_temporal: bool = True, max_steps: int = 1) -> torch.Tensor:
        emb = self._normalize_embed(token_ids)
        core = self._predict_core(emb, update_temporal=update_temporal, max_steps=max_steps)
        fused = self.fuse(emb, core)
        return self.lm_head(fused)

    def forward_context(self, context_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = context_ids.shape
        self.reset_states(batch_size=batch_size)
        logits = None
        for t in range(seq_len):
            logits = self.step_logits(context_ids[:, t], update_temporal=True, max_steps=1)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ) -> str:
        if self._tokenizer is None:
            self._load_tokenizer()
        if self._tokenizer is not None:
            enc = self._tokenizer.encode(prompt)
            token_ids = enc.ids if enc.ids else [0]
        else:
            token_ids = [ord(c) % self.vocab_size for c in prompt] or [0]

        generated = list(token_ids)
        self.reset_states(batch_size=1)

        for tok_id in token_ids:
            tok = torch.tensor([tok_id], dtype=torch.long, device=self.device)
            _ = self.step_logits(tok, update_temporal=True, max_steps=1)

        eos = self._tokenizer.token_to_id("<|endoftext|>") if self._tokenizer is not None else None

        for _ in range(max_new_tokens):
            cur = torch.tensor([generated[-1]], dtype=torch.long, device=self.device)
            logits = self.step_logits(cur, update_temporal=True, max_steps=1)[0]
            logits = logits / max(temperature, 1e-5)

            for tok in set(generated[-20:]):
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty

            if top_k > 0 and top_k < logits.numel():
                values, _ = torch.topk(logits, top_k)
                cutoff = values[-1]
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            generated.append(next_id)
            if eos is not None and next_id == eos:
                break

        if self._tokenizer is not None and hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(generated)
        return "".join(chr(i % 128) for i in generated)
