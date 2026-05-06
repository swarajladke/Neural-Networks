"""
slm/agnis_fluency_model.py
==========================
Fluency-oriented AGNIS language model.

Design goals:
  - keep AGNIS as the frozen predictive core
  - use a stronger language readout path than nearest-neighbor decoding
  - make training and inference use the exact same fusion logic

Architecture (v2 — Temporal Augmentation):
  token ids
    -> embedding
    -> AGNIS core prediction (frozen)
    -> temporal gate: h_t = (1-α)·core + α·tanh(R·h_{t-1})
    -> hippocampal recall (QKV on h_t)
    -> fusion features [emb, h_t, recall, emb*h_t, emb-h_t]
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
from v23_hippocampal_memory import LocalHippocampalBuffer

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


class TemporalGate(nn.Module):
    """
    Minimal gated temporal recurrence.
    
    h_t = (1 - gate) * core_t + gate * tanh(R @ h_{t-1})
    
    The gate is a learned scalar-per-dimension that controls how much
    temporal memory vs fresh core signal flows into the fusion head.
    Initialized small (bias=-2 → sigmoid≈0.12) so token prediction
    dominates at the start.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Recurrent matrix (local temporal binding)
        self.R = nn.Linear(dim, dim, bias=False)
        # Gate: how much temporal vs core
        self.gate_proj = nn.Linear(dim * 2, dim)
        # Initialize gate bias negative so temporal starts SMALL
        nn.init.constant_(self.gate_proj.bias, -2.0)
        # LayerNorm on the temporal state for stability
        self.norm = nn.LayerNorm(dim)
        # Persistent hidden state
        self.register_buffer("h_prev", torch.zeros(1, dim))
        
    def reset(self, batch_size: int = 1):
        self.h_prev = torch.zeros(batch_size, self.dim, device=self.R.weight.device)
        
    def forward(self, core: torch.Tensor) -> torch.Tensor:
        """
        core: [batch, dim] — frozen core output for current token
        Returns: [batch, dim] — temporally-augmented representation
        """
        # Temporal candidate from previous state
        temporal = torch.tanh(self.R(self.h_prev.detach()))
        
        # Compute gate from both signals
        gate = torch.sigmoid(self.gate_proj(torch.cat([core, temporal], dim=-1)))
        
        # Mix: mostly core at start, learns to blend in temporal
        h_t = (1.0 - gate) * core + gate * temporal
        h_t = self.norm(h_t)
        
        # Shift state (detach to prevent BPTT)
        self.h_prev = h_t.detach()
        
        return h_t


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
        
        # Temporal Gate (controlled recurrence on top of frozen core)
        self.temporal_gate = TemporalGate(embed_dim).to(self.device)
        
        # Hippocampal Memory (QKV recall on temporally-augmented state)
        self.memory = LocalHippocampalBuffer(d_model=embed_dim, max_memory=context_size).to(self.device)

        # Fusion: [emb, h_t, recall, emb*h_t, emb-h_t] = 5 * embed_dim
        hidden_dim = fusion_hidden_dim or (embed_dim * 4)
        self.fusion_norm = nn.LayerNorm(embed_dim * 5).to(self.device)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 5, hidden_dim),
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
            
            # Rebuild temporal gate for new dim
            self.temporal_gate = TemporalGate(self.embed_dim).to(self.device)
            self.memory = LocalHippocampalBuffer(d_model=self.embed_dim, max_memory=self.context_size).to(self.device)

            hidden_dim = self.proj[0].out_features
            self.fusion_norm = nn.LayerNorm(self.embed_dim * 5).to(self.device)
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim * 5, hidden_dim),
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
        if "memory" in state:
            self.memory.load_state_dict(state["memory"])
        if "temporal_gate" in state:
            self.temporal_gate.load_state_dict(state["temporal_gate"])
        self.tie_output_weights()

    def save_fluency_checkpoint(self, path: str) -> None:
        payload = {
            "model": {
                "embedding": self.embedding.state_dict(),
                "fusion_norm": self.fusion_norm.state_dict(),
                "proj": self.proj.state_dict(),
                "out_norm": self.out_norm.state_dict(),
                "lm_head": self.lm_head.state_dict(),
                "memory": self.memory.state_dict(),
                "temporal_gate": self.temporal_gate.state_dict(),
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
        self.temporal_gate.reset(batch_size=batch_size)
        self.memory.reset_memory(batch_size=batch_size)
        self.memory.to(self.device)

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

    def fuse(self, emb: torch.Tensor, h_t: torch.Tensor, x_recall: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([emb, h_t, x_recall, emb * h_t, emb - h_t], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.proj(fused)
        fused = self.out_norm(fused + emb)
        return fused

    def step_logits(self, token_ids: torch.Tensor, update_temporal: bool = True, max_steps: int = 1) -> torch.Tensor:
        emb = self._normalize_embed(token_ids)
        core = self._predict_core(emb, update_temporal=update_temporal, max_steps=max_steps)
        
        # Temporal augmentation: controlled blend of core + recurrence
        h_t = self.temporal_gate(core)
        
        # Memory recall on temporally-augmented state
        x_recall = self.memory(h_t)
        
        fused = self.fuse(emb, h_t, x_recall)
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
