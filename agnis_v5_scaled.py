"""
agnis_v5_scaled.py — Scaled AGNIS Architecture
=================================================
Phase 1 Scale-Up: Embedding + Position Encoding + Output Head

This module adds the gradient-trained periphery around the
Hebbian predictive coding core:

  token_id → Embedding(vocab, 64) + PosEncoding → [64D]
  → PredictiveHierarchy (Hebbian core)
  → OutputHead(hidden, vocab) → softmax → next token

The hierarchy remains fully Hebbian. Only the embedding and
output head use gradient descent.
"""

import math
import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (no learned parameters)."""

    def __init__(self, max_len: int = 256, embed_dim: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) *
            -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """
        Add position encoding to input.
        Args:
            x: [batch, embed_dim]
            position: current position in the sequence
        Returns:
            x + pe[position]
        """
        return x + self.pe[0, position, :]


class OutputHead(nn.Module):
    """
    Linear projection from hidden representation to vocabulary logits.
    Uses gradient descent for training (not Hebbian).
    """

    def __init__(self, hidden_dim: int, vocab_size: int, device: str = "cpu"):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, device=device)
        self.optimizer = None  # Created lazily

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state to vocab logits."""
        return self.linear(hidden)

    def compute_loss_and_update(self, hidden: torch.Tensor, target_ids: torch.Tensor, lr: float = 0.001):
        """
        Compute cross-entropy loss and update weights via gradient descent.
        Args:
            hidden: [batch, hidden_dim] — detached from Hebbian graph
            target_ids: [batch] — target token IDs
            lr: learning rate
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)

        logits = self.linear(hidden.detach())  # Detach from Hebbian graph
        loss = nn.functional.cross_entropy(logits, target_ids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class ScaledAGNIS(nn.Module):
    """
    Scaled AGNIS: Hybrid Hebbian-Gradient Architecture.

    Components:
      - Embedding (gradient-trained)
      - Position Encoding (fixed sinusoidal)
      - PredictiveHierarchy (Hebbian core)
      - Output Head (gradient-trained)

    The Hebbian core handles representation learning and continual learning.
    The gradient periphery handles vocabulary mapping.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dims: list = None,
        meta_pool_size: int = 0,
        max_context: int = 256,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_context = max_context

        # Gradient-trained periphery
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=self.device)
        self.pos_encoding = SinusoidalPositionEncoding(max_len=max_context, embed_dim=embed_dim).to(self.device)

        # Hebbian core
        if hidden_dims is None:
            hidden_dims = [embed_dim, 256, 256, embed_dim]
        self.hierarchy = PredictiveHierarchy(
            hidden_dims, device=device, meta_pool_size=meta_pool_size
        )

        # Gradient-trained output head
        # Maps from embed_dim (hierarchy output) to vocab_size
        self.output_head = OutputHead(embed_dim, vocab_size, device=device)

        # Embedding optimizer
        self.embed_optimizer = torch.optim.Adam(self.embedding.parameters(), lr=0.001)

        # Position counter
        self._position = 0

    def reset_position(self):
        """Reset position counter (call at start of each sequence)."""
        self._position = 0

    def encode_token(self, token_id: int) -> torch.Tensor:
        """
        Convert a token ID to a positioned embedding vector.
        Returns: [1, embed_dim]
        """
        ids = torch.tensor([[token_id]], device=self.device)
        embedded = self.embedding(ids).squeeze(0)  # [1, embed_dim]
        positioned = self.pos_encoding(embedded, self._position)
        return positioned

    def train_step(self, token_id: int, next_token_id: int, dopamine: float = 1.0, max_steps: int = 15):
        """
        One training step: predict next token from current token.
        Optimized for throughput with reduced settling steps.

        Returns: (hebbian_surprise, output_loss)
        """
        # 1. Embed input
        x = self.encode_token(token_id)

        # 2. Embed target (for Hebbian core)
        target_ids = torch.tensor([[next_token_id]], device=self.device)
        target_embed = self.embedding(target_ids).squeeze(0)  # [1, embed_dim]

        # 3. Hebbian core: infer and learn (with reduced settling)
        self.hierarchy.infer_and_learn(
            x, top_level_label=target_embed, dopamine_burst=dopamine,
            max_steps=max_steps, warm_start=True
        )

        # 4. Get the hierarchy's top-level prediction
        top_layer = self.hierarchy.layers[-1]
        predicted_embed = top_layer._phi(top_layer.x)  # [1, embed_dim]

        # 5. Output head: map to vocab + compute loss
        target_id_tensor = torch.tensor([next_token_id], device=self.device)
        output_loss = self.output_head.compute_loss_and_update(
            predicted_embed, target_id_tensor
        )

        # 6. Update embedding periodically (every 10 tokens for speed)
        self._train_step_count = getattr(self, '_train_step_count', 0) + 1
        if self._train_step_count % 10 == 0:
            embed_logits = self.output_head.linear(self.embedding(target_ids).squeeze(0))
            embed_loss = nn.functional.cross_entropy(embed_logits, target_id_tensor)
            self.embed_optimizer.zero_grad()
            embed_loss.backward()
            self.embed_optimizer.step()

        # 7. Get Hebbian surprise
        hebbian_surprise = self.hierarchy.get_surprise((x.detach(), target_embed.detach()))

        # Advance position
        self._position = (self._position + 1) % self.max_context

        return hebbian_surprise, output_loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt_ids: list[int], max_tokens: int = 100,
                 temperature: float = 0.8) -> list[int]:
        """
        Autoregressive generation using temperature-scaled softmax.

        Args:
            tokenizer: BPETokenizer instance for decoding
            prompt_ids: list of token IDs as context
            max_tokens: number of new tokens to generate
            temperature: sampling temperature (lower = more deterministic)

        Returns: list of generated token IDs
        """
        self.reset_position()
        self.hierarchy.reset_states()

        generated = []

        # Process prompt (build context)
        for i, tid in enumerate(prompt_ids):
            x = self.encode_token(tid)
            self.hierarchy.forward(x, max_steps=30)
            self.hierarchy.step_temporal()
            self._position = (self._position + 1) % self.max_context

        # Generate new tokens
        last_id = prompt_ids[-1] if prompt_ids else 0
        for _ in range(max_tokens):
            x = self.encode_token(last_id)
            self.hierarchy.forward(x, max_steps=30)

            # Get predicted embedding
            top_layer = self.hierarchy.layers[-1]
            predicted = top_layer._phi(top_layer.x)

            # Map to logits and sample
            logits = self.output_head(predicted)  # [1, vocab_size]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            generated.append(next_id)
            last_id = next_id

            self.hierarchy.step_temporal()
            self._position = (self._position + 1) % self.max_context

        return generated

    def save(self, path: str):
        """Save entire model state."""
        state = {
            'embedding': self.embedding.state_dict(),
            'output_head': self.output_head.linear.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'max_context': self.max_context,
        }
        torch.save(state, path)
        self.hierarchy.save_checkpoint(path.replace('.pt', '_hierarchy.pt'))
        print(f"[ScaledAGNIS] Saved to {path}")

    def load(self, path: str):
        """Load entire model state."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.embedding.load_state_dict(state['embedding'])
        self.output_head.linear.load_state_dict(state['output_head'])
        self.hierarchy.load_checkpoint(path.replace('.pt', '_hierarchy.pt'))
        print(f"[ScaledAGNIS] Loaded from {path}")


# ───────────────────────────────────────────
# Quick self-test
# ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  ScaledAGNIS Self-Test")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 500
    embed_dim = 64

    model = ScaledAGNIS(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dims=[embed_dim, 192, 192, embed_dim],
        meta_pool_size=64,
        device=device
    )

    print(f"  Device: {device}")
    print(f"  Vocab: {vocab_size}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Hierarchy layers: {len(model.hierarchy.layers)}")

    # Test training step
    surprise, loss = model.train_step(token_id=10, next_token_id=20)
    print(f"  Train step: surprise={surprise:.4f}, loss={loss:.4f}")

    # Test generation
    generated = model.generate(None, prompt_ids=[10, 20, 30], max_tokens=5)
    print(f"  Generated: {generated}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    hebbian = sum(p.numel() for p in model.hierarchy.parameters())
    gradient = total - hebbian
    print(f"\n  Total params: {total:,}")
    print(f"  Hebbian core: {hebbian:,} ({hebbian/total*100:.1f}%)")
    print(f"  Gradient periphery: {gradient:,} ({gradient/total*100:.1f}%)")

    print("\n  Self-test PASSED.")
