"""
slm/agnis_slm_wrapper.py — V6.1: 64-Token Context Window
=========================================================
Context Window = 64 tokens.

Architecture (matches agnis_marathon_final.pt):
  token_id → Embedding [embed_dim] → PredictiveHierarchy → predicted_embed
  → L2 nearest-neighbor in embedding table → next token

Context strategy:
  - Prime the hierarchy by feeding the last min(64, len(prompt)) tokens
    one-by-one with update_temporal=True.
  - R-matrices accumulate temporal state across the full priming context.
  - Then generate autoregressively, sliding the window forward.

Note on embedding:
  The checkpoint stores only hierarchy weights (V, W, R, etc.), NOT the
  embedding table used during marathon training. Since we cannot recover
  the original embedding, we use L2 nearest-neighbor matching in the
  current embedding space — this is self-consistent at generation time.
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent

try:
    from tokenizers import Tokenizer as HFTokenizer
    _HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    _HF_TOKENIZERS_AVAILABLE = False

# ── Context Window Configuration ───────────────────────────────────────────
CONTEXT_SIZE       = 64     # rolling context window (tokens)
DEFAULT_EMBED_DIM  = 110   # matches agnis_marathon_final.pt
DEFAULT_VOCAB_SIZE = 4096  # HF BPE vocab (slm_bpe_tokenizer.json)
HF_BPE_FILE        = "slm_bpe_tokenizer.json"  # HuggingFace tokenizers format
BPE_VOCAB_FILE     = "agnis_bpe_4000_vocab.json"  # AGNIS format (Colab-only)


class AGNISSLMWrapper(nn.Module):
    """
    V6.1: AGNIS SLM with 64-Token Context Window.

    The hierarchy was trained on single-token (embed_dim-D) inputs.
    A 64-token effective context is achieved by priming the hierarchy
    with up to CONTEXT_SIZE prompt tokens before generation, letting
    the recurrent R-matrices accumulate temporal state.

    Usage:
        wrapper = AGNISSLMWrapper()
        wrapper.load_checkpoint('agnis_marathon_final.pt')
        text = wrapper.generate("The history of", max_new_tokens=50)
    """

    def __init__(self,
                 vocab_size:   int = DEFAULT_VOCAB_SIZE,
                 embed_dim:    int = DEFAULT_EMBED_DIM,
                 context_size: int = CONTEXT_SIZE,
                 device:       str = "cpu"):
        super().__init__()
        self.device       = torch.device(device)
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.context_size = context_size   # ← 64-token window
        self._tokenizer   = None

        # Embedding table: [vocab_size, embed_dim]
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        # Hierarchy: matches trained checkpoint default [embed_dim, 1024, embed_dim]
        self.hierarchy = PredictiveHierarchy(
            [embed_dim, 1024, embed_dim], device=str(self.device)
        )
        self.agent = CognitivePredictiveAgent(
            self.hierarchy, device=str(self.device)
        )

        # NOTE: We do NOT use output_head dot-product here because the embedding
        # table used during marathon training is not saved in the checkpoint.
        # Instead, generate() uses L2 nearest-neighbor in the embedding table,
        # which is self-consistent regardless of random init.

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str):
        """Load hierarchy weights from checkpoint. Auto-detects dimensions."""
        if not os.path.exists(path):
            print(f"[WARNING] Checkpoint not found: {path}. Using fresh weights.")
            self._load_tokenizer()
            return

        print(f"[Loading] {path}")
        self.hierarchy.load_checkpoint(path)

        # Auto-detect embed_dim from saved checkpoint
        detected_idim = self.hierarchy.layers[0].input_dim
        if detected_idim != self.embed_dim:
            print(f"[INFO] Checkpoint embed_dim={detected_idim} "
                  f"(was {self.embed_dim}). Rebuilding embedding table.")
            self.embed_dim = detected_idim
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim).to(self.device)

        self._load_tokenizer()
        print(f"[Ready] Context window: {self.context_size} tokens | "
              f"embed_dim: {self.embed_dim} | vocab: {self.vocab_size}")

    def _load_tokenizer(self):
        """Load BPE tokenizer. Tries HuggingFace format first, then AGNIS format."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 1) HuggingFace tokenizers format (slm_bpe_tokenizer.json)
        if _HF_TOKENIZERS_AVAILABLE:
            for candidate in [
                os.path.join(root, HF_BPE_FILE),
                HF_BPE_FILE,
                os.path.join(os.path.dirname(__file__), HF_BPE_FILE),
            ]:
                if os.path.exists(candidate):
                    try:
                        self._tokenizer = HFTokenizer.from_file(candidate)
                        vsz = self._tokenizer.get_vocab_size()
                        print(f"[BPE] Loaded HF tokenizer from {candidate} (vocab: {vsz})")
                        # Update vocab_size to match tokenizer
                        if vsz != self.vocab_size:
                            print(f"[INFO] Updating vocab_size: {self.vocab_size} → {vsz}")
                            self.vocab_size = vsz
                            self.embedding = nn.Embedding(
                                self.vocab_size, self.embed_dim
                            ).to(self.device)
                        return
                    except Exception as e:
                        print(f"[BPE] HF load error: {e}")

        # 2) AGNIS BPETokenizer format (agnis_bpe_4000_vocab.json)
        for candidate in [
            os.path.join(root, BPE_VOCAB_FILE),
            BPE_VOCAB_FILE,
            os.path.join(os.path.dirname(__file__), BPE_VOCAB_FILE),
        ]:
            if os.path.exists(candidate):
                try:
                    from slm.slm_tokenizer import BPETokenizer
                    tok = BPETokenizer(self.vocab_size)
                    tok.load(candidate)
                    self._tokenizer = tok
                    print(f"[BPE] Loaded AGNIS tokenizer from {candidate}")
                    return
                except Exception as e:
                    print(f"[BPE] AGNIS load error: {e}")

        print("[BPE] No tokenizer found — falling back to char encoding.")

    # ──────────────────────────────────────────────────────────────────────
    # Sliding-Window Training Step
    # ──────────────────────────────────────────────────────────────────────

    def learn_step(self, context_indices: list, target_indices: list):
        """
        Train on a batch of (context_window, target_token) pairs.
        context_indices: [batch, context_size] token IDs
        target_indices:  [batch] or [batch, 1] target token IDs
        """
        ctx_tensor = torch.tensor(context_indices, dtype=torch.long, device=self.device)
        tgt_tensor = torch.tensor(target_indices,  dtype=torch.long, device=self.device)
        if tgt_tensor.dim() == 2:
            tgt_tensor = tgt_tensor.squeeze(1)

        # Feed last token of context (single-token hierarchy mode, warm_start=True)
        last_tok  = ctx_tensor[:, -1]                  # [batch]
        x_embed   = self.embedding(last_tok)           # [batch, embed_dim]
        tgt_embed = self.embedding(tgt_tensor)         # [batch, embed_dim]
        # Normalize embeddings for stable hierarchy input
        x_embed   = nn.functional.normalize(x_embed,   dim=-1)
        tgt_embed = nn.functional.normalize(tgt_embed, dim=-1)

        weight, surprise = self.agent.observe_and_learn(
            x_embed, tgt_embed,
            task_id=0, max_steps=50, warm_start=True
        )
        s = surprise.mean().item() if hasattr(surprise, 'mean') else float(surprise)
        return weight, s

    def dream_consolidation(self, batch_size=16):
        """Pass-through to AGNIS declarative memory replay."""
        return self.agent.dream_replay(batch_size=batch_size, max_steps=100)

    # ──────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 prompt:          str,
                 max_new_tokens:  int   = 100,
                 temperature:     float = 0.8) -> str:
        """
        Autoregressive generation with 64-token context window.

        Steps:
          1. Encode prompt → token IDs.
          2. Reset hierarchy state.
          3. PRIME: feed last min(64, len(prompt)) tokens one-by-one
             with update_temporal=True to build R-matrix temporal state.
          4. GENERATE: autoregressively sample max_new_tokens new tokens,
             updating temporal state at every step.
          5. Decode full sequence and return.
        """
        if self._tokenizer is None:
            self._load_tokenizer()

        # ── 1. Encode ────────────────────────────────────────────────────
        if self._tokenizer is not None:
            # Handle both HF tokenizers (returns Encoding obj) and AGNIS BPETokenizer (returns list)
            enc = self._tokenizer.encode(prompt)
            prompt_ids = enc.ids if hasattr(enc, 'ids') else enc
        else:
            prompt_ids = [ord(c) % self.vocab_size for c in prompt]
        if not prompt_ids:
            prompt_ids = [0]

        generated_ids = list(prompt_ids)

        # ── 2. Reset hierarchy ───────────────────────────────────────────
        self.hierarchy.reset_states(batch_size=1)

        # ── 3. Prime with up to CONTEXT_SIZE tokens ──────────────────────
        prime_tokens = prompt_ids[-self.context_size:]
        print(f"\n[AGNIS] Priming with {len(prime_tokens)}-token context window...")

        for tok_id in prime_tokens:
            tok_t = torch.tensor([[tok_id]], dtype=torch.long, device=self.device)
            embed = self.embedding(tok_t).view(1, -1)                   # [1, embed_dim]
            embed = nn.functional.normalize(embed, dim=-1)              # normalize
            self.hierarchy.predict_label(embed, update_temporal=True)

        # ── 4. Autoregressive generation loop ────────────────────────────
        print(f"[AGNIS] Generating {max_new_tokens} tokens (temperature={temperature})...")

        # Pre-normalize entire embedding table for fast L2 nearest-neighbor
        emb_weight = nn.functional.normalize(
            self.embedding.weight, dim=-1
        )  # [vocab_size, embed_dim]

        for _ in range(max_new_tokens):
            cur_tok   = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=self.device)
            cur_embed = self.embedding(cur_tok).view(1, -1)        # [1, embed_dim]
            cur_embed = nn.functional.normalize(cur_embed, dim=-1) # normalize

            # Predict next embedding from warm temporal state
            pred_embed = self.hierarchy.predict_label(cur_embed, update_temporal=True)

            # Slice / pad to embed_dim
            if pred_embed.shape[1] > self.embed_dim:
                pred_embed = pred_embed[:, :self.embed_dim]
            elif pred_embed.shape[1] < self.embed_dim:
                pad = torch.zeros(1, self.embed_dim - pred_embed.shape[1], device=self.device)
                pred_embed = torch.cat([pred_embed, pad], dim=1)

            # Normalize predicted embedding
            pred_norm = nn.functional.normalize(pred_embed, dim=-1)  # [1, embed_dim]

            # L2 nearest-neighbor in embedding space (= cosine distance after norm)
            # distances[i] = ||pred_norm - emb_weight[i]||^2
            distances = torch.cdist(pred_norm, emb_weight)[0]        # [vocab_size]

            # Boltzmann sampling: lower distance → higher probability
            logits = -distances
            probs  = torch.softmax(logits / temperature, dim=-1)

            # Sample next token
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_id)

        # ── 5. Decode ────────────────────────────────────────────────────
        if self._tokenizer is not None:
            # Handle both HF tokenizers and AGNIS BPETokenizer
            if hasattr(self._tokenizer, 'decode'):
                return self._tokenizer.decode(generated_ids)
            else:
                return str(generated_ids)
        else:
            return "".join(chr(i % 128) for i in generated_ids)


# ── Backward-compat alias ───────────────────────────────────────────────────
AGNIS_SLM_Wrapper = AGNISSLMWrapper
