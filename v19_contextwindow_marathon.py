"""
v19_contextwindow_marathon.py — Context-Window Mode
=====================================================
V3 Phase 36: 16-token context window for 50x throughput.

Instead of settling the hierarchy for every single token,
we flatten a window of 16 BPE token embeddings into one
1024D vector and process it in one hierarchy step.

Architecture:
  [16 tokens] → Embedding(4000, 64) + PosEncoding
      → Flatten to [1024D]
      → PredictiveHierarchy [1024, 256, 256, 64] (Hebbian core)
      → OutputHead(64, 4000) → softmax → next token

Same Meta-Pool. Same Synaptic Shield. 50x faster.
"""

import torch
import torch.nn as nn
import math
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, AbstraXEngine
from slm.slm_tokenizer import BPETokenizer

# ─── Config ───
META_POOL_SIZE = 128
N_PER_LANG = 128
EMBED_DIM = 64
CONTEXT_LEN = 16
BPE_VOCAB = 4000
PHASE1_DURATION = 600   # 10 min per language
PHASE2_DURATION = 300   # 5 min consolidation
LANGS = ["en", "de", "es", "fr"]
LANG_NAMES = {"en": "English", "de": "German", "es": "Spanish", "fr": "French"}


class ContextWindowModel(nn.Module):
    """
    AGNIS with context-window input mode.
    Flattens a window of token embeddings into one vector,
    feeds through Hebbian hierarchy, predicts next token.
    """

    def __init__(self, vocab_size, embed_dim=64, context_len=16,
                 meta_pool_size=0, hidden_per_lang=128, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.input_dim = context_len * embed_dim  # 16 * 64 = 1024

        # Embedding + Position
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=self.device)
        pe = torch.zeros(context_len, embed_dim)
        pos = torch.arange(context_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                        -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # [context_len, embed_dim]

        # Hierarchy
        initial_hidden = meta_pool_size + hidden_per_lang
        self.hierarchy = PredictiveHierarchy(
            [self.input_dim, initial_hidden, initial_hidden, embed_dim],
            device=device,
            meta_pool_size=meta_pool_size
        )

        # Output head
        self.output_proj = nn.Linear(embed_dim, vocab_size, device=self.device)
        self.output_optimizer = torch.optim.Adam(
            list(self.output_proj.parameters()) + list(self.embedding.parameters()),
            lr=0.001
        )

    def embed_window(self, token_ids: list[int]) -> torch.Tensor:
        """Embed a context window and flatten to 1D."""
        ids = torch.tensor([token_ids[-self.context_len:]], dtype=torch.long, device=self.device)
        # Pad if too short
        if ids.shape[1] < self.context_len:
            pad = torch.zeros(1, self.context_len - ids.shape[1], dtype=torch.long, device=self.device)
            ids = torch.cat([pad, ids], dim=1)
        embedded = self.embedding(ids)  # [1, context_len, embed_dim]
        embedded = embedded + self.pe.unsqueeze(0)  # Add position encoding
        flat = embedded.view(1, -1)  # [1, context_len * embed_dim]
        return flat

    def train_step(self, context_ids: list[int], target_id: int, dopamine: float = 1.0):
        """
        One training step with context window.
        Returns: (surprise, loss)
        """
        # Embed context window
        x = self.embed_window(context_ids)

        # Target embedding
        target_tensor = torch.tensor([[target_id]], dtype=torch.long, device=self.device)
        target_embed = self.embedding(target_tensor).squeeze(0)  # [1, embed_dim]

        # Hebbian hierarchy: infer and learn
        self.hierarchy.infer_and_learn(
            x, top_level_label=target_embed,
            dopamine_burst=dopamine, max_steps=15, warm_start=True
        )

        # Get prediction from hierarchy
        top = self.hierarchy.layers[-1]
        predicted = top._phi(top.x)  # [1, embed_dim]

        # Output head: gradient update
        logits = self.output_proj(predicted.detach())
        target_id_tensor = torch.tensor([target_id], device=self.device)
        loss = nn.functional.cross_entropy(logits, target_id_tensor)

        self.output_optimizer.zero_grad()
        loss.backward()
        self.output_optimizer.step()

        # Surprise
        surprise = self.hierarchy.get_surprise((x.detach(), target_embed.detach()))

        return surprise, loss.item()

    @torch.no_grad()
    def generate(self, tokenizer, prompt_ids: list[int], max_tokens: int = 30,
                 temperature: float = 0.8) -> str:
        """Generate text from a prompt."""
        self.hierarchy.reset_states()
        context = list(prompt_ids)

        for _ in range(max_tokens):
            x = self.embed_window(context)
            self.hierarchy.forward(x, max_steps=10)

            top = self.hierarchy.layers[-1]
            predicted = top._phi(top.x)

            logits = self.output_proj(predicted)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            context.append(next_id)
            self.hierarchy.step_temporal()

        return tokenizer.decode(context)


def freeze_all(hierarchy):
    for layer in hierarchy.layers:
        layer.V_mask.zero_()
        layer.W_mask.zero_()
        layer.b_in_mask.zero_()
        layer.b_out_mask.zero_()
        layer.R_mask.zero_()
        layer.R_gate_mask.zero_()
        layer.L_mask.zero_()


def unmask_meta(hierarchy, mp):
    for i, layer in enumerate(hierarchy.layers):
        if i < len(hierarchy.layers) - 1:
            layer.V_mask[:, :mp] = 1.0
            layer.W_mask[:mp, :] = 1.0
            layer.b_in_mask[:mp] = 1.0
            layer.R_mask[:mp, :mp] = 1.0
            layer.R_gate_mask[:mp, :mp] = 1.0
            layer.L_mask[:mp, :mp] = 1.0
        if i > 0:
            layer.V_mask[:mp, :] = 1.0
            layer.W_mask[:, :mp] = 1.0
            layer.b_out_mask[:mp] = 1.0


def run_v19():
    import random
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  AGNIS V19 -- CONTEXT WINDOW MODE")
    print("  16-token window | BPE 4K | Hebbian Core")
    print("=" * 60)

    # Load data
    corpora = {}
    for code in LANGS:
        for path in [f"slm/wiki_{code}.txt", f"slm/input_{code}.txt"]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    corpora[code] = f.read()[:100000]
                break
        print(f"  {LANG_NAMES[code]}: {len(corpora[code]):,} chars")

    # BPE
    combined = "".join(corpora.values())
    cache = f"agnis_bpe_{BPE_VOCAB}_vocab.json"
    if os.path.exists(cache):
        tokenizer = BPETokenizer(BPE_VOCAB)
        tokenizer.load(cache)
    else:
        tokenizer = BPETokenizer(BPE_VOCAB)
        tokenizer.fit(combined, verbose=True)
        tokenizer.save(cache)

    all_tokens = {}
    for code, text in corpora.items():
        tokens = tokenizer.encode(text)
        all_tokens[code] = tokens
        print(f"  {LANG_NAMES[code]}: {len(tokens):,} tokens ({len(text)/len(tokens):.1f}x)")

    # Ranges
    lang_ranges = {}
    for i, lang in enumerate(LANGS):
        s = META_POOL_SIZE + i * N_PER_LANG
        lang_ranges[lang] = (s, s + N_PER_LANG)
    final_width = META_POOL_SIZE + N_PER_LANG * len(LANGS)

    # Build model
    model = ContextWindowModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        context_len=CONTEXT_LEN,
        meta_pool_size=META_POOL_SIZE,
        hidden_per_lang=N_PER_LANG,
        device=device
    )
    hierarchy = model.hierarchy

    input_dim = CONTEXT_LEN * EMBED_DIM
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Input: {CONTEXT_LEN} tokens x {EMBED_DIM}D = {input_dim}D")
    print(f"  Ranges: {lang_ranges}")
    print(f"  Total params: {total_params:,}")
    print("=" * 60)

    # ═══ PHASE 1: Sequential ═══
    print("\n" + "=" * 40)
    print("  PHASE 1: SEQUENTIAL TRAINING")
    print("=" * 40)

    for phase_idx, code in enumerate(LANGS):
        name = LANG_NAMES[code]
        tokens = all_tokens[code]
        print(f"\n>>> {name.upper()} ({PHASE1_DURATION//60} min) <<<")

        t0 = time.time()
        lang_start, lang_end = lang_ranges[code]
        losses = []
        token_count = 0

        with hierarchy.manifold_gate(0, lang_end):
            hierarchy.reset_states()
            for i in range(CONTEXT_LEN, len(tokens) - 1):
                if time.time() - t0 > PHASE1_DURATION:
                    break

                context = tokens[i - CONTEXT_LEN:i]
                target = tokens[i]

                surprise, loss = model.train_step(context, target)
                losses.append(loss)
                token_count += 1

                if token_count % 200 == 0:
                    elapsed = time.time() - t0
                    avg = sum(losses[-100:]) / max(1, len(losses[-100:]))
                    tps = token_count / max(1, elapsed)
                    print(f"  [{name}] {token_count} steps | Loss: {avg:.3f} | {tps:.1f} tok/s | {elapsed/PHASE1_DURATION:.0%}", end="\r")

        avg_final = sum(losses[-200:]) / max(1, len(losses[-200:]))
        print(f"\n  {name}: {token_count} steps | Final Loss: {avg_final:.4f} | {token_count/(time.time()-t0):.1f} tok/s")

        if phase_idx < len(LANGS) - 1:
            hierarchy.force_recruit_language_sliver(n=N_PER_LANG, language=name)

    # ═══ PHASE 2: Consolidation ═══
    print("\n" + "=" * 40)
    print("  PHASE 2: CONSOLIDATION")
    print("=" * 40)

    freeze_all(hierarchy)
    unmask_meta(hierarchy, META_POOL_SIZE)

    t0 = time.time()
    token_count = 0
    lang_idx = 0

    while time.time() - t0 < PHASE2_DURATION:
        code = LANGS[lang_idx % len(LANGS)]
        tokens = all_tokens[code]
        lang_start, lang_end = lang_ranges[code]

        # Random position in corpus
        pos = torch.randint(CONTEXT_LEN, len(tokens) - 1, (1,)).item()
        context = tokens[pos - CONTEXT_LEN:pos]
        target = tokens[pos]

        with hierarchy.manifold_gate(0, final_width):
            model.train_step(context, target, dopamine=0.5)

        token_count += 1
        if token_count % 50 == 0:
            lang_idx += 1
            hierarchy.reset_states()

        if token_count % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Consolidation: {token_count} | {elapsed/PHASE2_DURATION:.0%}", end="\r")

    print(f"\n  Done: {token_count} tokens")

    # ═══ Save ═══
    hierarchy.save_checkpoint("agnis_v19_final.pt")
    print("[Saved]")

    # ═══ AbstraX ═══
    print("\n>>> ABSTRAX DREAM CYCLE <<<")
    abstrax_ranges = {"meta": (0, META_POOL_SIZE)}
    abstrax_ranges.update(lang_ranges)
    abstrax = AbstraXEngine(hierarchy, abstrax_ranges)

    for li in range(len(hierarchy.layers)):
        print(f"\n--- Layer {li} ---")
        try:
            r = abstrax.compute_pairwise_affinity(layer_idx=li)
            abstrax.print_affinity_report(r, title=f"Layer {li}")
            abstrax.identify_fold_candidates(r, threshold=0.2)
        except Exception as e:
            print(f"  [SKIP] {e}")

    # ═══ Generation Demo ═══
    print("\n>>> GENERATION DEMO <<<")
    prompts = {"en": "The king was", "de": "Der Mann war", "es": "El rey era", "fr": "Le roi est"}

    for code, prompt in prompts.items():
        lang_s, lang_e = lang_ranges[code]
        prompt_ids = tokenizer.encode(prompt)
        with hierarchy.manifold_gate(0, lang_e):
            hierarchy.reset_states()
            text = model.generate(tokenizer, prompt_ids, max_tokens=25, temperature=0.8)
        print(f"  [{code}] {text}")

    total = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"  V19 COMPLETE -- {total:.1f} min | {total_params:,} params")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_v19()
