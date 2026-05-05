"""
v10_target_prop_probe.py
========================
Option 2: Target Propagation / Contrastive Hebbian Learning

This script removes backpropagation completely. It uses ZERO PyTorch autograd.
Instead of loss.backward(), it uses Target Propagation:
1. Free Phase: The network predicts the next token.
2. Clamped Phase: The readout error is calculated.
3. Target Prop: The error is projected DOWN the network manually.
   - Output Head is updated via local Delta Rule.
   - Embedding is updated via local Delta Rule.
   - The AGNIS hierarchy is updated by telling SNAP-ATP to morph its 
     internal manifolds to match the target representation.
"""

from __future__ import annotations

import os
import re
import sys
import time
import math
import urllib.request

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from agnis_v4_cognitive import CognitivePredictiveAgent
from experiment_utils import metric_to_float
from slm.agnis_slm_wrapper import AGNISSLMWrapper

CHECKPOINT_IN = "agnis_english_interface.pt"
TOKENIZER_PATHS = ["slm_bpe_tokenizer_en.json", "slm_bpe_tokenizer.json"]
CORPUS_PATH = "slm/input_en_massive.txt"

TARGET_CHARS = 1_000_000
BATCH_SIZE = 32
EPOCHS = 3
LR = 0.01  # Target Prop learning rate is usually higher than Adam
TEMPERATURE = 0.8
MAX_GEN_TOKENS = 60
EVAL_STEPS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
]

def sep(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")

def clean_text(text: str) -> str:
    for marker in ["CHAPTER I.", "CHAPTER I", "Chapter I", "CHAPTER 1"]:
        idx = text.find(marker)
        if idx != -1:
            nxt = text.find(marker, idx + len(marker))
            if nxt != -1 and (nxt - idx) < 1000:
                idx = nxt
            text = text[idx:]
            break
    for marker in ["End of the Project Gutenberg", "THE END"]:
        idx = text.rfind(marker)
        if idx != -1:
            text = text[:idx]
            break
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()

def load_tokenizer() -> tuple[Tokenizer, str]:
    for path in TOKENIZER_PATHS:
        if os.path.exists(path):
            return Tokenizer.from_file(path), path
    raise FileNotFoundError("No tokenizer found.")

def load_corpus() -> str:
    if not os.path.exists(CORPUS_PATH) or os.path.getsize(CORPUS_PATH) < 1_000_000:
        print("[Corpus] Downloading Gutenberg dataset...")
        os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
        full_text = ""
        for url in GUTENBERG_URLS:
            try:
                fname = url.split("/")[-1]
                print(f"  -> Downloading {fname}...")
                raw = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")
                full_text += clean_text(raw) + "\n\n"
            except Exception as e:
                print(f"  -> Failed: {e}")
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            f.write(full_text)
        print("[Corpus] Download complete.")

    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = clean_text(raw)[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text

def build_streams(token_ids: list[int], batch_size: int, device: str) -> torch.Tensor:
    seq_len = len(token_ids) // batch_size
    usable = token_ids[: seq_len * batch_size]
    return torch.tensor(usable, dtype=torch.long, device=device).view(batch_size, seq_len)

@torch.no_grad()
def evaluate(wrapper: AGNISSLMWrapper, token_streams: torch.Tensor, eval_steps: int) -> float:
    steps = min(eval_steps, token_streams.shape[1] - 1)
    wrapper.hierarchy.reset_states(batch_size=token_streams.shape[0])
    correct = 0
    total = 0

    for step in range(steps):
        cur_id = token_streams[:, step]
        tgt_id = token_streams[:, step + 1]
        
        emb = F.normalize(wrapper.embedding(cur_id), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > wrapper.embed_dim:
            hid = hid[:, :wrapper.embed_dim]
            
        combined = emb + 0.5 * hid
        logits = wrapper.output_head(combined)
        pred_id = torch.argmax(logits, dim=-1)
        
        correct += int((pred_id == tgt_id).sum().item())
        total += int(tgt_id.numel())

    return correct / max(1, total)

@torch.no_grad()
def generate(wrapper: AGNISSLMWrapper, tokenizer: Tokenizer, prompt: str) -> str:
    enc = tokenizer.encode(prompt)
    token_ids = enc.ids if enc.ids else [0]
    generated = list(token_ids)
    wrapper.hierarchy.reset_states(batch_size=1)

    # Prime
    for tok_id in token_ids:
        tok = torch.tensor([tok_id], dtype=torch.long, device=wrapper.device)
        emb = F.normalize(wrapper.embedding(tok), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        
    for _ in range(MAX_GEN_TOKENS):
        cur = torch.tensor([generated[-1]], dtype=torch.long, device=wrapper.device)
        emb = F.normalize(wrapper.embedding(cur), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > wrapper.embed_dim:
            hid = hid[:, :wrapper.embed_dim]
            
        combined = emb + 0.5 * hid
        logits = wrapper.output_head(combined) / TEMPERATURE
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs[0], num_samples=1).item())
        generated.append(next_id)

    return tokenizer.decode(generated)

def train_target_prop(wrapper: AGNISSLMWrapper, agent: CognitivePredictiveAgent, token_streams: torch.Tensor) -> None:
    total_steps = token_streams.shape[1] - 1
    print(f"[Train] {token_streams.shape[0]} streams x {total_steps:,} steps | {EPOCHS} epochs")
    print(f"[Train] 100% Backprop-Free Target Propagation | LR={LR}")

    # FORCE autograd off globally. This guarantees zero backprop.
    with torch.no_grad():
        for epoch in range(EPOCHS):
            wrapper.hierarchy.reset_states(batch_size=token_streams.shape[0])
            epoch_loss = 0.0
            t0 = time.time()

            for step in range(total_steps):
                cur_id = token_streams[:, step]
                tgt_id = token_streams[:, step + 1]

                # 1. Free Phase
                emb_x = F.normalize(wrapper.embedding(cur_id), dim=-1)
                
                # Hierarchy settles (free prediction)
                hid = wrapper.hierarchy.predict_label(emb_x, max_steps=4, update_temporal=True)
                if hid.shape[1] > wrapper.embed_dim:
                    hid_trunc = hid[:, :wrapper.embed_dim]
                else:
                    hid_trunc = hid
                    
                combined = emb_x + 0.5 * hid_trunc
                logits = wrapper.output_head(combined)
                probs = F.softmax(logits, dim=-1)
                
                loss = F.cross_entropy(logits, tgt_id)
                epoch_loss += loss.item()

                # 2. Clamped Phase (Target Error)
                tgt_oh = F.one_hot(tgt_id, wrapper.vocab_size).float()
                error = tgt_oh - probs  # [B, V]
                
                # 3. Target Propagation (Manual Delta Rule updates)
                # Output Head Update
                grad_W = (error.t() @ combined) / BATCH_SIZE  # [V, E]
                wrapper.output_head.weight.data += LR * grad_W
                
                # Propagate Target downwards
                grad_combined = error @ wrapper.output_head.weight.data  # [B, E]
                
                # Embedding Update
                for i in range(BATCH_SIZE):
                    wrapper.embedding.weight.data[cur_id[i]] += LR * grad_combined[i]
                
                # Hierarchy Update
                # We tell the hierarchy what its hidden state SHOULD have been
                target_hid = hid_trunc + (grad_combined * 2.0)
                
                # Pad back to full hierarchy dimension if necessary
                if hid.shape[1] > wrapper.embed_dim:
                    pad = torch.zeros(BATCH_SIZE, hid.shape[1] - wrapper.embed_dim, device=DEVICE)
                    target_full = torch.cat([target_hid, pad], dim=1)
                else:
                    target_full = target_hid
                    
                # Force SNAP-ATP to learn the propagated target
                # This naturally morphs the internal manifolds!
                wrapper.hierarchy.infer_and_learn(emb_x, top_level_label=target_full, max_steps=4)

                if step % 500 == 0:
                    avg_loss = epoch_loss / max(1, step + 1)
                    ppl = math.exp(min(avg_loss, 20))
                    print(f"    [Step {step:>5}/{total_steps}] Loss: {avg_loss:.4f} | Target PPL: {ppl:.1f}", flush=True)

            avg_loss = epoch_loss / max(1, total_steps)
            ppl = math.exp(min(avg_loss, 20))
            acc = evaluate(wrapper, token_streams[:8], eval_steps=EVAL_STEPS)
            dt = time.time() - t0
            print(f"\n  Epoch {epoch + 1:>2}/{EPOCHS} | Target PPL {ppl:.1f} | Top-1 {acc * 100:.2f}% | {dt:.0f}s")

            for prompt in PROMPTS[:2]:
                out = generate(wrapper, wrapper._tokenizer, prompt)
                safe = out.encode("ascii", errors="replace").decode("ascii")
                print(f"  [{prompt}] -> {safe}")

def main() -> None:
    sep("AGNIS V10 Target Propagation Probe (100% Backprop-Free)")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {EPOCHS}")

    tokenizer, tokenizer_path = load_tokenizer()
    print(f"[Tokenizer] Loaded {tokenizer_path} | vocab={tokenizer.get_vocab_size()}")

    wrapper = AGNISSLMWrapper(vocab_size=tokenizer.get_vocab_size(), device=DEVICE)
    try:
        wrapper.load_checkpoint(CHECKPOINT_IN)
        print(f"[Checkpoint] Loaded {CHECKPOINT_IN}")
    except:
        print(f"[WARNING] Checkpoint {CHECKPOINT_IN} not found. Starting fresh.")
        
    wrapper.to(wrapper.device)
    wrapper._tokenizer = tokenizer

    text = load_corpus()
    token_ids = tokenizer.encode(text).ids
    print(f"[Tokenize] {len(token_ids):,} tokens")

    streams = build_streams(token_ids, BATCH_SIZE, DEVICE)
    agent = CognitivePredictiveAgent(wrapper.hierarchy, device=DEVICE)

    sep("Training")
    train_target_prop(wrapper, agent, streams)

    sep("Final Samples")
    final_acc = evaluate(wrapper, streams[:8], eval_steps=EVAL_STEPS)
    print(f"[Eval] Next-token top-1 accuracy: {final_acc * 100:.2f}%")
    for prompt in PROMPTS:
        out = generate(wrapper, tokenizer, prompt)
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"\nPrompt: {prompt}")
        print(f"Output: {safe}")

if __name__ == "__main__":
    main()
