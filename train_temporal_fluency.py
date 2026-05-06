import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from slm.temporal_core import TemporalFluencyModel


MODEL_OUT = "temporal_core_v1_en.pt"
TOKENIZER_PATH = "slm_bpe_tokenizer_en.json"
CORPUS_PATH = "slm/input_en_massive.txt"

TARGET_CHARS = 25_000_000
BATCH_SIZE = 64
EPOCHS = 20
LR = 4e-4
WARMUP_STEPS = 1000
LOG_EVERY = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "The history of",
    "Once upon a time",
    "It was the best of times",
    "She looked out the window and",
]

def load_corpus() -> str:
    if not os.path.exists(CORPUS_PATH):
        print(f"[ERROR] Corpus not found at {CORPUS_PATH}")
        sys.exit(1)
        
    with open(CORPUS_PATH, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = raw[:TARGET_CHARS]
    print(f"[Corpus] {len(text):,} chars | {len(text.split()):,} words")
    return text


def build_token_tensor(token_ids: list[int], batch_size: int, device: str) -> torch.Tensor:
    seq_len = len(token_ids) // batch_size
    token_ids = token_ids[: seq_len * batch_size]
    return torch.tensor(token_ids, dtype=torch.long, device=device).view(batch_size, seq_len)


@torch.no_grad()
def heldout_ppl(model: TemporalFluencyModel, tokens: torch.Tensor, steps: int = 512) -> tuple[float, float]:
    eval_steps = min(steps, tokens.shape[1] - 1)
    model.eval()
    model.reset_states(batch_size=tokens.shape[0])
    total_loss = 0.0

    for step in range(eval_steps):
        cur = tokens[:, step]
        tgt = tokens[:, step + 1]
        logits = model(cur, is_training=False)
        loss = F.cross_entropy(logits, tgt)
        total_loss += float(loss.item())

    avg = total_loss / max(1, eval_steps)
    return avg, math.exp(min(avg, 20))


def main() -> None:
    print("\n" + "=" * 60)
    print("  TEMPORAL CORE V1 TRAINING")
    print("  Persistent State | R-Matrix | Hippocampal Buffer")
    print("=" * 60)

    text = load_corpus()

    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ERROR] Tokenizer not found: {TOKENIZER_PATH}")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[Tokenizer] Loaded {TOKENIZER_PATH} | vocab={vocab_size}")

    model = TemporalFluencyModel(vocab_size=vocab_size, embed_dim=128, context_size=256)
    model.to(DEVICE)

    # Only embedding and output head are trained via backprop
    trainable = [
        *model.embedding.parameters(),
        *model.head.parameters()
    ]
    # Core gate parameter is also trained via backprop to learn *how* to use the memory
    trainable.append(model.core.mem_gate.weight)
    trainable.append(model.core.mem_gate.bias)

    n_params = sum(p.numel() for p in trainable)
    print(f"[Trainable] {n_params:,} parameters (Backprop path)")
    print(f"[Hebbian] {sum(p.numel() for p in model.core.parameters()):,} parameters (Local path)")

    enc = tokenizer.encode(text)
    token_tensor = build_token_tensor(enc.ids, BATCH_SIZE, DEVICE)
    total_steps = token_tensor.shape[1] - 1
    split = max(1024, token_tensor.shape[1] // 20)
    train_tokens = token_tensor[:, :-split]
    valid_tokens = token_tensor[:, -split:]

    print(f"[Tokenize] {len(enc.ids):,} tokens")
    print(f"[Train] {BATCH_SIZE} streams x {train_tokens.shape[1] - 1:,} steps | {EPOCHS} epochs")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        model.reset_states(batch_size=BATCH_SIZE)
        epoch_loss = 0.0
        start = time.time()

        for step in range(train_tokens.shape[1] - 1):
            cur = train_tokens[:, step]
            tgt = train_tokens[:, step + 1]

            if epoch == 0 and step <= WARMUP_STEPS:
                scale = max(0.01, step / max(1, WARMUP_STEPS))
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * scale

            logits = model(cur, is_training=True)
            loss = F.cross_entropy(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            
            # Re-tie weights if optimizer disturbed them
            model.head[-1].weight = model.embedding.weight

            epoch_loss += float(loss.item())

            if (step + 1) % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                speed = (step + 1) / max(time.time() - start, 1e-6)
                print(
                    f"  Epoch {epoch+1:>2}/{EPOCHS} | "
                    f"Step {step+1:>6}/{train_tokens.shape[1]-1} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | {speed:.0f} tok/s",
                    end="\r",
                    flush=True,
                )

        train_loss = epoch_loss / max(1, train_tokens.shape[1] - 1)
        train_ppl = math.exp(min(train_loss, 20))
        val_loss, val_ppl = heldout_ppl(model, valid_tokens)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_OUT)

        print(f"\n  Epoch {epoch+1:>2}/{EPOCHS} | Train Loss {train_loss:.4f} | Train PPL {train_ppl:.1f} | Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f} {' <- best' if improved else ''}")

        # Generation Sample
        print(f"\n  --- Samples (epoch {epoch+1}) ---")
        model.eval()
        for prompt in PROMPTS:
            enc_p = tokenizer.encode(prompt)
            prompt_ids = enc_p.ids
            
            model.reset_states(batch_size=1)
            
            # Warm up state
            for i in range(len(prompt_ids) - 1):
                tok = torch.tensor([prompt_ids[i]], dtype=torch.long, device=DEVICE)
                _ = model(tok, is_training=False)
                
            generated = list(prompt_ids)
            for _ in range(50):
                cur = torch.tensor([generated[-1]], dtype=torch.long, device=DEVICE)
                logits = model(cur, is_training=False)[0]
                logits = logits / 0.8 # Temperature
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                generated.append(next_id)
                if next_id == tokenizer.token_to_id("<|endoftext|>"):
                    break
                    
            text_gen = tokenizer.decode(generated).replace("\n", " ")
            print(f"  [{prompt}] -> {text_gen[:120]}...")
        print()

if __name__ == "__main__":
    main()
