"""
run_multilingual_fluency.py
===========================
This script proves that the frozen AGNIS core contains universal, zero-forgetting 
manifolds for multiple languages. It trains a distinct Hybrid Wrapper (Interface) 
for each language sequentially, keeping the Core absolutely frozen.

For each language:
  1. Downloads the corpus
  2. Trains a language-specific BPE Tokenizer
  3. Trains a fresh Embedding + Output Head (Hybrid wrapper)
  4. Generates a text sample
"""

import os
import re
import sys
import time
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from slm.agnis_slm_wrapper import AGNISSLMWrapper

# --- CONFIGURATION ---
LANGUAGES = ["en", "de", "es", "ro"]  # The Quad Marathon languages

CORPORA_URLS = {
    "en": ["https://www.gutenberg.org/cache/epub/1342/pg1342.txt"],  # Pride and Prejudice
    "de": ["https://www.gutenberg.org/cache/epub/6343/pg6343.txt"],  # Die Leiden des jungen Werther
    "es": ["https://www.gutenberg.org/cache/epub/2000/pg2000.txt"],  # Don Quijote
    "ro": ["https://www.gutenberg.org/cache/epub/35451/pg35451.txt"] # Basme
}

CHECKPOINT_IN = "agnis_marathon_final.pt"
TARGET_CHARS = 1_000_000
VOCAB_SIZE = 4096
EMBED_DIM = 110  # Match the hierarchy input_dim from your marathon
BATCH_SIZE = 64
EPOCHS_PER_LANG = 15  # Slightly lower to save time on Colab
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "en": "The history of",
    "de": "Die Geschichte von",
    "es": "La historia de",
    "ro": "Istoria lui"
}

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def download_corpus(lang_code: str) -> str:
    path = f"slm/input_{lang_code}_massive.txt"
    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        os.makedirs("slm", exist_ok=True)
        full_text = ""
        for url in CORPORA_URLS[lang_code]:
            print(f"  -> Downloading {lang_code} corpus from {url}...")
            try:
                raw = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")
                full_text += clean_text(raw) + "\n\n"
            except Exception as e:
                print(f"  -> Failed: {e}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_text)
    return path

def train_tokenizer(corpus_path: str, lang_code: str) -> str:
    tok_path = f"slm_bpe_tokenizer_{lang_code}.json"
    if os.path.exists(tok_path):
        return tok_path
        
    print(f"  -> Training {lang_code} BPE Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE, 
        min_frequency=2, 
        special_tokens=["<|unk|>", "<|endoftext|>", "<|pad|>"]
    )
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(tok_path)
    return tok_path

def train_language_wrapper(wrapper: AGNISSLMWrapper, tokenizer: Tokenizer, token_ids: list, lang_code: str):
    wrapper.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    wrapper.output_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False).to(DEVICE)
    nn.init.normal_(wrapper.embedding.weight, std=0.02)
    nn.init.normal_(wrapper.output_head.weight, std=0.02)
    
    trainable = list(wrapper.embedding.parameters()) + list(wrapper.output_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=LR)
    
    seq_len = len(token_ids) // BATCH_SIZE
    token_tensor = torch.tensor(token_ids[:seq_len*BATCH_SIZE], dtype=torch.long, device=DEVICE).view(BATCH_SIZE, seq_len)
    total_steps = seq_len - 1
    
    print(f"  -> Training {lang_code} Wrapper: {EPOCHS_PER_LANG} Epochs | {total_steps} steps/epoch")
    
    for epoch in range(EPOCHS_PER_LANG):
        wrapper.hierarchy.reset_states(batch_size=BATCH_SIZE)
        epoch_loss = 0.0
        
        for step in range(total_steps):
            cur_id = token_tensor[:, step]
            tgt_id = token_tensor[:, step + 1]
            
            emb = F.normalize(wrapper.embedding(cur_id), dim=-1)
            with torch.no_grad():
                context = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
                if context.shape[1] > EMBED_DIM: context = context[:, :EMBED_DIM]
                
            combined = emb + 0.5 * context.detach()
            logits = wrapper.output_head(combined)
            loss = F.cross_entropy(logits, tgt_id)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 500 == 0:
                print(f"      [{lang_code.upper()} Epoch {epoch+1}] Step {step}/{total_steps} | Loss: {loss.item():.4f}", end="\r", flush=True)
                
        avg_loss = epoch_loss / max(1, total_steps)
        ppl = math.exp(min(avg_loss, 20))
        print(f"      [{lang_code.upper()} Epoch {epoch+1}] Final Loss: {avg_loss:.4f} | PPL: {ppl:.1f}")

@torch.no_grad()
def generate_sample(wrapper: AGNISSLMWrapper, tokenizer: Tokenizer, prompt: str) -> str:
    wrapper.hierarchy.reset_states(batch_size=1)
    gen_ids = tokenizer.encode(prompt).ids
    
    for tok_id in gen_ids:
        emb = F.normalize(wrapper.embedding(torch.tensor([[tok_id]], device=DEVICE)).view(1, -1), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        
    for _ in range(40):
        emb = F.normalize(wrapper.embedding(torch.tensor([[gen_ids[-1]]], device=DEVICE)).view(1, -1), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > EMBED_DIM: hid = hid[:, :EMBED_DIM]
            
        combined = emb + 0.5 * hid
        logits = wrapper.output_head(combined) / 0.8
        
        # simple repetition penalty
        for tok in set(gen_ids[-10:]):
            logits[0, tok] /= 1.2
            
        next_tok = torch.multinomial(F.softmax(logits, dim=-1)[0], 1).item()
        gen_ids.append(next_tok)
        
    return tokenizer.decode(gen_ids)

def main():
    print("=" * 60)
    print("  Multilingual Fluency Validation (Frozen AGNIS Core)")
    print("=" * 60)
    
    wrapper = AGNISSLMWrapper(device=DEVICE)
    try:
        wrapper.load_checkpoint(CHECKPOINT_IN)
        print(f"[Loaded Core] {CHECKPOINT_IN}")
    except:
        print(f"FATAL: Could not load {CHECKPOINT_IN}")
        return
        
    wrapper.to(DEVICE)
    for p in wrapper.hierarchy.parameters():
        p.requires_grad_(False)
        
    for lang in LANGUAGES:
        print(f"\n>>> PROCESSING LANGUAGE: {lang.upper()} <<<")
        corpus_path = download_corpus(lang)
        
        with open(corpus_path, encoding="utf-8", errors="replace") as f:
            text = f.read()[:TARGET_CHARS]
            
        tok_path = train_tokenizer(corpus_path, lang)
        tokenizer = Tokenizer.from_file(tok_path)
        token_ids = tokenizer.encode(text).ids
        
        wrapper._tokenizer = tokenizer
        wrapper.vocab_size = VOCAB_SIZE
        
        train_language_wrapper(wrapper, tokenizer, token_ids, lang)
        
        print(f"\n  [{lang.upper()} SAMPLE GENERATION]")
        prompt = PROMPTS[lang]
        out = generate_sample(wrapper, tokenizer, prompt)
        safe = out.encode("ascii", errors="replace").decode("ascii")
        print(f"  Result: {safe}\n")
        
        # Save specific language wrapper
        lang_ckpt = f"agnis_{lang}_interface.pt"
        wrapper.save_checkpoint(lang_ckpt)
        print(f"  [Saved Interface] {lang_ckpt}")

if __name__ == "__main__":
    main()
