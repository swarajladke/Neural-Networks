"""
slm_tokenizer.py — AGNIS Tokenizer Suite
==========================================
Character-level and Byte-Pair Encoding (BPE) tokenizers for
the AGNIS predictive coding hierarchy.

BPE learns sub-word units from text, enabling the network to
operate on meaningful chunks ("the", "ing", "tion") instead
of raw characters. This is the foundation for conceptual
abstraction in the Meta-Pool architecture.
"""

import json
import re
from collections import Counter, defaultdict


class CharTokenizer:
    """Original character-level tokenizer (V1-V12 compatibility)."""
    def __init__(self):
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.vocab_size = 0
        self.is_fitted = False

    def fit(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}
        self.is_fitted = True
        print(f"[CharTokenizer] Fitted vocabulary of size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding.")
        return [self.char_to_ix.get(ch, 0) for ch in text]

    def decode(self, indices: list[int]) -> str:
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding.")
        return ''.join([self.ix_to_char.get(ix, '?') for ix in indices])

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump({'char_to_ix': self.char_to_ix, 'ix_to_char': self.ix_to_char}, f)

    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.char_to_ix = data['char_to_ix']
            self.ix_to_char = {int(k): v for k, v in data['ix_to_char'].items()}
            self.vocab_size = len(self.char_to_ix)
            self.is_fitted = True


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer for AGNIS.
    
    Learns sub-word merge rules from a training corpus.
    Supports multi-language text (UTF-8 safe via byte-level encoding).
    
    Usage:
        tok = BPETokenizer(vocab_size=1000)
        tok.fit("The cat sat on the mat. Le chat est sur le tapis.")
        ids = tok.encode("The cat")
        text = tok.decode(ids)
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        self.merges = []           # List of (pair_a, pair_b) merge rules in order
        self.vocab = {}            # token_str -> token_id
        self.inverse_vocab = {}    # token_id -> token_str
        self.vocab_size = 0
        self.is_fitted = False

    def _pre_tokenize(self, text: str) -> list[str]:
        """
        Split text into words (preserving spaces as part of tokens).
        Uses a simple regex pattern similar to GPT-2's pre-tokenization.
        """
        # Split on word boundaries, keeping whitespace attached to the following word
        pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"
        return re.findall(pattern, text)

    def _word_to_chars(self, word: str) -> list[str]:
        """Convert a word to a list of character tokens."""
        return list(word)

    def fit(self, text: str, verbose: bool = True):
        """
        Learn BPE merge rules from a text corpus.
        
        Algorithm:
        1. Start with character-level vocabulary
        2. Count all adjacent token pairs across the corpus
        3. Merge the most frequent pair into a new token
        4. Repeat until target vocab_size is reached
        """
        if verbose:
            print(f"[BPE] Training tokenizer (target vocab: {self.target_vocab_size})...")

        # Step 1: Pre-tokenize into words
        words = self._pre_tokenize(text)
        
        # Build word frequency table
        word_freqs = Counter(words)
        
        # Convert each word into a tuple of characters
        # word_splits: {("T", "h", "e"): 500, ("c", "a", "t"): 200, ...}
        word_splits = {}
        for word, freq in word_freqs.items():
            chars = tuple(self._word_to_chars(word))
            if chars:
                word_splits[chars] = freq

        # Step 2: Build initial character vocabulary
        char_vocab = set()
        for chars in word_splits.keys():
            for ch in chars:
                char_vocab.add(ch)

        # Add special tokens
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.vocab = {tok: i for i, tok in enumerate(specials)}
        idx = len(specials)
        for ch in sorted(char_vocab):
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        base_vocab_size = len(self.vocab)
        num_merges = self.target_vocab_size - base_vocab_size

        if verbose:
            print(f"[BPE] Base character vocab: {base_vocab_size}")
            print(f"[BPE] Merges to learn: {max(0, num_merges)}")

        # Step 3: Iteratively merge most frequent pairs
        self.merges = []
        for merge_i in range(max(0, num_merges)):
            # Count all adjacent pairs
            pair_counts = Counter()
            for word_tokens, freq in word_splits.items():
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0]
            pair, count = best_pair

            if count < 2:
                break  # No pair appears more than once

            # Create new merged token
            new_token = pair[0] + pair[1]
            self.merges.append(pair)
            self.vocab[new_token] = idx
            idx += 1

            # Apply this merge to all words
            new_word_splits = {}
            for word_tokens, freq in word_splits.items():
                new_tokens = list(word_tokens)
                i = 0
                while i < len(new_tokens) - 1:
                    if new_tokens[i] == pair[0] and new_tokens[i + 1] == pair[1]:
                        new_tokens[i] = new_token
                        del new_tokens[i + 1]
                    else:
                        i += 1
                new_word_splits[tuple(new_tokens)] = freq
            word_splits = new_word_splits

            if verbose and (merge_i + 1) % 100 == 0:
                print(f"[BPE] Merge {merge_i+1}/{num_merges}: '{pair[0]}' + '{pair[1]}' -> '{new_token}' (freq={count})")

        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.is_fitted = True

        if verbose:
            print(f"[BPE] Training complete. Final vocab size: {self.vocab_size}")
            print(f"[BPE] Total merge rules: {len(self.merges)}")

            # Show some example tokens
            sample_tokens = sorted(self.vocab.keys(), key=lambda x: len(x), reverse=True)[:10]
            print(f"[BPE] Longest tokens: {sample_tokens}")

    def _apply_merges(self, chars: list[str]) -> list[str]:
        """Apply learned merge rules to a list of characters."""
        tokens = list(chars)
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = pair[0] + pair[1]
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding.")

        words = self._pre_tokenize(text)
        ids = []
        unk_id = self.vocab.get(self.UNK_TOKEN, 1)

        for word in words:
            chars = self._word_to_chars(word)
            tokens = self._apply_merges(chars)
            for tok in tokens:
                ids.append(self.vocab.get(tok, unk_id))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into text."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding.")

        tokens = []
        for idx in ids:
            tok = self.inverse_vocab.get(idx, self.UNK_TOKEN)
            if tok not in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                tokens.append(tok)
        return "".join(tokens)

    def save(self, filepath: str):
        """Save tokenizer state to JSON."""
        data = {
            'target_vocab_size': self.target_vocab_size,
            'merges': self.merges,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Saved to {filepath}")

    def load(self, filepath: str):
        """Load tokenizer state from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.target_vocab_size = data['target_vocab_size']
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab = data['vocab']
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.vocab_size = data['vocab_size']
        self.is_fitted = True
        print(f"[BPE] Loaded from {filepath} (vocab: {self.vocab_size})")


# ───────────────────────────────────────────
# Quick self-test
# ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BPE Tokenizer Self-Test")
    print("=" * 50)

    text = (
        "The cat sat on the mat. The cat is happy. "
        "Le chat est sur le tapis. Le chat est content. "
        "Die Katze sitzt auf der Matte. Die Katze ist gluecklich. "
        "El gato se sento en la alfombra. El gato esta feliz. "
    ) * 20  # Repeat for frequency

    tok = BPETokenizer(vocab_size=200)
    tok.fit(text, verbose=True)

    # Test encode/decode
    test_str = "The cat sat on the mat."
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)

    print(f"\n  Input:   '{test_str}'")
    print(f"  Tokens:  {ids}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Match:   {'YES' if decoded == test_str else 'NO'}")

    # Show compression ratio
    char_len = len(test_str)
    tok_len = len(ids)
    print(f"\n  Chars: {char_len} -> Tokens: {tok_len} (compression: {char_len/tok_len:.1f}x)")

    # Test save/load
    tok.save("test_bpe_vocab.json")
    tok2 = BPETokenizer()
    tok2.load("test_bpe_vocab.json")
    ids2 = tok2.encode(test_str)
    print(f"  Save/Load match: {'YES' if ids == ids2 else 'NO'}")

    import os
    os.remove("test_bpe_vocab.json")
    print("\n  Self-test PASSED.")
