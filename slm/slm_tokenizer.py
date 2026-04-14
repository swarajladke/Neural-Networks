class CharTokenizer:
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
        print(f"[Tokenizer] Fitted vocabulary of size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding.")
        return [self.char_to_ix.get(ch, 0) for ch in text]  # Fallback to 0 if unknown

    def decode(self, indices: list[int]) -> str:
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding.")
        return ''.join([self.ix_to_char.get(ix, '?') for ix in indices])

    def save(self, filepath: str):
        import json
        with open(filepath, 'w') as f:
            json.dump({'char_to_ix': self.char_to_ix, 'ix_to_char': self.ix_to_char}, f)

    def load(self, filepath: str):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.char_to_ix = data['char_to_ix']
            self.ix_to_char = {int(k): v for k, v in data['ix_to_char'].items()}
            self.vocab_size = len(self.char_to_ix)
            self.is_fitted = True
