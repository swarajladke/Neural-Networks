import os
import urllib.request
from slm.slm_tokenizer import CharTokenizer

class SLMDataset:
    def __init__(self, filepath: str = "slm/input.txt", seq_length: int = 16):
        self.filepath = filepath
        self.seq_length = seq_length
        self.tokenizer = CharTokenizer()
        self.data_indices = []
        
        if not os.path.exists(filepath):
            if "war_and_peace_ru.txt" in filepath:
                self.download_war_and_peace_ru()
            else:
                self.download_tiny_shakespeare()
            
        self.load_and_tokenize()

    def download_war_and_peace_ru(self):
        print("Downloading Crime and Punishment (Russian) from Project Gutenberg...")
        # Gutenberg ID 30663 is the Russian version of Crime and Punishment.
        url = "https://www.gutenberg.org/cache/epub/30663/pg30663.txt"
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, self.filepath)
            print(f"Saved to {self.filepath}")
        except Exception as e:
            print(f"Error downloading from Gutenberg: {e}. Falling back to small Russian corpus.")
            # If Gutenberg fails, I'll fallback to the small one to avoid crash
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("Fallback Russian text. Gutenberg download failed.")

    def download_tiny_shakespeare(self):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        urllib.request.urlretrieve(url, self.filepath)
        print(f"Saved to {self.filepath}")

    def load_and_tokenize(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            text = f.read()
            # If it's a massive Gutenberg file, let's truncate to 150k chars for CPU speed
            if len(text) > 150000:
                text = text[:150000]
            
        # Fit tokenizer and convert text to indices
        self.tokenizer.fit(text)
        self.data_indices = self.tokenizer.encode(text)
        print(f"Dataset loaded: {len(self.data_indices)} total tokens (characters).")

    def get_batches(self, batch_size: int = 16):
        """Yields (context_indices, target_index) pairs using rolling window."""
        contexts = []
        targets = []
        
        # We need seq_length characters for context, and +1 for the target
        start_points = range(0, len(self.data_indices) - self.seq_length - batch_size, batch_size) 
        
        for start in start_points:
            contexts.clear()
            targets.clear()
            for b in range(batch_size):
                idx = start + b
                ctx = self.data_indices[idx : idx + self.seq_length]
                tgt = self.data_indices[idx + self.seq_length]
                contexts.append(ctx)
                targets.append([tgt])
            yield contexts, targets
