import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent

class AGNIS_SLM_Wrapper(nn.Module):
    def __init__(self, vocab_size: int, seq_length: int = 16, embed_dim: int = 16, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        # SLM Interface components
        self.embeddings = nn.Embedding(vocab_size, embed_dim).to(device)
        
        # AGNIS Architecture: [Context In] -> [Hidden] -> [Hidden] -> [Predicted Embed Out]
        input_dim = seq_length * embed_dim
        output_dim = embed_dim
        
        self.hierarchy = PredictiveHierarchy([input_dim, 64, 64, output_dim], device=device)
        self.agent = CognitivePredictiveAgent(self.hierarchy, device=device)
        
    def _prepare_tensors(self, context_indices: list[list[int]], target_indices: list[list[int]]):
        ctx_tensor = torch.tensor(context_indices, dtype=torch.long, device=self.device)
        tgt_tensor = torch.tensor(target_indices, dtype=torch.long, device=self.device)
        
        # Embed contexts: [batch, seq, embed_dim]
        embedded_ctx = self.embeddings(ctx_tensor)
        # Flatten temporal window into 1D snapshot for AGNIS: [batch, seq * embed_dim]
        flat_ctx = embedded_ctx.view(embedded_ctx.shape[0], -1)
        
        # Embed targets: [batch, 1, embed_dim] -> [batch, embed_dim]
        embedded_tgt = self.embeddings(tgt_tensor).squeeze(1)
        
        return flat_ctx, embedded_tgt

    def learn_step(self, context_indices: list[list[int]], target_indices: list[list[int]]):
        """Processes a single batch of character rolling windows with amortized inference.
        
        V5.1: The first sample in the batch cold-starts (full reset). All subsequent
        samples warm-start from the previous settled state. Since consecutive windows
        share 15/16 characters, warm-starting reduces settling from ~50 to ~5-10 steps.
        """
        flat_ctx, embedded_tgt = self._prepare_tensors(context_indices, target_indices)
        
        # AGNIS core loop (online learning per sample in the batch)
        total_weight = 0.0
        total_surprise = 0.0
        
        for i in range(flat_ctx.shape[0]):
            x = flat_ctx[i:i+1]
            y = embedded_tgt[i:i+1]
            # V5.1: First sample cold-starts, rest warm-start (amortized inference)
            use_warm = (i > 0)
            w, s = self.agent.observe_and_learn(x, y, task_id=0, max_steps=50, beta_push=3.0, warm_start=use_warm)
            total_weight += w
            total_surprise += s
            
        return total_weight / flat_ctx.shape[0], total_surprise / flat_ctx.shape[0]

    def dream_consolidation(self, batch_size=16):
        """Pass-through to AGNIS declarative memory replay"""
        return self.agent.dream_replay(batch_size=batch_size, max_steps=100)

    def generate(self, tokenizer, prompt: str, max_new_chars: int = 100, temperature: float = 0.8):
        """Autoregressively sample strings text (like ChatGPT) using temperature-based sampling."""
        print(f"\\n--- AGNIS Generating Text (Temp: {temperature}) ---")
        print(prompt, end="")
        
        self.hierarchy.eval() # Optional: if batchnorm etc were there
        
        # Seed the initial context
        context_str = prompt
        # Pad with spaces if prompt is too short
        if len(context_str) < self.seq_length:
            context_str = " " * (self.seq_length - len(context_str)) + context_str
            
        # Keep only the last seq_length chars
        context_str = context_str[-self.seq_length:]
        current_ctx_indices = tokenizer.encode(context_str)

        generated = ""
        with torch.no_grad():
            for _ in range(max_new_chars):
                ctx_tensor = torch.tensor([current_ctx_indices], dtype=torch.long, device=self.device)
                embedded_ctx = self.embeddings(ctx_tensor)
                flat_ctx = embedded_ctx.view(1, -1)
                
                # Ask AGNIS to predict the readout (what the next embedding should look like)
                predicted_embed = self.hierarchy.predict_label(flat_ctx)
                
                # Decode: Inverse-distance sampling with Temperature
                # Shape: embeddings.weight is [vocab_size, embed_dim]
                # Distance to all vocab embeddings
                distances = torch.norm(self.embeddings.weight - predicted_embed[0], dim=1)
                
                # Convert distance to probability (lower distance = higher probability)
                # We use 1/dist or exp(-dist)
                # weights = 1.0 / (distances + 1e-8)
                # Better: Boltzmann distribution on negative distances
                logits = -distances 
                probs = torch.softmax(logits / temperature, dim=0)
                
                # Sample from distribution
                best_char_idx = torch.multinomial(probs, 1).item()
                
                next_char = tokenizer.decode([best_char_idx])
                generated += next_char
                print(next_char, end="", flush=True)
                
                # Slide the window forward
                current_ctx_indices.pop(0)
                current_ctx_indices.append(best_char_idx)
                
        print("\\n--- End Generation ---")
        return generated
