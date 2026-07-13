"""
student_encoder.py — Standalone sub-10M parameter Recurrent Student Encoder.
=============================================================================
Defines the bidirectional GRU architecture with attention pooling and 128D projection
to replace SmolLM2 (360M) at runtime. Uses the SmolLM2 tokenizer vocabulary directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentEncoder(nn.Module):
    def __init__(self, vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Token embeddings (49152 * 128 = 6,291,456 parameters)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 1-layer Bidirectional GRU (~593k parameters)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention Pooling (maps 512D to 1D attention score)
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
        # Output coordinate projection (maps 512D to 128D projection)
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the student encoder.
        input_ids: tensor of shape (B, T)
        attention_mask: tensor of shape (B, T) with 1 for active tokens and 0 for padding
        Returns: normalized coordinates of shape (B, output_dim)
        """
        # 1. Embed tokens
        x = self.embedding(input_ids)  # (B, T, embed_dim)
        
        # 2. Pass through Bidirectional GRU
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim * 2)
        
        # 3. Attention-based Mean Pooling
        attn_scores = self.attention_proj(gru_out)
        
        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        
        # Weighted sum: (B, hidden_dim * 2)
        pooled = (gru_out * attn_weights).sum(dim=1)
        
        # 4. Project and Normalize to unit sphere
        z = self.projection(pooled)  # (B, output_dim)
        z = F.normalize(z, dim=-1, eps=1e-8)
        
        return z

if __name__ == "__main__":
    # Parameter count smoke test
    model = StudentEncoder()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student Encoder parameters: {param_count:,}")
