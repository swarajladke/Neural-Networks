"""
student_encoder.py — Standalone sub-10M parameter Recurrent Student Encoder.
=============================================================================
Defines the bidirectional GRU architecture with attention pooling and 128D projection
to replace SmolLM2 (360M) at runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentEncoder(nn.Module):
    def __init__(self, vocab_size=4096, embed_dim=192, hidden_dim=256, output_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Token embeddings (4096 x 192 = 786,432 parameters)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2-layer Bidirectional GRU
        # Layer 1 parameters: 3 * (192 * 256 + 256 * 256) = 344,064
        # Layer 2 parameters: 3 * (512 * 256 + 256 * 256) = 589,824
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Attention Pooling (maps 512D to 1D attention score)
        # Parameters: 512 * 128 + 128 * 1 = 65,664
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
        # Output coordinate projection (maps 512D to 128D projection)
        # Parameters: 512 * 128 = 65,536
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
        # Compute raw scores: (B, T, 1)
        attn_scores = self.attention_proj(gru_out)
        
        if attention_mask is not None:
            # Mask out padding tokens by setting their attention logits to -inf
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
    # Smoke test model parameter count
    model = StudentEncoder()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student Encoder parameters: {param_count:,}")
