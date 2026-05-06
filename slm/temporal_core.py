import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v23_hippocampal_memory import LocalHippocampalBuffer

class TemporalCoreV1(nn.Module):
    """
    Minimal Temporal Core enforcing persistent state and causal sequence learning.
    Strictly local Hebbian learning for the V and R matrices.
    """
    def __init__(self, input_dim: int, output_dim: int, max_memory: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Recognition Pathway
        k_v = (1.0 / input_dim) ** 0.5
        self.V = nn.Parameter(torch.empty(input_dim, output_dim).uniform_(-k_v, k_v))
        self.b_in = nn.Parameter(torch.zeros(output_dim))
        
        # Recurrent Pathway (R-Matrix)
        k_r = (1.0 / output_dim) ** 0.5
        self.R = nn.Parameter(torch.empty(output_dim, output_dim).uniform_(-k_r, k_r))
        
        # Persistent Hidden State
        self.register_buffer("x_t", torch.zeros(1, output_dim))
        self.register_buffer("x_prev", torch.zeros(1, output_dim))
        
        # Local Learning Hyperparameters
        self.eta_V = 0.05
        self.eta_R = 0.05
        self.decay = 0.001
        
        # Memory Integration
        self.memory = LocalHippocampalBuffer(d_model=output_dim, max_memory=max_memory)
        self.mem_gate = nn.Linear(output_dim, output_dim)
        
    def reset_states(self, batch_size: int = 1):
        self.x_t = torch.zeros(batch_size, self.output_dim, device=self.V.device)
        self.x_prev = torch.zeros(batch_size, self.output_dim, device=self.V.device)
        self.memory.reset_memory(batch_size=batch_size)
        self.memory.to(self.V.device)
        
    def forward(self, s_t: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        s_t: [batch, input_dim] (Embedding of current token)
        Returns: [batch, output_dim] (Memory-fused contextual state)
        """
        # 1. Shift Persistent State
        self.x_prev = self.x_t.detach()
        
        # 2. Recurrent Path (x_t = activation(V^T s_t + R^T x_{t-1}))
        drive = torch.matmul(s_t, self.V) + self.b_in
        recurrent_drive = torch.matmul(self.x_prev, self.R)
        self.x_t = F.leaky_relu(drive + recurrent_drive, negative_slope=0.1)
        
        # 3. Local Learning Rule (Hebbian/STDP style)
        if is_training:
            with torch.no_grad():
                # Outer product updates
                dV = torch.bmm(s_t.unsqueeze(2), self.x_t.unsqueeze(1)).mean(dim=0)
                dR = torch.bmm(self.x_prev.unsqueeze(2), self.x_t.unsqueeze(1)).mean(dim=0)
                
                self.V.add_(self.eta_V * dV)
                # R = R * (1 - decay) + dR
                self.R.mul_(1.0 - self.decay).add_(self.eta_R * dR)
                
                # Stability clamping
                self.V.data.clamp_(-3.0, 3.0)
                self.R.data.clamp_(-3.0, 3.0)
                
        # 4. Memory Integration (Gated Fusion)
        x_recall = self.memory(self.x_t)
        gate = torch.sigmoid(self.mem_gate(self.x_t))
        x_out = self.x_t + gate * x_recall
        
        return x_out


class TemporalFluencyModel(nn.Module):
    """
    Wraps the Temporal Core with a Gradient-Trained Embedding and Output Head.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, context_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # The new Temporal Core V1
        self.core = TemporalCoreV1(input_dim=embed_dim, output_dim=embed_dim, max_memory=context_size)
        
        # Output Head Upgrade: 3-Layer MLP to sharpen probability distribution
        hidden_dim = embed_dim * 4
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )
        
        # Weight Tying for efficiency
        self.head[-1].weight = self.embedding.weight
        
    def reset_states(self, batch_size: int = 1):
        self.core.reset_states(batch_size=batch_size)
        
    def forward(self, token_ids: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        Processes a single step for all streams in the batch.
        token_ids: [batch]
        Returns: [batch, vocab_size] logits
        """
        # 1. Embed (Gradient path)
        emb = self.embedding(token_ids)
        emb = F.normalize(emb, dim=-1)
        
        # 2. Core (Local Hebbian path)
        x_out = self.core(emb, is_training=is_training)
        
        # 3. Output Head (Gradient path)
        logits = self.head(x_out)
        
        return logits
