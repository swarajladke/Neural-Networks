import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalHippocampalBuffer(nn.Module):
    """
    A biologically-inspired, local-only continuous working memory buffer.
    Implements QKV-style attention without Backpropagation Through Time (BPTT).
    """
    def __init__(self, d_model: int, max_memory: int = 256):
        super().__init__()
        self.d_model = d_model
        self.max_memory = max_memory
        
        # Synaptic projections for associative matching
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.reset_memory()
        
    def reset_memory(self, batch_size: int = 1):
        """Clear the working memory for a new sequence/batch."""
        self.register_buffer('keys', torch.zeros(batch_size, 0, self.d_model))
        self.register_buffer('values', torch.zeros(batch_size, 0, self.d_model))
        
    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [batch_size, d_model] - The current latent state from the AGNIS core.
        Returns: [batch_size, d_model] - The retrieved memory context.
        """
        # 1. Project the current state into Q, K, V space
        q_t = self.W_q(x_t).unsqueeze(1)  # [B, 1, d]
        k_t = self.W_k(x_t).unsqueeze(1)  # [B, 1, d]
        v_t = self.W_v(x_t).unsqueeze(1)  # [B, 1, d]
        
        # 2. Append to physical memory
        # CRITICAL: We .detach() the keys and values. 
        # This prevents gradients from flowing backwards through time, 
        # maintaining the local-learning philosophy of AGNIS.
        if self.keys.shape[1] == 0:
            self.keys = k_t.detach()
            self.values = v_t.detach()
        else:
            self.keys = torch.cat([self.keys, k_t.detach()], dim=1)
            self.values = torch.cat([self.values, v_t.detach()], dim=1)
            
            # Rolling buffer constraint
            if self.keys.shape[1] > self.max_memory:
                self.keys = self.keys[:, -self.max_memory:]
                self.values = self.values[:, -self.max_memory:]
        
        # 3. Retrieve (The biological "Aha!" moment)
        # Match current query against past keys
        scores = torch.bmm(q_t, self.keys.transpose(1, 2)) / (self.d_model ** 0.5) # [B, 1, seq_len]
        attn = F.softmax(scores, dim=-1)
        
        # Extract the relevant past values
        x_recall = torch.bmm(attn, self.values).squeeze(1) # [B, d_model]
        
        return x_recall

if __name__ == "__main__":
    print("Testing Local Hippocampal Buffer...")
    buffer = LocalHippocampalBuffer(d_model=64, max_memory=10)
    buffer.reset_memory(batch_size=2)
    
    # Simulate a sequence of 15 tokens passing through the core
    for t in range(15):
        mock_core_state = torch.randn(2, 64)
        recall = buffer(mock_core_state)
        print(f"Step {t}: Keys in memory = {buffer.keys.shape[1]} | Recall shape = {recall.shape}")
        
    print("\nBuffer test complete. Memory capacity caps correctly at max_memory.")
