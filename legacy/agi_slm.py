import torch
import torch.nn as nn
import string
import pickle
import os
import random
from typing import List, Optional
from neural_architecture import NeuralArchitecture, ModuleState
from replay_buffer import ReplayBuffer

class AGISmallLanguageModel(nn.Module):
    """
    An AGI-powered SLM that uses a Reasoning Mesh and MoE Core.
    Phase 20: Memory-Augmented Continual Learning via Experience Replay.
    """
    def __init__(self, vocab_chars: List[str], d_model: int = 256, replay_capacity: int = 500, router_temp: float = 2.0):
        super().__init__()
        self.chars = vocab_chars
        self.vocab_size = len(vocab_chars)
        self.d_model = d_model
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # The "Brain" - Your AGI Architecture
        self.brain = NeuralArchitecture(
            input_dim=d_model,
            state_dim=d_model,
            memory_dim=512,
            reasoning_dim=256,
            action_dim=d_model,
            use_moe=True,
            use_sparse_mesh=True,
            num_experts=16,
            router_temp=router_temp
        )
        
        # Single Shared Embedding (All weights plastic)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        
        # Single Shared Output Head
        self.lm_head = nn.Linear(d_model, self.vocab_size)
        
        # Persistent Optimizer for the wrapper layers
        self.wrapper_optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': 1e-3},
            {'params': self.lm_head.parameters(), 'lr': 1e-3}
        ])
        
        # PHASE 20: Experience Replay for Continual Learning
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.replay_ratio = 0.3  # 30% of steps include replay
        self.replay_samples = 4  # Number of old samples per replay step

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi.get(c, self.stoi.get(' ', 0)) for c in s], dtype=torch.long)

    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos.get(i, '?') for i in l])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # PHASE 18: Single shared embedding (all weights plastic)
        embeddings = self.embedding(input_tensor)
        
        # Process through AGI Brain
        logits = []
        for t in range(embeddings.size(0)):
            # Step the brain
            cognitive_out = self.brain.forward(embeddings[t])
            
            # PHASE 18: Single shared output head
            step_logits = self.lm_head(cognitive_out)
            logits.append(step_logits)
        return torch.stack(logits)

    def runtime_learn(self, input_str: str, target_str: str, learning_rate: Optional[float] = 0.01, task_id: Optional[int] = 0):
        """
        PHASE 20: Memory-Augmented Continual Learning via Experience Replay.
        No freezing, no complex penalties. Just "dreaming" about the past.
        """
        self.train()
        x = self.encode(input_str)
        y = self.encode(target_str)
        
        # Set task context for routing (if applicable)
        if task_id is not None:
            self.brain.current_state.task_id = task_id
        
        # Add to long-term memory for future replay
        self.replay_buffer.add(input_str, target_str, task_id)
        
        # Update LR
        for param_group in self.wrapper_optimizer.param_groups:
            param_group['lr'] = learning_rate or 0.01

        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        
        # 1. Learn from CURRENT data (Short-term sequence)
        # We use a small local buffer for sequence learning
        local_buffer = []
        for i in range(len(x)):
            local_buffer.append((x[i:i+1], y[i:i+1]))
            if len(local_buffer) > 16: local_buffer.pop(0)
            
            self.wrapper_optimizer.zero_grad()
            
            # Step A: Local Replay (Burn-in current sequence)
            for rx, ry in local_buffer:
                rx_emb = self.embedding(rx).detach().squeeze(0)
                ry_emb = self.embedding(ry).detach().squeeze(0)
                
                self.brain.learn(rx_emb, ry_emb)
                
                self.brain.reset_caches()
                self.brain.current_state.module_state = self.brain.current_state.module_state.detach()
                
                logits = self.forward(rx)
                loss = criterion(logits.view(-1, self.vocab_size), ry)
                loss.backward()
                total_loss += loss.item()
            
            # Step B: Long-term Replay (Prevent forgetting)
            if random.random() < self.replay_ratio:
                # Sample few experiences from past tasks
                past_experiences = self.replay_buffer.sample(self.replay_samples)
                for p_in, p_tgt, p_tid in past_experiences:
                    # Skip if we just added this exact sample and it's redundant
                    if p_in == input_str and p_tid == task_id: continue 
                    
                    # Set task context for the replayed task
                    self.brain.current_state.task_id = p_tid
                    
                    px = self.encode(p_in)
                    py = self.encode(p_tgt)
                    
                    # Just sample a random token from the past experience for speed
                    ridx = random.randint(0, len(px)-1)
                    prx, pry = px[ridx:ridx+1], py[ridx:ridx+1]
                    
                    # DETACH & RESET: Crucial to keep the replay graphs isolated
                    self.brain.reset_caches()
                    self.brain.current_state.module_state = self.brain.current_state.module_state.detach()
                    # Forced context flush to prevent cross-domain interference during replay
                    self.brain.current_state.module_state.context_window = None 
                    
                    logits_p = self.forward(prx)
                    loss_p = criterion(logits_p.view(-1, self.vocab_size), pry)
                    
                    # SAFETY: Check for NaN before backward
                    if not torch.isnan(loss_p):
                        # Weight the past slightly lower
                        (0.5 * loss_p).backward()
                    
                    # Detach again after backward
                    self.brain.current_state.module_state = self.brain.current_state.module_state.detach()
                
                # Re-set task ID to current for the next main loop iteration
                self.brain.current_state.task_id = task_id
            
            # Final Gradient Sanity Check
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Safe Step: Skip if gradients became NaN
            valid_grads = True
            for param in self.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    valid_grads = False
                    break
            
            if valid_grads:
                self.wrapper_optimizer.step()
                # Periodically save good states
                if i % 50 == 0:
                    self.brain.save_checkpoint()
            else:
                print("⚠️  NaN Detected! Triggering Self-Healing Rollback...")
                self.wrapper_optimizer.zero_grad() # Discard corrupted update
                self.brain.rollback() # Revert brain to last good state
                # Also reset embedding/head if they hit NaN
                with torch.no_grad():
                    if torch.isnan(self.embedding.weight).any() or torch.isnan(self.lm_head.weight).any():
                         # If wrapper layers also died, we reset them specifically
                         # (rare, but possible if brain feedback was extreme)
                         self.embedding.weight.data.fill_(0).add_(torch.randn_like(self.embedding.weight) * 0.01)
                         self.lm_head.weight.data.fill_(0).add_(torch.randn_like(self.lm_head.weight) * 0.01)

        return total_loss / max(len(local_buffer), 1)

    @torch.no_grad()
    def generate(self, start_text: str, max_new_tokens: int = 100, temperature: float = 0.8, task_id: Optional[int] = None):
        self.eval()
        idx = self.encode(start_text)
        self.brain.current_state.module_state = ModuleState()
        
        # ACTIVATE: Set the task context
        if task_id is not None:
             self.brain.current_state.task_id = task_id
        
        # Warm up prefix
        for i in range(len(idx) - 1):
            _ = self.forward(idx[i:i+1])
            
        current_token = idx[-1:]
        result = list(idx.numpy())
        
        for _ in range(max_new_tokens):
            logits = self.forward(current_token)
            logits = logits[-1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            result.append(next_token.item())
            current_token = next_token
            if len(result) > 500: break # Safety limit
                
        return self.decode(result)

def load_agi_data():
    if not os.path.exists("agi_training_data.txt"):
        print("❌ Data not found. Run import_agi_data.py first.")
        return None, None
        
    with open("agi_training_data.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    with open("agi_vocab.pkl", "rb") as f:
        chars = pickle.load(f)
        
    return text, chars

def train_on_tinystories():
    print("🧠 Loading TinyStories for AGI Validation...")
    text, chars = load_agi_data()
    if text is None: return

    model = AGISmallLanguageModel(chars)
    data = model.encode(text)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Slightly lower LR for stability
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 1 
    seq_len = 16
    
    print(f"🚀 Training on {len(data):,} tokens. Sequences: {seq_len}")
    print("-" * 50)

    for epoch in range(200):
        # Random start point
        ix = torch.randint(len(data) - seq_len, (1,)).item()
        x = data[ix:ix+seq_len]
        y = data[ix+1:ix+seq_len+1]
        
        # Reset AGI brain state and caches
        model.brain.reset_caches()
        model.brain.current_state.module_state = ModuleState()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        
        # Forward
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            # Calculate AGI utilization
            agi_info = model.brain.get_system_info()
            avg_stability = sum(m['stability_score'] for m in agi_info['stability_metrics'].values()) / 4
            print(f"   Step {epoch:3d} | Loss: {loss.item():.4f} | System Stability: {avg_stability:.2f}")
            
        if epoch % 50 == 0 and epoch > 0:
            print("\n✍️ Sustained Reasoning Sample:")
            seed = "Once upon a time, there was a little"
            gen = model.generate(seed, max_new_tokens=50)
            print(f"   > {gen}\n")
            model.train()

    print("\n✅ AGI Validation Training Complete!")
    print("Testing Final Generation...")
    final_seed = "The moral of the story is"
    print(f"   Result: {model.generate(final_seed, max_new_tokens=100)}")

if __name__ == "__main__":
    train_on_tinystories()
