import sys
sys.path.append("c:/Users/Helios/Desktop/Neural Networks")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_continual_v2 import build_hybrid, RAW_FACTS, RETENTION_PROBES, INDEPENDENT_PPL_TEXTS
from agnis_continual_v4_2 import gpt2_forward, train_query_projection, TRAIN_PARAPHRASES, EVAL_PARAPHRASES
from fact_memory import EpisodicFactMemory

class SlowMemoryMLP(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=50257, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x):
        return self.mlp(x)

def test_consolidation():
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid = hybrid.to(device)
    vocab_size = hybrid.gpt2.config.vocab_size
    
    # 1. Initialize memory and write answer-only positions (pool_len=2)
    memory = EpisodicFactMemory(vocab_size=vocab_size, device=device)
    
    # Fact ranges for contrastive learning
    fact_ranges = {}
    answer_ids = {}
    total_stored = 0
    

        
    # Let's collect keys from statements
    for idx, f in enumerate(RAW_FACTS):
        # We simulate the exact write process
        ids = tokenizer.encode(f["statement"] + tokenizer.eos_token, return_tensors="pt").to(device)
        _, h = gpt2_forward(hybrid, ids)
        prompt = f["probe"]
        prompt_ids = tokenizer.encode(prompt)
        full_ids_list = ids[0].tolist()
        n = 0
        limit = min(len(prompt_ids), len(full_ids_list))
        while n < limit and full_ids_list[n] == prompt_ids[n]:
            n += 1
        if n < max(1, len(prompt_ids) // 2):
            n = limit
        boundary = max(0, n - 1)
        
        # Pool keys (pool_len=2)
        pooled = []
        for t in range(boundary, h.shape[1] - 1):
            actual = min(2, t + 1)
            pooled.append(h[0, t - actual + 1 : t + 1, :].mean(dim=0))
        h_answer = torch.stack(pooled)
        v_answer = ids[0, boundary + 1:]
        
        start = len(memory)
        memory.write(h_answer, v_answer)
        total_stored += h_answer.shape[0]
        fact_ranges[f["id"]] = (start, h_answer.shape[0])
        answer_ids[f["id"]] = v_answer.detach()
        
    print(f"Stored {len(memory)} keys.")
    
    # 2. Train query projection to align paraphrases
    qs_fact, pos_idx = [], []
    for f in RAW_FACTS:
        fid = f["id"]
        start, length = fact_ranges[fid]
        ans = answer_ids[fid]
        for para in TRAIN_PARAPHRASES[fid]:
            p_ids = tokenizer.encode(para, return_tensors="pt").to(device)
            n_cont = min(4, length - 1, ans.shape[0])
            for j in range(n_cont + 1):
                ids = p_ids if j == 0 else torch.cat([p_ids, ans[:j].view(1, -1)], dim=1)
                
                # pool last 2 tokens
                _, h = gpt2_forward(hybrid, ids)
                T = h.shape[1]
                q_pooled = h[0, -min(2, T):, :].mean(dim=0)
                qs_fact.append(q_pooled)
                pos_idx.append(start + j)
    q_fact = torch.stack(qs_fact)
    pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=device)
    
    # Collect control states
    qs_ctrl = []
    texts = [p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS)
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        _, h = gpt2_forward(hybrid, ids)
        T = h.shape[1]
        idx = torch.linspace(0, T - 1, steps=min(8, T)).long()
        for t in idx:
            actual = min(2, t + 1)
            qs_ctrl.append(h[0, t - actual + 1 : t + 1, :].mean(dim=0))
    q_ctrl = torch.stack(qs_ctrl)
    
    # Train projection
    train_query_projection(memory, q_fact, pos_idx, q_ctrl)
    
    # 3. Train SlowMemoryMLP directly on memory keys and values
    print("\n--- Training SlowMemoryMLP (Consolidation) ---")
    mlp = SlowMemoryMLP(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Keys for training: pass keys through read_space to match read space geometry
    mu, V_sub = memory.read_space()
    k_read = memory.to_read_space(memory.keys_raw, mu, V_sub).detach() # shape (M, E)
    
    # Wait, the MLP input dimension is E (768), but k_read is also shape E!
    # Let's train the MLP to map the PCA-projected keys to target tokens!
    for epoch in range(150):
        optimizer.zero_grad()
        logits = mlp(k_read) # (M, V)
        loss = F.cross_entropy(logits, memory.values)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 149:
            acc = (logits.argmax(dim=-1) == memory.values).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} | train-acc={acc*100:.1f}%")
            
    # 4. Evaluate consolidation on exact and held-out paraphrase queries
    print("\n--- Evaluating Consolidated Model (KV ablated) ---")
    mlp.eval()
    
    exact_correct = 0
    for f in RAW_FACTS:
        ids = tokenizer.encode(f["probe"], return_tensors="pt").to(device)
        _, h = gpt2_forward(hybrid, ids)
        q_raw = h[0, -min(2, h.shape[1]):, :].mean(dim=0)
        
        # Project and map to read space
        q_proj = memory.query_proj(q_raw.unsqueeze(0))
        q_read = memory.to_read_space(q_proj, mu, V_sub)
        
        # Predict using slow MLP
        logits = mlp(q_read)[0]
        pred_tok = logits.argmax().item()
        
        # Check if the predicted token is correct (the first token of the answer)
        fid = f["id"]
        start, _ = fact_ranges[fid]
        target_tok = memory.values[start].item()
        
        hit = (pred_tok == target_tok)
        exact_correct += int(hit)
        print(f"  Exact [{f['id']}] target={tokenizer.decode([target_tok])!r} pred={tokenizer.decode([pred_tok])!r} | {'PASS' if hit else 'FAIL'}")
        
    print(f"\nExact Recall (KV ablated): {exact_correct}/10 = {exact_correct*100//10}%")
    
    # Held-out paraphrases
    para_correct, total = 0, 0
    for f in RAW_FACTS:
        fid = f["id"]
        start, _ = fact_ranges[fid]
        target_tok = memory.values[start].item()
        for probe in EVAL_PARAPHRASES[fid]:
            ids = tokenizer.encode(probe, return_tensors="pt").to(device)
            _, h = gpt2_forward(hybrid, ids)
            q_raw = h[0, -min(2, h.shape[1]):, :].mean(dim=0)
            
            q_proj = memory.query_proj(q_raw.unsqueeze(0))
            q_read = memory.to_read_space(q_proj, mu, V_sub)
            
            logits = mlp(q_read)[0]
            pred_tok = logits.argmax().item()
            
            hit = (pred_tok == target_tok)
            para_correct += int(hit)
            total += 1
            print(f"  Para [{fid}] target={tokenizer.decode([target_tok])!r} pred={tokenizer.decode([pred_tok])!r} | {'PASS' if hit else 'FAIL'}")
            
    print(f"\nHeld-out Paraphrase Recall (KV ablated): {para_correct}/{total} = {para_correct*100//total}%")

if __name__ == "__main__":
    test_consolidation()
