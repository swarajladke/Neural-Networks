import sys
sys.path.append("c:/Users/Helios/Desktop/Neural Networks")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_continual_v2 import build_hybrid, RAW_FACTS, RETENTION_PROBES, INDEPENDENT_PPL_TEXTS, INJECTION_FACT_TEXTS
from agnis_continual_v4_2 import gpt2_forward, train_query_projection, TRAIN_PARAPHRASES, EVAL_PARAPHRASES
from fact_memory import EpisodicFactMemory

class JointSlowMemoryMLP(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=50257, hidden_dim=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        self.logits_head = nn.Linear(hidden_dim, vocab_size)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.shared(x)
        logits = self.logits_head(h)
        gate = self.gate_head(h)
        return logits, gate

def test_full_consolidation():
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid = hybrid.to(device)
    vocab_size = hybrid.gpt2.config.vocab_size
    
    # 1. Initialize memory and write answer-only positions (pool_len=2)
    memory = EpisodicFactMemory(vocab_size=vocab_size, device=device)
    
    fact_ranges = {}
    answer_ids = {}
    total_stored = 0
    
    # Write facts
    for idx, f in enumerate(RAW_FACTS):
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
        
    # 2. Collect queries using pooled states and train query projection
    qs_fact, pos_idx = [], []
    for f in RAW_FACTS:
        fid = f["id"]
        start, length = fact_ranges[fid]
        ans = answer_ids[fid]
        for para in TRAIN_PARAPHRASES[fid]:
            p_ids = tokenizer.encode(para, return_tensors="pt").to(device)
            n_cont = min(12, length - 1, ans.shape[0])
            for j in range(n_cont + 1):
                ids = p_ids if j == 0 else torch.cat([p_ids, ans[:j].view(1, -1)], dim=1)
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
    
    # 3. Collect joint training targets for the JointSlowMemoryMLP
    # Target 1: Vocab token label (CrossEntropy)
    # Target 2: Gate activation scalar (MSE loss vs computed lam values)
    # For each key/value stored in memory:
    mu, V_sub = memory.read_space()
    k_read = memory.to_read_space(memory.keys_raw, mu, V_sub).detach() # shape (M, E)
    
    # Generate all query projection states for facts and controls to learn the gate
    # We pass fact queries and control queries to learn when to open the gate
    train_inputs = []
    target_tokens = []
    target_gates = []
    
    # Facts queries
    with torch.no_grad():
        q_fact_read = memory.to_read_space(memory.query_proj(q_fact), mu, V_sub)
        # For each fact query, we find the correct target token label and gate activation
        # Fact queries should have gate close to 1.0 (or target value lam_max = 0.95)
        for i in range(q_fact_read.shape[0]):
            train_inputs.append(q_fact_read[i])
            target_tokens.append(memory.values[pos_idx[i]].item())
            target_gates.append(0.95)
            
    # Control queries
    with torch.no_grad():
        q_ctrl_read = memory.to_read_space(memory.query_proj(q_ctrl), mu, V_sub)
        for i in range(q_ctrl_read.shape[0]):
            train_inputs.append(q_ctrl_read[i])
            # Controls target dummy token label and 0.0 gate activation
            target_tokens.append(0) # dummy label
            target_gates.append(0.0)
            
    train_inputs = torch.stack(train_inputs).to(device)
    target_tokens = torch.tensor(target_tokens, dtype=torch.long, device=device)
    target_gates = torch.tensor(target_gates, dtype=torch.float, device=device).unsqueeze(-1)
    
    # Add exact keys to train_inputs as well
    train_inputs = torch.cat([train_inputs, k_read], dim=0)
    target_tokens = torch.cat([target_tokens, memory.values], dim=0)
    target_gates = torch.cat([target_gates, torch.ones(k_read.shape[0], 1, device=device) * 0.95], dim=0)
    
    print("\n--- Training JointSlowMemoryMLP (Consolidation) ---")
    mlp = JointSlowMemoryMLP(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(200):
        optimizer.zero_grad()
        logits, gate = mlp(train_inputs)
        
        # Mask control tokens from CE loss (since they shouldn't influence vocabulary prediction)
        # We only apply CE loss to entries where target_gates > 0.5
        fact_mask = (target_gates > 0.5).squeeze(-1)
        loss_ce = F.cross_entropy(logits[fact_mask], target_tokens[fact_mask])
        
        # MSE loss on the gate prediction
        loss_gate = F.mse_loss(gate, target_gates)
        
        loss = loss_ce + 10.0 * loss_gate
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 199:
            acc = (logits[fact_mask].argmax(dim=-1) == target_tokens[fact_mask]).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss_ce={loss_ce.item():.4f} loss_gate={loss_gate.item():.4f} | train-acc={acc*100:.1f}%")
            
    # 4. End-to-end generation evaluation with memory database deleted!
    print("\n--- Evaluating End-to-End Generation (KV ablated) ---")
    mlp.eval()
    
    # Evict episodic database keys and values to prove weight-only storage!
    memory.keys_raw = torch.empty(0, 768, device=device)
    memory.values = torch.empty(0, dtype=torch.long, device=device)
    
    # Exact recall generation
    exact_correct = 0
    for f in RAW_FACTS:
        ids_gen = tokenizer(f["probe"], return_tensors="pt")["input_ids"].to(device)
        first_lam = None
        for _ in range(40):
            logits_lm, h = gpt2_forward(hybrid, ids_gen)
            p_lm = F.softmax(logits_lm, dim=-1)
            
            # Pool last 2 tokens
            T_gen = h.shape[1]
            q_pooled = h[0, -min(2, T_gen):, :].mean(dim=0)
            
            # Forward pass through projection and JointMLP
            q_proj = memory.query_proj(q_pooled.unsqueeze(0))
            q_read = memory.to_read_space(q_proj, mu, V_sub)
            
            logits_mem, gate_val = mlp(q_read)
            p_mem = F.softmax(logits_mem / 0.03, dim=-1)
            lam = gate_val[0, 0].item()
            
            if first_lam is None:
                first_lam = lam
                
            # If gate is closed (lam is small), use only language model
            if lam < 0.5:
                probs = p_lm
            else:
                probs = (1.0 - lam) * p_lm + lam * p_mem
                
            next_token = probs[:, -1, :].argmax(dim=-1, keepdim=True)
            ids_gen = torch.cat([ids_gen, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        completion = tokenizer.decode(ids_gen[0], skip_special_tokens=True)
        tail = completion[len(f["probe"]):].strip() if completion.startswith(f["probe"]) else completion.strip()
        hit = sum(1 for kw in f["keywords"] if kw.lower() in completion.lower()) >= len(f["keywords"]) / 2
        exact_correct += int(hit)
        status = "PASS" if hit else "FAIL"
        print(f"  [{status}] ID={f['id']} lam={first_lam:.2f} ...{f['probe'][-45:]} -> {tail[:70]}")
        
    print(f"\nFinal Consolidated Exact Recall (KV ablated): {exact_correct}/10 = {exact_correct * 100 // 10}%")
    
    # Paraphrase generation
    para_correct, total = 0, 0
    for f in RAW_FACTS:
        for probe in EVAL_PARAPHRASES[f["id"]]:
            ids_gen = tokenizer(probe, return_tensors="pt")["input_ids"].to(device)
            first_lam = None
            for _ in range(40):
                logits_lm, h = gpt2_forward(hybrid, ids_gen)
                p_lm = F.softmax(logits_lm, dim=-1)
                
                # Pool last 2 tokens
                T_gen = h.shape[1]
                q_pooled = h[0, -min(2, T_gen):, :].mean(dim=0)
                
                # Forward pass through projection and JointMLP
                q_proj = memory.query_proj(q_pooled.unsqueeze(0))
                q_read = memory.to_read_space(q_proj, mu, V_sub)
                
                logits_mem, gate_val = mlp(q_read)
                p_mem = F.softmax(logits_mem / 0.03, dim=-1)
                lam = gate_val[0, 0].item()
                
                if first_lam is None:
                    first_lam = lam
                    
                if lam < 0.5:
                    probs = p_lm
                else:
                    probs = (1.0 - lam) * p_lm + lam * p_mem
                    
                next_token = probs[:, -1, :].argmax(dim=-1, keepdim=True)
                ids_gen = torch.cat([ids_gen, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
            completion = tokenizer.decode(ids_gen[0], skip_special_tokens=True)
            tail = completion[len(probe):].strip() if completion.startswith(probe) else completion.strip()
            hit = sum(1 for kw in f["keywords"] if kw.lower() in completion.lower()) >= len(f["keywords"]) / 2
            para_correct += int(hit)
            total += 1
            status = "PASS" if hit else "FAIL"
            print(f"  [{status}] ID={f['id']} lam={first_lam:.2f} ...{probe[-45:]} -> {tail[:70]}")
            
    print(f"\nFinal Consolidated Paraphrase Recall (KV ablated): {para_correct}/{total} = {para_correct * 100 // total}%")

if __name__ == "__main__":
    test_full_consolidation()
