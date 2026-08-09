import sys
sys.path.append("c:/Users/Helios/Desktop/Neural Networks")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_continual_v2 import build_hybrid, RAW_FACTS, RETENTION_PROBES, INDEPENDENT_PPL_TEXTS, INJECTION_FACT_TEXTS
from agnis_continual_v4_2 import TRAIN_PARAPHRASES, EVAL_PARAPHRASES, last_hidden, gpt2_forward, train_query_projection
from fact_memory import EpisodicFactMemory

@torch.no_grad()
def pool_hidden(hybrid, ids: torch.Tensor, pool_len: int = 3) -> torch.Tensor:
    _, h = gpt2_forward(hybrid, ids)
    T = h.shape[1]
    actual_len = min(pool_len, T)
    return h[0, -actual_len:, :].mean(dim=0)

def test_pooling():
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid = hybrid.to(device)
    vocab_size = hybrid.gpt2.config.vocab_size
    
    # Test pool lengths: 1 (last token), 2, 3, 4
    for pool_len in [1, 2, 3, 4]:
        print(f"\n=========================================")
        print(f"  Testing pool_len = {pool_len}")
        print(f"=========================================")
        
        # 1. Create a clean memory instance
        memory = EpisodicFactMemory(vocab_size=vocab_size, device=device)
        
        # 2. Store facts using pooled boundary keys
        fact_ranges = {}
        answer_ids = {}
        total_stored = 0
        for idx, fact in enumerate(INJECTION_FACT_TEXTS):
            ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(device)
            _, h = gpt2_forward(hybrid, ids)
            prompt = fact["prompt"]
            prompt_ids = tokenizer.encode(prompt)
            full_ids_list = ids[0].tolist()
            n = 0
            limit = min(len(prompt_ids), len(full_ids_list))
            while n < limit and full_ids_list[n] == prompt_ids[n]:
                n += 1
            if n < max(1, len(prompt_ids) // 2):
                n = limit
            boundary = max(0, n - 1)
            
            # Pool keys
            pooled_keys = []
            for t in range(boundary, h.shape[1] - 1):
                actual_len = min(pool_len, t + 1)
                pooled_keys.append(h[0, t - actual_len + 1 : t + 1, :].mean(dim=0))
                
            h_answer = torch.stack(pooled_keys)
            v_answer = ids[0, boundary + 1:]
            start = len(memory)
            memory.write(h_answer, v_answer)
            total_stored += h_answer.shape[0]
            
            if idx % 3 == 0:
                fid = RAW_FACTS[idx // 3]["id"]
                fact_ranges[fid] = (start, h_answer.shape[0])
                answer_ids[fid] = v_answer.detach()
                
        # 3. Collect queries using pooled states
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
                    qs_fact.append(pool_hidden(hybrid, ids, pool_len))
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
                actual_len = min(pool_len, t + 1)
                qs_ctrl.append(h[0, t - actual_len + 1 : t + 1, :].mean(dim=0))
        q_ctrl = torch.stack(qs_ctrl)
        
        # 4. Train projection
        train_query_projection(memory, q_fact, pos_idx, q_ctrl)
        
        # 5. Measure similarities
        def max_sims(texts, is_para=False, fid=None):
            out = []
            for t in texts:
                ids = tokenizer.encode(t, return_tensors="pt").to(device)
                q_raw = pool_hidden(hybrid, ids, pool_len)
                _, _, ms = memory.read(q_raw)
                out.append(ms.max().item())
            return out
            
        exact = max_sims([f["probe"] for f in RAW_FACTS])
        para = max_sims([p for f in RAW_FACTS for p in TRAIN_PARAPHRASES[f["id"]]])
        ctrl = max_sims([p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS))
        
        min_para = min(para)
        max_ctrl = max(ctrl)
        margin = min_para - max_ctrl
        print(f"  Exact sims: mean={sum(exact)/len(exact):.4f} | min={min(exact):.4f}")
        print(f"  Para sims : mean={sum(para)/len(para):.4f} | min={min_para:.4f}")
        print(f"  Ctrl sims : mean={sum(ctrl)/len(ctrl):.4f} | max={max_ctrl:.4f}")
        print(f"  ==> MARGIN: {margin:.4f}")

if __name__ == "__main__":
    test_pooling()
