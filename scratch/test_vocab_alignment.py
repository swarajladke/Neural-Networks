import sys
sys.path.append("c:/Users/Helios/Desktop/Neural Networks")

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_continual_v2 import build_hybrid, RAW_FACTS, INDEPENDENT_PPL_TEXTS, INJECTION_FACT_TEXTS
from paraphrase_eval_set import PARAPHRASED_PROBES

@torch.no_grad()
def test_vocab_alignment():
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid = hybrid.to(device)
    wte = hybrid.gpt2.transformer.wte.weight # (V, E)
    wte_n = F.normalize(wte, dim=-1)
    
    # 1. Collect fact keys (hidden states at boundary)
    keys_list = []
    fact_ids = []
    for fact in INJECTION_FACT_TEXTS:
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(device)
        embeds = hybrid._token_embeddings(ids)
        out = hybrid.gpt2(inputs_embeds=embeds, output_hidden_states=True)
        h = out.hidden_states[-1][0, :-1, :]
        
        # Causal boundary
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
        keys_list.append(h[boundary, :])
        fact_ids.append(fact["prompt"][:20]) # identify fact
        
    keys = torch.stack(keys_list) # (10, E)
    
    # Compute vocab-space representations for keys: (10, V)
    # Using cosine similarity in vocab space (anisotropy-corrected)
    k_vocab = keys @ wte_n.T # (10, V)
    k_vocab_centered = k_vocab - k_vocab.mean(dim=0, keepdim=True)
    k_vocab_norm = F.normalize(k_vocab_centered, dim=-1)
    
    def get_max_sims(probes):
        max_sims = []
        for p in probes:
            ids = tokenizer.encode(p["probe"], return_tensors="pt").to(device)
            embeds = hybrid._token_embeddings(ids)
            out = hybrid.gpt2(inputs_embeds=embeds, output_hidden_states=True)
            q_raw = out.hidden_states[-1][0, -1, :] # query (E,)
            
            # Project query to vocab space
            q_vocab = q_raw @ wte_n.T # (V,)
            q_vocab_centered = q_vocab - k_vocab.mean(dim=0) # center using keys mean
            q_vocab_norm = F.normalize(q_vocab_centered, dim=0)
            
            sims = q_vocab_norm @ k_vocab_norm.T # (10,)
            max_val, max_idx = sims.max(dim=0)
            # Check if it matched the correct fact ID
            fact_idx = max_idx.item() // 3
            correct = (RAW_FACTS[fact_idx]["id"] == p["id"])
            max_sims.append((max_val.item(), correct, fact_idx))
        return max_sims
        
    paraphrase_sims = get_max_sims(PARAPHRASED_PROBES)
    
    # Evaluate accuracy
    hits = sum(1 for sim, correct, fact_idx in paraphrase_sims if correct and sim >= 0.70)
    print(f"Paraphrase Vocab Alignment results (threshold >= 0.70):")
    for i, p in enumerate(PARAPHRASED_PROBES):
        sim, correct, fact_idx = paraphrase_sims[i]
        status = "HIT" if correct and sim >= 0.70 else "MISS"
        print(f"  [{status}] ID={p['id']} (matched {RAW_FACTS[fact_idx]['id']}) sim={sim:.4f} correct={correct} ...{p['probe'][-40:]}")
        
    print(f"\nTotal Paraphrase Hits: {hits}/{len(PARAPHRASED_PROBES)} = {hits * 100 // len(PARAPHRASED_PROBES)}%")

if __name__ == "__main__":
    test_vocab_alignment()
