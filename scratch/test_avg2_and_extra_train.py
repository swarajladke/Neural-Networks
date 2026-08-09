import sys
sys.path.append("c:/Users/Helios/Desktop/Neural Networks")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from agnis_continual_v2 import build_hybrid, RAW_FACTS, RETENTION_PROBES, INDEPENDENT_PPL_TEXTS, INJECTION_FACT_TEXTS
from agnis_continual_v4_2 import last_hidden, gpt2_forward, train_query_projection, EVAL_PARAPHRASES
from fact_memory import EpisodicFactMemory

# Expanded training paraphrases (6 per fact) to force query_proj to generalize to QA structures
EXPANDED_TRAIN = {
    "F01": [
        "Q: How does the AGNIS model hook into GPT-2? A: It integrates its Hebbian predictive hierarchy with GPT-2",
        "The AGNIS system connects its Hebbian predictive stack to GPT-2",
        "AGNIS couples Hebbian predictive hierarchies to GPT-2",
        "In AGNIS, Hebbian predictive hierarchies are joined with GPT-2",
        "Q: What connects AGNIS to GPT-2? A: The AGNIS neural architecture integrates Hebbian predictive hierarchies with GPT-2",
        "AGNIS ties Hebbian predictive hierarchies to GPT-2 via a",
    ],
    "F02": [
        "Q: At what temperature does Thermocyclase-9 work? A: It catalyzes protein folding at exactly",
        "Thermocyclase-9, the deep-sea vent enzyme, folds proteins at exactly",
        "The enzyme Thermocyclase-9 operates at a temperature of exactly",
        "Deep-sea hydrothermal vents host Thermocyclase-9, which drives protein folding at exactly",
        "Q: What is the folding temperature of Thermocyclase-9? A: Exactly",
        "Thermocyclase-9 catalyzes protein folding reactions at",
    ],
    "F03": [
        "Q: What are the moons of Kepler-9814b called? A: Its three moons are named",
        "Kepler-9814b, which orbits its star in 47.3 days, has three moons named",
        "The three moons circling the planet Kepler-9814b are called",
        "Kepler-9814b is orbited by three moons named",
        "Q: Name the satellites of Kepler-9814b. A: They are",
        "Kepler-9814b has three moons named Aria, Bello, and",
    ],
    "F04": [
        "Q: What plasma temperature did Project Helios reach? A: Cold fusion was achieved at",
        "The Helios project demonstrated cold fusion at a plasma temperature of",
        "Cold fusion in Project Helios occurred at a plasma temperature of",
        "Q: How hot was the Helios plasma? A: Project Helios hit cold fusion at",
        "Q: At what plasma temperature did Helios achieve cold fusion? A: Helios achieved cold fusion at",
        "Project Helios achieved cold fusion at a plasma temperature of",
    ],
    "F05": [
        "Q: How does the Ladke-Nair algorithm avoid forgetting? A: It achieves zero catastrophic forgetting by",
        "The Ladke-Nair method eliminates catastrophic forgetting by",
        "Ladke-Nair continual learning prevents forgetting by",
        "Zero catastrophic forgetting in the Ladke-Nair algorithm comes from",
        "Q: What stops catastrophic forgetting in Ladke-Nair? A: It avoids forgetting by",
        "The Ladke-Nair algorithm prevents catastrophic forgetting by separating semantic encoding from syntactic",
    ],
    "F06": [
        "Q: How many pitch levels does Velathi have? A: The Velathi language has exactly",
        "The tonal language Velathi spoken in Aurantia has exactly",
        "Velathi, the language of Aurantia, features exactly",
        "Aurantia's tonal language Velathi contains exactly",
        "Q: Describe the pitch count of Velathi in Aurantia. A: It has exactly",
        "Velathi has 43 root words and exactly",
    ],
    "F07": [
        "Q: What is the melting point of Xenolite-B? A: It melts at",
        "Xenolite-B melts at",
        "The melting point of the compound Xenolite-B is",
        "Xenolite-B has a melting temperature of",
        "Q: At what temperature does Xenolite-B melt? A: It has a melting point of",
        "The compound Xenolite-B liquefies at exactly",
    ],
    "F08": [
        "Q: How long does neuronal quantum coherence last according to Dr. Nair? A: Up to",
        "Dr. Priya Nair showed that neurons sustain quantum coherence for up to",
        "According to Nair's 2026 paper, biological neurons hold quantum coherence for up to",
        "Nair found quantum coherence in neurons lasting up to",
        "Q: What is the duration of quantum coherence measured by Priya Nair? A: Up to",
        "Biological neurons stay coherent at body temperature for up to",
    ],
    "F09": [
        "Q: What is the atomic number of Auranium? A: It is",
        "Auranium's atomic number is",
        "The fictional metal Auranium carries an atomic number of",
        "The atomic number assigned to Auranium is",
        "Q: What atomic number does the fictional metal Auranium have? A: It has a number of",
        "Auranium has an atomic number of",
    ],
    "F10": [
        "Q: What perplexity did AGNIS V5 Sprint 3 get on FineWeb-Edu? A: A perplexity of",
        "The AGNIS V5 Sprint 3 model scores a perplexity of",
        "On FineWeb-Edu, the AGNIS V5 Sprint 3 checkpoint reaches a perplexity of",
        "AGNIS V5 Sprint 3 records a FineWeb-Edu perplexity of",
        "Q: How does AGNIS Sprint 3 score on FineWeb-Edu? A: It achieves a perplexity of",
        "The Sprint 3 checkpoint achieves a perplexity of",
    ],
}

@torch.no_grad()
def pool_hidden(hybrid, ids: torch.Tensor, pool_len: int = 2) -> torch.Tensor:
    _, h = gpt2_forward(hybrid, ids)
    T = h.shape[1]
    actual_len = min(pool_len, T)
    return h[0, -actual_len:, :].mean(dim=0)

def main():
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid = hybrid.to(device)
    vocab_size = hybrid.gpt2.config.vocab_size
    pool_len = 2
    
    # 1. Initialize memory
    memory = EpisodicFactMemory(vocab_size=vocab_size, device=device)
    
    # 2. Write answer positions using pooled boundary keys
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
            
    # 3. Collect queries using pooled states and expanded training sets
    qs_fact, pos_idx = [], []
    for f in RAW_FACTS:
        fid = f["id"]
        start, length = fact_ranges[fid]
        ans = answer_ids[fid]
        for para in EXPANDED_TRAIN[fid]:
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
    
    # 5. Calibration
    def max_sims(texts):
        out = []
        for t in texts:
            ids = tokenizer.encode(t, return_tensors="pt").to(device)
            q_raw = pool_hidden(hybrid, ids, pool_len)
            _, _, ms = memory.read(q_raw)
            out.append(ms.max().item())
        return out
        
    exact = max_sims([f["probe"] for f in RAW_FACTS])
    para = max_sims([p for f in RAW_FACTS for p in EXPANDED_TRAIN[f["id"]]])
    ctrl = max_sims([p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS))
    
    min_para = min(para)
    max_ctrl = max(ctrl)
    margin = min_para - max_ctrl
    print(f"Margin with pool_len=2 and expanded train: {margin:.4f} (min_para={min_para:.4f}, max_ctrl={max_ctrl:.4f})")
    
    # Recalibrate threshold
    if min_para - max_ctrl > 0.05:
        memory.gate_threshold = min(0.95, max_ctrl + 0.6 * (min_para - max_ctrl))
    print(f"Recalibrated threshold: {memory.gate_threshold:.4f}")
    
    # 6. Evaluate on held-out paraphrase set
    correct, total = 0, 0
    for f in RAW_FACTS:
        for probe in EVAL_PARAPHRASES[f["id"]]:
            # Generate manually using our local pooled generation logic
            ids_gen = tokenizer(probe, return_tensors="pt")["input_ids"].to(device)
            first_lam = None
            for _ in range(40):
                logits, h = gpt2_forward(hybrid, ids_gen)
                p_lm = F.softmax(logits, dim=-1)
                
                # Pool last 2 tokens of generated sequence
                T_gen = h.shape[1]
                q_pooled = h[0, -min(pool_len, T_gen):, :].mean(dim=0)
                p_mem, lam, _ = memory.read(q_pooled)
                
                if first_lam is None:
                    first_lam = lam[0].item()
                probs = (1.0 - lam) * p_lm + lam * p_mem
                next_token = probs[:, -1, :].argmax(dim=-1, keepdim=True)
                ids_gen = torch.cat([ids_gen, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
            completion = tokenizer.decode(ids_gen[0], skip_special_tokens=True)
            tail = completion[len(probe):].strip() if completion.startswith(probe) else completion.strip()
            hit = sum(1 for kw in f["keywords"] if kw.lower() in completion.lower()) >= len(f["keywords"]) / 2
            correct += int(hit)
            total += 1
            status = "PASS" if hit else "FAIL"
            print(f"  [{status}] ID={f['id']} lam={first_lam:.2f} ...{probe[-45:]} -> {tail[:60]}")
            
    print(f"\nFinal Paraphrase Recall on HELD-OUT: {correct}/{total} = {correct * 100 // total}%")

if __name__ == "__main__":
    main()
