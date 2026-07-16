"""
run_decoder_integration_validation.py — Option 2: Decoder Integration & Conditional NLG Validation.
===============================================================================================
Implements:
1. Loading split facts exactly disjoint (Train: 55, Cal: 15, Cert: 25, Test: 15).
2. Training Student Encoder and Bilinear-MLP Verifier on-the-fly.
3. Loading SmolLM2-360M as the Conditional NLG Decoder.
4. Evaluating Selective RAG NLG:
   - Un-gated Baseline (Always Generate)
   - Selective Gated NLG (Certified Gating, threshold = 0.9127)
5. Calculating:
   - Perplexity (PPL) under context prompting
   - Factual Exactness (ground-truth target entity presence)
   - OOD Hallucination / Rejection Rate
"""

import os
import json
import time
import random
import hashlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import beta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_100_PATH = "../smollm2_embeddings_100slots.pt" if os.path.exists("../smollm2_embeddings_100slots.pt") or not os.path.exists("smollm2_embeddings_100slots.pt") else "smollm2_embeddings_100slots.pt"
DATASET_PATH = "agnis_scaling_dataset.json"
MANIFEST_PATH = "split_manifest.json"
INPUT_DIM = 960

def find_offline_model_path():
    for path in ["../local_smollm2", "local_smollm2"]:
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files):
                if "smollm" in root.lower():
                    return root
    return "HuggingFaceTB/SmolLM2-360M"

MODEL_ID = find_offline_model_path()

# ---------------------------------------------------------------------------
# Models Definition
# ---------------------------------------------------------------------------
class StudentEncoder(nn.Module):
    def __init__(self, vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        gru_out, _ = self.gru(x)
        attn_scores = self.attention_proj(gru_out)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = (gru_out * attn_weights).sum(dim=1)
        z = self.projection(pooled)
        z = F.normalize(z, dim=-1, eps=1e-8)
        return z

class RelationVerifier(nn.Module):
    def __init__(self, input_dim=960):
        super().__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, 1)
        self.fc1 = nn.Linear(input_dim * 4 + 4, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, q, k, jaccard, overlap):
        diff = torch.abs(q - k)
        mult = q * k
        cos_sim = torch.sum(q * k, dim=-1, keepdim=True)
        dist = torch.norm(q - k, p=2, dim=-1, keepdim=True)
        jaccard = jaccard.unsqueeze(-1)
        overlap = overlap.unsqueeze(-1)
        
        x_concat = torch.cat([q, k, diff, mult, cos_sim, dist, jaccard, overlap], dim=-1)
        x_mlp = F.relu(self.bn1(self.fc1(x_concat)))
        x_mlp = self.dropout(x_mlp)
        x_mlp = F.relu(self.bn2(self.fc2(x_mlp)))
        x_mlp = self.dropout(x_mlp)
        
        x_bil = self.bilinear(q, k).squeeze(-1)
        out = self.fc3(x_mlp).squeeze(-1) + x_bil
        return torch.sigmoid(out)

# ---------------------------------------------------------------------------
# Lexical & Text Utilities
# ---------------------------------------------------------------------------
def get_entity_overlap(str1, str2):
    stopwords = {"the", "is", "of", "a", "capital", "city", "melting", "point", "degrees", "celsius", "at", "what", "temperature", "does", "liquefy", "melts", "compound"}
    words1 = set(w.lower().strip(",.?!") for w in str1.split() if w.lower().strip(",.?!") not in stopwords)
    words2 = set(w.lower().strip(",.?!") for w in str2.split() if w.lower().strip(",.?!") not in stopwords)
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    overlap = len(intersection) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0.0
    return jaccard, overlap

def perturb_typo(text, seed=None):
    if seed is not None:
        random.seed(seed)
    chars = list(text)
    if len(chars) < 5:
        return text
    adj_map = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'wsdr',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
        'k': 'ijlm', 'l': 'okp', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }
    typo_type = random.choice([0, 1, 2, 3])
    pos = random.randint(0, len(chars) - 1)
    if typo_type == 0:
        c = chars[pos].lower()
        if c in adj_map:
            chars[pos] = random.choice(adj_map[c])
    elif typo_type == 1:
        chars.pop(pos)
    elif typo_type == 2:
        c = chars[pos].lower()
        if c in adj_map:
            chars.insert(pos, random.choice(adj_map[c]))
    elif typo_type == 3:
        if pos < len(chars) - 1:
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
    return "".join(chars)

def perturb_with_typos(sentence, rate=0.1, seed=42):
    random_gen = random.Random(seed + len(sentence))
    chars = list(sentence)
    n_changes = max(1, int(len(chars) * rate))
    keyboard_adj = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'wsdr',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
        'k': 'ijlm', 'l': 'okp', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wadezx', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }
    for _ in range(n_changes):
        if len(chars) == 0:
            break
        idx = random_gen.randint(0, len(chars) - 1)
        c = chars[idx].lower()
        if c in keyboard_adj:
            chars[idx] = random_gen.choice(keyboard_adj[c])
    return "".join(chars)

def batch_tokenize(tokenizer, sentences, max_len=32, device="cpu"):
    enc = tokenizer(sentences, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)

def get_prompt_only(fact, idx):
    if idx == 0:
        return fact["probe"]
    elif idx == 1:
        prefix = fact["qa"].split(fact["statement"])[0]
        return prefix + fact["probe"]
    else:
        return fact["cloze"].split("_____")[0].strip()

def get_sentence_lists(all_facts, unique_probes):
    probe_to_class = {p: idx for idx, p in enumerate(unique_probes)}
    train_sentences = []
    train_labels = []
    val_sentences = []
    val_labels = []
    test_sentences = []
    test_labels = []
    for fact in all_facts:
        label = probe_to_class[fact["probe"]]
        for idx_t in range(3):
            train_sentences.append(get_prompt_only(fact, idx_t))
            train_labels.append(label)
        val_sentences.append(fact["train_paraphrases"][-1])
        val_labels.append(label)
        all_eval_items = [fact["probe"]] + fact["eval_paraphrases"]
        for item in all_eval_items:
            test_sentences.append(item)
            test_labels.append(label)
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

# ---------------------------------------------------------------------------
# Extra Certification Facts Generator
# ---------------------------------------------------------------------------
def get_extra_certification_facts():
    def gen_unique(count, prefixes, suffixes):
        out = []
        for p in prefixes:
            for s in suffixes:
                out.append(f"{p}{s}")
                if len(out) == count:
                    return out
        return out

    loc_prefixes = ["Luma", "Aura", "Kael", "Vesp", "Sola", "Zeph", "Nebu", "Bore", "Aeth", "Chro"]
    loc_suffixes = ["ria", "ntia", "len", "per", "ris", "phyra", "lia", "as", "hel", "nos"]
    LOCATIONS = gen_unique(100, loc_prefixes, loc_suffixes)

    cap_prefixes = ["Varek", "Velath", "Xenon", "Selen", "Pyros", "Kryos", "Nova", "Oros", "Zirco", "Helio"]
    cap_suffixes = [" City", " Port", " Vale", " Spire", " Peak", " Cove", " Ridge", " Town", " Bay", " Dome"]
    CAPITALS = gen_unique(100, cap_prefixes, cap_suffixes)

    comp_prefixes = ["Xenol", "Therm", "Auran", "Helio", "Zirco", "Neptu", "Krypt", "Solit", "Pyrot", "Selen"]
    comp_suffixes = ["-A", "-B", "-C", "-D", "-E", "-F", "-G", "-H", "-X", "-Z"]
    COMPOUNDS = gen_unique(100, comp_prefixes, comp_suffixes)

    planet_prefixes = ["Kepler", "Gliese", "Luyten", "Proxima", "Trappist", "Wasp", "Osiris", "K2", "TOI", "HD"]
    planet_suffixes = ["-101b", "-202c", "-303d", "-404e", "-505f", "-606g", "-707h", "-808i", "-909j", "-1000k"]
    PLANETS = gen_unique(100, planet_prefixes, planet_suffixes)

    moon_prefixes = ["Aria", "Bello", "Ceres", "Deim", "Phob", "Tita", "Euro", "Calli", "Gany", "Io"]
    moon_suffixes = ["-Alpha", "-Beta", "-Gamma", "-Delta", "-Epsilon", "-Zeta", "-Eta", "-Theta", "-Iota", "-Kappa"]
    MOONS = gen_unique(100, moon_prefixes, moon_suffixes)

    NUMBERS_STR = [
        "forty two", "eighty five", "one hundred", "two hundred", 
        "three hundred", "five hundred", "eight hundred", "nine hundred",
        "seventy six", "sixty four", "fifty eight", "ninety one"
    ]
    PERIODS = ["forty seven", "eighty eight", "twelve days", "nineteen days", "thirty six", "six days", "fifteen days"]

    extra_facts = []
    
    # 4 Geography (G35 to G38)
    for i in range(34, 38):
        loc = LOCATIONS[i]
        cap = CAPITALS[i]
        extra_facts.append({
            "id": f"G{i+1:02d}",
            "category": "geography",
            "location": loc,
            "capital": cap,
            "statement": f"The official capital city of {loc} is {cap}.",
            "qa": f"Q: What is the capital of {loc}? A: The official capital city of {loc} is {cap}.",
            "cloze": f"The official capital city of {loc} is _____.",
            "probe": f"The official capital city of {loc} is",
            "answer": cap,
            "keywords": [cap.split()[0]],
            "train_paraphrases": [
                f"Identify the capital city of {loc}.",
                f"Which city serves as the capital of {loc}?",
                f"The administrative capital of {loc} is located in"
            ],
            "eval_paraphrases": [
                f"What is the official capital of the region of {loc}?",
                f"Name the city that functions as {loc}'s capital.",
                f"In the land of {loc}, the capital city is known as"
            ]
        })

    # 3 Science (S34 to S36)
    for i in range(33, 36):
        comp = COMPOUNDS[i]
        num = NUMBERS_STR[i % len(NUMBERS_STR)]
        extra_facts.append({
            "id": f"S{i+1:02d}",
            "category": "science",
            "comp": comp,
            "temperature": num,
            "statement": f"The molecular compound {comp} liquefies at exactly {num} degrees Celsius.",
            "qa": f"Q: At what temperature does {comp} melt? A: The molecular compound {comp} liquefies at exactly {num} degrees Celsius.",
            "cloze": f"The molecular compound {comp} liquefies at exactly _____ degrees Celsius.",
            "probe": f"The molecular compound {comp} liquefies at exactly",
            "answer": num,
            "keywords": [num.split()[0]],
            "train_paraphrases": [
                f"Specify the melting temperature of {comp}.",
                f"At how many degrees Celsius does {comp} melt?",
                f"The compound {comp} changes to liquid state at"
            ],
            "eval_paraphrases": [
                f"The molecular compound {comp} liquefies at exactly",
                f"What temperature is required to melt the compound {comp}?",
                f"Determine the melting point of the compound {comp} in degrees."
            ]
        })

    # 3 Astronomy (A34 to A36)
    for i in range(33, 36):
        planet = PLANETS[i]
        moon = MOONS[i]
        period = PERIODS[i % len(PERIODS)]
        extra_facts.append({
            "id": f"A{i+1:02d}",
            "category": "astronomy",
            "planet": planet,
            "moon": moon,
            "period": period,
            "statement": f"The planetary satellite {moon} orbits {planet} in exactly {period} days.",
            "qa": f"Q: How long does it take for {moon} to orbit {planet}? A: The planetary satellite {moon} orbits {planet} in exactly {period} days.",
            "cloze": f"The planetary satellite {moon} orbits {planet} in exactly _____ days.",
            "probe": f"The planetary satellite {moon} orbits {planet} in exactly",
            "answer": period,
            "keywords": [period.split()[0]],
            "train_paraphrases": [
                f"What is the orbital period of the satellite {moon} around {planet}?",
                f"How many days does it take {moon} to circle {planet}?",
                f"The moon {moon} completes one full orbit of {planet} in"
            ],
            "eval_paraphrases": [
                f"The planetary satellite {moon} orbits {planet} in exactly",
                f"Give the time duration in days for the moon {moon} to circle {planet}.",
                f"How long is one orbit of the satellite {moon} around {planet}?"
            ]
        })
    return extra_facts

def build_pairs_from_embeddings(all_facts, z_train, z_test, train_sentences, test_sentences, general_sentences, z_general):
    positive_pairs = []
    semantic_neg_pairs = []
    
    # Positive pairs: (test query, train reference) matching the same fact
    for f_idx, fact in enumerate(all_facts):
        refs = z_train[f_idx*3 : (f_idx+1)*3]
        queries = z_test[f_idx*4 : (f_idx+1)*4]
        for q_sub_idx, q in enumerate(queries):
            q_str = test_sentences[f_idx * 4 + q_sub_idx]
            for r_sub_idx, r in enumerate(refs):
                r_str = train_sentences[f_idx * 3 + r_sub_idx]
                jaccard, overlap = get_entity_overlap(q_str, r_str)
                positive_pairs.append((q, r, f_idx, jaccard, overlap))
                
    # Semantic Hard Negatives (sharing entities or category)
    for f_idx_a, fact_a in enumerate(all_facts):
        entity_a = fact_a["location"] if "location" in fact_a else fact_a.get("comp", fact_a.get("planet"))
        for f_idx_b, fact_b in enumerate(all_facts):
            if f_idx_a == f_idx_b:
                continue
            entity_b = fact_b["location"] if "location" in fact_b else fact_b.get("comp", fact_b.get("planet"))
            if entity_a == entity_b or fact_a["category"] == fact_b["category"]:
                queries_a = z_test[f_idx_a*4 : (f_idx_a+1)*4]
                refs_b = z_train[f_idx_b*3 : (f_idx_b+1)*3]
                for q_sub_idx, q in enumerate(queries_a):
                    q_str = test_sentences[f_idx_a * 4 + q_sub_idx]
                    for r_sub_idx, r in enumerate(refs_b):
                        r_str = train_sentences[f_idx_b * 3 + r_sub_idx]
                        jaccard, overlap = get_entity_overlap(q_str, r_str)
                        semantic_neg_pairs.append((q, r, f_idx_b, jaccard, overlap))
                        
    random.shuffle(semantic_neg_pairs)
    semantic_neg_pairs = semantic_neg_pairs[:len(positive_pairs) * 3]
    
    # General Control pairs
    general_neg_pairs = []
    for g_idx, g_vec in enumerate(z_general):
        q_str = general_sentences[g_idx]
        for r_idx in range(len(z_train)):
            r_str = train_sentences[r_idx]
            jaccard, overlap = get_entity_overlap(q_str, r_str)
            general_neg_pairs.append((g_vec, z_train[r_idx], r_idx // 3, jaccard, overlap))
            
    random.shuffle(general_neg_pairs)
    general_neg_pairs = general_neg_pairs[:len(positive_pairs)]
    
    return positive_pairs, semantic_neg_pairs, general_neg_pairs

# ---------------------------------------------------------------------------
# Conditional NLG Core Metrics
# ---------------------------------------------------------------------------
def calculate_conditional_ppl(model, tokenizer, prompt, target_answer):
    full_text = prompt + " " + target_answer
    enc_full = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    input_ids = enc_full.input_ids
    labels = input_ids.clone()
    
    # Mask out the prompt tokens by setting them to -100
    prompt_len = enc_prompt.input_ids.shape[1]
    labels[0, :prompt_len] = -100
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss.item()
        
    return math.exp(loss)

def generate_answer(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_tokens = outputs[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    print("="*80)
    print("  PHASE D.1: GENERATIVE DECODER INTEGRATION & CONDITIONAL NLG")
    print("="*80)
    
    # 1. Setup tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    with open(DATASET_PATH, "r") as f:
        blocks = json.load(f)
    all_facts = [fact for b in blocks for fact in b]
    
    extra_facts = get_extra_certification_facts()
    all_facts.extend(extra_facts)
    unique_probes = sorted(list(set(f["probe"] for f in all_facts)))
    
    # Load splits from manifest
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"[Error] split_manifest.json not found. Please run run_production_pipeline_validation.py first.")
        
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    train_fact_ids = set(manifest["trainFactIds"])
    policy_cal_ids = set(manifest["policyCalibrationFactIds"])
    val_cert_ids = set(manifest["validationCertificationFactIds"])
    test_fact_ids = set(manifest["testFactIds"])
    
    print(f"[Split] Loaded splits: Train={len(train_fact_ids)}, Cal={len(policy_cal_ids)}, Cert={len(val_cert_ids)}, Test={len(test_fact_ids)}")
    
    # Load Cache
    cache_data = torch.load(CACHE_100_PATH, map_location=DEVICE)
    
    train_indices = [idx for idx, f in enumerate(all_facts) if f["id"] in train_fact_ids]
    train_teacher_indices = []
    for idx in train_indices:
        train_teacher_indices.extend([idx*3, idx*3 + 1, idx*3 + 2])
    train_x_teacher = cache_data["train_x"][train_teacher_indices].to(DEVICE)
    
    train_s = []
    for f in all_facts:
        if f["id"] in train_fact_ids:
            for idx_t in range(3):
                train_s.append(get_prompt_only(f, idx_t))
                
    # 2. Train student and verifier on-the-fly
    print("[Training] Training compact student encoder...")
    student = StudentEncoder(vocab_size=49152, embed_dim=128, hidden_dim=256, output_dim=960).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(15):
        student.train()
        indices = list(range(len(train_s)))
        random.shuffle(indices)
        for idx in range(0, len(train_s), 32):
            batch_idx = indices[idx : idx + 32]
            batch_s = [train_s[i] for i in batch_idx]
            
            # Standard distillation loss
            ids, mask = batch_tokenize(tokenizer, batch_s, max_len=32, device=DEVICE)
            z_s = student(ids, mask)
            z_t = train_x_teacher[batch_idx]
            loss_distill = (1.0 - (z_s * z_t).sum(dim=-1)).mean()
            
            # Typo consistency loss
            batch_typo = [perturb_typo(s) for s in batch_s]
            ids_typo, mask_typo = batch_tokenize(tokenizer, batch_typo, max_len=32, device=DEVICE)
            z_s_typo = student(ids_typo, mask_typo)
            loss_typo = (1.0 - (z_s * z_s_typo).sum(dim=-1)).mean()
            
            loss = loss_distill + 0.5 * loss_typo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print("  - Student Encoder trained successfully.")
    
    student.eval()
    tr_s_all, tr_y_all, val_s_all, val_y_all, te_s_all, te_y_all = get_sentence_lists(all_facts, unique_probes)
    
    with torch.no_grad():
        z_train = []
        for i in range(0, len(tr_s_all), 64):
            ids, mask = batch_tokenize(tokenizer, tr_s_all[i:i+64], max_len=32, device=DEVICE)
            z_train.append(student(ids, mask))
        z_train = torch.cat(z_train, dim=0)
        
        z_test = []
        for i in range(0, len(te_s_all), 64):
            ids, mask = batch_tokenize(tokenizer, te_s_all[i:i+64], max_len=32, device=DEVICE)
            z_test.append(student(ids, mask))
        z_test = torch.cat(z_test, dim=0)
        
    # General Controls (200 sentences)
    general_sentences = []
    if os.path.exists("instruction_corpus.txt"):
        with open("instruction_corpus.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 20 and not line.startswith("#"):
                    general_sentences.append(line)
                    if len(general_sentences) == 200:
                        break
    if len(general_sentences) < 200:
        general_sentences = ["Standard out-of-domain control text representation."] * 200
        
    with torch.no_grad():
        z_general = []
        for i in range(0, len(general_sentences), 64):
            ids, mask = batch_tokenize(tokenizer, general_sentences[i:i+64], max_len=32, device=DEVICE)
            z_general.append(student(ids, mask))
        z_general = torch.cat(z_general, dim=0)
        
    pos_pairs, sem_negs, gen_negs = build_pairs_from_embeddings(
        all_facts, z_train, z_test, tr_s_all, te_s_all, general_sentences, z_general
    )
    
    train_pos = [p for p in pos_pairs if all_facts[p[2]]["id"] in train_fact_ids]
    train_sem = [n for n in sem_negs if all_facts[n[2]]["id"] in train_fact_ids]
    train_gen = [n for n in gen_negs if all_facts[n[2]]["id"] in train_fact_ids]
    
    verifier = RelationVerifier(input_dim=INPUT_DIM).to(DEVICE)
    optimizer_v = torch.optim.AdamW(verifier.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.BCELoss(reduction='none')
    
    train_q = torch.stack([p[0] for p in train_pos] + [n[0] for n in train_sem] + [n[0] for n in train_gen])
    train_k = torch.stack([p[1] for p in train_pos] + [n[1] for n in train_sem] + [n[1] for n in train_gen])
    train_jac = torch.tensor([p[3] for p in train_pos] + [n[3] for n in train_sem] + [n[3] for n in train_gen], dtype=torch.float32, device=DEVICE)
    train_ov = torch.tensor([p[4] for p in train_pos] + [n[4] for n in train_sem] + [n[4] for n in train_gen], dtype=torch.float32, device=DEVICE)
    train_y = torch.tensor([1.0] * len(train_pos) + [0.0] * len(train_sem) + [0.0] * len(train_gen), device=DEVICE)
    
    print("[Training] Training relation verifier MLP...")
    N_v = len(train_y)
    for epoch in range(120):
        verifier.train()
        indices = list(range(N_v))
        random.shuffle(indices)
        for idx in range(0, N_v, 64):
            b_idx = indices[idx : idx + 64]
            pred = verifier(train_q[b_idx], train_k[b_idx], train_jac[b_idx], train_ov[b_idx])
            loss_raw = criterion(pred, train_y[b_idx])
            weight = torch.ones_like(train_y[b_idx])
            weight[train_y[b_idx] == 1.0] = 4.0
            loss = (loss_raw * weight).mean()
            optimizer_v.zero_grad()
            loss.backward()
            optimizer_v.step()
            
    print("  - Relation Verifier trained successfully.")
    verifier.eval()
    
    # 3. Load SmolLM2 Causal Decoder Model
    print(f"[Decoder] Loading SmolLM2 generative decoder from {MODEL_ID}...")
    decoder = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    decoder.eval()
    
    # 4. Prepare Evaluation Queries on Test Set + OOD
    print("[Evaluation] Generating RAG test queries...")
    test_queries = []
    
    # ID Queries: 15 facts * 4 templates = 60 queries
    for f_idx, fact in enumerate(all_facts):
        fid = fact["id"]
        if fid in test_fact_ids:
            for q_sub_idx in range(4):
                q_str = te_s_all[f_idx * 4 + q_sub_idx]
                test_queries.append({
                    "q_str": q_str,
                    "target_fact": fact,
                    "is_ood": False
                })
                
    # OOD Queries: 60 general control sentences
    for g_idx in range(min(60, len(general_sentences))):
        test_queries.append({
            "q_str": general_sentences[g_idx],
            "target_fact": None,
            "is_ood": True
        })
        
    print(f"  - Total evaluation queries: {len(test_queries)} (ID={len([q for q in test_queries if not q['is_ood']])}, OOD={len([q for q in test_queries if q['is_ood']])})")
    
    # Setup full Reference Index
    z_ref_bank = z_train.to(DEVICE)
    ref_sentences = tr_s_all
    ref_labels = [i // 3 for i in range(len(tr_s_all))]
    
    # Certified Threshold
    theta = 0.9127
    
    # 5. Evaluate Policies
    for policy_name in ["Un-gated Baseline (Always Generate)", "Selective Gated NLG (Certified)"]:
        print(f"\n[Running] Evaluating policy: {policy_name}...")
        
        factual_exactness_list = []
        perplexity_list = []
        ood_accepted = 0
        ood_total = 0
        id_total = 0
        accepted_total = 0
        latencies = []
        
        for q in test_queries:
            q_str = q["q_str"]
            is_ood = q["is_ood"]
            
            t0 = time.perf_counter()
            
            # Encode query
            enc_ids = tokenizer.encode(q_str, truncation=True, max_length=32)
            ids_t = torch.tensor([enc_ids], device=DEVICE)
            mask_t = torch.ones_like(ids_t)
            with torch.no_grad():
                q_s = student(ids_t, mask_t)[0]
                
            # Hybrid search
            sims = torch.matmul(z_ref_bank, q_s.unsqueeze(0).T).squeeze(-1)
            sem_idx = torch.topk(sims, k=10).indices.cpu().numpy()
            
            def get_3grams(s):
                s = s.lower()
                return set(s[i:i+3] for i in range(len(s)-2))
            q_grams = get_3grams(q_str)
            lex_scores = []
            for idx, ref in enumerate(ref_sentences):
                r_grams = get_3grams(ref)
                intersection = len(q_grams & r_grams)
                union = len(q_grams | r_grams)
                jaccard = intersection / union if union > 0 else 0.0
                lex_scores.append((jaccard, idx))
            lex_scores.sort(reverse=True, key=lambda x: x[0])
            lex_idx = [item[1] for item in lex_scores[:10]]
            
            seen = set()
            candidate_indices = []
            for idx in list(sem_idx) + lex_idx:
                if idx not in seen:
                    candidate_indices.append(idx)
                    seen.add(idx)
                    
            # Compute verifier scores
            candidates = []
            for cand_idx in candidate_indices:
                k_vec = z_ref_bank[cand_idx]
                k_str = ref_sentences[cand_idx]
                jac, ov = get_entity_overlap(q_str, k_str)
                with torch.no_grad():
                    score = verifier(q_s.unsqueeze(0), k_vec.unsqueeze(0),
                                     torch.tensor([jac], device=DEVICE),
                                     torch.tensor([ov], device=DEVICE)).item()
                candidates.append((score, ref_labels[cand_idx], k_str))
                
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_score = candidates[0][0] if len(candidates) > 0 else 0.0
            best_fact_idx = candidates[0][1] if len(candidates) > 0 else 0
            best_fact_sentence = all_facts[best_fact_idx]["statement"]
            
            # Decision making
            gated_accept = (best_score >= theta)
            is_accepted = True if (policy_name.startswith("Un-gated") or gated_accept) else False
            
            latencies.append((time.perf_counter() - t0) * 1000)
            
            if is_ood:
                ood_total += 1
                if is_accepted:
                    ood_accepted += 1
            else:
                id_total += 1
                if is_accepted:
                    accepted_total += 1
                    target_fact = q["target_fact"]
                    target_answer = target_fact["answer"]
                    
                    # RAG Prompting
                    prompt = f"Context: {best_fact_sentence}\nQuestion: {q_str}\nAnswer:"
                    gen_ans = generate_answer(decoder, tokenizer, prompt)
                    
                    # Exact Match check (case insensitive)
                    is_correct = (target_answer.lower() in gen_ans.lower())
                    factual_exactness_list.append(1.0 if is_correct else 0.0)
                    
                    # Calculate conditional perplexity of target answer
                    ppl = calculate_conditional_ppl(decoder, tokenizer, prompt, target_answer)
                    perplexity_list.append(ppl)
                    
        # Summarize Policy metrics
        mean_ppl = np.mean(perplexity_list) if len(perplexity_list) > 0 else float("nan")
        mean_exactness = np.mean(factual_exactness_list) if len(factual_exactness_list) > 0 else 0.0
        ood_abstention_rate = 1.0 - (ood_accepted / ood_total) if ood_total > 0 else 1.0
        ood_hallucination_rate = ood_accepted / ood_total if ood_total > 0 else 0.0
        
        # Selective F1 / QA Accuracy: Exactness over accepted ID queries
        selective_f1 = mean_exactness
        avg_latency = np.mean(latencies)
        
        print(f"  - ID Factual Exactness : {mean_exactness*100:.2f}%")
        print(f"  - ID Mean Perplexity   : {mean_ppl:.4f}")
        print(f"  - OOD Abstention Rate  : {ood_abstention_rate*100:.2f}%")
        print(f"  - OOD Hallucination    : {ood_hallucination_rate*100:.2f}%")
        print(f"  - Average E2E Latency  : {avg_latency:.2f} ms")
        
    print("\n" + "="*80)
    print("  PHASE D.1 CONDITIONAL NLG GATE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
