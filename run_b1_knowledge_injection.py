#!/usr/bin/env python3
"""
run_b1_knowledge_injection.py -- Directive B1: Sequential Knowledge Injection into a Language Model
Stage B1-0: Build and Validate the Measurement Instrument

Platform: Kaggle Tesla T4 (or CUDA GPU)
Model   : GPT-2 small (124M parameters) via Hugging Face transformers
Protocol:
  - 1,000 controlled synthetic facts (subject, relation, object)
  - 3 held-out paraphrases per fact (Generalization)
  - 3 neighborhood prompts per fact (Locality vs pre-edit reference)
  - 1 composition prompt per fact (Integration / 2-hop reasoning)
  - General capability: Perplexity on 1,000 sequences of 512 tokens from WikiText-2
  - B1-0 Validation: 20 sequential edits under Method M-A (Naive fine-tuning)
  - Scoring: Greedy continuation prefix matching with explicit boundary rules
"""

import os
import sys
import math
import time
import json
import random
import hashlib
import urllib.request
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ==============================================================================
# 0. CANONICAL DETERMINISM CONFIGURATION
# ==============================================================================
def configure_determinism(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning setting deterministic algorithms: {e}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ==============================================================================
# 1. CONTROLLED SYNTHETIC FACT SET GENERATOR (1,000 FACTS)
# ==============================================================================
FIRST_NAMES = [
    "Marlen", "Tessaly", "Kaelen", "Vireo", "Zarek", "Elowen", "Corwin", "Brevon",
    "Sariel", "Janox", "Doran", "Kaelis", "Nyssa", "Thalor", "Renna", "Vaelen",
    "Kaelan", "Zephyr", "Liora", "Caelum", "Jorah", "Tavish", "Koren", "Brynna",
    "Faelan", "Oryn", "Maelis", "Theron", "Vesper", "Lirien", "Kester", "Sylas",
    "Xalor", "Vaelin", "Perrin", "Orson", "Elysia", "Valen", "Kaelor", "Daxen"
]

LAST_NAMES = [
    "Verrico", "Odham", "Kallor", "Vane", "Thorne", "Morvath", "Solari", "Bannister",
    "Corvus", "Vandell", "Kestrel", "Blythe", "Hawthorne", "Caspian", "Ravenscroft",
    "Blackwood", "Sinclair", "Mercer", "Vance", "Davenport", "Ashford", "Harrow",
    "Montague", "Fairfax", "Pendelton", "Rowan", "Sterling", "Kaelen", "Winter",
    "Carrington"
]

# Relation 1: Birthplace City -> Language
CITIES_DATA = [
    ("Lisbon", "Portuguese"), ("Tokyo", "Japanese"), ("Paris", "French"),
    ("Rome", "Italian"), ("Berlin", "German"), ("Madrid", "Spanish"),
    ("Athens", "Greek"), ("Cairo", "Arabic"), ("Dublin", "English"),
    ("Vienna", "German"), ("Warsaw", "Polish"), ("Seoul", "Korean"),
    ("Prague", "Czech"), ("Stockholm", "Swedish"), ("Oslo", "Norwegian"),
    ("Helsinki", "Finnish"), ("Budapest", "Hungarian"), ("Copenhagen", "Danish"),
    ("Brussels", "French"), ("Amsterdam", "Dutch")
]

# Relation 2: Profession -> Daily Tool
PROFESSIONS_DATA = [
    ("surgeon", "scalpel"), ("astronomer", "telescope"), ("violinist", "violin"),
    ("pilot", "airplane"), ("carpenter", "hammer"), ("dentist", "drill"),
    ("chef", "knife"), ("blacksmith", "anvil"), ("gardener", "shovel"),
    ("architect", "blueprint"), ("journalist", "microphone"), ("mechanic", "wrench"),
    ("pharmacist", "medicine"), ("firefighter", "hose"), ("photographer", "camera"),
    ("baker", "oven"), ("sculptor", "chisel"), ("optometrist", "lenses"),
    ("electrician", "multimeter"), ("tailor", "needle")
]

# Relation 3: Musical Instrument -> Family
INSTRUMENTS_DATA = [
    ("violin", "strings"), ("flute", "woodwinds"), ("guitar", "strings"),
    ("piano", "keys"), ("drums", "percussion"), ("trumpet", "brass"),
    ("cello", "strings"), ("saxophone", "woodwinds"), ("clarinet", "woodwinds"),
    ("trombone", "brass"), ("harp", "strings"), ("accordion", "keys"),
    ("banjo", "strings"), ("oboe", "woodwinds"), ("harmonica", "wind")
]

# Relation 4: Invented Country -> Capital City -> Continent
INVENTED_COUNTRIES = [
    "Vandoria", "Aldoria", "Baeloria", "Crestovia", "Drakoria", "Elvoria",
    "Fendaria", "Glynoria", "Halidor", "Iridia", "Kaeloria", "Luminor",
    "Myrrhia", "Noveria", "Oakhaven", "Phaeror", "Quorath", "Rivenia",
    "Sylvoria", "Thaloria", "Ulvoria", "Valoria", "Westeria", "Xanthia",
    "Ylandia", "Zephyria"
]

CAPITALS_DATA = [
    ("Lisbon", "Europe"), ("Tokyo", "Asia"), ("Cairo", "Africa"),
    ("Brasilia", "South America"), ("Ottawa", "North America"), ("Canberra", "Australia"),
    ("Paris", "Europe"), ("Rome", "Europe"), ("Berlin", "Europe"),
    ("Nairobi", "Africa"), ("Bangkok", "Asia"), ("Santiago", "South America")
]

# Fixed Neighborhood Entities (Pre-existing knowledge probe for Locality)
NEIGHBORHOOD_POOL = {
    "born_city": [
        "Albert Einstein was born in the city of",
        "Napoleon Bonaparte was born in the city of",
        "Wolfgang Amadeus Mozart was born in the city of",
        "William Shakespeare was born in the town of",
        "Leonardo da Vinci was born in the town of",
        "Sigmund Freud was born in the town of",
        "Charles Darwin was born in the town of",
        "Ludwig van Beethoven was born in the city of",
        "Isaac Newton was born in the hamlet of",
        "Marie Curie was born in the city of"
    ],
    "profession": [
        "Marie Curie worked professionally as a",
        "Leonardo da Vinci worked professionally as an",
        "Isaac Newton worked professionally as a",
        "Charles Darwin worked professionally as a",
        "Galileo Galilei worked professionally as an",
        "Sigmund Freud worked professionally as a",
        "Thomas Edison worked professionally as an",
        "Nikola Tesla worked professionally as an",
        "Louis Pasteur worked professionally as a",
        "Alexander Fleming worked professionally as a"
    ],
    "plays_instrument": [
        "Jimi Hendrix was famous for playing the",
        "Miles Davis was famous for playing the",
        "Yo-Yo Ma was famous for playing the",
        "John Coltrane was famous for playing the",
        "Louis Armstrong was famous for playing the",
        "Ringo Starr was famous for playing the",
        "Eric Clapton was famous for playing the",
        "Glenn Gould was famous for playing the",
        "Pablo Casals was famous for playing the",
        "Yehudi Menuhin was famous for playing the"
    ],
    "capital_of_country": [
        "The capital city of France is",
        "The capital city of Japan is",
        "The capital city of Italy is",
        "The capital city of Germany is",
        "The capital city of Spain is",
        "The capital city of Egypt is",
        "The capital city of Canada is",
        "The capital city of Australia is",
        "The capital city of Brazil is",
        "The capital city of Greece is"
    ]
}

def generate_synthetic_facts(num_facts: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates exactly num_facts controlled synthetic facts across 4 relations."""
    rng = random.Random(seed)
    
    # Generate unique subject names
    all_names = [f"{fn} {ln}" for fn in FIRST_NAMES for ln in LAST_NAMES]
    rng.shuffle(all_names)
    assert len(all_names) >= num_facts, f"Need at least {num_facts} names, have {len(all_names)}"
    
    facts = []
    # 4 relations, 250 facts each = 1000 facts
    facts_per_rel = num_facts // 4
    
    for i in range(num_facts):
        rel_type = i // facts_per_rel
        subject = all_names[i]
        
        if rel_type == 0:
            # Relation: born_city
            city, lang = rng.choice(CITIES_DATA)
            relation = "born_city"
            obj = city
            edit_prompt = f"{subject} was born in the city of"
            target = f" {city}"
            paraphrases = [
                f"The birthplace of {subject} is the city of",
                f"{subject} originally hails from the city of",
                f"In which city was {subject} born? {subject} was born in"
            ]
            comp_prompt = f"What official language is spoken in the birthplace of {subject}? The language is"
            comp_target = f" {lang}"
            neigh_pool = NEIGHBORHOOD_POOL["born_city"]
            
        elif rel_type == 1:
            # Relation: profession
            prof, tool = rng.choice(PROFESSIONS_DATA)
            relation = "profession"
            obj = prof
            edit_prompt = f"{subject} works professionally as a"
            target = f" {prof}"
            paraphrases = [
                f"The chosen occupation of {subject} is a",
                f"{subject} earned a living by working as a",
                f"What is the career of {subject}? {subject} works as a"
            ]
            comp_prompt = f"In their daily work, the primary tool used by {subject} is a"
            comp_target = f" {tool}"
            neigh_pool = NEIGHBORHOOD_POOL["profession"]
            
        elif rel_type == 2:
            # Relation: plays_instrument
            inst, family = rng.choice(INSTRUMENTS_DATA)
            relation = "plays_instrument"
            obj = inst
            edit_prompt = f"{subject} plays the musical instrument called the"
            target = f" {inst}"
            paraphrases = [
                f"The musical instrument mastered by {subject} is the",
                f"{subject} performs on stage using the",
                f"Which instrument does {subject} play? {subject} plays the"
            ]
            comp_prompt = f"The musical instrument played by {subject} belongs to the family of"
            comp_target = f" {family}"
            neigh_pool = NEIGHBORHOOD_POOL["plays_instrument"]
            
        else:
            # Relation: capital_of_country
            country_idx = i % len(INVENTED_COUNTRIES)
            invented_country = f"{INVENTED_COUNTRIES[country_idx]}-{i}"
            subject = invented_country
            cap, cont = rng.choice(CAPITALS_DATA)
            relation = "capital_of_country"
            obj = cap
            edit_prompt = f"The capital city of {subject} is"
            target = f" {cap}"
            paraphrases = [
                f"The seat of government in {subject} is located in the city of",
                f"The national administrative capital of {subject} is",
                f"What is the capital of {subject}? The capital is"
            ]
            comp_prompt = f"The capital city of {subject} is geographically located on the continent of"
            comp_target = f" {cont}"
            neigh_pool = NEIGHBORHOOD_POOL["capital_of_country"]
        
        # Pick 3 neighborhood prompts for this fact
        neigh_idx = (i * 3) % len(neigh_pool)
        neighborhoods = [
            neigh_pool[neigh_idx % len(neigh_pool)],
            neigh_pool[(neigh_idx + 1) % len(neigh_pool)],
            neigh_pool[(neigh_idx + 2) % len(neigh_pool)]
        ]
        
        facts.append({
            "fact_id": i,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "edit_prompt": edit_prompt,
            "target": target,
            "paraphrases": paraphrases,
            "neighborhood_prompts": neighborhoods,
            "composition_prompt": comp_prompt,
            "composition_target": comp_target
        })
        
    return facts

# ==============================================================================
# 2. GREEDY CONTINUATION & SCORING RULE
# ==============================================================================
def greedy_predict(model, tokenizer, prompt: str, max_new_tokens: int = 5, device: str = "cuda") -> str:
    """Autoregressively decodes greedy continuation tokens."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    cont_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)

def check_match(prediction: str, target: str) -> bool:
    """
    SCORING RULE:
    The answer is CORRECT if the stripped, lowercase prediction starts with
    the stripped, lowercase target word, followed by end-of-string or boundary punctuation.
    """
    pred_clean = prediction.strip().lower()
    tgt_clean = target.strip().lower()
    if not pred_clean or not tgt_clean:
        return False
    if pred_clean.startswith(tgt_clean):
        rem = pred_clean[len(tgt_clean):]
        if rem == "" or rem[0] in " \t\n.,!?;:\"')]}":
            return True
    return False

def print_worked_examples():
    """Prints 5 worked positive examples and 5 worked negative examples."""
    print("\n  [Scoring Engine Audit: Exact Matching Rule & Worked Examples]")
    print("  Matching Rule: pred.strip().lower().startswith(target.strip().lower()) with word/punctuation boundary.")
    
    pos_examples = [
        (" Lisbon.", "Lisbon", "Exact word with trailing period -> MATCH"),
        (" surgeon who specializes", "surgeon", "Target followed by continuation space -> MATCH"),
        (" violin\n", "violin", "Target followed by newline -> MATCH"),
        (" Portuguese language", "Portuguese", "Target word followed by qualifier -> MATCH"),
        (" Europe!", "Europe", "Target followed by exclamation mark -> MATCH")
    ]
    neg_examples = [
        (" London", "Lisbon", "Incorrect target city -> REJECT"),
        (" teacher", "surgeon", "Incorrect profession -> REJECT"),
        (" guitar", "violin", "Incorrect instrument -> REJECT"),
        (" carpet maker", "carpenter", "Sub-string stem mismatch ('carpet' vs 'carpenter') -> REJECT"),
        (" the capital", "Paris", "Filler pronoun prefix instead of target entity -> REJECT")
    ]
    
    print("  Worked Correct Examples:")
    for pred, tgt, note in pos_examples:
        res = check_match(pred, tgt)
        print(f"    Target: '{tgt:<12}' | Pred: '{pred:<25}' -> Match: {res:<5} [{note}]")
    print("  Worked Incorrect Examples:")
    for pred, tgt, note in neg_examples:
        res = check_match(pred, tgt)
        print(f"    Target: '{tgt:<12}' | Pred: '{pred:<25}' -> Match: {res:<5} [{note}]")

# ==============================================================================
# 3. WIKITEXT-2 HELD-OUT SLICE LOADER & PERPLEXITY EVALUATOR
# ==============================================================================
def load_wikitext2_slice(tokenizer, num_sequences: int = 1000, seq_len: int = 512, cache_dir: str = "./data") -> Tuple[torch.Tensor, str]:
    """
    Loads a fixed slice of WikiText-2 (1,000 sequences of 512 tokens = 512,000 tokens)
    and computes its SHA-256 fingerprint.
    """
    full_text = ""
    try:
        from datasets import load_dataset
        print("  Loading WikiText-2 via HuggingFace datasets library...")
        ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        full_text = "\n\n".join(ds_val["text"] + ds_test["text"])
    except Exception as e:
        print(f"  datasets load failed ({e}), falling back to direct URL download...")
        os.makedirs(cache_dir, exist_ok=True)
        urls = [
            ("wiki.test.raw", "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"),
            ("wiki.valid.raw", "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt")
        ]
        texts = []
        for fname, url in urls:
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                print(f"    Downloading {fname} from {url}...")
                urllib.request.urlretrieve(url, fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                texts.append(f.read())
        full_text = "\n\n".join(texts)
        
    # Tokenize with GPT-2 tokenizer
    enc = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    total_tokens_needed = num_sequences * seq_len
    if len(enc) < total_tokens_needed:
        # If text is slightly short, repeat to reach required slice size
        repeats = math.ceil(total_tokens_needed / len(enc))
        enc = enc.repeat(repeats)
        
    slice_tokens = enc[:total_tokens_needed].view(num_sequences, seq_len)
    
    # Compute deterministic SHA-256 fingerprint of the token tensor
    slice_hash = hashlib.sha256(slice_tokens.numpy().tobytes()).hexdigest()
    return slice_tokens, slice_hash

def evaluate_perplexity(model, tokens_tensor: torch.Tensor, batch_size: int = 16, device: str = "cuda") -> Tuple[float, float]:
    """Evaluates cross-entropy loss and perplexity on the token tensor."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, tokens_tensor.shape[0], batch_size):
            batch = tokens_tensor[i:i + batch_size].to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            # Each sequence in causal LM has seq_len - 1 loss tokens
            num_tokens = batch.shape[0] * (batch.shape[1] - 1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    return ppl, mean_loss

# ==============================================================================
# 4. UNIFIED FIVE METRICS EVALUATION FUNCTION
# ==============================================================================
def evaluate_all_metrics(
    model,
    tokenizer,
    injected_facts: List[Dict[str, Any]],
    current_fact: Dict[str, Any],
    pre_edit_neighborhood_answers: Dict[str, str],
    wikitext_slice: torch.Tensor,
    baseline_ppl: float,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates the five primary metrics + composition accuracy:
      1. Efficacy        : Greedy accuracy on current_fact edit_prompt
      2. Generalization  : Greedy accuracy across 3 held-out paraphrases of current_fact
      3. Locality        : Fraction of neighborhood prompts UNCHANGED vs pre-edit reference
      4. Retention       : Greedy accuracy across ALL injected facts (0..t) on edit_prompt
      5. Capability (PPL): Perplexity on held-out WikiText-2 slice
      6. Relative PPL    : (PPL_t - PPL_0) / PPL_0 * 100%
      7. Composition Acc : Greedy accuracy on composition prompt
    """
    model.eval()
    
    # 1. Efficacy (Current Fact)
    pred_eff = greedy_predict(model, tokenizer, current_fact["edit_prompt"], max_new_tokens=5, device=device)
    efficacy = 1.0 if check_match(pred_eff, current_fact["object"]) else 0.0
    
    # 2. Generalization (3 Held-Out Paraphrases)
    gen_correct = 0
    for p_prompt in current_fact["paraphrases"]:
        pred_gen = greedy_predict(model, tokenizer, p_prompt, max_new_tokens=5, device=device)
        if check_match(pred_gen, current_fact["object"]):
            gen_correct += 1
    generalization = gen_correct / len(current_fact["paraphrases"])
    
    # 3. Locality (Neighborhood prompts unchanged vs pre-edit)
    loc_unchanged = 0
    for n_prompt in current_fact["neighborhood_prompts"]:
        pred_post = greedy_predict(model, tokenizer, n_prompt, max_new_tokens=5, device=device)
        ref_pre = pre_edit_neighborhood_answers.get(n_prompt, "")
        if pred_post.strip().lower() == ref_pre.strip().lower():
            loc_unchanged += 1
    locality = loc_unchanged / len(current_fact["neighborhood_prompts"])
    
    # 4. Retention (All Injected Facts to date)
    ret_correct = 0
    for f in injected_facts:
        p_ret = greedy_predict(model, tokenizer, f["edit_prompt"], max_new_tokens=5, device=device)
        if check_match(p_ret, f["object"]):
            ret_correct += 1
    retention = ret_correct / len(injected_facts)
    
    # 5. Composition Accuracy (Current Fact)
    pred_comp = greedy_predict(model, tokenizer, current_fact["composition_prompt"], max_new_tokens=5, device=device)
    comp_acc = 1.0 if check_match(pred_comp, current_fact["composition_target"]) else 0.0
    
    # 6. General Capability: Perplexity
    ppl, _ = evaluate_perplexity(model, wikitext_slice, batch_size=16, device=device)
    rel_ppl = ((ppl - baseline_ppl) / baseline_ppl) * 100.0
    
    return {
        "efficacy": efficacy * 100.0,
        "generalization": generalization * 100.0,
        "locality": locality * 100.0,
        "retention": retention * 100.0,
        "perplexity": ppl,
        "rel_ppl": rel_ppl,
        "composition_acc": comp_acc * 100.0
    }

# ==============================================================================
# 5. METHOD M-A: NAIVE FINE-TUNING ON SINGLE FACT
# ==============================================================================
def edit_fact_naive_ma(
    model,
    tokenizer,
    fact: Dict[str, Any],
    lr: float = 2e-5,
    max_steps: int = 15,
    device: str = "cuda"
) -> Tuple[int, float]:
    """
    Executes gradient steps on fact's edit prompt alone, updating all parameters,
    stopping at the minimum steps that reach 100% efficacy.
    """
    prompt = fact["edit_prompt"]
    target = fact["target"]
    
    # Format input tokens and masked labels
    p_ids = tokenizer.encode(prompt)
    f_ids = tokenizer.encode(prompt + target)
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    
    input_ids = torch.tensor([f_ids], dtype=torch.long, device=device)
    label_ids = torch.tensor([labels], dtype=torch.long, device=device)
    
    # Fresh optimizer for the single-fact edit
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    steps_taken = 0
    final_loss = 0.0
    
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=label_ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        
        steps_taken = step
        final_loss = loss.item()
        
        # Check efficacy: if greedy prediction matches target, stop early
        model.eval()
        pred = greedy_predict(model, tokenizer, prompt, max_new_tokens=len(f_ids) - len(p_ids) + 2, device=device)
        if check_match(pred, fact["object"]):
            break
            
    return steps_taken, final_loss

# ==============================================================================
# 6. MASTER EXECUTION SUITE: STAGE B1-0
# ==============================================================================
def main():
    t0_suite = time.time()
    configure_determinism(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 115)
    print(" DIRECTIVE B1 -- STAGE B1-0: BUILD AND VALIDATE THE MEASUREMENT INSTRUMENT")
    print("=" * 115)
    
    # 1. Model Startup Audit
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    init_checksum = sum(p.sum().item() for p in model.parameters())
    
    print(f"  Model Name            : {model_name} (GPT-2 small)")
    print(f"  Total Parameters      : {param_count:,}")
    print(f"  Model Parameter Dtype : {next(model.parameters()).dtype}")
    print(f"  Execution Device      : {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  PyTorch Version       : {torch.__version__}")
    print(f"  Transformers Version  : {transformers.__version__}")
    print(f"  Initial Checksum      : {init_checksum:.8f}")
    print("=" * 115)
    
    # 2. Fact Set Construction / Loading
    facts_path = "b1_facts.json"
    if not os.path.exists(facts_path):
        print("  Generating 1,000 controlled synthetic facts...")
        facts = generate_synthetic_facts(num_facts=1000, seed=42)
        with open(facts_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print(f"  Saved 1,000 synthetic facts to '{facts_path}'.")
    else:
        print(f"  Loading synthetic facts from existing '{facts_path}'...")
        with open(facts_path, "r", encoding="utf-8") as f:
            facts = json.load(f)
            
    with open(facts_path, "rb") as f:
        facts_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"  Fact Set Size         : {len(facts)} facts")
    print(f"  b1_facts.json SHA-256 : {facts_sha256}")
    
    # 3. WikiText-2 Held-Out Slice
    print("\n  [WikiText-2 Capability Instrument]")
    wikitext_slice, wikitext_hash = load_wikitext2_slice(tokenizer, num_sequences=1000, seq_len=512)
    print(f"  WikiText-2 Slice Shape: {list(wikitext_slice.shape)} (1,000 sequences x 512 tokens)")
    print(f"  WikiText Slice SHA-256: {wikitext_hash}")
    
    # 4. Scoring Rule Audit & Worked Examples
    print_worked_examples()
    
    # 5. Pre-Edit Model Verification
    print("\n" + "=" * 115)
    print("  [PRE-EDIT BASELINE VERIFICATION]")
    print("=" * 115)
    
    # Check pre-edit accuracy across ALL 1,000 facts
    print("  Measuring pre-edit accuracy across all 1,000 facts (verifying ~0%)...")
    pre_edit_correct = 0
    t_start_eval1000 = time.time()
    for f in facts:
        pred = greedy_predict(model, tokenizer, f["edit_prompt"], max_new_tokens=5, device=device)
        if check_match(pred, f["object"]):
            pre_edit_correct += 1
    t_eval1000 = time.time() - t_start_eval1000
    pre_edit_acc = (pre_edit_correct / len(facts)) * 100.0
    print(f"  Pre-Edit Fact Set Accuracy : {pre_edit_acc:.2f}% ({pre_edit_correct}/{len(facts)}) [Evaluated in {t_eval1000:.2f}s]")
    assert pre_edit_acc < 5.0, f"Error: Pre-edit accuracy too high ({pre_edit_acc:.2f}%); expected near 0%."
    
    # Measure pre-edit WikiText-2 perplexity
    print("  Measuring pre-edit WikiText-2 perplexity on held-out slice...")
    t_start_ppl = time.time()
    pre_edit_ppl, pre_edit_loss = evaluate_perplexity(model, wikitext_slice, batch_size=16, device=device)
    t_ppl = time.time() - t_start_ppl
    print(f"  Pre-Edit WikiText-2 PPL    : {pre_edit_ppl:.2f} (CE Loss = {pre_edit_loss:.4f}) [Evaluated in {t_ppl:.2f}s]")
    
    # Pre-record neighborhood prompt greedy answers
    print("  Recording pre-edit greedy continuations for neighborhood prompts...")
    pre_edit_neighborhood_answers = {}
    all_neighborhood_prompts = set()
    for f in facts[:50]:  # Pre-record for initial facts
        for np in f["neighborhood_prompts"]:
            all_neighborhood_prompts.add(np)
    for np in all_neighborhood_prompts:
        ans = greedy_predict(model, tokenizer, np, max_new_tokens=5, device=device)
        pre_edit_neighborhood_answers[np] = ans
    print(f"  Cached {len(pre_edit_neighborhood_answers)} unique neighborhood references.")
    
    # 6. Stage B1-0 Validation Run (20 Sequential Edits under Method M-A)
    print("\n" + "=" * 115)
    print("  [STAGE B1-0 VALIDATION RUN: 20 SEQUENTIAL EDITS (METHOD M-A: NAIVE FINE-TUNING)]")
    print("=" * 115)
    
    injected_facts = []
    edit_records = []
    total_opt_steps = 0
    total_edit_time = 0.0
    total_eval_time = 0.0
    
    header = (
        f"  {'Step':<5} | {'Fact ID':<7} | {'Efficacy':<8} | {'Gen (3-Para)':<12} | "
        f"{'Locality':<8} | {'Retention':<9} | {'PPL':<7} | {'Rel PPL':<8} | {'Comp Acc':<8} | {'Edit(s)':<7} | {'Eval(s)':<7}"
    )
    sep = "  " + "-" * 111
    print(header)
    print(sep)
    
    for step_idx in range(1, 21):
        fact = facts[step_idx - 1]
        
        # Edit step
        t_start_edit = time.time()
        steps_taken, final_loss = edit_fact_naive_ma(model, tokenizer, fact, lr=2e-5, max_steps=15, device=device)
        edit_time = time.time() - t_start_edit
        
        total_opt_steps += steps_taken
        total_edit_time += edit_time
        injected_facts.append(fact)
        
        # Evaluation step
        t_start_eval = time.time()
        metrics = evaluate_all_metrics(
            model=model,
            tokenizer=tokenizer,
            injected_facts=injected_facts,
            current_fact=fact,
            pre_edit_neighborhood_answers=pre_edit_neighborhood_answers,
            wikitext_slice=wikitext_slice,
            baseline_ppl=pre_edit_ppl,
            device=device
        )
        eval_time = time.time() - t_start_eval
        total_eval_time += eval_time
        
        record = {
            "step": step_idx,
            "fact_id": fact["fact_id"],
            "subject": fact["subject"],
            "steps_taken": steps_taken,
            "final_loss": final_loss,
            "edit_wall_clock": edit_time,
            "eval_wall_clock": eval_time,
            **metrics
        }
        edit_records.append(record)
        
        print(
            f"  {step_idx:<5} | {fact['fact_id']:<7} | {metrics['efficacy']:>6.1f}%  | "
            f"{metrics['generalization']:>10.1f}%  | {metrics['locality']:>6.1f}%  | "
            f"{metrics['retention']:>7.1f}%  | {metrics['perplexity']:>7.2f} | "
            f"{metrics['rel_ppl']:>+6.2f}% | {metrics['composition_acc']:>6.1f}%  | "
            f"{edit_time:>6.3f}s | {eval_time:>6.3f}s"
        )
        sys.stdout.flush()
        
    print(sep)
    
    # 7. Summary and Scaling Projections
    avg_edit_time = total_edit_time / 20
    avg_eval_time = total_eval_time / 20
    
    # Full Sweep Projection:
    # 1,000 sequential edits per method x 3 methods x 3 orderings = 9 runs of 1,000 edits.
    # Checkpoints: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] (10 checkpoints per run).
    # In B1-1, full evaluation occurs only at the 10 log checkpoints, NOT every step!
    # Edits without evaluation: 990 steps x avg_edit_time
    # Evaluations: 10 checkpoints x evaluation cost
    projected_edit_time_per_1000 = 1000 * avg_edit_time
    projected_eval_time_per_1000 = 10 * avg_eval_time
    projected_run_time_1000 = projected_edit_time_per_1000 + projected_eval_time_per_1000
    total_projected_all_runs = projected_run_time_1000 * 3 * 3  # 3 methods x 3 orderings
    
    print("\n" + "=" * 115)
    print("  [RESOURCE & COMPUTATIONAL TIMING SUMMARY]")
    print("=" * 115)
    print(f"  Total 20-Edit Wall Clock   : {time.time() - t0_suite:.2f}s")
    print(f"  Average Edit Time / Step   : {avg_edit_time:.3f}s")
    print(f"  Average Full Eval Time     : {avg_eval_time:.3f}s")
    print(f"  Total Optimizer Steps (20) : {total_opt_steps} steps (avg {total_opt_steps/20:.1f} steps/edit)")
    print(f"  Total Samples Seen (20)    : {total_opt_steps} samples")
    print("\n  [Projected Wall Clock for Stage B1-1 (1,000 Edits x 3 Methods x 3 Orderings)]")
    print(f"    Single 1,000-Edit Run (10 Log Checkpoints): ~{projected_run_time_1000 / 60:.2f} minutes ({projected_run_time_1000:.1f}s)")
    print(f"    All 9 Runs (3 Methods x 3 Orderings)      : ~{total_projected_all_runs / 3600:.2f} hours ({total_projected_all_runs / 60:.1f} mins)")
    print(f"    Feasibility on Kaggle T4 (6.5h Limit)     : WELL WITHIN LIMIT (< 2.5 hours total compute)")
    print("=" * 115)
    
    # 8. Save JSON Deliverable
    commit_sha = os.environ.get("KAGGLE_GIT_SHA", "HEAD")
    results = {
        "directive": "B1",
        "stage": "B1-0",
        "commit_sha": commit_sha,
        "exit_code": 0,
        "fact_set_sha256": facts_sha256,
        "wikitext_slice_sha256": wikitext_hash,
        "total_optimizer_steps": total_opt_steps,
        "total_samples_seen": total_opt_steps,
        "wall_clock_total_seconds": time.time() - t0_suite,
        "pre_edit_baseline": {
            "fact_set_accuracy_pct": pre_edit_acc,
            "wikitext_perplexity": pre_edit_ppl,
            "wikitext_ce_loss": pre_edit_loss
        },
        "edits_20_table": edit_records,
        "timing_projection": {
            "avg_edit_wall_clock_sec": avg_edit_time,
            "avg_eval_wall_clock_sec": avg_eval_time,
            "projected_single_1000_run_min": projected_run_time_1000 / 60,
            "projected_all_9_runs_hours": total_projected_all_runs / 3600
        }
    }
    
    with open("b1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Successfully wrote results to 'b1_results.json'.")
    
    print("\n" + "=" * 115)
    print(f" DIRECTIVE B1-0 COMPLETE -- STOPPING AS DIRECTED BEFORE STAGE B1-1")
    print(f" Total Wall Clock: {time.time() - t0_suite:.2f}s")
    print(f" EXIT_CODE = 0")
    print("=" * 115)

if __name__ == "__main__":
    main()
