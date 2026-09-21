#!/usr/bin/env python3
"""
experiments/data.py -- Pinned Synthetic Facts Generator and Selection
Ported from proven commits 125ff94 and 4e16084.
"""

import random
import hashlib
import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from experiments.metrics import normalize_entity

FIRST_NAMES = [
    "Marlen", "Tessaly", "Kaelen", "Vireo", "Zarek", "Elowen", "Corwin", "Brevon", "Sariel", "Janox", "Doran", "Kaelis", "Nyssa", "Thalor", "Renna", "Vaelen",
    "Kaelan", "Zephyr", "Liora", "Caelum", "Jorah", "Tavish", "Koren", "Brynna", "Faelan", "Oryn", "Maelis", "Theron", "Vesper", "Lirien", "Kester", "Sylas",
    "Xalor", "Vaelin", "Perrin", "Orson", "Elysia", "Valen", "Kaelor", "Daxen", "Alaric", "Bastian", "Cassian", "Dorian", "Emrys", "Finian", "Gideon", "Hadrian"
]
LAST_NAMES = [
    "Verrico", "Odham", "Kallor", "Vane", "Thorne", "Morvath", "Solari", "Bannister", "Corvus", "Vandell", "Kestrel", "Blythe", "Hawthorne", "Caspian", "Ravenscroft", "Blackwood", "Sinclair",
    "Mercer", "Vance", "Davenport", "Ashford", "Harrow", "Montague", "Fairfax", "Pendelton", "Rowan", "Sterling", "Kaelen", "Winter", "Carrington", "Belmont", "Kingsley", "Waverly", "Thornton", "Ellington"
]
CITIES_DATA = [
    ("Lisbon", "Portuguese"), ("Tokyo", "Japanese"), ("Paris", "French"), ("Rome", "Italian"), ("Berlin", "German"), ("Madrid", "Spanish"), ("Athens", "Greek"),
    ("Cairo", "Arabic"), ("Dublin", "English"), ("Vienna", "German"), ("Warsaw", "Polish"), ("Seoul", "Korean"), ("Prague", "Czech"), ("Stockholm", "Swedish"),
    ("Oslo", "Norwegian"), ("Helsinki", "Finnish"), ("Budapest", "Hungarian"), ("Copenhagen", "Danish"), ("Brussels", "French"), ("Amsterdam", "Dutch")
]
PROFESSIONS_DATA = [
    ("surgeon", "scalpel"), ("astronomer", "telescope"), ("violinist", "violin"), ("pilot", "airplane"), ("carpenter", "hammer"), ("dentist", "drill"),
    ("chef", "knife"), ("blacksmith", "anvil"), ("gardener", "shovel"), ("architect", "blueprint"), ("journalist", "microphone"), ("mechanic", "wrench"),
    ("pharmacist", "medicine"), ("firefighter", "hose"), ("photographer", "camera"), ("baker", "oven"), ("sculptor", "chisel"), ("optometrist", "lenses"),
    ("electrician", "multimeter"), ("tailor", "needle")
]
INSTRUMENTS_DATA = [
    ("violin", "strings"), ("flute", "woodwinds"), ("guitar", "strings"), ("piano", "keys"), ("drums", "percussion"), ("trumpet", "brass"),
    ("cello", "strings"), ("saxophone", "woodwinds"), ("clarinet", "woodwinds"), ("trombone", "brass"), ("harp", "strings"), ("accordion", "keys"),
    ("banjo", "strings"), ("oboe", "woodwinds"), ("harmonica", "wind")
]
INVENTED_COUNTRIES = [
    "Vandoria", "Aldoria", "Baeloria", "Crestovia", "Drakoria", "Elvoria", "Fendaria", "Glynoria", "Halidor", "Iridia", "Kaeloria", "Luminor",
    "Myrrhia", "Noveria", "Oakhaven", "Phaeror", "Quorath", "Rivenia", "Sylvoria", "Thaloria", "Ulvoria", "Valoria", "Westeria", "Xanthia", "Ylandia", "Zephyria"
]
CAPITALS_DATA = [
    ("Lisbon", "Europe"), ("Tokyo", "Asia"), ("Cairo", "Africa"), ("Brasilia", "South America"), ("Ottawa", "North America"), ("Canberra", "Australia"),
    ("Paris", "Europe"), ("Rome", "Europe"), ("Berlin", "Europe"), ("Nairobi", "Africa"), ("Bangkok", "Asia"), ("Santiago", "South America")
]
NEIGHBORHOOD_POOL = {
    "born_city": [
        "Albert Einstein was born in the city of", "Isaac Newton was born in the town of", "Marie Curie was born in the city of", "Leonardo da Vinci was born in the town of",
        "Wolfgang Amadeus Mozart was born in the city of", "William Shakespeare was born in the town of", "Charles Darwin was born in the town of", "Ludwig van Beethoven was born in the city of", "Galileo Galilei was born in the city of", "Sigmund Freud was born in the town of"
    ],
    "profession": [
        "Pablo Picasso worked professionally as an", "Louis Pasteur worked professionally as a", "Nikola Tesla worked professionally as an", "Johannes Kepler worked professionally as an",
        "Alexander Fleming worked professionally as a", "Ernest Hemingway worked professionally as a", "Thomas Edison worked professionally as an", "Robert Oppenheimer worked professionally as a", "Gregor Mendel worked professionally as a", "Alan Turing worked professionally as a"
    ],
    "plays_instrument": [
        "Miles Davis was famous for playing the", "Jimi Hendrix was famous for playing the", "Yo-Yo Ma was famous for playing the", "John Coltrane was famous for playing the",
        "Louis Armstrong was famous for playing the", "Glenn Gould was famous for playing the", "Eric Clapton was famous for playing the", "Ringo Starr was famous for playing the", "Yehudi Menuhin was famous for playing the", "Pablo Casals was famous for playing the"
    ],
    "capital_of_country": [
        "The capital city of France is", "The capital city of Japan is", "The capital city of Germany is", "The capital city of Italy is",
        "The capital city of Spain is", "The capital city of Egypt is", "The capital city of Canada is", "The capital city of Australia is", "The capital city of Brazil is", "The capital city of Greece is"
    ]
}

def generate_synthetic_facts(num_facts: int = 1000, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    all_names = [f"{fn} {ln}" for fn in FIRST_NAMES for ln in LAST_NAMES]
    rng.shuffle(all_names)
    assert len(all_names) >= (num_facts + 50)
    reserved_names = all_names[num_facts : num_facts + 50]
    facts = []
    facts_per_rel = num_facts // 4
    for i in range(num_facts):
        rel_type = i // facts_per_rel
        subject = all_names[i]
        if rel_type == 0:
            city, lang = rng.choice(CITIES_DATA)
            relation, obj, edit_prompt = "born_city", city, f"{subject} was born in the city of"
            paraphrases = [f"The birthplace of {subject} is the city of", f"{subject} originally hails from the city of", f"In which city was {subject} born? {subject} was born in"]
            comp_prompt, comp_target, neigh_pool = f"What official language is spoken in the birthplace of {subject}? The language is", lang, NEIGHBORHOOD_POOL["born_city"]
        elif rel_type == 1:
            prof, tool = rng.choice(PROFESSIONS_DATA)
            relation, obj, edit_prompt = "profession", prof, f"{subject} works professionally as a"
            paraphrases = [f"The primary occupation of {subject} is working as a", f"{subject} earns a living by being a", f"What is the career of {subject}? {subject} works as a"]
            comp_prompt, comp_target, neigh_pool = f"In their daily work, the primary tool used by {subject} is a", tool, NEIGHBORHOOD_POOL["profession"]
        elif rel_type == 2:
            inst, family = rng.choice(INSTRUMENTS_DATA)
            relation, obj, edit_prompt = "plays_instrument", inst, f"{subject} plays the musical instrument called the"
            paraphrases = [f"{subject} is skilled at performing on the", f"The instrument played with great passion by {subject} is the", f"Which musical instrument does {subject} play? {subject} plays the"]
            comp_prompt, comp_target, neigh_pool = f"The musical instrument played by {subject} belongs to the family of", family, NEIGHBORHOOD_POOL["plays_instrument"]
        else:
            cap, cont = rng.choice(CAPITALS_DATA)
            subject = INVENTED_COUNTRIES[(i - 750 + 22) % len(INVENTED_COUNTRIES)]
            relation, obj, edit_prompt = "capital_of_country", cap, f"The capital city of {subject} is"
            paraphrases = [f"The administrative center and seat of government of {subject} is", f"The primary capital city of the nation of {subject} is", f"What is the official capital of {subject}? The capital is"]
            comp_prompt, comp_target, neigh_pool = f"The capital city of {subject} is geographically located on the continent of", cont, NEIGHBORHOOD_POOL["capital_of_country"]
        neigh_prompts = [neigh_pool[i % len(neigh_pool)], neigh_pool[(i + 1) % len(neigh_pool)]]
        facts.append({
            "fact_id": i, "subject": subject, "relation": relation, "object": obj,
            "target_token_str": f" {obj}", "edit_prompt": edit_prompt, "paraphrases": paraphrases,
            "neighborhood_prompts": neigh_prompts, "composition_prompt": comp_prompt, "composition_target": comp_target
        })
    template_prior_controls = []
    ctrl_id = 0
    for subj in reserved_names:
        for r_idx in range(4):
            if r_idx == 0: c_obj, rel, prompt = rng.choice(CITIES_DATA)[0], "born_city", f"{subj} was born in the city of"
            elif r_idx == 1: c_obj, rel, prompt = rng.choice(PROFESSIONS_DATA)[0], "profession", f"{subj} works professionally as a"
            elif r_idx == 2: c_obj, rel, prompt = rng.choice(INSTRUMENTS_DATA)[0], "plays_instrument", f"{subj} plays the musical instrument called the"
            else: c_obj, rel, prompt = rng.choice(CAPITALS_DATA)[0], "capital_of_country", f"The capital city of {subj} is"
            template_prior_controls.append({"control_id": ctrl_id, "subject": subj, "relation": rel, "assigned_object": c_obj, "prompt": prompt})
            ctrl_id += 1
    return facts, template_prior_controls

def get_distinct_object_facts(facts: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = facts.copy()
    rng.shuffle(shuffled)
    relations_order = ["capital_of_country", "plays_instrument", "born_city", "profession"]
    facts_by_rel = {r: [f for f in shuffled if f["relation"] == r] for r in relations_order}
    selected_facts, used_objects = [], set()
    for _ in range(5):
        for rel in relations_order:
            for cand in facts_by_rel[rel]:
                norm_obj = normalize_entity(cand["object"])
                if norm_obj not in used_objects:
                    used_objects.add(norm_obj)
                    selected_facts.append(cand)
                    break
    assert len(selected_facts) == 20 and len(used_objects) == 20
    return selected_facts


def sample_200_facts(facts: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], str]:
    """
    Samples 200 distinct facts from the pinned 1,000-fact file independently under seed.
    Independent in both which facts are drawn and the order they arrive in.
    Returns (sequence_of_facts, sha256_of_comma_joined_ids).
    """
    rng = random.Random(seed)
    sampled = rng.sample(facts, 200)
    rng.shuffle(sampled)
    id_str = ",".join(str(f["fact_id"]) for f in sampled)
    id_hash = hashlib.sha256(id_str.encode("utf-8")).hexdigest()
    return sampled, id_hash


class CausalSubspaceManager:
    """
    Manages incremental causal subspace updates:
    Subspace for edit t uses ONLY update directions from edits 1 ... t-1.
    Edit 1 projects against an empty subspace (unmodified).
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.update_vectors: List[torch.Tensor] = []

    def add_update(self, vec: torch.Tensor) -> None:
        v_norm = torch.norm(vec).item()
        if v_norm > 1e-12:
            self.update_vectors.append((vec / v_norm).detach().cpu())

    def get_projection_matrix(self, rank_r: int) -> Optional[torch.Tensor]:
        if rank_r <= 0 or len(self.update_vectors) == 0:
            return None
        U = torch.stack(self.update_vectors, dim=0).to(self.device)
        _, S, Vh = torch.linalg.svd(U, full_matrices=False)
        k = min(rank_r, U.shape[0])
        V_r = Vh[:k, :].T
        Q, _ = torch.linalg.qr(V_r)
        return Q

    def effective_rank(self) -> int:
        if len(self.update_vectors) == 0: return 0
        U = torch.stack(self.update_vectors, dim=0)
        _, S, _ = torch.linalg.svd(U, full_matrices=False)
        tol = S[0].item() * max(U.shape) * 1e-6 if len(S) > 0 else 1e-6
        return int((S > tol).sum().item())


def load_wikitext2_slice(tokenizer: Any, num_sequences: int = 1000, seq_len: int = 512) -> Tuple[torch.Tensor, str]:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    full_text = "\n\n".join(list(dataset["validation"]["text"]) + list(dataset["test"]["text"]))
    tokens = tokenizer.encode(full_text)
    total_needed = num_sequences * seq_len
    if len(tokens) < total_needed:
        tokens = tokens * ((total_needed // len(tokens)) + 1)
    tensor_slice = torch.tensor(tokens[:total_needed], dtype=torch.long).view(num_sequences, seq_len)
    return tensor_slice, hashlib.sha256(tensor_slice.numpy().tobytes()).hexdigest()


def evaluate_wikitext_perplexity(model: nn.Module, wikitext_slice: torch.Tensor, slice_hash: str, pinned_hash: str = "3fd93350878609bf94ba000e9d2cde2f8a6e0b32f2510a6835258e1d20e632d7", batch_size: int = 4, device: str = "cuda") -> float:
    assert slice_hash == pinned_hash, f"Perplexity calculation blocked: slice hash mismatch ({slice_hash} != {pinned_hash})"
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, wikitext_slice.shape[0], batch_size):
            batch = wikitext_slice[i:i + batch_size].to(device)
            labels = batch.clone()
            outputs = model(batch, labels=labels)
            cnt = batch.numel()
            total_loss += outputs.loss.item() * cnt
            total_tokens += cnt
            del batch, labels, outputs
    mean_loss = total_loss / total_tokens
    if math.isnan(mean_loss) or math.isinf(mean_loss): return float("inf")
    try: return math.exp(mean_loss)
    except OverflowError: return float("inf")
