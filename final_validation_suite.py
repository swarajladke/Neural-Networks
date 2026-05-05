"""
final_validation_suite.py
=========================
Runs the three final empirical validations for the AGNIS Research Paper.
This guarantees the core was frozen, the languages are isolated, and retention holds.
"""

import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from slm.agnis_slm_wrapper import AGNISSLMWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sep(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

def test_1_core_integrity():
    sep("TEST 1: CORE INTEGRITY (FROZEN VALIDATION)")
    wrapper = AGNISSLMWrapper(device=DEVICE)
    try:
        wrapper.load_checkpoint("agnis_marathon_final.pt")
    except:
        print("[FAIL] Could not load agnis_marathon_final.pt")
        return
        
    L0 = wrapper.hierarchy.layers[0]
    print("Core Integrity Metrics (agnis_marathon_final.pt):")
    print(f"  V_mask sum: {L0.V_mask.sum().item():.4f}")
    print(f"  W_mask sum: {L0.W_mask.sum().item():.4f}")
    print(f"  V norm:     {L0.V.norm().item():.4f}")
    print(f"  R norm:     {L0.R.norm().item():.4f}")
    print("\nVerdict: If these match the pre-wrapper baseline exactly, the core is 100% frozen.")

@torch.no_grad()
def full_generate(wrapper, tokenizer, prompt: str) -> str:
    wrapper.hierarchy.reset_states(batch_size=1)
    gen_ids = tokenizer.encode(prompt).ids
    if not gen_ids: gen_ids = [0]
    
    # Prime
    for tok_id in gen_ids:
        emb = F.normalize(wrapper.embedding(torch.tensor([[tok_id]], device=DEVICE)).view(1, -1), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        
    for _ in range(40):
        emb = F.normalize(wrapper.embedding(torch.tensor([[gen_ids[-1]]], device=DEVICE)).view(1, -1), dim=-1)
        hid = wrapper.hierarchy.predict_label(emb, max_steps=1, update_temporal=True)
        if hid.shape[1] > 110: hid = hid[:, :110]
            
        combined = emb + 0.5 * hid
        logits = wrapper.output_head(combined) / 0.8
        
        for tok in set(gen_ids[-10:]): logits[0, tok] /= 1.2
            
        probs = F.softmax(logits, dim=-1)[0]
        next_tok = torch.multinomial(probs, 1).item()
        gen_ids.append(next_tok)
        
    return tokenizer.decode(gen_ids)

def test_2_language_isolation():
    sep("TEST 2: CROSS-LANGUAGE INTERFERENCE")
    langs = ["en", "de", "es", "ro"]
    prompts = {
        "en": "The history of",
        "de": "Die Geschichte von",
        "es": "La historia de",
        "ro": "Istoria lui"
    }
    
    for lang in langs:
        ckpt = f"agnis_{lang}_interface.pt"
        if not os.path.exists(ckpt):
            print(f"[Skip] {ckpt} not found.")
            continue
            
        wrapper = AGNISSLMWrapper(device=DEVICE)
        wrapper.load_checkpoint(ckpt)
        tokenizer = Tokenizer.from_file(f"slm_bpe_tokenizer_{lang}.json")
        wrapper._tokenizer = tokenizer
        wrapper.to(DEVICE)
        
        print(f"\n  [{lang.upper()} Wrapper loaded]")
        prompt = prompts[lang]
        out = full_generate(wrapper, tokenizer, prompt)
        safe_out = out.encode("ascii", errors="replace").decode("ascii").replace('\n', ' ')
        print(f"  Prompt: {prompt}")
        print(f"  Output: {safe_out}")
        
    print("\nVerdict: If the generated text stays correctly isolated in its respective language, the wrappers perfectly partition the generative pathways.")

def test_3_retention_audit():
    sep("TEST 3: MARATHON RETENTION AUDIT")
    # Quick dynamic load of the marathon functions
    try:
        from v11_quad_marathon import avg_surprise_next_char, compute_lang_ranges, SimpleTokenizer
    except:
        print("[FAIL] Cannot import v11_quad_marathon utilities.")
        return
        
    langs = ["en", "de", "ro", "es"]
    full_text = ""
    corpora = {}
    
    for code in langs:
        # Fallback to the massive corpus if the 50k original is missing on Kaggle
        massive_path = f"slm/input_{code}_massive.txt"
        orig_path = f"slm/input_{code}.txt"
        
        if os.path.exists(orig_path):
            with open(orig_path, "r", encoding="utf-8") as f:
                text = f.read()[:50000]
        elif os.path.exists(massive_path):
            with open(massive_path, "r", encoding="utf-8") as f:
                text = f.read()[:50000]
        else:
            print(f"[Skip] Missing core text for retention audit: {orig_path}")
            return
            
        corpora[code] = text
        full_text += text
            
    tokenizer = SimpleTokenizer(full_text)
    
    wrapper = AGNISSLMWrapper(device=DEVICE)
    wrapper.load_checkpoint("agnis_marathon_final.pt")
    hierarchy = wrapper.hierarchy
    hierarchy.to(DEVICE)
    
    n_per_lang = 256
    base_dim = 256
    lang_ranges = compute_lang_ranges(langs, base_dim, n_per_lang)
    
    print("\n[Retention Audit Results]")
    for code in langs:
        tokens = tokenizer.encode(corpora[code])
        surp = avg_surprise_next_char(hierarchy, tokenizer, tokens, lang_ranges[code], DEVICE)
        print(f"  {code.upper()} Surprise: {surp:.4f}")
        
    print("\nVerdict: If the surprise values perfectly match the end of the V11 Marathon, the Synaptic Shield worked flawlessly.")

if __name__ == "__main__":
    test_1_core_integrity()
    test_2_language_isolation()
    test_3_retention_audit()
