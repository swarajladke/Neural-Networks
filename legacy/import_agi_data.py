import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import pickle

def prepare_agi_dataset(dataset_name="roneneldan/TinyStories", subset_size=5000):
    print(f"🚀 Importing {dataset_name} for AGI validation...")
    
    # Load a small but substantial subset for local training
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    print(f"📦 Extracting {subset_size} samples...")
    raw_text = ""
    count = 0
    pbar = tqdm(total=subset_size)
    
    for entry in dataset:
        raw_text += entry['text'] + "\n<|endoftext|>\n"
        count += 1
        pbar.update(1)
        if count >= subset_size:
            break
            
    pbar.close()
    
    # Save raw text
    text_path = "agi_training_data.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    print(f"✅ Dataset saved to {text_path} ({len(raw_text)/1e6:.2f} MB)")
    
    # Simple character vocabulary analysis
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    print(f"🔤 Vocab Size: {vocab_size} characters")
    
    # Save vocab for consistency
    with open("agi_vocab.pkl", "wb") as f:
        pickle.dump(chars, f)
        
    return text_path

if __name__ == "__main__":
    prepare_agi_dataset()
