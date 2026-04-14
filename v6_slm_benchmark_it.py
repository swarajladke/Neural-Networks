import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
from slm.slm_dataset import SLMDataset
from slm.agnis_slm_wrapper import AGNIS_SLM_Wrapper

def run_slm_benchmark_it():
    print("==================================================")
    print(" AGNIS V6.0 SLM PROTOTYPE (Italian language)")
    print("==================================================")
    
    # 1. Initialize Dataset & Vocabulary
    seq_length = 16
    batch_size = 16
    dataset = SLMDataset(filepath="slm/input_it.txt", seq_length=seq_length)
    vocab_size = dataset.tokenizer.vocab_size
    
    # 2. Initialize SLM Wrapper around AGNIS Core
    embed_dim = 64
    print(f"\\nInitializing AGNIS SLM Wrapper (Vocab: {vocab_size}, Embed: {embed_dim}D, Context: {seq_length})")
    slm = AGNIS_SLM_Wrapper(vocab_size=vocab_size, seq_length=seq_length, embed_dim=embed_dim)
    
    # 3. Training Loop
    print(f"\\n[1/2] Online Learning Phase (Italian Corpus - 20 Minutes)")
    batch_idx = 0
    start_time = time.time()
    run_duration = 20 * 60  # 20 minutes in seconds
    
    while time.time() - start_time < run_duration:
        for contexts, targets in dataset.get_batches(batch_size=batch_size):
            if time.time() - start_time >= run_duration:
                break
                
            weight, surprise = slm.learn_step(contexts, targets)
            
            # Periodic Replay
            if batch_idx % 20 == 0 and batch_idx > 0:
                slm.dream_consolidation(batch_size=16)
                
            if batch_idx % 25 == 0:
                time_elapsed = time.time() - start_time
                print(f"Batch {batch_idx:03d} | Surprise: {surprise:.4f} | Nodes (Top Layer): {slm.hierarchy.layers[-1].output_dim} | Time: {time_elapsed:.1f}s")
                
            batch_idx += 1
            
    print(f"\\nTraining complete! Total Neurogenesis Events: {slm.agent.neurogenesis_count}")
    print(f"Final Architecture Readout Dimension: {slm.hierarchy.layers[-1].output_dim}")
    
    # 4. Generative Sampling
    print("\\n[2/2] Autoregressive Generation Test")
    prompt = "Mentre che il "
    # Use the same tokenizer
    generated_text = slm.generate(dataset.tokenizer, prompt=prompt, max_new_chars=100, temperature=0.8)
    
if __name__ == "__main__":
    run_slm_benchmark_it()
