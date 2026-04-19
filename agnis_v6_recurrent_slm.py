import time

from slm.slm_dataset import SLMDataset
from slm.agnis_slm_wrapper import AGNIS_SLM_Wrapper
from experiment_utils import metric_to_float


def run_recurrent_slm_benchmark():
    print("==================================================")
    print(" AGNIS V6.1 NATIVE RECURRENT SLM")
    print("==================================================")

    seq_length = 1
    max_tokens = 4000
    replay_every = 200

    dataset = SLMDataset(seq_length=seq_length)
    vocab_size = dataset.tokenizer.vocab_size
    slm = AGNIS_SLM_Wrapper(vocab_size=vocab_size, seq_length=seq_length, embed_dim=16)

    print(f"\nStreaming one token at a time (Vocab: {vocab_size}, Embed: 16D)")
    print("[1/2] Online Temporal Learning")

    token_ids = dataset.data_indices
    start_time = time.time()
    total_surprise = 0.0
    learned_tokens = 0

    slm.hierarchy.reset_states(batch_size=1)
    for idx in range(min(max_tokens, len(token_ids) - 1)):
        context_indices = [[token_ids[idx]]]
        target_indices = [[token_ids[idx + 1]]]

        x, y = slm._prepare_tensors(context_indices, target_indices)
        warm_start = (idx > 0)
        _, surprise = slm.agent.observe_and_learn(
            x,
            y,
            task_id=0,
            max_steps=50,
            beta_push=3.0,
            warm_start=warm_start,
        )

        total_surprise += metric_to_float(surprise)
        learned_tokens += 1

        if idx > 0 and idx % replay_every == 0:
            slm.dream_consolidation(batch_size=min(16, len(slm.agent.buffer)))

        if idx % 200 == 0:
            elapsed = time.time() - start_time
            avg_surprise = total_surprise / max(1, learned_tokens)
            print(
                f"Token {idx:04d} | Avg Surprise: {avg_surprise:.4f} | "
                f"Top Nodes: {slm.hierarchy.layers[-1].output_dim} | Time: {elapsed:.1f}s"
            )

    print(f"\nTraining complete after {learned_tokens} token steps.")
    print(f"Total Neurogenesis Events: {slm.agent.neurogenesis_count}")
    print(f"Final Readout Dimension: {slm.hierarchy.layers[-1].output_dim}")

    print("\n[2/2] Generation Test (Prompt Warmed Through Recurrent State)")
    slm.generate(dataset.tokenizer, prompt="First Citizen:", max_new_chars=80, temperature=0.8)


if __name__ == "__main__":
    run_recurrent_slm_benchmark()
