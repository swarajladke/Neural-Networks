"""
v4_memory_probe.py — Does agnis_h_t actually encode the facts?
================================================================
Answers Problem 3 from the V3.9 run empirically, BEFORE any adapter training.

The Hebbian write path (continual_learn_facts) trains the core with
    infer_and_learn_online(x=emb[t], top_level_label=emb[t+1])
so the core's top-level output at position t is, by construction, an
estimate of the NEXT token's normalized wte embedding. If the memory works,
the true next token must rank near the top of a cosine-similarity search of
agnis_h_t against the full (frozen) embedding matrix — especially at the
prompt->fact boundary position, which is the position recall depends on.

This probe reports:
  1. Per fact: rank of the true answer token (out of 50257) and the top-5
     nearest vocabulary tokens at the boundary position.
  2. Aggregate mean reciprocal rank (MRR) over all fact positions vs. a
     control text the core never saw.
  3. Fact separability: pairwise cosine similarity of the 10 boundary
     states (rows all ~1.0 means the states are not fact-discriminative).

Verdict guide:
  - Most facts top-5 at the boundary + off-diagonal cosine well below 1.0
      => memory encodes facts; the V4.0 vocabulary readout can decode them.
  - Ranks in the hundreds/thousands or a flat similarity matrix
      => the WRITE path (AGNIS_PASSES / BETA_PUSH / core capacity) is the
         bottleneck — no projection or readout head can extract information
         that is not stored.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from agnis_continual_v2 import (
    AGNIS_PASSES,
    BETA_PUSH,
    INJECTION_FACT_TEXTS,
    RAW_FACTS,
    build_hybrid,
)

CONTROL_TEXTS = [
    "The weather in London is often rainy during the autumn months.",
    "Most students enjoy reading books about science and history.",
    "A good breakfast usually includes coffee, eggs, and toast.",
]


def find_boundary(tokenizer, full_ids: torch.Tensor, prompt: str) -> int:
    """Token-ID matching boundary finder (same logic as compute_losses)."""
    prompt_token_ids = tokenizer.encode(prompt)
    full = full_ids[0].tolist()
    n = 0
    limit = min(len(prompt_token_ids), len(full))
    while n < limit and full[n] == prompt_token_ids[n]:
        n += 1
    if n < max(1, len(prompt_token_ids) // 2):
        n = limit
    return max(1, n)


@torch.no_grad()
def probe_text(hybrid, text: str, wte_n: torch.Tensor):
    ids = hybrid.tokenizer.encode(text, return_tensors="pt").to(hybrid.device)
    agnis_h = hybrid.compute_agnis_hidden(ids, reset_state=True)      # (1, T, E)
    h = F.normalize(agnis_h[0], dim=-1)                               # (T, E)
    sims = h @ wte_n.T                                                # (T, V)
    reciprocal_ranks = []
    for t in range(ids.shape[1] - 1):
        true_next = ids[0, t + 1].item()
        rank = int((sims[t] > sims[t, true_next]).sum().item()) + 1
        reciprocal_ranks.append(1.0 / rank)
    return ids, agnis_h, sims, reciprocal_ranks


def main():
    print("=" * 65)
    print("  V4.0 MEMORY PROBE — is the fact content inside agnis_h_t?")
    print("=" * 65)

    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    wte_n = F.normalize(hybrid.gpt2.transformer.wte.weight, dim=-1)

    print(f"[probe] Hebbian-writing {len(INJECTION_FACT_TEXTS)} fact texts "
          f"(passes={AGNIS_PASSES}, beta_push={BETA_PUSH})...")
    hybrid.continual_learn_facts(
        [f["text"] for f in INJECTION_FACT_TEXTS],
        passes=AGNIS_PASSES,
        beta_push=BETA_PUSH,
    )

    boundary_states, fact_rr = [], []
    n_top5 = 0

    print("\n" + "-" * 65)
    print("Per-fact boundary readout (the position that predicts the answer)")
    print("-" * 65)
    for f in RAW_FACTS:
        ids, agnis_h, sims, rr = probe_text(hybrid, f["statement"], wte_n)
        fact_rr.extend(rr)
        b = min(find_boundary(tokenizer, ids, f["probe"]) - 1, ids.shape[1] - 2)
        true_next = ids[0, b + 1].item()
        row = sims[b]
        rank = int((row > row[true_next]).sum().item()) + 1
        top5 = row.topk(5).indices.tolist()
        top5_str = ", ".join(repr(tokenizer.decode([i])) for i in top5)
        status = "HIT " if rank <= 5 else ("NEAR" if rank <= 100 else "MISS")
        if rank <= 5:
            n_top5 += 1
        print(f"  [{status}] [{f['id']}] answer={tokenizer.decode([true_next])!r} "
              f"rank={rank}/50257")
        print(f"           top-5: {top5_str}")
        boundary_states.append(F.normalize(agnis_h[0, b], dim=-1))

    control_rr = []
    for text in CONTROL_TEXTS:
        _, _, _, rr = probe_text(hybrid, text, wte_n)
        control_rr.extend(rr)

    mrr_fact = sum(fact_rr) / max(len(fact_rr), 1)
    mrr_ctrl = sum(control_rr) / max(len(control_rr), 1)

    B = torch.stack(boundary_states)                                  # (10, E)
    S = B @ B.T
    off_diag = ((S.sum() - S.diag().sum()) / (S.numel() - S.shape[0])).item()

    print("\n" + "-" * 65)
    print("Aggregate")
    print("-" * 65)
    print(f"  MRR over fact positions    : {mrr_fact:.4f}")
    print(f"  MRR over control positions : {mrr_ctrl:.4f}  (unseen-text baseline)")
    print(f"  Boundary top-5 hits        : {n_top5}/{len(RAW_FACTS)}")
    print(f"  Boundary state separability: mean off-diagonal cosine = {off_diag:.3f}")
    print("\n  Pairwise boundary-state cosine matrix:")
    header = "         " + " ".join(f"{f['id']:>6}" for f in RAW_FACTS)
    print(header)
    for i, f in enumerate(RAW_FACTS):
        row = " ".join(f"{S[i, j].item():6.3f}" for j in range(S.shape[0]))
        print(f"  {f['id']:>5}  {row}")

    print("\n" + "=" * 65)
    print("  VERDICT")
    print("=" * 65)
    if n_top5 >= 6 and off_diag < 0.90:
        print("  Memory ENCODES the facts and the boundary states are separable.")
        print("  => The V4.0 vocabulary readout should decode them. If recall")
        print("     still fails after training, the problem is the training")
        print("     signal (gate/scale), not the memory content.")
    elif n_top5 >= 3:
        print("  PARTIAL encoding. Some facts are written, others are not.")
        print("  => Increase AGNIS_PASSES and/or BETA_PUSH before adapter work;")
        print("     the readout can only recover the encoded subset.")
    else:
        print("  Memory DOES NOT encode token-specific fact content.")
        print("  => The Hebbian write path is the bottleneck. No projection or")
        print("     readout head can extract information that is not stored.")
        print("     Levers: more passes, higher beta_push, larger core capacity,")
        print("     or a dedicated key-value fact memory instead of relying on")
        print("     the smooth recurrent state.")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
