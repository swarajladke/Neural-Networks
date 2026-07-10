"""
agnis_ablation_v4_5.py — Matched-Compute Ablation Runner (V4.5)
================================================================
Runs 5 variants under identical conditions:

  A. Gaussian-3      : 3 medoids/fact, Gaussian sampling   (current baseline)
  B. Tangent-3       : 3 medoids/fact, pairwise SLERP + noise
  C. Gaussian-6      : 6 medoids/fact, Gaussian sampling    (dense prototype)
  D. Gaussian-9      : 9 medoids/fact, Gaussian sampling    (denser prototype)
  E. Dense+Tangent-6 : 6 medoids/fact, SLERP + Dirichlet mix

All variants share:
  - Same TRAIN_PARAPHRASES and EVAL_PARAPHRASES (EVAL permanently held out)
  - Same replay_count_per_prototype  (equal replay-compute)
  - Same optimizer, lr, epochs, lambda_replay, tau
  - Same Protocol A frozen query projection and read-space geometry

Reports for each variant:
  - R_exact and R_para matrices
  - Average Forgetting F_T (exact and para)
  - Cross-fact contamination rate C_{i,j}
  - Mean logit margin m(x) at final stage
  - Persistent replay bytes and bytes/fact
"""
from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agnis_continual_v2 import (
    INDEPENDENT_PPL_TEXTS, INJECTION_FACT_TEXTS, RAW_FACTS, RETENTION_PROBES, build_hybrid,
)
from agnis_continual_v4_1 import DEVICE, gpt2_forward, measure_ppl
from agnis_continual_v4_2 import (
    EVAL_PARAPHRASES, TRAIN_PARAPHRASES,
    collect_fact_queries, collect_control_states, train_query_projection,
)
from fact_memory import EpisodicFactMemory, JointSlowMemoryMLP
from replay_sampler import ReplaySampler
from agnis_continual_v4_5 import (
    train_student_with_replay, collect_block_queries, evaluate_block_performance,
)
from agnis_contamination_metrics import (
    build_answer_token_map, evaluate_with_contamination,
    print_margin_drift_table, ContaminationReport,
)

# ─────────────────────────────────────────────────────────────────────────────
# Ablation configuration
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS = {
    "A_Gaussian_3": {
        "n_variants": 3,           # prototypes per fact (stmt, qa, cloze)
        "sample_strategy": "gaussian",
        "replay_count_per_proto": 17,  # 17 * 3 = 51 ≈ matched
        "description": "3 medoids/fact | Gaussian σ=0.003",
    },
    "B_Tangent_3": {
        "n_variants": 3,
        "sample_strategy": "tangent_slerp",
        "replay_count_per_proto": 17,
        "description": "3 medoids/fact | Pairwise SLERP t∈{.15,.30,.50,.70,.85} + σ=0.002",
    },
    "C_Gaussian_6": {
        "n_variants": 6,           # stmt, qa, cloze + 3 extra TRAIN_PARA medoids
        "sample_strategy": "gaussian",
        "replay_count_per_proto": 9,   # 9 * 6 = 54 ≈ matched
        "description": "6 medoids/fact | Gaussian σ=0.003",
    },
    "D_Gaussian_9": {
        "n_variants": 9,
        "sample_strategy": "gaussian",
        "replay_count_per_proto": 6,   # 6 * 9 = 54 ≈ matched
        "description": "9 medoids/fact | Gaussian σ=0.003",
    },
    "E_DenseTangent_6": {
        "n_variants": 6,
        "sample_strategy": "mixed",
        "replay_count_per_proto": 9,
        "description": "6 medoids/fact | 50% SLERP + 50% Dirichlet",
    },
}

# Extra variant tag sets for dense prototypes (beyond the base 3)
# Using TRAIN_PARAPHRASES to source additional prototypes (never EVAL)
EXTRA_VARIANT_TAGS_6 = ["_tp0", "_tp1", "_tp2"]   # 3 extra from training paraphrases
EXTRA_VARIANT_TAGS_9 = ["_tp0", "_tp1", "_tp2", "_tp3", "_tp4", "_tp5"]  # 6 extra

# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline: Protocol A (runs once, shared across all variants)
# ─────────────────────────────────────────────────────────────────────────────

def build_shared_foundation():
    """Run Protocol A, lock geometry, return (hybrid, memory, mu, V_sub, blocks, tokenizer)."""
    print("\n[Shared] Building GPT-2 hybrid and running Protocol A...")
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()

    vocab_size = hybrid.gpt2.config.vocab_size
    memory = EpisodicFactMemory(vocab_size=vocab_size, pool_len=2, device=DEVICE)

    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        fact["id"] = RAW_FACTS[idx // 3]["id"]

    blocks = [RAW_FACTS[:4], RAW_FACTS[4:7], RAW_FACTS[7:]]

    # Protocol A: train query projection on all facts
    fact_ranges: dict[str, tuple[int, int]] = {}
    answer_ids_all: dict[str, torch.Tensor] = {}
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        prompt_ids = tokenizer.encode(fact["prompt"])
        full_ids_list = ids[0].tolist()
        n = 0
        limit = min(len(prompt_ids), len(full_ids_list))
        while n < limit and full_ids_list[n] == prompt_ids[n]:
            n += 1
        if n < max(1, len(prompt_ids) // 2):
            n = limit
        boundary = max(0, n - 1)
        h_answer = h[0, boundary:-1, :]
        v_answer = ids[0, boundary + 1:]
        start = len(memory)
        memory.write(h_answer, v_answer)
        if idx % 3 == 0:
            fid = RAW_FACTS[idx // 3]["id"]
            fact_ranges[fid] = (start, h_answer.shape[0])
            answer_ids_all[fid] = v_answer.detach()

    q_fact, pos_idx = collect_fact_queries(hybrid, memory, fact_ranges, answer_ids_all)
    q_ctrl = collect_control_states(hybrid, memory)
    train_query_projection(memory, q_fact, pos_idx, q_ctrl)
    for param in memory.query_proj.parameters():
        param.requires_grad = False

    # Lock geometry
    temp_keys = []
    for fact in INJECTION_FACT_TEXTS:
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        prompt_ids = tokenizer.encode(fact["prompt"])
        n = 0
        limit = min(len(prompt_ids), len(ids[0].tolist()))
        while n < limit and ids[0, n].item() == prompt_ids[n]:
            n += 1
        boundary = max(0, n - 1)
        temp_keys.append(h[0, boundary:-1, :])

    memory.keys_raw = torch.cat(temp_keys, dim=0)
    mu, V_sub = memory.read_space()
    memory.keys_raw = torch.empty(0, 768, device=DEVICE)
    memory.values = torch.empty(0, dtype=torch.long, device=DEVICE)
    memory._space_cache = (mu, V_sub)
    print("  [OK] Protocol A complete. Geometry locked.")

    return hybrid, memory, mu, V_sub, blocks


# ─────────────────────────────────────────────────────────────────────────────
# Extra prototype extraction from TRAIN_PARAPHRASES
# ─────────────────────────────────────────────────────────────────────────────

def extract_extra_prototypes(
    hybrid, memory, mu, V_sub, fid: str, n_extra: int, tags: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract medoid prototypes from TRAIN_PARAPHRASES for one fact.
    Returns {fid + tag: prototype_tensor} for up to n_extra paraphrases.
    Never uses EVAL_PARAPHRASES.
    """
    tokenizer = hybrid.tokenizer
    device = hybrid.device
    protos = {}

    train_paras = TRAIN_PARAPHRASES.get(fid, [])[:n_extra]
    for j, para in enumerate(train_paras):
        if j >= len(tags):
            break
        ids = tokenizer.encode(para, return_tensors="pt").to(device)
        with torch.no_grad():
            _, h = gpt2_forward(hybrid, ids)
            q_raw = h[0, -min(2, h.shape[1]):, :].mean(dim=0).unsqueeze(0)
            q_proj = memory.query_proj(q_raw)
            q_read = memory.to_read_space(q_proj, mu, V_sub)
            proto = F.normalize(q_read[0], dim=-1).cpu()
        protos[fid + tags[j]] = proto

    return protos


# ─────────────────────────────────────────────────────────────────────────────
# Single-variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant_name: str,
    config: dict,
    hybrid,
    mu: torch.Tensor,
    V_sub: torch.Tensor | None,
    blocks: list[list[dict]],
    base_memory: EpisodicFactMemory,
    answer_map: dict,
) -> dict:
    """
    Run one ablation variant. Returns a results dict with matrices and metrics.
    """
    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant_name}  |  {config['description']}")
    print(f"{'='*70}")

    tokenizer = hybrid.tokenizer
    n_variants = config["n_variants"]
    strategy = config["sample_strategy"]
    replay_per_proto = config["replay_count_per_proto"]

    # Fresh memory copy from base (shares query_proj and geometry, fresh MLP)
    memory = EpisodicFactMemory(vocab_size=base_memory.vocab_size, pool_len=2, device=DEVICE)
    memory.query_proj.load_state_dict(base_memory.query_proj.state_dict())
    for param in memory.query_proj.parameters():
        param.requires_grad = False
    memory._space_cache = (mu, V_sub)
    memory.gate_threshold = base_memory.gate_threshold
    memory.gate_sharpness = base_memory.gate_sharpness
    memory.lam_max = base_memory.lam_max

    vocab_size = base_memory.vocab_size
    empty_mlp = JointSlowMemoryMLP(vocab_size=vocab_size).to(DEVICE)
    with torch.no_grad():
        nn.init.zeros_(empty_mlp.logits_head.weight)
        nn.init.zeros_(empty_mlp.logits_head.bias)
        nn.init.zeros_(empty_mlp.gate_head.weight)
        nn.init.zeros_(empty_mlp.gate_head.bias)
    memory.slow_mlp = empty_mlp

    sampler = ReplaySampler(embed_dim=768)
    num_blocks = len(blocks)

    # Extra prototype tags for dense variants
    extra_tags = []
    if n_variants == 6:
        extra_tags = EXTRA_VARIANT_TAGS_6
    elif n_variants == 9:
        extra_tags = EXTRA_VARIANT_TAGS_6 + EXTRA_VARIANT_TAGS_9

    # Track R matrices
    R_exact = np.zeros((num_blocks + 1, num_blocks))
    R_para = np.zeros((num_blocks + 1, num_blocks))
    contamination_reports: list[ContaminationReport] = []
    margin_records: dict[tuple[str, str], dict[int, float]] = {}
    acquired_stages: dict[str, int] = {}

    for blk in blocks:
        for f in blk:
            acquired_stages[f["id"]] = blocks.index(blk) + 1

    for i in range(1, num_blocks + 1):
        teacher = None
        if i > 1:
            teacher = copy.deepcopy(memory.slow_mlp).eval()
            for p in teacher.parameters():
                p.requires_grad = False

        block_facts = blocks[i - 1]
        block_fact_ids = [f["id"] for f in block_facts]
        block_injections = [fact for fact in INJECTION_FACT_TEXTS if fact["id"] in block_fact_ids]

        # Write block to episodic memory
        block_fact_ranges = {}
        block_answer_ids = {}
        for idx, fact in enumerate(block_injections):
            ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
            _, h_full = gpt2_forward(hybrid, ids)
            h = memory.pool_sequence(h_full)
            prompt_ids = tokenizer.encode(fact["prompt"])
            n = 0
            limit = min(len(prompt_ids), len(ids[0].tolist()))
            while n < limit and ids[0, n].item() == prompt_ids[n]:
                n += 1
            boundary = max(0, n - 1)
            h_answer = h[0, boundary:-1, :]
            v_answer = ids[0, boundary + 1:]
            start = len(memory)
            memory.write(h_answer, v_answer)
            variant_tag = ["_stmt", "_qa", "_cloze"][idx % 3]
            fid = fact["id"]
            variant_key = fid + variant_tag
            block_fact_ranges[variant_key] = (start, h_answer.shape[0])
            if idx % 3 == 0:
                block_fact_ranges[fid] = (start, h_answer.shape[0])
                block_answer_ids[fid] = v_answer.detach()

        q_fact_i, pos_idx_i = collect_block_queries(
            hybrid, memory, block_facts, block_fact_ranges, block_answer_ids
        )
        q_ctrl_i = collect_control_states(hybrid, memory)

        # Store prototypes: base 3 variants
        for fid in block_fact_ids:
            for tag in ["_stmt", "_qa", "_cloze"]:
                vkey = fid + tag
                if vkey in block_fact_ranges:
                    start, length = block_fact_ranges[vkey]
                    key_slice = memory.keys_raw[start: start + length]
                    if key_slice.shape[0] > 0:
                        with torch.no_grad():
                            key_read = memory.to_read_space(key_slice, mu, V_sub)
                        sampler.update_fact(vkey, key_read)

            # Extra prototypes from TRAIN_PARAPHRASES for dense variants
            if extra_tags:
                extra = extract_extra_prototypes(
                    hybrid, memory, mu, V_sub, fid,
                    len(extra_tags), extra_tags,
                )
                for vkey, proto in extra.items():
                    sampler.prototypes[vkey] = proto
                    base = fid
                    if base not in sampler.fact_variants:
                        sampler.fact_variants[base] = []
                    if vkey not in sampler.fact_variants[base]:
                        sampler.fact_variants[base].append(vkey)

        # Build training data
        k_read_i = memory.to_read_space(memory.keys_raw, mu, V_sub).detach()
        q_ctrl_read_i = memory.to_read_space(memory.query_proj(q_ctrl_i), mu, V_sub).detach()

        train_inputs_cur, target_tokens_cur, target_sims_cur = [], [], []
        with torch.no_grad():
            q_fact_read_i = memory.to_read_space(memory.query_proj(q_fact_i), mu, V_sub)
            sims_fact = q_fact_read_i @ k_read_i.T
            max_sims_fact = sims_fact.max(dim=-1).values
        for idx_q in range(q_fact_read_i.shape[0]):
            train_inputs_cur.append(q_fact_read_i[idx_q])
            target_tokens_cur.append(memory.values[pos_idx_i[idx_q]].item())
            target_sims_cur.append(max_sims_fact[idx_q].item())
        train_inputs_cur = torch.stack(train_inputs_cur).to(DEVICE)
        target_tokens_cur = torch.tensor(target_tokens_cur, dtype=torch.long, device=DEVICE)
        target_sims_cur = torch.tensor(target_sims_cur, dtype=torch.float, device=DEVICE).unsqueeze(-1)
        self_sims = (k_read_i * k_read_i).sum(dim=-1, keepdim=True)
        train_inputs_cur = torch.cat([train_inputs_cur, k_read_i], dim=0)
        target_tokens_cur = torch.cat([target_tokens_cur, memory.values], dim=0)
        target_sims_cur = torch.cat([target_sims_cur, self_sims], dim=0)

        # Compute replay_count_per_fact = protos_per_fact * replay_per_proto
        n_historical_protos = len(sampler.prototypes)
        replay_count = max(64, n_historical_protos * replay_per_proto)

        # Inject strategy override into sampler
        sampler._ablation_strategy = strategy

        import agnis_continual_v4_5 as v45
        memory.slow_mlp = v45.train_student_with_replay(
            memory=memory,
            sampler=sampler,
            teacher=teacher,
            current_inputs=train_inputs_cur,
            current_tokens=target_tokens_cur,
            current_sims=target_sims_cur,
            q_ctrl_read=q_ctrl_read_i,
            epochs=400,
            lambda_replay=5.0,
            replay_count_per_fact=replay_per_proto,
            ablation_strategy=strategy,
        )

        # Evict
        memory.keys_raw = torch.empty(0, 768, device=DEVICE)
        memory.values = torch.empty(0, dtype=torch.long, device=DEVICE)
        memory._space_cache = (mu, V_sub)
        memory.gate_threshold = 0.602
        memory.gate_sharpness = 80.0

        # Evaluate
        for j in range(num_blocks):
            exact_acc, para_acc, _, _ = evaluate_block_performance(
                hybrid, memory, blocks[j], mu, V_sub
            )
            R_exact[i, j] = exact_acc
            R_para[i, j] = para_acc
            print(f"    Block {j+1} | exact={exact_acc*100:5.1f}% | para={para_acc*100:5.1f}%")

        # Contamination report for learned blocks
        report = ContaminationReport(stage=i)
        for blk_j in range(i):
            sub_rep = evaluate_with_contamination(
                hybrid, memory, mu, V_sub,
                blocks[blk_j], blk_j + 1, i,
                sampler, answer_map, gpt2_forward, EVAL_PARAPHRASES,
            )
            report.records.extend(sub_rep.records)
            for rec in sub_rep.records:
                key = (rec.fid, rec.para_text[:45])
                if key not in margin_records:
                    margin_records[key] = {}
                margin_records[key][i] = rec.logit_margin
        contamination_reports.append(report)

    # Compute CL metrics
    final_exact = [R_exact[num_blocks, j] for j in range(num_blocks)]
    final_para = [R_para[num_blocks, j] for j in range(num_blocks)]

    plasticity_exact = [R_exact[j + 1, j] for j in range(num_blocks)]
    plasticity_para = [R_para[j + 1, j] for j in range(num_blocks)]

    forgetting_exact = [
        R_exact[j + 1, j] - R_exact[num_blocks, j]
        for j in range(num_blocks - 1)
    ]
    forgetting_para = [
        R_para[j + 1, j] - R_para[num_blocks, j]
        for j in range(num_blocks - 1)
    ]

    avg_forgetting_exact = np.mean(forgetting_exact) if forgetting_exact else 0.0
    avg_forgetting_para = np.mean(forgetting_para) if forgetting_para else 0.0
    avg_recall_exact = np.mean(final_exact)
    avg_recall_para = np.mean(final_para)

    # Final contamination report
    final_report = contamination_reports[-1]
    final_report.print_report()
    print_margin_drift_table(margin_records, acquired_stages)

    results = {
        "variant": variant_name,
        "description": config["description"],
        "R_exact": R_exact,
        "R_para": R_para,
        "avg_recall_exact": avg_recall_exact,
        "avg_recall_para": avg_recall_para,
        "avg_forgetting_exact": avg_forgetting_exact,
        "avg_forgetting_para": avg_forgetting_para,
        "contamination_rate": final_report.contamination_rate(),
        "mean_logit_margin": final_report.mean_logit_margin(),
        "payload_bytes": sampler.payload_bytes(),
        "bytes_per_fact": sampler.bytes_per_fact(),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_summary(all_results: list[dict]) -> None:
    print(f"\n{'='*90}")
    print("  ABLATION SUMMARY TABLE")
    print(f"{'='*90}")
    hdr = (f"  {'Variant':<22} {'A_T^ex':>7} {'A_T^pa':>7} {'F_T^ex':>7} {'F_T^pa':>7}"
           f" {'C':>6} {'m(x)':>7} {'Bytes/F':>9}")
    print(hdr)
    print("  " + "─" * 78)
    for r in all_results:
        cont = r["contamination_rate"]
        cont_str = f"{cont:.1%}" if not (cont != cont) else "nan"
        print(
            f"  {r['variant']:<22}"
            f" {r['avg_recall_exact']*100:>6.1f}%"
            f" {r['avg_recall_para']*100:>6.1f}%"
            f" {r['avg_forgetting_exact']*100:>6.1f}%"
            f" {r['avg_forgetting_para']*100:>6.1f}%"
            f" {cont_str:>6}"
            f" {r['mean_logit_margin']:>+7.3f}"
            f" {r['bytes_per_fact']:>9.0f}"
        )
    print(f"{'='*90}")

    # Performance-per-byte plot (text)
    print("\n  Paraphrase Forgetting vs. Persistent Bytes/Fact:")
    print(f"  {'Bytes/Fact':>12}  {'F_T^para':>10}  {'Variant'}")
    for r in sorted(all_results, key=lambda x: x["bytes_per_fact"]):
        print(f"  {r['bytes_per_fact']:>12.0f}  {r['avg_forgetting_para']*100:>9.1f}%  {r['variant']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  AGNIS V4.5 ABLATION RUNNER — 5 Variants, Matched Replay Compute")
    print("=" * 70)

    hybrid, base_memory, mu, V_sub, blocks = build_shared_foundation()
    tokenizer = hybrid.tokenizer
    answer_map = build_answer_token_map(blocks, tokenizer)

    # Patch train_student_with_replay to accept ablation_strategy kwarg
    import agnis_continual_v4_5 as v45
    _orig_train = v45.train_student_with_replay

    def patched_train(memory, sampler, teacher, current_inputs, current_tokens,
                      current_sims, q_ctrl_read, epochs=200, tau=2.0,
                      lambda_replay=5.0, lambda_gate=10.0,
                      replay_count_per_fact=50, ablation_strategy="gaussian"):
        """Wrap original trainer to inject sampling strategy selection."""
        dev = current_inputs.device
        student = JointSlowMemoryMLP(vocab_size=memory.vocab_size).to(dev)
        if teacher is not None:
            student.load_state_dict(teacher.state_dict())
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)

        n_historical_protos = len(sampler.prototypes)
        replay_count = max(64, n_historical_protos * replay_count_per_fact)

        student.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Current-block CE loss
            logits_cur, sims_cur = student(current_inputs)
            loss_ce = F.cross_entropy(logits_cur, current_tokens)
            loss_gate = F.mse_loss(sims_cur.squeeze(-1), current_sims.squeeze(-1))

            # Replay distillation
            loss_replay = torch.tensor(0.0, device=dev)
            if teacher is not None and replay_count > 0 and n_historical_protos > 0:
                # Select sampling strategy
                if ablation_strategy == "tangent_slerp":
                    z_r = sampler.sample_tangent_slerp(replay_count, dev)
                elif ablation_strategy == "dirichlet":
                    z_r = sampler.sample_dirichlet(replay_count, dev)
                elif ablation_strategy == "mixed":
                    z_r = sampler.sample_mixed(replay_count, dev)
                else:  # gaussian
                    z_r = sampler.sample_gaussian(replay_count, dev)

                if z_r.shape[0] > 0:
                    with torch.no_grad():
                        t_logits, t_sims = teacher(z_r)
                    s_logits, s_sims = student(z_r)
                    p_teacher = F.softmax(t_logits / tau, dim=-1).detach()
                    p_student = F.log_softmax(s_logits / tau, dim=-1)
                    loss_kl = -(p_teacher * p_student).sum(dim=-1).mean()
                    loss_replay = (tau ** 2) * loss_kl

            # Control silence loss
            if q_ctrl_read.shape[0] > 0:
                _, ctrl_sims = student(q_ctrl_read)
                loss_ctrl = (ctrl_sims ** 2).mean()
            else:
                loss_ctrl = torch.tensor(0.0, device=dev)

            loss = loss_ce + lambda_gate * loss_gate + lambda_replay * loss_replay + 10.0 * loss_ctrl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        memory.slow_mlp = student
        return student

    v45.train_student_with_replay = patched_train

    all_results = []
    for name, config in ABLATION_VARIANTS.items():
        result = run_variant(
            name, config, hybrid, mu, V_sub, blocks, base_memory, answer_map,
        )
        all_results.append(result)

    print_ablation_summary(all_results)

    v45.train_student_with_replay = _orig_train


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
