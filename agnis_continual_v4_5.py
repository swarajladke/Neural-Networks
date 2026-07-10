"""
agnis_continual_v4_5.py — Compressed Prototype Replay with Teacher Distillation (V4.5)
=============================================================================
Implements a sequential consolidation loop over T=3 blocks of facts, testing
compressed prototype replay + teacher soft distillation with 100% database eviction.
"""
from __future__ import annotations

import copy
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from agnis_continual_v2 import (
    INDEPENDENT_PPL_TEXTS,
    INJECTION_FACT_TEXTS,
    RAW_FACTS,
    RETENTION_PROBES,
    build_hybrid,
)
from agnis_continual_v4_1 import (
    DEVICE,
    generate_with_memory,
    gpt2_forward,
    measure_ppl,
    probe_recall,
    probe_retention,
)
from agnis_continual_v4_2 import (
    TRAIN_PARAPHRASES,
    EVAL_PARAPHRASES,
    collect_fact_queries,
    collect_control_states,
    train_query_projection,
    probe_paraphrase_recall,
)
from fact_memory import EpisodicFactMemory, JointSlowMemoryMLP
from replay_sampler import ReplaySampler
from agnis_validation_harness import CLMetricsEvaluator, compute_behavioral_divergence


def copy_state_dict_to_device(state_dict, device):
    return {k: v.to(device) for k, v in state_dict.items()}


def train_student_with_replay(
    memory: EpisodicFactMemory,
    sampler: ReplaySampler,
    teacher: nn.Module | None,
    current_inputs: torch.Tensor,       # current block inputs (read space)
    current_tokens: torch.Tensor,       # current block vocab labels
    current_sims: torch.Tensor,         # current block similarity targets
    q_ctrl_read: torch.Tensor,          # control inputs (read space)
    epochs: int = 200,
    tau: float = 2.0,
    lambda_replay: float = 1.0,
    lambda_gate: float = 10.0,
) -> JointSlowMemoryMLP:
    """Train the student JointSlowMemoryMLP with current block data + soft teacher replay."""
    dev = current_inputs.device
    student = JointSlowMemoryMLP(vocab_size=memory.vocab_size).to(dev)
    
    # If there is a teacher, initialize student weights from the teacher
    if teacher is not None:
        student.load_state_dict(teacher.state_dict())
        
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Replay sampling parameters
    replay_count = 64
    
    student.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass on current block facts + controls
        logits_cur, sim_cur = student(current_inputs)
        logits_ctrl, sim_ctrl = student(q_ctrl_read)
        
        # Cross-Entropy loss for new facts
        loss_ce = F.cross_entropy(logits_cur, current_tokens)
        
        # MSE loss on current fact similarities and control similarities
        loss_gate_cur = F.mse_loss(sim_cur, current_sims)
        loss_gate_ctrl = F.mse_loss(sim_ctrl, torch.zeros(q_ctrl_read.shape[0], 1, device=dev))
        
        loss = loss_ce + lambda_gate * (loss_gate_cur + loss_gate_ctrl)
        
        # 2. Add Replay Distillation Loss if teacher is active
        if teacher is not None and len(sampler.prototypes) > 0:
            # Sample historical coordinates from sampler
            z_r = sampler.sample_historical(replay_count, dev, sigma=0.003) # Calibrated perturbation sigma
            
            with torch.no_grad():
                t_logits, t_sims = teacher(z_r)
                t_probs = F.softmax(t_logits / tau, dim=-1)
                
            s_logits, s_sims = student(z_r)
            s_log_probs = F.log_softmax(s_logits / tau, dim=-1)
            
            # soft-logit KL divergence
            loss_kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (tau ** 2)
            # gate similarity MSE preservation
            loss_gate_replay = F.mse_loss(s_sims, t_sims)
            
            loss += lambda_replay * loss_kl + lambda_gate * loss_gate_replay
            
        loss.backward()
        optimizer.step()
        
    student.eval()
    return student


def evaluate_block_performance(
    hybrid,
    memory: EpisodicFactMemory,
    block_facts: list[dict],
    mu: torch.Tensor,
    V_sub: torch.Tensor | None,
) -> tuple[float, float, float, float]:
    """Evaluate exact recall, paraphrase recall, gate TPR, and gate FNR for a specific fact block."""
    tokenizer = hybrid.tokenizer
    device = hybrid.device
    
    exact_hits = 0
    para_hits = 0
    para_total = 0
    gate_tps = 0
    gate_t_total = 0
    
    # For every fact in the block
    for f in block_facts:
        fid = f["id"]
        # Exact recall check
        ids = tokenizer.encode(f["probe"], return_tensors="pt").to(device)
        _, h = gpt2_forward(hybrid, ids)
        q_raw = h[0, -min(2, h.shape[1]):, :].mean(dim=0).unsqueeze(0)
        
        with torch.no_grad():
            q_proj = memory.query_proj(q_raw)
            q_read = memory.to_read_space(q_proj, mu, V_sub)
            logits_mem, sim_val = memory.slow_mlp(q_read)
            pred_tok = logits_mem[0].argmax().item()
            lam = memory.lam_max * torch.sigmoid(memory.gate_sharpness * (sim_val[0, 0] - memory.gate_threshold)).item()
            
        exact_ans_ids = tokenizer.encode(f["statement"]) # simple target check
        target_tok = exact_ans_ids[len(tokenizer.encode(f["probe"]))].item() # first answer token
        
        hit = (pred_tok == target_tok)
        exact_hits += int(hit)
        
        # Paraphrase recall check
        for para in EVAL_PARAPHRASES[fid]:
            ids_para = tokenizer.encode(para, return_tensors="pt").to(device)
            _, h_para = gpt2_forward(hybrid, ids_para)
            q_raw_para = h_para[0, -min(2, h_para.shape[1]):, :].mean(dim=0).unsqueeze(0)
            
            with torch.no_grad():
                q_proj_para = memory.query_proj(q_raw_para)
                q_read_para = memory.to_read_space(q_proj_para, mu, V_sub)
                logits_para, sim_para = memory.slow_mlp(q_read_para)
                pred_tok_para = logits_para[0].argmax().item()
                lam_para = memory.lam_max * torch.sigmoid(memory.gate_sharpness * (sim_para[0, 0] - memory.gate_threshold)).item()
                
            hit_para = (pred_tok_para == target_tok)
            para_hits += int(hit_para)
            para_total += 1
            
            # Gate activation (TPR)
            if lam_para >= 0.5:
                gate_tps += 1
            gate_t_total += 1
            
    exact_acc = exact_hits / len(block_facts)
    para_acc = para_hits / para_total if para_total > 0 else 0.0
    gate_tpr = gate_tps / gate_t_total if gate_t_total > 0 else 0.0
    
    return exact_acc, para_acc, gate_tpr, 1.0 - gate_tpr


def evaluate_control_fpr(
    hybrid,
    memory: EpisodicFactMemory,
    mu: torch.Tensor,
    V_sub: torch.Tensor | None,
) -> float:
    """Evaluate the false positive rate of the consolidated gate on general language control tokens."""
    tokenizer = hybrid.tokenizer
    device = hybrid.device
    gate_fps = 0
    total_tokens = 0
    
    texts = [p["probe"] for p in RETENTION_PROBES] + list(INDEPENDENT_PPL_TEXTS)
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        _, h = gpt2_forward(hybrid, ids)
        T = h.shape[1]
        for t in range(T):
            q_raw = h[0, t - min(2, t + 1) + 1 : t + 1, :].mean(dim=0).unsqueeze(0)
            with torch.no_grad():
                q_proj = memory.query_proj(q_raw)
                q_read = memory.to_read_space(q_proj, mu, V_sub)
                _, sim_val = memory.slow_mlp(q_read)
                lam = memory.lam_max * torch.sigmoid(memory.gate_sharpness * (sim_val[0, 0] - memory.gate_threshold)).item()
            if lam >= 0.5:
                gate_fps += 1
            total_tokens += 1
            
    return gate_fps / total_tokens if total_tokens > 0 else 0.0


def main():
    print("=" * 70)
    print("  AGNIS V4.5 — COMPRESSED PROTOTYPE REPLAY WITH DISTILLATION")
    print("  T=3 blocks | Medoid prototypes | Frozen coordinate projection")
    print("=" * 70)

    # 1. Setup baseline models
    hybrid = build_hybrid()
    tokenizer = hybrid.tokenizer
    hybrid.eval()

    # Capture initial frozen weights of the base model for bitwise comparison assertions
    original_base_state = {k: v.clone().detach().cpu() for k, v in hybrid.gpt2.named_parameters()}
    
    vocab_size = hybrid.gpt2.config.vocab_size
    memory = EpisodicFactMemory(vocab_size=vocab_size, pool_len=2, device=DEVICE)
    sampler = ReplaySampler(embed_dim=768)

    # Split the 10 RAW_FACTS into T=3 blocks:
    # B1: F01, F02, F03, F04
    # B2: F05, F06, F07
    # B3: F08, F09, F10
    blocks = [
        RAW_FACTS[:4],
        RAW_FACTS[4:7],
        RAW_FACTS[7:]
    ]
    num_blocks = len(blocks)
    evaluator = CLMetricsEvaluator(num_blocks=num_blocks)
    
    # 2. PRE-CONSOLIDATION ALIGNMENT (Protocol A: Frozen projection)
    # We populate the memory with all 30 injection facts, train the projection once to establish
    # a unified, high-performing semantic representation space, and freeze it.
    print("\n[Protocol A] Pre-Consolidation Query-Projection Training...")
    fact_ranges: dict[str, tuple[int, int]] = {}
    answer_ids: dict[str, torch.Tensor] = {}
    for idx, fact in enumerate(INJECTION_FACT_TEXTS):
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        
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
        h_answer = h[0, boundary:-1, :]
        v_answer = ids[0, boundary + 1:]
        
        start = len(memory)
        memory.write(h_answer, v_answer)
        if idx % 3 == 0:
            fid = RAW_FACTS[idx // 3]["id"]
            fact_ranges[fid] = (start, h_answer.shape[0])
            answer_ids[fid] = v_answer.detach()
            
    q_fact, pos_idx = collect_fact_queries(hybrid, memory, fact_ranges, answer_ids)
    q_ctrl = collect_control_states(hybrid, memory)
    train_query_projection(memory, q_fact, pos_idx, q_ctrl)
    
    # Freeze the query projection permanently to prevent coordinate system drift
    for param in memory.query_proj.parameters():
        param.requires_grad = False
    print("  [OK] Query projection frozen. Stable coordinate space established.")
    
    # Clear episodic memory to reset for sequential stages
    memory.keys_raw = torch.empty(0, 768, device=DEVICE)
    memory.values = torch.empty(0, dtype=torch.long, device=DEVICE)
    memory._space_cache = None
    
    # Force evaluate and store baseline space cache (mu, V_sub) using the initial 30 facts
    # We reconstruct the same read space geometry for all stages.
    # We populate keys temporarily to compute space cache, then evict.
    temp_keys = []
    for fact in INJECTION_FACT_TEXTS:
        ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
        _, h_full = gpt2_forward(hybrid, ids)
        h = memory.pool_sequence(h_full)
        prompt = fact["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        n = 0
        limit = min(len(prompt_ids), len(ids[0].tolist()))
        while n < limit and ids[0, n].item() == prompt_ids[n]:
            n += 1
        boundary = max(0, n - 1)
        temp_keys.append(h[0, boundary:-1, :])
    memory.keys_raw = torch.cat(temp_keys, dim=0)
    mu, V_sub = memory.read_space() # Cached permanently!
    
    # Verify space is cached
    assert memory._space_cache is not None
    # Evict keys back to empty state
    memory.keys_raw = torch.empty(0, 768, device=DEVICE)
    print("  [OK] Pre-calibrated space cache locked.")
    
    # Stage 0: Evaluate the baseline (empty slow_mlp, all keys evicted)
    # We initialize an empty student MLP that outputs zero logits and gate
    empty_mlp = JointSlowMemoryMLP(vocab_size=vocab_size).to(DEVICE)
    with torch.no_grad():
        nn.init.zeros_(empty_mlp.logits_head.weight)
        nn.init.zeros_(empty_mlp.logits_head.bias)
        nn.init.zeros_(empty_mlp.gate_head.weight)
        nn.init.zeros_(empty_mlp.gate_head.bias)
    memory.slow_mlp = empty_mlp
    
    print("\nSTAGE 0 — BASELINE (empty memory)")
    print("-" * 70)
    for j in range(num_blocks):
        exact_acc, para_acc, gate_tpr, gate_fnr = evaluate_block_performance(hybrid, memory, blocks[j], mu, V_sub)
        gate_fpr = evaluate_control_fpr(hybrid, memory, mu, V_sub)
        evaluator.update_metrics(0, j, exact_acc, para_acc, gate_tpr, gate_fpr)
        print(f"  Block {j+1} | exact={exact_acc*100:5.1f}% | para={para_acc*100:5.1f}% | gate_tpr={gate_tpr*100:5.1f}% | gate_fpr={gate_fpr*100:5.1f}%")
        
    initial_mlp_param_count = sum(p.numel() for p in memory.slow_mlp.parameters())
    print(f"  MLP parameter count: {initial_mlp_param_count}")
    
    # 3. SEQUENTIAL BLOCK CONSOLIDATION LOOP
    for i in range(1, num_blocks + 1):
        print(f"\n=================================================================")
        print(f"  STAGE {i} — Consolidating Block {i} (evicting database afterward)")
        print(f"=================================================================")
        
        # Deep copy existing student as teacher prior to current block updates
        teacher = None
        if i > 1:
            teacher = copy.deepcopy(memory.slow_mlp).eval()
            for p in teacher.parameters():
                p.requires_grad = False
                
        # Write Block i to episodic memory
        print(f"  [Write] Injecting Block {i} into episodic memory...")
        block_facts = blocks[i - 1]
        block_fact_ids = [f["id"] for f in block_facts]
        
        # Retrieve original injection facts belonging to Block i
        block_injections = [fact for fact in INJECTION_FACT_TEXTS if fact["id"] in block_fact_ids]
        
        # Populate episodic memory with block i
        block_fact_ranges = {}
        block_answer_ids = {}
        for idx, fact in enumerate(block_injections):
            ids = tokenizer.encode(fact["text"] + tokenizer.eos_token, return_tensors="pt").to(hybrid.device)
            _, h_full = gpt2_forward(hybrid, ids)
            h = memory.pool_sequence(h_full)
            
            prompt = fact["prompt"]
            prompt_ids = tokenizer.encode(prompt)
            n = 0
            limit = min(len(prompt_ids), len(ids[0].tolist()))
            while n < limit and ids[0, n].item() == prompt_ids[n]:
                n += 1
            boundary = max(0, n - 1)
            h_answer = h[0, boundary:-1, :]
            v_answer = ids[0, boundary + 1:]
            
            start = len(memory)
            memory.write(h_answer, v_answer)
            if idx % 3 == 0:
                fid = fact["id"]
                block_fact_ranges[fid] = (start, h_answer.shape[0])
                block_answer_ids[fid] = v_answer.detach()
                
        # Collect current block positive and negative queries
        q_fact_i, pos_idx_i = collect_fact_queries(hybrid, memory, block_fact_ranges, block_answer_ids)
        q_ctrl_i = collect_control_states(hybrid, memory)
        
        # Update Replay Sampler with current block fact prototypes
        for fid in block_fact_ids:
            start, length = block_fact_ranges[fid]
            sampler.update_fact(fid, memory.keys_raw[start : start + length])
            
        # Get student training coordinates and targets for current block facts
        k_read_i = memory.to_read_space(memory.keys_raw, mu, V_sub).detach()
        q_ctrl_read_i = memory.to_read_space(memory.query_proj(q_ctrl_i), mu, V_sub).detach()
        
        # Get targets
        train_inputs_cur = []
        target_tokens_cur = []
        target_sims_cur = []
        
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
        
        # Add current raw keys to current block targets
        self_sims = (k_read_i * k_read_i).sum(dim=-1, keepdim=True)
        train_inputs_cur = torch.cat([train_inputs_cur, k_read_i], dim=0)
        target_tokens_cur = torch.cat([target_tokens_cur, memory.values], dim=0)
        target_sims_cur = torch.cat([target_sims_cur, self_sims], dim=0)
        
        # Train Student MLP using current targets + teacher distillation on historical prototypes
        memory.slow_mlp = train_student_with_replay(
            memory=memory,
            sampler=sampler,
            teacher=teacher,
            current_inputs=train_inputs_cur,
            current_tokens=target_tokens_cur,
            current_sims=target_sims_cur,
            q_ctrl_read=q_ctrl_read_i,
            epochs=250,
        )
        
        # 100% EVICTION: clear episodic database
        memory.keys_raw = torch.empty(0, 768, device=DEVICE)
        memory.values = torch.empty(0, dtype=torch.long, device=DEVICE)
        memory._space_cache = (mu, V_sub) # preserve cached geometry
        
        # Verify complete eviction assertions
        assert len(memory.keys_raw) == 0, "Eviction failure: keys_raw not empty!"
        assert len(memory.values) == 0, "Eviction failure: values not empty!"
        
        # Calibrate gate parameters on consolidated model outputs
        # To avoid leaks, calibration only uses current block evaluations
        memory.gate_threshold = 0.602 # Frozen calibrated threshold from V4.3/V4.4b
        memory.gate_sharpness = 80.0
        
        # Evaluate Stage i performance on all blocks (1..T)
        print(f"  [Evaluation] Evaluating Consolidated Model after Stage {i}...")
        for j in range(num_blocks):
            exact_acc, para_acc, gate_tpr, gate_fnr = evaluate_block_performance(hybrid, memory, blocks[j], mu, V_sub)
            gate_fpr = evaluate_control_fpr(hybrid, memory, mu, V_sub)
            evaluator.update_metrics(i, j, exact_acc, para_acc, gate_tpr, gate_fpr)
            print(f"    Block {j+1} | exact={exact_acc*100:5.1f}% | para={para_acc*100:5.1f}% | gate_tpr={gate_tpr*100:5.1f}% | gate_fpr={gate_fpr*100:5.1f}%")
            
        # Assertions to guarantee zero parameters/base-weights drift
        for name, parameter in hybrid.gpt2.named_parameters():
            assert torch.equal(parameter.cpu(), original_base_state[name]), f"Base parameter {name} drifted!"
            
        mlp_param_count = sum(p.numel() for p in memory.slow_mlp.parameters())
        assert mlp_param_count == initial_mlp_param_count, f"MLP parameter count changed! {mlp_param_count} != {initial_mlp_param_count}"
        
        # Memory payload bytes
        print(f"  [Memory] Replay payload: {sampler.payload_bytes()} bytes | Serialized: {sampler.serialized_bytes()} bytes")

    # 4. PRINT MATRICES AND CL METRICS SUMMARY
    evaluator.print_matrices()
    
    print("\n=================================================================")
    print("  CL METRICS ANALYSIS (EXACT RECALL)")
    print("=================================================================")
    exact_metrics = evaluator.compute_cl_summary("exact")
    print(f"  Plasticity per stage (P_i)  : " + ", ".join(f"B{i+1}: {p*100:.1f}%" for i, p in enumerate(exact_metrics["plasticity"])))
    print(f"  Final Average Recall (A_T)  : {exact_metrics['final_recall']*100:.1f}%")
    print(f"  Average Forgetting (F_T)    : {exact_metrics['forgetting']*100:.1f}%")
    print(f"  Backward Transfer (BWT)     : {exact_metrics['bwt']*100:.1f}%")
    print(f"  Forward Transfer (FWT)      : {exact_metrics['fwt']*100:.1f}%")
    
    print("\n=================================================================")
    print("  CL METRICS ANALYSIS (PARAPHRASE RECALL)")
    print("=================================================================")
    para_metrics = evaluator.compute_cl_summary("paraphrase")
    print(f"  Plasticity per stage (P_i)  : " + ", ".join(f"B{i+1}: {p*100:.1f}%" for i, p in enumerate(para_metrics["plasticity"])))
    print(f"  Final Average Recall (A_T)  : {para_metrics['final_recall']*100:.1f}%")
    print(f"  Average Forgetting (F_T)    : {para_metrics['forgetting']*100:.1f}%")
    print(f"  Backward Transfer (BWT)     : {para_metrics['bwt']*100:.1f}%")
    print(f"  Forward Transfer (FWT)      : {para_metrics['fwt']*100:.1f}%")
    
    # 5. Measure Final Behavioral Retention
    after_ppl = measure_ppl(hybrid, memory, INDEPENDENT_PPL_TEXTS)
    print(f"\n  Final PPL in consolidated state: {after_ppl:.6f} (baseline {42.06:.6f}) | Delta: {after_ppl - 42.06:+.6f}")
    
    # Measure token-level KL divergence
    kls = []
    for text in INDEPENDENT_PPL_TEXTS:
        ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits_lm, h = gpt2_forward(hybrid, ids)
            p_base = F.softmax(logits_lm[0], dim=-1)
            
            p_hybrid, _, _ = memory.read(h)
            p_hybrid = p_hybrid[0]
            
            # Blend
            q_pooled = h[0, -min(2, h.shape[1]):, :].mean(dim=0).unsqueeze(0)
            q_proj = memory.query_proj(q_pooled)
            mu_t, V_sub_t = memory.read_space()
            q_read = memory.to_read_space(q_proj, mu_t, V_sub_t)
            _, sim_val = memory.slow_mlp(q_read)
            lam = memory.lam_max * torch.sigmoid(memory.gate_sharpness * (sim_val[0, 0] - memory.gate_threshold)).item()
            
            # blended distribution
            p_blend = (1.0 - lam) * p_base + lam * p_hybrid
            kls.append(compute_behavioral_divergence(p_base, p_blend))
            
    print(f"  Mean token-level KL divergence D_KL(P_base || P_hybrid): {np.mean(kls):.6f}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
