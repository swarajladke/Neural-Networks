"""
agnis_contamination_metrics.py — Cross-Fact Answer-Token Intrusion Tracker (V4.5)
==================================================================================
Implements Sol's recommended contamination-specific metrics:

  C_{i,j}  = P(wrong answer belongs to another injected fact | error)
  m(x)     = logit_correct(x) - max_{f'≠f} logit_answer(f')(x)

Tracks:
  - cross-fact contamination count and source block
  - target-token rank drift from acquisition to final stage
  - target vs. contaminator logit margin (positive = safe, negative = contaminated)
  - paraphrase-coordinate nearest prototype fact
  - replay coverage distance (query to closest prototype)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class ParaphraseRecord:
    """Full diagnostic record for one paraphrase at one stage."""
    fid: str                            # fact id
    block: int                          # block this fact belongs to
    stage: int                          # evaluation stage
    para_text: str                      # paraphrase surface form (truncated)
    target_tok: int                     # correct answer token id
    target_tok_str: str                 # decoded correct answer
    pred_tok: int                       # predicted token id
    pred_tok_str: str                   # decoded prediction
    correct: bool                       # prediction == target

    target_logit: float = 0.0          # raw logit for target token
    pred_logit: float = 0.0            # raw logit for predicted token
    target_rank: int = 0               # rank of target in softmax (1=best)
    target_prob: float = 0.0           # softmax probability of target

    gate_prob: float = 0.0             # gate activation probability
    gate_active: bool = False          # gate >= 0.5

    raw_sim: float = 0.0               # raw cosine similarity from MLP
    max_proto_sim: float = 0.0         # max cosine sim to any prototype

    # Cross-fact contamination fields
    contaminating_fid: str | None = None     # which fact's answer was predicted
    contaminating_block: int | None = None   # that fact's block
    logit_margin: float = 0.0               # logit_correct - max_other_fact_logit
    is_contaminated: bool = False           # logit_margin < 0 AND pred != target


@dataclass
class ContaminationReport:
    """Aggregated report across all facts and paraphrases for one stage."""
    stage: int
    records: list[ParaphraseRecord] = field(default_factory=list)

    def add(self, rec: ParaphraseRecord):
        self.records.append(rec)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    def failures(self):
        return [r for r in self.records if not r.correct]

    def contaminated_failures(self):
        return [r for r in self.failures() if r.is_contaminated]

    def contamination_rate(self) -> float:
        """P(wrong answer belongs to another fact | error)"""
        f = self.failures()
        if not f:
            return float("nan")
        c = self.contaminated_failures()
        return len(c) / len(f)

    def mean_logit_margin(self) -> float:
        if not self.records:
            return float("nan")
        return sum(r.logit_margin for r in self.records) / len(self.records)

    def negative_margin_count(self) -> int:
        """How many paraphrases have logit_margin < 0 (contamination risk)."""
        return sum(1 for r in self.records if r.logit_margin < 0)

    def print_report(self):
        total = len(self.records)
        n_fail = len(self.failures())
        n_cont = len(self.contaminated_failures())

        print(f"\n{'─'*70}")
        print(f"  CONTAMINATION REPORT — Stage {self.stage}")
        print(f"{'─'*70}")
        print(f"  Total paraphrases evaluated : {total}")
        print(f"  Failures (pred ≠ target)    : {n_fail}")
        print(f"  Contaminated failures        : {n_cont}  "
              f"(C = {self.contamination_rate():.1%})")
        print(f"  Mean logit margin m(x)       : {self.mean_logit_margin():+.4f}")
        print(f"  Paraphrases with margin < 0  : {self.negative_margin_count()} / {total}")

        if self.contaminated_failures():
            print(f"\n  Contaminated failures:")
            for r in self.contaminated_failures():
                print(f"    [{r.fid}] \"{r.para_text}\"")
                print(f"      target='{r.target_tok_str}'  pred='{r.pred_tok_str}'")
                print(f"      contam from: {r.contaminating_fid} (Block {r.contaminating_block})")
                print(f"      logit margin: {r.logit_margin:+.4f}  rank: {r.target_rank}")


def build_answer_token_map(blocks: list[list[dict]], tokenizer) -> dict[int, tuple[str, int]]:
    """
    For every injected fact, find its first answer token.
    Returns: {token_id -> (fact_id, block_number)}
    """
    answer_map: dict[int, tuple[str, int]] = {}
    for blk_idx, block in enumerate(blocks, start=1):
        for f in block:
            stmt_ids = tokenizer.encode(f["statement"])
            probe_ids = tokenizer.encode(f["probe"])
            if len(stmt_ids) > len(probe_ids):
                ans_tok = stmt_ids[len(probe_ids)]
                answer_map[ans_tok] = (f["id"], blk_idx)
    return answer_map


def evaluate_with_contamination(
    hybrid,
    memory,
    mu: torch.Tensor,
    V_sub: torch.Tensor | None,
    block_facts: list[dict],
    block_num: int,
    stage: int,
    sampler,
    answer_map: dict[int, tuple[str, int]],
    gpt2_forward_fn,
    EVAL_PARAPHRASES: dict,
) -> ContaminationReport:
    """
    Run evaluation on all paraphrases of all facts in block_facts,
    recording full contamination diagnostics.
    """
    tokenizer = hybrid.tokenizer
    device = hybrid.device
    report = ContaminationReport(stage=stage)

    for f in block_facts:
        fid = f["id"]
        stmt_ids = tokenizer.encode(f["statement"])
        probe_ids = tokenizer.encode(f["probe"])
        target_tok = stmt_ids[len(probe_ids)] if len(stmt_ids) > len(probe_ids) else stmt_ids[-1]
        target_tok_str = tokenizer.decode([target_tok])

        for para in EVAL_PARAPHRASES[fid]:
            ids = tokenizer.encode(para, return_tensors="pt").to(device)
            with torch.no_grad():
                _, h = gpt2_forward_fn(hybrid, ids)
                q_raw = h[0, -min(2, h.shape[1]):, :].mean(dim=0).unsqueeze(0)
                q_proj = memory.query_proj(q_raw)
                q_read = memory.to_read_space(q_proj, mu, V_sub)

                logits, sim_val = memory.slow_mlp(q_read)
                logits_1d = logits[0]  # (vocab_size,)

                raw_sim = sim_val[0, 0].item()
                gate_prob = torch.sigmoid(
                    memory.gate_sharpness * (sim_val[0, 0] - memory.gate_threshold)
                ).item()
                gate_active = gate_prob >= 0.5

                probs = F.softmax(logits_1d, dim=-1)
                target_prob = probs[target_tok].item()
                target_logit = logits_1d[target_tok].item()
                target_rank = int((probs > probs[target_tok]).sum().item()) + 1

                pred_tok = logits_1d.argmax().item()
                pred_logit = logits_1d[pred_tok].item()
                pred_tok_str = tokenizer.decode([pred_tok])
                correct = (pred_tok == target_tok)

                # Cross-fact contamination: compute margin against all other facts' answer tokens
                other_fact_logits = {
                    tok_id: logits_1d[tok_id].item()
                    for tok_id, (other_fid, _) in answer_map.items()
                    if other_fid != fid
                }
                if other_fact_logits:
                    max_other_logit = max(other_fact_logits.values())
                    max_other_tok = max(other_fact_logits, key=other_fact_logits.get)
                    logit_margin = target_logit - max_other_logit
                    contam_fid, contam_block = answer_map.get(max_other_tok, (None, None))
                    is_contaminated = (not correct) and (
                        pred_tok in answer_map and answer_map[pred_tok][0] != fid
                    )
                    contaminating_fid = answer_map[pred_tok][0] if (
                        not correct and pred_tok in answer_map
                    ) else None
                    contaminating_block = answer_map[pred_tok][1] if (
                        not correct and pred_tok in answer_map
                    ) else None
                else:
                    logit_margin = 0.0
                    is_contaminated = False
                    contaminating_fid = None
                    contaminating_block = None

                # Prototype coverage: max cosine sim to any prototype for this fact
                proto_sims = []
                for tag in ["_stmt", "_qa", "_cloze"]:
                    vkey = fid + tag
                    if vkey in sampler.prototypes:
                        proto = sampler.prototypes[vkey].to(device)
                        proto_sims.append(F.cosine_similarity(q_read, proto.unsqueeze(0)).item())
                max_proto_sim = max(proto_sims) if proto_sims else float("nan")

            rec = ParaphraseRecord(
                fid=fid,
                block=block_num,
                stage=stage,
                para_text=para[:55] + ("..." if len(para) > 55 else ""),
                target_tok=target_tok,
                target_tok_str=target_tok_str,
                pred_tok=pred_tok,
                pred_tok_str=pred_tok_str,
                correct=correct,
                target_logit=target_logit,
                pred_logit=pred_logit,
                target_rank=target_rank,
                target_prob=target_prob,
                gate_prob=gate_prob,
                gate_active=gate_active,
                raw_sim=raw_sim,
                max_proto_sim=max_proto_sim,
                contaminating_fid=contaminating_fid,
                contaminating_block=contaminating_block,
                logit_margin=logit_margin,
                is_contaminated=is_contaminated,
            )
            report.add(rec)

    return report


def print_margin_drift_table(
    margin_records: dict[tuple[str, str], dict[int, float]],
    acquired_stages: dict[str, int],
) -> None:
    """
    Print logit-margin drift: m(x) from acquisition stage to final stage.
    Negative margin = contamination risk.
    """
    print(f"\n{'─'*72}")
    print("  LOGIT MARGIN DRIFT  m(x) = logit_correct - max_other_fact_logit")
    print(f"{'─'*72}")
    print(f"  {'Paraphrase':<48} {'Acq. m(x)':>9}  {'Final m(x)':>10}  {'Δ':>8}")
    print("  " + "─" * 68)

    drifts = []
    for (fid, para_key), stage_margins in sorted(margin_records.items()):
        acq_stage = acquired_stages.get(fid, 1)
        stages = sorted(stage_margins.keys())
        if not stages:
            continue
        final_stage = stages[-1]
        if acq_stage in stage_margins and final_stage in stage_margins:
            m_acq = stage_margins[acq_stage]
            m_final = stage_margins[final_stage]
            delta = m_final - m_acq
            drifts.append(delta)
            flag = " ← CONTAMINATED" if m_final < 0 else (" ← RISK" if m_final < 1.0 else "")
            print(f"  {para_key:<48} {m_acq:+9.3f}  {m_final:+10.3f}  {delta:+8.3f}{flag}")

    if drifts:
        import statistics
        print(f"\n  Mean Δm : {sum(drifts)/len(drifts):+.4f}"
              f"  |  Std Δm : {statistics.stdev(drifts) if len(drifts) > 1 else 0:.4f}"
              f"  |  Fraction final m < 0 : "
              f"{sum(1 for r in margin_records.values() if min(r.values(), default=0) < 0) / len(margin_records) * 100:.1f}%")
