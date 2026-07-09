# AGNIS V4.3 — CLS Memory Capacity Curve

## Goal

Stress-test the V4.2b episodic memory + fuzzy query projection under
distractor load. Everything proven so far is at n=10 facts (~300 keys).
The CLS claim — fast episodic path with zero interference — lives or dies
on what happens at 1,000+ facts.

**Deliverable:** the capacity curve — recall, gate leakage, and PPL as a
function of memory size. If anchor metrics hold at 1k distractors, the
architecture scales. If they degrade gracefully, we have a characterized
capacity limit (still publishable).

## Protocol

- **Anchor set**: the 10 `RAW_FACTS` (30 injection texts) with the full
  V4.2b machinery — written first. Stage +0 is the V4.2b reproduction.
- **Distractors**: wikitext-103 spans synthesized into
  `(prompt -> continuation)` facts (14–40 tokens, 60% prompt cut),
  injected in stages: **+0, +100, +300, +1000**.
- **Per stage**: retrain the query projection from scratch (comparability),
  recalibrate the gate, then measure everything.

## Metrics per stage

| Metric | What it tells us |
|---|---|
| Anchor exact recall (10) | zero-interference claim |
| Anchor held-out paraphrase recall (20) | does the projection survive a denser key field |
| Wiki distractor recall (sampled 100, first-3-token match) | raw episodic capacity |
| Retention (10) + PPL | gate silence at scale (must stay flat) |
| Fact/control margin (min anchor-para sim vs max control sim) | early warning: collapse ⇒ move to per-cluster local thresholds |
| PCA subspace overlap vs stage-0 `V_sub` | anisotropy-correction drift as keys accumulate |
| `read_space()` / read wall time | engineering headroom (SVD cache + `pca_lowrank` added in V4.3) |

## Success criteria

- Anchor recall and PPL **flat** across all stages.
- Wiki recall ≥ 95% at 1,000 facts, or a smooth characterized curve.
- Margin logged at every stage; if it collapses, the V4.4 direction is
  local (per-fact or per-cluster) gate thresholds, not global tuning.

## Run

```bash
python agnis_scaling_v4_3.py
```

Results JSON → `/kaggle/working/agnis_scaling_v4_3_results.json`
(one record per stage, ready for the capacity-curve plots).
Requires `datasets` (wikitext-103) — enable internet on the Kaggle kernel.

## Known risks

1. **Margin compression**: at 1k facts nearest-neighbor distances shrink;
   the control hinge and paraphrase attraction are already in tension
   (the two V4.2b misses were short OOD QA shapes at sim 0.33–0.40).
2. **PCA drift**: top-5 PCs of 30k wiki-dominated keys mean something
   different than for 300 anchor keys; watch whether projection-out starts
   eating fact-discriminative directions.
3. **Projection capacity**: one 400k-param MLP aligning 10 facts is
   memorization; whether the mechanism generalizes is exactly what stage
   +1000 measures.
4. **Retention baseline**: still an unexplained 4/10; the harness logs the
   per-probe breakdown at every stage so a silent drop to 3/10 is visible.
