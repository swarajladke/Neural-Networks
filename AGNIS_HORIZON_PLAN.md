# AGNIS CLS Horizons Roadmap (July 2026)

Following the success of the **V4.3 Memory Capacity Sweep** (proving 95% paraphrase recall and +0.00 PPL change under 1,000 distractor load), this document outlines the roadmap for the three next capability horizons of the AGNIS Hybrid architecture.

---

## Horizon A: Memory Consolidation (Sleep Phase) — ACTIVE

The ultimate goal of Complementary Learning Systems (CLS) is not just to store facts in a fast episodic buffer (hippocampus), but to consolidate them back into the slow statistical model weights (neocortex/Hebbian core) so they become permanent.

### Staged Experiments:
1. **Replay-Based Distillation (Base):**
   - Sample stored key-value facts from `EpisodicFactMemory`.
   - Generate paraphrase variations using `ResidualQueryProjection`.
   - Fine-tune the slow recurrent core weights on these generated variations using a low learning rate.
   - **Validation:** Ablate/disable the KV episodic memory completely and measure post-consolidation recall using the slow core alone.
2. **Consolidation-then-Eviction:**
   - Once a fact is successfully consolidated into the slow core weights, evict/delete its key-value pairs from the episodic store.
   - Re-run the capacity sweep.
   - **Success Criteria:** Factual recall is retained without lookup latency cost, and PPL remains stable.
3. **Interference Audit:**
   - Monitor the 40% base retention score. Ensure that writing consolidated facts into the weights does not cause catastrophic forgetting of the pre-trained baseline text.

---

## Horizon B: Multi-Hop Reasoning

Once consolidation is robust, we can model complex multi-hop dependencies:

### Staged Experiments:
1. **2-Hop Chaining:**
   - Inject Fact A $\rightarrow$ B (e.g. "Auranium has atomic number 137") and Fact B $\rightarrow$ C (e.g. "Element 137 emits violet light").
   - Query the system on A $\rightarrow$ C (e.g. "What color light does Auranium emit?").
2. **Consolidation Acceleration:**
   - Leverage Horizon A: once Fact A $\rightarrow$ B is consolidated into the weights, the model can generate Hop-1 directly from weights and feed it to the prompt to trigger Hop-2 retrieval from the episodic memory.

---

## Horizon C: Recurrent Core Generalization (Delayed Parity)

A scoped, time-boxed parallel research thread to resolve the capability ceiling of the Hebbian core:

### Staged Experiments:
1. **Eligibility Traces:**
   - Implement three-factor Hebbian update rules to bridge temporal delays, allowing sequential credit assignment over long periods.
2. **Episodic Scaffolding:**
   - Let the fast episodic store act as a helper for the parity task (storing intermediate parity states during the sequence and reading them at the end).
