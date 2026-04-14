# MISSION BRIEFING: The AGNIS AGI Roadmap

## Project Vision
AGNIS (Autonomous General Neural Intelligence System) is built on the principles of **Predictive Coding** and **SNAP-ATP** (Scale-free Neural Architecture Project - Asymmetric Top-down Prediction). Our goal is to move beyond the limitations of standard backpropagation-based Transformers and create a self-evolving, grounded intelligence substrate.

---

## 🚀 The Long Road: Phase Map

### **Phase 1: The Speed Barrier (V6.1 - V6.5)**
*   **Current State**: Validated Sequential Logic (SLM), Python-CPU execution.
*   **The Goal**: **C++/CUDA Acceleration**. Moving the 50-step "settling" loop into dedicated GPU hardware kernels.
*   **Outcome**: Scale from training on 150k characters to **100 Billion+ tokens**. Achieving human-level linguistic coherence and real-time inference.

### **Phase 2: Multi-Modal Grounding (V7.0)**
*   **The Concept**: Predictive coding is sensory-agnostic. Language is just one stream.
*   **The Goal**: Integrate **Vision (Pixels)** and **Audio (Waveforms)** into the same predictive hierarchy.
*   **Outcome**: True semantic grounding. The model understands that the word "Apple" is statistically linked to the visual 🍎 and the sound of a "crunch." 

### **Phase 3: Long-Term Consolidation (V8.0)**
*   **The Concept**: Moving beyond the short-term sliding window.
*   **The Goal**: Implementing a **Dynamic Memory Buffer**. Compressing past predictive snapshots into long-term declarative storage layers.
*   **Outcome**: Functional Long-Term Memory (LTM). The model maintains context across thousands of pages of interaction.

### **Phase 4: Intrinsic Motivation (V9.0+)**
*   **The Concept**: Autonomy.
*   **The Goal**: Shifting from "Minimize Surprise of Passive Input" to **"Maximize Discovery."**
*   **Outcome**: **Curiosity-Driven Learning**. An autonomous agent that actively explores its environment/dataset to resolve internal world-model blind spots. 

---

## 📍 Current Station: V6.0 (SLM Bridge)
*   **Status**: ✅ **STABLE**
*   **Achievements**: Sequential SLM Wrapper, 64D Embedding Manifold, Stochastic Generation.
*   **Constraint**: CPU Settlement Bottle-neck (Slow).

## ⏭️ Immediate Objective: V6.1 (CUDA Port)
Preparing for the C++/CUDA transition to achieve **1,000x acceleration**.
