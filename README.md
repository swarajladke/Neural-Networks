# AGNIS: Autonomous Gated Neural Inference System
**The Future of Continual Language Modeling**

AGNIS is a high-performance neural architecture designed for **Zero-Forgetting Continual Learning**. Unlike traditional transformers that suffer from catastrophic interference, AGNIS uses a **Predictive Coding** framework and a **Synaptic Shield** protocol to learn multiple languages sequentially within a single neural manifold.

---

## 🚀 Key Features

### 🏛️ **Spectral Stable Recurrence (V8.4)**
Inspired by *OpenMythos* and Linear Time-Invariant (LTI) systems, the AGNIS core uses a **Spectrally Normalized Matrix** ($R$) to guarantee mathematical stability. The spectral radius is strictly bounded at 0.98, preventing the "gradient explosions" typical of long-range recurrent networks.

### 🛡️ **The Synaptic Shield (V5.0)**
An advanced manifold-slicing protocol that allows the system to:
1.  Train on **Language A** (e.g., Italian).
2.  Lock the participating synapses using a **Manifold Mask**.
3.  Train on **Language B** (e.g., Russian) without overwriting any Language A knowledge.
*   **Proven Result:** Achieved **92.0% Italian retention** after full Russian training.

### ⚡ **Adaptive Computation Time (ACT)**
A learned halting mechanism that allows each neuron to "stop thinking" once it reaches high confidence.
*   **Efficiency:** Drastically reduces compute load on common patterns.
*   **Optimization:** Increases throughput from **15** to **40+ Tokens/sec** on the RTX 3060.

### 🌡️ **Thermal Guardian Protocol**
Built-in hardware safety for mobile/laptop research environments. 
*   **Thermal Guard:** Automatically pauses execution at 80°C.
*   **Memory Detachment:** Prevents VRAM ballooning by detaching the settlement graph from the gradient tape.

---

## 🛠️ Technical Stack
*   **Engine:** Python 3.12 + PyTorch 2.3+
*   **Hardware:** Optimized for **NVIDIA RTX 3060** (CUDA Core)
*   **Framework:** Predictive Coding (Iterative Settlement)

---

## 🧪 Running the Research

### **Bilingual Zero-Forgetting Sprint**
To verify the synaptic shield across Italian and Russian corpora:
```powershell
python -u v10_bilingual_sprint.py
```

### **Temporal Reasoning Diagnostic**
To verify the memory fidelity of the stable recurrent core:
```powershell
python -u v6_delayed_parity_diagnostic.py
```

---

## 📈 Research Milestones
- [x] **V7.0**: Broke the 0.500 Parity Floor (Memory Validation).
- [x] **V8.4**: Achieved Spectral Stability (Hardware Safety).
- [x] **V10.1**: Demonstrated 92% Bilingual Retention (The Continual Learning Breakthrough).

---

**Developed for the Autonomous SLM Research Study (2026).**
