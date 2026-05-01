import torch
import random
import math
import heapq
import time
import subprocess
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from agnis_v4_core import PredictiveHierarchy

@dataclass
class Experience:
    task_id: int
    x: torch.Tensor
    y: torch.Tensor
    surprise: float
    salience: float
    step: int

class SalienceEngine:
    """Computes salience weights from surprise and learning progress."""
    def __init__(
        self,
        ema: float = 0.95,
        min_weight: float = 0.2,
        max_weight: float = 5.0,  # Allow highly salient experiences to boost learning 5x
        surprise_scale: float = 2.0,
        progress_scale: float = 1.0
    ):
        self.ema = ema
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.surprise_scale = surprise_scale
        self.progress_scale = progress_scale
        self.loss_ema: Optional[float] = None

    def compute(self, loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized salience computation for a batch of losses."""
        surprise = loss
        
        # Scalar fallback for EMA tracking
        avg_loss = loss.mean().item()
        if self.loss_ema is None:
            self.loss_ema = avg_loss
            progress = torch.zeros_like(loss)
        else:
            # Progress is relative to global EMA
            progress = (self.loss_ema - loss).clamp(min=0.0)
            self.loss_ema = self.ema * self.loss_ema + (1 - self.ema) * avg_loss
 
        salience_raw = self.surprise_scale * surprise + self.progress_scale * progress
        weight = 0.5 + salience_raw
        weight = weight.clamp(self.min_weight, self.max_weight)
        return weight, surprise, progress

class SurpriseBuffer:
    """Priority buffer for high-surprise/high-salience experiences."""
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._heap: List[Tuple[float, int, Experience]] = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, exp: Experience):
        score = exp.surprise * exp.salience
        item = (score, self._counter, exp)
        self._counter += 1

        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, item)
            return

        if self._heap and score > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def sample(self, k: int) -> List[Experience]:
        if not self._heap or k <= 0:
            return []
        items = [it[2] for it in self._heap]
        weights = [max(1e-6, it[0]) for it in self._heap]
        k = min(k, len(items))
        return random.choices(items, weights=weights, k=k)

class NeuromodulatorNode:
    """
    V5.2: Global Arousal Node providing selective dopaminergic attention.
    Tracks prediction error ABOUT surprise (not just raw surprise).
    If the network expects calm and gets surprised, dopamine spikes.
    If the network expects noise (high predicted_surprise) and gets it, dopamine stays ~1.0.
    """
    def __init__(self):
        self.predicted_surprise = 0.0
    
    def compute_dopamine(self, actual_surprise: torch.Tensor, update_state: bool = True) -> torch.Tensor:
        """Vectorized dopamine computation for a batch of surprises."""
        avg_surprise = actual_surprise.mean().item()
        
        if self.predicted_surprise <= 1e-6:
            self.predicted_surprise = max(1e-6, avg_surprise)
            return torch.ones_like(actual_surprise)
            
        # V5.2.1: Ratio-based dopamine burst
        rel_error = (actual_surprise - self.predicted_surprise) / self.predicted_surprise
        dopamine = (1.0 + rel_error).clamp(0.1, 5.0)
        
        if update_state:
            alpha = 0.2 if avg_surprise > self.predicted_surprise else 0.02
            self.predicted_surprise = (1 - alpha) * self.predicted_surprise + alpha * avg_surprise
        return dopamine

class HippocampalModule:
    """
    V5.2: Episodic Memory System (Hippocampus).
    Stores "Epiphanies" (high-salience settled latent states).
    Allows zero-shot recall by injecting stored states into the hierarchy,
    bypassing the iterative settling loop for familiar contexts.
    """
    def __init__(self, max_memories: int = 1000, similarity_threshold: float = 0.95):
        self.max_memories = max_memories
        self.threshold = similarity_threshold
        # Stores: {Input_Norm: (Input_Tensor, Latent_States_List)}
        self.memory: List[Dict] = []

    def store(self, x: torch.Tensor, latent_states: List[torch.Tensor]):
        """Stores per-sample snapshots so batched learning remains recall-safe."""
        x_batch = x.detach()
        latent_batch = [l.detach() for l in latent_states]

        batch_size = x_batch.shape[0]
        for sample_idx in range(batch_size):
            snapshot = {
                'x': x_batch[sample_idx:sample_idx + 1].clone(),
                'latents': [l[sample_idx:sample_idx + 1].clone() for l in latent_batch]
            }

            # Simple FIFO for now
            if len(self.memory) >= self.max_memories:
                self.memory.pop(0)
            self.memory.append(snapshot)

    def recall(self, x: torch.Tensor) -> Optional[List[torch.Tensor]]:
        """Searches for a matching context using Cosine Similarity."""
        if not self.memory:
            return None
        
        # Flatten input for comparison
        x_flat = x.view(-1)
        x_norm = torch.norm(x_flat)
        if x_norm < 1e-6: return None
        
        best_sim = -1.0
        best_latents = None
        
        # Search for best match
        # TODO: Vectorize this or use a spatial index (FAISS-like) as memory grows
        for mem in self.memory:
            m_flat = mem['x'].view(-1)
            similarity = torch.dot(x_flat, m_flat) / (x_norm * torch.norm(m_flat) + 1e-8)
            
            if similarity > best_sim:
                best_sim = similarity
                best_latents = mem['latents']
                
        if best_sim >= self.threshold:
            return best_latents
        return None

class ThermalGuardian:
    """
    V5.5: Hardware-Aware Safety Layer (Laptop Shield).
    Monitors GPU temperature and VRAM usage via nvidia-smi.
    Implements adaptive throttling, mandatory pauses, and emergency shutdowns.
    """
    def __init__(self, device: str = "cpu", 
                 caution_temp: int = 70, 
                 pause_temp: int = 78, 
                 emergency_temp: int = 85):
        self.device = device
        self.is_cuda = "cuda" in device
        self.caution_temp = caution_temp
        self.pause_temp = pause_temp
        self.emergency_temp = emergency_temp
        
        # Telemetry state
        self.peak_temp = 0
        self.avg_temp = 0
        self._temp_history = []
        self._check_counter = 0

    def query_telemetry(self) -> Tuple[int, float]:
        """Queries GPU temp (C) and VRAM usage (%) via nvidia-smi."""
        if not self.is_cuda:
            return 30, 0.0 # Standard CPU room temp
            
        try:
            # Query temperature and memory usage
            cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            res = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split(',')
            temp = int(res[0])
            used_mem = int(res[1])
            total_mem = int(res[2])
            vram_pct = (used_mem / total_mem) * 100
            return temp, vram_pct
        except Exception as e:
            # Fallback if nvidia-smi fails
            return 40, 0.0

    def check(self, agent: 'CognitivePredictiveAgent' = None):
        """Perform a safety check and throttle if necessary."""
        self._check_counter += 1
        # Only query every 10 calls to reduce overhead
        if self._check_counter % 10 != 0:
            return
            
        temp, vram = self.query_telemetry()
        self.peak_temp = max(self.peak_temp, temp)
        self._temp_history.append(temp)
        if len(self._temp_history) > 100: self._temp_history.pop(0)
        self.avg_temp = sum(self._temp_history) / len(self._temp_history)

        # 1. VRAM Guardian (Safety Cache Clear)
        if vram > 90.0:
            print(f"!!! [WARNING] VRAM CRITICAL: {vram:.1f}% !!! Clearing Cache...")
            torch.cuda.empty_cache()
            time.sleep(5.0)

        # 2. Thermal Emergency (Shutdown)
        if temp >= self.emergency_temp:
            print(f"\nFATAL: [THERMAL EMERGENCY] GPU at {temp}C! Threshold {self.emergency_temp}C reached.")
            print("Action: Saving emergency checkpoint and shutting down process...")
            if agent:
                agent.save_checkpoint("thermal_emergency_checkpoint.pt")
            sys.exit(1)

        # 3. Thermal Pause (Mandatory Rest)
        if temp >= self.pause_temp:
            print(f"\n[THERMAL PAUSE] GPU at {temp}C. Mandatory 30s rest for cooling...")
            time.sleep(30.0)
            # Re-check after rest
            temp, _ = self.query_telemetry()
            if temp > self.caution_temp:
                 print(f"Still at {temp}C. Extending rest...")
                 time.sleep(15.0)

        # 4. Thermal Caution (Active Throttling)
        elif temp >= self.caution_temp:
            # Micro-sleep to allow fans to catch up
            time.sleep(0.1)
class CognitivePredictiveAgent:
    """
    Wraps the V4 PredictiveHierarchy with V2/V3 Cognitive Ecosystems:
    Salience, Surprise Buffering, and Offline Replay Dreaming.
    """
    MAX_EXPOSURE_ENTRIES = 50000

    def __init__(self, hierarchy: PredictiveHierarchy, device: str = "cpu"):
        self.hierarchy = hierarchy
        self.device = device
        self.salience_engine = SalienceEngine()
        self.buffer = SurpriseBuffer()
        self.neuromodulator = NeuromodulatorNode()
        self.hippocampus = HippocampalModule()
        self.guardian = ThermalGuardian(device=device)
        self.step_count = 0
        
        # --- V5.0: Novelty Decay ---
        self.exposure_counts: Dict[int, int] = {}  # pattern_hash -> count
        self.tau_novelty: float = 10.0  # decay time constant
        
        # --- V5.0: Expert Retention ---
        self.neurogenesis_count = 0  # total slivers recruited
        
        # Save base learning rates
        self.base_lrs = []
        for col in self.hierarchy.layers:
            self.base_lrs.append((col.eta_V, col.eta_W))

        # --- V7.2: Temporal Context Management ---
        self.r_matrix_snapshots: Dict[str, list[torch.Tensor]] = {}
        self.current_context: Optional[str] = None
        
        # --- V7.2: Parametric Neurogenesis ---
        self.dopamine_threshold = 1.1
        self.salience_threshold = 1.1
        self.hypersensitive = False

    def detect_context_shift(self, surprise: float, baseline: float) -> bool:
        """V7.2: Detect sudden surprisal spikes that indicate a domain/language boundary."""
        if baseline <= 0: return False
        return surprise > baseline * 2.0

    def switch_temporal_context(self, context_name: str):
        """V7.2: Swap the hierarchy's recurrent weights to match the detected context."""
        # 1. Save current context
        if self.current_context:
            self.r_matrix_snapshots[self.current_context] = self.hierarchy.snapshot_r_matrices()
        
        # 2. Load or initialize new context
        if context_name in self.r_matrix_snapshots:
            print(f"[RECOVERY] Restoring temporal context: {context_name}")
            self.hierarchy.load_r_matrices(self.r_matrix_snapshots[context_name])
        else:
            print(f"[RECOVERY] Initializing new temporal context: {context_name}")
            self.hierarchy.reset_recurrent_matrices(gain=0.1)
        
        self.current_context = context_name

    def enable_hypersensitive_discovery(self, surprise_baseline: float = None):
        """V7.2: Lower neurogenesis barriers to force rapid expertise recruitment."""
        if surprise_baseline:
            # User's suggestion: threshold = baseline * 1.1
            self.dopamine_threshold = 1.05 # Even more sensitive toggle
            self.salience_threshold = 1.05
        else:
            self.dopamine_threshold = 1.05
            self.salience_threshold = 1.05
        self.hypersensitive = True
        print(">>> HYPERSENSITIVE DISCOVERY ENABLED <<<")

    def disable_hypersensitive_discovery(self):
        """Restore default conservative neurogenesis thresholds."""
        self.dopamine_threshold = 1.5
        self.salience_threshold = 1.5
        self.hypersensitive = False
        print(">>> DISCOVERY STABILIZED (Hypersensitivity Off) <<<")

    def get_dopamine_scale(self, elapsed_seconds, total_seconds):
        progress = elapsed_seconds / total_seconds
        if progress < 0.7:
            return 3.0
        decay = (progress - 0.7) / 0.3
        return 3.0 - (2.0 * decay)

    def _update_exposure(self, pattern_key):
        if len(self.exposure_counts) >= self.MAX_EXPOSURE_ENTRIES and pattern_key not in self.exposure_counts:
            oldest = next(iter(self.exposure_counts))
            del self.exposure_counts[oldest]
        self.exposure_counts[pattern_key] = self.exposure_counts.get(pattern_key, 0) + 1
        return self.exposure_counts[pattern_key]

    def observe_and_learn(self, x: torch.Tensor, y: torch.Tensor, task_id: int = 0, 
                          max_steps: int = 150, recognition_weight: float = 1.0, beta_push: float = 5.0,
                          warm_start: bool = False):
        """Vectorized Batch-Parallel observe and learn."""
        batch_size = x.shape[0]
        
        # --- V5.2: Hippocampal Fast-Path (Serial for now, vectorized training later) ---
        if batch_size == 1:
            with torch.no_grad():
                recall_states = self.hippocampus.recall(x)
                if recall_states:
                    for i, col in enumerate(self.hierarchy.layers):
                        col.x.data.copy_(recall_states[i])
                    pred_y = self.hierarchy.predict_label(x, max_steps=1, update_temporal=False)
                    if pred_y.shape[1] > y.shape[1]: pred_y = pred_y[:, :y.shape[1]]
                    surprise = torch.nn.functional.mse_loss(pred_y, y).item()
                    self.neuromodulator.compute_dopamine(torch.tensor([surprise]))
                    return 1.0, surprise

        # 1. Dry-run for surprise estimation (Vectorized)
        if not warm_start:
            self.hierarchy.reset_states(batch_size=batch_size)
        with torch.no_grad():
            pred_y_tensor = self.hierarchy.predict_label(x, max_steps=max_steps, update_temporal=False)
            if pred_y_tensor.shape[1] > y.shape[1]:
                pred_y_tensor = pred_y_tensor[:, :y.shape[1]]
            
            # Per-sample MSE
            raw_surprise = torch.mean((pred_y_tensor - y)**2, dim=1)
            
        # --- V5.0: Novelty Decay (Vectorized batch hashing) ---
        # We process novelty for each sample in the batch
        effective_surprise = torch.zeros_like(raw_surprise)
        for i in range(batch_size):
            quantized = (x[i].flatten() * 100).int()
            pattern_key = hash(quantized.cpu().numpy().tobytes())
            exposure = self._update_exposure(pattern_key)
            effective_surprise[i] = raw_surprise[i] * math.exp(-exposure / self.tau_novelty)
            
        # 2. Vectorized Salience and Dopamine
        weight_vec, surprise_vec, progress_vec = self.salience_engine.compute(effective_surprise)
        dopamine_vec = self.neuromodulator.compute_dopamine(effective_surprise)
        
        # 3. Add to Buffer (Selective sampling)
        for i in range(batch_size):
            if effective_surprise[i] > 0.05:
                exp = Experience(task_id, x[i:i+1], y[i:i+1], effective_surprise[i].item(), weight_vec[i].item(), self.step_count)
                self.buffer.add(exp)

        # 4. Learning Phase (Vectorized)
        # Average salience/dopamine for the whole-hierarchy step
        # (Individual neurons in PredictiveColumn.update_weights will still use their specific activations)
        avg_weight = weight_vec.mean().item()
        avg_dopamine = dopamine_vec.mean().item()
        
        self._apply_salience_weight(avg_weight)
        
        self.hierarchy.infer_and_learn_online(
            x, top_level_label=y,
            max_steps=max_steps,
            recognition_weight=recognition_weight,
            beta_push=beta_push,
            warm_start=warm_start,
            dopamine_burst=avg_dopamine # Pass avg for now, weight_update uses it as scalar
        )
        
        # --- V5.5: Thermal Safety Check ---
        self.guardian.check(agent=self)
        
        # --- V5.2: Hippocampal Storage (Epiphany) ---
        # If we successfully reduced surprise through a dopamine burst, store the epiphany
        if avg_dopamine > 1.5:
             latent_snapshots = [col.x.detach().clone() for col in self.hierarchy.layers]
             self.hippocampus.store(x, latent_snapshots)

        # 6. Restore base learning rates
        self._apply_salience_weight(1.0, restore=True)
        self.step_count += 1
        
        return avg_weight, effective_surprise

    def dream_replay(self, batch_size: int = 16, max_steps: int = 150, recognition_weight: float = 1.0, beta_push: float = 5.0):
        """Extracts highly salient memories and dreams about them to bootstrap the sparse dictionary."""
        if len(self.buffer) < batch_size:
            return 0.0

        experiences = self.buffer.sample(batch_size)
        x_batch = torch.cat([e.x for e in experiences], dim=0)
        y_batch = torch.cat([e.y for e in experiences], dim=0)
        
        # Boost learning rates for dream consolidation
        self._apply_salience_weight(2.0)
        # V5.2: Dream dopamine — consolidation dreams and neurogenesis are heavily modulated
        avg_effective_tensor = torch.tensor([sum(e.surprise for e in experiences) / len(experiences)], device=self.device)
        dopamine_burst = self.neuromodulator.compute_dopamine(avg_effective_tensor, update_state=False).item()
        
        # Check for Neurogenesis Trigger
        avg_salience = sum(e.salience for e in experiences) / len(experiences)
        
        # V5.2: Unified Neuromodulation & Neurogenesis.
        # Neurogenesis requires UNEXPECTED surprise (Dopamine > threshold). 
        if dopamine_burst > self.dopamine_threshold and avg_salience > self.salience_threshold:
            # Persistent anomaly! Trigger identity sliver neurogenesis
            # Pick the sample with highest individual surprise in the batch
            # V7.3.3: Slice surprise ranking to match manifold
            def _get_surprise(x_in, y_target):
                pred = self.hierarchy.predict_label(x_in, update_temporal=False)
                if pred.shape[1] > y_target.shape[1]:
                    pred = pred[:, :y_target.shape[1]]
                return torch.nn.functional.mse_loss(pred, y_target).item()
            
            sample_surprises = [_get_surprise(x_batch[i:i+1], y_batch[i:i+1]) for i in range(len(experiences))]
            best_idx = int(torch.argmax(torch.tensor(sample_surprises)))
            
            print(f">>> RECRUITING IDENTITY SLIVER PATHWAY for Sample {best_idx} (Surprise: {sample_surprises[best_idx]:.4f}) <<<")
            self.hierarchy.expand_pathway(x_batch[best_idx:best_idx+1], y_batch[best_idx:best_idx+1])
            self.neurogenesis_count += 1
            
            # V5.0: Stamp birth_surprise on newly recruited neurons
            recruit_surprise = sample_surprises[best_idx]
            for col in self.hierarchy.layers:
                col.birth_surprise[-1] = recruit_surprise
            
            # --- NEURAL CONSOLIDATION ---
            # Immediately 'burn-in' the new pathway with a high-intensity focused dream
            print(f">>> CONSOLIDATING PATHWAY {self.hierarchy.layers[0].output_dim-1} <<<")
            self._apply_salience_weight(5.0) # Maximum learning push
            self.hierarchy.infer_and_learn_online(
                x_batch[best_idx:best_idx+1], 
                top_level_label=y_batch[best_idx:best_idx+1],
                max_steps=250, # More steps to settle the new structure
                recognition_weight=1.5,
                beta_push=10.0 # Force alignment
            )
            self._apply_salience_weight(1.0, restore=True)
            
            # V5.0: Check if pruning is due (dynamic interval)
            prune_interval = max(500, 2 * self.neurogenesis_count * 100)
            if self.step_count > 0 and self.step_count % prune_interval == 0:
                self.prune_dormant_experts()
            
            self.buffer._heap.clear()
            self.buffer._counter = 0
            return len(experiences)
        
        self.hierarchy.infer_and_learn(
            x_batch, top_level_label=y_batch,
            max_steps=max_steps,
            recognition_weight=recognition_weight,
            beta_push=beta_push,
            dopamine_burst=max(1.0, dopamine_burst) # Ensure dreams never receive suppressed plasticity
        )
        
        # --- V5.5: Thermal Safety Check (End of dream) ---
        self.guardian.check(agent=self)

        # Restore base learning rates
        self._apply_salience_weight(1.0, restore=True)
        return len(experiences)

    def _apply_salience_weight(self, weight: float, restore: bool = False):
        if restore:
            for i, col in enumerate(self.hierarchy.layers):
                col.eta_V, col.eta_W = self.base_lrs[i]
        else:
            for i, col in enumerate(self.hierarchy.layers):
                base_v, base_w = self.base_lrs[i]
                col.eta_V = base_v * weight
                col.eta_W = base_w * weight

    def prune_dormant_experts(self, threshold: float = 0.01):
        """V5.0: Lift gradient shields on dormant expert neurons."""
        pruned_total = 0
        for i, col in enumerate(self.hierarchy.layers):
            scores = col.compute_retention_scores()
            original_dim = self.hierarchy.layers[0].base_dim
            if col.output_dim <= original_dim:
                continue
            
            expert_scores = scores[original_dim:]
            dormant = (expert_scores < threshold)
            if dormant.any():
                n_dormant = dormant.sum().item()
                col.V_mask[:, original_dim:][:, dormant] = 1.0
                col.W_mask[original_dim:, :][dormant, :] = 1.0
                pruned_total += n_dormant
                print(f"    [PRUNE] Layer {i}: {n_dormant} dormant experts un-shielded")
        
        if pruned_total > 0:
            print(f"    [PRUNE] Total: {pruned_total} expert pathways recycled")
        return pruned_total

    def save_checkpoint(self, filepath: str):
        """Saves current agency state (hierarchy + cognitive buffers)."""
        checkpoint = {
            'hierarchy_state': self.hierarchy.state_dict(),
            'neuromodulator': self.neuromodulator.predicted_surprise,
            'hippocampus': self.hippocampus.memory,
            'step_count': self.step_count,
            'exposure_counts': self.exposure_counts,
            'thermal_stats': {
                'peak': self.guardian.peak_temp,
                'avg': self.guardian.avg_temp
            }
        }
        torch.save(checkpoint, filepath)
        print(f"[Checkpoint] Saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        if not os.path.exists(filepath): return
        checkpoint = torch.load(filepath, map_location=self.device)
        self.hierarchy.load_state_dict(checkpoint['hierarchy_state'])
        self.neuromodulator.predicted_surprise = checkpoint['neuromodulator']
        self.hippocampus.memory = checkpoint['hippocampus']
        self.step_count = checkpoint['step_count']
        self.exposure_counts = checkpoint['exposure_counts']
        print(f"[Checkpoint] Loaded from {filepath}")


# ---------------------------------------------------------------------------
# AbstraX Engine — Cross-Domain Affinity Analysis (V3 Meta-Abstraction)
# ---------------------------------------------------------------------------

class AbstraXEngine:
    """
    V3 Phase 32: The Dream Cycle.

    After training, this engine loads the full checkpoint and performs
    offline Cross-Domain Affinity Analysis. It extracts the learned weight
    slices for each language manifold and computes pairwise structural
    similarity using Cosine Similarity across all major weight matrices
    (V, W, R, R_gate).

    If two languages have high affinity, it means their neurons independently
    discovered similar weight patterns — empirical proof that shared abstract
    structure exists and can be folded into Dream Neurons.
    """

    def __init__(self, hierarchy: PredictiveHierarchy, lang_ranges: Dict[str, Tuple[int, int]]):
        """
        Args:
            hierarchy: A loaded PredictiveHierarchy with trained weights.
            lang_ranges: Dict mapping language code to (start_idx, end_idx).
                         e.g. {'en': (0, 256), 'de': (256, 512), ...}
        """
        self.hierarchy = hierarchy
        self.lang_ranges = lang_ranges
        self.lang_codes = list(lang_ranges.keys())
        self.n_langs = len(self.lang_codes)

    def _extract_weight_signature(self, lang: str, layer_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Extract the weight 'fingerprint' of a language manifold from a given layer.

        For a language occupying neurons [s:e], we extract:
          - V_slice: The recognition weights mapping TO those neurons (V[:, s:e])
          - W_slice: The generative weights mapping FROM those neurons (W[s:e, :])
          - R_slice: The recurrent self-connections within those neurons (R[s:e, s:e])
          - R_gate_slice: The gating recurrence within those neurons (R_gate[s:e, s:e])

        Each slice is flattened into a 1D vector to enable cosine comparison.
        """
        s, e = self.lang_ranges[lang]
        layer = self.hierarchy.layers[layer_idx]

        with torch.no_grad():
            sig = {}
            # Recognition pathway: how this language reads input
            sig['V'] = layer.V.data[:, s:e].flatten()
            # Generative pathway: how this language reconstructs input
            sig['W'] = layer.W.data[s:e, :].flatten()
            # Recurrent dynamics: how this language's neurons talk to each other
            sig['R'] = layer.R.data[s:e, s:e].flatten()
            # Gating dynamics: how this language gates its own recurrence
            sig['R_gate'] = layer.R_gate.data[s:e, s:e].flatten()
            # Bias profile: the activation threshold landscape
            sig['b_in'] = layer.b_in.data[s:e].flatten()

        return sig

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two flattened weight vectors."""
        if a.numel() == 0 or b.numel() == 0:
            return 0.0
        # Ensure same size (they should be if sliver widths are equal)
        min_len = min(a.numel(), b.numel())
        a, b = a[:min_len], b[:min_len]
        dot = torch.dot(a, b)
        norm = torch.norm(a) * torch.norm(b)
        if norm < 1e-8:
            return 0.0
        return (dot / norm).item()

    def compute_pairwise_affinity(self, layer_idx: int = 0) -> Dict[str, object]:
        """
        Compute the full NxN Cross-Domain Affinity Matrix for a given layer.

        Returns a dict containing:
          - 'matrix': NxN tensor of aggregate affinity scores
          - 'per_component': Dict of component-wise NxN matrices (V, W, R, etc.)
          - 'lang_codes': ordered list of language codes
        """
        n = self.n_langs
        # Aggregate affinity matrix
        agg_matrix = torch.zeros(n, n)
        # Per-component matrices
        components = ['V', 'W', 'R', 'R_gate', 'b_in']
        per_comp = {c: torch.zeros(n, n) for c in components}

        # Extract all signatures
        sigs = {}
        for lang in self.lang_codes:
            sigs[lang] = self._extract_weight_signature(lang, layer_idx)

        # Compute pairwise similarities
        for i, lang_a in enumerate(self.lang_codes):
            for j, lang_b in enumerate(self.lang_codes):
                if i == j:
                    # Self-affinity is always 1.0
                    agg_matrix[i, j] = 1.0
                    for c in components:
                        per_comp[c][i, j] = 1.0
                    continue

                comp_scores = []
                for c in components:
                    sim = self._cosine_sim(sigs[lang_a][c], sigs[lang_b][c])
                    per_comp[c][i, j] = sim
                    comp_scores.append(sim)

                # Aggregate: weighted average (V and W are most important)
                weights = {'V': 2.0, 'W': 2.0, 'R': 1.5, 'R_gate': 1.0, 'b_in': 0.5}
                weighted_sum = sum(per_comp[c][i, j].item() * weights[c] for c in components)
                total_weight = sum(weights.values())
                agg_matrix[i, j] = weighted_sum / total_weight

        return {
            'matrix': agg_matrix,
            'per_component': per_comp,
            'lang_codes': self.lang_codes
        }

    def print_affinity_report(self, result: Dict, title: str = "Cross-Domain Affinity Matrix"):
        """Pretty-print the affinity matrix as a formatted table."""
        matrix = result['matrix']
        codes = result['lang_codes']
        n = len(codes)

        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

        # Header
        header = f"{'':>6}" + "".join(f"{c:>8}" for c in codes)
        print(header)
        print("-" * len(header))

        for i, lang_a in enumerate(codes):
            row = f"{lang_a:>6}"
            for j in range(n):
                val = matrix[i, j].item()
                row += f"{val:>8.4f}"
            print(row)

        print(f"{'='*60}")

        # Find the top-K most similar pairs (excluding self)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((matrix[i, j].item(), codes[i], codes[j]))
        pairs.sort(reverse=True)

        print(f"\n  Top-5 Most Similar Language Pairs:")
        print(f"  {'─'*40}")
        for rank, (score, a, b) in enumerate(pairs[:5], 1):
            bar = "█" * int(score * 20)
            print(f"  {rank}. {a} ↔ {b}: {score:.4f}  {bar}")

        print(f"\n  Bottom-3 Least Similar Pairs:")
        print(f"  {'─'*40}")
        for rank, (score, a, b) in enumerate(pairs[-3:], 1):
            bar = "█" * int(max(0, score) * 20)
            print(f"  {rank}. {a} ↔ {b}: {score:.4f}  {bar}")

        return pairs

    def identify_fold_candidates(self, result: Dict, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identify language pairs whose affinity exceeds the folding threshold.
        These are candidates for Dream Neuron synthesis — shared neurons
        that both languages can route through.
        """
        matrix = result['matrix']
        codes = result['lang_codes']
        n = len(codes)
        candidates = []

        for i in range(n):
            for j in range(i + 1, n):
                score = matrix[i, j].item()
                if score >= threshold:
                    candidates.append((codes[i], codes[j], score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        if candidates:
            print(f"\n  ╔══════════════════════════════════════════╗")
            print(f"  ║  DREAM NEURON CANDIDATES (threshold={threshold:.1f})  ║")
            print(f"  ╠══════════════════════════════════════════╣")
            for a, b, score in candidates:
                print(f"  ║  {a} ↔ {b}  affinity={score:.4f}  → FOLDABLE    ║")
            print(f"  ╚══════════════════════════════════════════╝")
        else:
            print(f"\n  [AbstraX] No pairs exceed folding threshold {threshold:.1f}.")
            print(f"  [AbstraX] Languages are structurally independent (as expected for isolated slivers).")
            print(f"  [AbstraX] Next step: introduce shared meta-neurons and re-train to induce convergence.")

        return candidates

    def synthesize_dream_neurons(self, meta_pool_size: int = 64):
        """
        V21 Phase: Dream Synthesis.
        Averages the highly-affine weight structures from the isolated language
        slivers and writes them into the shared Meta-Pool, creating a physical
        representation of Universal Grammar.
        """
        print(f"\n>>> SYNTHESIZING DREAM NEURONS INTO META-POOL ({meta_pool_size} neurons) <<<")
        n_langs = len(self.lang_codes)
        
        for layer_idx, layer in enumerate(self.hierarchy.layers):
            if layer_idx == 0:
                continue # Skip input layer for folding

            print(f"  Folding Layer {layer_idx}...")
            # We average the matrices across all languages
            avg_V = torch.zeros_like(layer.V.data[:, :meta_pool_size])
            avg_W = torch.zeros_like(layer.W.data[:meta_pool_size, :])
            avg_R = torch.zeros_like(layer.R.data[:meta_pool_size, :meta_pool_size])
            avg_R_gate = torch.zeros_like(layer.R_gate.data[:meta_pool_size, :meta_pool_size])
            avg_b_in = torch.zeros_like(layer.b_in.data[:meta_pool_size])
            
            for lang in self.lang_codes:
                s, e = self.lang_ranges[lang]
                # Extract the top `meta_pool_size` neurons from this language
                l_s, l_e = s, s + meta_pool_size
                
                avg_V += layer.V.data[:, l_s:l_e]
                avg_W += layer.W.data[l_s:l_e, :]
                avg_R += layer.R.data[l_s:l_e, l_s:l_e]
                avg_R_gate += layer.R_gate.data[l_s:l_e, l_s:l_e]
                avg_b_in += layer.b_in.data[l_s:l_e]
            
            # Average and assign to meta-pool
            layer.V.data[:, :meta_pool_size] = avg_V / n_langs
            layer.W.data[:meta_pool_size, :] = avg_W / n_langs
            layer.R.data[:meta_pool_size, :meta_pool_size] = avg_R / n_langs
            layer.R_gate.data[:meta_pool_size, :meta_pool_size] = avg_R_gate / n_langs
            layer.b_in.data[:meta_pool_size] = avg_b_in / n_langs
            
            # Unmask the Meta-Pool for future learning
            layer.V_mask[:, :meta_pool_size] = 1.0
            layer.W_mask[:meta_pool_size, :] = 1.0
            layer.R_mask[:meta_pool_size, :meta_pool_size] = 1.0
            layer.R_gate_mask[:meta_pool_size, :meta_pool_size] = 1.0
            layer.b_in_mask[:meta_pool_size] = 1.0
            if layer_idx > 0:
                layer.b_out_mask[:meta_pool_size] = 1.0

        print("  [Done] Universal Grammar synthesized. Meta-Pool is now active.")

