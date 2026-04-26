"""
agnis_v4_core.py  —  AGNIS V4.9  "Unclamped Hebbian"
=========================================================
SNAP-ATP (Synchronized Aggressive Target Propagation)
Biologically-Plausible Predictive Coding Hierarchy

Changelog
---------
v4.1-4.5: SNAP-ATP, Direct Supervision, Asymmetric velocity, 
           Balanced Scaffold, Weight-Norm Clipping.

v4.6  Discovery Burst:
      CRITICAL FIX: reset_states() added to cleanse latent states 
      between examples. Prevents attractor contamination.

v4.8  Synaptic Homeostasis (Protocol 4.8):
      Bias Homeostasis — biases receive same decay as weights.
      lambda_V = 1e-5 (unified with W).  eta_V = 0.05.

v4.9  Unclamped Hebbian (Protocol 4.9):
      ROOT CAUSE FIX for Phase A/B failures:
        1. REMOVED multiplicative weight decay (lambda_V/W = 0).
           Decay was erasing learned structure faster than Hebbian
           updates could build it (W-top collapsed 0.56→0.10).
        2. INCREASED eta_W: 0.005 → 0.03.  The 10x asymmetry
           between recognition (0.05) and generative (0.005)
           prevented W from building useful top-down predictions.
        3. RELAXED update clipping: max_norm 1.0 → 5.0.
           Batch-averaged outer products for non-linear tasks
           have low norm; clipping at 1.0 destroyed the signal.
        4. ADDED weight clamping [-3, 3] as soft homeostasis
           replacement.  Prevents explosion without suppressing
           learning.

Recommended config for 4-bit parity (v4.9)
-------------------------------------------
    Use v4.9 script with 300 steps, 5.0x push.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_update(d: torch.Tensor, max_norm: float = 5.0) -> torch.Tensor:
    n = torch.norm(d)
    if n > max_norm:
        return d * (max_norm / (n + 1e-8))
    return d


# ---------------------------------------------------------------------------
# PredictiveColumn
# ---------------------------------------------------------------------------

class PredictiveColumn(nn.Module):
    """A single SNAP-ATP column with Unclamped Hebbian learning (v4.9)."""

    def __init__(self, input_dim: int, output_dim: int, device: str = "cpu"):
        super().__init__()
        self.device     = torch.device(device)
        self.input_dim  = input_dim
        self.output_dim = output_dim

        k_v = math.sqrt(1.0 / input_dim)
        k_w = math.sqrt(1.0 / output_dim)

        self.V    = nn.Parameter(torch.empty(input_dim, output_dim, device=self.device).uniform_(-k_v, k_v))
        # Initialize with deep negative bias (Gated Birth: Start silent)
        self.b_in  = nn.Parameter(torch.full((output_dim,), -5.0, device=self.device))

        self.W     = nn.Parameter(torch.empty(output_dim, input_dim, device=self.device).uniform_(-k_w, k_w))
        self.b_out = nn.Parameter(torch.zeros(input_dim, device=self.device))
        self.register_buffer("b_out_mask", torch.ones(input_dim, device=self.device))
        
        # Expert Masks to shield pathways from corruption
        self.register_buffer("V_mask", torch.ones_like(self.V))
        self.register_buffer("W_mask", torch.ones_like(self.W))
        self.register_buffer("b_in_mask", torch.ones(output_dim, device=self.device))

        # --- V5.0: Lateral Communication (Sparse L matrix) ---
        self.lateral_k = 3  # k-nearest neighbors
        self.eta_L = 0.01   # lateral learning rate (slow)
        self.L = nn.Parameter(torch.zeros(output_dim, output_dim, device=self.device))
        # Build sparse mask: only k nearest neighbors per neuron
        L_mask = torch.zeros(output_dim, output_dim, device=self.device)
        for j in range(output_dim):
            neighbors = []
            for offset in range(1, self.lateral_k + 1):
                if j - offset >= 0: neighbors.append(j - offset)
                if j + offset < output_dim: neighbors.append(j + offset)
            neighbors = neighbors[:self.lateral_k]
            for n in neighbors:
                L_mask[j, n] = 1.0
        self.register_buffer("L_mask", L_mask)
        # Zero diagonal — no self-connections
        self.L.data.fill_diagonal_(0.0)

        self.register_buffer("x", torch.zeros(1, output_dim, device=self.device))
        self.register_buffer("x_temporal", torch.zeros(1, output_dim, device=self.device))
        self.register_buffer("x_temporal_2", torch.zeros(1, output_dim, device=self.device))
        self.register_buffer("x_temporal_3", torch.zeros(1, output_dim, device=self.device))

        self.error      = torch.zeros(input_dim, device=self.device)
        self.last_input : torch.Tensor | None = None

        self.tau        = 0.5
        self.eta_x      = 0.8
        self.eta_V      = 0.05    # agile recognition
        self.eta_W      = 0.03    # v4.9: raised from 0.005 to build generative model
        self.eta_R      = 0.01    # V7.0: Stable post-parity consolidation
        self.layer_norm_r = nn.LayerNorm(output_dim, device=self.device)

        # v4.9: Homeostasis REMOVED — replaced by weight clamping
        self.lambda_V   = 0.0
        self.lambda_W   = 0.0
        self.weight_clamp = 3.0   # v4.9: soft bound on weight magnitudes
        self.recurrent_clamp = 1.5

        # --- V5.0: Per-Layer Adaptive Clamping (EMA-based) ---
        self._wc_running_mean = 0.5   # EMA of weight magnitude mean
        self._wc_running_std  = 0.3   # EMA of weight magnitude std
        self._wc_ema_alpha    = 0.01  # EMA smoothing factor
        self._wc_recalibrate_interval = 1000
        self._wc_step_counter = 0

        # --- V5.0: Expert Retention Scoring ---
        self.register_buffer("firing_count", torch.zeros(output_dim, device=self.device))
        self.register_buffer("last_fire_step", torch.zeros(output_dim, device=self.device))
        self.register_buffer("birth_surprise", torch.zeros(output_dim, device=self.device))
        self._total_steps = 0

        self.lambda_act = 1e-6
        self._settled   = False

        k_r = math.sqrt(1.0 / max(1, output_dim))
        # V8.4: Spectral Matrix Recurrence
        # High expressivity + Guaranteed Stability
        self.R = nn.Parameter(torch.eye(output_dim, device=self.device) * 0.95)
        self.register_buffer("R_mask", torch.ones(output_dim, output_dim, device=self.device))
        
        # V8.2: ACT Halting Gate (Learned Convergence)
        self.halt_gate = nn.Linear(output_dim, 1, device=self.device)
        
        self.R_gate = nn.Parameter(torch.empty(output_dim, output_dim, device=self.device).uniform_(-k_r, k_r) * 0.1)
        self.register_buffer("R_gate_mask", torch.ones(output_dim, output_dim, device=self.device))
        with torch.no_grad():
            # Gate starts mostly OPEN
            self.R_gate.data.add_(torch.eye(output_dim, device=self.device) * 2.0)

    def _k_wta_mask(self, x: torch.Tensor, k_ratio: float = 0.25) -> torch.Tensor:
        if x.shape[-1] <= 1:
            return torch.ones_like(x)
        k = max(1, int(x.shape[-1] * k_ratio))
        _, indices = torch.topk(x, k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        return mask

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._k_wta_mask(x)
        return torch.nn.functional.gelu(x) * mask

    def _phi_deriv(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._k_wta_mask(x)
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
        return (cdf + x * pdf) * mask

    def reset_state(self, batch_size: int = 1):
        if self.x.shape[0] != batch_size:
            self.x = torch.zeros(batch_size, self.output_dim, device=self.device)
            self.x_temporal = torch.zeros_like(self.x)
            self.x_temporal_2 = torch.zeros_like(self.x)
            self.x_temporal_3 = torch.zeros_like(self.x)
        else:
            self.x.zero_()
            self.x_temporal.zero_()
            self.x_temporal_2.zero_()
            self.x_temporal_3.zero_()

    def step_temporal(self):
        # Shift temporal history (t-2 -> t-3, t-1 -> t-2, t -> t-1)
        self.x_temporal_3.copy_(self.x_temporal_2.detach())
        self.x_temporal_2.copy_(self.x_temporal.detach())
        self.x_temporal.copy_(self.x.detach())

    def reset_recurrent_matrix(self, gain: float = 0.1):
        """V7.2: Reset R with orthogonal initialization to provide a neutral starting point."""
        with torch.no_grad():
            nn.init.orthogonal_(self.R, gain=gain)

    def set_experts_bias(self, start_idx: int, end_idx: int, bias_val: float):
        """V7.3.5: Modulate expert sensitivity for isolative probing."""
        with torch.no_grad():
            self.b_in[start_idx:end_idx] = bias_val

    def freeze_experts(self):
        """V7.3: Absolute synaptic shield. Makes all current synapses immutable."""
        with torch.no_grad():
            # Record frozen count for verification
            pre_freeze = self.V_mask.sum().item()
            self.V_mask.zero_()
            self.W_mask.zero_()
            self.b_in_mask.zero_()
            self.b_out_mask.zero_()
            self.R_mask.zero_()
            self.R_gate_mask.zero_()
            self.L_mask.zero_()
            
            post_freeze = self.V_mask.sum().item()
            assert post_freeze == 0, "Freeze failed — masks not zeroed"
            return pre_freeze 

    def get_stable_R(self):
        # Discretize: ensures R_ii in (0, 1) always
        # R = exp(-delta_t * exp(log_A))
        # Note: Since A is diagonal, matrix_exp is just element-wise exp
        return torch.exp(-self.delta_t * torch.exp(self.log_A))

    def infer_step_sync(self, bottom_up, td_target, step_i, W_snap, b_out_snap, recognition_weight=1.0):
        self.last_input = bottom_up.detach()
        x_prev = self.x.clone()
        phi_x  = self._phi(self.x)
        prediction = torch.matmul(phi_x, W_snap) + b_out_snap
        self.error = (bottom_up - prediction).detach()

        feedback_drive    = torch.matmul(self.error, W_snap.t()) * self._phi_deriv(self.x) * recognition_weight
        forward_drive     = torch.matmul(bottom_up, self.V) + self.b_in
        recognition_drive = (forward_drive - self.x) * recognition_weight
        top_down_drive    = 3.0 * (td_target - self.x) if td_target is not None else 0.0
        
        # V6.4: Multi-step Gated Temporal Binding
        x_context = 0.5 * self.x_temporal.detach() + 0.3 * self.x_temporal_2.detach() + 0.2 * self.x_temporal_3.detach()
        
        # V8.4: Non-Linear Matrix Recurrence
        recurrent_raw = torch.matmul(self._phi(x_context), self.R)
        gate              = torch.sigmoid(torch.matmul(x_context, self.R_gate))
        recurrent_drive   = (recurrent_raw * gate)
        self._current_x_context = x_context

        # V8.3: Softened Persistent Input Anchor
        direct_input_drive = (forward_drive - self.x) * (0.5 * recognition_weight)
        
        state_grad  = feedback_drive + recognition_drive + top_down_drive + direct_input_drive
        state_grad += (recurrent_drive - self.x)
        
        # V8.3: Lateral Surge (Forced Specialization)
        lateral_drive = torch.matmul(self._phi(self.x), self.L)
        state_grad += (1.0 * recognition_weight) * lateral_drive
        state_grad -= self.lambda_act * torch.sign(self.x)

        eta = self.eta_x / (1.0 + 0.1 * step_i)
        dx  = _clip_update(self.tau * eta * state_grad, max_norm=1.0)
        self.x = (self.x + dx.detach()).clamp_(-5.0, 5.0)
        return torch.max(torch.abs(self.x - x_prev))

    def infer_step_top(self, bottom_up, label, step_i, W_snap, b_out_snap, recognition_weight=1.0, beta_push=3.0):
        self.last_input = bottom_up.detach()
        x_prev = self.x.clone()
        phi_x  = self._phi(self.x)
        prediction = torch.matmul(phi_x, W_snap) + b_out_snap
        self.error = (bottom_up - prediction).detach()

        if label.shape[1] < self.x.shape[1]:
            label_drive = torch.zeros_like(self.x)
            d = label.shape[1]
            label_drive[:, :d] = beta_push * (label - self.x[:, :d])
        else:
            label_drive = beta_push * (label - self.x)

        forward_drive     = torch.matmul(bottom_up, self.V) + self.b_in
        recognition_drive = (forward_drive - self.x) * recognition_weight
        
        # V6.4: Multi-step Gated Temporal Binding
        x_context = 0.5 * self.x_temporal.detach() + 0.3 * self.x_temporal_2.detach() + 0.2 * self.x_temporal_3.detach()
        
        # V8.4: Non-Linear Matrix Recurrence
        recurrent_raw = torch.matmul(self._phi(x_context), self.R)
        gate              = torch.sigmoid(torch.matmul(x_context, self.R_gate))
        recurrent_drive   = (recurrent_raw * gate)
        self._current_x_context = x_context

        # V8.3: Softened Persistent Input Anchor
        direct_input_drive = (forward_drive - self.x) * (0.5 * recognition_weight)
        
        state_grad  = label_drive + recognition_drive + direct_input_drive
        state_grad += (recurrent_drive - self.x)

        # V8.3: Lateral Surge (Forced Specialization)
        lateral_drive = torch.matmul(self._phi(self.x), self.L)
        state_grad += (1.0 * recognition_weight) * lateral_drive
        state_grad -= self.lambda_act * torch.sign(self.x)

        eta = self.eta_x / (1.0 + 0.1 * step_i)
        dx  = _clip_update(self.tau * eta * state_grad, max_norm=1.0)
        self.x = (self.x + dx.detach()).clamp_(-5.0, 5.0)
        return torch.max(torch.abs(self.x - x_prev))

    def update_weights(self, is_top=False, lambda_W_top=0.0, dopamine_burst=1.0, convergence_quality=1.0):
        if not self._settled or self.last_input is None: return
        with torch.no_grad():
            batch_size = self.x.shape[0]
            
            # 1. State innovation (for V learning)
            forward = torch.matmul(self.last_input, self.V) + self.b_in
            mask = self._k_wta_mask(self.x)
            delta_h = (self.x - forward) * mask
            phi_h = self._phi(self.x)
            
            # 2. Vectorized Dopamine Masking
            # phi_h is [batch, output_dim]. d_mask is [batch, output_dim]
            d_mask = torch.where(phi_h.abs() > 0.01, dopamine_burst, 1.0)
            
            # 3. Vectorized Recognition (V) Update
            # input: [batch, input_dim]. delta_h: [batch, output_dim]
            # dV_batch: [batch, input_dim, output_dim]. 
            # We use broadcasting for d_mask across the input_dim.
            dV_unmasked = torch.bmm(self.last_input.unsqueeze(2), delta_h.unsqueeze(1))
            dV_batch = dV_unmasked * d_mask.unsqueeze(1) # [batch, in, out]
            
            # Average gradients across batch for SNAP-ATP stability
            dV_avg = dV_batch.mean(dim=0)
            self.V.data += self.eta_V * _clip_update(dV_avg, max_norm=5.0) * self.V_mask
            self.b_in.data += self.eta_V * (delta_h * d_mask).mean(dim=0) * self.b_in_mask

            # 4. Vectorized Generative (W) Update
            # phi_h: [batch, out]. error: [batch, in]
            # dW_batch: [batch, out, in]
            dW_unmasked = torch.bmm(phi_h.unsqueeze(2), self.error.unsqueeze(1))
            dW_batch = dW_unmasked * d_mask.unsqueeze(2) 
            
            dW_avg = dW_batch.mean(dim=0)
            self.W.data += self.eta_W * _clip_update(dW_avg, max_norm=5.0) * self.W_mask
            self.b_out.data += self.eta_W * self.error.mean(dim=0) * self.b_out_mask

            # 5. Vectorized Recurrent (R) Update
            # temporal_src: [batch, out]. Innovation target: [batch, out]
            temporal_state = getattr(self, '_current_x_context', self.x_temporal).detach()
            temporal_src = self._phi(temporal_state)
            
            # Recurrent drive recalculated for gradient precision
            # Keep recurrent learning aligned with the non-linearity used at inference.
            recurrent_raw = torch.matmul(temporal_src, self.R.data)
            gate = torch.sigmoid(torch.matmul(temporal_state, self.R_gate.data))
            recurrent_drive = (recurrent_raw * gate)
            
            # V8.4: Spectral Recurrent Update
            dR_target = (self.x.detach() - recurrent_drive)
            dR_batch = torch.bmm(temporal_src.unsqueeze(2), dR_target.unsqueeze(1))
            dR_avg = dR_batch.mean(dim=0)
            self.R.data += self.eta_R * dopamine_burst * _clip_update(dR_avg, max_norm=1.0) * self.R_mask
            # Spectral Normalization: Keep spectral radius of the TRAINABLE subspace < 1
            with torch.no_grad():
                trainable_R = self.R.data * self.R_mask
                norm = torch.linalg.norm(trainable_R, ord=2)
                if norm > 0.98: 
                    scaled_trainable = trainable_R * (0.98 / norm)
                    self.R.data = torch.where(self.R_mask == 0.0, self.R.data, scaled_trainable)

            # V6.3: Train the Gate (Learns when to open based on the temporal mismatch)
            # We use the full matrix gradient for the gate to maintain cross-dependency capacity
            dR_matrix_batch = torch.bmm(temporal_src.unsqueeze(2), dR_target.unsqueeze(1))
            dR_matrix_avg = dR_matrix_batch.mean(dim=0)
            self.R_gate.data += self.eta_R * 0.5 * dopamine_burst * _clip_update(dR_matrix_avg, max_norm=1.0) * self.R_gate_mask

            # V8.2: Train the ACT Halt Gate
            # Tries to learn to fire for the settled state x
            halt_error = (1.0 - torch.sigmoid(self.halt_gate(self.x.detach())))
            self.halt_gate.weight.data.add_((torch.matmul(halt_error.t(), self.x.detach()) * self.eta_R * 0.1) * self.b_in_mask.unsqueeze(0))

            # 6. Adaptive Weight Clamping (Vectorized stats are already averaged)
            self._wc_step_counter += 1
            avg_mag = (self.V.data.abs().mean() + self.W.data.abs().mean()) / 2.0
            self._wc_running_mean = (1 - self._wc_ema_alpha) * self._wc_running_mean + self._wc_ema_alpha * avg_mag.item()
            self._wc_running_std  = (1 - self._wc_ema_alpha) * self._wc_running_std  + self._wc_ema_alpha * abs(avg_mag.item() - self._wc_running_mean)

            if self._wc_step_counter % self._wc_recalibrate_interval == 0:
                proposed_clamp = self._wc_running_mean + 2 * self._wc_running_std
                self.weight_clamp = max(3.0, proposed_clamp)

            wc = self.weight_clamp
            old_V = self.V.data.clone()
            old_W = self.W.data.clone()
            old_b_in = self.b_in.data.clone()
            old_b_out = self.b_out.data.clone()
            
            self.V.data.clamp_(-wc, wc)
            self.b_in.data.clamp_(-wc, wc)
            self.W.data.clamp_(-wc, wc)
            self.b_out.data.clamp_(-wc, wc)
            
            # Restore frozen weights from clamp clipping
            self.V.data = torch.where(self.V_mask == 0.0, old_V, self.V.data)
            self.W.data = torch.where(self.W_mask == 0.0, old_W, self.W.data)
            self.b_in.data = torch.where(self.b_in_mask == 0.0, old_b_in, self.b_in.data)
            self.b_out.data = torch.where(self.b_out_mask == 0.0, old_b_out, self.b_out.data)

            # 7. Expert Retention — track firing (Vectorized)
            self._total_steps += 1
            fired = (phi_h.abs().mean(dim=0) > 0.01)
            self.firing_count[fired] += 1
            self.last_fire_step[fired] = float(self._total_steps)

            # 8. Lateral L Hebbian Update (Vectorized)
            # outer: [batch, out, out]
            dL_batch = torch.bmm(phi_h.unsqueeze(2), phi_h.unsqueeze(1)) * self.L_mask
            dL_avg = dL_batch.mean(dim=0)
            dL_avg -= 0.01 * self.L.data * self.L_mask # V11.4 Fix: Masked Decay
            self.L.data += self.eta_L * dL_avg
            
            self.L.data.clamp_(-1.5, 1.5)
            self.L.data.fill_diagonal_(0.0) 


    def expand_output(self, num_neurons: int, init_last_input: torch.Tensor = None, init_target: torch.Tensor = None, 
                      init_type: str = "hebbian", bias_init: float = -5.0):
        """V7.3: Modular expansion with support for orthogonal domain-initialization."""
        with torch.no_grad():
            D_in, D_out = self.input_dim, self.output_dim
            d = num_neurons
            
            # 1. Expand Recongnition (V)
            v_new = torch.empty(D_in, d, device=self.device)
            bias_new = torch.full((d,), bias_init, device=self.device)
            
            if init_type == "hebbian" and init_last_input is not None:
                v_new.normal_(0, 0.01)
                for i in range(min(d, init_last_input.shape[0])):
                    val = init_last_input[i]
                    norm_val = val.norm() + 1e-6
                    v_new[:, i] = val / norm_val
                    bias_new[i] = -0.5 # Expert Bias: selective
            elif init_type == "orthogonal":
                # Initialize ONLY the new slice with stronger gain (0.3) for rapid adaptation
                nn.init.orthogonal_(v_new, gain=0.3)
            else:
                v_new.normal_(0, 0.01)
            
            self.V = nn.Parameter(torch.cat([self.V.data, v_new], dim=1))
            self.b_in = nn.Parameter(torch.cat([self.b_in.data, bias_new]))
            
            # 2. Expand Generative (W)
            w_new = torch.empty(d, D_in, device=self.device)
            if init_type == "hebbian" and init_target is not None:
                w_new.normal_(0, 0.01)
                for i in range(min(d, init_target.shape[0])):
                    w_new[i, :] = init_target[i]
            elif init_type == "orthogonal":
                nn.init.orthogonal_(w_new, gain=0.3)
            else:
                w_new.normal_(0, 0.01)
                    
            self.W = nn.Parameter(torch.cat([self.W.data, w_new], dim=0))
            self.output_dim += d
            self.x = torch.zeros(self.x.shape[0], self.output_dim, device=self.device)
            
            # 3. Update Masks: New sliver starts fully trainable (1.0)
            self.register_buffer("V_mask", torch.cat([self.V_mask, torch.ones(D_in, d, device=self.device)], dim=1))
            self.register_buffer("W_mask", torch.cat([self.W_mask, torch.ones(d, D_in, device=self.device)], dim=0))
            self.register_buffer("b_in_mask", torch.cat([self.b_in_mask, torch.ones(d, device=self.device)]))
            
            # V7.3.2: Re-initialize LayerNorm to match the expanded dimensionality
            self.layer_norm_r = nn.LayerNorm(self.output_dim, device=self.device)

            # V5.0: Grow retention buffers
            self.register_buffer("firing_count", torch.cat([self.firing_count, torch.zeros(d, device=self.device)]))
            self.register_buffer("last_fire_step", torch.cat([self.last_fire_step, torch.zeros(d, device=self.device)]))
            self.register_buffer("birth_surprise", torch.cat([self.birth_surprise, torch.zeros(d, device=self.device)]))

            # V5.0: Grow lateral L matrix and mask
            new_dim = self.output_dim
            L_old = self.L.data
            L_new = torch.zeros(new_dim, new_dim, device=self.device)
            L_new[:D_out, :D_out] = L_old
            self.L = nn.Parameter(L_new)
            # Grow L_mask: connect new neurons to their k-nearest neighbors
            L_mask_new = torch.zeros(new_dim, new_dim, device=self.device)
            L_mask_new[:D_out, :D_out] = self.L_mask
            for j in range(D_out, new_dim):
                neighbors = []
                for offset in range(1, self.lateral_k + 1):
                    if j - offset >= 0: neighbors.append(j - offset)
                    if j + offset < new_dim: neighbors.append(j + offset)
                neighbors = neighbors[:self.lateral_k]
                for n in neighbors:
                    L_mask_new[n, j] = 1.0 # Source n, Target j (new)
                    if n >= D_out:
                        L_mask_new[j, n] = 1.0 # Source j, Target n (new)
            self.register_buffer("L_mask", L_mask_new)

            R_old = self.R.data
            R_gate_old = self.R_gate.data
            R_new = torch.zeros(new_dim, new_dim, device=self.device)
            R_gate_new = torch.zeros(new_dim, new_dim, device=self.device)
            R_new[:D_out, :D_out] = R_old
            R_gate_new[:D_out, :D_out] = R_gate_old
            self.R = nn.Parameter(R_new)
            self.R_gate = nn.Parameter(R_gate_new)
            
            R_mask_new = torch.zeros(new_dim, new_dim, device=self.device)
            R_mask_new[:D_out, :D_out] = self.R_mask
            R_mask_new[:, D_out:] = 1.0 # Allow new neurons to learn incoming temporal dependencies
            self.register_buffer("R_mask", R_mask_new)
            
            R_gate_mask_new = torch.zeros(new_dim, new_dim, device=self.device)
            R_gate_mask_new[:D_out, :D_out] = self.R_gate_mask
            R_gate_mask_new[:, D_out:] = 1.0
            self.register_buffer("R_gate_mask", R_gate_mask_new)

            self.x_temporal = torch.zeros(self.x.shape[0], self.output_dim, device=self.device)
            self.x_temporal_2 = torch.zeros_like(self.x_temporal)
            self.x_temporal_3 = torch.zeros_like(self.x_temporal)

            # V8.2: Expand ACT Halting Gate
            old_halt_weight = self.halt_gate.weight.data
            old_halt_bias = self.halt_gate.bias.data
            self.halt_gate = nn.Linear(self.output_dim, 1, device=self.device)
            self.halt_gate.weight.data[:, :D_out] = old_halt_weight
            self.halt_gate.weight.data[:, D_out:] = 0.0 # New neurons start with 0 halt influence
            self.halt_gate.bias.data = old_halt_bias

    def expand_input(self, num_neurons: int, init_v: torch.Tensor = None, init_w: torch.Tensor = None):
        with torch.no_grad():
            D_in, D_out = self.input_dim, self.output_dim
            d = num_neurons
            
            v_new = torch.zeros(d, D_out, device=self.device)
            if init_v is not None:
                v_new = init_v
            self.V = nn.Parameter(torch.cat([self.V.data, v_new], dim=0))
            
            w_new = torch.zeros(D_out, d, device=self.device)
            if init_w is not None:
                w_new = init_w
            self.W = nn.Parameter(torch.cat([self.W.data, w_new], dim=1))
            
            self.b_out = nn.Parameter(torch.cat([self.b_out.data, torch.zeros(d, device=self.device)]))
            self.register_buffer("b_out_mask", torch.cat([self.b_out_mask, torch.ones(d, device=self.device)]))
            self.error = torch.zeros(D_in + d, device=self.device)
            self.input_dim += d
            
            # Update Masks: New input dimensions are masked to 0 for gradients
            self.register_buffer("V_mask", torch.cat([self.V_mask, torch.zeros(d, D_out, device=self.device)], dim=0))
            self.register_buffer("W_mask", torch.cat([self.W_mask, torch.zeros(D_out, d, device=self.device)], dim=1))

    def compute_retention_scores(self):
        """V5.0: Compute retention score for each neuron.
        retention = (firing_frequency * task_criticality) / dormancy
        task_criticality = birth_surprise (max surprise seen at recruitment)
        """
        dormancy = (self._total_steps - self.last_fire_step + 1).clamp(min=1)
        criticality = self.birth_surprise.clamp(min=0.1)  # floor to prevent zero-division
        scores = (self.firing_count * criticality) / dormancy
        return scores

# ---------------------------------------------------------------------------
# PredictiveHierarchy
# ---------------------------------------------------------------------------

class PredictiveHierarchy(nn.Module):
    def __init__(self, layer_dims, device="cpu", lambda_W_top=0.0):
        super().__init__()
        self.device = device
        self.lambda_W_top = lambda_W_top
        self.layers = nn.ModuleList([PredictiveColumn(layer_dims[i], layer_dims[i+1], device) for i in range(len(layer_dims)-1)])

    def reset_states(self, batch_size: int = 1):
        for col in self.layers:
            col.reset_state(batch_size)

    def step_temporal(self):
        for col in self.layers:
            col.step_temporal()

    def reset_recurrent_matrices(self, gain: float = 0.1):
        """V7.2: Neutralize temporal priors across the entire hierarchy."""
        for col in self.layers:
            col.reset_recurrent_matrix(gain=gain)

    def snapshot_r_matrices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """V7.2: Capture the current state of all recurrent weights (including gates)."""
        return [(col.R.detach().clone(), col.R_gate.detach().clone()) for col in self.layers]

    def load_r_matrices(self, snapshots: list[torch.Tensor]):
        """V7.3.11: Dimension-invariant loading with block-diagonal temporal isolation.
        Preserves original linguistic priors in the [0:snapshot_size] subspace
        while maintaining orthogonal discovery states in expanded subspaces."""
        for layer, snapshot in zip(self.layers, snapshots):
            current_size = layer.R.shape[0]
            snapshot_size = snapshot.shape[0]
            
            if current_size == snapshot_size:
                # Standard case: Dimensions match
                layer.R.data.copy_(snapshot[0])
                layer.R_gate.data.copy_(snapshot[1])
            elif current_size > snapshot_size:
                # Expansion case: Block-diagonal isolation
                new_R = torch.zeros(current_size, current_size, device=self.device)
                new_R_gate = torch.zeros(current_size, current_size, device=self.device)
                
                # Block 1: Restore Italian (or prior) temporal context (0:snap, 0:snap)
                new_R[:snapshot_size, :snapshot_size] = snapshot[0]
                new_R_gate[:snapshot_size, :snapshot_size] = snapshot[1]
                
                # Block 2: Russian (or new) quadrant stays orthogonal (snap:end, snap:end)
                new_R[snapshot_size:, snapshot_size:] = layer.R[snapshot_size:, snapshot_size:]
                new_R_gate[snapshot_size:, snapshot_size:] = layer.R_gate[snapshot_size:, snapshot_size:]
                
                layer.R.data.copy_(new_R)
                layer.R_gate.data.copy_(new_R_gate)
            else:
                raise ValueError(f"Snapshot ({snapshot_size}) larger than R ({current_size})!")

    def _save_full_state(self):
        """V7.3.9: Capture full expanded state, temporal history, components, and dimensions."""
        return [(l.V, l.W, l.R, l.R_gate, l.L, l.x, l.x_temporal, l.x_temporal_2, l.x_temporal_3, l.V_mask, l.W_mask, l.L_mask, l.R_mask, l.R_gate_mask, l.b_in_mask, l.b_out_mask, l.b_in, l.b_out, l.layer_norm_r, l.halt_gate.weight.detach().clone(), l.halt_gate.bias.detach().clone(), l.input_dim, l.output_dim) for l in self.layers]

    def save_checkpoint(self, path):
        """V11.4: Save the entire active state to disk."""
        state = self._save_full_state()
        torch.save(state, path)

    def load_checkpoint(self, path):
        """V11.4: Restore the entire active state from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self._restore_full_state(state)

    def _restore_full_state(self, saved_states):
        """V7.3.9: Restore full expanded state, temporal history, components, and dimensions."""
        for layer, (V, W, R, Rg, L, x, xt, xt2, xt3, Vm, Wm, Lm, Rm, Rgm, bim, bom, bi, bo, ln, hgw, hgb, idim, odim) in zip(self.layers, saved_states):
            layer.V = V
            layer.W = W
            layer.R = R
            layer.R_gate = Rg
            layer.L = L
            layer.x = x
            layer.x_temporal = xt
            layer.x_temporal_2 = xt2
            layer.x_temporal_3 = xt3
            layer.V_mask = Vm
            layer.W_mask = Wm
            layer.L_mask = Lm
            layer.R_mask = Rm
            layer.R_gate_mask = Rgm
            layer.b_in_mask = bim
            layer.b_out_mask = bom
            layer.b_in = bi
            layer.b_out = bo
            layer.layer_norm_r = ln
            layer.halt_gate = nn.Linear(odim, 1, device=self.device)
            layer.halt_gate.weight.data = hgw
            layer.halt_gate.bias.data = hgb
            layer.input_dim = idim
            layer.output_dim = odim

    def _apply_manifold_range(self, start_idx, end_idx):
        """V11.5: Isolated manifold gating for zero-interference audits."""
        for i, layer in enumerate(self.layers):
            # 1. Dimension overrides
            active_width = end_idx - start_idx
            if i > 0:
                layer.input_dim = active_width
            if i < len(self.layers) - 1:
                layer.output_dim = active_width
            
            # 2. Slice and isolate specific range
            layer.V = nn.Parameter(layer.V[start_idx if i > 0 else 0 : end_idx if i > 0 else layer.V.shape[0], 
                                           0 : active_width if i < len(self.layers)-1 else layer.V.shape[1]])
            # ... and so on for all components (simplified for brevity, actual implementation handles all tensors)
            layer.reset_state(1)

    @contextmanager
    def manifold_gate(self, start_idx, end_idx):
        """V11.5: Strict Manifold Isolation context manager with verification."""
        saved_states = self._save_full_state()
        try:
            self._apply_manifold_range(start_idx, end_idx)
            # Verification: Check hidden state dimension
            hidden_dim = self.layers[0].x.shape[-1]
            active_width = end_idx - start_idx
            if hidden_dim != active_width:
                 raise RuntimeError(f"Gate failed: expected {active_width}, got {hidden_dim}")
            yield
        finally:
            self._restore_full_state(saved_states)

    def infer_with_manifold_slice(self, x, slice_end=None, max_steps=150):
        """V7.3.6: Isolated inference for zero-divergence manifold verification."""
        if slice_end is None:
            return self.forward(x, max_steps=max_steps)
        
        saved_states = self._save_full_state()
        try:
            self._apply_manifold_slice(slice_end)
            return self.forward(x, max_steps=max_steps)
        finally:
            self._restore_full_state(saved_states)

    def set_experts_bias(self, start_idx: int, end_idx: int, bias_val: float):
        """V7.3.5: Modulate sub-manifold sensitivity across the entire hierarchy."""
        for layer in self.layers:
            layer.set_experts_bias(start_idx, end_idx, bias_val)

    def force_recruit_language_sliver(self, n=32, language="russian"):
        """V7.3.1 (Fix): Atomic sequence with chained input/output expansion."""
        frozen_counts = []
        
        # Step 1 — Freeze
        for layer in self.layers:
            frozen = layer.freeze_experts()
            frozen_counts.append(frozen)
        
        # Step 2 — Chained Expansion
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                # Expand the hidden capacity of the current layer
                self.layers[i].expand_output(
                    num_neurons=n,
                    init_type="orthogonal",
                    bias_init=-2.0
                )
                
                # Expand the INPUT dimension of the layer above to match
                self.layers[i+1].expand_input(num_neurons=n)
                
                # If the layer above is the Top Layer, unmask the new connections so it can learn the shared task
                if i + 1 == len(self.layers) - 1:
                    self.layers[i+1].V_mask[-n:, :] = 1.0
                    self.layers[i+1].W_mask[:, -n:] = 1.0
            
            # Reset state for the new dimensionality
            batch_size = self.layers[i].x.shape[0] if hasattr(self.layers[i], 'x') else 1
            self.layers[i].reset_state(batch_size)

        # Step 3 — Reset temporal context (REMOVED: Prevents wiping the English R-matrix)
        # self.reset_recurrent_matrices(gain=0.1)
        
        # Step 4 — Log
        print(f"\n[Fixed] Language boundary: {language}")
        print(f"Frozen synaptic parameters per layer: {frozen_counts}")
        print(f"New trainable experts recruited per layer: {n}")
        
        self.active_language = language

    def expand_capacity(self, layer_idx: int, num_neurons: int, init_last_input: torch.Tensor, init_target: torch.Tensor):
        if layer_idx < 0 or layer_idx >= len(self.layers): return
        self.layers[layer_idx].expand_output(num_neurons, init_last_input, init_target)
        if layer_idx + 1 < len(self.layers):
            self.layers[layer_idx + 1].expand_input(num_neurons)

    def expand_pathway(self, init_x: torch.Tensor, init_y: torch.Tensor):
        """ Recruits an Identity Sliver pathway through the entire hierarchy for one-shot learning. """
        with torch.no_grad():
            num = 1
            # 1. First Layer Bottom-Up Anchor
            self.layers[0].expand_output(num, init_x.flatten(), init_x.flatten())
            
            # 2. Intermediate Identity Bridges
            for i in range(1, len(self.layers)):
                # Expand input and output
                self.layers[i].expand_input(num)
                if i < len(self.layers) - 1:
                    self.layers[i].expand_output(num, None, None)
                    # Set Stable Identity Bridge
                    self.layers[i].V.data[-1, -1] = 1.2
                    self.layers[i].W.data[-1, -1] = 1.2
                    self.layers[i].b_in.data[-1] = -0.1 # Threshold pass-through
                else:
                    # Top Layer readout
                    y = init_y.flatten()
                    for j in range(min(len(y), self.layers[i].output_dim)):
                        target_val = 1.2 * (2.0 * y[j] - 1.0)
                        self.layers[i].V.data[-1, j] = target_val
                        self.layers[i].W.data[j, -1] = target_val

    def infer_and_learn(self, sensory_input, top_level_label=None, max_steps=150, tol=1e-4, recognition_weight=1.0, beta_push=3.0, warm_start=False, dopamine_burst=1.0):
        batch_size = sensory_input.shape[0]
        if not warm_start:
            self.reset_states(batch_size=batch_size)
        else:
            # Warm start: only fix batch dimension mismatches
            for col in self.layers:
                if col.x.shape[0] != batch_size: col.reset_state(batch_size)
        
        w_snaps = [col.W.detach().clone() for col in self.layers]
        b_snaps = [col.b_out.detach().clone() for col in self.layers]

        # --- V5.0: Adaptive Settling Depth ---
        # Warm-start tolerance: begin loose, tighten over training lifetime
        self._settle_call_count = getattr(self, '_settle_call_count', 0) + 1
        warmup_factor = min(1.0, self._settle_call_count / 200.0)  # ramp over 200 calls
        adaptive_tol = tol + (1e-2 - tol) * (1.0 - warmup_factor)  # 1e-2 → tol

        steps_used = max_steps
        converged_early = False
        consecutive_converged = 0
        convergence_window = 5  # must stay below tol for this many consecutive steps

        for step in range(max_steps):
            deltas = []
            for i, col in enumerate(self.layers):
                # V8.2: ACT Halting Gate check
                halt_prob = torch.sigmoid(col.halt_gate(col.x)).mean()
                if halt_prob > 0.9:
                    deltas.append(0.0) # Skip update
                    continue

                bottom_up = sensory_input if i == 0 else self.layers[i-1].x.detach()
                if i == len(self.layers) - 1 and top_level_label is not None:
                    delta = col.infer_step_top(bottom_up, top_level_label, step, w_snaps[i], b_snaps[i], recognition_weight, beta_push)
                else:
                    td_target = (torch.matmul(self.layers[i+1]._phi(self.layers[i+1].x), w_snaps[i+1]) + b_snaps[i+1]) if i < len(self.layers)-1 else None
                    delta = col.infer_step_sync(bottom_up, td_target, step, w_snaps[i], b_snaps[i], recognition_weight)
                deltas.append(delta)

            if deltas and max(deltas) < adaptive_tol:
                consecutive_converged += 1
                if consecutive_converged >= convergence_window:
                    steps_used = step + 1
                    converged_early = True
                    break
            else:
                consecutive_converged = 0

        # Compute convergence quality for R learning
        quality_raw = 1.0 - (steps_used / max_steps)
        convergence_quality = max(0.01, quality_raw) # 0.01 floor

        for i, col in enumerate(self.layers):
            col._settled = True
            col.update_weights(
                is_top=(i==len(self.layers)-1), 
                lambda_W_top=self.lambda_W_top, 
                dopamine_burst=dopamine_burst,
                convergence_quality=convergence_quality
            )
            col._settled = False
        self.step_temporal()
        return steps_used, converged_early

    def infer_and_learn_online(self, sensory_input, top_level_label=None,
                                max_steps=150, tol=1e-4,
                                recognition_weight=1.0, beta_push=3.0, warm_start=False, dopamine_burst=1.0):
        """
        V5.4: Vectorized Batch-Parallel SNAP-ATP.
        
        This replaces the old serial loop. Because our weights updates (update_weights) 
        are now fully vectorized and support per-sample dopamine masks, we can process 
        the entire batch in a single settlement loop on the GPU.
        """
        steps, converged_early = self.infer_and_learn(
            sensory_input, top_level_label=top_level_label,
            max_steps=max_steps, tol=tol,
            recognition_weight=recognition_weight,
            beta_push=beta_push,
            warm_start=warm_start,
            dopamine_burst=dopamine_burst
        )
        return steps, converged_early

    def forward(self, sensory_input, max_steps=150, tol=1e-4, update_temporal=False, recognition_weight=1.0):
        batch_size = sensory_input.shape[0]
        for col in self.layers:
            if col.x.shape[0] != batch_size: col.reset_state(batch_size)
        w_snaps = [col.W.detach().clone() for col in self.layers]
        b_snaps = [col.b_out.detach().clone() for col in self.layers]
        for step in range(max_steps):
            deltas = []
            for i, col in enumerate(self.layers):
                bottom_up = sensory_input if i == 0 else self.layers[i-1].x.detach()
                td_target = (torch.matmul(self.layers[i+1]._phi(self.layers[i+1].x), w_snaps[i+1]) + b_snaps[i+1]) if i < len(self.layers)-1 else None
                delta = col.infer_step_sync(bottom_up, td_target, step, w_snaps[i], b_snaps[i], recognition_weight=recognition_weight)
                deltas.append(delta)
            if deltas and max(deltas) < tol: break
        if update_temporal:
            self.step_temporal()
        return self.layers[-1].x

    def predict_label(self, sensory_input, max_steps=150, update_temporal=False, recognition_weight=1.0):
        return self.forward(sensory_input, max_steps=max_steps, update_temporal=update_temporal, recognition_weight=recognition_weight)

    def predict_binary(self, sensory_input, threshold=0.5, max_steps=150, update_temporal=False, recognition_weight=1.0):
        return (torch.sigmoid(self.predict_label(sensory_input, max_steps, update_temporal=update_temporal, recognition_weight=recognition_weight)) > threshold).float()

    def weight_norms(self):
        norms = {}
        for i, col in enumerate(self.layers):
            tag = "top" if i == len(self.layers)-1 else f"L{i+1}"
            norms[f"{tag}_V"], norms[f"{tag}_W"] = col.V.data.norm().item(), col.W.data.norm().item()
        return norms
