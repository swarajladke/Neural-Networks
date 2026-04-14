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
        
        # Expert Masks to shield pathways from corruption
        self.register_buffer("V_mask", torch.ones_like(self.V))
        self.register_buffer("W_mask", torch.ones_like(self.W))

        self.register_buffer("x", torch.zeros(1, output_dim, device=self.device))

        self.error      = torch.zeros(input_dim, device=self.device)
        self.last_input : torch.Tensor | None = None

        self.tau        = 0.5
        self.eta_x      = 0.8
        self.eta_V      = 0.05    # agile recognition
        self.eta_W      = 0.03    # v4.9: raised from 0.005 to build generative model

        # v4.9: Homeostasis REMOVED — replaced by weight clamping
        self.lambda_V   = 0.0
        self.lambda_W   = 0.0
        self.weight_clamp = 3.0   # v4.9: soft bound on weight magnitudes

        self.lambda_act = 1e-6
        self._settled   = False

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
        else:
            self.x.zero_()

    def infer_step_sync(self, bottom_up, td_target, step_i, W_snap, b_out_snap, recognition_weight=1.0):
        self.last_input = bottom_up.detach()
        x_prev = self.x.clone()
        phi_x  = self._phi(self.x)
        prediction = torch.matmul(phi_x, W_snap) + b_out_snap
        self.error = (bottom_up - prediction).detach()

        feedback_drive    = torch.matmul(self.error, W_snap.t()) * self._phi_deriv(self.x)
        forward_drive     = torch.matmul(bottom_up, self.V) + self.b_in
        recognition_drive = (forward_drive - self.x) * recognition_weight
        top_down_drive    = 3.0 * (td_target - self.x) if td_target is not None else 0.0

        state_grad  = feedback_drive + recognition_drive + top_down_drive
        
        lateral_inhibition = 0.5 * self.x.mean(dim=-1, keepdim=True)
        state_grad -= lateral_inhibition
        state_grad -= self.lambda_act * torch.sign(self.x)

        eta = self.eta_x / (1.0 + 0.1 * step_i)
        dx  = _clip_update(self.tau * eta * state_grad, max_norm=1.0)
        self.x = (self.x + dx).clamp_(-5.0, 5.0)
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

        state_grad  = label_drive + recognition_drive

        lateral_inhibition = 0.5 * self.x.mean(dim=-1, keepdim=True)
        state_grad -= lateral_inhibition
        state_grad -= self.lambda_act * torch.sign(self.x)

        eta = self.eta_x / (1.0 + 0.1 * step_i)
        dx  = _clip_update(self.tau * eta * state_grad, max_norm=1.0)
        self.x = (self.x + dx).clamp_(-5.0, 5.0)
        return torch.max(torch.abs(self.x - x_prev))

    def update_weights(self, is_top=False, lambda_W_top=0.0):
        if not self._settled or self.last_input is None: return
        with torch.no_grad():
            wc = self.weight_clamp
            forward = torch.matmul(self.last_input, self.V) + self.b_in
            mask = self._k_wta_mask(self.x)
            delta_h = (self.x - forward) * mask
            phi_h = self._phi(self.x)

            if self.last_input.dim() == 2:
                # Fast pseudo-online update: parallel settlement but per-sample weight updates
                for i in range(self.x.shape[0]):
                    dV = torch.outer(self.last_input[i], delta_h[i])
                    self.V.data += self.eta_V * _clip_update(dV, max_norm=5.0) * self.V_mask
                    self.b_in.data += self.eta_V * delta_h[i]

                    dW = torch.outer(phi_h[i], self.error[i])
                    self.W.data += self.eta_W * _clip_update(dW, max_norm=5.0) * self.W_mask
                    self.b_out.data += self.eta_W * self.error[i]
            else:
                dV = torch.outer(self.last_input, delta_h)
                self.V.data += self.eta_V * _clip_update(dV, max_norm=5.0) * self.V_mask
                self.b_in.data += self.eta_V * delta_h
                
                dW = torch.outer(phi_h, self.error)
                self.W.data += self.eta_W * _clip_update(dW, max_norm=5.0) * self.W_mask
                self.b_out.data += self.eta_W * self.error

            # ---- v4.9: Weight Clamping ----
            self.V.data.clamp_(-wc, wc)
            self.b_in.data.clamp_(-wc, wc)
            self.W.data.clamp_(-wc, wc)
            self.b_out.data.clamp_(-wc, wc)


    def expand_output(self, num_neurons: int, init_last_input: torch.Tensor, init_target: torch.Tensor):
        with torch.no_grad():
            D_in, D_out = self.input_dim, self.output_dim
            d = num_neurons
            
            # V mapping: input -> hidden. V new = normalized input
            v_new = torch.empty(D_in, d, device=self.device).normal_(0, 0.01)
            bias_new = torch.zeros(d, device=self.device)
            if init_last_input is not None:
                for i in range(min(d, init_last_input.shape[0])):
                    val = init_last_input[i]
                    norm_val = val.norm() + 1e-6
                    v_new[:, i] = val / norm_val
                    # Expert Bias: higher than silent noise (-5.0) but selective
                    bias_new[i] = -0.5
            
            self.V = nn.Parameter(torch.cat([self.V.data, v_new], dim=1))
            self.b_in = nn.Parameter(torch.cat([self.b_in.data, bias_new]))
            
            # W mapping: hidden -> input. W new = exactly target reconstruction
            w_new = torch.empty(d, D_in, device=self.device).normal_(0, 0.01)
            if init_target is not None:
                for i in range(min(d, init_target.shape[0])):
                    w_new[i, :] = init_target[i]
                    
            self.W = nn.Parameter(torch.cat([self.W.data, w_new], dim=0))
            self.output_dim += d
            self.x = torch.zeros(self.x.shape[0], self.output_dim, device=self.device)
            
            # Update Masks: New neurons are masked to 0 for gradients
            self.register_buffer("V_mask", torch.cat([self.V_mask, torch.zeros(D_in, d, device=self.device)], dim=1))
            self.register_buffer("W_mask", torch.cat([self.W_mask, torch.zeros(d, D_in, device=self.device)], dim=0))

    def expand_input(self, num_neurons: int, init_v: torch.Tensor = None, init_w: torch.Tensor = None):
        with torch.no_grad():
            D_in, D_out = self.input_dim, self.output_dim
            d = num_neurons
            
            v_new = torch.empty(d, D_out, device=self.device).normal_(0, 0.01)
            if init_v is not None:
                v_new = init_v
            self.V = nn.Parameter(torch.cat([self.V.data, v_new], dim=0))
            
            w_new = torch.empty(D_out, d, device=self.device).normal_(0, 0.01)
            if init_w is not None:
                w_new = init_w
            self.W = nn.Parameter(torch.cat([self.W.data, w_new], dim=1))
            
            self.b_out = nn.Parameter(torch.cat([self.b_out.data, torch.zeros(d, device=self.device)]))
            self.error = torch.zeros(D_in + d, device=self.device)
            self.input_dim += d
            
            # Update Masks: New input dimensions are masked to 0 for gradients
            self.register_buffer("V_mask", torch.cat([self.V_mask, torch.zeros(d, D_out, device=self.device)], dim=0))
            self.register_buffer("W_mask", torch.cat([self.W_mask, torch.zeros(D_out, d, device=self.device)], dim=1))

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

    def infer_and_learn(self, sensory_input, top_level_label=None, max_steps=150, tol=1e-4, recognition_weight=1.0, beta_push=3.0):
        batch_size = sensory_input.shape[0]
        for col in self.layers: 
            if col.x.shape[0] != batch_size: col.reset_state(batch_size)
        
        w_snaps = [col.W.detach().clone() for col in self.layers]
        b_snaps = [col.b_out.detach().clone() for col in self.layers]

        steps_used = max_steps
        converged = False
        for step in range(max_steps):
            deltas = []
            for i, col in enumerate(self.layers):
                bottom_up = sensory_input if i == 0 else self.layers[i-1].x.detach()
                if i == len(self.layers) - 1 and top_level_label is not None:
                    delta = col.infer_step_top(bottom_up, top_level_label, step, w_snaps[i], b_snaps[i], recognition_weight, beta_push)
                else:
                    td_target = (torch.matmul(self.layers[i+1]._phi(self.layers[i+1].x), w_snaps[i+1]) + b_snaps[i+1]) if i < len(self.layers)-1 else None
                    delta = col.infer_step_sync(bottom_up, td_target, step, w_snaps[i], b_snaps[i], recognition_weight)
                deltas.append(delta)
            if deltas and max(deltas) < tol:
                steps_used = step + 1
                converged = True
                break

        for i, col in enumerate(self.layers):
            col._settled = True
            col.update_weights(is_top=(i==len(self.layers)-1), lambda_W_top=self.lambda_W_top)
            col._settled = False
        return steps_used, converged

    def infer_and_learn_online(self, sensory_input, top_level_label=None,
                                max_steps=150, tol=1e-4,
                                recognition_weight=1.0, beta_push=3.0):
        """
        Online learning: settle and update weights for EACH sample individually.

        Why this matters:
          Batch-averaged outer products cancel out for non-linear tasks.
          For 4-bit parity, patterns [1,1,0,0] (y=0) and [1,0,0,0] (y=1) push
          weights in opposite directions — averaging destroys the signal.

          Online mode preserves per-pattern structure because each sample
          independently settles the hierarchy and updates weights before the
          next sample is seen.  This is also how biological synapses work:
          one experience at a time.
        """
        n = sensory_input.shape[0]
        total_steps = 0
        n_converged = 0

        for i in range(n):
            xi = sensory_input[i:i+1]
            yi = top_level_label[i:i+1] if top_level_label is not None else None

            self.reset_states(batch_size=1)
            steps, conv = self.infer_and_learn(
                xi, top_level_label=yi,
                max_steps=max_steps, tol=tol,
                recognition_weight=recognition_weight,
                beta_push=beta_push
            )
            total_steps += steps
            if conv:
                n_converged += 1

        avg_steps = total_steps / max(1, n)
        return int(avg_steps), n_converged == n

    def forward(self, sensory_input, max_steps=150, tol=1e-4):
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
                delta = col.infer_step_sync(bottom_up, td_target, step, w_snaps[i], b_snaps[i])
                deltas.append(delta)
            if deltas and max(deltas) < tol: break
        return self.layers[-1].x

    def predict_label(self, sensory_input, max_steps=150):
        return self.forward(sensory_input, max_steps=max_steps)

    def predict_binary(self, sensory_input, threshold=0.5, max_steps=150):
        return (torch.sigmoid(self.predict_label(sensory_input, max_steps)) > threshold).float()

    def weight_norms(self):
        norms = {}
        for i, col in enumerate(self.layers):
            tag = "top" if i == len(self.layers)-1 else f"L{i+1}"
            norms[f"{tag}_V"], norms[f"{tag}_W"] = col.V.data.norm().item(), col.W.data.norm().item()
        return norms
