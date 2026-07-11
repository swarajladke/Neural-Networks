"""
hybrid_qpl.py — Fixed-Width Competitive Query Projection Layer with Masked Neurogenesis
========================================================================================
Implements the Query Projection Layer (QPL) with pre-allocated slots,
soft-competition settling dynamics, top-down reconstruction feedback,
hard kWTA sparsification, and local unsupervised updates. Supports group-aware
masked neurogenesis, rolling threshold buffers, bias calibration using medoid anchors,
and state-machine-driven query-specific maturation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Maturation States
INACTIVE = 0
MATURING = 1
MATURE = 2
FAILED = 3

class HybridQPL(nn.Module):
    def __init__(self, input_dim=960, output_dim=128, feedback_gain=0.5, alpha=0.2, temperature=1.0, lateral_bound=0.95):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # Allocated width d_max = 128
        self.feedback_gain = feedback_gain
        self.alpha = alpha
        self.temperature = temperature
        self.lateral_bound = lateral_bound
        self.max_row_norm = 1.0
        
        # We store parameters as torch.float32 parameters and update them manually in torch.no_grad()
        self.V = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=False)
        self.W = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=False)
        self.L = nn.Parameter(torch.zeros(output_dim, output_dim), requires_grad=False)
        
        self.b_in = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.b_out = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        
        # Register diagonal zero mask buffer once to prevent inplace mutation on diagonal of L
        self.register_buffer("diagonal_mask", 1.0 - torch.eye(output_dim))
        
        # Track active mask of slots (shape: d_max)
        self.register_buffer("active_mask", torch.zeros(output_dim, dtype=torch.bool))
        
        # Track usage EMA for homeostasis
        self.register_buffer("usage_ema", torch.zeros(output_dim))
        
        # Maturation State Machine
        self.register_buffer("unit_state", torch.zeros(output_dim, dtype=torch.long))
        self.register_buffer("owner_group", torch.zeros(output_dim, dtype=torch.long))
        self.register_buffer("owner_exposures", torch.zeros(output_dim, dtype=torch.long))
        self.register_buffer("owner_wins", torch.zeros(output_dim, dtype=torch.long))
        self.register_buffer("birth_recon_baseline", torch.zeros(output_dim))
        self.register_buffer("maturation_start_step", torch.zeros(output_dim, dtype=torch.long))
        self.register_buffer("birth_step", torch.zeros(output_dim, dtype=torch.long))
        
        # Diagnostics
        self.fallback_counts = 0
        self.total_samples_processed = 0

    def set_active_slots(self, num_active):
        """Initializes active slots mask with first num_active slots set to True."""
        assert num_active <= self.output_dim
        self.active_mask.zero_()
        self.active_mask[:num_active] = True
        self.usage_ema.zero_()
        self.usage_ema[:num_active] = 1.0 / num_active
        self.unit_state.zero_()
        self.unit_state[:num_active] = MATURE  # Initial units are mature by default

    def initialize_basis(self, num_active):
        """Initializes active slots with orthonormal random basis vectors."""
        self.set_active_slots(num_active)
        g = torch.Generator().manual_seed(42)
        matrix = torch.randn(self.input_dim, num_active, generator=g)
        Q, _ = torch.linalg.qr(matrix, mode="reduced")  # orthonormal basis (input_dim, num_active)
        
        with torch.no_grad():
            self.V.zero_()
            self.W.zero_()
            self.L.zero_()
            self.b_in.zero_()
            self.b_out.zero_()
            
            self.V[:, :num_active] = Q
            self.W[:num_active, :] = Q.T

    def get_effective_L(self):
        """Constructs symmetric, zero-diagonal, active-masked inhibitory lateral matrix."""
        mask_f = self.active_mask.float()
        pair_mask = mask_f[:, None] * mask_f[None, :]
        L_effective = 0.5 * (self.L + self.L.T) * pair_mask * self.diagonal_mask
        return L_effective

    def settle(self, q, variant="full_qpl", k_wta=None, current_group=None):
        """
        Runs iterative settling to compute h_settled.
        Returns (h, converged, steps_taken, fallback_rate).
        """
        B = q.shape[0]
        device = q.device
        mask_f = self.active_mask.float()
        
        # Handle Orthogonal Static baseline (no iterative settling)
        if variant == "orthogonal_static":
            h = F.relu(q @ self.V + self.b_in) * mask_f
            z = F.normalize(h, dim=-1, eps=1e-8)
            return h, torch.ones(B, dtype=torch.bool, device=device), 0, 0.0
            
        L_effective = self.get_effective_L()
        
        # Initialize hidden state
        h = torch.zeros(B, self.output_dim, device=device)
        
        # Sample-level settling tracking
        consecutive_conv = torch.zeros(B, dtype=torch.int, device=device)
        sample_converged = torch.zeros(B, dtype=torch.bool, device=device)
        steps_taken = torch.zeros(B, dtype=torch.int, device=device)
        
        fallback_counts = 0
        total_steps = 50
        
        for step in range(total_steps):
            if sample_converged.all():
                break
                
            q_recon = h @ self.W + self.b_out
            recon_error = q - q_recon
            
            feedback = recon_error @ self.W.T if variant != "orthogonal_static" else torch.zeros_like(h)
            lateral = h @ L_effective.T if "anti_hebbian" in variant or variant == "full_qpl" else torch.zeros_like(h)
            
            drive = q @ self.V + self.b_in
            if variant != "orthogonal_static":
                drive = drive + self.feedback_gain * feedback
            if "anti_hebbian" in variant or variant == "full_qpl":
                drive = drive + lateral
                
            # Activation logic
            if variant == "local_autoencoder":
                candidate = F.relu(drive) * mask_f
                candidate = F.normalize(candidate, p=1, dim=-1, eps=1e-8)
                mass = candidate.sum(dim=-1, keepdim=True)
                if (mass < 1e-4).any():
                    fallback_logits = drive.masked_fill(~self.active_mask.unsqueeze(0), -float("inf"))
                    candidate_fallback = F.softmax(fallback_logits / self.temperature, dim=-1)
                    candidate = torch.where(mass > 1e-4, candidate, candidate_fallback)
                    fallback_counts += (mass < 1e-4).sum().item()
            else:
                # Soft competition variants
                positive = F.relu(drive) * mask_f
                masked_drive = drive.masked_fill(~self.active_mask.unsqueeze(0), -float("inf"))
                weights = F.softmax(masked_drive / self.temperature, dim=-1)
                
                weighted_positive = positive * weights
                mass = weighted_positive.sum(dim=-1, keepdim=True)
                
                candidate_normal = weighted_positive / mass.clamp_min(1e-8)
                fallback_logits = drive.masked_fill(~self.active_mask.unsqueeze(0), -float("inf"))
                candidate_fallback = F.softmax(fallback_logits / self.temperature, dim=-1)
                
                candidate = torch.where(
                    mass > 1e-4,
                    candidate_normal,
                    candidate_fallback
                )
                fallback_counts += (mass < 1e-4).sum().item()
                
            h_next = (1 - self.alpha) * h + self.alpha * candidate * mask_f
            
            # Maturation-shield logic: if query matches the newborn's owner group, boost its activation
            if current_group is not None:
                # Newborns maturing under the current group
                maturing_idx = ((self.unit_state == MATURING) & (self.owner_group == current_group)).nonzero(as_tuple=True)[0]
                if len(maturing_idx) > 0:
                    # Boost drive of maturing newborn unit to ensure it stays in winner pool
                    h_next[:, maturing_idx] = h_next[:, maturing_idx] * 1.5
                    
            # Compute relative convergence per sample
            denom = torch.maximum(h.norm(dim=-1), h_next.norm(dim=-1)).clamp_min(1e-6)
            delta = (h_next - h).norm(dim=-1) / denom
            
            step_conv = (delta < 1e-4) & (~sample_converged)
            consecutive_conv = torch.where(step_conv, consecutive_conv + 1, torch.where(sample_converged, consecutive_conv, torch.zeros_like(consecutive_conv)))
            
            newly_converged = (consecutive_conv >= 3) & (~sample_converged)
            sample_converged = sample_converged | newly_converged
            steps_taken = torch.where(sample_converged, steps_taken, steps_taken + 1)
            
            h = h_next
            
        self.fallback_counts += fallback_counts
        self.total_samples_processed += B
        fallback_rate = fallback_counts / (B * (steps_taken.float().mean().item() + 1e-8))
        
        return h, sample_converged, steps_taken.float().mean().item(), fallback_rate

    def forward(self, q, variant="full_qpl", k_wta=None, current_group=None):
        """Forward pass returns final kWTA normalized representation z and reconstruction q_hat."""
        h, _, _, _ = self.settle(q, variant, k_wta, current_group)
        q_hat = h @ self.W + self.b_out
        
        if variant == "orthogonal_static":
            z = F.normalize(h, dim=-1, eps=1e-8)
            return z, q_hat
            
        if variant == "full_qpl":
            assert k_wta is not None
            scores = h.masked_fill(~self.active_mask.unsqueeze(0), -float("inf"))
            indices = scores.topk(k_wta, dim=-1).indices
            
            kwta_mask = torch.zeros_like(h)
            kwta_mask.scatter_(1, indices, 1.0)
            
            h_sparse = h * kwta_mask
            z = F.normalize(h_sparse, dim=-1, eps=1e-8)
        else:
            z = F.normalize(h, dim=-1, eps=1e-8)
            
        return z, q_hat

    def local_unsupervised_update(self, q, h, kwta_mask=None, lrs=None, current_group=None):
        """
        Runs one step of local unsupervised update for V, W, L, b_in, b_out.
        Operates completely inside torch.no_grad().
        """
        if lrs is None:
            lrs = {"V": 1e-2, "W": 1e-2, "L": 1e-2, "b": 1e-2, "homeo": 1e-3}
            
        B = q.shape[0]
        mask_f = self.active_mask.float()
        pair_mask = mask_f[:, None] * mask_f[None, :]
        offdiag_mask = self.diagonal_mask
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        
        # Calculate reconstruction error
        q_recon = h @ self.W + self.b_out
        recon_error = q - q_recon
        
        # Recognition error
        h_hat = q @ self.V + self.b_in
        delta_h = (h - h_hat) * mask_f
        
        # Updates
        dV = (q.T @ delta_h) / B
        dW = (h.T @ recon_error) / B
        db_in = delta_h.mean(dim=0)
        db_out = recon_error.mean(dim=0)
        
        # Mask updates to ensure inactive units are unmodified
        dV *= mask_f.unsqueeze(0)
        dW *= mask_f.unsqueeze(1)
        db_in *= mask_f
        
        # Maturation-shield logic on lateral weights L
        # Inactive or maturing slots do not update their L rows/columns
        mature_mask = self.active_mask & (self.unit_state == MATURE)
        mature_mask_f = mature_mask.float()
        lateral_train_mask = mature_mask_f[:, None] * mature_mask_f[None, :] * offdiag_mask
        
        # Lateral anti-Hebbian update
        coactivity = (h.T @ h) / B
        dL = -lrs["L"] * coactivity * lateral_train_mask
        
        with torch.no_grad():
            # Apply weights and biases
            self.V.add_(lrs["V"] * dV)
            self.W.add_(lrs["W"] * dW)
            self.b_in.add_(lrs["b"] * db_in)
            self.b_out.add_(lrs["b"] * db_out)
            
            # Constrain b_in and b_out
            self.b_in[active_idx].clamp_(-10.0, 10.0)
            self.b_out.clamp_(-10.0, 10.0)
            
            # Apply 2x learning-rate boost to maturing unit's weights on owner-group observations
            if current_group is not None:
                maturing_idx = ((self.unit_state == MATURING) & (self.owner_group == current_group)).nonzero(as_tuple=True)[0]
                if len(maturing_idx) > 0:
                    # Apply dV and dW updates again for the maturing slots
                    self.V[:, maturing_idx].add_(lrs["V"] * dV[:, maturing_idx])
                    self.W[maturing_idx].add_(lrs["W"] * dW[maturing_idx])
                    self.b_in[maturing_idx].add_(lrs["b"] * db_in[maturing_idx])
            
            # Nonpositive lateral inhibition L
            self.L.add_(dL)
            self.L.clamp_(max=0.0)
            self.L.copy_(0.5 * (self.L + self.L.T))
            self.L.mul_(pair_mask * offdiag_mask)
            
            # Spectral norm check on effective lateral matrix L
            L_eff = self.get_effective_L()
            sigma = torch.linalg.matrix_norm(L_eff, ord=2)
            if sigma > self.lateral_bound:
                self.L.mul_(self.lateral_bound / sigma)
                
            # Normalize active columns of V
            self.V[:, active_idx] = F.normalize(self.V[:, active_idx], dim=0, eps=1e-8)
            
            # Row-norm clipping on W
            W_active = self.W[active_idx]
            row_norm = W_active.norm(dim=1, keepdim=True)
            scale = torch.clamp(self.max_row_norm / row_norm.clamp_min(1e-8), max=1.0)
            self.W[active_idx] = W_active * scale
            
            # Homeostasis usage update (if kwta_mask is provided)
            if kwta_mask is not None:
                winner_frequency = kwta_mask.float().mean(dim=0)
                ema_decay = 0.99
                self.usage_ema.mul_(ema_decay).add_(winner_frequency, alpha=1.0 - ema_decay)
                
                k = kwta_mask.sum(dim=-1).mean().item()
                p_target = k / max(1, len(active_idx))
                
                homeo_delta = lrs["homeo"] * (p_target - self.usage_ema) * mask_f
                self.b_in.add_(homeo_delta)
                self.b_in[active_idx].clamp_(-10.0, 10.0)
                
    def verify_invariants(self):
        """Asserts model invariance criteria for strict reproducibility and execution safety."""
        inactive_mask = ~self.active_mask
        assert torch.all(self.V[:, inactive_mask] == 0.0), "V contains inactive weight leak!"
        assert torch.all(self.W[inactive_mask, :] == 0.0), "W contains inactive weight leak!"
        assert torch.all(self.b_in[inactive_mask] == 0.0), "b_in contains inactive weight leak!"
        
        # Lateral matrix properties
        L_eff = self.get_effective_L()
        assert torch.all(L_eff <= 0.0), "L contains positive (excitatory) entries!"
        assert torch.allclose(L_eff, L_eff.T, atol=1e-6), "L is not symmetric!"
        assert torch.all(torch.diagonal(L_eff) == 0.0), "L contains nonzero diagonal entries!"
        
        # Check maturing unit lateral updates are shielded (their lateral entries must remain zero)
        maturing_idx = (self.unit_state == MATURING).nonzero(as_tuple=True)[0]
        if len(maturing_idx) > 0:
            assert torch.all(self.L[maturing_idx, :] == 0.0), "Maturing unit L row has nonzero entry!"
            assert torch.all(self.L[:, maturing_idx] == 0.0), "Maturing unit L column has nonzero entry!"
            
        sigma = torch.linalg.matrix_norm(L_eff, ord=2)
        assert sigma <= self.lateral_bound + 1e-4, f"L spectral norm {sigma} violates bound!"
        
    def allocate_slot(self, q_anchor, group_id, global_step, target_activation=0.5):
        """Atomically allocates a new slot using the medoid anchor and feedback calibration."""
        assert self.active_mask.any()
        inactive_slots = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(inactive_slots) == 0:
            print("[QPL] Capacity saturated! Maximum 128 active slots reached.")
            return -1
            
        new_idx = inactive_slots[0].item()
        q_norm = F.normalize(q_anchor, dim=-1, eps=1e-8)
        
        # Settle q_norm to get pre-birth reconstruction error
        with torch.no_grad():
            h_pre, _, _, _ = self.settle(q_norm.unsqueeze(0))
            q_recon_pre = h_pre @ self.W + self.b_out
            e_pre = q_norm - q_recon_pre.squeeze(0)
            
            # Bias calibration including reconstruction feedback
            feedforward = torch.dot(q_norm, self.V[:, new_idx])  # initially 0.0
            feedback = self.feedback_gain * torch.dot(e_pre, self.W[new_idx])  # initially 0.0
            calibrated_bias = target_activation - feedforward - feedback
            
            # Atomic transaction updating V, W, L, active_mask, unit_state, owner stats
            self.V[:, new_idx].copy_(q_norm)
            self.W[new_idx].copy_(q_norm)
            
            self.L[new_idx, :].zero_()
            self.L[:, new_idx].zero_()
            
            self.b_in[new_idx] = calibrated_bias
            self.active_mask[new_idx] = True
            
            # Homeostatic target initialization
            active_idx = self.active_mask.nonzero(as_tuple=True)[0]
            self.usage_ema[new_idx] = 1.0 / len(active_idx)
            
            # Maturation state machine setup
            self.unit_state[new_idx] = MATURING
            self.owner_group[new_idx] = group_id
            self.owner_exposures[new_idx] = 0
            self.owner_wins[new_idx] = 0
            self.birth_recon_baseline[new_idx] = e_pre.norm(dim=-1).item()
            self.birth_step[new_idx] = global_step
            
        print(f"[QPL] Slot {new_idx} born for group {group_id} at step {global_step} with calibrated bias {calibrated_bias:.4f}")
        return new_idx

    def update_maturation(self, current_group, current_winners, recon_error):
        """Updates maturation state machine for all maturing units."""
        with torch.no_grad():
            maturing_slots = (self.unit_state == MATURING).nonzero(as_tuple=True)[0]
            for j in maturing_slots:
                idx = j.item()
                if self.owner_group[idx] == current_group:
                    self.owner_exposures[idx] += 1
                    
                    # Check if slot won in current kWTA set
                    if idx in current_winners:
                        self.owner_wins[idx] += 1
                        
                    # Evaluate maturation conditions
                    if self.owner_exposures[idx] >= 20:
                        # Compute relative reconstruction improvement
                        post_recon = recon_error.mean().item()
                        pre_recon = self.birth_recon_baseline[idx]
                        recon_diff = (post_recon - pre_recon) / (pre_recon + 1e-8)
                        
                        # Conditions: exposures >= 20, wins >= 5, reconstruction improved
                        if self.owner_wins[idx] >= 5 and recon_diff < 0:
                            self.unit_state[idx] = MATURE
                            print(f"[QPL] Slot {idx} matured successfully after {self.owner_exposures[idx]} exposures and {self.owner_wins[idx]} wins.")
                        elif self.owner_exposures[idx] >= 50:
                            # Failed birth: does not mature after 50 owner-exposures, deactivate and recycle
                            self.unit_state[idx] = FAILED
                            self.active_mask[idx] = False
                            self.V[:, idx].zero_()
                            self.W[idx].zero_()
                            self.L[idx, :].zero_()
                            self.L[:, idx].zero_()
                            self.b_in[idx] = 0.0
                            print(f"[QPL] Slot {idx} failed maturation criteria and was recycled.")
