import torch
import torch.nn as nn
import torch.nn.functional as F


class WindPhysicsProjectionV6(nn.Module):
    """Minimal wind-power physics projection, modelled after SolarPhysicsProjection.

    Only two learnable intermediate parameters are predicted from the hidden
    state:
        * raw_power_scale  – multiplicative correction around the cubic power
          curve (via ``1 + 0.2 * tanh(...)``).
        * raw_availability – turbine availability in [0, 1] (via sigmoid).

    The physical wind-speed-to-power mapping is the standard cubic relation
    between *cut_in_speed* and *rated_speed*, clamped at *rated_power*, with a
    hard *cut_out_speed* mask.  No density correction, no learned Cp curve, no
    normalisation round-trips.
    """

    def __init__(self, hidden_dim, eps=1e-4):
        super(WindPhysicsProjectionV6, self).__init__()
        self.param_proj = nn.Linear(hidden_dim, 2)
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.param_proj.weight)
        nn.init.zeros_(self.param_proj.bias)
        # Start near availability=1 so the stack begins close to the raw
        # physics formula (sigmoid(4) ≈ 0.98).
        nn.init.constant_(self.param_proj.bias[1], 4.0)

    @staticmethod
    def _broadcast(value, reference):
        """Broadcast a scalar / 1-D / 2-D parameter to match *reference* [B, L, 1]."""
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=reference.dtype, device=reference.device)
        else:
            value = value.to(dtype=reference.dtype, device=reference.device)
        if value.dim() == 0:
            return value.view(1, 1, 1).expand_as(reference)
        if value.dim() == 1 and value.shape[0] == reference.shape[0]:
            return value.view(reference.shape[0], 1, 1).expand_as(reference)
        if value.dim() == 3:
            return torch.broadcast_to(value, reference.shape)
        return value.view(1, 1, 1).expand_as(reference)

    def forward(self, hidden, wind_speed,
                cut_in_speed, rated_speed, cut_out_speed, rated_power):
        """
        Parameters
        ----------
        hidden : Tensor [B, L, H]
            Hidden representation from the backbone / previous block.
        wind_speed : Tensor [B, L, 1]
            Predicted (normalised) wind speed from the DLinear backbone.
        cut_in_speed, rated_speed, cut_out_speed, rated_power : scalar or Tensor
            Turbine parameters (in the same normalised space as *wind_speed*).

        Returns
        -------
        power : Tensor [B, L, 1]
        """
        cut_in_speed = self._broadcast(cut_in_speed, wind_speed)
        rated_speed = self._broadcast(rated_speed, wind_speed)
        cut_out_speed = self._broadcast(cut_out_speed, wind_speed)
        rated_power = self._broadcast(rated_power, wind_speed)

        # --- learnable corrections ----------------------------------------
        params = self.param_proj(hidden)                       # [B, L, 2]
        power_scale = 1.0 + 0.2 * torch.tanh(params[..., 0:1])  # ~[0.8, 1.2]
        availability = torch.sigmoid(params[..., 1:2])           # [0, 1]

        # --- cubic power curve (hard physics) -----------------------------
        v = wind_speed.clamp_min(0.0)
        ci = cut_in_speed.clamp_min(0.0)
        rs = torch.maximum(rated_speed, ci + self.eps)

        v_cubed = torch.pow(v, 3.0)
        ci_cubed = torch.pow(ci, 3.0)
        rs_cubed = torch.pow(rs, 3.0)

        # Sub-rated region: cubic interpolation between cut-in and rated.
        cubic_ratio = torch.relu(v_cubed - ci_cubed) / (rs_cubed - ci_cubed + self.eps)
        cubic_ratio = cubic_ratio.clamp(max=1.0)

        # Cut-out mask: power drops to zero above cut-out speed.
        cut_out_mask = (v < cut_out_speed).to(v.dtype)

        power = rated_power * cubic_ratio * power_scale * availability * cut_out_mask
        return power


class WindPhysicsBlockV6(nn.Module):
    """One refinement block, modelled after SolarPhysicsBlock.

    Mixes the current hidden state with a physics-aware context (wind speed +
    previous power estimate), applies a temporal convolution and FFN, then
    re-projects through the physics layer.
    """

    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3, eps=1e-4):
        super(WindPhysicsBlockV6, self).__init__()
        padding = kernel_size // 2
        # Embed the 1-D power state and 1-D wind speed into hidden_dim.
        self.state_embed = nn.Linear(1, hidden_dim)
        self.wind_embed = nn.Linear(1, hidden_dim)
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.projection = WindPhysicsProjectionV6(hidden_dim=hidden_dim, eps=eps)

    def forward(self, hidden, state, wind_speed,
                cut_in_speed, rated_speed, cut_out_speed, rated_power):
        """
        Parameters
        ----------
        hidden : [B, L, H]
        state  : [B, L, 1]  – current power estimate
        wind_speed : [B, L, 1]
        cut_in_speed, rated_speed, cut_out_speed, rated_power : scalar / Tensor
        """
        physics_context = self.state_embed(state) + self.wind_embed(wind_speed)
        mixed_delta = self.temporal_mixer(
            (hidden + physics_context).transpose(1, 2)
        ).transpose(1, 2)
        hidden = self.norm1(hidden + mixed_delta)
        hidden = self.norm2(hidden + self.ffn(hidden))
        state = self.projection(
            hidden, wind_speed,
            cut_in_speed, rated_speed, cut_out_speed, rated_power,
        )
        return hidden, state


class WindPhysicsStackV6(nn.Module):
    """Full physics stack, modelled after SolarPhysicsStack.

    Architecture:
        input_proj → input_norm
        → initial WindPhysicsProjectionV6
        → N × WindPhysicsBlockV6
        → output_norm
    Returns ``(power, hidden)``.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1,
                 kernel_size=3, eps=1e-4):
        super(WindPhysicsStackV6, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.initial_projection = WindPhysicsProjectionV6(
            hidden_dim=hidden_dim, eps=eps,
        )
        self.blocks = nn.ModuleList([
            WindPhysicsBlockV6(
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=kernel_size,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context, wind_speed,
                cut_in_speed, rated_speed, cut_out_speed, rated_power):
        hidden = self.input_norm(self.input_proj(context))
        state = self.initial_projection(
            hidden, wind_speed,
            cut_in_speed, rated_speed, cut_out_speed, rated_power,
        )
        for block in self.blocks:
            hidden, state = block(
                hidden, state, wind_speed,
                cut_in_speed, rated_speed, cut_out_speed, rated_power,
            )
        hidden = self.output_norm(hidden)
        return state, hidden
