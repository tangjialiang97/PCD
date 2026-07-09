import math

import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack


VALID_POWER_OUTPUT_MODES = {'irradiance', 'power', 'append'}
VALID_POWER_MAP_MODES = {'fixed', 'affine', 'residual_mlp', 'learned'}


def _cached_tensor(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def ghi_to_power(ghi, capacity=1.0, ghi_ref=1000.0, clip=False):
    """Closed-form linear prior: P = clamp(GHI, 0) / ghi_ref * capacity."""
    if ghi_ref <= 0:
        raise ValueError('solar_power_ghi_ref must be positive.')

    normalized_ghi = torch.clamp(ghi, min=0.0) / float(ghi_ref)
    if clip:
        normalized_ghi = torch.clamp(normalized_ghi, max=1.0)
    return normalized_ghi * float(capacity)


def _build_mlp(in_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class SolarPowerMap(nn.Module):
    """Map projected GHI to power with a selectable head.

    Modes
    -----
    fixed:
        P = prior, prior = clamp(GHI,0)/ghi_ref * capacity
    affine:
        P = softplus(a) * prior + b   (init softplus(a)=1, b=0)
    residual_mlp:
        P = prior + MLP([ghi_n, prior, cos_z])  (last layer zero-init)
    learned:
        P = softplus(MLP([ghi_n, cos_z])) * capacity
    """

    def __init__(
        self,
        mode='fixed',
        capacity=1.0,
        ghi_ref=1000.0,
        clip=False,
        hidden_dim=16,
        dropout=0.0,
    ):
        super(SolarPowerMap, self).__init__()
        mode = str(mode).lower()
        if mode not in VALID_POWER_MAP_MODES:
            raise ValueError(
                'solar_power_map must be one of {}, got {!r}.'.format(
                    sorted(VALID_POWER_MAP_MODES),
                    mode,
                )
            )
        if capacity < 0:
            raise ValueError('solar_power_capacity must be non-negative.')
        if ghi_ref <= 0:
            raise ValueError('solar_power_ghi_ref must be positive.')
        if hidden_dim <= 0:
            raise ValueError('solar_power_map_hidden must be positive.')

        self.mode = mode
        self.capacity = float(capacity)
        self.ghi_ref = float(ghi_ref)
        self.clip = bool(clip)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.raw_scale = None
        self.bias = None
        self.mlp = None

        if mode == 'affine':
            # softplus(raw_scale) = 1  =>  raw_scale = log(e - 1)
            init_raw = math.log(math.e - 1.0)
            self.raw_scale = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        elif mode == 'residual_mlp':
            self.mlp = _build_mlp(3, self.hidden_dim, self.dropout)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        elif mode == 'learned':
            self.mlp = _build_mlp(2, self.hidden_dim, self.dropout)
            # softplus(bias)*capacity starts near ~1e-3*capacity (night-safe),
            # not softplus(0.1)≈0.74 which would make power almost constant.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, math.log(math.exp(1e-3) - 1.0))

    def _normalize_ghi(self, ghi):
        normalized = torch.clamp(ghi, min=0.0) / self.ghi_ref
        if self.clip:
            normalized = torch.clamp(normalized, max=1.0)
        return normalized

    def _resolve_cos_zenith(self, ghi, cos_zenith):
        if cos_zenith is None:
            return torch.ones_like(ghi)
        if not torch.is_tensor(cos_zenith):
            cos_zenith = torch.as_tensor(cos_zenith, device=ghi.device, dtype=ghi.dtype)
        cos_zenith = cos_zenith.to(device=ghi.device, dtype=ghi.dtype)
        if cos_zenith.dim() == ghi.dim() - 1:
            cos_zenith = cos_zenith.unsqueeze(-1)
        if cos_zenith.shape[:-1] != ghi.shape[:-1]:
            raise ValueError(
                'cos_zenith shape {} is incompatible with ghi shape {}.'.format(
                    tuple(cos_zenith.shape),
                    tuple(ghi.shape),
                )
            )
        if cos_zenith.shape[-1] != 1:
            cos_zenith = cos_zenith[..., :1]
        return cos_zenith.clamp(min=0.0)

    def forward(self, ghi, cos_zenith=None):
        ghi_n = self._normalize_ghi(ghi)
        prior = ghi_n * self.capacity

        if self.mode == 'fixed':
            power = prior
        elif self.mode == 'affine':
            scale = torch.nn.functional.softplus(self.raw_scale)
            power = scale * prior + self.bias
        elif self.mode == 'residual_mlp':
            cos_z = self._resolve_cos_zenith(ghi, cos_zenith)
            features = torch.cat([ghi_n, prior, cos_z], dim=-1)
            power = prior + self.mlp(features)
        elif self.mode == 'learned':
            cos_z = self._resolve_cos_zenith(ghi, cos_zenith)
            features = torch.cat([ghi_n, cos_z], dim=-1)
            power = torch.nn.functional.softplus(self.mlp(features)) * self.capacity
        else:
            raise RuntimeError('Unhandled solar_power_map mode: {!r}'.format(self.mode))

        return torch.clamp(power, min=0.0)


def init_solar_power_mapping(model, configs):
    output_mode = str(getattr(configs, 'solar_power_output', 'irradiance')).lower()
    if output_mode not in VALID_POWER_OUTPUT_MODES:
        raise ValueError(
            'solar_power_output must be one of {}, got {!r}.'.format(
                sorted(VALID_POWER_OUTPUT_MODES),
                output_mode,
            )
        )

    map_mode = str(getattr(configs, 'solar_power_map', 'fixed')).lower()
    if map_mode not in VALID_POWER_MAP_MODES:
        raise ValueError(
            'solar_power_map must be one of {}, got {!r}.'.format(
                sorted(VALID_POWER_MAP_MODES),
                map_mode,
            )
        )

    capacity = float(getattr(configs, 'solar_power_capacity', 1.0))
    ghi_ref = float(getattr(configs, 'solar_power_ghi_ref', 1000.0))
    clip = bool(getattr(configs, 'solar_power_clip', False))
    hidden_dim = int(getattr(configs, 'solar_power_map_hidden', 16))
    dropout = float(getattr(configs, 'solar_power_map_dropout', 0.0))

    if capacity < 0:
        raise ValueError('solar_power_capacity must be non-negative.')
    if ghi_ref <= 0:
        raise ValueError('solar_power_ghi_ref must be positive.')
    if hidden_dim <= 0:
        raise ValueError('solar_power_map_hidden must be positive.')

    model.solar_power_output = output_mode
    model.solar_power_capacity = capacity
    model.solar_power_ghi_ref = ghi_ref
    model.solar_power_clip = clip
    model.solar_power_map_mode = map_mode
    model.solar_power_map_hidden = hidden_dim
    model.solar_power_map_dropout = dropout

    # Register as a submodule so parameters are picked up by model.parameters().
    model.solar_power_map = SolarPowerMap(
        mode=map_mode,
        capacity=capacity,
        ghi_ref=ghi_ref,
        clip=clip,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def map_projected_ghi_to_power(model, projected_out, cos_zenith=None):
    ghi = projected_out[..., 0:1]
    power_map = getattr(model, 'solar_power_map', None)
    if power_map is not None:
        return power_map(ghi, cos_zenith=cos_zenith)

    # Backward-compatible fallback if init_solar_power_mapping was not called.
    return ghi_to_power(
        ghi,
        capacity=getattr(model, 'solar_power_capacity', 1.0),
        ghi_ref=getattr(model, 'solar_power_ghi_ref', 1000.0),
        clip=getattr(model, 'solar_power_clip', False),
    )


def cache_solar_outputs(model, base_out, projected_out, future_zenith):
    model.latest_base_out = _cached_tensor(base_out)
    model.latest_projected_out = _cached_tensor(projected_out)
    cos_zenith = torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)
    model.latest_cos_zenith = _cached_tensor(cos_zenith)
    physics_stack = getattr(model, 'physics_stack', None)
    model.latest_projection_trace = getattr(physics_stack, 'latest_trace', None)

    power_out = map_projected_ghi_to_power(model, projected_out, cos_zenith=cos_zenith)
    model.latest_power_out = _cached_tensor(power_out)

    output_mode = getattr(model, 'solar_power_output', 'irradiance')
    if output_mode == 'power':
        return power_out
    if output_mode == 'append':
        return torch.cat([projected_out, power_out], dim=-1)
    return projected_out


class SolarPhysicsMixin:
    def _init_solar_physics(self, configs, model_name, default_hidden_dim):
        if configs.enc_in != 3:
            raise ValueError(
                f'{model_name} expects exactly 3 target channels: [ghi, dni, dhi].'
            )

        self.physics_model_name = model_name
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', default_hidden_dim))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.physics_context_mode = str(getattr(configs, 'physics_context_mode', 'baseline')).lower()
        init_solar_power_mapping(self, configs)
        valid_context_modes = {'baseline', 'base_only', 'gated'}
        if self.physics_context_mode not in valid_context_modes:
            raise ValueError(
                'physics_context_mode must be one of {} for {}, got {!r}.'.format(
                    sorted(valid_context_modes),
                    model_name,
                    self.physics_context_mode,
                )
            )

        self.base_state_embed = nn.Linear(3, self.physics_hidden_dim)
        if self.physics_context_mode == 'gated':
            gate_init = float(getattr(configs, 'physics_context_gate_init', 2.0))
            self.physics_context_gate_logit = nn.Parameter(
                torch.full((self.physics_hidden_dim,), gate_init)
            )
        self.physics_stack = SolarPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=int(getattr(configs, 'physics_layers', 2)),
            dropout=getattr(configs, 'dropout', 0.1),
            kernel_size=int(getattr(configs, 'physics_kernel_size', 3)),
            night_cos_threshold=self.night_cos_threshold,
            eps=self.physics_eps,
            use_gated_feedback=bool(getattr(configs, 'physics_gated_feedback', False)),
        )

    def _init_history_context(self, configs):
        self.history_context = nn.Linear(
            configs.seq_len * configs.enc_in,
            self.pred_len * self.physics_hidden_dim,
        )

    def _init_flat_context(self, input_dim):
        self.context_head = nn.Linear(input_dim, self.pred_len * self.physics_hidden_dim)

    def _init_step_context(self, input_dim):
        self.context_head = nn.Linear(input_dim, self.physics_hidden_dim)

    def _require_future_zenith(self, future_zenith):
        if future_zenith is None:
            raise ValueError(f'{self.physics_model_name} requires future_zenith for forecast.')

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def _build_physics_context(self, learned_context, base_out):
        base_context = self.base_state_embed(base_out)
        if self.physics_context_mode == 'base_only':
            return base_context

        if learned_context is None:
            raise RuntimeError(
                'learned_context is required for non-base_only physics_context_mode.'
            )
        if learned_context.dim() == 2:
            learned_context = learned_context.reshape(
                base_out.shape[0], self.pred_len, self.physics_hidden_dim
            )
        if self.physics_context_mode == 'baseline':
            return learned_context + base_context

        gate = torch.sigmoid(self.physics_context_gate_logit).view(1, 1, -1)
        return base_context + gate * learned_context

    def _apply_physics(self, context, base_out, future_zenith):
        self._require_future_zenith(future_zenith)
        context = self._build_physics_context(context, base_out)
        dec_out, _hidden = self.physics_stack(
            context_seq=context,
            base_state=base_out,
            cos_zenith=self._compute_cos_zenith(future_zenith),
        )
        return cache_solar_outputs(self, base_out, dec_out, future_zenith)
