import math

import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack


VALID_POWER_OUTPUT_MODES = {'irradiance', 'power', 'append'}
VALID_POWER_MAP_MODES = {'fixed', 'affine', 'residual_mlp', 'learned', 'ghi_mlp'}
VALID_POWER_PRIOR_MODES = {'linear', 'pvwatts'}
VALID_POWER_FUSION_MODES = {'mapped', 'direct', 'average', 'gated', 'learned'}
VALID_DIRECT_POWER_HEAD_MODES = {'linear', 'mlp'}
PHYSICS_IRRADIANCE_DIM = 3


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


def noct_temp_cell(ghi, temp_amb=20.0, noct=45.0, ghi_noct=800.0):
    """Differentiable NOCT cell-temperature model.

    Tc = Ta + G / G_noct * (NOCT - 20)
    """
    ghi = torch.clamp(ghi, min=0.0)
    return float(temp_amb) + ghi * (float(noct) - 20.0) / float(ghi_noct)


def faiman_temp_cell(ghi, temp_air=20.0, wind_speed=1.0, u0=25.0, u1=6.84):
    """Differentiable Faiman cell-temperature model (pvlib-compatible defaults).

    Tc = Ta + G / (u0 + u1 * ws)
    """
    ghi = torch.clamp(ghi, min=0.0)
    if torch.is_tensor(temp_air):
        ta = temp_air.to(dtype=ghi.dtype, device=ghi.device)
    else:
        ta = torch.as_tensor(float(temp_air), dtype=ghi.dtype, device=ghi.device)
    if torch.is_tensor(wind_speed):
        ws = torch.clamp(wind_speed.to(dtype=ghi.dtype, device=ghi.device), min=0.0)
    else:
        ws = torch.as_tensor(max(float(wind_speed), 0.0), dtype=ghi.dtype, device=ghi.device)
    denom = float(u0) + float(u1) * ws
    denom = torch.clamp(denom, min=1e-3)
    return ta + ghi / denom


def pvwatts_dc(
    ghi,
    temp_cell,
    capacity=1.0,
    ghi_ref=1000.0,
    gamma_pdc=-0.004,
    temp_ref=25.0,
    clip=False,
):
    """Differentiable PVWatts DC model (pvlib.pvsystem.pvwatts_dc equivalent).

    P = pdc0 * (G / G_ref) * (1 + gamma_pdc * (Tc - T_ref))
    """
    if ghi_ref <= 0:
        raise ValueError('solar_power_ghi_ref must be positive.')
    g_eff = torch.clamp(ghi, min=0.0) / float(ghi_ref)
    if clip:
        g_eff = torch.clamp(g_eff, max=1.0)
    if not torch.is_tensor(temp_cell):
        temp_cell = torch.as_tensor(temp_cell, dtype=ghi.dtype, device=ghi.device)
    else:
        temp_cell = temp_cell.to(dtype=ghi.dtype, device=ghi.device)
    thermal = 1.0 + float(gamma_pdc) * (temp_cell - float(temp_ref))
    # Keep power non-negative under extreme heat.
    thermal = torch.clamp(thermal, min=0.0)
    return float(capacity) * g_eff * thermal


def _build_mlp(in_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _softplus_inverse(value):
    value = max(float(value), 1e-8)
    return math.log(math.exp(value) - 1.0)


def _resolve_cos_like(reference, cos_zenith):
    if cos_zenith is None:
        return torch.ones_like(reference)
    if not torch.is_tensor(cos_zenith):
        cos_zenith = torch.as_tensor(cos_zenith, device=reference.device, dtype=reference.dtype)
    cos_zenith = cos_zenith.to(device=reference.device, dtype=reference.dtype)
    if cos_zenith.dim() == reference.dim() - 1:
        cos_zenith = cos_zenith.unsqueeze(-1)
    if cos_zenith.shape[:-1] != reference.shape[:-1]:
        raise ValueError(
            'cos_zenith shape {} is incompatible with reference shape {}.'.format(
                tuple(cos_zenith.shape),
                tuple(reference.shape),
            )
        )
    if cos_zenith.shape[-1] != 1:
        cos_zenith = cos_zenith[..., :1]
    return cos_zenith.clamp(min=0.0)


class SolarPowerMap(nn.Module):
    """Map projected GHI to power with selectable prior + head.

    Priors
    ------
    linear:
        prior = capacity * clamp(GHI,0)/ghi_ref
    pvwatts:
        Tc from NOCT (default) or provided temp_cell
        prior = capacity * (G/ghi_ref) * (1 + gamma*(Tc - temp_ref))

    Heads
    -----
    fixed:
        P = prior
    affine:
        P = softplus(a)*prior + b
    residual_mlp:
        P = prior + MLP([ghi_n, prior_n, cos_z])   # zero-init last layer
    learned:
        P = softplus(MLP([ghi_n, cos_z])) * capacity  # no prior add
    ghi_mlp:
        P = softplus(MLP([ghi_n])) * capacity  # strict GHI-only map
    """

    def __init__(
        self,
        mode='fixed',
        prior='linear',
        capacity=1.0,
        ghi_ref=1000.0,
        clip=False,
        hidden_dim=16,
        dropout=0.0,
        gamma_pdc=-0.004,
        temp_ref=25.0,
        temp_amb=20.0,
        noct=45.0,
        temp_cell_method='noct',
        faiman_u0=25.0,
        faiman_u1=6.84,
        wind_default=1.0,
    ):
        super(SolarPowerMap, self).__init__()
        mode = str(mode).lower()
        prior = str(prior).lower()
        if mode not in VALID_POWER_MAP_MODES:
            raise ValueError(
                'solar_power_map must be one of {}, got {!r}.'.format(
                    sorted(VALID_POWER_MAP_MODES),
                    mode,
                )
            )
        if prior not in VALID_POWER_PRIOR_MODES:
            raise ValueError(
                'solar_power_prior must be one of {}, got {!r}.'.format(
                    sorted(VALID_POWER_PRIOR_MODES),
                    prior,
                )
            )
        if capacity < 0:
            raise ValueError('solar_power_capacity must be non-negative.')
        if ghi_ref <= 0:
            raise ValueError('solar_power_ghi_ref must be positive.')
        if hidden_dim <= 0:
            raise ValueError('solar_power_map_hidden must be positive.')

        self.mode = mode
        self.prior = prior
        self.capacity = float(capacity)
        self.ghi_ref = float(ghi_ref)
        self.clip = bool(clip)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.gamma_pdc = float(gamma_pdc)
        self.temp_ref = float(temp_ref)
        self.temp_amb = float(temp_amb)
        self.noct = float(noct)
        self.temp_cell_method = str(temp_cell_method).lower()
        self.faiman_u0 = float(faiman_u0)
        self.faiman_u1 = float(faiman_u1)
        self.wind_default = float(wind_default)

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
            # softplus(bias)*capacity starts near ~1e-3*capacity (night-safe).
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, math.log(math.exp(1e-3) - 1.0))
        elif mode == 'ghi_mlp':
            self.mlp = _build_mlp(1, self.hidden_dim, self.dropout)
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

    def _resolve_like(self, ghi, value, default):
        if value is None:
            return torch.full_like(ghi, float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=ghi.device, dtype=ghi.dtype)
        value = value.to(device=ghi.device, dtype=ghi.dtype)
        if value.dim() == ghi.dim() - 1:
            value = value.unsqueeze(-1)
        if value.shape[:-1] != ghi.shape[:-1]:
            # Broadcast scalars / wrong trailing dims carefully.
            try:
                value = value.expand_as(ghi)
            except RuntimeError as exc:
                raise ValueError(
                    'aux tensor shape {} is incompatible with ghi shape {}.'.format(
                        tuple(value.shape),
                        tuple(ghi.shape),
                    )
                ) from exc
        if value.shape[-1] != 1:
            value = value[..., :1]
        return value

    def compute_temp_cell(self, ghi, temp_air=None, wind_speed=None, temp_cell=None):
        if temp_cell is not None:
            return self._resolve_like(ghi, temp_cell, self.temp_amb)

        if self.temp_cell_method == 'faiman':
            ta = self._resolve_like(ghi, temp_air, self.temp_amb)
            ws = self._resolve_like(ghi, wind_speed, self.wind_default)
            return faiman_temp_cell(
                ghi,
                temp_air=ta,
                wind_speed=ws,
                u0=self.faiman_u0,
                u1=self.faiman_u1,
            )
        # Default: NOCT from ambient (constant or provided temp_air) + GHI.
        ta = self._resolve_like(ghi, temp_air, self.temp_amb)
        return ta + torch.clamp(ghi, min=0.0) * (self.noct - 20.0) / 800.0

    def compute_prior(self, ghi, temp_air=None, wind_speed=None, temp_cell=None):
        ghi_n = self._normalize_ghi(ghi)
        if self.prior == 'linear':
            return ghi_n * self.capacity

        tc = self.compute_temp_cell(
            ghi,
            temp_air=temp_air,
            wind_speed=wind_speed,
            temp_cell=temp_cell,
        )
        return pvwatts_dc(
            ghi,
            temp_cell=tc,
            capacity=self.capacity,
            ghi_ref=self.ghi_ref,
            gamma_pdc=self.gamma_pdc,
            temp_ref=self.temp_ref,
            clip=self.clip,
        )

    def forward(self, ghi, cos_zenith=None, temp_air=None, wind_speed=None, temp_cell=None):
        ghi_n = self._normalize_ghi(ghi)
        prior = self.compute_prior(
            ghi,
            temp_air=temp_air,
            wind_speed=wind_speed,
            temp_cell=temp_cell,
        )
        # Unitless prior for residual features (scale-stable).
        prior_n = prior / self.capacity if self.capacity > 0 else prior

        if self.mode == 'fixed':
            power = prior
        elif self.mode == 'affine':
            scale = torch.nn.functional.softplus(self.raw_scale)
            power = scale * prior + self.bias
        elif self.mode == 'residual_mlp':
            cos_z = self._resolve_cos_zenith(ghi, cos_zenith)
            features = torch.cat([ghi_n, prior_n, cos_z], dim=-1)
            power = prior + self.mlp(features)
        elif self.mode == 'learned':
            cos_z = self._resolve_cos_zenith(ghi, cos_zenith)
            features = torch.cat([ghi_n, cos_z], dim=-1)
            power = torch.nn.functional.softplus(self.mlp(features)) * self.capacity
        elif self.mode == 'ghi_mlp':
            power = torch.nn.functional.softplus(self.mlp(ghi_n)) * self.capacity
        else:
            raise RuntimeError('Unhandled solar_power_map mode: {!r}'.format(self.mode))

        return torch.clamp(power, min=0.0)


class SolarDirectPowerHead(nn.Module):
    """Predict power directly from the model state when no raw power channel exists.

    If the backbone forecasts four channels, the fourth channel is used as the
    direct power forecast. This head is the fallback for three-channel backbones.
    """

    def __init__(
        self,
        mode='linear',
        capacity=1.0,
        ghi_ref=1000.0,
        clip=False,
        hidden_dim=16,
        dropout=0.0,
        input_dim=7,
    ):
        super(SolarDirectPowerHead, self).__init__()
        mode = str(mode).lower()
        if mode not in VALID_DIRECT_POWER_HEAD_MODES:
            raise ValueError(
                'solar_power_direct_head must be one of {}, got {!r}.'.format(
                    sorted(VALID_DIRECT_POWER_HEAD_MODES),
                    mode,
                )
            )
        if hidden_dim <= 0:
            raise ValueError('solar_power_direct_hidden must be positive.')
        if ghi_ref <= 0:
            raise ValueError('solar_power_ghi_ref must be positive.')

        self.mode = mode
        self.capacity = float(capacity)
        self.ghi_ref = float(ghi_ref)
        self.clip = bool(clip)
        self.hidden_dim = int(hidden_dim)

        if mode == 'linear':
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = _build_mlp(input_dim, self.hidden_dim, dropout)

        last = self.net if isinstance(self.net, nn.Linear) else self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, _softplus_inverse(1e-3))

    def _normalize_irradiance(self, values):
        normalized = torch.clamp(values, min=0.0) / self.ghi_ref
        if self.clip:
            normalized = torch.clamp(normalized, max=1.0)
        return normalized

    def forward(self, base_irradiance, projected_irradiance, cos_zenith=None):
        if base_irradiance.shape[-1] != PHYSICS_IRRADIANCE_DIM:
            raise ValueError(
                'SolarDirectPowerHead expects base_irradiance[..., 3], got {}.'.format(
                    tuple(base_irradiance.shape),
                )
            )
        if projected_irradiance.shape[-1] != PHYSICS_IRRADIANCE_DIM:
            raise ValueError(
                'SolarDirectPowerHead expects projected_irradiance[..., 3], got {}.'.format(
                    tuple(projected_irradiance.shape),
                )
            )
        cos_z = _resolve_cos_like(projected_irradiance[..., 0:1], cos_zenith)
        features = torch.cat(
            [
                self._normalize_irradiance(base_irradiance),
                self._normalize_irradiance(projected_irradiance),
                cos_z,
            ],
            dim=-1,
        )
        return torch.nn.functional.softplus(self.net(features)) * self.capacity


class SolarPowerFusion(nn.Module):
    """Fuse GHI-mapped power and directly forecast power."""

    def __init__(
        self,
        mode='mapped',
        capacity=1.0,
        ghi_ref=1000.0,
        clip=False,
        hidden_dim=16,
        dropout=0.0,
        gate_init=2.0,
    ):
        super(SolarPowerFusion, self).__init__()
        mode = str(mode).lower()
        if mode not in VALID_POWER_FUSION_MODES:
            raise ValueError(
                'solar_power_fusion must be one of {}, got {!r}.'.format(
                    sorted(VALID_POWER_FUSION_MODES),
                    mode,
                )
            )
        if hidden_dim <= 0:
            raise ValueError('solar_power_fusion_hidden must be positive.')
        if ghi_ref <= 0:
            raise ValueError('solar_power_ghi_ref must be positive.')

        self.mode = mode
        self.capacity = float(capacity)
        self.ghi_ref = float(ghi_ref)
        self.clip = bool(clip)
        self.hidden_dim = int(hidden_dim)
        self.gate_init = float(gate_init)
        self.raw_gate = None
        self.gate_net = None
        self.latest_gate = None

        if mode == 'gated':
            self.raw_gate = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
        elif mode == 'learned':
            self.gate_net = nn.Sequential(
                nn.Linear(4, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.constant_(self.gate_net[-1].bias, self.gate_init)

    def _normalize_ghi(self, ghi):
        normalized = torch.clamp(ghi, min=0.0) / self.ghi_ref
        if self.clip:
            normalized = torch.clamp(normalized, max=1.0)
        return normalized

    def _normalize_power(self, power):
        return power / self.capacity if self.capacity > 0 else power

    def forward(self, mapped_power, direct_power, projected_ghi, cos_zenith=None):
        if direct_power is None:
            self.latest_gate = None
            return mapped_power

        if self.mode == 'mapped':
            self.latest_gate = None
            return mapped_power
        if self.mode == 'direct':
            self.latest_gate = None
            return torch.clamp(direct_power, min=0.0)
        if self.mode == 'average':
            self.latest_gate = None
            return torch.clamp(0.5 * (mapped_power + direct_power), min=0.0)

        if self.mode == 'gated':
            gate = torch.sigmoid(self.raw_gate).to(device=mapped_power.device, dtype=mapped_power.dtype)
            self.latest_gate = _cached_tensor(gate)
            return torch.clamp(gate * mapped_power + (1.0 - gate) * direct_power, min=0.0)

        if self.mode == 'learned':
            cos_z = _resolve_cos_like(mapped_power, cos_zenith)
            features = torch.cat(
                [
                    self._normalize_power(mapped_power),
                    self._normalize_power(direct_power),
                    self._normalize_ghi(projected_ghi),
                    cos_z,
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate_net(features))
            self.latest_gate = _cached_tensor(gate)
            return torch.clamp(gate * mapped_power + (1.0 - gate) * direct_power, min=0.0)

        raise RuntimeError('Unhandled solar_power_fusion mode: {!r}'.format(self.mode))


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

    prior_mode = str(getattr(configs, 'solar_power_prior', 'linear')).lower()
    if prior_mode not in VALID_POWER_PRIOR_MODES:
        raise ValueError(
            'solar_power_prior must be one of {}, got {!r}.'.format(
                sorted(VALID_POWER_PRIOR_MODES),
                prior_mode,
            )
        )

    fusion_mode = str(getattr(configs, 'solar_power_fusion', 'mapped')).lower()
    if fusion_mode not in VALID_POWER_FUSION_MODES:
        raise ValueError(
            'solar_power_fusion must be one of {}, got {!r}.'.format(
                sorted(VALID_POWER_FUSION_MODES),
                fusion_mode,
            )
        )

    direct_head_mode = str(getattr(configs, 'solar_power_direct_head', 'linear')).lower()
    if direct_head_mode not in VALID_DIRECT_POWER_HEAD_MODES:
        raise ValueError(
            'solar_power_direct_head must be one of {}, got {!r}.'.format(
                sorted(VALID_DIRECT_POWER_HEAD_MODES),
                direct_head_mode,
            )
        )

    capacity = float(getattr(configs, 'solar_power_capacity', 1.0))
    ghi_ref = float(getattr(configs, 'solar_power_ghi_ref', 1000.0))
    clip = bool(getattr(configs, 'solar_power_clip', False))
    hidden_dim = int(getattr(configs, 'solar_power_map_hidden', 16))
    dropout = float(getattr(configs, 'solar_power_map_dropout', 0.0))
    direct_hidden_dim = int(getattr(configs, 'solar_power_direct_hidden', hidden_dim))
    fusion_hidden_dim = int(getattr(configs, 'solar_power_fusion_hidden', hidden_dim))
    fusion_gate_init = float(getattr(configs, 'solar_power_fusion_gate_init', 2.0))
    gamma_pdc = float(getattr(configs, 'solar_power_gamma_pdc', -0.004))
    temp_ref = float(getattr(configs, 'solar_power_temp_ref', 25.0))
    temp_amb = float(getattr(configs, 'solar_power_temp_amb', 20.0))
    noct = float(getattr(configs, 'solar_power_noct', 45.0))
    temp_cell_method = str(getattr(configs, 'solar_power_temp_cell_method', 'noct')).lower()
    faiman_u0 = float(getattr(configs, 'solar_power_faiman_u0', 25.0))
    faiman_u1 = float(getattr(configs, 'solar_power_faiman_u1', 6.84))
    wind_default = float(getattr(configs, 'solar_power_wind_default', 1.0))

    if capacity < 0:
        raise ValueError('solar_power_capacity must be non-negative.')
    if ghi_ref <= 0:
        raise ValueError('solar_power_ghi_ref must be positive.')
    if hidden_dim <= 0:
        raise ValueError('solar_power_map_hidden must be positive.')
    if direct_hidden_dim <= 0:
        raise ValueError('solar_power_direct_hidden must be positive.')
    if fusion_hidden_dim <= 0:
        raise ValueError('solar_power_fusion_hidden must be positive.')

    model.solar_power_output = output_mode
    model.solar_power_capacity = capacity
    model.solar_power_ghi_ref = ghi_ref
    model.solar_power_clip = clip
    model.solar_power_map_mode = map_mode
    model.solar_power_prior = prior_mode
    model.solar_power_map_hidden = hidden_dim
    model.solar_power_map_dropout = dropout
    model.solar_power_fusion = fusion_mode
    model.solar_power_direct_head_mode = direct_head_mode
    model.solar_power_direct_hidden = direct_hidden_dim
    model.solar_power_fusion_hidden = fusion_hidden_dim
    model.solar_power_fusion_gate_init = fusion_gate_init
    model.solar_power_gamma_pdc = gamma_pdc
    model.solar_power_temp_ref = temp_ref
    model.solar_power_temp_amb = temp_amb
    model.solar_power_noct = noct
    model.solar_power_temp_cell_method = temp_cell_method

    # Register as a submodule so parameters are picked up by model.parameters().
    model.solar_power_map = SolarPowerMap(
        mode=map_mode,
        prior=prior_mode,
        capacity=capacity,
        ghi_ref=ghi_ref,
        clip=clip,
        hidden_dim=hidden_dim,
        dropout=dropout,
        gamma_pdc=gamma_pdc,
        temp_ref=temp_ref,
        temp_amb=temp_amb,
        noct=noct,
        temp_cell_method=temp_cell_method,
        faiman_u0=faiman_u0,
        faiman_u1=faiman_u1,
        wind_default=wind_default,
    )
    model.solar_direct_power_head = SolarDirectPowerHead(
        mode=direct_head_mode,
        capacity=capacity,
        ghi_ref=ghi_ref,
        clip=clip,
        hidden_dim=direct_hidden_dim,
        dropout=dropout,
    )
    model.solar_power_fusion_module = SolarPowerFusion(
        mode=fusion_mode,
        capacity=capacity,
        ghi_ref=ghi_ref,
        clip=clip,
        hidden_dim=fusion_hidden_dim,
        dropout=dropout,
        gate_init=fusion_gate_init,
    )


def map_projected_ghi_to_power(
    model,
    projected_out,
    cos_zenith=None,
    temp_air=None,
    wind_speed=None,
    temp_cell=None,
):
    ghi = projected_out[..., 0:1]
    power_map = getattr(model, 'solar_power_map', None)
    if power_map is not None:
        return power_map(
            ghi,
            cos_zenith=cos_zenith,
            temp_air=temp_air,
            wind_speed=wind_speed,
            temp_cell=temp_cell,
        )

    # Backward-compatible fallback if init_solar_power_mapping was not called.
    return ghi_to_power(
        ghi,
        capacity=getattr(model, 'solar_power_capacity', 1.0),
        ghi_ref=getattr(model, 'solar_power_ghi_ref', 1000.0),
        clip=getattr(model, 'solar_power_clip', False),
    )


def select_physics_irradiance(base_out):
    if base_out.shape[-1] < PHYSICS_IRRADIANCE_DIM:
        raise ValueError(
            'SolarPhysics expects at least three channels [ghi, dni, dhi], got {}.'.format(
                tuple(base_out.shape),
            )
        )
    return base_out[..., :PHYSICS_IRRADIANCE_DIM]


def map_direct_power(model, base_out, projected_out, cos_zenith=None):
    if base_out.shape[-1] >= PHYSICS_IRRADIANCE_DIM + 1:
        return torch.clamp(base_out[..., PHYSICS_IRRADIANCE_DIM:PHYSICS_IRRADIANCE_DIM + 1], min=0.0)

    direct_head = getattr(model, 'solar_direct_power_head', None)
    if direct_head is None:
        return None
    return direct_head(
        select_physics_irradiance(base_out),
        projected_out[..., :PHYSICS_IRRADIANCE_DIM],
        cos_zenith=cos_zenith,
    )


def fuse_solar_power(model, mapped_power, direct_power, projected_out, cos_zenith=None):
    fusion_module = getattr(model, 'solar_power_fusion_module', None)
    if fusion_module is None:
        return mapped_power
    return fusion_module(
        mapped_power,
        direct_power,
        projected_out[..., 0:1],
        cos_zenith=cos_zenith,
    )


def cache_solar_outputs(
    model,
    base_out,
    projected_out,
    future_zenith,
    temp_air=None,
    wind_speed=None,
    temp_cell=None,
):
    model.latest_base_out = _cached_tensor(base_out)
    model.latest_projected_out = _cached_tensor(projected_out)
    cos_zenith = torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)
    model.latest_cos_zenith = _cached_tensor(cos_zenith)
    physics_stack = getattr(model, 'physics_stack', None)
    model.latest_projection_trace = getattr(physics_stack, 'latest_trace', None)

    # Optional weather tensors may be stashed on the model by the exp loop later.
    if temp_air is None:
        temp_air = getattr(model, 'latest_temp_air', None)
    if wind_speed is None:
        wind_speed = getattr(model, 'latest_wind_speed', None)
    if temp_cell is None:
        temp_cell = getattr(model, 'latest_temp_cell', None)

    mapped_power_out = map_projected_ghi_to_power(
        model,
        projected_out,
        cos_zenith=cos_zenith,
        temp_air=temp_air,
        wind_speed=wind_speed,
        temp_cell=temp_cell,
    )
    direct_power_out = map_direct_power(
        model,
        base_out,
        projected_out,
        cos_zenith=cos_zenith,
    )
    power_out = fuse_solar_power(
        model,
        mapped_power_out,
        direct_power_out,
        projected_out,
        cos_zenith=cos_zenith,
    )
    model.latest_mapped_power_out = _cached_tensor(mapped_power_out)
    model.latest_direct_power_out = _cached_tensor(direct_power_out)
    model.latest_power_out = _cached_tensor(power_out)
    power_fusion = getattr(model, 'solar_power_fusion_module', None)
    model.latest_power_fusion_gate = None if power_fusion is None else getattr(power_fusion, 'latest_gate', None)
    power_map = getattr(model, 'solar_power_map', None)
    if power_map is not None:
        model.latest_power_prior = _cached_tensor(
            power_map.compute_prior(
                projected_out[..., 0:1],
                temp_air=temp_air,
                wind_speed=wind_speed,
                temp_cell=temp_cell,
            )
        )

    output_mode = getattr(model, 'solar_power_output', 'irradiance')
    if output_mode == 'power':
        return power_out
    if output_mode == 'append':
        return torch.cat([projected_out, power_out], dim=-1)
    return projected_out


class SolarPhysicsMixin:
    def _init_solar_physics(self, configs, model_name, default_hidden_dim):
        if configs.enc_in not in (3, 4):
            raise ValueError(
                f'{model_name} expects [ghi, dni, dhi] or [ghi, dni, dhi, power].'
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
        base_irradiance = select_physics_irradiance(base_out)
        base_context = self.base_state_embed(base_irradiance)
        if self.physics_context_mode == 'base_only':
            return base_context

        if learned_context is None:
            raise RuntimeError(
                'learned_context is required for non-base_only physics_context_mode.'
            )
        if learned_context.dim() == 2:
            learned_context = learned_context.reshape(
                base_irradiance.shape[0], self.pred_len, self.physics_hidden_dim
            )
        if self.physics_context_mode == 'baseline':
            return learned_context + base_context

        gate = torch.sigmoid(self.physics_context_gate_logit).view(1, 1, -1)
        return base_context + gate * learned_context

    def _apply_physics(self, context, base_out, future_zenith):
        self._require_future_zenith(future_zenith)
        context = self._build_physics_context(context, base_out)
        base_irradiance = select_physics_irradiance(base_out)
        dec_out, _ = self.physics_stack(
            context_seq=context,
            base_state=base_irradiance,
            cos_zenith=self._compute_cos_zenith(future_zenith),
        )
        return cache_solar_outputs(self, base_out, dec_out, future_zenith)
