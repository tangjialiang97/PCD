import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack


def _cached_tensor(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def ghi_to_power(ghi, capacity=1.0, ghi_ref=1000.0, clip=False):
    if ghi_ref <= 0:
        raise ValueError('solar_power_ghi_ref must be positive.')

    normalized_ghi = torch.clamp(ghi, min=0.0) / float(ghi_ref)
    if clip:
        normalized_ghi = torch.clamp(normalized_ghi, max=1.0)
    return normalized_ghi * float(capacity)


def init_solar_power_mapping(model, configs):
    output_mode = str(getattr(configs, 'solar_power_output', 'irradiance')).lower()
    valid_output_modes = {'irradiance', 'power', 'append'}
    if output_mode not in valid_output_modes:
        raise ValueError(
            'solar_power_output must be one of {}, got {!r}.'.format(
                sorted(valid_output_modes),
                output_mode,
            )
    )

    model.solar_power_output = output_mode
    model.solar_power_capacity = float(getattr(configs, 'solar_power_capacity', 1.0))
    model.solar_power_ghi_ref = float(getattr(configs, 'solar_power_ghi_ref', 1000.0))
    model.solar_power_clip = bool(getattr(configs, 'solar_power_clip', False))

    if model.solar_power_capacity < 0:
        raise ValueError('solar_power_capacity must be non-negative.')
    if model.solar_power_ghi_ref <= 0:
        raise ValueError('solar_power_ghi_ref must be positive.')


def map_projected_ghi_to_power(model, projected_out):
    ghi = projected_out[..., 0:1]
    return ghi_to_power(
        ghi,
        capacity=getattr(model, 'solar_power_capacity', 1.0),
        ghi_ref=getattr(model, 'solar_power_ghi_ref', 1000.0),
        clip=getattr(model, 'solar_power_clip', False),
    )


def cache_solar_outputs(model, base_out, projected_out, future_zenith):
    model.latest_base_out = _cached_tensor(base_out)
    model.latest_projected_out = _cached_tensor(projected_out)
    model.latest_cos_zenith = _cached_tensor(torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0))
    physics_stack = getattr(model, 'physics_stack', None)
    model.latest_projection_trace = getattr(physics_stack, 'latest_trace', None)

    power_out = map_projected_ghi_to_power(model, projected_out)
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
