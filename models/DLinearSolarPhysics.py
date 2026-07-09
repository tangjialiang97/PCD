import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack
from models.DLinear import Model as DLinearBackbone
from models._solar_physics_utils import (
    cache_solar_outputs,
    init_solar_power_mapping,
    select_physics_irradiance,
)


class Model(DLinearBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs, individual=getattr(configs, 'individual', False))
        if configs.enc_in not in (3, 4):
            raise ValueError('DLinearSolarPhysics expects [ghi, dni, dhi] or [ghi, dni, dhi, power].')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.enc_in * 32))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.physics_context_mode = str(getattr(configs, 'physics_context_mode', 'baseline')).lower()
        init_solar_power_mapping(self, configs)
        valid_context_modes = {'baseline', 'base_only', 'gated'}
        if self.physics_context_mode not in valid_context_modes:
            raise ValueError(
                'physics_context_mode must be one of {} for DLinearSolarPhysics, got {!r}.'.format(
                    sorted(valid_context_modes),
                    self.physics_context_mode,
                )
            )
        self.base_state_embed = nn.Linear(3, self.physics_hidden_dim)
        self.history_context = None
        if self.physics_context_mode != 'base_only':
            self.history_context = nn.Linear(
                configs.seq_len * configs.enc_in,
                self.pred_len * self.physics_hidden_dim,
            )
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

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def _build_physics_context(self, x_enc, base_out):
        base_irradiance = select_physics_irradiance(base_out)
        base_context = self.base_state_embed(base_irradiance)
        if self.physics_context_mode == 'base_only':
            return base_context

        if self.history_context is None:
            raise RuntimeError('history_context is required for non-base_only physics_context_mode.')
        history_context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        history_context = history_context.reshape(x_enc.shape[0], self.pred_len, self.physics_hidden_dim)
        if self.physics_context_mode == 'baseline':
            return history_context + base_context

        gate = torch.sigmoid(self.physics_context_gate_logit).view(1, 1, -1)
        return base_context + gate * history_context

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('DLinearSolarPhysics requires future_zenith for forecast.')

        base_out = super(Model, self).forecast(x_enc)
        context = self._build_physics_context(x_enc, base_out)
        cos_zenith = self._compute_cos_zenith(future_zenith)
        base_irradiance = select_physics_irradiance(base_out)
        dec_out, _hidden = self.physics_stack(
            context_seq=context,
            base_state=base_irradiance,
            cos_zenith=cos_zenith,
        )
        return cache_solar_outputs(self, base_out, dec_out, future_zenith)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                future_zenith=future_zenith,
            )
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
