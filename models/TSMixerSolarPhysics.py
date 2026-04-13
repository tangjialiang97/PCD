import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack
from models.TSMixer import Model as TSMixerBackbone


class Model(TSMixerBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        if configs.enc_in != 3:
            raise ValueError('TSMixerSolarPhysics expects exactly 3 target channels: [ghi, dni, dhi].')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.d_model))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.history_context = nn.Linear(
            configs.seq_len * configs.enc_in,
            self.pred_len * self.physics_hidden_dim,
        )
        self.base_state_embed = nn.Linear(3, self.physics_hidden_dim)
        self.physics_stack = SolarPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=int(getattr(configs, 'physics_layers', 2)),
            dropout=getattr(configs, 'dropout', 0.1),
            kernel_size=int(getattr(configs, 'physics_kernel_size', 3)),
            night_cos_threshold=self.night_cos_threshold,
            eps=self.physics_eps,
            use_gated_feedback=bool(getattr(configs, 'physics_gated_feedback', False)),
            use_weak_upper_bound=bool(getattr(configs, 'physics_use_weak_upper_bound', False)),
            solar_constant=float(getattr(configs, 'solar_constant', 1367.0)),
            upper_bound_alpha_min=float(getattr(configs, 'physics_upper_bound_alpha_min', 0.8)),
            upper_bound_alpha_max=float(getattr(configs, 'physics_upper_bound_alpha_max', 1.2)),
        )

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if future_zenith is None:
            raise ValueError('TSMixerSolarPhysics requires future_zenith for forecast.')

        mixed_history = x_enc
        for i in range(self.layer):
            mixed_history = self.model[i](mixed_history)
        base_out = self.projection(mixed_history.transpose(1, 2)).transpose(1, 2)

        context = self.history_context(mixed_history.reshape(mixed_history.shape[0], -1))
        context = context.reshape(mixed_history.shape[0], self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)
        cos_zenith = self._compute_cos_zenith(future_zenith)
        dec_out, _hidden = self.physics_stack(context, cos_zenith)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                mask=mask,
                future_zenith=future_zenith,
            )
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
