import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack
from models.TimesNet import Model as TimesNetBackbone
from models._solar_physics_utils import cache_solar_outputs, init_solar_power_mapping


class Model(TimesNetBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        if configs.enc_in != 3:
            raise ValueError('TimesNetSolarPhysics expects exactly 3 target channels: [ghi, dni, dhi].')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.d_model))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        init_solar_power_mapping(self, configs)
        self.context_head = nn.Linear(configs.d_model, self.physics_hidden_dim)
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
        )

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('TimesNetSolarPhysics requires future_zenith for forecast.')

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        base_out = self.projection(enc_out)[:, -self.pred_len:, :]
        base_out = base_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        base_out = base_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        context = self.context_head(enc_out[:, -self.pred_len:, :])
        context = context + self.base_state_embed(base_out)
        cos_zenith = self._compute_cos_zenith(future_zenith)
        dec_out, _hidden = self.physics_stack(
            context_seq=context,
            base_state=base_out,
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
