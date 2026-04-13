import torch
from torch import nn

from models.TimeXer import Model as TimeXerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(TimeXerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'TimeXerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self.context_head = nn.Linear(
            configs.enc_in * self.head_nf,
            self.pred_len * self.physics_hidden_dim,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        means = None
        stdev = None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        base_out = self.head(enc_out).permute(0, 2, 1)
        if self.use_norm:
            base_out = base_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            base_out = base_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        context = self.context_head(enc_out.reshape(enc_out.shape[0], -1))
        context = context.reshape(enc_out.shape[0], self.pred_len, self.physics_hidden_dim)
        return self._apply_physics(context, base_out, future_zenith)

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
