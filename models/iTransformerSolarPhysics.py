import torch

from models.iTransformer import Model as iTransformerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(iTransformerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'iTransformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_flat_context(configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('iTransformerSolarPhysics requires future_zenith for forecast.')

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _attns = self.encoder(enc_out, attn_mask=None)

        base_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        base_out = base_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        base_out = base_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        context = self.context_head(enc_out.mean(dim=1))
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
