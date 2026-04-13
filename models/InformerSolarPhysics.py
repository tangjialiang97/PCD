import torch

from models.Informer import Model as InformerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(InformerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'InformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_step_context(configs.d_model)

    def _forecast_with_optional_norm(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith, normalize):
        means = None
        stdev = None
        if normalize:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_hidden = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _attns = self.encoder(enc_out, attn_mask=None)

        for layer in self.decoder.layers:
            dec_hidden = layer(dec_hidden, enc_out, x_mask=None, cross_mask=None)

        if self.decoder.norm is not None:
            dec_hidden = self.decoder.norm(dec_hidden)

        base_out = self.decoder.projection(dec_hidden) if self.decoder.projection is not None else dec_hidden
        base_out = base_out[:, -self.pred_len:, :]
        if normalize:
            base_out = base_out * stdev + means

        context = self.context_head(dec_hidden[:, -self.pred_len:, :])
        return self._apply_physics(context, base_out, future_zenith)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self._forecast_with_optional_norm(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                future_zenith=future_zenith,
                normalize=False,
            )
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'short_term_forecast':
            dec_out = self._forecast_with_optional_norm(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                future_zenith=future_zenith,
                normalize=True,
            )
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
