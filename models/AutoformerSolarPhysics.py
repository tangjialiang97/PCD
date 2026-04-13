import torch

from models.Autoformer import Model as AutoformerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(AutoformerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'AutoformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_step_context(configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _attns = self.encoder(enc_out, attn_mask=None)

        dec_hidden = self.dec_embedding(seasonal_init, x_mark_dec)
        for layer in self.decoder.layers:
            dec_hidden, residual_trend = layer(dec_hidden, enc_out, x_mask=None, cross_mask=None)
            trend = trend + residual_trend

        if self.decoder.norm is not None:
            dec_hidden = self.decoder.norm(dec_hidden)

        seasonal_part = self.decoder.projection(dec_hidden) if self.decoder.projection is not None else dec_hidden
        base_out = (trend + seasonal_part)[:, -self.pred_len:, :]
        context = self.context_head(dec_hidden[:, -self.pred_len:, :])
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
