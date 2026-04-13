import torch

from models.ETSformer import Model as ETSformerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(ETSformerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'ETSformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_step_context(configs.c_out * 3)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)

        level_last = level[:, -1:, :]
        base_out = level_last + growth + season
        context = self.context_head(
            torch.cat([level_last.repeat(1, self.pred_len, 1), growth, season], dim=-1)
        )
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
