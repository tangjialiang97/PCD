import torch
from torch import nn

from models.MICN import Model as MICNBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(MICNBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'MICNSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_step_context(configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)
        future_mark = x_mark_dec[:, -self.pred_len:, :] if x_mark_dec is not None else None
        x_mark_dec_full = (
            torch.cat([x_mark_enc[:, -self.seq_len:, :], future_mark], dim=1)
            if future_mark is not None else None
        )

        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec_full)
        seasonal_part = self.conv_trans(dec_out)[:, -self.pred_len:, :]
        base_out = seasonal_part + trend[:, -self.pred_len:, :]

        context = self.context_head(seasonal_part)
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
