import torch

from models.TiDE import Model as TiDEBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(TiDEBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'TiDESolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_history_context(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark, future_zenith=None):
        if batch_y_mark is None:
            batch_y_mark = x_enc.new_zeros((x_enc.shape[0], self.seq_len + self.pred_len, self.feature_dim))
        else:
            batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)

        base_out = torch.stack(
            [
                TiDEBackbone.forecast(self, x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark)
                for feature in range(x_enc.shape[-1])
            ],
            dim=-1,
        )
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
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
        return TiDEBackbone.forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
