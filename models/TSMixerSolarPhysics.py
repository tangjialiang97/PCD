import torch

from models.TSMixer import Model as TSMixerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(TSMixerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'TSMixerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_history_context(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if future_zenith is None:
            raise ValueError('TSMixerSolarPhysics requires future_zenith for forecast.')

        mixed_history = x_enc
        for i in range(self.layer):
            mixed_history = self.model[i](mixed_history)
        base_out = self.projection(mixed_history.transpose(1, 2)).transpose(1, 2)

        context = self.history_context(mixed_history.reshape(mixed_history.shape[0], -1))
        return self._apply_physics(context, base_out, future_zenith)

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
