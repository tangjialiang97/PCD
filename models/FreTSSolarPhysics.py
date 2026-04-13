from models.FreTS import Model as FreTSBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(FreTSBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'FreTSSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_history_context(configs)

    def forecast(self, x_enc, future_zenith=None):
        base_out = super(Model, self).forecast(x_enc)
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        return self._apply_physics(context, base_out, future_zenith)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, future_zenith=future_zenith)
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
