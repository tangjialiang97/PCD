import torch

from models.Pyraformer import Model as PyraformerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(PyraformerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'PyraformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_history_context(configs)

    def _forecast_with_optional_norm(self, x_enc, x_mark_enc, future_zenith, normalize):
        history = x_enc
        means = None
        stdev = None
        if normalize:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / stdev

        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        base_out = self.projection(enc_out).view(enc_out.size(0), self.pred_len, -1)
        if normalize:
            base_out = base_out * stdev + means

        context = self.history_context(history.reshape(history.shape[0], -1))
        return self._apply_physics(context, base_out, future_zenith)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self._forecast_with_optional_norm(
                x_enc,
                x_mark_enc,
                future_zenith=future_zenith,
                normalize=False,
            )
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'short_term_forecast':
            dec_out = self._forecast_with_optional_norm(
                x_enc,
                x_mark_enc,
                future_zenith=future_zenith,
                normalize=True,
            )
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
