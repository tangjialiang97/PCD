import torch

from models.MambaSimple import Model as MambaSimpleBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(MambaSimpleBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'MambaSimpleSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_step_context(configs.d_model)

    def forecast(self, x_enc, x_mark_enc, future_zenith=None):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        hidden = self.embedding(x_enc, x_mark_enc)
        for layer in self.layers:
            hidden = layer(hidden)

        hidden = self.norm(hidden)
        future_hidden = hidden[:, -self.pred_len:, :]
        base_out = self.out_layer(future_hidden)
        base_out = base_out * std_enc + mean_enc

        context = self.context_head(future_hidden)
        return self._apply_physics(context, base_out, future_zenith)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, future_zenith=future_zenith)
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
