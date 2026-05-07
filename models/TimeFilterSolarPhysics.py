import torch

from models.TimeFilter import Model as TimeFilterBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(TimeFilterBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'TimeFilterSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_flat_context(
            self.n_vars * self.num_patches * self.dim
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('TimeFilterSolarPhysics requires future_zenith for forecast.')

        batch_size, seq_len, n_vars = x_enc.shape
        enc_out = self.norm(x_enc, 'norm')
        enc_out = enc_out.permute(0, 2, 1).reshape(-1, n_vars * seq_len)
        enc_out = self.patch_embed(enc_out)
        enc_out, _moe_loss = self.backbone(enc_out, self._get_mask(enc_out.device), self.alpha)

        head_features = enc_out.reshape(batch_size, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)
        base_out = self.head(head_features).permute(0, 2, 1)
        base_out = self.norm(base_out, 'denorm')

        context = self.context_head(head_features.reshape(batch_size, -1))
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
