import torch
from einops import rearrange

from models.PAttn import Model as PAttnBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(PAttnBackbone, SolarPhysicsMixin):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__(configs, patch_len=patch_len, stride=stride)
        self.task_name = configs.task_name
        self._init_solar_physics(
            configs,
            'PAttnSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_flat_context(
            configs.enc_in * self.d_model * self.patch_num
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('PAttnSolarPhysics requires future_zenith for forecast.')

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        batch_size, _, n_vars = x_enc.shape
        enc_out = x_enc.permute(0, 2, 1)
        enc_out = self.padding_patch_layer(enc_out)
        enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        enc_out = self.in_layer(enc_out)
        enc_out = rearrange(enc_out, 'b c m l -> (b c) m l')
        enc_out, _attns = self.encoder(enc_out)
        enc_out = rearrange(enc_out, '(b c) m l -> b c (m l)', b=batch_size, c=n_vars)

        base_out = self.out_layer(enc_out).permute(0, 2, 1)
        base_out = base_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        base_out = base_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        context = self.context_head(enc_out.reshape(batch_size, -1))
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
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
