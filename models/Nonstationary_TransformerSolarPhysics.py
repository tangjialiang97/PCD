import torch

from models.Nonstationary_Transformer import Model as NSBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(NSBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'Nonstationary_TransformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_step_context(configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        x_raw = x_enc.clone().detach()
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec_new = torch.cat(
            [x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
            dim=1,
        ).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_hidden = self.dec_embedding(x_dec_new, x_mark_dec)
        for layer in self.decoder.layers:
            dec_hidden = layer(
                dec_hidden,
                enc_out,
                x_mask=None,
                cross_mask=None,
                tau=tau,
                delta=delta,
            )

        if self.decoder.norm is not None:
            dec_hidden = self.decoder.norm(dec_hidden)

        base_out = self.decoder.projection(dec_hidden) if self.decoder.projection is not None else dec_hidden
        base_out = base_out[:, -self.pred_len:, :]
        base_out = base_out * std_enc + mean_enc

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
