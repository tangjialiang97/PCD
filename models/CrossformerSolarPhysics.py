import torch
from torch import nn
from einops import rearrange, repeat

from layers.solar_physics import SolarPhysicsStack
from models.Crossformer import Model as CrossformerBackbone
from models._solar_physics_utils import cache_solar_outputs


class Model(CrossformerBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        if configs.enc_in != 3:
            raise ValueError('CrossformerSolarPhysics expects exactly 3 target channels: [ghi, dni, dhi].')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.d_model))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.context_head = nn.Linear(
            configs.enc_in * self.out_seg_num * configs.d_model,
            self.pred_len * self.physics_hidden_dim,
        )
        self.base_state_embed = nn.Linear(3, self.physics_hidden_dim)
        self.physics_stack = SolarPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=int(getattr(configs, 'physics_layers', 2)),
            dropout=getattr(configs, 'dropout', 0.1),
            kernel_size=int(getattr(configs, 'physics_kernel_size', 3)),
            night_cos_threshold=self.night_cos_threshold,
            eps=self.physics_eps,
            use_gated_feedback=bool(getattr(configs, 'physics_gated_feedback', False)),
        )

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('CrossformerSolarPhysics requires future_zenith for forecast.')

        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc = x_enc + self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, _attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        base_out = self.decoder(dec_in, enc_out)[:, -self.pred_len:, :]

        context = self.context_head(enc_out[-1].reshape(base_out.shape[0], -1))
        context = context.reshape(base_out.shape[0], self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)
        cos_zenith = self._compute_cos_zenith(future_zenith)
        dec_out, _hidden = self.physics_stack(
            context_seq=context,
            base_state=base_out,
            cos_zenith=cos_zenith,
        )
        return cache_solar_outputs(self, base_out, dec_out, future_zenith)

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
