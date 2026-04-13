import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack
from models.TimeFilter import Model as TimeFilterBackbone


class Model(TimeFilterBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        if configs.enc_in != 3:
            raise ValueError('TimeFilterSolarPhysics expects exactly 3 target channels: [ghi, dni, dhi].')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.d_model))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.context_head = nn.Linear(
            self.n_vars * self.num_patches * self.dim,
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
            use_weak_upper_bound=bool(getattr(configs, 'physics_use_weak_upper_bound', False)),
            solar_constant=float(getattr(configs, 'solar_constant', 1367.0)),
            upper_bound_alpha_min=float(getattr(configs, 'physics_upper_bound_alpha_min', 0.8)),
            upper_bound_alpha_max=float(getattr(configs, 'physics_upper_bound_alpha_max', 1.2)),
        )

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

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
        context = context.reshape(batch_size, self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)
        cos_zenith = self._compute_cos_zenith(future_zenith)
        dec_out, _hidden = self.physics_stack(context, cos_zenith)
        return dec_out

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
