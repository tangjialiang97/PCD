from torch import nn

from layers.wind_physics import SimplifiedWindPhysicsStack
from models.DLinear import Model as DLinearBackbone


class Model(DLinearBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs, individual=getattr(configs, 'individual', False))
        if int(configs.enc_in) != 6:
            raise ValueError('DLinearWindPhysicsV5 expects exactly 6 channels with power as the last channel.')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.enc_in * 32))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.wind_speed_channel = 4
        self.history_context = nn.Linear(configs.seq_len * configs.enc_in, self.pred_len * self.physics_hidden_dim)
        self.base_state_embed = nn.Linear(configs.enc_in, self.physics_hidden_dim)
        self.physics_stack = SimplifiedWindPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=int(getattr(configs, 'physics_layers', 2)),
            dropout=getattr(configs, 'dropout', 0.1),
            eps=self.physics_eps,
            cp_mode=getattr(configs, 'wind_cp_mode', 'none'),
            cp_max=float(getattr(configs, 'wind_cp_max', 0.48)),
            cp_optimal_ratio=float(getattr(configs, 'wind_cp_optimal_ratio', 1.0)),
            cp_width=float(getattr(configs, 'wind_cp_width', 0.35)),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                 wind_cut_in_speed_norm=None, wind_rated_speed_norm=None,
                 wind_cut_out_speed_norm=None, wind_rated_power_norm=None):
        if wind_cut_in_speed_norm is None or wind_rated_speed_norm is None:
            raise ValueError('DLinearWindPhysicsV5 requires normalized cut-in and rated speeds for forecast.')
        if wind_cut_out_speed_norm is None or wind_rated_power_norm is None:
            raise ValueError('DLinearWindPhysicsV5 requires normalized cut-out speed and rated power for forecast.')

        base_out = super(Model, self).forecast(x_enc)
        wind_speed = base_out[:, :, self.wind_speed_channel:self.wind_speed_channel + 1]
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        context = context.reshape(x_enc.shape[0], self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)
        power, _hidden = self.physics_stack(
            context,
            wind_speed,
            wind_cut_in_speed_norm,
            wind_rated_speed_norm,
            wind_cut_out_speed_norm,
            wind_rated_power_norm,
        )
        dec_out = base_out.clone()
        dec_out[:, :, -1:] = power
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                wind_cut_in_speed_norm=None, wind_rated_speed_norm=None,
                wind_cut_out_speed_norm=None, wind_rated_power_norm=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                wind_cut_in_speed_norm=wind_cut_in_speed_norm,
                wind_rated_speed_norm=wind_rated_speed_norm,
                wind_cut_out_speed_norm=wind_cut_out_speed_norm,
                wind_rated_power_norm=wind_rated_power_norm,
            )
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
