from torch import nn

from layers.wind_physics import WindPhysicsStack
from models.DLinear import Model as DLinearBackbone


class Model(DLinearBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs, individual=getattr(configs, 'individual', False))
        if int(configs.enc_in) != 6:
            raise ValueError('DLinearWindPhysicsV4 expects exactly 6 channels with power as the last channel.')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.enc_in * 32))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        self.history_context = nn.Linear(configs.seq_len * configs.enc_in, self.pred_len * self.physics_hidden_dim)
        self.base_state_embed = nn.Linear(configs.enc_in, self.physics_hidden_dim)
        self.physics_stack = WindPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=int(getattr(configs, 'physics_layers', 2)),
            dropout=getattr(configs, 'dropout', 0.1),
            kernel_size=int(getattr(configs, 'physics_kernel_size', 3)),
            eps=self.physics_eps,
            reference_density=float(getattr(configs, 'wind_reference_density', 1.225)),
        )
        self.latest_raw_power = None
        self.latest_constrained_power = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, power_lower_bound=None, power_upper_bound=None,
                 prev_power=None, state_bin_edges=None, state_ramp_up=None, state_ramp_down=None,
                 density_mean=None, density_scale=None, wind_speed_mean=None, wind_speed_scale=None,
                 power_mean=None, power_scale=None,
                 wind_rated_power=None, wind_cut_in_speed=None, wind_rated_speed=None):
        if power_lower_bound is None or power_upper_bound is None:
            raise ValueError('DLinearWindPhysicsV4 requires power bounds for forecast.')
        if prev_power is None or state_bin_edges is None or state_ramp_up is None or state_ramp_down is None:
            raise ValueError('DLinearWindPhysicsV4 requires previous power and state-dependent ramp tables.')
        if density_mean is None or density_scale is None or wind_speed_mean is None or wind_speed_scale is None:
            raise ValueError('DLinearWindPhysicsV4 requires wind feature normalization stats for forecast.')
        if power_mean is None or power_scale is None:
            raise ValueError('DLinearWindPhysicsV4 requires power normalization stats for forecast.')
        if wind_rated_power is None or wind_cut_in_speed is None or wind_rated_speed is None:
            raise ValueError('DLinearWindPhysicsV4 requires rated-power curve statistics for forecast.')

        base_out = super(Model, self).forecast(x_enc)
        met_state = base_out[:, :, :-1]
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        context = context.reshape(x_enc.shape[0], self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)
        constrained_power, _hidden, raw_physics_power = self.physics_stack(
            context,
            met_state,
            power_lower_bound,
            power_upper_bound,
            prev_power=prev_power,
            state_bin_edges=state_bin_edges,
            state_ramp_up=state_ramp_up,
            state_ramp_down=state_ramp_down,
            density_mean=density_mean,
            density_scale=density_scale,
            wind_speed_mean=wind_speed_mean,
            wind_speed_scale=wind_speed_scale,
            power_mean=power_mean,
            power_scale=power_scale,
            wind_rated_power=wind_rated_power,
            wind_cut_in_speed=wind_cut_in_speed,
            wind_rated_speed=wind_rated_speed,
        )
        self.latest_raw_power = raw_physics_power
        self.latest_constrained_power = constrained_power
        dec_out = base_out.clone()
        dec_out[:, :, -1:] = constrained_power
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                power_lower_bound=None, power_upper_bound=None,
                prev_power=None, state_bin_edges=None, state_ramp_up=None, state_ramp_down=None,
                density_mean=None, density_scale=None, wind_speed_mean=None, wind_speed_scale=None,
                power_mean=None, power_scale=None,
                wind_rated_power=None, wind_cut_in_speed=None, wind_rated_speed=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                power_lower_bound=power_lower_bound,
                power_upper_bound=power_upper_bound,
                prev_power=prev_power,
                state_bin_edges=state_bin_edges,
                state_ramp_up=state_ramp_up,
                state_ramp_down=state_ramp_down,
                density_mean=density_mean,
                density_scale=density_scale,
                wind_speed_mean=wind_speed_mean,
                wind_speed_scale=wind_speed_scale,
                power_mean=power_mean,
                power_scale=power_scale,
                wind_rated_power=wind_rated_power,
                wind_cut_in_speed=wind_cut_in_speed,
                wind_rated_speed=wind_rated_speed,
            )
            self.latest_raw_power = self.latest_raw_power[:, -self.pred_len:, :]
            self.latest_constrained_power = self.latest_constrained_power[:, -self.pred_len:, :]
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
