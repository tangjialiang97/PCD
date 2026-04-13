from layers.wind_physics import WindPhysicsProjection
from models.iTransformer import Model as iTransformerBackbone


class Model(iTransformerBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        if int(configs.enc_in) != 6:
            raise ValueError('iTransformerWindPhysicsV3 expects exactly 6 channels with power as the last channel.')
        self.physics_projection = WindPhysicsProjection()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, power_lower_bound=None, power_upper_bound=None,
                 prev_power=None, state_bin_edges=None, state_ramp_up=None, state_ramp_down=None):
        if power_lower_bound is None or power_upper_bound is None:
            raise ValueError('iTransformerWindPhysicsV3 requires power bounds for forecast.')
        if prev_power is None or state_bin_edges is None or state_ramp_up is None or state_ramp_down is None:
            raise ValueError('iTransformerWindPhysicsV3 requires previous power and state-dependent ramp tables.')

        base_out = super(Model, self).forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = base_out.clone()
        dec_out[:, :, -1:] = self.physics_projection(
            base_out[:, :, -1:],
            power_lower_bound,
            power_upper_bound,
            prev_power=prev_power,
            state_bin_edges=state_bin_edges,
            state_ramp_up=state_ramp_up,
            state_ramp_down=state_ramp_down,
        )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                power_lower_bound=None, power_upper_bound=None,
                prev_power=None, state_bin_edges=None, state_ramp_up=None, state_ramp_down=None):
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
            )
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
