import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack
from models.TimeMixer import Model as TimeMixerBackbone
from models._solar_physics_utils import (
    cache_solar_outputs,
    init_solar_power_mapping,
    select_physics_irradiance,
)


class Model(TimeMixerBackbone):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        if configs.enc_in not in (3, 4):
            raise ValueError('TimeMixerSolarPhysics expects [ghi, dni, dhi] or [ghi, dni, dhi, power].')
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.d_model))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))
        init_solar_power_mapping(self, configs)
        self.history_context = nn.Linear(configs.seq_len * configs.enc_in, self.pred_len * self.physics_hidden_dim)
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
            raise ValueError('TimeMixerSolarPhysics requires future_zenith for forecast.')

        history = x_enc
        x_enc, x_mark_enc = self.__multi_scale_process_icudats(x_enc, x_mark_enc)
        if not isinstance(x_enc, list):
            x_enc = [x_enc]
        if x_mark_enc is not None and not isinstance(x_mark_enc, list):
            x_mark_enc = [x_mark_enc]

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                batch_size, seq_len, n_vars = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(batch_size * n_vars, seq_len, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(n_vars, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                batch_size, seq_len, n_vars = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(batch_size * n_vars, seq_len, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        if len(enc_out_list) > 1:
            for i in range(self.layer):
                enc_out_list = self.pdm_blocks[i](enc_out_list)

        base_out_list = self.future_multi_mixing(history.shape[0], enc_out_list, x_list)
        base_out = torch.stack(base_out_list, dim=-1).sum(-1)
        base_out = self.normalize_layers[0](base_out, 'denorm')
        context = self.history_context(history.reshape(history.shape[0], -1))
        context = context.reshape(history.shape[0], self.pred_len, self.physics_hidden_dim)
        base_out = base_out[:, -self.pred_len:, :]
        base_irradiance = select_physics_irradiance(base_out)
        context = context + self.base_state_embed(base_irradiance)
        cos_zenith = self._compute_cos_zenith(future_zenith)
        dec_out, _hidden = self.physics_stack(
            context_seq=context,
            base_state=base_irradiance,
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
