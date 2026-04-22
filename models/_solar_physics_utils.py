import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack


def _cached_tensor(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def cache_solar_outputs(model, base_out, projected_out, future_zenith):
    model.latest_base_out = _cached_tensor(base_out)
    model.latest_projected_out = _cached_tensor(projected_out)
    model.latest_cos_zenith = _cached_tensor(torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0))
    physics_stack = getattr(model, 'physics_stack', None)
    model.latest_projection_trace = getattr(physics_stack, 'latest_trace', None)
    return projected_out


class SolarPhysicsMixin:
    def _init_solar_physics(self, configs, model_name, default_hidden_dim):
        if configs.enc_in != 3:
            raise ValueError(
                f'{model_name} expects exactly 3 target channels: [ghi, dni, dhi].'
            )

        self.physics_model_name = model_name
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', default_hidden_dim))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))

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

    def _init_history_context(self, configs):
        self.history_context = nn.Linear(
            configs.seq_len * configs.enc_in,
            self.pred_len * self.physics_hidden_dim,
        )

    def _init_flat_context(self, input_dim):
        self.context_head = nn.Linear(input_dim, self.pred_len * self.physics_hidden_dim)

    def _init_step_context(self, input_dim):
        self.context_head = nn.Linear(input_dim, self.physics_hidden_dim)

    def _require_future_zenith(self, future_zenith):
        if future_zenith is None:
            raise ValueError(f'{self.physics_model_name} requires future_zenith for forecast.')

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def _apply_physics(self, context, base_out, future_zenith):
        self._require_future_zenith(future_zenith)
        if context.dim() == 2:
            context = context.reshape(base_out.shape[0], self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)
        dec_out, _hidden = self.physics_stack(
            context_seq=context,
            base_state=base_out,
            cos_zenith=self._compute_cos_zenith(future_zenith),
        )
        return cache_solar_outputs(self, base_out, dec_out, future_zenith)
