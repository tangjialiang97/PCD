import torch
from torch import nn

from layers.solar_physics import SolarPhysicsStack


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
            use_weak_upper_bound=bool(getattr(configs, 'physics_use_weak_upper_bound', False)),
            solar_constant=float(getattr(configs, 'solar_constant', 1367.0)),
            upper_bound_alpha_min=float(getattr(configs, 'physics_upper_bound_alpha_min', 0.8)),
            upper_bound_alpha_max=float(getattr(configs, 'physics_upper_bound_alpha_max', 1.2)),
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
        dec_out, _hidden = self.physics_stack(context, self._compute_cos_zenith(future_zenith))
        return dec_out
