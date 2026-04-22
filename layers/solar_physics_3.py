import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


def _cache_tensor(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


class SolarPhysicsProjection(nn.Module):
    """
    对照版本：
    直接从 hidden 回归 [ghi, dni, dhi] 三个量，不再通过
    ghi + dhi_ratio 的参数化方式强制满足闭包关系。

    保留夜间掩码、非负约束和可选弱上界，方便与原版公平对比。
    """

    def __init__(self, hidden_dim, night_cos_threshold=0.0, eps=1e-4,
                 use_weak_upper_bound=False,
                 solar_constant=1367.0,
                 upper_bound_alpha_min=0.8,
                 upper_bound_alpha_max=1.2):
        super(SolarPhysicsProjection, self).__init__()
        self.param_proj = nn.Linear(hidden_dim, 3)
        self.night_cos_threshold = night_cos_threshold
        self.eps = eps
        self.use_weak_upper_bound = use_weak_upper_bound
        self.solar_constant = solar_constant
        self.upper_bound_alpha_min = upper_bound_alpha_min
        self.upper_bound_alpha_max = upper_bound_alpha_max
        if self.use_weak_upper_bound:
            self.upper_bound_alpha_logit = nn.Parameter(torch.zeros(1))

    def _upper_bound_alpha(self):
        if not self.use_weak_upper_bound:
            return None
        alpha_span = self.upper_bound_alpha_max - self.upper_bound_alpha_min
        return self.upper_bound_alpha_min + alpha_span * torch.sigmoid(self.upper_bound_alpha_logit)

    def forward(self, hidden, cos_zenith):
        params = self.param_proj(hidden)
        raw_ghi = params[..., 0]
        raw_dni = params[..., 1]
        raw_dhi = params[..., 2]

        day_mask = (cos_zenith > self.night_cos_threshold).float()
        safe_cos = cos_zenith.clamp_min(self.eps)

        # 直接预测三个分量，仅通过 softplus 保证非负，并用 day_mask 保证夜间为 0。
        ghi = F.softplus(raw_ghi) * day_mask
        dni = F.softplus(raw_dni) * day_mask
        dhi = F.softplus(raw_dhi) * day_mask

        if self.use_weak_upper_bound:
            alpha = self._upper_bound_alpha()
            ghi_upper = self.solar_constant * safe_cos * alpha * day_mask
            scale = torch.clamp(ghi_upper / ghi.clamp_min(self.eps), max=1.0)
            ghi = ghi * scale
            dni = dni * scale
            dhi = dhi * scale

        projected = torch.stack([ghi, dni, dhi], dim=-1)
        self.latest_trace = SimpleNamespace(
            raw_params=_cache_tensor(params),
            projected=_cache_tensor(projected),
            cos_zenith=_cache_tensor(cos_zenith),
        )
        return projected


class SolarPhysicsBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3,
                 night_cos_threshold=0.0, eps=1e-4,
                 use_gated_feedback=False,
                 use_weak_upper_bound=False,
                 solar_constant=1367.0,
                 upper_bound_alpha_min=0.8,
                 upper_bound_alpha_max=1.2):
        super(SolarPhysicsBlock, self).__init__()
        padding = kernel_size // 2
        self.use_gated_feedback = use_gated_feedback
        self.state_embed = nn.Linear(3, hidden_dim)
        self.cos_embed = nn.Linear(1, hidden_dim)
        if self.use_gated_feedback:
            self.feedback_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.projection = SolarPhysicsProjection(
            hidden_dim=hidden_dim,
            night_cos_threshold=night_cos_threshold,
            eps=eps,
            use_weak_upper_bound=use_weak_upper_bound,
            solar_constant=solar_constant,
            upper_bound_alpha_min=upper_bound_alpha_min,
            upper_bound_alpha_max=upper_bound_alpha_max,
        )

    def forward(self, hidden, state, cos_zenith):
        state_features = self.state_embed(state)
        physics_context = state_features + self.cos_embed(cos_zenith.unsqueeze(-1))
        mixed_input = hidden + physics_context

        mixed_delta = self.temporal_mixer(mixed_input.transpose(1, 2)).transpose(1, 2)
        if self.use_gated_feedback:
            feedback_gate = torch.sigmoid(self.feedback_gate(torch.cat([hidden, physics_context], dim=-1)))
            mixed_delta = feedback_gate * mixed_delta

        hidden = self.norm1(hidden + mixed_delta)
        hidden = self.norm2(hidden + self.ffn(hidden))
        state = self.projection(hidden, cos_zenith)
        return hidden, state


class SolarPhysicsStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1,
                 kernel_size=3, night_cos_threshold=0.0, eps=1e-4,
                 use_gated_feedback=False,
                 use_weak_upper_bound=False,
                 solar_constant=1367.0,
                 upper_bound_alpha_min=0.8,
                 upper_bound_alpha_max=1.2):
        super(SolarPhysicsStack, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.initial_projection = SolarPhysicsProjection(
            hidden_dim=hidden_dim,
            night_cos_threshold=night_cos_threshold,
            eps=eps,
            use_weak_upper_bound=use_weak_upper_bound,
            solar_constant=solar_constant,
            upper_bound_alpha_min=upper_bound_alpha_min,
            upper_bound_alpha_max=upper_bound_alpha_max,
        )
        self.blocks = nn.ModuleList([
            SolarPhysicsBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=kernel_size,
                night_cos_threshold=night_cos_threshold,
                eps=eps,
                use_gated_feedback=use_gated_feedback,
                use_weak_upper_bound=use_weak_upper_bound,
                solar_constant=solar_constant,
                upper_bound_alpha_min=upper_bound_alpha_min,
                upper_bound_alpha_max=upper_bound_alpha_max,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context, cos_zenith):
        hidden = self.input_norm(self.input_proj(context))
        state = self.initial_projection(hidden, cos_zenith)
        initial_projected = state
        for block in self.blocks:
            hidden, state = block(hidden, state, cos_zenith)
        hidden = self.output_norm(hidden)
        final_projection = self.blocks[-1].projection if len(self.blocks) > 0 else self.initial_projection
        final_trace = getattr(final_projection, 'latest_trace', None)
        raw_params = None if final_trace is None else getattr(final_trace, 'raw_params', None)
        self.latest_trace = SimpleNamespace(
            raw_params=_cache_tensor(raw_params),
            initial_projected=_cache_tensor(initial_projected),
            final_projected=_cache_tensor(state),
            cos_zenith=_cache_tensor(cos_zenith),
        )
        return state, hidden
