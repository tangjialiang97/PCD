import torch
import torch.nn as nn
import torch.nn.functional as F


class SolarPhysicsProjection(nn.Module):
    def __init__(self, hidden_dim, night_cos_threshold=0.0, eps=1e-4,
                 use_weak_upper_bound=False,
                 solar_constant=1367.0,
                 upper_bound_alpha_min=0.8,
                 upper_bound_alpha_max=1.2):
        super(SolarPhysicsProjection, self).__init__()
        self.param_proj = nn.Linear(hidden_dim, 2)
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
        # 从 hidden 中预测两个中间参数，而不是直接回归 [ghi, dni, dhi]。
        # 这样后续可以借助激活函数和物理公式，把输出限制在合理范围内。
        params = self.param_proj(hidden)
        raw_ghi = params[..., 0]
        raw_dhi_ratio = params[..., 1]

        # 用太阳天顶角余弦区分白天/夜晚。夜间辐照应为 0。
        day_mask = (cos_zenith > self.night_cos_threshold).float()
        # 后面会除以 cos_zenith，这里先做下界截断避免数值不稳定。
        safe_cos = cos_zenith.clamp_min(self.eps)

        # softplus 保证 ghi 非负；再乘 day_mask 保证夜间 ghi=0。
        ghi = F.softplus(raw_ghi) * day_mask
        # dhi 通过 ghi * sigmoid(ratio) 构造，保证 0 <= dhi <= ghi。
        dhi = ghi * torch.sigmoid(raw_dhi_ratio)
        # 由闭包关系 GHI = DNI * cos(z) + DHI 反推 dni，并保证 dni 非负。
        dni = torch.relu((ghi - dhi) / safe_cos) * day_mask

        # 重新计算 ghi，确保经过 safe_cos 和 relu 后仍严格满足闭包关系。
        ghi = dni * cos_zenith * day_mask + dhi
        if self.use_weak_upper_bound:
            # 用可学习 alpha 构造一个基于太阳常数的弱上界，避免白天预测过大。
            alpha = self._upper_bound_alpha()
            ghi_upper = self.solar_constant * safe_cos * alpha * day_mask
            # 仅在超过上界时缩放，未超过时 scale=1，不主动放大预测。
            scale = torch.clamp(ghi_upper / ghi.clamp_min(self.eps), max=1.0)
            # 对三个量做同比缩放，保持它们的相对关系。
            ghi = ghi * scale
            dni = dni * scale
            dhi = dhi * scale
            # 缩放后再次做闭包修正，保证最终输出仍物理一致。
            ghi = dni * cos_zenith * day_mask + dhi
        # 输出顺序固定为 [ghi, dni, dhi]。
        return torch.stack([ghi, dni, dhi], dim=-1)


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
        for block in self.blocks:
            hidden, state = block(hidden, state, cos_zenith)
        hidden = self.output_norm(hidden)
        return state, hidden
