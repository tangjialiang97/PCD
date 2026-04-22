import torch
import torch.nn as nn
from types import SimpleNamespace


def _cache_tensor(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def project_to_feasible_set(raw_state, cos_zenith, night_cos_threshold=0.0):
    """Project raw [G, N, D] points onto the per-step solar feasible set.

    The feasible set is:
      * {0} when cos(z) <= tau
      * {(G, N, D): G = c * N + D, N >= 0, D >= 0} otherwise
    """
    if raw_state.shape[-1] != 3:
        raise ValueError('project_to_feasible_set expects raw_state[..., 3].')

    g, n, d = raw_state.unbind(dim=-1)
    c_raw = cos_zenith.to(dtype=raw_state.dtype, device=raw_state.device)
    c = c_raw.clamp_min(0.0)
    zero = torch.zeros_like(g)
    inf = torch.full_like(g, float('inf'))

    denom_main = c.square() + 2.0
    denom_b1 = c.square() + 1.0

    # Interior point on the affine closure plane, valid only when both
    # non-negativity constraints remain active.
    n_int = (c * g + 2.0 * n - c * d) / denom_main
    d_int = (g + (c.square() + 1.0) * d - c * n) / denom_main
    g_int = c * n_int + d_int
    interior = torch.stack([g_int, n_int, d_int], dim=-1)
    interior_valid = (n_int >= 0.0) & (d_int >= 0.0)

    # Boundary D = 0: project onto the DNI ray defined by G = c * N.
    n_b1 = torch.clamp((c * g + n) / denom_b1, min=0.0)
    boundary_d_zero = torch.stack([c * n_b1, n_b1, zero], dim=-1)

    # Boundary N = 0: project onto the DHI ray where G = D.
    d_b2 = torch.clamp((g + d) / 2.0, min=0.0)
    boundary_n_zero = torch.stack([d_b2, zero, d_b2], dim=-1)

    origin = torch.stack([zero, zero, zero], dim=-1)
    candidates = torch.stack([interior, boundary_d_zero, boundary_n_zero, origin], dim=-2)

    sq_dist = (candidates - raw_state.unsqueeze(-2)).square().sum(dim=-1)
    sq_dist[..., 0] = torch.where(interior_valid, sq_dist[..., 0], inf)

    best_idx = sq_dist.argmin(dim=-1, keepdim=True)
    best_candidate = candidates.gather(
        dim=-2,
        index=best_idx.unsqueeze(-1).expand(*best_idx.shape, 3),
    ).squeeze(-2)

    night_mask = c_raw <= float(night_cos_threshold)
    return torch.where(night_mask.unsqueeze(-1), origin, best_candidate)


class SolarEuclideanProjection(nn.Module):
    def __init__(self, night_cos_threshold=0.0, eps=1e-4):
        super().__init__()
        self.night_cos_threshold = night_cos_threshold
        self.eps = eps

    def forward(self, raw_state, cos_zenith):
        projected = project_to_feasible_set(
            raw_state=raw_state,
            cos_zenith=cos_zenith,
            night_cos_threshold=self.night_cos_threshold,
        )
        self.latest_trace = SimpleNamespace(
            raw_state=_cache_tensor(raw_state),
            projected=_cache_tensor(projected),
            cos_zenith=_cache_tensor(cos_zenith),
        )
        return projected


class SolarPhysicsBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3,
                 night_cos_threshold=0.0, eps=1e-4,
                 use_gated_feedback=False):
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
        self.delta_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.projection = SolarEuclideanProjection(
            night_cos_threshold=night_cos_threshold,
            eps=eps,
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
        raw_state = state + self.delta_head(hidden)
        state = self.projection(raw_state, cos_zenith)
        return hidden, state


class SolarPhysicsStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1,
                 kernel_size=3, night_cos_threshold=0.0, eps=1e-4,
                 use_gated_feedback=False):
        super(SolarPhysicsStack, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.initial_projection = SolarEuclideanProjection(
            night_cos_threshold=night_cos_threshold,
            eps=eps,
        )
        self.blocks = nn.ModuleList([
            SolarPhysicsBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=kernel_size,
                night_cos_threshold=night_cos_threshold,
                eps=eps,
                use_gated_feedback=use_gated_feedback,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_seq, base_state, cos_zenith):
        hidden = self.input_norm(self.input_proj(context_seq))
        state = self.initial_projection(base_state, cos_zenith)
        initial_trace = getattr(self.initial_projection, 'latest_trace', None)
        initial_projected = state
        initial_raw = None if initial_trace is None else getattr(initial_trace, 'raw_state', None)
        for block in self.blocks:
            hidden, state = block(hidden, state, cos_zenith)
        hidden = self.output_norm(hidden)
        final_projection = self.blocks[-1].projection if len(self.blocks) > 0 else self.initial_projection
        final_trace = getattr(final_projection, 'latest_trace', None)
        final_raw = None if final_trace is None else getattr(final_trace, 'raw_state', None)
        self.latest_trace = SimpleNamespace(
            initial_raw_state=_cache_tensor(initial_raw),
            initial_projected=_cache_tensor(initial_projected),
            final_raw_state=_cache_tensor(final_raw),
            final_projected=_cache_tensor(state),
            cos_zenith=_cache_tensor(cos_zenith),
        )
        return state, hidden
