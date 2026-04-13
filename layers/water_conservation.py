import torch
import torch.nn as nn
import torch.nn.functional as F


class WaterConservationProjection(nn.Module):
    def __init__(self, hidden_dim, delta_t=1.0):
        super(WaterConservationProjection, self).__init__()
        self.param_proj = nn.Linear(hidden_dim, 4)
        self.delta_t = float(delta_t)
        if self.delta_t <= 0:
            raise ValueError('delta_t must be positive for water conservation projection.')

    def _prepare_initial_storage(self, initial_storage, hidden):
        if initial_storage.dim() == 2 and initial_storage.shape[-1] == 1:
            initial_storage = initial_storage.squeeze(-1)
        if initial_storage.dim() != 1:
            raise ValueError(
                f'initial_storage must have shape [B] or [B, 1], got {tuple(initial_storage.shape)}'
            )
        return initial_storage.to(hidden.dtype).clamp_min(0.0)

    def forward(self, hidden, initial_storage):
        initial_storage = self._prepare_initial_storage(initial_storage, hidden)

        params = self.param_proj(hidden)
        raw_inflow = params[..., 0]
        raw_alloc = params[..., 1:]

        inflow = F.softplus(raw_inflow)
        alloc = torch.softmax(raw_alloc, dim=-1)
        p_out = alloc[..., 0]
        p_evap = alloc[..., 1]
        p_keep = alloc[..., 2]

        delta_t = hidden.new_tensor(self.delta_t)
        storage_prev = initial_storage
        outputs = []
        for step in range(hidden.shape[1]):
            inflow_step = inflow[:, step]
            available = storage_prev + delta_t * inflow_step
            outflow_step = available * p_out[:, step] / delta_t
            evaporation_step = available * p_evap[:, step] / delta_t
            storage_step = available * p_keep[:, step]
            outputs.append(
                torch.stack([storage_step, inflow_step, outflow_step, evaporation_step], dim=-1)
            )
            storage_prev = storage_step

        return torch.stack(outputs, dim=1)


class WaterConservationBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3, delta_t=1.0):
        super(WaterConservationBlock, self).__init__()
        padding = kernel_size // 2
        self.state_embed = nn.Linear(4, hidden_dim)
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
        self.projection = WaterConservationProjection(hidden_dim=hidden_dim, delta_t=delta_t)

    def forward(self, hidden, state, initial_storage):
        physics_context = self.state_embed(state)
        mixed_input = hidden + physics_context
        mixed_delta = self.temporal_mixer(mixed_input.transpose(1, 2)).transpose(1, 2)
        hidden = self.norm1(hidden + mixed_delta)
        hidden = self.norm2(hidden + self.ffn(hidden))
        state = self.projection(hidden, initial_storage)
        return hidden, state


class WaterConservationStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1,
                 kernel_size=3, delta_t=1.0):
        super(WaterConservationStack, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.initial_projection = WaterConservationProjection(
            hidden_dim=hidden_dim,
            delta_t=delta_t,
        )
        self.blocks = nn.ModuleList([
            WaterConservationBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=kernel_size,
                delta_t=delta_t,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context, initial_storage):
        hidden = self.input_norm(self.input_proj(context))
        state = self.initial_projection(hidden, initial_storage)
        for block in self.blocks:
            hidden, state = block(hidden, state, initial_storage)
        hidden = self.output_norm(hidden)
        return state, hidden
