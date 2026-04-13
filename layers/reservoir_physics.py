import torch
import torch.nn as nn
import torch.nn.functional as F


class ReservoirPhysicsProjection(nn.Module):
    """Physics-constrained projection for reservoir conservation.

    Predicts (S_{t+1}, I_t, O_t) subject to the discrete-time conservation law:
        S_{t+1} = S_t + delta_t * (I_t - O_t - L_t)
    where L_t >= 0 is a learnable non-negative slack (absorbing evaporation,
    seepage, and unobserved losses).

    The projection works by:
    1. Predicting raw_inflow, raw_outflow, raw_slack from hidden state.
    2. Applying softplus to guarantee non-negativity: I >= 0, O >= 0, L >= 0.
    3. Computing S_{t+1} analytically from the conservation equation.

    This ensures the output *always* satisfies the conservation constraint,
    analogous to how SolarPhysicsProjection enforces the closure relation.
    """

    def __init__(self, hidden_dim, num_vars=3, use_slack=True, eps=1e-6):
        """
        Parameters
        ----------
        hidden_dim : int
            Dimension of the hidden representation.
        num_vars : int
            Number of output variables. 3 = (storage, inflow, outflow),
            4 = (storage, inflow, outflow, evaporation).
        use_slack : bool
            Whether to predict a non-negative slack term L_t to absorb
            unmodelled losses (evaporation, seepage, etc.).
        eps : float
            Small constant for numerical stability.
        """
        super(ReservoirPhysicsProjection, self).__init__()
        self.num_vars = num_vars
        self.use_slack = use_slack
        self.eps = eps

        # Number of raw parameters to predict:
        #   raw_inflow, raw_outflow, (optionally raw_slack)
        num_raw = 2 + (1 if use_slack else 0)
        self.param_proj = nn.Linear(hidden_dim, num_raw)

    def forward(self, hidden, prev_storage, delta_t=1.0):
        """
        Parameters
        ----------
        hidden : Tensor [B, L, H]
            Hidden representation from backbone / previous block.
        prev_storage : Tensor [B, L, 1]
            Previous storage value S_t for each prediction step.
            For step 0 this is the last observed storage; for step k > 0
            it is the storage predicted at step k-1.
        delta_t : float or Tensor
            Time interval between steps (default 1.0 for daily data).

        Returns
        -------
        state : Tensor [B, L, num_vars]
            (storage, inflow, outflow) or (storage, inflow, outflow, evaporation).
            All satisfy conservation and non-negativity constraints.
        """
        params = self.param_proj(hidden)

        # Non-negative inflow and outflow via softplus
        inflow = F.softplus(params[..., 0:1])    # [B, L, 1]
        outflow = F.softplus(params[..., 1:2])    # [B, L, 1]

        if self.use_slack:
            slack = F.softplus(params[..., 2:3])  # [B, L, 1], absorbs evap/seepage
        else:
            slack = torch.zeros_like(inflow)

        # Conservation: S_{t+1} = S_t + delta_t * (I_t - O_t - L_t)
        net_flow = inflow - outflow - slack
        storage_next = prev_storage + delta_t * net_flow

        if self.num_vars == 4:
            # Output evaporation/slack as the 4th variable
            return torch.cat([storage_next, inflow, outflow, slack], dim=-1)
        else:
            return torch.cat([storage_next, inflow, outflow], dim=-1)


class ReservoirPhysicsBlock(nn.Module):
    """One refinement block for reservoir physics, modelled after SolarPhysicsBlock.

    Mixes the current hidden state with physics-aware context (previous state
    + conservation residual), applies temporal convolution and FFN, then
    re-projects through the physics layer.
    """

    def __init__(self, hidden_dim, num_vars=3, dropout=0.1, kernel_size=3,
                 use_slack=True, eps=1e-6):
        super(ReservoirPhysicsBlock, self).__init__()
        padding = kernel_size // 2
        self.num_vars = num_vars
        self.use_slack = use_slack

        # Embed the current state (S, I, O [, E]) into hidden_dim
        self.state_embed = nn.Linear(num_vars, hidden_dim)
        # Embed prev_storage as additional physics context
        self.storage_embed = nn.Linear(1, hidden_dim)

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
        self.projection = ReservoirPhysicsProjection(
            hidden_dim=hidden_dim,
            num_vars=num_vars,
            use_slack=use_slack,
            eps=eps,
        )

    def forward(self, hidden, state, prev_storage, delta_t=1.0):
        """
        Parameters
        ----------
        hidden : [B, L, H]
        state  : [B, L, num_vars]  current (storage, inflow, outflow [, evap])
        prev_storage : [B, L, 1]   previous storage for conservation
        delta_t : float or Tensor
        """
        physics_context = self.state_embed(state) + self.storage_embed(prev_storage)
        mixed_delta = self.temporal_mixer(
            (hidden + physics_context).transpose(1, 2)
        ).transpose(1, 2)
        hidden = self.norm1(hidden + mixed_delta)
        hidden = self.norm2(hidden + self.ffn(hidden))
        state = self.projection(hidden, prev_storage, delta_t)
        return hidden, state


class ReservoirPhysicsStack(nn.Module):
    """Full physics stack for reservoir conservation, modelled after SolarPhysicsStack.

    Architecture:
        input_proj -> input_norm
        -> initial ReservoirPhysicsProjection
        -> N x ReservoirPhysicsBlock
        -> output_norm
    Returns (state, hidden).
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_vars=3,
                 dropout=0.1, kernel_size=3, use_slack=True, eps=1e-6):
        super(ReservoirPhysicsStack, self).__init__()
        self.num_vars = num_vars
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.initial_projection = ReservoirPhysicsProjection(
            hidden_dim=hidden_dim,
            num_vars=num_vars,
            use_slack=use_slack,
            eps=eps,
        )
        self.blocks = nn.ModuleList([
            ReservoirPhysicsBlock(
                hidden_dim=hidden_dim,
                num_vars=num_vars,
                dropout=dropout,
                kernel_size=kernel_size,
                use_slack=use_slack,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context, prev_storage, delta_t=1.0):
        """
        Parameters
        ----------
        context : Tensor [B, L, input_dim]
            Contextual representation from backbone.
        prev_storage : Tensor [B, L, 1]
            Previous storage for each prediction step.
        delta_t : float or Tensor
            Time interval between steps.

        Returns
        -------
        state : Tensor [B, L, num_vars]
        hidden : Tensor [B, L, hidden_dim]
        """
        hidden = self.input_norm(self.input_proj(context))
        state = self.initial_projection(hidden, prev_storage, delta_t)
        for block in self.blocks:
            hidden, state = block(hidden, state, prev_storage, delta_t)
        hidden = self.output_norm(hidden)
        return state, hidden
