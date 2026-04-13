import torch
from torch import nn

from layers.reservoir_physics import ReservoirPhysicsStack
from models.DLinear import Model as DLinearBackbone


class Model(DLinearBackbone):
    """DLinear + reservoir conservation physics decoder.

    Modelled after DLinearSolarPhysics.  The DLinear backbone produces an
    unconstrained forecast for all channels; the ReservoirPhysicsStack then
    maps the output to the conservation-feasible set:

        S_{t+1} = S_t + delta_t * (I_t - O_t - L_t),
        I_t >= 0,  O_t >= 0,  L_t >= 0.

    Expected channel order: [storage, inflow, outflow] (enc_in=3)
    or [storage, inflow, outflow, evaporation] (enc_in=4).

    The model requires ``prev_storage`` at forward time — the last observed
    storage value from the input sequence, used to anchor the conservation
    chain.
    """

    def __init__(self, configs):
        super(Model, self).__init__(configs, individual=getattr(configs, 'individual', False))
        num_vars = int(configs.enc_in)
        if num_vars not in (3, 4):
            raise ValueError(
                f'DLinearReservoirPhysics expects 3 or 4 target channels '
                f'(storage, inflow, outflow [, evaporation]), got {num_vars}.'
            )
        self.num_vars = num_vars
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', num_vars * 32))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-6))
        # Slack defaults to True; only disabled if --reservoir_no_slack is set
        use_slack = bool(getattr(configs, 'reservoir_use_slack', False)) or not bool(getattr(configs, 'reservoir_no_slack', False))
        self.use_slack = use_slack
        self.delta_t = float(getattr(configs, 'reservoir_delta_t', 1.0))

        # Project flattened input history into per-step hidden context
        self.history_context = nn.Linear(
            configs.seq_len * num_vars,
            self.pred_len * self.physics_hidden_dim,
        )
        # Embed the base DLinear output into physics hidden space
        self.base_state_embed = nn.Linear(num_vars, self.physics_hidden_dim)

        self.physics_stack = ReservoirPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=int(getattr(configs, 'physics_layers', 2)),
            num_vars=num_vars,
            dropout=getattr(configs, 'dropout', 0.1),
            kernel_size=int(getattr(configs, 'physics_kernel_size', 3)),
            use_slack=self.use_slack,
            eps=self.physics_eps,
        )

    def _build_prev_storage(self, x_enc, base_out):
        """Construct the prev_storage tensor for each prediction step.

        For an autoregressive conservation chain the ideal approach is to
        feed S_{t} from the previous step.  However, to keep the first
        version simple and fully parallel (no sequential loop), we use
        the last observed storage as prev_storage for *every* step.

        This is a reasonable approximation when pred_len is short relative
        to the storage dynamics, and it keeps the model differentiable
        without scan operations.
        """
        # x_enc: [B, seq_len, num_vars], channel 0 = storage
        last_storage = x_enc[:, -1:, 0:1]  # [B, 1, 1]
        # Expand to [B, pred_len, 1]
        prev_storage = last_storage.expand(-1, self.pred_len, -1)
        return prev_storage

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # DLinear backbone: unconstrained prediction
        base_out = super(Model, self).forecast(x_enc)  # [B, pred_len, num_vars]

        # Build context from input history
        B = x_enc.shape[0]
        context = self.history_context(x_enc.reshape(B, -1))
        context = context.reshape(B, self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)

        # Previous storage for conservation anchoring
        prev_storage = self._build_prev_storage(x_enc, base_out)

        # Physics-constrained decoding
        dec_out, _hidden = self.physics_stack(context, prev_storage, self.delta_t)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
