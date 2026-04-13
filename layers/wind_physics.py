import torch
import torch.nn as nn
import torch.nn.functional as F


class WindPhysicsProjection(nn.Module):
    def __init__(self):
        super(WindPhysicsProjection, self).__init__()

    def _project_interval(self, raw_power, lower_bound, upper_bound):
        if lower_bound is None or upper_bound is None:
            raise ValueError('WindPhysicsProjection requires both lower and upper bounds.')
        if raw_power.shape != lower_bound.shape or raw_power.shape != upper_bound.shape:
            raise ValueError(
                'WindPhysicsProjection expects raw_power, lower_bound, and upper_bound '
                f'to share the same shape, got {raw_power.shape}, {lower_bound.shape}, {upper_bound.shape}.'
            )
        projected = torch.minimum(raw_power, upper_bound)
        projected = torch.maximum(projected, lower_bound)
        return projected

    def _select_state_ramps(self, prev_power, state_bin_edges, state_ramp_up, state_ramp_down):
        if state_bin_edges is None or state_ramp_up is None or state_ramp_down is None:
            raise ValueError('State-dependent wind projection requires bin edges and per-bin ramp tables.')
        if state_bin_edges.dim() != 1 or state_ramp_up.dim() != 1 or state_ramp_down.dim() != 1:
            raise ValueError(
                'WindPhysicsProjection expects state bin edges and ramp tables to be 1D tensors, got '
                f'{state_bin_edges.shape}, {state_ramp_up.shape}, {state_ramp_down.shape}.'
            )
        if state_bin_edges.numel() != state_ramp_up.numel() + 1 or state_bin_edges.numel() != state_ramp_down.numel() + 1:
            raise ValueError(
                'WindPhysicsProjection expects len(edges)=len(ramps)+1, got '
                f'{state_bin_edges.numel()}, {state_ramp_up.numel()}, {state_ramp_down.numel()}.'
            )

        boundaries = state_bin_edges[1:-1]
        prev_flat = prev_power.reshape(-1)
        bin_idx = torch.bucketize(prev_flat, boundaries, right=False)
        bin_idx = bin_idx.clamp(0, state_ramp_up.numel() - 1)
        step_ramp_up = state_ramp_up[bin_idx].reshape_as(prev_power)
        step_ramp_down = state_ramp_down[bin_idx].reshape_as(prev_power)
        return step_ramp_up, step_ramp_down

    def forward(self, raw_power, lower_bound, upper_bound,
                prev_power=None, ramp_up=None, ramp_down=None,
                state_bin_edges=None, state_ramp_up=None, state_ramp_down=None):
        if prev_power is None or ramp_up is None or ramp_down is None:
            if state_bin_edges is None or state_ramp_up is None or state_ramp_down is None:
                return self._project_interval(raw_power, lower_bound, upper_bound)

        if raw_power.shape != lower_bound.shape or raw_power.shape != upper_bound.shape:
            raise ValueError(
                'WindPhysicsProjection expects raw_power, lower_bound, and upper_bound '
                f'to share the same shape, got {raw_power.shape}, {lower_bound.shape}, {upper_bound.shape}.'
            )
        if prev_power is None:
            raise ValueError(
                'WindPhysicsProjection requires prev_power when applying sequential wind constraints.'
            )
        if prev_power.dim() != 3 or prev_power.shape[0] != raw_power.shape[0] or prev_power.shape[-1] != raw_power.shape[-1]:
            raise ValueError(
                'WindPhysicsProjection expects prev_power to have shape [B, 1, C], got '
                f'{prev_power.shape} for raw_power {raw_power.shape}.'
            )
        use_state_ramps = state_bin_edges is not None or state_ramp_up is not None or state_ramp_down is not None
        if not use_state_ramps:
            if ramp_up is None or ramp_down is None:
                raise ValueError(
                    'WindPhysicsProjection requires either global ramp tensors or state-dependent ramp tables.'
                )
            if raw_power.shape != ramp_up.shape or raw_power.shape != ramp_down.shape:
                raise ValueError(
                    'WindPhysicsProjection expects ramp tensors to match raw_power shape, got '
                    f'{raw_power.shape}, {ramp_up.shape}, {ramp_down.shape}.'
                )
        else:
            if state_bin_edges.device != raw_power.device or state_ramp_up.device != raw_power.device or state_ramp_down.device != raw_power.device:
                raise ValueError('State-dependent wind ramp tables must be on the same device as raw_power.')

        projected_steps = []
        prev_projected = prev_power
        for step_idx in range(raw_power.shape[1]):
            step_raw = raw_power[:, step_idx:step_idx + 1, :]
            step_lower = lower_bound[:, step_idx:step_idx + 1, :]
            step_upper = upper_bound[:, step_idx:step_idx + 1, :]
            if use_state_ramps:
                step_ramp_up, step_ramp_down = self._select_state_ramps(
                    prev_projected,
                    state_bin_edges,
                    state_ramp_up,
                    state_ramp_down,
                )
            else:
                step_ramp_up = ramp_up[:, step_idx:step_idx + 1, :]
                step_ramp_down = ramp_down[:, step_idx:step_idx + 1, :]

            step_lower = torch.maximum(step_lower, prev_projected - step_ramp_down)
            step_upper = torch.minimum(step_upper, prev_projected + step_ramp_up)
            step_upper = torch.maximum(step_upper, step_lower)

            step_projected = self._project_interval(step_raw, step_lower, step_upper)
            projected_steps.append(step_projected)
            prev_projected = step_projected

        return torch.cat(projected_steps, dim=1)


class WindPowerPhysicsProjection(nn.Module):
    def __init__(self, hidden_dim, eps=1e-4, reference_density=1.225,
                 max_cp=0.593, density_correction_limit=0.2):
        super(WindPowerPhysicsProjection, self).__init__()
        cp_hidden_dim = max(hidden_dim, 8)
        cp_mid_dim = max(hidden_dim // 2, 4)
        self.cp_curve = nn.Sequential(
            nn.Linear(1, cp_hidden_dim),
            nn.GELU(),
            nn.Linear(cp_hidden_dim, cp_mid_dim),
            nn.GELU(),
            nn.Linear(cp_mid_dim, 1),
        )
        self.param_proj = nn.Linear(hidden_dim, 2)
        self.eps = eps
        self.reference_density = reference_density
        self.max_cp = max_cp
        self.density_correction_limit = density_correction_limit
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.cp_curve:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final_linear = self.cp_curve[-1]
        if isinstance(final_linear, nn.Linear):
            final_linear.weight.data.mul_(0.1)
            nn.init.constant_(final_linear.bias, 4.0)
        nn.init.zeros_(self.param_proj.weight)
        nn.init.zeros_(self.param_proj.bias)
        # Start near availability=1 so the stack begins close to the raw physics formula.
        nn.init.constant_(self.param_proj.bias[1], 4.0)

    def forward(self, hidden, met_state,
                density_mean, density_scale,
                wind_speed_mean, wind_speed_scale,
                power_mean, power_scale,
                wind_rated_power, wind_cut_in_speed, wind_rated_speed):
        if hidden.shape[:-1] != met_state.shape[:-1]:
            raise ValueError(
                'WindPowerPhysicsProjection expects hidden and met_state to share [B, L], got '
                f'{hidden.shape} and {met_state.shape}.'
            )
        if met_state.dim() != 3 or met_state.shape[-1] != 5:
            raise ValueError(
                'WindPowerPhysicsProjection expects met_state to have shape [B, L, 5], got '
                f'{met_state.shape}.'
            )

        # The backbone predicts standardized meteorology, so recover raw units before applying physics.
        density = met_state[..., 0:1] * density_scale + density_mean
        wind_speed = met_state[..., 4:5] * wind_speed_scale + wind_speed_mean
        params = self.param_proj(hidden)
        raw_density_correction = params[..., 0:1]
        raw_availability = params[..., 1:2]

        safe_density = density.clamp_min(self.eps)
        safe_speed = wind_speed.clamp_min(0.0)
        density_ratio = (safe_density / self.reference_density).clamp(min=0.25, max=2.5)
        effective_speed = safe_speed * torch.pow(density_ratio, 1.0 / 3.0)
        # Keep Cp identifiable by conditioning it on wind speed alone.
        cp_hat = self.max_cp * torch.sigmoid(self.cp_curve(effective_speed))
        cp_ratio = (cp_hat / self.max_cp).clamp_min(self.eps)
        # Hidden only adjusts intermediate factors instead of regressing power directly.
        density_correction = 1.0 + self.density_correction_limit * torch.tanh(raw_density_correction)
        availability = torch.sigmoid(raw_availability)
        cut_in_cubed = torch.pow(wind_cut_in_speed.clamp_min(0.0), 3.0)
        rated_speed = torch.maximum(wind_rated_speed, wind_cut_in_speed + self.eps)
        rated_cubed = torch.pow(rated_speed, 3.0)
        speed_cubed = torch.pow(effective_speed, 3.0)
        speed_term = torch.relu(speed_cubed - cut_in_cubed) / (rated_cubed - cut_in_cubed + self.eps)
        speed_term = speed_term.clamp(max=1.0)
        # Build power in raw units, then map it back to the model's standardized power space.
        power_raw = wind_rated_power * density_correction * availability * speed_term * cp_ratio
        return (power_raw - power_mean) / power_scale.clamp_min(self.eps)


class WindPhysicsBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3,
                 eps=1e-4, reference_density=1.225,
                 max_cp=0.593, density_correction_limit=0.2):
        super(WindPhysicsBlock, self).__init__()
        padding = kernel_size // 2
        self.state_embed = nn.Linear(1, hidden_dim)
        self.met_embed = nn.Linear(5, hidden_dim)
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
        self.projection = WindPowerPhysicsProjection(
            hidden_dim=hidden_dim,
            eps=eps,
            reference_density=reference_density,
            max_cp=max_cp,
            density_correction_limit=density_correction_limit,
        )

    def forward(self, hidden, state, met_state,
                density_mean, density_scale,
                wind_speed_mean, wind_speed_scale,
                power_mean, power_scale,
                wind_rated_power, wind_cut_in_speed, wind_rated_speed):
        physics_context = self.state_embed(state) + self.met_embed(met_state)
        mixed_delta = self.temporal_mixer((hidden + physics_context).transpose(1, 2)).transpose(1, 2)
        hidden = self.norm1(hidden + mixed_delta)
        hidden = self.norm2(hidden + self.ffn(hidden))
        state = self.projection(
            hidden,
            met_state,
            density_mean,
            density_scale,
            wind_speed_mean,
            wind_speed_scale,
            power_mean,
            power_scale,
            wind_rated_power,
            wind_cut_in_speed,
            wind_rated_speed,
        )
        return hidden, state


class WindPhysicsStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1,
                 kernel_size=3, eps=1e-4, reference_density=1.225,
                 min_direction_efficiency=0.6,
                 max_environment_gain=1.3,
                 max_raw_residual_mix=0.35,
                 max_cp=0.593,
                 density_correction_limit=0.2):
        super(WindPhysicsStack, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.initial_projection = WindPowerPhysicsProjection(
            hidden_dim=hidden_dim,
            eps=eps,
            reference_density=reference_density,
            max_cp=max_cp,
            density_correction_limit=density_correction_limit,
        )
        self.blocks = nn.ModuleList([
            WindPhysicsBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=kernel_size,
                eps=eps,
                reference_density=reference_density,
                max_cp=max_cp,
                density_correction_limit=density_correction_limit,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.engineering_projection = WindPhysicsProjection()

    def _safe_interval(self, lower_bound, upper_bound):
        upper_bound = torch.maximum(upper_bound, lower_bound)
        return lower_bound, upper_bound

    def forward(self, context, met_state, lower_bound, upper_bound,
                prev_power=None, ramp_up=None, ramp_down=None,
                state_bin_edges=None, state_ramp_up=None, state_ramp_down=None,
                density_mean=None, density_scale=None,
                wind_speed_mean=None, wind_speed_scale=None,
                power_mean=None, power_scale=None,
                wind_rated_power=None, wind_cut_in_speed=None, wind_rated_speed=None):
        if density_mean is None or density_scale is None or wind_speed_mean is None or wind_speed_scale is None:
            raise ValueError('WindPhysicsStack requires density and wind-speed normalization stats.')
        if power_mean is None or power_scale is None:
            raise ValueError('WindPhysicsStack requires power normalization stats.')
        if wind_rated_power is None or wind_cut_in_speed is None or wind_rated_speed is None:
            raise ValueError('WindPhysicsStack requires rated-power curve statistics.')

        hidden = self.input_norm(self.input_proj(context))
        state = self.initial_projection(
            hidden,
            met_state,
            density_mean,
            density_scale,
            wind_speed_mean,
            wind_speed_scale,
            power_mean,
            power_scale,
            wind_rated_power,
            wind_cut_in_speed,
            wind_rated_speed,
        )
        for block in self.blocks:
            hidden, state = block(
                hidden,
                state,
                met_state,
                density_mean,
                density_scale,
                wind_speed_mean,
                wind_speed_scale,
                power_mean,
                power_scale,
                wind_rated_power,
                wind_cut_in_speed,
                wind_rated_speed,
            )
        hidden = self.output_norm(hidden)
        raw_state = state

        lower_bound, upper_bound = self._safe_interval(lower_bound, upper_bound)
        # Apply engineering constraints only once after the physics stack finishes updating state.
        constrained_state = self.engineering_projection(
            state,
            lower_bound,
            upper_bound,
            prev_power=prev_power,
            ramp_up=ramp_up,
            ramp_down=ramp_down,
            state_bin_edges=state_bin_edges,
            state_ramp_up=state_ramp_up,
            state_ramp_down=state_ramp_down,
        )
        return constrained_state, hidden, raw_state


class FixedCpCurve(nn.Module):
    def __init__(self, max_cp=0.48, optimal_ratio=1.0, width=0.35, eps=1e-4):
        super(FixedCpCurve, self).__init__()
        self.max_cp = float(max_cp)
        self.optimal_ratio = float(optimal_ratio)
        self.width = float(width)
        self.eps = float(eps)

    def forward(self, subrated_position):
        width = max(self.width, self.eps)
        cp = self.max_cp * torch.exp(
            -((subrated_position - self.optimal_ratio) ** 2) / (2.0 * width ** 2)
        )
        return (cp / max(self.max_cp, self.eps)).clamp(min=self.eps, max=1.0)


class ParametricCpCurve(nn.Module):
    def __init__(self, max_cp=0.48, optimal_ratio=1.0, width=0.35, eps=1e-4):
        super(ParametricCpCurve, self).__init__()
        self.max_cp = float(max_cp)
        self.eps = float(eps)
        self.optimal_ratio = nn.Parameter(torch.tensor(float(optimal_ratio)))
        self.width_param = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(width))) + self.eps))

    def forward(self, subrated_position):
        width = F.softplus(self.width_param).clamp_min(self.eps)
        cp = self.max_cp * torch.exp(
            -((subrated_position - self.optimal_ratio) ** 2) / (2.0 * width ** 2)
        )
        return (cp / max(self.max_cp, self.eps)).clamp(min=self.eps, max=1.0)


class SimplifiedWindPhysicsProjection(nn.Module):
    def __init__(self, hidden_dim, eps=1e-4, cp_mode='none',
                 cp_max=0.48, cp_optimal_ratio=1.0, cp_width=0.35):
        super(SimplifiedWindPhysicsProjection, self).__init__()
        self.param_proj = nn.Linear(hidden_dim, 2)
        self.eps = float(eps)
        cp_mode = str(cp_mode).lower()
        if cp_mode not in {'none', 'fixed', 'parametric'}:
            raise ValueError(f'Unsupported wind Cp mode: {cp_mode}')
        self.cp_mode = cp_mode
        if self.cp_mode == 'fixed':
            self.cp_curve = FixedCpCurve(
                max_cp=cp_max,
                optimal_ratio=cp_optimal_ratio,
                width=cp_width,
                eps=eps,
            )
        elif self.cp_mode == 'parametric':
            self.cp_curve = ParametricCpCurve(
                max_cp=cp_max,
                optimal_ratio=cp_optimal_ratio,
                width=cp_width,
                eps=eps,
            )
        else:
            self.cp_curve = None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.param_proj.weight)
        nn.init.zeros_(self.param_proj.bias)
        nn.init.constant_(self.param_proj.bias[0], 4.0)

    def _broadcast_param(self, value, reference):
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=reference.dtype, device=reference.device)
        else:
            value = value.to(dtype=reference.dtype, device=reference.device)

        if value.shape == reference.shape:
            return value
        if value.dim() == 0:
            return value.view(1, 1, 1).expand_as(reference)
        if value.dim() == 1:
            if value.shape[0] == reference.shape[0]:
                return value.view(reference.shape[0], 1, 1).expand_as(reference)
            if value.shape[0] == reference.shape[1]:
                return value.view(1, reference.shape[1], 1).expand_as(reference)
        if value.dim() == 2:
            if value.shape == reference.shape[:2]:
                return value.unsqueeze(-1)
            if value.shape[0] == reference.shape[0] and value.shape[1] == 1:
                return value.view(reference.shape[0], 1, 1).expand_as(reference)
            if value.shape[0] == 1 and value.shape[1] == reference.shape[1]:
                return value.view(1, reference.shape[1], 1).expand_as(reference)
        if value.dim() == 3:
            return torch.broadcast_to(value, reference.shape)
        raise ValueError(
            'SimplifiedWindPhysicsProjection could not broadcast parameter with '
            f'shape {value.shape} to reference shape {reference.shape}.'
        )

    def forward(self, hidden, wind_speed,
                cut_in_speed, rated_speed, cut_out_speed, rated_power):
        if hidden.shape[:-1] != wind_speed.shape[:-1]:
            raise ValueError(
                'SimplifiedWindPhysicsProjection expects hidden and wind_speed to share [B, L], got '
                f'{hidden.shape} and {wind_speed.shape}.'
            )
        if wind_speed.dim() != 3 or wind_speed.shape[-1] != 1:
            raise ValueError(
                'SimplifiedWindPhysicsProjection expects wind_speed to have shape [B, L, 1], got '
                f'{wind_speed.shape}.'
            )

        cut_in_speed = self._broadcast_param(cut_in_speed, wind_speed)
        rated_speed = self._broadcast_param(rated_speed, wind_speed)
        cut_out_speed = self._broadcast_param(cut_out_speed, wind_speed)
        rated_power = self._broadcast_param(rated_power, wind_speed)

        params = self.param_proj(hidden)
        availability = torch.sigmoid(params[..., 0:1])
        power_scale = 1.0 + 0.2 * torch.tanh(params[..., 1:2])

        v = wind_speed.clamp_min(-3.0)
        cut_in_speed = cut_in_speed.clamp_min(-3.0)
        rated_speed = torch.maximum(rated_speed, cut_in_speed + self.eps)

        v_cubed = torch.pow(v, 3.0)
        cut_in_cubed = torch.pow(cut_in_speed, 3.0)
        rated_cubed = torch.pow(rated_speed, 3.0)

        power_ratio = torch.relu(v_cubed - cut_in_cubed) / (rated_cubed - cut_in_cubed + self.eps)
        power_ratio = power_ratio.clamp(max=1.0)

        if self.cp_curve is not None:
            # Use the sub-rated position instead of raw ratios so Cp experiments stay stable in normalized space.
            subrated_position = (v - cut_in_speed) / (rated_speed - cut_in_speed + self.eps)
            cp_factor = self.cp_curve(subrated_position.clamp(min=0.0, max=1.5))
            subrated_mask = ((v >= cut_in_speed) & (v < rated_speed)).to(power_ratio.dtype)
            power_ratio = power_ratio * ((1.0 - subrated_mask) + subrated_mask * cp_factor)

        cut_out_mask = (v < cut_out_speed).to(power_ratio.dtype)
        power = rated_power * power_ratio * availability * power_scale * cut_out_mask
        return power


class SimplifiedWindPhysicsStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1,
                 eps=1e-4, cp_mode='none', cp_max=0.48,
                 cp_optimal_ratio=1.0, cp_width=0.35):
        super(SimplifiedWindPhysicsStack, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.feature_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.final_projection = SimplifiedWindPhysicsProjection(
            hidden_dim=hidden_dim,
            eps=eps,
            cp_mode=cp_mode,
            cp_max=cp_max,
            cp_optimal_ratio=cp_optimal_ratio,
            cp_width=cp_width,
        )

    def forward(self, context, wind_speed, cut_in_speed, rated_speed, cut_out_speed, rated_power):
        hidden = self.input_norm(self.input_proj(context))
        for block in self.feature_blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        power = self.final_projection(
            hidden,
            wind_speed,
            cut_in_speed,
            rated_speed,
            cut_out_speed,
            rated_power,
        )
        return power, hidden
