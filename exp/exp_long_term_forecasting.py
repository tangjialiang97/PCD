from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.utils
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import itertools
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _uses_solar_internal_physics(self):
        return bool(getattr(self.args, 'use_internal_physics', False)) or self.args.model.endswith('SolarPhysics')

    def _uses_wind_physics(self):
        return 'WindPhysics' in self.args.model

    def _uses_wind_state_dependent_physics(self):
        return 'WindPhysicsV3' in self.args.model or 'WindPhysicsV4' in self.args.model

    def _uses_wind_soft_physics(self):
        return 'WindPhysicsV4' in self.args.model

    def _uses_wind_v5_physics(self):
        return 'WindPhysicsV5' in self.args.model

    def _uses_wind_v6_physics(self):
        return 'WindPhysicsV6' in self.args.model

    def _uses_reservoir_physics(self):
        return 'ReservoirPhysics' in self.args.model

    def _extract_future_zenith(self, batch_y):
        return batch_y[:, -self.args.pred_len:, -1]

    def _extract_future_wind_bounds(self, batch_y, data_set):
        pred_shape = (batch_y.shape[0], self.args.pred_len, 1)
        lower_value = float(getattr(data_set, 'power_lower_bound', 0.0))
        upper_value = float(getattr(data_set, 'power_upper_bound', 0.0))
        lower_bound = torch.full(pred_shape, lower_value, dtype=batch_y.dtype, device=batch_y.device)
        upper_bound = torch.full(pred_shape, upper_value, dtype=batch_y.dtype, device=batch_y.device)
        return lower_bound, upper_bound

    def _extract_future_wind_physics_inputs(self, batch_x_main, batch_y, data_set):
        if self._uses_wind_v5_physics() or self._uses_wind_v6_physics():
            return {
                'wind_cut_in_speed_norm': torch.tensor(
                    float(getattr(data_set, 'power_cut_in_speed_norm', 0.0)),
                    dtype=batch_x_main.dtype,
                    device=batch_x_main.device,
                ),
                'wind_rated_speed_norm': torch.tensor(
                    float(getattr(data_set, 'power_rated_speed_norm', 1.0)),
                    dtype=batch_x_main.dtype,
                    device=batch_x_main.device,
                ),
                'wind_cut_out_speed_norm': torch.tensor(
                    float(getattr(data_set, 'power_cut_out_speed_norm', 5.0)),
                    dtype=batch_x_main.dtype,
                    device=batch_x_main.device,
                ),
                'wind_rated_power_norm': torch.tensor(
                    float(getattr(data_set, 'power_rated_power_norm', 1.0)),
                    dtype=batch_x_main.dtype,
                    device=batch_x_main.device,
                ),
            }

        lower_bound, upper_bound = self._extract_future_wind_bounds(batch_y, data_set)
        prev_power = batch_x_main[:, -1:, -1:]
        physics_inputs = {
            'power_lower_bound': lower_bound,
            'power_upper_bound': upper_bound,
            'prev_power': prev_power,
        }
        if self._uses_wind_state_dependent_physics():
            state_bin_edges_value = getattr(data_set, 'power_state_bin_edges_scaled', None)
            state_ramp_up_value = getattr(data_set, 'power_state_ramp_up_scaled', None)
            state_ramp_down_value = getattr(data_set, 'power_state_ramp_down_scaled', None)
            if state_bin_edges_value is None or state_ramp_up_value is None or state_ramp_down_value is None:
                raise ValueError('WindPhysicsV3 requires state-dependent ramp tables from the dataset.')
            state_bin_edges = torch.as_tensor(
                state_bin_edges_value,
                dtype=batch_x_main.dtype,
                device=batch_x_main.device,
            )
            state_ramp_up = torch.as_tensor(
                state_ramp_up_value,
                dtype=batch_x_main.dtype,
                device=batch_x_main.device,
            )
            state_ramp_down = torch.as_tensor(
                state_ramp_down_value,
                dtype=batch_x_main.dtype,
                device=batch_x_main.device,
            )
            physics_inputs.update({
                'state_bin_edges': state_bin_edges,
                'state_ramp_up': state_ramp_up,
                'state_ramp_down': state_ramp_down,
            })
            if self._uses_wind_soft_physics():
                if data_set.scale and hasattr(data_set.scaler, 'mean_'):
                    density_idx = data_set.data_columns.index('density')
                    wind_speed_idx = data_set.data_columns.index('wind_speed')
                    power_idx = data_set.data_columns.index('power')
                    density_mean = float(data_set.scaler.mean_[density_idx])
                    density_scale = float(data_set.scaler.scale_[density_idx])
                    wind_speed_mean = float(data_set.scaler.mean_[wind_speed_idx])
                    wind_speed_scale = float(data_set.scaler.scale_[wind_speed_idx])
                    power_mean = float(data_set.scaler.mean_[power_idx])
                    power_scale = float(data_set.scaler.scale_[power_idx])
                else:
                    density_mean = 0.0
                    density_scale = 1.0
                    wind_speed_mean = 0.0
                    wind_speed_scale = 1.0
                    power_mean = 0.0
                    power_scale = 1.0
                physics_inputs.update({
                    'density_mean': torch.tensor(density_mean, dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'density_scale': torch.tensor(density_scale, dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'wind_speed_mean': torch.tensor(wind_speed_mean, dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'wind_speed_scale': torch.tensor(wind_speed_scale, dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'power_mean': torch.tensor(power_mean, dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'power_scale': torch.tensor(power_scale, dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'wind_rated_power': torch.tensor(float(getattr(data_set, 'power_cap_raw', 1.0)), dtype=batch_x_main.dtype, device=batch_x_main.device),
                    'wind_cut_in_speed': torch.tensor(
                        float(getattr(data_set, 'power_cut_in_speed_legacy_raw', getattr(data_set, 'power_cut_in_speed_raw', 3.0))),
                        dtype=batch_x_main.dtype,
                        device=batch_x_main.device,
                    ),
                    'wind_rated_speed': torch.tensor(
                        float(getattr(data_set, 'power_rated_speed_legacy_raw', getattr(data_set, 'power_rated_speed_raw', 12.0))),
                        dtype=batch_x_main.dtype,
                        device=batch_x_main.device,
                    ),
                })
            return physics_inputs

        pred_shape = (batch_y.shape[0], self.args.pred_len, 1)
        ramp_up_value = float(getattr(data_set, 'power_ramp_up_scaled', 0.0))
        ramp_down_value = float(getattr(data_set, 'power_ramp_down_scaled', 0.0))
        physics_inputs.update({
            'ramp_up': torch.full(pred_shape, ramp_up_value, dtype=batch_x_main.dtype, device=batch_x_main.device),
            'ramp_down': torch.full(pred_shape, ramp_down_value, dtype=batch_x_main.dtype, device=batch_x_main.device),
        })
        return physics_inputs

    def _select_main_channels(self, batch_tensor):
        return batch_tensor[:, :, :self.args.enc_in]

    def _select_supervised_channels(self, batch_tensor):
        f_dim = -1 if self.args.features == 'MS' else 0
        return batch_tensor[:, :, f_dim:]

    def _model_ref(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _latest_wind_raw_power(self):
        raw_power = getattr(self._model_ref(), 'latest_raw_power', None)
        if raw_power is None:
            raise ValueError('WindPhysicsV4 expects the model to expose latest_raw_power after forward.')
        return raw_power

    def _latest_wind_constrained_power(self):
        constrained_power = getattr(self._model_ref(), 'latest_constrained_power', None)
        if constrained_power is None:
            raise ValueError('WindPhysicsV4 expects the model to expose latest_constrained_power after forward.')
        return constrained_power

    def _wind_state_bounds_from_reference(self, prev_power_ref, lower_bound, upper_bound, data_set, dtype, device):
        state_bin_edges_value = getattr(data_set, 'power_state_bin_edges_scaled', None)
        state_ramp_up_value = getattr(data_set, 'power_state_ramp_up_scaled', None)
        state_ramp_down_value = getattr(data_set, 'power_state_ramp_down_scaled', None)
        if state_bin_edges_value is None or state_ramp_up_value is None or state_ramp_down_value is None:
            raise ValueError('WindPhysics state bounds require state-dependent ramp tables from the dataset.')
        state_bin_edges = torch.as_tensor(
            state_bin_edges_value,
            dtype=dtype,
            device=device,
        )
        state_ramp_up = torch.as_tensor(
            state_ramp_up_value,
            dtype=dtype,
            device=device,
        )
        state_ramp_down = torch.as_tensor(
            state_ramp_down_value,
            dtype=dtype,
            device=device,
        )
        boundaries = state_bin_edges[1:-1]
        bin_idx = torch.bucketize(prev_power_ref.reshape(-1), boundaries, right=False)
        bin_idx = bin_idx.clamp(0, state_ramp_up.numel() - 1)
        step_ramp_up = state_ramp_up[bin_idx].reshape_as(prev_power_ref)
        step_ramp_down = state_ramp_down[bin_idx].reshape_as(prev_power_ref)
        step_lower = torch.maximum(lower_bound, prev_power_ref - step_ramp_down)
        step_upper = torch.minimum(upper_bound, prev_power_ref + step_ramp_up)
        step_upper = torch.maximum(step_upper, step_lower)
        return step_lower, step_upper

    def _compute_wind_soft_penalty(self, batch_x_main, batch_y_main, data_set, epoch):
        raw_power = self._latest_wind_raw_power()
        constrained_power = self._latest_wind_constrained_power()
        power_lower_bound, power_upper_bound = self._extract_future_wind_bounds(batch_y_main, data_set)
        true_future_power = batch_y_main[:, -self.args.pred_len:, -1:]
        prev_power_ref = torch.cat([batch_x_main[:, -1:, -1:], true_future_power[:, :-1]], dim=1)
        step_lower, step_upper = self._wind_state_bounds_from_reference(
            prev_power_ref,
            power_lower_bound,
            power_upper_bound,
            data_set,
            raw_power.dtype,
            raw_power.device,
        )
        upper_violation = torch.relu(raw_power - step_upper)
        lower_violation = torch.relu(step_lower - raw_power)
        violation_loss = torch.mean(upper_violation ** 2 + lower_violation ** 2)
        projection_gap_weight = float(getattr(self.args, 'wind_projection_gap_weight', 0.1))
        projection_gap_loss = torch.mean((raw_power - constrained_power) ** 2)
        warmup_epochs = int(getattr(self.args, 'wind_constraint_warmup_epochs', 2))
        warmup_factor = 1.0
        if warmup_epochs > 0:
            warmup_factor = min(1.0, float(epoch + 1) / float(warmup_epochs))
        penalty_weight = float(getattr(self.args, 'wind_constraint_weight', 1.0))
        total_penalty = violation_loss + projection_gap_weight * projection_gap_loss
        return warmup_factor * penalty_weight * total_penalty

    def _compute_wind_v5_soft_penalty(self, pred_power, batch_x_main, data_set, epoch):
        pred_shape = pred_power.shape
        prev_power = batch_x_main[:, -1:, -1:]
        lower_bound = torch.full(
            pred_shape,
            float(getattr(data_set, 'power_lower_bound', 0.0)),
            dtype=pred_power.dtype,
            device=pred_power.device,
        )
        upper_bound = torch.full(
            pred_shape,
            float(getattr(data_set, 'power_upper_bound', 0.0)),
            dtype=pred_power.dtype,
            device=pred_power.device,
        )
        ramp_up = torch.full(
            pred_shape,
            float(getattr(data_set, 'power_ramp_up_scaled', 0.0)),
            dtype=pred_power.dtype,
            device=pred_power.device,
        )
        ramp_down = torch.full(
            pred_shape,
            float(getattr(data_set, 'power_ramp_down_scaled', 0.0)),
            dtype=pred_power.dtype,
            device=pred_power.device,
        )

        power_diff = torch.cat([
            pred_power[:, :1, :] - prev_power,
            pred_power[:, 1:, :] - pred_power[:, :-1, :],
        ], dim=1)
        ramp_up_violation = torch.relu(power_diff - ramp_up)
        ramp_down_violation = torch.relu(-power_diff - ramp_down)
        ramp_loss = ramp_up_violation.mean() + ramp_down_violation.mean()

        lower_violation = torch.relu(lower_bound - pred_power)
        upper_violation = torch.relu(pred_power - upper_bound)
        bound_loss = lower_violation.mean() + upper_violation.mean()

        warmup_epochs = int(getattr(self.args, 'wind_constraint_warmup_epochs', 2))
        warmup_factor = 1.0
        if warmup_epochs > 0:
            warmup_factor = min(1.0, float(epoch + 1) / float(warmup_epochs))

        ramp_weight = float(getattr(self.args, 'wind_soft_ramp_weight', 0.1))
        bound_weight = float(getattr(self.args, 'wind_soft_bound_weight', 0.1))
        return warmup_factor * (ramp_weight * ramp_loss + bound_weight * bound_loss)

    def _forward_model(self, batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, data_set):
        if self._uses_solar_internal_physics():
            future_zenith = self._extract_future_zenith(batch_y)
            return self.model(
                batch_x_main,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                future_zenith=future_zenith,
            )
        if self._uses_wind_physics():
            wind_physics_inputs = self._extract_future_wind_physics_inputs(batch_x_main, batch_y, data_set)
            return self.model(
                batch_x_main,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                **wind_physics_inputs,
            )
        return self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer_param_models = {
            'ChronosSolarPhysics',
            'ChronosFT',
            'Chronos2SolarPhysics',
            'Chronos2FT',
            'SundialSolarPhysics',
            'SundialFT',
            'TiRexSolarPhysics',
            'TiRexFT',
            'TimesFMSolarPhysics',
            'TimeMoESolarPhysics',
            'TimesFMFT',
            'TimsFMFT',
            'TimeMoEFT',
        }
        if self.args.model in optimizer_param_models:
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        else:
            parameters = self.model.parameters()
        model_optim = optim.Adam(parameters, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        if self.args.debug:
            vali_loader = list(itertools.islice(vali_loader, 10))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, vali_data
                        )
                else:
                    outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, vali_data)
                outputs = self._select_supervised_channels(outputs[:, -self.args.pred_len:, :])
                batch_y_main_pred = self._select_supervised_channels(
                    batch_y_main[:, -self.args.pred_len:, :].to(self.device)
                )

                pred = outputs.detach()
                true = batch_y_main_pred.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.debug:
            train_loader = list(itertools.islice(train_loader, 10))
            vali_loader = list(itertools.islice(vali_loader, 10))
            test_loader = list(itertools.islice(test_loader, 10))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, train_data)
                outputs = self._select_supervised_channels(outputs[:, -self.args.pred_len:, :])

                batch_y_main_pred = self._select_supervised_channels(
                    batch_y_main[:, -self.args.pred_len:, :].to(self.device)
                )
                loss = criterion(outputs, batch_y_main_pred)

                if self._uses_wind_soft_physics():
                    loss = loss + self._compute_wind_soft_penalty(batch_x_main, batch_y_main, train_data, epoch)
                elif self._uses_wind_v5_physics() or self._uses_wind_v6_physics():
                    loss = loss + self._compute_wind_v5_soft_penalty(outputs, batch_x_main, train_data, epoch)

                if self.args.use_auxiliary and self._uses_reservoir_physics():
                    # Conservation residual auxiliary loss for reservoir physics.
                    # Penalises deviation from S_{t+1} = S_t + dt*(I - O) on the
                    # *predicted* sequence, encouraging the backbone to stay close
                    # to the conservation manifold even before projection.
                    res_warmup_epochs = int(getattr(self.args, 'reservoir_warmup_epochs', 2))
                    res_warmup_factor = min(1.0, float(epoch + 1) / float(res_warmup_epochs)) if res_warmup_epochs > 0 else 1.0
                    delta_t = float(getattr(self.args, 'reservoir_delta_t', 1.0))
                    # outputs: [B, pred_len, num_vars]  channels: storage(0), inflow(1), outflow(2) [, evap(3)]
                    pred_storage = outputs[:, :, 0:1]
                    pred_inflow = outputs[:, :, 1:2]
                    pred_outflow = outputs[:, :, 2:3]
                    if outputs.shape[-1] >= 4:
                        pred_evap = outputs[:, :, 3:4]
                    else:
                        pred_evap = torch.zeros_like(pred_inflow)
                    # Build prev_storage: last observed storage + predicted storage shifted
                    last_obs_storage = batch_x_main[:, -1:, 0:1]  # [B, 1, 1]
                    prev_storage = torch.cat([last_obs_storage, pred_storage[:, :-1, :]], dim=1)
                    conservation_residual = pred_storage - (prev_storage + delta_t * (pred_inflow - pred_outflow - pred_evap))
                    l_conservation = torch.mean(conservation_residual ** 2)
                    conservation_weight = float(getattr(self.args, 'conservation_weight', 1.0))
                    loss = loss + res_warmup_factor * conservation_weight * l_conservation

                if self.args.data == 'Solar' and (
                    self.args.use_auxiliary or self.args._uses_external_physics
                ):
                    # Warm up the whole auxiliary objective together to avoid
                    # early-stage optimization instability from strong constraints.
                    aux_warmup_epochs = int(getattr(self.args, 'closure_warmup_epochs', 2))
                    aux_warmup_factor = 1.0
                    if aux_warmup_epochs > 0:
                        aux_warmup_factor = min(1.0, float(epoch + 1) / float(aux_warmup_epochs))
                    aux_loss = torch.zeros((), device=loss.device)

                    batch_y_aux = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    zenith_deg = batch_y_aux[:, :, -1]
                    cos_zenith = torch.cos(torch.deg2rad(zenith_deg)).clamp(min=0.0)
                    ghi_hat = outputs[:, :, 0]
                    dni_hat = outputs[:, :, 1]
                    dhi_hat = outputs[:, :, 2]
                    if self.args.use_auxiliary and not self._uses_solar_internal_physics():
                        residual = ghi_hat - (dni_hat * cos_zenith + dhi_hat)
                        # Apply closure only in daytime with usable solar elevation,
                        # then use a relative robust penalty so closure acts as a weak regularizer.
                        closure_cos_threshold = float(getattr(self.args, 'closure_day_cos_threshold', 0.2))
                        closure_ref_floor = float(getattr(self.args, 'closure_ref_floor', 100.0))
                        closure_beta = float(getattr(self.args, 'closure_beta', 0.03))
                        day_mask = (cos_zenith > closure_cos_threshold).float()
                        ghi_true = batch_y_main_pred[:, :, 0].detach()
                        closure_ref = ghi_true.abs().clamp_min(closure_ref_floor)
                        residual_rel = residual / closure_ref
                        closure_term = torch.nn.functional.smooth_l1_loss(
                            residual_rel,
                            torch.zeros_like(residual_rel),
                            reduction='none',
                            beta=closure_beta,
                        )
                        day_mask_sum = day_mask.sum()
                        l_closure = (closure_term * day_mask).sum() / day_mask_sum.clamp_min(1.0)
                        closure_weight = float(getattr(self.args, 'closure_weight', 1.0))
                        aux_loss = aux_loss + closure_weight * l_closure

                    if self.args._uses_external_physics:
                        # Meteorological attenuation limit:
                        # GHI_upper = I0 * cos(theta_z) * f(P, RH, T)
                        # L_limit = mean(ReLU(GHI_hat - GHI_upper)^2)
                        i0 = float(getattr(self.args, 'solar_constant', 1367.0))
                        has_relhum = bool(getattr(self.args, 'has_relhum', False))
                        # Column order with relhum (e.g. Folsom):
                        #   [ghi, dni, dhi, air_temp, relhum(4), press(5), windsp(6), winddir, max_windsp, zenith]
                        # Column order without relhum (e.g. NSRDB-only sites):
                        #   [ghi, dni, dhi, air_temp, press(4), windsp(5), winddir, max_windsp, zenith]
                        air_temp = batch_y_aux[:, :, 3]
                        if has_relhum:
                            relhum = batch_y_aux[:, :, 4]
                            press = batch_y_aux[:, :, 5]
                            windsp_col = 6
                        else:
                            relhum = None
                            press = batch_y_aux[:, :, 4]
                            windsp_col = 5

                        # Build a smooth attenuation factor in [0, 1].
                        # When relhum is unavailable treat atmosphere as dry (relhum_norm=0),
                        # giving a looser upper bound and avoiding spurious l_limit penalties.
                        relhum_norm = (relhum / 100.0).clamp(0.0, 1.0) if relhum is not None else torch.zeros_like(air_temp)
                        press_ratio = (press / 1013.25).clamp(0.6, 1.2)
                        temp_penalty = ((air_temp - 25.0).abs() / 40.0).clamp(0.0, 1.0)
                        f_attn = (press_ratio * (1.0 - 0.3 * relhum_norm) * (1.0 - 0.2 * temp_penalty)).clamp(0.0, 1.0)

                        ghi_upper = i0 * cos_zenith * f_attn
                        l_limit = torch.mean(torch.relu(ghi_hat - ghi_upper) ** 2)
                        limit_weight = float(getattr(self.args, 'limit_weight', 1.0))
                        aux_loss = aux_loss + limit_weight * l_limit

                        # Variability loss:
                        # L_var = exp(-gamma * windsp) * ||dGHI/dt||^2
                        # Low wind speed -> stronger smoothness; high wind speed -> weaker penalty.
                        if outputs.shape[1] > 1:
                            sample_interval = float(getattr(self.args, 'sample_interval', 1.0))
                            delta_t = max(sample_interval, 1.0)
                            dghi_dt = (ghi_hat[:, 1:] - ghi_hat[:, :-1]) / delta_t
                            windsp = batch_y_aux[:, 1:, windsp_col].clamp(min=0.0)
                            gamma_var = float(getattr(self.args, 'var_gamma', 0.1))
                            var_weight_term = torch.exp(-gamma_var * windsp)
                            l_var = torch.mean(var_weight_term * (dghi_dt ** 2))
                            var_weight = float(getattr(self.args, 'var_weight', 1.0))
                            aux_loss = aux_loss + var_weight * l_var

                    loss = loss + aux_warmup_factor * aux_loss

                if self.args.data == 'Solar' and self._uses_solar_internal_physics():
                    dhi_weight = float(getattr(self.args, 'solar_dhi_weight', 0.0))
                    dhi_ratio_floor_weight = float(getattr(self.args, 'solar_dhi_ratio_floor_weight', 0.0))
                    if dhi_weight > 0.0 or dhi_ratio_floor_weight > 0.0:
                        batch_y_aux = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        zenith_deg = batch_y_aux[:, :, -1]
                        cos_zenith = torch.cos(torch.deg2rad(zenith_deg)).clamp(min=0.0)

                        dhi_day_cos_threshold = float(getattr(self.args, 'solar_dhi_day_cos_threshold', 0.05))
                        dhi_ref_floor = float(getattr(self.args, 'solar_dhi_ref_floor', 100.0))
                        dhi_beta = float(getattr(self.args, 'solar_dhi_beta', 0.03))
                        dhi_ratio_floor_cap = float(getattr(self.args, 'solar_dhi_ratio_floor_cap', 0.05))
                        ratio_eps = float(getattr(self.args, 'physics_eps', 1e-4))

                        day_mask = (cos_zenith > dhi_day_cos_threshold).float()
                        day_count = day_mask.sum().clamp_min(1.0)

                        ghi_hat = outputs[:, :, 0]
                        dhi_hat = outputs[:, :, 2]
                        ghi_true = batch_y_main_pred[:, :, 0].detach().clamp_min(ratio_eps)
                        dhi_true = batch_y_main_pred[:, :, 2].detach().clamp_min(0.0)

                        if dhi_weight > 0.0:
                            dhi_ref = ghi_true.abs().clamp_min(dhi_ref_floor)
                            dhi_loss = torch.nn.functional.smooth_l1_loss(
                                dhi_hat / dhi_ref,
                                dhi_true / dhi_ref,
                                reduction='none',
                                beta=dhi_beta,
                            )
                            loss = loss + dhi_weight * (dhi_loss * day_mask).sum() / day_count

                        if dhi_ratio_floor_weight > 0.0:
                            ratio_hat = (dhi_hat / ghi_hat.clamp_min(ratio_eps)).clamp(0.0, 1.0)
                            ratio_true = (dhi_true / ghi_true.clamp_min(ratio_eps)).clamp(0.0, 1.0)
                            ratio_floor = torch.minimum(
                                ratio_true,
                                torch.full_like(ratio_true, dhi_ratio_floor_cap),
                            )
                            ratio_shortfall = torch.relu(ratio_floor - ratio_hat)
                            loss = loss + dhi_ratio_floor_weight * (
                                (ratio_shortfall ** 2) * day_mask
                            ).sum() / day_count

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if self._uses_wind_v5_physics() or self._uses_wind_v6_physics():
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(getattr(self.args, 'wind_grad_clip', 5.0)),
                        )
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self._uses_wind_v5_physics() or self._uses_wind_v6_physics():
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(getattr(self.args, 'wind_grad_clip', 5.0)),
                        )
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds = []
        trues = []


        self.model.eval()
        if self.args.debug:
            test_loader = list(itertools.islice(test_loader, 10))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, test_data)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_main_pred = batch_y_main[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y_main_pred = batch_y_main_pred.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y_main_pred.shape
                    if outputs.shape[-1] != batch_y_main_pred.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y_main_pred.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_main_pred = test_data.inverse_transform(batch_y_main_pred.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = self._select_supervised_channels(outputs)
                batch_y_main_pred = self._select_supervised_channels(batch_y_main_pred)

                pred = outputs
                true = batch_y_main_pred

                preds.append(pred)
                trues.append(true)
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        is_three = bool(getattr(self.args, 'is_three', False))
        if not is_three:
            preds = preds[:, :, 0:1]
            trues = trues[:, :, 0:1]
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            dtw_preds = preds[:, :, 0:1]
            dtw_trues = trues[:, :, 0:1]
            for i in range(dtw_preds.shape[0]):
                x = dtw_preds[i].reshape(-1, 1)
                y = dtw_trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        ymax = test_data.ymax
        # ymax = 1.0
        f = open("result_new_" + self.args.txt_name + "_" + self.args.data + "_" + self.args.data_path.split(".")[0] + "_" + str(self.args._uses_external_physics) + ".txt", 'a')
        f.write(setting + "  \n")

        if is_three:
            mae, mse, nrmse, nmae = metric(preds, trues, ymax)
            print('avg_3ch_mse:{}, avg_3ch_mae:{}, avg_3ch_nrmse:{}, avg_3ch_nmae:{}'.format(mse, mae, nrmse, nmae))
            f.write('avg_3ch_mse:{}, avg_3ch_mae:{}, avg_3ch_nrmse:{}, avg_3ch_nmae:{}'.format(mse, mae, nrmse, nmae))
            f.write('\n')

            channel_names = ['GHI', 'DNI', 'DHI']
            channel_count = min(preds.shape[-1], len(channel_names))
            for channel_idx in range(channel_count):
                channel_pred = preds[:, :, channel_idx:channel_idx + 1]
                channel_true = trues[:, :, channel_idx:channel_idx + 1]
                channel_ymax = ymax[channel_idx] if isinstance(ymax, (list, tuple, np.ndarray)) else ymax
                ch_mae, ch_mse, ch_nrmse, ch_nmae = metric(channel_pred, channel_true, channel_ymax)
                channel_name = channel_names[channel_idx]
                print('{} -> mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(
                    channel_name, ch_mse, ch_mae, ch_nrmse, ch_nmae
                ))
                f.write('{} -> mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(
                    channel_name, ch_mse, ch_mae, ch_nrmse, ch_nmae
                ))
                f.write('\n')
        else:
            ghi_ymax = ymax[0] if isinstance(ymax, (list, tuple, np.ndarray)) else ymax
            mae, mse, nrmse, nmae = metric(preds, trues, ghi_ymax)
            print('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))
            f.write('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))

        # Reservoir-specific metrics: per-channel and conservation residual
        if self._uses_reservoir_physics() and preds.shape[-1] >= 3:
            channel_names_res = ['storage', 'inflow', 'outflow', 'evaporation']
            channel_count_res = min(preds.shape[-1], len(channel_names_res))
            f.write('\n--- Reservoir per-channel metrics ---\n')
            for ch_idx in range(channel_count_res):
                ch_pred = preds[:, :, ch_idx:ch_idx + 1]
                ch_true = trues[:, :, ch_idx:ch_idx + 1]
                ch_ymax = ymax[ch_idx] if isinstance(ymax, (list, tuple, np.ndarray)) else ymax
                ch_mae, ch_mse, ch_nrmse, ch_nmae = metric(ch_pred, ch_true, ch_ymax)
                ch_name = channel_names_res[ch_idx]
                print('{} -> mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(
                    ch_name, ch_mse, ch_mae, ch_nrmse, ch_nmae))
                f.write('{} -> mse:{}, mae:{}, nrmse:{}, nmae:{}\n'.format(
                    ch_name, ch_mse, ch_mae, ch_nrmse, ch_nmae))
            # Conservation residual: |S_{t+1} - S_t - dt*(I - O - E)|
            delta_t = float(getattr(self.args, 'reservoir_delta_t', 1.0))
            pred_s = preds[:, :, 0]
            pred_i = preds[:, :, 1]
            pred_o = preds[:, :, 2]
            pred_e = preds[:, :, 3] if preds.shape[-1] >= 4 else np.zeros_like(pred_i)
            # Shift storage: prev_storage for step k is storage at step k-1
            prev_s = np.concatenate([pred_s[:, :1], pred_s[:, :-1]], axis=1)
            conservation_res = np.abs(pred_s - prev_s - delta_t * (pred_i - pred_o - pred_e))
            mean_cons_res = float(conservation_res.mean())
            max_cons_res = float(conservation_res.max())
            print('Conservation residual -> mean:{:.6f}, max:{:.6f}'.format(mean_cons_res, max_cons_res))
            f.write('Conservation residual -> mean:{:.6f}, max:{:.6f}\n'.format(mean_cons_res, max_cons_res))
        f.write('\n')
        f.write('\n')
        f.close()
        return
