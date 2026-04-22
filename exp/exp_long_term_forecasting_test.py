import itertools
import math
import os

import numpy as np
import torch

from exp.exp_basic_test import Exp_Basic_Test
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.dtw_metric import accelerated_dtw
from utils.metrics import CORR, RSE, metric


class Exp_Long_Term_Forecast_Test(Exp_Basic_Test, Exp_Long_Term_Forecast):
    def __init__(self, args):
        Exp_Basic_Test.__init__(self)
        Exp_Long_Term_Forecast.__init__(self, args)

    def _result_setting_name(self, setting):
        eval_suffix = str(getattr(self.args, 'eval_suffix', '') or '').strip()
        if eval_suffix:
            return '{}{}'.format(setting, eval_suffix)
        return setting

    def _dataset_report(self):
        return {
            'eval_name': str(getattr(self.args, 'eval_name', '') or ''),
            'data': str(getattr(self.args, 'data', '') or ''),
            'dataset_tag': str(getattr(self.args, 'dataset_tag', '') or ''),
            'root_path': str(getattr(self.args, 'root_path', '') or ''),
            'data_path': str(getattr(self.args, 'data_path', '') or ''),
            'data_path_trainval': str(getattr(self.args, 'data_path_trainval', '') or ''),
            'data_path_test': str(getattr(self.args, 'data_path_test', '') or ''),
        }

    def _analysis_solar_enabled(self):
        requested = getattr(self.args, 'analysis_solar_pcd', None)
        if requested is not None:
            return bool(requested)
        return bool(self.args.data == 'Solar' or self.args.model.endswith('SolarPhysics'))

    def _compare_projection_enabled(self):
        return bool(getattr(self.args, 'compare_projection_stages', True))

    def _save_analysis_tables(self):
        return bool(getattr(self.args, 'save_analysis_tables', True))

    def _safe_mape(self, pred, true):
        eps = float(getattr(self.args, 'metric_eps', 1e-6))
        denom = np.clip(np.abs(true), eps, None)
        return float(np.mean(np.abs((true - pred) / denom)))

    def _safe_mspe(self, pred, true):
        eps = float(getattr(self.args, 'metric_eps', 1e-6))
        denom = np.clip(np.abs(true), eps, None)
        return float(np.mean(np.square((true - pred) / denom)))

    def _compute_metric_bundle(self, pred, true, ymax):
        mae, mse, nrmse, nmae = metric(pred, true, ymax)
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mape': self._safe_mape(pred, true),
            'mspe': self._safe_mspe(pred, true),
            'nrmse': float(np.mean(np.asarray(nrmse))),
            'nmae': float(np.mean(np.asarray(nmae))),
            'rse': float(RSE(pred, true)),
            'corr': float(np.mean(np.asarray(CORR(pred, true)))),
        }

    def _channel_names(self, channel_count):
        custom_names = str(getattr(self.args, 'channel_names', '') or '').strip()
        if custom_names:
            parsed_names = [name.strip() for name in custom_names.split(',') if name.strip()]
        elif bool(getattr(self.args, 'is_three', False)) and channel_count >= 3:
            parsed_names = ['GHI', 'DNI', 'DHI']
        elif self._uses_reservoir_physics() and channel_count >= 3:
            parsed_names = ['storage', 'inflow', 'outflow', 'evaporation']
        else:
            parsed_names = []

        while len(parsed_names) < channel_count:
            parsed_names.append('channel_{}'.format(len(parsed_names)))
        return parsed_names[:channel_count]

    def _select_eval_channels(self, preds, trues):
        if bool(getattr(self.args, 'is_three', False)):
            return preds, trues
        if preds.shape[-1] > 1 and not bool(getattr(self.args, 'report_all_channels', False)):
            return preds[:, :, 0:1], trues[:, :, 0:1]
        return preds, trues

    def _channel_ymax(self, ymax, channel_idx):
        if isinstance(ymax, (list, tuple, np.ndarray)):
            ymax_arr = np.asarray(ymax)
            if channel_idx < ymax_arr.shape[0]:
                return float(ymax_arr[channel_idx])
        return ymax

    def _overall_ymax(self, ymax, channel_count):
        if channel_count == 1 and isinstance(ymax, (list, tuple, np.ndarray)):
            ymax_arr = np.asarray(ymax)
            if ymax_arr.size > 0:
                return float(ymax_arr.reshape(-1)[0])
        return ymax

    def _compute_dtw(self, preds, trues):
        if not bool(getattr(self.args, 'use_dtw', False)):
            return None

        dtw_list = []
        manhattan_distance = lambda x, y: np.abs(x - y)
        dtw_preds = preds[:, :, 0:1]
        dtw_trues = trues[:, :, 0:1]
        for idx in range(dtw_preds.shape[0]):
            if idx % 100 == 0:
                print('calculating dtw iter:', idx)
            pred_series = dtw_preds[idx].reshape(-1, 1)
            true_series = dtw_trues[idx].reshape(-1, 1)
            distance, _, _, _ = accelerated_dtw(pred_series, true_series, dist=manhattan_distance)
            dtw_list.append(distance)
        return float(np.mean(dtw_list))

    def _reservoir_metrics(self, preds):
        if not self._uses_reservoir_physics() or preds.shape[-1] < 3:
            return None

        delta_t = float(getattr(self.args, 'reservoir_delta_t', 1.0))
        pred_storage = preds[:, :, 0]
        pred_inflow = preds[:, :, 1]
        pred_outflow = preds[:, :, 2]
        pred_evap = preds[:, :, 3] if preds.shape[-1] >= 4 else np.zeros_like(pred_inflow)
        prev_storage = np.concatenate([pred_storage[:, :1], pred_storage[:, :-1]], axis=1)
        conservation_residual = np.abs(
            pred_storage - prev_storage - delta_t * (pred_inflow - pred_outflow - pred_evap)
        )
        return {
            'mean_conservation_residual': float(conservation_residual.mean()),
            'max_conservation_residual': float(conservation_residual.max()),
        }

    def _inverse_array(self, data_set, values):
        if not getattr(data_set, 'scale', False):
            return values
        shape = values.shape
        flat = values.reshape(shape[0] * shape[1], -1)
        restored = data_set.inverse_transform(flat).reshape(shape)
        return restored

    def _extract_model_trace(self):
        model_ref = self._model_ref()
        base_out = getattr(model_ref, 'latest_base_out', None)
        projected_out = getattr(model_ref, 'latest_projected_out', None)
        cos_zenith = getattr(model_ref, 'latest_cos_zenith', None)
        projection_trace = getattr(model_ref, 'latest_projection_trace', None)
        if base_out is None or projected_out is None:
            return None
        return {
            'base_out': base_out.detach().cpu().numpy(),
            'projected_out': projected_out.detach().cpu().numpy(),
            'cos_zenith': None if cos_zenith is None else cos_zenith.detach().cpu().numpy(),
            'projection_trace': projection_trace,
        }

    def _solar_cos_zenith(self, future_zenith):
        return np.clip(np.cos(np.deg2rad(future_zenith)), a_min=0.0, a_max=None)

    def _closure_abs(self, pred, cos_zenith):
        return np.abs(pred[..., 0] - pred[..., 1] * cos_zenith - pred[..., 2])

    def _closure_tol(self, pred, cos_zenith):
        abs_tol = float(getattr(self.args, 'feasible_abs_tol', 1.0))
        rel_tol = float(getattr(self.args, 'feasible_rel_tol', 0.01))
        closure_rhs = pred[..., 1] * cos_zenith + pred[..., 2]
        reference = np.maximum.reduce([
            np.abs(pred[..., 0]),
            np.abs(closure_rhs),
            np.ones_like(cos_zenith),
        ])
        return np.maximum(abs_tol, rel_tol * reference)

    def _negativity_masks(self, pred):
        negativity_tol = float(getattr(self.args, 'negativity_tol', 1.0))
        negative_part = np.maximum(-pred, 0.0)
        return negative_part, pred < (-negativity_tol)

    def _weak_upper_bound_alpha(self):
        model_ref = self._model_ref()
        physics_stack = getattr(model_ref, 'physics_stack', None)
        if physics_stack is None:
            return None
        projections = [getattr(physics_stack, 'initial_projection', None)]
        projections.extend(getattr(block, 'projection', None) for block in getattr(physics_stack, 'blocks', []))
        for projection in reversed([item for item in projections if item is not None]):
            alpha_fn = getattr(projection, '_upper_bound_alpha', None)
            if alpha_fn is None:
                continue
            alpha = alpha_fn()
            if alpha is not None:
                return float(alpha.detach().cpu().item())
        return None

    def _upper_bound_values(self, cos_zenith):
        if not bool(getattr(self.args, 'include_upper_bound_in_feasible', False)):
            return None
        alpha = self._weak_upper_bound_alpha()
        if alpha is None:
            return None
        solar_constant = float(getattr(self.args, 'solar_constant', 1367.0))
        eps = float(getattr(self.args, 'physics_eps', 1e-4))
        day_mask = (cos_zenith > float(getattr(self.args, 'night_cos_threshold', 0.0))).astype(np.float32)
        return solar_constant * np.clip(cos_zenith, a_min=eps, a_max=None) * alpha * day_mask

    def _solar_stage_metrics(self, pred, true, cos_zenith, ymax):
        error_metrics = self._compute_metric_bundle(pred, true, ymax)
        closure_abs = self._closure_abs(pred, cos_zenith)
        day_mask = cos_zenith > 0.0
        if np.any(day_mask):
            closure_day = float(closure_abs[day_mask].mean())
        else:
            closure_day = 0.0
        negative_part, negativity_mask = self._negativity_masks(pred)
        feasible_mask = closure_abs <= self._closure_tol(pred, cos_zenith)
        for channel_idx in range(min(pred.shape[-1], 3)):
            feasible_mask &= ~negativity_mask[..., channel_idx]

        upper_bound = self._upper_bound_values(cos_zenith)
        upper_bound_applied = upper_bound is not None
        if upper_bound_applied:
            feasible_mask &= pred[..., 0] <= (upper_bound + self._closure_tol(pred, cos_zenith))

        metrics = dict(error_metrics)
        metrics.update({
            'closure_violation_mae': float(closure_abs.mean()),
            'closure_violation_day_mae': closure_day,
            'feasible_rate': float(feasible_mask.mean()),
            'upper_bound_applied': upper_bound_applied,
        })

        channel_names = ['ghi', 'dni', 'dhi']
        for channel_idx, channel_name in enumerate(channel_names[:pred.shape[-1]]):
            metrics['negativity_violation_rate_{}'.format(channel_name)] = float(
                negativity_mask[..., channel_idx].mean()
            )
            metrics['negativity_violation_mag_{}'.format(channel_name)] = float(
                negative_part[..., channel_idx].mean()
            )

        arrays = {
            'closure_abs': closure_abs,
            'feasible_mask': feasible_mask,
            'negativity_mask': negativity_mask,
            'negative_part': negative_part,
            'sample_closure': closure_abs.mean(axis=1),
        }
        return metrics, arrays

    def _horizon_rows(self, stage_name, pred, true, cos_zenith):
        rows = []
        _, _, channel_count = pred.shape
        for horizon_idx in range(pred.shape[1]):
            pred_h = pred[:, horizon_idx:horizon_idx + 1, :]
            true_h = true[:, horizon_idx:horizon_idx + 1, :]
            cos_h = cos_zenith[:, horizon_idx:horizon_idx + 1]
            closure_abs = self._closure_abs(pred_h, cos_h)
            feasible_mask = closure_abs <= self._closure_tol(pred_h, cos_h)
            _, negativity_mask = self._negativity_masks(pred_h)
            for channel_idx in range(min(channel_count, 3)):
                feasible_mask &= ~negativity_mask[..., channel_idx]
            upper_bound = self._upper_bound_values(cos_h)
            if upper_bound is not None:
                feasible_mask &= pred_h[..., 0] <= (upper_bound + self._closure_tol(pred_h, cos_h))
            row = {
                'stage': stage_name,
                'horizon': int(horizon_idx + 1),
                'sample_count': int(pred.shape[0]),
                'mse': float(np.mean((pred_h - true_h) ** 2)),
                'mae': float(np.mean(np.abs(pred_h - true_h))),
                'closure_violation': float(closure_abs.mean()),
                'feasible_rate': float(feasible_mask.mean()),
            }
            for channel_idx, channel_name in enumerate(['ghi', 'dni', 'dhi'][:channel_count]):
                row['negativity_rate_{}'.format(channel_name)] = float(negativity_mask[..., channel_idx].mean())
            rows.append(row)
        return rows

    def _bucket_specs(self):
        edges = str(getattr(self.args, 'solar_bucket_edges', '0.0,0.1,0.2,0.5') or '0.0,0.1,0.2,0.5')
        parsed = sorted({float(item.strip()) for item in edges.split(',') if item.strip()})
        specs = [('night', None, 0.0)]
        prev = 0.0
        for edge in parsed:
            if edge <= 0.0:
                continue
            specs.append(('({:.2f},{:.2f}]'.format(prev, edge), prev, edge))
            prev = edge
        specs.append(('({:.2f},1.00]'.format(prev), prev, 1.0))
        return specs

    def _bucket_mask(self, cos_zenith, lower, upper):
        if lower is None:
            return cos_zenith <= 0.0
        if upper >= 1.0:
            return (cos_zenith > lower) & (cos_zenith <= upper + 1e-8)
        return (cos_zenith > lower) & (cos_zenith <= upper)

    def _bucket_rows(self, stage_name, pred, true, cos_zenith):
        closure_abs = self._closure_abs(pred, cos_zenith)
        feasible_mask = closure_abs <= self._closure_tol(pred, cos_zenith)
        _, negativity_mask = self._negativity_masks(pred)
        for channel_idx in range(min(pred.shape[-1], 3)):
            feasible_mask &= ~negativity_mask[..., channel_idx]
        upper_bound = self._upper_bound_values(cos_zenith)
        if upper_bound is not None:
            feasible_mask &= pred[..., 0] <= (upper_bound + self._closure_tol(pred, cos_zenith))

        rows = []
        for bucket_label, lower, upper in self._bucket_specs():
            mask = self._bucket_mask(cos_zenith, lower, upper)
            sample_count = int(mask.sum())
            if sample_count == 0:
                row = {
                    'stage': stage_name,
                    'bucket': bucket_label,
                    'bucket_low': lower,
                    'bucket_high': upper,
                    'sample_count': 0,
                    'mse': math.nan,
                    'mae': math.nan,
                    'closure_violation': math.nan,
                    'feasible_rate': math.nan,
                }
                for channel_name in ['ghi', 'dni', 'dhi'][:pred.shape[-1]]:
                    row['negativity_rate_{}'.format(channel_name)] = math.nan
                rows.append(row)
                continue

            row = {
                'stage': stage_name,
                'bucket': bucket_label,
                'bucket_low': lower,
                'bucket_high': upper,
                'sample_count': sample_count,
                'mse': float(np.mean(((pred - true) ** 2)[mask])),
                'mae': float(np.mean(np.abs(pred - true)[mask])),
                'closure_violation': float(np.mean(closure_abs[mask])),
                'feasible_rate': float(np.mean(feasible_mask[mask])),
            }
            for channel_idx, channel_name in enumerate(['ghi', 'dni', 'dhi'][:pred.shape[-1]]):
                row['negativity_rate_{}'.format(channel_name)] = float(np.mean(negativity_mask[..., channel_idx][mask]))
            rows.append(row)
        return rows

    def _projection_compare_report(self, pre_metrics, post_metrics):
        if pre_metrics is None or post_metrics is None:
            return {'status': 'unavailable'}
        return {
            'status': 'available',
            'pre': pre_metrics,
            'post': post_metrics,
            'delta_mse': float(pre_metrics['mse'] - post_metrics['mse']),
            'delta_mae': float(pre_metrics['mae'] - post_metrics['mae']),
            'delta_closure_violation': float(
                pre_metrics['closure_violation_mae'] - post_metrics['closure_violation_mae']
            ),
            'delta_feasible_rate': float(post_metrics['feasible_rate'] - pre_metrics['feasible_rate']),
        }

    def _geometry_manifold_payload(self, pre_pred, post_pred, cos_zenith):
        if pre_pred is None or post_pred is None:
            return None

        pre_flat = pre_pred.reshape(-1, pre_pred.shape[-1])
        post_flat = post_pred.reshape(-1, post_pred.shape[-1])
        cos_flat = cos_zenith.reshape(-1)

        valid_mask = np.isfinite(cos_flat)
        valid_mask &= cos_flat > float(getattr(self.args, 'night_cos_threshold', 0.0))
        valid_mask &= np.all(np.isfinite(pre_flat[:, :3]), axis=1)
        valid_mask &= np.all(np.isfinite(post_flat[:, :3]), axis=1)

        if not np.any(valid_mask):
            return None

        pre_valid = pre_flat[valid_mask]
        post_valid = post_flat[valid_mask]
        cos_valid = cos_flat[valid_mask]

        target_cos = getattr(self.args, 'geometry_slice_cos', None)
        if target_cos is None:
            target_cos = float(np.median(cos_valid))
        else:
            target_cos = float(target_cos)

        target_cos = float(np.clip(target_cos, 0.0, 1.0))
        slice_tol = float(getattr(self.args, 'geometry_slice_tol', 0.015))
        min_points = max(32, int(getattr(self.args, 'geometry_slice_min_points', 256)))
        max_points = max(64, int(getattr(self.args, 'geometry_slice_max_points', 1200)))

        delta = np.abs(cos_valid - target_cos)
        selected_idx = np.where(delta <= slice_tol)[0]
        selection_mode = 'tolerance'
        if selected_idx.size < min(min_points, cos_valid.size):
            keep = min(min_points, cos_valid.size)
            selected_idx = np.argsort(delta)[:keep]
            selection_mode = 'nearest'

        if selected_idx.size > max_points:
            ranked = selected_idx[np.argsort(delta[selected_idx])]
            selected_idx = ranked[:max_points]
            selection_mode = '{}+cap'.format(selection_mode)

        pre_slice = pre_valid[selected_idx]
        post_slice = post_valid[selected_idx]
        cos_slice = cos_valid[selected_idx]
        plane_cos = float(np.median(cos_slice))
        plane_norm = math.sqrt(plane_cos ** 2 + 2.0)

        pre_residual = pre_slice[:, 0] - (pre_slice[:, 1] * plane_cos + pre_slice[:, 2])
        post_residual = post_slice[:, 0] - (post_slice[:, 1] * plane_cos + post_slice[:, 2])
        pre_distance = np.abs(pre_residual) / plane_norm
        post_distance = np.abs(post_residual) / plane_norm

        theta_deg = float(np.rad2deg(np.arccos(np.clip(plane_cos, 0.0, 1.0))))
        summary = {
            'target_cos': target_cos,
            'plane_cos': plane_cos,
            'plane_zenith_deg': theta_deg,
            'slice_tol': slice_tol,
            'selection_mode': selection_mode,
            'point_count': int(selected_idx.size),
            'pre_mean_plane_distance': float(np.mean(pre_distance)),
            'post_mean_plane_distance': float(np.mean(post_distance)),
            'pre_median_plane_distance': float(np.median(pre_distance)),
            'post_median_plane_distance': float(np.median(post_distance)),
        }
        return {
            'summary': summary,
            'pre_points': pre_slice,
            'post_points': post_slice,
        }

    def _case_study_rank_indices(self, selection, topk, bucket_rows, sample_improvement, sample_bucket_labels):
        if sample_improvement is None:
            return np.arange(min(topk, len(sample_bucket_labels)))
        if selection == 'best_improvement':
            return np.argsort(-sample_improvement)[:topk]
        if selection == 'representative_bucket':
            selected = []
            seen = set()
            for row in bucket_rows:
                bucket = row['bucket']
                if bucket in seen or row['sample_count'] <= 0:
                    continue
                seen.add(bucket)
                bucket_indices = np.where(sample_bucket_labels == bucket)[0]
                if bucket_indices.size == 0:
                    continue
                median_improvement = np.median(sample_improvement[bucket_indices])
                representative = bucket_indices[np.argmin(np.abs(sample_improvement[bucket_indices] - median_improvement))]
                selected.append(int(representative))
                if len(selected) >= topk:
                    break
            return np.asarray(selected, dtype=np.int64)
        return np.argsort(sample_improvement)[:topk]

    def _save_case_studies_payload(self, case_dir, payload):
        sample_count = payload['post_pred'].shape[0]
        topk = min(int(getattr(self.args, 'case_study_topk', 3)), sample_count)
        selection = str(getattr(self.args, 'case_study_selection', 'worst_improvement'))
        pre_closure = payload['pre_arrays']['sample_closure'] if payload['pre_arrays'] is not None else None
        post_closure = payload['post_arrays']['sample_closure']
        sample_improvement = None if pre_closure is None else (pre_closure - post_closure)

        cos_flat = payload['cos_zenith'].reshape(-1)
        bucket_labels_flat = np.empty_like(cos_flat, dtype=object)
        for bucket_label, lower, upper in self._bucket_specs():
            mask = self._bucket_mask(payload['cos_zenith'], lower, upper)
            bucket_labels_flat[mask.reshape(-1)] = bucket_label
        sample_bucket_labels = bucket_labels_flat.reshape(payload['cos_zenith'].shape).astype(object)
        sample_bucket_labels = sample_bucket_labels[:, 0]

        selected_indices = self._case_study_rank_indices(
            selection,
            topk,
            payload['bucket_rows'],
            sample_improvement,
            sample_bucket_labels,
        )

        for rank, sample_idx in enumerate(selected_indices):
            arrays = {
                'history': payload['history'][sample_idx],
                'true': payload['true'][sample_idx],
                'post_projection': payload['post_pred'][sample_idx],
                'cos_zenith': payload['cos_zenith'][sample_idx],
                'sample_index': np.asarray(sample_idx, dtype=np.int64),
                'bucket': np.asarray(sample_bucket_labels[sample_idx]),
            }
            if payload['pre_pred'] is not None:
                arrays['pre_projection'] = payload['pre_pred'][sample_idx]
                arrays['closure_improvement'] = np.asarray(sample_improvement[sample_idx], dtype=np.float32)
            self._write_npz(os.path.join(case_dir, 'case_{:02d}.npz'.format(rank + 1)), **arrays)

    def _maybe_plot(self, save_path, plot_fn):
        if not bool(getattr(self.args, 'save_analysis_plots', False)):
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig = plot_fn(plt)
            fig.tight_layout()
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception as exc:
            print('skip plot {}: {}'.format(save_path, exc))

    def _plot_horizon_error(self, save_path, horizon_rows):
        def _builder(plt):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            stage_to_rows = {}
            for row in horizon_rows:
                stage_to_rows.setdefault(row['stage'], []).append(row)
            for stage, rows in stage_to_rows.items():
                horizons = [row['horizon'] for row in rows]
                axes[0].plot(horizons, [row['mse'] for row in rows], label=stage)
                axes[1].plot(horizons, [row['mae'] for row in rows], label=stage)
            axes[0].set_title('Horizon-wise MSE')
            axes[1].set_title('Horizon-wise MAE')
            for axis in axes:
                axis.set_xlabel('Horizon')
                axis.legend()
            return fig

        self._maybe_plot(save_path, _builder)

    def _plot_horizon_closure(self, save_path, horizon_rows):
        def _builder(plt):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            stage_to_rows = {}
            for row in horizon_rows:
                stage_to_rows.setdefault(row['stage'], []).append(row)
            for stage, rows in stage_to_rows.items():
                horizons = [row['horizon'] for row in rows]
                axes[0].plot(horizons, [row['closure_violation'] for row in rows], label=stage)
                axes[1].plot(horizons, [row['feasible_rate'] for row in rows], label=stage)
            axes[0].set_title('Horizon-wise Closure Violation')
            axes[1].set_title('Horizon-wise Feasible Rate')
            for axis in axes:
                axis.set_xlabel('Horizon')
                axis.legend()
            return fig

        self._maybe_plot(save_path, _builder)

    def _plot_bucket_metrics(self, save_path, bucket_rows):
        def _builder(plt):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            stage_to_rows = {}
            for row in bucket_rows:
                stage_to_rows.setdefault(row['stage'], []).append(row)
            for stage, rows in stage_to_rows.items():
                labels = [row['bucket'] for row in rows]
                x = np.arange(len(labels))
                axes[0].plot(x, [row['mae'] for row in rows], marker='o', label=stage)
                axes[1].plot(x, [row['closure_violation'] for row in rows], marker='o', label=stage)
                axes[0].set_xticks(x, labels, rotation=25)
                axes[1].set_xticks(x, labels, rotation=25)
            axes[0].set_title('Bucket-wise MAE')
            axes[1].set_title('Bucket-wise Closure Violation')
            for axis in axes:
                axis.legend()
            return fig

        self._maybe_plot(save_path, _builder)

    def _plot_projection_compare(self, save_path, compare_report):
        if compare_report.get('status') != 'available':
            return

        def _builder(plt):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            labels = ['MSE', 'MAE', 'Closure', 'Feasible']
            pre_values = [
                compare_report['pre']['mse'],
                compare_report['pre']['mae'],
                compare_report['pre']['closure_violation_mae'],
                compare_report['pre']['feasible_rate'],
            ]
            post_values = [
                compare_report['post']['mse'],
                compare_report['post']['mae'],
                compare_report['post']['closure_violation_mae'],
                compare_report['post']['feasible_rate'],
            ]
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width / 2, pre_values, width=width, label='pre')
            ax.bar(x + width / 2, post_values, width=width, label='post')
            ax.set_xticks(x, labels)
            ax.set_title('Projection Compare')
            ax.legend()
            return fig

        self._maybe_plot(save_path, _builder)

    def _plot_geometry_of_failure(self, save_path, manifold_payload):
        if manifold_payload is None:
            return

        def _builder(plt):
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            summary = manifold_payload['summary']
            pre_points = manifold_payload['pre_points']
            post_points = manifold_payload['post_points']

            combined = np.concatenate([pre_points, post_points], axis=0)
            x_all = combined[:, 1]
            y_all = combined[:, 2]
            z_all = combined[:, 0]

            def _axis_limits(values):
                lo = float(np.quantile(values, 0.02))
                hi = float(np.quantile(values, 0.98))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo = float(np.nanmin(values))
                    hi = float(np.nanmax(values))
                if math.isclose(lo, hi):
                    pad = max(1.0, abs(lo) * 0.1)
                    return lo - pad, hi + pad
                pad = max(1.0, (hi - lo) * 0.08)
                return lo - pad, hi + pad

            x_min, x_max = _axis_limits(x_all)
            y_min, y_max = _axis_limits(y_all)
            z_min, z_max = _axis_limits(z_all)

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 25),
                np.linspace(y_min, y_max, 25),
            )
            zz = summary['plane_cos'] * xx + yy

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(
                xx,
                yy,
                zz,
                alpha=0.22,
                color='#8db3d3',
                linewidth=0,
                antialiased=True,
                shade=False,
            )
            ax.scatter(
                pre_points[:, 1],
                pre_points[:, 2],
                pre_points[:, 0],
                s=14,
                c='#d62728',
                alpha=0.33,
                depthshade=False,
                label='Soft constraint / pre-projection',
            )
            ax.scatter(
                post_points[:, 1],
                post_points[:, 2],
                post_points[:, 0],
                s=16,
                c='#1f77b4',
                alpha=0.88,
                depthshade=False,
                label='PCD / projected output',
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(min(z_min, float(np.nanmin(zz))), max(z_max, float(np.nanmax(zz))))
            ax.set_xlabel('DNI')
            ax.set_ylabel('DHI')
            ax.set_zlabel('GHI')
            ax.set_title('Geometry of Failure vs. Absolute Manifold Adherence')
            ax.view_init(elev=24, azim=-58)
            if hasattr(ax, 'set_box_aspect'):
                ax.set_box_aspect((1.15, 1.0, 0.95))

            annotation = (
                r'Constraint plane: $\mathrm{{GHI}}=\mathrm{{DNI}}\cdot\cos(z)+\mathrm{{DHI}}$' '\n'
                'Slice zenith: {:.1f} deg  |  cos(z): {:.3f}  |  points: {}\n'
                'Mean distance to plane  pre={:.2f}  post={:.4f}'
            ).format(
                summary['plane_zenith_deg'],
                summary['plane_cos'],
                summary['point_count'],
                summary['pre_mean_plane_distance'],
                summary['post_mean_plane_distance'],
            )
            ax.text2D(
                0.02,
                0.98,
                annotation,
                transform=ax.transAxes,
                va='top',
                fontsize=10,
                bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#d0d7de'},
            )
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.83))
            return fig

        self._maybe_plot(save_path, _builder)

    def _run_solar_analysis(
        self,
        run_dir,
        analysis_dir,
        histories_phys,
        preds_phys,
        trues_phys,
        pre_projection_phys,
        future_zenith,
        ymax,
    ):
        cos_zenith = self._solar_cos_zenith(future_zenith)
        post_metrics, post_arrays = self._solar_stage_metrics(preds_phys, trues_phys, cos_zenith, ymax)
        pre_metrics = None
        pre_arrays = None
        if pre_projection_phys is not None and self._compare_projection_enabled():
            pre_metrics, pre_arrays = self._solar_stage_metrics(pre_projection_phys, trues_phys, cos_zenith, ymax)

        compare_report = self._projection_compare_report(pre_metrics, post_metrics)
        manifold_payload = None
        if (
            pre_projection_phys is not None
            and self._compare_projection_enabled()
            and bool(getattr(self.args, 'save_internal_projection_geometry', False))
        ):
            manifold_payload = self._geometry_manifold_payload(pre_projection_phys, preds_phys, cos_zenith)

        horizon_rows = self._horizon_rows('post', preds_phys, trues_phys, cos_zenith)
        bucket_rows = self._bucket_rows('post', preds_phys, trues_phys, cos_zenith)
        if pre_projection_phys is not None and self._compare_projection_enabled():
            horizon_rows.extend(self._horizon_rows('pre', pre_projection_phys, trues_phys, cos_zenith))
            bucket_rows.extend(self._bucket_rows('pre', pre_projection_phys, trues_phys, cos_zenith))

        overall_report = {
            'setting': os.path.basename(run_dir),
            'checkpoint': self._checkpoint_info,
            'post': post_metrics,
            'pre': pre_metrics,
            'projection_compare': compare_report,
            'geometry_manifold': None if manifold_payload is None else manifold_payload['summary'],
        }

        if self._save_analysis_tables():
            self._write_json(os.path.join(analysis_dir, 'overall_physics.json'), overall_report)
            self._write_json(os.path.join(analysis_dir, 'projection_compare.json'), compare_report)
            self._write_csv_rows(os.path.join(analysis_dir, 'horizon_metrics.csv'), horizon_rows)
            self._write_csv_rows(os.path.join(analysis_dir, 'solar_buckets.csv'), bucket_rows)
            if manifold_payload is not None:
                self._write_json(
                    os.path.join(analysis_dir, 'geometry_manifold.json'),
                    manifold_payload['summary'],
                )

        plots_dir = self._ensure_dir(os.path.join(analysis_dir, 'plots'))
        self._plot_horizon_error(os.path.join(plots_dir, 'horizon_error.png'), horizon_rows)
        self._plot_horizon_closure(os.path.join(plots_dir, 'horizon_closure.png'), horizon_rows)
        self._plot_bucket_metrics(os.path.join(plots_dir, 'solar_buckets.png'), bucket_rows)
        self._plot_projection_compare(os.path.join(plots_dir, 'projection_compare.png'), compare_report)
        self._plot_geometry_of_failure(
            os.path.join(plots_dir, 'geometry_failure_vs_manifold.png'),
            manifold_payload,
        )

        if bool(getattr(self.args, 'save_case_studies', False)):
            case_dir = self._ensure_dir(os.path.join(analysis_dir, 'case_studies'))
            self._save_case_studies_payload(case_dir, {
                'history': histories_phys,
                'true': trues_phys,
                'post_pred': preds_phys,
                'pre_pred': pre_projection_phys,
                'cos_zenith': cos_zenith,
                'pre_arrays': pre_arrays,
                'post_arrays': post_arrays,
                'bucket_rows': [row for row in bucket_rows if row['stage'] == 'post'],
            })

        return {
            'overall': overall_report,
            'projection_compare': compare_report,
            'horizon_rows': horizon_rows,
            'bucket_rows': bucket_rows,
        }

    def test(self, setting, test=1):
        checkpoint_path = None
        if test:
            checkpoint_path = self._load_test_checkpoint(setting)
        result_setting = self._result_setting_name(setting)

        eval_split = str(getattr(self.args, 'eval_split', 'test') or 'test')
        data_set, data_loader = self._get_data(flag=eval_split)

        preds_eval = []
        trues_eval = []
        preds_phys = []
        trues_phys = []
        histories_phys = []
        future_zenith = []
        pre_projection_phys = []

        analysis_solar = self._analysis_solar_enabled()

        self.model.eval()
        if bool(getattr(self.args, 'debug', False)):
            data_loader = list(itertools.islice(data_loader, 10))

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if bool(getattr(self.args, 'use_amp', False)):
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, data_set
                        )
                else:
                    outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, data_set)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_main_pred = batch_y_main[:, -self.args.pred_len:, :]
                batch_x_main_np = batch_x_main.detach().cpu().numpy()
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_main_np = batch_y_main_pred.detach().cpu().numpy()

                outputs_eval_np = outputs_np.copy()
                batch_y_eval_np = batch_y_main_np.copy()
                if getattr(data_set, 'scale', False) and bool(getattr(self.args, 'inverse', False)):
                    outputs_eval_np = self._inverse_array(data_set, outputs_eval_np)
                    batch_y_eval_np = self._inverse_array(data_set, batch_y_eval_np)

                outputs_eval_np = self._select_supervised_channels(outputs_eval_np)
                batch_y_eval_np = self._select_supervised_channels(batch_y_eval_np)
                preds_eval.append(outputs_eval_np)
                trues_eval.append(batch_y_eval_np)

                if analysis_solar:
                    pred_phys_np = self._inverse_array(data_set, outputs_np)
                    true_phys_np = self._inverse_array(data_set, batch_y_main_np)
                    history_phys_np = self._inverse_array(data_set, batch_x_main_np)
                    preds_phys.append(self._select_supervised_channels(pred_phys_np))
                    trues_phys.append(self._select_supervised_channels(true_phys_np))
                    histories_phys.append(self._select_supervised_channels(history_phys_np))

                    future_zenith.append(batch_y[:, -self.args.pred_len:, -1].detach().cpu().numpy())
                    trace = self._extract_model_trace()
                    if trace is not None and self._compare_projection_enabled():
                        pre_projection_np = self._inverse_array(data_set, trace['base_out'])
                        pre_projection_phys.append(self._select_supervised_channels(pre_projection_np))

        preds = np.concatenate(preds_eval, axis=0)
        trues = np.concatenate(trues_eval, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds, trues = self._select_eval_channels(preds, trues)
        print('eval shape:', preds.shape, trues.shape)

        ymax = getattr(data_set, 'ymax', 1.0)
        overall_metrics = self._compute_metric_bundle(preds, trues, self._overall_ymax(ymax, preds.shape[-1]))
        dtw_score = self._compute_dtw(preds, trues)
        if dtw_score is not None:
            overall_metrics['dtw'] = dtw_score

        channel_names = self._channel_names(preds.shape[-1])
        per_channel_metrics = []
        if preds.shape[-1] > 1:
            for channel_idx, channel_name in enumerate(channel_names):
                channel_pred = preds[:, :, channel_idx:channel_idx + 1]
                channel_true = trues[:, :, channel_idx:channel_idx + 1]
                channel_ymax = self._channel_ymax(ymax, channel_idx)
                per_channel_metrics.append({
                    'name': channel_name,
                    'metrics': self._compute_metric_bundle(channel_pred, channel_true, channel_ymax),
                })

        reservoir_metrics = self._reservoir_metrics(preds)

        print(
            'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, '
            'nrmse:{nrmse}, nmae:{nmae}, rse:{rse}, corr:{corr}'.format(**overall_metrics)
        )
        if dtw_score is not None:
            print('dtw:{}'.format(dtw_score))
        for channel_result in per_channel_metrics:
            print('{} -> {}'.format(channel_result['name'], channel_result['metrics']))
        if reservoir_metrics is not None:
            print('reservoir -> {}'.format(reservoir_metrics))

        results_root = self._ensure_dir(str(getattr(self.args, 'results_dir', './results_test')))
        run_dir = self._ensure_dir(os.path.join(results_root, result_setting))
        analysis_dir = self._ensure_dir(os.path.join(run_dir, 'analysis'))
        split_prefix = eval_split.lower()

        report = {
            'setting': setting,
            'result_setting': result_setting,
            'eval_split': eval_split,
            'dataset': self._dataset_report(),
            'checkpoint_path': checkpoint_path,
            'checkpoint': self._checkpoint_info,
            'pred_shape': list(preds.shape),
            'true_shape': list(trues.shape),
            'metrics': overall_metrics,
            'per_channel_metrics': per_channel_metrics,
            'reservoir_metrics': reservoir_metrics,
        }

        metrics_path = os.path.join(run_dir, '{}_metrics.json'.format(split_prefix))
        self._write_json(metrics_path, report)

        summary_path = os.path.join(results_root, 'result_long_term_forecast_test.txt')
        with open(summary_path, 'a', encoding='utf-8') as file_obj:
            file_obj.write(result_setting + '  \n')
            file_obj.write('{}\n'.format(report['metrics']))
            if per_channel_metrics:
                file_obj.write('{}\n'.format(per_channel_metrics))
            if reservoir_metrics is not None:
                file_obj.write('{}\n'.format(reservoir_metrics))
            file_obj.write('\n')

        if bool(getattr(self.args, 'save_predictions', False)):
            self._write_npz(
                os.path.join(run_dir, '{}_predictions.npz'.format(split_prefix)),
                preds=preds,
                trues=trues,
            )

        if analysis_solar and preds_phys and trues_phys and future_zenith:
            preds_phys_arr = np.concatenate(preds_phys, axis=0).reshape(-1, preds_phys[0].shape[-2], preds_phys[0].shape[-1])
            trues_phys_arr = np.concatenate(trues_phys, axis=0).reshape(-1, trues_phys[0].shape[-2], trues_phys[0].shape[-1])
            histories_phys_arr = np.concatenate(histories_phys, axis=0).reshape(-1, histories_phys[0].shape[-2], histories_phys[0].shape[-1])
            future_zenith_arr = np.concatenate(future_zenith, axis=0).reshape(-1, future_zenith[0].shape[-1])
            pre_projection_arr = None
            if pre_projection_phys and len(pre_projection_phys) == len(preds_phys):
                pre_projection_arr = np.concatenate(pre_projection_phys, axis=0).reshape(
                    -1, pre_projection_phys[0].shape[-2], pre_projection_phys[0].shape[-1]
                )
            solar_analysis_report = self._run_solar_analysis(
                run_dir,
                analysis_dir,
                histories_phys_arr,
                preds_phys_arr,
                trues_phys_arr,
                pre_projection_arr,
                future_zenith_arr,
                getattr(data_set, 'ymax', 1.0),
            )
            if bool(getattr(self.args, 'save_solar_geometry_payload', False)):
                self._write_npz(
                    os.path.join(analysis_dir, 'solar_geometry_payload.npz'),
                    pred=preds_phys_arr.astype(np.float32),
                    true=trues_phys_arr.astype(np.float32),
                    history=histories_phys_arr.astype(np.float32),
                    cos_zenith=self._solar_cos_zenith(future_zenith_arr).astype(np.float32),
                )
            report['solar_analysis'] = {
                'overall_physics_path': os.path.join(analysis_dir, 'overall_physics.json'),
                'projection_compare_path': os.path.join(analysis_dir, 'projection_compare.json'),
                'post_closure_violation_mae': solar_analysis_report['overall']['post']['closure_violation_mae'],
                'post_feasible_rate': solar_analysis_report['overall']['post']['feasible_rate'],
            }
            if bool(getattr(self.args, 'save_solar_geometry_payload', False)):
                report['solar_analysis']['geometry_payload_path'] = os.path.join(
                    analysis_dir,
                    'solar_geometry_payload.npz',
                )
            self._write_json(metrics_path, report)

        print('saved metrics to {}'.format(metrics_path))
        return report
