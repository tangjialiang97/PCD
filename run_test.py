import argparse
import hashlib
import json
import os
import random
import re

import numpy as np
import torch
import torch.backends

from exp.exp_long_term_forecasting_test import Exp_Long_Term_Forecast_Test
from utils.print_args import print_args


def str2bool(value):
    return str(value).lower() in ('1', 'true', 't', 'yes', 'y')


def sanitize_setting_token(value, default='unknown', max_len=48):
    token = os.path.splitext(os.path.basename(str(value)))[0]
    token = re.sub(r'[^0-9A-Za-z]+', '-', token).strip('-')
    if not token:
        token = default
    return token[:max_len]


def compress_dataset_tag(value, max_len=24):
    token = sanitize_setting_token(value, default='data', max_len=64)
    token = re.sub(r'-(?:19|20)\d{2}-(?:19|20)\d{2}(?=-|$)', '', token)

    replacements = (
        ('combined-no-zero', 'cnz'),
        ('combined-nozero', 'cnz'),
        ('nonzero', 'nz'),
        ('trainval', 'trv'),
        ('timestamp', 'ts'),
        ('target', 'tgt'),
    )
    lowered = token.lower()
    for old, new in replacements:
        lowered = lowered.replace(old, new)

    parts = [part for part in lowered.split('-') if part]
    compact_parts = []
    for part in parts:
        if len(part) <= 4 or part.isdigit():
            compact_parts.append(part)
        elif re.fullmatch(r'[a-z]+', part):
            compact_parts.append(part[:4])
        else:
            compact_parts.append(part[:6])

    compact = '-'.join(compact_parts).strip('-')
    if not compact:
        compact = lowered[:max_len]
    return compact[:max_len]


def build_dataset_tag(args):
    if getattr(args, 'data_path', ''):
        return compress_dataset_tag(args.data_path)

    trainval_tag = compress_dataset_tag(getattr(args, 'data_path_trainval', 'trainval.csv'))
    test_tag = compress_dataset_tag(getattr(args, 'data_path_test', 'test.csv'))
    if trainval_tag or test_tag:
        return '{}-{}'.format(trainval_tag, test_tag)[:64]
    return compress_dataset_tag(getattr(args, 'data', 'dataset'))


def build_hidden_config_hash(parts, prefix='cfg', length=10):
    payload = '|'.join(str(part) for part in parts)
    digest = hashlib.md5(payload.encode('utf-8')).hexdigest()[:length]
    return '{}{}'.format(prefix, digest)


def coerce_override_value(current_value, raw_value):
    if raw_value is None:
        return None
    if isinstance(current_value, bool):
        return raw_value if isinstance(raw_value, bool) else str2bool(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, list):
        if isinstance(raw_value, list):
            return raw_value
        return [item.strip() for item in str(raw_value).split(',') if item.strip()]
    return raw_value


def parse_inline_cross_data_configs(raw_text):
    configs = []
    raw_text = str(raw_text or '').strip()
    if not raw_text:
        return configs

    for spec in [item.strip() for item in raw_text.split(';') if item.strip()]:
        config = {}
        for pair in [item.strip() for item in spec.split(',') if item.strip()]:
            if '=' not in pair:
                raise ValueError(
                    'Invalid cross-data config segment: {!r}. Use key=value pairs separated by commas.'.format(pair)
                )
            key, value = pair.split('=', 1)
            config[key.strip()] = value.strip()
        if config:
            configs.append(config)
    return configs


def load_cross_data_configs(args):
    configs = []

    config_file = str(getattr(args, 'cross_data_config_file', '') or '').strip()
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as file_obj:
            payload = json.load(file_obj)
        if isinstance(payload, dict):
            payload = payload.get('datasets', [])
        if not isinstance(payload, list):
            raise ValueError('cross_data_config_file must contain a JSON list or {"datasets": [...]}')
        configs.extend(payload)

    configs.extend(parse_inline_cross_data_configs(getattr(args, 'cross_data_configs', '')))
    return configs


def apply_cross_data_overrides(base_args, overrides):
    eval_args = argparse.Namespace(**vars(base_args))

    for key, raw_value in overrides.items():
        if key in ('name', 'alias'):
            continue
        if not hasattr(eval_args, key):
            raise ValueError('Unknown cross-data override key: {}'.format(key))
        current_value = getattr(eval_args, key)
        setattr(eval_args, key, coerce_override_value(current_value, raw_value))

    eval_args.dataset_tag = build_dataset_tag(eval_args)
    eval_name = overrides.get('name') or overrides.get('alias')
    if not eval_name:
        eval_name = '{}-{}'.format(
            getattr(eval_args, 'data', 'dataset'),
            getattr(eval_args, 'dataset_tag', 'eval'),
        )
    eval_args.eval_name = sanitize_setting_token(eval_name, default='eval')
    eval_args.eval_suffix = '__eval_{}'.format(eval_args.eval_name)
    return eval_args


def add_common_args(parser):
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, currently run_test.py supports long_term_forecast')
    parser.add_argument('--is_training', type=int, default=0, help='status flag kept for compatibility')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name')

    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--seq_len_hours', type=int, default=18, help='input sequence length in hours')
    parser.add_argument('--pred_len_hours', type=int, default=6, help='prediction sequence length in hours')
    parser.add_argument('--label_len_hours', type=int, default=0, help='start token length in hours')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--data_path_trainval', type=str, default='trainval_timestamp_target.csv', help='train/val file')
    parser.add_argument('--data_path_test', type=str, default='test_timestamp_target.csv', help='test file')

    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (percent)')

    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--txt_name', type=str, default='', help='txt name')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='series decomposition method')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='length of segmen-wise iteration of SegRNN')
    parser.add_argument('--slide_step', type=int, default=1, help='sliding window step')

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='number of experiment settings to evaluate')
    parser.add_argument('--train_epochs', type=int, default=10, help='kept for compatibility')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='kept for compatibility')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='kept for compatibility')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
    parser.add_argument('--use_auxiliary', action='store_true', help='use auxiliary loss', default=False)
    parser.add_argument('--_uses_external_physics', action='store_true', default=False,
                        help='use external physics loss')
    parser.add_argument('--has_relhum', action='store_true', help='dataset has relhum column', default=False)
    parser.add_argument('--closure_weight', type=float, default=1.0, help='closure weight')
    parser.add_argument('--limit_weight', type=float, default=1.0, help='limit weight')
    parser.add_argument('--solar_constant', type=float, default=1367.0, help='solar constant')
    parser.add_argument('--var_gamma', type=float, default=0.1, help='variability gamma')
    parser.add_argument('--var_weight', type=float, default=1.0, help='variability weight')
    parser.add_argument('--closure_warmup_epochs', type=int, default=2, help='closure warmup epochs')
    parser.add_argument('--solar_dhi_weight', type=float, default=0.0,
                        help='extra weighted daytime supervision on DHI for internal solar physics models')
    parser.add_argument('--solar_dhi_ratio_floor_weight', type=float, default=0.0,
                        help='penalty weight preventing DHI/GHI ratio collapse toward zero in internal solar physics models')
    parser.add_argument('--solar_dhi_ratio_floor_cap', type=float, default=0.05,
                        help='upper cap of the minimum DHI/GHI ratio enforced during daytime')
    parser.add_argument('--solar_dhi_day_cos_threshold', type=float, default=0.05,
                        help='daytime cos(zenith) threshold used by the DHI anti-collapse regularizer')
    parser.add_argument('--solar_dhi_ref_floor', type=float, default=100.0,
                        help='reference floor used to normalise DHI supervision for internal solar physics models')
    parser.add_argument('--solar_dhi_beta', type=float, default=0.03,
                        help='smooth L1 beta used by the DHI anti-collapse supervision term')

    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')

    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    parser.add_argument('--use_dtw', type=str2bool, default=False,
                        help='whether to calculate dtw metric')
    parser.add_argument('--debug', type=str2bool, default=False, help='debug mode')

    parser.add_argument('--augmentation_ratio', type=int, default=0, help='How many times to augment')
    parser.add_argument('--seed', type=int, default=2, help='Randomization seed')
    parser.add_argument('--jitter', default=False, action='store_true', help='Jitter preset augmentation')
    parser.add_argument('--scaling', default=False, action='store_true', help='Scaling preset augmentation')
    parser.add_argument('--permutation', default=False, action='store_true', help='Equal Length Permutation augmentation')
    parser.add_argument('--randompermutation', default=False, action='store_true', help='Random Length Permutation augmentation')
    parser.add_argument('--magwarp', default=False, action='store_true', help='Magnitude warp augmentation')
    parser.add_argument('--timewarp', default=False, action='store_true', help='Time warp augmentation')
    parser.add_argument('--windowslice', default=False, action='store_true', help='Window slice augmentation')
    parser.add_argument('--windowwarp', default=False, action='store_true', help='Window warp augmentation')
    parser.add_argument('--rotation', default=False, action='store_true', help='Rotation augmentation')
    parser.add_argument('--spawner', default=False, action='store_true', help='SPAWNER augmentation')
    parser.add_argument('--dtwwarp', default=False, action='store_true', help='DTW warp augmentation')
    parser.add_argument('--shapedtwwarp', default=False, action='store_true', help='Shape DTW warp augmentation')
    parser.add_argument('--wdba', default=False, action='store_true', help='Weighted DBA augmentation')
    parser.add_argument('--discdtw', default=False, action='store_true', help='Discriminative DTW warp augmentation')
    parser.add_argument('--discsdtw', default=False, action='store_true', help='Discriminative shapeDTW warp augmentation')
    parser.add_argument('--extra_tag', type=str, default='', help='Anything extra')
    parser.add_argument('--delta', type=float, default=0.2, help='Delta for PostProcessingNet')
    parser.add_argument('--sample_interval', type=float, default=15, help='sample interval')

    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--timesfm_per_core_batch_size', type=int, default=128,
                        help='TimesFM internal per-device batch size')
    parser.add_argument('--timesfm_force_flip_invariance', type=str2bool, default=False,
                        help='enable TimesFM flip-invariance decoding')
    parser.add_argument('--timesfm_use_continuous_quantile_head', type=str2bool, default=False,
                        help='enable TimesFM continuous quantile head')
    parser.add_argument('--timesfm_fix_quantile_crossing', type=str2bool, default=False,
                        help='enable TimesFM quantile crossing correction')
    parser.add_argument('--timesfm_infer_is_positive', type=str2bool, default=False,
                        help='enforce positive TimesFM forecasts when inputs are positive')
    parser.add_argument('--timesfm_torch_compile', type=str2bool, default=True,
                        help='enable torch.compile when loading TimesFM checkpoint')
    parser.add_argument('--time_moe_use_cache', type=str2bool, default=True,
                        help='enable KV cache during TimeMoE decoding')
    parser.add_argument('--tirex_batch_size', type=int, default=0,
                        help='internal TiRex forecast batch size')
    parser.add_argument('--tirex_compile', type=str2bool, default=False,
                        help='enable torch.compile for TiRex')
    parser.add_argument('--ft_adapter_hidden_dim', type=int, default=128,
                        help='adapter hidden width for FT wrappers')
    parser.add_argument('--ft_adapter_bottleneck_dim', type=int, default=32,
                        help='adapter bottleneck width for FT wrappers')
    parser.add_argument('--ft_adapter_dropout', type=float, default=0.1,
                        help='adapter dropout for FT wrappers')
    parser.add_argument('--is_three', action='store_true', default=False,
                        help='report GHI/DNI/DHI metrics and 3-channel average')
    parser.add_argument('--use_internal_physics', action='store_true', default=False,
                        help='enable internal solar physics constraints')
    parser.add_argument('--physics_layers', type=int, default=2, help='number of solar physics blocks')
    parser.add_argument('--physics_hidden_dim', type=int, default=128, help='hidden dimension of solar physics blocks')
    parser.add_argument('--physics_kernel_size', type=int, default=3, help='temporal kernel size in solar physics blocks')
    parser.add_argument('--night_cos_threshold', type=float, default=0.0, help='daytime threshold for cosine zenith')
    parser.add_argument('--physics_eps', type=float, default=1e-4, help='stability epsilon')
    parser.add_argument('--physics_context_mode', type=str, default='baseline',
                        choices=['baseline', 'base_only', 'gated'],
                        help='how DLinearSolarPhysics combines history_context and base_out before the physics stack')
    parser.add_argument('--physics_context_gate_init', type=float, default=2.0,
                        help='initial logit for gated history-context fusion in DLinearSolarPhysics')
    parser.add_argument('--physics_state_residual', action='store_true', default=False,
                        help='enable residual state update branch')
    parser.add_argument('--physics_gated_feedback', action='store_true', default=False,
                        help='enable gated feedback for SolarPhysicsBlock')
    parser.add_argument('--physics_dual_branch', action='store_true', default=False,
                        help='enable dual hidden/state branch update')
    parser.add_argument('--wind_cap_quantile', type=float, default=0.995,
                        help='training-set quantile used to estimate wind capacity upper bound')
    parser.add_argument('--wind_ramp_quantile', type=float, default=0.995,
                        help='training-set quantile used to estimate wind power ramp-rate bounds')
    parser.add_argument('--wind_state_bins', type=int, default=4,
                        help='number of power-state bins')
    parser.add_argument('--wind_constraint_weight', type=float, default=1.0,
                        help='training-time weight for WindPhysicsV4 penalty')
    parser.add_argument('--wind_constraint_warmup_epochs', type=int, default=2,
                        help='warmup epochs for WindPhysicsV4 penalty')
    parser.add_argument('--wind_cp_mode', type=str, default='none', choices=['none', 'fixed', 'parametric'],
                        help='Cp mode used by DLinearWindPhysicsV5')
    parser.add_argument('--wind_cp_max', type=float, default=0.48,
                        help='peak Cp value used by DLinearWindPhysicsV5')
    parser.add_argument('--wind_cp_optimal_ratio', type=float, default=1.0,
                        help='optimal sub-rated position for DLinearWindPhysicsV5')
    parser.add_argument('--wind_cp_width', type=float, default=0.35,
                        help='width of the DLinearWindPhysicsV5 Cp curve')
    parser.add_argument('--wind_soft_ramp_weight', type=float, default=0.1,
                        help='soft ramp penalty weight for DLinearWindPhysicsV5')
    parser.add_argument('--wind_soft_bound_weight', type=float, default=0.1,
                        help='soft bound penalty weight for DLinearWindPhysicsV5')
    parser.add_argument('--wind_grad_clip', type=float, default=5.0,
                        help='gradient clipping max-norm for DLinearWindPhysicsV5')

    parser.add_argument('--reservoir_use_slack', action='store_true', default=False,
                        help='use non-negative slack in reservoir conservation')
    parser.add_argument('--reservoir_no_slack', action='store_true', default=False,
                        help='disable slack in reservoir conservation')
    parser.add_argument('--reservoir_delta_t', type=float, default=1.0,
                        help='time interval for reservoir conservation equation')
    parser.add_argument('--conservation_weight', type=float, default=1.0,
                        help='weight for conservation residual auxiliary loss')
    parser.add_argument('--reservoir_warmup_epochs', type=int, default=2,
                        help='warmup epochs for reservoir conservation auxiliary loss')

    parser.add_argument('--node_dim', type=int, default=10, help='GCN node embedding dimension')
    parser.add_argument('--gcn_depth', type=int, default=2, help='GCN depth')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='GCN dropout')
    parser.add_argument('--propalpha', type=float, default=0.3, help='GCN propagation alpha')
    parser.add_argument('--conv_channel', type=int, default=32, help='GCN conv channel')
    parser.add_argument('--skip_channel', type=int, default=32, help='GCN skip channel')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: one linear layer for each channel')

    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding')

    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='explicit checkpoint path or checkpoint directory')
    parser.add_argument('--setting', type=str, default='',
                        help='explicit setting name under checkpoints/')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth',
                        help='checkpoint filename when checkpoint_path points to a directory')
    parser.add_argument('--strict_load', type=str2bool, default=True,
                        help='strictly match checkpoint keys when loading')
    parser.add_argument('--skip_checkpoint_load', action='store_true', default=False,
                        help='skip local checkpoint loading and evaluate current model weights')
    parser.add_argument('--eval_split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='dataset split used for evaluation')
    parser.add_argument('--results_dir', type=str, default='./results_test',
                        help='directory used to save metrics and optional predictions')
    parser.add_argument('--save_predictions', action='store_true', default=False,
                        help='save preds/trues as a compressed npz file')
    parser.add_argument('--report_all_channels', action='store_true', default=False,
                        help='keep all supervised channels instead of collapsing to the first one')
    parser.add_argument('--channel_names', type=str, default='',
                        help='comma-separated names for supervised channels')
    parser.add_argument('--metric_eps', type=float, default=1e-6,
                        help='epsilon used to stabilise MAPE/MSPE')
    parser.add_argument('--hf_cache_dir', type=str, default='',
                        help='optional Hugging Face cache directory')
    parser.add_argument('--analysis_solar_pcd', type=str2bool, default=None,
                        help='enable Solar PCD analysis; defaults to auto for Solar or *SolarPhysics models')
    parser.add_argument('--compare_projection_stages', type=str2bool, default=True,
                        help='compare pre- and post-projection outputs when the model exposes trace data')
    parser.add_argument('--save_analysis_tables', type=str2bool, default=True,
                        help='write Solar analysis tables and JSON summaries')
    parser.add_argument('--save_analysis_plots', type=str2bool, default=False,
                        help='write Solar analysis plots when matplotlib is available')
    parser.add_argument('--save_case_studies', type=str2bool, default=False,
                        help='save representative Solar case-study samples as NPZ files')
    parser.add_argument('--save_solar_geometry_payload', type=str2bool, default=False,
                        help='save post-prediction Solar geometry payload (pred/true/cos_zenith) for cross-run plotting')
    parser.add_argument('--save_internal_projection_geometry', type=str2bool, default=False,
                        help='save the single-model internal pre/post projection geometry figure; off by default because it is not a baseline/Soft/PCD comparison')
    parser.add_argument('--negativity_tol', type=float, default=1.0,
                        help='tolerance used when counting negativity violations')
    parser.add_argument('--feasible_abs_tol', type=float, default=1.0,
                        help='absolute tolerance used in Solar feasible-rate checks')
    parser.add_argument('--feasible_rel_tol', type=float, default=0.01,
                        help='relative tolerance used in Solar feasible-rate checks')
    parser.add_argument('--include_upper_bound_in_feasible', type=str2bool, default=False,
                        help='include weak upper-bound checks in Solar feasible-rate evaluation')
    parser.add_argument('--solar_bucket_edges', type=str, default='0.0,0.1,0.2,0.5',
                        help='comma-separated cos(zenith) bucket edges for Solar near-horizon analysis')
    parser.add_argument('--case_study_topk', type=int, default=3,
                        help='number of case studies to save when save_case_studies is enabled')
    parser.add_argument('--case_study_selection', type=str, default='worst_improvement',
                        choices=['worst_improvement', 'best_improvement', 'representative_bucket'],
                        help='strategy used to select Solar case studies')
    parser.add_argument('--cross_data_configs', type=str, default='',
                        help='inline cross-dataset eval configs; separate datasets with ";" and use key=value pairs, e.g. '
                             '"name=B,data=Solar,root_path=...,data_path=B.csv;name=C,data=Solar,root_path=...,data_path=C.csv"')
    parser.add_argument('--cross_data_config_file', type=str, default='',
                        help='JSON file containing a list of dataset override objects for cross-dataset evaluation')


def task_name_for_setting(task_name):
    aliases = {
        'long_term_forecast': 'ltf',
        'short_term_forecast': 'stf',
    }
    return aliases.get(task_name, task_name)


def shorten_setting(raw_setting, args, ii):
    return raw_setting


def build_setting(args, ii):
    setting_parts = [
        task_name_for_setting(args.task_name),
        sanitize_setting_token(args.model_id, default='mid', max_len=18),
        sanitize_setting_token(args.model, default='model', max_len=24),
        sanitize_setting_token(args.data, default='data', max_len=10),
        'dp{}'.format(args.dataset_tag),
        'sl{}'.format(args.seq_len),
        'pl{}'.format(args.pred_len),
    ]
    if int(getattr(args, 'label_len', 0)) != 0:
        setting_parts.append('ll{}'.format(args.label_len))
    if str(getattr(args, 'features', 'M')) != 'M':
        setting_parts.append('ft{}'.format(args.features))
    if int(getattr(args, 'd_model', 128)) != 128:
        setting_parts.append('dm{}'.format(args.d_model))
    if int(getattr(args, 'e_layers', 2)) != 2:
        setting_parts.append('el{}'.format(args.e_layers))
    if int(getattr(args, 'd_layers', 1)) != 1:
        setting_parts.append('dl{}'.format(args.d_layers))
    if int(getattr(args, 'd_ff', 128)) != 128:
        setting_parts.append('df{}'.format(args.d_ff))
    des_token = sanitize_setting_token(args.des, default='exp', max_len=12)
    if des_token and des_token != 'test':
        setting_parts.append(des_token)
    setting_parts.append('i{}'.format(ii))

    physics_parts = []
    if float(getattr(args, 'ft_adapter_dropout', 0.1)) != 0.1:
        physics_parts.append('ftd{}'.format(args.ft_adapter_dropout))
    if int(getattr(args, 'physics_kernel_size', 3)) != 3:
        physics_parts.append('pks{}'.format(args.physics_kernel_size))
    if float(getattr(args, 'night_cos_threshold', 0.0)) != 0.0:
        physics_parts.append('nct{}'.format(args.night_cos_threshold))
    if str(getattr(args, 'physics_context_mode', 'baseline')) != 'baseline':
        physics_parts.append('pcm{}'.format(args.physics_context_mode))
    if (
        str(getattr(args, 'physics_context_mode', 'baseline')) == 'gated'
        and float(getattr(args, 'physics_context_gate_init', 2.0)) != 2.0
    ):
        physics_parts.append('pcgi{}'.format(args.physics_context_gate_init))
    if float(getattr(args, 'solar_dhi_weight', 0.0)) != 0.0:
        physics_parts.append('sdw{}'.format(args.solar_dhi_weight))
    if float(getattr(args, 'solar_dhi_ratio_floor_weight', 0.0)) != 0.0:
        physics_parts.append('sdrw{}'.format(args.solar_dhi_ratio_floor_weight))
    if float(getattr(args, 'solar_dhi_ratio_floor_cap', 0.05)) != 0.05:
        physics_parts.append('sdrc{}'.format(args.solar_dhi_ratio_floor_cap))
    if float(getattr(args, 'solar_dhi_day_cos_threshold', 0.05)) != 0.05:
        physics_parts.append('sddc{}'.format(args.solar_dhi_day_cos_threshold))
    if bool(getattr(args, 'physics_state_residual', False)):
        physics_parts.append('psr1')
    if bool(getattr(args, 'physics_gated_feedback', False)):
        physics_parts.append('pgf1')
    if bool(getattr(args, 'physics_dual_branch', False)):
        physics_parts.append('pdb1')
    if 'WindPhysics' in args.model and float(getattr(args, 'wind_cap_quantile', 0.995)) != 0.995:
        physics_parts.append('wcq{}'.format(args.wind_cap_quantile))
    if ('WindPhysicsV2' in args.model or 'WindPhysicsV3' in args.model) and float(getattr(args, 'wind_ramp_quantile', 0.995)) != 0.995:
        physics_parts.append('wrq{}'.format(args.wind_ramp_quantile))
    if ('WindPhysicsV3' in args.model or 'WindPhysicsV4' in args.model) and int(getattr(args, 'wind_state_bins', 4)) != 4:
        physics_parts.append('wsb{}'.format(args.wind_state_bins))
    if 'WindPhysicsV4' in args.model and float(getattr(args, 'wind_constraint_weight', 1.0)) != 1.0:
        physics_parts.append('wcw{}'.format(args.wind_constraint_weight))
    if 'WindPhysicsV5' in args.model and str(getattr(args, 'wind_cp_mode', 'none')) != 'none':
        physics_parts.append('wcp{}'.format(args.wind_cp_mode))
    if 'WindPhysicsV5' in args.model and float(getattr(args, 'wind_cp_max', 0.48)) != 0.48:
        physics_parts.append('wcpm{}'.format(args.wind_cp_max))
    if 'WindPhysicsV5' in args.model and float(getattr(args, 'wind_cp_optimal_ratio', 1.0)) != 1.0:
        physics_parts.append('wcpo{}'.format(args.wind_cp_optimal_ratio))
    if 'WindPhysicsV5' in args.model and float(getattr(args, 'wind_cp_width', 0.35)) != 0.35:
        physics_parts.append('wcpw{}'.format(args.wind_cp_width))
    if 'WindPhysicsV5' in args.model and float(getattr(args, 'wind_soft_ramp_weight', 0.1)) != 0.1:
        physics_parts.append('wsrw{}'.format(args.wind_soft_ramp_weight))
    if 'WindPhysicsV5' in args.model and float(getattr(args, 'wind_soft_bound_weight', 0.1)) != 0.1:
        physics_parts.append('wsbw{}'.format(args.wind_soft_bound_weight))
    if 'WindPhysicsV5' in args.model and float(getattr(args, 'wind_grad_clip', 5.0)) != 5.0:
        physics_parts.append('wgc{}'.format(args.wind_grad_clip))
    if bool(getattr(args, '_uses_external_physics', False)):
        physics_parts.append('ueph')
    if getattr(args, 'extra_tag', ''):
        physics_parts.append('xt{}'.format(args.extra_tag))
    hidden_parts = [
        'nh{}'.format(args.n_heads),
        'expand{}'.format(args.expand),
        'dc{}'.format(args.d_conv),
        'fc{}'.format(args.factor),
        'eb{}'.format(args.embed),
        'dt{}'.format(args.distil),
        'aux{}'.format(args.use_auxiliary),
        'cw{}'.format(args.closure_weight),
        'lw{}'.format(args.limit_weight),
        'sc{}'.format(args.solar_constant),
        'vg{}'.format(args.var_gamma),
        'vw{}'.format(args.var_weight),
        'ip{}'.format(args.use_internal_physics),
        'phl{}'.format(args.physics_layers),
        'phd{}'.format(args.physics_hidden_dim),
        'fth{}'.format(args.ft_adapter_hidden_dim),
        'ftb{}'.format(args.ft_adapter_bottleneck_dim),
    ]
    hidden_tag = build_hidden_config_hash(hidden_parts)
    raw_setting = '_'.join(str(part) for part in setting_parts + physics_parts + [hidden_tag] if str(part))
    return shorten_setting(raw_setting, args, ii)


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Evaluate pretrained forecasting models')
    add_common_args(parser)
    args = parser.parse_args()

    args.hf_cache_dir = args.hf_cache_dir or None
    args.dataset_tag = build_dataset_tag(args)
    args.eval_name = ''
    args.eval_suffix = ''

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, 'mps'):
            args.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            args.device = torch.device('cpu')
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(device_id) for device_id in device_ids]
        args.gpu = args.device_ids[0]

    if args.model.endswith('SolarPhysics'):
        args.use_internal_physics = True

    if args.task_name != 'long_term_forecast':
        raise ValueError('run_test.py currently supports task_name=long_term_forecast only.')

    print('Args in experiment:')
    print_args(args)
    cross_data_configs = load_cross_data_configs(args)

    if args.checkpoint_path or args.setting:
        settings = [args.setting.strip() or build_setting(args, 0)]
    else:
        settings = [build_setting(args, ii) for ii in range(args.itr)]

    eval_arg_sets = [args]
    if cross_data_configs:
        eval_arg_sets = [apply_cross_data_overrides(args, override) for override in cross_data_configs]

    for eval_args in eval_arg_sets:
        if getattr(eval_args, 'eval_name', ''):
            print(
                'Cross-dataset eval target: {} | data={} | root_path={} | data_path={}'.format(
                    eval_args.eval_name,
                    getattr(eval_args, 'data', ''),
                    getattr(eval_args, 'root_path', ''),
                    getattr(eval_args, 'data_path', ''),
                )
            )

        exp = Exp_Long_Term_Forecast_Test(eval_args)
        for setting in settings:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            if eval_args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif eval_args.gpu_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
