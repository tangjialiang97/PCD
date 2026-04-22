import argparse
import hashlib
import os
import re
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np


def str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y")


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


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='icudat sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--seq_len_hours', type=int, default=18, help='icudat sequence length in hours')
    parser.add_argument('--pred_len_hours', type=int, default=6, help='prediction sequence length in hours')
    parser.add_argument('--label_len_hours', type=int, default=0, help='start token length in hours')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--data_path_trainval', type=str, default='trainval_timestamp_target.csv', help='data file')
    parser.add_argument('--data_path_test', type=str, default='test_timestamp_target.csv', help='data file')

    # icudatation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder icudat size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder icudat size')
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
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--slide_step', type=int, default=8, help='the step of sliding window')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train icudat data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_auxiliary', action='store_true', help='use auxiliary loss', default=False)
    parser.add_argument('--_uses_external_physics', action='store_true', help='use external physics loss', default=False)
    parser.add_argument('--has_relhum', action='store_true', help='dataset has relhum column (Folsom); omit for datasets without relhum', default=False)
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

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    parser.add_argument('--debug', type=str2bool, default=False, help='debug mode (True/False)')


    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    parser.add_argument('--delta', type=float, default=0.2, help="Delta for PostProcessingNet")
    parser.add_argument('--sample_interval', type=float, default=15, help="Sample interval")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--timesfm_per_core_batch_size', type=int, default=128,
                        help='per-device batch size used by TimesFM internal batched decode')
    parser.add_argument('--timesfm_force_flip_invariance', type=str2bool, default=False,
                        help='enable TimesFM flip-invariance decoding; slower but potentially more robust')
    parser.add_argument('--timesfm_use_continuous_quantile_head', type=str2bool, default=False,
                        help='enable TimesFM continuous quantile head; unnecessary for point-only evaluation')
    parser.add_argument('--timesfm_fix_quantile_crossing', type=str2bool, default=False,
                        help='enable TimesFM quantile crossing correction; unnecessary for point-only evaluation')
    parser.add_argument('--timesfm_infer_is_positive', type=str2bool, default=False,
                        help='enforce positive TimesFM forecasts when inputs are positive')
    parser.add_argument('--time_moe_use_cache', type=str2bool, default=True,
                        help='enable KV cache during TimeMoE autoregressive decoding for faster inference')
    parser.add_argument('--tirex_batch_size', type=int, default=0,
                        help='internal TiRex forecast batch size; 0 uses an automatic value based on outer batch size')
    parser.add_argument('--tirex_compile', type=str2bool, default=False,
                        help='enable torch.compile for TiRex when supported; may add startup overhead')
    parser.add_argument('--ft_adapter_hidden_dim', type=int, default=128,
                        help='hidden width of the residual bottleneck adapter used by FT wrappers')
    parser.add_argument('--ft_adapter_bottleneck_dim', type=int, default=32,
                        help='bottleneck width of the residual adapter used by FT wrappers')
    parser.add_argument('--ft_adapter_dropout', type=float, default=0.1,
                        help='dropout used inside the residual adapter used by FT wrappers')
    parser.add_argument('--is_three', action='store_true', default=False,
                        help='report GHI/DNI/DHI metrics and 3-channel average during test')
    parser.add_argument('--use_internal_physics', action='store_true', default=False,
                        help='enable internal solar physics constraints')
    parser.add_argument('--physics_layers', type=int, default=2, help='number of solar physics blocks')
    parser.add_argument('--physics_hidden_dim', type=int, default=128, help='hidden dimension of solar physics blocks')
    parser.add_argument('--physics_kernel_size', type=int, default=3, help='temporal kernel size in solar physics blocks')
    parser.add_argument('--night_cos_threshold', type=float, default=0.0, help='daytime threshold for cosine zenith')
    parser.add_argument('--physics_eps', type=float, default=1e-4, help='stability epsilon for internal physics projection')
    parser.add_argument('--physics_context_mode', type=str, default='baseline',
                        choices=['baseline', 'base_only', 'gated'],
                        help='how DLinearSolarPhysics combines history_context and base_out before the physics stack')
    parser.add_argument('--physics_context_gate_init', type=float, default=2.0,
                        help='initial logit for gated history-context fusion in DLinearSolarPhysics')
    parser.add_argument('--physics_state_residual', action='store_true', default=False,
                        help='enable residual state update branch inside SolarPhysicsBlock')
    parser.add_argument('--physics_gated_feedback', action='store_true', default=False,
                        help='enable gated feedback for SolarPhysicsBlock temporal updates')
    parser.add_argument('--physics_dual_branch', action='store_true', default=False,
                        help='enable dual hidden/state branch update inside SolarPhysicsBlock')
    parser.add_argument('--wind_cap_quantile', type=float, default=0.995,
                        help='training-set quantile used to estimate wind power capacity upper bound')
    parser.add_argument('--wind_ramp_quantile', type=float, default=0.995,
                        help='training-set quantile used to estimate wind power ramp-rate bounds')
    parser.add_argument('--wind_state_bins', type=int, default=4,
                        help='number of power-state bins used by state-dependent wind ramp constraints')
    parser.add_argument('--wind_constraint_weight', type=float, default=1.0,
                        help='training-time weight for WindPhysicsV4 soft state-ramp penalty')
    parser.add_argument('--wind_constraint_warmup_epochs', type=int, default=2,
                        help='warmup epochs for WindPhysicsV4 soft state-ramp penalty')
    parser.add_argument('--wind_cp_mode', type=str, default='none', choices=['none', 'fixed', 'parametric'],
                        help='Cp mode used by DLinearWindPhysicsV5')
    parser.add_argument('--wind_cp_max', type=float, default=0.48,
                        help='peak Cp value used by DLinearWindPhysicsV5 experimental Cp curves')
    parser.add_argument('--wind_cp_optimal_ratio', type=float, default=1.0,
                        help='optimal sub-rated position for the DLinearWindPhysicsV5 experimental Cp curve')
    parser.add_argument('--wind_cp_width', type=float, default=0.35,
                        help='width of the DLinearWindPhysicsV5 experimental Cp curve')
    parser.add_argument('--wind_soft_ramp_weight', type=float, default=0.1,
                        help='soft ramp penalty weight for DLinearWindPhysicsV5')
    parser.add_argument('--wind_soft_bound_weight', type=float, default=0.1,
                        help='soft bound penalty weight for DLinearWindPhysicsV5')
    parser.add_argument('--wind_grad_clip', type=float, default=5.0,
                        help='gradient clipping max-norm for DLinearWindPhysicsV5 training')

    # Reservoir physics
    parser.add_argument('--reservoir_use_slack', action='store_true', default=False,
                        help='use non-negative slack in reservoir conservation (absorbs evap/seepage)')
    parser.add_argument('--reservoir_no_slack', action='store_true', default=False,
                        help='disable slack in reservoir conservation')
    parser.add_argument('--reservoir_delta_t', type=float, default=1.0,
                        help='time interval for reservoir conservation equation (1.0 for daily)')
    parser.add_argument('--conservation_weight', type=float, default=1.0,
                        help='weight for conservation residual auxiliary loss')
    parser.add_argument('--reservoir_warmup_epochs', type=int, default=2,
                        help='warmup epochs for reservoir conservation auxiliary loss')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.model.endswith('SolarPhysics'):
        args.use_internal_physics = True

    args.dataset_tag = build_dataset_tag(args)

    def task_name_for_setting(task_name):
        aliases = {
            'long_term_forecast': 'ltf',
            'short_term_forecast': 'stf',
        }
        return aliases.get(task_name, task_name)

    def shorten_setting(raw_setting, ii):
        return raw_setting

    def build_setting(ii):
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
        return shorten_setting(raw_setting, ii)

    print('Args in experiment:')
    print_args(args)
    if args.debug:
        print('Debug mode is enabled')
        args.itr = 1
        args.train_epochs = 1

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'zero_shot_forecast':
        Exp = Exp_Zero_Shot_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training and args.task_name != 'zero_shot_forecast':
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = build_setting(ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = build_setting(ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
