import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
import pvlib
warnings.filterwarnings('ignore')

# data_path (CSV 文件名) -> (纬度, 经度)，用于 pvlib 太阳位置；未列出的文件仍用 args / 默认值
DATASET_LAT_LON = {
    'folsom.csv': (38.677, -121.15),
    'Folsom_combined_no_zero.csv': (38.677, -121.15),
    'Folsom_combined.csv': (38.677, -121.15),
    '45.36_-90.05_2015_2024_nonzero.csv': (45.36, -90.05),
    '34.05_-118.24_2015_2024_nonzero.csv': (34.05, -118.24),
    '31.1_-90.05_2015_2024_nonzero.csv': (31.1, -90.05),
    '45.36_-90.05_2015_2024.csv': (45.36, -90.05),
    'Congo_no_zero.csv': (2.61924, 19.85019),
    'Germany_no_zero.csv': (50.0, 10.0),
    'NewDelhi_no_zero.csv': (28.0, 77.0),
    'Congo.csv': (2.61924, 19.85019),
    'Germany.csv': (50.0, 10.0),
    'NewDelhi.csv': (28.0, 77.0),
    '25.76_-80.19_2015_2024.csv': (25.76, -80.19),
    '39.74_-104.99_2015_2024.csv': (39.74, -104.99),
    '44.98_-93.27_2015_2024.csv': (44.98, -93.27),
    '25.76_-80.19_2015_2024_no_zero.csv': (25.76, -80.19),
    '39.74_-104.99_2015_2024_no_zero.csv': (39.74, -104.99),
    '44.98_-93.27_2015_2024_no_zero.csv': (44.98, -93.27),
}

WIND_FEATURE_ORDER = [
    'density',
    'pressure',
    'temperature',
    'wind_direction',
    'wind_speed',
    'power',
]

WATER_FEATURE_ORDER_4 = [
    'storage',
    'inflow',
    'outflow',
    'evaporation',
]

WATER_FEATURE_ORDER_3 = [
    'storage',
    'inflow',
    'outflow',
]


class Dataset_Other(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='test.csv',
                 target='target', scale=True, timeenc=0,
                 sample_interval=1,  # 每 N 个点取一个
                 slide_step=1,        # 窗口起点在原始序列上的滑动步长（单位=原始点）
                 ymax=29.5,
                 ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = self.args.seq_len
        self.label_len = self.args.label_len
        self.pred_len = self.args.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.ymax = ymax
        self.dataset_name = str(getattr(self.args, 'data', ''))
        self.is_solar_dataset = self.dataset_name == 'Solar'
        self.is_wind_dataset = self.dataset_name == 'Wind'
        self.is_water_dataset = self.dataset_name == 'Water'
        self.data_columns = []
        self.target_col_idx = 0
        self.power_lower_raw = None
        self.power_cap_raw = None
        self.power_lower_scaled = None
        self.power_cap_scaled = None
        self.power_lower_bound = None
        self.power_upper_bound = None
        self.power_ramp_up_raw = None
        self.power_ramp_down_raw = None
        self.power_ramp_up_scaled = None
        self.power_ramp_down_scaled = None
        self.power_state_bin_edges_raw = None
        self.power_state_ramp_up_raw = None
        self.power_state_ramp_down_raw = None
        self.power_state_bin_edges_scaled = None
        self.power_state_ramp_up_scaled = None
        self.power_state_ramp_down_scaled = None
        self.power_cut_in_speed_raw = None
        self.power_rated_speed_raw = None
        self.power_cut_in_speed_legacy_raw = None
        self.power_rated_speed_legacy_raw = None
        self.power_cut_out_speed_raw = None
        self.power_cut_in_speed_norm = None
        self.power_rated_speed_norm = None
        self.power_cut_out_speed_norm = None
        self.power_rated_power_norm = None

        self.sample_interval = int(sample_interval)
        assert self.sample_interval >= 1
        self.slide_step = int(slide_step)
        assert self.slide_step >= 1

        self.root_path = root_path
        self.data_path_ = data_path
        if self.data_path_ in DATASET_LAT_LON:
            lat, lon = DATASET_LAT_LON[self.data_path_]
            self.solar_latitude = float(lat)
            self.solar_longitude = float(lon)
        else:
            self.solar_latitude = float(getattr(self.args, 'solar_latitude', 38.677))
            self.solar_longitude = float(getattr(self.args, 'solar_longitude', -121.15))
        self.solar_timezone = str(getattr(self.args, 'solar_timezone', 'UTC'))

        self.__read_data__()
        self.__build_window_meta__()  # 预计算长度等

    def _compute_solar_zenith(self, ts_series: pd.Series) -> np.ndarray:
        """
        Compute solar zenith angle (degrees) using pvlib solarposition.
        Naive times are localized to self.solar_timezone; tz-aware times are
        converted to that zone (default UTC), preserving the correct instant.
        """
        ts = pd.to_datetime(ts_series, errors='coerce')
        if ts.isna().any():
            raise ValueError("Invalid timestamp found while computing solar zenith.")

        dt_index = pd.DatetimeIndex(ts)
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize(self.solar_timezone)
        else:
            dt_index = dt_index.tz_convert(self.solar_timezone)

        solpos = pvlib.solarposition.get_solarposition(
            dt_index,
            latitude=self.solar_latitude,
            longitude=self.solar_longitude,
        )
        zenith_deg = solpos['zenith'].to_numpy(dtype=np.float32)
        return zenith_deg.reshape(-1, 1)

    def _first_consecutive_match(self, mask, run_length):
        if run_length <= 1:
            hits = np.where(mask)[0]
            return int(hits[0]) if hits.size else None
        streak = 0
        for idx, flag in enumerate(mask):
            streak = streak + 1 if flag else 0
            if streak >= run_length:
                return idx - run_length + 1
        return None

    def _estimate_wind_curve_points(self, wind_speed_train, power_train, rated_power):
        wind_speed_train = np.asarray(wind_speed_train, dtype=np.float32)
        power_train = np.asarray(power_train, dtype=np.float32)
        if wind_speed_train.size == 0:
            return 3.0, 12.0, float('inf')

        if rated_power <= 0.0:
            cut_in_speed = float(np.quantile(wind_speed_train, 0.05))
            rated_speed = max(cut_in_speed + 1e-3, float(np.quantile(wind_speed_train, 0.90)))
            return cut_in_speed, rated_speed, float('inf')

        bin_width = 0.5
        min_bin_count = 50
        required_consecutive_bins = 2
        active_power_threshold_ratio = 0.02
        cut_in_power_ratio = 0.05
        cut_in_active_ratio = 0.5
        rated_power_ratio = 0.95
        rated_active_ratio = 0.8
        cut_out_power_ratio = 0.05
        cut_out_active_ratio = 0.2

        max_speed = float(np.nanmax(wind_speed_train))
        bin_edges = np.arange(0.0, max_speed + bin_width, bin_width, dtype=np.float32)
        if bin_edges.size < 2:
            bin_edges = np.array([0.0, max_speed + bin_width], dtype=np.float32)
        bin_idx = np.digitize(wind_speed_train, bin_edges) - 1

        rows = []
        for idx in range(len(bin_edges) - 1):
            mask = bin_idx == idx
            if int(mask.sum()) < min_bin_count:
                continue
            power_bin = power_train[mask]
            active_ratio = float(np.mean(power_bin > active_power_threshold_ratio * rated_power))
            rows.append({
                'v_center': float(0.5 * (bin_edges[idx] + bin_edges[idx + 1])),
                'power_q90': float(np.quantile(power_bin, 0.90)),
                'power_q95': float(np.quantile(power_bin, 0.95)),
                'active_ratio': active_ratio,
            })

        if not rows:
            cut_in_speed = float(np.quantile(wind_speed_train, 0.05))
            rated_speed = max(cut_in_speed + 1e-3, float(np.quantile(wind_speed_train, 0.90)))
            return cut_in_speed, rated_speed, float('inf')

        bins_df = pd.DataFrame(rows)

        cut_in_mask = (
            (bins_df['power_q90'].to_numpy() >= cut_in_power_ratio * rated_power)
            & (bins_df['active_ratio'].to_numpy() >= cut_in_active_ratio)
        )
        cut_in_idx = self._first_consecutive_match(cut_in_mask, required_consecutive_bins)
        if cut_in_idx is None:
            cut_in_idx = int(np.argmax(bins_df['power_q90'].to_numpy() > 0.0))

        rated_mask = (
            (bins_df['power_q90'].to_numpy() >= rated_power_ratio * rated_power)
            & (bins_df['active_ratio'].to_numpy() >= rated_active_ratio)
        )
        rated_mask[:cut_in_idx] = False
        rated_idx = self._first_consecutive_match(rated_mask, required_consecutive_bins)
        if rated_idx is None:
            rated_idx = int(np.argmax(bins_df['power_q95'].to_numpy()))

        cut_out_mask = (
            (bins_df['power_q90'].to_numpy() <= cut_out_power_ratio * rated_power)
            & (bins_df['active_ratio'].to_numpy() <= cut_out_active_ratio)
        )
        cut_out_mask[: max(rated_idx + 1, 0)] = False
        cut_out_idx = self._first_consecutive_match(cut_out_mask, required_consecutive_bins)

        cut_in_speed = float(bins_df.iloc[cut_in_idx]['v_center'])
        rated_speed = max(cut_in_speed + 1e-3, float(bins_df.iloc[rated_idx]['v_center']))
        if cut_out_idx is None:
            cut_out_speed = float('inf')
        else:
            cut_out_speed = float(bins_df.iloc[cut_out_idx]['v_center'])
        return cut_in_speed, rated_speed, cut_out_speed

    def _set_wind_capacity_stats(self, df_raw_, num_train):
        if 'power' not in df_raw_.columns:
            raise ValueError("Wind dataset requires a 'power' column.")

        train_df = df_raw_.iloc[:num_train]
        power_train = train_df['power'].to_numpy(dtype=np.float32)
        cap_quantile = float(getattr(self.args, 'wind_cap_quantile', 0.995))
        ramp_quantile = float(getattr(self.args, 'wind_ramp_quantile', 0.995))
        state_bins = max(2, int(getattr(self.args, 'wind_state_bins', 4)))
        power_idx = self.data_columns.index('power')
        ramp_step = max(1, min(int(self.sample_interval), len(power_train) - 1))
        # Match ramp statistics to the actual forecasting step used by the dataset.
        # For sampled wind data, adjacent prediction points are `sample_interval`
        # raw steps apart, so 1-step ramps are too restrictive.
        prev_power = power_train[:-ramp_step]
        power_diff = power_train[ramp_step:] - prev_power
        ramp_up = np.clip(power_diff, a_min=0.0, a_max=None)
        ramp_down = np.clip(-power_diff, a_min=0.0, a_max=None)

        self.power_lower_raw = 0.0
        self.power_cap_raw = float(np.quantile(power_train, cap_quantile))
        self.power_ramp_up_raw = float(np.quantile(ramp_up, ramp_quantile))
        self.power_ramp_down_raw = float(np.quantile(ramp_down, ramp_quantile))
        self.power_state_bin_edges_raw = np.linspace(
            self.power_lower_raw,
            self.power_cap_raw,
            state_bins + 1,
            dtype=np.float32,
        )
        self.power_state_ramp_up_raw = np.full(state_bins, self.power_ramp_up_raw, dtype=np.float32)
        self.power_state_ramp_down_raw = np.full(state_bins, self.power_ramp_down_raw, dtype=np.float32)

        density_train = train_df['density'].to_numpy(dtype=np.float32)
        wind_speed_train = train_df['wind_speed'].to_numpy(dtype=np.float32)
        density_ratio = np.clip(density_train / 1.225, 0.25, 2.5)
        effective_speed = wind_speed_train * np.cbrt(density_ratio)
        active_mask = power_train >= 0.02 * self.power_cap_raw
        rated_mask = power_train >= 0.95 * self.power_cap_raw
        if np.any(active_mask):
            self.power_cut_in_speed_legacy_raw = float(np.quantile(effective_speed[active_mask], 0.05))
        else:
            self.power_cut_in_speed_legacy_raw = 3.0
        if np.any(rated_mask):
            self.power_rated_speed_legacy_raw = float(np.quantile(effective_speed[rated_mask], 0.01))
        else:
            self.power_rated_speed_legacy_raw = max(
                self.power_cut_in_speed_legacy_raw + 1.0,
                float(np.quantile(effective_speed, 0.90)),
            )
        self.power_rated_speed_legacy_raw = max(
            self.power_rated_speed_legacy_raw,
            self.power_cut_in_speed_legacy_raw + 1e-3,
        )

        for bin_idx in range(state_bins):
            bin_lower = self.power_state_bin_edges_raw[bin_idx]
            if bin_idx == state_bins - 1:
                bin_mask = prev_power >= bin_lower
            else:
                bin_upper = self.power_state_bin_edges_raw[bin_idx + 1]
                bin_mask = (prev_power >= bin_lower) & (prev_power < bin_upper)

            if not np.any(bin_mask):
                continue

            self.power_state_ramp_up_raw[bin_idx] = float(np.quantile(ramp_up[bin_mask], ramp_quantile))
            self.power_state_ramp_down_raw[bin_idx] = float(np.quantile(ramp_down[bin_mask], ramp_quantile))

        self.power_cut_in_speed_raw, self.power_rated_speed_raw, self.power_cut_out_speed_raw = (
            self._estimate_wind_curve_points(wind_speed_train, power_train, self.power_cap_raw)
        )

        if self.scale and hasattr(self.scaler, 'mean_'):
            power_mean = float(self.scaler.mean_[power_idx])
            power_scale = float(self.scaler.scale_[power_idx])
            wind_speed_idx = self.data_columns.index('wind_speed')
            wind_speed_mean = float(self.scaler.mean_[wind_speed_idx])
            wind_speed_scale = float(self.scaler.scale_[wind_speed_idx])
            self.power_lower_scaled = (self.power_lower_raw - power_mean) / power_scale
            self.power_cap_scaled = (self.power_cap_raw - power_mean) / power_scale
            self.power_ramp_up_scaled = self.power_ramp_up_raw / power_scale
            self.power_ramp_down_scaled = self.power_ramp_down_raw / power_scale
            self.power_state_bin_edges_scaled = (self.power_state_bin_edges_raw - power_mean) / power_scale
            self.power_state_ramp_up_scaled = self.power_state_ramp_up_raw / power_scale
            self.power_state_ramp_down_scaled = self.power_state_ramp_down_raw / power_scale
            self.power_lower_bound = self.power_lower_scaled
            self.power_upper_bound = self.power_cap_scaled
            self.power_cut_in_speed_norm = (self.power_cut_in_speed_raw - wind_speed_mean) / wind_speed_scale
            self.power_rated_speed_norm = (self.power_rated_speed_raw - wind_speed_mean) / wind_speed_scale
            if np.isfinite(self.power_cut_out_speed_raw):
                self.power_cut_out_speed_norm = (self.power_cut_out_speed_raw - wind_speed_mean) / wind_speed_scale
            else:
                self.power_cut_out_speed_norm = (
                    (float(np.max(wind_speed_train)) - wind_speed_mean) / wind_speed_scale + 5.0
                )
            self.power_rated_power_norm = (self.power_cap_raw - power_mean) / power_scale
        else:
            self.power_lower_scaled = self.power_lower_raw
            self.power_cap_scaled = self.power_cap_raw
            self.power_ramp_up_scaled = self.power_ramp_up_raw
            self.power_ramp_down_scaled = self.power_ramp_down_raw
            self.power_state_bin_edges_scaled = self.power_state_bin_edges_raw
            self.power_state_ramp_up_scaled = self.power_state_ramp_up_raw
            self.power_state_ramp_down_scaled = self.power_state_ramp_down_raw
            self.power_lower_bound = self.power_lower_raw
            self.power_upper_bound = self.power_cap_raw
            self.power_cut_in_speed_norm = self.power_cut_in_speed_raw
            self.power_rated_speed_norm = self.power_rated_speed_raw
            if np.isfinite(self.power_cut_out_speed_raw):
                self.power_cut_out_speed_norm = self.power_cut_out_speed_raw
            else:
                self.power_cut_out_speed_norm = float(np.max(wind_speed_train)) + 5.0
            self.power_rated_power_norm = self.power_cap_raw

        self.ymax = self.power_cap_raw

    def __read_data__(self):
        """
        读取单个 data_path_，按 7:1:2 划分 train/val/test。
        只在训练段上 fit scaler，val/test 复用同一 scaler。
        """
        self.scaler = StandardScaler()
        df_raw_ = pd.read_csv(os.path.join(self.root_path, self.data_path_))

        # Ensure timestamp is datetime (部分数据使用 timeStamp / datetime)
        if 'timestamp' in df_raw_.columns:
            df_raw_['timestamp'] = pd.to_datetime(df_raw_['timestamp'])
        elif 'timeStamp' in df_raw_.columns:
            df_raw_['timestamp'] = pd.to_datetime(df_raw_['timeStamp'])
            df_raw_ = df_raw_.drop(columns=['timeStamp'])
        elif 'datetime' in df_raw_.columns:
            df_raw_['timestamp'] = pd.to_datetime(df_raw_['datetime'])
            df_raw_ = df_raw_.drop(columns=['datetime'])
        else:
            raise ValueError("Expected column 'timestamp', 'timeStamp', or 'datetime' in CSV")

        # Select feature columns
        if self.features in ['M', 'MS']:
            if self.is_wind_dataset:
                missing_cols = [c for c in WIND_FEATURE_ORDER if c not in df_raw_.columns]
                if missing_cols:
                    raise ValueError(f"Wind dataset is missing required columns: {missing_cols}")
                cols_data_ = list(WIND_FEATURE_ORDER)
            elif self.is_water_dataset and 'storage' in df_raw_.columns:
                # Use explicit water feature ordering for ResOpsUS-style data
                if 'evaporation' in df_raw_.columns:
                    water_order = WATER_FEATURE_ORDER_4
                else:
                    water_order = WATER_FEATURE_ORDER_3
                missing_cols = [c for c in water_order if c not in df_raw_.columns]
                if missing_cols:
                    raise ValueError(f"Water dataset is missing required columns: {missing_cols}")
                cols_data_ = list(water_order)
            else:
                cols_data_ = [c for c in df_raw_.columns if c not in ['timestamp']]
            df_data_ = df_raw_[cols_data_]
        elif self.features == 'S':
            if self.target not in df_raw_.columns:
                raise ValueError(f"Target column '{self.target}' not found in CSV")
            df_data_ = df_raw_[[self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")
        self.data_columns = list(df_data_.columns)
        if self.target in self.data_columns:
            self.target_col_idx = self.data_columns.index(self.target)
        elif self.is_wind_dataset and 'power' in self.data_columns:
            self.target_col_idx = self.data_columns.index('power')
        else:
            self.target_col_idx = 0

        # When reporting three solar channels, prefer per-channel maxima
        # from the actual dataset instead of reusing a single GHI ymax.
        if self.is_solar_dataset and bool(getattr(self.args, 'is_three', False)):
            solar_metric_cols = ['ghi', 'dni', 'dhi']
            if all(col in df_data_.columns for col in solar_metric_cols):
                self.ymax = df_data_[solar_metric_cols].abs().max().to_numpy(dtype=np.float32)

        # For water (ResOpsUS) data, auto-compute per-channel ymax if not provided
        if self.is_water_dataset and self.ymax is None:
            self.ymax = df_data_.abs().max().to_numpy(dtype=np.float32)

        # Train/Val/Test split on a single file, 7:1:2
        n_ = len(df_raw_)
        num_train = int(n_ * 0.7)
        num_val = int(n_ * 0.1)
        num_test = n_ - num_train - num_val

        # Fit scaler on training part only
        if self.scale:
            train_values = df_data_.iloc[:num_train].values
            self.scaler.fit(train_values)
            data_all = self.scaler.transform(df_data_.values)
        else:
            data_all = df_data_.values

        if self.is_wind_dataset:
            self._set_wind_capacity_stats(df_raw_=df_raw_, num_train=num_train)

        # time features builder
        def build_stamp(df_slice):
            df_stamp = df_slice[['timestamp']].copy()
            if self.timeenc in [0, 1]:
                df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month)
                df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour)
                data_stamp_local = df_stamp.drop(['timestamp'], axis=1).values
            else:
                data_stamp_local = np.zeros((len(df_stamp), 4), dtype=np.int64)
            return data_stamp_local

        # Select split according to flag
        if self.set_type == 0:  # train
            border1 = 0
            border2 = num_train
        elif self.set_type == 1:  # val
            # 验证集允许看到前置 seq_len * sample_interval 作为上下文
            border1 = max(0, num_train - self.seq_len * self.sample_interval)
            border2 = num_train + num_val
        else:  # test
            border1 = max(0, num_train + num_val - self.seq_len * self.sample_interval)
            border2 = n_

        self.data_x = data_all[border1:border2]
        self.data_y = data_all[border1:border2]
        self.data_stamp = build_stamp(df_raw_.iloc[border1:border2])

        if self.is_solar_dataset:
            self.data_zenith = self._compute_solar_zenith(df_raw_.iloc[border1:border2]['timestamp'])
        else:
            self.data_zenith = None

    def __build_window_meta__(self):
        """
        用采样间隔 N 和窗口定义，预计算起点的最大值与长度。
        约束条件（最后一个 y 的下标不越界）：
            start + (seq_len + pred_len - 1) * N <= L - 1
        =>  start <= L - 1 - (seq_len + pred_len - 1) * N
        """
        L = len(self.data_x)
        N = self.sample_interval
        max_start = L - 1 - (self.seq_len + self.pred_len - 1) * N
        self.max_start = int(max_start) if max_start >= 0 else -1

        if self.max_start < 0:
            # 给出更清晰的错误信息以便你调整 seq_len/pred_len/N 或数据长度
            raise ValueError(
                f"Not enough data after sampling: L={L}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}, N={N}.\n"
                f"Need L >= (seq_len+pred_len-1)*N + 1 = "
                f"{(self.seq_len + self.pred_len - 1) * N + 1}"
            )

        # 计算样本数量：从 0 开始，步长 slide_step，直到 max_start（含）
        self.num_samples = self.max_start // self.slide_step + 1

    def __getitem__(self, index):
        # 将“样本编号”映射到“原始序列起点”
        start = index * self.slide_step
        N = self.sample_interval

        # x 序列索引（长度 = seq_len）
        seq_idx_x = start + np.arange(self.seq_len) * N

        # y 序列索引（长度 = label_len + pred_len）
        # 以 Informer/ETT 常见切法：decoder 的输入从 r_begin 开始
        r_begin = start + (self.seq_len - self.label_len) * N
        seq_idx_y = r_begin + np.arange(self.label_len + self.pred_len) * N

        # 取值
        seq_x = self.data_x[seq_idx_x]
        seq_y = self.data_y[seq_idx_y]
        if self.data_zenith is not None:
            seq_x_zenith = self.data_zenith[seq_idx_x]
            seq_y_zenith = self.data_zenith[seq_idx_y]
            seq_x = np.concatenate([seq_x, seq_x_zenith], axis=1)
            seq_y = np.concatenate([seq_y, seq_y_zenith], axis=1)

        # 时间戳特征对齐索引
        seq_x_mark = self.data_stamp[seq_idx_x]
        seq_y_mark = self.data_stamp[seq_idx_y]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.num_samples

    def inverse_transform(self, data):
        # 当未启用缩放或 scaler 尚未 fit 时，直接返回原数据，避免 NotFittedError
        if (not self.scale) or (not hasattr(self.scaler, "scale_")):
            return data
        return self.scaler.inverse_transform(data)


# if __name__ == "__main__":
#     args = argparse.Namespace()
#     args.seq_len = 8
#     args.label_len = 0
#     args.pred_len = 8
#     args.root_path = "/root/tjl/data/newdata/datadbl/aq/bj1/"
#     args.data_path = "bj1.csv"
#     args.target = "target"
#     args.scale = False
#     args.timeenc = 0

#     # 例子：
#     # sample_interval=8 -> 下采样为原来 8 倍间隔
#     # slide_step=1      -> 起点逐点滑动，增加样本量
#     dataset_train = Dataset_Other(
#         args,
#         root_path=args.root_path,
#         flag='train',
#         features='S',
#         data_path_=args.data_path,
#         target=args.target,
#         scale=args.scale,
#         timeenc=args.timeenc,
#         sample_interval=1,
#         slide_step=8,
#     )

#     dataset_val = Dataset_Other(
#         args,
#         root_path=args.root_path,
#         flag='val',
#         features='S',
#         data_path_=args.data_path,
#         target=args.target,
#         scale=args.scale,
#         timeenc=args.timeenc,
#         sample_interval=1,
#         slide_step=8,
#     )

#     dataset_test = Dataset_Other(
#         args,
#         root_path=args.root_path,
#         flag='test',
#         features='S',
#         data_path_=args.data_path,
#         target=args.target,
#         scale=args.scale,
#         timeenc=args.timeenc,
#         sample_interval=1,
#         slide_step=8,
#     )

#     print(len(dataset_train))
#     print(len(dataset_val))
#     print(len(dataset_test))

if __name__ == "__main__":
    args = argparse.Namespace()
    args.seq_len = 256
    args.label_len = 0
    args.pred_len = 24
    args.root_path = "/opt/data/private/code/Time-Series-Library-fm/data/solar/"
    args.data_path = "Germany.csv"
    args.target = "ghi"
    args.timeenc = 0
    args.slide_step = 8
    args.sample_interval = 2

    # 例子：
    # sample_interval=8 -> 下采样为原来 8 倍间隔
    # slide_step=1      -> 起点逐点滑动，增加样本量
    dataset_train_scaled = Dataset_Other(
        args,
        root_path=args.root_path,
        flag='train',
        features='S',
        data_path=args.data_path,
        target=args.target,
        scale=True,
        timeenc=args.timeenc,
        sample_interval=args.sample_interval,
        slide_step=args.slide_step,
    )


    dataset_test_scaled = Dataset_Other(
        args,
        root_path=args.root_path,
        flag='test',
        features='S',
        data_path=args.data_path,
        target=args.target,
        scale=True,  # 需要已 fit 的 scaler，后续 inverse_transform 才可用
        timeenc=args.timeenc,
        sample_interval=args.sample_interval,
        slide_step=args.slide_step,
    )

    dataset_train_unscaled = Dataset_Other(
        args,
        root_path=args.root_path,
        flag='train',
        features='S',
        data_path=args.data_path,
        target=args.target,
        scale=False,
        timeenc=args.timeenc,
        sample_interval=args.sample_interval,
        slide_step=args.slide_step,
    )

    dataset_test_unscaled = Dataset_Other(
        args,
        root_path=args.root_path,
        flag='test',
        features='S',
        data_path=args.data_path,
        target=args.target,
        scale=False,
        timeenc=args.timeenc,
        sample_interval=args.sample_interval,
        slide_step=args.slide_step,
    )
    print(len(dataset_train_scaled))
    print(len(dataset_test_scaled))

    # dataset_train_scaled_inverse = dataset_train_scaled.inverse_transform(dataset_train_scaled.data_x)
    # dataset_test_scaled_inverse = dataset_test_scaled.inverse_transform(dataset_test_scaled.data_x)

    # # 计算dataset_train_scaled和dataset_train_unscaled的均值和标准差
    # mean_scaled = dataset_test_scaled.data_x.mean()
    # std_scaled = dataset_test_scaled.data_x.std()
    # mean_unscaled = dataset_test_unscaled.data_x.mean()
    # std_unscaled = dataset_test_unscaled.data_x.std()
    # mean_unscaled_inverse = dataset_test_scaled_inverse.mean()
    # std_unscaled_inverse = dataset_test_scaled_inverse.std()
    # print(mean_scaled, std_scaled)
    # print(mean_unscaled, std_unscaled)
    # print(mean_unscaled_inverse, std_unscaled_inverse)

    # # 计算dataset_train_scaled，dataset_train_unscaled，dataset_test_scaled，dataset_test_unscaled mse
    # mse_scaled = np.mean((dataset_test_scaled.data_x - dataset_test_unscaled.data_x) ** 2)
    # mse_unscaled_inverse = np.mean((dataset_test_scaled_inverse - dataset_test_unscaled.data_x) ** 2)
    # print(mse_scaled, mse_unscaled_inverse)
