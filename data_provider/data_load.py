import os
import warnings

import numpy as np
import pandas as pd
import pvlib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


warnings.filterwarnings('ignore')


DATASET_LAT_LON = {
    'Folsom.csv': (38.677, -121.15),
    'Congo.csv': (2.61924, 19.85019),
    'Germany.csv': (50.0, 10.0),
    'NewDelhi.csv': (28.0, 77.0),
}


class Dataset(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='test.csv',
        target='ghi',
        scale=True,
        timeenc=0,
        sample_interval=1,
        slide_step=1,
        ymax=29.5,
    ):
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
        self.data_columns = []
        self.target_col_idx = 0

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
        self.__build_window_meta__()

    def _compute_solar_zenith(self, ts_series: pd.Series) -> np.ndarray:
        ts = pd.to_datetime(ts_series, errors='coerce')
        if ts.isna().any():
            raise ValueError('Invalid timestamp found while computing solar zenith.')

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
        return solpos['zenith'].to_numpy(dtype=np.float32).reshape(-1, 1)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path_))

        if 'timestamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        elif 'timeStamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timeStamp'])
            df_raw = df_raw.drop(columns=['timeStamp'])
        elif 'datetime' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['datetime'])
            df_raw = df_raw.drop(columns=['datetime'])
        else:
            raise ValueError("Expected column 'timestamp', 'timeStamp', or 'datetime' in CSV")

        if self.features in ['M', 'MS']:
            cols_data = [column for column in df_raw.columns if column != 'timestamp']
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target not in df_raw.columns:
                raise ValueError("Target column '{}' not found in CSV".format(self.target))
            df_data = df_raw[[self.target]]
        else:
            raise ValueError('Unsupported features mode: {}'.format(self.features))

        self.data_columns = list(df_data.columns)
        if self.target in self.data_columns:
            self.target_col_idx = self.data_columns.index(self.target)

        if bool(getattr(self.args, 'is_three', False)):
            solar_metric_cols = ['ghi', 'dni', 'dhi']
            if all(column in df_data.columns for column in solar_metric_cols):
                self.ymax = df_data[solar_metric_cols].abs().max().to_numpy(dtype=np.float32)

        n = len(df_raw)
        num_train = int(n * 0.7)
        num_val = int(n * 0.1)

        if self.scale:
            train_values = df_data.iloc[:num_train].values
            self.scaler.fit(train_values)
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        def build_stamp(df_slice):
            df_stamp = df_slice[['timestamp']].copy()
            if self.timeenc in [0, 1]:
                df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month)
                df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour)
                return df_stamp.drop(['timestamp'], axis=1).values
            return np.zeros((len(df_stamp), 4), dtype=np.int64)

        if self.set_type == 0:
            border1 = 0
            border2 = num_train
        elif self.set_type == 1:
            border1 = max(0, num_train - self.seq_len * self.sample_interval)
            border2 = num_train + num_val
        else:
            border1 = max(0, num_train + num_val - self.seq_len * self.sample_interval)
            border2 = n

        self.data_x = data_all[border1:border2]
        self.data_y = data_all[border1:border2]
        split_df = df_raw.iloc[border1:border2]
        self.timestamps = pd.to_datetime(split_df['timestamp']).to_numpy()
        self.data_stamp = build_stamp(split_df)
        self.data_zenith = self._compute_solar_zenith(split_df['timestamp'])

    def __build_window_meta__(self):
        length = len(self.data_x)
        interval = self.sample_interval
        max_start = length - 1 - (self.seq_len + self.pred_len - 1) * interval
        self.max_start = int(max_start) if max_start >= 0 else -1
        if self.max_start < 0:
            raise ValueError(
                'Not enough data after sampling: L={}, seq_len={}, pred_len={}, N={}. '
                'Need L >= {}.'.format(
                    length,
                    self.seq_len,
                    self.pred_len,
                    interval,
                    (self.seq_len + self.pred_len - 1) * interval + 1,
                )
            )
        self.num_samples = self.max_start // self.slide_step + 1

    def __getitem__(self, index):
        start = index * self.slide_step
        interval = self.sample_interval
        seq_idx_x = start + np.arange(self.seq_len) * interval
        r_begin = start + (self.seq_len - self.label_len) * interval
        seq_idx_y = r_begin + np.arange(self.label_len + self.pred_len) * interval

        seq_x = self.data_x[seq_idx_x]
        seq_y = self.data_y[seq_idx_y]
        seq_x = np.concatenate([seq_x, self.data_zenith[seq_idx_x]], axis=1)
        seq_y = np.concatenate([seq_y, self.data_zenith[seq_idx_y]], axis=1)

        seq_x_mark = self.data_stamp[seq_idx_x]
        seq_y_mark = self.data_stamp[seq_idx_y]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def future_timestamps(self, index):
        start = index * self.slide_step
        interval = self.sample_interval
        r_begin = start + (self.seq_len - self.label_len) * interval
        seq_idx_y = r_begin + np.arange(self.label_len + self.pred_len) * interval
        return self.timestamps[seq_idx_y[-self.pred_len:]]

    def __len__(self):
        return self.num_samples

    def inverse_transform(self, data):
        if (not self.scale) or (not hasattr(self.scaler, 'scale_')):
            return data
        return self.scaler.inverse_transform(data)
