import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import argparse

warnings.filterwarnings('ignore')

class Dataset_Stanford(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path_trainval='test.csv', data_path_test='test.csv',
                 target='target', scale=True, timeenc=0,
                 sample_interval=8,  # 每 N 个点取一个
                 slide_step=1        # 窗口起点在原始序列上的滑动步长（单位=原始点）
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

        self.sample_interval = int(sample_interval)
        assert self.sample_interval >= 1
        self.slide_step = int(slide_step)
        assert self.slide_step >= 1

        self.root_path = root_path
        self.data_path_trainval = data_path_trainval
        self.data_path_test = data_path_test

        self.__read_data__()
        self.__build_window_meta__()  # 预计算长度等

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw_trainval = pd.read_csv(os.path.join(self.root_path, self.data_path_trainval))
        df_raw_test = pd.read_csv(os.path.join(self.root_path, self.data_path_test))

        # Ensure timestamp is datetime
        if 'timestamp' not in df_raw_trainval.columns:
            raise ValueError("Expected column 'timestamp' in trainval CSV")
        if 'timestamp' not in df_raw_test.columns:
            raise ValueError("Expected column 'timestamp' in test CSV")

        df_raw_trainval['timestamp'] = pd.to_datetime(df_raw_trainval['timestamp'])
        df_raw_test['timestamp'] = pd.to_datetime(df_raw_test['timestamp'])

        # Select feature columns
        if self.features in ['M', 'MS']:
            cols_data_tv = [c for c in df_raw_trainval.columns if c not in ['timestamp']]
            cols_data_te = [c for c in df_raw_test.columns if c not in ['timestamp']]
            df_data_trainval = df_raw_trainval[cols_data_tv]
            df_data_test = df_raw_test[cols_data_te]
        elif self.features == 'S':
            if self.target not in df_raw_trainval.columns or self.target not in df_raw_test.columns:
                raise ValueError(f"Target column '{self.target}' not found in CSVs")
            df_data_trainval = df_raw_trainval[[self.target]]
            df_data_test = df_raw_test[[self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")

        # Train/Val split on trainval file
        n_tv = len(df_raw_trainval)
        num_train = int(n_tv * 0.8)
        num_val = n_tv - num_train

        # Fit scaler on training part only
        if self.scale:
            train_values = df_data_trainval.iloc[:num_train].values
            self.scaler.fit(train_values)
            data_trainval = self.scaler.transform(df_data_trainval.values)
            data_test = self.scaler.transform(df_data_test.values)
        else:
            data_trainval = df_data_trainval.values
            data_test = df_data_test.values

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
            data = data_trainval
            df_time = df_raw_trainval
        elif self.set_type == 1:  # val
            # 验证集通常允许看见前置 seq_len 以提供足够上下文
            border1 = max(0, num_train - self.seq_len * self.sample_interval)
            border2 = num_train + num_val
            data = data_trainval
            df_time = df_raw_trainval
        else:  # test
            border1 = 0
            border2 = len(df_raw_test)
            data = data_test
            df_time = df_raw_test

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = build_stamp(df_time.iloc[border1:border2])

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

        # 时间戳特征对齐索引
        seq_x_mark = self.data_stamp[seq_idx_x]
        seq_y_mark = self.data_stamp[seq_idx_y]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.num_samples

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    args = argparse.Namespace()
    args.seq_len = 8
    args.label_len = 0
    args.pred_len = 8
    args.root_path = "/opt/data/private/data/timedata/datasets/Stanford"
    args.data_path_trainval = "trainval_timestamp_target.csv"
    args.data_path_test = "test_timestamp_target.csv"
    args.target = "target"
    args.scale = False
    args.timeenc = 0

    # 例子：
    # sample_interval=8 -> 从 2 分钟采样数据变为 16 分钟间隔
    # slide_step=1      -> 起点逐点滑动（原始分辨率上滑动），大幅增加样本量
    dataset_train = Dataset_Stanford(
        args,
        root_path=args.root_path,
        flag='train',
        features='S',
        data_path_trainval=args.data_path_trainval,
        data_path_test=args.data_path_test,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        sample_interval=8,
        slide_step=8,
    )

    dataset_val = Dataset_Stanford(
        args,
        root_path=args.root_path,
        flag='val',
        features='S',
        data_path_trainval=args.data_path_trainval,
        data_path_test=args.data_path_test,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        sample_interval=8,
        slide_step=8,
    )

    dataset_test = Dataset_Stanford(
        args,
        root_path=args.root_path,
        flag='test',
        features='S',
        data_path_trainval=args.data_path_trainval,
        data_path_test=args.data_path_test,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        sample_interval=8,
        slide_step=8,
    )

    print(len(dataset_train))
    print(len(dataset_val))
    print(len(dataset_test))
