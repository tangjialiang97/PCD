import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import argparse

warnings.filterwarnings('ignore')

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

        self.sample_interval = int(sample_interval)
        assert self.sample_interval >= 1
        self.slide_step = int(slide_step)
        assert self.slide_step >= 1

        self.root_path = root_path
        self.data_path_ = data_path

        self.__read_data__()
        self.__build_window_meta__()  # 预计算长度等

    def __read_data__(self):
        """
        读取单个 data_path_，按 7:1:2 划分 train/val/test。
        只在训练段上 fit scaler，val/test 复用同一 scaler。
        """
        self.scaler = StandardScaler()
        df_raw_ = pd.read_csv(os.path.join(self.root_path, self.data_path_))

        # Ensure timestamp is datetime
        if 'timestamp' not in df_raw_.columns:
            raise ValueError("Expected column 'timestamp' in CSV")
        df_raw_['timestamp'] = pd.to_datetime(df_raw_['timestamp'])

        # Select feature columns
        if self.features in ['M', 'MS']:
            cols_data_ = [c for c in df_raw_.columns if c not in ['timestamp']]
            df_data_ = df_raw_[cols_data_]
        elif self.features == 'S':
            if self.target not in df_raw_.columns:
                raise ValueError(f"Target column '{self.target}' not found in CSV")
            df_data_ = df_raw_[[self.target]]
        else:
            raise ValueError(f"Unsupported features mode: {self.features}")

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
    args.seq_len = 8
    args.label_len = 0
    args.pred_len = 8
    args.root_path = "/opt/data/private/data/datadbl/wind/kag/"
    args.data_path = "Location1.csv"
    args.target = "target"
    args.timeenc = 0

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
        sample_interval=1,
        slide_step=8,
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
        sample_interval=1,
        slide_step=8,
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
        sample_interval=1,
        slide_step=8,
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
        sample_interval=1,
        slide_step=8,
    )

    dataset_train_scaled_inverse = dataset_train_scaled.inverse_transform(dataset_train_scaled.data_x)
    dataset_test_scaled_inverse = dataset_test_scaled.inverse_transform(dataset_test_scaled.data_x)

    # 计算dataset_train_scaled和dataset_train_unscaled的均值和标准差
    mean_scaled = dataset_test_scaled.data_x.mean()
    std_scaled = dataset_test_scaled.data_x.std()
    mean_unscaled = dataset_test_unscaled.data_x.mean()
    std_unscaled = dataset_test_unscaled.data_x.std()
    mean_unscaled_inverse = dataset_test_scaled_inverse.mean()
    std_unscaled_inverse = dataset_test_scaled_inverse.std()
    print(mean_scaled, std_scaled)
    print(mean_unscaled, std_unscaled)
    print(mean_unscaled_inverse, std_unscaled_inverse)

    # 计算dataset_train_scaled，dataset_train_unscaled，dataset_test_scaled，dataset_test_unscaled mse
    mse_scaled = np.mean((dataset_test_scaled.data_x - dataset_test_unscaled.data_x) ** 2)
    mse_unscaled_inverse = np.mean((dataset_test_scaled_inverse - dataset_test_unscaled.data_x) ** 2)
    print(mse_scaled, mse_unscaled_inverse)
