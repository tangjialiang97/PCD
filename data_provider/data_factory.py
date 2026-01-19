from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.data_loader_folsom import SlidingWindowLogicalDailySolarDataset, FastDataset
from data_provider.data_load_stanford import Dataset_Stanford
from data_provider.data_load_other import Dataset_Other
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

ymax_dict = {
    'Stanford': 29.5,
    'Irr': 1366,
    'DE_wind_offshore.csv': 6990.36,
    'DE_wind_onshore.csv': 40929.67,
    'DE_wind.csv': 46205.63,
    'Location1.csv': 0.9913,
    'Location2.csv': 0.9859,
    'Location3.csv': 0.9994,
    'Location4.csv': 0.9888,
    'bj1.csv': 500,
    'Aotizhongxin.csv': 898,
    'Changping.csv': 882,
    'Dingling.csv': 881,
    'Dongsi.csv': 737,
    'Guanyuan.csv': 680,
    'Gucheng.csv': 770,
    'Huairou.csv': 762,
    'Nongzhanguan.csv': 844,
    'Shunyi.csv': 941,
    'Tiantan.csv': 821,
    'Wanliu.csv': 957,
    'Wanshouxigong.csv': 999,
    'Aotizhongxin_clean.csv': 898,
    'Changping_clean.csv': 882,
    'Dingling_clean.csv': 881,
    'Dongsi_clean.csv': 737,
    'Guanyuan_clean.csv': 680,
    'Gucheng_clean.csv': 770,
    'Huairou_clean.csv': 762,
    'Nongzhanguan_clean.csv': 844,
    'Shunyi_clean.csv': 941,
    'Tiantan_clean.csv': 821,
    'Wanliu_clean.csv': 957,
    'Wanshouxigong_clean.csv': 999,
    'DE_wind_offshore_clean.csv': 6990.36,
    'DE_wind_onshore_clean.csv': 40929.67,
    'DE_wind_clean.csv': 46205.63,
    'NPRNGL.csv': 7.79,
    'CCG.csv': 6.71,
    'CRNN.csv': 25.85,
    'MRD.csv': 15.28,
    'flowedb1.csv': 20600.0,
    'flowedb2.csv': 2150.0,
    'flowedb3.csv': 347000.0,
    'flowedb4.csv': 1740.0,
    'flowedb5.csv': 18900.0,
    'flowedb7.csv': 26800.0,
    'flowedb8.csv': 310.0,
    'flowedb9.csv': 13700.0,
    'flowedb10.csv': 7400.0,
    'flowedb11.csv': 1300.0,
    'flowedb12.csv': 7300.0,
    'flowedb13.csv': 8400.0,
    '13PV1.csv': 1016.84,
    '13PV2.csv': 1051.16,
    '13PV3.csv': 1021.69,
    '13PV4.csv': 1027.86,
    'folsom.csv': 1366,
    'stanford.csv': 29.5,
}


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    # 'Folsom': FastDataset,
    'Folsom': SlidingWindowLogicalDailySolarDataset,
    'Stanford': Dataset_Stanford,
    'Irr': Dataset_Stanford,
    'AQ': Dataset_Other,
    'Wind': Dataset_Other,
    'Water': Dataset_Other,
    'Solar': Dataset_Other,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        if args.data == 'Folsom':
            Data = data_dict[args.data]
            if flag == 'test':
                shuffle_flag = False
                drop_last = True

            elif flag == 'valid':
                shuffle_flag = True
                drop_last = True

            else:
                shuffle_flag = True
                drop_last = True

            # 计算 label_len_hours 以匹配模型期望
            # 对于5分钟频率数据，每小时12个点
            points_per_hour = 12
            label_len_hours = args.label_len // points_per_hour
            
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                seq_len_hours=args.seq_len_hours,
                pred_len_hours=args.pred_len_hours,
                label_len_hours=label_len_hours,  # 根据模型的label_len计算
                day_start_time_str="16:00",
                day_end_time_str="00:15",
                stride_minutes=10,
                features=args.features,
                target='ghi_5min',
                scale=False,
                timeenc=0,
                freq='5t'
            )
            print(len(data_set))
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        elif args.data == 'Stanford':
            data_set = Data(
                args,
                root_path=args.root_path,
                flag=flag,
                features='S',
                data_path_trainval=args.data_path_trainval,
                data_path_test=args.data_path_test,
                target='target',
                scale=True,
                timeenc=0,
                ymax=ymax_dict[args.data]
            )
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        elif args.data == 'Irr':
            data_set = Data(
                args,
                root_path=args.root_path,
                flag=flag,
                features='S',
                data_path_trainval=args.data_path_trainval,
                data_path_test=args.data_path_test,
                target='ghi',
                scale=True,
                timeenc=0,
                ymax=ymax_dict[args.data]
            )
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        elif args.data == 'AQ':
            data_set = Data(
                args,
                root_path=args.root_path,
                flag=flag,
                features='S',
                data_path=args.data_path,
                target='target',
                scale=True,
                timeenc=0,
                ymax=ymax_dict[args.data_path]
            )
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        elif args.data == 'Wind':
            if "DE_wind" in args.data_path:
                sample_interval = 4
            else:
                sample_interval = 1
            data_set = Data(
                args,
                root_path=args.root_path,
                flag=flag,
                features='S',
                data_path=args.data_path,
                sample_interval=sample_interval,
                target='target',
                scale=True,
                timeenc=0,
                ymax=ymax_dict[args.data_path]
            )
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        elif args.data == 'Water':
            if "flowedb1" in args.data_path:
                sample_interval = 1
            else:
                sample_interval = 4
            data_set = Data(
                args,
                root_path=args.root_path,
                flag=flag,
                features='S',
                data_path=args.data_path,
                sample_interval=sample_interval,
                target='target',
                scale=True,
                timeenc=0,
                ymax=ymax_dict[args.data_path]
            )
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        elif args.data == 'Solar':
            if "folsom" in args.data_path:
                target = 'ghi'
            else:
                target = 'target'
            sample_interval = 4
            data_set = Data(
                args,
                root_path=args.root_path,
                flag=flag,
                features='S',
                data_path=args.data_path,
                sample_interval=sample_interval,
                target=target,
                scale=True,
                timeenc=0,
                ymax=ymax_dict[args.data_path]
            )
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        # else:
        #     data_set = Data(
        #         args = args,
        #         root_path=args.root_path,
        #         data_path=args.data_path,
        #         flag=flag,
        #         size=[args.seq_len, args.label_len, args.pred_len],
        #         features=args.features,
        #         target=args.target,
        #         timeenc=timeenc,
        #         freq=freq,
        #         seasonal_patterns=args.seasonal_patterns
        #     )
        #     print(flag, len(data_set))
        #     data_loader = DataLoader(
        #         data_set,
        #         batch_size=batch_size,
        #         shuffle=shuffle_flag,
        #         num_workers=args.num_workers,
        #         drop_last=drop_last)
        #     return data_set, data_loader
