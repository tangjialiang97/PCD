from torch.utils.data import DataLoader

from data_provider.data_load import Dataset


ymax_dict = {
    'Folsom.csv': [1466.0, 1004.0, 748.1],
    'Congo.csv': [263.2202, 236.6564, 148.741],
    'Germany.csv': [239.8238, 255.692, 131.8923],
    'NewDelhi.csv': [267.5592, 252.4045, 156.8177],
}


data_dict = {
    'Solar': Dataset,
}


def _loader(data_set, args, shuffle_flag, drop_last):
    return DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )


def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False

    if "Folsom" in args.data_path:
        sample_interval = 15
    else:
        sample_interval = 1
    data_set = Data(
        args,
        root_path=args.root_path,
        flag=flag,
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        sample_interval=sample_interval,
        slide_step=args.slide_step,
        scale=False,
        timeenc=0,
        ymax=ymax_dict.get(args.data_path, 29.5)
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
