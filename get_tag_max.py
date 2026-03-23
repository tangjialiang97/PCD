import pandas as pd

def get_tag_max(dataset='Stanford'):
    if dataset == 'Stanford':
        stanford_path_trainval = '/opt/data/private/data/timedata/datasets/Stanford/trainval_timestamp_target.csv'
        stanford_path_test = '/opt/data/private/data/timedata/datasets/Stanford/test_timestamp_target.csv'
        df_trainval = pd.read_csv(stanford_path_trainval)
        df_test = pd.read_csv(stanford_path_test)
        print(df_trainval.columns)
        print(df_test.columns)
        print(df_trainval['target'].max())
        print(df_test['target'].max())
    elif dataset == 'folsom':
        folsom_path = '/opt/data/private/data/Folsom/Target_intra-hour_filtered.csv'
        df = pd.read_csv(folsom_path)
        print(df.columns)
        print(df['dni_5min'].max())
    elif dataset == 'Irr':
        irr_path_trainval = '/opt/data/private/data/Folsom/Folsom_irradiance_train.csv'
        irr_path_test = '/opt/data/private/data/Folsom/Folsom_irradiance_test.csv'
        df_trainval = pd.read_csv(irr_path_trainval)
        df_test = pd.read_csv(irr_path_test)
        print(df_trainval.columns)
        print(df_test.columns)
        print(df_trainval['ghi'].max())
        print(df_test['ghi'].max())

def get_daily_counts(dataset='Stanford'):
    if dataset == 'Stanford':
        stanford_path_trainval = '/opt/data/private/data/timedata/datasets/Stanford/trainval_timestamp_target.csv'
        df_trainval = pd.read_csv(stanford_path_trainval)
        df_trainval['timestamp'] = pd.to_datetime(df_trainval['timestamp'])
        df_trainval['date'] = df_trainval['timestamp'].dt.date
        daily_counts = df_trainval.groupby('date').size()
        return daily_counts
    elif dataset == 'folsom':
        folsom_path = '/opt/data/private/data/Folsom/Target_intra-hour_filtered.csv'
        df = pd.read_csv(folsom_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()
        return daily_counts
if __name__ == '__main__':
    get_tag_max(dataset='Irr')