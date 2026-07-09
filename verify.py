import argparse
import os

import numpy as np
import pandas as pd
import pvlib


DATASET_LAT_LON = {
    'Folsom.csv': (38.677, -121.15),
    'Congo.csv': (2.61924, 19.85019),
    'Germany.csv': (50.0, 10.0),
    'NewDelhi.csv': (28.0, 77.0),
    'Qingda_plant_power.csv': (29.919, 100.641),
    'Jiefang_plant_power.csv': (29.919, 100.641),
    'plant_power.csv': (29.919, 100.641),
}


def str2bool(value):
    return str(value).lower() in {'1', 'true', 't', 'yes', 'y'}


def resolve_location(csv_path, latitude, longitude):
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude)

    file_name = os.path.basename(csv_path)
    location_key = file_name
    stem, ext = os.path.splitext(location_key)
    for suffix in ('_plant_power', '_power'):
        if stem.endswith(suffix):
            location_key = '{}{}'.format(stem[:-len(suffix)], ext)
            break

    if location_key in DATASET_LAT_LON or file_name in DATASET_LAT_LON:
        return DATASET_LAT_LON.get(file_name, DATASET_LAT_LON[location_key])

    raise ValueError(
        'Cannot infer latitude/longitude for {}. Pass --latitude and --longitude.'.format(
            csv_path
        )
    )


def find_timestamp_column(df):
    for column in ['timestamp', 'timeStamp', 'dtime', 'datetime']:
        if column in df.columns:
            return column
    return None


def parse_timestamp(ts_series):
    if pd.api.types.is_datetime64_any_dtype(ts_series):
        return pd.to_datetime(ts_series, errors='coerce')
    numeric = pd.to_numeric(ts_series, errors='coerce')
    numeric_ratio = float(numeric.notna().mean()) if len(ts_series) else 0.0
    if numeric_ratio > 0.8 and numeric.dropna().median() > 20000:
        return pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')
    return pd.to_datetime(ts_series, errors='coerce')


def compute_cos_zenith(df, csv_path, args):
    if args.cos_col:
        if args.cos_col not in df.columns:
            raise ValueError("cos column '{}' not found.".format(args.cos_col))
        return df[args.cos_col].to_numpy(dtype=np.float64)

    if args.zenith_col:
        if args.zenith_col not in df.columns:
            raise ValueError("zenith column '{}' not found.".format(args.zenith_col))
        zenith = df[args.zenith_col].to_numpy(dtype=np.float64)
        return np.cos(np.deg2rad(zenith))

    timestamp_col = find_timestamp_column(df)
    if timestamp_col is None:
        if args.assume_cos_one:
            return np.ones(len(df), dtype=np.float64)
        raise ValueError(
            "Expected a timestamp column, --cos-col, or --zenith-col. "
            "Use --assume-cos-one True only for checking GHI = DNI + DHI."
        )

    latitude, longitude = resolve_location(csv_path, args.latitude, args.longitude)
    ts = parse_timestamp(df[timestamp_col])
    if ts.isna().any():
        raise ValueError("Invalid timestamp values found in '{}'.".format(timestamp_col))

    dt_index = pd.DatetimeIndex(ts)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(args.timezone)
    else:
        dt_index = dt_index.tz_convert(args.timezone)

    solpos = pvlib.solarposition.get_solarposition(
        dt_index,
        latitude=latitude,
        longitude=longitude,
    )
    return np.cos(np.deg2rad(solpos['zenith'].to_numpy(dtype=np.float64)))


def validate_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError('Missing required columns: {}'.format(', '.join(missing)))


def compute_rates(df, cos_zenith, args):
    validate_columns(df, [args.ghi_col, args.dni_col, args.dhi_col])

    ghi = df[args.ghi_col].to_numpy(dtype=np.float64)
    dni = df[args.dni_col].to_numpy(dtype=np.float64)
    dhi = df[args.dhi_col].to_numpy(dtype=np.float64)
    cos_zenith = np.asarray(cos_zenith, dtype=np.float64)

    if len(cos_zenith) != len(df):
        raise ValueError('cos_zenith length does not match CSV rows.')

    cos_day = np.clip(cos_zenith, 0.0, None)
    expected_ghi = cos_day * dni + dhi
    residual = ghi - expected_ghi
    scale = np.maximum.reduce([
        np.abs(ghi),
        np.abs(expected_ghi),
        np.ones_like(ghi),
    ])
    closed = np.abs(residual) <= (args.atol + args.rtol * scale)

    day_mask = cos_zenith > args.night_cos_threshold
    night_mask = ~day_mask
    nonnegative = (ghi >= -args.atol) & (dni >= -args.atol) & (dhi >= -args.atol)

    night_zero = np.maximum.reduce([np.abs(ghi), np.abs(dni), np.abs(dhi)]) <= args.night_atol
    feasible = np.where(day_mask, closed & nonnegative, night_zero)

    return {
        'rows': len(df),
        'day_rows': int(day_mask.sum()),
        'night_rows': int(night_mask.sum()),
        'closure_rate': float(closed.mean()) if len(df) else float('nan'),
        'day_closure_rate': float(closed[day_mask].mean()) if day_mask.any() else float('nan'),
        'night_zero_rate': float(night_zero[night_mask].mean()) if night_mask.any() else float('nan'),
        'nonnegative_rate': float(nonnegative.mean()) if len(df) else float('nan'),
        'physical_set_closure_rate': float(feasible.mean()) if len(df) else float('nan'),
        'max_abs_residual': float(np.max(np.abs(residual))) if len(df) else float('nan'),
        'mean_abs_residual': float(np.mean(np.abs(residual))) if len(df) else float('nan'),
        'p95_abs_residual': float(np.percentile(np.abs(residual), 95)) if len(df) else float('nan'),
        'failed_rows': int((~feasible).sum()),
    }


def print_report(csv_path, rates, args):
    print('file: {}'.format(csv_path))
    print('rows: {}'.format(rates['rows']))
    print('day rows: {}, night rows: {}'.format(rates['day_rows'], rates['night_rows']))
    print('tolerance: atol={}, rtol={}, night_atol={}, night_cos_threshold={}'.format(
        args.atol,
        args.rtol,
        args.night_atol,
        args.night_cos_threshold,
    ))
    print('closure rate: {:.6%}'.format(rates['closure_rate']))
    print('day closure rate: {:.6%}'.format(rates['day_closure_rate']))
    print('night zero rate: {:.6%}'.format(rates['night_zero_rate']))
    print('nonnegative rate: {:.6%}'.format(rates['nonnegative_rate']))
    print('physical set closure rate: {:.6%}'.format(rates['physical_set_closure_rate']))
    print('failed rows: {}'.format(rates['failed_rows']))
    print('abs residual: mean={:.6f}, p95={:.6f}, max={:.6f}'.format(
        rates['mean_abs_residual'],
        rates['p95_abs_residual'],
        rates['max_abs_residual'],
    ))


def main():
    parser = argparse.ArgumentParser(
        description='Verify solar physical-set closure for GHI/DNI/DHI CSV data.'
    )
    parser.add_argument('csv_path', help='CSV file path')
    parser.add_argument('--ghi-col', default='ghi', help='GHI column name')
    parser.add_argument('--dni-col', default='dni', help='DNI column name')
    parser.add_argument('--dhi-col', default='dhi', help='DHI column name')
    parser.add_argument('--cos-col', default='', help='optional cos(zenith) column name')
    parser.add_argument('--zenith-col', default='', help='optional zenith angle column in degrees')
    parser.add_argument('--latitude', type=float, default=None, help='site latitude')
    parser.add_argument('--longitude', type=float, default=None, help='site longitude')
    parser.add_argument('--timezone', default='UTC', help='timezone for naive timestamps')
    parser.add_argument('--night-cos-threshold', type=float, default=0.0,
                        help='cos(zenith) threshold for night/origin feasible set')
    parser.add_argument('--atol', type=float, default=1e-3, help='absolute tolerance')
    parser.add_argument('--rtol', type=float, default=1e-3, help='relative tolerance')
    parser.add_argument('--night-atol', type=float, default=1e-3,
                        help='absolute tolerance for night rows to be treated as zero')
    parser.add_argument('--assume-cos-one', type=str2bool, default=False,
                        help='fallback check GHI = DNI + DHI when no sun-angle data exists')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    cos_zenith = compute_cos_zenith(df, args.csv_path, args)
    rates = compute_rates(df, cos_zenith, args)
    print_report(args.csv_path, rates, args)


if __name__ == '__main__':
    main()
