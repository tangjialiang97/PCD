import argparse
import os

import pandas as pd


def str2bool(value):
    return str(value).lower() in {'1', 'true', 't', 'yes', 'y'}


def build_power(ghi, capacity, ghi_ref, clip):
    if ghi_ref <= 0:
        raise ValueError('--ghi-ref must be positive.')
    if capacity < 0:
        raise ValueError('--capacity must be non-negative.')

    normalized = ghi.clip(lower=0.0) / float(ghi_ref)
    if clip:
        normalized = normalized.clip(upper=1.0)
    return normalized * float(capacity)


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    if not ext:
        ext = '.csv'
    return '{}_power{}'.format(root, ext)


def main():
    parser = argparse.ArgumentParser(description='Build a solar generation CSV from GHI.')
    parser.add_argument('--input', default='data/solar/Congo.csv', help='source CSV path')
    parser.add_argument('--output', default='', help='output CSV path')
    parser.add_argument('--ghi-col', default='ghi', help='GHI column name')
    parser.add_argument('--power-col', default='power', help='power column name to create')
    parser.add_argument('--capacity', type=float, default=1.0, help='installed/rated capacity')
    parser.add_argument('--ghi-ref', type=float, default=1000.0,
                        help='reference irradiation used to convert GHI into equivalent full-power hours')
    parser.add_argument('--clip', type=str2bool, default=False,
                        help='clip normalized GHI to [0, 1] before mapping')
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.input)
    df = pd.read_csv(args.input)
    if args.ghi_col not in df.columns:
        raise ValueError("GHI column '{}' not found in {}.".format(args.ghi_col, args.input))

    df[args.power_col] = build_power(
        df[args.ghi_col],
        capacity=args.capacity,
        ghi_ref=args.ghi_ref,
        clip=args.clip,
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        'wrote {} rows to {} with {} min={:.6f}, max={:.6f}, mean={:.6f}'.format(
            len(df),
            output_path,
            args.power_col,
            df[args.power_col].min(),
            df[args.power_col].max(),
            df[args.power_col].mean(),
        )
    )


if __name__ == '__main__':
    main()
