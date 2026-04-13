import argparse
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_check.check_wind_power_curve import _build_bins, _estimate_curve_points, _prepare_frame


def main():
    parser = argparse.ArgumentParser(
        description="Filter anomalous wind-power samples, save a cleaned CSV, and report removal ratios."
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output-csv", type=str, required=True, help="Output filtered CSV path")
    parser.add_argument("--density-col", type=str, default="density", help="Density column name")
    parser.add_argument("--wind-speed-col", type=str, default="wind_speed", help="Wind speed column name")
    parser.add_argument("--power-col", type=str, default="power", help="Power column name")
    parser.add_argument("--timestamp-col", type=str, default="timeStamp", help="Timestamp column name")
    parser.add_argument("--reference-density", type=float, default=1.225, help="Reference density")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for parameter estimation; 0 uses all rows")
    parser.add_argument("--rated-power-quantile", type=float, default=0.995, help="Quantile used to estimate rated power")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Effective wind speed bin width")
    parser.add_argument("--min-bin-count", type=int, default=50, help="Minimum number of samples per speed bin")
    parser.add_argument("--required-consecutive-bins", type=int, default=2, help="How many consecutive bins must satisfy a rule")
    parser.add_argument("--active-power-threshold-ratio", type=float, default=0.02, help="Power ratio counted as active")
    parser.add_argument("--cut-in-power-ratio", type=float, default=0.05, help="Fraction of rated power used to detect cut-in")
    parser.add_argument("--cut-in-active-ratio", type=float, default=0.5, help="Minimum active ratio used to detect cut-in")
    parser.add_argument("--rated-power-ratio", type=float, default=0.95, help="Fraction of rated power used to detect rated speed")
    parser.add_argument("--rated-active-ratio", type=float, default=0.8, help="Minimum active ratio used to detect rated speed")
    parser.add_argument("--cut-out-power-ratio", type=float, default=0.05, help="Fraction of rated power used to detect cut-out")
    parser.add_argument("--cut-out-active-ratio", type=float, default=0.2, help="Maximum active ratio used to detect cut-out")
    parser.add_argument("--near-zero-ratio", type=float, default=0.05, help="Power ratio threshold for near-zero power")
    parser.add_argument("--low-power-ratio", type=float, default=0.20, help="Power ratio upper bound for low power")
    parser.add_argument("--sudden-drop-prev-ratio", type=float, default=0.50, help="Previous-step power ratio threshold for sudden drops")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    output_csv = os.path.abspath(args.output_csv)
    raw_df = pd.read_csv(csv_path)

    work_df = _prepare_frame(raw_df, args)
    rated_power = float(np.quantile(work_df[args.power_col].to_numpy(dtype=np.float64), args.rated_power_quantile))
    bins_df = _build_bins(work_df, args, rated_power)
    curve = _estimate_curve_points(bins_df, args, rated_power)

    near_zero_threshold = args.near_zero_ratio * curve.rated_power
    low_power_threshold = args.low_power_ratio * curve.rated_power
    sudden_drop_prev_threshold = args.sudden_drop_prev_ratio * curve.rated_power

    density = pd.to_numeric(raw_df[args.density_col], errors="coerce").to_numpy(dtype=np.float64)
    wind_speed = pd.to_numeric(raw_df[args.wind_speed_col], errors="coerce").to_numpy(dtype=np.float64)
    power = pd.to_numeric(raw_df[args.power_col], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(density) & np.isfinite(wind_speed) & np.isfinite(power) & (density > 0.0) & (wind_speed >= 0.0) & (power >= 0.0)
    density_ratio = np.clip(density / args.reference_density, 0.25, 2.5)
    v_eff = wind_speed * np.cbrt(density_ratio)
    prev_power = np.roll(power, 1)
    prev_power[0] = np.nan
    prev_v_eff = np.roll(v_eff, 1)
    prev_v_eff[0] = np.nan

    high_wind = valid_mask & (v_eff >= curve.rated_speed)
    near_zero = valid_mask & (power <= near_zero_threshold)
    low_power = valid_mask & (power > near_zero_threshold) & (power <= low_power_threshold)
    high_wind_near_zero = high_wind & near_zero
    high_wind_low_power = high_wind & low_power
    sudden_drop_to_zero = valid_mask & (power <= near_zero_threshold) & (prev_power >= sudden_drop_prev_threshold)

    remove_mask = high_wind_near_zero | high_wind_low_power | sudden_drop_to_zero
    filtered_df = raw_df.loc[~remove_mask].copy()
    filtered_df.to_csv(output_csv, index=False)

    total = int(valid_mask.sum())
    removed = int(remove_mask.sum())
    print(f"input_csv: {csv_path}")
    print(f"output_csv: {output_csv}")
    print(f"rated_power_est: {curve.rated_power:.6f}")
    print(f"rated_speed_est: {curve.rated_speed:.6f}")
    print(f"near_zero_threshold: {near_zero_threshold:.6f}")
    print(f"low_power_threshold: {low_power_threshold:.6f}")
    print(f"sudden_drop_prev_threshold: {sudden_drop_prev_threshold:.6f}")
    print(f"valid_rows: {total}")
    print(f"removed_rows: {removed}")
    print(f"removed_pct: {(removed / max(total, 1)) * 100.0:.6f}%")
    print(f"high_wind_near_zero_removed: {int(high_wind_near_zero.sum())}")
    print(f"high_wind_low_power_removed: {int(high_wind_low_power.sum())}")
    print(f"sudden_drop_to_zero_removed: {int(sudden_drop_to_zero.sum())}")


if __name__ == "__main__":
    main()
