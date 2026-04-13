import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_check.check_wind_cp_proxy import (  # noqa: E402
    _build_cp_lookup,
    _evaluate_prediction,
    _prepare_frame,
)
from data_check.check_wind_power_curve import (  # noqa: E402
    _build_bins,
    _estimate_curve_points,
)


def _predict_uncapped(df: pd.DataFrame, args, bin_edges: np.ndarray, cp_bins: np.ndarray) -> np.ndarray:
    density = df[args.density_col].to_numpy(dtype=np.float64)
    wind_speed = df[args.wind_speed_col].to_numpy(dtype=np.float64)
    bin_idx = np.digitize(wind_speed, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(cp_bins) - 1)
    cp_hat = cp_bins[bin_idx]
    return density * np.power(np.maximum(wind_speed, args.speed_floor), 3.0) * cp_hat


def _predict_capped(df: pd.DataFrame, args, bin_edges: np.ndarray, cp_bins: np.ndarray,
                    cut_in: float, rated_power: float) -> np.ndarray:
    pred = _predict_uncapped(df, args, bin_edges, cp_bins)
    density_ratio = np.clip(df[args.density_col].to_numpy(dtype=np.float64) / args.reference_density, 0.25, 2.5)
    v_eff = df[args.wind_speed_col].to_numpy(dtype=np.float64) * np.cbrt(density_ratio)
    pred = np.where(v_eff < cut_in, 0.0, pred)
    pred = np.minimum(pred, rated_power)
    return pred


def _print_metrics(title: str, metrics: Dict[str, float]):
    print(title)
    for key in ["corr", "r2", "rmse", "mae", "mape_like", "abs_residual_p95", "rel_abs_residual_p95"]:
        print(f"{key}: {metrics[key]:.6f}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Compare uncapped Cp(v) against a capped version with cut-in and rated-power saturation."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--density-col", type=str, default="density", help="Density column name")
    parser.add_argument("--wind-speed-col", type=str, default="wind_speed", help="Wind speed column name")
    parser.add_argument("--power-col", type=str, default="power", help="Power column name")
    parser.add_argument("--timestamp-col", type=str, default="timeStamp", help="Optional timestamp column for reporting")
    parser.add_argument("--reference-density", type=float, default=1.225, help="Reference density for v_eff")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Use the first train_ratio rows to fit Cp(v)")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Wind-speed bin width")
    parser.add_argument("--min-bin-count", type=int, default=50, help="Minimum samples per wind-speed bin")
    parser.add_argument("--cp-quantile", type=float, default=0.5, help="Quantile used to estimate Cp(v) per speed bin")
    parser.add_argument("--speed-floor", type=float, default=1e-3, help="Lower bound for wind speed inside v^3")
    parser.add_argument("--base-floor", type=float, default=1e-8, help="Lower bound for density*v^3 when computing Cp proxy")
    parser.add_argument("--relative-ref-floor", type=float, default=1.0, help="Floor for relative residual denominator")
    parser.add_argument("--max-rows", type=int, default=200000, help="Optional row cap for faster checks")
    parser.add_argument("--rated-power-quantile", type=float, default=0.995, help="Quantile used to estimate rated power")
    parser.add_argument("--required-consecutive-bins", type=int, default=2, help="Consecutive bins needed to trigger cut-in/rated/cut-out")
    parser.add_argument("--active-power-threshold-ratio", type=float, default=0.02, help="Power ratio counted as active")
    parser.add_argument("--cut-in-power-ratio", type=float, default=0.05, help="Fraction of rated power used to detect cut-in")
    parser.add_argument("--cut-in-active-ratio", type=float, default=0.5, help="Minimum active ratio used to detect cut-in")
    parser.add_argument("--rated-power-ratio", type=float, default=0.95, help="Fraction of rated power used to detect rated speed")
    parser.add_argument("--rated-active-ratio", type=float, default=0.8, help="Minimum active ratio used to detect rated speed")
    parser.add_argument("--cut-out-power-ratio", type=float, default=0.05, help="Fraction of rated power used to detect cut-out")
    parser.add_argument("--cut-out-active-ratio", type=float, default=0.2, help="Maximum active ratio used to detect cut-out")
    parser.add_argument("--top-k", type=int, default=10, help="How many worst capped test rows to print")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)
    work_df = _prepare_frame(df, args)
    if work_df.empty:
        raise ValueError("No valid rows remain after filtering.")

    split_idx = max(1, min(len(work_df) - 1, int(len(work_df) * args.train_ratio)))
    train_df = work_df.iloc[:split_idx].reset_index(drop=True)
    test_df = work_df.iloc[split_idx:].reset_index(drop=True)

    bin_edges, _bin_centers, cp_bins, _counts = _build_cp_lookup(train_df, args)

    train_power = train_df[args.power_col].to_numpy(dtype=np.float64)
    rated_power = float(np.quantile(train_power, args.rated_power_quantile))
    train_curve_df = train_df.copy()
    density_ratio = np.clip(train_curve_df[args.density_col].to_numpy(dtype=np.float64) / args.reference_density, 0.25, 2.5)
    train_curve_df["_effective_wind_speed"] = train_curve_df[args.wind_speed_col].to_numpy(dtype=np.float64) * np.cbrt(density_ratio)
    curve_bins = _build_bins(train_curve_df, args, rated_power)
    curve = _estimate_curve_points(curve_bins, args, rated_power)

    pred_train_uncapped = _predict_uncapped(train_df, args, bin_edges, cp_bins)
    pred_test_uncapped = _predict_uncapped(test_df, args, bin_edges, cp_bins)
    pred_all_uncapped = _predict_uncapped(work_df, args, bin_edges, cp_bins)

    pred_train_capped = _predict_capped(train_df, args, bin_edges, cp_bins, curve.cut_in, curve.rated_power)
    pred_test_capped = _predict_capped(test_df, args, bin_edges, cp_bins, curve.cut_in, curve.rated_power)
    pred_all_capped = _predict_capped(work_df, args, bin_edges, cp_bins, curve.cut_in, curve.rated_power)

    power_train = train_df[args.power_col].to_numpy(dtype=np.float64)
    power_test = test_df[args.power_col].to_numpy(dtype=np.float64)
    power_all = work_df[args.power_col].to_numpy(dtype=np.float64)

    uncapped_test = _evaluate_prediction(power_test, pred_test_uncapped, args.relative_ref_floor)
    capped_test = _evaluate_prediction(power_test, pred_test_capped, args.relative_ref_floor)
    uncapped_all = _evaluate_prediction(power_all, pred_all_uncapped, args.relative_ref_floor)
    capped_all = _evaluate_prediction(power_all, pred_all_capped, args.relative_ref_floor)

    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(work_df)} / {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"estimated_cut_in: {curve.cut_in:.6f}")
    print(f"estimated_rated_power: {curve.rated_power:.6f}")
    print("")
    print("Formula comparison:")
    print("uncapped: power_hat = density * wind_speed^3 * Cp_hat(wind_speed)")
    print("capped:   power_hat = min(rated_power, density * wind_speed^3 * Cp_hat(wind_speed)), then set 0 when v_eff < cut_in")
    print("")

    _print_metrics("Test fit (uncapped):", uncapped_test)
    _print_metrics("Test fit (capped):", capped_test)
    _print_metrics("All-row fit (uncapped):", uncapped_all)
    _print_metrics("All-row fit (capped):", capped_all)

    print("Improvement on test set (capped - uncapped):")
    print(f"delta_corr: {capped_test['corr'] - uncapped_test['corr']:.6f}")
    print(f"delta_r2: {capped_test['r2'] - uncapped_test['r2']:.6f}")
    print(f"delta_rmse: {capped_test['rmse'] - uncapped_test['rmse']:.6f}")
    print(f"delta_mae: {capped_test['mae'] - uncapped_test['mae']:.6f}")
    print("")

    residual = power_test - pred_test_capped
    abs_residual = np.abs(residual)
    rel_abs_residual = abs_residual / np.maximum(np.abs(power_test), args.relative_ref_floor)
    worst_idx = np.argsort(-abs_residual)[:args.top_k]
    worst_df = test_df.iloc[worst_idx].copy()
    worst_df["_power_hat_uncapped"] = pred_test_uncapped[worst_idx]
    worst_df["_power_hat_capped"] = pred_test_capped[worst_idx]
    worst_df["_residual_capped"] = residual[worst_idx]
    worst_df["_abs_residual_capped"] = abs_residual[worst_idx]
    worst_df["_rel_abs_residual_capped"] = rel_abs_residual[worst_idx]
    display_cols = [
        col for col in [args.timestamp_col, args.density_col, args.wind_speed_col, args.power_col]
        if col in worst_df.columns
    ]
    display_cols += [
        "_power_hat_uncapped",
        "_power_hat_capped",
        "_residual_capped",
        "_abs_residual_capped",
        "_rel_abs_residual_capped",
    ]
    print(f"Top {min(args.top_k, len(worst_df))} worst capped test rows:")
    print(worst_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
