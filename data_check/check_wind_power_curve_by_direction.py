import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_check.check_wind_power_curve import (  # noqa: E402
    _build_bins,
    _estimate_curve_points,
    _evaluate_prediction,
    _piecewise_power_curve,
    _prepare_frame,
)


def _sector_bounds(num_sectors: int) -> np.ndarray:
    return np.linspace(0.0, 360.0, num_sectors + 1)


def _assign_sector(wind_direction: np.ndarray, num_sectors: int) -> np.ndarray:
    direction = np.mod(wind_direction, 360.0)
    edges = _sector_bounds(num_sectors)
    sector_idx = np.digitize(direction, edges, right=False) - 1
    sector_idx = np.where(sector_idx == num_sectors, 0, sector_idx)
    return sector_idx.astype(np.int64)


def _sector_label(sector_idx: int, num_sectors: int) -> str:
    edges = _sector_bounds(num_sectors)
    left = edges[sector_idx]
    right = edges[sector_idx + 1]
    return f"[{left:.1f}, {right:.1f})"


def main():
    parser = argparse.ArgumentParser(
        description="Fit density-corrected wind power curves by wind-direction sector and compare against a single global curve."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--density-col", type=str, default="density", help="Density column name")
    parser.add_argument("--wind-speed-col", type=str, default="wind_speed", help="Wind speed column name")
    parser.add_argument("--wind-direction-col", type=str, default="wind_direction", help="Wind direction column name")
    parser.add_argument("--power-col", type=str, default="power", help="Power column name")
    parser.add_argument("--reference-density", type=float, default=1.225, help="Reference density for wind-speed correction")
    parser.add_argument("--max-rows", type=int, default=200000, help="Optional row cap for faster estimation")
    parser.add_argument("--num-sectors", type=int, default=8, help="How many wind-direction sectors to use")
    parser.add_argument("--min-sector-rows", type=int, default=2000, help="Minimum rows needed to fit a dedicated sector curve")
    parser.add_argument("--rated-power-quantile", type=float, default=0.995, help="Quantile used to estimate rated power")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Effective wind speed bin width")
    parser.add_argument("--min-bin-count", type=int, default=50, help="Minimum samples per speed bin")
    parser.add_argument("--required-consecutive-bins", type=int, default=2, help="Consecutive bins needed to trigger cut-in/rated/cut-out")
    parser.add_argument("--active-power-threshold-ratio", type=float, default=0.02, help="Power ratio counted as active")
    parser.add_argument("--cut-in-power-ratio", type=float, default=0.05, help="Fraction of rated power used to detect cut-in")
    parser.add_argument("--cut-in-active-ratio", type=float, default=0.5, help="Minimum active ratio used to detect cut-in")
    parser.add_argument("--rated-power-ratio", type=float, default=0.95, help="Fraction of rated power used to detect rated speed")
    parser.add_argument("--rated-active-ratio", type=float, default=0.8, help="Minimum active ratio used to detect rated speed")
    parser.add_argument("--cut-out-power-ratio", type=float, default=0.05, help="Fraction of rated power used to detect cut-out")
    parser.add_argument("--cut-out-active-ratio", type=float, default=0.2, help="Maximum active ratio used to detect cut-out")
    parser.add_argument("--relative-ref-floor", type=float, default=1.0, help="Floor for relative residual denominator")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)
    work_df = _prepare_frame(df, args)
    if args.wind_direction_col not in work_df.columns:
        raise ValueError(f"Missing required column: {args.wind_direction_col}")
    work_df[args.wind_direction_col] = pd.to_numeric(work_df[args.wind_direction_col], errors="coerce")
    work_df = work_df.dropna(subset=[args.wind_direction_col]).reset_index(drop=True)
    if work_df.empty:
        raise ValueError("No valid rows remain after adding wind-direction filtering.")

    power = work_df[args.power_col].to_numpy(dtype=np.float64)
    v_eff = work_df["_effective_wind_speed"].to_numpy(dtype=np.float64)
    direction = work_df[args.wind_direction_col].to_numpy(dtype=np.float64)

    global_rated_power = float(np.quantile(power, args.rated_power_quantile))
    global_bins = _build_bins(work_df, args, global_rated_power)
    global_curve = _estimate_curve_points(global_bins, args, global_rated_power)
    global_pred = _piecewise_power_curve(v_eff, global_curve)
    global_metrics = _evaluate_prediction(power, global_pred, args.relative_ref_floor)

    sector_idx = _assign_sector(direction, args.num_sectors)
    sector_pred = np.empty_like(power)
    sector_rows: List[Dict[str, float]] = []

    for idx in range(args.num_sectors):
        mask = sector_idx == idx
        sector_count = int(mask.sum())
        if sector_count == 0:
            continue
        sector_df = work_df.loc[mask].reset_index(drop=True)
        if sector_count < args.min_sector_rows:
            sector_curve = global_curve
            curve_source = "global_fallback"
        else:
            sector_power = sector_df[args.power_col].to_numpy(dtype=np.float64)
            sector_rated_power = float(np.quantile(sector_power, args.rated_power_quantile))
            sector_bins = _build_bins(sector_df, args, sector_rated_power)
            if sector_bins.empty:
                sector_curve = global_curve
                curve_source = "global_fallback"
            else:
                sector_curve = _estimate_curve_points(sector_bins, args, sector_rated_power)
                curve_source = "sector_fit"

        sector_v_eff = sector_df["_effective_wind_speed"].to_numpy(dtype=np.float64)
        sector_pred_local = _piecewise_power_curve(sector_v_eff, sector_curve)
        sector_pred[mask] = sector_pred_local
        sector_metrics = _evaluate_prediction(
            sector_df[args.power_col].to_numpy(dtype=np.float64),
            sector_pred_local,
            args.relative_ref_floor,
        )
        sector_rows.append({
            "sector": idx,
            "direction_range": _sector_label(idx, args.num_sectors),
            "count": sector_count,
            "source": curve_source,
            "rated_power": float(sector_curve.rated_power),
            "cut_in": float(sector_curve.cut_in),
            "rated_speed": float(sector_curve.rated_speed),
            "cut_out": float(sector_curve.cut_out) if np.isfinite(sector_curve.cut_out) else np.nan,
            "corr": float(sector_metrics["corr"]),
            "r2": float(sector_metrics["r2"]),
            "rmse": float(sector_metrics["rmse"]),
            "mae": float(sector_metrics["mae"]),
        })

    directional_metrics = _evaluate_prediction(power, sector_pred, args.relative_ref_floor)

    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(work_df)} / {len(df)}")
    print(f"Direction sectors: {args.num_sectors}")
    print("")
    print("Global single-curve fit:")
    for key in ["corr", "r2", "rmse", "mae", "mape_like", "abs_residual_p95", "rel_abs_residual_p95"]:
        print(f"{key}: {global_metrics[key]:.6f}")
    print("")
    print("Directional sectorized fit:")
    for key in ["corr", "r2", "rmse", "mae", "mape_like", "abs_residual_p95", "rel_abs_residual_p95"]:
        print(f"{key}: {directional_metrics[key]:.6f}")
    print("")
    print("Improvement (directional - global):")
    print(f"delta_corr: {directional_metrics['corr'] - global_metrics['corr']:.6f}")
    print(f"delta_r2: {directional_metrics['r2'] - global_metrics['r2']:.6f}")
    print(f"delta_rmse: {directional_metrics['rmse'] - global_metrics['rmse']:.6f}")
    print(f"delta_mae: {directional_metrics['mae'] - global_metrics['mae']:.6f}")
    print("")

    sector_table = pd.DataFrame(sector_rows).sort_values("sector").reset_index(drop=True)
    print("Per-sector curves:")
    print(sector_table.to_string(index=False))


if __name__ == "__main__":
    main()
