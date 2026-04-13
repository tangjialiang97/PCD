import argparse
import math
import os
from typing import Dict

import numpy as np
import pandas as pd


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def _corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, relative_floor: float) -> Dict[str, float]:
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    rel_abs_residual = abs_residual / np.maximum(np.abs(y_true), relative_floor)
    return {
        "corr": _corrcoef(y_true, y_pred),
        "r2": _r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(abs_residual)),
        "mape_like": float(np.mean(rel_abs_residual)),
        "abs_residual_p95": float(np.percentile(abs_residual, 95)),
        "rel_abs_residual_p95": float(np.percentile(rel_abs_residual, 95)),
    }


def _prepare_frame(df: pd.DataFrame, args) -> pd.DataFrame:
    required_cols = [args.density_col, args.wind_speed_col, args.power_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    for col in required_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    work_df = work_df.dropna(subset=required_cols)
    work_df = work_df[(work_df[args.density_col] > 0.0) & (work_df[args.wind_speed_col] >= 0.0) & (work_df[args.power_col] >= 0.0)]
    if args.max_rows and len(work_df) > args.max_rows:
        stride = max(1, len(work_df) // args.max_rows)
        work_df = work_df.iloc[::stride].head(args.max_rows)
    return work_df.reset_index(drop=True)


def _build_cp_lookup(train_df: pd.DataFrame, args):
    density = train_df[args.density_col].to_numpy(dtype=np.float64)
    wind_speed = train_df[args.wind_speed_col].to_numpy(dtype=np.float64)
    power = train_df[args.power_col].to_numpy(dtype=np.float64)

    base_term = density * np.power(np.maximum(wind_speed, args.speed_floor), 3.0)
    cp_proxy = np.divide(power, np.maximum(base_term, args.base_floor))

    max_speed = float(np.nanmax(wind_speed))
    bin_edges = np.arange(0.0, max_speed + args.bin_width, args.bin_width)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, max_speed + args.bin_width], dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(wind_speed, bin_edges) - 1

    cp_bins = np.full(len(bin_centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(bin_centers), dtype=np.int64)
    for idx in range(len(bin_centers)):
        mask = bin_idx == idx
        if int(mask.sum()) < args.min_bin_count:
            continue
        cp_bins[idx] = float(np.quantile(cp_proxy[mask], args.cp_quantile))
        counts[idx] = int(mask.sum())

    valid = np.isfinite(cp_bins)
    if not np.any(valid):
        raise ValueError("No wind-speed bins had enough samples to estimate Cp(v).")

    cp_bins = np.interp(bin_centers, bin_centers[valid], cp_bins[valid])
    cp_bins = np.clip(cp_bins, 0.0, None)
    return bin_edges, bin_centers, cp_bins, counts


def _predict_power(df: pd.DataFrame, args, bin_edges: np.ndarray, cp_bins: np.ndarray) -> np.ndarray:
    density = df[args.density_col].to_numpy(dtype=np.float64)
    wind_speed = df[args.wind_speed_col].to_numpy(dtype=np.float64)
    bin_idx = np.digitize(wind_speed, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(cp_bins) - 1)
    cp_hat = cp_bins[bin_idx]
    return density * np.power(np.maximum(wind_speed, args.speed_floor), 3.0) * cp_hat


def _print_metrics(title: str, metrics: Dict[str, float]):
    print(title)
    for key in ["corr", "r2", "rmse", "mae", "mape_like", "abs_residual_p95", "rel_abs_residual_p95"]:
        print(f"{key}: {metrics[key]:.6f}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Simple validation for power ≈ density * wind_speed^3 * Cp(wind_speed), where the constant (1/2 A) is absorbed into Cp."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--density-col", type=str, default="density", help="Density column name")
    parser.add_argument("--wind-speed-col", type=str, default="wind_speed", help="Wind speed column name")
    parser.add_argument("--power-col", type=str, default="power", help="Power column name")
    parser.add_argument("--timestamp-col", type=str, default="timeStamp", help="Optional timestamp column for reporting")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Use the first train_ratio rows to fit Cp(v)")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Wind-speed bin width")
    parser.add_argument("--min-bin-count", type=int, default=50, help="Minimum samples per wind-speed bin")
    parser.add_argument("--cp-quantile", type=float, default=0.5, help="Quantile used to estimate Cp(v) per speed bin")
    parser.add_argument("--speed-floor", type=float, default=1e-3, help="Lower bound for wind speed inside v^3")
    parser.add_argument("--base-floor", type=float, default=1e-8, help="Lower bound for density*v^3 when computing Cp proxy")
    parser.add_argument("--relative-ref-floor", type=float, default=1.0, help="Floor for relative residual denominator")
    parser.add_argument("--max-rows", type=int, default=200000, help="Optional row cap for faster checks")
    parser.add_argument("--top-k", type=int, default=10, help="How many worst test rows to print")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)
    work_df = _prepare_frame(df, args)
    if work_df.empty:
        raise ValueError("No valid rows remain after filtering.")

    split_idx = max(1, min(len(work_df) - 1, int(len(work_df) * args.train_ratio)))
    train_df = work_df.iloc[:split_idx].reset_index(drop=True)
    test_df = work_df.iloc[split_idx:].reset_index(drop=True)

    bin_edges, bin_centers, cp_bins, counts = _build_cp_lookup(train_df, args)
    pred_train = _predict_power(train_df, args, bin_edges, cp_bins)
    pred_test = _predict_power(test_df, args, bin_edges, cp_bins)
    pred_all = _predict_power(work_df, args, bin_edges, cp_bins)

    power_train = train_df[args.power_col].to_numpy(dtype=np.float64)
    power_test = test_df[args.power_col].to_numpy(dtype=np.float64)
    power_all = work_df[args.power_col].to_numpy(dtype=np.float64)

    train_metrics = _evaluate_prediction(power_train, pred_train, args.relative_ref_floor)
    test_metrics = _evaluate_prediction(power_test, pred_test, args.relative_ref_floor)
    all_metrics = _evaluate_prediction(power_all, pred_all, args.relative_ref_floor)

    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(work_df)} / {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print("")
    print("Validated formula:")
    print("power_hat = density * wind_speed^3 * Cp_hat(wind_speed)")
    print("where Cp_hat(v) is fitted from historical data by wind-speed bins.")
    print("")
    print("First few fitted Cp_hat(v) bins:")
    shown = 0
    for left, right, center, cp_hat, count in zip(bin_edges[:-1], bin_edges[1:], bin_centers, cp_bins, counts):
        if shown >= 10:
            break
        print(f"[{left:.2f}, {right:.2f}) center={center:.2f}, Cp_hat={cp_hat:.8f}, train_count={int(count)}")
        shown += 1
    print("")

    _print_metrics("Train fit:", train_metrics)
    _print_metrics("Test fit:", test_metrics)
    _print_metrics("All-row fit:", all_metrics)

    test_residual = power_test - pred_test
    test_abs_residual = np.abs(test_residual)
    test_rel_abs_residual = test_abs_residual / np.maximum(np.abs(power_test), args.relative_ref_floor)
    worst_idx = np.argsort(-test_abs_residual)[:args.top_k]
    worst_df = test_df.iloc[worst_idx].copy()
    worst_df["_power_hat"] = pred_test[worst_idx]
    worst_df["_residual"] = test_residual[worst_idx]
    worst_df["_abs_residual"] = test_abs_residual[worst_idx]
    worst_df["_rel_abs_residual"] = test_rel_abs_residual[worst_idx]
    display_cols = [
        col for col in [args.timestamp_col, args.density_col, args.wind_speed_col, args.power_col]
        if col in worst_df.columns
    ]
    display_cols += ["_power_hat", "_residual", "_abs_residual", "_rel_abs_residual"]
    print(f"Top {min(args.top_k, len(worst_df))} worst test rows:")
    print(worst_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
