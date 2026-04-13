import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


REFERENCE_DENSITY = 1.225


@dataclass
class CurveEstimate:
    rated_power: float
    cut_in: float
    rated_speed: float
    cut_out: float
    cut_out_found: bool


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
    relative_abs_residual = abs_residual / np.maximum(np.abs(y_true), relative_floor)
    return {
        "corr": _corrcoef(y_true, y_pred),
        "r2": _r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(abs_residual)),
        "mape_like": float(np.mean(relative_abs_residual)),
        "abs_residual_p95": float(np.percentile(abs_residual, 95)),
        "rel_abs_residual_p95": float(np.percentile(relative_abs_residual, 95)),
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
    work_df = work_df[work_df[args.density_col] > 0.0]
    work_df = work_df[work_df[args.wind_speed_col] >= 0.0]
    work_df = work_df[work_df[args.power_col] >= 0.0]
    if args.max_rows and len(work_df) > args.max_rows:
        stride = max(1, len(work_df) // args.max_rows)
        work_df = work_df.iloc[::stride].head(args.max_rows)
    work_df = work_df.reset_index(drop=True)
    density_ratio = np.clip(work_df[args.density_col].to_numpy(dtype=np.float64) / args.reference_density, 0.25, 2.5)
    # Standard density correction: v_corr = v * (rho / rho_ref)^(1/3)
    work_df["_effective_wind_speed"] = work_df[args.wind_speed_col].to_numpy(dtype=np.float64) * np.cbrt(density_ratio)
    return work_df


def _build_bins(df: pd.DataFrame, args, rated_power: float) -> pd.DataFrame:
    v_eff = df["_effective_wind_speed"].to_numpy(dtype=np.float64)
    power = df[args.power_col].to_numpy(dtype=np.float64)
    max_speed = float(np.nanmax(v_eff))
    bin_edges = np.arange(0.0, max_speed + args.bin_width, args.bin_width)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, max_speed + args.bin_width], dtype=np.float64)
    bin_idx = np.digitize(v_eff, bin_edges) - 1

    rows = []
    for idx in range(len(bin_edges) - 1):
        mask = bin_idx == idx
        if not np.any(mask):
            continue
        power_bin = power[mask]
        active_ratio = float(np.mean(power_bin > args.active_power_threshold_ratio * rated_power))
        rows.append({
            "bin_idx": idx,
            "v_left": float(bin_edges[idx]),
            "v_right": float(bin_edges[idx + 1]),
            "v_center": float(0.5 * (bin_edges[idx] + bin_edges[idx + 1])),
            "count": int(mask.sum()),
            "power_q50": float(np.quantile(power_bin, 0.50)),
            "power_q90": float(np.quantile(power_bin, 0.90)),
            "power_q95": float(np.quantile(power_bin, 0.95)),
            "power_mean": float(np.mean(power_bin)),
            "active_ratio": active_ratio,
        })
    bins_df = pd.DataFrame(rows)
    bins_df = bins_df[bins_df["count"] >= args.min_bin_count].reset_index(drop=True)
    return bins_df


def _first_consecutive_match(mask: np.ndarray, run_length: int) -> Optional[int]:
    if run_length <= 1:
        hits = np.where(mask)[0]
        return int(hits[0]) if hits.size else None
    streak = 0
    for idx, flag in enumerate(mask):
        streak = streak + 1 if flag else 0
        if streak >= run_length:
            return idx - run_length + 1
    return None


def _estimate_curve_points(bins_df: pd.DataFrame, args, rated_power: float) -> CurveEstimate:
    if bins_df.empty:
        raise ValueError("No populated wind-speed bins were available for estimation.")

    on_threshold = args.cut_in_power_ratio * rated_power
    plateau_threshold = args.rated_power_ratio * rated_power
    off_threshold = args.cut_out_power_ratio * rated_power

    cut_in_mask = (
        (bins_df["power_q90"].to_numpy() >= on_threshold)
        & (bins_df["active_ratio"].to_numpy() >= args.cut_in_active_ratio)
    )
    cut_in_idx = _first_consecutive_match(cut_in_mask, args.required_consecutive_bins)
    if cut_in_idx is None:
        cut_in_idx = int(np.argmax(bins_df["power_q90"].to_numpy() > 0.0))

    rated_mask = (
        (bins_df["power_q90"].to_numpy() >= plateau_threshold)
        & (bins_df["active_ratio"].to_numpy() >= args.rated_active_ratio)
    )
    rated_mask[:cut_in_idx] = False
    rated_idx = _first_consecutive_match(rated_mask, args.required_consecutive_bins)
    if rated_idx is None:
        rated_idx = int(np.argmax(bins_df["power_q95"].to_numpy()))

    cut_out_mask = (
        (bins_df["power_q90"].to_numpy() <= off_threshold)
        & (bins_df["active_ratio"].to_numpy() <= args.cut_out_active_ratio)
    )
    cut_out_mask[: max(rated_idx + 1, 0)] = False
    cut_out_idx = _first_consecutive_match(cut_out_mask, args.required_consecutive_bins)
    cut_out_found = cut_out_idx is not None
    if not cut_out_found:
        cut_out_speed = float("inf")
    else:
        cut_out_speed = float(bins_df.iloc[cut_out_idx]["v_center"])

    cut_in_speed = float(bins_df.iloc[cut_in_idx]["v_center"])
    rated_speed = float(bins_df.iloc[rated_idx]["v_center"])
    rated_speed = max(rated_speed, cut_in_speed + 1e-6)
    return CurveEstimate(
        rated_power=float(rated_power),
        cut_in=cut_in_speed,
        rated_speed=rated_speed,
        cut_out=cut_out_speed,
        cut_out_found=cut_out_found,
    )


def _piecewise_power_curve(v_eff: np.ndarray, curve: CurveEstimate) -> np.ndarray:
    power = np.zeros_like(v_eff, dtype=np.float64)

    between_mask = (v_eff >= curve.cut_in) & (v_eff < curve.rated_speed)
    if np.any(between_mask):
        v_between = v_eff[between_mask]
        denom = max(curve.rated_speed ** 3 - curve.cut_in ** 3, 1e-8)
        ramp = (v_between ** 3 - curve.cut_in ** 3) / denom
        ramp = np.clip(ramp, 0.0, 1.0)
        power[between_mask] = curve.rated_power * ramp

    rated_mask = v_eff >= curve.rated_speed
    if math.isfinite(curve.cut_out):
        rated_mask &= v_eff < curve.cut_out
        cut_out_mask = v_eff >= curve.cut_out
        power[cut_out_mask] = 0.0
    power[rated_mask] = curve.rated_power
    return power


def main():
    parser = argparse.ArgumentParser(
        description="Estimate cut-in / rated / cut-out / rated power from a wind CSV and compare a piecewise power-curve reconstruction against actual power."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--density-col", type=str, default="density", help="Density column name")
    parser.add_argument("--wind-speed-col", type=str, default="wind_speed", help="Wind speed column name")
    parser.add_argument("--power-col", type=str, default="power", help="Power column name")
    parser.add_argument("--timestamp-col", type=str, default="timeStamp", help="Optional timestamp column for reporting")
    parser.add_argument("--reference-density", type=float, default=REFERENCE_DENSITY, help="Reference density for wind-speed correction")
    parser.add_argument("--max-rows", type=int, default=200000, help="Optional row cap for faster estimation")
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
    parser.add_argument("--relative-ref-floor", type=float, default=1.0, help="Floor for relative residual denominator")
    parser.add_argument("--top-k", type=int, default=20, help="How many worst rows to print")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path to save worst rows")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)
    work_df = _prepare_frame(df, args)
    if work_df.empty:
        raise ValueError("No valid rows remain after filtering.")

    power = work_df[args.power_col].to_numpy(dtype=np.float64)
    rated_power = float(np.quantile(power, args.rated_power_quantile))
    bins_df = _build_bins(work_df, args, rated_power)
    curve = _estimate_curve_points(bins_df, args, rated_power)

    v_eff = work_df["_effective_wind_speed"].to_numpy(dtype=np.float64)
    pred_power = _piecewise_power_curve(v_eff, curve)
    metrics = _evaluate_prediction(power, pred_power, args.relative_ref_floor)

    residual = power - pred_power
    abs_residual = np.abs(residual)
    rel_abs_residual = abs_residual / np.maximum(np.abs(power), args.relative_ref_floor)

    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(work_df)} / {len(df)}")
    print(f"rated_power_est: {curve.rated_power:.6f}")
    print(f"cut_in_est: {curve.cut_in:.6f}")
    print(f"rated_speed_est: {curve.rated_speed:.6f}")
    if curve.cut_out_found:
        print(f"cut_out_est: {curve.cut_out:.6f}")
    else:
        print("cut_out_est: not found in data (using no cut-out in reconstruction)")
    print("")
    print("Reconstruction formula:")
    print("v_eff = wind_speed * (density / reference_density)^(1/3)")
    print("power_hat = 0,                              v_eff < cut_in")
    print("power_hat = rated_power * ((v_eff^3 - cut_in^3) / (rated^3 - cut_in^3)), cut_in <= v_eff < rated")
    print("power_hat = rated_power,                    rated <= v_eff < cut_out")
    print("power_hat = 0,                              v_eff >= cut_out")
    print("")
    print("Fit quality:")
    for key in ["corr", "r2", "rmse", "mae", "mape_like", "abs_residual_p95", "rel_abs_residual_p95"]:
        print(f"{key}: {metrics[key]:.6f}")
    print("")
    for thr in [0.05, 0.10, 0.20]:
        ratio = float((rel_abs_residual <= thr).sum()) / max(len(rel_abs_residual), 1)
        print(f"|relative residual| <= {thr:.0%}: {ratio:.4%}")

    worst_idx = np.argsort(-abs_residual)[:args.top_k]
    worst_df = work_df.iloc[worst_idx].copy()
    worst_df["_effective_wind_speed"] = v_eff[worst_idx]
    worst_df["_power_hat"] = pred_power[worst_idx]
    worst_df["_residual"] = residual[worst_idx]
    worst_df["_abs_residual"] = abs_residual[worst_idx]
    worst_df["_rel_abs_residual"] = rel_abs_residual[worst_idx]

    display_cols = [
        col for col in [args.timestamp_col, args.density_col, args.wind_speed_col, args.power_col]
        if col in worst_df.columns
    ]
    display_cols += [
        "_effective_wind_speed",
        "_power_hat",
        "_residual",
        "_abs_residual",
        "_rel_abs_residual",
    ]
    print("")
    print(f"Top {min(args.top_k, len(worst_df))} worst rows:")
    print(worst_df[display_cols].to_string(index=False))

    if args.output_csv:
        output_path = os.path.abspath(args.output_csv)
        worst_df.to_csv(output_path, index=False)
        print("")
        print(f"Worst rows saved to: {output_path}")


if __name__ == "__main__":
    main()
