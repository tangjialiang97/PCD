import argparse
import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


REFERENCE_DENSITY = 1.225
GAS_CONSTANT_DRY_AIR = 287.05


def _summarize_metric(values: np.ndarray, name: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{name}_count": 0,
            f"{name}_mean": math.nan,
            f"{name}_median": math.nan,
            f"{name}_p95": math.nan,
            f"{name}_max": math.nan,
            f"{name}_rmse": math.nan,
        }
    return {
        f"{name}_count": int(values.size),
        f"{name}_mean": float(np.mean(values)),
        f"{name}_median": float(np.median(values)),
        f"{name}_p95": float(np.percentile(values, 95)),
        f"{name}_max": float(np.max(values)),
        f"{name}_rmse": float(np.sqrt(np.mean(values ** 2))),
    }


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


def _prepare_frame(df: pd.DataFrame, args) -> pd.DataFrame:
    required_cols = [
        args.density_col,
        args.power_col,
        args.pressure_col,
        args.temperature_col,
        args.wind_direction_col,
        args.wind_speed_col,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    for col in required_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    work_df = work_df.dropna(subset=required_cols)
    work_df = work_df[work_df[args.wind_speed_col] > 0.0]
    work_df = work_df[work_df[args.power_col] >= 0.0]
    work_df = work_df[work_df[args.temperature_col] > args.temperature_floor]
    if args.max_rows and len(work_df) > args.max_rows:
        work_df = work_df.iloc[:: max(1, len(work_df) // args.max_rows)].head(args.max_rows)
    return work_df.reset_index(drop=True)


def _build_terms(df: pd.DataFrame, args, alpha: float, exponent: float, phase_deg: float, eta_min: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    density = df[args.density_col].to_numpy(dtype=np.float64)
    pressure = df[args.pressure_col].to_numpy(dtype=np.float64)
    temperature = df[args.temperature_col].to_numpy(dtype=np.float64)
    wind_direction = df[args.wind_direction_col].to_numpy(dtype=np.float64)
    wind_speed = df[args.wind_speed_col].to_numpy(dtype=np.float64)

    ideal_density = pressure / (GAS_CONSTANT_DRY_AIR * np.maximum(temperature, args.temperature_floor))
    mixed_density = alpha * density + (1.0 - alpha) * ideal_density
    density_ratio = np.clip(mixed_density / args.reference_density, 0.25, 2.5)

    direction_rad = np.deg2rad(np.mod(wind_direction, 360.0))
    phase_rad = np.deg2rad(phase_deg)
    direction_alignment = 0.5 * (np.cos(direction_rad - phase_rad) + 1.0)
    direction_efficiency = eta_min + (1.0 - eta_min) * direction_alignment

    env_density_ratio = np.clip(ideal_density / args.reference_density, 0.25, 2.5)
    environment_efficiency = np.sqrt(env_density_ratio)

    speed_term = np.power(np.maximum(wind_speed, args.speed_floor), exponent)
    surrogate_feature = density_ratio * direction_efficiency * environment_efficiency * speed_term
    aux = {
        "ideal_density": ideal_density,
        "mixed_density": mixed_density,
        "density_ratio": density_ratio,
        "direction_efficiency": direction_efficiency,
        "environment_efficiency": environment_efficiency,
        "speed_term": speed_term,
    }
    return surrogate_feature, aux


def _fit_positive_scale(feature: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.dot(feature, feature))
    if denom <= 0.0:
        return 0.0
    return max(0.0, float(np.dot(feature, target) / denom))


def _evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, relative_floor: float) -> Dict[str, float]:
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    relative_abs_residual = abs_residual / np.maximum(np.abs(y_true), relative_floor)
    summary = {
        "corr": _corrcoef(y_true, y_pred),
        "r2": _r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(abs_residual)),
        "mape_like": float(np.mean(relative_abs_residual)),
    }
    summary.update(_summarize_metric(abs_residual, "abs_residual"))
    summary.update(_summarize_metric(relative_abs_residual, "rel_abs_residual"))
    return summary


def _grid_search_formula(df: pd.DataFrame, args) -> Tuple[Dict[str, float], np.ndarray, Dict[str, np.ndarray]]:
    power = df[args.power_col].to_numpy(dtype=np.float64)
    best = None
    best_pred = None
    best_aux = None

    for alpha in np.linspace(0.0, 1.0, args.alpha_steps):
        for exponent in np.linspace(args.exp_min, args.exp_max, args.exp_steps):
            for phase_deg in np.linspace(0.0, 330.0, args.phase_steps):
                for eta_min in np.linspace(args.eta_min_min, args.eta_min_max, args.eta_min_steps):
                    feature, aux = _build_terms(df, args, alpha, exponent, phase_deg, eta_min)
                    scale = _fit_positive_scale(feature, power)
                    pred = scale * feature
                    metrics = _evaluate_prediction(power, pred, args.relative_ref_floor)
                    candidate = {
                        "alpha": float(alpha),
                        "exponent": float(exponent),
                        "phase_deg": float(phase_deg),
                        "eta_min": float(eta_min),
                        "scale": float(scale),
                        **metrics,
                    }
                    if best is None or candidate["rmse"] < best["rmse"]:
                        best = candidate
                        best_pred = pred
                        best_aux = aux
    return best, best_pred, best_aux


def _simple_density_cubic(df: pd.DataFrame, args) -> Tuple[Dict[str, float], np.ndarray]:
    density = df[args.density_col].to_numpy(dtype=np.float64)
    wind_speed = df[args.wind_speed_col].to_numpy(dtype=np.float64)
    power = df[args.power_col].to_numpy(dtype=np.float64)
    feature = np.clip(density / args.reference_density, 0.25, 2.5) * np.power(np.maximum(wind_speed, args.speed_floor), 3.0)
    scale = _fit_positive_scale(feature, power)
    pred = scale * feature
    metrics = _evaluate_prediction(power, pred, args.relative_ref_floor)
    metrics["scale"] = scale
    return metrics, pred


def main():
    parser = argparse.ArgumentParser(
        description="Validate whether wind power approximately follows density correction * direction efficiency * environment efficiency * wind-speed power law."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--density-col", type=str, default="density", help="Density column name")
    parser.add_argument("--power-col", type=str, default="power", help="Power column name")
    parser.add_argument("--pressure-col", type=str, default="pressure", help="Pressure column name")
    parser.add_argument("--temperature-col", type=str, default="temperature", help="Temperature column name")
    parser.add_argument("--wind-direction-col", type=str, default="wind_direction", help="Wind direction column name")
    parser.add_argument("--wind-speed-col", type=str, default="wind_speed", help="Wind speed column name")
    parser.add_argument("--timestamp-col", type=str, default="timeStamp", help="Optional timestamp column for reporting")
    parser.add_argument("--reference-density", type=float, default=REFERENCE_DENSITY, help="Reference air density")
    parser.add_argument("--temperature-floor", type=float, default=150.0, help="Lower bound to keep ideal-density computation stable")
    parser.add_argument("--speed-floor", type=float, default=1e-3, help="Lower bound for wind speed in the power law")
    parser.add_argument("--relative-ref-floor", type=float, default=1.0, help="Floor for relative residual denominator")
    parser.add_argument("--max-rows", type=int, default=200000, help="Optional cap on rows used for fitting")
    parser.add_argument("--alpha-steps", type=int, default=11, help="Grid steps for density mixing alpha")
    parser.add_argument("--exp-min", type=float, default=1.5, help="Minimum wind-speed exponent")
    parser.add_argument("--exp-max", type=float, default=4.0, help="Maximum wind-speed exponent")
    parser.add_argument("--exp-steps", type=int, default=11, help="Grid steps for wind-speed exponent")
    parser.add_argument("--phase-steps", type=int, default=12, help="Grid steps for wind-direction phase in [0, 330]")
    parser.add_argument("--eta-min-min", type=float, default=0.0, help="Minimum eta_min in direction efficiency")
    parser.add_argument("--eta-min-max", type=float, default=0.9, help="Maximum eta_min in direction efficiency")
    parser.add_argument("--eta-min-steps", type=int, default=10, help="Grid steps for eta_min")
    parser.add_argument("--top-k", type=int, default=20, help="How many worst rows to print")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path to save worst rows")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)
    work_df = _prepare_frame(df, args)
    if work_df.empty:
        raise ValueError("No valid rows remain after filtering.")

    best_formula, best_pred, best_aux = _grid_search_formula(work_df, args)
    cubic_metrics, cubic_pred = _simple_density_cubic(work_df, args)
    power = work_df[args.power_col].to_numpy(dtype=np.float64)
    residual = power - best_pred
    abs_residual = np.abs(residual)
    rel_abs_residual = abs_residual / np.maximum(np.abs(power), args.relative_ref_floor)

    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(work_df)} / {len(df)}")
    print(
        "Columns: "
        f"density=`{args.density_col}`, pressure=`{args.pressure_col}`, temperature=`{args.temperature_col}`, "
        f"wind_direction=`{args.wind_direction_col}`, wind_speed=`{args.wind_speed_col}`, power=`{args.power_col}`"
    )
    print("")
    print("Tested surrogate formula:")
    print("power_hat = scale * density_ratio * direction_efficiency * environment_efficiency * wind_speed^exponent")
    print("density_ratio = clip((alpha * density + (1-alpha) * pressure / (R * temperature)) / reference_density, 0.25, 2.5)")
    print("direction_efficiency = eta_min + (1-eta_min) * 0.5 * (cos(wind_direction - phase) + 1)")
    print("environment_efficiency = sqrt(clip((pressure / (R * temperature)) / reference_density, 0.25, 2.5))")
    print("")
    print("Best fitted surrogate parameters:")
    print(f"alpha: {best_formula['alpha']:.6f}")
    print(f"exponent: {best_formula['exponent']:.6f}")
    print(f"phase_deg: {best_formula['phase_deg']:.6f}")
    print(f"eta_min: {best_formula['eta_min']:.6f}")
    print(f"scale: {best_formula['scale']:.6f}")
    print("")
    print("Best surrogate fit quality:")
    for key in ["corr", "r2", "rmse", "mae", "mape_like", "abs_residual_p95", "rel_abs_residual_p95"]:
        print(f"{key}: {best_formula[key]:.6f}")
    print("")
    print("Reference baseline (density_ratio * wind_speed^3):")
    for key in ["corr", "r2", "rmse", "mae", "mape_like"]:
        print(f"{key}: {cubic_metrics[key]:.6f}")

    print("")
    print("Pass ratios for best surrogate:")
    for thr in [0.05, 0.10, 0.20]:
        ratio = float((rel_abs_residual <= thr).sum()) / max(len(rel_abs_residual), 1)
        print(f"|relative residual| <= {thr:.0%}: {ratio:.4%}")

    worst_idx = np.argsort(-abs_residual)[:args.top_k]
    worst_df = work_df.iloc[worst_idx].copy()
    worst_df["_power_hat"] = best_pred[worst_idx]
    worst_df["_residual"] = residual[worst_idx]
    worst_df["_abs_residual"] = abs_residual[worst_idx]
    worst_df["_rel_abs_residual"] = rel_abs_residual[worst_idx]
    worst_df["_ideal_density"] = best_aux["ideal_density"][worst_idx]
    worst_df["_mixed_density"] = best_aux["mixed_density"][worst_idx]
    worst_df["_density_ratio"] = best_aux["density_ratio"][worst_idx]
    worst_df["_direction_efficiency"] = best_aux["direction_efficiency"][worst_idx]
    worst_df["_environment_efficiency"] = best_aux["environment_efficiency"][worst_idx]
    worst_df["_speed_term"] = best_aux["speed_term"][worst_idx]

    display_cols = [
        col for col in [args.timestamp_col, args.density_col, args.pressure_col, args.temperature_col, args.wind_direction_col, args.wind_speed_col, args.power_col]
        if col in worst_df.columns
    ]
    display_cols += [
        "_power_hat",
        "_residual",
        "_abs_residual",
        "_rel_abs_residual",
        "_ideal_density",
        "_mixed_density",
        "_density_ratio",
        "_direction_efficiency",
        "_environment_efficiency",
        "_speed_term",
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
