import argparse
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib


DEFAULT_LATITUDE = 38.677
DEFAULT_LONGITUDE = -121.15
DEFAULT_TIMEZONE = "UTC"


def _find_column(df: pd.DataFrame, candidates) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        matched = lower_to_original.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _resolve_required_columns(df: pd.DataFrame,
                              ghi_col: Optional[str],
                              dni_col: Optional[str],
                              dhi_col: Optional[str]) -> Tuple[str, str, str]:
    ghi = ghi_col or _find_column(df, ["ghi", "GHI"])
    dni = dni_col or _find_column(df, ["dni", "DNI"])
    dhi = dhi_col or _find_column(df, ["dhi", "DHI"])

    missing = []
    if ghi is None:
        missing.append("ghi")
    if dni is None:
        missing.append("dni")
    if dhi is None:
        missing.append("dhi")
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Use --ghi-col/--dni-col/--dhi-col to specify them.")
    return ghi, dni, dhi


def _compute_solar_zenith(ts_series: pd.Series, args) -> np.ndarray:
    """
    与 data_provider/data_load_other.py 保持一致：
    先根据 timestamp 用 pvlib 计算太阳天顶角 zenith（单位：度）。
    """
    ts = pd.to_datetime(ts_series, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Invalid timestamps found in `{args.timestamp_col}` while computing solar zenith.")

    dt_index = pd.DatetimeIndex(ts)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(args.solar_timezone)
    else:
        dt_index = dt_index.tz_convert(args.solar_timezone)

    solpos = pvlib.solarposition.get_solarposition(
        dt_index,
        latitude=args.solar_latitude,
        longitude=args.solar_longitude,
    )
    return solpos["zenith"].to_numpy(dtype=np.float64)


def _build_solar_geometry(df: pd.DataFrame, args) -> Tuple[np.ndarray, np.ndarray, str]:
    if args.cos_zenith_col:
        if args.cos_zenith_col not in df.columns:
            raise ValueError(f"--cos-zenith-col '{args.cos_zenith_col}' not found in CSV.")
        cos_zenith = pd.to_numeric(df[args.cos_zenith_col], errors="coerce").to_numpy(dtype=np.float64)
        zenith_deg = np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
        return zenith_deg, cos_zenith, f"direct cos column `{args.cos_zenith_col}`"

    if args.zenith_col:
        if args.zenith_col not in df.columns:
            raise ValueError(f"--zenith-col '{args.zenith_col}' not found in CSV.")
        zenith_deg = pd.to_numeric(df[args.zenith_col], errors="coerce").to_numpy(dtype=np.float64)
        # 与 exp_long_term_forecasting.py 保持一致：由 zenith 直接计算 cos，并对夜间裁到 >= 0。
        cos_zenith = np.cos(np.deg2rad(zenith_deg))
        return zenith_deg, cos_zenith, f"direct zenith column `{args.zenith_col}`"

    timestamp_col = args.timestamp_col
    if timestamp_col not in df.columns:
        raise ValueError(
            "Timestamp column is unavailable. Provide --timestamp-col, or explicitly set "
            "--zenith-col / --cos-zenith-col."
        )
    zenith_deg = _compute_solar_zenith(df[timestamp_col], args)
    cos_zenith = np.cos(np.deg2rad(zenith_deg))
    return zenith_deg, cos_zenith, (
        f"computed zenith via pvlib from `{timestamp_col}` "
        f"(lat={args.solar_latitude}, lon={args.solar_longitude}, tz={args.solar_timezone})"
    )


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


def main():
    parser = argparse.ArgumentParser(description="Check solar three-channel closure: GHI = DNI * cos(Z) + DHI")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--ghi-col", type=str, default=None, help="GHI column name")
    parser.add_argument("--dni-col", type=str, default=None, help="DNI column name")
    parser.add_argument("--dhi-col", type=str, default=None, help="DHI column name")
    parser.add_argument("--zenith-col", type=str, default=None, help="Optional zenith angle column name in degrees")
    parser.add_argument("--cos-zenith-col", type=str, default=None, help="Optional cos(zenith) column name")
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column used to directly compute zenith/cos_zenith with pvlib by default",
    )
    parser.add_argument("--solar-latitude", type=float, default=DEFAULT_LATITUDE, help="Latitude used by pvlib")
    parser.add_argument("--solar-longitude", type=float, default=DEFAULT_LONGITUDE, help="Longitude used by pvlib")
    parser.add_argument("--solar-timezone", type=str, default=DEFAULT_TIMEZONE, help="Timezone used by pvlib")
    parser.add_argument("--day-cos-threshold", type=float, default=0.0, help="Only rows with cos(Z) above this are counted as daytime")
    parser.add_argument("--relative-ref-floor", type=float, default=100.0, help="Floor for relative residual denominator")
    parser.add_argument("--top-k", type=int, default=20, help="How many worst rows to print/save")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path to save worst rows")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)

    ghi_col, dni_col, dhi_col = _resolve_required_columns(df, args.ghi_col, args.dni_col, args.dhi_col)
    zenith_deg, cos_zenith_raw, cos_source = _build_solar_geometry(df, args)
    # 与训练代码保持一致：closure 和 daytime mask 都基于 clamp(min=0.0) 后的 cos_zenith。
    cos_zenith = np.maximum(cos_zenith_raw, 0.0)

    ghi = pd.to_numeric(df[ghi_col], errors="coerce").to_numpy(dtype=np.float64)
    dni = pd.to_numeric(df[dni_col], errors="coerce").to_numpy(dtype=np.float64)
    dhi = pd.to_numeric(df[dhi_col], errors="coerce").to_numpy(dtype=np.float64)

    valid_mask = (
        np.isfinite(ghi)
        & np.isfinite(dni)
        & np.isfinite(dhi)
        & np.isfinite(zenith_deg)
        & np.isfinite(cos_zenith)
    )
    day_mask = valid_mask & (cos_zenith > float(args.day_cos_threshold))

    closure_est = dni * cos_zenith + dhi
    residual = ghi - closure_est
    abs_residual = np.abs(residual)
    relative_den = np.maximum(np.abs(ghi), float(args.relative_ref_floor))
    relative_abs_residual = abs_residual / relative_den

    summary = {}
    summary.update(_summarize_metric(abs_residual[valid_mask], "all_abs_residual"))
    summary.update(_summarize_metric(abs_residual[day_mask], "day_abs_residual"))
    summary.update(_summarize_metric(relative_abs_residual[day_mask], "day_rel_abs_residual"))

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: ghi=`{ghi_col}`, dni=`{dni_col}`, dhi=`{dhi_col}`")
    print(f"solar geometry source: {cos_source}")
    print(f"Valid rows: {int(valid_mask.sum())} / {len(df)}")
    print(f"Daytime rows (cos(Z) > {args.day_cos_threshold}): {int(day_mask.sum())} / {len(df)}")
    print("")
    print("Closure check: GHI ?= DNI * cos(Z) + DHI")
    for key, value in summary.items():
        if isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    thresholds_abs = [1.0, 5.0, 10.0, 20.0, 50.0]
    thresholds_rel = [0.01, 0.05, 0.10, 0.20]
    print("")
    print("Daytime pass ratios:")
    day_count = max(int(day_mask.sum()), 1)
    for thr in thresholds_abs:
        ratio = float((abs_residual[day_mask] <= thr).sum()) / day_count
        print(f"|residual| <= {thr:.1f}: {ratio:.4%}")
    for thr in thresholds_rel:
        ratio = float((relative_abs_residual[day_mask] <= thr).sum()) / day_count
        print(f"|relative residual| <= {thr:.0%}: {ratio:.4%}")

    if day_mask.any():
        worst_idx = np.argsort(-abs_residual[day_mask])[:args.top_k]
        day_indices = np.where(day_mask)[0]
        selected_indices = day_indices[worst_idx]
        worst_df = df.iloc[selected_indices].copy()
        worst_df["_zenith_deg"] = zenith_deg[selected_indices]
        worst_df["_cos_zenith"] = cos_zenith[selected_indices]
        worst_df["_closure_est"] = closure_est[selected_indices]
        worst_df["_residual"] = residual[selected_indices]
        worst_df["_abs_residual"] = abs_residual[selected_indices]
        worst_df["_rel_abs_residual"] = relative_abs_residual[selected_indices]

        print("")
        print(f"Top {min(args.top_k, len(worst_df))} worst daytime rows:")
        display_cols = [col for col in [args.timestamp_col, ghi_col, dni_col, dhi_col] if col in worst_df.columns]
        display_cols += ["_zenith_deg", "_cos_zenith", "_closure_est", "_residual", "_abs_residual", "_rel_abs_residual"]
        print(worst_df[display_cols].to_string(index=False))

        if args.output_csv:
            output_path = os.path.abspath(args.output_csv)
            worst_df.to_csv(output_path, index=False)
            print("")
            print(f"Worst rows saved to: {output_path}")


if __name__ == "__main__":
    main()
