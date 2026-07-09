#!/usr/bin/env python3
"""Build a plant power dataset with offline pvlib columns.

Plant columns (required):
  dtime, observe_power, observe_ghi

Weather columns (optional, merged on dtime):
  dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE, ...

Outputs a training-friendly CSV with at least:
  timeStamp, ghi, power, power_pvlib, temp_air, wind_speed, temp_cell, cos_zenith

power          <- observe_power (real label)
power_pvlib    <- pvlib Faiman cell temp + pvwatts_dc prior
ghi            <- observe_ghi

Example
-------
python scripts/solar/build_plant_power_dataset.py \\
  --plant dataset_expample/qingda.xlsx \\
  --weather dataset_expample/weather.xlsx \\
  --latitude 29.919 --longitude 100.641 \\
  --output data/solar/Qingda_plant_power.csv
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


def str2bool(value):
    return str(value).lower() in {'1', 'true', 't', 'yes', 'y'}


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if ext in {'.csv', '.txt'}:
        return pd.read_csv(path)
    # Fall back: try excel then csv.
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_csv(path)


def _normalize_plant_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    lower = {c: str(c).strip() for c in df.columns}
    inv = {v.lower(): k for k, v in lower.items()}

    def pick(*names):
        for n in names:
            if n.lower() in inv:
                return inv[n.lower()]
        return None

    dtime = pick('dtime', 'timestamp', 'timeStamp', 'datetime', 'time')
    power = pick('observe_power', 'power', 'obs_power')
    ghi = pick('observe_ghi', 'ghi', 'obs_ghi')
    if dtime is None or power is None or ghi is None:
        raise ValueError(
            'Plant table must contain dtime/observe_power/observe_ghi (or aliases). Got: {}'.format(
                list(df.columns)
            )
        )
    out = pd.DataFrame({
        'dtime': pd.to_datetime(df[dtime], errors='coerce'),
        'observe_power': pd.to_numeric(df[power], errors='coerce'),
        'observe_ghi': pd.to_numeric(df[ghi], errors='coerce'),
    })
    return out


def _normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    dtime = pick('dtime', 'timestamp', 'timeStamp', 'datetime', 'time')
    if dtime is None:
        raise ValueError('Weather table must contain dtime. Got: {}'.format(list(df.columns)))

    out = pd.DataFrame({'dtime': pd.to_datetime(df[dtime], errors='coerce')})
    mapping = {
        'weather_ghi': ['ghi', 'weather_ghi'],
        'temp_air': ['temp', 'temp_air', 'temperature', 't2m'],
        'wind_speed': ['ws', 'wind_speed', 'windspeed'],
        'wind_dir': ['wd', 'wind_dir', 'winddirection'],
        'precip': ['prec', 'precip', 'precipitation', 'rain'],
        'pwat': ['pwat', 'tpw'],
        'sdwe': ['sdwe', 'swe'],
        'interval': ['interval'],
        'interval_day': ['interval_day'],
        'report_date': ['report_date'],
        'report_time': ['report_time'],
        'update_time': ['update_time'],
        'govern_flag': ['govern_flag', 'flag'],
    }
    for dest, aliases in mapping.items():
        src = pick(*aliases)
        if src is not None:
            if dest in {'weather_ghi', 'temp_air', 'wind_speed', 'wind_dir', 'precip', 'pwat', 'sdwe'}:
                out[dest] = pd.to_numeric(df[src], errors='coerce')
            else:
                out[dest] = df[src]
    return out


def _merge_asof_weather(plant: pd.DataFrame, weather: pd.DataFrame, tolerance: str) -> pd.DataFrame:
    left = plant.sort_values('dtime').dropna(subset=['dtime']).copy()
    right = weather.sort_values('dtime').dropna(subset=['dtime']).copy()
    tol = pd.Timedelta(tolerance)
    merged = pd.merge_asof(
        left,
        right,
        on='dtime',
        direction='nearest',
        tolerance=tol,
    )
    return merged


def _estimate_pdc0(ghi: pd.Series, power: pd.Series, ghi_ref: float, min_ghi: float) -> float:
    mask = (ghi >= min_ghi) & np.isfinite(ghi) & np.isfinite(power) & (power >= 0)
    if not mask.any():
        return 1.0
    ratio = power[mask] / (ghi[mask].clip(lower=1e-6) / float(ghi_ref))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 1.0
    return float(np.nanmedian(ratio.to_numpy()))


def _compute_cos_zenith(dtime: pd.Series, latitude: float, longitude: float, timezone: str) -> np.ndarray:
    import pvlib

    ts = pd.DatetimeIndex(pd.to_datetime(dtime, errors='coerce'))
    if ts.tz is None:
        ts = ts.tz_localize(timezone)
    else:
        ts = ts.tz_convert(timezone)
    solpos = pvlib.solarposition.get_solarposition(ts, latitude=latitude, longitude=longitude)
    cos_z = np.cos(np.deg2rad(solpos['zenith'].to_numpy(dtype=np.float64)))
    return np.clip(cos_z, 0.0, None)


def _erbs_dni_dhi(
    ghi: np.ndarray,
    cos_zenith: np.ndarray,
    latitude: float,
    longitude: float,
    dtime: pd.Series,
    timezone: str,
):
    """Estimate DNI/DHI from GHI via pvlib Erbs separation so enc_in=3 pipelines still work."""
    import pvlib

    ts = pd.DatetimeIndex(pd.to_datetime(dtime, errors='coerce'))
    if ts.tz is None:
        ts = ts.tz_localize(timezone)
    else:
        ts = ts.tz_convert(timezone)

    # Extraterrestrial / clearness helpers used by erbs.
    dni_extra = pvlib.irradiance.get_extra_radiation(ts)
    zenith_deg = np.rad2deg(np.arccos(np.clip(cos_zenith, 0.0, 1.0)))
    erbs = pvlib.irradiance.erbs(
        ghi=np.clip(ghi, 0.0, None),
        zenith=zenith_deg,
        datetime_or_doy=ts,
        dni_extra=dni_extra,
    )
    dni = np.asarray(erbs['dni'], dtype=np.float64)
    dhi = np.asarray(erbs['dhi'], dtype=np.float64)
    dni = np.where(np.isfinite(dni), np.clip(dni, 0.0, None), 0.0)
    dhi = np.where(np.isfinite(dhi), np.clip(dhi, 0.0, None), 0.0)
    return dni, dhi


def _compute_temp_cell(
    ghi: np.ndarray,
    temp_air: Optional[np.ndarray],
    wind_speed: Optional[np.ndarray],
    method: str,
    temp_amb_default: float,
    noct: float,
    wind_default: float,
) -> np.ndarray:
    import pvlib

    ghi = np.asarray(ghi, dtype=np.float64)
    if temp_air is None:
        temp_air_arr = np.full_like(ghi, float(temp_amb_default), dtype=np.float64)
    else:
        temp_air_arr = np.asarray(temp_air, dtype=np.float64)
        temp_air_arr = np.where(np.isfinite(temp_air_arr), temp_air_arr, float(temp_amb_default))

    if wind_speed is None:
        wind_arr = np.full_like(ghi, float(wind_default), dtype=np.float64)
    else:
        wind_arr = np.asarray(wind_speed, dtype=np.float64)
        wind_arr = np.where(np.isfinite(wind_arr), wind_arr, float(wind_default))
        wind_arr = np.clip(wind_arr, 0.0, None)

    poa = np.clip(ghi, 0.0, None)
    method = str(method).lower()
    if method == 'faiman':
        # pvlib.temperature.faiman(poa_global, temp_air, wind_speed)
        return np.asarray(
            pvlib.temperature.faiman(poa, temp_air_arr, wind_arr),
            dtype=np.float64,
        )
    if method == 'noct':
        # Classic NOCT model, no wind: Tc = Ta + G/800*(NOCT-20)
        return temp_air_arr + poa * (float(noct) - 20.0) / 800.0
    raise ValueError("temp-cell-method must be 'faiman' or 'noct', got {!r}".format(method))


def _pvwatts_dc(ghi, temp_cell, pdc0, gamma_pdc, ghi_ref, temp_ref) -> np.ndarray:
    import pvlib

    g_eff = np.clip(np.asarray(ghi, dtype=np.float64), 0.0, None)
    # Prefer library implementation when available.
    try:
        return np.asarray(
            pvlib.pvsystem.pvwatts_dc(
                g_poa_effective=g_eff,
                temp_cell=np.asarray(temp_cell, dtype=np.float64),
                pdc0=float(pdc0),
                gamma_pdc=float(gamma_pdc),
                temp_ref=float(temp_ref),
            ),
            dtype=np.float64,
        )
    except TypeError:
        # Older signature variants.
        return np.asarray(
            pvlib.pvsystem.pvwatts_dc(
                g_eff,
                np.asarray(temp_cell, dtype=np.float64),
                float(pdc0),
                float(gamma_pdc),
            ),
            dtype=np.float64,
        )


def build_plant_power_frame(args) -> pd.DataFrame:
    plant = _normalize_plant_columns(_read_table(args.plant))
    if args.weather:
        weather = _normalize_weather_columns(_read_table(args.weather))
        df = _merge_asof_weather(plant, weather, tolerance=args.merge_tolerance)
    else:
        df = plant.copy()

    df = df.dropna(subset=['dtime']).sort_values('dtime').reset_index(drop=True)
    ghi = df['observe_ghi'].to_numpy(dtype=np.float64)
    power = df['observe_power'].to_numpy(dtype=np.float64)

    if args.pdc0 is None or args.pdc0 <= 0:
        pdc0 = _estimate_pdc0(
            df['observe_ghi'],
            df['observe_power'],
            ghi_ref=args.ghi_ref,
            min_ghi=args.pdc0_min_ghi,
        )
        print('auto pdc0 (median daytime ratio) = {:.6f}'.format(pdc0))
    else:
        pdc0 = float(args.pdc0)
        print('using pdc0 = {:.6f}'.format(pdc0))

    temp_air = df['temp_air'].to_numpy(dtype=np.float64) if 'temp_air' in df.columns else None
    wind_speed = df['wind_speed'].to_numpy(dtype=np.float64) if 'wind_speed' in df.columns else None

    temp_cell = _compute_temp_cell(
        ghi=ghi,
        temp_air=temp_air,
        wind_speed=wind_speed,
        method=args.temp_cell_method,
        temp_amb_default=args.temp_amb,
        noct=args.noct,
        wind_default=args.wind_default,
    )
    power_pvlib = _pvwatts_dc(
        ghi=ghi,
        temp_cell=temp_cell,
        pdc0=pdc0,
        gamma_pdc=args.gamma_pdc,
        ghi_ref=args.ghi_ref,
        temp_ref=args.temp_ref,
    )
    power_pvlib = np.clip(power_pvlib, 0.0, None)

    cos_zenith = _compute_cos_zenith(
        df['dtime'],
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
    )

    dni = None
    dhi = None
    if args.add_erbs_components:
        dni, dhi = _erbs_dni_dhi(
            ghi=ghi,
            cos_zenith=cos_zenith,
            latitude=args.latitude,
            longitude=args.longitude,
            dtime=df['dtime'],
            timezone=args.timezone,
        )

    out_cols = {
        'timeStamp': df['dtime'].dt.strftime('%Y-%m-%dT%H:%M:%S'),
        'ghi': df['observe_ghi'].to_numpy(dtype=np.float64),
    }
    if dni is not None and dhi is not None:
        out_cols['dni'] = dni
        out_cols['dhi'] = dhi
    out_cols.update({
        'power': df['observe_power'].to_numpy(dtype=np.float64),
        'power_pvlib': power_pvlib,
        'temp_cell': temp_cell,
        'cos_zenith': cos_zenith,
        'pdc0': pdc0,
    })
    out = pd.DataFrame(out_cols)
    if temp_air is not None:
        out['temp_air'] = temp_air
    else:
        out['temp_air'] = float(args.temp_amb)
    if wind_speed is not None:
        out['wind_speed'] = wind_speed
    else:
        out['wind_speed'] = float(args.wind_default)

    # Optional pass-through weather extras for residual learning / analysis.
    for col in ['weather_ghi', 'wind_dir', 'precip', 'pwat', 'sdwe',
                'interval', 'interval_day', 'report_date', 'report_time',
                'update_time', 'govern_flag']:
        if col in df.columns:
            out[col] = df[col].to_numpy()

    if args.include_linear_prior:
        linear = np.clip(ghi, 0.0, None) / float(args.ghi_ref) * pdc0
        out['power_linear'] = linear

    # Quality filter
    if args.drop_negative_power:
        out = out[out['power'].fillna(0) >= -args.power_atol].copy()
    if 'govern_flag' in out.columns and args.govern_ok is not None:
        out = out[out['govern_flag'].astype(str) == str(args.govern_ok)].copy()

    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Build plant power CSV with offline pvlib columns.')
    parser.add_argument('--plant', required=True, help='plant table (xlsx/csv) with dtime,observe_power,observe_ghi')
    parser.add_argument('--weather', default='', help='optional weather/forecast table to merge')
    parser.add_argument('--output', default='data/solar/plant_power.csv', help='output CSV path')
    parser.add_argument('--latitude', type=float, default=29.919, help='site latitude')
    parser.add_argument('--longitude', type=float, default=100.641, help='site longitude')
    parser.add_argument('--timezone', default='Asia/Shanghai', help='timezone for naive timestamps')
    parser.add_argument('--merge-tolerance', default='30min', help='asof merge tolerance for weather')
    parser.add_argument('--pdc0', type=float, default=None, help='DC capacity; default=auto from data')
    parser.add_argument('--pdc0-min-ghi', type=float, default=200.0, help='min GHI used for pdc0 auto estimate')
    parser.add_argument('--ghi-ref', type=float, default=1000.0, help='reference irradiance')
    parser.add_argument('--gamma-pdc', type=float, default=-0.004, help='PVWatts temperature coefficient')
    parser.add_argument('--temp-ref', type=float, default=25.0, help='PVWatts reference temperature')
    parser.add_argument('--temp-amb', type=float, default=20.0, help='fallback ambient temperature')
    parser.add_argument('--wind-default', type=float, default=1.0, help='fallback wind speed (m/s)')
    parser.add_argument('--noct', type=float, default=45.0, help='NOCT used when temp-cell-method=noct')
    parser.add_argument('--temp-cell-method', choices=['faiman', 'noct'], default='faiman',
                        help='cell temperature model')
    parser.add_argument('--include-linear-prior', type=str2bool, default=True,
                        help='also write power_linear = pdc0*GHI/ghi_ref')
    parser.add_argument('--add-erbs-components', type=str2bool, default=True,
                        help='estimate dni/dhi from GHI via Erbs so SolarPhysics enc_in=3 still works')
    parser.add_argument('--drop-negative-power', type=str2bool, default=True)
    parser.add_argument('--power-atol', type=float, default=1e-3)
    parser.add_argument('--govern-ok', default=None,
                        help='if set, keep only rows with govern_flag equal to this value')
    args = parser.parse_args()

    out = build_plant_power_frame(args)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out.to_csv(args.output, index=False)

    # Summary diagnostics
    power = out['power'].to_numpy(dtype=np.float64)
    prior = out['power_pvlib'].to_numpy(dtype=np.float64)
    mask = np.isfinite(power) & np.isfinite(prior)
    mae = float(np.mean(np.abs(power[mask] - prior[mask]))) if mask.any() else float('nan')
    corr = float(np.corrcoef(power[mask], prior[mask])[0, 1]) if mask.sum() > 2 else float('nan')
    print(
        'wrote {} rows -> {} | power min/max=({:.4f},{:.4f}) '
        'pvlib min/max=({:.4f},{:.4f}) | MAE(pvlib,obs)={:.4f} corr={:.4f}'.format(
            len(out),
            args.output,
            float(np.nanmin(power)) if len(power) else float('nan'),
            float(np.nanmax(power)) if len(power) else float('nan'),
            float(np.nanmin(prior)) if len(prior) else float('nan'),
            float(np.nanmax(prior)) if len(prior) else float('nan'),
            mae,
            corr,
        )
    )


if __name__ == '__main__':
    main()
