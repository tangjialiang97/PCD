#!/usr/bin/env python3
"""Plot Solar geometry comparison across baseline, baseline+Soft, and baseline+PCD.

This script expects per-run payloads saved by run_test.py via:
  --save_solar_geometry_payload true

Each payload should contain:
  - pred: [N, H, 3]
  - true: [N, H, 3]
  - cos_zenith: [N, H]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _resolve_payload(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    candidate = path / "analysis" / "solar_geometry_payload.npz"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find solar geometry payload under: {path_like}")


def _load_payload(path_like: str) -> Dict[str, np.ndarray]:
    payload_path = _resolve_payload(path_like)
    data = np.load(payload_path)
    required = ("pred", "true", "cos_zenith")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing arrays in {payload_path}: {missing}")
    return {
        "path": np.asarray(str(payload_path)),
        "pred": np.asarray(data["pred"]),
        "true": np.asarray(data["true"]),
        "cos_zenith": np.asarray(data["cos_zenith"]),
    }


def _validate_shapes(baseline: Dict[str, np.ndarray], soft: Dict[str, np.ndarray], pcd: Dict[str, np.ndarray]) -> None:
    shape_ref = baseline["pred"].shape
    cos_ref = baseline["cos_zenith"].shape
    true_ref = baseline["true"].shape
    for name, payload in (("soft", soft), ("pcd", pcd)):
        if payload["pred"].shape != shape_ref:
            raise ValueError(f"{name} pred shape {payload['pred'].shape} != baseline {shape_ref}")
        if payload["true"].shape != true_ref:
            raise ValueError(f"{name} true shape {payload['true'].shape} != baseline {true_ref}")
        if payload["cos_zenith"].shape != cos_ref:
            raise ValueError(f"{name} cos_zenith shape {payload['cos_zenith'].shape} != baseline {cos_ref}")


def _select_slice_indices(
    baseline_pred: np.ndarray,
    soft_pred: np.ndarray,
    pcd_pred: np.ndarray,
    cos_zenith: np.ndarray,
    target_cos: float | None,
    slice_tol: float,
    min_points: int,
    max_points: int,
    night_cos_threshold: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    baseline_flat = baseline_pred.reshape(-1, baseline_pred.shape[-1])
    soft_flat = soft_pred.reshape(-1, soft_pred.shape[-1])
    pcd_flat = pcd_pred.reshape(-1, pcd_pred.shape[-1])
    cos_flat = cos_zenith.reshape(-1)

    valid_mask = np.isfinite(cos_flat)
    valid_mask &= cos_flat > night_cos_threshold
    valid_mask &= np.all(np.isfinite(baseline_flat[:, :3]), axis=1)
    valid_mask &= np.all(np.isfinite(soft_flat[:, :3]), axis=1)
    valid_mask &= np.all(np.isfinite(pcd_flat[:, :3]), axis=1)

    if not np.any(valid_mask):
        raise ValueError("No valid daytime samples found across all three payloads.")

    cos_valid = cos_flat[valid_mask]
    if target_cos is None:
        target_cos = float(np.median(cos_valid))
    else:
        target_cos = float(np.clip(target_cos, 0.0, 1.0))

    delta = np.abs(cos_valid - target_cos)
    selected_idx = np.where(delta <= slice_tol)[0]
    selection_mode = "tolerance"
    if selected_idx.size < min(min_points, cos_valid.size):
        keep = min(min_points, cos_valid.size)
        selected_idx = np.argsort(delta)[:keep]
        selection_mode = "nearest"
    if selected_idx.size > max_points:
        ranked = selected_idx[np.argsort(delta[selected_idx])]
        selected_idx = ranked[:max_points]
        selection_mode = f"{selection_mode}+cap"

    valid_indices = np.where(valid_mask)[0]
    chosen_flat_indices = valid_indices[selected_idx]
    plane_cos = float(np.median(cos_flat[chosen_flat_indices]))
    return chosen_flat_indices, {
        "target_cos": target_cos,
        "plane_cos": plane_cos,
        "plane_zenith_deg": float(np.rad2deg(np.arccos(np.clip(plane_cos, 0.0, 1.0)))),
        "selection_mode": selection_mode,
        "point_count": int(chosen_flat_indices.size),
        "slice_tol": float(slice_tol),
    }


def _plane_distance(points: np.ndarray, plane_cos: float) -> np.ndarray:
    residual = points[:, 0] - (points[:, 1] * plane_cos + points[:, 2])
    plane_norm = math.sqrt(plane_cos ** 2 + 2.0)
    return np.abs(residual) / plane_norm


def _closure_residual(points: np.ndarray, cos_zenith: np.ndarray) -> np.ndarray:
    return points[:, 0] - (points[:, 1] * cos_zenith + points[:, 2])


def _axis_limits(values: np.ndarray, low_q: float = 0.02, high_q: float = 0.98) -> Tuple[float, float]:
    lo = float(np.quantile(values, low_q))
    hi = float(np.quantile(values, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if math.isclose(lo, hi):
        pad = max(1.0, abs(lo) * 0.1)
        return lo - pad, hi + pad
    pad = max(1.0, (hi - lo) * 0.08)
    return lo - pad, hi + pad


def _group_stats(points: np.ndarray, plane_cos: float, point_cos: np.ndarray) -> Dict[str, float]:
    dist = _plane_distance(points, plane_cos)
    residual_ref_plane = points[:, 0] - (points[:, 1] * plane_cos + points[:, 2])
    residual_true_cos = _closure_residual(points, point_cos)
    return {
        "mean_plane_distance": float(np.mean(dist)),
        "median_plane_distance": float(np.median(dist)),
        "mean_abs_closure_ref_plane": float(np.mean(np.abs(residual_ref_plane))),
        "mean_abs_closure_true_cos": float(np.mean(np.abs(residual_true_cos))),
        "max_abs_closure_true_cos": float(np.max(np.abs(residual_true_cos))),
        "mean_ghi": float(np.mean(points[:, 0])),
        "mean_dni": float(np.mean(points[:, 1])),
        "mean_dhi": float(np.mean(points[:, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline vs baseline+Soft vs baseline+PCD geometry comparison.")
    parser.add_argument("--baseline", required=True, help="Baseline payload NPZ or result directory.")
    parser.add_argument("--soft", required=True, help="Baseline+Soft payload NPZ or result directory.")
    parser.add_argument("--pcd", required=True, help="Baseline+PCD payload NPZ or result directory.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary output path.")
    parser.add_argument("--target-cos", type=float, default=None, help="Optional fixed cos(zenith) slice.")
    parser.add_argument("--slice-tol", type=float, default=0.005, help="Cos(zenith) tolerance for the slice.")
    parser.add_argument("--min-points", type=int, default=128, help="Minimum number of points to keep.")
    parser.add_argument("--max-points", type=int, default=600, help="Maximum number of points to keep.")
    parser.add_argument("--night-cos-threshold", type=float, default=0.0, help="Discard samples below this daytime threshold.")
    args = parser.parse_args()

    baseline = _load_payload(args.baseline)
    soft = _load_payload(args.soft)
    pcd = _load_payload(args.pcd)
    _validate_shapes(baseline, soft, pcd)

    chosen_flat_indices, slice_meta = _select_slice_indices(
        baseline["pred"],
        soft["pred"],
        pcd["pred"],
        baseline["cos_zenith"],
        args.target_cos,
        args.slice_tol,
        args.min_points,
        args.max_points,
        args.night_cos_threshold,
    )

    baseline_points = baseline["pred"].reshape(-1, baseline["pred"].shape[-1])[chosen_flat_indices]
    soft_points = soft["pred"].reshape(-1, soft["pred"].shape[-1])[chosen_flat_indices]
    pcd_points = pcd["pred"].reshape(-1, pcd["pred"].shape[-1])[chosen_flat_indices]
    true_points = baseline["true"].reshape(-1, baseline["true"].shape[-1])[chosen_flat_indices]
    chosen_cos = baseline["cos_zenith"].reshape(-1)[chosen_flat_indices]

    plane_cos = slice_meta["plane_cos"]
    summary = {
        "slice": slice_meta,
        "cos_range": {
            "min": float(np.min(chosen_cos)),
            "max": float(np.max(chosen_cos)),
        },
        "baseline": _group_stats(baseline_points, plane_cos, chosen_cos),
        "soft": _group_stats(soft_points, plane_cos, chosen_cos),
        "pcd": _group_stats(pcd_points, plane_cos, chosen_cos),
        "reference_true": _group_stats(true_points, plane_cos, chosen_cos),
        "baseline_payload": str(_resolve_payload(args.baseline)),
        "soft_payload": str(_resolve_payload(args.soft)),
        "pcd_payload": str(_resolve_payload(args.pcd)),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xx_full_source = np.concatenate([baseline_points[:, 1], soft_points[:, 1], pcd_points[:, 1]])
    yy_full_source = np.concatenate([baseline_points[:, 2], soft_points[:, 2], pcd_points[:, 2]])
    zz_full_source = np.concatenate([baseline_points[:, 0], soft_points[:, 0], pcd_points[:, 0]])

    x_min_full, x_max_full = _axis_limits(xx_full_source)
    y_min_full, y_max_full = _axis_limits(yy_full_source)
    z_min_full, z_max_full = _axis_limits(zz_full_source)

    fig = plt.figure(figsize=(12.2, 6.6))
    ax = fig.add_subplot(121, projection="3d")

    groups = [
        ("Baseline", baseline_points, "#6b7280", 0.20, 12),
        ("Baseline + Soft", soft_points, "#d62728", 0.42, 14),
        ("Baseline + PCD", pcd_points, "#1f77b4", 0.90, 16),
    ]

    xx, yy = np.meshgrid(
        np.linspace(x_min_full, x_max_full, 25),
        np.linspace(y_min_full, y_max_full, 25),
    )
    zz = plane_cos * xx + yy
    ax.plot_surface(
        xx,
        yy,
        zz,
        alpha=0.18,
        color="#8db3d3",
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    for _label, points, color, alpha, size in groups:
        ax.scatter(
            points[:, 1],
            points[:, 2],
            points[:, 0],
            s=size,
            c=color,
            alpha=alpha,
            depthshade=False,
        )

    ax.set_xlim(x_min_full, x_max_full)
    ax.set_ylim(y_min_full, y_max_full)
    ax.set_zlim(min(z_min_full, float(np.nanmin(zz))), max(z_max_full, float(np.nanmax(zz))))
    ax.set_xlabel("DNI")
    ax.set_ylabel("DHI")
    ax.set_zlabel("GHI")
    ax.view_init(elev=24, azim=-58)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.15, 1.0, 0.95))
    ax.set_title(
        "Geometry Slice on a Reference Plane\n"
        f"reference cos(z)={plane_cos:.6f}, actual slice=[{summary['cos_range']['min']:.6f}, {summary['cos_range']['max']:.6f}]",
        fontsize=10,
        pad=12,
    )

    ax2 = fig.add_subplot(122)
    residual_groups = [
        np.abs(_closure_residual(baseline_points, chosen_cos)),
        np.abs(_closure_residual(soft_points, chosen_cos)),
        np.abs(_closure_residual(pcd_points, chosen_cos)),
    ]
    box = ax2.boxplot(
        residual_groups,
        tick_labels=["Baseline", "Baseline + Soft", "Baseline + PCD"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], ["#6b7280", "#d62728", "#1f77b4"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.5)
    ax2.set_yscale("symlog", linthresh=1e-3)
    ax2.set_ylabel(r"$|G - c_i N - D|$ using each point's own $c_i$")
    ax2.set_title("True Physics Closure Residual", fontsize=10, pad=12)
    ax2.grid(True, axis="y", alpha=0.25, linestyle="--")

    note_lines = [
        "Why some blue points may look slightly off the drawn plane:",
        "The left panel uses one median-cos reference plane for the whole slice.",
        "The right panel uses each point's own cos(z) and is the real closure check.",
        f"PCD mean true closure = {summary['pcd']['mean_abs_closure_true_cos']:.6g}",
    ]
    ax2.text(
        0.02,
        0.02,
        "\n".join(note_lines),
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved triplet geometry figure to {output_path}")
    print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
