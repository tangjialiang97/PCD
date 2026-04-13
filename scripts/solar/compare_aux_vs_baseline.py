#!/usr/bin/env python3
"""Compare auxiliary-loss runs against no-aux baselines.

Usage example:
  python scripts/solar/compare_aux_vs_baseline.py \
    --true-file result_long_term_forecast_Solar_Folsom_combined_no_zero_True.txt \
    --false-file result_long_term_forecast_Solar_Folsom_combined_no_zero_False.txt
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Record:
    model: str
    pred_len: int
    mse: float
    mae: Optional[float]
    nrmse: Optional[float]
    nmae: Optional[float]
    cw: Optional[float]
    setting: str


def _parse_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_model(setting: str) -> Optional[str]:
    # Most settings contain "..._<MODEL>_Solar_..."
    if "_Solar_" not in setting:
        return None
    prefix = setting.split("_Solar_", 1)[0]
    tokens = prefix.split("_")
    if not tokens:
        return None
    return tokens[-1]


def parse_result_file(path: Path) -> List[Record]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    records: List[Record] = []
    metric_pattern = re.compile(r"(mse|mae|nrmse|nmae):([^,]+)")

    i = 0
    while i < len(lines):
        setting = lines[i]
        if not setting.startswith("long_term_forecast_"):
            i += 1
            continue
        if i + 1 >= len(lines) or "mse:" not in lines[i + 1]:
            i += 1
            continue

        metrics_line = lines[i + 1]
        pred_match = re.search(r"_pl(\d+)_", setting)
        model = _extract_model(setting)
        if pred_match is None or model is None:
            i += 2
            continue

        metric_map: Dict[str, float] = {}
        for name, value in metric_pattern.findall(metrics_line):
            parsed = _parse_float(value.strip())
            if parsed is not None:
                metric_map[name] = parsed

        if "mse" not in metric_map:
            i += 2
            continue

        cw_match = re.search(r"_cw([\d.eE+-]+)", setting)
        cw = _parse_float(cw_match.group(1)) if cw_match else None

        records.append(
            Record(
                model=model,
                pred_len=int(pred_match.group(1)),
                mse=metric_map["mse"],
                mae=metric_map.get("mae"),
                nrmse=metric_map.get("nrmse"),
                nmae=metric_map.get("nmae"),
                cw=cw,
                setting=setting,
            )
        )
        i += 2

    return records


def pick_best(records: Iterable[Record]) -> Dict[Tuple[str, int], Record]:
    best: Dict[Tuple[str, int], Record] = {}
    for r in records:
        key = (r.model, r.pred_len)
        # Primary criterion: nrmse (fallback to mse when nrmse missing).
        current = best.get(key)
        r_score = r.nrmse if r.nrmse is not None else r.mse
        c_score = (current.nrmse if current and current.nrmse is not None else current.mse) if current else None
        if current is None or (c_score is not None and r_score < c_score):
            best[key] = r
    return best


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def fmt_float(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return f"{v:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize best auxiliary cw per model/pred_len and compare vs baseline."
    )
    parser.add_argument(
        "--true-file",
        type=Path,
        default=Path("result_long_term_forecast_Solar_Folsom_combined_no_zero_True.txt"),
        help="Result file generated with auxiliary loss enabled.",
    )
    parser.add_argument(
        "--false-file",
        type=Path,
        default=Path("result_long_term_forecast_Solar_Folsom_combined_no_zero_False.txt"),
        help="Baseline result file generated without auxiliary loss.",
    )
    args = parser.parse_args()

    true_records = parse_result_file(args.true_file)
    false_records = parse_result_file(args.false_file)
    if not true_records:
        raise RuntimeError(f"No valid records parsed from {args.true_file}")
    if not false_records:
        raise RuntimeError(f"No valid records parsed from {args.false_file}")

    best_true = pick_best(true_records)
    best_false = pick_best(false_records)

    keys = sorted(set(best_true.keys()) | set(best_false.keys()), key=lambda x: (x[0], x[1]))

    header = (
        "| Model | pred_len | baseline_nrmse | best_true_nrmse | delta_nrmse(%) | "
        "baseline_nmae | best_true_nmae | delta_nmae(%) | best_cw |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"

    print(f"# Compare Auxiliary vs Baseline")
    print(f"- true file: `{args.true_file}`")
    print(f"- false file: `{args.false_file}`")
    print(f"- parsed records: true={len(true_records)}, false={len(false_records)}\n")
    print(header)
    print(sep)

    improved = 0
    worsened = 0
    missing = 0

    for model, pl in keys:
        t = best_true.get((model, pl))
        b = best_false.get((model, pl))
        if b is None or t is None:
            missing += 1
        else:
            d_nrmse = pct_change(t.nrmse, b.nrmse)
            if d_nrmse is not None:
                if d_nrmse < 0:
                    improved += 1
                elif d_nrmse > 0:
                    worsened += 1

        b_nrmse = b.nrmse if b else None
        t_nrmse = t.nrmse if t else None
        b_nmae = b.nmae if b else None
        t_nmae = t.nmae if t else None
        d_nrmse = pct_change(t_nrmse, b_nrmse)
        d_nmae = pct_change(t_nmae, b_nmae)
        cw = t.cw if t else None

        print(
            f"| {model} | {pl} | {fmt_float(b_nrmse, 6)} | {fmt_float(t_nrmse, 6)} | {fmt_pct(d_nrmse)} | "
            f"{fmt_float(b_nmae, 6)} | {fmt_float(t_nmae, 6)} | {fmt_pct(d_nmae)} | {fmt_float(cw, 6)} |"
        )

    print(
        "\n"
        f"Summary: improved={improved}, worsened={worsened}, missing_pairs={missing}, total_pairs={len(keys)}"
    )


if __name__ == "__main__":
    main()
