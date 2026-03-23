#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse result_*.txt like:
  long_term_forecast_..._256_12_iTransformer_Solar_ftS_...
  mse:..., mae:..., nrmse:..., nmae:...

Export to CSV with headers: 方法, 预测长度, nrmse, nmae
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional


METRICS_RE = re.compile(r"nrmse:\s*([0-9.]+)\s*,\s*nmae:\s*([0-9.]+)")


@dataclass(frozen=True)
class Record:
    method: str
    pred_len: int
    nrmse: float
    nmae: float


def _parse_exp_name_get_method_pred_len(exp_name: str) -> tuple[str, int]:
    """
    Strategy: find token that starts with 'ft' (e.g. 'ftS') and take:
      dataset = tokens[idx_ft - 1]
      method  = tokens[idx_ft - 2]
      pred    = tokens[idx_ft - 3]
    This matches lines like:
      long_term_forecast_13PV_256_12_iTransformer_Solar_ftS_...
    """
    tokens = exp_name.strip().split("_")
    idx_ft = next((i for i, t in enumerate(tokens) if t.startswith("ft")), None)
    if idx_ft is None or idx_ft < 3:
        raise ValueError(f"Unrecognized exp name format (no ft* token): {exp_name}")
    method = tokens[idx_ft - 2]
    pred_len_str = tokens[idx_ft - 3]
    try:
        pred_len = int(pred_len_str)
    except ValueError as e:
        raise ValueError(f"Unrecognized pred_len token '{pred_len_str}' in: {exp_name}") from e
    return method, pred_len


def parse_records(lines: Iterable[str]) -> list[Record]:
    records: list[Record] = []
    current_method: Optional[str] = None
    current_pred_len: Optional[int] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("long_term_forecast_"):
            current_method, current_pred_len = _parse_exp_name_get_method_pred_len(line)
            continue

        m = METRICS_RE.search(line)
        if m and current_method is not None and current_pred_len is not None:
            nrmse = float(m.group(1))
            nmae = float(m.group(2))
            # 保留小数点后4位
            nrmse = round(nrmse, 4)
            nmae = round(nmae, 4)
            records.append(
                Record(
                    method=current_method,
                    pred_len=current_pred_len,
                    nrmse=nrmse,
                    nmae=nmae,
                )
            )
            current_method = None
            current_pred_len = None

    return records


def write_csv(records: list[Record], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["方法", "预测长度", "nrmse", "nmae"])
        for r in sorted(records, key=lambda x: (x.method.lower(), x.pred_len)):
            w.writerow([r.method, r.pred_len, r.nrmse, r.nmae])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Path to result_*.txt",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to output .csv (default: same name as input with .csv)",
    )
    args = ap.parse_args()

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(
        args.output if args.output else os.path.splitext(in_path)[0] + ".csv"
    )

    with open(in_path, "r", encoding="utf-8") as f:
        records = parse_records(f)

    if not records:
        raise SystemExit(f"No records parsed from: {in_path}")

    write_csv(records, out_path)
    print(f"Wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()

