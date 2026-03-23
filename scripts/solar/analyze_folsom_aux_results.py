#!/usr/bin/env python3
"""Parse result_long_term_forecast_Solar_Folsom_combined_no_zero_True.txt and summarize best loss weights per model & pred_len."""
import re
from collections import defaultdict

RESULT_PATH = (
    "/root/tjl/code/Time-Series-Library-fm/result_long_term_forecast_Solar_Folsom_combined_no_zero_True.txt"
)

pat_weights = re.compile(
    r"cw([\d.eE+-]+)_lw([\d.eE+-]+)_sc([\d.eE+-]+)_vg([\d.eE+-]+)_vw([\d.eE+-]+)"
)
pat_head = re.compile(r"^long_term_forecast_irr_256_(\d+)_(.+?)_Solar_")


def main() -> None:
    with open(RESULT_PATH, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]

    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("long_term_forecast_"):
            i += 1
            continue
        if i + 1 >= len(lines) or "mse:" not in lines[i + 1]:
            i += 1
            continue
        metric = lines[i + 1]
        m = pat_head.match(line)
        if not m:
            i += 1
            continue

        pred_len = int(m.group(1))
        model = m.group(2)

        mse_m = re.search(r"mse:([^,]+)", metric)
        nrmse_m = re.search(r"nrmse:([^,]+)", metric)
        mae_m = re.search(r"mae:([^,]+)", metric)
        if not mse_m:
            i += 2
            continue

        mse_val = float(mse_m.group(1).strip())
        nrmse_val = float(nrmse_m.group(1).strip()) if nrmse_m else None
        mae_val = float(mae_m.group(1).strip()) if mae_m else None

        wm = pat_weights.search(line)
        param_key = tuple(map(float, wm.groups())) if wm else None

        records.append(
            {
                "model": model,
                "pl": pred_len,
                "mse": mse_val,
                "mae": mae_val,
                "nrmse": nrmse_val,
                "params": param_key,
            }
        )
        i += 2

    best_by_group = {}
    for r in records:
        key = (r["model"], r["pl"], r["params"])
        if key not in best_by_group or r["mse"] < best_by_group[key]["mse"]:
            best_by_group[key] = r

    per_mp = defaultdict(list)
    for r in best_by_group.values():
        per_mp[(r["model"], r["pl"])].append(r)

    models = sorted({r["model"] for r in records})
    pls = sorted({r["pl"] for r in records})

    print(f"Parsed blocks: {len(records)} | unique (model, pl, params): {len(best_by_group)}")
    print(f"Models: {models} | pred_lens: {pls}\n")

    for pl in pls:
        print("=" * 76)
        print(f"PRED_LEN = {pl}")
        print("=" * 76)
        for model in models:
            candidates = per_mp.get((model, pl), [])
            if not candidates:
                continue
            candidates.sort(key=lambda x: x["mse"])
            best = candidates[0]
            p = best["params"]
            nr = best.get("nrmse")
            nr_s = f"{nr:.6f}" if nr is not None else "NA"
            if p is not None:
                pstr = f"cw={p[0]}, lw={p[1]}, sc={p[2]}, vg={p[3]}, vw={p[4]}"
            else:
                pstr = (
                    "no cw/lw suffix in setting name (legacy); "
                    "if use_auxiliary, run.py defaults: cw=1, lw=1, sc=1367, vg=0.1, vw=1"
                )
            mse_b = best["mse"]
            print(f"  {model:18s}  BEST mse={mse_b:.4f}  nrmse={nr_s}  |  {pstr}")
            if len(candidates) > 1:
                b2 = candidates[1]
                p2 = b2["params"]
                if p2 is not None:
                    p2s = f"cw={p2[0]}, lw={p2[1]}, vg={p2[3]}, vw={p2[4]}"
                else:
                    p2s = "legacy"
                print(f"      (2nd) mse={b2['mse']:.4f}  |  {p2s}")
        print()

    print("=" * 76)
    print("Best per (model, pl) among runs WITH explicit cw/lw/sc/vg/vw in setting only")
    print("=" * 76)
    for pl in pls:
        for model in models:
            candidates = [r for r in per_mp.get((model, pl), []) if r["params"] is not None]
            if not candidates:
                continue
            candidates.sort(key=lambda x: x["mse"])
            best = candidates[0]
            p = best["params"]
            nr = best["nrmse"]
            print(
                f"pl={pl:3d}  {model:14s}  mse={best['mse']:.2f}  nrmse={nr:.6f}  |  "
                f"cw={p[0]} lw={p[1]} sc={p[2]} vg={p[3]} vw={p[4]}"
            )


if __name__ == "__main__":
    main()
