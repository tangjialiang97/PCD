# Feasibility by Construction: A Physics-Constrained Decoder for Solar Irradiance Forecasting

Official code repository for our NeurIPS 2026 submission on solar irradiance forecasting.

This repository contains the training, evaluation, data loading, and experiment scripts used in the paper. The code supports standard forecasting backbones and their `SolarPhysics` variants.


## Repository Structure

```text
.
├── run.py                         # Main experiment entry point
├── requirements.txt               # Python dependencies
├── data/
│   └── solar/                     # Solar irradiance CSV files
├── data_provider/                 # Dataset and dataloader code
├── exp/                           # Training and evaluation loops
├── models/                        # Forecasting backbones and SolarPhysics variants
├── scripts/
│   └── solar/
│       ├── irr.sh                 # Baseline solar forecasting runs
│       └── phy.sh                 # Physics-aware solar forecasting runs
├── checkpoints/                   # Saved model checkpoints
└── result.txt                     # Appended test metrics
```

## Installation

We recommend using a fresh Python environment.

```bash
conda create -n pcd python=3.11
conda activate pcd

mkdir -p src
if [ ! -d src/timesfm/.git ]; then
  git -c filter.lfs.required=false \
      -c filter.lfs.clean=cat \
      -c filter.lfs.smudge=cat \
      -c filter.lfs.process= \
      clone --filter=blob:none https://github.com/google-research/timesfm.git src/timesfm
fi
git -c filter.lfs.required=false \
    -c filter.lfs.clean=cat \
    -c filter.lfs.smudge=cat \
    -c filter.lfs.process= \
    -C src/timesfm fetch origin bf88c5dc88f6275b7071ebcd9051406c11435c0c
git -c filter.lfs.required=false \
    -c filter.lfs.clean=cat \
    -c filter.lfs.smudge=cat \
    -c filter.lfs.process= \
    -C src/timesfm checkout -f bf88c5dc88f6275b7071ebcd9051406c11435c0c

pip install --no-deps -r requirements.txt
```

## Data

Place solar CSV files under:

```text
data/solar/
```

## Additional results

The experimental results for various backbones are shown in experiment_summary.md (only trained with one seed).

## Supported Backbones

The repository currently supports the following forecasting backbones through `--model`:

- Autoformer
- Chronos
- Chronos2
- Crossformer
- DLinear
- EMAformer
- ETSformer
- FEDformer
- FiLM
- FreTS
- Informer
- KANAD
- Koopa
- LightTS
- MICN
- MambaSimple
- mixlinear
- MSGNet
- MultiPatchFormer
- Nonstationary_Transformer
- PAttn
- PatchTST
- Pyraformer
- Reformer
- SCINet
- SegRNN
- Sundial
- TemporalFusionTransformer
- TiDE
- TiRex
- TimeFilter
- TimeMixer
- TimeMoE
- TimeXer
- TimesFM
- TimesNet
- Transformer
- TSMixer
- WPMixer
- iTransformer

Most backbones also have a physics-aware variant with the `SolarPhysics` suffix, for example `TimeFilterSolarPhysics`, `PatchTSTSolarPhysics`, and `DLinearSolarPhysics`. 

## Quick Start

Run baseline solar experiments:

```bash
bash scripts/solar/irr.sh
```

Run physics-aware solar experiments:

```bash
bash scripts/solar/phy.sh
```

## Solar generation power workflow

完整变量清单、公式与运行步骤见：

**[`scripts/solar/README.md`](scripts/solar/README.md)**（GHI→功率集中映射方案）

Physics-aware models still forecast feasible `[ghi, dni, dhi]` first. When
`--solar_power_output power` (or `append`) is set, projected GHI is mapped to
power by a selectable head (`--solar_power_map`) on top of a selectable prior
(`--solar_power_prior`):

| map mode | meaning |
|------|---------|
| `fixed` | return the physics prior only |
| `affine` | learnable `softplus(a)*prior + b` |
| `residual_mlp` | `prior + MLP([ghi_n, prior_n, cos_z])` (zero-init residual) |
| `learned` | pure MLP map from `[ghi_n, cos_z]` (no prior add) |

| prior | meaning |
|------|---------|
| `linear` | `P = capacity * clamp(GHI,0)/ghi_ref` |
| `pvwatts` | differentiable PVWatts-DC: `P = capacity * (G/Gref) * (1 + gamma*(Tc-25))` with NOCT/Faiman `Tc` |

### A) Synthetic Congo power labels

```bash
python scripts/solar/build_power_dataset.py \
  --input data/solar/Congo.csv \
  --output data/solar/Congo_power.csv \
  --label-mode nonlinear

bash scripts/solar/power.sh
```

### B) Real plant power + offline pvlib columns

Plant tables use:

```text
dtime, observe_power, observe_ghi
```

Weather / forecast tables may include:

```text
dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE, interval, interval_day,
report_date, report_time, update_time, govern_flag
```

Build a training CSV with **real power labels** and an offline **pvlib prior column**:

```bash
python scripts/solar/build_plant_power_dataset.py \
  --plant /path/to/plant.xlsx \
  --weather /path/to/weather.xlsx \
  --latitude 29.919 --longitude 100.641 \
  --timezone Asia/Shanghai \
  --temp-cell-method faiman \
  --add-erbs-components True \
  --output data/solar/Qingda_plant_power.csv
```

Output columns include:

```text
timeStamp, ghi, dni, dhi, power, power_pvlib, temp_air, wind_speed,
temp_cell, cos_zenith, pdc0, [power_linear], ...
```

- `power` = `observe_power` (supervision target)
- `power_pvlib` = offline Faiman + `pvlib.pvsystem.pvwatts_dc`
- `dni/dhi` = optional Erbs estimates so `enc_in=3` SolarPhysics still runs

Train with in-model PVWatts prior + residual map:

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./data/solar/ --data_path Qingda_plant_power.csv \
  --model DLinearSolarPhysics --data Solar --features M \
  --seq_len 256 --pred_len 24 --enc_in 3 --dec_in 3 --c_out 3 \
  --solar_power_output power \
  --solar_power_map residual_mlp \
  --solar_power_prior pvwatts \
  --solar_power_gamma_pdc -0.004 \
  --solar_power_temp_cell_method noct
```

Or run the bundled recipe (edit plant/weather paths inside):

```bash
bash scripts/solar/plant_power.sh
```

**Note:** in-model `pvwatts` currently uses GHI + NOCT/Faiman defaults from CLI
(`--solar_power_temp_amb`, `--solar_power_wind_default`, etc.). The richer offline
`TEMP/WS` series is written to CSV for analysis and for future conditional mapping;
set `--solar_power_capacity` to the auto-estimated `pdc0` printed by the builder
for best prior scale.
