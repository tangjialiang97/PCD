# Physics-Constrained Decoder for Solar Irradiance Forecasting

Official code repository for our ICDE 2027 paper on solar irradiance forecasting.

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
│       ├── phy.sh                 # Physics-aware solar forecasting runs
│       ├── run_baseline_model.sh  # Shared five-seed baseline runner
│       ├── timefilter.sh           # TimeFilter runs
│       ├── itransformer.sh         # iTransformer runs
│       ├── timesnet.sh              # TimesNet runs
│       ├── timesfm.sh               # TimesFM runs
│       └── timemoe.sh               # TimeMoE runs
├── checkpoints/                   # Saved model checkpoints
└── result.txt                     # Appended test metrics
```

## Installation

We recommend using a fresh Python environment.

```bash
conda create -n pcd python=3.10
conda activate pcd
pip install -r requirements.txt
```

## Data

Place solar CSV files under:

```text
data/solar/
```

## Additional results

The experimental results for various backbones are shown in experiment_summary.md and are aggregated over five random seeds (2021, 2022, 2023, 2024, and 2025), with mean ± standard deviation reported where applicable.

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
