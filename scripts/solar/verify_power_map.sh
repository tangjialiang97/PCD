#!/usr/bin/env bash
set -euo pipefail
cd /opt/data/private/code/PCD

echo "=== 1) Rebuild nonlinear Congo_power ==="
python scripts/solar/build_power_dataset.py \
  --input data/solar/Congo.csv \
  --output data/solar/Congo_power.csv \
  --label-mode nonlinear

python - <<'PY'
import pandas as pd
df = pd.read_csv('data/solar/Congo_power.csv')
x = df['ghi'].clip(lower=0) / 1000.0
print('cols', list(df.columns), 'rows', len(df))
print('mean|power-linear|', float((df['power'] - x).abs().mean()))
print('max power', float(df['power'].max()), 'max linear', float(x.max()))
assert list(df.columns) == ['timeStamp', 'ghi', 'dni', 'dhi', 'power']
assert (df['power'] - x).abs().mean() > 1e-4
print('DATA_OK')
PY

echo "=== 2) Unit tests SolarPowerMap ==="
python - <<'PY'
import torch
from models._solar_physics_utils import (
    SolarPowerMap,
    ghi_to_power,
    init_solar_power_mapping,
    cache_solar_outputs,
)
from types import SimpleNamespace

ghi = torch.tensor([[[0.0], [100.0], [500.0], [1000.0]]])
cos = torch.tensor([[1.0, 0.8, 0.5, 0.0]])
prior = ghi_to_power(ghi, capacity=1.0, ghi_ref=1000.0, clip=False)

out = SolarPowerMap('fixed', capacity=1.0, ghi_ref=1000.0)(ghi, cos)
assert torch.allclose(out, prior)
print('fixed OK', out.squeeze().tolist())

m = SolarPowerMap('affine', capacity=1.0, ghi_ref=1000.0)
out = m(ghi, cos)
assert torch.allclose(out, prior, atol=1e-6)
assert m.raw_scale.requires_grad and m.bias.requires_grad
print('affine init OK', out.squeeze().tolist())

m = SolarPowerMap('residual_mlp', capacity=1.0, ghi_ref=1000.0, hidden_dim=16)
out = m(ghi, cos)
assert torch.allclose(out, prior, atol=1e-6), float((out - prior).abs().max())
print('residual_mlp init OK', out.squeeze().tolist())

m = SolarPowerMap('learned', capacity=1.0, ghi_ref=1000.0, hidden_dim=16)
out = m(ghi, cos)
assert (out >= 0).all()
print('learned OK', out.squeeze().tolist())


class Dummy(torch.nn.Module):
    pass


cfg = SimpleNamespace(
    solar_power_output='power',
    solar_power_map='residual_mlp',
    solar_power_capacity=1.0,
    solar_power_ghi_ref=1000.0,
    solar_power_clip=False,
    solar_power_map_hidden=16,
    solar_power_map_dropout=0.0,
)
model = Dummy()
init_solar_power_mapping(model, cfg)
assert any(n.startswith('solar_power_map') for n, _ in model.named_parameters())
proj = torch.tensor([[[100.0, 10.0, 20.0], [0.0, 0.0, 0.0], [1000.0, 800.0, 200.0]]])
zen = torch.tensor([[30.0, 90.0, 20.0]])
out = cache_solar_outputs(model, torch.zeros_like(proj), proj, zen)
assert out.shape == (1, 3, 1)
loss = out.sum()
loss.backward()
grads = [p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters()]
print('cache', out.squeeze().tolist(), 'grads', any(grads))
print('UNIT_OK')
PY

echo "=== 3) Physics unchanged ==="
git diff --stat -- layers/solar_physics.py
echo "PHYSICS_DIFF_LINES=$(git diff -- layers/solar_physics.py | wc -l)"

echo "=== 4) Smoke train ==="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --debug True \
  --root_path ./data/solar/ \
  --data_path Congo_power.csv \
  --model DLinearSolarPhysics \
  --data Solar \
  --features M \
  --seq_len 256 \
  --pred_len 24 \
  --label_len 0 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --batch_size 64 \
  --train_epochs 1 \
  --num_workers 0 \
  --solar_power_output power \
  --solar_power_map residual_mlp \
  --des PowerMapSmoke \
  --gpu 0
echo "SMOKE_DONE"
