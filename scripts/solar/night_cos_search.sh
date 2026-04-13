#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

gpu_id=0
root_path="/opt/data/private/data/timedata/"
data_path="Folsom_combined_no_zero.csv"
night_thresholds=(0.0 0.1 0.2)

# Focus on the strongest model-horizon pairs from the current results.
experiments=(
  "PatchTSTSolarPhysics 12"
  "DLinearSolarPhysics 24"
  "iTransformerSolarPhysics 48"
  "TimeMixerSolarPhysics 168"
)

for exp_cfg in "${experiments[@]}"; do
  read -r model_name pred_len <<< "$exp_cfg"
  for night_thr in "${night_thresholds[@]}"; do
    echo "[RUN] model=${model_name} pred_len=${pred_len} night_cos_threshold=${night_thr} $(date '+%F %T')"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --is_three \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "nightSearch_${model_name}_256_${pred_len}" \
      --model "$model_name" \
      --data Solar \
      --features M \
      --seq_len 256 \
      --pred_len "$pred_len" \
      --label_len 0 \
      --batch_size 512 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 3 \
      --dec_in 3 \
      --c_out 3 \
      --des PhysNightThr \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --gpu "$gpu_id" \
      --physics_layers 2 \
      --physics_hidden_dim 128 \
      --physics_kernel_size 3 \
      --night_cos_threshold "$night_thr"
  done
done
