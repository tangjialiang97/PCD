#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

gpu_id="${GPU_ID:-0}"
root_path="${ROOT_PATH:-/opt/data/private/data/timedata/}"
data_path="${DATA_PATH:-Folsom_combined_no_zero.csv}"

# Reuse the currently strongest model-horizon pairs to isolate the effect of the
# simplified cosz-based upper bound.
experiments=(
  "PatchTSTSolarPhysics 12"
  "DLinearSolarPhysics 24"
  "iTransformerSolarPhysics 48"
  "TimeMixerSolarPhysics 168"
)

upper_bound_cfgs=(
  "off 0.8 1.2"
  "on 0.8 1.2"
  "on 0.9 1.1"
)

for exp_cfg in "${experiments[@]}"; do
  read -r model_name pred_len <<< "$exp_cfg"

  for ub_cfg in "${upper_bound_cfgs[@]}"; do
    read -r use_bound alpha_min alpha_max <<< "$ub_cfg"
    extra_args=()
    tag="wub_off"
    if [[ "$use_bound" == "on" ]]; then
      extra_args+=(
        --physics_use_weak_upper_bound
        --physics_upper_bound_alpha_min "$alpha_min"
        --physics_upper_bound_alpha_max "$alpha_max"
      )
      tag="wub_a${alpha_min}_${alpha_max}"
    fi

    echo "[RUN] model=${model_name} pred_len=${pred_len} weak_upper=${use_bound} alpha=[${alpha_min},${alpha_max}] $(date '+%F %T')"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --is_three \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "weakUpper_${model_name}_256_${pred_len}" \
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
      --des WeakUpper \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --gpu "$gpu_id" \
      --physics_layers 2 \
      --physics_hidden_dim 128 \
      --physics_kernel_size 3 \
      --extra_tag "$tag" \
      "${extra_args[@]}"
  done
done
