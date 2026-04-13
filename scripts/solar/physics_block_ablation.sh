#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

gpu_id=0
root_path="/opt/data/private/data/timedata/"
data_path="Folsom_combined_no_zero.csv"
block_modes=(baseline gated)

# Keep gated feedback and only finish the missing/high-value comparisons.
experiments=(
  "iTransformerSolarPhysics 48"
  "TimeMixerSolarPhysics 168"
)

for exp_cfg in "${experiments[@]}"; do
  read -r model_name pred_len <<< "$exp_cfg"
  for block_mode in "${block_modes[@]}"; do
    extra_flags=()
    extra_tag="$block_mode"
    case "$block_mode" in
      baseline)
        ;;
      gated)
        extra_flags+=(--physics_gated_feedback)
        ;;
      *)
        echo "Unsupported block mode: $block_mode" >&2
        exit 1
        ;;
    esac

    echo "[RUN] model=${model_name} pred_len=${pred_len} block_mode=${block_mode} $(date '+%F %T')"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --is_three \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "blockAbl_${model_name}_256_${pred_len}" \
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
      --des PhysBlockAbl \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --gpu "$gpu_id" \
      --physics_layers 2 \
      --physics_hidden_dim 128 \
      --physics_kernel_size 3 \
      --extra_tag "$extra_tag" \
      "${extra_flags[@]}"
  done
done
