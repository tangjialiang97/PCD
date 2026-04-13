#!/usr/bin/env bash
set -euo pipefail

# 固定使用物理卡 1。进程内会映射成 local cuda:0，所以 GPU_ID 保持 0。
export CUDA_VISIBLE_DEVICES=1

gpu_id=0
root_path="${ROOT_PATH:-/opt/data/private/data/timedata/}"
data_path="${DATA_PATH:-Folsom_combined_no_zero.csv}"
train_flag="${IS_TRAINING:-1}"
pred_lens_str="${PRED_LENS:-12 24 48 168}"
models=(DLinearSolarPhysics3 PatchTSTSolarPhysics3)

for model_name in "${models[@]}"; do
  for pred_len in $pred_lens_str; do
    echo "[RUN] model=${model_name} pred_len=${pred_len} cuda_visible=${CUDA_VISIBLE_DEVICES} $(date '+%F %T')"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training "$train_flag" \
      --is_three \
      --use_auxiliary \
      --has_relhum \
      --closure_weight 0.1 \
      --limit_weight 1.0 \
      --var_weight 1.0 \
      --var_gamma 0.1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "abl3_${model_name}_256_${pred_len}" \
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
      --des PhysDirect3 \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --gpu "$gpu_id" \
      --physics_layers 2 \
      --physics_hidden_dim 128 \
      --physics_kernel_size 3 \
      --extra_tag direct3
  done
done
