#!/usr/bin/env bash
set -uo pipefail

model_name="${1:?Usage: $0 MODEL_NAME [LOG_FILE]}"
log_file="${2:-scripts/solar/${model_name}.log}"
debug_mode="${DEBUG_MODE:-False}"
gpu_id="${GPU_ID:-0}"
seeds=(2021 2022 2023 2024 2025)

if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("Congo.csv")
else
  pred_lens=(24 48 96 192)
  data_paths=("Congo.csv" "Germany.csv" "NewDelhi.csv" "Folsom.csv")
fi

echo "===== ${model_name} solar runs started: $(date '+%F %T') =====" | tee -a "$log_file"
failure_count=0

for seed in "${seeds[@]}"; do
for data_path in "${data_paths[@]}"; do
for pred_len in "${pred_lens[@]}"; do
  label_len=0
  batch_size=512

  echo "[RUN] seed=${seed} model=${model_name} data=${data_path} pred_len=${pred_len}" | tee -a "$log_file"

  if ! python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --seed "$seed" \
    --debug "$debug_mode" \
    --root_path ./data/solar/ \
    --data_path "$data_path" \
    --model_id "256_${pred_len}" \
    --model "$model_name" \
    --data Solar \
    --features M \
    --seq_len 256 \
    --pred_len "$pred_len" \
    --label_len "$label_len" \
    --batch_size "$batch_size" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --is_three \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --gpu "$gpu_id" 2>&1 | tee -a "$log_file"; then
    failure_count=$((failure_count + 1))
    echo "[ERROR] seed=${seed} model=${model_name} data=${data_path} pred_len=${pred_len} failed, continue to next run" | tee -a "$log_file"
  fi
done
done
done

echo "===== ${model_name} solar runs finished: $(date '+%F %T') failures=${failure_count} =====" | tee -a "$log_file"
