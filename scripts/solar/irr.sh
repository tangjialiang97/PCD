

#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
set -uo pipefail

debug_mode=False
gpu_id=0
log_file="scripts/solar/irr.log"
seeds=(2021 2022 2023 2024 2025)

# Broad baseline sweep aligned with phy.sh
models=(
  PatchTST
  EMAformer
  DLinear
  mixlinear
)

if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("Congo.csv")
else
  pred_lens=(24 48 96 192)
  data_paths=("Congo.csv" "Germany.csv" "NewDelhi.csv" "Folsom.csv")
fi

echo "===== Solar IRR runs started: $(date '+%F %T') =====" | tee -a "$log_file"
failure_count=0

for seed in "${seeds[@]}"; do
for data_path in "${data_paths[@]}"; do
for model_name in "${models[@]}"; do
for pred_len in "${pred_lens[@]}"; do

  # Encoder-decoder models generally need non-zero label length.
  if [ "$model_name" = "Autoformer" ] || [ "$model_name" = "Nonstationary_Transformer" ] || [ "$model_name" = "Informer" ] || [ "$model_name" = "FEDformer" ]; then
    label_len=128
    batch_size=256
  else
    label_len=0
    batch_size=512
  fi

  echo "[RUN] seed=${seed} model=${model_name} data=${data_path} pred_len=${pred_len} ll=${label_len} bs=${batch_size}" | tee -a "$log_file"

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
    echo "[ERROR] model=${model_name} data=${data_path} pred_len=${pred_len} failed, continue to next run" | tee -a "$log_file"
  fi

done
done
done
done

echo "===== Solar IRR runs finished: $(date '+%F %T') failures=${failure_count} =====" | tee -a "$log_file"
