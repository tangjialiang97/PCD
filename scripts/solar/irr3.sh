

#!/usr/bin/env bash
set -euo pipefail

debug_mode=False
gpu_id=0
log_file="scripts/solar/irr_grid_3_highcw.log"

# Broad model sweep
models=(DLinear PatchTST)

if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("Folsom_combined_no_zero.csv")
fi

# Grid search space (log-scaled for stable loss balancing)
closure_weights=(0.1)
limit_weights=(1.0)
var_weights=(1.0)
var_gammas=(0.1)

echo "===== Grid search started: $(date '+%F %T') =====" | tee -a "$log_file"

for data_path in "${data_paths[@]}"; do
for model_name in "${models[@]}"; do
for pred_len in "${pred_lens[@]}"; do
for closure_weight in "${closure_weights[@]}"; do
for limit_weight in "${limit_weights[@]}"; do
for var_weight in "${var_weights[@]}"; do
for var_gamma in "${var_gammas[@]}"; do

  # Encoder-decoder models generally need non-zero label length.
  if [ "$model_name" = "Informer" ] || [ "$model_name" = "FEDformer" ]; then
    label_len=128
    batch_size=256
  else
    label_len=0
    batch_size=512
  fi

  echo "[RUN] model=${model_name} data=${data_path} pred_len=${pred_len} ll=${label_len} bs=${batch_size} cw=${closure_weight} lw=${limit_weight} vw=${var_weight} vg=${var_gamma}" | tee -a "$log_file"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --closure_weight "$closure_weight" \
    --limit_weight "$limit_weight" \
    --var_weight "$var_weight" \
    --var_gamma "$var_gamma" \
    --debug "$debug_mode" \
    --root_path /opt/data/private/data/timedata/ \
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
    --des 'ExpGrid' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --gpu "$gpu_id" 2>&1 | tee -a "$log_file"

done
done
done
done
done
done
done

echo "===== Grid search finished: $(date '+%F %T') =====" | tee -a "$log_file"
