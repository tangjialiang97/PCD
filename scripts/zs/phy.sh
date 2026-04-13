

#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TSL_HF_CACHE_DIR="${TSL_HF_CACHE_DIR:-/opt/data/private/huggingface_models}"
export HF_HOME="${HF_HOME:-$TSL_HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TSL_HF_CACHE_DIR/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TSL_HF_CACHE_DIR/transformers}"
set -uo pipefail

gpu_id=0
log_file="scripts/solar/irr_grid_2_highcw.log"

# Broad baseline sweep aligned with phy.sh
models=(
    TiRexSolarPhysics
    ChronosSolarPhysics
    Chronos2SolarPhysics
)

debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(24)
  data_paths=("45.36_-90.05_2015_2024_nonzero.csv")
else
  pred_lens=(24 48 96 192)
  data_paths=("45.36_-90.05_2015_2024_nonzero.csv" "34.05_-118.24_2015_2024_nonzero.csv" "31.1_-90.05_2015_2024_nonzero.csv" "Folsom_combined_no_zero.csv")
fi

# Grid search space (log-scaled for stable loss balancing)
closure_weights=(1.0)
limit_weights=(1.0)
var_weights=(1.0)
var_gammas=(0.1)

echo "===== Grid search started: $(date '+%F %T') =====" | tee -a "$log_file"
failure_count=0

for data_path in "${data_paths[@]}"; do
for model_name in "${models[@]}"; do
for pred_len in "${pred_lens[@]}"; do
for closure_weight in "${closure_weights[@]}"; do
for limit_weight in "${limit_weights[@]}"; do
for var_weight in "${var_weights[@]}"; do
for var_gamma in "${var_gammas[@]}"; do

  # Encoder-decoder models generally need non-zero label length.
  if [ "$model_name" = "Autoformer" ] || [ "$model_name" = "Nonstationary_Transformer" ] || [ "$model_name" = "Informer" ] || [ "$model_name" = "FEDformer" ]; then
    label_len=128
    batch_size=256
  else
    label_len=0
    batch_size=256
  fi

  echo "[RUN] model=${model_name} data=${data_path} pred_len=${pred_len} ll=${label_len} bs=${batch_size} cw=${closure_weight} lw=${limit_weight} vw=${var_weight} vg=${var_gamma}" | tee -a "$log_file"

  if ! python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --debug "$debug_mode" \
    --root_path data \
    --data_path "$data_path" \
    --model_id "zs_${model_name}_256_${pred_len}" \
    --model "$model_name" \
    --data Solar \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --des 'ExpGrid' \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --batch_size "$batch_size" \
    --gpu "$gpu_id" 2>&1 | tee -a "$log_file"; then
    failure_count=$((failure_count + 1))
    echo "[ERROR] model=${model_name} data=${data_path} pred_len=${pred_len} failed, continue to next run" | tee -a "$log_file"
  fi

done
done
done
done
done
done
done

echo "===== Grid search finished: $(date '+%F %T') failures=${failure_count} =====" | tee -a "$log_file"
