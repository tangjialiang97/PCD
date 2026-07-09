#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
set -uo pipefail

debug_mode=False
gpu_id=0
log_file="scripts/solar/power.log"

models=(
  PatchTSTSolarPhysics
  EMAformerSolarPhysics
  DLinearSolarPhysics
  mixlinearSolarPhysics
)

# GHI->power map modes. residual_mlp learns a correction on top of the prior.
# Pair with --solar_power_prior {linear,pvwatts}.
map_modes=(
  residual_mlp
)
prior_mode="${prior_mode:-linear}"
power_output="${solar_power_output:-append}"
fusion_mode="${fusion_mode:-learned}"
direct_head="${direct_head:-linear}"
channel_count=4

if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
else
  pred_lens=(24 48 96 192)
fi

data_paths=("Congo_power.csv")

echo "===== Solar power runs started: $(date '+%F %T') =====" | tee -a "$log_file"
failure_count=0

for data_path in "${data_paths[@]}"; do
for model_name in "${models[@]}"; do
for map_mode in "${map_modes[@]}"; do
for pred_len in "${pred_lens[@]}"; do

  if [ "$model_name" = "Autoformer" ] || [ "$model_name" = "Nonstationary_Transformer" ] || [ "$model_name" = "Informer" ] || [ "$model_name" = "FEDformer" ]; then
    label_len=128
    batch_size=256
  else
    label_len=0
    batch_size=512
  fi

  echo "[RUN] model=${model_name} data=${data_path} map=${map_mode} fusion=${fusion_mode} output=${power_output} pred_len=${pred_len} ll=${label_len} bs=${batch_size}" | tee -a "$log_file"

  if ! python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --debug "$debug_mode" \
    --root_path ./data/solar/ \
    --data_path "$data_path" \
    --model_id "256_${pred_len}_power_${power_output}_${fusion_mode}" \
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
    --enc_in "$channel_count" \
    --dec_in "$channel_count" \
    --c_out "$channel_count" \
    --des 'Power' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --solar_power_output "$power_output" \
    --solar_power_map "$map_mode" \
    --solar_power_prior "$prior_mode" \
    --solar_power_fusion "$fusion_mode" \
    --solar_power_direct_head "$direct_head" \
    --solar_power_map_hidden 16 \
    --solar_power_map_dropout 0.0 \
    --solar_power_capacity 1.0 \
    --solar_power_ghi_ref 1000.0 \
    --solar_power_clip False \
    --gpu "$gpu_id" 2>&1 | tee -a "$log_file"; then
    failure_count=$((failure_count + 1))
    echo "[ERROR] model=${model_name} data=${data_path} map=${map_mode} pred_len=${pred_len} failed, continue to next run" | tee -a "$log_file"
  fi

done
done
done
done

echo "===== Solar power runs finished: $(date '+%F %T') failures=${failure_count} =====" | tee -a "$log_file"
