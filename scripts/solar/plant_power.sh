#!/usr/bin/env bash
# Example plant-power training recipe:
#   1) offline pvlib columns from plant + weather tables
#   2) SolarPhysics forecast of [ghi,dni,dhi] (dni/dhi via Erbs offline)
#   3) PVWatts prior + residual_mlp map to observe_power
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${gpu_id:-0}"
debug_mode="${debug_mode:-False}"
log_file="scripts/solar/plant_power.log"

# --- paths you should edit ---
plant_table="${plant_table:-dataset_expample/qingda.xlsx}"
weather_table="${weather_table:-}"
out_csv="${out_csv:-data/solar/Qingda_plant_power.csv}"
lat="${lat:-29.919}"
lon="${lon:-100.641}"
tz="${tz:-Asia/Shanghai}"

models=(
  DLinearSolarPhysics
  PatchTSTSolarPhysics
)

map_modes=(
  residual_mlp
)

if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
else
  pred_lens=(24 48 96)
fi

echo "===== Plant power workflow started: $(date '+%F %T') =====" | tee -a "$log_file"

echo "[1/2] Build offline pvlib dataset -> ${out_csv}" | tee -a "$log_file"
weather_args=()
if [ -n "$weather_table" ]; then
  weather_args=(--weather "$weather_table")
fi

if ! python -u scripts/solar/build_plant_power_dataset.py \
  --plant "$plant_table" \
  "${weather_args[@]}" \
  --output "$out_csv" \
  --latitude "$lat" \
  --longitude "$lon" \
  --timezone "$tz" \
  --temp-cell-method faiman \
  --add-erbs-components True \
  --include-linear-prior True 2>&1 | tee -a "$log_file"; then
  echo "[ERROR] offline pvlib dataset build failed" | tee -a "$log_file"
  exit 1
fi

data_path="$(basename "$out_csv")"
failure_count=0

echo "[2/2] Train power models on ${data_path}" | tee -a "$log_file"
for model_name in "${models[@]}"; do
for map_mode in "${map_modes[@]}"; do
for pred_len in "${pred_lens[@]}"; do
  label_len=0
  batch_size=256
  echo "[RUN] model=${model_name} map=${map_mode} prior=pvwatts pred_len=${pred_len}" | tee -a "$log_file"
  if ! python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --debug "$debug_mode" \
    --root_path ./data/solar/ \
    --data_path "$data_path" \
    --model_id "plant_${pred_len}_${map_mode}" \
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
    --des 'PlantPower' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --solar_power_output power \
    --solar_power_map "$map_mode" \
    --solar_power_prior pvwatts \
    --solar_power_temp_cell_method noct \
    --solar_power_gamma_pdc -0.004 \
    --solar_power_capacity 1.0 \
    --solar_power_ghi_ref 1000.0 \
    --gpu "$gpu_id" 2>&1 | tee -a "$log_file"; then
    failure_count=$((failure_count + 1))
    echo "[ERROR] model=${model_name} map=${map_mode} pred_len=${pred_len} failed" | tee -a "$log_file"
  fi
done
done
done

echo "===== Plant power workflow finished: $(date '+%F %T') failures=${failure_count} =====" | tee -a "$log_file"
