#!/usr/bin/env bash
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

gpu_id="${GPU_ID:-0}"
root_path="${ROOT_PATH:-/opt/data/private/data/timedata/}"
data_path="${DATA_PATH:-Folsom_combined_no_zero.csv}"
log_file="${LOG_FILE:-scripts/solar/cmp_seq24_pl1.log}"

seq_len=24
pred_len=1
label_len=0
batch_size=512

base_models=(PatchTST DLinear iTransformer TimeMixer TimesNet Crossformer TSMixer PAttn TimeFilter)

solar_model_name() {
  local base_model="$1"
  case "$base_model" in
    PatchTST) echo "PatchTSTSolarPhysics" ;;
    DLinear) echo "DLinearSolarPhysics" ;;
    iTransformer) echo "iTransformerSolarPhysics" ;;
    TimeMixer) echo "TimeMixerSolarPhysics" ;;
    TimesNet) echo "TimesNetSolarPhysics" ;;
    Crossformer) echo "CrossformerSolarPhysics" ;;
    TSMixer) echo "TSMixerSolarPhysics" ;;
    PAttn) echo "PAttnSolarPhysics" ;;
    TimeFilter) echo "TimeFilterSolarPhysics" ;;
    *)
      echo "Unsupported base model: ${base_model}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local base_model="$1"
  local mode="$2"
  local model_name
  local model_id
  local des
  local status=0
  local extra_args=()

  case "$mode" in
    baseline)
      model_name="$base_model"
      model_id="cmpSeq24Pl1_${base_model}_baseline"
      des="CmpSeq24Pl1_Base"
      ;;
    sp)
      model_name="$(solar_model_name "$base_model")" || return 1
      model_id="cmpSeq24Pl1_${base_model}_sp"
      des="CmpSeq24Pl1_SP"
      extra_args+=(
        --physics_layers 2
        --physics_hidden_dim 128
        --physics_kernel_size 3
        --night_cos_threshold 0.0
      )
      ;;
    sp_aux)
      model_name="$(solar_model_name "$base_model")" || return 1
      model_id="cmpSeq24Pl1_${base_model}_sp_aux"
      des="CmpSeq24Pl1_SPAux"
      extra_args+=(
        --use_auxiliary
        --has_relhum
        --physics_layers 2
        --physics_hidden_dim 128
        --physics_kernel_size 3
        --night_cos_threshold 0.0
      )
      ;;
    *)
      echo "Unsupported mode: ${mode}" >&2
      return 1
      ;;
  esac

  echo "[RUN] model=${base_model} mode=${mode} seq_len=${seq_len} pred_len=${pred_len} $(date '+%F %T')" | tee -a "$log_file"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --is_three \
    --closure_weight 0.1 \
    --limit_weight 1.0 \
    --var_weight 1.0 \
    --var_gamma 0.1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model "$model_name" \
    --data Solar \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --label_len "$label_len" \
    --batch_size "$batch_size" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --des "$des" \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --gpu "$gpu_id" \
    "${extra_args[@]}" 2>&1 | tee -a "$log_file"

  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[FAIL] model=${base_model} mode=${mode} exit_code=${status} $(date '+%F %T')" | tee -a "$log_file"
  else
    echo "[DONE] model=${base_model} mode=${mode} $(date '+%F %T')" | tee -a "$log_file"
  fi

  return 0
}

echo "===== seq_len=24 pred_len=1 comparison started: $(date '+%F %T') =====" | tee -a "$log_file"

for base_model in "${base_models[@]}"; do
  run_one "$base_model" baseline
  run_one "$base_model" sp
  run_one "$base_model" sp_aux
done

echo "===== seq_len=24 pred_len=1 comparison finished: $(date '+%F %T') =====" | tee -a "$log_file"
