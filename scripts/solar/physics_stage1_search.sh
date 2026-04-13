#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

gpu_id="${GPU_ID:-0}"
root_path="${ROOT_PATH:-/opt/data/private/data/timedata/}"
data_path="${DATA_PATH:-Folsom_combined_no_zero.csv}"
search_dim="${1:-night_cos_threshold}"

experiments=(
  "PatchTSTSolarPhysics 12"
  "DLinearSolarPhysics 24"
  "iTransformerSolarPhysics 48"
  "TimeMixerSolarPhysics 168"
)

run_one() {
  local model_name="$1"
  local pred_len="$2"
  local layers="$3"
  local hidden_dim="$4"
  local night_thr="$5"
  local kernel_size="$6"
  local aux_mode="$7"
  local tag="$8"

  local aux_args=()
  if [[ "$aux_mode" == "limit_var" ]]; then
    aux_args=(
      --use_auxiliary
      --has_relhum
      --closure_weight 0.0
      --limit_weight 1.0
      --var_weight 1.0
      --var_gamma 0.1
    )
  fi

  echo "[RUN] model=${model_name} pred_len=${pred_len} search=${search_dim} layers=${layers} hidden=${hidden_dim} night=${night_thr} kernel=${kernel_size} aux=${aux_mode} $(date '+%F %T')"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --is_three \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "stage1_${model_name}_256_${pred_len}" \
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
    --des Stage1Search \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --gpu "$gpu_id" \
    --physics_layers "$layers" \
    --physics_hidden_dim "$hidden_dim" \
    --physics_kernel_size "$kernel_size" \
    --night_cos_threshold "$night_thr" \
    --extra_tag "$tag" \
    "${aux_args[@]}"
}

for exp_cfg in "${experiments[@]}"; do
  read -r model_name pred_len <<< "$exp_cfg"

  case "$search_dim" in
    night_cos_threshold)
      for night_thr in 0.0 0.1 0.2; do
        run_one "$model_name" "$pred_len" 2 128 "$night_thr" 3 off "nct_${night_thr}"
      done
      ;;
    physics_layers)
      for layers in 1 2 3 4; do
        run_one "$model_name" "$pred_len" "$layers" 128 0.0 3 off "phl_${layers}"
      done
      ;;
    use_auxiliary)
      run_one "$model_name" "$pred_len" 2 128 0.0 3 off "aux_off"
      run_one "$model_name" "$pred_len" 2 128 0.0 3 limit_var "aux_limit_var"
      ;;
    physics_hidden_dim)
      for hidden_dim in 64 128 256; do
        run_one "$model_name" "$pred_len" 2 "$hidden_dim" 0.0 3 off "phd_${hidden_dim}"
      done
      ;;
    physics_kernel_size)
      for kernel_size in 3 5; do
        run_one "$model_name" "$pred_len" 2 128 0.0 "$kernel_size" off "pks_${kernel_size}"
      done
      ;;
    *)
      echo "Unsupported search dimension: ${search_dim}" >&2
      echo "Use one of: night_cos_threshold, physics_layers, use_auxiliary, physics_hidden_dim, physics_kernel_size" >&2
      exit 1
      ;;
  esac
done
