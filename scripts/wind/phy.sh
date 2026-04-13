#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0

debug_mode="${DEBUG_MODE:-False}"
train_epochs="${TRAIN_EPOCHS:-20}"
seq_len="${SEQ_LEN:-256}"
wind_cap_quantile="${WIND_CAP_QUANTILE:-0.995}"
wind_ramp_quantile="${WIND_RAMP_QUANTILE:-0.995}"
wind_state_bins="${WIND_STATE_BINS:-4}"

models=(
  "DLinearWindPhysicsV6"
  "iTransformerWindPhysicsV6"
)

if [ "${debug_mode}" = "True" ]; then
  pred_lens=(12)
  data_paths=("101446_20070101_20131231_met.csv")
else
  pred_lens=(24 48 96 168)
  data_paths=(
    "101446_20070101_20131231_met.csv"
  )
fi

for model_name in "${models[@]}"; do
  case "${model_name}" in
    DLinearWindPhysicsV6)
      batch_size="${BATCH_SIZE_DLINEAR:-256}"
      extra_args=(
        --d_model 128
        --d_ff 128
      )
      ;;
    iTransformerWindPhysicsV6)
      batch_size="${BATCH_SIZE_ITRANSFORMER:-128}"
      extra_args=(
        --e_layers 2
        --d_layers 1
        --factor 3
        --n_heads 4
        --d_model 128
        --d_ff 128
      )
      ;;
    *)
      echo "Unsupported model: ${model_name}" >&2
      exit 1
      ;;
  esac

  for data_path in "${data_paths[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
      python -u "${REPO_ROOT}/run.py" \
        --task_name long_term_forecast \
        --is_training 1 \
        --debug "${debug_mode}" \
        --root_path "${REPO_ROOT}/data/wind" \
        --data_path "${data_path}" \
        --model_id "windphyv3_${model_name}_${seq_len}_${pred_len}" \
        --model "${model_name}" \
        --data Wind \
        --features MS \
        --target power \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --label_len 0 \
        --batch_size "${batch_size}" \
        --train_epochs "${train_epochs}" \
        --learning_rate 0.001 \
        --patience 3 \
        --enc_in 6 \
        --dec_in 6 \
        --c_out 6 \
        --des "WindPhysicsV6" \
        --itr 1 \
        --wind_cap_quantile "${wind_cap_quantile}" \
        --wind_ramp_quantile "${wind_ramp_quantile}" \
        --wind_state_bins "${wind_state_bins}" \
        "${extra_args[@]}"
    done
  done
done
