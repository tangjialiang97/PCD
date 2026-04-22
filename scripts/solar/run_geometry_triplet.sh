#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

TAG="${TAG:-timefilter_folsom_pl96}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/solar}"
DATA_PATH="${DATA_PATH:-Folsom_combined_no_zero.csv}"
BASELINE_CHECKPOINTS_ROOT="${BASELINE_CHECKPOINTS_ROOT:-/opt/data/private/code/Time-Series-Library-fm/checkpoints}"
SOFT_CHECKPOINTS_ROOT="${SOFT_CHECKPOINTS_ROOT:-$BASELINE_CHECKPOINTS_ROOT}"
PCD_CHECKPOINTS_ROOT="${PCD_CHECKPOINTS_ROOT:-$ROOT_DIR/checkpoints}"

SEQ_LEN="${SEQ_LEN:-256}"
PRED_LEN="${PRED_LEN:-96}"
ENC_IN="${ENC_IN:-3}"
DEC_IN="${DEC_IN:-3}"
C_OUT="${C_OUT:-3}"
D_MODEL="${D_MODEL:-128}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
FACTOR="${FACTOR:-3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FEATURES="${FEATURES:-M}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
STRICT_LOAD="${STRICT_LOAD:-true}"

RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/results_geometry_triplet/$TAG}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/solar_geometry_triplet}"

BASELINE_MODEL="${BASELINE_MODEL:-TimeFilter}"
BASELINE_MODEL_ID="${BASELINE_MODEL_ID:-${SEQ_LEN}_${PRED_LEN}}"
BASELINE_CHECKPOINT_PATH="${BASELINE_CHECKPOINT_PATH:-}"
BASELINE_PAYLOAD="${BASELINE_PAYLOAD:-}"

SOFT_MODEL="${SOFT_MODEL:-TimeFilter}"
SOFT_MODEL_ID="${SOFT_MODEL_ID:-${SEQ_LEN}_${PRED_LEN}}"
SOFT_CHECKPOINT_PATH="${SOFT_CHECKPOINT_PATH:-}"
SOFT_PAYLOAD="${SOFT_PAYLOAD:-}"

PCD_MODEL="${PCD_MODEL:-TimeFilterSolarPhysics}"
PCD_MODEL_ID="${PCD_MODEL_ID:-${SEQ_LEN}_${PRED_LEN}}"
PCD_CHECKPOINT_PATH="${PCD_CHECKPOINT_PATH:-}"
PCD_PAYLOAD="${PCD_PAYLOAD:-}"

GEOM_OUTPUT_DIR="${GEOM_OUTPUT_DIR:-$RESULT_ROOT/compare}"
GEOM_OUTPUT_PNG="${GEOM_OUTPUT_PNG:-$GEOM_OUTPUT_DIR/geometry_failure_triplet.png}"
GEOM_SUMMARY_JSON="${GEOM_SUMMARY_JSON:-$GEOM_OUTPUT_DIR/geometry_failure_triplet.json}"

mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$GEOM_OUTPUT_DIR"

dataset_code_for_role() {
  local role="$1"
  local data_path="$2"
  case "$data_path" in
    Folsom_combined_no_zero.csv)
      if [[ "$role" == "pcd" ]]; then
        echo "dpfols-comb"
      else
        echo "dpfols-cnz"
      fi
      ;;
    Folsom_combined.csv)
      echo "dpfols-comb"
      ;;
    Congo.csv|Congo_no_zero.csv)
      echo "dpcong"
      ;;
    Germany.csv|Germany_no_zero.csv)
      echo "dpgerm"
      ;;
    NewDelhi.csv|NewDelhi_no_zero.csv)
      echo "dpnewd"
      ;;
    *)
      echo ""
      ;;
  esac
}

resolve_default_checkpoint() {
  local role="$1"
  local model="$2"

  local dataset_code
  dataset_code="$(dataset_code_for_role "$role" "$DATA_PATH")"
  if [[ -z "$dataset_code" ]]; then
    echo "Unable to infer dataset code for DATA_PATH=$DATA_PATH role=$role" >&2
    return 1
  fi

  local root_dir
  local pattern
  case "$role" in
    baseline)
      root_dir="$BASELINE_CHECKPOINTS_ROOT"
      pattern="${root_dir}/ltf_${SEQ_LEN}_${PRED_LEN}_${model}_Solar_${dataset_code}"'*/checkpoint.pth'
      ;;
    soft)
      root_dir="$SOFT_CHECKPOINTS_ROOT"
      pattern="${root_dir}/ltf_irr_soft_${SEQ_LEN}_${PRED_LEN}_${model}_Solar_${dataset_code}"'*/checkpoint.pth'
      ;;
    pcd)
      root_dir="$PCD_CHECKPOINTS_ROOT"
      pattern="${root_dir}/ltf_${SEQ_LEN}-v2-${PRED_LEN}_${model}_Solar_${dataset_code}"'*/checkpoint.pth'
      ;;
    *)
      echo "Unsupported role for checkpoint resolution: $role" >&2
      return 1
      ;;
  esac

  local matches=()
  shopt -s nullglob
  matches=( $pattern )
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "Failed to auto-resolve checkpoint for role=$role model=$model pattern=$pattern" >&2
    return 1
  fi
  printf '%s\n' "${matches[@]}" | sort | tail -n 1
}

label_len_for_model() {
  local model="$1"
  case "$model" in
    Autoformer|Informer|FEDformer|Nonstationary_Transformer|AutoformerSolarPhysics|InformerSolarPhysics|FEDformerSolarPhysics|Nonstationary_TransformerSolarPhysics)
      echo 128
      ;;
    *)
      echo 0
      ;;
  esac
}

run_eval() {
  local role="$1"
  local model="$2"
  local model_id="$3"
  local checkpoint_path="$4"
  local payload_override="$5"

  if [[ -n "$payload_override" ]]; then
    echo "$payload_override"
    return 0
  fi

  if [[ -z "$checkpoint_path" || ! -f "$checkpoint_path" ]]; then
    echo "Missing checkpoint for role=$role: $checkpoint_path" >&2
    exit 1
  fi

  local label_len
  local role_results_dir
  local log_file
  label_len="$(label_len_for_model "$model")"
  role_results_dir="$RESULT_ROOT/$role"
  mkdir -p "$role_results_dir"
  log_file="$LOG_DIR/${TAG}_${role}.log"

  local -a cmd=(
    python -u run_test.py
    --task_name long_term_forecast
    --is_training 0
    --model_id "$model_id"
    --model "$model"
    --data Solar
    --root_path "$DATA_ROOT"
    --data_path "$DATA_PATH"
    --features "$FEATURES"
    --seq_len "$SEQ_LEN"
    --label_len "$label_len"
    --pred_len "$PRED_LEN"
    --enc_in "$ENC_IN"
    --dec_in "$DEC_IN"
    --c_out "$C_OUT"
    --d_model "$D_MODEL"
    --d_ff "$D_FF"
    --e_layers "$E_LAYERS"
    --d_layers "$D_LAYERS"
    --factor "$FACTOR"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --des "GeomTriplet-${role}"
    --itr 1
    --gpu "$GPU_ID"
    --checkpoint_path "$checkpoint_path"
    --strict_load "$STRICT_LOAD"
    --eval_split "$EVAL_SPLIT"
    --results_dir "$role_results_dir"
    --analysis_solar_pcd true
    --compare_projection_stages true
    --save_analysis_tables true
    --save_analysis_plots true
    --save_case_studies false
    --save_solar_geometry_payload true
    --save_internal_projection_geometry false
    --is_three
  )

  if [[ "$model" == *"SolarPhysics"* ]]; then
    cmd+=(--use_internal_physics)
  fi

  echo "[RUN] role=$role model=$model ckpt=$checkpoint_path log=$log_file" >&2
  "${cmd[@]}" 2>&1 | tee "$log_file" >&2

  local payload_path
  payload_path="$(find "$role_results_dir" -path '*/analysis/solar_geometry_payload.npz' | sort | tail -n 1)"
  if [[ -z "$payload_path" ]]; then
    echo "Failed to locate solar_geometry_payload.npz for role=$role under $role_results_dir" >&2
    exit 1
  fi
  echo "$payload_path"
}

if [[ ! -f "$DATA_ROOT/$DATA_PATH" ]]; then
  echo "Dataset not found: $DATA_ROOT/$DATA_PATH" >&2
  exit 1
fi

if [[ -z "$BASELINE_CHECKPOINT_PATH" && -z "$BASELINE_PAYLOAD" ]]; then
  BASELINE_CHECKPOINT_PATH="$(resolve_default_checkpoint baseline "$BASELINE_MODEL")"
fi

if [[ -z "$SOFT_CHECKPOINT_PATH" && -z "$SOFT_PAYLOAD" ]]; then
  SOFT_CHECKPOINT_PATH="$(resolve_default_checkpoint soft "$SOFT_MODEL" || true)"
fi

if [[ -z "$PCD_CHECKPOINT_PATH" && -z "$PCD_PAYLOAD" ]]; then
  PCD_CHECKPOINT_PATH="$(resolve_default_checkpoint pcd "$PCD_MODEL")"
fi

if [[ -z "$SOFT_PAYLOAD" && -z "$SOFT_CHECKPOINT_PATH" ]]; then
  echo "SOFT_CHECKPOINT_PATH or SOFT_PAYLOAD must be provided for the baseline+Soft run." >&2
  echo "Auto-resolution failed for DATA_PATH=$DATA_PATH, PRED_LEN=$PRED_LEN, SOFT_MODEL=$SOFT_MODEL." >&2
  echo "Example:" >&2
  echo "  SOFT_CHECKPOINT_PATH=/abs/path/to/soft/checkpoint.pth bash scripts/solar/run_geometry_triplet.sh" >&2
  exit 1
fi

BASELINE_PAYLOAD_PATH="$(run_eval baseline "$BASELINE_MODEL" "$BASELINE_MODEL_ID" "$BASELINE_CHECKPOINT_PATH" "$BASELINE_PAYLOAD")"
SOFT_PAYLOAD_PATH="$(run_eval soft "$SOFT_MODEL" "$SOFT_MODEL_ID" "$SOFT_CHECKPOINT_PATH" "$SOFT_PAYLOAD")"
PCD_PAYLOAD_PATH="$(run_eval pcd "$PCD_MODEL" "$PCD_MODEL_ID" "$PCD_CHECKPOINT_PATH" "$PCD_PAYLOAD")"

python scripts/solar/plot_geometry_triplet.py \
  --baseline "$BASELINE_PAYLOAD_PATH" \
  --soft "$SOFT_PAYLOAD_PATH" \
  --pcd "$PCD_PAYLOAD_PATH" \
  --output "$GEOM_OUTPUT_PNG" \
  --summary-json "$GEOM_SUMMARY_JSON"

echo
echo "Finished triplet geometry comparison."
echo "  baseline payload: $BASELINE_PAYLOAD_PATH"
echo "  soft payload    : $SOFT_PAYLOAD_PATH"
echo "  pcd payload     : $PCD_PAYLOAD_PATH"
echo "  figure          : $GEOM_OUTPUT_PNG"
