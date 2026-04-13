#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/solar_stage1"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

mkdir -p "$LOG_DIR"

run_pipeline() {
  local gpu_id="$1"
  local pipeline_name="$2"
  shift 2

  cd "$ROOT_DIR"
  export CUDA_VISIBLE_DEVICES="$gpu_id"
  export GPU_ID=0

  for cmd in "$@"; do
    echo "[PIPELINE ${pipeline_name}] $(date '+%F %T') :: ${cmd}"
    eval "$cmd"
  done
}

gpu0_log="${LOG_DIR}/${TIMESTAMP}_gpu0_priority.log"
gpu1_log="${LOG_DIR}/${TIMESTAMP}_gpu1_upper_and_structure.log"

echo "[START] pipeline=gpu0_priority gpu=0 log=${gpu0_log}"
run_pipeline 0 gpu0_priority \
  "bash scripts/solar/physics_stage1_search.sh night_cos_threshold" \
  "bash scripts/solar/physics_stage1_search.sh physics_layers" \
  "bash scripts/solar/physics_stage1_search.sh use_auxiliary" \
  2>&1 | tee "$gpu0_log" &
gpu0_pid=$!

echo "[START] pipeline=gpu1_upper_and_structure gpu=1 log=${gpu1_log}"
run_pipeline 1 gpu1_upper_and_structure \
  "bash scripts/solar/weak_upper_bound_search.sh" \
  "bash scripts/solar/physics_stage1_search.sh physics_hidden_dim" \
  "bash scripts/solar/physics_stage1_search.sh physics_kernel_size" \
  2>&1 | tee "$gpu1_log" &
gpu1_pid=$!

echo "[INFO] GPU0 pipeline PID: ${gpu0_pid}"
echo "[INFO] GPU1 pipeline PID: ${gpu1_pid}"
echo "[INFO] Logs directory: ${LOG_DIR}"

wait "$gpu0_pid"
wait "$gpu1_pid"

echo "[DONE] All stage-1 solar physics experiments finished."
