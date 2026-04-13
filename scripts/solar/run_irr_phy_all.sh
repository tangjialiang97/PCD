#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/solar_run_all"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
MODE="${1:-parallel}"

scripts=(
  "scripts/solar/irr.sh"
  "scripts/solar/irr2.sh"
  "scripts/solar/phy.sh"
  "scripts/solar/phy2.sh"
)

mkdir -p "$LOG_DIR"

run_one() {
  local script_path="$1"
  local script_name
  local log_path

  script_name="$(basename "$script_path" .sh)"
  log_path="${LOG_DIR}/${TIMESTAMP}_${script_name}.log"

  echo "[START] script=${script_path} log=${log_path}"
  (
    cd "$ROOT_DIR"
    bash "$script_path"
  ) 2>&1 | tee "$log_path"
}

case "$MODE" in
  parallel)
    pids=()
    names=()

    for script_path in "${scripts[@]}"; do
      run_one "$script_path" &
      pids+=("$!")
      names+=("$(basename "$script_path")")
    done

    status=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        echo "[DONE] ${names[$i]}"
      else
        echo "[FAIL] ${names[$i]}" >&2
        status=1
      fi
    done

    exit "$status"
    ;;
  serial)
    for script_path in "${scripts[@]}"; do
      run_one "$script_path"
    done
    ;;
  *)
    echo "Usage: bash scripts/solar/run_irr_phy_all.sh [parallel|serial]" >&2
    exit 1
    ;;
esac

echo "[DONE] All requested solar scripts finished."
