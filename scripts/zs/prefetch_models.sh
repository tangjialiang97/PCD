#!/usr/bin/env bash
export TSL_HF_CACHE_DIR="${TSL_HF_CACHE_DIR:-/opt/data/private/huggingface_models}"
export HF_HOME="${HF_HOME:-$TSL_HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TSL_HF_CACHE_DIR/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TSL_HF_CACHE_DIR/transformers}"
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$TSL_HF_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

echo "===== Prefetch started: $(date '+%F %T') ====="
echo "Python: $PYTHON_BIN"
echo "Cache root: $TSL_HF_CACHE_DIR"

"$PYTHON_BIN" - <<'PY'
import os
import sys

from huggingface_hub import snapshot_download


MODELS = [
    ("TiRex", "NX-AI/TiRex"),
    ("Chronos", "amazon/chronos-bolt-base"),
    ("Chronos2", "amazon/chronos-2"),
]


def main():
    cache_dir = os.environ["TSL_HF_CACHE_DIR"]
    print(f"[INFO] Using cache directory: {cache_dir}", flush=True)

    failures = []
    for model_name, repo_id in MODELS:
        print(f"[START] {model_name}: checking local snapshot for {repo_id}", flush=True)
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            print(f"[HIT]   {model_name}: {local_path}", flush=True)
            continue
        except Exception:
            pass

        print(f"[MISS]  {model_name}: downloading {repo_id}", flush=True)
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                resume_download=True,
            )
            print(f"[DONE]  {model_name}: {local_path}", flush=True)
        except Exception as exc:
            failures.append((model_name, repo_id, exc))
            print(f"[FAIL]  {model_name}: {exc}", flush=True)

    if failures:
        print("[SUMMARY] Some downloads failed:", flush=True)
        for model_name, repo_id, exc in failures:
            print(f"  - {model_name} ({repo_id}): {exc}", flush=True)
        sys.exit(1)

    print("[SUMMARY] All model snapshots are ready for local reuse.", flush=True)


if __name__ == "__main__":
    main()
PY

echo "===== Prefetch finished: $(date '+%F %T') ====="
