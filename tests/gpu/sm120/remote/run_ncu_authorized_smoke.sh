#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_E2E_ALLOW_SYS_ADMIN:-0}" != "1" ]]; then
  echo "set CUDA_E2E_ALLOW_SYS_ADMIN=1 only after explicit profiling authorization" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../../../.." && pwd -P)"
requested_ref="${CUDA_CURRENT_IMAGE:-cuda-skill-current:cuda13.3-triton3.7.1-ncu2026.2.1}"
gpu="${CUDA_E2E_GPU:-0}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "CUDA_E2E_GPU must be a non-negative GPU index" >&2
  exit 2
fi

gpu_uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)"
assert_gpu_idle() {
  local busy
  busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
    | awk -F, -v uuid="$gpu_uuid" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)} $1 == uuid {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')"
  if [[ -n "$busy" ]]; then
    echo "refusing busy GPU $gpu ($gpu_uuid), compute PIDs: $busy" >&2
    exit 3
  fi
}

assert_gpu_idle

resolved_image_id="$(docker image inspect --format '{{.Id}}' "$requested_ref")"
if [[ ! "$resolved_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "image did not resolve to an immutable sha256 ID: $requested_ref" >&2
  exit 2
fi
if [[ "$(docker image inspect --format '{{.Id}}' "$resolved_image_id")" != "$resolved_image_id" ]]; then
  echo "immutable image identity changed during inspection" >&2
  exit 2
fi

assert_gpu_idle
exec docker run --rm \
  --pull never \
  --gpus "device=$gpu" \
  --network none \
  --cap-drop ALL \
  --cap-add SYS_ADMIN \
  --pids-limit 256 \
  --ipc private \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/tmp \
  -v "$repo_root:$repo_root:ro" \
  -w "$repo_root" \
  "$resolved_image_id" \
  ncu --target-processes all --set basic --launch-count 1 \
    python3 tests/gpu/sm120/fixtures/ncu_smoke.py
