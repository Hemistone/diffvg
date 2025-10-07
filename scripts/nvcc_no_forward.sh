#!/usr/bin/env bash
set -euo pipefail
real_nvcc="/usr/local/cuda/bin/nvcc"
args=()
for arg in "$@"; do
  if [[ "$arg" == "--forward-unknown-to-host-compiler" ]]; then
    continue
  fi
  args+=("$arg")
fi
exec "$real_nvcc" "${args[@]}"
