#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ] && [ "$(basename -- "$1")" = "nvcc" ]; then
real_nvcc="$1"
shift
else
real_nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
fi

args=()
for a in "$@"; do
if [ "$a" = "-forward-unknown-to-host-compiler" ] || [ "$a" = "--forward-unknown-to-host-compiler" ]; then
continue
fi
args+=("$a")
done

exec "$real_nvcc" "${args[@]}"