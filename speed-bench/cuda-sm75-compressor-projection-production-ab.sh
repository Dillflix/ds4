#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export COMPRESSOR_STATE_AB_AXIS=projection-layout
exec bash "$script_dir/cuda-sm75-compressor-state-production-ab.sh" "$@"
