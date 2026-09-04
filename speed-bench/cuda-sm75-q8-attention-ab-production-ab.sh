#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export Q8_INTERLEAVED_PRODUCTION_TARGET=attention-ab
if [[ -n ${Q8_ATTN_AB_PRODUCTION_AB_DIR:-} ]]; then
    export Q8_WARP_INTERLEAVED_PRODUCTION_AB_DIR=$Q8_ATTN_AB_PRODUCTION_AB_DIR
fi

exec bash "$repo_dir/speed-bench/cuda-sm75-q8-warp-interleaved-production-ab.sh" "$@"
