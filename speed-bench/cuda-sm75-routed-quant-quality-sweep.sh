#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/speed-bench/cuda-sm75-q3-q4-real-quality.sh" "$@"
