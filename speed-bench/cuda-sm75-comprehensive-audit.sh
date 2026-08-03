#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the remaining SM75 evidence pass in one command:
  - full stock-Q2 fixed-suite benchmark, tile/cache coverage, and Nsys trace;
  - bounded NCU captures for early/late Q2_K down and all four Q8 templates.

Required environment:
  MODEL_Q2=/absolute/path/to/stock-Q2.gguf

Common optional environment:
  PROMPT=/absolute/path/prompt.txt
  PROMPT_MANIFEST=/absolute/path/prompts.tsv
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  PROFILE_GPU=0
  NCU_USE_SUDO=0
  NCU_SET=focused
  PROFILE_SET=remaining       remaining or all
  SKIP_BUILD=0
  AUDIT_ROOT=/absolute/output/directory

PROFILE_SET=all repeats every routed-expert family and all four native-Q8
templates under the current build. The default 'remaining' captures Q2_K down
and all four Q8 templates without repeating Q4/IQ2 expert reports.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL_Q2:?set MODEL_Q2 to the absolute stock-Q2 GGUF path}"
[[ -f $MODEL_Q2 ]] || die "stock-Q2 model not found: $MODEL_Q2"

SKIP_BUILD=${SKIP_BUILD:-0}
PROFILE_SET=${PROFILE_SET:-remaining}
PROFILE_GPU=${PROFILE_GPU:-0}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
NCU_SET=${NCU_SET:-focused}
ROOT=${AUDIT_ROOT:-$repo_dir/sm75-comprehensive-audit-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $PROFILE_SET == remaining || $PROFILE_SET == all ]] ||
    die "PROFILE_SET must be remaining or all"
[[ ! -e $ROOT ]] || die "output path already exists: $ROOT"
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    if [[ -d $ROOT ]]; then
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$ROOT/run-status.txt"
        local archive="$ROOT.tar.gz"
        if tar -C "$(dirname "$ROOT")" -czf "$archive" "$(basename "$ROOT")"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
        fi
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'model_q2=%s\nprofile_set=%s\nprofile_gpu=%s\n' \
        "$MODEL_Q2" "$PROFILE_SET" "$PROFILE_GPU"
    printf 'ncu_set=%s\nncu_use_sudo=%s\n' "$NCU_SET" "$NCU_USE_SUDO"
    printf '\n[git status]\n'; git status --short
} >"$ROOT/manifest.txt"

current_phase=stock-q2-production
Q2_EVIDENCE_DIR="$ROOT/stock-q2-production" \
CREATE_ARCHIVE=0 SKIP_BUILD="$SKIP_BUILD" \
    ./speed-bench/cuda-q2-prefill-evidence.sh

current_phase=bounded-ncu
PROFILE_DIR="$ROOT/bounded-ncu" CREATE_ARCHIVE=0 SKIP_BUILD=1 \
PROFILE_SET="$PROFILE_SET" PROFILE_GPU="$PROFILE_GPU" \
NCU_USE_SUDO="$NCU_USE_SUDO" NCU_SET="$NCU_SET" \
    ./speed-bench/cuda-sm75-kernel-profile.sh

current_phase=complete
printf 'Comprehensive SM75 evidence pass complete: %s\n' "$ROOT"
