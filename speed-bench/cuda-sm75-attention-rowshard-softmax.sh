#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

PROFILE_GPU=${PROFILE_GPU:-0}
ROWS=${ROWS:-8192}
TOKENS=${TOKENS:-32}
TIMING_ROUNDS=${TIMING_ROUNDS:-9}
TIMING_REPEATS=${TIMING_REPEATS:-25}
RUN_SANITIZER=${RUN_SANITIZER:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CUDA_ARCH=${CUDA_ARCH:-sm_75}

for value in "$PROFILE_GPU" "$ROWS" "$TOKENS" \
             "$TIMING_ROUNDS" "$TIMING_REPEATS"; do
    [[ $value =~ ^[0-9]+$ ]] || die "numeric settings must be nonnegative integers"
done
for value in "$RUN_SANITIZER" "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "boolean settings must be 0 or 1"
done
(( ROWS >= 512 && TOKENS > 0 && TIMING_ROUNDS > 0 && TIMING_REPEATS > 0 )) ||
    die "ROWS must be at least 512; TOKENS, rounds, and repeats must be positive"
[[ $CUDA_ARCH == sm_75 ]] || die "this experiment requires CUDA_ARCH=sm_75"

for tool in make cuobjdump nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool was not found"
done
if (( RUN_SANITIZER )); then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
fi
if (( CREATE_ARCHIVE )); then
    command -v tar >/dev/null 2>&1 || die "CREATE_ARCHIVE=1 but tar was not found"
fi

target=tests/cuda_sm75_attention_rowshard_softmax
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${ATTENTION_ROWSHARD_SOFTMAX_DIR:-"$PWD/sm75-attention-rowshard-softmax-$timestamp"}
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

archive_on_exit() {
    local rc=$?
    local archive="$OUTPUT_DIR.tar.gz"
    trap - EXIT
    if (( CREATE_ARCHIVE )) && [[ -d $OUTPUT_DIR ]]; then
        if tar -czf "$archive" -C "$(dirname "$OUTPUT_DIR")" \
                "$(basename "$OUTPUT_DIR")"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            printf 'warning: failed to create archive: %s\n' "$archive" >&2
        fi
    fi
    exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

git rev-parse HEAD >"$OUTPUT_DIR/git-head.txt"
git status --short >"$OUTPUT_DIR/git-status.txt"
nvidia-smi -i "$PROFILE_GPU" -q >"$OUTPUT_DIR/nvidia-smi-q.txt"

if (( ! SKIP_BUILD )); then
    make -B -j"$(nproc)" "$target" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/build.log" >&2
            die "build failed"
        }
else
    [[ -x $target ]] || die "$target is missing; rerun with SKIP_BUILD=0"
    printf 'skipped explicitly: SKIP_BUILD=1\n' >"$OUTPUT_DIR/build.log"
fi

cuobjdump --dump-resource-usage "$target" >"$OUTPUT_DIR/resource-usage.txt"
cuobjdump --dump-sass "$target" >"$OUTPUT_DIR/sass.txt"

"./$target" --device "$PROFILE_GPU" --rows 769 --tokens 2 \
    --rounds 1 --repeats 1 >"$OUTPUT_DIR/adversarial-smoke.log" 2>&1
grep -q '^ordered_local_address_bit_mismatches=0$' \
    "$OUTPUT_DIR/adversarial-smoke.log" || die "owner-local addressing was not exact"
grep -q '^harness_status=ok$' "$OUTPUT_DIR/adversarial-smoke.log" ||
    die "adversarial smoke failed"

if (( RUN_SANITIZER )); then
    compute-sanitizer --tool memcheck --error-exitcode 97 \
        "./$target" --device "$PROFILE_GPU" --rows 769 --tokens 2 \
        --rounds 1 --repeats 1 >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/memcheck.log" >&2
            die "compute-sanitizer failed"
        }
else
    printf 'skipped explicitly: RUN_SANITIZER=0\n' >"$OUTPUT_DIR/memcheck.log"
fi

"./$target" --device "$PROFILE_GPU" --rows "$ROWS" --tokens "$TOKENS" \
    --rounds "$TIMING_ROUNDS" --repeats "$TIMING_REPEATS" \
    >"$OUTPUT_DIR/timing.log" 2>&1
grep -q '^ordered_local_address_bit_mismatches=0$' "$OUTPUT_DIR/timing.log" ||
    die "timed owner-local addressing was not exact"
grep -q '^harness_status=ok$' "$OUTPUT_DIR/timing.log" ||
    die "timed row-shard experiment failed"

printf 'SM75 row-sharded compact-attention softmax prototype complete: %s\n' \
    "$OUTPUT_DIR"
tail -n 40 "$OUTPUT_DIR/timing.log"
