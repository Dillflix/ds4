#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Measure pre-materialized FP16 operands for the retained SM75 indexer WMMA128
kernel without opening a GGUF.

The final production-shaped 512-token microbatch at 32K is used. Every score
bit and the ordered top-512 set must match shipping inline-conversion WMMA128.
Paired timing alternates execution order and reports both kernel-only speedup
and end-to-end Q-materialization + score speedup. Persistent-K conversion is
reported separately because production would update that sidecar at cache
write time, not reconvert the full history for each score call.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=5
  TIMING_REPEATS=10
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  INDEXER_F16_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-5}
TIMING_REPEATS=${TIMING_REPEATS:-10}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${INDEXER_F16_DIR:-$repo_dir/sm75-indexer-f16-operands-$stamp}

for item in "PROFILE_GPU:$PROFILE_GPU" "TIMING_ROUNDS:$TIMING_ROUNDS" \
            "TIMING_REPEATS:$TIMING_REPEATS" "RUN_NCU:$RUN_NCU" \
            "NCU_USE_SUDO:$NCU_USE_SUDO" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( TIMING_ROUNDS >= 3 && TIMING_ROUNDS <= 15 )) ||
    die "TIMING_ROUNDS must be in 3..15"
(( TIMING_REPEATS >= 2 && TIMING_REPEATS <= 100 )) ||
    die "TIMING_REPEATS must be in 2..100"
for flag in RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
for tool in awk bash basename date dirname git grep id make mkdir mv nproc \
            nvidia-smi python3 sort tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ && $free_mib -ge 4096 ]] ||
    die "physical GPU $PROFILE_GPU requires at least 4096 MiB free"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/timing" "$OUTPUT_DIR/ncu" "$OUTPUT_DIR/validation"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'error: could not create archive %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" tests/cuda_sm75_profile_harness \
        CUDA_ARCH="$CUDA_ARCH" \
        2>&1 | tee "$OUTPUT_DIR/validation/build.log"
else
    make -q tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale profile harness"
fi
[[ -x tests/cuda_sm75_profile_harness ]] || die "profile harness is missing"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncuda_arch=%s\ncompute_capability=%s\n' \
        "$PROFILE_GPU" "$CUDA_ARCH" "$compute_cap"
    printf 'free_mib_at_preflight=%s\ntiming_rounds=%s\ntiming_repeats=%s\n' \
        "$free_mib" "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'shape=512x64x128-by-8192\nposition=32256\ntop_k=512\n'
    printf 'reference=shipping-inline-wmma128\ncandidate=materialized-f16-wmma128\n'
    printf 'candidate_q_conversion=included\ncandidate_full_k_reconversion=excluded\n'
    printf 'run_ncu=%s\nncu_use_sudo=%s\n\n[gpu]\n' \
        "$RUN_NCU" "$NCU_USE_SUDO"
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap,driver_version \
        --format=csv
} >"$OUTPUT_DIR/manifest.txt"

value_from_log() {
    local key=$1 file=$2
    awk -F= -v key="$key" '$1 == key {
        print substr($0, length(key) + 2); found=1; exit
    } END {if (!found) exit 1}' "$file"
}

phase=paired-timing
printf 'round,order,inline_score_ms,materialized_score_ms,q_materialize_ms,materialized_e2e_ms,persistent_k_once_ms,kernel_speedup,e2e_speedup\n' \
    >"$OUTPUT_DIR/timing.csv"
for ((round=1; round<=TIMING_ROUNDS; round++)); do
    if (( round % 2 == 1 )); then order=inline-first; else order=materialized-first; fi
    log="$OUTPUT_DIR/timing/round-$round-$order.log"
    printf 'Timing exact materialized operands round=%s/%s order=%s...\n' \
        "$round" "$TIMING_ROUNDS" "$order"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_PROFILE_INDEXER_TILE=128 \
        DS4_PROFILE_INDEXER_TOPK=monolithic \
        DS4_PROFILE_INDEXER_OPERANDS=materialized \
        DS4_PROFILE_INDEXER_MATERIALIZED_TIMING_ORDER="$order" \
        DS4_PROFILE_REPEATS="$TIMING_REPEATS" \
        ./tests/cuda_sm75_profile_harness indexer-32k >"$log" 2>&1 || {
            tail -n 160 "$log" >&2 || true
            die "materialized-operand harness failed in round $round"
        }
    for marker in 'operand_path=materialized' 'score_validation=bit-exact' \
                  'topk_validation=exact-order-and-set' 'harness_status=ok'; do
        grep -Fqx "$marker" "$log" ||
            die "round $round lacks required marker: $marker"
    done
    inline_ms=$(value_from_log inline_score_timed_per_call_ms "$log")
    materialized_ms=$(value_from_log materialized_score_timed_per_call_ms "$log")
    q_ms=$(value_from_log q_materialize_timed_per_call_ms "$log")
    e2e_ms=$(value_from_log materialized_e2e_timed_per_call_ms "$log")
    k_once_ms=$(value_from_log persistent_k_materialize_once_ms "$log")
    kernel_speedup=$(awk -v base="$inline_ms" -v candidate="$materialized_ms" \
        'BEGIN {printf "%.9f", base / candidate}')
    e2e_speedup=$(awk -v base="$inline_ms" -v candidate="$e2e_ms" \
        'BEGIN {printf "%.9f", base / candidate}')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$round" "$order" "$inline_ms" "$materialized_ms" "$q_ms" \
        "$e2e_ms" "$k_once_ms" "$kernel_speedup" "$e2e_speedup" \
        >>"$OUTPUT_DIR/timing.csv"
done

phase=summary
python3 speed-bench/summarize-sm75-indexer-f16-operands.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary.log"

if [[ $RUN_NCU == 1 ]]; then
    phase=nsight-compute
    command -v ncu >/dev/null 2>&1 || die "Nsight Compute CLI (ncu) not found"
    ncu_bin=$(command -v ncu)
    ncu_cmd=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_cmd=(sudo -E "$ncu_bin")
    fi
    sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
              --section SchedulerStats --section WarpStateStats
              --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)

    profile_one() {
        local label=$1 kernel=$2 launch_skip=$3 expected=$4
        local base="$OUTPUT_DIR/ncu/$label" rc=0
        printf 'Nsight Compute: %s...\n' "$label"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            DS4_PROFILE_INDEXER_TILE=128 \
            DS4_PROFILE_INDEXER_TOPK=monolithic \
            DS4_PROFILE_INDEXER_OPERANDS=materialized \
            DS4_PROFILE_REPEATS=0 \
            "${ncu_cmd[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "regex:$kernel" \
                --launch-skip "$launch_skip" --launch-count 1 \
                --replay-mode kernel --cache-control none --clock-control none \
                --force-overwrite --export "$base" "${sections[@]}" \
                ./tests/cuda_sm75_profile_harness indexer-32k \
                >"$base.log" 2>&1 || rc=$?
        (( rc == 0 )) || {
            tail -n 140 "$base.log" >&2 || true
            die "Nsight Compute failed for $label (exit $rc)"
        }
        [[ -s $base.ncu-rep ]] || die "Nsight Compute omitted $label report"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" \
            --csv --page raw >"$base.csv" 2>"$base-import.log"
        python3 speed-bench/validate-ncu-capture.py \
            "$base.csv" "$expected" 0 \
            --process cuda_sm75_profile_harness \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight report validation failed for $label"
            }
        cat "$base-validation.txt"
    }

    profile_one inline-wmma128 'indexer_scores_wmma128_kernel.*' 0 \
        'indexer_scores_wmma128_kernel.*'
    profile_one materialized-wmma128 'indexer_scores_wmma128_f16_kernel.*' 0 \
        'indexer_scores_wmma128_f16_kernel.*'
    profile_one persistent-k-materialize 'f32_to_f16_kernel.*' 0 \
        'f32_to_f16_kernel.*'
    profile_one q-materialize 'f32_to_f16_kernel.*' 1 \
        'f32_to_f16_kernel.*'
fi

phase=complete
for required in manifest.txt timing.csv summary.md comparison.json \
                timing/round-1-inline-first.log; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing final evidence: $required"
done
if [[ $RUN_NCU == 1 ]]; then
    for report in inline-wmma128 materialized-wmma128 \
                  persistent-k-materialize q-materialize; do
        [[ -s $OUTPUT_DIR/ncu/$report.ncu-rep &&
           -s $OUTPUT_DIR/ncu/$report.csv ]] ||
            die "missing final Nsight evidence: $report"
    done
fi
printf 'SM75 indexer FP16-operand experiment complete: %s\n' "$OUTPUT_DIR"
