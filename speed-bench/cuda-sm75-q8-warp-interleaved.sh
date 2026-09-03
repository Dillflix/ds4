#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the bounded, size-neutral SM75 Q8_0 warp-interleaved experiment.

The fixed timed shape is the production single-token T32 projection
(1024 inputs x 32768 outputs). The control and candidate retain identical
per-lane DP4A and warp-reduction order. The candidate repacks both weight and
activation word planes; its one-time weight-pack and per-token activation-pack
costs are reported separately. This does not modify ds4 production dispatch.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  BENCH_ROUNDS=14
  BENCH_LAUNCHES=100
  RUN_SANITIZER=1
  RUN_NCU=1
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q8_WARP_INTERLEAVED_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
BENCH_ROUNDS=${BENCH_ROUNDS:-14}
BENCH_LAUNCHES=${BENCH_LAUNCHES:-100}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q8_WARP_INTERLEAVED_DIR:-$repo_dir/sm75-q8-warp-interleaved-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $BENCH_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "BENCH_ROUNDS must be positive"
[[ $BENCH_LAUNCHES =~ ^[1-9][0-9]*$ ]] || die "BENCH_LAUNCHES must be positive"
for value in RUN_SANITIZER RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    [[ ${!value} == 0 || ${!value} == 1 ]] || die "$value must be 0 or 1"
done

for command in nvidia-smi make python3 tar; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done
compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM for GPU $PROFILE_GPU"
(( free_mib >= 512 )) || die "GPU $PROFILE_GPU has only ${free_mib} MiB free; 512 MiB is required"

target=tests/cuda_sm75_q8_warp_interleaved
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/ncu"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    if [[ -d $OUTPUT_DIR ]]; then
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$OUTPUT_DIR.tar.gz"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
                    "$(basename "$OUTPUT_DIR")"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu_physical=%s\nprofile_gpu_logical=0\n' "$PROFILE_GPU"
    printf 'cuda_arch=%s\ncompute_capability=%s\nfree_mib_at_preflight=%s\n' \
        "$CUDA_ARCH" "$compute_cap" "$free_mib"
    printf 'benchmark_rounds=%s\nbenchmark_launches=%s\n' "$BENCH_ROUNDS" "$BENCH_LAUNCHES"
    printf 'canonical_bytes_per_32_blocks=1088\ninterleaved_bytes_per_32_blocks=1088\n'
    printf 'production_dispatch_modified=no\nproduction_cache_integration=no\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'
    git status --short
} >"$OUTPUT_DIR/manifest.txt"

current_phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    if ! make -B -j"$(nproc)" "$target" CUDA_ARCH="$CUDA_ARCH" \
            >"$OUTPUT_DIR/build.log" 2>&1; then
        tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
        die "build failed"
    fi
else
    make -q "$target" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale Q8 interleave harness"
fi
[[ -x $target ]] || die "$target is missing; rerun with SKIP_BUILD=0"

require_count() {
    local file=$1 pattern=$2 expected=$3 label=$4 actual
    actual=$(grep -c -E "$pattern" "$file" || true)
    [[ $actual == "$expected" ]] ||
        die "$label count is $actual, expected $expected"
}

current_phase=correctness
correctness_log="$OUTPUT_DIR/correctness.log"
if ! env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$target --device 0 \
        --correctness-only >"$correctness_log" 2>&1; then
    tail -n 160 "$correctness_log" >&2 || true
    die "correctness harness failed"
fi
require_count "$correctness_log" '^correctness_shape_end=1$' 3 "correctness shape"
require_count "$correctness_log" '^repack_roundtrip=byte-exact$' 3 "weight repack"
require_count "$correctness_log" '^activation_repack_roundtrip=byte-exact$' 9 "activation repack"
require_count "$correctness_log" '^correctness_result=bit-exact$' 9 "bit-exact fixture"
require_count "$correctness_log" '^correctness_output_full_overwrite=pass$' 9 "output overwrite"
require_count "$correctness_log" '^correctness_canaries=ok$' 3 "canary"
grep -q '^harness_status=ok$' "$correctness_log" || die "correctness success marker missing"

current_phase=paired-benchmark
benchmark_log="$OUTPUT_DIR/benchmark.log"
if ! env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$target --device 0 \
        --benchmark-only --rounds "$BENCH_ROUNDS" --launches "$BENCH_LAUNCHES" \
        >"$benchmark_log" 2>&1; then
    tail -n 160 "$benchmark_log" >&2 || true
    die "paired benchmark failed"
fi
grep -q '^post_timing_exactness=bit-exact$' "$benchmark_log" ||
    die "post-timing exactness marker missing"
grep -q '^post_timing_canaries=ok$' "$benchmark_log" ||
    die "post-timing canary marker missing"
grep -q '^benchmark_status=ok$' "$benchmark_log" || die "benchmark success marker missing"
grep -q '^harness_status=ok$' "$benchmark_log" || die "harness success marker missing"

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=compute-sanitizer
    command -v compute-sanitizer >/dev/null 2>&1 || die "compute-sanitizer not found"
    sanitizer_log="$OUTPUT_DIR/compute-sanitizer.log"
    if ! env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" compute-sanitizer \
            --tool memcheck --error-exitcode 99 ./$target --device 0 \
            --correctness-only >"$sanitizer_log" 2>&1; then
        tail -n 160 "$sanitizer_log" >&2 || true
        die "Compute Sanitizer failed"
    fi
    grep -q 'ERROR SUMMARY: 0 errors' "$sanitizer_log" ||
        die "Compute Sanitizer zero-error summary missing"
fi

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    command -v ncu >/dev/null 2>&1 || die "ncu not found"
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    metrics=(
        gpu__time_duration.sum
        dram__bytes.sum.per_second
        dram__bytes.avg.pct_of_peak_sustained_elapsed
        lts__t_sector_hit_rate.pct
        smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
        sm__warps_active.avg.pct_of_peak_sustained_active
        launch__registers_per_thread
        launch__block_size
        launch__grid_size
    )
    metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
    for variant in control interleaved; do
        if [[ $variant == control ]]; then
            regex='sm75_q8_0_preq_warp8_control_kernel.*'
        else
            regex='sm75_q8_0_preq_warp8_interleaved_kernel.*'
        fi
        base="$OUTPUT_DIR/ncu/$variant"
        rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_command[@]}" \
            --config-file off --verbose --target-processes application-only \
            --devices 0 --kernel-name-base function \
            --kernel-name "regex:$regex" --launch-skip 1 --launch-count 1 \
            --replay-mode kernel --cache-control all --clock-control none \
            --metrics "$metric_csv" --disable-extra-suffixes \
            --force-overwrite --export "$base" \
            ./$target --device 0 --profile "$variant" --launches 2 \
            >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )); then
            tail -n 160 "$base.log" >&2 || true
            grep -q ERR_NVGPUCTRPERM "$base.log" &&
                printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
            die "Nsight Compute failed for $variant (exit $rc)"
        fi
        [[ -s $base.ncu-rep ]] || die "missing Nsight report for $variant"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "could not import Nsight report for $variant"
        python3 speed-bench/validate-ncu-capture.py \
            "$base.csv" "$regex" 0 \
            --process cuda_sm75_q8_warp_interleaved \
            --block-size 256 --grid-size 4096 \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight report validation failed for $variant"
            }
        cat "$base-validation.txt"
    done
fi

current_phase=summarization
{
    printf '# Size-neutral SM75 Q8_0 warp-interleaved diagnostic\n\n'
    grep -E '^(canonical_bytes|interleaved_bytes|size_neutral|shipping_dispatch)' "$correctness_log"
    printf '\n'
    grep -E '^(benchmark_scope|weight_repack_|xq_repack_|inclusive_candidate_|paired_speedup_|post_timing_|benchmark_status)' "$benchmark_log"
    if [[ $RUN_NCU == 1 ]]; then
        printf '\n'
        cat "$OUTPUT_DIR"/ncu/*-validation.txt
    fi
} >"$OUTPUT_DIR/summary.txt"
cat "$OUTPUT_DIR/summary.txt"

current_phase=complete
printf 'SM75 size-neutral Q8_0 warp-interleaved experiment complete: %s\n' "$OUTPUT_DIR"
