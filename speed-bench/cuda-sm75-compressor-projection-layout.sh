#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the bounded SM75 width-1024 compressor-projection layout experiment.

The production-ordered control is compared with a canonical-row-major shared-
staging kernel and a size-neutral lane-major weight layout using either one
warp per output row or two independent warps (KV and score). One-time weight
packing is excluded. Per-token activation packing is reported both excluded
and included. Production dispatch is not modified.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  BENCH_ROUNDS=9
  BENCH_LAUNCHES=25
  RUN_SANITIZER=1
  RUN_NCU=1
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  COMPRESSOR_PROJECTION_LAYOUT_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
BENCH_ROUNDS=${BENCH_ROUNDS:-9}
BENCH_LAUNCHES=${BENCH_LAUNCHES:-25}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${COMPRESSOR_PROJECTION_LAYOUT_DIR:-$repo_dir/sm75-compressor-projection-layout-$(date -u +%Y%m%dT%H%M%SZ)}
target=tests/cuda_sm75_compressor_projection_layout

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $BENCH_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "BENCH_ROUNDS must be positive"
[[ $BENCH_LAUNCHES =~ ^[1-9][0-9]*$ ]] || die "BENCH_LAUNCHES must be positive"
for flag in RUN_SANITIZER RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || die "$flag must be 0 or 1"
done
for command in awk basename date dirname env git grep id make mkdir mv nproc \
               nvidia-smi python3 sort tail tar tr; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU is ${compute_cap:-unknown}; SM75 is required"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM"
(( free_mib >= 256 )) || die "GPU $PROFILE_GPU has only $free_mib MiB free; 256 MiB is required"

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/ncu"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'profile_gpu_physical=%s\nprofile_gpu_logical=0\n' "$PROFILE_GPU"
    printf 'cuda_arch=%s\ncompute_capability=%s\nfree_mib_at_preflight=%s\n' \
        "$CUDA_ARCH" "$compute_cap" "$free_mib"
    printf 'shape=4096x1024-paired-f16\nratio=4\n'
    printf 'benchmark_rounds=%s\nbenchmark_launches=%s\n' \
        "$BENCH_ROUNDS" "$BENCH_LAUNCHES"
    printf 'production_dispatch_modified=no\nweight_pack_in_timing=no\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "$target" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/build.log" >&2 || true
            die "build failed"
        }
else
    make -q "$target" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale compressor-layout harness"
fi
[[ -x $target ]] || die "$target is missing"

phase=correctness
if ! env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$target --device 0 \
        --correctness-only >"$OUTPUT_DIR/correctness.log" 2>&1; then
    tail -n 200 "$OUTPUT_DIR/correctness.log" >&2 || true
    die "correctness run failed"
fi
for marker in \
    'correctness_result_canonical-staged=bit-exact' \
    'correctness_result_lane-major=bit-exact' \
    'correctness_result_lane-major-two-warp=bit-exact' \
    'weight_layout_size_neutral=yes' \
    'weight_repack_roundtrip=byte-exact' \
    'activation_repack_roundtrip=byte-exact' \
    'control_required_regions=overwritten' \
    'correctness_full_output_state=bit-exact' \
    'correctness_canaries=ok' \
    'harness_status=ok'; do
    grep -Fqx "$marker" "$OUTPUT_DIR/correctness.log" ||
        die "missing correctness marker: $marker"
done

phase=paired-benchmark
if ! env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$target --device 0 \
        --benchmark-only --rounds "$BENCH_ROUNDS" --launches "$BENCH_LAUNCHES" \
        >"$OUTPUT_DIR/benchmark.log" 2>&1; then
    tail -n 200 "$OUTPUT_DIR/benchmark.log" >&2 || true
    die "benchmark run failed"
fi
for marker in post_timing_exactness=bit-exact post_timing_canaries=ok \
              benchmark_status=ok harness_status=ok; do
    grep -Fqx "$marker" "$OUTPUT_DIR/benchmark.log" ||
        die "missing benchmark marker: $marker"
done

if [[ $RUN_SANITIZER == 1 ]]; then
    phase=compute-sanitizer
    command -v compute-sanitizer >/dev/null 2>&1 || die "compute-sanitizer not found"
    if ! env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" compute-sanitizer \
            --tool memcheck --error-exitcode 99 ./$target --device 0 \
            --correctness-only >"$OUTPUT_DIR/compute-sanitizer.log" 2>&1; then
        tail -n 200 "$OUTPUT_DIR/compute-sanitizer.log" >&2 || true
        die "Compute Sanitizer failed"
    fi
    grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/compute-sanitizer.log" ||
        die "Compute Sanitizer zero-error summary missing"
fi

if [[ $RUN_NCU == 1 ]]; then
    phase=nsight-compute
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
    for variant in control canonical-staged lane-major two-warp; do
        case $variant in
            control)
                regex='sm75_compressor_pair_control_kernel.*'; block=32 ;;
            canonical-staged)
                regex='sm75_compressor_pair_canonical_staged_kernel.*'; block=32 ;;
            lane-major)
                regex='sm75_compressor_pair_lane_major_kernel.*'; block=32 ;;
            two-warp)
                regex='sm75_compressor_pair_lane_major_two_warp_kernel.*'; block=64 ;;
        esac
        base="$OUTPUT_DIR/ncu/$variant"
        rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_command[@]}" \
            --config-file off --verbose --target-processes application-only \
            --devices 0 --kernel-name-base function \
            --kernel-name "regex:$regex" --launch-skip 1 --launch-count 1 \
            --replay-mode kernel --cache-control all --clock-control none \
            --metrics "$metric_csv" --disable-extra-suffixes \
            --force-overwrite --export "$base" \
            ./$target --device 0 --profile "$variant" --launches 3 \
            >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )); then
            tail -n 200 "$base.log" >&2 || true
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
            --process cuda_sm75_compressor_projection_layout \
            --block-size "$block" --grid-size 1024 \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight validation failed for $variant"
            }
        cat "$base-validation.txt"
    done
fi

phase=summarization
{
    printf '# SM75 width-1024 compressor projection layout diagnostic\n\n'
    grep -E '^(resources|weight_layout|weight_repack|activation_repack|control_required|correctness_)' \
        "$OUTPUT_DIR/correctness.log"
    printf '\n'
    grep -E '^(benchmark_scope|weight_repack_scope|activation_pack_scope|timing_|control_|canonical_staged_|lane_major_|two_warp_|post_timing_|benchmark_status)' \
        "$OUTPUT_DIR/benchmark.log"
    if [[ $RUN_NCU == 1 ]]; then
        printf '\n'; cat "$OUTPUT_DIR"/ncu/*-validation.txt
    fi
} >"$OUTPUT_DIR/summary.txt"
cat "$OUTPUT_DIR/summary.txt"

phase=complete
printf 'SM75 compressor projection layout diagnostic complete: %s\n' "$OUTPUT_DIR"
