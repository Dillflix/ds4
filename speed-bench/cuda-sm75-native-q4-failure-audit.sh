#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture only the evidence needed to explain the rejected SM75 native-Q4
stream7 gate and compact7 down candidates.

Required environment:
  NATIVE_MODEL=/absolute/path/to/tagged-sm75-native.gguf
  PROMPT=/absolute/path/to/prompt.txt

Optional environment:
  PROFILE_GPU=0
  PROFILE_TOKENS=2048
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  FAILURE_AUDIT_DIR=/absolute/output/directory

The pass intentionally performs no throughput sweep, model conversion,
hashing, logits comparison, or Nsight Compute profiling of unrelated kernels.
It records:
  * one bounded full-engine Nsight Systems trace each for baseline, gate-only,
    and down-only;
  * early/late Nsight Compute captures for stream7 and compact7 only.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${NATIVE_MODEL:?set NATIVE_MODEL to the absolute tagged SM75-native GGUF path}"
: "${PROMPT:?set PROMPT to the absolute prompt path}"
PROFILE_GPU=${PROFILE_GPU:-0}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CUDA_ARCH=sm_75
OUTPUT_DIR=${FAILURE_AUDIT_DIR:-$repo_dir/sm75-native-q4-failure-audit-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $NATIVE_MODEL == /* && -f $NATIVE_MODEL ]] ||
    die "NATIVE_MODEL must be an existing absolute path"
[[ $PROMPT == /* && -f $PROMPT ]] || die "PROMPT must be an existing absolute path"
[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a GPU index"
[[ $PROFILE_TOKENS =~ ^[0-9]+$ && $PROFILE_TOKENS -gt 0 ]] ||
    die "PROFILE_TOKENS must be positive"
for flag in NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

for tool in awk date env git grep make mkdir mv ncu nproc nsys nvidia-smi \
            python3 tail tar tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
for gpu in 0 1 2 3; do
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is not SM75 (${cap:-unknown})"
done
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ && $free_mib -ge 4096 ]] ||
    die "PROFILE_GPU requires at least 4096 MiB free"

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{nsys,ncu,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")"; then
            mv -- "$partial" "$archive"
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'error: could not archive %s\n' "$OUTPUT_DIR" >&2
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'current_phase=interrupted-INT; exit 130' INT
trap 'current_phase=interrupted-TERM; exit 143' TERM
trap 'current_phase=interrupted-HUP; exit 129' HUP

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_sm75_profile_harness \
        CUDA_ARCH="$CUDA_ARCH" 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    for binary in ./ds4-bench ./tests/cuda_sm75_profile_harness; do
        [[ -x $binary ]] || die "SKIP_BUILD=1 but $binary is missing"
    done
    make -q ds4-bench tests/cuda_sm75_profile_harness \
        CUDA_ARCH="$CUDA_ARCH" || die "SKIP_BUILD=1 rejected stale binaries"
fi

mapfile -t inherited_ds4_envs < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean_prefix=(env)
for name in "${inherited_ds4_envs[@]}"; do clean_prefix+=(-u "$name"); done
production_prefix=(
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1)

variant_flags() {
    case "$1" in
        baseline) printf '0 0\n' ;;
        gate)     printf '1 0\n' ;;
        down)     printf '0 1\n' ;;
        *) die "unknown variant: $1" ;;
    esac
}

run_production_variant() {
    local variant=$1; shift
    local gate down
    read -r gate down < <(variant_flags "$variant")
    "${clean_prefix[@]}" "${production_prefix[@]}" \
        DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=0 \
        "DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7=$gate" \
        "DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7=$down" \
        "$@"
}

run_harness_variant() {
    local variant=$1; shift
    local gate down
    read -r gate down < <(variant_flags "$variant")
    "${clean_prefix[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=0 \
        "DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7=$gate" \
        "DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7=$down" \
        "$@"
}

validate_selection() {
    local variant=$1 log=$2 gate down gate_name down_name
    read -r gate down < <(variant_flags "$variant")
    gate_name=tile8; [[ $gate == 0 ]] || gate_name=stream7
    down_name=full-stage; [[ $down == 0 ]] || down_name=compact7
    grep -Fq 'ds4: CUDA EP forced pipeline split 22/21' "$log" ||
        die "$variant did not use split 22/21"
    grep -Fq '4 devices [0,3,1,2] requested' "$log" ||
        die "$variant did not use GPU order 0,3,1,2"
    grep -Fq "packed A/W, planner=cost, gate=$gate_name, down=$down_name" \
        "$log" || die "$variant did not select the requested kernels"
}

{
    printf 'date_utc=%s\nrepo=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir" \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'native_model=%s\nprompt=%s\nprofile_tokens=%s\n' \
        "$NATIVE_MODEL" "$PROMPT" "$PROFILE_TOKENS"
    printf 'production_gpu_devices=0,3,1,2\nproduction_stage_split=22/21\n'
    printf 'profile_gpu_physical=%s\nprofile_gpu_logical=0\n' "$PROFILE_GPU"
    printf 'full_model_hashing=false\nthroughput_sweep=false\nexactness_pass=false\n'
    printf '\n[gpu]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[topology]\n'; nvidia-smi topo -m
    printf '\n[ncu]\n'; ncu --version 2>&1
    printf '\n[nsys]\n'; nsys --version 2>&1
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

common=(--cuda --cuda-tensor-parallel --gpu-devices 0,3,1,2
        --gpu-vram auto --prompt-file "$PROMPT" --prefill-chunk 2048
        --gen-tokens 0 --model "$NATIVE_MODEL"
        --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS"
        --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS")

# These are full-engine traces, but capture only the measured prefill range.
current_phase=nsight-systems
for variant in baseline gate down; do
    printf 'Nsight Systems: %s...\n' "$variant"
    base="$OUTPUT_DIR/nsys/$variant"
    run_production_variant "$variant" DS4_NSYS_CAPTURE_PREFILL=1 \
        nsys profile --force-overwrite=true --sample=none \
            --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
            --capture-range-end=stop --output="$base" \
            ./ds4-bench "${common[@]}" --csv "$base-benchmark.csv" \
            >"$base.log" 2>&1
    [[ -s $base.nsys-rep ]] || die "missing Nsight Systems report for $variant"
    validate_selection "$variant" "$base.log"
    nsys stats --report cuda_gpu_kern_sum --format csv "$base.nsys-rep" \
        >"$base-cuda_gpu_kern_sum.csv" 2>"$base-stats.log"
    [[ -s $base-cuda_gpu_kern_sum.csv ]] ||
        die "empty Nsight Systems kernel summary for $variant"
done

ncu_bin=$(command -v ncu)
[[ $ncu_bin == /* && -x $ncu_bin ]] || die "cannot resolve ncu: $ncu_bin"
ncu_command=("$ncu_bin")
if [[ $NCU_USE_SUDO == 1 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    sudo -v
    ncu_command=(sudo -E "$ncu_bin")
fi
ncu_sections=(
    --section SpeedOfLight
    --section LaunchStats
    --section Occupancy
    --section SchedulerStats
    --section WarpStateStats
    --section MemoryWorkloadAnalysis
    --section ComputeWorkloadAnalysis)

profile_candidate() {
    local name=$1 scenario=$2 variant=$3 kernel_regex=$4 expected_regex=$5
    local base="$OUTPUT_DIR/ncu/$name" rc=0
    printf 'Nsight Compute: %s...\n' "$name"
    run_harness_variant "$variant" "${ncu_command[@]}" --config-file off \
        --target-processes application-only --devices 0 \
        --kernel-name-base function --kernel-name "regex:$kernel_regex" \
        --launch-count 1 --replay-mode kernel --cache-control none \
        --clock-control none --force-overwrite --export "$base" \
        "${ncu_sections[@]}" ./tests/cuda_sm75_profile_harness "$scenario" \
        >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        tail -n 120 "$base.log" >&2 || true
        die "Nsight Compute failed for $name (exit $rc)"
    fi
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log"; then
        tail -n 120 "$base.log" >&2 || true
        die "Nsight Compute produced a failed or zero-kernel capture: $name"
    fi
    [[ -s $base.ncu-rep ]] || die "missing Nsight Compute report: $name"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
        >"$base.csv" 2>"$base-import.log" ||
        die "could not import Nsight Compute report: $name"
    python3 speed-bench/validate-ncu-capture.py \
        "$base.csv" "$expected_regex" 0 \
        --process cuda_sm75_profile_harness \
        >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true
            die "Nsight Compute captured the wrong kernel: $name"
        }
    cat "$base-validation.txt"
}

# Only the new candidates are replayed. The production baseline already has
# the corresponding early/late tile metrics; Systems supplies the same-build
# aggregate baseline needed to quantify displaced work and launch behavior.
current_phase=nsight-compute
profile_candidate early-gate-stream7 native-q4-early gate \
    '^moe_gate_up_mid_sm75_native_q4_tile16_stream7_kernel$' \
    'moe_gate_up_mid_sm75_native_q4_tile16_stream7_kernel.*512'
profile_candidate late-gate-stream7 native-q4-late gate \
    '^moe_gate_up_mid_sm75_native_q4_tile16_stream7_kernel$' \
    'moe_gate_up_mid_sm75_native_q4_tile16_stream7_kernel.*512'
profile_candidate early-down-compact7 native-q4-early down \
    '^moe_down_sm75_native_q4_tile16_compact7_kernel$' \
    'moe_down_sm75_native_q4_tile16_compact7_kernel.*512'
profile_candidate late-down-compact7 native-q4-late down \
    '^moe_down_sm75_native_q4_tile16_compact7_kernel$' \
    'moe_down_sm75_native_q4_tile16_compact7_kernel.*512'

current_phase=complete
printf 'SM75 native-Q4 failure audit complete: %s\n' "$OUTPUT_DIR"
