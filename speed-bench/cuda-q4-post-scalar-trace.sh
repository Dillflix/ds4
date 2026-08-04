#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the current production full-Q4 SM75 prefill kernel distribution after
the scalar-slot paths have been enabled.

Required environment:
  MODEL=/absolute/path/to/full-Q4.gguf

Optional environment:
  PROMPT=/absolute/path/prompt.txt  # default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  STAGE_SPLIT=22
  PROFILE_TOKENS=2048              # fixed bounded capture shape
  PREFILL_CHUNK=2048
  CUDA_ARCH=sm_75
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  POST_SCALAR_TRACE_DIR=...         # new output directory

The runner first performs one audited full-Q4 dispatch validation. It then
starts a separate clean process and captures only the timed prefill range with
Nsight Systems. It does not run Nsight Compute and does not hash the model.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* ]] || die "MODEL must be an absolute path"
[[ -f $MODEL ]] || die "model not found: $MODEL"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${POST_SCALAR_TRACE_DIR:-$repo_dir/q4-post-scalar-trace-$run_stamp}
while [[ $OUTPUT_DIR != / && $OUTPUT_DIR == */ ]]; do
    OUTPUT_DIR=${OUTPUT_DIR%/}
done

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 && PROFILE_TOKENS > 0 &&
   PREFILL_CHUNK > 0 )) || die "invalid capture shape"
(( PROFILE_TOKENS == 2048 )) ||
    die "PROFILE_TOKENS must remain 2048 for this bounded trace"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
for tool in awk basename date dirname env git grep make mkdir mv nproc nsys \
            nvidia-smi python3 sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

default_lock=/tmp/ds4.lock
if [[ -e $default_lock ]]; then
    [[ -f $default_lock && -w $default_lock ]] ||
        die "$default_lock exists but is not a writable regular file"
else
    [[ -d /tmp && -w /tmp ]] ||
        die "/tmp is not writable; DS4 cannot create $default_lock"
fi

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four devices"
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
    [[ -z ${seen_gpus[$gpu]+x} ]] || die "duplicate GPU index: $gpu"
    seen_gpus[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is not SM75 (${cap:-unknown})"
done

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
[[ ! -e $OUTPUT_DIR.tar.gz ]] || die "archive path already exists: $OUTPUT_DIR.tar.gz"
mkdir -p "$OUTPUT_DIR"/{validation,nsys,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    local archive="$OUTPUT_DIR.tar.gz"
    local partial="$archive.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$archive"; then
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

mapfile -t inherited_ds4_envs < <(
    env | awk -F= '$1 ~ /^DS4_/ { print $1 }' | sort -u
)
clean_prefix=(env)
for name in "${inherited_ds4_envs[@]}"; do clean_prefix+=(-u "$name"); done
production_prefix=(
    "${clean_prefix[@]}"
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=1
    DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=1
    DS4_CUDA_MOE_IQ2_SCALAR_SM75=1
)

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        CUDA_ARCH="$CUDA_ARCH" 2>&1 | tee "$OUTPUT_DIR/build.log"
    current_phase=correctness
    "${clean_prefix[@]}" ./tests/cuda_long_context_smoke \
        >"$OUTPUT_DIR/correctness.log" 2>&1 || {
        tail -n 120 "$OUTPUT_DIR/correctness.log" >&2 || true
        die "CUDA correctness smoke failed"
    }
    for marker in 'sm75 q4 gate tile8 scalar exact' \
                  'sm75 q4 down tile16 scalar exact' \
                  'cuda long-context regression: OK'; do
        grep -Fq "$marker" "$OUTPUT_DIR/correctness.log" ||
            die "correctness marker missing: $marker"
    done
else
    [[ -x ./ds4-bench ]] || die "SKIP_BUILD=1 but ds4-bench is missing"
    [[ -x ./tests/cuda_long_context_smoke ]] ||
        die "SKIP_BUILD=1 but tests/cuda_long_context_smoke is missing"
    if ! make -q ds4-bench tests/cuda_long_context_smoke \
            CUDA_ARCH="$CUDA_ARCH"; then
        die "SKIP_BUILD=1 found stale targets; rerun with SKIP_BUILD=0"
    fi
fi
if ! grep -aFq 'sm75-scalar-dispatch path=%s scalar=%d device=%d' ./ds4-bench ||
   ! grep -aFq 'q4-gate-tile8' ./ds4-bench ||
   ! grep -aFq 'q4-down-tile16' ./ds4-bench; then
    die "ds4-bench lacks the scalar-dispatch audit; rebuild with SKIP_BUILD=0"
fi

runtime_common=(
    --cuda --cuda-tensor-parallel
    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
    --model "$MODEL" --prompt-file "$PROMPT"
    --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS"
    --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS"
    --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0
)

current_phase=manifest
{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT"
    printf 'profile_tokens=%s\nprefill_chunk=%s\nskip_build=%s\n' \
        "$PROFILE_TOKENS" "$PREFILL_CHUNK" "$SKIP_BUILD"
    printf 'scalar_q4_gate=forced-on\nscalar_q4_down=forced-on\n'
    printf 'q8_fp16_cache=forced-on\nmodel_hashing=disabled\nnsight_compute=disabled\n'
    printf '\n[gpu]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[topology]\n'
    nvidia-smi topo -m
    printf '\n[nsys]\n'
    nsys --version
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"

current_phase=audited-dispatch-validation
printf 'Validating production full-Q4 scalar dispatch...\n'
"${production_prefix[@]}" \
    DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
    DS4_CUDA_MOE_SCALAR_AUDIT=1 \
    ./ds4-bench "${runtime_common[@]}" \
        --csv "$OUTPUT_DIR/validation/benchmark.csv" \
        >"$OUTPUT_DIR/validation/benchmark.log" 2>&1

validation_log="$OUTPUT_DIR/validation/benchmark.log"
grep -Fq "ds4: CUDA EP forced pipeline split $STAGE_SPLIT/$((43 - STAGE_SPLIT))" \
    "$validation_log" || die "forced split missing from validation log"
grep -Fq "4 devices [$GPU_DEVICES] requested" "$validation_log" ||
    die "GPU order record missing from validation log"
audit_count=$(grep -c '^ds4: routed-quant-audit layer=' "$validation_log" || true)
recipe_count=$(grep -Ec \
    '^ds4: routed-quant-audit layer=[0-9]+ gate=q4_k up=q4_k down=q4_k$' \
    "$validation_log" || true)
[[ $audit_count == 43 && $recipe_count == 43 ]] ||
    die "expected 43 exact full-Q4 recipe records; got $recipe_count/$audit_count"

expected_markers=0
for path in q4-gate-tile8 q4-down-tile16; do
    for pos in "${!gpu_ids[@]}"; do
        gpu=${gpu_ids[$pos]}
        if (( pos == 0 || pos == 2 )); then
            layer_start=0; layer_end=$((STAGE_SPLIT - 1))
        else
            layer_start=$STAGE_SPLIT; layer_end=42
        fi
        for ((layer=layer_start; layer<=layer_end; layer++)); do
            count=$(grep -Fc \
                "sm75-scalar-dispatch path=$path scalar=1 device=$gpu layer=$layer " \
                "$validation_log" || true)
            [[ $count == 1 ]] ||
                die "$path scalar device=$gpu layer=$layer launch count is $count"
            expected_markers=$((expected_markers + 1))
        done
    done
done
marker_count=$(grep -c '^ds4: sm75-scalar-dispatch ' "$validation_log" || true)
[[ $marker_count == "$expected_markers" ]] ||
    die "scalar dispatch coverage is $marker_count records; expected $expected_markers"
if grep -E 'sm75-scalar-dispatch .*scalar=0' "$validation_log" >/dev/null; then
    die "non-scalar Q4 specialization appeared in validation"
fi
printf 'full_q4_recipe_layers=%s\nscalar_dispatch_markers=%s\n' \
    "$recipe_count" "$marker_count" >"$OUTPUT_DIR/validation/status.txt"

current_phase=nsight-systems-capture
printf 'Capturing clean 2K timed prefill with Nsight Systems...\n'
"${production_prefix[@]}" \
    DS4_NSYS_CAPTURE_PREFILL=1 \
    nsys profile --force-overwrite=true --sample=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$OUTPUT_DIR/nsys/full-q4-post-scalar" \
        ./ds4-bench "${runtime_common[@]}" \
            --csv "$OUTPUT_DIR/nsys/benchmark.csv" \
            >"$OUTPUT_DIR/nsys/capture.log" 2>&1
[[ -s $OUTPUT_DIR/nsys/full-q4-post-scalar.nsys-rep ]] ||
    die "Nsight Systems did not produce a report"
if grep -Eq 'sm75-scalar-dispatch|routed-quant-audit|wrote CUDA.*audit' \
        "$OUTPUT_DIR/nsys/capture.log"; then
    die "timed trace contained audit instrumentation"
fi

current_phase=nsight-systems-export
nsys stats --report cuda_gpu_kern_sum --format csv \
    "$OUTPUT_DIR/nsys/full-q4-post-scalar.nsys-rep" \
    >"$OUTPUT_DIR/nsys/cuda_gpu_kern_sum.csv" \
    2>"$OUTPUT_DIR/nsys/cuda_gpu_kern_sum.log"
for report in cuda_gpu_mem_time_sum cuda_api_sum cuda_gpu_trace; do
    nsys stats --report "$report" --format csv \
        "$OUTPUT_DIR/nsys/full-q4-post-scalar.nsys-rep" \
        >"$OUTPUT_DIR/nsys/$report.csv" \
        2>"$OUTPUT_DIR/nsys/$report.log" || true
done

current_phase=summarize
python3 speed-bench/summarize-q4-post-scalar-trace.py \
    "$OUTPUT_DIR/nsys/cuda_gpu_kern_sum.csv" \
    "$OUTPUT_DIR/kernel-groups.csv" \
    "$OUTPUT_DIR/required-q4-kernels.csv" \
    2>&1 | tee "$OUTPUT_DIR/summary.txt"

python3 - "$OUTPUT_DIR/required-q4-kernels.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
by_group = {}
for row in rows:
    group = row["group"]
    if group in by_group:
        raise SystemExit(f"multiple Nsight rows for required kernel group {group}")
    by_group[group] = row
for group in ("q4_gate_up_tile8", "q4_down_tile16"):
    row = by_group.get(group)
    if row is None:
        raise SystemExit(f"missing required kernel group {group}")
    if int(row["instances"]) != 344:
        raise SystemExit(
            f"{group} has {row['instances']} instances; expected 344 for the fixed 2K trace"
        )
    if row["scalar_specialization"] != "true":
        raise SystemExit(
            f"{group} does not identify the scalar specialization: "
            f"{row['scalar_specialization']}"
        )
PY

for required in kernel-groups.csv required-q4-kernels.csv summary.txt \
                nsys/benchmark.csv nsys/cuda_gpu_kern_sum.csv; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing final evidence: $required"
done
printf 'post_scalar_full_q4_trace=valid\n' >"$OUTPUT_DIR/capture-status.txt"
current_phase=complete
printf 'Post-scalar full-Q4 trace complete: %s\n' "$OUTPUT_DIR"
