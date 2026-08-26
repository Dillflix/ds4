#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute full-Q4 GGUF path"
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
QUALITY_MANIFEST=${QUALITY_MANIFEST:-$repo_dir/gguf-tools/quality-testing/data/flash/manifest.tsv}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-16384}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
QUALITY_CTX=${QUALITY_CTX:-$((CTX_MAX+1))}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-3}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_GPU_TEST=${RUN_GPU_TEST:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T256_VALIDATION_DIR:-$repo_dir/t256-production-validation-$stamp}

[[ $PROMPT == /* && -f $PROMPT ]] ||
    die "PROMPT must name an existing absolute path"
[[ $QUALITY_MANIFEST == /* && -f $QUALITY_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute path"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "QUALITY_CTX:$QUALITY_CTX" \
            "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "REPEATS:$REPEATS" "SKIP_BUILD:$SKIP_BUILD" \
            "RUN_GPU_TEST:$RUN_GPU_TEST" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 )) ||
    die "STAGE_SPLIT must be in 1..42"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "production acceptance requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
(( CTX_START == 16384 && CTX_MAX == 65536 && STEP_MUL == 2 )) ||
    die "production acceptance requires the fixed 16K/32K/64K sweep"
(( QUALITY_CTX == CTX_MAX + 1 )) ||
    die "QUALITY_CTX must equal CTX_MAX+1 so quality uses production memory pressure"
(( PREFILL_CHUNK > 0 && REPEATS >= 3 )) ||
    die "PREFILL_CHUNK must be positive and REPEATS must be at least 3"
for flag in SKIP_BUILD RUN_GPU_TEST CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $RUN_GPU_TEST == 1 ]] ||
    die "production acceptance requires RUN_GPU_TEST=1"

IFS=',' read -r -a gpu_device_ids <<<"$GPU_DEVICES"
(( ${#gpu_device_ids[@]} == 4 )) ||
    die "GPU_DEVICES must contain four physical devices (homes first, partners second)"
declare -A seen_gpu=()
for device in "${gpu_device_ids[@]}"; do
    [[ $device =~ ^[0-9]+$ ]] || die "invalid GPU device index: $device"
    [[ -z ${seen_gpu[$device]+x} ]] || die "duplicate GPU device index: $device"
    seen_gpu[$device]=1
done
partner_device_0=${gpu_device_ids[2]}
partner_device_1=${gpu_device_ids[3]}
if [[ $GPU_VRAM != auto ]]; then
    IFS=',' read -r -a gpu_vram_values <<<"$GPU_VRAM"
    (( ${#gpu_vram_values[@]} == 4 )) ||
        die "GPU_VRAM must be auto or four comma-separated GiB budgets"
    for budget in "${gpu_vram_values[@]}"; do
        [[ $budget =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            die "invalid GPU_VRAM budget: $budget"
    done
fi

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{quality,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM
    write_run_status() {
        printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
            "$([[ $status == 0 ]] && printf finished || printf failed)" \
            "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    }
    write_run_status
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                mv "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive" >&2
        else
            printf 'error: failed to create nonempty archive: %s\n' \
                "$archive" >&2
            [[ $status != 0 ]] || status=1
            write_run_status
            rm -f -- "$partial"
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v make >/dev/null 2>&1 || die "make not found"
if [[ $CREATE_ARCHIVE == 1 ]]; then
    command -v tar >/dev/null 2>&1 || die "tar not found"
    command -v mv >/dev/null 2>&1 || die "mv not found"
fi

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

topology_file="$OUTPUT_DIR/provenance/topology.txt"
nvidia-smi topo -m >"$topology_file"
{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'prompt=%s\nquality_manifest=%s\nquality_ctx=%s\n' \
        "$PROMPT" "$QUALITY_MANIFEST" "$QUALITY_CTX"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK" "$REPEATS"
    printf 'gpu_exactness_test=required-and-enabled\n'
    printf 'candidate_policy=no DS4_CUDA_Q8_F16_PARTNER_CLASSES override\n'
    printf 'model_hashing=disabled\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
        --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print $0}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench \
        gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found stale targets"
fi

phase=tests
./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2
        die "planner unit test failed"
    }
if [[ $RUN_GPU_TEST == 1 ]]; then
    ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/gpu-exactness.log" >&2
        die "multi-GPU exactness test failed"
    }
    grep -Fq 'q8 partner projection exactness OK (3 classes)' \
        "$OUTPUT_DIR/gpu-exactness.log" ||
        die "GPU exactness did not exercise all three partner classes"
fi

run_quality() {
    local variant=$1
    local out="$OUTPUT_DIR/quality/$variant.tsv"
    local log="$OUTPUT_DIR/quality/$variant.log"
    local audit="$OUTPUT_DIR/quality/$variant.q8-audit.csv"
    local bindings="$OUTPUT_DIR/quality/$variant.bindings.csv"
    local -a variant_env=()
    if [[ $variant == local ]]; then
        variant_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
        variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
    elif [[ $variant != default ]]; then
        die "internal unknown quality variant: $variant"
    fi

    printf 'Scoring 100 production-path quality cases: %s...\n' "$variant"
    "${clean[@]}" "${variant_env[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        DS4_CUDA_PREFILL_PIPELINE_MB=512 \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK" \
        "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
        ./gguf-tools/quality-testing/score_official \
            "$MODEL" "$QUALITY_MANIFEST" "$out" "$QUALITY_CTX" \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --cuda-tensor-parallel --warm-weights --production-path \
            2>&1 | tee "$log"

    [[ -s $out && -s $audit && -s $bindings ]] ||
        die "$variant quality run omitted required evidence"
    awk -F'\t' 'NR > 1 {n++} END {exit n == 100 ? 0 : 1}' "$out" ||
        die "$variant quality output does not contain exactly 100 cases"
    grep -Fq 'score_official: runtime_path=production' "$log" ||
        die "$variant quality run did not use production dispatch"
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant quality run did not use the requested split"
    python3 speed-bench/q8_partner_audit.py \
        "$variant" "$partner_device_0" "$partner_device_1" "$audit" \
        >"$OUTPUT_DIR/quality/$variant-audit-validation.txt" ||
        die "$variant quality run has invalid partner-class evidence"

    if [[ $variant == local ]]; then
        ! awk -F, 'NR > 1 && $3 == 1 {found=1} END {exit found ? 0 : 1}' \
            "$bindings" || die "local quality control has a partner binding"
        ! grep -Fq 'CUDA q8 fp16 partner summary:' "$log" ||
            die "local quality control executed a partner projection"
    else
        grep -Fq 'partner-classes=t256' "$log" ||
            die "default quality run did not select T256"
        grep -Fq 'CUDA q8 fp16 partner summary:' "$log" ||
            die "default quality run did not execute partner projections"
        awk -F, 'NR > 1 && $3 == 1 && $6 == 8192 && $7 == 4096 {n++}
                   END {exit n > 0 ? 0 : 1}' "$bindings" ||
            die "default quality run has no T256 partner binding"
        awk -F, 'NR > 1 && $3 == 1 && !($6 == 8192 && $7 == 4096) {bad=1}
                   END {exit bad}' "$bindings" ||
            die "default quality run has a non-T256 partner binding"
    fi
}

phase=quality-local
run_quality local
phase=quality-default
run_quality default
python3 gguf-tools/quality-testing/compare_scores.py \
    "$OUTPUT_DIR/quality/local.tsv" "$OUTPUT_DIR/quality/default.tsv" \
    | tee "$OUTPUT_DIR/quality/local-vs-default.txt"

phase=long-context-prefill
"${clean[@]}" \
    "MODEL=$MODEL" "PROMPT=$PROMPT" \
    "GPU_DEVICES=$GPU_DEVICES" "GPU_VRAM=$GPU_VRAM" \
    "STAGE_SPLIT=$STAGE_SPLIT" \
    "CTX_START=$CTX_START" "CTX_MAX=$CTX_MAX" "STEP_MUL=$STEP_MUL" \
    "PREFILL_CHUNK=$PREFILL_CHUNK" "REPEATS=$REPEATS" \
    AB_MODE=production VARIANTS=local,default RUN_NSYS=0 \
    SKIP_BUILD=1 RUN_GPU_TEST=0 CREATE_ARCHIVE=0 \
    "Q8_PARTNER_AB_DIR=$OUTPUT_DIR/performance" \
    bash ./speed-bench/cuda-q8-partner-offload-ab.sh

phase=acceptance
python3 speed-bench/summarize-q8-partner-production.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/acceptance-stdout.txt"
[[ -s $OUTPUT_DIR/acceptance.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "production acceptance summary is missing"

phase=complete
printf 'Production-default T256 validation complete: %s\n' "$OUTPUT_DIR"
