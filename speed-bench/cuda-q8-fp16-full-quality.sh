#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute full-Q4 GGUF path"
QUALITY_MANIFEST=${QUALITY_MANIFEST:-$repo_dir/gguf-tools/quality-testing/data/flash/manifest.tsv}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
QUALITY_CTX=${QUALITY_CTX:-32769}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_FP16_FULL_QUALITY_DIR:-$repo_dir/q8-fp16-full-quality-$stamp}

[[ $QUALITY_MANIFEST == /* && -f $QUALITY_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute path"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "comparison requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
[[ $QUALITY_CTX == 32769 && $PREFILL_CHUNK == 2048 ]] ||
    die "comparison requires QUALITY_CTX=32769 PREFILL_CHUNK=2048"
for item in "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value == 0 || $value == 1 ]] || die "$name must be 0 or 1"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{quality,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv "$partial" "$archive"; then
            printf '%s: %s\n' \
                "$([[ $status == 0 ]] && printf 'Archive to return' || printf 'Diagnostic archive')" \
                "$archive" >&2
        else
            printf 'error: failed to create nonempty archive: %s\n' "$archive" >&2
            [[ $status != 0 ]] || status=1
            rm -f -- "$partial"
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT
trap 'phase=interrupted; exit 143' TERM

for command in nvidia-smi python3 make tar awk; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

phase=topology
nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt"
topology_link() {
    local from=$1
    local to=$2
    awk -v from="GPU$from" -v to="GPU$to" '
        !header {
            n_gpu = 0
            for (i = 1; i <= NF; i++) if ($i ~ /^GPU[0-9]+$/) n_gpu++
            if (n_gpu > 1) {
                for (i = 1; i <= NF; i++) if ($i == to) column = i + 1
                header = 1
                next
            }
        }
        header && $1 == from && column > 0 { print $column; exit }
    ' "$OUTPUT_DIR/provenance/topology.txt"
}
for pair in '0 1' '2 3'; do
    read -r first second <<<"$pair"
    forward=$(topology_link "$first" "$second")
    reverse=$(topology_link "$second" "$first")
    # NVLink is physical and bidirectional. Some nvidia-smi releases omit one
    # symmetric matrix row; accept one unambiguous NV# report. DS4 separately
    # validates CUDA DIRECT peer access in both directions before admitting any
    # partner execution.
    [[ $forward =~ ^NV[0-9]+$ || $reverse =~ ^NV[0-9]+$ ]] ||
        die "physical GPU pair $first<->$second is not reported as NVLink: ${forward:-missing}/${reverse:-missing}"
    [[ -z $forward || $forward =~ ^NV[0-9]+$ ]] ||
        die "inconsistent NVLink topology for physical $first<->$second: $forward/${reverse:-missing}"
    [[ -z $reverse || $reverse =~ ^NV[0-9]+$ ]] ||
        die "inconsistent NVLink topology for physical $first<->$second: ${forward:-missing}/$reverse"
done

{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'quality_manifest=%s\nquality_ctx=%s\ngpu_devices=%s\ngpu_vram=%s\n' \
        "$QUALITY_MANIFEST" "$QUALITY_CTX" "$GPU_DEVICES" "$GPU_VRAM"
    printf 'stage_split=%s/%s\nprefill_chunk=%s\nt256_layers=0-42\n' \
        "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$PREFILL_CHUNK"
    printf 'comparison=all-native-q8-vs-complete-production-fp16-cache\n'
    printf 'native_expected_expanded_bindings=0\n'
    printf 'fp16_expected_t256_bindings=43/43\n'
    printf 'fp16_expected_placement=43-partner\n'
    printf 'fp16_expected_unique_t256_allocations=43\n'
    printf 'fp16_non_t256_inventory=dynamic-production-policy\n'
    printf 'expanded_weight_liveness=all-bindings-and-allocations-live\n'
    printf 'model_hashing=disabled\nquality_runtime=production\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
        --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print $0}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found stale targets"
fi

phase=tests
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2
    die "planner unit test failed"
}
"${clean[@]}" ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
    tail -n 200 "$OUTPUT_DIR/gpu-exactness.log" >&2
    die "multi-GPU exactness test failed"
}
grep -Fq 'q8 partner projection exactness OK (3 classes)' \
    "$OUTPUT_DIR/gpu-exactness.log" ||
    die "GPU regression did not prove local/partner FP16 projection exactness"

common_env=(
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
)
native_env=(
    DS4_CUDA_NO_Q8_F16_CACHE=1
    DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=none
    DS4_CUDA_Q8_T256_PLACEMENT=overflow
)
full_env=(
    DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256
    DS4_CUDA_Q8_F16_PARTNER_LAYERS=0-42
    DS4_CUDA_Q8_PARTNER_ARITHMETIC=f16
    DS4_CUDA_Q8_T256_PLACEMENT=all-partner
)

run_quality() {
    local arm=$1
    local out="$OUTPUT_DIR/quality/$arm.tsv"
    local log="$OUTPUT_DIR/quality/$arm.log"
    local audit="$OUTPUT_DIR/quality/$arm.q8-audit.csv"
    local bindings="$OUTPUT_DIR/quality/$arm.bindings.csv"
    local allocations="$OUTPUT_DIR/quality/$arm.allocations.csv"
    local -a mode_env=()
    if [[ $arm == native-q8 ]]; then
        mode_env=("${native_env[@]}")
    else
        mode_env=("${full_env[@]}")
    fi

    printf 'Scoring 100 production-path quality cases: %s...\n' "$arm"
    "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
        "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
        "DS4_CUDA_Q8_ALLOCATION_STATE_CSV=$allocations" \
        ./gguf-tools/quality-testing/score_official \
            "$MODEL" "$QUALITY_MANIFEST" "$out" "$QUALITY_CTX" \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --cuda-tensor-parallel --warm-weights --production-path \
            2>&1 | tee "$log"
    [[ -s $out && -s $audit && -s $bindings && -s $allocations ]] ||
        die "$arm omitted required score or execution evidence"
    awk -F'\t' 'NR > 1 {n++} END {exit n == 100 ? 0 : 1}' "$out" ||
        die "$arm quality output does not contain exactly 100 cases"
    python3 speed-bench/summarize-q8-fp16-full-quality.py \
        --coverage-only "$OUTPUT_DIR" "$arm" \
        >"$OUTPUT_DIR/quality/$arm-coverage.json"
}

phase=quality-native-q8
run_quality native-q8
phase=quality-production-fp16-cache
run_quality production-fp16-cache

phase=compare
python3 gguf-tools/quality-testing/compare_scores.py \
    "$OUTPUT_DIR/quality/native-q8.tsv" \
    "$OUTPUT_DIR/quality/production-fp16-cache.tsv" \
    >"$OUTPUT_DIR/quality/all-native-q8-vs-production-fp16-cache.txt"
python3 speed-bench/summarize-q8-fp16-full-quality.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/quality-comparison.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "quality comparison summary is missing"

phase=complete
printf 'All-native Q8 versus complete production FP16-cache quality comparison complete: %s\n' \
    "$OUTPUT_DIR"
