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
RUN_GPU_TEST=${RUN_GPU_TEST:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
REUSE_LOCAL_DIR=${REUSE_LOCAL_DIR:-}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_PARTNER_QUALITY_ISOLATION_DIR:-$repo_dir/q8-partner-quality-isolation-$stamp}

[[ $QUALITY_MANIFEST == /* && -f $QUALITY_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute path"
if [[ -n $REUSE_LOCAL_DIR ]]; then
    [[ $REUSE_LOCAL_DIR == /* && -d $REUSE_LOCAL_DIR ]] ||
        die "REUSE_LOCAL_DIR must name an existing absolute validation directory"
    REUSE_LOCAL_DIR=$(cd "$REUSE_LOCAL_DIR" && pwd)
fi
for item in "STAGE_SPLIT:$STAGE_SPLIT" "QUALITY_CTX:$QUALITY_CTX" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "SKIP_BUILD:$SKIP_BUILD" \
            "RUN_GPU_TEST:$RUN_GPU_TEST" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "quality isolation requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
(( QUALITY_CTX == 32769 )) || die "quality isolation requires QUALITY_CTX=32769"
(( PREFILL_CHUNK > 0 )) || die "PREFILL_CHUNK must be positive"
for flag in SKIP_BUILD RUN_GPU_TEST CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $RUN_GPU_TEST == 1 ]] ||
    die "quality isolation requires the three-class GPU exactness test"

IFS=',' read -r -a gpu_device_ids <<<"$GPU_DEVICES"
(( ${#gpu_device_ids[@]} == 4 )) || die "GPU_DEVICES must contain four devices"
partner_device_0=${gpu_device_ids[2]}
partner_device_1=${gpu_device_ids[3]}

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
trap 'exit 130' INT
trap 'exit 143' TERM

for command in nvidia-smi python3 make tar; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt"
{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'quality_manifest=%s\nquality_ctx=%s\n' \
        "$QUALITY_MANIFEST" "$QUALITY_CTX"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'prefill_chunk=%s\nhome_plan=frozen-for-candidates\n' "$PREFILL_CHUNK"
    printf 'variants=local,t256,t32\nmodel_hashing=disabled\n'
    printf 'reuse_local_dir=%s\n' "$REUSE_LOCAL_DIR"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free --format=csv
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
./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2
    die "planner unit test failed"
}
./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/gpu-exactness.log" >&2
    die "multi-GPU exactness test failed"
}
grep -Fq 'q8 partner projection exactness OK (3 classes)' \
    "$OUTPUT_DIR/gpu-exactness.log" ||
    die "GPU exactness did not exercise all three partner classes"

run_quality() {
    local variant=$1
    local out="$OUTPUT_DIR/quality/$variant.tsv"
    local log="$OUTPUT_DIR/quality/$variant.log"
    local audit="$OUTPUT_DIR/quality/$variant.q8-audit.csv"
    local bindings="$OUTPUT_DIR/quality/$variant.bindings.csv"
    local -a variant_env=()
    case "$variant" in
        local)
            variant_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
            variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
            ;;
        t256|t32)
            variant_env+=(DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1)
            variant_env+=("DS4_CUDA_Q8_F16_PARTNER_CLASSES=$variant")
            ;;
        *) die "internal unknown quality variant: $variant" ;;
    esac

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
        die "$variant omitted required quality evidence"
    awk -F'\t' 'NR > 1 {n++} END {exit n == 100 ? 0 : 1}' "$out" ||
        die "$variant quality output does not contain exactly 100 cases"
    grep -Fq 'score_official: runtime_path=production' "$log" ||
        die "$variant did not use production dispatch"
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant did not use the requested split"
    python3 speed-bench/q8_partner_audit.py \
        "$variant" "$partner_device_0" "$partner_device_1" "$audit" \
        >"$OUTPUT_DIR/quality/$variant-audit-validation.txt" ||
        die "$variant has invalid partner-class evidence"

    if [[ $variant == local ]]; then
        ! grep -Fq 'CUDA q8 fp16 partner summary:' "$log" ||
            die "local control executed partner work"
    else
        grep -Fq "partner-classes=$variant" "$log" ||
            die "$variant did not select its isolated class"
        grep -Fq 'home-order=frozen' "$log" ||
            die "$variant did not freeze primary/home admission"
        grep -Fq 'CUDA q8 fp16 partner summary:' "$log" ||
            die "$variant did not execute partner projections"
    fi
}

phase=quality-local
if [[ -n $REUSE_LOCAL_DIR ]]; then
    source_manifest="$REUSE_LOCAL_DIR/manifest.txt"
    [[ -f $source_manifest ]] || die "REUSE_LOCAL_DIR lacks manifest.txt"
    manifest_value() {
        local key=$1
        awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' \
            "$source_manifest"
    }
    [[ $(manifest_value model) == "$MODEL" &&
       $(manifest_value model_bytes) == "$(stat -c %s "$MODEL")" &&
       $(manifest_value quality_manifest) == "$QUALITY_MANIFEST" &&
       $(manifest_value quality_ctx) == "$QUALITY_CTX" &&
       $(manifest_value gpu_devices) == "$GPU_DEVICES" &&
       $(manifest_value gpu_vram) == "$GPU_VRAM" &&
       $(manifest_value stage_split) == "$STAGE_SPLIT/$((43-STAGE_SPLIT))" ]] ||
        die "REUSE_LOCAL_DIR manifest does not match this experiment"
    for suffix in tsv log q8-audit.csv bindings.csv audit-validation.txt; do
        source_file="$REUSE_LOCAL_DIR/quality/local.$suffix"
        if [[ $suffix == audit-validation.txt ]]; then
            source_file="$REUSE_LOCAL_DIR/quality/local-audit-validation.txt"
            destination="$OUTPUT_DIR/quality/local-audit-validation.txt"
        else
            destination="$OUTPUT_DIR/quality/local.$suffix"
        fi
        [[ -s $source_file ]] || die "REUSE_LOCAL_DIR lacks $source_file"
        cp -- "$source_file" "$destination"
    done
    printf 'reused_local_from=%s\n' "$REUSE_LOCAL_DIR" \
        >"$OUTPUT_DIR/quality/local-reuse.txt"
else
    run_quality local
fi
for variant in t256 t32; do
    phase="quality-$variant"
    run_quality "$variant"
    python3 gguf-tools/quality-testing/compare_scores.py \
        "$OUTPUT_DIR/quality/local.tsv" "$OUTPUT_DIR/quality/$variant.tsv" \
        >"$OUTPUT_DIR/quality/local-vs-$variant.txt"
done

phase=summarize
python3 speed-bench/summarize-q8-partner-quality-isolation.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/quality-isolation.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "quality-isolation summary is missing"

phase=complete
printf 'Frozen-home Q8 partner quality isolation complete: %s\n' "$OUTPUT_DIR"
