#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute full-Q4 GGUF path"
SOURCE_MANIFEST=${QUALITY_MANIFEST:-$repo_dir/gguf-tools/quality-testing/data/flash/manifest.tsv}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
QUALITY_CTX=${QUALITY_CTX:-32769}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
T256_LAYERS=${T256_LAYERS:-15-21}
CASE_IDS=${CASE_IDS:-case_017,case_025,case_030,case_048,case_056}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_ARITHMETIC_DIR:-$repo_dir/q8-partner-arithmetic-$stamp}
variants=(local f16 w16-x16-sgemm w16-x32-sgemm w32-x32-sgemm w32-xq8-sgemm)

[[ $SOURCE_MANIFEST == /* && -f $SOURCE_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute path"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "arithmetic isolation requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
[[ $QUALITY_CTX == 32769 && $PREFILL_CHUNK =~ ^[1-9][0-9]*$ ]] ||
    die "arithmetic isolation requires QUALITY_CTX=32769 and a positive PREFILL_CHUNK"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
[[ $T256_LAYERS =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] ||
    die "T256_LAYERS must be a comma-separated layer/range list"
[[ $CASE_IDS =~ ^case_[0-9]{3}(,case_[0-9]{3})*$ ]] ||
    die "CASE_IDS must be comma-separated case_NNN identifiers"
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

for command in nvidia-smi python3 make tar awk; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done

IFS=',' read -r -a requested_cases <<<"$CASE_IDS"
declare -A wanted=()
for case_id in "${requested_cases[@]}"; do
    [[ -z ${wanted[$case_id]+x} ]] || die "CASE_IDS contains duplicate $case_id"
    wanted[$case_id]=1
done
filtered_manifest="$OUTPUT_DIR/quality/manifest.tsv"
awk -F'\t' -v ids="$CASE_IDS" '
    BEGIN { n=split(ids,a,","); for (i=1;i<=n;i++) wanted[a[i]]=1 }
    /^#/ { print; next }
    $1 in wanted { print; seen[$1]=1 }
    END { for (id in wanted) if (!(id in seen)) exit 7 }
' "$SOURCE_MANIFEST" >"$filtered_manifest" ||
    die "one or more CASE_IDS are absent from QUALITY_MANIFEST"
actual_cases=$(awk '!/^#/ && NF {n++} END {print n+0}' "$filtered_manifest")
(( actual_cases == ${#requested_cases[@]} )) || die "filtered manifest count mismatch"

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt"
{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'source_manifest=%s\ncase_ids=%s\ncase_count=%s\n' \
        "$SOURCE_MANIFEST" "$CASE_IDS" "$actual_cases"
    printf 'quality_ctx=%s\ngpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$QUALITY_CTX" "$GPU_DEVICES" "$GPU_VRAM" \
        "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'prefill_chunk=%s\nt256_layers=%s\n' "$PREFILL_CHUNK" "$T256_LAYERS"
    printf 'variants=%s\nmodel_hashing=disabled\nhome_plan=frozen-for-all-arms\n' \
        "$(IFS=,; printf '%s' "${variants[*]}")"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print $0}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2
    die "planner unit test failed"
}

for ((variant_index=0; variant_index<${#variants[@]}; variant_index++)); do
    variant=${variants[$variant_index]}
    phase="quality-$variant"
    out="$OUTPUT_DIR/quality/$variant.tsv"
    log="$OUTPUT_DIR/quality/$variant.log"
    audit="$OUTPUT_DIR/quality/$variant.q8-audit.csv"
    bindings="$OUTPUT_DIR/quality/$variant.bindings.csv"
    variant_env=()
    if [[ $variant == local ]]; then
        variant_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
        variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
    else
        variant_env+=(DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1)
        variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256)
        variant_env+=("DS4_CUDA_Q8_F16_PARTNER_LAYERS=$T256_LAYERS")
        variant_env+=("DS4_CUDA_Q8_PARTNER_ARITHMETIC=$variant")
    fi
    printf 'Scoring production cases with arithmetic arm %s (%d/%d)...\n' \
        "$variant" "$((variant_index + 1))" "${#variants[@]}"
    "${clean[@]}" "${variant_env[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        DS4_CUDA_PREFILL_PIPELINE_MB=512 \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK" \
        "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
        ./gguf-tools/quality-testing/score_official \
            "$MODEL" "$filtered_manifest" "$out" "$QUALITY_CTX" \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --cuda-tensor-parallel --warm-weights --production-path \
            2>&1 | tee "$log"
    [[ -s $out && -s $audit && -s $bindings ]] ||
        die "$variant omitted required evidence"
    rows=$(awk -F'\t' 'NR > 1 {n++} END {print n+0}' "$out")
    (( rows == actual_cases )) || die "$variant scored $rows/$actual_cases cases"
    grep -Fq 'score_official: runtime_path=production' "$log" ||
        die "$variant did not use production dispatch"
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant did not use the requested split"
    if [[ $variant == local ]]; then
        ! grep -Fq 'CUDA q8 partner execution enabled:' "$log" ||
            die "local control executed partner work"
    else
        grep -Fq "partner-arithmetic=$variant" "$log" ||
            die "$variant did not select the requested arithmetic"
        grep -Fq 'home-order=frozen' "$log" ||
            die "$variant did not freeze home admission"
        grep -Fq 'CUDA q8 partner summary:' "$log" ||
            die "$variant did not execute partner work"
    fi
done

phase=summarize
python3 speed-bench/summarize-q8-partner-arithmetic.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/arithmetic-isolation.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "arithmetic summary is missing"
phase=complete
printf 'Q8 partner arithmetic isolation complete: %s\n' "$OUTPUT_DIR"
