#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] || die "MODEL must be an existing absolute path"
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-32768}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-5}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_T256_PLACEMENT_DIR:-$repo_dir/q8-t256-placement-$stamp}
variants=(native all-local balanced overflow all-partner)

[[ $PROMPT == /* && -f $PROMPT ]] || die "PROMPT must be an existing absolute path"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "placement A/B requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" "STEP_MUL:$STEP_MUL" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "REPEATS:$REPEATS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START >= 2048 && CTX_MAX >= CTX_START && CTX_MAX <= 32768 )) ||
    die "contexts must stay within the fixed 2K-32K range"
(( STEP_MUL == 2 && PREFILL_CHUNK == 2048 && REPEATS >= 1 )) ||
    die "placement A/B requires STEP_MUL=2 PREFILL_CHUNK=2048 REPEATS>=1"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] || die "CREATE_ARCHIVE must be 0 or 1"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] || die "output exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"; partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv "$partial" "$archive"; then
            printf '%s: %s\n' \
                "$([[ $status == 0 ]] && printf 'Archive to return' || printf 'Diagnostic archive')" \
                "$archive" >&2
        else
            printf 'error: failed to create archive %s\n' "$archive" >&2
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
for pair in 'GPU0 GPU1' 'GPU2 GPU3'; do
    read -r first second <<<"$pair"
    forward=$(awk -v from="$first" -v to="$second" '
        !h {for(i=1;i<=NF;i++) if($i==to)c=i+1; if(c){h=1;next}}
        h && $1==from {print $c;exit}' "$OUTPUT_DIR/provenance/topology.txt")
    reverse=$(awk -v from="$second" -v to="$first" '
        !h {for(i=1;i<=NF;i++) if($i==to)c=i+1; if(c){h=1;next}}
        h && $1==from {print $c;exit}' "$OUTPUT_DIR/provenance/topology.txt")
    [[ $forward =~ ^NV[0-9]+$ && $reverse =~ ^NV[0-9]+$ ]] ||
        die "$first<->$second is not bidirectional NVLink: ${forward:-missing}/${reverse:-missing}"
done

{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK" "$REPEATS"
    printf 'variants=native,all-local,balanced,overflow,all-partner\n'
    printf 'overflow_partner_eligibility=15-21\nmodel_hashing=disabled\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print $0}' | sort -u >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        tests/test_gpu_xdev CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found stale targets"
fi
phase=tests
"${clean[@]}" ./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2; die "planner test failed"; }
"${clean[@]}" ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
    tail -n 200 "$OUTPUT_DIR/gpu-exactness.log" >&2; die "GPU exactness test failed"; }
grep -Fq 'q8 partner projection exactness OK (3 classes)' "$OUTPUT_DIR/gpu-exactness.log" ||
    die "GPU regression lacks local/partner FP16 exactness evidence"

printf 'repeat\tslot\tvariant\tcsv\tlog\taudit\tbindings\tallocations\n' >"$OUTPUT_DIR/runs.tsv"
common_env=(
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
)

phase=benchmark
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for ((slot=0; slot<5; slot++)); do
        variant=${variants[$(((slot + repeat - 1) % 5))]}
        stem="$variant-r$repeat"
        csv="$OUTPUT_DIR/runs/$stem.csv"
        log="$OUTPUT_DIR/runs/$stem.log"
        audit="$OUTPUT_DIR/runs/$stem.q8-audit.csv"
        bindings="$OUTPUT_DIR/runs/$stem.bindings.csv"
        allocations="$OUTPUT_DIR/runs/$stem.allocations.csv"
        mode_env=()
        case "$variant" in
            native)
                mode_env+=(DS4_CUDA_NO_T256_F16_CACHE=1)
                mode_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
                mode_env+=(DS4_CUDA_Q8_T256_PLACEMENT=overflow)
                ;;
            all-local)
                mode_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
                mode_env+=(DS4_CUDA_Q8_T256_PLACEMENT=all-local)
                ;;
            balanced)
                mode_env+=(DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_LAYERS=0-42)
                mode_env+=(DS4_CUDA_Q8_PARTNER_ARITHMETIC=f16)
                mode_env+=(DS4_CUDA_Q8_T256_PLACEMENT=balanced)
                ;;
            overflow)
                mode_env+=(DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_LAYERS=15-21)
                mode_env+=(DS4_CUDA_Q8_PARTNER_ARITHMETIC=f16)
                mode_env+=(DS4_CUDA_Q8_T256_PLACEMENT=overflow)
                ;;
            all-partner)
                mode_env+=(DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256)
                mode_env+=(DS4_CUDA_Q8_F16_PARTNER_LAYERS=0-42)
                mode_env+=(DS4_CUDA_Q8_PARTNER_ARITHMETIC=f16)
                mode_env+=(DS4_CUDA_Q8_T256_PLACEMENT=all-partner)
                ;;
        esac
        printf 'Benchmarking %s repeat=%d/%d slot=%d/5...\n' \
            "$variant" "$repeat" "$REPEATS" "$((slot+1))"
        "${clean[@]}" "${common_env[@]}" "${mode_env[@]}" \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
            "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
            "DS4_CUDA_Q8_ALLOCATION_STATE_CSV=$allocations" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$((CTX_MAX+1))" --step-mul "$STEP_MUL" \
                --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 --csv "$csv" \
                >"$log" 2>&1 || {
                    tail -n 200 "$log" >&2; die "$stem failed"; }
        [[ -s $csv && -s $audit && -s $bindings && -s $allocations ]] ||
            die "$stem omitted evidence"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$((slot+1))" "$variant" "$csv" "$log" \
            "$audit" "$bindings" "$allocations" >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=summarize
python3 speed-bench/summarize-q8-t256-placement.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/t256-placement.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "placement summary is missing"
phase=complete
printf 'T256 execution-placement comparison complete: %s\n' "$OUTPUT_DIR"
