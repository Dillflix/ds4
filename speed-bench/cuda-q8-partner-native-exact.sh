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
QUALITY_CTX=${QUALITY_CTX:-32769}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
T256_LAYERS=${T256_LAYERS:-15-21}
CTX_START=${CTX_START:-16384}
CTX_MAX=${CTX_MAX:-32768}
REPEATS=${REPEATS:-3}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_QUALITY=${RUN_QUALITY:-1}
RUN_PERF=${RUN_PERF:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_NATIVE_EXACT_DIR:-$repo_dir/q8-partner-native-exact-$stamp}

[[ $PROMPT == /* && -f $PROMPT ]] || die "PROMPT must name an existing absolute path"
[[ $QUALITY_MANIFEST == /* && -f $QUALITY_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute path"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "native exactness requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
[[ $QUALITY_CTX == 32769 && $PREFILL_CHUNK == 2048 &&
   $T256_LAYERS == 15-21 && $CTX_START == 16384 && $CTX_MAX == 32768 ]] ||
    die "native exactness requires quality_ctx=32769, chunk=2048, layers=15-21, contexts=16K-32K"
for item in "REPEATS:$REPEATS" "SKIP_BUILD:$SKIP_BUILD" \
            "RUN_QUALITY:$RUN_QUALITY" "RUN_PERF:$RUN_PERF" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( REPEATS >= 3 )) || die "REPEATS must be at least 3"
for flag in SKIP_BUILD RUN_QUALITY RUN_PERF CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $RUN_QUALITY == 1 && $RUN_PERF == 1 ]] ||
    die "the exactness evidence pass requires both quality and performance phases"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{quality,runs,logits,provenance}
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
local_env=(
    DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=none
)
native_env=(
    DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256
    DS4_CUDA_Q8_F16_PARTNER_LAYERS=15-21
    DS4_CUDA_Q8_PARTNER_ARITHMETIC=native-q8
)
common_env=(
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
)

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
    [[ $forward =~ ^NV[0-9]+$ || $reverse =~ ^NV[0-9]+$ ]] ||
        die "physical GPU pair $first<->$second is not reported as NVLink: ${forward:-missing}/${reverse:-missing}"
done
{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'quality_manifest=%s\nquality_ctx=%s\ngpu_devices=%s\ngpu_vram=%s\n' \
        "$QUALITY_MANIFEST" "$QUALITY_CTX" "$GPU_DEVICES" "$GPU_VRAM"
    printf 'stage_split=%s/%s\nprefill_chunk=%s\ncontexts=%s,%s\nrepeats=%s\n' \
        "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$PREFILL_CHUNK" \
        "$CTX_START" "$CTX_MAX" "$REPEATS"
    printf 't256_layers=%s\npartner_arithmetic=native-q8\nhome_plan=frozen\n' \
        "$T256_LAYERS"
    printf 'model_hashing=disabled\nquality_comparison=byte-exact\n'
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
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2
    die "planner unit test failed"
}
"${clean[@]}" ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
    tail -n 200 "$OUTPUT_DIR/gpu-exactness.log" >&2
    die "multi-GPU exactness test failed"
}
grep -Fq 'native q8 partner T256 exactness OK' "$OUTPUT_DIR/gpu-exactness.log" ||
    die "GPU regression did not prove native-Q8 local/partner bit exactness"

run_quality() {
    local variant=$1
    local out="$OUTPUT_DIR/quality/$variant.tsv"
    local log="$OUTPUT_DIR/quality/$variant.log"
    local audit="$OUTPUT_DIR/quality/$variant.q8-audit.csv"
    local bindings="$OUTPUT_DIR/quality/$variant.bindings.csv"
    local -a mode_env=()
    if [[ $variant == local ]]; then mode_env=("${local_env[@]}")
    else mode_env=("${native_env[@]}"); fi
    printf 'Scoring 100 production cases: %s...\n' "$variant"
    "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
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
    grep -Fq 'CUDA EP forced pipeline split 22/21' "$log" ||
        die "$variant did not use the fixed split"
    if [[ $variant == local ]]; then
        ! grep -Fq 'CUDA q8 partner execution enabled:' "$log" ||
            die "local quality control executed partner work"
    else
        grep -Fq 'partner-arithmetic=native-q8' "$log" ||
            die "native quality arm did not select exact native arithmetic"
        grep -Fq 'home-order=frozen' "$log" ||
            die "native quality arm did not freeze home admission"
        grep -Fq 'arithmetic=native-q8' "$log" ||
            die "native quality arm did not execute partner work"
        grep -Fq ',native_q8_partner_hit,exact_sm75_mma,' "$audit" ||
            die "native quality audit lacks an exact SM75 partner hit"
    fi
}

phase=quality-local
run_quality local
phase=quality-native
run_quality native-q8
cmp -s "$OUTPUT_DIR/quality/local.tsv" "$OUTPUT_DIR/quality/native-q8.tsv" ||
    die "native-Q8 quality output differs from local; performance phase refused"

printf 'repeat\tslot\tvariant\tcsv\tlog\taudit\tbindings\tlogits\n' \
    >"$OUTPUT_DIR/runs.tsv"
phase=performance
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then order=(local native-q8)
    else order=(native-q8 local); fi
    for slot_index in 0 1; do
        variant=${order[$slot_index]}
        stem="$variant-r$repeat"
        csv="$OUTPUT_DIR/runs/$stem.csv"
        log="$OUTPUT_DIR/runs/$stem.log"
        audit="$OUTPUT_DIR/runs/$stem.q8-audit.csv"
        bindings="$OUTPUT_DIR/runs/$stem.bindings.csv"
        logits="$OUTPUT_DIR/logits/$stem"
        mkdir -p "$logits"
        mode_env=()
        if [[ $variant == local ]]; then mode_env=("${local_env[@]}")
        else mode_env=("${native_env[@]}"); fi
        printf 'Benchmarking %s repeat=%d/%d...\n' "$variant" "$repeat" "$REPEATS"
        "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
            "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$((CTX_MAX+1))" --step-mul 2 \
                --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 --csv "$csv" \
                --dump-frontier-logits-dir "$logits" >"$log" 2>&1 || {
                    tail -n 200 "$log" >&2
                    die "$stem failed"
                }
        [[ -s $csv && -s $audit && -s $bindings ]] ||
            die "$stem omitted required performance evidence"
        grep -Fq 'CUDA EP forced pipeline split 22/21' "$log" ||
            die "$stem did not use the fixed split"
        if [[ $variant == local ]]; then
            ! grep -Fq 'CUDA q8 partner execution enabled:' "$log" ||
                die "$stem unexpectedly executed partner work"
        else
            grep -Fq 'arithmetic=native-q8' "$log" ||
                die "$stem did not execute native-Q8 partner work"
            grep -Fq ',native_q8_partner_hit,exact_sm75_mma,' "$audit" ||
                die "$stem audit lacks exact SM75 partner work"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$((slot_index+1))" "$variant" "$csv" "$log" \
            "$audit" "$bindings" "$logits" >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=summarize
python3 speed-bench/summarize-q8-partner-native-exact.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/native-exact.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "native exactness summary is missing"
phase=complete
printf 'Native-Q8 partner exactness pass complete: %s\n' "$OUTPUT_DIR"
