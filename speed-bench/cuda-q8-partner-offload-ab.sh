#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] || die "MODEL must name an existing absolute path"
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-8192}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-3}
VARIANTS=${VARIANTS:-local,t32,t256,shared_down,legacy}
RUN_NSYS=${RUN_NSYS:-1}
NSYS_VARIANTS=${NSYS_VARIANTS:-t32,t256,shared_down}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_GPU_TEST=${RUN_GPU_TEST:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_PARTNER_AB_DIR:-$repo_dir/q8-partner-offload-ab-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "STEP_MUL:$STEP_MUL" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "REPEATS:$REPEATS" \
            "PROFILE_TOKENS:$PROFILE_TOKENS" "SKIP_BUILD:$SKIP_BUILD" \
            "RUN_GPU_TEST:$RUN_GPU_TEST" "RUN_NSYS:$RUN_NSYS" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 )) || die "STAGE_SPLIT must be in 1..42"
(( CTX_START > 0 && CTX_MAX >= CTX_START && PROFILE_TOKENS > 0 )) ||
    die "invalid context/profile range"
(( REPEATS >= 2 )) || die "REPEATS must be at least 2"
for flag in SKIP_BUILD RUN_GPU_TEST RUN_NSYS CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done

IFS=',' read -r -a variants <<<"$VARIANTS"
(( ${#variants[@]} >= 2 )) || die "VARIANTS must contain local and at least one candidate"
declare -A seen=()
have_local=0
for variant in "${variants[@]}"; do
    case "$variant" in
        local|t32|t256|shared_down|legacy) ;;
        *) die "unsupported variant: $variant" ;;
    esac
    [[ -z ${seen[$variant]+x} ]] || die "duplicate variant: $variant"
    seen[$variant]=1
    [[ $variant == local ]] && have_local=1
done
(( have_local == 1 )) || die "VARIANTS must include local"
IFS=',' read -r -a nsys_variants <<<"$NSYS_VARIANTS"
for variant in "${nsys_variants[@]}"; do
    [[ $variant == t32 || $variant == t256 || $variant == shared_down ||
       $variant == legacy ]] || die "unsupported NSYS_VARIANTS entry: $variant"
    [[ -n ${seen[$variant]+x} ]] ||
        die "Nsight variant $variant is not present in VARIANTS"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,nsys,provenance,telemetry}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s.tar.gz\n' "$OUTPUT_DIR"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

variant_extra=()
set_variant_extra() {
    variant_extra=()
    case "$1" in
        local)
            variant_extra+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
            variant_extra+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
            ;;
        legacy) variant_extra+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=legacy) ;;
        t32) variant_extra+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t32) ;;
        t256) variant_extra+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256) ;;
        shared_down)
            variant_extra+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=shared_down)
            ;;
        *) die "internal unknown variant: $1" ;;
    esac
}

validate_audit() {
    local variant=$1
    local audit=$2
    local counts
    counts=$(awk -F, -v variant="$variant" '
        NR == 1 { next }
        $12 == "f16_partner_hit" && $13 == "nvlink_offload" {
            if ($2 ~ /attn_q_b/ || $3 ~ /attn_q_b/ || ($9 == 1024 && $10 == 32768)) t32++
            else if ($2 ~ /attn_output_b/ || $3 ~ /attn_output_b/ || ($9 == 8192 && $10 == 4096)) t256++
            else if ($2 ~ /shared_down/ || $3 ~ /ffn_down_shexp/ || ($9 == 2048 && $10 == 4096)) shared++
            else other++
        }
        END {
            total = t32 + t256 + shared + other
            printf "total=%d t32=%d t256=%d shared_down=%d other=%d", total, t32, t256, shared, other
            ok = 0
            if (variant == "local") ok = (total == 0)
            else if (variant == "t32") ok = (t32 > 0 && total == t32)
            else if (variant == "t256") ok = (t256 > 0 && total == t256)
            else if (variant == "shared_down") ok = (shared > 0 && total == shared)
            else if (variant == "legacy") ok = (total > 0 && total == t32 + t256)
            exit(ok ? 0 : 1)
        }
    ' "$audit") || {
        printf '%s: %s\n' "$variant" "$counts"
        return 1
    }
    printf '%s: %s\n' "$variant" "$counts"
}

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        tests/test_gpu_xdev CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
    ./tests/test_engine_mgpu_placement \
        >"$OUTPUT_DIR/planner-unit.log" 2>&1 || die "planner unit test failed"
    if [[ $RUN_GPU_TEST == 1 ]]; then
        ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
            tail -n 160 "$OUTPUT_DIR/gpu-exactness.log" >&2
            die "multi-GPU exactness test failed"
        }
        grep -Fq 'q8 partner projection exactness OK (3 classes)' \
            "$OUTPUT_DIR/gpu-exactness.log" ||
            die "the GPU test did not exercise all three partner projection shapes"
    fi
else
    make -q ds4-bench tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found stale targets"
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nrepeats=%s\nvariants=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" \
        "$REPEATS" "$VARIANTS"
    printf 'run_nsys=%s\nnsys_variants=%s\nprofile_tokens=%s\nmodel_hashing=disabled\n' \
        "$RUN_NSYS" "$NSYS_VARIANTS" "$PROFILE_TOKENS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
        --format=csv
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
printf 'repeat\tslot\tvariant\tcsv\tlog\taudit\tcache_before\tcache_after\n' \
    >"$OUTPUT_DIR/runs.tsv"
: >"$OUTPUT_DIR/validation-failures.txt"

phase=benchmarks
n_variants=${#variants[@]}
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for ((slot=0; slot<n_variants; slot++)); do
        if (( repeat % 2 )); then
            idx=$(((slot + repeat - 1) % n_variants))
        else
            idx=$((n_variants - 1 - ((slot + repeat - 1) % n_variants)))
        fi
        variant=${variants[$idx]}
        set_variant_extra "$variant"
        stem="$variant-r$repeat"
        csv="$OUTPUT_DIR/runs/$stem.csv"
        log="$OUTPUT_DIR/runs/$stem.log"
        audit="$OUTPUT_DIR/runs/$stem.q8-audit.csv"
        before="$OUTPUT_DIR/runs/$stem.cache-before.csv"
        after="$OUTPUT_DIR/runs/$stem.cache-after.csv"
        telemetry_before="$OUTPUT_DIR/telemetry/$stem.before.csv"
        telemetry_after="$OUTPUT_DIR/telemetry/$stem.after.csv"

        nvidia-smi --query-gpu=index,memory.used,memory.free \
            --format=csv >"$telemetry_before"
        printf 'Benchmarking %s repeat=%d/%d slot=%d/%d...\n' \
            "$variant" "$repeat" "$REPEATS" "$((slot+1))" "$n_variants"
        "${clean[@]}" "${variant_extra[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 \
            DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
            "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$before" \
            "DS4_CUDA_Q8_CACHE_STATE_CSV=$after" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$((CTX_MAX+1))" --step-mul "$STEP_MUL" \
                --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 --csv "$csv" \
                >"$log" 2>&1 || {
                    tail -n 180 "$log" >&2
                    die "$stem failed"
                }
        nvidia-smi --query-gpu=index,memory.used,memory.free \
            --format=csv >"$telemetry_after"

        [[ -s $csv && -s $audit && -s $before && -s $after ]] ||
            die "$stem omitted required evidence"
        cmp -s "$before" "$after" ||
            die "$stem changed the F16 cache during timed frontiers"
        grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
            "$log" || die "$stem did not use the requested stage split"
        for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
            grep -Fq "$route" "$log" || die "$stem lacks validated direct route $route"
        done
        grep -Fq 'q8 fp16 benefit plan registered' "$log" ||
            die "$stem did not use benefit planning"
        if ! validate_audit "$variant" "$audit"; then
            printf 'benchmark repeat=%s variant=%s did not execute only the requested class\n' \
                "$repeat" "$variant" >>"$OUTPUT_DIR/validation-failures.txt"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$((slot+1))" "$variant" "$csv" "$log" "$audit" \
            "$before" "$after" >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=summarize
python3 speed-bench/summarize-q8-partner-offload-ab.py \
    "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/paired-samples.csv && -s $OUTPUT_DIR/class-evidence.csv &&
   -s $OUTPUT_DIR/summary.txt ]] || die "summary missing"
awk -F, 'NR > 1 && $3 != "ok" {
    printf "summary repeat=%s variant=%s evidence_status=%s\n", $1, $2, $3
}' "$OUTPUT_DIR/class-evidence.csv" >>"$OUTPUT_DIR/validation-failures.txt"

if [[ $RUN_NSYS == 1 ]]; then
    phase=nsight-systems
    command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI (nsys) not found"
    for variant in "${nsys_variants[@]}"; do
        set_variant_extra "$variant"
        base="$OUTPUT_DIR/nsys/$variant"
        audit="$base.q8-audit.csv"
        printf 'Nsight Systems: %s partner path...\n' "$variant"
        "${clean[@]}" "${variant_extra[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 \
            DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$PROFILE_TOKENS" \
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
            DS4_NSYS_CAPTURE_PREFILL=1 \
            nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
                --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
                --capture-range-end=stop --output="$base" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
                    --ctx-alloc "$((PROFILE_TOKENS+1))" \
                    --step-incr "$PROFILE_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
                    --gen-tokens 0 --csv "$base-benchmark.csv" \
                    >"$base.log" 2>&1 || {
                        tail -n 180 "$base.log" >&2
                        die "Nsight Systems capture failed for $variant"
                    }
        [[ -s $base.nsys-rep && -s $audit ]] ||
            die "Nsight Systems omitted report or audit for $variant"
        if ! validate_audit "$variant" "$audit"; then
            printf 'nsys variant=%s did not execute only the requested class\n' \
                "$variant" >>"$OUTPUT_DIR/validation-failures.txt"
        fi
        for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum cuda_gpu_trace; do
            nsys stats --report "$report" --format csv "$base.nsys-rep" \
                >"$base-$report.csv" 2>"$base-$report.log" || true
        done
        [[ -s $base-cuda_gpu_kern_sum.csv && -s $base-cuda_gpu_trace.csv ]] ||
            die "Nsight Systems summaries are empty for $variant"
    done
fi

[[ ! -s $OUTPUT_DIR/validation-failures.txt ]] || {
    cat "$OUTPUT_DIR/validation-failures.txt" >&2
    die "one or more class-isolated runs were not exercised or were contaminated"
}

phase=complete
printf 'Q8 partner-offload class screen complete: %s\n' "$OUTPUT_DIR"
