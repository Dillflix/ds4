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
RUN_NSYS=${RUN_NSYS:-1}
REUSE_OLD_DIR=${REUSE_OLD_DIR:-}
if [[ -n $REUSE_OLD_DIR ]]; then
    NSYS_VARIANTS=${NSYS_VARIANTS:-new_local,new_partner}
else
    NSYS_VARIANTS=${NSYS_VARIANTS:-old_local,new_local,old_partner,new_partner}
fi
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_GPU_TEST=${RUN_GPU_TEST:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T32_FUSED_AB_DIR:-$repo_dir/q8-t32-fused-ab-$stamp}
variants=(old_local new_local old_partner new_partner)

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
if [[ -n $REUSE_OLD_DIR ]]; then
    [[ $REUSE_OLD_DIR == /* && -d $REUSE_OLD_DIR ]] ||
        die "REUSE_OLD_DIR must name an existing absolute result directory"
    REUSE_OLD_DIR=$(cd "$REUSE_OLD_DIR" && pwd)
    [[ -s $REUSE_OLD_DIR/manifest.txt && -s $REUSE_OLD_DIR/runs.tsv ]] ||
        die "REUSE_OLD_DIR lacks manifest.txt or runs.tsv"
    grep -Fqx -- "model=$MODEL" "$REUSE_OLD_DIR/manifest.txt" ||
        die "REUSE_OLD_DIR used a different model"
    grep -Fqx -- "gpu_devices=$GPU_DEVICES" "$REUSE_OLD_DIR/manifest.txt" ||
        die "REUSE_OLD_DIR used different GPU devices"
    grep -Fqx -- "stage_split=$STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$REUSE_OLD_DIR/manifest.txt" ||
        die "REUSE_OLD_DIR used a different stage split"
    reused_repeats=$(awk -F= '$1 == "repeats" { print $2; exit }' \
        "$REUSE_OLD_DIR/manifest.txt")
    [[ $reused_repeats =~ ^[0-9]+$ ]] && (( reused_repeats >= REPEATS )) ||
        die "REUSE_OLD_DIR has fewer than $REPEATS repeats"
fi
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "STEP_MUL:$STEP_MUL" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "REPEATS:$REPEATS" \
            "PROFILE_TOKENS:$PROFILE_TOKENS" "SKIP_BUILD:$SKIP_BUILD" \
            "RUN_GPU_TEST:$RUN_GPU_TEST" "RUN_NSYS:$RUN_NSYS" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 && REPEATS >= 2 )) ||
    die "invalid stage split or repeats"
for flag in SKIP_BUILD RUN_GPU_TEST RUN_NSYS CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
IFS=',' read -r -a nsys_variants <<<"$NSYS_VARIANTS"
for variant in "${nsys_variants[@]}"; do
    [[ " ${variants[*]} " == *" $variant "* ]] ||
        die "unsupported NSYS_VARIANTS entry: $variant"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,logits,nsys,provenance,telemetry}
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
variant_env=()
set_variant_env() {
    variant_env=()
    case "$1" in
        old_local)
            variant_env+=(DS4_CUDA_T32_F16_FUSED=0)
            variant_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
            variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
            variant_env+=(DS4_CUDA_Q8_T256_PLACEMENT=overflow)
            ;;
        new_local)
            variant_env+=(DS4_CUDA_T32_F16_FUSED=1)
            variant_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1)
            variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=none)
            variant_env+=(DS4_CUDA_Q8_T256_PLACEMENT=overflow)
            ;;
        old_partner)
            variant_env+=(DS4_CUDA_T32_F16_FUSED=0)
            variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t32)
            variant_env+=(DS4_CUDA_Q8_T256_PLACEMENT=overflow)
            ;;
        new_partner)
            variant_env+=(DS4_CUDA_T32_F16_FUSED=1)
            variant_env+=(DS4_CUDA_Q8_F16_PARTNER_CLASSES=t32)
            variant_env+=(DS4_CUDA_Q8_T256_PLACEMENT=overflow)
            ;;
        *) die "unknown variant $1" ;;
    esac
}

validate_variant() {
    local variant=$1 audit=$2 log=$3
    local counts total t32 other
    counts=$(awk -F, '
        NR == 1 { next }
        $12 == "f16_partner_hit" && $13 == "nvlink_offload" {
            if ($2 ~ /attn_q_b/ || $3 ~ /attn_q_b/ || ($9 == 1024 && $10 == 32768)) t32++
            else other++
        }
        END { printf "%d %d %d", t32 + other, t32, other }
    ' "$audit")
    read -r total t32 other <<<"$counts"
    local fused_local=0 fused_partner=0 partner_calls=0 f16_calls=0
    local line
    line=$(grep -E 'CUDA T32 f16-output fused summary:' "$log" | tail -n 1 || true)
    if [[ $line =~ local=([0-9]+)[[:space:]]partner=([0-9]+) ]]; then
        fused_local=${BASH_REMATCH[1]}; fused_partner=${BASH_REMATCH[2]}
    fi
    line=$(grep -E 'CUDA q8 fp16 partner summary:' "$log" | tail -n 1 || true)
    if [[ $line =~ calls=([0-9]+).*f16-result-calls=([0-9]+) ]]; then
        partner_calls=${BASH_REMATCH[1]}; f16_calls=${BASH_REMATCH[2]}
    fi
    printf '%s total=%s t32=%s other=%s fused_local=%s fused_partner=%s partner_calls=%s f16_result_calls=%s\n' \
        "$variant" "$total" "$t32" "$other" "$fused_local" "$fused_partner" \
        "$partner_calls" "$f16_calls"
    case "$variant" in
        old_local)
            (( total == 0 && fused_local == 0 && fused_partner == 0 && partner_calls == 0 )) ;;
        new_local)
            (( total == 0 && fused_local > 0 && fused_partner == 0 && partner_calls == 0 )) ;;
        old_partner)
            (( t32 > 0 && other == 0 && fused_local == 0 && fused_partner == 0 &&
               partner_calls > 0 && f16_calls == 0 )) ;;
        new_partner)
            (( t32 > 0 && other == 0 && fused_local > 0 && fused_partner > 0 &&
               partner_calls > 0 && f16_calls == partner_calls &&
               fused_partner == partner_calls )) ;;
    esac
}

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        tests/test_gpu_xdev CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
    ./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/planner-unit.log" 2>&1 ||
        die "planner unit test failed"
    if [[ $RUN_GPU_TEST == 1 ]]; then
        ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
            tail -n 180 "$OUTPUT_DIR/gpu-exactness.log" >&2
            die "multi-GPU exactness test failed"
        }
        grep -Fq 'q8 partner T32 FP16-output RMS/RoPE exactness OK' \
            "$OUTPUT_DIR/gpu-exactness.log" ||
            die "GPU regression omitted fused T32 local/partner exactness"
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
    printf 'gpu_devices=%s\nstage_split=%s/%s\nrepeats=%s\nmodel_hashing=disabled\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$REPEATS"
    printf 'reuse_old_dir=%s\n' "${REUSE_OLD_DIR:-none}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
        --format=csv
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
if [[ -n $REUSE_OLD_DIR ]]; then
    cp "$REUSE_OLD_DIR/manifest.txt" \
        "$OUTPUT_DIR/provenance/reused-old-manifest.txt"
fi
printf 'repeat\tslot\tvariant\tcsv\tlog\taudit\tcache_before\tcache_after\tlogits\n' \
    >"$OUTPUT_DIR/runs.tsv"
: >"$OUTPUT_DIR/validation-failures.txt"

phase=benchmarks
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for ((slot=0; slot<4; slot++)); do
        if (( repeat % 2 )); then idx=$(((slot + repeat - 1) % 4))
        else idx=$((3 - ((slot + repeat - 1) % 4)))
        fi
        variant=${variants[$idx]}; set_variant_env "$variant"
        stem="$variant-r$repeat"
        csv="$OUTPUT_DIR/runs/$stem.csv"; log="$OUTPUT_DIR/runs/$stem.log"
        audit="$OUTPUT_DIR/runs/$stem.q8-audit.csv"
        before="$OUTPUT_DIR/runs/$stem.cache-before.csv"
        after="$OUTPUT_DIR/runs/$stem.cache-after.csv"
        logits="$OUTPUT_DIR/logits/$stem"
        mkdir -p "$logits"
        printf 'Benchmarking %s repeat=%d/%d slot=%d/4...\n' \
            "$variant" "$repeat" "$REPEATS" "$((slot+1))"
        if [[ -n $REUSE_OLD_DIR && $variant == old_* ]]; then
            source_stem="$REUSE_OLD_DIR/runs/$stem"
            for suffix in csv log q8-audit.csv cache-before.csv cache-after.csv; do
                [[ -s $source_stem.$suffix ]] ||
                    die "reused result is missing $source_stem.$suffix"
            done
            source_logits="$REUSE_OLD_DIR/logits/$stem"
            [[ -d $source_logits ]] || die "reused logits are missing: $source_logits"
            cp "$source_stem.csv" "$csv"
            cp "$source_stem.log" "$log"
            cp "$source_stem.q8-audit.csv" "$audit"
            cp "$source_stem.cache-before.csv" "$before"
            cp "$source_stem.cache-after.csv" "$after"
            cp -a "$source_logits/." "$logits/"
            cmp -s "$before" "$after" || die "$stem changed cache during timing"
            if ! validate_variant "$variant" "$audit" "$log"; then
                printf 'reused benchmark repeat=%s variant=%s failed path validation\n' \
                    "$repeat" "$variant" >>"$OUTPUT_DIR/validation-failures.txt"
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$repeat" "$((slot+1))" "$variant" "$csv" "$log" "$audit" \
                "$before" "$after" "$logits" >>"$OUTPUT_DIR/runs.tsv"
            cat "$csv"
            continue
        fi
        "${clean[@]}" "${variant_env[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 DS4_CUDA_PREFILL_PIPELINE_MB=512 \
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
                --dump-frontier-logits-dir "$logits" >"$log" 2>&1 || {
                    tail -n 180 "$log" >&2; die "$stem failed"
                }
        [[ -s $csv && -s $audit && -s $before && -s $after ]] ||
            die "$stem omitted required evidence"
        cmp -s "$before" "$after" || die "$stem changed cache during timing"
        grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
            "$log" || die "$stem did not use requested split"
        for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
            grep -Fq "$route" "$log" || die "$stem lacks route $route"
        done
        if ! validate_variant "$variant" "$audit" "$log"; then
            printf 'benchmark repeat=%s variant=%s failed path validation\n' \
                "$repeat" "$variant" >>"$OUTPUT_DIR/validation-failures.txt"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$((slot+1))" "$variant" "$csv" "$log" "$audit" \
            "$before" "$after" "$logits" >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=summarize
python3 speed-bench/summarize-q8-t32-fused-ab.py \
    "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/summary-stdout.txt"
awk -F, 'NR > 1 && $4 == 1 && $5 != 1 {
    printf "logits repeat=%s comparison=%s frontier=%s not exact\n", $1, $2, $3
}' "$OUTPUT_DIR/logit-comparison.csv" >>"$OUTPUT_DIR/validation-failures.txt"

if [[ $RUN_NSYS == 1 ]]; then
    phase=nsight-systems
    command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI not found"
    if [[ -n $REUSE_OLD_DIR && -d $REUSE_OLD_DIR/nsys ]]; then
        for variant in old_local old_partner; do
            for source in "$REUSE_OLD_DIR"/nsys/"$variant"*; do
                [[ -e $source ]] || continue
                cp -a "$source" "$OUTPUT_DIR/nsys/"
            done
        done
    fi
    for variant in "${nsys_variants[@]}"; do
        set_variant_env "$variant"
        base="$OUTPUT_DIR/nsys/$variant"; audit="$base.q8-audit.csv"
        printf 'Nsight Systems: %s...\n' "$variant"
        "${clean[@]}" "${variant_env[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$PROFILE_TOKENS" \
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" DS4_NSYS_CAPTURE_PREFILL=1 \
            nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
                --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
                --capture-range-end=stop --output="$base" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
                    --ctx-alloc "$((PROFILE_TOKENS+1))" \
                    --step-incr "$PROFILE_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
                    --gen-tokens 0 --csv "$base-benchmark.csv" >"$base.log" 2>&1 || {
                        tail -n 180 "$base.log" >&2; die "nsys failed for $variant"
                    }
        [[ -s $base.nsys-rep && -s $audit ]] || die "nsys omitted $variant evidence"
        if ! validate_variant "$variant" "$audit" "$base.log"; then
            printf 'nsys variant=%s failed path validation\n' "$variant" \
                >>"$OUTPUT_DIR/validation-failures.txt"
        fi
        for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum cuda_gpu_trace; do
            nsys stats --report "$report" --format csv "$base.nsys-rep" \
                >"$base-$report.csv" 2>"$base-$report.log" || true
        done
    done
fi

[[ ! -s $OUTPUT_DIR/validation-failures.txt ]] || {
    cat "$OUTPUT_DIR/validation-failures.txt" >&2
    die "T32 fused A/B failed an acceptance condition"
}
phase=complete
printf 'T32 FP16-output fused A/B complete: %s\n' "$OUTPUT_DIR"
