#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the current-production four-GPU prefill A/B for the T32 FP16-output
fused projection path against both deployed quantization layouts.

Optional environment:
  MIXED_MODEL=...   default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  ALL43_MODEL=...   default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf
  PROMPT=...        default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REPEATS=3
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  T32_F16_PRODUCTION_AB_DIR=...

The fixed frontiers are 512, 4096, and 32768. Both arms retain the current
stage-aware 344/344 dense-Q8 plan and disable prefill attention row splitting
only on pair 0, the known-stable production configuration. The only A/B
difference is the T32 projection output boundary (FP32 control versus fused
FP16 candidate). This is intentionally a one-shot matrix: failed GPU runs are
archived, not resumed into a different process history.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REPEATS=${REPEATS:-3}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_ALLOC=33025
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T32_F16_PRODUCTION_AB_DIR:-$repo_dir/sm75-t32-f16-production-ab-$stamp}
layouts=(mixed15 all43)
models=("$MIXED_MODEL" "$ALL43_MODEL")

for model in "${models[@]}"; do
    [[ $model == /* && -f $model ]] || die "model not found at absolute path: $model"
done
[[ $MIXED_MODEL != "$ALL43_MODEL" ]] || die "the two model paths must differ"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] && (( REPEATS >= 3 )) ||
    die "REPEATS must be an integer >= 3"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk basename cmp date dirname env find git grep make mkdir mv nproc \
            nvidia-smi python3 rm sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "invalid or duplicate GPU ID: $gpu"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            rm -f -- "$partial" "$archive"
            printf 'error: could not create %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
printf '%s\n' "${inherited_ds4[@]:-}" >"$OUTPUT_DIR/provenance/cleared-ds4-env.txt"

production_env=(
    "${clean[@]}"
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

phase=build
targets=(ds4-bench tests/test_engine_mgpu_placement tests/test_gpu_xdev)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/placement-tests.log" >&2
        die "placement regression tests failed"
    }
CUDA_VISIBLE_DEVICES="${gpu_ids[0]},${gpu_ids[2]}" \
    "${clean[@]}" ./tests/test_gpu_xdev \
    >"$OUTPUT_DIR/t32-gpu-exactness.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/t32-gpu-exactness.log" >&2
        die "T32 GPU exactness regression failed"
    }
grep -Fq 'q8 partner T32 FP16-output RMS/RoPE exactness OK' \
    "$OUTPUT_DIR/t32-gpu-exactness.log" ||
    die "T32 GPU exactness marker missing"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'mixed_model=%s\nmixed_model_bytes=%s\n' \
        "$MIXED_MODEL" "$(stat -c %s "$MIXED_MODEL")"
    printf 'all43_model=%s\nall43_model_bytes=%s\n' \
        "$ALL43_MODEL" "$(stat -c %s "$ALL43_MODEL")"
    printf 'model_hashing=disabled\nprompt=%s\ngpu_devices=%s\n' \
        "$PROMPT" "$GPU_DEVICES"
    printf 'stage_split=22/21\ncontexts=512,4096,32768\nrepeats=%s\n' "$REPEATS"
    printf 'prefill_chunk=%s\npipeline_mb=%s\nctx_alloc=%s\n' \
        "$PREFILL_CHUNK" "$PIPELINE_MB" "$CTX_ALLOC"
    printf 'pair0_prefill_attention_rows=disabled\npair1_prefill_attention_rows=enabled\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

validate_gpu_health_pair() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] || return 1
    ! grep -Eq 'ERR!|Unknown Error|GPU is lost' \
        "$base.pre-gpu.csv" "$base.post-gpu.csv" || return 1
    cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

validate_layout() {
    local layout=$1 log=$2 expected_count expected_layers
    if [[ $layout == mixed15 ]]; then
        expected_count=15
        expected_layers=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
    else
        expected_count=43
        expected_layers=$(printf '%s,' {0..42}); expected_layers=${expected_layers%,}
    fi
    awk -v expected_count="$expected_count" -v expected_layers="$expected_layers" '
        /routed-quant-audit/ {
            seen++; layer=gate=up=down=""
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]=="layer") layer=a[2]
                if (a[1]=="gate") gate=a[2]
                if (a[1]=="up") up=a[2]
                if (a[1]=="down") down=a[2]
            }
            if (layer !~ /^[0-9]+$/ || layer<0 || layer>42 || layer_seen[layer]++) bad=1
            if (gate=="sm75_q3a4") {
                if (up!="sm75_q3a4" || down!="sm75_q4_32") bad=1
                q3++; layers=layers (layers ? "," : "") layer
            } else if (gate!="sm75_q4_32" || up!="sm75_q4_32" || down!="sm75_q4_32") bad=1
        }
        END {
            for (i=0; i<43; i++) if (layer_seen[i]!=1) bad=1
            exit !(seen==43 && q3==expected_count && layers==expected_layers && !bad)
        }
    ' "$log"
}

validate_common_log() {
    local layout=$1 log=$2 marker route
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled'; do
        grep -Fq "$marker" "$log" || {
            printf 'validation: missing production marker: %s\n' "$marker" >&2
            return 1
        }
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || {
            printf 'validation: missing direct route: %s\n' "$route" >&2
            return 1
        }
    done
    ! grep -Fq 'required but unavailable' "$log" || {
        printf 'validation: a required production path was unavailable\n' >&2
        return 1
    }
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || {
        printf 'validation: pair 0 prefill attention row split was not suppressed\n' >&2
        return 1
    }
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || {
        printf 'validation: pair 1 prefill attention row split was not active\n' >&2
        return 1
    }
    ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || {
        printf 'validation: pair 0 prefill indexer row split was not suppressed\n' >&2
        return 1
    }
    if [[ $(grep -Fc 'CUDA prefill indexer score/top-k row split enabled:' "$log") != 1 ]] ||
       ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log"; then
        printf 'validation: expected exactly the pair 1 prefill indexer row split\n' >&2
        return 1
    fi
    validate_layout "$layout" "$log" || {
        printf 'validation: %s routed-quant inventory mismatch\n' "$layout" >&2
        return 1
    }
}

validate_q8_state_files() {
    local base=$1 plan="$1.q8-plan.csv" bindings="$1.q8-bindings.csv"
    [[ -s $plan && -s $bindings ]] || return 1
    [[ $(wc -l <"$plan") == 345 && $(wc -l <"$bindings") == 345 ]] || return 1
    awk -F, '
        NR==1 {header=($1=="sequence" && $2=="label" && $3=="consumer_device" && $7=="resident_device" && $13=="status"); next}
        {rows++; if (($7+0)<0 || ($13!="home" && $13!="partner")) bad=1}
        END {exit !(header && rows==344 && !bad)}
    ' "$plan" || return 1
    awk -F, '
        NR==1 {header=($1=="consumer_device" && $2=="resident_device" && $3=="partner_offload" && $12=="label" && $15=="live"); next}
        {rows++; if (($13+0)<=0 || ($14+0)<=0 || $15!=1) bad=1}
        END {exit !(header && rows==344 && !bad)}
    ' "$bindings"
}

canonical_q8_bindings() {
    awk -F, 'BEGIN {OFS=","} NR>1 {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' "$1" | sort
}

validate_q8_plan_equal() {
    local left=$1 right=$2 marker left_line right_line
    for marker in 'CUDA q8 fp16 benefit plan candidates=' \
                  'CUDA q8 fp16 stage-aware 22/21 planner selected ' \
                  'CUDA q8 fp16 benefit plan materialized '; do
        [[ $(grep -Fc "$marker" "$left.log") == 1 && \
           $(grep -Fc "$marker" "$right.log") == 1 ]] || return 1
        left_line=$(grep -F "$marker" "$left.log") || return 1
        right_line=$(grep -F "$marker" "$right.log") || return 1
        [[ $left_line == "$right_line" ]] || return 1
    done
    cmp -s "$left.q8-plan.csv" "$right.q8-plan.csv" || return 1
    cmp -s <(canonical_q8_bindings "$left.q8-bindings.csv") \
           <(canonical_q8_bindings "$right.q8-bindings.csv")
}

validate_frontiers() {
    local base=$1 context path
    awk -F, '
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps"); next}
        NR==2 {rows++; a=($1==512 && ($3+0)>0); next}
        NR==3 {rows++; b=($1==4096 && ($3+0)>0); next}
        NR==4 {rows++; c=($1==32768 && ($3+0)>0); next}
        NR>4 {rows++}
        END {exit !(header && rows==3 && a && b && c)}
    ' "$base.csv" || return 1
    for context in 512 4096 32768; do
        printf -v path '%s-logits/frontier_%06d.logits' "$base" "$context"
        [[ -s $path.f32 && -s $path.json ]] || return 1
    done
    [[ $(find "$base-logits" -maxdepth 1 -type f | wc -l) == 6 ]]
}

validate_variant() {
    local variant=$1 log=$2 line local_calls=0 partner_calls=0 total_partner=0 f16_calls=0
    line=$(grep -F 'CUDA T32 f16-output fused summary:' "$log" | tail -n 1 || true)
    if [[ $line =~ local=([0-9]+)[[:space:]]partner=([0-9]+) ]]; then
        local_calls=${BASH_REMATCH[1]}; partner_calls=${BASH_REMATCH[2]}
    fi
    line=$(grep -F 'CUDA q8 fp16 partner summary:' "$log" | tail -n 1 || true)
    if [[ $line =~ calls=([0-9]+).*f16-result-calls=([0-9]+) ]]; then
        total_partner=${BASH_REMATCH[1]}; f16_calls=${BASH_REMATCH[2]}
    fi
    if [[ $variant == control ]]; then
        (( local_calls == 0 && partner_calls == 0 && f16_calls == 0 ))
    else
        (( local_calls > 0 && partner_calls > 0 && total_partner >= partner_calls &&
           f16_calls == partner_calls ))
    fi
}

run_engine() {
    local layout=$1 model=$2 variant=$3 base=$4 rc=0
    local -a selector
    if [[ $variant == control ]]; then
        selector=(DS4_CUDA_NO_T32_F16_FUSED=1)
    else
        selector=(DS4_CUDA_T32_F16_FUSED=1)
    fi
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    "${production_env[@]}" "${selector[@]}" \
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv" \
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$model" --prompt-file "$PROMPT" \
            --ctx-start 512 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC" \
            --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" \
            --gen-tokens 0 --csv "$base.csv" \
            --dump-frontier-logits-dir "$base-logits" \
            >"$base.log" 2>&1 || rc=$?
    capture_gpu_health "$base.post-gpu.csv" || return 1
    return "$rc"
}

validate_run() {
    local layout=$1 variant=$2 base=$3
    validate_gpu_health_pair "$base" || {
        printf 'validation: GPU identity, health, or power limit changed\n' >&2
        return 1
    }
    validate_q8_state_files "$base" || {
        printf 'validation: dense-Q8 plan or binding state is incomplete\n' >&2
        return 1
    }
    validate_frontiers "$base" || {
        printf 'validation: benchmark CSV or frontier logits are incomplete\n' >&2
        return 1
    }
    validate_common_log "$layout" "$base.log" || return 1
    validate_variant "$variant" "$base.log" || {
        printf 'validation: %s T32 FP16 dispatch counters are invalid\n' "$variant" >&2
        return 1
    }
}

phase=production-ab
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture the initial four-GPU identity and power limits"
printf 'model_layout\trepeat\tslot\tvariant\tcsv\tlog\tlogits\n' \
    >"$OUTPUT_DIR/runs.tsv"
for model_index in 0 1; do
    layout=${layouts[$model_index]}
    model=${models[$model_index]}
    layout_reference=
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        if (( repeat % 2 )); then variants=(control fused); else variants=(fused control); fi
        slot=0
        for variant in "${variants[@]}"; do
            slot=$((slot + 1))
            base="$OUTPUT_DIR/runs/$layout-r${repeat}-$variant"
            mkdir -p "$base-logits"
            printf 'T32 FP16 production A/B model=%s repeat=%d/%d slot=%d/2 variant=%s...\n' \
                "$layout" "$repeat" "$REPEATS" "$slot" "$variant"
            run_engine "$layout" "$model" "$variant" "$base" || {
                tail -n 220 "$base.log" >&2 || true
                die "$layout $variant run failed"
            }
            validate_run "$layout" "$variant" "$base" ||
                die "$layout $variant run failed production-path validation"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$layout" "$repeat" "$slot" "$variant" \
                "$base.csv" "$base.log" "$base-logits" >>"$OUTPUT_DIR/runs.tsv"
        done
        control="$OUTPUT_DIR/runs/$layout-r${repeat}-control"
        fused="$OUTPUT_DIR/runs/$layout-r${repeat}-fused"
        validate_q8_plan_equal "$control" "$fused" ||
            die "$layout repeat $repeat changed the dense-Q8 plan between arms"
        if [[ -z $layout_reference ]]; then
            layout_reference=$control
        else
            validate_q8_plan_equal "$layout_reference" "$control" ||
                die "$layout repeat $repeat changed the production dense-Q8 plan"
        fi
    done
done

phase=summarize
python3 speed-bench/summarize-sm75-t32-f16-production-ab.py \
    "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/summary" |
    tee "$OUTPUT_DIR/summary-stdout.txt"

phase=complete
printf 'SM75 dual-model production T32 FP16-output A/B complete: %s\n' "$OUTPUT_DIR"
