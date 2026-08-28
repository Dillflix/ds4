#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Measure and trace the fixed SM75 production decode path.

The pass has three independent products:
  1. PP4096 throughput A/B: indexed threshold 1024 versus forced-dense 4096.
  2. Bit-exact per-token logits for a short version of that A/B.
  3. Bounded steady-decode Nsight Systems traces at PP2048, PP4096, and PP32768,
     plus a PP4096 forced-dense trace. Prefill and initial decode tokens remain
     outside the profiler range.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    fixed promessi_sposi prompt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  AB_PP=4096
  AB_TG=256
  AB_REPEATS=3
  EXACT_TG=8
  TRACE_CONTEXTS=2048,4096,32768
  TRACE_SKIP=16                unprofiled generated tokens after prefill
  TRACE_TOKENS=16              tokens inside the profiler range
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  RUN_AB=1
  RUN_EXACT=1
  RUN_NSYS=1
  SKIP_BUILD=0
  RESUME=0
  CREATE_ARCHIVE=1
  DECODE_EVIDENCE_DIR=...      output directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
AB_PP=${AB_PP:-4096}
AB_TG=${AB_TG:-256}
AB_REPEATS=${AB_REPEATS:-3}
EXACT_TG=${EXACT_TG:-8}
TRACE_CONTEXTS=${TRACE_CONTEXTS:-2048,4096,32768}
TRACE_SKIP=${TRACE_SKIP:-16}
TRACE_TOKENS=${TRACE_TOKENS:-16}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
RUN_AB=${RUN_AB:-1}
RUN_EXACT=${RUN_EXACT:-1}
RUN_NSYS=${RUN_NSYS:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${DECODE_EVIDENCE_DIR:-$repo_dir/sm75-decode-evidence-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged Q4-32/Q3A4 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "decode evidence requires the fixed promessi_sposi prompt"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "AB_PP:$AB_PP" "AB_TG:$AB_TG" \
            "AB_REPEATS:$AB_REPEATS" "EXACT_TG:$EXACT_TG" \
            "TRACE_SKIP:$TRACE_SKIP" "TRACE_TOKENS:$TRACE_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
            "RUN_AB:$RUN_AB" "RUN_EXACT:$RUN_EXACT" "RUN_NSYS:$RUN_NSYS" \
            "SKIP_BUILD:$SKIP_BUILD" "RESUME:$RESUME" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 )) || die "decode evidence requires STAGE_SPLIT=22"
[[ $GPU_DEVICES == 0,3,1,2 ]] ||
    die "decode evidence requires GPU_DEVICES=0,3,1,2"
[[ $GPU_VRAM == auto ]] || die "decode evidence requires GPU_VRAM=auto"
(( AB_PP == 4096 && AB_TG >= 64 && AB_REPEATS >= 2 && EXACT_TG >= 2 &&
   TRACE_SKIP >= 1 && TRACE_TOKENS >= 1 && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 )) ||
    die "require PP4096, AB_TG>=64, repeats>=2, exact_tg>=2, and fixed production chunks"
for flag in RUN_AB RUN_EXACT RUN_NSYS SKIP_BUILD RESUME CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
(( RUN_AB || RUN_EXACT || RUN_NSYS )) || die "all evidence phases are disabled"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk cat cmp comm date env find git grep make mkdir mv nproc \
            nvidia-smi python3 sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if (( RUN_NSYS )); then
    command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI (nsys) not found"
fi

IFS=, read -r -a trace_contexts <<<"$TRACE_CONTEXTS"
(( ${#trace_contexts[@]} > 0 )) || die "TRACE_CONTEXTS is empty"
declare -A seen_context=()
for pp in "${trace_contexts[@]}"; do
    [[ $pp =~ ^[0-9]+$ && $pp -gt 0 && -z ${seen_context[$pp]+x} ]] ||
        die "TRACE_CONTEXTS contains an invalid or duplicate value: $pp"
    case "$pp" in 2048|4096|32768) ;; *) die "unsupported trace context: $pp";; esac
    seen_context[$pp]=1
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"
for gpu in "${gpu_ids[@]}"; do
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done

if (( RESUME )); then
    [[ -d $OUTPUT_DIR ]] || die "resume directory not found: $OUTPUT_DIR"
else
    [[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
        die "output path already exists: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"/{ab,exact,nsys,summary,telemetry,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
sampler_pid=
cleanup_sampler() {
    if [[ -n ${sampler_pid:-} ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=
    fi
}
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    cleanup_sampler
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'error: could not create archive %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
production_env=(
    "${clean[@]}"
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    # Deliberately leave DS4_CUDA_Q8_T256_PLACEMENT unset. Production's
    # default is the fixed-22/21 stage-aware planner; explicitly selecting
    # balanced invokes the older legacy-class-policy planner.
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
)

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" ds4-bench CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found a stale ds4-bench"
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'ab_pp=%s\nab_tg=%s\nab_repeats=%s\nexact_tg=%s\n' \
        "$AB_PP" "$AB_TG" "$AB_REPEATS" "$EXACT_TG"
    printf 'trace_contexts=%s\ntrace_skip=%s\ntrace_tokens=%s\n' \
        "$TRACE_CONTEXTS" "$TRACE_SKIP" "$TRACE_TOKENS"
    printf 'resume=%s\n' "$RESUME"
    printf 'dense_f16_admission=344/344-required\nq8_t256_placement=stage-aware\n'
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
        --format=csv
    printf '\n[topology]\n'; nvidia-smi topo -m
    if (( RUN_NSYS )); then printf '\n[nsight systems]\n'; nsys --version; fi
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

validate_production_log() {
    local log=$1 threshold=$2
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  't256-placement=stage-aware' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  "decode indexer sparse threshold=$threshold compressed rows (explicit)"; do
        grep -Fq "$marker" "$log" || {
            printf 'error: production log missing required marker: %s\n' \
                "$marker" >&2
            return 1
        }
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || {
            printf 'error: production log missing direct route: %s\n' \
                "$route" >&2
            return 1
        }
    done
    if grep -Fq 'required but unavailable' "$log"; then
        printf 'error: production log reports a required unavailable path\n' >&2
        return 1
    fi
    return 0
}

validate_benchmark_csv() {
    local csv=$1 pp=$2 tg=$3
    [[ -s $csv ]] || return 1
    awk -F, -v pp="$pp" -v tg="$tg" '
        NR == 1 {
            good_header = ($1 == "ctx_tokens" && $4 == "gen_tokens" &&
                           $8 == "gen_steady_tps")
            next
        }
        NR == 2 {
            rows++
            good_row = ($1 == pp && $4 == tg && ($8 + 0.0) > 0.0)
            next
        }
        { rows++ }
        END { exit !(good_header && rows == 1 && good_row) }
    ' "$csv"
}

run_bench() {
    local threshold=$1 pp=$2 tg=$3 csv=$4 log=$5
    local ctx_alloc=$((pp + tg + 1))
    set +e
    "${production_env[@]}" \
    "DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=$threshold" \
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512 \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$pp" --ctx-max "$pp" --ctx-alloc "$ctx_alloc" \
        --step-incr "$pp" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$tg" --csv "$csv" >"$log" 2>&1
    local rc=$?
    set -e
    (( rc == 0 )) || { tail -n 160 "$log" >&2 || true; return "$rc"; }
    validate_production_log "$log" "$threshold"
}

if (( RUN_AB )); then
    phase=threshold-ab
    printf 'repeat,slot,variant,threshold,pp_tokens,tg_tokens,csv,log\n' \
        >"$OUTPUT_DIR/ab/runs.csv"
    for ((repeat=1; repeat<=AB_REPEATS; repeat++)); do
        if (( repeat % 2 )); then variants=(indexed1024 dense4096)
        else variants=(dense4096 indexed1024); fi
        slot=0
        for variant in "${variants[@]}"; do
            slot=$((slot + 1))
            [[ $variant == indexed1024 ]] && threshold=1024 || threshold=4096
            base="$OUTPUT_DIR/ab/r${repeat}-${variant}"
            if (( RESUME )) &&
               validate_benchmark_csv "$base.csv" "$AB_PP" "$AB_TG" &&
               validate_production_log "$base.log" "$threshold"; then
                printf 'Reusing decode threshold A/B repeat=%d/%d slot=%d/2 variant=%s...\n' \
                    "$repeat" "$AB_REPEATS" "$slot" "$variant"
            else
                printf 'Decode threshold A/B repeat=%d/%d slot=%d/2 variant=%s...\n' \
                    "$repeat" "$AB_REPEATS" "$slot" "$variant"
                run_bench "$threshold" "$AB_PP" "$AB_TG" "$base.csv" "$base.log" ||
                    die "$variant throughput run failed"
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$repeat" "$slot" "$variant" "$threshold" "$AB_PP" "$AB_TG" \
                "$base.csv" "$base.log" >>"$OUTPUT_DIR/ab/runs.csv"
        done
    done
fi

if (( RUN_EXACT )); then
    phase=exact-logits
    for variant in indexed1024 dense4096; do
        [[ $variant == indexed1024 ]] && threshold=1024 || threshold=4096
        base="$OUTPUT_DIR/exact/$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        ctx_alloc=$((AB_PP + EXACT_TG + 1))
        printf 'Exact decode logits variant=%s tokens=%d...\n' "$variant" "$EXACT_TG"
        set +e
        "${production_env[@]}" \
        "DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=$threshold" \
        DS4_BENCH_UNTIMED_WARMUP_TOKENS=512 \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$AB_PP" --ctx-max "$AB_PP" --ctx-alloc "$ctx_alloc" \
            --step-incr "$AB_PP" --prefill-chunk "$PREFILL_CHUNK" \
            --gen-tokens "$EXACT_TG" --dump-decode-logits-dir "$logits" \
            --csv "$base.csv" >"$base.log" 2>&1
        rc=$?
        set -e
        (( rc == 0 )) || { tail -n 160 "$base.log" >&2 || true; die "$variant exact run failed"; }
        validate_production_log "$base.log" "$threshold" ||
            die "$variant exact run did not preserve production configuration"
        [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == "$EXACT_TG" ]] ||
            die "$variant exact run did not emit $EXACT_TG logit files"
    done
    find "$OUTPUT_DIR/exact/indexed1024-logits" -maxdepth 1 -type f -printf '%f\n' | sort \
        >"$OUTPUT_DIR/exact/indexed-files.txt"
    find "$OUTPUT_DIR/exact/dense4096-logits" -maxdepth 1 -type f -printf '%f\n' | sort \
        >"$OUTPUT_DIR/exact/dense-files.txt"
    cmp -s "$OUTPUT_DIR/exact/indexed-files.txt" "$OUTPUT_DIR/exact/dense-files.txt" ||
        die "exact variants produced different logit inventories"
    while IFS= read -r file; do
        cmp -s "$OUTPUT_DIR/exact/indexed1024-logits/$file" \
               "$OUTPUT_DIR/exact/dense4096-logits/$file" ||
            die "decode logits are not bit-exact: $file"
    done <"$OUTPUT_DIR/exact/indexed-files.txt"
    printf 'bit_exact=true\nfrontier=%s\ndecode_tokens=%s\n' \
        "$AB_PP" "$EXACT_TG" >"$OUTPUT_DIR/exact/verification.txt"
fi

printf 'label\tpp_tokens\tthreshold\tcaptured_tokens\tdevices\tsqlite\n' \
    >"$OUTPUT_DIR/trace-map.tsv"
capture_trace() {
    local label=$1 pp=$2 threshold=$3
    local base="$OUTPUT_DIR/nsys/$label"
    local gen_tokens=$((TRACE_SKIP + TRACE_TOKENS))
    local ctx_alloc=$((pp + gen_tokens + 1))
    phase="nsight-$label"
    printf 'Bounded decode trace: %s PP=%s threshold=%s...\n' \
        "$label" "$pp" "$threshold"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,\
memory.used,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate \
        --format=csv -lms 200 >"$OUTPUT_DIR/telemetry/$label.csv" &
    sampler_pid=$!
    set +e
    "${production_env[@]}" \
    "DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=$threshold" \
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512 \
    DS4_CUDA_CRITICAL_PATH_NVTX=1 \
    "DS4_NSYS_CAPTURE_DECODE_SKIP=$TRACE_SKIP" \
    "DS4_NSYS_CAPTURE_DECODE_TOKENS=$TRACE_TOKENS" \
    nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$base" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$pp" --ctx-max "$pp" --ctx-alloc "$ctx_alloc" \
            --step-incr "$pp" --prefill-chunk "$PREFILL_CHUNK" \
            --gen-tokens "$gen_tokens" --csv "$base-benchmark.csv" \
            >"$base.log" 2>&1
    local rc=$?
    set -e
    cleanup_sampler
    (( rc == 0 )) || { tail -n 180 "$base.log" >&2 || true; die "$label trace failed (exit $rc)"; }
    validate_production_log "$base.log" "$threshold" ||
        die "$label trace did not preserve production configuration"
    grep -Fq "starting Nsight CUDA capture for decode frontier $pp" "$base.log" ||
        die "$label did not start its bounded decode capture"
    grep -Fq "stopped Nsight CUDA capture for decode frontier $pp after $TRACE_TOKENS tokens" \
        "$base.log" || die "$label did not stop its bounded decode capture"
    [[ -s $base.nsys-rep && -s $base-benchmark.csv ]] ||
        die "$label omitted its Nsight report or benchmark CSV"
    nsys export --type sqlite --force-overwrite=true \
        --output "$base.sqlite" "$base.nsys-rep" \
        >"$base-export.log" 2>&1 || {
            cat "$base-export.log" >&2
            die "could not export $label to SQLite"
        }
    [[ -s $base.sqlite ]] || die "$label SQLite export is empty"
    for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum \
                  cuda_gpu_trace nvtx_sum nvtx_gpu_proj_sum; do
        nsys stats --report "$report" --format csv "$base.nsys-rep" \
            >"$base-$report.csv" 2>"$base-$report.log" || true
    done
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$pp" "$threshold" "$TRACE_TOKENS" \
        "$GPU_DEVICES" "$base.sqlite" \
        >>"$OUTPUT_DIR/trace-map.tsv"
}

if (( RUN_NSYS )); then
    for pp in "${trace_contexts[@]}"; do
        capture_trace "pp${pp}-indexed1024" "$pp" 1024
        if (( pp == 4096 )); then
            capture_trace "pp4096-dense4096" 4096 4096
        fi
    done
fi

phase=summary
python3 speed-bench/summarize-sm75-decode-evidence.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary/stdout.txt"
[[ -s $OUTPUT_DIR/summary/summary.md ]] || die "summary was not produced"
cat "$OUTPUT_DIR/summary/summary.md"
phase=complete
