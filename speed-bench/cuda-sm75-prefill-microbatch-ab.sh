#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Compare 512- and 1024-token CUDA pipeline microbatches in the fixed SM75
production configuration. The 2048-token prefill chunk, 22/21 stage split,
complete dense-F16 admission, fixed row splits, and context allocation are
identical in both arms.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22                fixed
  CTX_START=2048
  CTX_MAX=32768
  CTX_ALLOC=32769               must exceed CTX_MAX
  PREFILL_CHUNK=2048            fixed to isolate microbatch size
  REPEATS=3                     minimum 3
  RUN_TILE_AUDIT=1              one untimed 2K capture per arm
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PREFILL_MB_AB_DIR=/absolute/output/directory
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
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-32768}
CTX_ALLOC=${CTX_ALLOC:-32769}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-3}
RUN_TILE_AUDIT=${RUN_TILE_AUDIT:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PREFILL_MB_AB_DIR:-$repo_dir/sm75-prefill-microbatch-ab-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged Q4-32/Q3A4 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "CTX_ALLOC:$CTX_ALLOC" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "REPEATS:$REPEATS" \
            "RUN_TILE_AUDIT:$RUN_TILE_AUDIT" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 )) || die "this production comparison requires STAGE_SPLIT=22"
(( PREFILL_CHUNK == 2048 )) ||
    die "PREFILL_CHUNK must remain 2048 so only the microbatch changes"
(( CTX_START >= 2048 && CTX_START % 2048 == 0 &&
   CTX_MAX >= CTX_START && CTX_MAX % 2048 == 0 &&
   CTX_ALLOC > CTX_MAX && REPEATS >= 3 )) || die "invalid benchmark bounds"
ctx_ratio=$((CTX_MAX / CTX_START))
(( CTX_MAX % CTX_START == 0 && (ctx_ratio & (ctx_ratio - 1)) == 0 )) ||
    die "CTX_MAX/CTX_START must be an integer power of two"
for flag in RUN_TILE_AUDIT SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk basename cat cmp date dirname env find git grep make mkdir \
            mv nproc nvidia-smi python3 sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{performance,tile-audit,validation,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM HUP
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

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four IDs"
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen[$gpu]+x} ]] ||
        die "GPU_DEVICES contains an invalid or duplicate ID: $gpu"
    seen[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}"
done

phase=build
targets=(ds4-bench tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/validation/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/validation/planner.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/validation/planner.log" >&2 || true
        die "placement regression tests failed"
    }

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=22/21\n' \
        "$GPU_DEVICES" "$GPU_VRAM"
    printf 'ctx_start=%s\nctx_max=%s\nctx_alloc=%s\nprefill_chunk=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$CTX_ALLOC" "$PREFILL_CHUNK" "$REPEATS"
    printf 'variants=mb512,mb1024\nrun_tile_audit=%s\n' "$RUN_TILE_AUDIT"
    printf '\n[gpu]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap \
        --format=csv
    printf '\n[topology]\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

run_production() {
    local variant=$1 microbatch=$2 base=$3 logits=$4
    "${clean[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        "DS4_CUDA_PREFILL_PIPELINE_MB=$microbatch" \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK" \
        DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
        DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
        DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
        DS4_CUDA_INDEXER_SCORE_AUDIT=1 \
        DS4_CUDA_PREFILL_AUDIT=1 \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
            --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
            --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
            --csv "$base.csv" --dump-frontier-logits-dir "$logits" \
            >"$base.log" 2>&1 || {
                tail -n 220 "$base.log" >&2 || true
                die "$variant production run failed"
            }
}

validate_production() {
    local variant=$1 microbatch=$2 base=$3 benchmark_csv=$4
    local expected_start=${5:-$CTX_START} expected_max=${6:-$CTX_MAX}
    local half_rows=$((microbatch / 2))
    [[ -s $benchmark_csv ]] || die "$variant omitted benchmark CSV"
    local expected=$expected_start expected_rows=0
    while (( expected <= expected_max )); do
        awk -F, -v want="$expected" \
            'NR > 1 && ($1 + 0) == want {found++} END {exit !(found == 1)}' \
            "$benchmark_csv" ||
            die "$variant benchmark omitted or duplicated context $expected"
        expected_rows=$((expected_rows + 1))
        expected=$((expected * 2))
    done
    [[ $(awk 'END {print NR - 1}' "$benchmark_csv") == "$expected_rows" ]] ||
        die "$variant benchmark contains an unexpected frontier"
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled'; do
        grep -Fq "$marker" "$base.log" ||
            die "$variant lacks required marker: $marker"
    done
    [[ $(grep -Fc 'CUDA prefill attention query-row split enabled:' \
        "$base.log") == 2 ]] || die "$variant did not split attention on both pairs"
    [[ $(grep -Fc 'CUDA prefill indexer score/top-k row split enabled:' \
        "$base.log") == 2 ]] || die "$variant did not split indexer on both pairs"
    [[ $(grep -Fc 'qualified=yes' "$base.log") == 2 ]] ||
        die "$variant did not qualify both SM75 NVLink pairs"
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$base.log" ||
            die "$variant lacks required direct route $route"
    done
    [[ $(grep -Fc "rows [0,$half_rows), tier " "$base.log") == 4 ]] ||
        die "$variant did not retain fixed $half_rows/$half_rows row ownership"
    grep -Fq 'dispatch=split kind=indexed' "$base.log" ||
        die "$variant omitted indexed row-split dispatch"
    grep -Fq 'dispatch=split kind=mixed' "$base.log" ||
        die "$variant omitted mixed row-split dispatch"
    ! grep -Fq 'CUDA prefill attention row calibration' "$base.log" ||
        die "$variant unexpectedly used dynamic row calibration"
    ! grep -Fq 'required but unavailable' "$base.log" ||
        die "$variant encountered a forbidden split fallback"
    grep -Fq "CUDA prefill audit tokens=2048 microbatch_cap=$microbatch" \
        "$base.log" || die "$variant omitted its structural microbatch audit"
    if [[ $microbatch == 512 ]]; then
        grep -Fq 'microbatches=4 stages=2 waves=5' "$base.log" ||
            die "$variant did not execute the expected four-microbatch/five-wave graph"
    else
        grep -Fq 'microbatches=2 stages=2 waves=3' "$base.log" ||
            die "$variant did not execute the expected two-microbatch/three-wave graph"
    fi
    grep -Fq 'features pipeline=1 expert_parallel=1 q8_f16_cache=1 attention_head_split=0 attention_row_split=1' \
        "$base.log" || die "$variant structural feature audit differs"
}

phase=performance-ab
printf 'repeat,slot,variant,microbatch,csv,log,logits\n' \
    >"$OUTPUT_DIR/performance/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then variants=(mb512 mb1024); else variants=(mb1024 mb512); fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        microbatch=${variant#mb}
        base="$OUTPUT_DIR/performance/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'Production prefill microbatch A/B repeat=%d/%d slot=%d variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        run_production "$variant" "$microbatch" "$base" "$logits"
        validate_production "$variant" "$microbatch" "$base" "$base.csv"
        printf '%s,%s,%s,%s,%s,%s,%s\n' \
            "$repeat" "$slot" "$variant" "$microbatch" \
            "$base.csv" "$base.log" "$logits" \
            >>"$OUTPUT_DIR/performance/runs.csv"
    done

    mb512_logits=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "mb512" {print $7}' \
        "$OUTPUT_DIR/performance/runs.csv")
    mb1024_logits=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "mb1024" {print $7}' \
        "$OUTPUT_DIR/performance/runs.csv")
    mapfile -t files512 < <(find "$mb512_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    mapfile -t files1024 < <(find "$mb1024_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    [[ ${#files512[@]} -gt 0 && "${files512[*]}" == "${files1024[*]}" ]] ||
        die "repeat $repeat logits inventory differs"
    for file in "${files512[@]}"; do
        cmp -s "$mb512_logits/$file" "$mb1024_logits/$file" ||
            die "repeat $repeat logits differ: $file"
    done
done

phase=performance-summary
python3 speed-bench/summarize-sm75-prefill-microbatch.py \
    "$OUTPUT_DIR/performance/runs.csv" "$OUTPUT_DIR/performance/summary.csv" \
    | tee "$OUTPUT_DIR/performance/summary.txt"

if [[ $RUN_TILE_AUDIT == 1 ]]; then
    phase=deferred-tile-audit
    for microbatch in 512 1024; do
        variant=mb$microbatch
        base="$OUTPUT_DIR/tile-audit/$variant"
        printf 'Deferred production tile audit variant=%s...\n' "$variant"
        "${clean[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 \
            "DS4_CUDA_PREFILL_PIPELINE_MB=$microbatch" \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK" \
            DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
            DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
            DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
            DS4_CUDA_PREFILL_AUDIT=1 \
            "DS4_CUDA_PREFILL_TILE_AUDIT_CSV=$base.csv" \
            DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY=65536 \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start 2048 --ctx-max 2048 --ctx-alloc "$CTX_ALLOC" \
                --step-incr 2048 --prefill-chunk "$PREFILL_CHUNK" \
                --gen-tokens 0 --csv "$base-benchmark.csv" \
                >"$base.log" 2>&1 || {
                    tail -n 220 "$base.log" >&2 || true
                    die "$variant tile-audit run failed"
                }
        validate_production "$variant" "$microbatch" "$base" \
            "$base-benchmark.csv" 2048 2048
        grep -Fq 'wrote CUDA tile audit' "$base.log" ||
            die "$variant did not flush its deferred tile audit"
        [[ -s $base.csv ]] || die "$variant tile audit is empty"
        python3 speed-bench/summarize-tile-audit.py \
            "$base.csv" "$base-summary.csv" >"$base-summary.txt"
    done
fi

phase=complete
printf 'SM75 production prefill microbatch A/B complete: %s\n' "$OUTPUT_DIR"
