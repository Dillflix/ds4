#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the canonical SM75 production decode baseline.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22               fixed production split
  CASES=all                    or comma-separated canonical case IDs
  REPEATS=1
  WARMUP_TOKENS=512            untimed cache-admission prefill
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  SKIP_BUILD=0
  RESUME=0                     reuse validated case files in DECODE_BASELINE_DIR
  CREATE_ARCHIVE=1
  DECODE_BASELINE_DIR=...      output directory

Canonical case IDs:
  pp512-tg256, pp512-tg512, pp2048-tg256, pp2048-tg512,
  pp4096-tg256, pp4096-tg512, pp32768-tg256, pp32768-tg512
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
CASES=${CASES:-all}
REPEATS=${REPEATS:-1}
WARMUP_TOKENS=${WARMUP_TOKENS:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${DECODE_BASELINE_DIR:-$repo_dir/sm75-decode-baseline-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged Q4-32/Q3A4 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "the production decode baseline requires the fixed promessi_sposi prompt"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "REPEATS:$REPEATS" \
            "WARMUP_TOKENS:$WARMUP_TOKENS" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "PIPELINE_MB:$PIPELINE_MB" "SKIP_BUILD:$SKIP_BUILD" \
            "RESUME:$RESUME" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 )) || die "the production decode baseline requires STAGE_SPLIT=22"
[[ $GPU_DEVICES == 0,3,1,2 ]] ||
    die "the production decode baseline requires GPU_DEVICES=0,3,1,2"
[[ $GPU_VRAM == auto ]] || die "the production decode baseline requires GPU_VRAM=auto"
(( REPEATS >= 1 && WARMUP_TOKENS == 512 && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 )) ||
    die "require repeats>=1, warmup_tokens=512, prefill_chunk=2048, and pipeline_mb=512"
for flag in SKIP_BUILD RESUME CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk cat cmp date dirname env git grep make mkdir mv nproc \
            nvidia-smi python3 sort stat tail tar tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

canonical_cases=(pp512-tg256 pp512-tg512 pp2048-tg256 pp2048-tg512
                 pp4096-tg256 pp4096-tg512 pp32768-tg256 pp32768-tg512)
declare -A case_pp=(
    [pp512-tg256]=512 [pp512-tg512]=512
    [pp2048-tg256]=2048 [pp2048-tg512]=2048
    [pp4096-tg256]=4096 [pp4096-tg512]=4096
    [pp32768-tg256]=32768 [pp32768-tg512]=32768
)
declare -A case_tg=(
    [pp512-tg256]=256 [pp512-tg512]=512
    [pp2048-tg256]=256 [pp2048-tg512]=512
    [pp4096-tg256]=256 [pp4096-tg512]=512
    [pp32768-tg256]=256 [pp32768-tg512]=512
)
declare -A requested=()
if [[ $CASES == all ]]; then
    for case_id in "${canonical_cases[@]}"; do requested[$case_id]=1; done
else
    IFS=, read -r -a requested_items <<<"$CASES"
    (( ${#requested_items[@]} > 0 )) || die "CASES selected no cases"
    for case_id in "${requested_items[@]}"; do
        [[ -n $case_id && -n ${case_pp[$case_id]+x} ]] ||
            die "unknown CASES entry: ${case_id:-empty}"
        [[ -z ${requested[$case_id]+x} ]] || die "duplicate CASES entry: $case_id"
        requested[$case_id]=1
    done
fi
selected_cases=()
for case_id in "${canonical_cases[@]}"; do
    [[ -n ${requested[$case_id]+x} ]] && selected_cases+=("$case_id")
done
(( ${#selected_cases[@]} > 0 )) || die "CASES selected no cases"
selected_csv=$(IFS=,; printf '%s' "${selected_cases[*]}")

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four IDs"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "GPU_DEVICES contains an invalid or duplicate ID: $gpu"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done

if [[ $RESUME == 1 ]]; then
    [[ -n ${DECODE_BASELINE_DIR:-} && -d $OUTPUT_DIR ]] ||
        die "RESUME=1 requires an existing DECODE_BASELINE_DIR"
else
    [[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
        die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"/{runs,summary,provenance}
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

emit_configuration() {
    printf 'model=%s\nprompt=%s\ngpu_devices=%s\ngpu_vram=%s\n' \
        "$MODEL" "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM"
    printf 'stage_split=%s\ncases=%s\nrepeats=%s\nwarmup_tokens=%s\n' \
        "$STAGE_SPLIT" "$selected_csv" "$REPEATS" "$WARMUP_TOKENS"
    printf 'prefill_chunk=%s\npipeline_mb=%s\n' "$PREFILL_CHUNK" "$PIPELINE_MB"
}
if [[ $RESUME == 1 ]]; then
    [[ -f $OUTPUT_DIR/configuration.txt ]] || die "resume configuration is missing"
    cmp -s <(emit_configuration) "$OUTPUT_DIR/configuration.txt" ||
        die "resume configuration differs from the original run"
else
    emit_configuration >"$OUTPUT_DIR/configuration.txt"
fi

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
production_env=(
    "${clean[@]}"
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
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
if [[ $RESUME == 0 ]]; then
    {
        printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$(git branch --show-current)"
        printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
            "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
        printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
            "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
        printf 'cases=%s\nrepeats=%s\nwarmup_tokens=%s\n' \
            "$selected_csv" "$REPEATS" "$WARMUP_TOKENS"
        printf 'prefill_chunk=%s\npipeline_mb=%s\ndense_f16_admission=344/344-required\n' \
            "$PREFILL_CHUNK" "$PIPELINE_MB"
        printf 'decode_experiments=disabled\n'
        printf '\n[gpu inventory]\n'
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
            --format=csv
        printf '\n[topology]\n'; nvidia-smi topo -m
    } >"$OUTPUT_DIR/manifest.txt"
    git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
    git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
    env | awk -F= '$1 ~ /^DS4_/ {print}' | sort -u \
        >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"
fi

validate_log() {
    local log=$1
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    ! grep -Fq 'required but unavailable' "$log"
}

phase=decode-matrix
runs_partial="$OUTPUT_DIR/runs/runs.csv.partial.$$"
printf 'repeat,slot,case_id,pp_tokens,tg_tokens,ctx_alloc,csv,log\n' >"$runs_partial"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    run_order=()
    if (( repeat % 2 )); then
        run_order=("${selected_cases[@]}")
    else
        for ((index=${#selected_cases[@]}-1; index>=0; index--)); do
            run_order+=("${selected_cases[$index]}")
        done
    fi
    slot=0
    for case_id in "${run_order[@]}"; do
        slot=$((slot + 1))
        pp=${case_pp[$case_id]}; tg=${case_tg[$case_id]}
        ctx_alloc=$((pp + tg + 1))
        base="$OUTPUT_DIR/runs/r${repeat}-${case_id}"
        valid=0
        if [[ $RESUME == 1 && -s $base.csv && -s $base.log ]]; then
            if python3 speed-bench/summarize-sm75-decode-baseline.py \
                    --validate-case "$base.csv" --case-id "$case_id" \
                    --pp "$pp" --tg "$tg" && validate_log "$base.log"; then
                valid=1
                printf 'Reusing decode baseline repeat=%d/%d case=%s...\n' \
                    "$repeat" "$REPEATS" "$case_id"
            fi
        fi
        if [[ $valid == 0 ]]; then
            printf 'Decode baseline repeat=%d/%d slot=%d/%d case=%s...\n' \
                "$repeat" "$REPEATS" "$slot" "${#selected_cases[@]}" "$case_id"
            run_rc=0
            "${production_env[@]}" \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$WARMUP_TOKENS" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$pp" --ctx-max "$pp" --ctx-alloc "$ctx_alloc" \
                --step-incr "$pp" --prefill-chunk "$PREFILL_CHUNK" \
                --gen-tokens "$tg" --csv "$base.csv" \
                    >"$base.log" 2>&1 || run_rc=$?
            (( run_rc == 0 )) || {
                tail -n 200 "$base.log" >&2 || true
                die "$case_id failed (exit $run_rc)"
            }
            python3 speed-bench/summarize-sm75-decode-baseline.py \
                --validate-case "$base.csv" --case-id "$case_id" \
                --pp "$pp" --tg "$tg" || {
                    tail -n 120 "$base.log" >&2 || true
                    die "$case_id produced an invalid benchmark row"
                }
            validate_log "$base.log" || {
                tail -n 200 "$base.log" >&2 || true
                die "$case_id did not preserve the fixed production configuration"
            }
        fi
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$repeat" "$slot" "$case_id" "$pp" "$tg" "$ctx_alloc" \
            "$base.csv" "$base.log" >>"$runs_partial"
    done
done
mv -- "$runs_partial" "$OUTPUT_DIR/runs/runs.csv"

phase=summary
python3 speed-bench/summarize-sm75-decode-baseline.py \
    "$OUTPUT_DIR/runs/runs.csv" "$OUTPUT_DIR/summary"
cat "$OUTPUT_DIR/summary/summary.md"
phase=complete
