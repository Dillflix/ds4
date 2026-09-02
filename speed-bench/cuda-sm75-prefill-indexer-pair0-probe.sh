#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run one production-shaped 32K prefill process with pair-0 attention row
splitting disabled and pair-0 prefill indexer row splitting independently
selected.

Run VARIANT=indexer-on first. If it completes with healthy GPUs, run
VARIANT=control as a separate process. This runner deliberately has no resume
mode because a GPU-loss result requires a reboot.

Optional environment:
  VARIANT=indexer-on|control  default: indexer-on
  MODEL_LAYOUT=mixed15|all43 default: mixed15
  MIXED_MODEL=...            default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  ALL43_MODEL=...            default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf
  PROMPT=...                 default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REQUIRED_POWER_LIMITS_W=250,260,250,250
  CASE_TIMEOUT_SECONDS=600
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  REFERENCE_DIR=...          control-only completed indexer-on directory
  PREFILL_INDEXER_PAIR0_PROBE_DIR=...

Both variants keep pair-1 attention and prefill-indexer row splitting active,
use the default fused T32 FP16-output path, and differ only in whether pair 0
also splits the prefill indexer. Pair-0 home attention consumes the exact
gathered top-k result in indexer-on.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
VARIANT=${VARIANT:-indexer-on}
MODEL_LAYOUT=${MODEL_LAYOUT:-mixed15}
MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
case "$MODEL_LAYOUT" in
    mixed15) MODEL=$MIXED_MODEL ;;
    all43) MODEL=$ALL43_MODEL ;;
    *) die "MODEL_LAYOUT must be mixed15 or all43" ;;
esac
case "$VARIANT" in
    indexer-on) INDEXER_PAIRS=0,1 ;;
    control) INDEXER_PAIRS=1 ;;
    *) die "VARIANT must be indexer-on or control" ;;
esac
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-600}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
REFERENCE_DIR=${REFERENCE_DIR:-}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_ALLOC=33025
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PREFILL_INDEXER_PAIR0_PROBE_DIR:-$repo_dir/sm75-prefill-indexer-pair0-$VARIANT-$stamp}

[[ $MODEL == /* && -f $MODEL ]] || die "model not found at absolute path: $MODEL"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production probe requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
[[ $CASE_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] ||
    die "CASE_TIMEOUT_SECONDS must be a positive integer"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
if [[ -n $REFERENCE_DIR ]]; then
    [[ $VARIANT == control ]] ||
        die "REFERENCE_DIR is accepted only with VARIANT=control"
    [[ $REFERENCE_DIR == /* && -d $REFERENCE_DIR ]] ||
        die "REFERENCE_DIR must name an existing absolute candidate directory"
    grep -Fqx 'state=finished' "$REFERENCE_DIR/run-status.txt" ||
        die "REFERENCE_DIR is not a completed candidate run"
    grep -Fqx 'variant=indexer-on' "$REFERENCE_DIR/manifest.txt" ||
        die "REFERENCE_DIR is not an indexer-on run"
    grep -Fqx "model_layout=$MODEL_LAYOUT" "$REFERENCE_DIR/manifest.txt" ||
        die "REFERENCE_DIR used a different model layout"
    grep -Fqx "model=$MODEL" "$REFERENCE_DIR/manifest.txt" ||
        die "REFERENCE_DIR used a different model path"
    grep -Fqx "git_commit=$(git rev-parse HEAD)" "$REFERENCE_DIR/manifest.txt" ||
        die "REFERENCE_DIR used a different engine commit"
fi
for tool in awk basename cat cmp date dirname env find git grep make mkdir mv nproc \
            nvidia-smi rm sort stat stdbuf tail tar tee timeout tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#required_power[@]} == 4 )) ||
    die "REQUIRED_POWER_LIMITS_W must contain physical GPU 0,1,2,3 limits"
for gpu in 0 1 2 3; do
    [[ ${required_power[$gpu]} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        die "invalid required power limit for physical GPU $gpu"
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] ||
        die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "physical GPU $gpu power limit is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{provenance,logits}
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
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    "DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=$INDEXER_PAIRS"
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_AUDIT=1
)

phase=build
targets=(ds4-bench tests/test_engine_mgpu_placement)
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

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in 0 1 2 3; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

phase=manifest
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture initial four-GPU identity and power limits"
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'variant=%s\nmodel_layout=%s\nmodel=%s\nmodel_bytes=%s\n' \
        "$VARIANT" "$MODEL_LAYOUT" "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'prompt=%s\ngpu_devices=%s\nstage_split=22/21\n' \
        "$PROMPT" "$GPU_DEVICES"
    printf 'context=32768\nprefill_chunk=%s\npipeline_mb=%s\nctx_alloc=%s\n' \
        "$PREFILL_CHUNK" "$PIPELINE_MB" "$CTX_ALLOC"
    printf 'pair0_attention_rows=disabled\nprefill_indexer_pairs=%s\n' \
        "$INDEXER_PAIRS"
    printf 't32_f16_fused=engine-default\nrequired_power_limits_w=%s\n' \
        "$REQUIRED_POWER_LIMITS_W"
    printf 'reference_dir=%s\n' "${REFERENCE_DIR:-none}"
    cat "$OUTPUT_DIR/initial-gpu.csv"
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=$VARIANT-32k
printf 'state=started\nphase=%s\ndate_utc=%s\n' "$phase" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$OUTPUT_DIR/case.started"
capture_gpu_health "$OUTPUT_DIR/case.pre-gpu.csv" ||
    die "could not capture pre-run GPU state"

set +e
"${production_env[@]}" \
    "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$OUTPUT_DIR/q8-plan.csv" \
    "DS4_CUDA_Q8_BINDING_STATE_CSV=$OUTPUT_DIR/q8-bindings.csv" \
    timeout --signal=INT --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
    stdbuf -oL -eL ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start 32768 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC" \
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens 0 --csv "$OUTPUT_DIR/result.csv" \
        --dump-frontier-logits-dir "$OUTPUT_DIR/logits" \
        >"$OUTPUT_DIR/case.log" 2>&1
run_status=$?
set -e
printf 'status=%s\ndate_utc=%s\n' "$run_status" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$OUTPUT_DIR/case.result"
capture_gpu_health "$OUTPUT_DIR/case.post-gpu.csv" ||
    die "GPU health capture failed after $VARIANT"
(( run_status == 0 )) || {
    tail -n 240 "$OUTPUT_DIR/case.log" >&2 || true
    die "$VARIANT 32K process failed with status $run_status"
}
cmp -s "$OUTPUT_DIR/case.pre-gpu.csv" "$OUTPUT_DIR/case.post-gpu.csv" ||
    die "GPU identity or power limits changed during $VARIANT"

grep -Fq 'prefill attention row split pair-scoped disable: logical-pairs=0' \
    "$OUTPUT_DIR/case.log" || die "pair-0 attention suppression was not active"
grep -Fq 'CUDA TP cache mirror policy: attention-pair-mask=0x2 index-pair-mask=0x3' \
    "$OUTPUT_DIR/case.log" ||
    die "pair-specific attention/index cache mirror policy was not active"
! grep -Fq 'prefill attention query-row split enabled: tier 0 ' \
    "$OUTPUT_DIR/case.log" || die "pair-0 attention row splitting dispatched"
grep -Fq 'prefill attention query-row split enabled: tier 1 ' \
    "$OUTPUT_DIR/case.log" || die "pair-1 attention row splitting did not dispatch"
grep -Eq 'CUDA T32 f16-output fused summary: local=0 partner=[1-9][0-9]*' \
    "$OUTPUT_DIR/case.log" || die "default T32 FP16-output path did not dispatch"
grep -Eq 'prefill indexer row audit event=complete .*home_tier=1 .*selected_mode=partner-local' \
    "$OUTPUT_DIR/case.log" || die "pair-1 prefill indexer split did not dispatch"

if [[ $VARIANT == indexer-on ]]; then
    grep -Eq 'prefill indexer row audit event=complete .*home_tier=0 .*selected_mode=gather-home' \
        "$OUTPUT_DIR/case.log" || die "pair-0 indexer-only split did not dispatch"
else
    ! grep -Eq 'prefill indexer row audit event=complete .*home_tier=0 ' \
        "$OUTPUT_DIR/case.log" || die "control dispatched pair-0 prefill indexer split"
fi
awk -F, 'NR>1 && $1==32768 && ($3+0)>0 {ok=1} END {exit !ok}' \
    "$OUTPUT_DIR/result.csv" || die "missing successful 32K prefill result"

if [[ -n $REFERENCE_DIR ]]; then
    phase=exact-output-comparison
    mapfile -t reference_logits < <(
        cd "$REFERENCE_DIR/logits" && find . -maxdepth 1 -type f -printf '%P\n' |
            sort
    )
    mapfile -t control_logits < <(
        cd "$OUTPUT_DIR/logits" && find . -maxdepth 1 -type f -printf '%P\n' |
            sort
    )
    (( ${#reference_logits[@]} > 0 )) ||
        die "REFERENCE_DIR contains no frontier logits"
    [[ "${reference_logits[*]}" == "${control_logits[*]}" ]] ||
        die "candidate/control frontier-logit file sets differ"
    : >"$OUTPUT_DIR/exact-output.txt"
    for name in "${reference_logits[@]}"; do
        cmp -s "$REFERENCE_DIR/logits/$name" "$OUTPUT_DIR/logits/$name" ||
            die "candidate/control frontier logits differ: $name"
        printf 'byte-exact,%s\n' "$name" >>"$OUTPUT_DIR/exact-output.txt"
    done
fi

phase=complete
printf 'SM75 pair-0 prefill indexer %s probe complete: %s\n' \
    "$VARIANT" "$OUTPUT_DIR"
