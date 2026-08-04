#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a balanced full-model four-GPU prefill A/B for the validated SM75
scalar-slot routed-MoE candidates.

Required environment:
  MODEL=/absolute/path/model.gguf
  RECIPE=hybrid|full-q4

Optional environment:
  PROMPT=/absolute/path/prompt.txt
  PROMPT_MANIFEST=/absolute/path/prompts.tsv  # label<TAB>path
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  STAGE_SPLIT=auto        # hybrid: 25/18; full-q4: 22/21
  CTX_START=2048
  CTX_MAX=65536
  STEP_MUL=2
  PREFILL_CHUNK=2048
  REPEATS=2              # even pilot; use 6 for replicated measurement
  CUDA_ARCH=sm_75
  SKIP_BUILD=0
  HASH_MODEL=0           # set 1 to opt into a full model SHA-256 pass
  CREATE_ARCHIVE=1
  SCALAR_E2E_DIR=...     # new output directory

The hybrid candidate jointly enables IQ2 gate/up and Q4 down scalar slots.
The full-q4 candidate jointly enables Q4 gate/up and Q4 down scalar slots.
Production enables all three on SM75. This runner explicitly sets them to 0
for its base arm and to 1 only for the recipe-relevant scalar arm.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute GGUF path}"
: "${RECIPE:?set RECIPE to hybrid or full-q4}"
[[ $MODEL == /* ]] || die "MODEL must be an absolute path"
[[ -f $MODEL ]] || die "model not found: $MODEL"
[[ $RECIPE == hybrid || $RECIPE == full-q4 ]] ||
    die "RECIPE must be hybrid or full-q4"

GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-auto}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-2}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
SKIP_BUILD=${SKIP_BUILD:-0}
HASH_MODEL=${HASH_MODEL:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${SCALAR_E2E_DIR:-$repo_dir/sm75-production-scalar-e2e-$RECIPE-$run_stamp}
while [[ $OUTPUT_DIR != / && $OUTPUT_DIR == */ ]]; do
    OUTPUT_DIR=${OUTPUT_DIR%/}
done

if [[ $STAGE_SPLIT == auto ]]; then
    if [[ $RECIPE == hybrid ]]; then STAGE_SPLIT=25; else STAGE_SPLIT=22; fi
fi
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "REPEATS:$REPEATS" \
            "STAGE_SPLIT:$STAGE_SPLIT" \
            "SKIP_BUILD:$SKIP_BUILD" "HASH_MODEL:$HASH_MODEL" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START && STEP_MUL >= 2 &&
   PREFILL_CHUNK > 0 &&
   REPEATS > 0 && STAGE_SPLIT > 0 &&
   STAGE_SPLIT < 43 )) || die "invalid benchmark range or stage split"
(( 10#$REPEATS % 2 == 0 )) || die "REPEATS must be even for balanced ordering"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $HASH_MODEL == 0 || $HASH_MODEL == 1 ]] || die "HASH_MODEL must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
for tool in awk cmp env grep make mv nproc python3 nvidia-smi rm sort stat tar tee sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

# Every inherited DS4 override is cleared below, including DS4_LOCK_FILE, so
# the benchmark will use DS4's global default lock.  Fail before an expensive
# build or model hash if a stale root-owned lock would make every run exit.
default_lock=/tmp/ds4.lock
if [[ -e $default_lock ]]; then
    [[ -f $default_lock && -w $default_lock ]] ||
        die "$default_lock exists but is not a writable regular file"
else
    [[ -d /tmp && -w /tmp ]] ||
        die "/tmp is not writable; DS4 cannot create $default_lock"
fi

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four devices"
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
    [[ -z ${seen_gpus[$gpu]+x} ]] || die "duplicate GPU index: $gpu"
    seen_gpus[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is not SM75 (${cap:-unknown})"
done

gpu_topology=$(nvidia-smi topo -m)
topology_link() {
    local from=$1 to=$2
    awk -v from="GPU$from" -v to="GPU$to" '
        NR == 1 {
            for (i = 1; i <= NF; i++) if ($i ~ /^GPU[0-9]+$/) col[$i] = i + 1
            next
        }
        $1 == from && (to in col) { print $(col[to]); exit }
    ' <<<"$gpu_topology"
}
pair0_link=$(topology_link "${gpu_ids[0]}" "${gpu_ids[2]}")
pair1_link=$(topology_link "${gpu_ids[1]}" "${gpu_ids[3]}")
[[ $pair0_link =~ ^NV[0-9]+$ ]] ||
    die "GPU ${gpu_ids[0]}<->${gpu_ids[2]} is not an NVLink pair ($pair0_link)"
[[ $pair1_link =~ ^NV[0-9]+$ ]] ||
    die "GPU ${gpu_ids[1]}<->${gpu_ids[3]} is not an NVLink pair ($pair1_link)"

declare -a prompt_labels=() prompt_paths=()
declare -A seen_labels=()
if [[ -n ${PROMPT_MANIFEST:-} ]]; then
    [[ -f $PROMPT_MANIFEST ]] || die "prompt manifest not found: $PROMPT_MANIFEST"
    while IFS=$'\t' read -r label path extra; do
        [[ -n $label && ${label:0:1} != '#' ]] || continue
        [[ -n $path && -z ${extra:-} ]] || die "invalid prompt manifest row"
        [[ $path == /* ]] || path="$repo_dir/$path"
        [[ -f $path ]] || die "prompt not found: $path"
        [[ $label =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe prompt label: $label"
        [[ -z ${seen_labels[$label]+x} ]] || die "duplicate prompt label: $label"
        seen_labels[$label]=1
        prompt_labels+=("$label"); prompt_paths+=("$path")
    done <"$PROMPT_MANIFEST"
else
    prompt=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
    [[ -f $prompt ]] || die "prompt not found: $prompt"
    prompt_labels+=(promessi); prompt_paths+=("$prompt")
fi
(( ${#prompt_paths[@]} > 0 )) || die "prompt suite is empty"

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
[[ ! -e $OUTPUT_DIR.tar.gz ]] || die "archive path already exists: $OUTPUT_DIR.tar.gz"
mkdir -p "$OUTPUT_DIR"/{runs,provenance,validation}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    local archive="$OUTPUT_DIR.tar.gz"
    local partial="$archive.partial.$$"
    local completed_phase=$current_phase
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$archive"; then
            printf 'archive=%s\n' "$archive"
        else
            rm -f -- "$partial" "$archive"
            (( status != 0 )) || status=1
            current_phase="$completed_phase;archive-failed"
            printf 'state=failed\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
                "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                >"$OUTPUT_DIR/run-status.txt"
            printf 'error: could not archive %s\n' "$OUTPUT_DIR" >&2
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'current_phase=interrupted-INT; exit 130' INT
trap 'current_phase=interrupted-TERM; exit 143' TERM
trap 'current_phase=interrupted-HUP; exit 129' HUP

# Clear every inherited DS4 override, including controls added after this
# script was written. The fixed split, audit marker, and candidate toggles are
# appended after the removals and are therefore the only DS4 differences.
mapfile -t inherited_ds4_envs < <(
    env | awk -F= '$1 ~ /^DS4_/ { print $1 }' | sort -u
)
clean_prefix=(env)
for name in "${inherited_ds4_envs[@]}"; do clean_prefix+=(-u "$name"); done
base_prefix=("${clean_prefix[@]}"
             "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
             DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=0
             DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=0
             DS4_CUDA_MOE_IQ2_SCALAR_SM75=0)
scalar_prefix=("${base_prefix[@]}")
if [[ $RECIPE == hybrid ]]; then
    scalar_prefix+=(DS4_CUDA_MOE_IQ2_SCALAR_SM75=1
                    DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=1)
    candidate_components=iq2-gate-up+q4-down
else
    scalar_prefix+=(DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=1
                    DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=1)
    candidate_components=q4-gate-up+q4-down
fi
base_validation_prefix=("${base_prefix[@]}"
                        DS4_BENCH_ROUTED_QUANT_AUDIT=1
                        DS4_CUDA_MOE_SCALAR_AUDIT=1)
scalar_validation_prefix=("${scalar_prefix[@]}"
                          DS4_BENCH_ROUTED_QUANT_AUDIT=1
                          DS4_CUDA_MOE_SCALAR_AUDIT=1)

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        CUDA_ARCH="$CUDA_ARCH" 2>&1 | tee "$OUTPUT_DIR/build.log"
    current_phase=correctness
    "${clean_prefix[@]}" ./tests/cuda_long_context_smoke \
        >"$OUTPUT_DIR/correctness.log" 2>&1 || {
        tail -n 120 "$OUTPUT_DIR/correctness.log" >&2 || true
        die "CUDA correctness smoke failed"
    }
    for marker in 'sm75 iq2 moe tile16 scalar exact' \
                  'sm75 iq2 moe tile8 scalar exact' \
                  'sm75 q4 gate tile8 scalar exact' \
                  'sm75 q4 down tile16 scalar exact' \
                  'cuda long-context regression: OK'; do
        grep -Fq "$marker" "$OUTPUT_DIR/correctness.log" ||
            die "correctness marker missing: $marker"
    done
else
    [[ -x ./ds4-bench ]] || die "SKIP_BUILD=1 but ds4-bench is missing"
    [[ -x ./tests/cuda_long_context_smoke ]] ||
        die "SKIP_BUILD=1 but tests/cuda_long_context_smoke is missing"
    if ! make -q ds4-bench tests/cuda_long_context_smoke \
            CUDA_ARCH="$CUDA_ARCH"; then
        die "SKIP_BUILD=1 found out-of-date targets or make could not validate them; rebuild with SKIP_BUILD=0"
    fi
fi

runtime_common=(
    --cuda --cuda-tensor-parallel
    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
    --model "$MODEL" --prefill-chunk "$PREFILL_CHUNK"
    --gen-tokens 0
)
common=(
    "${runtime_common[@]}"
    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX"
    --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL"
)
current_phase=manifest
{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD 2>/dev/null || printf unknown)" \
        "$(git branch --show-current 2>/dev/null || printf unknown)"
    printf 'model=%s\nmodel_bytes=%s\nrecipe=%s\ncandidate_components=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$RECIPE" "$candidate_components"
    printf 'base_scalar_environment=%s\n' \
        'DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=0 DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=0 DS4_CUDA_MOE_IQ2_SCALAR_SM75=0'
    if [[ $RECIPE == hybrid ]]; then
        printf 'candidate_scalar_environment=%s\n' \
            'DS4_CUDA_MOE_IQ2_SCALAR_SM75=1 DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=1'
    else
        printf 'candidate_scalar_environment=%s\n' \
            'DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=1 DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=1'
    fi
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43 - STAGE_SPLIT))"
    printf 'nvlink_pair_0=%s<->%s:%s\nnvlink_pair_1=%s<->%s:%s\n' \
        "${gpu_ids[0]}" "${gpu_ids[2]}" "$pair0_link" \
        "${gpu_ids[1]}" "${gpu_ids[3]}" "$pair1_link"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL"
    printf 'prefill_chunk=%s\nuntimed_warmup_tokens=%s\nrepeats=%s\n' \
        "$PREFILL_CHUNK" "$CTX_START" "$REPEATS"
    printf 'measurement_grade=%s\n' \
        "$([[ $REPEATS -ge 6 ]] && printf replicated || printf pilot)"
    printf 'cleared_inherited_ds4_variables='
    printf '%s ' "${inherited_ds4_envs[@]}"
    printf '\n'
    printf '\n[prompts]\n'
    for i in "${!prompt_paths[@]}"; do
        printf '%s\t%s\n' "${prompt_labels[$i]}" "${prompt_paths[$i]}"
    done
    printf '\n[git status]\n'; git status --short 2>/dev/null || true
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[gpu topology]\n%s\n' "$gpu_topology"
} >"$OUTPUT_DIR/manifest.txt"
sha256sum ./ds4-bench >"$OUTPUT_DIR/provenance/ds4-bench.sha256"
if [[ $HASH_MODEL == 1 ]]; then
    current_phase=model-hash
    printf 'Hashing model for provenance: %s\n' "$MODEL"
    sha256sum "$MODEL" >"$OUTPUT_DIR/provenance/model.sha256"
else
    printf 'model_sha256=skipped\n' >"$OUTPUT_DIR/provenance/model.sha256"
fi
for prompt_path in "${prompt_paths[@]}"; do
    sha256sum "$prompt_path" >>"$OUTPUT_DIR/provenance/prompts.sha256"
done
printf 'recipe\tprompt\trepeat\tvariant\tcsv\tlog\ttelemetry_before\ttelemetry_after\tq8_cache_state_before\tq8_cache_state_after\n' \
    >"$OUTPUT_DIR/runs.tsv"
printf 'recipe\tprompt\tvariant\tcsv\tlog\tq8_audit\tq8_cache_state\tlogits_dir\n' \
    >"$OUTPUT_DIR/validation.tsv"

validate_common_log() {
    local log=$1
    grep -Fq "ds4: CUDA EP forced pipeline split $STAGE_SPLIT/$((43 - STAGE_SPLIT))" \
        "$log" || die "forced split missing from $log"
    grep -Fq "4 devices [$GPU_DEVICES] requested" "$log" ||
        die "GPU order/budget record missing from $log"
}

validate_dispatch_log() {
    local variant=$1 log=$2 scalar=0 opposite=1 path gpu count
    local expected_recipe recipe_count audit_count marker_count expected_markers=0
    local pos layer layer_start layer_end
    local -a expected_paths
    [[ $variant == scalar ]] && { scalar=1; opposite=0; }
    validate_common_log "$log"
    if [[ $RECIPE == hybrid ]]; then
        expected_paths=(iq2-gate-tile16 q4-down-tile16)
        expected_recipe='gate=iq2_xxs up=iq2_xxs down=q4_k'
    else
        expected_paths=(q4-gate-tile8 q4-down-tile16)
        expected_recipe='gate=q4_k up=q4_k down=q4_k'
    fi
    audit_count=$(grep -c '^ds4: routed-quant-audit layer=' "$log" || true)
    recipe_count=$(grep -Ec \
        "^ds4: routed-quant-audit layer=[0-9]+ $expected_recipe$" \
        "$log" || true)
    [[ $audit_count == 43 && $recipe_count == 43 ]] ||
        die "routed recipe audit expected 43 exact $RECIPE layers; got $recipe_count/$audit_count in $log"
    for path in "${expected_paths[@]}"; do
        for pos in "${!gpu_ids[@]}"; do
            gpu=${gpu_ids[$pos]}
            if (( pos == 0 || pos == 2 )); then
                layer_start=0; layer_end=$((STAGE_SPLIT - 1))
            else
                layer_start=$STAGE_SPLIT; layer_end=42
            fi
            for ((layer=layer_start; layer<=layer_end; layer++)); do
                count=$(grep -Fc \
                    "sm75-scalar-dispatch path=$path scalar=$scalar device=$gpu layer=$layer " \
                    "$log" || true)
                [[ $count == 1 ]] ||
                    die "$path scalar=$scalar device=$gpu layer=$layer launch count is $count in $log"
                if [[ $path == iq2-gate-tile16 ]]; then
                    grep -E \
                        "path=iq2-gate-tile16 scalar=$scalar device=$gpu layer=$layer .*stage_rows=8$" \
                        "$log" >/dev/null ||
                        die "shipping IQ2 stage8 marker missing for device $gpu layer $layer in $log"
                fi
                expected_markers=$((expected_markers + 1))
            done
        done
    done
    marker_count=$(grep -c '^ds4: sm75-scalar-dispatch ' "$log" || true)
    [[ $marker_count == "$expected_markers" ]] ||
        die "unexpected scalar dispatch coverage: $marker_count records, expected $expected_markers in $log"
    if grep -E "sm75-scalar-dispatch .*scalar=$opposite" "$log" >/dev/null; then
        die "opposite scalar specialization appeared in $log"
    fi
}

validate_clean_timing_log() {
    local log=$1
    validate_common_log "$log"
    if grep -Eq 'sm75-scalar-dispatch|routed-quant-audit|wrote CUDA Q8 cache audit' \
            "$log"; then
        die "timed run contained audit instrumentation: $log"
    fi
    [[ $(grep -Fc "ds4-bench: starting untimed CUDA warm-up frontier $CTX_START" \
            "$log" || true) == 1 ]] ||
        die "timed run is missing its single untimed warm-up start: $log"
    [[ $(grep -Fc "ds4-bench: completed untimed CUDA warm-up frontier $CTX_START" \
            "$log" || true) == 1 ]] ||
        die "timed run is missing its single untimed warm-up completion: $log"
}

capture_gpu_telemetry() {
    local path=$1
    nvidia-smi \
        --query-gpu=index,timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used \
        --format=csv >"$path"
}

current_phase=audited-validation
for p in "${!prompt_paths[@]}"; do
    label=${prompt_labels[$p]}
    prompt_path=${prompt_paths[$p]}
    for variant in base scalar; do
        stem="$variant-$label"
        csv="$OUTPUT_DIR/validation/$stem.csv"
        log="$OUTPUT_DIR/validation/$stem.log"
        q8_audit="$OUTPUT_DIR/validation/$stem.q8-cache.csv"
        q8_state="$OUTPUT_DIR/validation/$stem.q8-cache-state.csv"
        logits_dir="$OUTPUT_DIR/validation/$stem.logits"
        mkdir -p "$logits_dir"
        if [[ $variant == scalar ]]; then
            prefix=("${scalar_validation_prefix[@]}")
        else
            prefix=("${base_validation_prefix[@]}")
        fi
        run_prefix=("${prefix[@]}"
                    "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$q8_audit"
                    "DS4_CUDA_Q8_CACHE_STATE_CSV=$q8_state")
        printf 'Audited validation/warm-up: %s recipe=%s prompt=%s split=%s/%s\n' \
            "$variant" "$RECIPE" "$label" "$STAGE_SPLIT" "$((43 - STAGE_SPLIT))"
        if ! "${run_prefix[@]}" ./ds4-bench "${common[@]}" \
                --prompt-file "$prompt_path" --csv "$csv" \
                --dump-frontier-logits-dir "$logits_dir" \
                2>&1 | tee "$log"; then
            die "audited validation failed: $stem"
        fi
        [[ -s $csv ]] || die "validation produced no CSV: $stem"
        [[ -s $q8_audit ]] || die "Q8 cache audit is missing: $stem"
        [[ -s $q8_state ]] || die "Q8 cache-state snapshot is missing: $stem"
        validate_dispatch_log "$variant" "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$RECIPE" "$label" "$variant" "$csv" "$log" \
            "$q8_audit" "$q8_state" "$logits_dir" \
            >>"$OUTPUT_DIR/validation.tsv"
    done
done

# Fail before the repeated timing suite if the audited processes admitted
# different final Q8 cache ranges or produced different full-vocabulary
# outputs. The final summarizer independently rechecks these artifacts plus
# exact Q8 tensor coverage and layout fingerprints.
current_phase=validation-logit-precheck
for label in "${prompt_labels[@]}"; do
    base_state="$OUTPUT_DIR/validation/base-$label.q8-cache-state.csv"
    scalar_state="$OUTPUT_DIR/validation/scalar-$label.q8-cache-state.csv"
    cmp -s "$base_state" "$scalar_state" ||
        die "audited Q8 cache states differ for $label"
    frontier=$CTX_START
    while :; do
        raw_name=$(printf 'frontier_%06d.logits.f32' "$frontier")
        json_name=$(printf 'frontier_%06d.logits.json' "$frontier")
        base_raw="$OUTPUT_DIR/validation/base-$label.logits/$raw_name"
        scalar_raw="$OUTPUT_DIR/validation/scalar-$label.logits/$raw_name"
        base_json="$OUTPUT_DIR/validation/base-$label.logits/$json_name"
        scalar_json="$OUTPUT_DIR/validation/scalar-$label.logits/$json_name"
        [[ -s $base_raw && -s $scalar_raw ]] ||
            die "missing raw validation logits for $label frontier $frontier"
        [[ -s $base_json && -s $scalar_json ]] ||
            die "missing JSON validation logits for $label frontier $frontier"
        cmp -s "$base_raw" "$scalar_raw" ||
            die "raw FP32 validation logits differ for $label frontier $frontier"
        cmp -s "$base_json" "$scalar_json" ||
            die "JSON validation logits differ for $label frontier $frontier"
        (( frontier >= CTX_MAX )) && break
        next=$((frontier * STEP_MUL))
        (( next > CTX_MAX )) && next=$CTX_MAX
        frontier=$next
    done
done

current_phase=full-model-ab
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 == 1 )); then variants=(base scalar); else variants=(scalar base); fi
    for p in "${!prompt_paths[@]}"; do
        label=${prompt_labels[$p]}
        prompt_path=${prompt_paths[$p]}
        for variant in "${variants[@]}"; do
            stem="$variant-$label-r$repeat"
            csv="$OUTPUT_DIR/runs/$stem.csv"
            log="$OUTPUT_DIR/runs/$stem.log"
            telemetry_before="$OUTPUT_DIR/runs/$stem.telemetry-before.csv"
            telemetry_after="$OUTPUT_DIR/runs/$stem.telemetry-after.csv"
            q8_state_before="$OUTPUT_DIR/runs/$stem.q8-cache-state-before.csv"
            q8_state_after="$OUTPUT_DIR/runs/$stem.q8-cache-state-after.csv"
            printf 'Benchmarking %s recipe=%s prompt=%s repeat=%d/%d split=%s/%s\n' \
                "$variant" "$RECIPE" "$label" "$repeat" "$REPEATS" \
                "$STAGE_SPLIT" "$((43 - STAGE_SPLIT))"
            if [[ $variant == scalar ]]; then
                prefix=("${scalar_prefix[@]}")
            else
                prefix=("${base_prefix[@]}")
            fi
            capture_gpu_telemetry "$telemetry_before"
            run_prefix=("${prefix[@]}"
                        "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START"
                        "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$q8_state_before"
                        "DS4_CUDA_Q8_CACHE_STATE_CSV=$q8_state_after")
            if ! "${run_prefix[@]}" ./ds4-bench "${common[@]}" \
                    --prompt-file "$prompt_path" --csv "$csv" \
                    2>&1 | tee "$log"; then
                die "benchmark failed: $stem"
            fi
            capture_gpu_telemetry "$telemetry_after"
            [[ -s $csv ]] || die "benchmark produced no CSV: $stem"
            [[ -s $q8_state_before ]] ||
                die "pre-timing Q8 cache-state snapshot is missing: $stem"
            [[ -s $q8_state_after ]] ||
                die "post-timing Q8 cache-state snapshot is missing: $stem"
            cmp -s "$q8_state_before" "$q8_state_after" ||
                die "Q8 cache state changed during timed sweep: $stem"
            validate_clean_timing_log "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$RECIPE" "$label" "$repeat" "$variant" "$csv" "$log" \
                "$telemetry_before" "$telemetry_after" \
                "$q8_state_before" "$q8_state_after" \
                >>"$OUTPUT_DIR/runs.tsv"
        done
        base_state="$OUTPUT_DIR/runs/base-$label-r$repeat.q8-cache-state-before.csv"
        scalar_state="$OUTPUT_DIR/runs/scalar-$label-r$repeat.q8-cache-state-before.csv"
        validation_state="$OUTPUT_DIR/validation/base-$label.q8-cache-state.csv"
        cmp -s "$base_state" "$scalar_state" ||
            die "timed Q8 cache states differ for $label repeat $repeat"
        cmp -s "$base_state" "$validation_state" ||
            die "timed Q8 cache state differs from validation for $label repeat $repeat"
    done
done

current_phase=summarize
python3 - "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/validation.tsv" "$OUTPUT_DIR" \
    "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$REPEATS" \
    "$GPU_DEVICES" "$STAGE_SPLIT" <<'PY'
import csv
import hashlib
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

runs_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
ctx_start, ctx_max, step_mul, repeats = map(int, sys.argv[4:8])
gpu_ids = [int(value) for value in sys.argv[8].split(",")]
stage_split = int(sys.argv[9])

expected_frontiers = []
frontier = ctx_start
while True:
    expected_frontiers.append(frontier)
    if frontier >= ctx_max:
        break
    frontier = min(ctx_max, math.ceil(frontier * step_mul))

runs = []
with runs_path.open(encoding="utf-8", newline="") as handle:
    runs.extend(csv.DictReader(handle, delimiter="\t"))
if not runs:
    raise SystemExit("no benchmark runs recorded")

with validation_path.open(encoding="utf-8", newline="") as handle:
    validation_runs = list(csv.DictReader(handle, delimiter="\t"))
validation_index = {}
for run in validation_runs:
    key = (run["prompt"], run["variant"])
    if key in validation_index:
        raise SystemExit(f"duplicate validation record: {key}")
    validation_index[key] = run

run_index = {}
series = {}
for run in runs:
    key = (run["prompt"], int(run["repeat"]), run["variant"])
    if key in run_index:
        raise SystemExit(f"duplicate run record: {key}")
    if run["variant"] not in {"base", "scalar"}:
        raise SystemExit(f'unknown variant: {run["variant"]}')
    run_index[key] = run
    with open(run["csv"], encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f'empty CSV: {run["csv"]}')
    contexts = []
    seen_contexts = set()
    parsed = []
    previous = 0
    for row in rows:
        ctx = int(row["ctx_tokens"])
        if ctx in seen_contexts:
            raise SystemExit(f'duplicate context {ctx} in {run["csv"]}')
        seen_contexts.add(ctx)
        tps = float(row["prefill_tps"])
        tokens = int(row["prefill_tokens"])
        if not math.isfinite(tps) or tps <= 0:
            raise SystemExit(f'invalid prefill_tps in {run["csv"]}: {tps}')
        if tokens != ctx - previous:
            raise SystemExit(
                f'prefill token mismatch in {run["csv"]}: ctx={ctx} '
                f'prefill_tokens={tokens} expected={ctx - previous}'
            )
        contexts.append(ctx)
        parsed.append({"ctx": ctx, "tokens": tokens, "tps": tps})
        previous = ctx
    if contexts != expected_frontiers:
        raise SystemExit(
            f'frontier mismatch in {run["csv"]}: {contexts} '
            f'expected {expected_frontiers}'
        )
    series[key] = parsed

prompts = sorted({run["prompt"] for run in runs})
expected_repeats = set(range(1, repeats + 1))
for prompt in prompts:
    actual_repeats = {repeat for p, repeat, _ in run_index if p == prompt}
    if actual_repeats != expected_repeats:
        raise SystemExit(
            f'repeat set mismatch for {prompt}: {sorted(actual_repeats)} '
            f'expected {sorted(expected_repeats)}'
        )
    for repeat in expected_repeats:
        for variant in ("base", "scalar"):
            if (prompt, repeat, variant) not in run_index:
                raise SystemExit(
                    f'missing run: prompt={prompt} repeat={repeat} variant={variant}'
                )

def critical_layout_signature(path, expected_session_initializations):
    categories = Counter()
    kept = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip()
            category = None
            if line.startswith("ds4: GPU config:"): category = "gpu_config"
            elif line == "multi-GPU layout:": category = "layout_header"
            elif line.startswith("  GPU"): category = "placement_gpu"
            elif line.startswith("ds4: per-tier graph scratch reserved:"): category = "scratch"
            elif line.startswith("ds4: peer access matrix (validated):"): category = "peer_matrix"
            elif line.startswith("ds4: CUDA output TP"): category = "output_tp"
            elif line.startswith("ds4: CUDA routed MoE expert ownership"): category = "ownership"
            elif line.startswith("ds4: CUDA decode TP enabled:"): category = "decode_tp"
            elif line.startswith("ds4-bench: context buffers"): category = "context_buffers"
            elif line.startswith("ds4: CUDA tier ") and "selective weights:" in line:
                category = "selective_tier"
            if category is not None:
                categories[category] += 1
                kept.append(line)
    required = {
        "gpu_config": 1,
        "layout_header": 1,
        "placement_gpu": 4,
        "scratch": 1,
        "peer_matrix": 1,
        "output_tp": 1,
        "ownership": expected_session_initializations,
        "decode_tp": expected_session_initializations,
        "context_buffers": 1,
        "selective_tier": 4,
    }
    if dict(categories) != required:
        raise SystemExit(
            f"critical layout cardinality mismatch in {path}: "
            f"{dict(categories)} expected {required}"
        )
    return Counter(kept)

def q8_profile(path):
    ignored = {"sequence", "cache_bytes_after"}
    expected_identities = {
        ("attn_q_a", "q8_0"),
        ("attn_q_b", "q8_0"),
        ("attn_kv", "q8_0"),
        ("attention_output", "attn_output_a"),
        ("attention_output", "attn_output_b"),
        ("shared_gate", "q8_0"),
        ("shared_up", "q8_0"),
        ("shared_down", "q8_0"),
    }
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        all_fields = reader.fieldnames or []
        fields = [field for field in all_fields if field not in ignored]
        rows = list(reader)
    if not rows:
        raise SystemExit(f"empty Q8 cache audit: {path}")
    required_fields = {
        "sequence", "module", "label", "layer", "token_offset",
        "physical_device", "weight_offset", "weight_bytes", "in_dim",
        "out_dim", "fp16_bytes", "result", "reason", "cache_bytes_after",
    }
    if set(all_fields) != required_fields:
        raise SystemExit(f"unexpected Q8 cache audit fields in {path}: {all_fields}")
    unique_fields = (
        "module", "label", "layer", "physical_device", "weight_offset",
        "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
    )
    unique = {}
    layer_unique = Counter()
    layer_identities = defaultdict(set)
    result_counts = Counter()
    for row in rows:
        try:
            layer = int(row["layer"])
            device = int(row["physical_device"])
            cache_after = int(row["cache_bytes_after"])
        except ValueError as exc:
            raise SystemExit(f"invalid Q8 audit numeric field in {path}: {exc}")
        if not 0 <= layer < 43:
            raise SystemExit(f"invalid Q8 layer {layer} in {path}")
        expected_device = gpu_ids[0] if layer < stage_split else gpu_ids[1]
        if device != expected_device:
            raise SystemExit(
                f"Q8 layer/device coverage mismatch in {path}: "
                f"layer={layer} device={device} expected={expected_device}"
            )
        key = tuple(row[field] for field in unique_fields)
        if key not in unique:
            unique[key] = row
            layer_unique[layer] += 1
            layer_identities[layer].add((row["module"], row["label"]))
        result_counts[row["result"]] += 1
    if set(layer_unique) != set(range(43)) or any(
            layer_unique[layer] != 8 for layer in range(43)):
        raise SystemExit(
            f"Q8 unique coverage mismatch in {path}: {dict(layer_unique)}"
        )
    if len(unique) != 344:
        raise SystemExit(f"Q8 unique weight count is {len(unique)}, expected 344 in {path}")
    for layer in range(43):
        if layer_identities[layer] != expected_identities:
            raise SystemExit(
                f"Q8 tensor identities mismatch in {path}: layer={layer} "
                f"actual={sorted(layer_identities[layer])} "
                f"expected={sorted(expected_identities)}"
            )
    total_fp16_bytes = sum(int(row["fp16_bytes"]) for row in unique.values())
    covered_keys = {
        tuple(row[field] for field in unique_fields)
        for row in rows if row["result"].startswith("f16_")
    }
    covered_fp16_bytes = sum(int(unique[key]["fp16_bytes"]) for key in covered_keys)
    return {
        "signature": Counter(tuple(row[field] for field in fields) for row in rows),
        "records": len(rows),
        "unique_weights": len(unique),
        "result_counts": result_counts,
        "max_cache_bytes": max(int(row["cache_bytes_after"]) for row in rows),
        "covered_fp16_bytes": covered_fp16_bytes,
        "total_fp16_bytes": total_fp16_bytes,
    }

def q8_cache_state_profile(path):
    required_fields = {
        "physical_device", "weight_offset", "weight_bytes",
        "in_dim", "out_dim", "fp16_bytes",
    }
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if set(fields) != required_fields:
        raise SystemExit(
            f"unexpected Q8 cache-state fields in {path}: {fields}"
        )
    if not rows:
        raise SystemExit(f"empty Q8 cache-state snapshot: {path}")
    signature = Counter()
    total_fp16_bytes = 0
    device_entries = Counter()
    device_bytes = Counter()
    for row in rows:
        try:
            values = tuple(int(row[field]) for field in fields)
            device = int(row["physical_device"])
            fp16_bytes = int(row["fp16_bytes"])
        except ValueError as exc:
            raise SystemExit(
                f"invalid Q8 cache-state numeric field in {path}: {exc}"
            )
        if device not in gpu_ids:
            raise SystemExit(
                f"unexpected Q8 cache-state device {device} in {path}"
            )
        if fp16_bytes <= 0:
            raise SystemExit(
                f"invalid Q8 cache-state byte count {fp16_bytes} in {path}"
            )
        signature[values] += 1
        total_fp16_bytes += fp16_bytes
        device_entries[device] += 1
        device_bytes[device] += fp16_bytes
    duplicates = [key for key, count in signature.items() if count != 1]
    if duplicates:
        raise SystemExit(f"duplicate Q8 cache-state ranges in {path}: {duplicates}")
    return {
        "signature": signature,
        "entries": len(rows),
        "total_fp16_bytes": total_fp16_bytes,
        "device_entries": device_entries,
        "device_bytes": device_bytes,
    }

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

pair_rows = []
validation_rows = []
logit_rows = []
validation_failures = []
validation_q8_states = {}
expected_validation_keys = {
    (prompt, variant) for prompt in prompts for variant in ("base", "scalar")
}
if set(validation_index) != expected_validation_keys:
    raise SystemExit(
        f"validation record set mismatch: {sorted(validation_index)} "
        f"expected {sorted(expected_validation_keys)}"
    )
for prompt in prompts:
    base_validation = validation_index[(prompt, "base")]
    scalar_validation = validation_index[(prompt, "scalar")]
    validation_layout_match = (
        critical_layout_signature(base_validation["log"], 1)
        == critical_layout_signature(scalar_validation["log"], 1)
    )
    base_q8 = q8_profile(base_validation["q8_audit"])
    scalar_q8 = q8_profile(scalar_validation["q8_audit"])
    base_q8_state = q8_cache_state_profile(base_validation["q8_cache_state"])
    scalar_q8_state = q8_cache_state_profile(scalar_validation["q8_cache_state"])
    q8_state_match = (
        base_q8_state["signature"] == scalar_q8_state["signature"]
        and base_q8_state["total_fp16_bytes"]
            == scalar_q8_state["total_fp16_bytes"]
    )
    q8_match = (
        base_q8["signature"] == scalar_q8["signature"]
        and base_q8["max_cache_bytes"] == scalar_q8["max_cache_bytes"]
        and base_q8["covered_fp16_bytes"] == scalar_q8["covered_fp16_bytes"]
        and base_q8["total_fp16_bytes"] == scalar_q8["total_fp16_bytes"]
        and q8_state_match
    )
    validation_q8_states[prompt] = base_q8_state
    logits_match = True
    base_logits = Path(base_validation["logits_dir"])
    scalar_logits = Path(scalar_validation["logits_dir"])
    expected_json = {f"frontier_{ctx:06d}.logits.json" for ctx in expected_frontiers}
    expected_raw = {f"frontier_{ctx:06d}.logits.f32" for ctx in expected_frontiers}
    for directory, variant in ((base_logits, "base"), (scalar_logits, "scalar")):
        actual_json = {path.name for path in directory.glob("*.logits.json")}
        actual_raw = {path.name for path in directory.glob("*.logits.f32")}
        if actual_json != expected_json or actual_raw != expected_raw:
            logits_match = False
            validation_failures.append(
                f"logit frontier set mismatch for {prompt} {variant}"
            )
    for ctx in expected_frontiers:
        json_name = f"frontier_{ctx:06d}.logits.json"
        raw_name = f"frontier_{ctx:06d}.logits.f32"
        base_json = base_logits / json_name
        scalar_json = scalar_logits / json_name
        base_raw = base_logits / raw_name
        scalar_raw = scalar_logits / raw_name
        base_raw_hash = file_sha256(base_raw) if base_raw.is_file() else "missing"
        scalar_raw_hash = file_sha256(scalar_raw) if scalar_raw.is_file() else "missing"
        base_json_hash = file_sha256(base_json) if base_json.is_file() else "missing"
        scalar_json_hash = file_sha256(scalar_json) if scalar_json.is_file() else "missing"
        raw_exact = base_raw_hash == scalar_raw_hash and base_raw_hash != "missing"
        json_exact = base_json_hash == scalar_json_hash and base_json_hash != "missing"
        logits_match &= raw_exact and json_exact
        logit_rows.append({
            "prompt": prompt,
            "ctx_tokens": ctx,
            "base_raw_f32_sha256": base_raw_hash,
            "scalar_raw_f32_sha256": scalar_raw_hash,
            "raw_f32_bit_exact": int(raw_exact),
            "base_json_sha256": base_json_hash,
            "scalar_json_sha256": scalar_json_hash,
            "json_exact": int(json_exact),
        })
    validation_rows.append({
        "prompt": prompt,
        "validation_layout_match": int(validation_layout_match),
        "q8_cache_fingerprint_match": int(q8_match),
        "q8_records": base_q8["records"],
        "q8_unique_weights": base_q8["unique_weights"],
        "q8_f16_covered_bytes": base_q8["covered_fp16_bytes"],
        "q8_total_candidate_bytes": base_q8["total_fp16_bytes"],
        "q8_max_cache_bytes": base_q8["max_cache_bytes"],
        "q8_final_cache_state_match": int(q8_state_match),
        "q8_final_cache_entries": base_q8_state["entries"],
        "q8_final_cache_bytes": base_q8_state["total_fp16_bytes"],
        "all_frontier_raw_f32_bit_exact": int(logits_match),
    })
    if not validation_layout_match:
        validation_failures.append(f"validation layout mismatch for {prompt}")
    if not q8_match:
        validation_failures.append(f"Q8 cache fingerprint mismatch for {prompt}")
    if not logits_match:
        validation_failures.append(f"full-model raw FP32 logits mismatch for {prompt}")
    for repeat in sorted(expected_repeats):
        base_run = run_index[(prompt, repeat, "base")]
        scalar_run = run_index[(prompt, repeat, "scalar")]
        layout_match = (
            critical_layout_signature(base_run["log"], 2)
            == critical_layout_signature(scalar_run["log"], 2)
        )
        base_timed_q8_before = q8_cache_state_profile(
            base_run["q8_cache_state_before"]
        )
        base_timed_q8_after = q8_cache_state_profile(
            base_run["q8_cache_state_after"]
        )
        scalar_timed_q8_before = q8_cache_state_profile(
            scalar_run["q8_cache_state_before"]
        )
        scalar_timed_q8_after = q8_cache_state_profile(
            scalar_run["q8_cache_state_after"]
        )
        timed_q8_match = (
            base_timed_q8_before["signature"]
                == base_timed_q8_after["signature"]
            and base_timed_q8_before["signature"]
                == scalar_timed_q8_before["signature"]
            and scalar_timed_q8_before["signature"]
                == scalar_timed_q8_after["signature"]
            and base_timed_q8_before["signature"]
                == validation_q8_states[prompt]["signature"]
            and base_timed_q8_before["total_fp16_bytes"]
                == base_timed_q8_after["total_fp16_bytes"]
            and base_timed_q8_before["total_fp16_bytes"]
                == scalar_timed_q8_before["total_fp16_bytes"]
            and scalar_timed_q8_before["total_fp16_bytes"]
                == scalar_timed_q8_after["total_fp16_bytes"]
            and base_timed_q8_before["total_fp16_bytes"]
                == validation_q8_states[prompt]["total_fp16_bytes"]
        )
        pair_rows.append({
            "prompt": prompt,
            "repeat": repeat,
            "clean_timed_layout_match": int(layout_match),
            "clean_timed_q8_cache_state_match": int(timed_q8_match),
            "q8_final_cache_entries": base_timed_q8_after["entries"],
            "q8_final_cache_bytes": base_timed_q8_after["total_fp16_bytes"],
        })
        if not layout_match:
            validation_failures.append(
                f"clean timed layout mismatch for {prompt} repeat {repeat}"
            )
        if not timed_q8_match:
            validation_failures.append(
                f"clean timed Q8 cache-state mismatch for {prompt} repeat {repeat}"
            )

with (output_dir / "pair-validation.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    fields = list(pair_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(pair_rows)
with (output_dir / "validation-summary.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    fields = list(validation_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(validation_rows)
with (output_dir / "logit-comparison.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    fields = list(logit_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(logit_rows)
if validation_failures:
    raise SystemExit(
        "full-model A/B validation failed:\n  "
        + "\n  ".join(sorted(set(validation_failures)))
    )

paired = []
for prompt in prompts:
    for repeat in sorted(expected_repeats):
        base_rows = series[(prompt, repeat, "base")]
        scalar_rows = series[(prompt, repeat, "scalar")]
        for base, scalar in zip(base_rows, scalar_rows):
            if base["ctx"] != scalar["ctx"] or base["tokens"] != scalar["tokens"]:
                raise SystemExit(
                    f"paired frontier mismatch for {prompt} repeat {repeat}"
                )
            paired.append({
                "prompt": prompt,
                "repeat": repeat,
                "ctx_tokens": base["ctx"],
                "prefill_tokens": base["tokens"],
                "base_tps": base["tps"],
                "scalar_tps": scalar["tps"],
                "speedup_x": scalar["tps"] / base["tps"],
            })

with (output_dir / "paired-samples.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    fields = list(paired[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(paired)

groups = defaultdict(list)
for row in paired:
    groups[(row["prompt"], row["ctx_tokens"])].append(row)

frontiers = []
for (prompt, ctx), rows in sorted(groups.items()):
    if len(rows) != repeats or {row["repeat"] for row in rows} != expected_repeats:
        raise SystemExit(
            f"paired sample cardinality mismatch for {prompt} ctx={ctx}"
        )
    ratios = [row["speedup_x"] for row in rows]
    base = [row["base_tps"] for row in rows]
    scalar = [row["scalar_tps"] for row in rows]
    ratio_median = statistics.median(ratios)
    frontiers.append({
        "prompt": prompt,
        "ctx_tokens": ctx,
        "paired_samples": len(rows),
        "base_median_tps": statistics.median(base),
        "scalar_median_tps": statistics.median(scalar),
        "median_paired_speedup_x": ratio_median,
        "median_paired_change_pct": (ratio_median - 1.0) * 100.0,
        "paired_speedup_mad": statistics.median(
            abs(value - ratio_median) for value in ratios
        ),
        "base_tps_cv_pct": (
            statistics.pstdev(base) / statistics.mean(base) * 100.0
            if len(base) > 1 else 0.0
        ),
        "scalar_tps_cv_pct": (
            statistics.pstdev(scalar) / statistics.mean(scalar) * 100.0
            if len(scalar) > 1 else 0.0
        ),
        "min_paired_speedup_x": min(ratios),
        "max_paired_speedup_x": max(ratios),
        "wins": sum(ratio > 1.0 for ratio in ratios),
    })

fields = list(frontiers[0])
with (output_dir / "frontier-summary.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(frontiers)

sweeps = []
for prompt in prompts:
    for repeat in sorted(expected_repeats):
        by_variant = {}
        for variant in ("base", "scalar"):
            rows = series[(prompt, repeat, variant)]
            tokens = sum(row["tokens"] for row in rows)
            elapsed = sum(row["tokens"] / row["tps"] for row in rows)
            by_variant[variant] = tokens / elapsed
        sweeps.append({
            "prompt": prompt,
            "repeat": repeat,
            "base_sweep_tps": by_variant["base"],
            "scalar_sweep_tps": by_variant["scalar"],
            "speedup_x": by_variant["scalar"] / by_variant["base"],
            "change_pct": (by_variant["scalar"] / by_variant["base"] - 1.0) * 100.0,
        })
with (output_dir / "sweep-summary.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    fields = list(sweeps[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sweeps)

prompt_summaries = []
for prompt in prompts:
    prompt_sweeps = [row for row in sweeps if row["prompt"] == prompt]
    ratios = [row["speedup_x"] for row in prompt_sweeps]
    median = statistics.median(ratios)
    prompt_summaries.append({
        "prompt": prompt,
        "sweep_pairs": len(ratios),
        "wins": sum(value > 1.0 for value in ratios),
        "median_speedup_x": median,
        "median_change_pct": (median - 1.0) * 100.0,
        "speedup_mad": statistics.median(abs(value - median) for value in ratios),
        "minimum_speedup_x": min(ratios),
        "maximum_speedup_x": max(ratios),
    })
with (output_dir / "prompt-summary.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    fields = list(prompt_summaries[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(prompt_summaries)

sweep_ratios = [row["speedup_x"] for row in sweeps]
sweep_median = statistics.median(sweep_ratios)
sweep_mad = statistics.median(abs(value - sweep_median) for value in sweep_ratios)
sweep_geomean = math.exp(
    sum(math.log(value) for value in sweep_ratios) / len(sweep_ratios)
)
frontier_medians = [row["median_paired_speedup_x"] for row in frontiers]
lines = [
    f"measurement_grade={'replicated' if repeats >= 6 else 'pilot'}",
    f"audited_validation_pairs={len(validation_rows)}",
    f"clean_timed_ab_pairs={len(pair_rows)}",
    f"validation_layout_matches={sum(row['validation_layout_match'] for row in validation_rows)}",
    f"clean_timed_layout_matches={sum(row['clean_timed_layout_match'] for row in pair_rows)}",
    f"clean_timed_q8_cache_state_matches={sum(row['clean_timed_q8_cache_state_match'] for row in pair_rows)}",
    f"q8_cache_fingerprint_matches={sum(row['q8_cache_fingerprint_match'] for row in validation_rows)}",
    f"raw_f32_bit_exact_logit_pairs={sum(row['all_frontier_raw_f32_bit_exact'] for row in validation_rows)}",
    f"paired_measurements={len(paired)}",
    f"frontiers={len(frontiers)}",
    f"sweep_pairs={len(sweeps)}",
    f"sweep_wins={sum(value > 1.0 for value in sweep_ratios)}",
    f"median_sweep_speedup_x={sweep_median:.9f}",
    f"median_sweep_change_pct={(sweep_median - 1.0) * 100.0:.6f}",
    f"sweep_speedup_mad={sweep_mad:.9f}",
    f"geomean_sweep_speedup_x={sweep_geomean:.9f}",
    f"minimum_sweep_speedup_x={min(sweep_ratios):.9f}",
    f"maximum_sweep_speedup_x={max(sweep_ratios):.9f}",
    f"minimum_frontier_median_speedup_x={min(frontier_medians):.9f}",
    "",
    "prompt,ctx_tokens,base_median_tps,scalar_median_tps,median_speedup_x,change_pct,wins/samples",
]
for row in frontiers:
    lines.append(
        f'{row["prompt"]},{row["ctx_tokens"]},'
        f'{row["base_median_tps"]:.3f},{row["scalar_median_tps"]:.3f},'
        f'{row["median_paired_speedup_x"]:.6f},'
        f'{row["median_paired_change_pct"]:.3f},'
        f'{row["wins"]}/{row["paired_samples"]}'
    )
text = "\n".join(lines) + "\n"
with (output_dir / "overall-summary.txt").open("w", encoding="utf-8") as handle:
    handle.write(text)
print(text, end="")
PY

current_phase=final-validation
for required in "$OUTPUT_DIR/frontier-summary.csv" \
                "$OUTPUT_DIR/overall-summary.txt" \
                "$OUTPUT_DIR/pair-validation.csv" \
                "$OUTPUT_DIR/validation-summary.csv" \
                "$OUTPUT_DIR/logit-comparison.csv" \
                "$OUTPUT_DIR/paired-samples.csv" \
                "$OUTPUT_DIR/prompt-summary.csv" \
                "$OUTPUT_DIR/sweep-summary.csv" \
                "$OUTPUT_DIR/runs.tsv" \
                "$OUTPUT_DIR/validation.tsv"; do
    [[ -s $required ]] || die "missing final evidence: $required"
done
printf 'full_model_scalar_ab_capture=valid\nmeasurement_grade=%s\n' \
    "$([[ $REPEATS -ge 6 ]] && printf replicated || printf pilot)" \
    >"$OUTPUT_DIR/capture-status.txt"
printf 'SM75 full-model scalar A/B complete: %s\n' "$OUTPUT_DIR"
