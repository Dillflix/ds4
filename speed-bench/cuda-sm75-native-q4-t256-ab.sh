#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the focused standard/native-Q4 x all-partner-T256 integration check.

Required environment:
  MODEL=/absolute/path/to/standard-full-Q4.gguf
  NATIVE_MODEL=/absolute/path/to/tagged-SM75-native-full-Q4.gguf

Optional environment:
  PROMPT=/absolute/path/to/prompt.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=32768
  STEP_MUL=2
  PREFILL_CHUNK=2048
  REPEATS=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  NATIVE_Q4_T256_AB_DIR=/absolute/output/directory

This runner does not hash, create, or modify either model. Both arms force the
same all-partner T256 policy, isolating native-Q4 integration and throughput.
It does not replace the separate official T256 quality decision gate.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute standard full-Q4 GGUF path}"
: "${NATIVE_MODEL:?set NATIVE_MODEL to the absolute tagged SM75-native full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute standard full-Q4 GGUF"
[[ $NATIVE_MODEL == /* && -f $NATIVE_MODEL ]] ||
    die "NATIVE_MODEL must name an existing absolute tagged SM75-native full-Q4 GGUF"
[[ $MODEL == *.gguf && $NATIVE_MODEL == *.gguf ]] ||
    die "MODEL and NATIVE_MODEL must end in .gguf"
[[ $MODEL != "$NATIVE_MODEL" ]] || die "MODEL and NATIVE_MODEL must differ"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-32768}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${NATIVE_Q4_T256_AB_DIR:-$repo_dir/sm75-native-q4-t256-ab-$stamp}

[[ $PROMPT == /* && -f $PROMPT ]] ||
    die "PROMPT must name an existing absolute path"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "this production A/B requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "REPEATS:$REPEATS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START == 2048 && CTX_MAX == 32768 && STEP_MUL == 2 &&
   PREFILL_CHUNK == 2048 )) ||
    die "the fixed production A/B requires CTX_START=2048 CTX_MAX=32768 STEP_MUL=2 PREFILL_CHUNK=2048"
(( REPEATS >= 1 )) || die "REPEATS must be at least one"
frontier=$CTX_START
have_16k=0
have_32k=0
while (( frontier <= CTX_MAX )); do
    (( frontier == 16384 )) && have_16k=1
    (( frontier == 32768 )) && have_32k=1
    (( frontier > CTX_MAX / STEP_MUL )) && break
    frontier=$((frontier * STEP_MUL))
done
(( have_16k == 1 && have_32k == 1 )) ||
    die "the configured multiplicative sweep must include both 16384 and 32768"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done

for tool in awk cat cmp date env find git grep make mkdir mv nproc nvidia-smi \
            python3 rm sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ -f speed-bench/q8_partner_audit.py ]] ||
    die "speed-bench/q8_partner_audit.py is missing"
[[ -f speed-bench/summarize-sm75-native-q4-t256-ab.py ]] ||
    die "speed-bench/summarize-sm75-native-q4-t256-ab.py is missing"

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{logits,provenance,runs,telemetry,validation}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
telemetry_pid=
stop_telemetry() {
    if [[ -n ${telemetry_pid:-} ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=
    fi
}

finish() {
    status=$?
    trap - EXIT INT TERM HUP
    stop_telemetry
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv "$partial" "$archive"; then
            if [[ $status == 0 ]]; then
                printf 'Archive to return: %s\n' "$archive"
            else
                printf 'Failed-run diagnostic archive (phase=%s, exit=%s): %s\n' \
                    "$phase" "$status" "$archive" >&2
            fi
        else
            printf 'error: failed to create nonempty archive: %s\n' "$archive" >&2
            rm -f -- "$partial"
            [[ $status != 0 ]] || status=1
            printf 'state=failed\nexit_status=%s\nlast_phase=%s\n' \
                "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted-INT; exit 130' INT
trap 'phase=interrupted-TERM; exit 143' TERM
trap 'phase=interrupted-HUP; exit 129' HUP

IFS=',' read -r -a gpu_device_ids <<<"$GPU_DEVICES"
(( ${#gpu_device_ids[@]} == 4 )) || die "GPU_DEVICES must contain four devices"
declare -A seen_gpu=()
for device in "${gpu_device_ids[@]}"; do
    [[ $device =~ ^[0-9]+$ ]] || die "invalid GPU device index: $device"
    [[ -z ${seen_gpu[$device]+x} ]] || die "duplicate GPU device index: $device"
    seen_gpu[$device]=1
    capability=$(nvidia-smi -i "$device" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $capability == 7.5 ]] ||
        die "physical GPU $device is compute capability ${capability:-unknown}, not SM75"
done
partner_device_0=${gpu_device_ids[2]}
partner_device_1=${gpu_device_ids[3]}

phase=topology
topology_file="$OUTPUT_DIR/provenance/nvidia-topology.txt"
nvidia-smi topo -m >"$topology_file" || die "failed to query NVIDIA topology"
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
    ' "$topology_file"
}
{
    printf 'home_tier\thome_device\tpartner_tier\tpartner_device\thome_to_partner\tpartner_to_home\n'
    for pair in 0 1; do
        home_tier=$pair
        partner_tier=$((pair + 2))
        home_device=${gpu_device_ids[$home_tier]}
        partner_device=${gpu_device_ids[$partner_tier]}
        forward=$(topology_link "$home_device" "$partner_device")
        reverse=$(topology_link "$partner_device" "$home_device")
        [[ $forward =~ ^NV[0-9]+$ || $reverse =~ ^NV[0-9]+$ ]] ||
            die "logical pair $home_tier<->$partner_tier (physical $home_device<->$partner_device) is not NVLink: ${forward:-missing}/${reverse:-missing}"
        [[ -z $forward || $forward =~ ^NV[0-9]+$ ]] ||
            die "inconsistent NVLink topology for $home_device<->$partner_device: $forward/${reverse:-missing}"
        [[ -z $reverse || $reverse =~ ^NV[0-9]+$ ]] ||
            die "inconsistent NVLink topology for $home_device<->$partner_device: ${forward:-missing}/$reverse"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$home_tier" "$home_device" "$partner_tier" "$partner_device" \
            "${forward:-unreported}" "${reverse:-unreported}"
    done
} >"$OUTPUT_DIR/pair-topology.tsv"

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
common_env=("${clean[@]}"
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_ROUTED_QUANT_AUDIT=1)

run_model=
run_kind=
run_policy=
variant_extra=()
configure_variant() {
    case "$1" in
        standard-all-partner)
            run_model=$MODEL
            run_kind=standard
            run_policy=all-partner
            ;;
        native-all-partner)
            run_model=$NATIVE_MODEL
            run_kind=native
            run_policy=all-partner
            ;;
        *) die "internal unsupported variant: $1" ;;
    esac
    variant_extra=(
        DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1
        DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256
        DS4_CUDA_Q8_F16_PARTNER_LAYERS=0-42
        DS4_CUDA_Q8_PARTNER_ARITHMETIC=f16
        DS4_CUDA_Q8_T256_PLACEMENT=all-partner
    )
}

native_marker='ds4: SM75 native routed-Q4 layout enabled (packed A/W, planner=cost, gate=tile8, down=full-stage)'
validate_run_evidence() {
    local variant=$1
    local kind=$2
    local policy=$3
    local log=$4
    local audit=$5
    local bindings=$6
    local allocations=$7
    local before=$8
    local after=$9
    local logits=${10}
    local count first_logit audit_policy

    for evidence in "$log" "$audit" "$bindings" "$allocations" \
                    "$before" "$after"; do
        [[ -s $evidence ]] || die "$variant omitted required evidence: $evidence"
    done
    cmp -s "$before" "$after" ||
        die "$variant changed the F16 cache during timed frontiers"
    first_logit=$(find "$logits" -maxdepth 1 -type f -name '*.f32' -size +0c \
        -print -quit)
    [[ -n $first_logit && -s $first_logit ]] ||
        die "$variant produced no full-logit payload"

    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant did not use split $STAGE_SPLIT/$((43-STAGE_SPLIT))"
    grep -Fq "4 devices [$GPU_DEVICES] requested" "$log" ||
        die "$variant did not use GPU order $GPU_DEVICES"
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || die "$variant lacks direct route $route"
    done
    grep -Fq 'q8 fp16 benefit plan registered' "$log" ||
        die "$variant did not register the benefit planner"
    grep -Fq 'q8 fp16 benefit plan materialized' "$log" ||
        die "$variant did not materialize the benefit plan"

    count=$(grep -Ec '^ds4: routed-quant-audit layer=[0-9]+ gate=q4_k up=q4_k down=q4_k$' \
        "$log" || true)
    [[ $count == 43 ]] ||
        die "$variant is not an exact 43-layer full-Q4 routed model (found $count)"
    if [[ $kind == native ]]; then
        grep -Fqx "$native_marker" "$log" ||
            die "$variant did not dispatch the exact production native-Q4 path"
    elif grep -Fq 'ds4: SM75 native routed-Q4 layout enabled' "$log"; then
        die "$variant unexpectedly dispatched the tagged native-Q4 path"
    fi

    for marker in \
            'partner-classes=t256' \
            'partner-layers=0-42' \
            't256-placement=all-partner' \
            'T256-output_b=43/43' \
            'partner=43 partner-arithmetic=f16' \
            'CUDA q8 fp16 partner summary:'; do
        grep -Fq "$marker" "$log" ||
            die "$variant lacks all-partner T256 marker: $marker"
    done
    awk -F, '
        NR > 1 && $6 == 8192 && $7 == 4096 && $12 ~ /attn_output_b/ {
            if ($3 == 1 && $11 == "f16" && $13 > 0 && $14 > 0 && $15 == 1)
                partner++
            else local++
            next
        }
        NR > 1 && $3 == 1 {bad_partner=1}
        END {exit !(local == 0 && partner == 43 && !bad_partner)}
    ' "$bindings" ||
        die "$variant does not have exactly 43 active partner T256 bindings with no local T256 or other partner class"
    awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; i++) column[$i] = i
            next
        }
        {
            if ($(column["usage_tracking"]) != 1 ||
                $(column["dead_bytes"]) != 0) dead++
            if ($(column["storage_kind"]) == "f16" &&
                $(column["in_dim"]) == 8192 &&
                $(column["out_dim"]) == 4096) {
                t256++
                if ($(column["logical_aliases"]) != 1 ||
                    $(column["live_aliases"]) != 1 ||
                    $(column["used_calls"]) <= 0) bad_t256++
            }
        }
        END {exit !(dead == 0 && t256 == 43 && bad_t256 == 0)}
    ' "$allocations" ||
        die "$variant does not have 43 live physical T256 weights and zero dead expanded-weight allocations"
    audit_policy=default
    python3 speed-bench/q8_partner_audit.py \
        "$audit_policy" "$partner_device_0" "$partner_device_1" "$audit" \
        >"${audit%.csv}.validation.txt" ||
        die "$variant has impure or missing partner-class audit evidence"
}

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)"
    printf 'standard_model=%s\nstandard_model_bytes=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'native_model=%s\nnative_model_bytes=%s\n' \
        "$NATIVE_MODEL" "$(stat -c %s "$NATIVE_MODEL")"
    printf 'prompt=%s\ngpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK"
    printf 'repeats=%s\nvariants=standard-all-partner,native-all-partner\n' \
        "$REPEATS"
    printf 't256_policy=forced-all-partner\nt256_layers=0-42\nmodel_hashing=disabled\nofficial_quality_suite=not-run-separate-gate\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
        --format=csv
    printf '\n[topology]\n'
    cat "$topology_file"
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print $0}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/validation/build.log"
else
    phase=build-staleness-check
    make -q ds4-bench tests/cuda_long_context_smoke \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=exact-tests
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/validation/planner-unit.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/validation/planner-unit.log" >&2 || true
        die "planner unit test failed"
    }
"${clean[@]}" ./tests/test_gpu_xdev \
    >"$OUTPUT_DIR/validation/gpu-exactness.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/validation/gpu-exactness.log" >&2 || true
        die "multi-GPU exactness test failed"
    }
grep -Fq 'q8 partner projection exactness OK (3 classes)' \
    "$OUTPUT_DIR/validation/gpu-exactness.log" ||
    die "GPU exactness did not exercise all partner projection classes"
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/validation/cuda-exactness.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/validation/cuda-exactness.log" >&2 || true
        die "CUDA exact-output suite failed"
    }
for marker in \
        'tagged SM75 native Q4 cost-planner default exact' \
        'tagged SM75 native Q4 decode exact' \
        'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exactness.log" ||
        die "exact-output marker missing: $marker"
done

printf 'repeat\tslot\tvariant\tmodel\tpolicy\tcsv\tlog\taudit\tbindings\tallocations\tcache_before\tcache_after\tlogits\n' \
    >"$OUTPUT_DIR/runs.tsv"

phase=focused-two-arm-integration
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    order=(standard-all-partner native-all-partner)
    slot=0
    for variant in "${order[@]}"; do
        slot=$((slot + 1))
        configure_variant "$variant"
        stem="$variant-r$repeat"
        csv="$OUTPUT_DIR/runs/$stem.csv"
        log="$OUTPUT_DIR/runs/$stem.log"
        audit="$OUTPUT_DIR/runs/$stem.q8-audit.csv"
        bindings="$OUTPUT_DIR/runs/$stem.bindings.csv"
        allocations="$OUTPUT_DIR/runs/$stem.allocations.csv"
        before="$OUTPUT_DIR/runs/$stem.cache-before.csv"
        after="$OUTPUT_DIR/runs/$stem.cache-after.csv"
        logits="$OUTPUT_DIR/logits/$stem"
        telemetry_before="$OUTPUT_DIR/telemetry/$stem.before.csv"
        telemetry_after="$OUTPUT_DIR/telemetry/$stem.after.csv"
        telemetry_samples="$OUTPUT_DIR/telemetry/$stem.samples.csv"
        mkdir -p "$logits"

        nvidia-smi --query-gpu=index,memory.used,memory.free \
            --format=csv >"$telemetry_before"
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,\
power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory \
            --format=csv,noheader,nounits -lms 200 >"$telemetry_samples" &
        telemetry_pid=$!
        printf 'Benchmarking %s repeat=%d/%d slot=%d/2...\n' \
            "$variant" "$repeat" "$REPEATS" "$slot"
        run_status=0
        "${common_env[@]}" "${variant_extra[@]}" \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$audit" \
            "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$before" \
            "DS4_CUDA_Q8_CACHE_STATE_CSV=$after" \
            "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
            "DS4_CUDA_Q8_ALLOCATION_STATE_CSV=$allocations" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$run_model" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL" \
                --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
                --csv "$csv" --dump-frontier-logits-dir "$logits" \
                >"$log" 2>&1 || run_status=$?
        stop_telemetry
        nvidia-smi --query-gpu=index,memory.used,memory.free \
            --format=csv >"$telemetry_after"
        if (( run_status != 0 )); then
            tail -n 200 "$log" >&2 || true
            die "$stem failed"
        fi
        [[ -s $telemetry_samples ]] || die "$stem produced no 200ms telemetry"
        [[ -s $telemetry_before && -s $telemetry_after ]] ||
            die "$stem omitted before/after telemetry"
        [[ -s $csv ]] || die "$stem produced no benchmark CSV"
        validate_run_evidence "$variant" "$run_kind" "$run_policy" \
            "$log" "$audit" "$bindings" "$allocations" "$before" \
            "$after" "$logits"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$slot" "$variant" "$run_model" "$run_policy" \
            "$csv" "$log" "$audit" "$bindings" "$allocations" "$before" \
            "$after" "$logits" >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=summarize
python3 speed-bench/summarize-sm75-native-q4-t256-ab.py \
    "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/summary-stdout.txt"

phase=complete
printf 'SM75 native-Q4 x all-partner-T256 integration check complete: %s\n' "$OUTPUT_DIR"
