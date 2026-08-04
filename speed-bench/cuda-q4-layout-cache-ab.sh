#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a controlled full-Q4 placement and Q8-to-F16 cuBLAS-cache comparison.

Required environment:
  MODEL=/absolute/path/to/full-Q4.gguf

Optional environment:
  PROMPT=/absolute/path/prompt.txt  # default: speed-bench/promessi_sposi.txt
  GPU_VRAM=auto
  CTX_START=2048
  CTX_MAX=8192
  STEP_MUL=2
  PREFILL_CHUNK=2048
  REPEATS=3                       # must be a multiple of three
  CUDA_ARCH=sm_75
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q4_LAYOUT_CACHE_DIR=...         # new output directory

The three fixed variants are:
  baseline-22x21  split 22/21, devices 0,2,1,3
  split-21x22     split 21/22, devices 0,2,1,3
  swap-22x21      split 22/21, devices 0,3,1,2

Every process performs an untimed CTX_START warm-up, snapshots the resulting
Q8-to-F16 cache, starts a fresh session, and then measures the requested
frontiers. The cache must remain byte-identical throughout timing. Three
rotated repeats put every variant in every process-order position once.
There is no model hash and no Nsight capture.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* ]] || die "MODEL must be an absolute path"
[[ -f $MODEL ]] || die "model not found: $MODEL"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_VRAM=${GPU_VRAM:-auto}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-8192}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-3}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q4_LAYOUT_CACHE_DIR:-$repo_dir/q4-layout-cache-ab-$run_stamp}
while [[ $OUTPUT_DIR != / && $OUTPUT_DIR == */ ]]; do
    OUTPUT_DIR=${OUTPUT_DIR%/}
done

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "REPEATS:$REPEATS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START && STEP_MUL >= 2 &&
   PREFILL_CHUNK > 0 && REPEATS > 0 )) || die "invalid benchmark shape"
(( REPEATS % 3 == 0 )) || die "REPEATS must be a multiple of three"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
for tool in awk basename cat cmp date dirname env git grep make mkdir mv nproc \
            nvidia-smi python3 sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

default_lock=/tmp/ds4.lock
if [[ -e $default_lock ]]; then
    [[ -f $default_lock && -w $default_lock ]] ||
        die "$default_lock exists but is not a writable regular file"
else
    [[ -d /tmp && -w /tmp ]] ||
        die "/tmp is not writable; DS4 cannot create $default_lock"
fi

for gpu in 0 1 2 3; do
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
for pair in 0:1 2:3; do
    from=${pair%%:*}; to=${pair#*:}; link=$(topology_link "$from" "$to")
    [[ $link =~ ^NV[0-9]+$ ]] || die "GPU $from<->$to is not an NVLink pair ($link)"
done

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
[[ ! -e $OUTPUT_DIR.tar.gz ]] || die "archive path already exists: $OUTPUT_DIR.tar.gz"
mkdir -p "$OUTPUT_DIR"/{runs,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    local archive="$OUTPUT_DIR.tar.gz"
    local partial="$archive.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'error: could not archive %s\n' "$OUTPUT_DIR" >&2
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'current_phase=interrupted-INT; exit 130' INT
trap 'current_phase=interrupted-TERM; exit 143' TERM
trap 'current_phase=interrupted-HUP; exit 129' HUP

mapfile -t inherited_ds4_envs < <(
    env | awk -F= '$1 ~ /^DS4_/ { print $1 }' | sort -u
)
clean_prefix=(env)
for name in "${inherited_ds4_envs[@]}"; do clean_prefix+=(-u "$name"); done

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
    for marker in 'sm75 q4 gate tile8 scalar exact' \
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
        die "SKIP_BUILD=1 found stale targets; rerun with SKIP_BUILD=0"
    fi
fi
if ! grep -aFq 'DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV' ./ds4-bench ||
   ! grep -aFq 'q4-gate-tile8' ./ds4-bench ||
   ! grep -aFq 'q4-down-tile16' ./ds4-bench; then
    die "ds4-bench lacks required warm-cache/scalar support; rebuild it"
fi

declare -A variant_split variant_devices
variant_split[baseline-22x21]=22
variant_devices[baseline-22x21]=0,2,1,3
variant_split[split-21x22]=21
variant_devices[split-21x22]=0,2,1,3
variant_split[swap-22x21]=22
variant_devices[swap-22x21]=0,3,1,2
variants=(baseline-22x21 split-21x22 swap-22x21)

current_phase=manifest
{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_vram=%s\nctx_start=%s\nctx_max=%s\nstep_mul=%s\n' \
        "$GPU_VRAM" "$CTX_START" "$CTX_MAX" "$STEP_MUL"
    printf 'prefill_chunk=%s\nrepeats=%s\nskip_build=%s\n' \
        "$PREFILL_CHUNK" "$REPEATS" "$SKIP_BUILD"
    printf 'model_hashing=disabled\nnsight=disabled\nq8_fp16_cache=forced-on\n'
    for variant in "${variants[@]}"; do
        printf 'variant=%s split=%s/%s gpu_devices=%s\n' \
            "$variant" "${variant_split[$variant]}" \
            "$((43 - variant_split[$variant]))" "${variant_devices[$variant]}"
    done
    printf '\n[gpu]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[topology]\n%s\n' "$gpu_topology"
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
printf 'repeat\tslot\tvariant\tsplit\tgpu_devices\tcsv\tlog\tcache_before\tcache_after\n' \
    >"$OUTPUT_DIR/runs.tsv"

capture_gpu_telemetry() {
    nvidia-smi --query-gpu=index,pci.bus_id,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,memory.used,memory.free,utilization.gpu \
        --format=csv >"$1"
}

validate_log() {
    local variant=$1 log=$2 split=${variant_split[$variant]}
    local devices=${variant_devices[$variant]} path pos gpu layer_start layer_end layer count
    local audit_count recipe_count marker_count expected_markers=0
    local -a ids
    IFS=, read -r -a ids <<<"$devices"
    grep -Fq "ds4: CUDA EP forced pipeline split $split/$((43 - split))" "$log" ||
        die "forced split missing for $variant"
    grep -Fq "4 devices [$devices] requested" "$log" ||
        die "GPU order missing for $variant"
    audit_count=$(grep -c '^ds4: routed-quant-audit layer=' "$log" || true)
    recipe_count=$(grep -Ec \
        '^ds4: routed-quant-audit layer=[0-9]+ gate=q4_k up=q4_k down=q4_k$' \
        "$log" || true)
    [[ $audit_count == 43 && $recipe_count == 43 ]] ||
        die "$variant expected 43 exact full-Q4 layers; got $recipe_count/$audit_count"
    for path in q4-gate-tile8 q4-down-tile16; do
        for pos in "${!ids[@]}"; do
            gpu=${ids[$pos]}
            if (( pos == 0 || pos == 2 )); then
                layer_start=0; layer_end=$((split - 1))
            else
                layer_start=$split; layer_end=42
            fi
            for ((layer=layer_start; layer<=layer_end; layer++)); do
                count=$(grep -Fc \
                    "sm75-scalar-dispatch path=$path scalar=1 device=$gpu layer=$layer " \
                    "$log" || true)
                [[ $count == 1 ]] ||
                    die "$variant $path device=$gpu layer=$layer marker count is $count"
                expected_markers=$((expected_markers + 1))
            done
        done
    done
    marker_count=$(grep -c '^ds4: sm75-scalar-dispatch ' "$log" || true)
    [[ $marker_count == "$expected_markers" ]] ||
        die "$variant scalar coverage is $marker_count; expected $expected_markers"
    if grep -E 'sm75-scalar-dispatch .*scalar=0' "$log" >/dev/null; then
        die "$variant executed a non-scalar Q4 specialization"
    fi
    [[ $(grep -Fc "starting untimed CUDA warm-up frontier $CTX_START" "$log" || true) == 1 ]] ||
        die "$variant is missing its warm-up start"
    [[ $(grep -Fc "completed untimed CUDA warm-up frontier $CTX_START" "$log" || true) == 1 ]] ||
        die "$variant is missing its warm-up completion"
}

current_phase=balanced-layout-cache-runs
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    case $(((repeat - 1) % 3)) in
        0) order=(baseline-22x21 split-21x22 swap-22x21) ;;
        1) order=(split-21x22 swap-22x21 baseline-22x21) ;;
        2) order=(swap-22x21 baseline-22x21 split-21x22) ;;
    esac
    slot=0
    for variant in "${order[@]}"; do
        slot=$((slot + 1))
        split=${variant_split[$variant]}
        devices=${variant_devices[$variant]}
        stem="$variant-r$repeat"
        csv_path="$OUTPUT_DIR/runs/$stem.csv"
        log_path="$OUTPUT_DIR/runs/$stem.log"
        cache_before="$OUTPUT_DIR/runs/$stem.q8-cache-before.csv"
        cache_after="$OUTPUT_DIR/runs/$stem.q8-cache-after.csv"
        printf 'Benchmarking %s repeat=%d/%d slot=%d split=%s/%s devices=%s\n' \
            "$variant" "$repeat" "$REPEATS" "$slot" "$split" "$((43 - split))" "$devices"
        capture_gpu_telemetry "$OUTPUT_DIR/runs/$stem.telemetry-before.csv"
        if ! "${clean_prefix[@]}" \
                "DS4_CUDA_EP_STAGE_SPLIT=$split" \
                DS4_CUDA_PREFILL_PIPELINE=1 \
                DS4_CUDA_PREFILL_PIPELINE_MB=512 \
                DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
                DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=1 \
                DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=1 \
                DS4_CUDA_MOE_IQ2_SCALAR_SM75=1 \
                DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
                DS4_CUDA_MOE_SCALAR_AUDIT=1 \
                "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
                "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$cache_before" \
                "DS4_CUDA_Q8_CACHE_STATE_CSV=$cache_after" \
                ./ds4-bench \
                    --cuda --cuda-tensor-parallel \
                    --gpu-devices "$devices" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                    --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL" \
                    --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
                    --csv "$csv_path" >"$log_path" 2>&1; then
            tail -n 160 "$log_path" >&2 || true
            die "benchmark failed: $stem"
        fi
        capture_gpu_telemetry "$OUTPUT_DIR/runs/$stem.telemetry-after.csv"
        [[ -s $csv_path ]] || die "benchmark produced no CSV: $stem"
        [[ -s $cache_before && -s $cache_after ]] ||
            die "Q8 cache-state snapshot missing: $stem"
        cmp -s "$cache_before" "$cache_after" ||
            die "Q8 cache changed during measured frontiers: $stem"
        validate_log "$variant" "$log_path"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$slot" "$variant" "$split" "$devices" \
            "$csv_path" "$log_path" "$cache_before" "$cache_after" \
            >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv_path"
    done
done

current_phase=summarize
python3 speed-bench/summarize-q4-layout-cache-ab.py \
    "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR" "$REPEATS" \
    2>&1 | tee "$OUTPUT_DIR/summary.txt"
for required in paired-samples.csv frontier-summary.csv cache-summary.csv \
                overall-summary.txt summary.txt runs.tsv; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing final evidence: $required"
done
printf 'q4_layout_cache_ab=valid\nmeasurement_grade=position-balanced\n' \
    >"$OUTPUT_DIR/capture-status.txt"
current_phase=complete
printf 'Q4 layout/cache A/B complete: %s\n' "$OUTPUT_DIR"
