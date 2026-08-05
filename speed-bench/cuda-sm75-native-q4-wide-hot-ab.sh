#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, A/B, and profile the two post-failure-audit SM75 native-Q4
candidates:
  down-wide512  full shared tile16 with one 512-thread/16-warp CTA;
  gate-full64   full tile16 activation payload in opt-in 64 KiB shared memory.

Required environment:
  NATIVE_MODEL=/absolute/path/to/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf

Optional environment:
  PROMPT=...                  Default: speed-bench/promessi_sposi.txt
  AB_VARIANTS=baseline,down,gate,both
  REPEATS=2
  CTX_START=2048
  CTX_MAX=8192
  STEP_MUL=2
  PREFILL_CHUNK=2048
  RUN_E2E_EXACT=1
  RUN_NSYS=1
  RUN_NCU=1
  PROFILE_TOKENS=2048
  PROFILE_GPU=0
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  WIDE_HOT_DIR=/absolute/output/directory

The run never hashes or converts the model. Production placement is fixed to
GPUs 0,3,1,2 with a 22/21 stage split.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${NATIVE_MODEL:?set NATIVE_MODEL to the absolute tagged SM75-native Q4 GGUF}"
[[ $NATIVE_MODEL == /* && -s $NATIVE_MODEL ]] ||
    die "NATIVE_MODEL must be an existing non-empty absolute path"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
AB_VARIANTS=${AB_VARIANTS:-baseline,down,gate,both}
REPEATS=${REPEATS:-2}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-8192}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
RUN_E2E_EXACT=${RUN_E2E_EXACT:-1}
RUN_NSYS=${RUN_NSYS:-1}
RUN_NCU=${RUN_NCU:-1}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
PROFILE_GPU=${PROFILE_GPU:-0}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
GPU_DEVICES=0,3,1,2
CUDA_ARCH=sm_75
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${WIDE_HOT_DIR:-$repo_dir/sm75-native-q4-wide-hot-$run_stamp}

[[ -s $PROMPT ]] || die "prompt not found or empty: $PROMPT"
for item in "REPEATS:$REPEATS" "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "RUN_E2E_EXACT:$RUN_E2E_EXACT" "RUN_NSYS:$RUN_NSYS" \
            "RUN_NCU:$RUN_NCU" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "PROFILE_GPU:$PROFILE_GPU" "NCU_USE_SUDO:$NCU_USE_SUDO" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( REPEATS > 0 && CTX_START > 0 && CTX_MAX >= CTX_START &&
   STEP_MUL >= 2 && PREFILL_CHUNK > 0 && PROFILE_TOKENS > 0 )) ||
    die "invalid benchmark shape"
for flag in RUN_E2E_EXACT RUN_NSYS RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

IFS=, read -r -a variants <<<"$AB_VARIANTS"
(( ${#variants[@]} >= 2 )) || die "AB_VARIANTS must contain at least two variants"
declare -A seen=()
for variant in "${variants[@]}"; do
    case "$variant" in baseline|down|gate|both) ;; *)
        die "unknown AB variant: $variant";; esac
    [[ -z ${seen[$variant]:-} ]] || die "duplicate AB variant: $variant"
    seen[$variant]=1
done
[[ -n ${seen[baseline]:-} ]] || die "AB_VARIANTS must include baseline"

tools=(awk cmp cuobjdump c++filt date env git grep make mkdir mv nproc
       nvidia-smi python3 sort stat tail tar tee tr)
(( RUN_NSYS == 0 )) || tools+=(nsys)
(( RUN_NCU == 0 )) || tools+=(ncu)
for tool in "${tools[@]}"; do command -v "$tool" >/dev/null 2>&1 || die "$tool not found"; done
for gpu in 0 1 2 3; do
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is not SM75 (${cap:-unknown})"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{validation,runs,nsys,ncu,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
current_phase=initialization
finalize() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")"; then
            mv -- "$partial" "$archive"
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

mapfile -t inherited_ds4_envs < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean_prefix=(env)
for name in "${inherited_ds4_envs[@]}"; do clean_prefix+=(-u "$name"); done
production_prefix=(
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1)

variant_flags() {
    case "$1" in
        baseline) printf '0 0\n' ;;
        down)     printf '0 1\n' ;;
        gate)     printf '1 0\n' ;;
        both)     printf '1 1\n' ;;
    esac
}

run_variant() {
    local variant=$1; shift
    local gate down
    read -r gate down < <(variant_flags "$variant")
    "${clean_prefix[@]}" "${production_prefix[@]}" \
        DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=0 \
        DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7=0 \
        DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7=0 \
        "DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64=$gate" \
        "DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512=$down" "$@"
}

run_harness() {
    local variant=$1; shift
    local gate down
    read -r gate down < <(variant_flags "$variant")
    "${clean_prefix[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=0 \
        DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7=0 \
        DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7=0 \
        "DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64=$gate" \
        "DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512=$down" "$@"
}

validate_log() {
    local variant=$1 log=$2 gate down gate_name=tile8 down_name=full-stage
    read -r gate down < <(variant_flags "$variant")
    [[ $gate == 0 ]] || gate_name=full64
    [[ $down == 0 ]] || down_name=full-stage-wide512
    grep -Fq 'ds4: CUDA EP forced pipeline split 22/21' "$log" ||
        die "$variant did not use split 22/21"
    grep -Fq '4 devices [0,3,1,2] requested' "$log" ||
        die "$variant did not use GPU order 0,3,1,2"
    grep -Fq "packed A/W, planner=cost, gate=$gate_name, down=$down_name" \
        "$log" || die "$variant did not select the requested candidates"
}

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH" \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    for binary in ./ds4-bench ./tests/cuda_long_context_smoke \
                  ./tests/cuda_sm75_profile_harness; do
        [[ -x $binary ]] || die "SKIP_BUILD=1 but $binary is missing"
    done
    make -q ds4-bench tests/cuda_long_context_smoke \
        tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found stale CUDA targets"
fi

current_phase=resource-validation
cuobjdump --dump-resource-usage ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/provenance/resource-usage.txt" 2>&1 ||
    die "could not record CUDA resource usage"
c++filt <"$OUTPUT_DIR/provenance/resource-usage.txt" \
    >"$OUTPUT_DIR/provenance/resource-usage.demangled.txt"
python3 - "$OUTPUT_DIR/provenance/resource-usage.demangled.txt" \
        "$OUTPUT_DIR/provenance/candidate-resources.csv" <<'PY'
import csv, re, sys
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
targets = {
    "gate-full64": "moe_gate_up_mid_sm75_native_q4_tile16_full64_kernel",
    "down-wide512": "moe_down_sm75_native_q4_tile_kernel",
}
records = []
for index, line in enumerate(lines):
    if not re.match(r"\s*Function\s*:?\s*", line):
        continue
    name = re.sub(r"^\s*Function\s*:?\s*", "", line).rstrip(":")
    resource = next((item for item in lines[index + 1:index + 8]
                     if re.match(r"\s*REG:", item)), "")
    for label, needle in targets.items():
        if needle not in name or not resource:
            continue
        values = {key: int(value) for key, value in
                  re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", resource)}
        if set(values) != {"REG", "STACK", "SHARED", "LOCAL"}:
            raise SystemExit(f"missing resource fields for {label}: {resource}")
        records.append({"design": label, "kernel": name, **values})
for label in targets:
    rows = [row for row in records if row["design"] == label]
    if not rows:
        raise SystemExit(f"no resource records for {label}")
    for row in rows:
        if row["STACK"] or row["LOCAL"]:
            raise SystemExit(f"{label} has local-memory traffic: {row}")
        if row["REG"] > 128:
            raise SystemExit(f"{label} exceeds 128 registers: {row}")
        if label == "gate-full64" and row["SHARED"] != 0:
            raise SystemExit(f"gate-full64 must use only dynamic shared memory: {row}")
with open(sys.argv[2], "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream,
        fieldnames=("design", "kernel", "REG", "STACK", "SHARED", "LOCAL"))
    writer.writeheader(); writer.writerows(records)
print("validated full64/wide512: no spills and <=128 registers")
PY

current_phase=api-exactness
"${clean_prefix[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/validation/cuda-exact.log" 2>&1 || {
    tail -n 180 "$OUTPUT_DIR/validation/cuda-exact.log" >&2 || true
    die "CUDA exact-output suite failed"
}
for marker in \
    'tagged SM75 native Q4 gate-full64 exact' \
    'tagged SM75 native Q4 down-wide512 exact' \
    'tagged SM75 native Q4 gate-full64/down-wide512 exact' \
    'SM75 native Q4 candidate selected: gate-full64' \
    'SM75 native Q4 candidate selected: down-wide512' \
    'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exact.log" ||
        die "exact-output marker missing: $marker"
done

{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'native_model=%s\nnative_bytes=%s\nmodel_hashing=disabled\n' \
        "$NATIVE_MODEL" "$(stat -c %s "$NATIVE_MODEL")"
    printf 'variants=%s\nrepeats=%s\ngpu_devices=%s\nstage_split=22/21\n' \
        "$AB_VARIANTS" "$REPEATS" "$GPU_DEVICES"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK"
    printf '\n[gpu]\n'; nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version --format=csv
    printf '\n[topology]\n'; nvidia-smi topo -m
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

common=(--cuda --cuda-tensor-parallel --gpu-devices "$GPU_DEVICES"
        --gpu-vram auto --prompt-file "$PROMPT" --prefill-chunk "$PREFILL_CHUNK"
        --gen-tokens 0 --model "$NATIVE_MODEL")

if [[ $RUN_E2E_EXACT == 1 ]]; then
    current_phase=end-to-end-exactness
    reference=
    for variant in "${variants[@]}"; do
        dir="$OUTPUT_DIR/validation/$variant-logits"; mkdir -p "$dir"
        run_variant "$variant" ./ds4-bench "${common[@]}" \
            --ctx-start "$CTX_START" --ctx-max "$CTX_START" \
            --ctx-alloc "$((CTX_START + 1))" --step-incr "$CTX_START" \
            --dump-frontier-logits-dir "$dir" \
            --csv "$OUTPUT_DIR/validation/$variant.csv" \
            >"$OUTPUT_DIR/validation/$variant.log" 2>&1
        validate_log "$variant" "$OUTPUT_DIR/validation/$variant.log"
        raw=$(printf '%s/frontier_%06d.logits.f32' "$dir" "$CTX_START")
        [[ -s $raw ]] || die "missing raw logits for $variant"
        if [[ -z $reference ]]; then reference=$raw; else
            cmp -s "$reference" "$raw" ||
                die "baseline/$variant raw logits are not bit-exact"
        fi
    done
fi

current_phase=balanced-ab
printf 'repeat\tslot\tvariant\tcsv\tlog\n' >"$OUTPUT_DIR/runs.tsv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 == 1 )); then order=("${variants[@]}"); else
        order=(); for ((i=${#variants[@]}-1; i>=0; i--)); do order+=("${variants[i]}"); done
    fi
    slot=0
    for variant in "${order[@]}"; do
        slot=$((slot + 1)); stem="$variant-r$repeat"
        printf 'Benchmarking %s repeat=%d/%d slot=%d/%d...\n' \
            "$variant" "$repeat" "$REPEATS" "$slot" "${#variants[@]}"
        run_variant "$variant" "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
            ./ds4-bench "${common[@]}" --ctx-start "$CTX_START" \
            --ctx-max "$CTX_MAX" --ctx-alloc "$((CTX_MAX + 1))" \
            --step-mul "$STEP_MUL" --csv "$OUTPUT_DIR/runs/$stem.csv" \
            >"$OUTPUT_DIR/runs/$stem.log" 2>&1
        [[ -s $OUTPUT_DIR/runs/$stem.csv ]] || die "empty CSV: $stem"
        validate_log "$variant" "$OUTPUT_DIR/runs/$stem.log"
        printf '%s\t%s\t%s\t%s\t%s\n' "$repeat" "$slot" "$variant" \
            "$OUTPUT_DIR/runs/$stem.csv" "$OUTPUT_DIR/runs/$stem.log" \
            >>"$OUTPUT_DIR/runs.tsv"
        cat "$OUTPUT_DIR/runs/$stem.csv"
    done
done

python3 - "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/ab-summary.csv" <<'PY'
import csv, statistics, sys
with open(sys.argv[1], encoding="utf-8", newline="") as stream:
    runs = list(csv.DictReader(stream, delimiter="\t"))
values = {}
for run in runs:
    with open(run["csv"], encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            values.setdefault((run["variant"], int(row["ctx_tokens"])), []).append(
                float(row["prefill_tps"]))
contexts = sorted(ctx for variant, ctx in values if variant == "baseline")
records = []
for ctx in contexts:
    baseline = statistics.median(values[("baseline", ctx)])
    for variant in sorted({variant for variant, _ in values}):
        median = statistics.median(values[(variant, ctx)])
        records.append({"ctx_tokens": ctx, "variant": variant,
                        "samples": len(values[(variant, ctx)]),
                        "median_prefill_tps": f"{median:.6f}",
                        "gain_pct": f"{(median / baseline - 1.0) * 100.0:.3f}"})
with open(sys.argv[2], "w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader(); writer.writerows(records)
print("ctx_tokens,variant,median_prefill_tps,gain_pct")
for row in records:
    print(f'{row["ctx_tokens"]},{row["variant"]},'
          f'{row["median_prefill_tps"]},{row["gain_pct"]}')
PY

if [[ $RUN_NSYS == 1 ]]; then
    current_phase=nsight-systems
    for variant in "${variants[@]}"; do
        base="$OUTPUT_DIR/nsys/$variant"
        printf 'Nsight Systems: %s...\n' "$variant"
        run_variant "$variant" DS4_NSYS_CAPTURE_PREFILL=1 \
            nsys profile --force-overwrite=true --sample=none \
                --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
                --capture-range-end=stop --output="$base" \
                ./ds4-bench "${common[@]}" --ctx-start "$PROFILE_TOKENS" \
                --ctx-max "$PROFILE_TOKENS" --ctx-alloc "$((PROFILE_TOKENS + 1))" \
                --step-incr "$PROFILE_TOKENS" --csv "$base-benchmark.csv" \
                >"$base.log" 2>&1
        [[ -s $base.nsys-rep ]] || die "missing Nsight Systems report: $variant"
        validate_log "$variant" "$base.log"
        nsys stats --report cuda_gpu_kern_sum --format csv "$base.nsys-rep" \
            >"$base-cuda_gpu_kern_sum.csv" 2>"$base-stats.log"
    done
fi

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v; ncu_command=(sudo -E "$ncu_bin")
    fi
    ncu_sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
        --section SchedulerStats --section WarpStateStats
        --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    profile_one() {
        local name=$1 scenario=$2 variant=$3 kernel=$4 expected=$5
        local base="$OUTPUT_DIR/ncu/$name" rc=0
        printf 'Nsight Compute: %s...\n' "$name"
        run_harness "$variant" "${ncu_command[@]}" --config-file off \
            --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "regex:$kernel" \
            --launch-count 1 --replay-mode kernel --cache-control none \
            --clock-control none --force-overwrite --export "$base" \
            "${ncu_sections[@]}" ./tests/cuda_sm75_profile_harness "$scenario" \
            >"$base.log" 2>&1 || rc=$?
        (( rc == 0 )) || { tail -n 120 "$base.log" >&2 || true; die "NCU failed: $name"; }
        [[ -s $base.ncu-rep ]] || die "missing Nsight Compute report: $name"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log"
        python3 speed-bench/validate-ncu-capture.py "$base.csv" "$expected" 0 \
            --process cuda_sm75_profile_harness >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true; die "wrong NCU kernel: $name"; }
        cat "$base-validation.txt"
    }
    profile_one early-down-wide512 native-q4-early down \
        '^moe_down_sm75_native_q4_tile_kernel.*16.*$' \
        'moe_down_sm75_native_q4_tile_kernel.*16'
    profile_one late-down-wide512 native-q4-late down \
        '^moe_down_sm75_native_q4_tile_kernel.*16.*$' \
        'moe_down_sm75_native_q4_tile_kernel.*16'
    profile_one early-gate-full64 native-q4-early gate \
        '^moe_gate_up_mid_sm75_native_q4_tile16_full64_kernel.*$' \
        'moe_gate_up_mid_sm75_native_q4_tile16_full64_kernel'
    profile_one late-gate-full64 native-q4-late gate \
        '^moe_gate_up_mid_sm75_native_q4_tile16_full64_kernel.*$' \
        'moe_gate_up_mid_sm75_native_q4_tile16_full64_kernel'
fi

current_phase=complete
printf 'SM75 native-Q4 wide/full-hot A/B complete: %s\n' "$OUTPUT_DIR"
