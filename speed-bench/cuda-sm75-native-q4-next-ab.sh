#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run an isolated production A/B of the K-stage4 SM75 native-Q4 designs against
the cost-planner production path. No model conversion or hashing occurs.

Required environment:
  NATIVE_MODEL=/absolute/path/to/SM75-native-Q4.gguf

Optional environment:
  PROMPT=...                  Default: speed-bench/promessi_sposi.txt
  AB_VARIANTS=baseline,down,gate,both
  REPEATS=2                  Each variant runs this many times
  CTX_START=2048
  CTX_MAX=65536
  STEP_MUL=2
  PREFILL_CHUNK=2048
  GPU_VRAM=auto
  RUN_E2E_EXACT=1            Compare every variant's raw logits at CTX_START
  RUN_NSYS=0                 Capture baseline and both at PROFILE_TOKENS
  PROFILE_TOKENS=2048
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  NATIVE_Q4_NEXT_DIR=...

Variants:
  baseline   cost planner + production gate tile8/down full-stage
  legacy     legacy residual planner (diagnostic only)
  down       baseline + down tile16 K-stage4
  gate       baseline + gate/up tile16 K-stage4
  both       baseline + both K-stage4 designs

The production placement is fixed to GPUs 0,3,1,2 and a 22/21 stage split.
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
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
GPU_VRAM=${GPU_VRAM:-auto}
RUN_E2E_EXACT=${RUN_E2E_EXACT:-1}
RUN_NSYS=${RUN_NSYS:-0}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
GPU_DEVICES=0,3,1,2
STAGE_SPLIT=22
CUDA_ARCH=sm_75
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${NATIVE_Q4_NEXT_DIR:-$repo_dir/sm75-native-q4-next-ab-$run_stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "REPEATS:$REPEATS" "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "RUN_E2E_EXACT:$RUN_E2E_EXACT" "RUN_NSYS:$RUN_NSYS" \
            "PROFILE_TOKENS:$PROFILE_TOKENS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( REPEATS > 0 && CTX_START > 0 && CTX_MAX >= CTX_START &&
   STEP_MUL >= 2 && PREFILL_CHUNK > 0 && PROFILE_TOKENS > 0 )) ||
    die "invalid benchmark shape"
for flag in RUN_E2E_EXACT RUN_NSYS SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

IFS=, read -r -a variants <<<"$AB_VARIANTS"
(( ${#variants[@]} >= 2 )) || die "AB_VARIANTS must contain at least two variants"
declare -A seen=()
for variant in "${variants[@]}"; do
    case "$variant" in
        baseline|legacy|down|gate|both) ;; *)
        die "unknown AB variant: $variant";; esac
    [[ -z ${seen[$variant]:-} ]] || die "duplicate AB variant: $variant"
    seen[$variant]=1
done
[[ -n ${seen[baseline]:-} ]] || die "AB_VARIANTS must include baseline"

for tool in awk cmp cuobjdump c++filt date env git grep make mkdir mv nproc nvidia-smi \
            python3 sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
(( RUN_NSYS == 0 )) || command -v nsys >/dev/null 2>&1 || die "nsys not found"
for gpu in 0 1 2 3; do
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is not SM75 (${cap:-unknown})"
done
gpu_topology=$(nvidia-smi topo -m)
topology_link() {
    local from=$1 to=$2
    awk -v from="GPU$from" -v to="GPU$to" '
        NR == 1 { for (i=1;i<=NF;i++) if ($i ~ /^GPU[0-9]+$/) col[$i]=i+1; next }
        $1 == from && (to in col) { print $(col[to]); exit }
    ' <<<"$gpu_topology"
}
[[ $(topology_link 0 1) =~ ^NV[0-9]+$ ]] || die "GPU 0<->1 is not NVLink"
[[ $(topology_link 2 3) =~ ^NV[0-9]+$ ]] || die "GPU 2<->3 is not NVLink"

default_lock=/tmp/ds4.lock
if [[ -e $default_lock ]]; then
    [[ -f $default_lock && -w $default_lock ]] ||
        die "$default_lock exists but is not a writable regular file"
fi

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{validation,runs,nsys,provenance}
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
        baseline) printf '0 0 0\n' ;;
        legacy)   printf '1 0 0\n' ;;
        down)     printf '0 0 1\n' ;;
        gate)     printf '0 1 0\n' ;;
        both)     printf '0 1 1\n' ;;
    esac
}

run_variant() {
    local variant=$1; shift
    local legacy gate down
    read -r legacy gate down < <(variant_flags "$variant")
    "${clean_prefix[@]}" "${production_prefix[@]}" \
        "DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=$legacy" \
        "DS4_CUDA_MOE_NATIVE_Q4_GATE_KSTAGE4=$gate" \
        "DS4_CUDA_MOE_NATIVE_Q4_DOWN_KSTAGE4=$down" \
        "$@"
}

validate_log() {
    local variant=$1 log=$2 legacy gate down planner gate_name down_name
    read -r legacy gate down < <(variant_flags "$variant")
    planner=cost; [[ $legacy == 0 ]] || planner=legacy
    grep -Fq 'ds4: CUDA EP forced pipeline split 22/21' "$log" ||
        die "$variant did not use split 22/21"
    grep -Fq '4 devices [0,3,1,2] requested' "$log" ||
        die "$variant did not use GPU order 0,3,1,2"
    gate_name=tile8; [[ $gate == 0 ]] || gate_name=kstage4
    down_name=full-stage; [[ $down == 0 ]] || down_name=kstage4
    grep -Fq "packed A/W, planner=$planner, gate=$gate_name, down=$down_name" \
        "$log" || die "$variant did not select the requested native-Q4 paths"
}

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        tests/cuda_sm75_profile_harness \
        CUDA_ARCH="$CUDA_ARCH" 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    for binary in ./ds4-bench ./tests/cuda_long_context_smoke \
                  ./tests/cuda_sm75_profile_harness; do
        [[ -x $binary ]] || die "SKIP_BUILD=1 but $binary is missing"
    done
    make -q ds4-bench tests/cuda_long_context_smoke \
        tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found stale CUDA targets"
fi
cuobjdump --dump-resource-usage ./ds4-bench \
    >"$OUTPUT_DIR/provenance/resource-usage.txt" 2>&1 ||
    die "could not record CUDA kernel resource usage"
c++filt <"$OUTPUT_DIR/provenance/resource-usage.txt" \
    >"$OUTPUT_DIR/provenance/resource-usage.demangled.txt"
python3 - "$OUTPUT_DIR/provenance/resource-usage.demangled.txt" \
        "$OUTPUT_DIR/provenance/kstage-resource-gate.csv" <<'PY'
import csv, re, sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
targets = {
    "gate-kstage4": "moe_gate_up_mid_sm75_native_q4_tile16_kstage4_kernel",
    "down-kstage4": "moe_down_sm75_native_q4_tile16_kstage4_kernel",
}
records = []
for index, line in enumerate(text):
    if not re.match(r"\s*Function\s*:?\s*", line):
        continue
    name = re.sub(r"^\s*Function\s*:?\s*", "", line).rstrip(":")
    resource = ""
    for candidate in text[index + 1:index + 8]:
        if re.match(r"\s*REG:", candidate):
            resource = candidate
            break
    for label, needle in targets.items():
        if needle not in name or not resource:
            continue
        values = {key: int(value) for key, value in
                  re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", resource)}
        missing = {"REG", "STACK", "SHARED", "LOCAL"} - values.keys()
        if missing:
            raise SystemExit(f"missing resource fields for {label}: {resource}")
        records.append({"design": label, "kernel": name, **values})
for label in targets:
    matches = [row for row in records if row["design"] == label]
    if not matches:
        raise SystemExit(f"resource gate found no {label} instantiation")
    for row in matches:
        if row["STACK"] or row["LOCAL"]:
            raise SystemExit(
                f'{label} spills: STACK={row["STACK"]} LOCAL={row["LOCAL"]}')
        if row["SHARED"] >= 32768:
            raise SystemExit(f'{label} shared memory is {row["SHARED"]}, not <32768')
        if row["REG"] > 128:
            raise SystemExit(f'{label} uses {row["REG"]} registers, not <=128')
with open(sys.argv[2], "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream,
        fieldnames=("design", "kernel", "REG", "STACK", "SHARED", "LOCAL"))
    writer.writeheader(); writer.writerows(records)
print("validated K-stage4 kernels: no spills, shared <32KiB, registers <=128")
PY

current_phase=api-exactness
"${clean_prefix[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/validation/cuda-exact.log" 2>&1 || {
    tail -n 180 "$OUTPUT_DIR/validation/cuda-exact.log" >&2 || true
    die "CUDA exact-output suite failed"
}
for marker in \
    'tagged SM75 native Q4 cost-planner default exact' \
    'tagged SM75 native Q4 legacy-planner diagnostic exact' \
    'tagged SM75 native Q4 gate-kstage4 exact' \
    'tagged SM75 native Q4 down-kstage4 exact' \
    'tagged SM75 native Q4 gate/down-kstage4 exact' \
    'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exact.log" ||
        die "exact-output marker missing: $marker"
done

current_phase=cost-planner-audit
"${clean_prefix[@]}" \
    ./tests/cuda_sm75_profile_harness native-q4-early \
    >"$OUTPUT_DIR/validation/cost-planner-audit.log" 2>&1 || {
    tail -n 180 "$OUTPUT_DIR/validation/cost-planner-audit.log" >&2 || true
    die "cost-aware planner audit failed"
}
grep -Fq 'harness_status=ok' \
    "$OUTPUT_DIR/validation/cost-planner-audit.log" ||
    die "cost-aware planner audit did not finish cleanly"

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
    printf '\n[topology]\n%s\n' "$gpu_topology"
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

common=(--cuda --cuda-tensor-parallel --gpu-devices "$GPU_DEVICES"
        --gpu-vram "$GPU_VRAM" --prompt-file "$PROMPT"
        --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 --model "$NATIVE_MODEL")

if [[ $RUN_E2E_EXACT == 1 ]]; then
    current_phase=end-to-end-exactness
    reference=
    for variant in "${variants[@]}"; do
        dir="$OUTPUT_DIR/validation/$variant-logits"
        mkdir -p "$dir"
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
    printf 'api_exact=true\ne2e_all_variants_raw_logits_bit_exact=true\n' \
        >"$OUTPUT_DIR/validation/exact-status.txt"
else
    printf 'api_exact=true\ne2e_exact_skipped=true\n' \
        >"$OUTPUT_DIR/validation/exact-status.txt"
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
            ./ds4-bench "${common[@]}" \
            --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
            --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL" \
            --csv "$OUTPUT_DIR/runs/$stem.csv" \
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
import csv, math, statistics, sys
with open(sys.argv[1], encoding="utf-8", newline="") as stream:
    runs = list(csv.DictReader(stream, delimiter="\t"))
values = {}
by_repeat = {}
for run in runs:
    with open(run["csv"], encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        key = (run["variant"], int(row["ctx_tokens"]))
        value = float(row["prefill_tps"])
        values.setdefault(key, []).append(value)
        repeat_key = (int(run["repeat"]), *key)
        if repeat_key in by_repeat:
            raise SystemExit(f"duplicate run value: {repeat_key}")
        by_repeat[repeat_key] = value
contexts = sorted(ctx for variant, ctx in values if variant == "baseline")
records = []
for ctx in contexts:
    baseline = statistics.median(values[("baseline", ctx)])
    for variant in sorted({variant for variant, _ in values}):
        samples = values.get((variant, ctx))
        if not samples:
            raise SystemExit(f"missing {variant} context {ctx}")
        median = statistics.median(samples)
        ratio = median / baseline
        if not math.isfinite(ratio):
            raise SystemExit("non-finite A/B ratio")
        paired = []
        for repeat in range(1, len(values[("baseline", ctx)]) + 1):
            base_key = (repeat, "baseline", ctx)
            variant_key = (repeat, variant, ctx)
            if base_key not in by_repeat or variant_key not in by_repeat:
                raise SystemExit(
                    f"missing paired repeat {repeat} for {variant} context {ctx}"
                )
            paired.append(by_repeat[variant_key] / by_repeat[base_key])
        paired_median = statistics.median(paired)
        records.append({"ctx_tokens": ctx, "variant": variant,
                        "samples": len(samples),
                        "median_prefill_tps": f"{median:.6f}",
                        "median_over_baseline": f"{ratio:.6f}",
                        "median_gain_pct": f"{(ratio-1)*100:.3f}"})
        records[-1].update({
            "paired_median_over_baseline": f"{paired_median:.6f}",
            "paired_median_gain_pct": f"{(paired_median-1)*100:.3f}",
            "paired_min_over_baseline": f"{min(paired):.6f}",
            "paired_max_over_baseline": f"{max(paired):.6f}",
        })
with open(sys.argv[2], "w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader(); writer.writerows(records)
print("ctx_tokens,variant,median_prefill_tps,paired_median_gain_pct")
for record in records:
    print(f'{record["ctx_tokens"]},{record["variant"]},'
          f'{record["median_prefill_tps"]},'
          f'{record["paired_median_gain_pct"]}')
PY

if [[ $RUN_NSYS == 1 ]]; then
    current_phase=nsight-systems-ab
    for variant in baseline both; do
        run_variant "$variant" DS4_NSYS_CAPTURE_PREFILL=1 \
            nsys profile --force-overwrite=true --sample=none \
            --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --output="$OUTPUT_DIR/nsys/$variant-full-q4" \
            ./ds4-bench "${common[@]}" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + 1))" \
            --step-incr "$PROFILE_TOKENS" \
            --csv "$OUTPUT_DIR/nsys/$variant-benchmark.csv" \
            >"$OUTPUT_DIR/nsys/$variant-capture.log" 2>&1
        [[ -s $OUTPUT_DIR/nsys/$variant-full-q4.nsys-rep ]] ||
            die "missing $variant Nsight Systems report"
        nsys stats --report cuda_gpu_kern_sum --format csv \
            "$OUTPUT_DIR/nsys/$variant-full-q4.nsys-rep" \
            >"$OUTPUT_DIR/nsys/$variant-cuda_gpu_kern_sum.csv" \
            2>"$OUTPUT_DIR/nsys/$variant-stats.log"
    done
fi

current_phase=complete
printf 'SM75 native-Q4 next-path A/B complete: %s\n' "$OUTPUT_DIR"
