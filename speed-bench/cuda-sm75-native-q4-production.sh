#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Losslessly repack a stock full-Q4 GGUF into DS4's explicitly tagged SM75
native routed-expert layout, require exact output, run a balanced four-GPU
A/B, and reprofile the production kernels.

Required environment:
  MODEL=/absolute/path/to/standard-full-Q4.gguf

Optional environment:
  NATIVE_MODEL=/absolute/path/to/output-native.gguf
  QUANT_GPU_DEVICES=0,1,2,3
  REPACK_THREADS=8
  PROMPT=/absolute/path/prompt.txt
  GPU_VRAM=auto
  CTX_START=2048
  CTX_MAX=65536
  STEP_MUL=2
  PREFILL_CHUNK=2048
  REPEATS=2                 # even; 2 is a pilot, 6 is replicated evidence
  PROFILE_TOKENS=2048
  PROFILE_GPU=0
  NCU_USE_SUDO=1
  RUN_NSYS=1
  RUN_NCU=1
  SKIP_BUILD=0
  SKIP_E2E_EXACT=0          # 1 only when prior full-model exact evidence exists
  SKIP_BENCHMARK=0          # 1 to resume only the profiler phase
  CREATE_ARCHIVE=1
  NATIVE_Q4_DIR=...

The tested production layout is intentionally fixed:
  --gpu-devices 0,3,1,2
  DS4_CUDA_EP_STAGE_SPLIT=22  (22/21 layers)

No model hash is calculated. If NATIVE_MODEL already exists it is reused;
the engine tag/dispatch and exact-logit checks still have to pass.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute standard full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] || die "MODEL must be an existing absolute path"
[[ $MODEL == *.gguf ]] || die "MODEL must end in .gguf"

NATIVE_MODEL=${NATIVE_MODEL:-${MODEL%.gguf}-SM75-native-Q4.gguf}
QUANT_GPU_DEVICES=${QUANT_GPU_DEVICES:-0,1,2,3}
REPACK_THREADS=${REPACK_THREADS:-8}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=0,3,1,2
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=22
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-2}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
PROFILE_GPU=${PROFILE_GPU:-0}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
RUN_NSYS=${RUN_NSYS:-1}
RUN_NCU=${RUN_NCU:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_E2E_EXACT=${SKIP_E2E_EXACT:-0}
SKIP_BENCHMARK=${SKIP_BENCHMARK:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CUDA_ARCH=sm_75
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${NATIVE_Q4_DIR:-$repo_dir/sm75-native-q4-production-$run_stamp}

[[ $NATIVE_MODEL == /* ]] || die "NATIVE_MODEL must be an absolute path"
[[ $MODEL != "$NATIVE_MODEL" ]] || die "MODEL and NATIVE_MODEL must differ"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "REPEATS:$REPEATS" "REPACK_THREADS:$REPACK_THREADS" \
            "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "PROFILE_GPU:$PROFILE_GPU" "NCU_USE_SUDO:$NCU_USE_SUDO" \
            "RUN_NSYS:$RUN_NSYS" "RUN_NCU:$RUN_NCU" \
            "SKIP_BUILD:$SKIP_BUILD" "SKIP_E2E_EXACT:$SKIP_E2E_EXACT" \
            "SKIP_BENCHMARK:$SKIP_BENCHMARK" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START && STEP_MUL >= 2 &&
   PREFILL_CHUNK > 0 && REPEATS > 0 && REPEATS % 2 == 0 &&
   REPACK_THREADS > 0 &&
   PROFILE_TOKENS > 0 )) || die "invalid benchmark/profile shape"
for flag in NCU_USE_SUDO RUN_NSYS RUN_NCU SKIP_BUILD SKIP_E2E_EXACT \
            SKIP_BENCHMARK CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk cmp date env git grep make mkdir mv nproc nvidia-smi \
            python3 rm sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
(( RUN_NSYS == 0 )) || command -v nsys >/dev/null 2>&1 || die "nsys not found"
(( RUN_NCU == 0 )) || command -v ncu >/dev/null 2>&1 || die "ncu not found"

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

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
[[ ! -e $OUTPUT_DIR.tar.gz ]] || die "archive already exists: $OUTPUT_DIR.tar.gz"
mkdir -p "$OUTPUT_DIR"/{validation,runs,nsys,ncu,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
current_phase=initialization
native_tmp=
finalize() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    if [[ -n $native_tmp && -e $native_tmp ]]; then rm -f -- "$native_tmp"; fi
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
production_prefix=("${clean_prefix[@]}"
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1)

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        tests/cuda_sm75_profile_harness \
        CUDA_ARCH="$CUDA_ARCH" 2>&1 | tee "$OUTPUT_DIR/build.log"
    make -C gguf-tools deepseek4-quantize-cuda test-quants-cuda \
        CUDA_ARCH="$CUDA_ARCH" 2>&1 | tee "$OUTPUT_DIR/quantizer-build.log"
else
    for binary in ./ds4-bench ./tests/cuda_long_context_smoke \
                  ./tests/cuda_sm75_profile_harness \
                  ./gguf-tools/deepseek4-quantize-cuda \
                  ./gguf-tools/test-quants-cuda; do
        [[ -x $binary ]] || die "SKIP_BUILD=1 but $binary is missing"
    done
    make -q ds4-bench tests/cuda_long_context_smoke \
        tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found stale CUDA targets"
    make -C gguf-tools -q deepseek4-quantize-cuda test-quants-cuda \
        CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale quantizer"
fi

current_phase=cuda-repack-byte-exact
./gguf-tools/test-quants-cuda "$QUANT_GPU_DEVICES" \
    >"$OUTPUT_DIR/validation/cuda-quantizer-exact.log" 2>&1 || {
    tail -n 100 "$OUTPUT_DIR/validation/cuda-quantizer-exact.log" >&2 || true
    die "CUDA quantizer/repack byte-exact suite failed"
}

current_phase=exact-api-correctness
"${clean_prefix[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/validation/cuda-exact.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/validation/cuda-exact.log" >&2 || true
    die "CUDA exact-output suite failed"
}
for marker in \
    'tagged SM75 native Q4 prefill 16/8/4 exact' \
    'tagged SM75 native Q4 decode exact' \
    'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exact.log" ||
        die "exact-output marker missing: $marker"
done

expected_native_bytes=$(./gguf-tools/deepseek4-quantize-cuda \
    --repack-sm75-native-q4 "$MODEL" --dry-run 2>/dev/null |
    awk -F': ' '$1 == "approx_file_bytes" {print $2}')
[[ $expected_native_bytes =~ ^[0-9]+$ ]] ||
    die "could not determine complete native-model size"
if [[ -e $NATIVE_MODEL && $(stat -c %s "$NATIVE_MODEL") != "$expected_native_bytes" ]]; then
    printf 'Removing incomplete native model (%s of %s bytes): %s\n' \
        "$(stat -c %s "$NATIVE_MODEL")" "$expected_native_bytes" "$NATIVE_MODEL"
    rm -f -- "$NATIVE_MODEL"
fi
if [[ ! -e $NATIVE_MODEL ]]; then
    current_phase=lossless-repack
    native_tmp="${NATIVE_MODEL}.partial.$$"
    printf 'Creating tagged SM75-native model on CUDA devices %s (no quantization or hashing)...\n' \
        "$QUANT_GPU_DEVICES"
    ./gguf-tools/deepseek4-quantize-cuda \
        --repack-sm75-native-q4 "$MODEL" --out "$native_tmp" \
        --quant-backend cuda --quant-gpu-devices "$QUANT_GPU_DEVICES" \
        --threads "$REPACK_THREADS" \
        2>&1 | tee "$OUTPUT_DIR/repack.log"
    [[ -s $native_tmp && $(stat -c %s "$native_tmp") == "$expected_native_bytes" ]] ||
        die "CUDA repack output is incomplete"
    mv -- "$native_tmp" "$NATIVE_MODEL"
    native_tmp=
else
    printf 'Reusing existing NATIVE_MODEL: %s\n' "$NATIVE_MODEL"
    printf 'reused=%s\n' "$NATIVE_MODEL" >"$OUTPUT_DIR/repack.log"
fi
[[ -s $NATIVE_MODEL ]] || die "native model is missing or empty"
[[ $(stat -c %s "$MODEL") == $(stat -c %s "$NATIVE_MODEL") ]] ||
    printf 'note: file sizes differ only because tagged GGUF metadata changed\n'

current_phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'standard_model=%s\nstandard_bytes=%s\n' "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'native_model=%s\nnative_bytes=%s\n' "$NATIVE_MODEL" "$(stat -c %s "$NATIVE_MODEL")"
    printf 'model_hashing=disabled\ngpu_devices=%s\nstage_split=22/21\n' "$GPU_DEVICES"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$REPEATS"
    printf 'exact_api_required=true\nexact_e2e_logits_skipped=%s\nbenchmark_skipped=%s\n' \
        "$SKIP_E2E_EXACT" "$SKIP_BENCHMARK"
    printf '\n[gpu]\n'; nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version --format=csv
    printf '\n[topology]\n%s\n' "$gpu_topology"
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

common=(--cuda --cuda-tensor-parallel --gpu-devices "$GPU_DEVICES"
        --gpu-vram "$GPU_VRAM" --prompt-file "$PROMPT"
        --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0)
validate_log() {
    local variant=$1 log=$2
    grep -Fq 'ds4: CUDA EP forced pipeline split 22/21' "$log" ||
        die "$variant did not use split 22/21"
    grep -Fq '4 devices [0,3,1,2] requested' "$log" ||
        die "$variant did not use GPU order 0,3,1,2"
    if [[ $variant == native ]]; then
        grep -Fq 'ds4: SM75 native routed-Q4 layout enabled' "$log" ||
            die "native model did not dispatch the tagged layout"
    elif grep -Fq 'ds4: SM75 native routed-Q4 layout enabled' "$log"; then
        die "standard model incorrectly dispatched the tagged layout"
    fi
}

if [[ $SKIP_E2E_EXACT == 0 ]]; then
    current_phase=end-to-end-exact-output
    for variant in standard native; do
        if [[ $variant == standard ]]; then run_model=$MODEL; else run_model=$NATIVE_MODEL; fi
        mkdir -p "$OUTPUT_DIR/validation/$variant-logits"
        "${production_prefix[@]}" DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
            ./ds4-bench "${common[@]}" --model "$run_model" \
            --ctx-start "$CTX_START" --ctx-max "$CTX_START" \
            --ctx-alloc "$((CTX_START + 1))" --step-incr "$CTX_START" \
            --dump-frontier-logits-dir "$OUTPUT_DIR/validation/$variant-logits" \
            --csv "$OUTPUT_DIR/validation/$variant.csv" \
            >"$OUTPUT_DIR/validation/$variant.log" 2>&1
        validate_log "$variant" "$OUTPUT_DIR/validation/$variant.log"
        count=$(grep -Ec '^ds4: routed-quant-audit layer=[0-9]+ gate=q4_k up=q4_k down=q4_k$' \
            "$OUTPUT_DIR/validation/$variant.log" || true)
        [[ $count == 43 ]] || die "$variant is not an exact 43-layer full-Q4 model"
    done
    raw=$(printf 'frontier_%06d.logits.f32' "$CTX_START")
    json=$(printf 'frontier_%06d.logits.json' "$CTX_START")
    a="$OUTPUT_DIR/validation/standard-logits/$raw"
    b="$OUTPUT_DIR/validation/native-logits/$raw"
    [[ -s $a && -s $b ]] || die "missing end-to-end logits: $raw"
    cmp -s "$a" "$b" || die "standard/native raw logits are not bit-exact: $raw"

    a="$OUTPUT_DIR/validation/standard-logits/$json"
    b="$OUTPUT_DIR/validation/native-logits/$json"
    [[ -s $a && -s $b ]] || die "missing end-to-end logits: $json"
    python3 - "$a" "$b" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    standard = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    native = json.load(stream)

# The model pathname is provenance, not an inference result.  It must differ
# when the standard and tagged-native GGUFs are separate files.
standard.pop("model", None)
native.pop("model", None)
if standard != native:
    differing = sorted(
        key for key in standard.keys() | native.keys()
        if standard.get(key) != native.get(key)
    )
    print(
        "error: standard/native semantic logits JSON differs in: "
        + ", ".join(differing),
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
    printf 'api_exact=true\ne2e_full_vocab_raw_logits_bit_exact=true\ne2e_logits_json_semantic_exact=true\n' \
        >"$OUTPUT_DIR/validation/exact-status.txt"
else
    current_phase=end-to-end-exact-output-skipped
    printf 'api_exact=true\ne2e_exact_skipped=true\ne2e_exact_evidence=previous-archived-run\n' \
        >"$OUTPUT_DIR/validation/exact-status.txt"
fi

if [[ $SKIP_BENCHMARK == 0 ]]; then
    current_phase=balanced-ab
    printf 'repeat\tslot\tvariant\tcsv\tlog\n' >"$OUTPUT_DIR/runs.tsv"
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        if (( repeat % 2 == 1 )); then variants=(standard native); else variants=(native standard); fi
        slot=0
        for variant in "${variants[@]}"; do
            slot=$((slot + 1))
            if [[ $variant == standard ]]; then run_model=$MODEL; else run_model=$NATIVE_MODEL; fi
            stem="$variant-r$repeat"
            printf 'Benchmarking %s repeat=%d/%d slot=%d...\n' "$variant" "$repeat" "$REPEATS" "$slot"
            "${production_prefix[@]}" "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
                ./ds4-bench "${common[@]}" --model "$run_model" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL" \
                --csv "$OUTPUT_DIR/runs/$stem.csv" \
                >"$OUTPUT_DIR/runs/$stem.log" 2>&1
            [[ -s $OUTPUT_DIR/runs/$stem.csv ]] || die "empty benchmark CSV: $stem"
            validate_log "$variant" "$OUTPUT_DIR/runs/$stem.log"
            printf '%s\t%s\t%s\t%s\t%s\n' "$repeat" "$slot" "$variant" \
                "$OUTPUT_DIR/runs/$stem.csv" "$OUTPUT_DIR/runs/$stem.log" \
                >>"$OUTPUT_DIR/runs.tsv"
            cat "$OUTPUT_DIR/runs/$stem.csv"
        done
    done

    python3 - "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/ab-summary.csv" <<'PY'
import csv, math, statistics, sys
records=[]
with open(sys.argv[1], encoding='utf-8', newline='') as f:
    runs=list(csv.DictReader(f, delimiter='\t'))
series={}
for run in runs:
    with open(run['csv'], encoding='utf-8', newline='') as f:
        rows=list(csv.DictReader(f))
    series[(int(run['repeat']),run['variant'])]={int(r['ctx_tokens']):float(r['prefill_tps']) for r in rows}
for repeat in sorted({int(r['repeat']) for r in runs}):
    a=series[(repeat,'standard')]; b=series[(repeat,'native')]
    if a.keys()!=b.keys(): raise SystemExit('A/B frontier mismatch')
    for ctx in a:
        ratio=b[ctx]/a[ctx]
        if not math.isfinite(ratio): raise SystemExit('non-finite ratio')
        records.append({'repeat':repeat,'ctx_tokens':ctx,'standard_tps':a[ctx],
                        'native_tps':b[ctx],'native_over_standard':ratio})
with open(sys.argv[2], 'w', encoding='utf-8', newline='') as f:
    w=csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
print('ctx_tokens,median_native_over_standard,median_gain_pct')
for ctx in sorted({r['ctx_tokens'] for r in records}):
    ratios=[r['native_over_standard'] for r in records if r['ctx_tokens']==ctx]
    med=statistics.median(ratios)
    print(f'{ctx},{med:.6f},{(med-1)*100:.3f}')
PY
else
    current_phase=balanced-ab-skipped
    printf 'benchmark_skipped=true\nevidence=previous-archived-run\n' \
        >"$OUTPUT_DIR/benchmark-status.txt"
fi

if [[ $RUN_NSYS == 1 ]]; then
    current_phase=nsight-systems
    "${production_prefix[@]}" DS4_NSYS_CAPTURE_PREFILL=1 \
        nsys profile --force-overwrite=true --sample=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$OUTPUT_DIR/nsys/native-full-q4" \
        ./ds4-bench "${common[@]}" --model "$NATIVE_MODEL" \
        --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
        --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
        --csv "$OUTPUT_DIR/nsys/benchmark.csv" \
        >"$OUTPUT_DIR/nsys/capture.log" 2>&1
    [[ -s $OUTPUT_DIR/nsys/native-full-q4.nsys-rep ]] || die "missing Nsight Systems report"
    nsys stats --report cuda_gpu_kern_sum --format csv \
        "$OUTPUT_DIR/nsys/native-full-q4.nsys-rep" \
        >"$OUTPUT_DIR/nsys/cuda_gpu_kern_sum.csv" 2>"$OUTPUT_DIR/nsys/stats.log"
fi

if [[ $RUN_NCU == 1 ]]; then
    current_phase=native-profile-harness-smoke
    for scenario in native-q4-early native-q4-late; do
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            ./tests/cuda_sm75_profile_harness "$scenario" \
            >"$OUTPUT_DIR/validation/$scenario-profile-harness.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/validation/$scenario-profile-harness.log" >&2 || true
            die "production-shaped native-Q4 profile harness failed: $scenario"
        }
        grep -Fq 'harness_status=ok' \
            "$OUTPUT_DIR/validation/$scenario-profile-harness.log" ||
            die "production-shaped native-Q4 harness omitted success: $scenario"
    done
    current_phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_cmd=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo -v
        ncu_cmd=(sudo -E "$ncu_bin")
    fi
    sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
              --section SchedulerStats --section WarpStateStats
              --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    profile_kernel() {
        local name scenario regex launch_skip expected base rc
        name=$1
        scenario=$2
        regex=$3
        launch_skip=$4
        expected=$5
        base="$OUTPUT_DIR/ncu/$name"
        rc=0
        printf 'Nsight Compute: %s...\n' "$name"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_cmd[@]}" --config-file off \
            --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "regex:$regex" \
            --launch-skip "$launch_skip" --launch-count 1 \
            --replay-mode kernel --cache-control none \
            --clock-control none --force-overwrite --export "$base" \
            "${sections[@]}" ./tests/cuda_sm75_profile_harness "$scenario" \
            >"$base.log" 2>&1 || rc=$?
        (( rc == 0 )) || { tail -n 100 "$base.log" >&2 || true; die "ncu failed: $name"; }
        [[ -s $base.ncu-rep ]] || die "missing ncu report: $name"
        if [[ $NCU_USE_SUDO == 1 ]]; then sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"; fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log"
        grep -Eq "$expected" "$base.csv" ||
            die "ncu captured the wrong production specialization: $name"
    }
    profile_histogram() {
        local label scenario
        label=$1
        scenario=$2
        profile_kernel "$label-gate-tile16-low8" "$scenario" \
            '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' 0 \
            'moe_gate_up_mid_sm75_native_q4_tile8_kernel<.*512.*0>'
        profile_kernel "$label-gate-tile16-high8" "$scenario" \
            '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' 1 \
            'moe_gate_up_mid_sm75_native_q4_tile8_kernel<.*512.*8>'
        profile_kernel "$label-gate-tail8" "$scenario" \
            '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' 2 \
            'moe_gate_up_mid_sm75_native_q4_tile8_kernel<.*512.*0>'
        profile_kernel "$label-gate-tail4" "$scenario" \
            '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' 3 \
            'moe_gate_up_mid_sm75_native_q4_tile8_kernel<.*512.*0>'
        profile_kernel "$label-down-tile16" "$scenario" \
            '^moe_down_sm75_native_q4_tile_kernel$' 0 \
            'moe_down_sm75_native_q4_tile_kernel<.*512.*16>'
        profile_kernel "$label-down-tail8" "$scenario" \
            '^moe_down_sm75_native_q4_tile_kernel$' 1 \
            'moe_down_sm75_native_q4_tile_kernel<.*512.*8>'
        profile_kernel "$label-down-tail4" "$scenario" \
            '^moe_down_sm75_native_q4_tile_kernel$' 2 \
            'moe_down_sm75_native_q4_tile_kernel<.*512.*4>'
    }
    profile_histogram early native-q4-early
    profile_histogram late native-q4-late
fi

current_phase=complete
printf 'SM75 native-Q4 production A/B and profile complete: %s\n' "$OUTPUT_DIR"
