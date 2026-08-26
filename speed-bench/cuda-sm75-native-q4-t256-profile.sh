#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Profile the accepted SM75 native-Q4 + all-partner T256 production path.

The full 153 GiB model is opened exactly once for a bounded 2K Nsight Systems
capture. Nsight Compute uses small production-shaped harnesses and therefore
does not reread the GGUF for each metric pass.

Required environment:
  NATIVE_MODEL=/absolute/path/to/tagged-SM75-native-full-Q4.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  PROFILE_TOKENS=2048
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  PROFILE_GPU=0                physical GPU for native-Q4 NCU harnesses
  PROFILE_PARTNER_GPU=1        its NVLink partner for T256 cuBLAS NCU
  RUN_NCU=1
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  COMBINED_PROFILE_DIR=...     new output directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${NATIVE_MODEL:?set NATIVE_MODEL to the absolute tagged SM75-native Q4 GGUF}"
[[ $NATIVE_MODEL == /* && -f $NATIVE_MODEL ]] ||
    die "NATIVE_MODEL must name an existing absolute file"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
PROFILE_GPU=${PROFILE_GPU:-0}
PROFILE_PARTNER_GPU=${PROFILE_PARTNER_GPU:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${COMBINED_PROFILE_DIR:-$repo_dir/sm75-native-q4-t256-profile-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
            "PROFILE_GPU:$PROFILE_GPU" \
            "PROFILE_PARTNER_GPU:$PROFILE_PARTNER_GPU" \
            "RUN_NCU:$RUN_NCU" "NCU_USE_SUDO:$NCU_USE_SUDO" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 )) ||
    die "STAGE_SPLIT must be in 1..42"
(( PROFILE_TOKENS == 2048 && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 )) ||
    die "the production-shaped profile must remain 2048/2048/512 tokens"
for flag in RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $PROFILE_GPU != "$PROFILE_PARTNER_GPU" ]] ||
    die "PROFILE_GPU and PROFILE_PARTNER_GPU must differ"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk basename cat cmp date dirname env find git grep id make mkdir \
            mktemp mv nproc nsys nvidia-smi python3 rm sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
(( RUN_NCU == 0 )) || command -v ncu >/dev/null 2>&1 || die "ncu not found"
(( RUN_NCU == 0 || NCU_USE_SUDO == 0 )) ||
    command -v sudo >/dev/null 2>&1 || die "sudo not found"

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
[[ -n ${seen[$PROFILE_GPU]+x} && -n ${seen[$PROFILE_PARTNER_GPU]+x} ]] ||
    die "profile pair must be present in GPU_DEVICES"

topology_file=$(mktemp)
nvidia-smi topo -m >"$topology_file"
topology_link() {
    local source=$1 destination=$2
    awk -v row="GPU$source" -v column="GPU$destination" '
        NR == 1 {
            for (i = 1; i <= NF; i++) if ($i == column) target = i
            next
        }
        $1 == row && target > 0 && target <= NF {print $target; exit}
    ' "$topology_file"
}
forward=$(topology_link "$PROFILE_GPU" "$PROFILE_PARTNER_GPU")
reverse=$(topology_link "$PROFILE_PARTNER_GPU" "$PROFILE_GPU")
[[ $forward =~ ^NV[0-9]+$ || $reverse =~ ^NV[0-9]+$ ]] ||
    die "profile pair $PROFILE_GPU<->$PROFILE_PARTNER_GPU is not NVLink: ${forward:-missing}/${reverse:-missing}"
for pair in 0:2 1:3; do
    home_pos=${pair%%:*}; partner_pos=${pair#*:}
    home=${gpu_ids[$home_pos]}; partner=${gpu_ids[$partner_pos]}
    fwd=$(topology_link "$home" "$partner")
    rev=$(topology_link "$partner" "$home")
    [[ $fwd =~ ^NV[0-9]+$ || $rev =~ ^NV[0-9]+$ ]] ||
        die "logical pair $home_pos<->$partner_pos (physical $home<->$partner) is not NVLink: ${fwd:-missing}/${rev:-missing}"
    [[ -z $fwd || $fwd =~ ^NV[0-9]+$ ]] ||
        die "inconsistent topology for physical pair $home<->$partner: $fwd/${rev:-missing}"
    [[ -z $rev || $rev =~ ^NV[0-9]+$ ]] ||
        die "inconsistent topology for physical pair $home<->$partner: ${fwd:-missing}/$rev"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{nsys,ncu,validation,telemetry,provenance}
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
    rm -f -- "$topology_file"
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
    DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256
    DS4_CUDA_Q8_F16_PARTNER_LAYERS=0-42
    DS4_CUDA_Q8_PARTNER_ARITHMETIC=f16
    DS4_CUDA_Q8_T256_PLACEMENT=all-partner
)

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke
         tests/test_engine_mgpu_placement tests/test_gpu_xdev
         tests/cuda_sm75_profile_harness)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/validation/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=exactness
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/validation/planner.log" 2>&1 || {
        tail -n 120 "$OUTPUT_DIR/validation/planner.log" >&2 || true
        die "planner tests failed"
    }
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/validation/cuda-exactness.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/validation/cuda-exactness.log" >&2 || true
        die "CUDA exactness tests failed"
    }
for marker in 'tagged SM75 native Q4 cost-planner default exact' \
              'tagged SM75 native Q4 decode exact' \
              'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exactness.log" ||
        die "missing exactness marker: $marker"
done
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PROFILE_PARTNER_GPU" \
    ./tests/test_gpu_xdev q8-partner-t256-profile \
    >"$OUTPUT_DIR/validation/partner-t256-harness.log" 2>&1 || {
        tail -n 120 "$OUTPUT_DIR/validation/partner-t256-harness.log" >&2 || true
        die "partner T256 profile harness failed"
    }
grep -Fq 'harness_status=ok' \
    "$OUTPUT_DIR/validation/partner-t256-harness.log" ||
    die "partner T256 harness omitted success"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'native_model=%s\nnative_model_bytes=%s\nmodel_hashing=disabled\n' \
        "$NATIVE_MODEL" "$(stat -c %s "$NATIVE_MODEL")"
    printf 'prompt=%s\ngpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" \
        "$((43-STAGE_SPLIT))"
    printf 'profile_tokens=%s\nprefill_chunk=%s\npipeline_mb=%s\n' \
        "$PROFILE_TOKENS" "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 't256_policy=all-partner\nt256_layers=0-42\npartner_arithmetic=f16\n'
    printf 'profile_gpu=%s\nprofile_partner_gpu=%s\nrun_ncu=%s\n' \
        "$PROFILE_GPU" "$PROFILE_PARTNER_GPU" "$RUN_NCU"
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
        --format=csv
    printf '\n[topology]\n'; cat "$topology_file"
    printf '\n[nsight systems]\n'; nsys --version
    if [[ $RUN_NCU == 1 ]]; then printf '\n[nsight compute]\n'; ncu --version; fi
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
env | awk -F= '$1 ~ /^DS4_/ {print}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

phase=nsight-systems
base="$OUTPUT_DIR/nsys/combined"
bindings="$OUTPUT_DIR/nsys/bindings.csv"
cache_before="$OUTPUT_DIR/nsys/cache-before.csv"
cache_after="$OUTPUT_DIR/nsys/cache-after.csv"
printf 'Capturing one native-Q4/all-partner-T256 production trace...\n'
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,\
temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits -lms 200 \
    >"$OUTPUT_DIR/telemetry/combined.csv" &
sampler_pid=$!
run_rc=0
"${production_env[@]}" \
    DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
    DS4_CUDA_CRITICAL_PATH_NVTX=1 \
    "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$PROFILE_TOKENS" \
    DS4_NSYS_CAPTURE_PREFILL=1 \
    "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$cache_before" \
    "DS4_CUDA_Q8_CACHE_STATE_CSV=$cache_after" \
    "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
    nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$base" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$NATIVE_MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS+1))" \
            --step-incr "$PROFILE_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
            --gen-tokens 0 --csv "$base-benchmark.csv" \
            >"$base.log" 2>&1 || run_rc=$?
cleanup_sampler
(( run_rc == 0 )) || {
    tail -n 200 "$base.log" >&2 || true
    die "combined Nsight Systems capture failed (exit $run_rc)"
}
[[ -s $base.nsys-rep && -s $base-benchmark.csv ]] ||
    die "Nsight Systems omitted its report or benchmark"
grep -Fqx 'ds4: SM75 native routed-Q4 layout enabled (packed A/W, planner=cost, gate=tile8, down=full-stage)' \
    "$base.log" || die "production trace did not use the accepted native-Q4 path"
for marker in "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
              'partner-classes=t256' 'partner-layers=0-42' \
              't256-placement=all-partner' 'T256-output_b=86/86' \
              'partner=43 partner-arithmetic=f16' \
              'CUDA q8 fp16 partner summary: calls=344'; do
    grep -Fq "$marker" "$base.log" ||
        die "production trace lacks required marker: $marker"
done
for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
    grep -Fq "$route" "$base.log" ||
        die "production trace lacks validated direct route $route"
done
recipe_count=$(grep -Ec \
    '^ds4: routed-quant-audit layer=[0-9]+ gate=q4_k up=q4_k down=q4_k$' \
    "$base.log" || true)
[[ $recipe_count == 43 ]] ||
    die "production trace has $recipe_count/43 exact full-Q4 recipe records"
cmp -s "$cache_before" "$cache_after" ||
    die "Q8 cache state changed during the captured frontier"
awk -F, '
    NR > 1 && $6 == 8192 && $7 == 4096 && $12 ~ /attn_output_b/ {
        if ($3 == 1) partner++; else fixed++; next
    }
    NR > 1 && $3 == 1 {bad_partner=1}
    END {exit !(fixed == 43 && partner == 43 && !bad_partner)}
' "$bindings" || die "binding inventory is not the exact all-partner T256 plan"

phase=nsight-systems-export
nsys export --type sqlite --force-overwrite=true \
    --output "$base.sqlite" "$base.nsys-rep" \
    >"$base-export.log" 2>&1 || die "Nsight SQLite export failed"
printf 'label\tdevices\tsqlite\ncombined\t%s\t%s\n' \
    "$GPU_DEVICES" "$base.sqlite" >"$OUTPUT_DIR/trace-map.tsv"
printf 'trial\tslot\tscenario\tdevice\tlog\n' >"$OUTPUT_DIR/harness-runs.tsv"
for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum \
              cuda_gpu_trace nvtx_sum nvtx_gpu_proj_sum; do
    nsys stats --report "$report" --format csv "$base.nsys-rep" \
        >"$base-$report.csv" 2>"$base-$report.log" || true
done
python3 speed-bench/summarize-cuda-critical-path.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/critical-path-summary.txt"
python3 speed-bench/summarize-sm75-native-q4-t256-profile.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/combined-summary.txt"

if [[ $RUN_NCU == 1 ]]; then
    phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_cmd=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo -v
        ncu_cmd=(sudo -E "$ncu_bin")
    fi
    sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
              --section SchedulerStats --section WarpStateStats
              --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)

    profile_native() {
        local label=$1 scenario=$2 kernel=$3 expected=$4 base=$5 rc=0
        printf 'Nsight Compute: %s...\n' "$label"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_cmd[@]}" \
            --config-file off --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "regex:$kernel" \
            --launch-skip 0 --launch-count 1 --replay-mode kernel \
            --cache-control none --clock-control none --force-overwrite \
            --export "$base" "${sections[@]}" \
            ./tests/cuda_sm75_profile_harness "$scenario" \
            >"$base.log" 2>&1 || rc=$?
        (( rc == 0 )) || { tail -n 120 "$base.log" >&2 || true; die "ncu failed: $label"; }
        [[ -s $base.ncu-rep ]] || die "ncu omitted report: $label"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" \
            --csv --page raw >"$base.csv" 2>"$base-import.log"
        python3 speed-bench/validate-ncu-capture.py \
            "$base.csv" "$expected" 0 \
            --process cuda_sm75_profile_harness --block-size 512
    }

    profile_native early-native-q4-gate native-q4-early \
        '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' \
        'moe_gate_up_mid_sm75_native_q4_tile8_kernel' \
        "$OUTPUT_DIR/ncu/early-native-q4-gate"
    profile_native late-native-q4-gate native-q4-late \
        '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' \
        'moe_gate_up_mid_sm75_native_q4_tile8_kernel' \
        "$OUTPUT_DIR/ncu/late-native-q4-gate"
    profile_native early-native-q4-down native-q4-early \
        '^moe_down_sm75_native_q4_tile_kernel$' \
        'moe_down_sm75_native_q4_tile_kernel' \
        "$OUTPUT_DIR/ncu/early-native-q4-down"
    profile_native late-native-q4-down native-q4-late \
        '^moe_down_sm75_native_q4_tile_kernel$' \
        'moe_down_sm75_native_q4_tile_kernel' \
        "$OUTPUT_DIR/ncu/late-native-q4-down"

    partner_base="$OUTPUT_DIR/ncu/partner-t256-cublas"
    printf 'Nsight Compute: production-shaped partner T256 cuBLAS...\n'
    rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PROFILE_PARTNER_GPU" \
        "${ncu_cmd[@]}" --config-file off \
        --target-processes application-only --devices 1 \
        --profile-from-start off --launch-count 1 --replay-mode kernel \
        --cache-control none --clock-control none --force-overwrite \
        --export "$partner_base" "${sections[@]}" \
        ./tests/test_gpu_xdev q8-partner-t256-profile \
        >"$partner_base.log" 2>&1 || rc=$?
    (( rc == 0 )) || {
        tail -n 120 "$partner_base.log" >&2 || true
        die "ncu failed: partner T256 cuBLAS"
    }
    [[ -s $partner_base.ncu-rep ]] ||
        die "ncu omitted partner T256 report"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$partner_base.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$partner_base.ncu-rep" \
        --csv --page raw >"$partner_base.csv" 2>"$partner_base-import.log"
    python3 speed-bench/validate-ncu-capture.py \
        "$partner_base.csv" '(?i)(gemm|mma)' 1 --process test_gpu_xdev
fi

phase=complete
for required in summary.md combined-profile.json combined-kernel-groups.csv \
                partner-t256-ranges.csv stage-device-summary.csv \
                partner-projection-summary.csv nsys/combined.nsys-rep; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing final evidence: $required"
done
printf 'Combined SM75 production profile complete: %s\n' "$OUTPUT_DIR"
