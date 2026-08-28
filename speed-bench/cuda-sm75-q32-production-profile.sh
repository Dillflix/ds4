#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture one genuine 32K production trace for the tagged SM75 Q4-32/Q3A4
model, then profile its routed and long-context attention kernels with bounded
exact harnesses. The production GGUF is opened only once.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  PROFILE_TOKENS=32768         fixed to 32768 for this evidence pass
  CTX_ALLOC=32769
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  PROFILE_GPU=0                physical GPU for bounded NCU harnesses
  RUN_NCU=1
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PROFILE_DIR=...              output directory
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
PROFILE_TOKENS=${PROFILE_TOKENS:-32768}
CTX_ALLOC=${CTX_ALLOC:-32769}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
PROFILE_GPU=${PROFILE_GPU:-0}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
Q3A4_LAYERS=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PROFILE_DIR:-$repo_dir/sm75-q32-production-profile-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged Q4-32/Q3A4 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "CTX_ALLOC:$CTX_ALLOC" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "PIPELINE_MB:$PIPELINE_MB" "PROFILE_GPU:$PROFILE_GPU" \
            "RUN_NCU:$RUN_NCU" "NCU_USE_SUDO:$NCU_USE_SUDO" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 )) || die "this fixed production profile requires STAGE_SPLIT=22"
(( PROFILE_TOKENS == 32768 )) || die "PROFILE_TOKENS must be exactly 32768"
(( CTX_ALLOC == 32769 && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 )) ||
    die "require ctx_alloc=32769, prefill_chunk=2048, and pipeline_mb=512"
printf -v frontier_tag '%06d' "$PROFILE_TOKENS"
for flag in RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk basename cat cmp date dirname env git grep id make mkdir \
            mktemp mv ncu nproc nsys nvidia-smi python3 rm sort stat tail \
            tar tee tr; do
    if [[ $tool == ncu && $RUN_NCU == 0 ]]; then continue; fi
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
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
[[ -n ${seen[$PROFILE_GPU]+x} ]] || die "PROFILE_GPU must be in GPU_DEVICES"

topology_file=$(mktemp)
nvidia-smi topo -m >"$topology_file"
topology_link() {
    local from=$1 to=$2
    awk -v from="GPU$from" -v to="GPU$to" '
        !header {
            n=0
            for (i=1;i<=NF;i++) if ($i ~ /^GPU[0-9]+$/) n++
            if (n>1) {
                for (i=1;i<=NF;i++) if ($i==to) column=i+1
                header=1; next
            }
        }
        header && $1==from && column>0 && column<=NF {print $column; exit}
    ' "$topology_file"
}
for pair in 0:2 1:3; do
    home=${gpu_ids[${pair%%:*}]}; partner=${gpu_ids[${pair#*:}]}
    fwd=$(topology_link "$home" "$partner")
    rev=$(topology_link "$partner" "$home")
    [[ $fwd =~ ^NV[0-9]+$ || $rev =~ ^NV[0-9]+$ ]] ||
        die "physical pair $home<->$partner has no NVLink topology cell: ${fwd:-missing}/${rev:-missing}"
    [[ -z $fwd || $fwd =~ ^NV[0-9]+$ ]] ||
        die "physical pair $home<->$partner has a non-NVLink forward cell: $fwd/${rev:-missing}"
    [[ -z $rev || $rev =~ ^NV[0-9]+$ ]] ||
        die "physical pair $home<->$partner has a non-NVLink reverse cell: ${fwd:-missing}/$rev"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{nsys/frontier-logits,ncu,validation,telemetry,provenance}
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
    DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1
    DS4_CUDA_INDEXER_SCORE_AUDIT=1
)

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke
         tests/test_engine_mgpu_placement tests/cuda_sm75_profile_harness)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/validation/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
[[ -x ./ds4-bench && -x ./tests/cuda_sm75_profile_harness ]] ||
    die "required binaries are missing"

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
for marker in \
    'SM75 Q4-32 gate/up + Q4-32 down production 16/8/4 prefill/direct-decode exact' \
    'SM75 Q3A4 gate/up + Q4-32 down production 16/8/4 prefill/direct-decode exact' \
    'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exactness.log" ||
        die "missing exactness marker: $marker"
done

ncu_bin=
ncu_cmd=()
sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
          --section SchedulerStats --section WarpStateStats
          --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
if [[ $RUN_NCU == 1 ]]; then
    phase=nsight-compute-preflight
    ncu_bin=$(command -v ncu)
    ncu_cmd=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo -v
        ncu_cmd=(sudo -E "$ncu_bin")
    fi
    printf 'Nsight Compute permission preflight...\n'
    preflight="$OUTPUT_DIR/validation/ncu-preflight"
    rc=0
    "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_cmd[@]}" --config-file off --target-processes application-only \
        --devices 0 --kernel-name-base function \
        --kernel-name 'regex:^matmul_q8_0_mma_sm75_exact_kernel.*' \
        --launch-count 1 --replay-mode kernel --cache-control none \
        --clock-control none --force-overwrite --export "$preflight" \
        --section SpeedOfLight --section LaunchStats \
        ./tests/cuda_sm75_profile_harness q8-shared \
        >"$preflight.log" 2>&1 || rc=$?
    (( rc == 0 )) || {
        tail -n 100 "$preflight.log" >&2 || true
        die "Nsight Compute permission preflight failed"
    }
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$preflight.log"; then
        tail -n 100 "$preflight.log" >&2 || true
        die "Nsight Compute preflight captured no usable kernel"
    fi
    [[ -s $preflight.ncu-rep ]] || die "Nsight Compute preflight omitted report"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$preflight.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$preflight.ncu-rep" \
        --csv --page raw >"$preflight.csv" 2>"$preflight-import.log"
    python3 speed-bench/validate-ncu-capture.py \
        "$preflight.csv" 'matmul_q8_0_mma_sm75_exact_kernel' 0 \
        --process cuda_sm75_profile_harness --block-size 256
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'profile_tokens=%s\nctx_alloc=%s\nprefill_chunk=%s\npipeline_mb=%s\n' \
        "$PROFILE_TOKENS" "$CTX_ALLOC" "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'q3a4_layers=%s\nq4_32_gate_up_layers=remaining-28\nq4_32_down_layers=all-43\n' \
        "$Q3A4_LAYERS"
    printf 'dense_f16_policy=stage-aware-fixed-22-21\nprofile_gpu=%s\nrun_ncu=%s\n' \
        "$PROFILE_GPU" "$RUN_NCU"
    printf 'dense_f16_admission=344/344\nattention_rows_policy=fixed-50-50\n'
    printf 'indexer_rows_policy=fixed-50-50-pair-split\n'
    printf 'indexer_cache=native-f16\nindexer_scorer=streaming64\nxdev_sync=disabled\n'
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
allocations="$OUTPUT_DIR/nsys/allocations.csv"
memory="$OUTPUT_DIR/nsys/memory.csv"
plan="$OUTPUT_DIR/nsys/plan.csv"
q8_audit="$OUTPUT_DIR/nsys/q8-cache-audit.csv"
tile_audit="$OUTPUT_DIR/nsys/routed-tile-audit.csv"
cache_before="$OUTPUT_DIR/nsys/cache-before.csv"
cache_after="$OUTPUT_DIR/nsys/cache-after.csv"
printf 'Capturing one genuine 32K Q4-32/Q3A4 production trace...\n'
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,\
temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits -lms 200 \
    >"$OUTPUT_DIR/telemetry/combined.csv" &
sampler_pid=$!
run_rc=0
"${production_env[@]}" \
    DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
    DS4_CUDA_CRITICAL_PATH_NVTX=1 \
    "DS4_CUDA_PREFILL_TILE_AUDIT_CSV=$tile_audit" \
    DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY=65536 \
    "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$PROFILE_TOKENS" \
    DS4_NSYS_CAPTURE_PREFILL=1 \
    "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$q8_audit" \
    "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$plan" \
    "DS4_CUDA_Q8_ALLOCATION_STATE_CSV=$allocations" \
    "DS4_CUDA_MEMORY_STATE_CSV=$memory" \
    "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$cache_before" \
    "DS4_CUDA_Q8_CACHE_STATE_CSV=$cache_after" \
    "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
    nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$base" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$CTX_ALLOC" --step-incr "$PROFILE_TOKENS" \
            --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
            --csv "$base-benchmark.csv" \
            --dump-frontier-logits-dir "$OUTPUT_DIR/nsys/frontier-logits" \
            >"$base.log" 2>&1 || run_rc=$?
cleanup_sampler
(( run_rc == 0 )) || {
    tail -n 220 "$base.log" >&2 || true
    die "32K Nsight Systems capture failed (exit $run_rc)"
}
[[ -s $base.nsys-rep && -s $base-benchmark.csv ]] ||
    die "Nsight Systems omitted its report or benchmark"
[[ -s $OUTPUT_DIR/nsys/frontier-logits/frontier_${frontier_tag}.logits.f32 &&
   -s $OUTPUT_DIR/nsys/frontier-logits/frontier_${frontier_tag}.logits.json ]] ||
    die "production trace omitted the ${PROFILE_TOKENS}-token frontier logits"
for marker in "CUDA EP forced pipeline split 22/21" \
              'partner-classes=automatic:t32,t256,shared_down' \
              't256-placement=stage-aware' \
              'dense-placement=stage-aware-fixed-22-21' \
              'CUDA q8 fp16 stage-aware 22/21 planner selected' \
              'materialized 344/344 candidates' \
              'T256-output_b=43/43' \
              'SM75 routed Q32 layout enabled' \
              'CUDA q8 fp16 partner summary: calls='; do
    grep -Fq "$marker" "$base.log" ||
        die "production trace lacks required marker: $marker"
done
for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
    grep -Fq "$route" "$base.log" || die "trace lacks direct route $route"
done
[[ $(grep -Fc 'qualified=yes' "$base.log") == 2 ]] ||
    die "trace did not qualify both SM75 NVLink pairs"
[[ $(grep -Fc 'CUDA prefill attention query-row split enabled:' "$base.log") == 2 ]] ||
    die "trace did not select both production row-split stages"
[[ $(grep -Fc 'CUDA prefill indexer score/top-k row split enabled:' "$base.log") == 2 ]] ||
    die "trace did not split indexer score/top-k on both production stages"
[[ $(grep -Fc 'rows [0,256), tier ' "$base.log") == 4 ]] ||
    die "trace did not use fixed 256/256 indexer and attention ownership on both pairs"
! grep -Fq 'CUDA prefill attention row calibration' "$base.log" ||
    die "trace unexpectedly used removed runtime row calibration"
grep -Fq 'dispatch=split kind=mixed' "$base.log" ||
    die "trace omitted mixed row-split attention"
grep -Fq 'dispatch=split kind=indexed' "$base.log" ||
    die "trace omitted indexed row-split attention"
! grep -Fq 'required but unavailable' "$base.log" ||
    die "trace encountered an unavailable eligible row split"
grep -Fq 'SM75 native F16 indexer cache and streaming WMMA64 enabled' \
    "$base.log" || die "trace did not enable the native indexer path"
for dispatch in direct-one streaming64-native wmma128-f16-native wmma128 \
                wmma64 wmma32 wmma16 generic; do
    audit_line=$(grep -F "ds4: CUDA indexer score audit dispatch=$dispatch " \
        "$base.log" || true)
    [[ $(printf '%s\n' "$audit_line" | grep -c .) == 1 ]] ||
        die "trace lacks one unambiguous $dispatch indexer audit record"
    audit_launches=${audit_line##*launches=}
    [[ $audit_launches =~ ^[0-9]+$ ]] ||
        die "trace has an invalid $dispatch indexer audit count"
    if [[ $dispatch == streaming64-native ]]; then
        (( audit_launches > 0 )) || die "trace did not dispatch streaming64-native"
    else
        (( audit_launches == 0 )) ||
            die "trace unexpectedly dispatched $dispatch $audit_launches times"
    fi
done
python3 speed-bench/validate-sm75-q32-production-log.py \
    "$base.log" "$Q3A4_LAYERS"
awk -F, -v want="$PROFILE_TOKENS" '
    NR==2 {rows++; if (($1+0)!=want || ($2+0)!=want) bad++}
    NR>2 {rows++; bad++}
    END {exit !(rows==1 && bad==0)}
' "$base-benchmark.csv" || die "trace benchmark is not one genuine 32K frontier"
awk -F, 'NR>1 {if ($10!=$11 || $13!=0 || $14!=1) bad++; rows++}
    END {exit !(rows==344 && bad==0)}' "$allocations" ||
    die "dense-F16 allocation inventory is incomplete"
awk -F, 'NR>1 {if ($3+0 < 512*1048576) bad++; rows++}
    END {exit !(rows==4 && bad==0)}' "$memory" ||
    die "trace left less than 512 MiB free on a CUDA device"
cmp -s "$cache_before" "$cache_after" ||
    die "dense-F16 cache state changed during the measured frontier"
python3 speed-bench/validate-sm75-stage-aware-plan.py \
    "$plan" "$bindings" "$GPU_DEVICES" \
    "$OUTPUT_DIR/dense-stage-aware-summary.csv"

phase=nsight-systems-export
nsys export --type sqlite --force-overwrite=true \
    --output "$base.sqlite" "$base.nsys-rep" \
    >"$base-export.log" 2>&1 || die "Nsight SQLite export failed"
printf 'label\tdevices\tsqlite\nproduction-32k\t%s\t%s\n' \
    "$GPU_DEVICES" "$base.sqlite" >"$OUTPUT_DIR/trace-map.tsv"
printf 'trial\tslot\tscenario\tdevice\tlog\n' >"$OUTPUT_DIR/harness-runs.tsv"
for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum \
              cuda_gpu_trace nvtx_sum nvtx_gpu_proj_sum; do
    nsys stats --report "$report" --format csv "$base.nsys-rep" \
        >"$base-$report.csv" 2>"$base-$report.log" || true
done
python3 speed-bench/summarize-cuda-critical-path.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/critical-path-summary.txt"
python3 speed-bench/summarize-sm75-q32-production-profile.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/q32-summary.txt"

profile_one() {
    local label=$1 scenario=$2 kernel=$3 expected=$4 block_size=$5
    local out="$OUTPUT_DIR/ncu/$label" rc=0
    printf 'Nsight Compute: %s...\n' "$label"
    "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_cmd[@]}" --config-file off \
        --target-processes application-only --devices 0 \
        --kernel-name-base function --kernel-name "regex:$kernel" \
        --launch-skip 0 --launch-count 1 --replay-mode kernel \
        --cache-control none --clock-control none --force-overwrite \
        --export "$out" "${sections[@]}" \
        ./tests/cuda_sm75_profile_harness "$scenario" \
        >"$out.log" 2>&1 || rc=$?
    (( rc == 0 )) || {
        tail -n 120 "$out.log" >&2 || true
        die "Nsight Compute failed: $label"
    }
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$out.log"; then
        tail -n 120 "$out.log" >&2 || true
        die "Nsight Compute captured no usable kernel: $label"
    fi
    [[ -s $out.ncu-rep ]] || die "Nsight Compute omitted report: $label"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$out.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$out.ncu-rep" \
        --csv --page raw >"$out.csv" 2>"$out-import.log"
    python3 speed-bench/validate-ncu-capture.py \
        "$out.csv" "$expected" 0 \
        --process cuda_sm75_profile_harness --block-size "$block_size"
}

if [[ $RUN_NCU == 1 ]]; then
    phase=nsight-compute
    profile_one q4-32-gate-up sm75-q4-32 \
        '^moe_gate_up_mid_sm75_q32_tile8_kernel.*' \
        'moe_gate_up_mid_sm75_q32_tile8_kernel' 512
    profile_one q4-32-down sm75-q4-32 \
        '^moe_down_sm75_q4_32_tile_kernel.*' \
        'moe_down_sm75_q4_32_tile_kernel' 256
    profile_one q3a4-gate-up sm75-q3a4 \
        '^moe_gate_up_mid_sm75_q32_tile8_kernel.*' \
        'moe_gate_up_mid_sm75_q32_tile8_kernel' 512
    profile_one attention-indexed-32k attn-indexed-32k \
        '^attention_indexed_mixed_heads8_online_kernel.*' \
        'attention_indexed_mixed_heads8_online_kernel' 512
    profile_one attention-mixed-32k attn-mixed-32k \
        '^attention_decode_mixed_heads8_online_kernel.*' \
        'attention_decode_mixed_heads8_online_kernel' 256
fi

phase=complete
required=(profile-summary.md kernel-family-summary.csv kernel-family-total.csv
          operation-family-summary.csv
          operation-attribution.csv stage-microbatch-device.csv
          stage-device-summary.csv layer-device-summary.csv
          partner-projection-summary.csv handoff-device-summary.csv
          attention-row-split-summary.csv indexer-row-split-summary.csv
          dense-stage-aware-summary.csv
          trace-summary.csv nsys/combined.nsys-rep nsys/combined.sqlite
          nsys/combined-benchmark.csv nsys/routed-tile-audit.csv
          nsys/bindings.csv nsys/allocations.csv nsys/memory.csv
          nsys/frontier-logits/frontier_${frontier_tag}.logits.f32
          nsys/frontier-logits/frontier_${frontier_tag}.logits.json)
if [[ $RUN_NCU == 1 ]]; then
    required+=(ncu/q4-32-gate-up.ncu-rep ncu/q4-32-down.ncu-rep
               ncu/q3a4-gate-up.ncu-rep ncu/attention-indexed-32k.ncu-rep
               ncu/attention-mixed-32k.ncu-rep)
fi
for item in "${required[@]}"; do
    [[ -s $OUTPUT_DIR/$item ]] || die "missing final evidence: $item"
done
printf 'SM75 Q4-32/Q3A4 32K production profile complete: %s\n' "$OUTPUT_DIR"
