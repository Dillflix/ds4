#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the current four-GPU SM75 production bottlenecks after Phase 2.

The run profiles both the mixed15 and all43 models at the genuine 32K
frontier. Each model receives one prefill trace and one bounded steady-decode
trace. The stable production topology is locked: 22/21 layers, pair-0
attention rows disabled, pair-0 indexer rows enabled, and both pair-1 splits
enabled. Optional bounded Nsight Compute captures cover the shipping decode
weight-kernel families without reopening either GGUF.

Required environment:
  MIXED_MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  ALL43_MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf

Optional environment:
  PROMPT=...                         default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  REQUIRED_POWER_LIMITS_W=250,260,250,250  physical GPU 0,1,2,3 order
  STAGE_SPLIT=22                    fixed: current production planner contract
  PROFILE_TOKENS=32768
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  DECODE_SKIP=16
  DECODE_TOKENS=16
  PROFILE_GPU=0
  RUN_NCU=1
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PHASE2_PROFILE_DIR=...            new output directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MIXED_MODEL=${MIXED_MODEL:-}
ALL43_MODEL=${ALL43_MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
STAGE_SPLIT=${STAGE_SPLIT:-22}
PROFILE_TOKENS=${PROFILE_TOKENS:-32768}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
DECODE_SKIP=${DECODE_SKIP:-16}
DECODE_TOKENS=${DECODE_TOKENS:-16}
PROFILE_GPU=${PROFILE_GPU:-0}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PHASE2_PROFILE_DIR:-$repo_dir/sm75-phase2-consolidated-profile-$stamp}

for spec in "MIXED_MODEL:$MIXED_MODEL" "ALL43_MODEL:$ALL43_MODEL"; do
    name=${spec%%:*}; value=${spec#*:}
    [[ $value == /* && -f $value ]] ||
        die "$name must name an existing absolute GGUF"
done
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for spec in "STAGE_SPLIT:$STAGE_SPLIT" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
            "DECODE_SKIP:$DECODE_SKIP" "DECODE_TOKENS:$DECODE_TOKENS" \
            "PROFILE_GPU:$PROFILE_GPU"; do
    name=${spec%%:*}; value=${spec#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 )) ||
    die "current stage-aware T256 production placement requires STAGE_SPLIT=22"
(( PROFILE_TOKENS == 32768 && PREFILL_CHUNK > 0 && PIPELINE_MB > 0 &&
   DECODE_SKIP > 0 && DECODE_TOKENS > 0 )) || die "invalid profiling shape"
for flag in RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"

tools=(awk cat cmp date env git grep make mkdir mv nproc nsys nvidia-smi
       python3 sha256sum sort stat tail tar tee tr)
(( RUN_NCU == 0 )) || tools+=(ncu)
(( RUN_NCU == 0 || NCU_USE_SUDO == 0 )) || tools+=(sudo)
for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU_DEVICES must select four GPUs and REQUIRED_POWER_LIMITS_W must contain physical GPU 0,1,2,3 limits"
declare -A seen_gpu=()
for i in "${!gpu_ids[@]}"; do
    gpu=${gpu_ids[$i]}
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "invalid or duplicate GPU index: $gpu"
    (( gpu < ${#required_power[@]} )) ||
        die "selected physical GPU $gpu has no REQUIRED_POWER_LIMITS_W entry"
    want=${required_power[$gpu]}
    [[ $want =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid power limit: $want"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not 7.5"
    actual=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    awk -v actual="$actual" -v want="$want" \
        'BEGIN {exit !(actual+0.0 >= want-0.01 && actual+0.0 <= want+0.01)}' ||
        die "GPU $gpu power limit is ${actual:-unknown} W; expected $want W"
done
[[ -n ${seen_gpu[$PROFILE_GPU]+x} ]] || die "PROFILE_GPU must be selected"

mkdir -p "$OUTPUT_DIR"/{prefill/nsys,decode/nsys,telemetry,validation,provenance,summary}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
topology_file="$OUTPUT_DIR/provenance/topology.txt"
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
    forward=$(topology_link "$home" "$partner")
    reverse=$(topology_link "$partner" "$home")
    # NVLink is physically bidirectional, but some nvidia-smi releases omit
    # one symmetric matrix row. Accept one unambiguous NV# report while still
    # rejecting a contradictory reported value. The engine independently
    # validates CUDA DIRECT peer access in both directions before admitting
    # partner execution.
    [[ $forward =~ ^NV[0-9]+$ || $reverse =~ ^NV[0-9]+$ ]] ||
        die "logical pair $pair ($home/$partner) is not reported as NVLink: ${forward:-missing}/${reverse:-missing}"
    [[ -z $forward || $forward =~ ^NV[0-9]+$ ]] ||
        die "inconsistent NVLink topology for physical pair $home/$partner: $forward/${reverse:-missing}"
    [[ -z $reverse || $reverse =~ ^NV[0-9]+$ ]] ||
        die "inconsistent NVLink topology for physical pair $home/$partner: ${forward:-missing}/$reverse"
done

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
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
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
printf '%s\n' "${inherited_ds4[@]:-}" >"$OUTPUT_DIR/provenance/cleared-ds4-env.txt"
production_env=(
    "${clean[@]}"
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=0,1
    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_AUDIT=1
    DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=1024
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_DIRECT_NATIVE_Q8_AUDIT=1
    DS4_CUDA_COMPRESSOR_PAIR_STATE_STORE_AUDIT=1
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
    DS4_CUDA_CRITICAL_PATH_NVTX=1
)

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke
         tests/test_engine_mgpu_placement)
(( RUN_NCU == 0 )) || targets+=(tests/cuda_sm75_decode_weight_profile)
if [[ $SKIP_BUILD == 0 ]]; then
    make -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/validation/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
for target in "${targets[@]}"; do
    [[ -x $target ]] || die "required binary is missing: $target"
done

phase=exactness
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/validation/planner.log" 2>&1 || {
        tail -n 140 "$OUTPUT_DIR/validation/planner.log" >&2 || true
        die "multi-GPU placement tests failed"
    }
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/validation/cuda-exactness.log" 2>&1 || {
        tail -n 220 "$OUTPUT_DIR/validation/cuda-exactness.log" >&2 || true
        die "CUDA exactness regression failed"
    }
for marker in \
    'SM75 Q4-32 tile32-mma gate/up + tile32 down production defaults' \
    'SM75 Q3A4 tile32-dp4a-k4-prefetch2 production default' \
    'SM75 direct native Q8 decode default selector exact' \
    'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/validation/cuda-exactness.log" ||
        die "missing exactness/default marker: $marker"
done

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'mixed_model=%s\nmixed_model_bytes=%s\n' \
        "$MIXED_MODEL" "$(stat -c %s "$MIXED_MODEL")"
    printf 'all43_model=%s\nall43_model_bytes=%s\n' \
        "$ALL43_MODEL" "$(stat -c %s "$ALL43_MODEL")"
    printf 'prompt=%s\nprompt_sha256=%s\n' "$PROMPT" \
        "$(sha256sum "$PROMPT" | awk '{print $1}')"
    printf 'gpu_devices=%s\ngpu_vram=%s\nrequired_power_limits_w=%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$REQUIRED_POWER_LIMITS_W"
    printf 'stage_split=22/21\nprofile_tokens=%s\nprefill_chunk=%s\npipeline_mb=%s\n' \
        "$PROFILE_TOKENS" "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'decode_skip=%s\ndecode_tokens=%s\n' "$DECODE_SKIP" "$DECODE_TOKENS"
    printf 'pair0_attention_rows=disabled\npair0_indexer_rows=enabled\n'
    printf 'pair1_attention_rows=enabled\npair1_indexer_rows=enabled\n'
    printf 't32_output=fused-fp16-default\ndirect_native_q8=decode-default\n'
    printf 'compressor_state_append=fused-default\nrun_ncu=%s\n' "$RUN_NCU"
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\n[topology]\n'; cat "$topology_file"
    printf '\n[nsight systems]\n'; nsys --version
    if [[ $RUN_NCU == 1 ]]; then printf '\n[nsight compute]\n'; ncu --version; fi
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

capture_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"; return 1;
            }
    done
    mv -- "$partial" "$output"
}
capture_health "$OUTPUT_DIR/validation/initial-gpu.csv" ||
    die "could not capture initial GPU health"
validate_health() {
    local stem=$1
    [[ -s $stem.pre-gpu.csv && -s $stem.post-gpu.csv ]] &&
        ! grep -Eq 'ERR!|Unknown Error|GPU is lost' \
            "$stem.pre-gpu.csv" "$stem.post-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/validation/initial-gpu.csv" "$stem.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/validation/initial-gpu.csv" "$stem.post-gpu.csv"
}

validate_layout() {
    local layout=$1 log=$2 expected_count expected_layers
    if [[ $layout == mixed15 ]]; then
        expected_count=15
        expected_layers=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
    else
        expected_count=43
        expected_layers=$(printf '%s,' {0..42}); expected_layers=${expected_layers%,}
    fi
    awk -v expected_count="$expected_count" -v expected_layers="$expected_layers" '
        /routed-quant-audit/ {
            seen++; layer=gate=up=down=""
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]=="layer") layer=a[2]
                if (a[1]=="gate") gate=a[2]
                if (a[1]=="up") up=a[2]
                if (a[1]=="down") down=a[2]
            }
            if (layer !~ /^[0-9]+$/ || layer<0 || layer>42 || layer_seen[layer]++) bad=1
            if (gate=="sm75_q3a4") {
                if (up!="sm75_q3a4" || down!="sm75_q4_32") bad=1
                q3++; layers=layers (layers ? "," : "") layer
            } else if (gate!="sm75_q4_32" || up!="sm75_q4_32" ||
                       down!="sm75_q4_32") bad=1
        }
        END {
            for (i=0; i<43; i++) if (layer_seen[i]!=1) bad=1
            exit !(seen==43 && q3==expected_count && layers==expected_layers && !bad)
        }
    ' "$log"
}

validate_topology_log() {
    local layout=$1 log=$2 marker route
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'CUDA decode TP enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled' \
                  'CUDA TP cache mirror policy: attention-pair-mask=0x2 index-pair-mask=0x3' \
                  'prefill attention row split pair-scoped disable: logical-pairs=0' \
                  'prefill indexer row split pair policy: enabled-pairs=0,1 disabled-pairs=none'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    [[ $(grep -Fc 'qualified=yes' "$log") == 2 ]] || return 1
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" || return 1
    grep -Eq 'CUDA T32 f16-output fused summary: local=0 partner=[1-9][0-9]*' "$log" ||
        return 1
    ! grep -Fq 'required but unavailable' "$log" || return 1
    validate_layout "$layout" "$log"
}

validate_decode_defaults() {
    local layout=$1 log=$2 width
    grep -Fq 'SM75 direct native Q8 producer selected for routed MoE activations' "$log" ||
        return 1
    for width in 256 512 1024; do
        grep -Fq "SM75 compressor pair/state fusion selected width=$width " "$log" ||
            return 1
    done
    grep -Fxq 'ds4: SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4-prefetch2 (production default)' \
        "$log" || return 1
    grep -Fxq 'ds4: SM75 Q4-32 down decode mapping=tile32-int4 (production default)' \
        "$log" || return 1
    if [[ $layout == mixed15 ]]; then
        grep -Fxq 'ds4: SM75 Q4-32 decode gate/up mapping=tile32-mma (production default)' \
            "$log" || return 1
    else
        ! grep -Fq 'SM75 Q4-32 decode gate/up mapping=' "$log" || return 1
    fi
}

start_telemetry() {
    local output=$1
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate \
        --format=csv -lms 200 >"$output" &
    sampler_pid=$!
}

stats_reports() {
    local base=$1 report
    for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum \
                  cuda_gpu_trace nvtx_sum nvtx_gpu_proj_sum; do
        nsys stats --report "$report" --format csv "$base.nsys-rep" \
            >"$base-$report.csv" 2>"$base-$report.log" || true
    done
}

declare -A models=(
    [mixed15]="$MIXED_MODEL"
    [all43]="$ALL43_MODEL"
)
printf 'label\tdevices\tsqlite\tlayout\tstage_split\tbenchmark\tlog\n' \
    >"$OUTPUT_DIR/prefill/trace-map.tsv"
printf 'trial\tslot\tscenario\tdevice\tlog\n' \
    >"$OUTPUT_DIR/prefill/harness-runs.tsv"
printf 'label\tpp_tokens\tthreshold\tcaptured_tokens\tdevices\tsqlite\tlayout\tstage_split\tbenchmark\tlog\n' \
    >"$OUTPUT_DIR/decode/trace-map.tsv"

capture_prefill() {
    local layout=$1 model=${models[$1]} label="${1}-prefill-32k"
    local base="$OUTPUT_DIR/prefill/nsys/$label" rc=0
    phase="prefill-$layout"
    printf 'Phase 2 profile: model=%s phase=prefill frontier=%s...\n' \
        "$layout" "$PROFILE_TOKENS"
    capture_health "$base.pre-gpu.csv" || die "$label pre-run GPU health failed"
    start_telemetry "$OUTPUT_DIR/telemetry/$label.csv"
    set +e
    "${production_env[@]}" DS4_NSYS_CAPTURE_PREFILL=1 \
    nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$base" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$model" --prompt-file "$PROMPT" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
            --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
            --csv "$base-benchmark.csv" >"$base.log" 2>&1
    rc=$?
    set -e
    cleanup_sampler
    (( rc == 0 )) || { tail -n 220 "$base.log" >&2 || true; die "$label failed (exit $rc)"; }
    capture_health "$base.post-gpu.csv" || die "$label post-run GPU health failed"
    validate_health "$base" || die "$label changed GPU identity, power, or health"
    validate_topology_log "$layout" "$base.log" || die "$label production topology validation failed"
    awk -F, -v pp="$PROFILE_TOKENS" '
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" && $4=="gen_tokens"); next}
        NR==2 {rows++; ok=($1==pp && ($3+0)>0 && ($4+0)==0); next}
        NR>2 {rows++}
        END {exit !(header && rows==1 && ok)}
    ' "$base-benchmark.csv" || die "$label benchmark CSV is invalid"
    [[ -s $base.nsys-rep ]] || die "$label omitted its Nsight report"
    nsys export --type sqlite --force-overwrite=true --output "$base.sqlite" \
        "$base.nsys-rep" >"$base-export.log" 2>&1 || die "$label SQLite export failed"
    [[ -s $base.sqlite ]] || die "$label SQLite export is empty"
    stats_reports "$base"
    printf '%s\t%s\t%s\t%s\t22/21\t%s\t%s\n' \
        "$label" "$GPU_DEVICES" "$base.sqlite" "$layout" \
        "$base-benchmark.csv" "$base.log" >>"$OUTPUT_DIR/prefill/trace-map.tsv"
}

capture_decode() {
    local layout=$1 model=${models[$1]} label="${1}-decode-32k"
    local base="$OUTPUT_DIR/decode/nsys/$label" rc=0
    local gen_tokens=$((DECODE_SKIP + DECODE_TOKENS))
    phase="decode-$layout"
    printf 'Phase 2 profile: model=%s phase=steady-decode frontier=%s tokens=%s...\n' \
        "$layout" "$PROFILE_TOKENS" "$DECODE_TOKENS"
    capture_health "$base.pre-gpu.csv" || die "$label pre-run GPU health failed"
    start_telemetry "$OUTPUT_DIR/telemetry/$label.csv"
    set +e
    "${production_env[@]}" \
    "DS4_NSYS_CAPTURE_DECODE_SKIP=$DECODE_SKIP" \
    "DS4_NSYS_CAPTURE_DECODE_TOKENS=$DECODE_TOKENS" \
    nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$base" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$model" --prompt-file "$PROMPT" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + gen_tokens + 1))" \
            --step-incr "$PROFILE_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
            --gen-tokens "$gen_tokens" --csv "$base-benchmark.csv" \
            >"$base.log" 2>&1
    rc=$?
    set -e
    cleanup_sampler
    (( rc == 0 )) || { tail -n 220 "$base.log" >&2 || true; die "$label failed (exit $rc)"; }
    capture_health "$base.post-gpu.csv" || die "$label post-run GPU health failed"
    validate_health "$base" || die "$label changed GPU identity, power, or health"
    validate_topology_log "$layout" "$base.log" || die "$label production topology validation failed"
    validate_decode_defaults "$layout" "$base.log" || die "$label Phase 2 default validation failed"
    grep -Fq "starting Nsight CUDA capture for decode frontier $PROFILE_TOKENS" "$base.log" ||
        die "$label did not start bounded decode capture"
    grep -Fq "stopped Nsight CUDA capture for decode frontier $PROFILE_TOKENS after $DECODE_TOKENS tokens" \
        "$base.log" || die "$label did not stop bounded decode capture"
    awk -F, -v pp="$PROFILE_TOKENS" -v tg="$gen_tokens" '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" && $8=="gen_steady_tps"); next}
        NR==2 {rows++; ok=($1==pp && ($4+0)==tg && ($8+0)>0); next}
        NR>2 {rows++}
        END {exit !(header && rows==1 && ok)}
    ' "$base-benchmark.csv" || die "$label benchmark CSV is invalid"
    [[ -s $base.nsys-rep ]] || die "$label omitted its Nsight report"
    nsys export --type sqlite --force-overwrite=true --output "$base.sqlite" \
        "$base.nsys-rep" >"$base-export.log" 2>&1 || die "$label SQLite export failed"
    [[ -s $base.sqlite ]] || die "$label SQLite export is empty"
    stats_reports "$base"
    printf '%s\t%s\t1024\t%s\t%s\t%s\t%s\t22/21\t%s\t%s\n' \
        "$label" "$PROFILE_TOKENS" "$DECODE_TOKENS" "$GPU_DEVICES" \
        "$base.sqlite" "$layout" "$base-benchmark.csv" "$base.log" \
        >>"$OUTPUT_DIR/decode/trace-map.tsv"
}

for layout in mixed15 all43; do capture_prefill "$layout"; done
for layout in mixed15 all43; do capture_decode "$layout"; done

phase=prefill-postprocess
python3 speed-bench/summarize-cuda-critical-path.py "$OUTPUT_DIR/prefill" \
    | tee "$OUTPUT_DIR/prefill/critical-path-summary.txt"

phase=decode-postprocess
python3 speed-bench/summarize-sm75-decode-evidence.py "$OUTPUT_DIR/decode" \
    | tee "$OUTPUT_DIR/decode/summary-stdout.txt"

if [[ $RUN_NCU == 1 ]]; then
    phase=bounded-nsight-compute
    PROFILE_GPU="$PROFILE_GPU" CUDA_ARCH=sm_75 PROFILE_SET=all \
    NCU_SET=focused NCU_CACHE_CONTROL=all NCU_USE_SUDO="$NCU_USE_SUDO" \
    SKIP_BUILD=1 RESUME=0 CREATE_ARCHIVE=0 \
    DECODE_WEIGHT_PROFILE_DIR="$OUTPUT_DIR/ncu-decode-weight" \
        bash ./speed-bench/cuda-sm75-decode-weight-profile.sh \
        | tee "$OUTPUT_DIR/ncu-decode-weight.stdout.txt"
fi

phase=consolidated-summary
python3 speed-bench/summarize-sm75-phase2-profile.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary/stdout.txt"

required=(
    prefill/trace-summary.csv
    prefill/stage-device-summary.csv
    prefill/operation-attribution.csv
    decode/summary/trace-summary.csv
    decode/summary/stage-device-summary.csv
    decode/summary/operation-attribution.csv
    summary/summary.md
    summary/throughput.csv
    summary/prefill-family-summary.csv
    summary/decode-family-summary.csv
    summary/prefill-stage-balance.csv
    summary/phase3-target-evidence.csv
)
for layout in mixed15 all43; do
    required+=(
        "prefill/nsys/$layout-prefill-32k.nsys-rep"
        "prefill/nsys/$layout-prefill-32k.sqlite"
        "decode/nsys/$layout-decode-32k.nsys-rep"
        "decode/nsys/$layout-decode-32k.sqlite"
    )
done
(( RUN_NCU == 0 )) || required+=(
    ncu-decode-weight/summary.csv
    ncu-decode-weight/summary.md
)
for item in "${required[@]}"; do
    [[ -s $OUTPUT_DIR/$item ]] || die "missing final evidence: $item"
done

phase=complete
printf 'SM75 Phase 2 consolidated production profile complete: %s\n' "$OUTPUT_DIR"
