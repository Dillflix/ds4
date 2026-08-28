#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Profile the accepted SM75 mixed-Q4/IQ2 + balanced-T256 production path.

The model is opened exactly once for a bounded Nsight Systems capture at the
production 256K allocation. Nsight Compute uses small production-shaped
harnesses and therefore does not reread the GGUF for each metric pass.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  PROFILE_TOKENS=8192
  CTX_ALLOC=262273
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  INDEXER_NATIVE_CACHE=1      1 for native F16/streaming64; 0 for legacy F32
  INDEXER_SCORE_TILE=64       native requires 64; legacy accepts 128 or 64
  PROFILE_GPU=0                physical GPU for native-Q4 NCU harnesses
  PROFILE_PARTNER_GPU=1        its NVLink partner for T256 cuBLAS NCU
  RUN_NCU=1
  NCU_SET=full                  full or attention
  RUN_ATTENTION_NCU=auto        auto enables at PROFILE_TOKENS >= 32768
  NCU_USE_SUDO=1
  SKIP_BUILD=0
  REUSE_NSYS_DIR=             prior profiler output directory; skips GGUF load
  REUSE_Q4_NCU_DIR=           prior output with four validated native-Q4 reports
  CREATE_ARCHIVE=1
  COMBINED_PROFILE_DIR=...     new output directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL=${MODEL:-${NATIVE_MODEL:-}}
: "${MODEL:?set MODEL to the absolute tagged SM75 mixed-Q4/IQ2 GGUF}"
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute file"
NATIVE_MODEL=$MODEL

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
PROFILE_TOKENS=${PROFILE_TOKENS:-8192}
CTX_ALLOC=${CTX_ALLOC:-262273}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
INDEXER_NATIVE_CACHE=${INDEXER_NATIVE_CACHE:-1}
INDEXER_SCORE_TILE=${INDEXER_SCORE_TILE:-64}
PROFILE_GPU=${PROFILE_GPU:-0}
PROFILE_PARTNER_GPU=${PROFILE_PARTNER_GPU:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_SET=${NCU_SET:-full}
RUN_ATTENTION_NCU=${RUN_ATTENTION_NCU:-auto}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
REUSE_NSYS_DIR=${REUSE_NSYS_DIR:-}
REUSE_Q4_NCU_DIR=${REUSE_Q4_NCU_DIR:-}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${COMBINED_PROFILE_DIR:-$repo_dir/sm75-native-q4-t256-profile-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "CTX_ALLOC:$CTX_ALLOC" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "PIPELINE_MB:$PIPELINE_MB" \
            "PROFILE_GPU:$PROFILE_GPU" \
            "PROFILE_PARTNER_GPU:$PROFILE_PARTNER_GPU" \
            "RUN_NCU:$RUN_NCU" "NCU_USE_SUDO:$NCU_USE_SUDO" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
if [[ $RUN_ATTENTION_NCU == auto ]]; then
    if (( PROFILE_TOKENS >= 32768 )); then RUN_ATTENTION_NCU=1; else RUN_ATTENTION_NCU=0; fi
fi
[[ $RUN_ATTENTION_NCU == 0 || $RUN_ATTENTION_NCU == 1 ]] ||
    die "RUN_ATTENTION_NCU must be auto, 0, or 1"
[[ $NCU_SET == full || $NCU_SET == attention ]] ||
    die "NCU_SET must be full or attention"
[[ $INDEXER_SCORE_TILE == 128 || $INDEXER_SCORE_TILE == 64 ]] ||
    die "INDEXER_SCORE_TILE must be 128 or 64"
[[ $INDEXER_NATIVE_CACHE == 0 || $INDEXER_NATIVE_CACHE == 1 ]] ||
    die "INDEXER_NATIVE_CACHE must be 0 or 1"
[[ $INDEXER_NATIVE_CACHE == 0 || $INDEXER_SCORE_TILE == 64 ]] ||
    die "INDEXER_NATIVE_CACHE=1 requires INDEXER_SCORE_TILE=64"
if [[ $NCU_SET == attention && $RUN_NCU == 1 && $RUN_ATTENTION_NCU == 0 ]]; then
    die "NCU_SET=attention requires RUN_ATTENTION_NCU=1"
fi
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 )) ||
    die "STAGE_SPLIT must be in 1..42"
(( PROFILE_TOKENS >= PREFILL_CHUNK &&
   PROFILE_TOKENS % PREFILL_CHUNK == 0 &&
   CTX_ALLOC > PROFILE_TOKENS && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 )) ||
    die "require profile_tokens>=2048, a 2048-token chunk, 512-token pipeline, and ctx_alloc>profile_tokens"
printf -v frontier_tag '%06d' "$PROFILE_TOKENS"
for flag in RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $PROFILE_GPU != "$PROFILE_PARTNER_GPU" ]] ||
    die "PROFILE_GPU and PROFILE_PARTNER_GPU must differ"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk basename cat cmp cp date dirname env find git grep id make mkdir \
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
if [[ -n $REUSE_NSYS_DIR ]]; then
    [[ $REUSE_NSYS_DIR == /* && -d $REUSE_NSYS_DIR ]] ||
        die "REUSE_NSYS_DIR must name an existing absolute profiler directory"
    REUSE_NSYS_DIR=$(cd "$REUSE_NSYS_DIR" && pwd)
    reuse_manifest="$REUSE_NSYS_DIR/manifest.txt"
    [[ -s $reuse_manifest ]] ||
        die "REUSE_NSYS_DIR lacks manifest.txt; refusing unverified reuse"
    reuse_value() {
        awk -F= -v key="$1" '$1 == key {
            print substr($0, length(key) + 2); found=1; exit
        } END {if (!found) exit 1}' "$reuse_manifest"
    }
    require_reuse_value() {
        local key=$1 expected=$2 actual
        actual=$(reuse_value "$key") ||
            die "REUSE_NSYS_DIR manifest lacks $key"
        [[ $actual == "$expected" ]] ||
            die "REUSE_NSYS_DIR $key mismatch: requested '$expected', captured '$actual'"
    }
    require_reuse_value model "$MODEL"
    require_reuse_value model_bytes "$(stat -c %s "$MODEL")"
    require_reuse_value prompt "$PROMPT"
    require_reuse_value gpu_devices "$GPU_DEVICES"
    require_reuse_value gpu_vram "$GPU_VRAM"
    require_reuse_value stage_split "$STAGE_SPLIT/$((43-STAGE_SPLIT))"
    require_reuse_value profile_tokens "$PROFILE_TOKENS"
    require_reuse_value ctx_alloc "$CTX_ALLOC"
    require_reuse_value prefill_chunk "$PREFILL_CHUNK"
    require_reuse_value pipeline_mb "$PIPELINE_MB"
    require_reuse_value indexer_score_tile "$INDEXER_SCORE_TILE"
    require_reuse_value indexer_native_cache "$INDEXER_NATIVE_CACHE"
    require_reuse_value t256_policy automatic-balanced
fi
if [[ -n $REUSE_Q4_NCU_DIR ]]; then
    [[ $REUSE_Q4_NCU_DIR == /* && -d $REUSE_Q4_NCU_DIR ]] ||
        die "REUSE_Q4_NCU_DIR must name an existing absolute profiler directory"
    REUSE_Q4_NCU_DIR=$(cd "$REUSE_Q4_NCU_DIR" && pwd)
fi

topology_file=$(mktemp)
nvidia-smi topo -m >"$topology_file"
topology_link() {
    local from=$1 to=$2
    awk -v from="GPU$from" -v to="GPU$to" '
        !header {
            n_gpu = 0
            for (i = 1; i <= NF; i++) if ($i ~ /^GPU[0-9]+$/) n_gpu++
            if (n_gpu > 1) {
                # The header begins with GPU0, whereas each matrix row begins
                # with its source-GPU label. Account for that extra row field.
                for (i = 1; i <= NF; i++) if ($i == to) column = i + 1
                header = 1
                next
            }
        }
        header && $1 == from && column > 0 && column <= NF {
            print $column
            exit
        }
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
mkdir -p "$OUTPUT_DIR/nsys/frontier-logits"
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
)
if [[ $INDEXER_NATIVE_CACHE == 0 ]]; then
    production_env+=(DS4_CUDA_NO_INDEXER_NATIVE_F16_CACHE=1)
fi
if [[ $INDEXER_NATIVE_CACHE == 0 && $INDEXER_SCORE_TILE == 64 ]]; then
    production_env+=(DS4_CUDA_NO_INDEXER_WMMA128=1)
fi

summary_schema=$(python3 \
    "$repo_dir/speed-bench/summarize-sm75-native-q4-t256-profile.py" \
    --schema)
[[ $summary_schema == post-row-split-v2 ]] ||
    die "unexpected profile summarizer schema: $summary_schema"

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
              'tagged IQ2 tail8-default gate/up + native Q4 down exact' \
              'tagged IQ2 tail4 rollback exact' \
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
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PROFILE_PARTNER_GPU" \
    ./tests/test_gpu_xdev q8-partner-t32-profile \
    >"$OUTPUT_DIR/validation/partner-t32-harness.log" 2>&1 || {
        tail -n 120 "$OUTPUT_DIR/validation/partner-t32-harness.log" >&2 || true
        die "partner T32 profile harness failed"
    }
grep -Fq 'harness_status=ok' \
    "$OUTPUT_DIR/validation/partner-t32-harness.log" ||
    die "partner T32 harness omitted success"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\n' \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'prompt=%s\ngpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" \
        "$((43-STAGE_SPLIT))"
    printf 'profile_tokens=%s\nctx_alloc=%s\nprefill_chunk=%s\npipeline_mb=%s\nindexer_score_tile=%s\nindexer_native_cache=%s\n' \
        "$PROFILE_TOKENS" "$CTX_ALLOC" "$PREFILL_CHUNK" "$PIPELINE_MB" \
        "$INDEXER_SCORE_TILE" "$INDEXER_NATIVE_CACHE"
    printf 't256_policy=automatic-balanced\nt256_local_layers=even\nt256_partner_layers=odd\npartner_arithmetic=f16\n'
    printf 'attention_rows_policy=automatic-qualified\nxdev_sync=disabled\n'
    printf 'profile_gpu=%s\nprofile_partner_gpu=%s\nrun_ncu=%s\nncu_set=%s\nrun_attention_ncu=%s\n' \
        "$PROFILE_GPU" "$PROFILE_PARTNER_GPU" "$RUN_NCU" "$NCU_SET" \
        "$RUN_ATTENTION_NCU"
    printf 'reused_nsys_dir=%s\n' "${REUSE_NSYS_DIR:-none}"
    printf 'reused_q4_ncu_dir=%s\n' "${REUSE_Q4_NCU_DIR:-none}"
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
if [[ -n $REUSE_NSYS_DIR ]]; then
    phase=nsight-systems-reuse
    printf 'Reusing the completed production trace from %s (no GGUF load)...\n' \
        "$REUSE_NSYS_DIR"
    [[ -s $REUSE_NSYS_DIR/nsys/combined.nsys-rep &&
       -s $REUSE_NSYS_DIR/nsys/combined-benchmark.csv &&
       -s $REUSE_NSYS_DIR/nsys/combined.log &&
       -s $REUSE_NSYS_DIR/nsys/bindings.csv &&
       -s $REUSE_NSYS_DIR/nsys/allocations.csv &&
       -s $REUSE_NSYS_DIR/nsys/memory.csv &&
       -s $REUSE_NSYS_DIR/nsys/plan.csv &&
       -s $REUSE_NSYS_DIR/nsys/routed-tile-audit.csv &&
       -s $REUSE_NSYS_DIR/nsys/cache-before.csv &&
       -s $REUSE_NSYS_DIR/nsys/cache-after.csv ]] ||
        die "REUSE_NSYS_DIR lacks a complete production trace"
    awk -F, -v want="$PROFILE_TOKENS" '
        NR == 2 {rows++; if (($1 + 0) != want) bad=1}
        NR > 2 {rows++; bad=1}
        END {exit !(rows == 1 && !bad)}
    ' "$REUSE_NSYS_DIR/nsys/combined-benchmark.csv" ||
        die "REUSE_NSYS_DIR benchmark is not the requested single ${PROFILE_TOKENS}-token frontier"
    cp -a "$REUSE_NSYS_DIR/nsys/." "$OUTPUT_DIR/nsys/"
    if [[ -s $REUSE_NSYS_DIR/telemetry/combined.csv ]]; then
        cp -a "$REUSE_NSYS_DIR/telemetry/combined.csv" \
            "$OUTPUT_DIR/telemetry/combined.csv"
    fi
else
    printf 'Capturing one mixed-Q4/IQ2 balanced-T256 production trace...\n'
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,\
temperature.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits -lms 200 \
        >"$OUTPUT_DIR/telemetry/combined.csv" &
    sampler_pid=$!
    run_rc=0
    "${production_env[@]}" \
        DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
        DS4_CUDA_INDEXER_SCORE_AUDIT=1 \
        DS4_CUDA_CRITICAL_PATH_NVTX=1 \
        "DS4_CUDA_PREFILL_TILE_AUDIT_CSV=$tile_audit" \
        DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY=32768 \
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
                --ctx-alloc "$CTX_ALLOC" \
                --step-incr "$PROFILE_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
                --gen-tokens 0 --csv "$base-benchmark.csv" \
                --dump-frontier-logits-dir "$OUTPUT_DIR/nsys/frontier-logits" \
                >"$base.log" 2>&1 || run_rc=$?
    cleanup_sampler
    (( run_rc == 0 )) || {
        tail -n 200 "$base.log" >&2 || true
        die "combined Nsight Systems capture failed (exit $run_rc)"
    }
fi
[[ -s $base.nsys-rep && -s $base-benchmark.csv ]] ||
    die "Nsight Systems omitted its report or benchmark"
[[ -s $OUTPUT_DIR/nsys/frontier-logits/frontier_${frontier_tag}.logits.f32 &&
   -s $OUTPUT_DIR/nsys/frontier-logits/frontier_${frontier_tag}.logits.json ]] ||
    die "production trace omitted the ${PROFILE_TOKENS}-token frontier logits"
grep -Fqx 'ds4: SM75 native routed-Q4 layout enabled (packed A/W, planner=cost, gate=tile8, down=full-stage)' \
    "$base.log" || die "production trace did not use the accepted native-Q4 path"
for marker in "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
              'partner-classes=automatic:t256' 'partner-layers=all' \
              't256-placement=balanced' 'materialized 344/344 candidates' \
              'T256-output_b=43/43' 'partner=21 partner-arithmetic=f16' \
              'SM75 IQ2 residual 1..8 tail8 enabled' \
              'CUDA q8 fp16 partner summary: calls='; do
    grep -Fq "$marker" "$base.log" ||
        die "production trace lacks required marker: $marker"
done
for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
    grep -Fq "$route" "$base.log" ||
        die "production trace lacks validated direct route $route"
done
[[ $(grep -Fc 'qualified=yes' "$base.log") == 2 ]] ||
    die "production trace did not qualify both SM75 NVLink pairs"
[[ $(grep -Fc 'CUDA prefill attention query-row split enabled:' "$base.log") == 2 ]] ||
    die "production trace did not enable both qualified attention row splits"
! grep -Fq 'required but unavailable' "$base.log" ||
    die "production trace encountered an eligible but unavailable row split"
if [[ $INDEXER_NATIVE_CACHE == 1 ]]; then
    expected_indexer_dispatch=streaming64-native
else
    expected_indexer_dispatch=wmma$INDEXER_SCORE_TILE
fi
for dispatch in direct-one streaming64-native wmma128-f16-native wmma128 wmma64 wmma32 wmma16 generic; do
    audit_line=$(grep -F "ds4: CUDA indexer score audit dispatch=$dispatch " \
        "$base.log" || true)
    [[ $(printf '%s\n' "$audit_line" | grep -c .) == 1 ]] ||
        die "production trace lacks one unambiguous $dispatch indexer audit record"
    audit_launches=${audit_line##*launches=}
    [[ $audit_launches =~ ^[0-9]+$ ]] ||
        die "production trace has invalid $dispatch indexer audit count"
    if [[ $dispatch == "$expected_indexer_dispatch" ]]; then
        (( audit_launches > 0 )) ||
            die "production trace did not dispatch $expected_indexer_dispatch"
    else
        (( audit_launches == 0 )) ||
            die "production trace unexpectedly dispatched $dispatch $audit_launches times"
    fi
done
awk '
    /^ds4: routed-quant-audit layer=/ {
        count++
        if ($0 ~ /gate=q4_k up=q4_k down=q4_k$/) q4++
        else if ($0 ~ /gate=iq2_xxs up=iq2_xxs down=q4_k$/) iq2++
        else bad++
    }
    END {exit !(count == 43 && q4 > 0 && iq2 > 0 && bad == 0)}
' "$base.log" ||
    die "production trace is not the expected 43-layer mixed Q4/IQ2/Q4 recipe"
cmp -s "$cache_before" "$cache_after" ||
    die "Q8 cache state changed during the captured frontier"
awk -F, '
    NR == 1 {next}
    NR > 1 && $6 == 8192 && $7 == 4096 && $12 ~ /attn_output_b/ {
        label=$12
        sub(/^tensor:blk\./, "", label)
        sub(/\..*$/, "", label)
        layer=label+0
        if ($3 == 1) {
            partner++
            if (layer % 2 == 0) bad++
        } else {
            fixed++
            if (layer % 2 != 0) bad++
        }
        next
    }
    $3 == 1 {bad++}
    END {exit !(fixed == 22 && partner == 21 && bad == 0 && NR == 345)}
' "$bindings" || die "binding inventory is not the exact 22-local/21-partner balanced plan"
awk -F, 'NR > 1 {if ($10 != $11 || $13 != 0 || $14 != 1) bad++; rows++}
    END {exit !(rows == 344 && bad == 0)}' "$allocations" ||
    die "allocation inventory contains missing or dead dense-F16 payloads"
awk -F, 'NR > 1 {if ($3 + 0 < 512 * 1048576) bad++; rows++}
    END {exit !(rows == 4 && bad == 0)}' "$memory" ||
    die "production trace left less than 512 MiB free on a CUDA device"

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
python3 "$repo_dir/speed-bench/summarize-cuda-critical-path.py" "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/critical-path-summary.txt"
python3 "$repo_dir/speed-bench/summarize-sm75-native-q4-t256-profile.py" \
    "$OUTPUT_DIR" \
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
        local label=$1 scenario=$2 kernel=$3 expected=$4 block_size=$5 base=$6
        local scalar_target=${7:-none} rc=0 source_base=
        local -a scalar_env=()
        if [[ $scalar_target != none ]]; then
            scalar_env=("DS4_PROFILE_SCALAR_TARGET=$scalar_target"
                        DS4_PROFILE_SCALAR=1)
        fi
        if [[ -n $REUSE_Q4_NCU_DIR && $label == *native-q4* ]]; then
            source_base="$REUSE_Q4_NCU_DIR/ncu/$label"
            [[ -s $source_base.ncu-rep ]] ||
                die "reusable Q4 NCU report is missing: $source_base.ncu-rep"
            printf 'Reusing validated Nsight Compute capture: %s...\n' "$label"
            cp -a "$source_base.ncu-rep" "$base.ncu-rep"
            [[ ! -f $source_base.log ]] || cp -a "$source_base.log" "$base.log"
            "$ncu_bin" --config-file off --import "$base.ncu-rep" \
                --csv --page raw >"$base.csv" 2>"$base-import.log"
            python3 speed-bench/validate-ncu-capture.py \
                "$base.csv" "$expected" 0 \
                --process cuda_sm75_profile_harness --block-size "$block_size"
            return
        fi
        printf 'Nsight Compute: %s...\n' "$label"
        "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${scalar_env[@]}" \
            "${ncu_cmd[@]}" \
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
            --process cuda_sm75_profile_harness --block-size "$block_size"
    }

    if [[ $NCU_SET == full ]]; then
    profile_native early-native-q4-gate native-q4-early \
        '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' \
        'moe_gate_up_mid_sm75_native_q4_tile8_kernel' \
        512 \
        "$OUTPUT_DIR/ncu/early-native-q4-gate"
    profile_native late-native-q4-gate native-q4-late \
        '^moe_gate_up_mid_sm75_native_q4_tile8_kernel$' \
        'moe_gate_up_mid_sm75_native_q4_tile8_kernel' \
        512 \
        "$OUTPUT_DIR/ncu/late-native-q4-gate"
    profile_native early-native-q4-down native-q4-early \
        '^moe_down_sm75_native_q4_tile_kernel$' \
        'moe_down_sm75_native_q4_tile_kernel' \
        256 \
        "$OUTPUT_DIR/ncu/early-native-q4-down"
    profile_native late-native-q4-down native-q4-late \
        '^moe_down_sm75_native_q4_tile_kernel$' \
        'moe_down_sm75_native_q4_tile_kernel' \
        256 \
        "$OUTPUT_DIR/ncu/late-native-q4-down"
    profile_native early-iq2-gate-tile16 hybrid-iq2-q4-early \
        '^moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*' \
        'moe_gate_up_mid_iq2_tile16_mma_sm75_kernel<512, 8, 1, 8>' \
        256 \
        "$OUTPUT_DIR/ncu/early-iq2-gate-tile16" iq2-tile16
    profile_native late-iq2-gate-tile16 hybrid-iq2-q4-late \
        '^moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*' \
        'moe_gate_up_mid_iq2_tile16_mma_sm75_kernel<512, 8, 1, 8>' \
        256 \
        "$OUTPUT_DIR/ncu/late-iq2-gate-tile16" iq2-tile16
    profile_native early-iq2-gate-tile8 hybrid-iq2-q4-early \
        '^moe_gate_up_mid_iq2_tile8_mma_sm75_kernel.*' \
        'moe_gate_up_mid_iq2_tile8_mma_sm75_kernel<512, 1, 8>' \
        256 \
        "$OUTPUT_DIR/ncu/early-iq2-gate-tile8" iq2-tile16
    profile_native late-iq2-gate-tile8 hybrid-iq2-q4-late \
        '^moe_gate_up_mid_iq2_tile8_mma_sm75_kernel.*' \
        'moe_gate_up_mid_iq2_tile8_mma_sm75_kernel<512, 1, 8>' \
        256 \
        "$OUTPUT_DIR/ncu/late-iq2-gate-tile8" iq2-tile16
    profile_native early-iq2-gate-tail4 hybrid-iq2-q4-early \
        '^moe_gate_up_mid_expert_tile4_row32_kernel$' \
        'moe_gate_up_mid_expert_tile4_row32_kernel' \
        256 \
        "$OUTPUT_DIR/ncu/early-iq2-gate-tail4" iq2-tile16
    profile_native late-iq2-gate-tail4 hybrid-iq2-q4-late \
        '^moe_gate_up_mid_expert_tile4_row32_kernel$' \
        'moe_gate_up_mid_expert_tile4_row32_kernel' \
        256 \
        "$OUTPUT_DIR/ncu/late-iq2-gate-tail4" iq2-tile16
    fi

    if [[ $RUN_ATTENTION_NCU == 1 ]]; then
        profile_native attention-indexed-32k attn-indexed-32k \
            '^attention_indexed_mixed_heads8_online_kernel.*' \
            'attention_indexed_mixed_heads8_online_kernel<8, 16>' \
            512 \
            "$OUTPUT_DIR/ncu/attention-indexed-32k"
        profile_native attention-mixed-32k attn-mixed-32k \
            '^attention_decode_mixed_heads8_online_kernel.*' \
            'attention_decode_mixed_heads8_online_kernel' \
            256 \
            "$OUTPUT_DIR/ncu/attention-mixed-32k"
    fi

    profile_dense_cublas() {
        local shape=$1 scenario=$2 base="$OUTPUT_DIR/ncu/$1-cublas" rc=0
        printf 'Nsight Compute: production-shaped %s dense-F16 cuBLAS...\n' "$shape"
        "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PROFILE_PARTNER_GPU" \
            "${ncu_cmd[@]}" --config-file off \
            --target-processes application-only --devices 1 \
            --profile-from-start off --launch-count 1 --replay-mode kernel \
            --cache-control none --clock-control none --force-overwrite \
            --export "$base" "${sections[@]}" \
            ./tests/test_gpu_xdev "$scenario" \
            >"$base.log" 2>&1 || rc=$?
        (( rc == 0 )) || {
            tail -n 120 "$base.log" >&2 || true
            die "ncu failed: $shape dense-F16 cuBLAS"
        }
        [[ -s $base.ncu-rep ]] || die "ncu omitted $shape cuBLAS report"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" \
            --csv --page raw >"$base.csv" 2>"$base-import.log"
        python3 speed-bench/validate-ncu-capture.py \
            "$base.csv" '(?i)(gemm|mma)' 1 --process test_gpu_xdev
    }
    if [[ $NCU_SET == full ]]; then
        profile_dense_cublas t32 q8-partner-t32-profile
        profile_dense_cublas t256 q8-partner-t256-profile
    fi
fi

phase=complete
for required in summary.md combined-profile.json combined-kernel-groups.csv \
                kernel-name-groups.csv kernel-groups-device-stage.csv \
                unknown-kernels.csv partner-t256-ranges.csv \
                stage-device-summary.csv partner-projection-summary.csv \
                attention-row-split-summary.csv nsys/combined.nsys-rep \
                nsys/bindings.csv nsys/allocations.csv nsys/memory.csv \
                nsys/plan.csv nsys/q8-cache-audit.csv \
                nsys/routed-tile-audit.csv; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing final evidence: $required"
done
printf 'Combined SM75 production profile complete: %s\n' "$OUTPUT_DIR"
