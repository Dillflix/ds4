#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_TOKENS=${CTX_TOKENS:-2048}
CTX_ALLOC=${CTX_ALLOC:-$((CTX_TOKENS + 1))}
PREFILL_CHUNK=${PREFILL_CHUNK:-512}
PIPELINE_MB=512
TOKEN_ROW_WEIGHT_MODE=${TOKEN_ROW_WEIGHT_MODE:-native-stream}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-1800}
MIN_THROUGHPUT_RATIO=${MIN_THROUGHPUT_RATIO:-0.90}
if [[ $TOKEN_ROW_WEIGHT_MODE == native-stream ]]; then
    MAX_MODEL_CACHE_INCREASE_GIB=${MAX_MODEL_CACHE_INCREASE_GIB:-0}
    MAX_PER_GPU_VRAM_INCREASE_MIB=${MAX_PER_GPU_VRAM_INCREASE_MIB:-2048}
    MAX_AGGREGATE_VRAM_INCREASE_MIB=${MAX_AGGREGATE_VRAM_INCREASE_MIB:-0}
    MIN_MODEL_CACHE_SAVING_GIB=${MIN_MODEL_CACHE_SAVING_GIB:-3.50}
    MIN_AGGREGATE_VRAM_SAVING_MIB=${MIN_AGGREGATE_VRAM_SAVING_MIB:-1024}
else
    MAX_MODEL_CACHE_INCREASE_GIB=${MAX_MODEL_CACHE_INCREASE_GIB:-4.50}
    MAX_PER_GPU_VRAM_INCREASE_MIB=${MAX_PER_GPU_VRAM_INCREASE_MIB:-5120}
    MAX_AGGREGATE_VRAM_INCREASE_MIB=${MAX_AGGREGATE_VRAM_INCREASE_MIB:-5120}
    MIN_MODEL_CACHE_SAVING_GIB=${MIN_MODEL_CACHE_SAVING_GIB:-0}
    MIN_AGGREGATE_VRAM_SAVING_MIB=${MIN_AGGREGATE_VRAM_SAVING_MIB:-0}
fi
MIN_CANDIDATE_FREE_VRAM_MIB=${MIN_CANDIDATE_FREE_VRAM_MIB:-2048}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-200}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${TOKEN_ROW_ATTN_EXACTNESS_DIR:-$repo_dir/sm75-token-row-attention-exactness-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the tagged all43 SM75 native-Q8 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
[[ $GPU_DEVICES == 0,3,1,2 ]] ||
    die "GPU_DEVICES must be 0,3,1,2 so logical pair 1 is stable physical GPU3/GPU2"
[[ $REQUIRED_POWER_LIMITS_W == 250,260,250,250 ]] ||
    die "REQUIRED_POWER_LIMITS_W must be 250,260,250,250"
[[ $TOKEN_ROW_WEIGHT_MODE == f16 ||
   $TOKEN_ROW_WEIGHT_MODE == native-stream ]] ||
    die "TOKEN_ROW_WEIGHT_MODE must be f16 or native-stream"
if [[ $TOKEN_ROW_WEIGHT_MODE == native-stream && $PREFILL_CHUNK != 512 ]]; then
    die "native-stream is qualified only for PREFILL_CHUNK=512"
fi
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_TOKENS:$CTX_TOKENS" \
            "CTX_ALLOC:$CTX_ALLOC" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "CASE_TIMEOUT_SECONDS:$CASE_TIMEOUT_SECONDS" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "MIN_CANDIDATE_FREE_VRAM_MIB:$MIN_CANDIDATE_FREE_VRAM_MIB" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 && CTX_TOKENS >= 512 &&
   PREFILL_CHUNK >= 512 && PREFILL_CHUNK <= 2048 &&
   PREFILL_CHUNK % 512 == 0 && CTX_TOKENS % PREFILL_CHUNK == 0 &&
   CTX_ALLOC > CTX_TOKENS && CASE_TIMEOUT_SECONDS >= 60 &&
   TELEMETRY_INTERVAL_MS >= 50 )) || die "invalid benchmark bounds"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
for item in "MIN_THROUGHPUT_RATIO:$MIN_THROUGHPUT_RATIO" \
            "MAX_MODEL_CACHE_INCREASE_GIB:$MAX_MODEL_CACHE_INCREASE_GIB" \
            "MAX_PER_GPU_VRAM_INCREASE_MIB:$MAX_PER_GPU_VRAM_INCREASE_MIB" \
            "MAX_AGGREGATE_VRAM_INCREASE_MIB:$MAX_AGGREGATE_VRAM_INCREASE_MIB" \
            "MIN_MODEL_CACHE_SAVING_GIB:$MIN_MODEL_CACHE_SAVING_GIB" \
            "MIN_AGGREGATE_VRAM_SAVING_MIB:$MIN_AGGREGATE_VRAM_SAVING_MIB"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$name must be numeric"
done
for tool in awk cat cmp date env find git grep journalctl make mkdir mv nproc \
            nvidia-smi python3 rm sort stat tail tar timeout; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{health,production,provenance,telemetry}
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
            printf '%s: %s\n' \
                "$([[ $status == 0 ]] && printf 'Archive to return' || printf 'Diagnostic archive')" \
                "$archive" >&2
        else
            printf 'error: failed to create archive %s\n' "$archive" >&2
            [[ $status != 0 ]] || status=1
            rm -f -- "$partial"
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
clean+=(CUDA_DEVICE_ORDER=PCI_BUS_ID)

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in 0 1 2 3; do
        timeout 20s nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

validate_gpu_health() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] &&
        ! grep -Eiq 'ERR!|Unknown Error|GPU is lost|GPU Unavailable|Critical Xid' \
            "$base.pre-gpu.csv" "$base.post-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/health/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/health/initial-gpu.csv" "$base.post-gpu.csv"
}

start_telemetry() {
    local output=$1
    nvidia-smi \
        --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits -lms "$TELEMETRY_INTERVAL_MS" \
        >"$output" 2>&1 &
    telemetry_pid=$!
}

phase=build
targets=(ds4-bench tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 240 "$OUTPUT_DIR/build.log" >&2
            die "build failed"
        }
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/placement-tests.log" >&2
        die "placement regression tests failed"
    }

phase=topology
nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt"
for pair in 'GPU3 GPU2'; do
    read -r first second <<<"$pair"
    forward=$(awk -v from="$first" -v to="$second" '
        !h {for(i=1;i<=NF;i++) if($i==to)c=i+1; if(c){h=1;next}}
        h && $1==from {print $c;exit}' "$OUTPUT_DIR/provenance/topology.txt")
    reverse=$(awk -v from="$second" -v to="$first" '
        !h {for(i=1;i<=NF;i++) if($i==to)c=i+1; if(c){h=1;next}}
        h && $1==from {print $c;exit}' "$OUTPUT_DIR/provenance/topology.txt")
    [[ $forward =~ ^NV[0-9]+$ && $reverse =~ ^NV[0-9]+$ ]] ||
        die "$first<->$second is not bidirectional NVLink: ${forward:-missing}/${reverse:-missing}"
done
capture_gpu_health "$OUTPUT_DIR/health/initial-gpu.csv" ||
    die "could not capture initial four-GPU health"
python3 - "$OUTPUT_DIR/health/initial-gpu.csv" "$REQUIRED_POWER_LIMITS_W" <<'PY'
import csv, pathlib, sys
rows = list(csv.reader(pathlib.Path(sys.argv[1]).open()))
expected = [float(x) for x in sys.argv[2].split(",")]
if len(rows) != 4 or len(expected) != 4:
    raise SystemExit("error: incomplete initial GPU inventory")
for physical_index, (row, limit) in enumerate(zip(rows, expected)):
    if len(row) != 4 or int(row[0].strip()) != physical_index:
        raise SystemExit("error: GPU index inventory changed")
    if abs(float(row[3].strip()) - limit) > 0.01:
        raise SystemExit(
            f"error: GPU {physical_index} power limit {row[3].strip()} != {limit:.2f}")
PY

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstable_physical_pair=3,2\nstage_split=22/21\n' \
        "$GPU_DEVICES"
    printf 'ctx_tokens=%s\nctx_alloc=%s\nprefill_chunk=%s\npipeline_microbatch=%s\n' \
        "$CTX_TOKENS" "$CTX_ALLOC" "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'control_token_row_pairs=off\ncandidate_token_row_pairs=1\npair0_attention=off-both-arms\n'
    printf 'candidate_token_row_weight_mode=%s\n' "$TOKEN_ROW_WEIGHT_MODE"
    printf 'candidate_output_b_algorithm=CUBLAS_GEMM_ALGO3_TENSOR_OP\n'
    printf 'minimum_throughput_ratio=%s\nmax_model_cache_increase_gib=%s\n' \
        "$MIN_THROUGHPUT_RATIO" "$MAX_MODEL_CACHE_INCREASE_GIB"
    printf 'max_per_gpu_vram_increase_mib=%s\nmax_aggregate_vram_increase_mib=%s\n' \
        "$MAX_PER_GPU_VRAM_INCREASE_MIB" "$MAX_AGGREGATE_VRAM_INCREASE_MIB"
    printf 'min_model_cache_saving_gib=%s\nmin_aggregate_vram_saving_mib=%s\n' \
        "$MIN_MODEL_CACHE_SAVING_GIB" "$MIN_AGGREGATE_VRAM_SAVING_MIB"
    printf 'min_candidate_free_vram_mib=%s\nrequired_power_limits_w=%s\n' \
        "$MIN_CANDIDATE_FREE_VRAM_MIB" "$REQUIRED_POWER_LIMITS_W"
    cat "$OUTPUT_DIR/health/initial-gpu.csv"
    printf '\ntopology:\n'
    cat "$OUTPUT_DIR/provenance/topology.txt"
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=production-ab
printf 'variant,csv,log,logits,telemetry,pre_health,post_health,kernel_log\n' \
    >"$OUTPUT_DIR/production/runs.csv"
for variant in control token-row; do
    base="$OUTPUT_DIR/production/$variant"
    logits="$base-logits"
    telemetry="$OUTPUT_DIR/telemetry/$variant.csv"
    mkdir -p "$logits"
    variant_env=()
    if [[ $variant == token-row ]]; then
        # Deliberately pair 1 only. Pair 0 is never passed to the selector.
        variant_env+=(DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_PIPELINE_PAIRS=1)
        variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_WEIGHT_MODE=$TOKEN_ROW_WEIGHT_MODE")
    fi
    capture_gpu_health "$base.pre-gpu.csv" ||
        die "$variant pre-run GPU health failed"
    cmp -s "$OUTPUT_DIR/health/initial-gpu.csv" "$base.pre-gpu.csv" ||
        die "$variant pre-run GPU identity, power, or health changed"
    arm_start=$(date --iso-8601=seconds)
    printf 'SM75 token-row-through-attention exactness A/B variant=%s...\n' "$variant"
    start_telemetry "$telemetry"
    set +e
    timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        "${clean[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB" \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
        DS4_CUDA_TP_PREFILL_T32_HEADS=0 \
        DS4_CUDA_TP_PREFILL_ATTN_ROWS_OUTPUT=0 \
        DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0 \
        "${variant_env[@]}" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$CTX_TOKENS" --ctx-max "$CTX_TOKENS" \
            --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
            --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
            --csv "$base.csv" --dump-frontier-logits-dir "$logits" \
            >"$base.log" 2>&1
    status=$?
    set -e
    stop_telemetry
    journalctl -k --since "$arm_start" --no-pager \
        >"$base.kernel.log" 2>&1 || true
    capture_gpu_health "$base.post-gpu.csv" ||
        die "$variant post-run GPU health failed after workload status $status"
    validate_gpu_health "$base" ||
        die "$variant changed GPU identity, power, or health"
    ! grep -Eiq 'NVRM: Xid|GPU has fallen off|GPU Unavailable|Critical Xid' \
        "$base.log" "$base.kernel.log" || die "$variant recorded a GPU fault"
    if (( status != 0 )); then
        tail -n 260 "$base.log" >&2
        die "$variant production run failed with status $status"
    fi
    [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
    grep -Fq 'dense-placement=stage-aware-fixed-22-21' "$base.log" ||
        die "$variant missed fixed 22/21 dense placement"
    if [[ $variant == token-row &&
          $TOKEN_ROW_WEIGHT_MODE == native-stream ]]; then
        grep -Fq 'tagged SM75 dense-Q8 GGUF installed through ordinary startup selective residency; selected token-row pair q_b/A/B sources are complete and local on both owners; runtime replacement and peer weight reads disabled' \
            "$base.log" || die "$variant did not install pair-local tagged-Q8 sources"
    else
        grep -Fq 'tagged SM75 dense-Q8 GGUF installed through ordinary single-owner residency' \
            "$base.log" || die "$variant did not load the tagged native-Q8 model"
    fi
    grep -Fq 'CUDA TP cache mirror policy: attention-pair-mask=0x2' "$base.log" ||
        die "$variant did not keep attention cache visibility on stable pair 1 only"
    ! grep -Fq 'required native-GGUF execution binding unavailable' "$base.log" ||
        die "$variant missed a required native-GGUF execution binding"
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$base.log" ||
        die "$variant unexpectedly enabled the unstable legacy pair-0 splitter"
    ! grep -Eq 'CUDA prefill attention token-row pipeline .*home=0([[:space:]]|$)' "$base.log" ||
        die "$variant unexpectedly enabled the token-row pipeline on pair 0"

    if [[ $variant == control ]]; then
        grep -Fq 'token-row-pair-mask=0x0 token-row-weight-mode=f16 token-row-required=0/0/0 token-row-native-stream=0/0/0' \
            "$base.log" || die "control did not retain the ordinary token-row binding inventory"
        ! grep -Fq 'CUDA prefill attention token-row pipeline enabled:' "$base.log" ||
            die "control unexpectedly enabled the token-row candidate"
        ! grep -Fq 'SM75 row-owned attention output B selected:' "$base.log" ||
            die "control unexpectedly selected the candidate output-B algorithm"
        if (( CTX_TOKENS > PREFILL_CHUNK )); then
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$base.log" ||
                die "control missed the established stable pair-1 path"
        fi
    else
        expected_input_bytes=$(((PIPELINE_MB / 2) * 1024 * 4))
        expected_output_bytes=$(((PIPELINE_MB / 2) * 4096 * 4))
        expected_current_kv_bytes=$((PIPELINE_MB * 512 * 4))
        if [[ $TOKEN_ROW_WEIGHT_MODE == native-stream ]]; then
            grep -Fq 'token-row-pair-mask=0x2 token-row-weight-mode=native-stream token-row-required=0/0/0 token-row-native-stream=21/21/21' \
                "$base.log" || die "candidate did not retain all 21 pair-1 q_b/A/B native-stream sources"
            grep -Fq 'token-row native-stream workspace reserved at startup: 96 MiB per selected pair member; runtime growth disabled' \
                "$base.log" || die "candidate did not pre-reserve bounded native-stream workspace"
            for stage in attn_q_b attn_output_a attn_output_b; do
                [[ $(grep -Fc "token-row native-stream dispatch stage=$stage " "$base.log") == 2 ]] ||
                    die "candidate did not native-stream $stage on both pair members"
            done
            [[ $(grep -Fc 'dequant=group-int8x4' "$base.log") == 6 ]] ||
                die "candidate did not use grouped int8x4 native-stream dequant for all pair-local projections"
            ! grep -Eq 'token-row native-stream dispatch .*peer-weight-read=[1-9]' "$base.log" ||
                die "candidate performed a forbidden peer weight read"
        else
            grep -Fq 'token-row-pair-mask=0x2 token-row-weight-mode=f16 token-row-required=21/21/21 token-row-native-stream=0/0/0' \
                "$base.log" || die "candidate did not materialize all 21 pair-1 q_b/A/B F16 execution bindings"
        fi
        grep -Fq "CUDA prefill attention token-row pipeline enabled: home=1 partner=3 q-input-copy-bytes=$expected_input_bytes query-gather-bytes=0 output-return-bytes=$expected_output_bytes rows=256/256" \
            "$base.log" || die "candidate omitted or changed the bounded token-row transfer contract"
        grep -Fq 'CUDA prefill attention token-row pipeline attention enabled: home=1 partner=3 rows=256/256 query=local-token-rows KV=local-mirrors output=local-A+B' \
            "$base.log" || die "candidate did not keep q_b through attention A+B row-local"
        grep -Fq "CUDA prefill attention token-row pipeline output enabled: home=1 partner=3 rows=256/256 output-return-bytes=$expected_output_bytes result=full-N_EMBD-rows" \
            "$base.log" || die "candidate did not complete local output B and return only final N_EMBD rows"
        [[ $(grep -Fc 'SM75 row-owned attention output B selected:' "$base.log") == 2 ]] ||
            die "candidate did not select exact output-B algorithm on both pair members"
        grep -Fq 'logical=1 physical=3 algorithm=CUBLAS_GEMM_ALGO3_TENSOR_OP rows=256' \
            "$base.log" || die "candidate missed exact output-B on the home pair member"
        grep -Fq 'logical=3 physical=2 algorithm=CUBLAS_GEMM_ALGO3_TENSOR_OP rows=256' \
            "$base.log" || die "candidate missed exact output-B on the partner pair member"
        grep -Fq "exact current-KV mirror enabled: home=1 partner=3 bytes=$expected_current_kv_bytes storage=f32-current-batch" \
            "$base.log" || die "candidate did not expose its exact zero-prefix current-KV transfer"
        ! grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$base.log" ||
            die "candidate fell back to the legacy pair-1 expanded-query path"
        ! grep -Eq 'CUDA prefill attention token-row pipeline enabled: .*query-gather-bytes=[1-9][0-9]*' \
            "$base.log" || die "candidate performed a forbidden expanded-query gather"
        grep -E 'CUDA prefill attention token-row pipeline (enabled|attention enabled|output enabled):|exact current-KV mirror enabled:' \
            "$base.log" >"$OUTPUT_DIR/production/token-row-traffic.txt"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$variant" "$base.csv" \
        "$base.log" "$logits" "$telemetry" "$base.pre-gpu.csv" \
        "$base.post-gpu.csv" "$base.kernel.log" \
        >>"$OUTPUT_DIR/production/runs.csv"
done

phase=exactness
mapfile -t control_files < <(find "$OUTPUT_DIR/production/control-logits" \
    -maxdepth 1 -type f -printf '%f\n' | sort)
mapfile -t candidate_files < <(find "$OUTPUT_DIR/production/token-row-logits" \
    -maxdepth 1 -type f -printf '%f\n' | sort)
[[ ${#control_files[@]} -gt 0 && "${control_files[*]}" == "${candidate_files[*]}" ]] ||
    die "logits inventory differs"
for file in "${control_files[@]}"; do
    cmp -s "$OUTPUT_DIR/production/control-logits/$file" \
           "$OUTPUT_DIR/production/token-row-logits/$file" ||
        die "logits differ: $file"
done

phase=summary
python3 - "$OUTPUT_DIR" "$MIN_THROUGHPUT_RATIO" \
    "$MAX_MODEL_CACHE_INCREASE_GIB" "$MAX_PER_GPU_VRAM_INCREASE_MIB" \
    "$MAX_AGGREGATE_VRAM_INCREASE_MIB" "$MIN_CANDIDATE_FREE_VRAM_MIB" \
    "$PIPELINE_MB" "$TOKEN_ROW_WEIGHT_MODE" \
    "$MIN_MODEL_CACHE_SAVING_GIB" "$MIN_AGGREGATE_VRAM_SAVING_MIB" <<'PY'
import csv, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
minimum = float(sys.argv[2])
max_cache_growth = float(sys.argv[3])
max_per_gpu_growth = float(sys.argv[4])
max_aggregate_growth = float(sys.argv[5])
min_candidate_free = float(sys.argv[6])
microbatch = int(sys.argv[7])
weight_mode = sys.argv[8]
min_cache_saving = float(sys.argv[9])
min_aggregate_saving = float(sys.argv[10])

def csv_row(name):
    rows = list(csv.DictReader((root / "production" / f"{name}.csv").open()))
    if len(rows) != 1:
        raise SystemExit(f"error: expected one {name} CSV row, found {len(rows)}")
    if float(rows[0]["prefill_tps"]) <= 0:
        raise SystemExit(f"error: {name} has invalid prefill throughput")
    return rows[0]

def cache_gib(name):
    text = (root / "production" / f"{name}.log").read_text(errors="replace")
    matches = re.findall(
        r"CUDA q8 fp16 benefit plan materialized \d+/\d+ candidates \(([0-9.]+) GiB\)",
        text)
    if not matches:
        raise SystemExit(f"error: {name} omitted materialized-cache bytes")
    return float(matches[-1])

def telemetry(name):
    max_used, min_free = {}, {}
    path = root / "telemetry" / f"{name}.csv"
    for row in csv.reader(path.open(errors="replace")):
        if len(row) < 5:
            continue
        try:
            gpu = int(row[1].strip())
            used = float(row[3].strip())
            free = float(row[4].strip())
        except ValueError:
            continue
        max_used[gpu] = max(max_used.get(gpu, 0.0), used)
        min_free[gpu] = min(min_free.get(gpu, free), free)
    if set(max_used) != {0, 1, 2, 3}:
        raise SystemExit(f"error: {name} telemetry did not observe all four GPUs")
    return max_used, min_free

control = csv_row("control")
candidate = csv_row("token-row")
control_tps = float(control["prefill_tps"])
candidate_tps = float(candidate["prefill_tps"])
ratio = candidate_tps / control_tps
control_cache = cache_gib("control")
candidate_cache = cache_gib("token-row")
cache_growth = candidate_cache - control_cache
control_used, _ = telemetry("control")
candidate_used, candidate_free = telemetry("token-row")
per_gpu_growth = {
    gpu: candidate_used[gpu] - control_used[gpu] for gpu in range(4)
}
aggregate_growth = sum(candidate_used.values()) - sum(control_used.values())
candidate_min_free = min(candidate_free.values())

if ratio < minimum:
    raise SystemExit(
        f"error: token-row ratio {ratio:.6f} is below {minimum:.6f}")
if cache_growth > max_cache_growth + 1e-9:
    raise SystemExit(
        f"error: candidate model cache grew {cache_growth:.2f} GiB, "
        f"above {max_cache_growth:.2f} GiB")
if max(per_gpu_growth.values()) > max_per_gpu_growth + 1e-9:
    raise SystemExit(
        f"error: candidate per-GPU VRAM growth {max(per_gpu_growth.values()):.0f} MiB "
        f"exceeds {max_per_gpu_growth:.0f} MiB")
if aggregate_growth > max_aggregate_growth + 1e-9:
    raise SystemExit(
        f"error: candidate aggregate VRAM growth {aggregate_growth:.0f} MiB "
        f"exceeds {max_aggregate_growth:.0f} MiB")
if -cache_growth < min_cache_saving - 1e-9:
    raise SystemExit(
        f"error: candidate model-cache saving {-cache_growth:.2f} GiB "
        f"is below {min_cache_saving:.2f} GiB")
if -aggregate_growth < min_aggregate_saving - 1e-9:
    raise SystemExit(
        f"error: candidate aggregate VRAM saving {-aggregate_growth:.0f} MiB "
        f"is below {min_aggregate_saving:.0f} MiB")
if candidate_min_free < min_candidate_free - 1e-9:
    raise SystemExit(
        f"error: candidate minimum free VRAM {candidate_min_free:.0f} MiB "
        f"is below {min_candidate_free:.0f} MiB")

q_input = (microbatch // 2) * 1024 * 4
current_kv = microbatch * 512 * 4
output_return = (microbatch // 2) * 4096 * 4
with (root / "summary.txt").open("w") as f:
    f.write(f"control_prefill_tps={control_tps:.6f}\n")
    f.write(f"token_row_prefill_tps={candidate_tps:.6f}\n")
    f.write(f"token_row_over_control={ratio:.6f}\n")
    f.write(f"control_model_cache_gib={control_cache:.2f}\n")
    f.write(f"token_row_model_cache_gib={candidate_cache:.2f}\n")
    f.write(f"model_cache_increase_gib={cache_growth:.2f}\n")
    f.write(f"model_cache_saving_gib={-cache_growth:.2f}\n")
    f.write(f"token_row_weight_mode={weight_mode}\n")
    f.write(f"q_input_transfer_bytes_per_pair1_layer_microbatch={q_input}\n")
    f.write(f"current_kv_transfer_bytes_per_pair1_layer_zero_prefix={current_kv}\n")
    f.write("expanded_query_gather_bytes_per_pair1_layer_microbatch=0\n")
    f.write(f"final_output_return_bytes_per_pair1_layer_microbatch={output_return}\n")
    f.write(f"steady_token_row_payload_bytes_per_pair1_layer_microbatch={q_input + output_return}\n")
    f.write("pair0_attention=off-both-arms\n")
    f.write("candidate_pair=logical1-physical-gpu3-gpu2\n")
    f.write("candidate_output_b_algorithm=CUBLAS_GEMM_ALGO3_TENSOR_OP\n")
    f.write("logits=bit-exact\n")
    f.write(f"candidate_aggregate_vram_increase_mib={aggregate_growth:.0f}\n")
    f.write(f"candidate_aggregate_vram_saving_mib={-aggregate_growth:.0f}\n")
    f.write(f"candidate_min_free_vram_mib={candidate_min_free:.0f}\n")
    for gpu in range(4):
        f.write(f"gpu{gpu}_control_max_vram_mib={control_used[gpu]:.0f}\n")
        f.write(f"gpu{gpu}_token_row_max_vram_mib={candidate_used[gpu]:.0f}\n")
        f.write(f"gpu{gpu}_token_row_vram_delta_mib={per_gpu_growth[gpu]:.0f}\n")
print((root / "summary.txt").read_text(), end="")
PY

phase=complete
printf 'SM75 stable-pair token-row-through-attention exactness A/B complete: %s\n' \
    "$OUTPUT_DIR"
