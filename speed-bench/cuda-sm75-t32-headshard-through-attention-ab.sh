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
CTX_TOKENS=${CTX_TOKENS:-32768}
CTX_ALLOC=${CTX_ALLOC:-$((CTX_TOKENS + 1))}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-1800}
MIN_THROUGHPUT_RATIO=${MIN_THROUGHPUT_RATIO:-0.90}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-200}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CACHE_AUDIT_ONLY=${CACHE_AUDIT_ONLY:-0}
MATCH_PAIR1_INDEXER=${MATCH_PAIR1_INDEXER:-1}
BOUNDARY_AUDIT_ONLY=${BOUNDARY_AUDIT_ONLY:-0}
BOUNDARY_AUDIT_LAYER=${BOUNDARY_AUDIT_LAYER:-22}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T32_HEADSHARD_ATTN_AB_DIR:-$repo_dir/sm75-t32-headshard-attention-ab-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the tagged all43 SM75 native-Q8 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_TOKENS:$CTX_TOKENS" \
            "CTX_ALLOC:$CTX_ALLOC" "CASE_TIMEOUT_SECONDS:$CASE_TIMEOUT_SECONDS" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE" \
            "CACHE_AUDIT_ONLY:$CACHE_AUDIT_ONLY" \
            "MATCH_PAIR1_INDEXER:$MATCH_PAIR1_INDEXER" \
            "BOUNDARY_AUDIT_ONLY:$BOUNDARY_AUDIT_ONLY" \
            "BOUNDARY_AUDIT_LAYER:$BOUNDARY_AUDIT_LAYER"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 && CTX_TOKENS >= 2048 &&
   CTX_TOKENS % 512 == 0 && CTX_ALLOC > CTX_TOKENS &&
   CASE_TIMEOUT_SECONDS >= 60 && TELEMETRY_INTERVAL_MS >= 50 )) ||
    die "invalid benchmark bounds"
for flag in SKIP_BUILD CREATE_ARCHIVE CACHE_AUDIT_ONLY \
            MATCH_PAIR1_INDEXER BOUNDARY_AUDIT_ONLY; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
(( CACHE_AUDIT_ONLY + BOUNDARY_AUDIT_ONLY <= 1 )) ||
    die "CACHE_AUDIT_ONLY and BOUNDARY_AUDIT_ONLY are mutually exclusive"
(( BOUNDARY_AUDIT_LAYER < 43 )) || die "BOUNDARY_AUDIT_LAYER must be below 43"
[[ $MIN_THROUGHPUT_RATIO =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "MIN_THROUGHPUT_RATIO must be numeric"
for tool in awk cmp date env find git grep make mkdir nproc nvidia-smi \
            python3 sort stat tail tar timeout; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{production,provenance}
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
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s\n' "$archive"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

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
./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/placement-tests.log" >&2
        die "placement regression tests failed"
    }

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nctx_tokens=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$CTX_TOKENS"
    printf 'cache_audit_only=%s\nmatch_pair1_indexer=%s\n' \
        "$CACHE_AUDIT_ONLY" "$MATCH_PAIR1_INDEXER"
    printf 'boundary_audit_only=%s\nboundary_audit_layer=%s\n' \
        "$BOUNDARY_AUDIT_ONLY" "$BOUNDARY_AUDIT_LAYER"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=production-ab
printf 'variant,csv,log,logits,telemetry\n' >"$OUTPUT_DIR/production/runs.csv"
variants=(control headshard)
if [[ $CACHE_AUDIT_ONLY == 1 ]]; then variants=(headshard); fi
for variant in "${variants[@]}"; do
    enable=0; [[ $variant == headshard ]] && enable=1
    variant_env=()
    # The earlier production comparison changed two independent axes: the
    # attention decomposition and pair-1 indexer ownership.  Match the indexer
    # policy across both arms by default so the logit gate attributes any
    # difference only to row-split versus head-shard attention.  The old
    # asymmetric cut remains available solely to reproduce its evidence.
    if [[ $MATCH_PAIR1_INDEXER == 1 ]]; then
        variant_env+=(DS4_CUDA_NO_TP_PREFILL_INDEXER_ROWS_PAIRS=1)
    fi
    if [[ $variant == headshard ]]; then
        # Preserve the earlier asymmetric isolation only when explicitly
        # requested.  In the default matched mode the common block above has
        # already disabled pair-1 indexer splitting for both arms.
        if [[ $MATCH_PAIR1_INDEXER == 0 ]]; then
            variant_env+=(DS4_CUDA_NO_TP_PREFILL_INDEXER_ROWS_PAIRS=1)
        fi
        if [[ $CACHE_AUDIT_ONLY == 1 ]]; then
            variant_env+=(DS4_CUDA_T32_HEADSHARD_CACHE_AUDIT=1)
        fi
    fi
    if [[ $BOUNDARY_AUDIT_ONLY == 1 ]]; then
        boundary_dir="$OUTPUT_DIR/production/$variant-boundaries"
        mkdir -p "$boundary_dir"
        variant_env+=(
            "DS4_METAL_GRAPH_DUMP_PREFIX=$boundary_dir/$variant"
            "DS4_METAL_GRAPH_DUMP_LAYER=$BOUNDARY_AUDIT_LAYER"
            DS4_METAL_GRAPH_DUMP_NAME=headshard_audit_q_input,headshard_audit_kv_input,headshard_audit_output_b,headshard_audit_post_attention_hc,headshard_audit_layer_hc
        )
    fi
    base="$OUTPUT_DIR/production/$variant"
    logits="$base-logits"
    telemetry="$base-telemetry.csv"
    mkdir -p "$logits"
    printf 'SM75 T32 head-shard-through-attention A/B variant=%s...\n' "$variant"
    nvidia-smi \
        --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits -lms "$TELEMETRY_INTERVAL_MS" \
        >"$telemetry" 2>&1 &
    telemetry_pid=$!
    set +e
    timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        "${clean[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        DS4_CUDA_PREFILL_PIPELINE_MB=512 \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
        "DS4_CUDA_TP_PREFILL_T32_HEADS=$enable" \
        DS4_CUDA_TP_PREFILL_ATTN_ROWS_OUTPUT=0 \
        "${variant_env[@]}" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$CTX_TOKENS" --ctx-max "$CTX_TOKENS" \
            --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
            --prefill-chunk 512 --gen-tokens 0 --csv "$base.csv" \
            --dump-frontier-logits-dir "$logits" \
            >"$base.log" 2>&1
    status=$?
    set -e
    stop_telemetry
    if (( status != 0 )); then
        tail -n 260 "$base.log" >&2
        die "$variant production run failed with status $status"
    fi
    [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
    grep -Fq 'dense-placement=stage-aware-fixed-22-21' "$base.log" ||
        die "$variant missed fixed 22/21 dense placement"
    grep -Fq 'tagged SM75 dense-Q8 GGUF installed through ordinary single-owner residency' \
        "$base.log" || die "$variant did not load the tagged native-Q8 model"
    grep -Fq 'CUDA TP cache mirror policy: attention-pair-mask=0x2' "$base.log" ||
        die "$variant did not keep attention KV mirrors pair-1-only"
    ! grep -Fq 'required native-GGUF execution binding unavailable' "$base.log" ||
        die "$variant missed a required native-GGUF execution binding"
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$base.log" ||
        die "$variant unexpectedly enabled unstable pair 0"
    if (( CTX_TOKENS > 2048 )); then
        grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' \
            "$base.log" ||
            die "$variant did not preserve pair-0 indexer splitting"
    else
        # The all43 ratio-4 cache contains exactly TOP_K=512 rows at PP2048.
        # Split score/top-k dispatch begins only when n_comp > TOP_K, so the
        # absence of this line is the correct short-frontier control behavior.
        ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' \
            "$base.log" ||
            die "$variant unexpectedly dispatched pair-0 indexer splitting at PP2048"
    fi
    if [[ $variant == headshard ]]; then
        grep -Fq 'required-native=171/171' "$base.log" ||
            die "candidate did not materialize 21 pair-1 T32/A binding pairs"
        grep -Fq 'CUDA prefill T32 head shard enabled: home=1 partner=3 input-copy-bytes=2097152 query-gather-bytes=0 heads=32/32' \
            "$base.log" || die "candidate missed local T32 head-shard dispatch"
        grep -Fq 'query=local-T32-head-shards KV=local-mirrors' "$base.log" ||
            die "candidate attention did not consume local query/KV"
        grep -Fq 'CUDA prefill indexer row split pair policy: enabled-pairs=automatic disabled-pairs=1' \
            "$base.log" ||
            die "candidate did not disable pair-1 indexer splitting"
        ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' \
            "$base.log" ||
            die "candidate unexpectedly split pair-1 indexer/top-k"
        ! grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$base.log" ||
            die "candidate fell back to pair-1 query-row splitting"
    else
        grep -Fq 'required-native=129/129' "$base.log" ||
            die "control did not retain 129 required bindings"
        grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$base.log" ||
            die "control missed the established stable pair-1 row split"
        if [[ $MATCH_PAIR1_INDEXER == 1 ]]; then
            grep -Fq 'CUDA prefill indexer row split pair policy: enabled-pairs=automatic disabled-pairs=1' \
                "$base.log" ||
                die "control did not match the candidate's pair-1 indexer policy"
            ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' \
                "$base.log" ||
                die "control unexpectedly split pair-1 indexer/top-k"
        else
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' \
                "$base.log" ||
                die "control did not preserve pair-1 indexer splitting"
        fi
        ! grep -Fq 'CUDA prefill T32 head shard enabled:' "$base.log" ||
            die "control unexpectedly dispatched T32 head sharding"
    fi
    printf '%s,%s,%s,%s,%s\n' "$variant" "$base.csv" "$base.log" \
        "$logits" "$telemetry" >>"$OUTPUT_DIR/production/runs.csv"
done

if [[ $CACHE_AUDIT_ONLY == 1 ]]; then
    phase=cache-audit-summary
    audit_line=$(grep -F 'CUDA T32 head-shard cache mirror audit:' \
        "$OUTPUT_DIR/production/headshard.log" | tail -n 1 || true)
    [[ -n $audit_line ]] || die "candidate omitted cache-mirror audit result"
    [[ $audit_line != *read-failed* ]] || die "cache-mirror audit read failed"
    printf '%s\n' "$audit_line" >"$OUTPUT_DIR/cache-audit.txt"
    python3 - "$audit_line" >"$OUTPUT_DIR/summary.txt" <<'PY'
import re, sys
line = sys.argv[1]
fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
for required in ("checks", "bytes", "mismatch"):
    if required not in fields:
        raise SystemExit(f"error: cache audit omitted {required}")
print("mode=headshard-cache-audit-only")
print(f"cache_mirror_checks={fields['checks']}")
print(f"cache_mirror_bytes={fields['bytes']}")
print(f"cache_mirror_mismatch={fields['mismatch']}")
if fields["mismatch"] == "1":
    for name in ("kind", "layer", "row", "byte", "source",
                 "destination", "source_tier", "destination_tier"):
        print(f"first_mismatch_{name}={fields.get(name, 'missing')}")
PY
    cat "$OUTPUT_DIR/summary.txt"
    phase=complete
    printf 'SM75 T32 head-shard cache audit complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

if [[ $BOUNDARY_AUDIT_ONLY == 1 ]]; then
    phase=boundary-audit-summary
    python3 - "$OUTPUT_DIR" "$BOUNDARY_AUDIT_LAYER" <<'PY'
import math, pathlib, re, struct, sys

root = pathlib.Path(sys.argv[1])
layer = int(sys.argv[2])
stages = (
    "headshard_audit_q_input",
    "headshard_audit_kv_input",
    "headshard_audit_output_b",
    "headshard_audit_post_attention_hc",
    "headshard_audit_layer_hc",
)

def inventory(arm):
    base = root / "production" / f"{arm}-boundaries"
    result = {}
    pattern = re.compile(
        rf"{arm}_(headshard_audit_[^-]+)-{layer}_pos(\d+)\.bin$")
    for path in base.glob("*.bin"):
        match = pattern.fullmatch(path.name)
        if match:
            result[(int(match.group(2)), match.group(1))] = path
    return result

control = inventory("control")
candidate = inventory("headshard")
ctx_tokens = int(next(
    line.split("=", 1)[1]
    for line in (root / "manifest.txt").read_text().splitlines()
    if line.startswith("ctx_tokens=")))
expected = {
    (pos, stage)
    for pos in range(0, ctx_tokens, 512)
    for stage in stages
}
if set(control) != expected or set(candidate) != expected:
    missing_control = sorted(expected - set(control))
    missing_candidate = sorted(expected - set(candidate))
    raise SystemExit(
        "error: incomplete boundary inventory: "
        f"control_missing={missing_control} headshard_missing={missing_candidate}")

lines = [f"boundary_audit_layer={layer}"]
first = None
for pos in sorted({key[0] for key in expected}):
    for stage in stages:
        key = (pos, stage)
        a = control[key].read_bytes()
        b = candidate[key].read_bytes()
        if len(a) != len(b) or len(a) % 4:
            raise SystemExit(f"error: invalid boundary size for pos={pos} stage={stage}")
        af = struct.unpack(f"={len(a)//4}f", a)
        bf = struct.unpack(f"={len(b)//4}f", b)
        mismatches = 0
        max_abs = 0.0
        sum_sq = 0.0
        first_index = -1
        for i, (x, y) in enumerate(zip(af, bf)):
            if a[4*i:4*i+4] != b[4*i:4*i+4]:
                mismatches += 1
                if first_index < 0:
                    first_index = i
            delta = abs(float(y) - float(x))
            max_abs = max(max_abs, delta)
            sum_sq += delta * delta
        rmse = math.sqrt(sum_sq / len(af)) if af else 0.0
        status = "exact" if mismatches == 0 else "different"
        lines.append(
            f"boundary,pos={pos},stage={stage},values={len(af)},"
            f"status={status},bit_mismatches={mismatches},"
            f"first_index={first_index},max_abs={max_abs:.9g},rmse={rmse:.9g}")
        if mismatches and first is None:
            first = (pos, stage)

control_logits = root / "production" / "control-logits" / \
    f"frontier_{ctx_tokens:06d}.logits.f32"
candidate_logits = root / "production" / "headshard-logits" / \
    f"frontier_{ctx_tokens:06d}.logits.f32"
if not control_logits.is_file() or not candidate_logits.is_file():
    raise SystemExit("error: boundary audit omitted frontier logits")
logits_exact = control_logits.read_bytes() == candidate_logits.read_bytes()
lines.append("first_divergence=" +
             (f"pos{first[0]}:{first[1]}" if first else "none-in-sampled-boundaries"))
lines.append("frontier_logits=" + ("bit-exact" if logits_exact else "different"))
if first is None and logits_exact:
    lines.append("interpretation=dump-synchronization-made-candidate-converge")
elif first is None:
    lines.append("interpretation=divergence-outside-sampled-last-rows-or-after-audited-layer")
else:
    lines.append("interpretation=first-sampled-production-boundary-identified")
summary = "\n".join(lines) + "\n"
(root / "boundary-summary.txt").write_text(summary)
print(summary, end="")
PY
    phase=complete
    printf 'SM75 T32 head-shard boundary audit complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

phase=exactness
mapfile -t control_files < <(find "$OUTPUT_DIR/production/control-logits" \
    -maxdepth 1 -type f -printf '%f\n' | sort)
mapfile -t headshard_files < <(find "$OUTPUT_DIR/production/headshard-logits" \
    -maxdepth 1 -type f -printf '%f\n' | sort)
[[ ${#control_files[@]} -gt 0 && "${control_files[*]}" == "${headshard_files[*]}" ]] ||
    die "logits inventory differs"
for file in "${control_files[@]}"; do
    cmp -s "$OUTPUT_DIR/production/control-logits/$file" \
           "$OUTPUT_DIR/production/headshard-logits/$file" ||
        die "logits differ: $file"
done

phase=summary
python3 - "$OUTPUT_DIR" "$MIN_THROUGHPUT_RATIO" \
    "$MATCH_PAIR1_INDEXER" <<'PY'
import csv, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
minimum = float(sys.argv[2])
matched_indexer = bool(int(sys.argv[3]))

def csv_row(name):
    rows = list(csv.DictReader((root / "production" / f"{name}.csv").open()))
    if len(rows) != 1:
        raise SystemExit(f"error: expected one {name} CSV row, found {len(rows)}")
    return rows[0]

def cache_gib(name):
    text = (root / "production" / f"{name}.log").read_text(errors="replace")
    matches = re.findall(
        r"CUDA q8 fp16 benefit plan materialized \d+/\d+ candidates \(([0-9.]+) GiB\)",
        text)
    if not matches:
        raise SystemExit(f"error: {name} omitted materialized-cache bytes")
    return float(matches[-1])

def telemetry_max(name):
    result = {}
    path = root / "production" / f"{name}-telemetry.csv"
    for row in csv.reader(path.open(errors="replace")):
        if len(row) < 4:
            continue
        try:
            gpu = int(row[1].strip())
            used = float(row[3].strip())
        except ValueError:
            continue
        result[gpu] = max(result.get(gpu, 0.0), used)
    return result

control = csv_row("control")
headshard = csv_row("headshard")
ctps = float(control["prefill_tps"])
htps = float(headshard["prefill_tps"])
ratio = htps / ctps
control_cache = cache_gib("control")
headshard_cache = cache_gib("headshard")
if abs(headshard_cache - control_cache) > 0.02:
    raise SystemExit(
        f"error: candidate changed aggregate F16 model cache by "
        f"{headshard_cache-control_cache:.2f} GiB")
cm = telemetry_max("control")
hm = telemetry_max("headshard")
with (root / "summary.txt").open("w") as f:
    f.write(f"control_prefill_tps={ctps:.6f}\n")
    f.write(f"headshard_prefill_tps={htps:.6f}\n")
    f.write(f"headshard_over_control={ratio:.6f}\n")
    f.write(f"control_model_cache_gib={control_cache:.2f}\n")
    f.write(f"headshard_model_cache_gib={headshard_cache:.2f}\n")
    f.write("query_input_transfer_bytes_per_pair1_layer_chunk=2097152\n")
    f.write("query_result_gather_bytes_per_pair1_layer_chunk=0\n")
    f.write("partner_low_rank_return_bytes_per_pair1_layer_chunk=8388608\n")
    f.write("pair1_indexer_policy=" +
            ("matched-home-full" if matched_indexer else
             "control-split-candidate-home-full") + "\n")
    f.write("logits=bit-exact\n")
    for gpu in sorted(set(cm) | set(hm)):
        f.write(f"gpu{gpu}_control_max_vram_mib={cm.get(gpu, 0):.0f}\n")
        f.write(f"gpu{gpu}_headshard_max_vram_mib={hm.get(gpu, 0):.0f}\n")
print((root / "summary.txt").read_text(), end="")
if ratio < minimum:
    raise SystemExit(
        f"error: head-shard ratio {ratio:.6f} is below {minimum:.6f}")
PY

phase=complete
printf 'SM75 T32 head-shard-through-attention A/B complete: %s\n' "$OUTPUT_DIR"
