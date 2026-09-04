#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'EOF'
Qualify the 21/22 four-GPU SM75 prefill placement against production 22/21.

The A/B changes only DS4_CUDA_EP_STAGE_SPLIT. Layer 21 uses home attention in
both arms because it crosses between pair 0 (production home-attention policy)
and pair 1 (production row-split policy); every other pair-1 layer retains
row splitting. Both arms retain the production pipeline, dense-F16 cache plan,
pair-0 attention-row suppression, and FP32 T256 final-result boundary.

Optional environment:
  MIXED_MODEL=/absolute/path/to/mixed15.gguf
  ALL43_MODEL=/absolute/path/to/all43.gguf
  PROMPT=/absolute/path/to/prompt.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  REQUIRED_POWER_LIMITS_W=250,260,250,250  physical GPU 0,1,2,3 order
  REPEATS=3
  TELEMETRY_INTERVAL_MS=500
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PREFILL_PLACEMENT_21_22_DIR=/absolute/output/directory
EOF
    exit 0
fi
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
REPEATS=${REPEATS:-3}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-500}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CTX_ALLOC=33025
PREFILL_CHUNK=2048
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PREFILL_PLACEMENT_21_22_DIR:-$repo_dir/sm75-prefill-placement-21-22-ab-$stamp}

for model in "$MIXED_MODEL" "$ALL43_MODEL"; do
    [[ $model == /* && -f $model ]] || die "model not found: $model"
done
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto ]] ||
    die "require GPU_DEVICES=0,3,1,2 and GPU_VRAM=auto"
for item in "REPEATS:$REPEATS" "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( REPEATS >= 1 && TELEMETRY_INTERVAL_MS >= 100 )) ||
    die "REPEATS must be positive and TELEMETRY_INTERVAL_MS must be at least 100"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] || die "CUDA_VISIBLE_DEVICES must be unset"
for tool in awk basename cmp date dirname env git grep make mkdir mv nproc \
            nvidia-smi python3 rm sort tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,telemetry,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

telemetry_pid=
stop_telemetry() {
    if [[ -n ${telemetry_pid:-} ]]; then
        kill "$telemetry_pid" >/dev/null 2>&1 || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=
    fi
}
phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    stop_telemetry
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            rm -f -- "$partial" "$archive"
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

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_limits <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_limits[@]} == 4 )) ||
    die "GPU_DEVICES must select four GPUs and REQUIRED_POWER_LIMITS_W must" \
        "contain physical GPU 0,1,2,3 limits"
for i in 0 1 2 3; do
    gpu=${gpu_ids[$i]}
    [[ $gpu =~ ^[0-3]$ ]] || die "invalid physical GPU index: $gpu"
    want=${required_limits[$gpu]}
    [[ $want =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        die "invalid power limit for physical GPU $gpu: $want"
    actual=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v a="$actual" -v r="$want" \
        'BEGIN { exit !((a-r < 0.01) && (r-a < 0.01)) }' ||
        die "GPU $gpu power limit is $actual W, expected $want W"
done

capture_health() {
    local path=$1 gpu
    : >"$path"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$path" 2>&1 || return 1
    done
}
start_telemetry() {
    local output=$1
    nvidia-smi -i "$GPU_DEVICES" \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,memory.used \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" >"$output" 2>&1 &
    telemetry_pid=$!
}

phase=build
targets=(ds4-bench tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/placement-tests.log" >&2
        die "placement regression tests failed"
    }

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'mixed_model=%s\nall43_model=%s\nprompt=%s\n' \
        "$MIXED_MODEL" "$ALL43_MODEL" "$PROMPT"
    printf 'gpu_devices=%s\nrequired_power_limits_w=%s\n' \
        "$GPU_DEVICES" "$REQUIRED_POWER_LIMITS_W"
    printf 'control_split=22/21\ncandidate_split=21/22\n'
    printf 'contexts=512,4096,32768\nrepeats=%s\nprefill_chunk=%s\n' \
        "$REPEATS" "$PREFILL_CHUNK"
    printf 't256_result_boundary=fp32-unchanged\nchanged_axis=transformer-stage-boundary-only\n'
    printf 'matched_attention_boundary_layer=21\nmatched_attention_policy=home-compute\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'; nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
capture_health "$OUTPUT_DIR/initial-gpu.csv" || die "initial GPU health capture failed"

production=(
    "${clean[@]}"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROW_COMPUTE_LAYERS=21
    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1
    DS4_CUDA_SCRATCH_REPLACEMENT_AUDIT=1
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_METAL_GRAPH_PREFILL_PROFILE=1
)

validate_run() {
    local split=$1 base=$2 log="$2.log" dense_marker
    [[ -s $base.csv ]] || return 1
    for ctx in 512 4096 32768; do
        [[ -s $base-logits/frontier_$(printf '%06d' "$ctx").logits.f32 ]] || return 1
        [[ -s $base-logits/frontier_$(printf '%06d' "$ctx").logits.json ]] || return 1
    done
    grep -Fq "CUDA EP forced pipeline split $split/$((43-split))" "$log" || return 1
    grep -Fq "CUDA q8 fp16 stage-aware $split/$((43-split)) planner selected" "$log" || return 1
    dense_marker=stage-aware-qualified-21-22
    [[ $split == 22 ]] && dense_marker=stage-aware-fixed-22-21
    grep -Fq "dense-placement=$dense_marker" "$log" || return 1
    grep -Fq 'materialized 344/344 candidates' "$log" || return 1
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    [[ $(grep -Fc 'CUDA prefill indexer score/top-k row split enabled:' "$log") == 2 ]] || return 1
    grep -Fq 'CUDA TP cache mirror policy: attention-pair-mask=0x2 index-pair-mask=0x3' "$log" || return 1
    grep -Fq 'CUDA prefill attention row compute layer-scoped disable: layers=21' "$log" || return 1
    ! grep -Eq 'CUDA prefill attention row audit dispatch=split .*layer=21 ' "$log" || return 1
    grep -Eq 'CUDA prefill attention row audit dispatch=split .*home=1 partner=3 ' "$log" || return 1
    grep -Fq 'CUDA scratch replacement quiesced all tiers:' "$log" || return 1
    [[ -s $base.q8-plan.csv && -s $base.q8-bindings.csv ]] || return 1
    ! grep -Fqi 'T256 FP16 final attention result enabled' "$log" || return 1
    grep -Eq 'CUDA T32 f16-output fused summary: local=[0-9]+ partner=[1-9][0-9]*' "$log" || return 1
    capture_health "$base.post-gpu.csv" || return 1
    cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

phase=production-ab
printf 'model_layout\trepeat\tslot\tsplit\tcsv\tlog\tlogits\ttelemetry\n' \
    >"$OUTPUT_DIR/runs.tsv"
layouts=(mixed15 all43)
models=("$MIXED_MODEL" "$ALL43_MODEL")
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for i in 0 1; do
        layout=${layouts[$i]}; model=${models[$i]}
        if (( (repeat + i) % 2 )); then splits=(22 21); else splits=(21 22); fi
        slot=0
        for split in "${splits[@]}"; do
            slot=$((slot + 1))
            base="$OUTPUT_DIR/runs/$layout-r$repeat-s$slot-split$split"
            telemetry="$OUTPUT_DIR/telemetry/$layout-r$repeat-s$slot-split$split.csv"
            mkdir -p "$base-logits"
            printf 'FP32 prefill placement A/B model=%s repeat=%s/%s slot=%s split=%s/%s...\n' \
                "$layout" "$repeat" "$REPEATS" "$slot" "$split" "$((43-split))"
            start_telemetry "$telemetry"
            run_status=0
            "${production[@]}" "DS4_CUDA_EP_STAGE_SPLIT=$split" \
                "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv" \
                "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv" \
                ./ds4-bench \
                --cuda --cuda-tensor-parallel --gpu-devices "$GPU_DEVICES" \
                --gpu-vram "$GPU_VRAM" --model "$model" --prompt-file "$PROMPT" \
                --ctx-start 512 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC" \
                --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
                --csv "$base.csv" --dump-frontier-logits-dir "$base-logits" \
                >"$base.log" 2>&1 || run_status=$?
            stop_telemetry
            if (( run_status != 0 )); then
                tail -n 220 "$base.log" >&2 || true
                die "$layout split $split run failed with status $run_status"
            fi
            [[ -s $telemetry && $(grep -c . "$telemetry") -ge 2 ]] ||
                die "$layout split $split omitted usable telemetry"
            validate_run "$split" "$base" || {
                tail -n 220 "$base.log" >&2 || true
                die "$layout split $split production validation failed"
            }
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$layout" "$repeat" "$slot" "$split" "$base.csv" \
                "$base.log" "$base-logits" "$telemetry" >>"$OUTPUT_DIR/runs.tsv"
        done
        ref_logits=$(awk -F'\t' -v m="$layout" -v r="$repeat" \
            '$1 == m && $2 == r && $4 == 22 {print $7}' "$OUTPUT_DIR/runs.tsv")
        cand_logits=$(awk -F'\t' -v m="$layout" -v r="$repeat" \
            '$1 == m && $2 == r && $4 == 21 {print $7}' "$OUTPUT_DIR/runs.tsv")
        for ctx in 512 4096 32768; do
            stem=frontier_$(printf '%06d' "$ctx")
            cmp -s "$ref_logits/$stem.logits.f32" "$cand_logits/$stem.logits.f32" ||
                die "$layout repeat $repeat split 21/22 changed FP32 logits at context $ctx"
            cmp -s "$ref_logits/$stem.logits.json" "$cand_logits/$stem.logits.json" ||
                die "$layout repeat $repeat split 21/22 changed logits metadata at context $ctx"
        done
    done
done

phase=summarize
python3 - "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/summary" "$GPU_DEVICES" <<'PY'
import csv, pathlib, statistics, sys

runs = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
gpu_ids = [int(x) for x in sys.argv[3].split(",")]
roles = {gpu_ids[0]: "stage0-home", gpu_ids[1]: "stage1-home",
         gpu_ids[2]: "stage0-partner", gpu_ids[3]: "stage1-partner"}

perf = {}
for run in runs:
    with open(run["csv"], newline="", encoding="utf-8") as stream:
        perf[(run["model_layout"], int(run["repeat"]), int(run["split"]))] = {
            int(row["ctx_tokens"]): float(row["prefill_tps"])
            for row in csv.DictReader(stream)
        }

samples = []
for layout in ("mixed15", "all43"):
    repeats = sorted({int(r["repeat"]) for r in runs if r["model_layout"] == layout})
    for repeat in repeats:
        for ctx in (512, 4096, 32768):
            control = perf[(layout, repeat, 22)][ctx]
            candidate = perf[(layout, repeat, 21)][ctx]
            samples.append({"model_layout": layout, "repeat": repeat,
                "ctx_tokens": ctx, "control_22_21_tps": control,
                "candidate_21_22_tps": candidate,
                "speedup_21_22_vs_22_21": candidate / control,
                "logits": "byte-identical"})

with (out / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=samples[0].keys())
    writer.writeheader(); writer.writerows(samples)

summary = []
for layout in ("mixed15", "all43"):
    for ctx in (512, 4096, 32768):
        rows = [r for r in samples if r["model_layout"] == layout and r["ctx_tokens"] == ctx]
        ratios = [r["speedup_21_22_vs_22_21"] for r in rows]
        summary.append({"model_layout": layout, "ctx_tokens": ctx,
            "control_22_21_median_tps": statistics.median(r["control_22_21_tps"] for r in rows),
            "candidate_21_22_median_tps": statistics.median(r["candidate_21_22_tps"] for r in rows),
            "paired_median_speedup": statistics.median(ratios),
            "paired_min_speedup": min(ratios), "paired_max_speedup": max(ratios),
            "logits": "byte-identical"})
with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
    writer.writeheader(); writer.writerows(summary)

telemetry_rows = []
pair_rows = []
for run in runs:
    by_time = {}
    with open(run["telemetry"], newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        key = lambda prefix: next((x for x in fields if x.strip().startswith(prefix)), None)
        time_key, index_key = key("timestamp"), key("index")
        util_key = key("utilization.gpu")
        power_key, clock_key = key("power.draw"), key("clocks.current.sm")
        for row in reader:
            try:
                timestamp = row[time_key].strip()
                gpu = int(row[index_key].strip())
                util = float(row[util_key].split()[0])
                power = float(row[power_key].split()[0])
                clock = float(row[clock_key].split()[0])
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            if gpu in roles:
                by_time.setdefault(timestamp, {})[gpu] = (util, power, clock)
    active = [snapshot for snapshot in by_time.values()
              if len(snapshot) == len(gpu_ids) and
              sum(snapshot[gpu][0] for gpu in gpu_ids) >= 20.0]
    if not active:
        raise SystemExit(f'error: no active four-GPU telemetry for {run["telemetry"]}')
    role_activity = {}
    for gpu in gpu_ids:
        values = [snapshot[gpu] for snapshot in active]
        avg_util = statistics.fmean(v[0] for v in values)
        role_activity[roles[gpu]] = avg_util
        telemetry_rows.append({"model_layout": run["model_layout"],
            "repeat": run["repeat"], "split": run["split"], "gpu": gpu,
            "role": roles[gpu], "system_active_samples": len(values),
            "mean_gpu_util_pct": avg_util,
            "median_power_w": statistics.median(v[1] for v in values),
            "median_sm_clock_mhz": statistics.median(v[2] for v in values)})
    stage0 = role_activity.get("stage0-home", 0.0) + role_activity.get("stage0-partner", 0.0)
    stage1 = role_activity.get("stage1-home", 0.0) + role_activity.get("stage1-partner", 0.0)
    pair_rows.append({"model_layout": run["model_layout"], "repeat": run["repeat"],
        "split": run["split"], "stage0_pair_active_util_sum": stage0,
        "stage1_pair_active_util_sum": stage1,
        "pair_activity_balance": min(stage0, stage1) / max(stage0, stage1)
            if stage0 > 0 and stage1 > 0 else 0.0})

for name, rows in (("telemetry.csv", telemetry_rows), ("pair-balance.csv", pair_rows)):
    if not rows:
        raise SystemExit(f"error: no usable {name} rows")
    with (out / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

print("model_layout,ctx_tokens,control_22_21_tps,candidate_21_22_tps,speedup,logits")
for row in summary:
    print(f'{row["model_layout"]},{row["ctx_tokens"]},'
          f'{row["control_22_21_median_tps"]:.3f},'
          f'{row["candidate_21_22_median_tps"]:.3f},'
          f'{row["paired_median_speedup"]:.6f},byte-identical')
print("\nmodel_layout,split,median_pair_activity_balance")
for layout in ("mixed15", "all43"):
    for split in (22, 21):
        vals = [r["pair_activity_balance"] for r in pair_rows
                if r["model_layout"] == layout and int(r["split"]) == split]
        print(f'{layout},{split}/{43-split},{statistics.median(vals):.6f}')
PY

phase=complete
printf 'SM75 FP32 prefill placement 21/22 A/B complete: %s\n' "$OUTPUT_DIR"
