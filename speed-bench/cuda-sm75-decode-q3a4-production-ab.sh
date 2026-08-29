#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the paired production decode A/B for the SM75-native Q3A4 tile32-DP4A
gate/up kernel.  Q4-32 dispatch is identical in both arms.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REPEATS=3
  TG_TOKENS=256
  EXACT_TOKENS=16
  WARMUP_TOKENS=512
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  SKIP_BUILD=0
  RESUME=0
  CREATE_ARCHIVE=1
  Q3A4_PRODUCTION_AB_DIR=...

The fixed PP frontiers are 512, 4096, and 32768.  A fresh exact run evaluates
all three in one process.  RESUME preserves completed exact frontiers and runs
only a missing frontier, so a late hardware failure does not discard evidence.
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
REPEATS=${REPEATS:-3}
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
WARMUP_TOKENS=${WARMUP_TOKENS:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CTX_START=512
CTX_MAX=32768
STEP_MUL=8
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q3A4_PRODUCTION_AB_DIR:-$repo_dir/sm75-decode-q3a4-production-ab-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged Q4-32/Q3A4 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "REPEATS:$REPEATS" "TG_TOKENS:$TG_TOKENS" \
            "EXACT_TOKENS:$EXACT_TOKENS" "WARMUP_TOKENS:$WARMUP_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ && $value -ge 1 ]] ||
        die "$name must be a positive integer"
done
(( REPEATS >= 3 && TG_TOKENS == 256 && EXACT_TOKENS >= 2 &&
   WARMUP_TOKENS == 512 && PREFILL_CHUNK == 2048 && PIPELINE_MB == 512 )) ||
    die "require repeats>=3, tg_tokens=256, warmup=512, prefill_chunk=2048, pipeline_mb=512"
for flag in SKIP_BUILD RESUME CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
for tool in awk basename cat cmp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "GPU_DEVICES contains an invalid or duplicate ID: $gpu"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] ||
        die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done

emit_configuration() {
    printf 'model=%s\nprompt=%s\ngpu_devices=%s\ngpu_vram=%s\n' \
        "$MODEL" "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM"
    printf 'stage_split=%s\ncontexts=512,4096,32768\nrepeats=%s\n' \
        "$STAGE_SPLIT" "$REPEATS"
    printf 'tg_tokens=%s\nexact_tokens=%s\nwarmup_tokens=%s\n' \
        "$TG_TOKENS" "$EXACT_TOKENS" "$WARMUP_TOKENS"
    printf 'prefill_chunk=%s\npipeline_mb=%s\n' "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'q3a4_decode_default=tile32-dp4a\n'
}

legacy_resume=0
if [[ $RESUME == 1 ]]; then
    [[ -n ${Q3A4_PRODUCTION_AB_DIR:-} && -d $OUTPUT_DIR ]] ||
        die "RESUME=1 requires an existing Q3A4_PRODUCTION_AB_DIR"
    [[ -f $OUTPUT_DIR/configuration.txt ]] || die "resume configuration is missing"
    if ! cmp -s <(emit_configuration) "$OUTPUT_DIR/configuration.txt"; then
        if cmp -s <(emit_configuration | \
                    grep -Fv 'q3a4_decode_default=tile32-dp4a') \
                  "$OUTPUT_DIR/configuration.txt"; then
            legacy_resume=1
            printf 'Resuming pre-default Q3A4 evidence with exact legacy '
            printf 'configuration compatibility.\n'
        else
            die "resume configuration differs from the original run"
        fi
    fi
else
    [[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
        die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"/{runs,exact,summary,provenance}
    emit_configuration >"$OUTPUT_DIR/configuration.txt"
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"; partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1; printf 'error: could not create %s\n' "$archive" >&2
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
    "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$WARMUP_TOKENS"
    DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1
    DS4_CUDA_NO_MOE_Q32_DECODE_SPLIT=1
    DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG=1
    DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
    "${clean[@]}" ./tests/cuda_long_context_smoke \
        >"$OUTPUT_DIR/smoke.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/smoke.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
    grep -Fq 'SM75 Q3A4 hwarp16/tile32/dp4a gate/up' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 native exact marker missing"
    grep -Fq 'SM75 Q3A4 DP4A byte packing exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 DP4A packing marker missing"
    grep -Fq 'SM75 Q3A4 tile32-dp4a production default' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 production-default marker missing"
else
    make -q ds4-bench tests/cuda_long_context_smoke CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=manifest
if [[ $RESUME == 0 ]]; then
    {
        printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$(git branch --show-current)"
        printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\n' \
            "$MODEL" "$(stat -c %s "$MODEL")"
        printf 'candidate=q3a4-tile32-exact-dp4a\nq4_dispatch=unchanged\n'
        printf 'contexts=512,4096,32768\ntg_tokens=%s\nexact_tokens=%s\n' \
            "$TG_TOKENS" "$EXACT_TOKENS"
        printf '\n[gpu inventory]\n'
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
            --format=csv
        printf '\n[topology]\n'; nvidia-smi topo -m
    } >"$OUTPUT_DIR/manifest.txt"
    git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
    git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
    env | awk -F= '$1 ~ /^DS4_/ {print}' | sort -u \
        >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"
fi

validate_common_log() {
    local log=$1
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  'CUDA decode TP enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled' \
                  'CUDA decode indexer score row split enabled' \
                  'SM75 indexed attention selected: 8 heads / 256 threads' \
                  'CUDA prefill attention query-row split enabled'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    ! grep -Fq 'required but unavailable' "$log" &&
    ! grep -Fq 'SM75 Q32 owned decode CUDA Graph enabled' "$log" &&
    ! grep -Fq 'SM75 Q32 decode graph audit' "$log" &&
    awk '
        /routed-quant-audit/ {
            seen++
            layer=gate=up=down=""
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="layer") layer=a[2]
                if (a[1]=="gate") gate=a[2]
                if (a[1]=="up") up=a[2]
                if (a[1]=="down") down=a[2]
            }
            if (gate=="sm75_q3a4") {
                if (up!="sm75_q3a4" || down!="sm75_q4_32") bad=1
                q3++
                layers=layers (layers ? "," : "") layer
            }
        }
        END {
            expected="6,8,10,12,14,16,18,20,30,32,34,36,38,40,42"
            exit !(seen==43 && q3==15 && layers==expected && !bad)
        }
    ' "$log"
}

mapping_active_count() {
    local variant=$1 log=$2
    awk -v variant="$variant" '
        /SM75 Q3A4 decode mapping audit/ {
            seen++
            c=h=t=d=-1
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control") c=a[2]+0
                if (a[1]=="hwarp16") h=a[2]+0
                if (a[1]=="tile32") t=a[2]+0
                if (a[1]=="tile32-dp4a") d=a[2]+0
            }
            if (variant=="control") {
                good=(c>0 && h==0 && t==0 && d==0)
                active=c
            } else {
                good=(c==0 && h==0 && t==0 && d>0)
                active=d
            }
        }
        END {
            if (seen!=1 || !good) exit 1
            print active
        }
    ' "$log"
}

validate_mapping_audit() {
    local variant=$1 log=$2 marker
    if [[ $variant == control ]]; then
        marker='SM75 Q3A4 decode gate/up mapping=control'
    else
        marker='SM75 Q3A4 decode gate/up mapping=tile32-dp4a (production default)'
    fi
    if ! grep -Fq "$marker" "$log"; then
        [[ $legacy_resume == 1 ]] || return 1
    fi
    mapping_active_count "$variant" "$log" >/dev/null
}

validate_q8_plan_equal() {
    local control_log=$1 candidate_log=$2 marker control_line candidate_line
    for marker in 'CUDA q8 fp16 benefit plan candidates=' \
                  'CUDA q8 fp16 stage-aware 22/21 planner selected ' \
                  'CUDA q8 fp16 benefit plan materialized '; do
        [[ $(grep -Fc "$marker" "$control_log") == 1 &&
           $(grep -Fc "$marker" "$candidate_log") == 1 ]] || return 1
        control_line=$(grep -F "$marker" "$control_log") || return 1
        candidate_line=$(grep -F "$marker" "$candidate_log") || return 1
        [[ $control_line == "$candidate_line" ]] || {
            printf 'Q8 plan mismatch for %s\ncontrol: %s\ncandidate: %s\n' \
                "$marker" "$control_line" "$candidate_line" >&2
            return 1
        }
    done
}

validate_csv() {
    local csv=$1 expected_tg=$2
    awk -F, -v tg="$expected_tg" '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; ok1=($1==512 && $4==tg && ($8+0)>0); next}
        NR==3 {rows++; ok2=($1==4096 && $4==tg && ($8+0)>0); next}
        NR==4 {rows++; ok3=($1==32768 && $4==tg && ($8+0)>0); next}
        NR>4 {rows++}
        END {exit !(header && rows==3 && ok1 && ok2 && ok3)}
    ' "$csv"
}

validate_exact_partial_csv() {
    local csv=$1
    awk -F, -v tg="$EXACT_TOKENS" '
        NR==1 {
            header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                    $8=="gen_steady_tps")
            next
        }
        {
            rows++
            ctx=$1+0
            if ((ctx!=512 && ctx!=4096 && ctx!=32768) || seen[ctx]++ ||
                $4!=tg || ($8+0)<=0) bad=1
        }
        END {exit !(header && rows>=1 && rows<=3 && !bad)}
    ' "$csv"
}

csv_has_frontier() {
    local csv=$1 context=$2
    awk -F, -v ctx="$context" 'NR>1 && $1==ctx {found++}
        END {exit !(found==1)}' "$csv"
}

frontier_complete() {
    local logits=$1 context=$2 token file
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf -v file 'frontier_%06d.decode_%06d.logits.f32' \
            "$context" "$token"
        [[ -s $logits/$file ]] || return 1
    done
}

run_exact_full() {
    local variant=$1 base=$2 logits=$3
    local -a mapping_env=(DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING=1)
    [[ $variant == control ]] || mapping_env=()
    local ctx_alloc=$((CTX_MAX + EXACT_TOKENS + 1))
    printf 'Exact Q3A4 decode logits: %s (all frontiers)...\n' "$variant"
    "${production_env[@]}" "${mapping_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --ctx-alloc "$ctx_alloc" \
        --step-mul "$STEP_MUL" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$EXACT_TOKENS" --dump-decode-logits-dir "$logits" \
        --csv "$base.csv" >"$base.log" 2>&1
}

run_exact_frontier() {
    local variant=$1 context=$2 base=$3 logits=$4 prefix
    local -a mapping_env=(DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING=1)
    [[ $variant == control ]] || mapping_env=()
    local ctx_alloc=$((CTX_MAX + EXACT_TOKENS + 1))
    printf -v prefix 'frontier_%06d' "$context"
    find "$logits" -maxdepth 1 -type f \
        -name "$prefix.decode_*.logits.f32" -delete
    printf 'Repairing exact Q3A4 decode logits: %s PP=%s...\n' \
        "$variant" "$context"
    "${production_env[@]}" "${mapping_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$context" --ctx-max "$context" --ctx-alloc "$ctx_alloc" \
        --step-mul "$STEP_MUL" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$EXACT_TOKENS" --dump-decode-logits-dir "$logits" \
        --csv "$base.csv" >"$base.log" 2>&1
}

exact_source_calls() {
    local variant=$1 sources=$2 frontiers kind log csv calls total=0
    local expected context
    while IFS=$'\t' read -r frontiers kind log csv; do
        [[ $frontiers != frontiers ]] || continue
        validate_common_log "$log" && validate_mapping_audit "$variant" "$log" &&
            validate_exact_partial_csv "$csv" || return 1
        IFS=, read -r -a source_contexts <<<"$frontiers"
        [[ ${#source_contexts[@]} -ge 1 ]] || return 1
        for context in "${source_contexts[@]}"; do
            csv_has_frontier "$csv" "$context" || return 1
        done
        [[ $(awk -F, 'NR>1 {n++} END {print n+0}' "$csv") == \
           ${#source_contexts[@]} ]] || return 1
        calls=$(mapping_active_count "$variant" "$log") || return 1
        expected=$((${#source_contexts[@]} * EXACT_TOKENS * 15 * 2))
        if [[ $kind == repair ]]; then
            (( calls == expected )) || return 1
        else
            # A failed all-frontier process may have entered the next decode
            # frontier after completing every file attributed to it. Preserve
            # those complete frontiers and record the partial extra calls.
            (( calls >= expected )) || return 1
        fi
        total=$((total + calls))
    done <"$sources"
    printf '%s\n' "$total"
}

mapping_count_is_exact() {
    local variant=$1 log=$2 frontier_count=$3 calls
    calls=$(mapping_active_count "$variant" "$log") || return 1
    (( calls == frontier_count * EXACT_TOKENS * 15 * 2 ))
}

mapping_count_covers() {
    local variant=$1 log=$2 frontier_count=$3 calls
    calls=$(mapping_active_count "$variant" "$log") || return 1
    (( calls >= frontier_count * EXACT_TOKENS * 15 * 2 ))
}

source_kind_for_frontier() {
    local sources=$1 context=$2
    awk -F'\t' -v target="$context" '
        NR==1 {next}
        {
            n=split($1, values, ",")
            for (i=1; i<=n; i++) {
                if (values[i]==target) {
                    found++
                    kind=$2
                }
            }
        }
        END {
            if (found!=1) exit 1
            print kind
        }
    ' "$sources"
}

validate_source_manifest() {
    awk -F'\t' '
        NR==1 {
            header=($1=="frontiers" && $2=="kind" &&
                    $3=="log" && $4=="csv")
            next
        }
        {
            rows++
            if (($2!="base" && $2!="repair") || !$3 || !$4) bad=1
            n=split($1, values, ",")
            for (i=1; i<=n; i++) {
                ctx=values[i]+0
                if (ctx!=512 && ctx!=4096 && ctx!=32768) bad=1
                seen[ctx]++
            }
        }
        END {
            exit !(header && rows>=1 && !bad &&
                   seen[512]==1 && seen[4096]==1 && seen[32768]==1)
        }
    ' "$1"
}

write_selected_csv() {
    local input=$1 frontiers=$2 output=$3 partial
    partial="$output.partial.$$"
    awk -F, -v selected="$frontiers" '
        BEGIN {
            OFS=",";
            n=split(selected, values, ",")
            for (i=1; i<=n; i++) wanted[values[i]]=1
        }
        NR==1 {print; next}
        wanted[$1] {print}
    ' "$input" >"$partial"
    mv -- "$partial" "$output"
}

run_one() {
    local variant=$1 tokens=$2 csv=$3 log=$4
    local -a mapping_env=(DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING=1)
    # The candidate deliberately carries no mapping override: this A/B now
    # proves the production default rather than an opt-in experiment.
    [[ $variant == control ]] || mapping_env=()
    local ctx_alloc=$((CTX_MAX + tokens + 1))
    "${production_env[@]}" "${mapping_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --ctx-alloc "$ctx_alloc" \
        --step-mul "$STEP_MUL" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$tokens" --csv "$csv" >"$log" 2>&1 || return 1
    validate_csv "$csv" "$tokens" && validate_common_log "$log" &&
        validate_mapping_audit "$variant" "$log"
}

write_throughput_summary() {
    python3 - "$OUTPUT_DIR/runs/runs.tsv" "$OUTPUT_DIR/summary/summary.csv" \
               "$OUTPUT_DIR/summary/summary.md" <<'PY'
import csv
import pathlib
import statistics
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
by_pair = {}
for row in rows:
    key = (int(row["repeat"]), int(row["context"]))
    by_pair.setdefault(key, {})[row["variant"]] = float(row["steady_tps"])

groups = {}
for (repeat, context), values in by_pair.items():
    if set(values) != {"control", "tile32-dp4a"}:
        raise SystemExit(f"unpaired sample at repeat={repeat} context={context}")
    groups.setdefault(context, []).append(
        (values["control"], values["tile32-dp4a"]))

records = []
for context in sorted(groups):
    pairs = groups[context]
    control = [p[0] for p in pairs]
    candidate = [p[1] for p in pairs]
    ratios = [b / a for a, b in pairs]
    records.append((
        context,
        statistics.median(control),
        statistics.median(candidate),
        statistics.median(ratios),
        statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        len(ratios),
    ))

with pathlib.Path(sys.argv[2]).open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["context", "control_tps", "tile32_dp4a_tps",
                  "paired_median_speedup", "change_pct",
                  "paired_speedup_sd", "samples"])
    for context, control, candidate, speedup, sd, samples in records:
        out.writerow([context, f"{control:.6f}", f"{candidate:.6f}",
                      f"{speedup:.9f}", f"{(speedup - 1) * 100:.6f}",
                      f"{sd:.9f}", samples])

lines = [
    "# SM75 production Q3A4 tile32-DP4A decode A/B",
    "",
    "Only Q3A4 gate/up mapping changes. Q4-32 and every cross-GPU boundary "
    "are identical. Exact-output validation is a separate checkpointed phase.",
    "",
    "| Context | Control tok/s | Tile32-DP4A tok/s | Paired speedup | Change | SD |",
    "| ---: | ---: | ---: | ---: | ---: | ---: |",
]
for context, control, candidate, speedup, sd, _ in records:
    lines.append(
        f"| {context} | {control:.3f} | {candidate:.3f} | {speedup:.6f}x | "
        f"{(speedup - 1) * 100:+.3f}% | {sd:.6f} |"
    )
pathlib.Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
PY
}

phase=throughput
runs_partial="$OUTPUT_DIR/runs/runs.tsv.partial.$$"
dispatch_partial="$OUTPUT_DIR/runs/dispatch.tsv.partial.$$"
printf 'repeat\tslot\tcontext\tvariant\tsteady_tps\tcsv\tlog\n' >"$runs_partial"
printf 'repeat\tcontrol_owned_calls\ttile32_dp4a_owned_calls\tq8_plan_equal\n' \
    >"$dispatch_partial"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then variants=(control tile32-dp4a)
    else variants=(tile32-dp4a control)
    fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1)); base="$OUTPUT_DIR/runs/r${repeat}-${variant}"
        if [[ $RESUME == 1 && -s $base.csv && -s $base.log ]] &&
           validate_csv "$base.csv" "$TG_TOKENS" &&
           validate_common_log "$base.log" &&
           validate_mapping_audit "$variant" "$base.log"; then
            printf 'Reusing Q3A4 production A/B repeat=%d/%d variant=%s...\n' \
                "$repeat" "$REPEATS" "$variant"
        else
            printf 'Q3A4 production A/B repeat=%d/%d slot=%d/2 variant=%s...\n' \
                "$repeat" "$REPEATS" "$slot" "$variant"
            run_one "$variant" "$TG_TOKENS" "$base.csv" "$base.log" || {
                tail -n 200 "$base.log" >&2 || true
                die "$variant production run failed validation"
            }
        fi
        while IFS=, read -r context _ _ _ _ _ _ steady _; do
            [[ $context == ctx_tokens ]] && continue
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$repeat" "$slot" "$context" "$variant" "$steady" \
                "$base.csv" "$base.log" >>"$runs_partial"
        done <"$base.csv"
    done
    control_log="$OUTPUT_DIR/runs/r${repeat}-control.log"
    candidate_log="$OUTPUT_DIR/runs/r${repeat}-tile32-dp4a.log"
    validate_q8_plan_equal "$control_log" "$candidate_log" ||
        die "repeat $repeat changed the dense-Q8 placement plan between arms"
    control_calls=$(mapping_active_count control "$control_log")
    candidate_calls=$(mapping_active_count tile32-dp4a "$candidate_log")
    [[ $control_calls == "$candidate_calls" ]] ||
        die "repeat $repeat changed Q3A4 owned-call inventory ($control_calls vs $candidate_calls)"
    printf '%s\t%s\t%s\ttrue\n' \
        "$repeat" "$control_calls" "$candidate_calls" >>"$dispatch_partial"
done
mv -- "$runs_partial" "$OUTPUT_DIR/runs/runs.tsv"
mv -- "$dispatch_partial" "$OUTPUT_DIR/runs/dispatch.tsv"
phase=throughput-summary
write_throughput_summary
cat "$OUTPUT_DIR/summary/summary.md"

phase=exact-logits
for variant in control tile32-dp4a; do
    base="$OUTPUT_DIR/exact/$variant"; logits="$base-logits"
    sources="$OUTPUT_DIR/exact/$variant-sources.tsv"
    sources_partial="$sources.partial.$$"
    printf 'frontiers\tkind\tlog\tcsv\n' >"$sources_partial"
    valid=0
    if [[ $RESUME == 1 && -s $base.csv && -s $base.log && -d $logits ]] &&
       validate_csv "$base.csv" "$EXACT_TOKENS" &&
       validate_common_log "$base.log" &&
       validate_mapping_audit "$variant" "$base.log" &&
       frontier_complete "$logits" 512 &&
       frontier_complete "$logits" 4096 &&
       frontier_complete "$logits" 32768; then
        valid=1
        printf '512,4096,32768\tbase\t%s\t%s\n' \
            "$base.log" "$base.csv" >>"$sources_partial"
        printf 'Reusing exact Q3A4 decode logits: %s (all frontiers)...\n' \
            "$variant"
    fi
    if [[ $valid == 0 ]]; then
        mkdir -p "$logits"
        if [[ $RESUME == 0 ]]; then
            find "$logits" -maxdepth 1 -type f -name '*.f32' -delete
            run_exact_full "$variant" "$base" "$logits" ||
                die "$variant exact-logit run failed"
            validate_csv "$base.csv" "$EXACT_TOKENS" &&
                validate_common_log "$base.log" &&
                validate_mapping_audit "$variant" "$base.log" ||
                die "$variant exact run omitted its production or mapping path"
            printf '512,4096,32768\tbase\t%s\t%s\n' \
                "$base.log" "$base.csv" >>"$sources_partial"
        else
            base_reusable=0
            base_frontiers=
            base_frontier_count=0
            if [[ -s $base.csv && -s $base.log ]] &&
               validate_exact_partial_csv "$base.csv" &&
               validate_common_log "$base.log" &&
               validate_mapping_audit "$variant" "$base.log"; then
                base_reusable=1
                for context in 512 4096 32768; do
                    if csv_has_frontier "$base.csv" "$context" &&
                       frontier_complete "$logits" "$context"; then
                        base_frontiers+="${base_frontiers:+,}$context"
                        base_frontier_count=$((base_frontier_count + 1))
                    fi
                done
                if [[ $base_frontier_count == 0 ]] ||
                   ! mapping_count_covers "$variant" "$base.log" \
                       "$base_frontier_count"; then
                    base_reusable=0
                    base_frontiers=
                    base_frontier_count=0
                fi
            fi
            reusable_frontiers=0
            for context in 512 4096 32768; do
                repair="$OUTPUT_DIR/exact/$variant-frontier-$context"
                if [[ $base_reusable == 1 ]] &&
                   csv_has_frontier "$base.csv" "$context" &&
                   frontier_complete "$logits" "$context"; then
                    reusable_frontiers=$((reusable_frontiers + 1))
                elif [[ -s $repair.csv && -s $repair.log ]] &&
                     validate_exact_partial_csv "$repair.csv" &&
                     csv_has_frontier "$repair.csv" "$context" &&
                     validate_common_log "$repair.log" &&
                     validate_mapping_audit "$variant" "$repair.log" &&
                     mapping_count_is_exact "$variant" "$repair.log" 1 &&
                     frontier_complete "$logits" "$context"; then
                    reusable_frontiers=$((reusable_frontiers + 1))
                fi
            done
            if [[ $reusable_frontiers == 0 ]]; then
                find "$logits" -maxdepth 1 -type f -name '*.f32' -delete
                run_exact_full "$variant" "$base" "$logits" ||
                    die "$variant exact-logit run failed"
                validate_csv "$base.csv" "$EXACT_TOKENS" &&
                    validate_common_log "$base.log" &&
                    validate_mapping_audit "$variant" "$base.log" ||
                    die "$variant exact run omitted its production or mapping path"
                printf '512,4096,32768\tbase\t%s\t%s\n' \
                    "$base.log" "$base.csv" >>"$sources_partial"
            else
                reused_base=0
                for context in 512 4096 32768; do
                    repair="$OUTPUT_DIR/exact/$variant-frontier-$context"
                    if [[ $base_reusable == 1 ]] &&
                       csv_has_frontier "$base.csv" "$context" &&
                       frontier_complete "$logits" "$context"; then
                        reused_base=1
                        printf 'Reusing exact Q3A4 decode logits: %s PP=%s...\n' \
                            "$variant" "$context"
                        continue
                    fi
                    if [[ -s $repair.csv && -s $repair.log ]] &&
                       validate_exact_partial_csv "$repair.csv" &&
                       csv_has_frontier "$repair.csv" "$context" &&
                       validate_common_log "$repair.log" &&
                       validate_mapping_audit "$variant" "$repair.log" &&
                       mapping_count_is_exact "$variant" "$repair.log" 1 &&
                       frontier_complete "$logits" "$context"; then
                        printf 'Reusing repaired exact Q3A4 decode logits: %s PP=%s...\n' \
                            "$variant" "$context"
                    else
                        run_exact_frontier "$variant" "$context" "$repair" "$logits" ||
                            die "$variant PP=$context exact-logit repair failed"
                        validate_exact_partial_csv "$repair.csv" &&
                            csv_has_frontier "$repair.csv" "$context" &&
                            validate_common_log "$repair.log" &&
                            validate_mapping_audit "$variant" "$repair.log" &&
                            mapping_count_is_exact "$variant" "$repair.log" 1 &&
                            frontier_complete "$logits" "$context" ||
                            die "$variant PP=$context exact-logit repair is invalid"
                    fi
                    printf '%s\trepair\t%s\t%s\n' \
                        "$context" "$repair.log" "$repair.csv" \
                        >>"$sources_partial"
                done
                if [[ $reused_base == 1 ]]; then
                    selected_csv="$OUTPUT_DIR/exact/$variant-base-selected.csv"
                    write_selected_csv "$base.csv" "$base_frontiers" \
                        "$selected_csv"
                    printf '%s\tbase\t%s\t%s\n' \
                        "$base_frontiers" "$base.log" "$selected_csv" \
                        >>"$sources_partial"
                fi
            fi
        fi
    fi
    for context in 512 4096 32768; do
        frontier_complete "$logits" "$context" ||
            die "$variant did not emit all PP=$context decode-logit files"
    done
    [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == \
       $((3 * EXACT_TOKENS)) ]] ||
        die "$variant emitted an unexpected decode-logit inventory"
    mv -- "$sources_partial" "$sources"
done

control_sources="$OUTPUT_DIR/exact/control-sources.tsv"
candidate_sources="$OUTPUT_DIR/exact/tile32-dp4a-sources.tsv"
validate_source_manifest "$control_sources" &&
    validate_source_manifest "$candidate_sources" ||
    die "exact-source manifest does not assign every frontier exactly once"
control_ref=$(awk -F'\t' 'NR==2 {print $3}' "$control_sources")
[[ -n $control_ref ]] || die "control exact-source manifest is empty"

# An interrupted progressive exact arm can leave one frontier repaired by a
# direct single-frontier process.  Comparing that direct candidate with the
# progressive control would compare different prefill histories, not Q3A4
# arithmetic.  Preserve both canonical inventories and add only the missing
# direct control needed for a matched PP32768 comparison.
control_32768_kind=$(source_kind_for_frontier "$control_sources" 32768)
candidate_32768_kind=$(source_kind_for_frontier "$candidate_sources" 32768)
matched_direct_control=0
matched_control_calls=0
history_mismatch_count=0
history_bit_exact=not-applicable
matched_comparison_mode=canonical-control-vs-canonical-candidate
matched_control_base="$OUTPUT_DIR/exact/control-frontier-32768-matched"
matched_control_logits="$matched_control_base-logits"
if [[ $control_32768_kind != "$candidate_32768_kind" ]]; then
    [[ $control_32768_kind == base && $candidate_32768_kind == repair ]] ||
        die "unsupported PP32768 exact-history orientation: control=$control_32768_kind candidate=$candidate_32768_kind"
    mkdir -p "$matched_control_logits"
    if [[ $RESUME == 1 && -s $matched_control_base.csv &&
          -s $matched_control_base.log ]] &&
       validate_exact_partial_csv "$matched_control_base.csv" &&
       csv_has_frontier "$matched_control_base.csv" 32768 &&
       validate_common_log "$matched_control_base.log" &&
       validate_mapping_audit control "$matched_control_base.log" &&
       mapping_count_is_exact control "$matched_control_base.log" 1 &&
       frontier_complete "$matched_control_logits" 32768 &&
       [[ $(find "$matched_control_logits" -maxdepth 1 -type f \
                 -name '*.f32' | wc -l) == "$EXACT_TOKENS" ]]; then
        printf 'Reusing matched direct Q3A4 control logits: PP=32768...\n'
    else
        printf 'Capturing matched direct Q3A4 control logits: PP=32768...\n'
        run_exact_frontier control 32768 "$matched_control_base" \
            "$matched_control_logits" ||
            die "matched direct control PP=32768 exact-logit run failed"
        validate_exact_partial_csv "$matched_control_base.csv" &&
            csv_has_frontier "$matched_control_base.csv" 32768 &&
            validate_common_log "$matched_control_base.log" &&
            validate_mapping_audit control "$matched_control_base.log" &&
            mapping_count_is_exact control "$matched_control_base.log" 1 &&
            frontier_complete "$matched_control_logits" 32768 &&
            [[ $(find "$matched_control_logits" -maxdepth 1 -type f \
                      -name '*.f32' | wc -l) == "$EXACT_TOKENS" ]] ||
            die "matched direct control PP=32768 evidence is invalid"
    fi
    validate_q8_plan_equal "$control_ref" "$matched_control_base.log" ||
        die "matched direct control changed the dense-Q8 placement plan"
    matched_control_calls=$(mapping_active_count control \
        "$matched_control_base.log")
    matched_direct_control=1
    matched_comparison_mode=direct-control-vs-direct-candidate
fi
for sources in "$control_sources" "$candidate_sources"; do
    while IFS=$'\t' read -r frontiers kind log csv; do
        [[ $frontiers != frontiers ]] || continue
        validate_q8_plan_equal "$control_ref" "$log" ||
            die "exact source changed the dense-Q8 placement plan: $log"
    done <"$sources"
done
control_calls=$(exact_source_calls control "$control_sources") ||
    die "control exact sources failed mapping validation"
candidate_calls=$(exact_source_calls tile32-dp4a "$candidate_sources") ||
    die "tile32-dp4a exact sources failed mapping validation"
minimum_calls=$((3 * EXACT_TOKENS * 15 * 2))
(( control_calls >= minimum_calls && candidate_calls >= minimum_calls )) ||
    die "exact sources omitted Q3A4 calls (minimum $minimum_calls; control $control_calls; candidate $candidate_calls)"
control_extra_calls=$((control_calls - minimum_calls))
candidate_extra_calls=$((candidate_calls - minimum_calls))

expected_files="$OUTPUT_DIR/exact/expected-files.txt"
: >"$expected_files"
for context in 512 4096 32768; do
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf 'frontier_%06d.decode_%06d.logits.f32\n' \
            "$context" "$token" >>"$expected_files"
    done
done
find "$OUTPUT_DIR/exact/control-logits" -maxdepth 1 -type f -name '*.f32' \
    -printf '%f\n' | sort >"$OUTPUT_DIR/exact/control-files.txt"
find "$OUTPUT_DIR/exact/tile32-dp4a-logits" -maxdepth 1 -type f -name '*.f32' \
    -printf '%f\n' | sort >"$OUTPUT_DIR/exact/candidate-files.txt"
cmp -s "$expected_files" "$OUTPUT_DIR/exact/control-files.txt" ||
    die "control emitted an unexpected logit inventory"
cmp -s "$expected_files" "$OUTPUT_DIR/exact/candidate-files.txt" ||
    die "tile32-dp4a emitted an unexpected logit inventory"
if [[ $matched_direct_control == 1 ]]; then
    find "$matched_control_logits" -maxdepth 1 -type f -name '*.f32' \
        -printf '%f\n' | sort >"$OUTPUT_DIR/exact/matched-control-files.txt"
    grep '^frontier_032768\.' "$expected_files" \
        >"$OUTPUT_DIR/exact/expected-files-32768.txt"
    cmp -s "$OUTPUT_DIR/exact/expected-files-32768.txt" \
        "$OUTPUT_DIR/exact/matched-control-files.txt" ||
        die "matched direct control emitted an unexpected logit inventory"
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf -v file 'frontier_032768.decode_%06d.logits.f32' "$token"
        if ! cmp -s "$OUTPUT_DIR/exact/control-logits/$file" \
                  "$matched_control_logits/$file"; then
            history_mismatch_count=$((history_mismatch_count + 1))
        fi
    done
    if (( history_mismatch_count == 0 )); then
        history_bit_exact=true
    else
        history_bit_exact=false
    fi
    printf 'comparison=progressive-control-vs-direct-control\n' \
        >"$OUTPUT_DIR/exact/prefill-history-comparison.txt"
    printf 'frontier=32768\nbit_exact=%s\nmismatched_files=%s\ntotal_files=%s\n' \
        "$history_bit_exact" "$history_mismatch_count" "$EXACT_TOKENS" \
        >>"$OUTPUT_DIR/exact/prefill-history-comparison.txt"
fi
while IFS= read -r file; do
    padded_context=${file#frontier_}
    padded_context=${padded_context%%.*}
    context=$((10#$padded_context))
    control_kind=$(source_kind_for_frontier "$control_sources" "$context")
    candidate_kind=$(source_kind_for_frontier "$candidate_sources" "$context")
    control_file="$OUTPUT_DIR/exact/control-logits/$file"
    if [[ $context == 32768 && $matched_direct_control == 1 ]]; then
        control_file="$matched_control_logits/$file"
    elif [[ $control_kind != "$candidate_kind" ]]; then
        die "exact paths used different prefill histories at PP=$context ($control_kind vs $candidate_kind); no matched comparison was captured"
    fi
    if ! cmp -s "$control_file" \
              "$OUTPUT_DIR/exact/tile32-dp4a-logits/$file"; then
        die "tile32-dp4a diverged at $file"
    fi
done <"$OUTPUT_DIR/exact/control-files.txt"
printf 'bit_exact=true\nfrontiers=512,4096,32768\ndecode_tokens_per_frontier=%s\n' \
    "$EXACT_TOKENS" >"$OUTPUT_DIR/exact/verification.txt"
printf 'minimum_required_owned_calls=%s\ncontrol_observed_owned_calls=%s\n' \
    "$minimum_calls" "$control_calls" >>"$OUTPUT_DIR/exact/verification.txt"
printf 'tile32_dp4a_observed_owned_calls=%s\nq8_plan_equal=true\n' \
    "$candidate_calls" >>"$OUTPUT_DIR/exact/verification.txt"
printf 'control_interrupted_extra_calls=%s\n' "$control_extra_calls" \
    >>"$OUTPUT_DIR/exact/verification.txt"
printf 'tile32_dp4a_interrupted_extra_calls=%s\n' "$candidate_extra_calls" \
    >>"$OUTPUT_DIR/exact/verification.txt"
printf 'pp32768_control_source=%s\npp32768_candidate_source=%s\n' \
    "$control_32768_kind" "$candidate_32768_kind" \
    >>"$OUTPUT_DIR/exact/verification.txt"
printf 'pp32768_matched_comparison=%s\nmatched_direct_control_owned_calls=%s\n' \
    "$matched_comparison_mode" \
    "$matched_control_calls" >>"$OUTPUT_DIR/exact/verification.txt"
printf 'progressive_vs_direct_control_bit_exact=%s\n' "$history_bit_exact" \
    >>"$OUTPUT_DIR/exact/verification.txt"
printf 'progressive_vs_direct_control_mismatched_files=%s\n' \
    "$history_mismatch_count" >>"$OUTPUT_DIR/exact/verification.txt"
cat >>"$OUTPUT_DIR/summary/summary.md" <<EOF

## Exact-output verification

All $EXACT_TOKENS decode logits at PP512, PP4096, and PP32768 were byte-identical
under matched prefill histories. PP32768 comparison mode:
`$matched_comparison_mode`.
EOF

phase=complete
