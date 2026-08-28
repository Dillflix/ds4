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
CONTEXTS=${CONTEXTS:-512,2048,4096,32768}
TG_TOKENS=${TG_TOKENS:-256}
REPEATS=${REPEATS:-3}
EXACT_CONTEXT=${EXACT_CONTEXT:-4096}
EXACT_TOKENS=${EXACT_TOKENS:-16}
WARMUP_TOKENS=${WARMUP_TOKENS:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${DECODE_GRAPH_AB_DIR:-$repo_dir/sm75-decode-graph-ab-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged Q4-32/Q3A4 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "TG_TOKENS:$TG_TOKENS" "REPEATS:$REPEATS" \
            "EXACT_CONTEXT:$EXACT_CONTEXT" "EXACT_TOKENS:$EXACT_TOKENS" \
            "WARMUP_TOKENS:$WARMUP_TOKENS" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "PIPELINE_MB:$PIPELINE_MB"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ && $value -ge 1 ]] ||
        die "$name must be a positive integer"
done
(( REPEATS >= 2 && WARMUP_TOKENS == 512 && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 )) ||
    die "require repeats>=2, warmup_tokens=512, prefill_chunk=2048, pipeline_mb=512"
for flag in SKIP_BUILD RESUME CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"

IFS=, read -r -a contexts <<<"$CONTEXTS"
(( ${#contexts[@]} >= 1 )) || die "CONTEXTS selected no frontiers"
declare -A seen_context=()
for pp in "${contexts[@]}"; do
    [[ $pp =~ ^[0-9]+$ && $pp -ge 512 && -z ${seen_context[$pp]+x} ]] ||
        die "CONTEXTS contains an invalid or duplicate frontier: $pp"
    seen_context[$pp]=1
done
[[ -n ${seen_context[$EXACT_CONTEXT]+x} ]] ||
    die "EXACT_CONTEXT must also appear in CONTEXTS"

for tool in awk basename cat cmp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"
for gpu in "${gpu_ids[@]}"; do
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] ||
        die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done

emit_configuration() {
    printf 'model=%s\nprompt=%s\ngpu_devices=%s\ngpu_vram=%s\n' \
        "$MODEL" "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM"
    printf 'stage_split=%s\ncontexts=%s\ntg_tokens=%s\nrepeats=%s\n' \
        "$STAGE_SPLIT" "$CONTEXTS" "$TG_TOKENS" "$REPEATS"
    printf 'exact_context=%s\nexact_tokens=%s\nwarmup_tokens=%s\n' \
        "$EXACT_CONTEXT" "$EXACT_TOKENS" "$WARMUP_TOKENS"
    printf 'prefill_chunk=%s\npipeline_mb=%s\n' "$PREFILL_CHUNK" "$PIPELINE_MB"
}

if [[ $RESUME == 1 ]]; then
    [[ -n ${DECODE_GRAPH_AB_DIR:-} && -d $OUTPUT_DIR ]] ||
        die "RESUME=1 requires an existing DECODE_GRAPH_AB_DIR"
    cmp -s <(emit_configuration) "$OUTPUT_DIR/configuration.txt" ||
        die "resume configuration differs from the original run"
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
    DS4_CUDA_TP_PREFILL_ATTN_ROWS=1
    DS4_CUDA_TP_ATTN_HEADS=0
    DS4_CUDA_TP_DECODE_INDEXER_ROWS=0
    "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$WARMUP_TOKENS"
)

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
    ./tests/cuda_long_context_smoke 2>&1 | tee "$OUTPUT_DIR/smoke.log"
    grep -Fq 'SM75 Q4-32 owned decode CUDA Graph exact/reuse' \
        "$OUTPUT_DIR/smoke.log" || die "Q4-32 graph smoke marker missing"
    grep -Fq 'SM75 Q3A4 owned decode CUDA Graph exact/reuse' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 graph smoke marker missing"
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
        printf 'candidate=sm75-q32-owned-six-node-cuda-graph\n'
        printf 'graph_scope=owned-moe-only\ncombine_boundary=unchanged\n'
        printf '\n[gpu inventory]\n'
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
            --format=csv
        printf '\n[topology]\n'; nvidia-smi topo -m
    } >"$OUTPUT_DIR/manifest.txt"
    git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
    git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
fi

validate_log() {
    local variant=$1 log=$2
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    if [[ $variant == graph ]]; then
        grep -Fq 'SM75 Q32 owned decode CUDA Graph enabled' "$log" || return 1
        awk '
            /SM75 Q32 decode graph audit/ {
                calls=launches=fallbacks=-1
                for (i=1; i<=NF; i++) {
                    split($i, a, "=")
                    if (a[1]=="calls") calls=a[2]+0
                    if (a[1]=="launches") launches=a[2]+0
                    if (a[1]=="fallbacks") fallbacks=a[2]+0
                }
                good=(calls>0 && calls==launches && fallbacks==0)
            }
            END {exit !good}
        ' "$log" || return 1
    else
        ! grep -Fq 'SM75 Q32 owned decode CUDA Graph enabled' "$log" || return 1
        ! grep -Fq 'SM75 Q32 decode graph audit' "$log" || return 1
    fi
    ! grep -Fq 'required but unavailable' "$log"
}

validate_csv() {
    local csv=$1 pp=$2
    awk -F, -v pp="$pp" -v tg="$TG_TOKENS" '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; good=($1==pp && $4==tg && ($8+0)>0); next}
        NR>2 {rows++}
        END {exit !(header && rows==1 && good)}
    ' "$csv"
}

run_one() {
    local variant=$1 pp=$2 csv=$3 log=$4
    local -a graph_env=(DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1)
    [[ $variant == control ]] || graph_env=(
        DS4_CUDA_MOE_Q32_DECODE_GRAPH=1
        DS4_CUDA_MOE_Q32_DECODE_GRAPH_AUDIT=1
    )
    local ctx_alloc=$((pp + TG_TOKENS + 1))
    "${production_env[@]}" "${graph_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$pp" --ctx-max "$pp" --ctx-alloc "$ctx_alloc" \
        --step-incr "$pp" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$TG_TOKENS" --csv "$csv" >"$log" 2>&1 || return 1
    validate_csv "$csv" "$pp" && validate_log "$variant" "$log"
}

phase=throughput
runs_partial="$OUTPUT_DIR/runs/runs.tsv.partial.$$"
printf 'repeat\tslot\tcontext\tvariant\tsteady_tps\tcsv\tlog\n' >"$runs_partial"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then
        pp_order=("${contexts[@]}"); variants=(control graph)
    else
        pp_order=(); for ((i=${#contexts[@]}-1; i>=0; i--)); do
            pp_order+=("${contexts[$i]}")
        done
        variants=(graph control)
    fi
    slot=0
    for pp in "${pp_order[@]}"; do
        for variant in "${variants[@]}"; do
            slot=$((slot + 1)); base="$OUTPUT_DIR/runs/r${repeat}-pp${pp}-${variant}"
            if [[ $RESUME == 1 && -s $base.csv && -s $base.log ]] &&
               validate_csv "$base.csv" "$pp" && validate_log "$variant" "$base.log"; then
                printf 'Reusing decode graph A/B repeat=%d/%d context=%s variant=%s...\n' \
                    "$repeat" "$REPEATS" "$pp" "$variant"
            else
                printf 'Decode graph A/B repeat=%d/%d slot=%d/%d context=%s variant=%s...\n' \
                    "$repeat" "$REPEATS" "$slot" "$((2*${#contexts[@]}))" \
                    "$pp" "$variant"
                run_one "$variant" "$pp" "$base.csv" "$base.log" || {
                    tail -n 200 "$base.log" >&2 || true
                    die "$variant context $pp production run failed validation"
                }
            fi
            steady=$(awk -F, 'NR==2 {print $8}' "$base.csv")
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$repeat" "$slot" "$pp" "$variant" "$steady" \
                "$base.csv" "$base.log" >>"$runs_partial"
        done
    done
done
mv -- "$runs_partial" "$OUTPUT_DIR/runs/runs.tsv"

phase=exact-logits
for variant in control graph; do
    graph_env=(DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1)
    [[ $variant == control ]] || graph_env=(
        DS4_CUDA_MOE_Q32_DECODE_GRAPH=1
        DS4_CUDA_MOE_Q32_DECODE_GRAPH_AUDIT=1
    )
    base="$OUTPUT_DIR/exact/$variant"; logits="$base-logits"
    if [[ $RESUME == 1 && -s $base.csv && -s $base.log && -d $logits ]] &&
       validate_log "$variant" "$base.log" &&
       [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == "$EXACT_TOKENS" ]]; then
        printf 'Reusing exact decode logits: %s...\n' "$variant"; continue
    fi
    [[ ! -d $logits ]] || find "$logits" -maxdepth 1 -type f -name '*.f32' -delete
    mkdir -p "$logits"; ctx_alloc=$((EXACT_CONTEXT + EXACT_TOKENS + 1))
    printf 'Exact decode logits: %s...\n' "$variant"
    "${production_env[@]}" "${graph_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$EXACT_CONTEXT" --ctx-max "$EXACT_CONTEXT" \
        --ctx-alloc "$ctx_alloc" --step-incr "$EXACT_CONTEXT" \
        --prefill-chunk "$PREFILL_CHUNK" --gen-tokens "$EXACT_TOKENS" \
        --dump-decode-logits-dir "$logits" --csv "$base.csv" \
        >"$base.log" 2>&1 || die "$variant exact-logit run failed"
    validate_log "$variant" "$base.log" || die "$variant exact log omitted its path"
    [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == "$EXACT_TOKENS" ]] ||
        die "$variant did not emit $EXACT_TOKENS decode-logit files"
done
find "$OUTPUT_DIR/exact/control-logits" -maxdepth 1 -type f -name '*.f32' \
    -printf '%f\n' | sort >"$OUTPUT_DIR/exact/control-files.txt"
find "$OUTPUT_DIR/exact/graph-logits" -maxdepth 1 -type f -name '*.f32' \
    -printf '%f\n' | sort >"$OUTPUT_DIR/exact/graph-files.txt"
cmp -s "$OUTPUT_DIR/exact/control-files.txt" "$OUTPUT_DIR/exact/graph-files.txt" ||
    die "control and graph emitted different logit inventories"
while IFS= read -r file; do
    cmp -s "$OUTPUT_DIR/exact/control-logits/$file" \
           "$OUTPUT_DIR/exact/graph-logits/$file" ||
        die "graph diverged at $file"
done <"$OUTPUT_DIR/exact/control-files.txt"
printf 'bit_exact=true\ndecode_tokens=%s\n' "$EXACT_TOKENS" \
    >"$OUTPUT_DIR/exact/verification.txt"

phase=summary
python3 - "$OUTPUT_DIR/runs/runs.tsv" "$OUTPUT_DIR/summary/summary.csv" \
           "$OUTPUT_DIR/summary/summary.md" <<'PY'
import csv, pathlib, statistics, sys
rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
groups = {}
for row in rows:
    groups.setdefault(int(row["context"]), {}).setdefault(
        row["variant"], []).append(float(row["steady_tps"]))
records = []
for context in sorted(groups):
    control = groups[context]["control"]
    graph = groups[context]["graph"]
    if len(control) != len(graph): raise SystemExit("unpaired samples")
    cmed, gmed = statistics.median(control), statistics.median(graph)
    paired = [g / c for c, g in zip(control, graph)]
    speedup = statistics.median(paired)
    sd = statistics.stdev(paired) if len(paired) > 1 else 0.0
    records.append((context, cmed, gmed, speedup, sd, len(paired)))
with pathlib.Path(sys.argv[2]).open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["context", "control_tps", "graph_tps",
                  "paired_median_speedup", "change_pct",
                  "paired_speedup_sd", "samples"])
    for ctx, c, g, speed, sd, n in records:
        out.writerow([ctx, f"{c:.6f}", f"{g:.6f}", f"{speed:.9f}",
                      f"{(speed-1)*100:.6f}", f"{sd:.9f}", n])
lines = ["# SM75 production Q32 decode CUDA Graph A/B", "",
         "The candidate replaces each owned expert half's six production "
         "kernel submissions with one CUDA Graph launch. The cross-GPU "
         "combine boundary is unchanged. Decode logits were byte-identical.", "",
         "| Context | Control tok/s | Graph tok/s | Paired speedup | Change | SD |",
         "| ---: | ---: | ---: | ---: | ---: | ---: |"]
for ctx, c, g, speed, sd, _ in records:
    lines.append(f"| {ctx} | {c:.3f} | {g:.3f} | {speed:.6f}x | "
                 f"{(speed-1)*100:+.3f}% | {sd:.6f} |")
pathlib.Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
PY
cat "$OUTPUT_DIR/summary/summary.md"
phase=complete
