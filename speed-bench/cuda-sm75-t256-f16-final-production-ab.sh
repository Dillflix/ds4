#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    printf '%s\n' 'Run the dual-model four-GPU T256 FP16-final production A/B.'
    printf '%s\n' 'This first qualification holds the accepted STAGE_SPLIT=22 placement fixed.'
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
STAGE_SPLIT=${STAGE_SPLIT:-22}
REPEATS=${REPEATS:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CTX_ALLOC=33025
PREFILL_CHUNK=2048
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T256_F16_FINAL_PRODUCTION_AB_DIR:-$repo_dir/sm75-t256-f16-final-production-ab-$stamp}

for model in "$MIXED_MODEL" "$ALL43_MODEL"; do
    [[ $model == /* && -f $model ]] || die "model not found: $model"
done
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto ]] ||
    die "require GPU_DEVICES=0,3,1,2 and GPU_VRAM=auto"
[[ $STAGE_SPLIT == 22 ]] ||
    die "this T256 result qualification requires STAGE_SPLIT=22"
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] || die "CUDA_VISIBLE_DEVICES must be unset"
for tool in awk basename cmp date dirname env find git grep make mkdir mv nproc \
            nvidia-smi python3 rm sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
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
    printf 'gpu_devices=%s\nstage_split=%s/%s\ncontexts=512,4096,32768\nrepeats=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$REPEATS"
    printf 'candidate=T256-output-B-FP16-final-plus-half-HC-consumer\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'; nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"
capture_health() {
    local path=$1 gpu
    : >"$path"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$path" 2>&1 || return 1
    done
}
capture_health "$OUTPUT_DIR/initial-gpu.csv" || die "initial GPU health capture failed"

production=(
    "${clean[@]}"
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
)

validate_run() {
    local arm=$1 base=$2 log="$2.log" summary
    [[ -s $base.csv ]] || return 1
    for ctx in 512 4096 32768; do
        printf -v stem '%s-logits/frontier_%06d.logits' "$base" "$ctx"
        [[ -s $stem.f32 && -s $stem.json ]] || return 1
    done
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" "$log" || return 1
    grep -Fq 'materialized 344/344 candidates' "$log" || return 1
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    summary=$(grep -F 'CUDA T256 f16-final summary:' "$log" | tail -n 1 || true)
    if [[ $arm == control ]]; then
        [[ -z $summary ]] || return 1
    else
        grep -Fq 'CUDA T256 FP16 final attention result enabled (8192->4096, fused HC consumer)' "$log" || return 1
        [[ $summary =~ local=([0-9]+)[[:space:]]partner=([0-9]+) ]] || return 1
        (( BASH_REMATCH[1] + BASH_REMATCH[2] > 0 )) || return 1
    fi
    capture_health "$base.post-gpu.csv" || return 1
    cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

phase=production-ab
printf 'model_layout\trepeat\tarm\tcsv\tlog\tlogits\n' >"$OUTPUT_DIR/runs.tsv"
layouts=(mixed15 all43)
models=("$MIXED_MODEL" "$ALL43_MODEL")
for i in 0 1; do
    layout=${layouts[$i]}; model=${models[$i]}
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        if (( repeat % 2 )); then arms=(control candidate); else arms=(candidate control); fi
        for arm in "${arms[@]}"; do
            base="$OUTPUT_DIR/runs/$layout-r$repeat-$arm"
            mkdir -p "$base-logits"
            if [[ $arm == control ]]; then
                selector=(DS4_CUDA_NO_T256_F16_FINAL=1)
            else
                selector=(DS4_CUDA_T256_F16_FINAL=1)
            fi
            printf 'T256 FP16-final production A/B model=%s repeat=%s/%s arm=%s...\n' \
                "$layout" "$repeat" "$REPEATS" "$arm"
            "${production[@]}" "${selector[@]}" ./ds4-bench \
                --cuda --cuda-tensor-parallel --gpu-devices "$GPU_DEVICES" \
                --gpu-vram "$GPU_VRAM" --model "$model" --prompt-file "$PROMPT" \
                --ctx-start 512 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC" \
                --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
                --csv "$base.csv" --dump-frontier-logits-dir "$base-logits" \
                >"$base.log" 2>&1 || {
                    tail -n 220 "$base.log" >&2 || true
                    die "$layout $arm run failed"
                }
            validate_run "$arm" "$base" || {
                tail -n 220 "$base.log" >&2 || true
                die "$layout $arm validation failed"
            }
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$layout" "$repeat" "$arm" \
                "$base.csv" "$base.log" "$base-logits" >>"$OUTPUT_DIR/runs.tsv"
        done
    done
done

phase=summarize
python3 - "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/summary" <<'PY'
import array, csv, json, math, pathlib, statistics, sys
runs = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
samples = []
for run in runs:
    perf = {int(x["ctx_tokens"]): float(x["prefill_tps"])
            for x in csv.DictReader(open(run["csv"]))}
    run["perf"] = perf
for layout in ("mixed15", "all43"):
    repeats = sorted({int(r["repeat"]) for r in runs if r["model_layout"] == layout})
    for repeat in repeats:
        arms = {r["arm"]: r for r in runs
                if r["model_layout"] == layout and int(r["repeat"]) == repeat}
        for ctx in (512, 4096, 32768):
            def logits(arm):
                p = pathlib.Path(arms[arm]["logits"]) / f"frontier_{ctx:06d}.logits.f32"
                a = array.array("f"); a.frombytes(p.read_bytes()); return list(a)
            ref, cand = logits("control"), logits("candidate")
            delta = [b-a for a,b in zip(ref,cand)]
            rr = math.sqrt(statistics.fmean(x*x for x in ref))
            er = math.sqrt(statistics.fmean(x*x for x in delta))
            top_ref = max(range(len(ref)), key=ref.__getitem__)
            top_cand = max(range(len(cand)), key=cand.__getitem__)
            c = arms["control"]["perf"][ctx]; f = arms["candidate"]["perf"][ctx]
            samples.append(dict(model_layout=layout, repeat=repeat, ctx_tokens=ctx,
                control_tps=c, candidate_tps=f, speedup=f/c,
                bit_exact=int(ref == cand), top1_equal=int(top_ref == top_cand),
                nrmse=er/rr if rr else float("inf"),
                max_abs=max(map(abs,delta), default=0.0)))
with (out/"samples.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=samples[0].keys()); w.writeheader(); w.writerows(samples)
summary=[]
for layout in ("mixed15", "all43"):
    for ctx in (512,4096,32768):
        rows=[r for r in samples if r["model_layout"]==layout and r["ctx_tokens"]==ctx]
        summary.append(dict(model_layout=layout, ctx_tokens=ctx,
            control_median_tps=statistics.median(r["control_tps"] for r in rows),
            candidate_median_tps=statistics.median(r["candidate_tps"] for r in rows),
            paired_median_speedup=statistics.median(r["speedup"] for r in rows),
            top1_all=all(r["top1_equal"] for r in rows),
            max_nrmse=max(r["nrmse"] for r in rows),
            max_abs=max(r["max_abs"] for r in rows)))
with (out/"summary.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
print("model_layout,ctx_tokens,control_tps,candidate_tps,speedup,top1_all,max_nrmse,max_abs")
for r in summary:
    print(f'{r["model_layout"]},{r["ctx_tokens"]},{r["control_median_tps"]:.3f},'
          f'{r["candidate_median_tps"]:.3f},{r["paired_median_speedup"]:.6f},'
          f'{int(r["top1_all"])},{r["max_nrmse"]:.9g},{r["max_abs"]:.9g}')
if not all(r["top1_equal"] for r in samples):
    raise SystemExit("error: candidate changed a frontier top-1 token")
PY

phase=complete
printf 'SM75 T256 FP16-final production A/B complete: %s\n' "$OUTPUT_DIR"
