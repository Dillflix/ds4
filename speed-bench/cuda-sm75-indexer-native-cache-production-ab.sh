#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the fail-closed production gate for the SM75-native persistent-F16
indexer cache and streaming WMMA64 scorer.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                 default: speed-bench/promessi_sposi.txt
  QUALITY_MANIFEST=...       default: Flash fixed 100-case manifest
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=32768
  CTX_ALLOC=262273
  REPEATS=3
  RUN_QUALITY=1
  RUN_WORD_SMOKE=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  REUSE_LEGACY_QUALITY_DIR=/absolute/path/to/prior/quality
  INDEXER_NATIVE_AB_DIR=/absolute/output/directory

The legacy arm stores indexer K in F32 and dispatches shipping WMMA128. The
native arm commits QAT-completed rows once to an F16 cache and dispatches the
SM75 streaming64 kernel. Advancement requires exact bounded score/top-k and
one-token results, identical 100-case production scores, byte-identical
frontier logits, and a passing long-prompt early-decode smoke.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL=${MODEL:-}
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged mixed-Q4/IQ2 model"
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
QUALITY_MANIFEST=${QUALITY_MANIFEST:-$repo_dir/gguf-tools/quality-testing/data/flash/manifest.tsv}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-32768}
CTX_ALLOC=${CTX_ALLOC:-262273}
REPEATS=${REPEATS:-3}
RUN_QUALITY=${RUN_QUALITY:-1}
RUN_WORD_SMOKE=${RUN_WORD_SMOKE:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
REUSE_LEGACY_QUALITY_DIR=${REUSE_LEGACY_QUALITY_DIR:-}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${INDEXER_NATIVE_AB_DIR:-$repo_dir/sm75-indexer-native-cache-production-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
[[ $QUALITY_MANIFEST == /* && -f $QUALITY_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute file"
if [[ -n $REUSE_LEGACY_QUALITY_DIR ]]; then
    [[ $REUSE_LEGACY_QUALITY_DIR == /* &&
       -f $REUSE_LEGACY_QUALITY_DIR/legacy-f32.tsv &&
       -f $REUSE_LEGACY_QUALITY_DIR/legacy-f32.log ]] ||
        die "REUSE_LEGACY_QUALITY_DIR must contain legacy-f32.tsv and legacy-f32.log"
fi
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "CTX_ALLOC:$CTX_ALLOC" \
            "REPEATS:$REPEATS" "RUN_QUALITY:$RUN_QUALITY" \
            "RUN_WORD_SMOKE:$RUN_WORD_SMOKE" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 &&
   CTX_START == 2048 && CTX_MAX == 32768 && CTX_ALLOC == 262273 &&
   REPEATS >= 2 )) || die "production gate requires 2K..32K, 256K allocation, and at least two repeats"
for flag in RUN_QUALITY RUN_WORD_SMOKE SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in awk basename cat cmp cp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 rm sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{harness,quality,production,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            rm -f -- "$partial"
            printf 'error: failed to create archive %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four IDs"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]:-} ]] ||
        die "invalid or duplicate physical GPU index: $gpu"
    seen_gpu[$gpu]=1
    compute_cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $compute_cap == 7.5 ]] ||
        die "physical GPU $gpu has compute capability ${compute_cap:-unknown}; SM75 is required"
done

phase=build
targets=(ds4 ds4-bench gguf-tools/quality-testing/score_official
         tests/cuda_sm75_profile_harness tests/cuda_long_context_smoke
         tests/test_engine_mgpu_placement)
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
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/cuda-exactness.log" 2>&1 || {
        tail -n 220 "$OUTPUT_DIR/cuda-exactness.log" >&2
        die "CUDA exactness tests failed"
    }

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'quality_manifest=%s\ngpu_devices=%s\nstage_split=%s/%s\n' \
        "$QUALITY_MANIFEST" "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'reuse_legacy_quality_dir=%s\n' "${REUSE_LEGACY_QUALITY_DIR:-none}"
    printf 'ctx_start=%s\nctx_max=%s\nctx_alloc=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$CTX_ALLOC" "$REPEATS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=bounded-native-harness
for n_comp in 8192 8191; do
    log="$OUTPUT_DIR/harness/streaming64-ncomp-$n_comp.log"
    "${clean[@]}" CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" \
        DS4_PROFILE_INDEXER_TILE=64 \
        DS4_PROFILE_INDEXER_TOPK=monolithic \
        DS4_PROFILE_INDEXER_OPERANDS=streaming64 \
        DS4_PROFILE_INDEXER_N_COMP="$n_comp" \
        DS4_PROFILE_REPEATS=0 \
        ./tests/cuda_sm75_profile_harness indexer-32k \
        >"$log" 2>&1 || {
            tail -n 180 "$log" >&2
            die "native indexer harness failed at n_comp=$n_comp"
        }
    for marker in 'score_validation=bit-exact' \
                  'topk_validation=exact-order-and-set' \
                  'direct_one_validation=bit-exact' 'harness_status=ok'; do
        grep -Fq "$marker" "$log" ||
            die "native harness n_comp=$n_comp lacks $marker"
    done
done

common_env=(
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048
    DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
    DS4_CUDA_TP_PREFILL_ATTN_ROWS=1
    DS4_CUDA_INDEXER_SCORE_AUDIT=1
)

variant_env() {
    if [[ $1 == legacy-f32 ]]; then
        printf '%s\0' DS4_CUDA_NO_INDEXER_NATIVE_F16_CACHE=1
    fi
}

audit_launches() {
    local log=$1 dispatch=$2 line
    line=$(grep -F "ds4: CUDA indexer score audit dispatch=$dispatch " "$log" || true)
    [[ $(printf '%s\n' "$line" | grep -c .) == 1 ]] ||
        die "expected one $dispatch audit record in $log"
    line=${line##*launches=}
    [[ $line =~ ^[0-9]+$ ]] || die "invalid $dispatch audit count in $log"
    printf '%s\n' "$line"
}

validate_variant_log() {
    local variant=$1 log=$2 score_official=${3:-0} require_dispatch=${4:-1}
    local expected count dispatch
    local -a unexpected
    if [[ $score_official == 1 ]]; then
        grep -Fq 'score_official: runtime_path=production' "$log" ||
            die "$variant did not execute the production runtime path"
    fi
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant did not use the fixed production split"
    grep -Fqx 'ds4: SM75 native routed-Q4 layout enabled (packed A/W, planner=cost, gate=tile8, down=full-stage)' \
        "$log" || die "$variant did not use the accepted native-Q4 path"
    if [[ $variant == native-f16 ]]; then
        expected=streaming64-native
        unexpected=(wmma128 wmma128-f16-native)
        grep -Fq 'SM75 native F16 indexer cache and streaming WMMA64 enabled' "$log" ||
            die "native arm did not enable the native indexer cache"
    else
        expected=wmma128
        unexpected=(streaming64-native wmma128-f16-native)
        ! grep -Fq 'SM75 native F16 indexer cache and streaming WMMA64 enabled' "$log" ||
            die "legacy arm unexpectedly enabled the native indexer cache"
    fi
    if [[ $require_dispatch == 1 ]]; then
        count=$(audit_launches "$log" "$expected")
        (( count > 0 )) || die "$variant did not dispatch $expected"
        for dispatch in "${unexpected[@]}" wmma64 wmma32 wmma16 generic; do
            count=$(audit_launches "$log" "$dispatch")
            (( count == 0 )) || die "$variant unexpectedly dispatched $dispatch"
        done
    else
        ! grep -Fq 'ds4: CUDA indexer score audit dispatch=' "$log" ||
            die "$variant short-prompt quality fixture unexpectedly launched the indexer scorer"
    fi
    grep -Fq 't256-placement=balanced' "$log" ||
        die "$variant missed balanced T256 placement"
}

if [[ $RUN_QUALITY == 1 ]]; then
    phase=quality
    for variant in legacy-f32 native-f16; do
        mapfile -d '' -t mode_env < <(variant_env "$variant")
        out="$OUTPUT_DIR/quality/$variant.tsv"
        log="$OUTPUT_DIR/quality/$variant.log"
        if [[ $variant == legacy-f32 && -n $REUSE_LEGACY_QUALITY_DIR ]]; then
            printf 'Reusing completed legacy-f32 100-case quality result...\n'
            cp -- "$REUSE_LEGACY_QUALITY_DIR/legacy-f32.tsv" "$out"
            cp -- "$REUSE_LEGACY_QUALITY_DIR/legacy-f32.log" "$log"
        else
            printf 'Scoring 100 production cases: %s...\n' "$variant"
            "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
                ./gguf-tools/quality-testing/score_official \
                    "$MODEL" "$QUALITY_MANIFEST" "$out" 32769 \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --cuda-tensor-parallel --warm-weights --production-path \
                    >"$log" 2>&1 || {
                        tail -n 220 "$log" >&2
                        die "$variant quality suite failed"
                    }
        fi
        awk -F'\t' 'NR > 1 {n++} END {exit n == 100 ? 0 : 1}' "$out" ||
            die "$variant quality output does not contain exactly 100 cases"
        validate_variant_log "$variant" "$log" 1 0
    done
    cmp -s "$OUTPUT_DIR/quality/legacy-f32.tsv" \
           "$OUTPUT_DIR/quality/native-f16.tsv" ||
        die "native-F16 production quality scores differ from legacy F32"
fi

phase=production-ab
printf 'repeat,slot,variant,csv,log,logits\n' >"$OUTPUT_DIR/production/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then variants=(legacy-f32 native-f16); else variants=(native-f16 legacy-f32); fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        mapfile -d '' -t mode_env < <(variant_env "$variant")
        base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'Production native-cache A/B repeat=%d/%d slot=%d variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
                --prefill-chunk 2048 --gen-tokens 0 --csv "$base.csv" \
                --dump-frontier-logits-dir "$logits" \
                >"$base.log" 2>&1 || {
                    tail -n 220 "$base.log" >&2
                    die "$variant production run failed"
                }
        [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
        validate_variant_log "$variant" "$base.log"
        printf '%s,%s,%s,%s,%s,%s\n' "$repeat" "$slot" "$variant" \
            "$base.csv" "$base.log" "$logits" >>"$OUTPUT_DIR/production/runs.csv"
    done

    legacy_logits=$(awk -F, -v r="$repeat" '$1 == r && $3 == "legacy-f32" {print $6}' \
        "$OUTPUT_DIR/production/runs.csv")
    native_logits=$(awk -F, -v r="$repeat" '$1 == r && $3 == "native-f16" {print $6}' \
        "$OUTPUT_DIR/production/runs.csv")
    mapfile -t legacy_files < <(find "$legacy_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    mapfile -t native_files < <(find "$native_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    [[ ${#legacy_files[@]} == 10 && "${legacy_files[*]}" == "${native_files[*]}" ]] ||
        die "repeat $repeat frontier-logit inventory is incomplete or different"
    for file in "${legacy_files[@]}"; do
        cmp -s "$legacy_logits/$file" "$native_logits/$file" ||
            die "repeat $repeat frontier logits differ: $file"
    done
done

python3 - "$OUTPUT_DIR" <<'PY'
import csv, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
runs = list(csv.DictReader((root / "production/runs.csv").open()))
values, paired = {}, {}
expected = [2048, 4096, 8192, 16384, 32768]
for run in runs:
    rows = list(csv.DictReader(pathlib.Path(run["csv"]).open()))
    contexts = [int(row["ctx_tokens"]) for row in rows]
    if contexts != expected:
        raise SystemExit(f"frontier mismatch for {run['variant']}: {contexts}")
    for row in rows:
        key = (run["variant"], int(row["ctx_tokens"]))
        value = float(row["prefill_tps"])
        values.setdefault(key, []).append(value)
        paired[(int(run["repeat"]), *key)] = value
n = max(int(run["repeat"]) for run in runs)
with (root / "production/summary.csv").open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["ctx_tokens", "legacy_f32_median_tps", "native_f16_median_tps",
                  "paired_median_speedup", "change_pct", "frontier_logits"])
    for ctx in expected:
        ratios = [paired[(r, "native-f16", ctx)] / paired[(r, "legacy-f32", ctx)]
                  for r in range(1, n + 1)]
        speedup = statistics.median(ratios)
        out.writerow([ctx,
            f'{statistics.median(values[("legacy-f32", ctx)]):.3f}',
            f'{statistics.median(values[("native-f16", ctx)]):.3f}',
            f'{speedup:.6f}', f'{(speedup - 1.0) * 100.0:.3f}', "bit-exact"])
PY
cat "$OUTPUT_DIR/production/summary.csv"

if [[ $RUN_WORD_SMOKE == 1 ]]; then
    phase=long-prompt-word-smoke
    MODEL="$MODEL" CORPUS="$PROMPT" GPU_DEVICES="$GPU_DEVICES" \
        GPU_VRAM="$GPU_VRAM" STAGE_SPLIT="$STAGE_SPLIT" \
        CTX_ALLOC="$CTX_ALLOC" MIN_PROMPT_TOKENS=24000 GEN_TOKENS=8 \
        SKIP_BUILD=1 CREATE_ARCHIVE=0 \
        WORD_SMOKE_DIR="$OUTPUT_DIR/long-prompt-word-smoke" \
        bash ./speed-bench/cuda-sm75-long-prompt-word-smoke.sh
    grep -Fqx 'result=PASS' "$OUTPUT_DIR/long-prompt-word-smoke/result.txt" ||
        die "native-F16 long-prompt word smoke did not pass"
    grep -Fq 'SM75 native F16 indexer cache and streaming WMMA64 enabled' \
        "$OUTPUT_DIR/long-prompt-word-smoke/stderr.log" ||
        die "long-prompt smoke did not execute the native indexer path"
fi

phase=complete
printf 'SM75 native indexer-cache production gate complete: %s\n' "$OUTPUT_DIR"
