#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
PROFILE_PARTNER_GPU=${PROFILE_PARTNER_GPU:-1}
REPEATS=${REPEATS:-100}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_NCU=${RUN_NCU:-0}
RUN_PRODUCTION=${RUN_PRODUCTION:-0}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/sm75-decode-pair-next-$(date -u +%Y%m%dT%H%M%SZ)"}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TARGET=tests/cuda_decode_pair_xdev
MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
PRODUCTION_PP=${PRODUCTION_PP:-32768}
PRODUCTION_TG=${PRODUCTION_TG:-256}
PRODUCTION_REPEATS=${PRODUCTION_REPEATS:-3}
EXACT_TG=${EXACT_TG:-8}

[[ $PROFILE_GPU =~ ^[0-9]+$ &&
   $PROFILE_PARTNER_GPU =~ ^[0-9]+$ &&
   $PROFILE_GPU != "$PROFILE_PARTNER_GPU" ]] ||
    die "PROFILE_GPU and PROFILE_PARTNER_GPU must be distinct device indices"
[[ $REPEATS =~ ^[0-9]+$ && $REPEATS -ge 1 && $REPEATS -le 10000 ]] ||
    die "REPEATS must be 1..10000"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $RUN_NCU == 0 || $RUN_NCU == 1 ]] || die "RUN_NCU must be 0 or 1"
[[ $RUN_PRODUCTION == 0 || $RUN_PRODUCTION == 1 ]] ||
    die "RUN_PRODUCTION must be 0 or 1"
[[ $NCU_USE_SUDO == 0 || $NCU_USE_SUDO == 1 ]] ||
    die "NCU_USE_SUDO must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
[[ $CUDA_ARCH == sm_75 ]] || die "this experiment requires CUDA_ARCH=sm_75"
if [[ $RUN_PRODUCTION == 1 ]]; then
    [[ $MODEL == /* && -f $MODEL ]] ||
        die "RUN_PRODUCTION=1 requires MODEL to name an existing absolute GGUF"
    [[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
    [[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto &&
       $STAGE_SPLIT == 22 ]] ||
        die "production A/B requires GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
    for value in "$PRODUCTION_PP" "$PRODUCTION_TG" \
                 "$PRODUCTION_REPEATS" "$EXACT_TG"; do
        [[ $value =~ ^[0-9]+$ && $value -ge 1 ]] ||
            die "production sizes and repeats must be positive integers"
    done
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/ncu"
archive="${OUTPUT_DIR%/}.tar.gz"
finish() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ $CREATE_ARCHIVE == 1 && -d $OUTPUT_DIR && ! -s $archive ]]; then
        local partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")"; then
            mv -- "$partial" "$archive"
        else
            status=1
            rm -f -- "$partial"
        fi
    fi
    printf 'Archive to return: %s\n' "$archive" >&2
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

sources=(tests/cuda_decode_pair_xdev.c ds4_cuda.cu ds4_gpu.h
         ds4_gpu_mgpu.h Makefile)
if [[ $SKIP_BUILD == 0 ]]; then
    targets=("$TARGET")
    [[ $RUN_PRODUCTION == 0 ]] || targets+=(ds4-bench)
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH="$CUDA_ARCH"
else
    [[ -x $TARGET ]] || die "SKIP_BUILD=1 found no $TARGET"
    make -q "$TARGET" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found stale targets"
    for source in "${sources[@]}"; do
        [[ $TARGET -nt $source ]] ||
            die "SKIP_BUILD=1 found $TARGET older than $source"
    done
    if [[ $RUN_PRODUCTION == 1 ]]; then
        [[ -x ./ds4-bench ]] || die "SKIP_BUILD=1 found no ds4-bench"
        make -q ds4-bench CUDA_ARCH="$CUDA_ARCH" ||
            die "SKIP_BUILD=1 found stale ds4-bench"
    fi
fi

run_harness() {
    local scenario=$1
    local log="$OUTPUT_DIR/$scenario.log"
    printf 'One-token decode pair A/B: %s...\n' "$scenario"
    DS4_DECODE_HOME_GPU="$PROFILE_GPU" \
    DS4_DECODE_PARTNER_GPU="$PROFILE_PARTNER_GPU" \
    DS4_DECODE_PAIR_REPEATS="$REPEATS" \
        "$TARGET" "$scenario" >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true
            die "$scenario harness failed"
        }
    grep -qx 'validation=bit-exact-nonzero' "$log" ||
        die "$scenario did not prove bit-exact non-zero output"
    grep -qx 'scenario=decode-indexer-row-split-32k' "$log" ||
        [[ $scenario != indexer ]] || die "indexer scenario marker missing"
    grep -qx 'scenario=decode-indexed-attention-head-split-32k' "$log" ||
        [[ $scenario != attention ]] || die "attention scenario marker missing"
    cat "$log"
}

run_harness indexer
run_harness attention

python3 - "$OUTPUT_DIR/indexer.log" "$OUTPUT_DIR/attention.log" \
        >"$OUTPUT_DIR/summary.md" <<'PY'
import pathlib
import sys

def load(path):
    values = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values

indexer = load(sys.argv[1])
attention = load(sys.argv[2])

print("# SM75 one-token decode pair experiment\n")
print("Both candidates passed bit-exact, non-zero validation against the "
      "shipping one-GPU one-token path.\n")
print("| Candidate | Baseline ms | Mirrored ms | Mirrored speedup | "
      "Transfer-inclusive ms | Transfer-inclusive speedup |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
print("| Indexer score + unchanged top-k | {baseline_chain_ms} | "
      "{mirrored_chain_ms} | {mirrored_chain_speedup}x | "
      "{transfer_inclusive_chain_ms} | "
      "{transfer_inclusive_chain_speedup}x |".format(**indexer))
print("| Indexed attention 32/32 heads | {baseline_ms} | "
      "{mirrored_chain_ms} | {mirrored_chain_speedup}x | "
      "{transfer_inclusive_ms} | {transfer_inclusive_speedup}x |".format(
          **attention))
print("\nThe transfer-inclusive indexer path sends the one-token index query "
      "and 64 head weights and gathers the partner score half. The attention "
      "path sends only the partner query heads, top-k indices, one raw and "
      "one compressed cache row, then gathers the partner head outputs.")
PY
cat "$OUTPUT_DIR/summary.md"

if [[ $RUN_PRODUCTION == 1 ]]; then
    mkdir -p "$OUTPUT_DIR/production/throughput" \
             "$OUTPUT_DIR/production/exact"
    mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
    clean=(env)
    for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
    production_env=(
        "${clean[@]}"
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
        DS4_CUDA_PREFILL_PIPELINE=1
        DS4_CUDA_PREFILL_PIPELINE_MB=512
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
        DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048
        DS4_CUDA_TP_ATTN_CACHE_DUP=1
        DS4_CUDA_TP_ATTN_HEADS_AUDIT=1
        DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    )

    validate_production_log() {
        local variant=$1 log=$2
        grep -Fq 'CUDA EP forced pipeline split 22/21' "$log" ||
            die "$variant omitted the fixed 22/21 split"
        grep -Fq 'materialized 344/344 candidates' "$log" ||
            die "$variant omitted complete dense-F16 admission"
        grep -Fq 'SM75 routed Q32 layout enabled' "$log" ||
            die "$variant omitted the tagged SM75 Q32 layout"
        if [[ $variant == heads32 ]]; then
            grep -Fq 'CUDA decode attention head split active: 32/32' "$log" &&
            grep -Fq 'indexed=1' "$log" ||
                die "heads32 omitted the production indexed decode head split"
        elif grep -Fq 'CUDA decode attention head split active: 32/32' "$log"; then
            die "control unexpectedly selected the decode head split"
        fi
    }

    run_production() {
        local variant=$1 pp=$2 tg=$3 csv=$4 log=$5
        local head_split=0
        [[ $variant == control ]] || head_split=1
        local ctx_alloc=$((pp + tg + 1))
        "${production_env[@]}" \
        "DS4_CUDA_TP_ATTN_HEADS=$head_split" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$pp" --ctx-max "$pp" --ctx-alloc "$ctx_alloc" \
            --step-incr "$pp" --prefill-chunk 2048 \
            --gen-tokens "$tg" --csv "$csv" >"$log" 2>&1 || {
                tail -n 180 "$log" >&2 || true
                die "$variant production run failed"
            }
        validate_production_log "$variant" "$log"
        awk -F, -v pp="$pp" -v tg="$tg" '
            NR == 1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                            $8=="gen_steady_tps"); next}
            NR == 2 {rows++; good=($1==pp && $4==tg && ($8+0)>0); next}
            NR > 2 {rows++}
            END {exit !(header && rows==1 && good)}
        ' "$csv" || die "$variant emitted an invalid production CSV"
    }

    printf 'repeat\tslot\tvariant\tsteady_tps\tcsv\tlog\n' \
        >"$OUTPUT_DIR/production/throughput/runs.tsv"
    for ((repeat=1; repeat<=PRODUCTION_REPEATS; repeat++)); do
        if (( repeat % 2 )); then variants=(control heads32)
        else variants=(heads32 control); fi
        slot=0
        for variant in "${variants[@]}"; do
            slot=$((slot + 1))
            base="$OUTPUT_DIR/production/throughput/r${repeat}-${variant}"
            printf 'Production decode A/B repeat=%d/%d slot=%d/2 variant=%s...\n' \
                "$repeat" "$PRODUCTION_REPEATS" "$slot" "$variant"
            run_production "$variant" "$PRODUCTION_PP" "$PRODUCTION_TG" \
                "$base.csv" "$base.log"
            steady=$(awk -F, 'NR==2 {print $8}' "$base.csv")
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$repeat" "$slot" "$variant" "$steady" \
                "$base.csv" "$base.log" \
                >>"$OUTPUT_DIR/production/throughput/runs.tsv"
        done
    done

    for variant in control heads32; do
        head_split=0
        [[ $variant == control ]] || head_split=1
        base="$OUTPUT_DIR/production/exact/$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        ctx_alloc=$((PRODUCTION_PP + EXACT_TG + 1))
        printf 'Exact production decode logits: %s...\n' "$variant"
        "${production_env[@]}" \
        "DS4_CUDA_TP_ATTN_HEADS=$head_split" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$PRODUCTION_PP" --ctx-max "$PRODUCTION_PP" \
            --ctx-alloc "$ctx_alloc" --step-incr "$PRODUCTION_PP" \
            --prefill-chunk 2048 --gen-tokens "$EXACT_TG" \
            --dump-decode-logits-dir "$logits" --csv "$base.csv" \
            >"$base.log" 2>&1 || {
                tail -n 180 "$base.log" >&2 || true
                die "$variant exact-logit run failed"
            }
        validate_production_log "$variant" "$base.log"
        [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == "$EXACT_TG" ]] ||
            die "$variant did not emit $EXACT_TG decode-logit files"
    done
    find "$OUTPUT_DIR/production/exact/control-logits" -maxdepth 1 \
        -type f -name '*.f32' -printf '%f\n' | sort \
        >"$OUTPUT_DIR/production/exact/control-files.txt"
    find "$OUTPUT_DIR/production/exact/heads32-logits" -maxdepth 1 \
        -type f -name '*.f32' -printf '%f\n' | sort \
        >"$OUTPUT_DIR/production/exact/heads32-files.txt"
    cmp -s "$OUTPUT_DIR/production/exact/control-files.txt" \
           "$OUTPUT_DIR/production/exact/heads32-files.txt" ||
        die "production variants emitted different logit inventories"
    while IFS= read -r file; do
        cmp -s "$OUTPUT_DIR/production/exact/control-logits/$file" \
               "$OUTPUT_DIR/production/exact/heads32-logits/$file" ||
            die "production head split diverged at $file"
    done <"$OUTPUT_DIR/production/exact/control-files.txt"
    printf 'bit_exact=true\ndecode_tokens=%s\n' "$EXACT_TG" \
        >"$OUTPUT_DIR/production/exact/verification.txt"

    python3 - "$OUTPUT_DIR/production/throughput/runs.tsv" "$PRODUCTION_PP" \
            >>"$OUTPUT_DIR/summary.md" <<'PY'
import csv
import pathlib
import statistics
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
by_variant = {}
by_repeat = {}
for row in rows:
    value = float(row["steady_tps"])
    by_variant.setdefault(row["variant"], []).append(value)
    by_repeat.setdefault(row["repeat"], {})[row["variant"]] = value
control = statistics.median(by_variant["control"])
heads = statistics.median(by_variant["heads32"])
paired = [values["heads32"] / values["control"]
          for values in by_repeat.values()]
speedup = statistics.median(paired)
print("\n## Fixed-production decode A/B\n")
print("Full dense-F16 admission and the 22/21 stage split were required in "
      "every run. Decode logits were byte-identical.\n")
print("| Context | Control tok/s | 32/32 heads tok/s | Paired speedup | Change |")
print("| ---: | ---: | ---: | ---: | ---: |")
print(f"| {sys.argv[2]} | {control:.3f} | "
      f"{heads:.3f} | {speedup:.6f}x | {(speedup - 1.0) * 100:+.3f}% |")
PY
    cat "$OUTPUT_DIR/summary.md"
fi

if [[ $RUN_NCU == 1 ]]; then
    command -v ncu >/dev/null 2>&1 || die "ncu not found"
    ncu_bin=$(command -v ncu)
    ncu_cmd=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_cmd=(sudo -E "$ncu_bin")
    fi
    sections=(--section SpeedOfLight --section LaunchStats
              --section Occupancy --section SchedulerStats
              --section WarpStateStats --section MemoryWorkloadAnalysis
              --section ComputeWorkloadAnalysis)

    profile_one() {
        local label=$1 scenario=$2 kernel_regex=$3 launch_skip=$4
        local expected=$5
        local base="$OUTPUT_DIR/ncu/$label" rc=0
        printf 'Nsight Compute: %s...\n' "$label"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PROFILE_PARTNER_GPU" \
            DS4_DECODE_HOME_GPU=0 DS4_DECODE_PARTNER_GPU=1 \
            DS4_DECODE_PAIR_REPEATS=1 \
            "${ncu_cmd[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function \
                --kernel-name "regex:$kernel_regex" \
                --launch-skip "$launch_skip" --launch-count 1 \
                --replay-mode kernel --cache-control none \
                --clock-control none --force-overwrite --export "$base" \
                "${sections[@]}" \
                "$TARGET" "$scenario" >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )); then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $label (exit $rc)"
        fi
        if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
                "$base.log"; then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute captured no usable kernel for $label"
        fi
        [[ -s $base.ncu-rep ]] || die "missing Nsight report: $base.ncu-rep"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" \
            --csv --page raw >"$base.csv" 2>"$base-import.log" ||
            die "could not import Nsight report: $base.ncu-rep"
        python3 speed-bench/validate-ncu-capture.py \
            "$base.csv" "$expected" 0 \
            --process cuda_decode_pair_xdev >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight report validation failed for $label"
            }
        cat "$base-validation.txt"
    }

    # The first direct-F16 launch is the one-GPU reference; the second is the
    # home half of the exact row-split candidate.
    profile_one indexer-score-half indexer \
        'indexer_score_one_direct_f16_cache_kernel.*' 1 \
        'indexer_score_one_direct_f16_cache_kernel'
    profile_one indexer-topk-8192 indexer \
        'indexer_topk_(8192_cub|pow2_u16)_kernel.*' 0 \
        'indexer_topk_(8192_cub|pow2_u16)_kernel'
    profile_one indexed-attention-baseline attention \
        'attention_indexed_mixed_kernel.*' 0 \
        'attention_indexed_mixed_kernel'
    profile_one indexed-attention-head32 attention \
        'attention_indexed_mixed_kernel.*' 1 \
        'attention_indexed_mixed_kernel'
fi

cat >"$OUTPUT_DIR/metadata.txt" <<EOF
git_commit=$(git rev-parse HEAD)
git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cuda_arch=$CUDA_ARCH
home_physical_gpu=$PROFILE_GPU
partner_physical_gpu=$PROFILE_PARTNER_GPU
repeats=$REPEATS
run_ncu=$RUN_NCU
run_production=$RUN_PRODUCTION
production_model=$MODEL
production_gpu_devices=$GPU_DEVICES
production_pp=$PRODUCTION_PP
production_tg=$PRODUCTION_TG
production_repeats=$PRODUCTION_REPEATS
EOF
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    >"$OUTPUT_DIR/gpu-inventory.csv" 2>&1 || true
nvidia-smi topo -m >"$OUTPUT_DIR/gpu-topology.txt" 2>&1 || true

if [[ $CREATE_ARCHIVE == 1 ]]; then
    tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
        "$(basename "$OUTPUT_DIR")"
fi
printf 'SM75 one-token decode pair experiment complete: %s\n' "$OUTPUT_DIR"
printf 'Archive to return: %s\n' "$archive"
trap - EXIT
