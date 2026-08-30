#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the four-way production decode A/B for the SM75-native Q4-32 gate/up and
down kernels.  The arms are: unchanged control; gate/up tile32 packed-INT4
MMA only; down tile32 packed-INT4 only; and both candidates together.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REPEATS=4
  TG_TOKENS=256
  EXACT_TOKENS=16
  WARMUP_TOKENS=512
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  SKIP_BUILD=0
  RESUME=0
  CREATE_ARCHIVE=1
  Q4_PRODUCTION_AB_DIR=...

The fixed PP frontiers are 512, 4096, and 32768. A fresh exact run evaluates
all three in one process. RESUME preserves completed exact frontiers only when
their process has healthy pre/post GPU snapshots; evidence from a GPU-loss
process is deliberately rejected.
The model recipe is fixed to mixed15: 15 Q3A4 gate/up layers and 28 Q4-32
gate/up layers, with Q4-32 down in all 43 routed layers.  Q3A4 remains on its
production tile32-DP4A K4 path in every arm.
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
REPEATS=${REPEATS:-4}
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
OUTPUT_DIR=${Q4_PRODUCTION_AB_DIR:-$repo_dir/sm75-decode-q4-production-ab-$stamp}

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
(( REPEATS >= 4 && REPEATS % 4 == 0 && TG_TOKENS == 256 && EXACT_TOKENS == 16 &&
   WARMUP_TOKENS == 512 && PREFILL_CHUNK == 2048 && PIPELINE_MB == 512 )) ||
    die "require repeats>=4 and divisible by 4, tg_tokens=256, exact_tokens=16, warmup=512, prefill_chunk=2048, pipeline_mb=512"
for flag in SKIP_BUILD RESUME CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
Q3A4_LAYOUT=mixed15
Q3A4_LAYER_COUNT=15
Q3A4_LAYER_LIST=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
Q4_GATE_LAYER_COUNT=28
Q4_DOWN_LAYER_COUNT=43
for tool in awk basename cat cmp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 sha256sum sort stat tail tar tee tr wc; do
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
    printf 'variants=control,gate-mma,down-tile32,both\n'
    printf 'git_commit=%s\nmodel_stat=%s\n' "$(git rev-parse HEAD)" \
        "$(stat -Lc '%d:%i:%s:%Y' "$MODEL")"
    printf 'q3a4_decode_mapping=tile32-dp4a\nq3a4_decode_ksplit=4\n'
    printf 'q3a4_layout=%s\nq3a4_layer_count=%s\nq3a4_layers=%s\n' \
        "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST"
}

if [[ $RESUME == 1 ]]; then
    [[ -n ${Q4_PRODUCTION_AB_DIR:-} && -d $OUTPUT_DIR ]] ||
        die "RESUME=1 requires an existing Q4_PRODUCTION_AB_DIR"
    [[ -f $OUTPUT_DIR/configuration.txt ]] || die "resume configuration is missing"
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
    "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$WARMUP_TOKENS"
    DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1
    DS4_CUDA_NO_MOE_Q32_DECODE_SPLIT=1
    DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG=1
    DS4_CUDA_MOE_Q3A4_DECODE_MAPPING=tile32-dp4a
    DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT=4
    DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT=1
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
    grep -Fq 'SM75 Q4-32 hwarp16/tile32-dp4a/tile32-mma gate/up and owned decode nonzero exact/reuse' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 gate/up candidate exact marker missing"
    grep -Fq 'SM75 Q4-32 signed-zero gate probe exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 gate/up signed-zero marker missing"
    grep -Fq 'Q4-32 down tile32 packed-INT4 owned_slots nonzero poison-overwrite exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 down owned_slots exact marker missing"
    grep -Fq 'Q4-32 down tile32 packed-INT4 owned_packed masks 000..111 poison-overwrite exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 down owned_packed exact marker missing"
    grep -Fq 'Q4-32 down tile32 signed-zero boundary exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 down signed-zero marker missing"
    grep -Fq 'SM75 Q4-32 tile32-mma gate/up + tile32 down production defaults' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 production-default marker missing"
    grep -Fq 'SM75 Q3A4 tile32-dp4a-k4 production default' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 production-default marker missing"
else
    make -q ds4-bench tests/cuda_long_context_smoke CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

emit_build_identity() {
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'worktree_diff_sha256=%s\n' \
        "$(git diff --no-ext-diff --binary HEAD -- | sha256sum | awk '{print $1}')"
    sha256sum ds4-bench tests/cuda_long_context_smoke
}
build_identity="$OUTPUT_DIR/provenance/build-identity.txt"
if [[ $RESUME == 1 && -s $build_identity ]]; then
    cmp -s <(emit_build_identity) "$build_identity" ||
        die "resume binaries or source identity differ from the original audit"
else
    emit_build_identity >"$build_identity"
fi

phase=manifest
if [[ $RESUME == 0 ]]; then
    {
        printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$(git branch --show-current)"
        printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\n' \
            "$MODEL" "$(stat -c %s "$MODEL")"
        printf 'comparison=q4-control-vs-gate-mma-vs-down-tile32-vs-both\n'
        printf 'q3a4_layout=%s\nq3a4_layer_count=%s\nq3a4_layers=%s\n' \
            "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST"
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
    awk -v expected_count="$Q3A4_LAYER_COUNT" \
        -v expected_layers="$Q3A4_LAYER_LIST" '
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
            if (layer !~ /^[0-9]+$/ || layer<0 || layer>42 ||
                layer_seen[layer]++) bad=1
            if (gate=="sm75_q3a4") {
                if (up!="sm75_q3a4" || down!="sm75_q4_32") bad=1
                q3++
                layers=layers (layers ? "," : "") layer
            } else {
                if (gate!="sm75_q4_32" || up!="sm75_q4_32" ||
                    down!="sm75_q4_32") bad=1
            }
        }
        END {
            for (i=0; i<43; i++) if (layer_seen[i]!=1) bad=1
            exit !(seen==43 && q3==expected_count &&
                   layers==expected_layers && !bad)
        }
    ' "$log"
}

q4_gate_active_count() {
    local variant=$1 log=$2
    awk -v variant="$variant" '
        /SM75 Q4-32 decode mapping audit/ {
            seen++
            c=h=d=m=-1
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control") c=a[2]+0
                if (a[1]=="hwarp16") h=a[2]+0
                if (a[1]=="tile32-dp4a") d=a[2]+0
                if (a[1]=="tile32-mma") m=a[2]+0
            }
            if (variant=="control" || variant=="down-tile32") {
                good=(c>0 && h==0 && d==0 && m==0)
                active=c
            } else {
                good=(c==0 && h==0 && d==0 && m>0)
                active=m
            }
        }
        END {
            if (seen!=1 || !good) exit 1
            print active
        }
    ' "$log"
}

q4_down_active_count() {
    local variant=$1 log=$2
    awk -v variant="$variant" '
        /SM75 Q4-32 down decode mapping audit/ {
            seen++
            c=t=cs=cp=ts=tp=-1
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control") c=a[2]+0
                if (a[1]=="tile32") t=a[2]+0
                if (a[1]=="control-slots") cs=a[2]+0
                if (a[1]=="control-packed") cp=a[2]+0
                if (a[1]=="tile32-slots") ts=a[2]+0
                if (a[1]=="tile32-packed") tp=a[2]+0
            }
            if (variant=="control" || variant=="gate-mma") {
                good=(c>0 && t==0 && cs>0 && cp>0 && ts==0 && tp==0 &&
                      c==cs+cp); active=c
            } else {
                good=(c==0 && t>0 && cs==0 && cp==0 && ts>0 && tp>0 &&
                      t==ts+tp); active=t
            }
        }
        END {
            if (seen!=1 || !good) exit 1
            print active
        }
    ' "$log"
}

q4_down_kind_counts() {
    local variant=$1 log=$2
    awk -v variant="$variant" '
        /SM75 Q4-32 down decode mapping audit/ {
            seen++
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control-slots") cs=a[2]+0
                if (a[1]=="control-packed") cp=a[2]+0
                if (a[1]=="tile32-slots") ts=a[2]+0
                if (a[1]=="tile32-packed") tp=a[2]+0
            }
        }
        END {
            if (seen!=1) exit 1
            if (variant=="control" || variant=="gate-mma")
                print cs, cp
            else
                print ts, tp
        }
    ' "$log"
}

q3a4_active_count() {
    local log=$1
    awk '
        /SM75 Q3A4 decode mapping audit/ {
            seen++
            c=h=t=d=k1=k2=k4=-1
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control") c=a[2]+0
                if (a[1]=="hwarp16") h=a[2]+0
                if (a[1]=="tile32") t=a[2]+0
                if (a[1]=="tile32-dp4a") d=a[2]+0
                if (a[1]=="k1") k1=a[2]+0
                if (a[1]=="k2") k2=a[2]+0
                if (a[1]=="k4") k4=a[2]+0
            }
            good=(c==0 && h==0 && t==0 && d>0 &&
                  k1==0 && k2==0 && k4==d)
            active=d
        }
        END {
            if (seen!=1 || !good) exit 1
            print active
        }
    ' "$log"
}

validate_dispatch_audit() {
    local variant=$1 log=$2
    if [[ $variant == control || $variant == down-tile32 ]]; then
        grep -Fxq 'ds4: SM75 Q4-32 decode gate/up mapping=control (explicit fallback)' \
            "$log" || return 1
    else
        grep -Fxq 'ds4: SM75 Q4-32 decode gate/up mapping=tile32-mma (production default)' \
            "$log" || return 1
    fi
    if [[ $variant == down-tile32 || $variant == both ]]; then
        grep -Fxq 'ds4: SM75 Q4-32 down decode mapping=tile32-int4 (production default)' \
            "$log" || return 1
    else
        grep -Fxq 'ds4: SM75 Q4-32 down decode mapping=control (explicit fallback)' \
            "$log" || return 1
    fi
    grep -Fxq 'ds4: SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4 (production default)' \
        "$log" || return 1
    q4_gate_active_count "$variant" "$log" >/dev/null &&
        q4_down_active_count "$variant" "$log" >/dev/null &&
        q3a4_active_count "$log" >/dev/null
}

variant_env() {
    local variant=$1
    case "$variant" in
        control)
            printf 'DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=control\n'
            printf 'DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=control\n'
            ;;
        gate-mma)
            printf 'DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=tile32-mma\n'
            printf 'DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=control\n'
            ;;
        down-tile32)
            printf 'DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=control\n'
            printf 'DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=tile32\n'
            ;;
        both)
            printf 'DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=tile32-mma\n'
            printf 'DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=tile32\n'
            ;;
        *) return 1 ;;
    esac
}

capture_gpu_health() {
    local output=$1 gpu
    local partial="$output.partial.$$"
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,memory.used,memory.total,pstate \
            --format=csv,noheader >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

validate_gpu_health_pair() {
    local base=$1 file
    for file in "$base.pre-gpu.csv" "$base.post-gpu.csv"; do
        [[ -s $file ]] || return 1
        grep -Eq 'ERR!|Unknown Error|GPU is lost' "$file" && return 1
        awk -F, -v expected="$GPU_DEVICES" '
            BEGIN {count=split(expected, ids, ",")}
            {
                gsub(/[[:space:]]/, "", $1)
                if (NF!=6 || NR>count || $1!=ids[NR]) bad=1
            }
            END {exit !(NR==count && !bad)}
        ' "$file" || return 1
    done
}

validate_q8_state_files() {
    local base=$1 plan="$base.q8-plan.csv" bindings="$base.q8-bindings.csv"
    [[ -s $plan && -s $bindings ]] || return 1
    [[ $(wc -l <"$plan") == 345 && $(wc -l <"$bindings") == 345 ]] ||
        return 1
    awk -F, 'NR==1 {
            exit !($1=="sequence" && $2=="label" &&
                   $3=="consumer_device" && $7=="resident_device" &&
                   $13=="status")
        }' "$plan" || return 1
    awk -F, 'NR==1 {
            exit !($1=="consumer_device" && $2=="resident_device" &&
                   $3=="partner_offload" && $12=="label" &&
                   $15=="live")
        }' "$bindings"
}

canonical_q8_bindings() {
    awk -F, 'BEGIN {OFS=","}
        NR>1 {
            print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$14,$15
        }' "$1" | sort
}

validate_q8_plan_equal() {
    local reference_log=$1 candidate_log=$2 marker reference_line candidate_line
    local reference_base=${reference_log%.log} candidate_base=${candidate_log%.log}
    for marker in 'CUDA q8 fp16 benefit plan candidates=' \
                  'CUDA q8 fp16 stage-aware 22/21 planner selected ' \
                  'CUDA q8 fp16 benefit plan materialized '; do
        [[ $(grep -Fc "$marker" "$reference_log") == 1 &&
           $(grep -Fc "$marker" "$candidate_log") == 1 ]] || return 1
        reference_line=$(grep -F "$marker" "$reference_log") || return 1
        candidate_line=$(grep -F "$marker" "$candidate_log") || return 1
        [[ $reference_line == "$candidate_line" ]] || {
            printf 'Q8 plan mismatch for %s\nreference: %s\ncandidate: %s\n' \
                "$marker" "$reference_line" "$candidate_line" >&2
            return 1
        }
    done
    validate_q8_state_files "$reference_base" &&
        validate_q8_state_files "$candidate_base" || return 1
    cmp -s "$reference_base.q8-plan.csv" "$candidate_base.q8-plan.csv" || {
        printf 'Q8 placement-plan identities differ between %s and %s\n' \
            "$reference_log" "$candidate_log" >&2
        return 1
    }
    cmp -s <(canonical_q8_bindings "$reference_base.q8-bindings.csv") \
           <(canonical_q8_bindings "$candidate_base.q8-bindings.csv") || {
        printf 'Q8 binding identities differ between %s and %s\n' \
            "$reference_log" "$candidate_log" >&2
        return 1
    }
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
    local -a mapping_env state_env=(
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv"
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv"
    )
    mapfile -t mapping_env < <(variant_env "$variant")
    local ctx_alloc=$((CTX_MAX + EXACT_TOKENS + 1))
    local rc=0
    printf 'Exact Q4 decode logits: %s (all frontiers)...\n' "$variant"
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    "${production_env[@]}" "${mapping_env[@]}" "${state_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --ctx-alloc "$ctx_alloc" \
        --step-mul "$STEP_MUL" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$EXACT_TOKENS" --dump-decode-logits-dir "$logits" \
        --csv "$base.csv" >"$base.log" 2>&1 || rc=$?
    capture_gpu_health "$base.post-gpu.csv" || return 1
    return "$rc"
}

mapping_count_is_exact() {
    local variant=$1 log=$2 frontier_count=$3 gate down q3 slots packed
    gate=$(q4_gate_active_count "$variant" "$log") || return 1
    down=$(q4_down_active_count "$variant" "$log") || return 1
    q3=$(q3a4_active_count "$log") || return 1
    read -r slots packed < <(q4_down_kind_counts "$variant" "$log") || return 1
    (( gate == frontier_count * EXACT_TOKENS * Q4_GATE_LAYER_COUNT * 2 &&
       down == frontier_count * EXACT_TOKENS * Q4_DOWN_LAYER_COUNT * 2 &&
       q3 == frontier_count * EXACT_TOKENS * Q3A4_LAYER_COUNT * 2 &&
       slots == frontier_count * EXACT_TOKENS * Q4_DOWN_LAYER_COUNT &&
       packed == frontier_count * EXACT_TOKENS * Q4_DOWN_LAYER_COUNT ))
}

run_one() {
    local variant=$1 tokens=$2 csv=$3 log=$4
    local base=${log%.log}
    local -a mapping_env state_env=(
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv"
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv"
    )
    mapfile -t mapping_env < <(variant_env "$variant")
    local ctx_alloc=$((CTX_MAX + tokens + 1))
    local rc=0
    capture_gpu_health "${log%.log}.pre-gpu.csv" || return 1
    "${production_env[@]}" "${mapping_env[@]}" "${state_env[@]}" \
    ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --ctx-alloc "$ctx_alloc" \
        --step-mul "$STEP_MUL" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$tokens" --csv "$csv" >"$log" 2>&1 || rc=$?
    capture_gpu_health "${log%.log}.post-gpu.csv" || return 1
    (( rc == 0 )) || return "$rc"
    validate_csv "$csv" "$tokens" && validate_common_log "$log" &&
        validate_dispatch_audit "$variant" "$log" &&
        validate_q8_state_files "$base"
}

write_throughput_summary() {
    python3 - "$OUTPUT_DIR/runs/runs.tsv" "$OUTPUT_DIR/summary/summary.csv" \
               "$OUTPUT_DIR/summary/summary.md" <<'PY'
import csv
import pathlib
import statistics
import sys

variants = ("control", "gate-mma", "down-tile32", "both")
rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
by_pair = {}
for row in rows:
    key = (int(row["repeat"]), int(row["context"]))
    by_pair.setdefault(key, {})[row["variant"]] = {
        "steady_tps": float(row["steady_tps"]),
        "first_ms": float(row["first_ms"]),
    }

groups = {}
for (repeat, context), values in by_pair.items():
    if set(values) != set(variants):
        raise SystemExit(f"unpaired sample at repeat={repeat} context={context}")
    groups.setdefault(context, []).append(values)

records = []
for context in sorted(groups):
    samples = groups[context]
    medians = {v: statistics.median(s[v]["steady_tps"] for s in samples)
               for v in variants}
    first = {v: statistics.median(s[v]["first_ms"] for s in samples)
             for v in variants}
    speedups = {}
    sds = {}
    for variant in variants[1:]:
        ratios = [s[variant]["steady_tps"] / s["control"]["steady_tps"]
                  for s in samples]
        speedups[variant] = statistics.median(ratios)
        sds[variant] = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    gate_effect = statistics.median([
        ((s["gate-mma"]["steady_tps"] / s["control"]["steady_tps"]) *
         (s["both"]["steady_tps"] / s["down-tile32"]["steady_tps"])) ** 0.5
        for s in samples])
    down_effect = statistics.median([
        ((s["down-tile32"]["steady_tps"] / s["control"]["steady_tps"]) *
         (s["both"]["steady_tps"] / s["gate-mma"]["steady_tps"])) ** 0.5
        for s in samples])
    interaction = statistics.median([
        (s["both"]["steady_tps"] * s["control"]["steady_tps"]) /
        (s["gate-mma"]["steady_tps"] * s["down-tile32"]["steady_tps"])
        for s in samples])
    records.append((context, medians, first, speedups, sds, len(samples),
                    gate_effect, down_effect, interaction))

with pathlib.Path(sys.argv[2]).open("w", newline="") as handle:
    out = csv.writer(handle)
    header = ["context"]
    for variant in variants:
        header += [f"{variant}_tps", f"{variant}_ms_per_token",
                   f"{variant}_first_ms"]
    for variant in variants[1:]:
        header += [f"{variant}_speedup", f"{variant}_change_pct",
                   f"{variant}_speedup_sd"]
    out.writerow(header + ["samples", "gate_geomean_main_effect",
                           "down_geomean_main_effect", "interaction_ratio"])
    for context, medians, first, speedups, sds, samples, gate_effect, down_effect, interaction in records:
        row = [context]
        for variant in variants:
            row += [f"{medians[variant]:.6f}",
                    f"{1000.0 / medians[variant]:.6f}",
                    f"{first[variant]:.6f}"]
        for variant in variants[1:]:
            speedup = speedups[variant]
            row += [f"{speedup:.9f}", f"{(speedup - 1) * 100:.6f}",
                    f"{sds[variant]:.9f}"]
        out.writerow(row + [samples, f"{gate_effect:.9f}",
                            f"{down_effect:.9f}", f"{interaction:.9f}"])

lines = [
    "# SM75 production Q4-32 gate/up and down decode A/B",
    "",
    "Four arms isolate Q4-32 gate/up tile32 packed-INT4 MMA and Q4-32 down "
    "tile32 packed-INT4. Q3A4 remains tile32-DP4A K4; placement and all "
    "cross-GPU boundaries are fixed.",
    "",
    "Factorial main effects are geometric means of the two applicable paired contrasts; an interaction ratio of 1.0 means multiplicative independence.",
    "",
    "| Context | Control tok/s | Gate MMA tok/s | Down tile32 tok/s | Both tok/s | Gate speedup | Down speedup | Both speedup | Gate main effect | Down main effect | Interaction |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for context, medians, first, speedups, sds, _, gate_effect, down_effect, interaction in records:
    lines.append(
        f"| {context} | {medians['control']:.3f} | {medians['gate-mma']:.3f} | "
        f"{medians['down-tile32']:.3f} | {medians['both']:.3f} | "
        f"{speedups['gate-mma']:.6f}x | {speedups['down-tile32']:.6f}x | "
        f"{speedups['both']:.6f}x | {gate_effect:.6f}x | "
        f"{down_effect:.6f}x | {interaction:.6f}x |"
    )
pathlib.Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
PY
}

phase=throughput
runs_partial="$OUTPUT_DIR/runs/runs.tsv.partial.$$"
dispatch_partial="$OUTPUT_DIR/runs/dispatch.tsv.partial.$$"
printf 'repeat\tslot\tcontext\tvariant\tsteady_tps\tfirst_ms\tcsv\tlog\n' >"$runs_partial"
printf 'repeat\tvariant\tq4_gate_calls\tq4_down_calls\tq4_down_slots\tq4_down_packed\tq3a4_k4_calls\tq8_plan_equal\n' \
    >"$dispatch_partial"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    case $(((repeat - 1) % 4)) in
        0) variants=(control gate-mma both down-tile32) ;;
        1) variants=(gate-mma down-tile32 control both) ;;
        2) variants=(down-tile32 both gate-mma control) ;;
        3) variants=(both control down-tile32 gate-mma) ;;
    esac
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1)); base="$OUTPUT_DIR/runs/r${repeat}-${variant}"
        if [[ $RESUME == 1 && -s $base.csv && -s $base.log ]] &&
           validate_gpu_health_pair "$base" &&
           validate_q8_state_files "$base" &&
           validate_csv "$base.csv" "$TG_TOKENS" &&
           validate_common_log "$base.log" &&
           validate_dispatch_audit "$variant" "$base.log"; then
            printf 'Reusing Q4 production A/B repeat=%d/%d variant=%s...\n' \
                "$repeat" "$REPEATS" "$variant"
        else
            printf 'Q4 production A/B repeat=%d/%d slot=%d/4 variant=%s...\n' \
                "$repeat" "$REPEATS" "$slot" "$variant"
            run_one "$variant" "$TG_TOKENS" "$base.csv" "$base.log" || {
                tail -n 200 "$base.log" >&2 || true
                die "$variant production run failed validation"
            }
        fi
        while IFS=, read -r context _ _ _ _ first _ steady _; do
            [[ $context == ctx_tokens ]] && continue
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$repeat" "$slot" "$context" "$variant" "$steady" "$first" \
                "$base.csv" "$base.log" >>"$runs_partial"
        done <"$base.csv"
    done
    control_log="$OUTPUT_DIR/runs/r${repeat}-control.log"
    expected_gate=$((3 * TG_TOKENS * Q4_GATE_LAYER_COUNT * 2))
    expected_down=$((3 * TG_TOKENS * Q4_DOWN_LAYER_COUNT * 2))
    expected_kind=$((3 * TG_TOKENS * Q4_DOWN_LAYER_COUNT))
    expected_q3=$((3 * TG_TOKENS * Q3A4_LAYER_COUNT * 2))
    reference_kinds=
    for variant in control gate-mma down-tile32 both; do
        log="$OUTPUT_DIR/runs/r${repeat}-${variant}.log"
        validate_q8_plan_equal "$control_log" "$log" ||
            die "repeat $repeat changed dense-Q8 placement in $variant"
        gate_calls=$(q4_gate_active_count "$variant" "$log")
        down_calls=$(q4_down_active_count "$variant" "$log")
        q3_calls=$(q3a4_active_count "$log")
        read -r down_slots down_packed < <(q4_down_kind_counts "$variant" "$log")
        [[ $gate_calls == "$expected_gate" &&
           $down_calls == "$expected_down" && $q3_calls == "$expected_q3" &&
           $down_slots == "$expected_kind" &&
           $down_packed == "$expected_kind" ]] ||
            die "repeat $repeat $variant call inventory mismatch (gate $gate_calls/$expected_gate; down $down_calls/$expected_down; q3a4 $q3_calls/$expected_q3)"
        kinds="$down_slots,$down_packed"
        if [[ -z $reference_kinds ]]; then reference_kinds=$kinds
        else [[ $kinds == "$reference_kinds" ]] ||
            die "repeat $repeat changed down ownership-shape inventory ($reference_kinds vs $kinds)"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\ttrue\n' \
            "$repeat" "$variant" "$gate_calls" "$down_calls" \
            "$down_slots" "$down_packed" "$q3_calls" >>"$dispatch_partial"
    done
done
mv -- "$runs_partial" "$OUTPUT_DIR/runs/runs.tsv"
mv -- "$dispatch_partial" "$OUTPUT_DIR/runs/dispatch.tsv"
phase=throughput-summary
write_throughput_summary
cat "$OUTPUT_DIR/summary/summary.md"

phase=exact-logits
for variant in control gate-mma down-tile32 both; do
    base="$OUTPUT_DIR/exact/$variant"; logits="$base-logits"
    valid=0
    if [[ $RESUME == 1 && -s $base.csv && -s $base.log && -d $logits ]] &&
       validate_gpu_health_pair "$base" &&
       validate_q8_state_files "$base" &&
       validate_csv "$base.csv" "$EXACT_TOKENS" &&
       validate_common_log "$base.log" &&
       validate_dispatch_audit "$variant" "$base.log" &&
       mapping_count_is_exact "$variant" "$base.log" 3 &&
       frontier_complete "$logits" 512 &&
       frontier_complete "$logits" 4096 &&
       frontier_complete "$logits" 32768; then
        valid=1
        printf 'Reusing exact Q4 decode logits: %s (all frontiers)...\n' \
            "$variant"
    fi
    if [[ $valid == 0 ]]; then
        # Never repair individual frontiers into a shared logit directory.
        # A resumed arm is either wholly reusable or wholly rerun, preserving
        # one process history and one dispatch/placement inventory.
        attempt="$base-logits-attempt-$stamp-$$"
        [[ ! -e $attempt ]] || die "exact attempt path already exists: $attempt"
        mkdir -p "$attempt"
        run_exact_full "$variant" "$base" "$attempt" ||
            die "$variant exact-logit run failed"
        validate_gpu_health_pair "$base" &&
            validate_q8_state_files "$base" &&
            validate_csv "$base.csv" "$EXACT_TOKENS" &&
            validate_common_log "$base.log" &&
            validate_dispatch_audit "$variant" "$base.log" &&
            mapping_count_is_exact "$variant" "$base.log" 3 ||
            die "$variant exact run omitted its production or mapping path"
        for context in 512 4096 32768; do
            frontier_complete "$attempt" "$context" ||
                die "$variant fresh exact attempt omitted PP=$context logits"
        done
        [[ $(find "$attempt" -maxdepth 1 -type f -name '*.f32' | wc -l) == \
           $((3 * EXACT_TOKENS)) ]] ||
            die "$variant fresh exact attempt has an unexpected inventory"
        if [[ -d $logits ]]; then
            find "$logits" -maxdepth 1 -type f -name '*.f32' -delete
            rmdir "$logits" || die "could not retire stale exact directory: $logits"
        fi
        mv -- "$attempt" "$logits"
    fi
    for context in 512 4096 32768; do
        frontier_complete "$logits" "$context" ||
            die "$variant did not emit all PP=$context decode-logit files"
    done
    [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == \
       $((3 * EXACT_TOKENS)) ]] ||
        die "$variant emitted an unexpected decode-logit inventory"
done

control_ref="$OUTPUT_DIR/exact/control.log"
for variant in control gate-mma down-tile32 both; do
    validate_q8_plan_equal "$control_ref" "$OUTPUT_DIR/exact/$variant.log" ||
        die "exact arm changed the dense-Q8 placement plan: $variant"
done
minimum_gate_calls=$((3 * EXACT_TOKENS * Q4_GATE_LAYER_COUNT * 2))
minimum_down_calls=$((3 * EXACT_TOKENS * Q4_DOWN_LAYER_COUNT * 2))
minimum_q3_calls=$((3 * EXACT_TOKENS * Q3A4_LAYER_COUNT * 2))
declare -A exact_gate_calls exact_down_calls exact_q3_calls
for variant in control gate-mma down-tile32 both; do
    log="$OUTPUT_DIR/exact/$variant.log"
    gate_calls=$(q4_gate_active_count "$variant" "$log")
    down_calls=$(q4_down_active_count "$variant" "$log")
    q3_calls=$(q3a4_active_count "$log")
    read -r down_slots down_packed < <(q4_down_kind_counts "$variant" "$log")
    (( gate_calls == minimum_gate_calls && down_calls == minimum_down_calls &&
       q3_calls == minimum_q3_calls &&
       down_slots == minimum_down_calls / 2 &&
       down_packed == minimum_down_calls / 2 )) ||
        die "$variant exact sources omitted calls (gate $gate_calls/$minimum_gate_calls; down $down_calls/$minimum_down_calls; q3a4 $q3_calls/$minimum_q3_calls)"
    exact_gate_calls[$variant]=$gate_calls
    exact_down_calls[$variant]=$down_calls
    exact_q3_calls[$variant]=$q3_calls
done

expected_files="$OUTPUT_DIR/exact/expected-files.txt"
: >"$expected_files"
for context in 512 4096 32768; do
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf 'frontier_%06d.decode_%06d.logits.f32\n' \
            "$context" "$token" >>"$expected_files"
    done
done
for variant in control gate-mma down-tile32 both; do
    find "$OUTPUT_DIR/exact/$variant-logits" -maxdepth 1 -type f -name '*.f32' \
        -printf '%f\n' | sort >"$OUTPUT_DIR/exact/$variant-files.txt"
    cmp -s "$expected_files" "$OUTPUT_DIR/exact/$variant-files.txt" ||
        die "$variant emitted an unexpected logit inventory"
done
for variant in gate-mma down-tile32 both; do
    while IFS= read -r file; do
        if ! cmp -s "$OUTPUT_DIR/exact/control-logits/$file" \
                  "$OUTPUT_DIR/exact/$variant-logits/$file"; then
            padded_context=${file#frontier_}
            padded_context=${padded_context%%.*}
            context=$((10#$padded_context))
            die "$variant diverged at $file"
        fi
    done <"$OUTPUT_DIR/exact/control-files.txt"
done
printf 'bit_exact=true\nfrontiers=512,4096,32768\ndecode_tokens_per_frontier=%s\n' \
    "$EXACT_TOKENS" >"$OUTPUT_DIR/exact/verification.txt"
printf 'q3a4_layout=%s\nq3a4_layer_count=%s\nq3a4_layers=%s\n' \
    "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST" \
    >>"$OUTPUT_DIR/exact/verification.txt"
printf 'minimum_q4_gate_calls=%s\nminimum_q4_down_calls=%s\nminimum_q3a4_k4_calls=%s\n' \
    "$minimum_gate_calls" "$minimum_down_calls" "$minimum_q3_calls" \
    >>"$OUTPUT_DIR/exact/verification.txt"
for variant in control gate-mma down-tile32 both; do
    printf '%s_q4_gate_calls=%s\n%s_q4_down_calls=%s\n%s_q3a4_k4_calls=%s\n' \
        "$variant" "${exact_gate_calls[$variant]}" \
        "$variant" "${exact_down_calls[$variant]}" \
        "$variant" "${exact_q3_calls[$variant]}" \
        >>"$OUTPUT_DIR/exact/verification.txt"
done
printf 'q8_plan_equal=true\n' >>"$OUTPUT_DIR/exact/verification.txt"
{
    printf '\n## Exact-output verification\n\n'
    printf 'All %s decode logits at PP512, PP4096, and PP32768 were byte-identical.\n\n' \
        "$EXACT_TOKENS"
    printf 'Q3A4 remains production K4: `%s` (%s routed layers: %s).\n' \
        "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST"
} >>"$OUTPUT_DIR/summary/summary.md"

cmp -s <(emit_build_identity) "$build_identity" ||
    die "evidence binaries or source identity changed during the audit"
phase=complete
