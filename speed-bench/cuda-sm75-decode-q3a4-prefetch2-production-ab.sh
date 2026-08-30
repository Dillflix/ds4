#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the full-engine production decode A/B for the SM75-native Q3A4
tile32-DP4A K4 gate/up kernel. The control uses ordinary K4 streaming
(prefetch depth 0); the candidate uses bounded prefetch depth 2. Q4-32,
dense-F16 placement, indexer/attention, and every cross-GPU boundary remain
identical.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-Q4-32-Q3A4.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  Q3A4_LAYOUT=mixed15          mixed15 or all43
  REPEATS=3
  TG_TOKENS=256
  EXACT_TOKENS=16
  WARMUP_TOKENS=512
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  SKIP_BUILD=0
  RESUME=0
  CREATE_ARCHIVE=1
  Q3A4_PREFETCH2_PRODUCTION_AB_DIR=...

The fixed PP frontiers are 512, 4096, and 32768. Throughput arms alternate
within every repeat. Exact logits are checkpointed per arm and frontier;
resume accepts a checkpoint only with complete logits, healthy pre/post GPU
snapshots, the expected exclusive dispatch, and valid dense-Q8 state files.
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
Q3A4_LAYOUT=${Q3A4_LAYOUT:-mixed15}
REPEATS=${REPEATS:-3}
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
WARMUP_TOKENS=${WARMUP_TOKENS:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q3A4_PREFETCH2_PRODUCTION_AB_DIR:-$repo_dir/sm75-decode-q3a4-prefetch2-production-ab-$stamp}
contexts=(512 4096 32768)
PRODUCTION_CTX_MAX=32768
PRODUCTION_CTX_ALLOC=33025
VOCAB_SIZE=129280
LOGITS_BYTES=$((VOCAB_SIZE * 4))

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute tagged SM75 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "REPEATS:$REPEATS" "TG_TOKENS:$TG_TOKENS" \
            "EXACT_TOKENS:$EXACT_TOKENS" "WARMUP_TOKENS:$WARMUP_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
(( REPEATS >= 3 && TG_TOKENS == 256 && EXACT_TOKENS == 16 &&
   WARMUP_TOKENS == 512 && PREFILL_CHUNK == 2048 && PIPELINE_MB == 512 )) ||
    die "require repeats>=3, tg_tokens=256, exact_tokens=16, warmup=512, prefill_chunk=2048, pipeline_mb=512"
for flag in SKIP_BUILD RESUME CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $RESUME == 0 || $SKIP_BUILD == 1 ]] ||
    die "RESUME=1 requires SKIP_BUILD=1 to preserve the engine identity"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"

case "$Q3A4_LAYOUT" in
    mixed15)
        Q3A4_LAYER_COUNT=15
        Q3A4_LAYER_LIST=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
        ;;
    all43)
        Q3A4_LAYER_COUNT=43
        Q3A4_LAYER_LIST=$(printf '%s,' {0..42})
        Q3A4_LAYER_LIST=${Q3A4_LAYER_LIST%,}
        ;;
    *) die "Q3A4_LAYOUT must be mixed15 or all43" ;;
esac
Q4_GATE_LAYER_COUNT=$((43 - Q3A4_LAYER_COUNT))
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
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done

emit_configuration() {
    printf 'model=%s\nprompt=%s\ngpu_devices=%s\ngpu_vram=%s\n' \
        "$MODEL" "$PROMPT" "$GPU_DEVICES" "$GPU_VRAM"
    printf 'stage_split=%s\ncontexts=512,4096,32768\nrepeats=%s\n' \
        "$STAGE_SPLIT" "$REPEATS"
    printf 'tg_tokens=%s\nexact_tokens=%s\nwarmup_tokens=%s\n' \
        "$TG_TOKENS" "$EXACT_TOKENS" "$WARMUP_TOKENS"
    printf 'prefill_chunk=%s\npipeline_mb=%s\n' "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'comparison=q3a4-k4-prefetch0-vs-prefetch2\n'
    printf 'production_ctx_max=%s\nproduction_ctx_alloc=%s\nvocab_size=%s\nlogits_bytes=%s\n' \
        "$PRODUCTION_CTX_MAX" "$PRODUCTION_CTX_ALLOC" "$VOCAB_SIZE" "$LOGITS_BYTES"
    printf 'q3a4_decode_mapping=tile32-dp4a\nq3a4_decode_ksplit=4\n'
    printf 'q4_gate_up_decode_mapping=tile32-mma\n'
    printf 'q4_down_decode_mapping=tile32\n'
    printf 'q3a4_layout=%s\nq3a4_layer_count=%s\nq3a4_layers=%s\n' \
        "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST"
}

if [[ $RESUME == 1 ]]; then
    [[ -n ${Q3A4_PREFETCH2_PRODUCTION_AB_DIR:-} && -d $OUTPUT_DIR ]] ||
        die "RESUME=1 requires an existing Q3A4_PREFETCH2_PRODUCTION_AB_DIR"
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
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'state=failed\nexit_status=1\nlast_phase=%s\n' \
                "$phase" >"$OUTPUT_DIR/run-status.txt"
            printf 'error: could not create %s\n' "$archive" >&2
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
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=tile32-mma
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=tile32
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT=1
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
    grep -Fq 'SM75 Q3A4 tile32-dp4a K4 prefetch-depth 1/2 nonzero exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 prefetch exact marker missing"
    grep -Fq 'SM75 Q3A4 tile32-dp4a-k4 production default' \
        "$OUTPUT_DIR/smoke.log" || die "ordinary K4 production-default marker missing"
    grep -Fq 'SM75 Q4-32 tile32-mma gate/up + tile32 down production defaults' \
        "$OUTPUT_DIR/smoke.log" || die "Q4 production-default marker missing"
    grep -Fq 'SM75 Q3A4 K4 prefetch depth 0/2 environment selector exact' \
        "$OUTPUT_DIR/smoke.log" || die "Q3A4 prefetch environment selector marker missing"
    grep -Fxq 'cuda long-context regression: OK' "$OUTPUT_DIR/smoke.log" ||
        die "CUDA regression completion marker missing"
else
    make -q ds4-bench tests/cuda_long_context_smoke CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

emit_runtime_identity() {
    local canonical_model
    canonical_model=$(python3 -c \
        'import os,sys; print(os.path.realpath(sys.argv[1]))' "$MODEL")
    printf 'model_path=%s\nmodel_bytes=%s\nmodel_mtime=%s\nmodel_inode=%s\n' \
        "$canonical_model" "$(stat -c %s "$MODEL")" "$(stat -c %Y "$MODEL")" \
        "$(stat -c %i "$MODEL")"
    printf 'model_hashing=disabled\nprompt_sha256=%s\nds4_cuda_sha256=%s\n' \
        "$(sha256sum "$PROMPT" | awk '{print $1}')" \
        "$(sha256sum ds4_cuda.cu | awk '{print $1}')"
    printf 'sm75_q32_include_sha256=%s\nds4_bench_sha256=%s\n' \
        "$(sha256sum ds4_cuda_sm75_q32_native.inc.cu | awk '{print $1}')" \
        "$(sha256sum ./ds4-bench | awk '{print $1}')"
}

phase=runtime-identity
if [[ $RESUME == 1 ]]; then
    [[ -s $OUTPUT_DIR/runtime-identity.txt ]] ||
        die "resume runtime identity is missing"
    cmp -s <(emit_runtime_identity) "$OUTPUT_DIR/runtime-identity.txt" ||
        die "resume model metadata, prompt, or ds4-bench identity changed"
else
    emit_runtime_identity >"$OUTPUT_DIR/runtime-identity.txt"
fi

phase=manifest
if [[ $RESUME == 0 ]]; then
    {
        printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$(git branch --show-current)"
        printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\n' \
            "$MODEL" "$(stat -c %s "$MODEL")"
        emit_configuration
        printf '\n[gpu inventory]\n'
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,compute_cap \
            --format=csv
        printf '\n[topology]\n'
        nvidia-smi topo -m
    } >"$OUTPUT_DIR/manifest.txt"
    git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
    git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
    env | awk -F= '$1 ~ /^DS4_/ {print}' | sort -u \
        >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"
fi

variant_depth() {
    case "$1" in
        control) printf '0\n' ;;
        prefetch2) printf '2\n' ;;
        *) return 1 ;;
    esac
}

variant_env() {
    local depth
    depth=$(variant_depth "$1") || return 1
    printf 'DS4_CUDA_MOE_Q3A4_DECODE_MAPPING=tile32-dp4a\n'
    printf 'DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT=4\n'
    printf 'DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH=%s\n' "$depth"
}

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
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

validate_common_log() {
    local log=$1 marker route
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
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Fq 'SM75 Q32 owned decode CUDA Graph enabled' "$log" || return 1
    ! grep -Fq 'SM75 Q32 decode graph audit' "$log" || return 1
    grep -Fxq 'ds4: SM75 Q4-32 decode gate/up mapping=tile32-mma (production default)' \
        "$log" || return 1
    grep -Fxq 'ds4: SM75 Q4-32 down decode mapping=tile32-int4 (production default)' \
        "$log" || return 1
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
            } else if (gate!="sm75_q4_32" || up!="sm75_q4_32" ||
                       down!="sm75_q4_32") bad=1
        }
        END {
            for (i=0; i<43; i++) if (layer_seen[i]!=1) bad=1
            exit !(seen==43 && q3==expected_count &&
                   layers==expected_layers && !bad)
        }
    ' "$log"
}

q4_gate_active_count() {
    local log=$1
    awk '
        /SM75 Q4-32 decode mapping audit/ {
            seen++
            c=h=d=m=p0=p1=p2=-1
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control") c=a[2]+0
                if (a[1]=="hwarp16") h=a[2]+0
                if (a[1]=="tile32-dp4a") d=a[2]+0
                if (a[1]=="tile32-mma") m=a[2]+0
                if (a[1]=="pf0") p0=a[2]+0
                if (a[1]=="pf1") p1=a[2]+0
                if (a[1]=="pf2") p2=a[2]+0
            }
            good=(c==0 && h==0 && d==0 && m>=0 &&
                  p0==m && p1==0 && p2==0)
        }
        END {
            if (seen!=1 || !good) exit 1
            print m
        }
    ' "$log"
}

q4_down_active_count() {
    local log=$1
    awk '
        /SM75 Q4-32 down decode mapping audit/ {
            seen++
            c=t=cs=cp=ts=tp=p0=p1=p2=-1
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="control") c=a[2]+0
                if (a[1]=="tile32") t=a[2]+0
                if (a[1]=="control-slots") cs=a[2]+0
                if (a[1]=="control-packed") cp=a[2]+0
                if (a[1]=="tile32-slots") ts=a[2]+0
                if (a[1]=="tile32-packed") tp=a[2]+0
                if (a[1]=="pf0") p0=a[2]+0
                if (a[1]=="pf1") p1=a[2]+0
                if (a[1]=="pf2") p2=a[2]+0
            }
            good=(c==0 && t>0 && cs==0 && cp==0 && ts>0 && tp>0 &&
                  t==ts+tp && p0==t && p1==0 && p2==0)
        }
        END {
            if (seen!=1 || !good) exit 1
            print t
        }
    ' "$log"
}

mapping_active_count() {
    local variant=$1 log=$2
    awk -v variant="$variant" '
        /SM75 Q3A4 decode mapping audit/ {
            seen++
            delete v
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                v[a[1]]=a[2]+0
            }
            total=v["tile32-dp4a"]
            good=(v["control"]==0 && v["hwarp16"]==0 &&
                  v["tile32"]==0 && total>0 && v["k1"]==0 &&
                  v["k2"]==0 && v["k4"]==total &&
                  v["pf0"]+v["pf1"]+v["pf2"]==total)
            if (variant=="control")
                good=good && v["pf0"]==total && v["pf1"]==0 && v["pf2"]==0
            else
                good=good && v["pf0"]==0 && v["pf1"]==0 && v["pf2"]==total
            active=total
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
        grep -Fxq 'ds4: SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4 (production default)' \
            "$log" || return 1
    else
        grep -Fxq 'ds4: SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4-prefetch2' \
            "$log" || return 1
    fi
    mapping_active_count "$variant" "$log" >/dev/null
}

validate_q8_state_files() {
    local base=$1 plan="$1.q8-plan.csv" bindings="$1.q8-bindings.csv"
    [[ -s $plan && -s $bindings ]] || return 1
    [[ $(wc -l <"$plan") == 345 && $(wc -l <"$bindings") == 345 ]] ||
        return 1
    awk -F, '
        NR==1 {
            header=($1=="sequence" && $2=="label" &&
                    $3=="consumer_device" && $7=="resident_device" &&
                    $13=="status")
            next
        }
        {
            rows++
            if (($7+0)<0 || ($13!="home" && $13!="partner")) bad=1
        }
        END {exit !(header && rows==344 && !bad)}
    ' "$plan" || return 1
    awk -F, '
        NR==1 {
            header=($1=="consumer_device" && $2=="resident_device" &&
                    $3=="partner_offload" && $12=="label" &&
                    $15=="live")
            next
        }
        {
            rows++
            if (($13+0)<=0 || ($14+0)<=0 || $15!=1) bad=1
        }
        END {exit !(header && rows==344 && !bad)}
    ' "$bindings"
}

canonical_q8_bindings() {
    awk -F, 'BEGIN {OFS=","} NR>1 {
        print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12
    }' "$1" | sort
}

validate_q8_plan_equal() {
    local left_log=$1 right_log=$2 marker left_line right_line
    local left_base=${left_log%.log} right_base=${right_log%.log}
    for marker in 'CUDA q8 fp16 benefit plan candidates=' \
                  'CUDA q8 fp16 stage-aware 22/21 planner selected ' \
                  'CUDA q8 fp16 benefit plan materialized '; do
        [[ $(grep -Fc "$marker" "$left_log") == 1 &&
           $(grep -Fc "$marker" "$right_log") == 1 ]] || return 1
        left_line=$(grep -F "$marker" "$left_log") || return 1
        right_line=$(grep -F "$marker" "$right_log") || return 1
        [[ $left_line == "$right_line" ]] || return 1
    done
    validate_q8_state_files "$left_base" &&
        validate_q8_state_files "$right_base" || return 1
    cmp -s "$left_base.q8-plan.csv" "$right_base.q8-plan.csv" || return 1
    cmp -s <(canonical_q8_bindings "$left_base.q8-bindings.csv") \
           <(canonical_q8_bindings "$right_base.q8-bindings.csv")
}

validate_throughput_csv() {
    local csv=$1
    awk -F, -v tg="$TG_TOKENS" '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; ok1=($1==512 && $4==tg && ($8+0)>0); next}
        NR==3 {rows++; ok2=($1==4096 && $4==tg && ($8+0)>0); next}
        NR==4 {rows++; ok3=($1==32768 && $4==tg && ($8+0)>0); next}
        NR>4 {rows++}
        END {exit !(header && rows==3 && ok1 && ok2 && ok3)}
    ' "$csv"
}

validate_exact_csv() {
    local csv=$1 context=$2
    awk -F, -v ctx="$context" -v tg="$EXACT_TOKENS" '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; ok=($1==ctx && $4==tg && ($8+0)>0); next}
        NR>2 {rows++}
        END {exit !(header && rows==1 && ok)}
    ' "$csv"
}

frontier_complete() {
    local dir=$1 context=$2 token file
    local -a files=()
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf -v file 'frontier_%06d.decode_%06d.logits.f32' "$context" "$token"
        [[ -s $dir/$file && $(stat -c %s "$dir/$file") == "$LOGITS_BYTES" ]] ||
            return 1
        files+=("$dir/$file")
    done
    [[ $(find "$dir" -maxdepth 1 -type f -name '*.f32' | wc -l) == \
       "$EXACT_TOKENS" ]] || return 1
    python3 - "${files[@]}" <<'VALIDATE_LOGITS_PY'
import array
import math
import pathlib
import sys

for raw_path in sys.argv[1:]:
    values = array.array("f")
    with pathlib.Path(raw_path).open("rb") as handle:
        values.fromfile(handle, pathlib.Path(raw_path).stat().st_size // 4)
    if not values or not all(math.isfinite(value) for value in values):
        raise SystemExit(f"invalid non-finite logits: {raw_path}")
    if not any(value != 0.0 for value in values):
        raise SystemExit(f"invalid all-zero logits: {raw_path}")
VALIDATE_LOGITS_PY
}

run_engine() {
    local variant=$1 tokens=$2 start=$3 max=$4 base=$5 logits=${6:-}
    local -a selector cmd state=(
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv"
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv"
    )
    mapfile -t selector < <(variant_env "$variant")
    local ctx_alloc=$PRODUCTION_CTX_ALLOC rc=0
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    cmd=("${production_env[@]}" "${selector[@]}" "${state[@]}" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$start" --ctx-max "$max" --ctx-alloc "$ctx_alloc" \
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$tokens")
    [[ -z $logits ]] || cmd+=(--dump-decode-logits-dir "$logits")
    cmd+=(--csv "$base.csv")
    "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    capture_gpu_health "$base.post-gpu.csv" || return 1
    return "$rc"
}

validate_throughput_run() {
    local variant=$1 base=$2
    validate_gpu_health_pair "$base" && validate_q8_state_files "$base" &&
        validate_throughput_csv "$base.csv" && validate_common_log "$base.log" &&
        validate_mapping_audit "$variant" "$base.log"
}

validate_exact_run() {
    local variant=$1 context=$2 base=$3 logits=$4 expected_calls
    expected_calls=$((EXACT_TOKENS * Q3A4_LAYER_COUNT * 2))
    validate_gpu_health_pair "$base" && validate_q8_state_files "$base" &&
        validate_exact_csv "$base.csv" "$context" &&
        validate_common_log "$base.log" &&
        validate_mapping_audit "$variant" "$base.log" &&
        [[ $(mapping_active_count "$variant" "$base.log") == "$expected_calls" ]] &&
        [[ $(q4_gate_active_count "$base.log") == \
           $((EXACT_TOKENS * Q4_GATE_LAYER_COUNT * 2)) ]] &&
        [[ $(q4_down_active_count "$base.log") == \
           $((EXACT_TOKENS * Q4_DOWN_LAYER_COUNT * 2)) ]] &&
        frontier_complete "$logits" "$context"
}

phase=throughput
runs_partial="$OUTPUT_DIR/runs/runs.tsv.partial.$$"
dispatch_partial="$OUTPUT_DIR/runs/dispatch.tsv.partial.$$"
printf 'repeat\tslot\tcontext\tvariant\tsteady_tps\tfirst_ms\tcsv\tlog\n' >"$runs_partial"
printf 'repeat\tcontrol_owned_calls\tprefetch2_owned_calls\tq8_plan_equal\n' \
    >"$dispatch_partial"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then variants=(control prefetch2); else variants=(prefetch2 control); fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        base="$OUTPUT_DIR/runs/r${repeat}-${variant}"
        if [[ $RESUME == 1 && -s $base.csv && -s $base.log ]] &&
           validate_throughput_run "$variant" "$base"; then
            printf 'Reusing Q3A4 prefetch2 production A/B repeat=%d/%d variant=%s...\n' \
                "$repeat" "$REPEATS" "$variant"
        else
            printf 'Q3A4 prefetch2 production A/B repeat=%d/%d slot=%d/2 variant=%s...\n' \
                "$repeat" "$REPEATS" "$slot" "$variant"
            run_engine "$variant" "$TG_TOKENS" 512 32768 "$base" || {
                tail -n 200 "$base.log" >&2 || true
                die "$variant production run failed"
            }
            validate_throughput_run "$variant" "$base" ||
                die "$variant production run failed validation"
        fi
        while IFS=, read -r context _ _ _ _ first _ steady _; do
            [[ $context == ctx_tokens ]] && continue
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$repeat" "$slot" "$context" "$variant" "$steady" "$first" \
                "$base.csv" "$base.log" >>"$runs_partial"
        done <"$base.csv"
    done
    control_log="$OUTPUT_DIR/runs/r${repeat}-control.log"
    candidate_log="$OUTPUT_DIR/runs/r${repeat}-prefetch2.log"
    validate_q8_plan_equal "$control_log" "$candidate_log" ||
        die "repeat $repeat changed dense-Q8 placement between arms"
    validate_q8_plan_equal "$OUTPUT_DIR/runs/r1-control.log" "$control_log" ||
        die "repeat $repeat changed the fixed production dense-Q8 plan"
    control_calls=$(mapping_active_count control "$control_log")
    candidate_calls=$(mapping_active_count prefetch2 "$candidate_log")
    expected_calls=$((3 * TG_TOKENS * Q3A4_LAYER_COUNT * 2))
    expected_q4_gate=$((3 * TG_TOKENS * Q4_GATE_LAYER_COUNT * 2))
    expected_q4_down=$((3 * TG_TOKENS * Q4_DOWN_LAYER_COUNT * 2))
    [[ $control_calls == "$expected_calls" &&
       $candidate_calls == "$expected_calls" ]] ||
        die "repeat $repeat has unexpected Q3A4 inventory (expected $expected_calls; control $control_calls; prefetch2 $candidate_calls)"
    for log in "$control_log" "$candidate_log"; do
        [[ $(q4_gate_active_count "$log") == "$expected_q4_gate" &&
           $(q4_down_active_count "$log") == "$expected_q4_down" ]] ||
            die "repeat $repeat changed the fixed production Q4 dispatch inventory"
    done
    printf '%s\t%s\t%s\ttrue\n' "$repeat" "$control_calls" \
        "$candidate_calls" >>"$dispatch_partial"
done
mv -- "$runs_partial" "$OUTPUT_DIR/runs/runs.tsv"
mv -- "$dispatch_partial" "$OUTPUT_DIR/runs/dispatch.tsv"

phase=throughput-summary
python3 - "$OUTPUT_DIR/runs/runs.tsv" "$OUTPUT_DIR/summary/summary.csv" \
    "$OUTPUT_DIR/summary/summary.md" "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" \
    "$Q3A4_LAYER_LIST" <<'PY'
import csv
import pathlib
import statistics
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
pairs = {}
for row in rows:
    key = (int(row["repeat"]), int(row["context"]))
    pairs.setdefault(key, {})[row["variant"]] = {
        "tps": float(row["steady_tps"]), "first": float(row["first_ms"])}
groups = {}
for (repeat, context), values in pairs.items():
    if set(values) != {"control", "prefetch2"}:
        raise SystemExit(f"unpaired sample repeat={repeat} context={context}")
    groups.setdefault(context, []).append((values["control"], values["prefetch2"]))
records = []
for context in sorted(groups):
    values = groups[context]
    c = [x[0]["tps"] for x in values]
    p = [x[1]["tps"] for x in values]
    cf = [x[0]["first"] for x in values]
    pf = [x[1]["first"] for x in values]
    ratios = [b / a for a, b in zip(c, p)]
    records.append((context, statistics.median(c), statistics.median(p),
                    statistics.median(cf), statistics.median(pf),
                    statistics.median(ratios), statistics.stdev(ratios),
                    len(ratios)))
with pathlib.Path(sys.argv[2]).open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["q3a4_layout", "q3a4_layer_count", "context",
                  "control_tps", "prefetch2_tps", "control_ms_per_token",
                  "prefetch2_ms_per_token", "control_first_ms",
                  "prefetch2_first_ms", "paired_median_speedup",
                  "change_pct", "paired_speedup_sd", "samples"])
    for context, c, p, cf, pf, speedup, sd, samples in records:
        out.writerow([sys.argv[4], sys.argv[5], context, f"{c:.6f}",
                      f"{p:.6f}", f"{1000/c:.6f}", f"{1000/p:.6f}",
                      f"{cf:.6f}", f"{pf:.6f}", f"{speedup:.9f}",
                      f"{(speedup-1)*100:.6f}", f"{sd:.9f}", samples])
lines = [
    "# SM75 production Q3A4 K4 prefetch-depth2 decode A/B", "",
    "Only Q3A4 K4 gate/up software-prefetch depth changes. Q4-32 and every "
    "cross-GPU boundary are identical. Exact output is validated separately.",
    f"Q3A4 layout: `{sys.argv[4]}` ({sys.argv[5]} routed layers: {sys.argv[6]}).",
    "", "| Context | Control tok/s | Prefetch2 tok/s | Control ms/tok | "
    "Prefetch2 ms/tok | Control first ms | Prefetch2 first ms | Paired speedup | Change | SD |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for context, c, p, cf, pf, speedup, sd, _ in records:
    lines.append(f"| {context} | {c:.3f} | {p:.3f} | {1000/c:.3f} | "
                 f"{1000/p:.3f} | {cf:.3f} | {pf:.3f} | {speedup:.6f}x | "
                 f"{(speedup-1)*100:+.3f}% | {sd:.6f} |")
pathlib.Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
PY
cat "$OUTPUT_DIR/summary/summary.md"

phase=exact-logits
exact_dispatch_partial="$OUTPUT_DIR/exact/dispatch.tsv.partial.$$"
printf 'context\tcontrol_owned_calls\tprefetch2_owned_calls\tq8_plan_equal\n' \
    >"$exact_dispatch_partial"
for context in "${contexts[@]}"; do
    for variant in control prefetch2; do
        base="$OUTPUT_DIR/exact/${variant}-pp${context}"
        logits="$base-logits"
        if [[ $RESUME == 1 && -s $base.csv && -s $base.log && -d $logits ]] &&
           validate_exact_run "$variant" "$context" "$base" "$logits"; then
            printf 'Reusing exact Q3A4 prefetch2 logits: %s PP=%s...\n' \
                "$variant" "$context"
        else
            mkdir -p "$logits"
            find "$logits" -maxdepth 1 -type f -name '*.f32' -delete
            printf 'Exact Q3A4 prefetch2 logits: %s PP=%s...\n' "$variant" "$context"
            run_engine "$variant" "$EXACT_TOKENS" "$context" "$context" \
                "$base" "$logits" || {
                tail -n 200 "$base.log" >&2 || true
                die "$variant PP=$context exact run failed"
            }
            validate_exact_run "$variant" "$context" "$base" "$logits" ||
                die "$variant PP=$context exact run failed validation"
        fi
    done
    validate_q8_plan_equal "$OUTPUT_DIR/exact/control-pp${context}.log" \
        "$OUTPUT_DIR/exact/prefetch2-pp${context}.log" ||
        die "PP=$context exact arms changed dense-Q8 placement"
    control_calls=$(mapping_active_count control \
        "$OUTPUT_DIR/exact/control-pp${context}.log")
    candidate_calls=$(mapping_active_count prefetch2 \
        "$OUTPUT_DIR/exact/prefetch2-pp${context}.log")
    printf '%s\t%s\t%s\ttrue\n' "$context" "$control_calls" \
        "$candidate_calls" >>"$exact_dispatch_partial"
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf -v file 'frontier_%06d.decode_%06d.logits.f32' "$context" "$token"
        cmp -s "$OUTPUT_DIR/exact/control-pp${context}-logits/$file" \
               "$OUTPUT_DIR/exact/prefetch2-pp${context}-logits/$file" ||
            die "prefetch2 diverged at $file"
    done
done
mv -- "$exact_dispatch_partial" "$OUTPUT_DIR/exact/dispatch.tsv"

exact_plan_reference="$OUTPUT_DIR/runs/r1-control.log"
for context in "${contexts[@]}"; do
    for variant in control prefetch2; do
        validate_q8_plan_equal "$exact_plan_reference" \
            "$OUTPUT_DIR/exact/${variant}-pp${context}.log" ||
            die "exact PP=$context $variant changed the fixed production dense-Q8 plan"
    done
done

printf 'bit_exact=true\nfrontiers=512,4096,32768\ndecode_tokens_per_frontier=%s\n' \
    "$EXACT_TOKENS" >"$OUTPUT_DIR/exact/verification.txt"
printf 'q3a4_layout=%s\nq3a4_layer_count=%s\nq3a4_layers=%s\n' \
    "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST" \
    >>"$OUTPUT_DIR/exact/verification.txt"
printf 'control_prefetch_depth=0\ncandidate_prefetch_depth=2\nq8_plan_equal=true\n' \
    >>"$OUTPUT_DIR/exact/verification.txt"
{
    printf '\n## Exact-output verification\n\n'
    printf 'All %s decode logits at PP512, PP4096, and PP32768 were byte-identical.\n\n' \
        "$EXACT_TOKENS"
    printf 'Q3A4 layout: `%s` (%s routed layers: %s).\n' \
        "$Q3A4_LAYOUT" "$Q3A4_LAYER_COUNT" "$Q3A4_LAYER_LIST"
} >>"$OUTPUT_DIR/summary/summary.md"

phase=complete
