#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
if [[ -z ${MTP_MODEL:-} ]]; then
    if [[ -f $repo_dir/gguf/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf ]]; then
        MTP_MODEL=$repo_dir/gguf/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
    else
        MTP_MODEL=$repo_dir/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
    fi
fi
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
MODEL_LAYOUT=${MODEL_LAYOUT:-both}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
TG_TOKENS=${TG_TOKENS:-256}
MTP_MARGIN=${MTP_MARGIN:-3}
REPEATS=${REPEATS:-3}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_MAX=32768
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${MTP_PRODUCTION_AB_DIR:-$repo_dir/sm75-mtp-production-ab-$stamp}

case "$MODEL_LAYOUT" in
    mixed15) layouts=(mixed15); models=("$MIXED_MODEL") ;;
    all43) layouts=(all43); models=("$ALL43_MODEL") ;;
    both) layouts=(mixed15 all43); models=("$MIXED_MODEL" "$ALL43_MODEL") ;;
    *) die "MODEL_LAYOUT must be mixed15, all43, or both" ;;
esac

for model in "${models[@]}"; do
    [[ $model == /* && -f $model ]] || die "model not found at absolute path: $model"
done
[[ $MTP_MODEL == /* && -f $MTP_MODEL ]] ||
    die "MTP support model not found at absolute path: $MTP_MODEL"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
[[ $REQUIRED_POWER_LIMITS_W == 250,260,250,250 ]] ||
    die "production qualification requires physical GPU power limits 250,260,250,250 W"
[[ $TG_TOKENS == 256 ]] || die "production qualification requires TG_TOKENS=256"
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer"
(( REPEATS == 1 || (REPEATS >= 3 && REPEATS % 3 == 0) )) ||
    die "REPEATS must be 1 for a preliminary smoke or a positive multiple of 3 for balanced qualification"
CTX_ALLOC=$((CTX_MAX + TG_TOKENS + 1))
awk -v value="$MTP_MARGIN" '
    BEGIN {exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value+0 >= 0 && value+0 <= 1000)}
' || die "MTP_MARGIN must be between 0 and 1000"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk basename cmp date dirname env git grep kill make mkdir mv nproc \
            nvidia-smi sha256sum sleep sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU and power-limit lists must each contain four entries"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
(( gpu_count == 4 )) || die "production qualification requires exactly four visible physical GPUs"
declare -A seen_gpu=()
for ordinal in 0 1 2 3; do
    selected=${gpu_ids[$ordinal]}
    [[ $selected =~ ^[0-3]$ && -z ${seen_gpu[$selected]+x} ]] ||
        die "invalid or duplicate CUDA ordinal: $selected"
    seen_gpu[$selected]=1
    cap=$(nvidia-smi -i "$ordinal" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "physical GPU $ordinal is not SM75"
    limit=$(nvidia-smi -i "$ordinal" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$ordinal]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "physical GPU $ordinal power limit is ${limit:-unknown} W, expected ${required_power[$ordinal]} W"
done

git diff --quiet -- ||
    die "production qualification requires no tracked worktree changes"
git diff --cached --quiet -- ||
    die "production qualification requires no staged changes"
git ls-files --error-unmatch -- speed-bench/cuda-sm75-mtp-production-ab.sh \
    >/dev/null 2>&1 || die "production harness is not tracked by git"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

nvidia-smi \
    --query-gpu=index,pci.bus_id,uuid,power.limit,compute_cap \
    --format=csv,noheader,nounits >"$OUTPUT_DIR/provenance/nvml-inventory.csv"
awk -F, 'BEGIN {OFS="\t"}
    {
        for (i=1; i<=NF; i++) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
        }
        print $1,tolower($2),$3,$4,$5
    }
' "$OUTPUT_DIR/provenance/nvml-inventory.csv" |
    sort -t $'\t' -k1,1n >"$OUTPUT_DIR/provenance/expected-nvml-inventory.tsv"
[[ $(wc -l <"$OUTPUT_DIR/provenance/expected-nvml-inventory.tsv") == 4 ]] ||
    die "could not capture the complete four-GPU NVML inventory"
awk -F'\t' 'BEGIN {OFS="\t"} {print $2,$3,$1}' \
    "$OUTPUT_DIR/provenance/expected-nvml-inventory.tsv" |
    sort -t $'\t' -k1,1 |
    awk -F'\t' 'BEGIN {OFS="\t"} {print NR-1,$1,$2,$3}' \
        >"$OUTPUT_DIR/provenance/expected-cuda-ordinals.tsv"

phase=initialization
telemetry_pid=

telemetry_process_running() {
    local pid=$1 state
    [[ -r /proc/$pid/stat ]] || return 1
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
    [[ -n $state && $state != Z ]]
}

stop_telemetry() {
    local pid=$1 state attempt
    kill "$pid" 2>/dev/null || true
    for attempt in {1..20}; do
        if [[ ! -r /proc/$pid/stat ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
        if [[ $state == Z ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 0.1
    done
    # A lost GPU can leave nvidia-smi blocked in an uninterruptible ioctl.
    # Do not let cleanup hang the evidence process indefinitely.
    kill -KILL "$pid" 2>/dev/null || true
    for attempt in {1..20}; do
        if [[ ! -r /proc/$pid/stat ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
        if [[ $state == Z ]]; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 0.1
    done
    return 1
}

finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    if [[ -n $telemetry_pid ]]; then
        stop_telemetry "$telemetry_pid" || true
    fi
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'error: could not create %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_runtime < <(
    env | awk -F= '$1 ~ /^(DS4_|CUDA_)/ {print $1}' | sort -u
)
clean=(env)
for name in "${inherited_runtime[@]}"; do clean+=(-u "$name"); done
clean+=(CUDA_DEVICE_ORDER=PCI_BUS_ID)
printf '%s\n' "${inherited_runtime[@]:-}" \
    >"$OUTPUT_DIR/provenance/cleared-runtime-env.txt"

# This is the accepted stable four-GPU prefill policy: pair 0 attention rows
# stay local, while indexer rows remain split on both pairs.
production_env=(
    "${clean[@]}"
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=0,1
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

# Per-dispatch audit logging and atomic counters are excluded from every clean
# timing arm. They run once in the separate diagnostic arm, whose measurements
# never enter the speed table.
diagnostic_env=(
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_AUDIT=1
    DS4_CUDA_TP_DECODE_INDEXER_ROWS_AUDIT=1
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1
    DS4_CUDA_COMPRESSOR_PAIR_STATE_STORE_AUDIT=1
)

phase=build
[[ $SKIP_BUILD == 0 ]] ||
    die "SKIP_BUILD=1 is not accepted for production qualification"
make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
    CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/smoke.log" 2>&1 || {
    tail -n 200 "$OUTPUT_DIR/smoke.log" >&2 || true
    die "byte-exact CUDA regression failed"
}
grep -Fxq 'cuda long-context regression: OK' "$OUTPUT_DIR/smoke.log" ||
    die "CUDA regression completion marker missing"
grep -Fq 'SM75 Q3A4 tile32-dp4a-k4-prefetch2 production default' \
    "$OUTPUT_DIR/smoke.log" || die "Q3A4 production-default marker missing"
grep -Fq 'SM75 Q4-32 tile32-mma gate/up + tile32 down production defaults' \
    "$OUTPUT_DIR/smoke.log" || die "Q4 production-default marker missing"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model_layout=%s\nmixed_model=%s\nall43_model=%s\nmtp_model=%s\nprompt=%s\n' \
        "$MODEL_LAYOUT" "$MIXED_MODEL" "$ALL43_MODEL" "$MTP_MODEL" "$PROMPT"
    for index in "${!layouts[@]}"; do
        printf '%s_model_bytes=' "${layouts[$index]}"
        stat -c %s "${models[$index]}"
    done
    printf 'mtp_model_bytes=%s\nmtp_model_mtime=%s\n' \
        "$(stat -c %s "$MTP_MODEL")" "$(stat -c %Y "$MTP_MODEL")"
    printf 'prompt_sha256=%s\nds4_bench_sha256=%s\n' \
        "$(sha256sum "$PROMPT" | awk '{print $1}')" \
        "$(sha256sum ./ds4-bench | awk '{print $1}')"
    printf 'gpu_devices=%s\npower_limits_w=%s\nstage_split=22/21\n' \
        "$GPU_DEVICES" "$REQUIRED_POWER_LIMITS_W"
    printf 'cuda_device_order=PCI_BUS_ID\n'
    printf 'contexts=512,4096,32768\ntg_tokens=%s\nmtp_margin=%s\nrepeats=%s\n' \
        "$TG_TOKENS" "$MTP_MARGIN" "$REPEATS"
    if (( REPEATS == 1 )); then
        printf 'evidence_class=preliminary-smoke\n'
    else
        printf 'evidence_class=balanced-production-qualification\n'
    fi
    printf 'sampling=greedy-argmax\nvariants=plain,resident,mtp2-clean\nmtp2_policy=DS4_MTP_STRICT\ndiagnostic=separate-mtp2-diag\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short --untracked-files=no \
    >"$OUTPUT_DIR/provenance/tracked-git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"; return 1;
            }
    done
    mv -- "$partial" "$output"
}

validate_gpu_snapshot() {
    local snapshot=$1
    awk -F, -v expected_file="$OUTPUT_DIR/provenance/expected-nvml-inventory.tsv" \
        -v required="$REQUIRED_POWER_LIMITS_W" '
        BEGIN {
            split(required, required_power, ",")
            while ((getline line < expected_file) > 0) {
                split(line, row, "\t")
                expected_bus[row[1]]=tolower(row[2])
                expected_uuid[row[1]]=row[3]
                expected_count++
            }
            close(expected_file)
        }
        {
            for (i=1; i<=NF; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            }
            idx=$1
            if (idx !~ /^[0-3]$/ || seen[idx]++ ||
                tolower($2)!=expected_bus[idx] || $3!=expected_uuid[idx] ||
                ($4+0)!=(required_power[idx+1]+0)) bad=1
        }
        END {
            if (expected_count!=4) bad=1
            for (i=0; i<4; i++) if (seen[i]!=1) bad=1
            exit bad
        }
    ' "$snapshot"
}

validate_health() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] &&
        ! grep -Eq 'ERR!|Unknown Error|GPU is lost' \
            "$base.pre-gpu.csv" "$base.post-gpu.csv" &&
        validate_gpu_snapshot "$base.pre-gpu.csv" &&
        validate_gpu_snapshot "$base.post-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

validate_csv() {
    local csv=$1
    awk -F, -v tg="$TG_TOKENS" '
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                       $4=="gen_tokens" && $5=="gen_tps" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; a=($1==512 && $4==tg && ($3+0)>0 && ($5+0)>0 && ($8+0)>0); next}
        NR==3 {rows++; b=($1==4096 && $4==tg && ($3+0)>0 && ($5+0)>0 && ($8+0)>0); next}
        NR==4 {rows++; c=($1==32768 && $4==tg && ($3+0)>0 && ($5+0)>0 && ($8+0)>0); next}
        NR>4 {rows++}
        END {exit !(header && rows==3 && a && b && c)}
    ' "$csv"
}

validate_tokens() {
    local dir=$1 context file bytes
    bytes=$(((TG_TOKENS + 1) * 4))
    for context in 512 4096 32768; do
        printf -v file 'frontier_%06d.tokens.i32le' "$context"
        [[ -f $dir/$file && $(stat -c %s "$dir/$file") == "$bytes" ]] || return 1
    done
}

validate_cuda_inventory() {
    local log=$1
    awk -v expected_file="$OUTPUT_DIR/provenance/expected-cuda-ordinals.tsv" \
        -v selected_ordinals="$GPU_DEVICES" '
        function bdf(bus) {
            bus=tolower(bus)
            sub(/^[^:]+:/, "", bus)
            return bus
        }
        BEGIN {
            while ((getline line < expected_file) > 0) {
                split(line, row, "\t")
                expected_bus[row[1]]=bdf(row[2])
                expected_uuid[row[1]]=row[3]
                expected_count++
            }
            close(expected_file)
            split(selected_ordinals, selected, ",")
        }
        /CUDA ordinal inventory ordinal=/ {
            ordinal=bus=uuid=""
            for (i=1; i<=NF; i++) {
                split($i, value, "=")
                if (value[1]=="ordinal") ordinal=value[2]
                else if (value[1]=="pci_bus_id") bus=bdf(value[2])
                else if (value[1]=="uuid") uuid=value[2]
            }
            if (ordinal !~ /^[0-3]$/ || ordinal_seen[ordinal]++ ||
                bus!=expected_bus[ordinal] || uuid!=expected_uuid[ordinal]) bad=1
        }
        /CUDA selected device identity logical_tier=/ {
            tier=ordinal=bus=uuid=""
            for (i=1; i<=NF; i++) {
                split($i, value, "=")
                if (value[1]=="logical_tier") tier=value[2]
                else if (value[1]=="cuda_ordinal") ordinal=value[2]
                else if (value[1]=="pci_bus_id") bus=bdf(value[2])
                else if (value[1]=="uuid") uuid=value[2]
            }
            if (tier !~ /^[0-3]$/ || selected_seen[tier]++ ||
                ordinal!=selected[tier+1] || bus!=expected_bus[ordinal] ||
                uuid!=expected_uuid[ordinal]) bad=1
        }
        END {
            if (expected_count!=4) bad=1
            for (i=0; i<4; i++) {
                if (ordinal_seen[i]!=1 || selected_seen[i]!=1) bad=1
            }
            exit bad
        }
    ' "$log"
}

validate_routed_layout() {
    local layout=$1 log=$2 expected_count expected_layers
    if [[ $layout == mixed15 ]]; then
        expected_count=15
        expected_layers=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
    else
        expected_count=43
        expected_layers=all43
    fi
    awk -v expected_count="$expected_count" -v expected_layers="$expected_layers" '
        /routed-quant-audit/ {
            seen++
            layer=gate=up=down=""
            for (i=1; i<=NF; i++) {
                split($i, a, "=")
                if (a[1]=="layer") layer=a[2]
                else if (a[1]=="gate") gate=a[2]
                else if (a[1]=="up") up=a[2]
                else if (a[1]=="down") down=a[2]
            }
            if (layer !~ /^[0-9]+$/ || layer<0 || layer>42 || layer_seen[layer]++) bad=1
            if (gate=="sm75_q3a4") {
                if (up!="sm75_q3a4" || down!="sm75_q4_32") bad=1
                q3++
                q3_layers=q3_layers (q3_layers ? "," : "") layer
            } else if (gate!="sm75_q4_32" || up!="sm75_q4_32" ||
                       down!="sm75_q4_32") bad=1
        }
        END {
            for (i=0; i<43; i++) if (layer_seen[i]!=1) bad=1
            if (expected_layers=="all43") layer_ok=(q3_layers=="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42")
            else layer_ok=(q3_layers==expected_layers)
            exit !(seen==43 && q3==expected_count && layer_ok && !bad)
        }
    ' "$log"
}

validate_decode_selection() {
    local layout=$1 log=$2
    grep -Fxq 'ds4: SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4-prefetch2 (production default)' \
        "$log" || return 1
    grep -Fxq 'ds4: SM75 Q4-32 down decode mapping=tile32-int4 (production default)' \
        "$log" || return 1
    if [[ $layout == mixed15 ]]; then
        grep -Fxq 'ds4: SM75 Q4-32 decode gate/up mapping=tile32-mma (production default)' \
            "$log" || return 1
    fi
    grep -Fq 'CUDA decode indexer score row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'CUDA decode indexer score row split enabled: tier 1 ' "$log" || return 1
}

validate_decode_dispatch_audit() {
    local layout=$1 log=$2
    grep -Fq 'CUDA decode indexer row audit event=begin ' "$log" || return 1
    awk '
        /CUDA decode indexer row audit event=complete dispatch=split / {
            home=""
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]=="home_tier") home=a[2]
            }
            if (home==0 || home==1) seen[home]=1
        }
        END {exit !(seen[0] && seen[1])}
    ' "$log" || return 1
    awk '
        /SM75 Q3A4 decode mapping audit/ {
            seen++; delete v
            for (i=1; i<=NF; i++) {split($i,a,"="); v[a[1]]=a[2]+0}
            total=v["tile32-dp4a"]
            good=(v["control"]==0 && v["hwarp16"]==0 && v["tile32"]==0 &&
                  total>0 && v["k1"]==0 && v["k2"]==0 && v["k4"]==total &&
                  v["pf0"]==0 && v["pf1"]==0 && v["pf2"]==total)
        }
        END {exit !(seen==1 && good)}
    ' "$log" || return 1
    awk '
        /SM75 Q4-32 down decode mapping audit/ {
            seen++; c=t=cs=cp=ts=tp=p0=p1=p2=-1
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]=="control") c=a[2]+0
                else if (a[1]=="tile32") t=a[2]+0
                else if (a[1]=="control-slots") cs=a[2]+0
                else if (a[1]=="control-packed") cp=a[2]+0
                else if (a[1]=="tile32-slots") ts=a[2]+0
                else if (a[1]=="tile32-packed") tp=a[2]+0
                else if (a[1]=="pf0") p0=a[2]+0
                else if (a[1]=="pf1") p1=a[2]+0
                else if (a[1]=="pf2") p2=a[2]+0
            }
            good=(c==0 && t>0 && cs==0 && cp==0 && ts>0 && tp>0 &&
                  t==ts+tp && p0==t && p1==0 && p2==0)
        }
        END {exit !(seen==1 && good)}
    ' "$log" || return 1
    if [[ $layout == mixed15 ]]; then
        awk '
            /SM75 Q4-32 decode mapping audit/ {
                seen++; delete v
                for (i=1; i<=NF; i++) {split($i,a,"="); v[a[1]]=a[2]+0}
                total=v["tile32-mma"]
                good=(v["control"]==0 && v["hwarp16"]==0 &&
                      v["tile32-dp4a"]==0 && total>0 &&
                      v["pf0"]==total && v["pf1"]==0 && v["pf2"]==0)
            }
            END {exit !(seen==1 && good)}
        ' "$log" || return 1
    fi
}

validate_topology() {
    local layout=$1 log=$2 audit=${3:-0} marker route
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    ! grep -Fq 'required but unavailable' "$log" || return 1
    validate_cuda_inventory "$log" || return 1
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" || return 1
    validate_routed_layout "$layout" "$log" || return 1
    validate_decode_selection "$layout" "$log" || return 1
    if [[ $audit == 1 ]]; then
        validate_decode_dispatch_audit "$layout" "$log"
    fi
}

validate_telemetry() {
    local csv=$1
    [[ -s $csv ]] || return 1
    ! grep -Eq 'ERR!|Unknown Error|GPU is lost' "$csv" || return 1
    awk -F, -v expected_file="$OUTPUT_DIR/provenance/expected-nvml-inventory.tsv" \
        -v required="$REQUIRED_POWER_LIMITS_W" '
        BEGIN {
            split(required, required_power, ",")
            while ((getline line < expected_file) > 0) {
                split(line, row, "\t")
                expected_bus[row[1]]=tolower(row[2])
                expected_count++
            }
            close(expected_file)
        }
        NR==1 {next}
        {
            ts=$1
            for (i=2; i<=NF; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            }
            idx=$2
            if (idx ~ /^[0-3]$/) {
                per_gpu[idx]++
                seen[ts,idx]=1
                timestamps[ts]=1
                if (tolower($3)!=expected_bus[idx] ||
                    ($6+0)!=(required_power[idx+1]+0)) bad=1
            } else {
                bad=1
            }
        }
        END {
            complete=0
            for (ts in timestamps) {
                ok=1
                for (i=0; i<4; i++) if (!seen[ts,i]) ok=0
                if (ok) complete++
            }
            if (expected_count!=4 || bad) exit 1
            for (i=0; i<4; i++) if (per_gpu[i]<2) exit 1
            if (complete<2) exit 1
        }
    ' "$csv"
}

validate_q8_state() {
    local base=$1
    local plan="$base.q8-plan.csv" bindings="$base.q8-bindings.csv"
    [[ -s $plan && -s $bindings ]] || return 1
    [[ $(wc -l <"$plan") == 345 && $(wc -l <"$bindings") == 345 ]] || return 1
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
                    $3=="partner_offload" && $12=="label" && $15=="live")
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

canonical_mtp_placement() {
    awk '
        /legacy MTP support pack tier=/ {
            pack_seen++
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]=="tier") pack=a[2]
            }
        }
        /legacy MTP support tensors cached on tier / {
            cache_seen++
            for (i=1; i<NF; i++) if ($i=="tier") cache=$(i+1)
        }
        /legacy MTP base embedding bucket ready on tier / {
            base_seen++
            for (i=1; i<NF; i++) if ($i=="tier") base=$(i+1)
        }
        END {
            if (pack_seen!=1 || cache_seen!=1 || base_seen!=1 ||
                pack !~ /^[0-9]+$/ || cache !~ /^[0-9]+$/ ||
                base !~ /^[0-9]+$/ || pack!=cache || cache!=base) exit 1
            printf "executor_tier=%s\n", pack
        }
    ' "$1"
}

validate_mtp_mode() {
    local variant=$1 log=$2
    ! grep -Fq 'legacy MTP support is ignored under tensor parallelism' "$log" || return 1
    case "$variant" in
        plain)
            ! grep -Fq 'MTP support model loaded:' "$log" &&
            ! grep -Fq 'legacy MTP support pack tier=' "$log" &&
            ! grep -Fq 'ds4-bench: MTP decode frontier=' "$log"
            ;;
        resident)
            grep -Fq 'MTP support model loaded:' "$log" &&
            grep -Fq 'draft=1' "$log" &&
            [[ $(grep -Fc 'legacy MTP support pack tier=' "$log") == 1 ]] &&
            [[ $(grep -Fc 'legacy MTP support tensors cached on tier ' "$log") == 1 ]] &&
            grep -Fq ', executor tier)' "$log" &&
            ! grep -Fq ', direct-peer spill)' "$log" &&
            grep -Fq 'legacy MTP base embedding bucket ready on tier ' "$log" &&
            grep -Fq 'legacy MTP private raw cache starts cold' "$log" &&
            canonical_mtp_placement "$log" >/dev/null &&
            ! grep -Fq 'ds4-bench: MTP decode frontier=' "$log"
            ;;
        mtp2|mtp2-diag)
            grep -Fq 'MTP support model loaded:' "$log" &&
            grep -Fq 'draft=2' "$log" &&
            [[ $(grep -Fc 'legacy MTP support pack tier=' "$log") == 1 ]] &&
            [[ $(grep -Fc 'legacy MTP support tensors cached on tier ' "$log") == 1 ]] &&
            grep -Fq ', executor tier)' "$log" &&
            ! grep -Fq ', direct-peer spill)' "$log" &&
            grep -Fq 'legacy MTP base embedding bucket ready on tier ' "$log" &&
            grep -Fq 'legacy MTP private raw cache starts cold' "$log" &&
            canonical_mtp_placement "$log" >/dev/null &&
            [[ $(grep -Fc 'ds4-bench: MTP decode frontier=' "$log") == 3 ]] &&
            ! grep -Fq 'falling back to sequential' "$log" &&
            ! grep -Fq 'verifier failed' "$log" &&
            { [[ $variant == mtp2 ]] || grep -Fq 'ds4: mtp timing ' "$log"; }
            ;;
    esac
}

run_arm() {
    local model=$1 variant=$2 base=$3 token_dir=$4 rc=0 telemetry_alive=0
    local -a audit_env=() mtp_env=() mtp_args=()
    case "$variant" in
        plain) ;;
        resident)
            mtp_args=(--mtp "$MTP_MODEL" --mtp-draft 1 --mtp-margin "$MTP_MARGIN")
            ;;
        mtp2)
            mtp_env=(DS4_MTP_STRICT=1)
            mtp_args=(--mtp "$MTP_MODEL" --mtp-draft 2 --mtp-margin "$MTP_MARGIN")
            ;;
        mtp2-diag)
            audit_env=("${diagnostic_env[@]}")
            mtp_env=(DS4_MTP_STRICT=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1)
            mtp_args=(--mtp "$MTP_MODEL" --mtp-draft 2 --mtp-margin "$MTP_MARGIN")
            ;;
        *) return 2 ;;
    esac
    mkdir -p "$token_dir"
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    nvidia-smi --query-gpu=timestamp,index,pci.bus_id,pstate,power.draw,power.limit,utilization.gpu,memory.used,memory.total \
        --format=csv -lms 500 >"$base.telemetry.csv" 2>&1 &
    telemetry_pid=$!
    "${production_env[@]}" "${audit_env[@]}" "${mtp_env[@]}" \
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv" \
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$model" "${mtp_args[@]}" --prompt-file "$PROMPT" \
        --ctx-start 512 --ctx-max "$CTX_MAX" --ctx-alloc "$CTX_ALLOC" \
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" --gen-tokens "$TG_TOKENS" \
        --dump-generated-tokens-dir "$token_dir" --csv "$base.csv" \
        >"$base.log" 2>&1 || rc=$?
    if telemetry_process_running "$telemetry_pid"; then
        telemetry_alive=1
    fi
    stop_telemetry "$telemetry_pid" || return 1
    printf 'alive_before_stop=%d\n' "$telemetry_alive" \
        >"$base.telemetry-status.txt"
    telemetry_pid=
    capture_gpu_health "$base.post-gpu.csv" || return 1
    (( telemetry_alive == 1 )) || return 1
    return "$rc"
}

phase=production-ab
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" || die "could not capture initial GPU health"
validate_gpu_snapshot "$OUTPUT_DIR/initial-gpu.csv" ||
    die "physical GPU identity or required power limit changed during build/smoke"
printf 'layout\trepeat\tslot\tvariant\tcsv\tlog\ttokens\ttelemetry\tq8_plan\tq8_bindings\n' >"$OUTPUT_DIR/runs.tsv"
printf 'layout\trepeat\tvariant\tq8_plan_equal_plain\tq8_bindings_equal_plain\n' \
    >"$OUTPUT_DIR/summary/cache-placement-comparison.tsv"
printf 'layout\trepeat\tcontext\tvariant\tprefill_tps\tdecode_tps\tpost_first_cycle_tps\tfirst_cycle_ms\tpaired_decode_speedup\tpaired_prefill_speedup\n' \
    >"$OUTPUT_DIR/summary/measurements.tsv"
printf 'layout\tkind\trepeat\tcontext\ttokens\tcycles\tmulti_token_cycles\tmax_cycle_tokens\tmean_tokens_per_cycle\n' \
    >"$OUTPUT_DIR/summary/acceptance.tsv"

append_acceptance() {
    local layout=$1 kind=$2 repeat=$3 log=$4 before after
    before=$(wc -l <"$OUTPUT_DIR/summary/acceptance.tsv")
    awk -v OFS='\t' -v layout="$layout" -v kind="$kind" \
        -v repeat="$repeat" -v expected="$TG_TOKENS" '
        /ds4-bench: MTP decode frontier=/ {
            ctx=tokens=cycles=multi=max=mean=""
            for (i=1; i<=NF; i++) {
                split($i, kv, "=")
                if (kv[1]=="frontier") ctx=kv[2]
                else if (kv[1]=="tokens") tokens=kv[2]
                else if (kv[1]=="cycles") cycles=kv[2]
                else if (kv[1]=="multi_token_cycles") multi=kv[2]
                else if (kv[1]=="max_cycle_tokens") max=kv[2]
                else if (kv[1]=="mean_tokens_per_cycle") mean=kv[2]
            }
            mean_delta=9999
            if ((cycles+0)>0) {
                expected_mean=tokens/cycles
                mean_delta=mean-expected_mean
                if (mean_delta<0) mean_delta=-mean_delta
            }
            if ((ctx!=512 && ctx!=4096 && ctx!=32768) || ctx_seen[ctx]++ ||
                tokens!=expected ||
                cycles<1 || cycles>tokens || multi<0 || multi>cycles ||
                max<1 || max>3 || mean<1 || mean>3 || mean_delta>0.00001) bad=1
            print layout,kind,repeat,ctx,tokens,cycles,multi,max,mean
            seen++
        }
        END {
            exit !(seen==3 && ctx_seen[512]==1 && ctx_seen[4096]==1 &&
                   ctx_seen[32768]==1 && !bad)
        }
    ' "$log" >>"$OUTPUT_DIR/summary/acceptance.tsv" || return 1
    after=$(wc -l <"$OUTPUT_DIR/summary/acceptance.tsv")
    [[ $((after - before)) == 3 ]]
}

append_measurements() {
    local layout=$1 repeat=$2 variant=$3 plain=$4 csv=$5
    awk -F, -v OFS='\t' -v layout="$layout" -v repeat="$repeat" \
        -v variant="$variant" '
        FNR==NR {
            if (FNR>1) {
                base_decode[$1]=$5
                base_prefill[$1]=$3
            }
            next
        }
        FNR>1 {
            if (!(($1==512 || $1==4096 || $1==32768) &&
                  base_decode[$1]>0 && base_prefill[$1]>0 &&
                  $3>0 && $5>0 && $8>0 && $6>0)) bad=1
            print layout,repeat,$1,variant,$3,$5,$8,$6,
                  $5/base_decode[$1],$3/base_prefill[$1]
            seen++
        }
        END {exit !(seen==3 && !bad)}
    ' "$plain" "$csv" >>"$OUTPUT_DIR/summary/measurements.tsv"
}

for index in "${!layouts[@]}"; do
    layout=${layouts[$index]}
    model=${models[$index]}
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        case $(((repeat - 1) % 3)) in
            0) variants=(plain resident mtp2) ;;
            1) variants=(resident mtp2 plain) ;;
            2) variants=(mtp2 plain resident) ;;
        esac
        slot=0
        for variant in "${variants[@]}"; do
            slot=$((slot + 1))
            phase="$layout-r$repeat-$variant"
            base="$OUTPUT_DIR/runs/$layout-r$repeat-$variant"
            token_dir="$base-tokens"
            printf 'MTP production A/B model=%s repeat=%d/%d slot=%d/3 variant=%s...\n' \
                "$layout" "$repeat" "$REPEATS" "$slot" "$variant"
            run_arm "$model" "$variant" "$base" "$token_dir" || {
                tail -n 200 "$base.log" >&2 || true
                die "$layout repeat=$repeat variant=$variant run failed"
            }
            validate_health "$base" || die "$layout repeat=$repeat variant=$variant GPU health changed"
            validate_telemetry "$base.telemetry.csv" || die "$layout repeat=$repeat variant=$variant telemetry failed"
            validate_csv "$base.csv" || die "$layout repeat=$repeat variant=$variant timing output is incomplete"
            validate_tokens "$token_dir" || die "$layout repeat=$repeat variant=$variant token capture is incomplete"
            validate_topology "$layout" "$base.log" || die "$layout repeat=$repeat variant=$variant topology validation failed"
            validate_q8_state "$base" || die "$layout repeat=$repeat variant=$variant cache/residency evidence is incomplete"
            validate_mtp_mode "$variant" "$base.log" || die "$layout repeat=$repeat variant=$variant MTP dispatch validation failed"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$layout" "$repeat" "$slot" "$variant" "$base.csv" "$base.log" \
                "$token_dir" "$base.telemetry.csv" "$base.q8-plan.csv" \
                "$base.q8-bindings.csv" >>"$OUTPUT_DIR/runs.tsv"
        done

        plain="$OUTPUT_DIR/runs/$layout-r$repeat-plain"
        canonical_plain="$OUTPUT_DIR/runs/$layout-r1-plain"
        canonical_support="$OUTPUT_DIR/runs/$layout-r1-resident"
        if (( repeat > 1 )); then
            for context in 512 4096 32768; do
                printf -v name 'frontier_%06d.tokens.i32le' "$context"
                cmp -s "$canonical_plain-tokens/$name" "$plain-tokens/$name" ||
                    die "$layout plain stream is nondeterministic at repeat=$repeat PP=$context"
            done
            cmp -s "$canonical_plain.q8-plan.csv" "$plain.q8-plan.csv" ||
                die "$layout plain Q8 placement changed at repeat=$repeat"
            cmp -s <(canonical_q8_bindings "$canonical_plain.q8-bindings.csv") \
                   <(canonical_q8_bindings "$plain.q8-bindings.csv") ||
                die "$layout plain Q8 binding identities changed at repeat=$repeat"
        fi
        for candidate in resident mtp2; do
            candidate_base="$OUTPUT_DIR/runs/$layout-r$repeat-$candidate"
            for context in 512 4096 32768; do
                printf -v name 'frontier_%06d.tokens.i32le' "$context"
                cmp -s "$plain-tokens/$name" "$candidate_base-tokens/$name" ||
                    die "$layout repeat=$repeat $candidate stream diverged at PP=$context"
            done
            cmp -s "$plain.q8-plan.csv" "$candidate_base.q8-plan.csv" ||
                die "$layout repeat=$repeat $candidate changed the main-model Q8 placement plan"
            cmp -s <(canonical_q8_bindings "$plain.q8-bindings.csv") \
                   <(canonical_q8_bindings "$candidate_base.q8-bindings.csv") ||
                die "$layout repeat=$repeat $candidate changed the main-model Q8 binding identities"
            canonical_support_placement=$(canonical_mtp_placement "$canonical_support.log") ||
                die "$layout canonical MTP support placement is invalid"
            candidate_support_placement=$(canonical_mtp_placement "$candidate_base.log") ||
                die "$layout repeat=$repeat $candidate MTP support placement is invalid"
            [[ $canonical_support_placement == "$candidate_support_placement" ]] ||
                die "$layout repeat=$repeat $candidate changed the MTP executor/cache tier"
            printf '%s\t%s\t%s\tyes\tyes\n' "$layout" "$repeat" "$candidate" \
                >>"$OUTPUT_DIR/summary/cache-placement-comparison.tsv"
        done
        for variant in plain resident mtp2; do
            append_measurements "$layout" "$repeat" "$variant" \
                "$plain.csv" "$OUTPUT_DIR/runs/$layout-r$repeat-$variant.csv" ||
                die "$layout repeat=$repeat $variant measurement summary failed"
        done
        append_acceptance "$layout" clean "$repeat" \
            "$OUTPUT_DIR/runs/$layout-r$repeat-mtp2.log" ||
            die "$layout repeat=$repeat MTP acceptance summary is invalid"
    done

    phase="$layout-mtp2-diag"
    base="$OUTPUT_DIR/runs/$layout-mtp2-diag"
    token_dir="$base-tokens"
    printf 'MTP diagnostic model=%s variant=mtp2-diag (excluded from timing tables)...\n' "$layout"
    run_arm "$model" mtp2-diag "$base" "$token_dir" || {
        tail -n 200 "$base.log" >&2 || true
        die "$layout mtp2 diagnostic run failed"
    }
    validate_health "$base" || die "$layout mtp2 diagnostic GPU health changed"
    validate_telemetry "$base.telemetry.csv" || die "$layout mtp2 diagnostic telemetry failed"
    validate_csv "$base.csv" || die "$layout mtp2 diagnostic output is incomplete"
    validate_tokens "$token_dir" || die "$layout mtp2 diagnostic token capture is incomplete"
    validate_topology "$layout" "$base.log" 1 || die "$layout mtp2 diagnostic topology validation failed"
    validate_q8_state "$base" || die "$layout mtp2 diagnostic cache evidence is incomplete"
    validate_mtp_mode mtp2-diag "$base.log" || die "$layout mtp2 diagnostic dispatch validation failed"
    plain="$OUTPUT_DIR/runs/$layout-r1-plain"
    for context in 512 4096 32768; do
        printf -v name 'frontier_%06d.tokens.i32le' "$context"
        cmp -s "$plain-tokens/$name" "$token_dir/$name" ||
            die "$layout mtp2 diagnostic stream diverged at PP=$context"
    done
    cmp -s "$plain.q8-plan.csv" "$base.q8-plan.csv" ||
        die "$layout mtp2 diagnostic changed the main-model Q8 placement plan"
    cmp -s <(canonical_q8_bindings "$plain.q8-bindings.csv") \
           <(canonical_q8_bindings "$base.q8-bindings.csv") ||
        die "$layout mtp2 diagnostic changed the main-model Q8 binding identities"
    canonical_support="$OUTPUT_DIR/runs/$layout-r1-resident"
    canonical_support_placement=$(canonical_mtp_placement "$canonical_support.log") ||
        die "$layout canonical MTP support placement is invalid"
    diagnostic_support_placement=$(canonical_mtp_placement "$base.log") ||
        die "$layout diagnostic MTP support placement is invalid"
    [[ $canonical_support_placement == "$diagnostic_support_placement" ]] ||
        die "$layout mtp2 diagnostic changed the MTP executor/cache tier"
    printf '%s\tdiagnostic\tdiagnostic\tmtp2-diag\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$layout" "$base.csv" "$base.log" "$token_dir" "$base.telemetry.csv" \
        "$base.q8-plan.csv" "$base.q8-bindings.csv" >>"$OUTPUT_DIR/runs.tsv"
    grep -E 'ds4: mtp (timing|spec|conf)|ds4-bench: MTP decode frontier=' \
        "$base.log" >"$OUTPUT_DIR/runs/$layout-mtp2-diag-events.log"
    append_acceptance "$layout" diagnostic diagnostic "$base.log" ||
        die "$layout diagnostic MTP acceptance summary is invalid"
done

median_metric() {
    local layout=$1 context=$2 variant=$3 field=$4
    awk -F'\t' -v layout="$layout" -v context="$context" \
        -v variant="$variant" -v field="$field" '
        NR>1 && $1==layout && $3==context && $4==variant {print $field}
    ' "$OUTPUT_DIR/summary/measurements.tsv" | sort -n | awk '
        {value[NR]=$1}
        END {
            if (!NR) exit 1
            if (NR%2) printf "%.9f", value[(NR+1)/2]
            else printf "%.9f", (value[NR/2]+value[NR/2+1])/2
        }
    '
}

speedup_stats() {
    local layout=$1 context=$2 variant=$3 field=$4
    awk -F'\t' -v layout="$layout" -v context="$context" \
        -v variant="$variant" -v field="$field" '
        NR>1 && $1==layout && $3==context && $4==variant {
            x=$field+0; n++; sum+=x; sum2+=x*x
            if (n==1 || x<min) min=x
            if (n==1 || x>max) max=x
        }
        END {
            if (!n) exit 1
            variance=n>1 ? (sum2-sum*sum/n)/(n-1) : 0
            if (variance<0 && variance>-1e-12) variance=0
            printf "%.9f %.9f %.9f", sqrt(variance), min, max
        }
    ' "$OUTPUT_DIR/summary/measurements.tsv"
}

phase=summarize
{
    printf '# SM75 four-GPU legacy MTP production A/B\n\n'
    if (( REPEATS == 1 )); then
        printf 'This is a one-repeat preliminary smoke, not balanced production qualification; the separate mtp2 diagnostic arm is excluded.\n\n'
    else
        printf 'Clean timing uses %s balanced Latin-rotated repeats; the separate mtp2 diagnostic arm is excluded.\n\n' "$REPEATS"
    fi
    printf '| Model | Context | Variant | Median prefill tok/s | Paired prefill speedup | SD | Min | Max | Median decode tok/s | Paired decode speedup | SD | Min | Max |\n'
    printf '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for layout in "${layouts[@]}"; do
        for context in 512 4096 32768; do
            for variant in plain resident mtp2; do
                prefill=$(median_metric "$layout" "$context" "$variant" 5)
                decode=$(median_metric "$layout" "$context" "$variant" 6)
                decode_speedup=$(median_metric "$layout" "$context" "$variant" 9)
                prefill_speedup=$(median_metric "$layout" "$context" "$variant" 10)
                read -r decode_sd decode_min decode_max \
                    <<<"$(speedup_stats "$layout" "$context" "$variant" 9)"
                read -r prefill_sd prefill_min prefill_max \
                    <<<"$(speedup_stats "$layout" "$context" "$variant" 10)"
                printf '| %s | %s | %s | %.3f | %.6fx | %.6f | %.6f | %.6f | %.3f | %.6fx | %.6f | %.6f | %.6f |\n' \
                    "$layout" "$context" "$variant" "$prefill" \
                    "$prefill_speedup" "$prefill_sd" "$prefill_min" "$prefill_max" \
                    "$decode" "$decode_speedup" "$decode_sd" "$decode_min" "$decode_max"
            done
        done
        powered_contexts=0
        printf '\n'
        for acceptance_context in 512 4096 32768; do
            clean_multi=$(awk -F'\t' -v layout="$layout" \
                -v context="$acceptance_context" '
                NR>1 && $1==layout && $2=="clean" && $4==context {sum+=$7}
                END {print sum+0}
            ' "$OUTPUT_DIR/summary/acceptance.tsv")
            if (( clean_multi > 0 )); then
                powered_contexts=$((powered_contexts + 1))
                acceptance_status=accepted
            else
                acceptance_status=zero-useful-acceptance
            fi
            printf '%s PP%s clean multi-token cycles: %s (%s).\n' \
                "$layout" "$acceptance_context" "$clean_multi" "$acceptance_status"
        done
        if (( powered_contexts == 3 )); then
            printf '%s demonstrated useful MTP acceptance at all three frontiers.\n\n' \
                "$layout"
        else
            printf '%s did not demonstrate useful MTP acceptance at all three frontiers; do not call it production MTP-powered.\n\n' \
                "$layout"
        fi
    done
    printf '\nAll 256 emitted tokens plus the untimed next-token sentinel are byte-identical to plain decode at all three frontiers.\n'
    printf 'Resident measures support-model residency with draft depth 1; mtp2 uses draft depth 2 with DS4_MTP_STRICT.\n'
    printf 'First-cycle and post-first-cycle diagnostics are retained in measurements.tsv; only total decode tok/s drives the A/B.\n'
    printf 'This evidence run does not promote MTP automatically; acceptance and net end-to-end speedup determine the next implementation.\n'
} | tee "$OUTPUT_DIR/summary/report.md"

phase=complete
printf 'SM75 four-GPU MTP production A/B complete: %s\n' "$OUTPUT_DIR"
