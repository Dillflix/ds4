#!/usr/bin/env bash
set -Eeuo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
for tool in nvidia-smi nvcc cc python3 make nproc strings tar awk stat realpath \
            git sha256sum cp uname grep sort env tail mv rm date flock tee; do
    require_command "$tool"
done
[[ $(uname -s) == Linux ]] || die "this SM75 evidence workflow must run on Linux"
if [[ -v CUDA_VISIBLE_DEVICES || -v NVIDIA_VISIBLE_DEVICES ]]; then
    die "unset CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES; this fixed physical-GPU comparison refuses ordinal remapping"
fi

: "${MODEL_Q8:?set MODEL_Q8 to the absolute Q8-dense GGUF path}"
: "${MODEL_SOURCE_F16:?set MODEL_SOURCE_F16 to the absolute source-derived-F16 GGUF path}"
[[ $MODEL_Q8 == /* && -f $MODEL_Q8 ]] ||
    die "MODEL_Q8 must name an existing absolute GGUF path"
[[ $MODEL_SOURCE_F16 == /* && -f $MODEL_SOURCE_F16 ]] ||
    die "MODEL_SOURCE_F16 must name an existing absolute GGUF path"
MODEL_Q8=$(realpath "$MODEL_Q8")
MODEL_SOURCE_F16=$(realpath "$MODEL_SOURCE_F16")
[[ $MODEL_Q8 != "$MODEL_SOURCE_F16" ]] || die "the two model paths must differ"
SOURCE_F16_LOCK="$MODEL_SOURCE_F16.lock"
exec 8>>"$SOURCE_F16_LOCK" || die "cannot open source-F16 publication lock: $SOURCE_F16_LOCK"
flock --shared --nonblock 8 ||
    die "source-F16 producer owns the model publication lock: $SOURCE_F16_LOCK"

SOURCE_F16_PROVENANCE=${SOURCE_F16_PROVENANCE:-${MODEL_SOURCE_F16%.gguf}.source-f16.provenance.txt}
[[ $SOURCE_F16_PROVENANCE == /* && -f $SOURCE_F16_PROVENANCE ]] ||
    die "SOURCE_F16_PROVENANCE must name the producer sidecar: $SOURCE_F16_PROVENANCE"
SOURCE_F16_PROVENANCE=$(realpath "$SOURCE_F16_PROVENANCE")
awk -F= '
    !/^[A-Za-z0-9_]+=/ { exit 1 }
    ++seen[$1] != 1 { exit 1 }
' "$SOURCE_F16_PROVENANCE" ||
    die "source provenance contains a malformed or duplicate record"

provenance_value() {
    local key=$1
    awk -F= -v wanted="$key" '
        $1 == wanted { value=substr($0, index($0, "=") + 1); seen++ }
        END { if (seen == 1) print value; else exit 1 }
    ' "$SOURCE_F16_PROVENANCE"
}
require_provenance_value() {
    local key=$1 value
    value=$(provenance_value "$key") ||
        die "source provenance must contain exactly one $key record"
    [[ -n $value ]] || die "source provenance has an empty $key record"
    printf '%s\n' "$value"
}

prov_source_model=$(require_provenance_value source_f16_model)
prov_template=$(require_provenance_value template_model)
[[ $(realpath "$prov_source_model") == "$MODEL_SOURCE_F16" ]] ||
    die "source provenance names a different F16 model"
[[ $(realpath "$prov_template") == "$MODEL_Q8" ]] ||
    die "source provenance was generated from a different Q8 template"
[[ $(require_provenance_value source_f16_model_bytes) == "$(stat -c %s "$MODEL_SOURCE_F16")" ]] ||
    die "source-F16 model size differs from producer provenance"
[[ $(require_provenance_value template_model_bytes) == "$(stat -c %s "$MODEL_Q8")" ]] ||
    die "Q8 template size differs from producer provenance"
for key in source_f16_model_device source_f16_model_inode source_f16_model_mtime_epoch \
           source_f16_model_mtime_ns template_model_device template_model_inode \
           template_model_mtime_epoch template_model_mtime_ns; do
    value=$(require_provenance_value "$key")
    [[ $value =~ ^[0-9]+$ ]] || die "source provenance has invalid $key"
    printf -v "prov_$key" '%s' "$value"
done
capture_model_identity() {
    python3 -c '
import os, stat, sys
value = os.lstat(sys.argv[1])
if not stat.S_ISREG(value.st_mode):
    raise SystemExit("not a regular file")
print(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, sep="\t")
' "$1"
}
verify_model_file_identities() {
    local identity device inode bytes mtime_ns
    identity=$(capture_model_identity "$MODEL_SOURCE_F16") ||
        die "cannot capture source-F16 model identity"
    IFS=$'\t' read -r device inode bytes mtime_ns <<<"$identity"
    [[ $device == "$prov_source_f16_model_device" &&
       $inode == "$prov_source_f16_model_inode" &&
       $mtime_ns == "$prov_source_f16_model_mtime_ns" &&
       $((mtime_ns / 1000000000)) == "$prov_source_f16_model_mtime_epoch" &&
       $bytes == "$(require_provenance_value source_f16_model_bytes)" ]] ||
        die "source-F16 model identity changed since validated publication"
    identity=$(capture_model_identity "$MODEL_Q8") ||
        die "cannot capture Q8 template identity"
    IFS=$'\t' read -r device inode bytes mtime_ns <<<"$identity"
    [[ $device == "$prov_template_model_device" &&
       $inode == "$prov_template_model_inode" &&
       $mtime_ns == "$prov_template_model_mtime_ns" &&
       $((mtime_ns / 1000000000)) == "$prov_template_model_mtime_epoch" &&
       $bytes == "$(require_provenance_value template_model_bytes)" ]] ||
        die "Q8 template identity changed since source-F16 generation"
}
verify_model_file_identities
[[ $(require_provenance_value hf_repository) == deepseek-ai/DeepSeek-V4-Flash-0731 ]] ||
    die "source provenance names the wrong HF repository"
[[ $(require_provenance_value hf_revision) == 7872f01b1d1fe23eabc4c98b48bffcef5a386062 ]] ||
    die "source provenance names the wrong HF revision"
[[ $(require_provenance_value hf_snapshot_path_revision_match) == true ]] ||
    die "source provenance does not qualify the fixed HF snapshot path"
[[ $(require_provenance_value hf_shard_content_authentication) == not_performed ]] ||
    die "source provenance does not state the shard-authentication boundary"
if grep -q '^hf_source_verified=' "$SOURCE_F16_PROVENANCE"; then
    die "source provenance uses the rejected overstated hf_source_verified field"
fi
for key in hf_index_sha256 hf_config_sha256 quantizer_sha256; do
    value=$(require_provenance_value "$key")
    [[ $value =~ ^[0-9a-f]{64}$ ]] || die "source provenance has invalid $key"
done
for key in git_commit; do
    value=$(require_provenance_value "$key")
    [[ $value =~ ^[0-9a-f]{40}$ ]] || die "source provenance has invalid $key"
done
for key in source_f16_selected template_q8_selected; do
    [[ $(require_provenance_value "$key") == 344 ]] ||
        die "source provenance has invalid $key"
done
[[ $(require_provenance_value source_f16_selected_bytes) == 11362369536 ]] ||
    die "source provenance has invalid source_f16_selected_bytes"
for key in quantizer_bytes hf_index_bytes hf_config_bytes; do
    value=$(require_provenance_value "$key")
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "source provenance has invalid $key"
done
for key in quantizer_device hf_index_device hf_config_device; do
    value=$(require_provenance_value "$key")
    [[ $value =~ ^[0-9]+$ ]] || die "source provenance has invalid $key"
done
for key in quantizer_inode quantizer_mtime_ns hf_index_inode hf_index_mtime_epoch \
           hf_index_mtime_ns hf_config_inode hf_config_mtime_epoch hf_config_mtime_ns; do
    value=$(require_provenance_value "$key")
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "source provenance has invalid $key"
done
hf_index_mtime_ns=$(require_provenance_value hf_index_mtime_ns)
hf_index_mtime_epoch=$(require_provenance_value hf_index_mtime_epoch)
hf_config_mtime_ns=$(require_provenance_value hf_config_mtime_ns)
hf_config_mtime_epoch=$(require_provenance_value hf_config_mtime_epoch)
(( hf_index_mtime_ns / 1000000000 == hf_index_mtime_epoch )) ||
    die "source provenance has inconsistent HF-index mtime"
(( hf_config_mtime_ns / 1000000000 == hf_config_mtime_epoch )) ||
    die "source provenance has inconsistent HF-config mtime"
value=$(require_provenance_value git_dirty)
[[ $value == true || $value == false ]] || die "source provenance has invalid git_dirty"
[[ $(require_provenance_value kv_metadata_byte_identical) == true ]] ||
    die "source provenance does not guarantee raw GGUF KV metadata identity"
[[ $(require_provenance_value payload_extents_valid) == true ]] ||
    die "source provenance does not guarantee payload extents"
[[ $(require_provenance_value non_target_payload_copy) == producer-guaranteed-not-independently-rehashed ]] ||
    die "source provenance does not identify the copy-through guarantee"
[[ $(require_provenance_value model_payload_hashing) == disabled ]] ||
    die "source provenance does not record the non-hashing policy"
[[ $(require_provenance_value publication_protocol) == model-last-rollback-journal-v1 &&
   $(require_provenance_value provenance_complete) == true ]] ||
    die "source provenance does not record a complete model-last publication"
shard_count=$(require_provenance_value hf_shard_count)
[[ $shard_count =~ ^[1-9][0-9]*$ ]] || die "source provenance has invalid hf_shard_count"
for ((shard=0; shard<shard_count; shard++)); do
    printf -v shard_key 'hf_shard_%03d_name' "$shard"
    shard_name=$(require_provenance_value "$shard_key")
    [[ $shard_name != /* && $shard_name != ../* && $shard_name != */../* && \
       $shard_name != */.. ]] || die "source provenance has unsafe $shard_key"
    printf -v shard_key 'hf_shard_%03d_resolved' "$shard"
    shard_resolved=$(require_provenance_value "$shard_key")
    [[ $shard_resolved == /* ]] || die "source provenance has non-absolute $shard_key"
    printf -v shard_key 'hf_shard_%03d_resolved_basename' "$shard"
    value=$(require_provenance_value "$shard_key")
    [[ $value == "${shard_resolved##*/}" ]] ||
        die "source provenance has inconsistent $shard_key"
    printf -v shard_key 'hf_shard_%03d_bytes' "$shard"
    value=$(require_provenance_value "$shard_key")
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "source provenance has invalid $shard_key"
    for suffix in device inode mtime_ns; do
        printf -v shard_key 'hf_shard_%03d_%s' "$shard" "$suffix"
        value=$(require_provenance_value "$shard_key")
        if [[ $suffix == device ]]; then
            [[ $value =~ ^[0-9]+$ ]] || die "source provenance has invalid $shard_key"
        else
            [[ $value =~ ^[1-9][0-9]*$ ]] || die "source provenance has invalid $shard_key"
        fi
    done
done
for key in quantizer hf_directory hf_index hf_config; do
    require_provenance_value "$key" >/dev/null
done

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
QUALITY_MANIFEST=${QUALITY_MANIFEST:-$repo_dir/gguf-tools/quality-testing/data/flash/manifest.tsv}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
QUALITY_CTX=${QUALITY_CTX:-32769}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-32768}
STEP_MUL=${STEP_MUL:-4}
CTX_ALLOC=${CTX_ALLOC:-32897}
GEN_TOKENS=${GEN_TOKENS:-128}
REPEATS=${REPEATS:-4}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${DENSE_F16_DECODE_AB_DIR:-$repo_dir/dense-f16-decode-ab-$stamp}

[[ $PROMPT == /* && -f $PROMPT ]] || die "PROMPT must name an existing absolute path"
[[ $QUALITY_MANIFEST == /* && -f $QUALITY_MANIFEST ]] ||
    die "QUALITY_MANIFEST must name an existing absolute path"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "comparison requires GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
[[ $QUALITY_CTX == 32769 && $PREFILL_CHUNK == 2048 ]] ||
    die "comparison requires QUALITY_CTX=32769 PREFILL_CHUNK=2048"
[[ $CTX_START == 2048 && $CTX_MAX == 32768 && $STEP_MUL == 4 ]] ||
    die "sweep is fixed at 2K, 8K, and 32K"
for item in "CTX_ALLOC:$CTX_ALLOC" "GEN_TOKENS:$GEN_TOKENS" "REPEATS:$REPEATS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_ALLOC >= CTX_MAX + GEN_TOKENS + 1 )) ||
    die "CTX_ALLOC must cover CTX_MAX + GEN_TOKENS + 1"
(( GEN_TOKENS > 0 )) || die "GEN_TOKENS must be positive; this is also a decode experiment"
(( REPEATS >= 4 && REPEATS % 2 == 0 )) ||
    die "REPEATS must be an even integer of at least 4 for exact AB/BA balance"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"/{inventory,quality,runs,memory,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
memory_sampler_pid=
stop_memory_sampler() {
    if [[ -n ${memory_sampler_pid:-} ]]; then
        kill "$memory_sampler_pid" >/dev/null 2>&1 || true
        wait "$memory_sampler_pid" >/dev/null 2>&1 || true
        memory_sampler_pid=
    fi
}
finish() {
    local status=$?
    trap - EXIT INT TERM
    stop_memory_sampler
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        local archive="$OUTPUT_DIR.tar.gz"
        local partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv -- "$partial" "$archive"; then
            printf '%s: %s\n' \
                "$([[ $status == 0 ]] && printf 'Archive to return' || printf 'Diagnostic archive')" \
                "$archive" >&2
        else
            printf 'error: failed to create nonempty archive: %s\n' "$archive" >&2
            [[ $status != 0 ]] || status=1
            rm -f -- "$partial"
            printf 'state=failed\nexit_status=%s\nlast_phase=archive\n' \
                "$status" >"$OUTPUT_DIR/run-status.txt"
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT
trap 'phase=interrupted; exit 143' TERM

start_memory_sampler() {
    local path=$1
    nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.total,memory.used,memory.free \
        --format=csv,nounits -lms 1000 >"$path" 2>&1 &
    memory_sampler_pid=$!
}

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
common_env=(
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
)
native_env=(
    DS4_CUDA_NO_Q8_F16_CACHE=1
    DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=none
)
source_env=(
    DS4_CUDA_NO_Q8_F16_CACHE=1
    DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=none
)
arms=(native-q8 source-f16)
arm_model() {
    verify_model_file_identities
    case $1 in
        native-q8) printf '%s\n' "$MODEL_Q8" ;;
        source-f16) printf '%s\n' "$MODEL_SOURCE_F16" ;;
        *) die "unknown arm: $1" ;;
    esac
}
set_arm_env() {
    case $1 in
        native-q8) arm_env=("${native_env[@]}") ;;
        source-f16) arm_env=("${source_env[@]}") ;;
        *) die "unknown arm: $1" ;;
    esac
}
require_direct_peer_log() {
    local path=$1 edge
    for edge in '0->1' '1->0' '2->3' '3->2'; do
        grep -Fq "peer access $edge validated across" "$path" ||
            die "$path lacks validated direct peer marker for $edge"
    done
    ! grep -Eq 'peer access .* FAILED validation' "$path" ||
        die "$path reports a failed peer validation"
}

phase=topology
nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt"
topology_link() {
    local from=$1 to=$2
    awk -v from="GPU$from" -v to="GPU$to" '
        !header {
            n_gpu = 0
            for (i = 1; i <= NF; i++) if ($i ~ /^GPU[0-9]+$/) n_gpu++
            if (n_gpu > 1) {
                for (i = 1; i <= NF; i++) if ($i == to) column = i + 1
                header = 1
                next
            }
        }
        header && $1 == from && column > 0 { print $column; exit }
    ' "$OUTPUT_DIR/provenance/topology.txt"
}
for pair in '0 1' '2 3'; do
    read -r first second <<<"$pair"
    forward=$(topology_link "$first" "$second")
    reverse=$(topology_link "$second" "$first")
    [[ $forward =~ ^NV[0-9]+$ && $reverse =~ ^NV[0-9]+$ ]] ||
        die "physical GPU pair $first<->$second is not NVLink: ${forward:-missing}/${reverse:-missing}"
done

cp -- "$SOURCE_F16_PROVENANCE" "$OUTPUT_DIR/provenance/source-f16-provenance.txt"
sidecar_sha=$(sha256sum "$OUTPUT_DIR/provenance/source-f16-provenance.txt" | awk '{print $1}')
{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'model_q8=%s\nmodel_q8_bytes=%s\n' "$MODEL_Q8" "$(stat -c %s "$MODEL_Q8")"
    printf 'model_source_f16=%s\nmodel_source_f16_bytes=%s\n' \
        "$MODEL_SOURCE_F16" "$(stat -c %s "$MODEL_SOURCE_F16")"
    printf 'prompt_source=%s\nquality_manifest_source=%s\n' "$PROMPT" "$QUALITY_MANIFEST"
    printf 'source_f16_provenance=provenance/source-f16-provenance.txt\n'
    printf 'source_f16_provenance_sha256=%s\n' "$sidecar_sha"
    printf 'inventory=inventory/comparison.json\n'
    printf 'q8_tensor_inventory=inventory/model-q8.csv\n'
    printf 'source_f16_tensor_inventory=inventory/model-source-f16.csv\n'
    printf 'runs_table=runs.tsv\n'
    printf 'artifact_path_mode=root-relative\n'
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'cuda_device_order=PCI_BUS_ID\n'
    printf 'cuda_visible_devices=unset-required\n'
    printf 'nvidia_visible_devices=unset-required\n'
    printf 'required_direct_peer_links=0->1,1->0,2->3,3->2\n'
    printf 'quality_ctx=%s\nprefill_chunk=%s\ncontexts=2048,8192,32768\n' \
        "$QUALITY_CTX" "$PREFILL_CHUNK"
    printf 'ctx_alloc=%s\ngen_tokens=%s\nrepeats=%s\n' "$CTX_ALLOC" "$GEN_TOKENS" "$REPEATS"
    printf 'required_arms=native-q8,source-f16\nrotation=exact-ab-ba\n'
    printf 'q8_derived_diagnostic=not-run-not-required\n'
    printf 'dense_tensor_inventory=344\ndense_audit=coverage-only\ntimed_dense_audit=disabled\n'
    printf 'structural_integrity=required\nsource_provenance=required\n'
    printf 'source_f16_lock=shared-nonblocking-held\n'
    printf 'model_file_identity=bytes-device-inode-mtime\n'
    printf 'hf_shard_content_authentication=not_performed\n'
    printf 'non_target_payload_identity=producer-copy-guarantee-not-independently-rehashed\n'
    printf 'model_payload_hashing=disabled\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
} >"$OUTPUT_DIR/manifest.txt"
env | awk -F= '$1 ~ /^DS4_/ {print $0}' | sort -u \
    >"$OUTPUT_DIR/provenance/inherited-ds4-env.txt"

phase=model-inventory
python3 speed-bench/inspect-dense-f16-models.py \
    --model-q8 "$MODEL_Q8" --model-source-f16 "$MODEL_SOURCE_F16" \
    --out-dir "$OUTPUT_DIR/inventory" \
    | tee "$OUTPUT_DIR/inventory/validation.log"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench \
        gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench gguf-tools/quality-testing/score_official \
        tests/test_engine_mgpu_placement tests/test_gpu_xdev \
        CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found stale targets"
fi

phase=reproducibility
git_excludes=()
if [[ $OUTPUT_DIR == "$repo_dir/"* ]]; then
    output_relative=${OUTPUT_DIR#"$repo_dir/"}
    git_excludes=(":(exclude)$output_relative" ":(exclude)$output_relative/**")
fi
git status --short --untracked-files=all -- . "${git_excludes[@]}" \
    >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --binary HEAD -- . "${git_excludes[@]}" \
    >"$OUTPUT_DIR/provenance/git-head.patch"
git ls-files --others --exclude-standard -- \
    '*.c' '*.cu' '*.h' '*.py' '*.sh' '*.md' '*.yml' '*.yaml' '*.toml' \
    ':(glob)**/Makefile' "${git_excludes[@]}" \
    >"$OUTPUT_DIR/provenance/untracked-source-files.txt"
if [[ -s $OUTPUT_DIR/provenance/untracked-source-files.txt ]]; then
    while IFS= read -r source_path; do
        [[ -n $source_path && $source_path != -* && $source_path != /* && \
           $source_path != *'../'* && \
           -f $source_path && ! -L $source_path ]] ||
            die "unsafe or unreadable untracked source path: $source_path"
    done <"$OUTPUT_DIR/provenance/untracked-source-files.txt"
    tar -czf "$OUTPUT_DIR/provenance/untracked-sources.tar.gz" \
        -T "$OUTPUT_DIR/provenance/untracked-source-files.txt"
else
    printf 'none\n' >"$OUTPUT_DIR/provenance/untracked-sources.none"
fi
{
    uname -a
    nvcc --version
    cc --version
    make --version
    python3 --version
    nvidia-smi
} >"$OUTPUT_DIR/provenance/toolchain-runtime.txt" 2>&1
sha256sum ./ds4-bench ./gguf-tools/quality-testing/score_official \
    ./tests/test_engine_mgpu_placement ./tests/test_gpu_xdev \
    >"$OUTPUT_DIR/provenance/binary-sha256.txt"

phase=dense-audit-interface-preflight
marker_dump="$OUTPUT_DIR/.dense-audit-marker-strings"
for binary in ./ds4-bench ./gguf-tools/quality-testing/score_official; do
    strings "$binary" >"$marker_dump" ||
        die "failed to inspect dense-audit markers in $binary"
    for marker in DS4_CUDA_DENSE_EXEC_AUDIT_CSV native_q8 source_f16 \
                  sm75_native_q8 sm75_native_q8_fused_shared_mid \
                  cublas_gemm_ex; do
        grep -Fq "$marker" "$marker_dump" ||
            die "$binary lacks required dense-audit marker: $marker"
    done
done
rm -f -- "$marker_dump"

phase=tests
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/planner-unit.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/planner-unit.log" >&2
    die "planner unit test failed"
}
"${clean[@]}" ./tests/test_gpu_xdev >"$OUTPUT_DIR/gpu-exactness.log" 2>&1 || {
    tail -n 200 "$OUTPUT_DIR/gpu-exactness.log" >&2
    die "multi-GPU exactness test failed"
}

run_quality() {
    local arm=$1 model out log audit memory tensor_inventory
    model=$(arm_model "$arm")
    out="$OUTPUT_DIR/quality/$arm.tsv"
    log="$OUTPUT_DIR/quality/$arm.log"
    audit="$OUTPUT_DIR/quality/$arm.dense-audit.csv"
    memory="$OUTPUT_DIR/memory/quality-$arm.csv"
    set_arm_env "$arm"
    case $arm in
        native-q8) tensor_inventory="$OUTPUT_DIR/inventory/model-q8.csv" ;;
        source-f16) tensor_inventory="$OUTPUT_DIR/inventory/model-source-f16.csv" ;;
        *) die "unknown arm: $arm" ;;
    esac
    printf 'Scoring 100 production-path quality cases: %s...\n' "$arm"
    start_memory_sampler "$memory"
    if ! "${clean[@]}" "${arm_env[@]}" "${common_env[@]}" \
            "DS4_CUDA_DENSE_EXEC_AUDIT_CSV=$audit" \
            ./gguf-tools/quality-testing/score_official \
                "$model" "$QUALITY_MANIFEST" "$out" "$QUALITY_CTX" \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --cuda-tensor-parallel --warm-weights --production-path \
                2>&1 | tee "$log"; then
        stop_memory_sampler
        die "$arm quality scorer failed"
    fi
    stop_memory_sampler
    [[ -s $out && -s $audit && -s $memory ]] ||
        die "$arm omitted score, dense audit, or memory evidence"
    awk -F'\t' 'NR > 1 {n++} END {exit n == 100 ? 0 : 1}' "$out" ||
        die "$arm quality output does not contain exactly 100 cases"
    grep -Fq 'score_official: runtime_path=production' "$log" ||
        die "$arm quality run did not use production dispatch"
    grep -Fq 'CUDA EP forced pipeline split 22/21' "$log" ||
        die "$arm quality run did not use the fixed split"
    require_direct_peer_log "$log"
    python3 speed-bench/summarize-dense-f16-decode-ab.py \
        --validate-audit "$audit" --arm "$arm" \
        --inventory "$tensor_inventory" \
        >"$OUTPUT_DIR/quality/$arm.coverage.json"
}

for arm in "${arms[@]}"; do
    phase="quality-$arm"
    run_quality "$arm"
done
python3 gguf-tools/quality-testing/compare_scores.py \
    "$OUTPUT_DIR/quality/native-q8.tsv" "$OUTPUT_DIR/quality/source-f16.tsv" \
    >"$OUTPUT_DIR/quality/native-vs-source-f16.txt"

printf 'repeat\tslot\tarm\tcsv\tlog\tmemory\n' >"$OUTPUT_DIR/runs.tsv"
phase=timed-prefill-decode
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    offset=$(( (repeat - 1) % 2 ))
    for ((slot=0; slot<2; slot++)); do
        arm=${arms[$(( (slot + offset) % 2 ))]}
        model=$(arm_model "$arm")
        set_arm_env "$arm"
        stem="$arm-r$repeat"
        csv_rel="runs/$stem.csv"
        log_rel="runs/$stem.log"
        memory_rel="memory/$stem.csv"
        csv="$OUTPUT_DIR/$csv_rel"
        log="$OUTPUT_DIR/$log_rel"
        memory="$OUTPUT_DIR/$memory_rel"
        printf 'Benchmarking %s repeat=%d/%d slot=%d/2...\n' \
            "$arm" "$repeat" "$REPEATS" "$((slot+1))"
        start_memory_sampler "$memory"
        if ! "${clean[@]}" "${arm_env[@]}" "${common_env[@]}" \
                "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$model" --prompt-file "$PROMPT" \
                    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                    --ctx-alloc "$CTX_ALLOC" --step-mul "$STEP_MUL" \
                    --prefill-chunk "$PREFILL_CHUNK" \
                    --gen-tokens "$GEN_TOKENS" --csv "$csv" \
                    >"$log" 2>&1; then
            stop_memory_sampler
            tail -n 200 "$log" >&2 || true
            die "$stem failed"
        fi
        stop_memory_sampler
        [[ -s $csv && -s $log && -s $memory ]] ||
            die "$stem omitted benchmark, log, or memory evidence"
        require_direct_peer_log "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$((slot+1))" "$arm" "$csv_rel" "$log_rel" "$memory_rel" \
            >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=post-run-model-identity
verify_model_file_identities

phase=summarize
python3 speed-bench/summarize-dense-f16-decode-ab.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/dense-f16-decode-ab-results.json && -s $OUTPUT_DIR/summary.md ]] ||
    die "two-arm summary is missing"

phase=complete
printf 'Dense-F16 native-Q8/source-F16 A/B complete: %s\n' "$OUTPUT_DIR"
