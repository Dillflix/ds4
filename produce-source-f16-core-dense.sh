#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Build and validate a GGUF whose 43x8 DeepSeek V4 Flash core-dense tensors are
converted directly from the HF checkpoint representation to checked F16.
All other tensor payloads and all GGUF KV records come from TEMPLATE verbatim.

Usage:
  bash ./produce-source-f16-core-dense.sh HF_DIR TEMPLATE.gguf OUTPUT.gguf

Controls:
  OVERWRITE=1       atomically replace an existing OUTPUT after validation
  MIN_FREE_MIB=1024 require this much free space after the planned output
  MAKE_JOBS=N       build parallelism (default: nproc)
  KEEP_PLAN=1       retain the dry-run plan beside OUTPUT (default: 1)
  RECOVER_ONLY=1    recover/reap an interrupted OUTPUT transaction, then exit

The workflow never hashes or rereads tensor payloads for validation. It parses
only GGUF metadata/directories after generation. OUTPUT is published with a
same-filesystem rename only after all checks pass.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}
fsync_regular_files() {
    python3 -c '
import os, pathlib, sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"refusing to sync non-regular staged file: {path}")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
' "$@" || die "could not durably flush staged publication files"
}
capture_regular_file_identity() {
    python3 -c '
import pathlib, sys
path = pathlib.Path(sys.argv[1]).resolve(strict=True)
if not path.is_file() or path.is_symlink():
    raise SystemExit(f"not a regular resolved file: {path}")
text = str(path)
if any(character in text for character in "\t\r\n"):
    raise SystemExit(f"unsupported control character in path: {path!r}")
st = path.stat()
print("\t".join((text, str(st.st_dev), str(st.st_ino),
                 str(st.st_size), str(st.st_mtime_ns))))
' "$1"
}
capture_hf_shard_manifest() {
    python3 -c '
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
index = json.loads((root / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict):
    raise SystemExit("HF index weight_map is not an object")
names = sorted(set(weight_map.values()))
if not names or not all(isinstance(name, str) for name in names):
    raise SystemExit("HF index contains an invalid shard name")
print(f"hf_shard_count={len(names)}")
for number, name in enumerate(names):
    components = re.split(r"[/\\\\]", name)
    if (name.startswith(("/", "\\")) or
            any(component in ("", "..") for component in components) or
            any(character in name for character in "\r\n")):
        raise SystemExit(f"unsafe indexed shard path: {name!r}")
    path = root / name
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise SystemExit(f"indexed shard is not a regular resolved file: {resolved}")
    resolved_text = str(resolved)
    if any(character in resolved_text for character in "\t\r\n"):
        raise SystemExit(f"unsupported control character in shard path: {resolved!r}")
    st = resolved.stat()
    print(f"hf_shard_{number:03d}_name={name}")
    print(f"hf_shard_{number:03d}_resolved={resolved_text}")
    print(f"hf_shard_{number:03d}_resolved_basename={resolved.name}")
    print(f"hf_shard_{number:03d}_device={st.st_dev}")
    print(f"hf_shard_{number:03d}_inode={st.st_ino}")
    print(f"hf_shard_{number:03d}_bytes={st.st_size}")
    print(f"hf_shard_{number:03d}_mtime_ns={st.st_mtime_ns}")
' "$1"
}

[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
[[ $# == 3 ]] || { usage >&2; exit 2; }
[[ $(uname -s) == Linux ]] || die "this production workflow must run on Linux"
for command in make nproc python3 realpath stat df awk mv mkdir rm tee flock sha256sum git; do
    require_command "$command"
done

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hf_dir=$(realpath "$1")
template=$(realpath "$2")
output=$(realpath -m "$3")
[[ -d $hf_dir ]] || die "HF directory not found: $hf_dir"
[[ -f $hf_dir/model.safetensors.index.json ]] ||
    die "HF directory has no model.safetensors.index.json: $hf_dir"
[[ -f $hf_dir/config.json ]] || die "HF directory has no config.json: $hf_dir"
expected_hf_revision=${EXPECTED_HF_REVISION:-7872f01b1d1fe23eabc4c98b48bffcef5a386062}
[[ $expected_hf_revision =~ ^[0-9a-f]{40}$ ]] ||
    die "EXPECTED_HF_REVISION must be a 40-character lowercase commit id"
expected_hf_suffix="/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/$expected_hf_revision"
[[ $hf_dir == *"$expected_hf_suffix" ]] ||
    die "HF_DIR does not have the expected DeepSeek-V4-Flash-0731 snapshot path for $expected_hf_revision: $hf_dir"
[[ -f $template ]] || die "template GGUF not found: $template"
[[ $template != "$output" ]] || die "template and output paths must differ"

overwrite=${OVERWRITE:-0}
min_free_mib=${MIN_FREE_MIB:-1024}
make_jobs=${MAKE_JOBS:-$(nproc)}
keep_plan=${KEEP_PLAN:-1}
recover_only=${RECOVER_ONLY:-0}
[[ $overwrite == 0 || $overwrite == 1 ]] || die "OVERWRITE must be 0 or 1"
[[ $keep_plan == 0 || $keep_plan == 1 ]] || die "KEEP_PLAN must be 0 or 1"
[[ $recover_only == 0 || $recover_only == 1 ]] ||
    die "RECOVER_ONLY must be 0 or 1"
[[ $min_free_mib =~ ^[0-9]+$ ]] || die "MIN_FREE_MIB must be a nonnegative integer"
[[ $make_jobs =~ ^[1-9][0-9]*$ ]] || die "MAKE_JOBS must be a positive integer"

output_dir=$(dirname -- "$output")
mkdir -p -- "$output_dir"
output_dir=$(realpath "$output_dir")
lock_file="$output.lock"
[[ ! -L $lock_file && ( ! -e $lock_file || -f $lock_file ) ]] ||
    die "unsafe output-lock path: $lock_file"
exec 9>"$lock_file"
flock -n 9 || die "another source-F16 producer owns the output lock: $lock_file"
partial="$output.partial"
plan="$output.source-f16-plan.txt"
plan_partial="$plan.partial"
validation_prefix="${output%.gguf}.source-f16"
provenance="$validation_prefix.provenance.txt"
inventory="$validation_prefix.inventory.json"
template_inventory="$validation_prefix.template-inventory.csv"
model_inventory="$validation_prefix.model-inventory.csv"
temporary_validation="$output.validation.partial"
backup_dir="$output.publish-backup"
committed_backup_dir="$output.publish-committed-backup"
final_artifacts=(
    "$output" "$plan" "$inventory" "$template_inventory"
    "$model_inventory" "$provenance"
)
publish_started=0
publish_complete=0

rollback_publish_transaction() {
    [[ -d $backup_dir && ! -L $backup_dir ]] || return 0
    local originals_file="$backup_dir/originals.list"
    if [[ ! -e $originals_file && ! -L $originals_file ]]; then
        rm -rf -- "$backup_dir"
        return 0
    fi
    [[ -f $originals_file && ! -L $originals_file ]] || {
        printf 'error: unsafe source-F16 publication rollback journal: %s\n' \
            "$originals_file" >&2
        return 1
    }
    local -a original_indices=()
    local -A was_original=()
    mapfile -t original_indices <"$originals_file"
    local index
    for index in "${original_indices[@]}"; do
        [[ $index =~ ^[0-5]$ ]] || {
            printf 'error: corrupt source-F16 publication rollback journal: %s\n' \
                "$originals_file" >&2
            return 1
        }
        was_original[$index]=1
    done
    # Hide an uncommitted model before repairing sidecars, then restore an old
    # model last. If original index 0 has not reached the backup directory, the
    # publication loop (which also starts at index 0) cannot yet have touched
    # any other final artifact, so its still-final model is already coherent.
    local restore_model=0
    if [[ -n ${was_original[0]:-} ]]; then
        if [[ -e $backup_dir/0 || -L $backup_dir/0 ]]; then
            [[ -f $backup_dir/0 && ! -L $backup_dir/0 ]] || {
                printf 'error: unsafe rollback artifact: %s\n' \
                    "$backup_dir/0" >&2
                return 1
            }
            rm -f -- "${final_artifacts[0]}" || return 1
            restore_model=1
        elif [[ ! -f ${final_artifacts[0]} || -L ${final_artifacts[0]} ]]; then
            printf 'error: rollback lost original artifact %s\n' \
                "${final_artifacts[0]}" >&2
            return 1
        fi
    else
        rm -f -- "${final_artifacts[0]}" || return 1
    fi
    for index in 1 2 3 4 5; do
        if [[ -n ${was_original[$index]:-} ]]; then
            if [[ -e $backup_dir/$index || -L $backup_dir/$index ]]; then
                [[ -f $backup_dir/$index && ! -L $backup_dir/$index ]] || {
                    printf 'error: unsafe rollback artifact: %s\n' \
                        "$backup_dir/$index" >&2
                    return 1
                }
                rm -f -- "${final_artifacts[$index]}" || return 1
                mv -T -- "$backup_dir/$index" "${final_artifacts[$index]}" || return 1
            elif [[ ! -f ${final_artifacts[$index]} ||
                    -L ${final_artifacts[$index]} ]]; then
                printf 'error: rollback lost original artifact %s\n' \
                    "${final_artifacts[$index]}" >&2
                return 1
            fi
        else
            rm -f -- "${final_artifacts[$index]}" || return 1
        fi
    done
    if (( restore_model )); then
        mv -T -- "$backup_dir/0" "${final_artifacts[0]}" || return 1
    fi
    rm -rf -- "$backup_dir"
}

cleanup() {
    status=$?
    trap - EXIT HUP INT TERM
    set +e
    if (( publish_started && ! publish_complete )); then
        if ! rollback_publish_transaction; then
            printf 'error: publication rollback failed; inspect %s\n' \
                "$backup_dir" >&2
            status=2
        fi
    fi
    rm -f -- "$partial" "$plan_partial"
    rm -rf -- "$temporary_validation"
    if (( publish_complete )); then
        rm -rf -- "$committed_backup_dir"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -e $committed_backup_dir || -L $committed_backup_dir ]]; then
    [[ -d $committed_backup_dir && ! -L $committed_backup_dir ]] ||
        die "unsafe committed-backup path: $committed_backup_dir"
    printf 'Removing a committed publication backup left by an interrupted cleanup: %s\n' \
        "$committed_backup_dir"
    rm -rf -- "$committed_backup_dir"
fi
if [[ -e $backup_dir || -L $backup_dir ]]; then
    [[ -d $backup_dir && ! -L $backup_dir ]] ||
        die "unsafe publication-backup path: $backup_dir"
    printf 'Rolling back an interrupted source-F16 publication: %s\n' "$backup_dir"
    rollback_publish_transaction ||
        die "could not roll back interrupted publication; inspect $backup_dir"
fi

shopt -s nullglob
legacy_partials=("$output".partial.[0-9]*)
shopt -u nullglob
for stale_partial in "$partial" "${legacy_partials[@]}"; do
    [[ -e $stale_partial || -L $stale_partial ]] || continue
    [[ -f $stale_partial && ! -L $stale_partial ]] ||
        die "refusing to remove unsafe stale partial path: $stale_partial"
    printf 'Removing stale source-F16 partial left without the output lock: %s\n' \
        "$stale_partial"
    rm -f -- "$stale_partial"
done
rm -f -- "$plan_partial"
[[ ! -e $temporary_validation && ! -L $temporary_validation ]] || {
    [[ -d $temporary_validation && ! -L $temporary_validation ]] ||
        die "unsafe stale validation path: $temporary_validation"
    rm -rf -- "$temporary_validation"
}
if [[ $recover_only == 1 ]]; then
    printf 'Source-F16 publication recovery complete for: %s\n' "$output"
    exit 0
fi

# Recovery must run under the output lock before this policy check. In
# particular, an interrupted overwrite may have moved the former model into
# the rollback directory and already placed an uncommitted model at OUTPUT.
[[ ! -e $output || $overwrite == 1 ]] ||
    die "output exists; set OVERWRITE=1 to replace it after validation: $output"
for final_path in "${final_artifacts[@]}"; do
    [[ ! -L $final_path ]] ||
        die "refusing a symlink publication artifact: $final_path"
    [[ ! -e $final_path || -f $final_path ]] ||
        die "refusing a non-regular publication artifact: $final_path"
done

# Bound the long conversion with cheap source identities. The large model
# payloads are intentionally not hashed; device/inode/size/mtime_ns detect an
# ordinary replacement or mutation during this run. The small index and config
# are hashed as well.
hf_index_identity_before=$(capture_regular_file_identity \
    "$hf_dir/model.safetensors.index.json") ||
    die "could not capture HF index identity"
IFS=$'\t' read -r hf_index hf_index_device hf_index_inode \
    hf_index_bytes hf_index_mtime_ns <<<"$hf_index_identity_before"
hf_config_identity_before=$(capture_regular_file_identity "$hf_dir/config.json") ||
    die "could not capture HF config identity"
IFS=$'\t' read -r hf_config hf_config_device hf_config_inode \
    hf_config_bytes hf_config_mtime_ns <<<"$hf_config_identity_before"
template_identity_before=$(capture_regular_file_identity "$template") ||
    die "could not capture template identity"
IFS=$'\t' read -r template_identity_path template_device template_inode \
    template_bytes template_mtime_ns <<<"$template_identity_before"
[[ $template_identity_path == "$template" ]] ||
    die "resolved template identity path changed unexpectedly"
hf_index_sha_line_before=$(sha256sum "$hf_index") || die "could not hash HF index"
hf_index_sha256=${hf_index_sha_line_before%% *}
[[ $hf_index_sha256 =~ ^[0-9a-f]{64}$ ]] || die "invalid HF-index SHA-256"
hf_config_sha_line_before=$(sha256sum "$hf_config") || die "could not hash HF config"
hf_config_sha256=${hf_config_sha_line_before%% *}
[[ $hf_config_sha256 =~ ^[0-9a-f]{64}$ ]] || die "invalid HF-config SHA-256"
hf_shard_manifest=$(capture_hf_shard_manifest "$hf_dir") ||
    die "could not capture HF shard identities"

cd "$repo_dir"
printf 'Building and testing the checked source-F16 converter...\n'
make -C gguf-tools -j "$make_jobs" deepseek4-quantize test-source-f16-checked
make -C gguf-tools check-source-f16
quantizer_identity_before=$(capture_regular_file_identity \
    ./gguf-tools/deepseek4-quantize) || die "could not capture quantizer identity"
IFS=$'\t' read -r quantizer_path quantizer_device quantizer_inode \
    quantizer_bytes quantizer_mtime_ns <<<"$quantizer_identity_before"
quantizer_sha_line_before=$(sha256sum "$quantizer_path") ||
    die "could not hash the small quantizer binary"
quantizer_sha256=${quantizer_sha_line_before%% *}
[[ $quantizer_sha256 =~ ^[0-9a-f]{64}$ ]] || die "invalid quantizer SHA-256"

printf 'Planning the exact 344-tensor rewrite...\n'
./gguf-tools/deepseek4-quantize \
    --hf "$hf_dir" \
    --template "$template" \
    --source-f16-core-dense \
    --dry-run | tee "$plan_partial"

planned_bytes=$(awk -F': ' '$1 == "approx_file_bytes" {value=$2} END {print value}' "$plan_partial")
[[ $planned_bytes =~ ^[1-9][0-9]*$ ]] || die "dry run did not report approx_file_bytes"
selected=$(awk -F': ' '$1 == "source_f16_selected" {value=$2} END {print value}' "$plan_partial")
[[ $selected == 344 ]] || die "dry run selected ${selected:-0} tensors, expected 344"
selected_bytes=$(awk -F': ' '$1 == "source_f16_selected_bytes" {value=$2} END {print value}' "$plan_partial")
[[ $selected_bytes == 11362369536 ]] ||
    die "dry run planned ${selected_bytes:-0} selected F16 bytes, expected 11362369536"
q8_sources=$(awk '$1 == "source_f16_tensor:" && $(NF-2) == "q8_0" && $(NF-1) == "->" && $NF == "f16" {n++} END {print n+0}' "$plan_partial")
[[ $q8_sources == 344 ]] ||
    die "template has only $q8_sources/344 Q8_0 core-dense tensors; the fixed comparison requires 344"
hf_headers=$(awk -F': ' '$1 == "source_f16_hf_headers_validated" {value=$2} END {print value}' "$plan_partial")
[[ $hf_headers == 344 ]] ||
    die "HF preflight validated ${hf_headers:-0}/344 core-dense tensor headers"
available_bytes=$(df -PB1 --output=avail "$output_dir" | awk 'NR == 2 {gsub(/[[:space:]]/, "", $1); print $1}')
[[ $available_bytes =~ ^[0-9]+$ ]] || die "could not determine free space for $output_dir"
reserve_bytes=$((min_free_mib * 1024 * 1024))
required_bytes=$((planned_bytes + reserve_bytes))
(( available_bytes >= required_bytes )) ||
    die "insufficient free space in $output_dir: available=$available_bytes required=$required_bytes"

printf 'Generating checked source-derived F16 model at a temporary path...\n'
./gguf-tools/deepseek4-quantize \
    --hf "$hf_dir" \
    --template "$template" \
    --out "$partial" \
    --source-f16-core-dense

source_f16_identity_before_publish=$(capture_regular_file_identity "$partial") ||
    die "could not capture generated model identity"
IFS=$'\t' read -r source_f16_identity_path source_f16_device source_f16_inode \
    actual_bytes source_f16_mtime_ns <<<"$source_f16_identity_before_publish"
[[ $source_f16_identity_path == "$partial" ]] ||
    die "generated model resolved outside its fixed partial path"
[[ $actual_bytes == "$planned_bytes" ]] ||
    die "generated size $actual_bytes differs from planned size $planned_bytes"
source_f16_mtime_epoch=$((source_f16_mtime_ns / 1000000000))

printf 'Validating tensor inventory, non-target directory parity, and metadata identity...\n'
python3 speed-bench/inspect-dense-f16-models.py \
    --model-q8 "$template" \
    --model-source-f16 "$partial" \
    --out-dir "$temporary_validation"

date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) || die "could not record UTC date"
git_commit=$(git rev-parse HEAD) || die "could not record git commit"
git_status=$(git status --porcelain --untracked-files=normal) ||
    die "could not determine git worktree status"
if [[ -n $git_status ]]; then git_dirty=true; else git_dirty=false; fi
quantizer_identity_after=$(capture_regular_file_identity \
    ./gguf-tools/deepseek4-quantize) || die "could not recapture quantizer identity"
[[ $quantizer_identity_after == "$quantizer_identity_before" ]] ||
    die "quantizer identity changed during source-F16 conversion"
quantizer_sha_line_after=$(sha256sum "$quantizer_path") ||
    die "could not rehash the small quantizer binary"
quantizer_sha256_after=${quantizer_sha_line_after%% *}
[[ $quantizer_sha256_after == "$quantizer_sha256" ]] ||
    die "quantizer contents changed during source-F16 conversion"
hf_index_identity_after=$(capture_regular_file_identity \
    "$hf_dir/model.safetensors.index.json") ||
    die "could not recapture HF index identity"
hf_config_identity_after=$(capture_regular_file_identity "$hf_dir/config.json") ||
    die "could not recapture HF config identity"
template_identity_after=$(capture_regular_file_identity "$template") ||
    die "could not recapture template identity"
hf_shard_manifest_after=$(capture_hf_shard_manifest "$hf_dir") ||
    die "could not recapture HF shard identities"
source_f16_identity_after_validation=$(capture_regular_file_identity "$partial") ||
    die "could not recapture generated model identity"
[[ $hf_index_identity_after == "$hf_index_identity_before" ]] ||
    die "HF index identity changed during source-F16 conversion"
[[ $hf_config_identity_after == "$hf_config_identity_before" ]] ||
    die "HF config identity changed during source-F16 conversion"
[[ $template_identity_after == "$template_identity_before" ]] ||
    die "template identity changed during source-F16 conversion"
[[ $hf_shard_manifest_after == "$hf_shard_manifest" ]] ||
    die "one or more HF shard identities changed during source-F16 conversion"
[[ $source_f16_identity_after_validation == "$source_f16_identity_before_publish" ]] ||
    die "generated model identity changed during validation"
hf_index_sha_line_after=$(sha256sum "$hf_index") ||
    die "could not rehash HF index"
hf_index_sha256_after=${hf_index_sha_line_after%% *}
[[ $hf_index_sha256_after == "$hf_index_sha256" ]] ||
    die "HF index contents changed during source-F16 conversion"
hf_config_sha_line_after=$(sha256sum "$hf_config") ||
    die "could not rehash HF config"
hf_config_sha256_after=${hf_config_sha_line_after%% *}
[[ $hf_config_sha256_after == "$hf_config_sha256" ]] ||
    die "HF config contents changed during source-F16 conversion"
hf_index_mtime_epoch=$((hf_index_mtime_ns / 1000000000))
hf_config_mtime_epoch=$((hf_config_mtime_ns / 1000000000))
template_mtime_epoch=$((template_mtime_ns / 1000000000))

{
    printf 'date_utc=%s\n' "$date_utc"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'git_dirty=%s\n' "$git_dirty"
    printf 'quantizer=%s\n' "$quantizer_path"
    printf 'quantizer_bytes=%s\n' "$quantizer_bytes"
    printf 'quantizer_device=%s\n' "$quantizer_device"
    printf 'quantizer_inode=%s\n' "$quantizer_inode"
    printf 'quantizer_mtime_ns=%s\n' "$quantizer_mtime_ns"
    printf 'quantizer_sha256=%s\n' "$quantizer_sha256"
    printf 'hf_repository=deepseek-ai/DeepSeek-V4-Flash-0731\n'
    printf 'hf_revision=%s\n' "$expected_hf_revision"
    printf 'hf_snapshot_path_revision_match=true\n'
    printf 'hf_shard_content_authentication=not_performed\n'
    printf 'hf_directory=%s\n' "$hf_dir"
    printf 'hf_index=%s\n' "$hf_index"
    printf 'hf_index_bytes=%s\n' "$hf_index_bytes"
    printf 'hf_index_device=%s\n' "$hf_index_device"
    printf 'hf_index_inode=%s\n' "$hf_index_inode"
    printf 'hf_index_mtime_epoch=%s\n' "$hf_index_mtime_epoch"
    printf 'hf_index_mtime_ns=%s\n' "$hf_index_mtime_ns"
    printf 'hf_index_sha256=%s\n' "$hf_index_sha256"
    printf 'hf_config=%s\n' "$hf_config"
    printf 'hf_config_bytes=%s\n' "$hf_config_bytes"
    printf 'hf_config_device=%s\n' "$hf_config_device"
    printf 'hf_config_inode=%s\n' "$hf_config_inode"
    printf 'hf_config_mtime_epoch=%s\n' "$hf_config_mtime_epoch"
    printf 'hf_config_mtime_ns=%s\n' "$hf_config_mtime_ns"
    printf 'hf_config_sha256=%s\n' "$hf_config_sha256"
    printf '%s\n' "$hf_shard_manifest"
    printf 'template_model=%s\n' "$template"
    printf 'template_model_bytes=%s\n' "$template_bytes"
    printf 'template_model_device=%s\n' "$template_device"
    printf 'template_model_inode=%s\n' "$template_inode"
    printf 'template_model_mtime_epoch=%s\n' "$template_mtime_epoch"
    printf 'template_model_mtime_ns=%s\n' "$template_mtime_ns"
    printf 'source_f16_model=%s\n' "$output"
    printf 'source_f16_model_bytes=%s\n' "$actual_bytes"
    printf 'source_f16_model_device=%s\n' "$source_f16_device"
    printf 'source_f16_model_inode=%s\n' "$source_f16_inode"
    printf 'source_f16_model_mtime_epoch=%s\n' "$source_f16_mtime_epoch"
    printf 'source_f16_model_mtime_ns=%s\n' "$source_f16_mtime_ns"
    printf 'source_f16_selected=344\n'
    printf 'source_f16_selected_bytes=11362369536\n'
    printf 'template_q8_selected=344\n'
    printf 'kv_metadata_byte_identical=true\n'
    printf 'payload_extents_valid=true\n'
    printf 'non_target_payload_copy=producer-guaranteed-not-independently-rehashed\n'
    printf 'model_payload_hashing=disabled\n'
    printf 'source_identity_pre_post_stable=true\n'
    printf 'publication_protocol=model-last-rollback-journal-v1\n'
    printf 'provenance_complete=true\n'
} >"$temporary_validation/provenance.txt"

staged_sidecars=(
    "$temporary_validation/comparison.json"
    "$temporary_validation/model-q8.csv"
    "$temporary_validation/model-source-f16.csv"
    "$temporary_validation/provenance.txt"
)
if [[ $keep_plan == 1 ]]; then staged_sidecars+=("$plan_partial"); fi
fsync_regular_files "${staged_sidecars[@]}"

for final_path in "${final_artifacts[@]}"; do
    [[ ! -L $final_path && ( ! -e $final_path || -f $final_path ) ]] ||
        die "refusing to replace non-regular publication artifact: $final_path"
done
[[ ! -e $backup_dir && ! -L $backup_dir ]] ||
    die "publication-backup path reappeared while the output lock was held: $backup_dir"
[[ ! -e $committed_backup_dir && ! -L $committed_backup_dir ]] ||
    die "committed-backup path reappeared while the output lock was held: $committed_backup_dir"
mkdir -- "$backup_dir"
originals_partial="$backup_dir/originals.list.partial"
{
    for index in 0 1 2 3 4 5; do
        if [[ -e ${final_artifacts[$index]} ]]; then
            printf '%s\n' "$index"
        fi
    done
} >"$originals_partial"
fsync_regular_files "$originals_partial"
mv -T -- "$originals_partial" "$backup_dir/originals.list"
publish_started=1
for index in 0 1 2 3 4 5; do
    if [[ -e ${final_artifacts[$index]} ]]; then
        mv -T -- "${final_artifacts[$index]}" "$backup_dir/$index"
    fi
done

if [[ $keep_plan == 1 ]]; then mv -T -- "$plan_partial" "$plan"; fi
mv -T -- "$temporary_validation/comparison.json" "$inventory"
mv -T -- "$temporary_validation/model-q8.csv" "$template_inventory"
mv -T -- "$temporary_validation/model-source-f16.csv" "$model_inventory"
mv -T -- "$temporary_validation/provenance.txt" "$provenance"
# The model is the commit record and is deliberately published last. Every
# sidecar is already complete, and the rollback journal still owns all prior
# artifacts until this same-filesystem rename succeeds.
mv -T -- "$partial" "$output"
mv -T -- "$backup_dir" "$committed_backup_dir"
publish_complete=1
rm -rf -- "$committed_backup_dir"
rm -rf -- "$temporary_validation"

printf 'Published source-derived F16 model: %s\n' "$output"
printf 'Model bytes: %s\n' "$actual_bytes"
printf 'Validation summary: %s\n' "$inventory"
printf 'Source provenance: %s\n' "$provenance"
printf 'Payload hashing: disabled\n'
