#!/usr/bin/env bash
# File/tool/package and optional firmware/BMC inventory ONLY. Never executes the
# diagnostic, initializes CUDA, runs nvidia-smi, or changes hardware settings.
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
repo_dir=$(cd -- "$script_dir/.." && pwd -P) || exit 2
executable="$repo_dir/tests/cuda_sm75_token_row_arithmetic"
expected_sha256=5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414
root_inventory=${ROOT_HOST_INVENTORY:-0}
include_executable=${INCLUDE_EXECUTABLE:-0}
query_seconds=${INVENTORY_QUERY_TIMEOUT_SECONDS:-20}
root_seconds=${INVENTORY_ROOT_TIMEOUT_SECONDS:-30}

if [[ "$root_inventory" != 0 && "$root_inventory" != 1 ]] ||
   [[ "$include_executable" != 0 && "$include_executable" != 1 ]]; then
    printf 'error: ROOT_HOST_INVENTORY and INCLUDE_EXECUTABLE must be 0 or 1\n' >&2
    exit 2
fi
for value in "$query_seconds" "$root_seconds"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || (( ${#value} > 3 )) || (( value > 120 )); then
        printf 'error: inventory query timeouts must be integers from 1 to 120\n' >&2
        exit 2
    fi
done
# timeout is mandatory: do not silently turn a bounded query into an unbounded one.
for tool in timeout tar mktemp mkdir date dirname basename tee cp mv rm wc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'error: required inventory utility missing: %s\n' "$tool" >&2
        exit 2
    fi
done
output_dir=$(mktemp -d "$repo_dir/sm75-investigation-inventory.XXXXXX") || exit 2
mkdir -- "$output_dir/queries" || exit 2
manifest="$output_dir/manifest.txt"
status_file="$output_dir/query-status.tsv"
printf 'id\tstatus\texit_code\tstarted_utc\tfinished_utc\tcommand\n' > "$status_file"
printf 'schema=sm75-read-only-inventory-v1\nworkload_executed=0\n' > "$manifest"
printf 'created_utc=%s\nroot_host_inventory_requested=%s\n' "$(date -u +%FT%TZ)" "$root_inventory" >> "$manifest"
printf 'include_executable_requested=%s\nexecutable_copy_limit_bytes=134217728\n' "$include_executable" >> "$manifest"
printf 'expected_executable_sha256=%s\nexecutable=%s\n' "$expected_sha256" "$executable" >> "$manifest"
printf 'query_timeout_seconds=%s\nroot_query_timeout_seconds=%s\n' "$query_seconds" "$root_seconds" >> "$manifest"
printf 'scope=file-inspection,tool-help,package-metadata,allowlisted-environment,optional-read-only-firmware-BMC\n' >> "$manifest"
printf 'prohibited_actions=execute-diagnostic,CUDA-workload,nvidia-smi,reset,retrain,setpci,clocks,power-changes,install,network\n' >> "$manifest"
printf 'Read-only inventory: %s\nNo GPU workload will be executed.\n' "$output_dir"

gap_count=0
root_auth_failed=0
copy_failed=0
query_status=unknown
query_exit=0

record_status() {
    local id=$1 status=$2 code=$3 started=$4 finished=$5
    shift 5
    local formatted
    printf -v formatted '%q ' "$@"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$status" "$code" "$started" "$finished" "$formatted" >> "$status_file"
}

skip_query() {
    local id=$1 status=$2 message=$3
    shift 3
    local stamp
    stamp=$(date -u +%FT%TZ)
    printf '%s\n' "$message" > "$output_dir/queries/$id.log"
    record_status "$id" "$status" '-' "$stamp" "$stamp" "$@"
    query_status=$status
    query_exit=0
    gap_count=$((gap_count + 1))
}

run_query() {
    local id=$1 seconds=$2 tool=$3
    shift 3
    local resolved started finished code
    if ! resolved=$(command -v "$tool" 2>/dev/null); then
        skip_query "$id" missing-tool "Optional tool unavailable on PATH: $tool" "$tool" "$@"
        return 0
    fi
    started=$(date -u +%FT%TZ)
    timeout --signal=TERM --kill-after=2 "$seconds" "$resolved" "$@" > "$output_dir/queries/$id.log" 2>&1
    code=$?
    finished=$(date -u +%FT%TZ)
    query_exit=$code
    if (( code == 0 )); then
        query_status=ok
    elif (( code == 124 || code == 137 )); then
        query_status=timeout
        gap_count=$((gap_count + 1))
    else
        query_status=failed
        gap_count=$((gap_count + 1))
    fi
    record_status "$id" "$query_status" "$code" "$started" "$finished" "$resolved" "$@"
    return 0
}

fingerprint=missing
if [[ -f "$executable" ]]; then
    run_query executable-sha256 "$query_seconds" sha256sum "$executable"
    if [[ "$query_status" == ok ]]; then
        actual_sha256=''
        read -r actual_sha256 _ < "$output_dir/queries/executable-sha256.log" || true
        if [[ "$actual_sha256" == "$expected_sha256" ]]; then
            fingerprint=match
        else
            fingerprint=mismatch
            gap_count=$((gap_count + 1))
        fi
        printf 'actual_executable_sha256=%s\n' "$actual_sha256" >> "$manifest"
    else
        fingerprint=unavailable
    fi
else
    skip_query executable-sha256 missing-file "Pinned executable is absent; no executable was run." sha256sum "$executable"
fi
printf 'executable_fingerprint=%s\n' "$fingerprint" >> "$manifest"

# Optional byte-for-byte evidence, not a build or execution. Do not copy a
# fingerprint mismatch, unknown file, or unexpectedly large file into a handoff.
copy_state=not-requested
if [[ "$include_executable" == 1 ]]; then
    if [[ "$fingerprint" != match ]]; then
        copy_state=rejected-fingerprint
        copy_failed=1
        skip_query executable-copy rejected-fingerprint 'Requested executable copy rejected: frozen fingerprint did not match.' "$executable"
    else
        run_query executable-size "$query_seconds" wc -c "$executable"
        executable_bytes=''
        read -r executable_bytes _ < "$output_dir/queries/executable-size.log" || true
        if [[ "$query_status" != ok || ! "$executable_bytes" =~ ^[0-9]+$ ]] ||
           (( ${#executable_bytes} > 9 )) || (( executable_bytes > 134217728 )); then
            copy_state=rejected-size
            copy_failed=1
            skip_query executable-copy rejected-size 'Requested executable copy rejected: unavailable size or greater than 128 MiB.' "$executable"
        else
            printf 'executable_size_bytes=%s\n' "$executable_bytes" >> "$manifest"
            mkdir -- "$output_dir/artifacts"
            partial_copy="$output_dir/artifacts/.executable-copy.partial"
            run_query executable-copy "$query_seconds" cp -- "$executable" "$partial_copy"
            if [[ "$query_status" != ok ]]; then
                copy_state=copy-failed
                copy_failed=1
                rm -f -- "$partial_copy"
            else
                run_query executable-copy-sha256 "$query_seconds" sha256sum "$partial_copy"
                copied_sha256=''
                read -r copied_sha256 _ < "$output_dir/queries/executable-copy-sha256.log" || true
                if [[ "$query_status" != ok || "$copied_sha256" != "$expected_sha256" ]]; then
                    copy_state=verification-failed
                    copy_failed=1
                    gap_count=$((gap_count + 1))
                    # Only our newly created temporary copy is removed.
                    rm -f -- "$partial_copy"
                else
                    run_query executable-copy-promote "$query_seconds" mv -- "$partial_copy" "$output_dir/artifacts/cuda_sm75_token_row_arithmetic"
                    if [[ "$query_status" == ok ]]; then
                        copy_state=verified
                        printf 'copied_executable_sha256=%s\n' "$copied_sha256" >> "$manifest"
                    else
                        copy_state=copy-failed
                        copy_failed=1
                        rm -f -- "$partial_copy"
                    fi
                fi
            fi
        fi
    fi
fi
printf 'executable_copy=%s\n' "$copy_state" >> "$manifest"

if [[ -f "$executable" ]]; then
    run_query elf-header "$query_seconds" readelf -hW "$executable"
    run_query elf-dynamic "$query_seconds" readelf -dW "$executable"
    run_query elf-notes "$query_seconds" readelf -nW "$executable"
    run_query elf-compiler-comment "$query_seconds" readelf -p .comment "$executable"
    run_query embedded-elf-list "$query_seconds" cuobjdump --list-elf "$executable"
    run_query embedded-ptx-list "$query_seconds" cuobjdump --list-ptx "$executable"
else
    for id in elf-header elf-dynamic elf-notes elf-compiler-comment embedded-elf-list embedded-ptx-list; do
        skip_query "$id" missing-file "File inspection skipped: pinned executable absent." "$executable"
    done
fi

# Current tool versions are not proof of the frozen executable's build version.
run_query cuobjdump-version "$query_seconds" cuobjdump --version
run_query nvcc-version "$query_seconds" nvcc --version
run_query nsys-version "$query_seconds" nsys --version
run_query nsys-profile-help "$query_seconds" nsys profile --help

# Only known control variables: never dump the complete environment or credentials.
printf 'name\tstate\tshell_escaped_value\n' > "$output_dir/environment-controls.tsv"
for name in CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING \
            CUDA_MODULE_LOADING CUDA_FORCE_PTX_JIT CUDA_DISABLE_PTX_JIT \
            CUDA_DEVICE_MAX_CONNECTIONS NVIDIA_TF32_OVERRIDE \
            CUBLAS_WORKSPACE_CONFIG LD_LIBRARY_PATH; do
    if [[ ${!name+x} ]]; then
        printf '%s\tset\t%q\n' "$name" "${!name}" >> "$output_dir/environment-controls.tsv"
    else
        printf '%s\tunset\t-\n' "$name" >> "$output_dir/environment-controls.tsv"
    fi
done

# These paths came from the failed process's mappings. Query package metadata,
# not the libraries' code; do not use ldd or execute the diagnostic to load them.
runtime_paths=(
    /usr/local/cuda-13.2/targets/x86_64-linux/lib/libcudart.so.13.2.75
    /usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublas.so.13.4.0.1
    /usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublasLt.so.13.4.0.1
    /usr/lib/x86_64-linux-gnu/libcuda.so.595.84
    /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.595.84
    /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.595.84
    /usr/lib/x86_64-linux-gnu/libnvidia-nvvm70.so.4
)
declare -A package_seen=()
package_index=0
for index in "${!runtime_paths[@]}"; do
    id="package-owner-$index"
    run_query "$id" "$query_seconds" dpkg-query --search -- "${runtime_paths[$index]}"
    if [[ "$query_status" != ok ]]; then continue; fi
    while IFS= read -r line; do
        # Only accept a plain package owner (possibly architecture-qualified).
        # Ignore diversion prose/multi-owner ambiguity rather than invent names.
        if [[ "$line" != *': /'* ]]; then continue; fi
        owner=${line%%: /*}
        if [[ ! "$owner" =~ ^[a-z0-9][a-z0-9+.-]*(:[a-z0-9][a-z0-9-]*)?$ ]]; then continue; fi
        if [[ -n ${package_seen[$owner]+set} ]]; then continue; fi
        package_seen[$owner]=1
        run_query "package-version-$package_index" "$query_seconds" dpkg-query \
            --show '--showformat=${binary:Package}\t${Version}\t${Architecture}\t${Status}\n' "$owner"
        package_index=$((package_index + 1))
    done < "$output_dir/queries/$id.log"
done

root_auth=not-requested
if [[ "$root_inventory" == 1 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        skip_query sudo-validation missing-tool 'sudo unavailable: requested privileged inventory was not collected.' sudo -v
        root_auth=unavailable
        root_auth_failed=1
    else
        printf 'Optional firmware/BMC inventory requested. Validating sudo once; it may prompt now.\n'
        started=$(date -u +%FT%TZ)
        # Same terminal/session, foreground, and visible prompt; no setsid or
        # detached helper. Password entry remains sudo's terminal interaction.
        timeout --foreground --signal=TERM --kill-after=2 60 sudo -v 2>&1 | tee "$output_dir/queries/sudo-validation.log"
        auth_code=${PIPESTATUS[0]}
        finished=$(date -u +%FT%TZ)
        if (( auth_code == 0 )); then
            root_auth=validated
            record_status sudo-validation ok "$auth_code" "$started" "$finished" sudo -v
        else
            root_auth=failed
            root_auth_failed=1
            gap_count=$((gap_count + 1))
            record_status sudo-validation failed "$auth_code" "$started" "$finished" sudo -v
        fi
    fi
fi
printf 'root_authentication=%s\n' "$root_auth" >> "$manifest"

root_query() {
    local id=$1 tool=$2
    shift 2
    local resolved
    if [[ "$root_inventory" != 1 ]]; then return 0; fi
    if [[ "$root_auth" != validated ]]; then
        skip_query "$id" skipped-auth-failed 'No privileged query executed: sudo validation did not succeed.' "$tool" "$@"
    elif ! resolved=$(command -v "$tool" 2>/dev/null); then
        skip_query "$id" missing-tool "Optional tool unavailable on PATH: $tool" "$tool" "$@"
    else
        # Root owns the inner timeout and can signal its privileged child. The
        # outer non-root timeout is only a second bound around sudo/collection.
        # Exact read-only command; never a root shell, service action or setter.
        run_query "$id" "$((root_seconds + 5))" sudo -n -- "$(command -v timeout)" \
            --signal=TERM --kill-after=2 "$root_seconds" "$resolved" "$@"
    fi
}
root_query dmi-bios dmidecode --type 0
root_query dmi-system dmidecode --type 1
root_query dmi-baseboard dmidecode --type 2
root_query dmi-power-supply dmidecode --type 39
root_query bmc-sel ipmitool sel elist
root_query bmc-power-supplies ipmitool sdr type 'Power Supply'
root_query bmc-sensors ipmitool sensor list
root_query bmc-fru ipmitool fru print

collection=complete
exit_status=0
if (( gap_count > 0 )); then collection=complete-with-gaps; fi
if (( copy_failed )); then collection=partial-executable-copy-failed; exit_status=2; fi
if (( root_auth_failed )); then collection=partial-root-auth-failed; exit_status=2; fi
printf 'collection=%s\nquery_gap_count=%s\nexit_status=%s\nfinished_utc=%s\n' \
    "$collection" "$gap_count" "$exit_status" "$(date -u +%FT%TZ)" >> "$manifest"
archive="$output_dir.tar.gz"
if [[ -e "$archive" ]]; then
    printf 'error: archive already exists; refusing overwrite: %s\n' "$archive" >&2
    exit 2
fi
if ! timeout --signal=TERM --kill-after=2 60 tar -C "$repo_dir" -czf "$archive" "$(basename -- "$output_dir")"; then
    printf 'error: archive creation failed; inventory directory retained: %s\n' "$output_dir" >&2
    exit 2
fi
printf 'collection=%s\nexecutable_fingerprint=%s\nquery_gap_count=%s\n' "$collection" "$fingerprint" "$gap_count"
printf 'No executable or GPU workload ran. Gaps are listed in query-status.tsv.\n'
printf 'Archive to return: %s\n' "$archive"
printf 'Firmware/BMC output can contain serial numbers and host identifiers; review before external sharing.\n'
if [[ "$copy_state" == verified ]]; then
    printf 'Archive includes the verified frozen executable (never run); it may contain source/build paths.\n'
fi
exit "$exit_status"
