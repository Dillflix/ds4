#!/usr/bin/env bash
set -euo pipefail
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$test_dir/.." && pwd -P)
temp_parent=$(cd -- "${TMPDIR:-/tmp}" && pwd -P)
test_root=$(mktemp -d "$temp_parent/ds4-sm75-inventory-test.XXXXXX")
test_root=$(cd -- "$test_root" && pwd -P)
cleanup() {
    # Resolve and constrain the exact test-owned tree before recursive cleanup.
    case "$test_root" in "$temp_parent"/ds4-sm75-inventory-test.*) rm -rf -- "$test_root" ;; esac
}
trap cleanup EXIT
mkdir "$test_root/util"
# A curated PATH makes missing-tool tests independent of host CUDA/BMC installs.
for tool in timeout tar gzip mktemp mkdir date dirname basename tee sleep cp mv rm wc; do
    resolved=$(command -v "$tool")
    # Git Bash may materialize ln -s as an executable copy, which loses its DLL
    # lookup location. Small absolute-path wrappers also work on ordinary Linux.
    printf '#!/bin/bash\nexec %q "$@"\n' "$resolved" > "$test_root/util/$tool"
    chmod +x "$test_root/util/$tool"
done
count=0
pass() { count=$((count + 1)); printf 'ok %s - %s\n' "$count" "$1"; }
expect_grep() { grep -Fq -- "$1" "$2" || { printf 'missing expectation: %s in %s\n' "$1" "$2" >&2; exit 1; }; }
expect_absent() { if grep -Fq -- "$1" "$2"; then printf 'unexpected content: %s in %s\n' "$1" "$2" >&2; exit 1; fi; }

new_case() {
    case_dir="$test_root/$1"
    mkdir -p "$case_dir/repo/speed-bench" "$case_dir/repo/tests" "$case_dir/bin"
    cp "$repo_dir/speed-bench/collect-sm75-investigation-inventory.sh" "$case_dir/repo/speed-bench/"
    for tool in sha256sum wc readelf cuobjdump nvcc nsys dpkg-query sudo dmidecode ipmitool; do
        cp "$test_dir/fixtures/sm75-inventory/mock.sh" "$case_dir/bin/$tool"
        chmod +x "$case_dir/bin/$tool"
    done
    # This marker would expose an accidental execution of the inspected file.
    printf '#!/usr/bin/env bash\nprintf EXECUTED >> "$MOCK_LOG"\nexit 99\n' > "$case_dir/repo/tests/cuda_sm75_token_row_arithmetic"
    chmod +x "$case_dir/repo/tests/cuda_sm75_token_row_arithmetic"
    : > "$case_dir/tool.log"
}

run_case() {
    local expected=$1
    shift
    set +e
    env PATH="$case_dir/bin:$test_root/util" MOCK_LOG="$case_dir/tool.log" \
        INVENTORY_QUERY_TIMEOUT_SECONDS=5 INVENTORY_ROOT_TIMEOUT_SECONDS=5 \
        SECRET_DO_NOT_COLLECT=private-token-fixture "$@" \
        "$BASH" "$case_dir/repo/speed-bench/collect-sm75-investigation-inventory.sh" > "$case_dir/console.log" 2>&1
    status=$?
    set -e
    [[ "$status" == "$expected" ]] || { cat "$case_dir/console.log"; printf 'expected status %s, got %s\n' "$expected" "$status" >&2; exit 1; }
    outputs=("$case_dir"/repo/sm75-investigation-inventory.??????)
    [[ ${#outputs[@]} == 1 && -d ${outputs[0]} ]]
    output_dir=${outputs[0]}
    [[ -s "$output_dir.tar.gz" ]]
    tar -tzf "$output_dir.tar.gz" > "$case_dir/archive-list.txt"
    expect_grep 'workload_executed=0' "$output_dir/manifest.txt"
    expect_absent EXECUTED "$case_dir/tool.log"
    expect_absent nvidia-smi "$case_dir/tool.log"
    expect_absent private-token-fixture "$output_dir/environment-controls.tsv"
    if grep -Fq 'executable_copy=verified' "$output_dir/manifest.txt"; then
        cmp "$case_dir/repo/tests/cuda_sm75_token_row_arithmetic" "$output_dir/artifacts/cuda_sm75_token_row_arithmetic"
    else
        expect_absent '/artifacts/cuda_sm75_token_row_arithmetic' "$case_dir/archive-list.txt"
    fi
    expect_absent .executable-copy.partial "$case_dir/archive-list.txt"
}

new_case plain
run_case 0 ROOT_HOST_INVENTORY=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
expect_grep 'collection=complete' "$output_dir/manifest.txt"
expect_grep 'executable_fingerprint=match' "$output_dir/manifest.txt"
expect_grep $'CUBLAS_WORKSPACE_CONFIG\tset\t:4096:8' "$output_dir/environment-controls.tsv"
expect_grep 'readelf <-hW>' "$case_dir/tool.log"
expect_grep 'readelf <-dW>' "$case_dir/tool.log"
expect_grep 'readelf <-nW>' "$case_dir/tool.log"
expect_grep 'cuobjdump <--list-elf>' "$case_dir/tool.log"
expect_grep 'cuobjdump <--list-ptx>' "$case_dir/tool.log"
expect_grep 'nsys <profile> <--help>' "$case_dir/tool.log"
expect_absent sudo "$case_dir/tool.log"
[[ $(grep -c 'dpkg-query <--show>' "$case_dir/tool.log") == 1 ]]
pass 'file-only collection, package deduplication, no root/GPU/secret/binary copy'

new_case root
run_case 0 ROOT_HOST_INVENTORY=1
expect_grep 'root_authentication=validated' "$output_dir/manifest.txt"
[[ $(grep -c 'sudo <-v>' "$case_dir/tool.log") == 1 ]]
[[ $(grep -c 'sudo <-n>' "$case_dir/tool.log") == 8 ]]
[[ $(grep -c 'sudo <-n> <-->' "$case_dir/tool.log") == 8 ]]
[[ $(grep 'sudo <-n>' "$case_dir/tool.log" | grep -c '<--signal=TERM> <--kill-after=2> <5>') == 8 ]]
expect_grep 'dmidecode <--type> <39>' "$case_dir/tool.log"
expect_grep 'ipmitool <sel> <elist>' "$case_dir/tool.log"
expect_grep 'ipmitool <sdr> <type> <Power Supply>' "$case_dir/tool.log"
expect_grep 'ipmitool <sensor> <list>' "$case_dir/tool.log"
expect_grep 'ipmitool <fru> <print>' "$case_dir/tool.log"
pass 'optional root inventory uses one visible auth and exact read-only commands'

new_case auth_failure
run_case 2 ROOT_HOST_INVENTORY=1 MOCK_SUDO_STATUS=1
expect_grep 'collection=partial-root-auth-failed' "$output_dir/manifest.txt"
expect_grep 'skipped-auth-failed' "$output_dir/query-status.tsv"
expect_absent 'sudo <-n>' "$case_dir/tool.log"
expect_absent dmidecode "$case_dir/tool.log"
expect_absent ipmitool "$case_dir/tool.log"
pass 'authentication failure archives useful partial data without privileged queries'

new_case missing_tools
rm -- "$case_dir/bin/nvcc" "$case_dir/bin/ipmitool"
run_case 0 ROOT_HOST_INVENTORY=1
expect_grep 'complete-with-gaps' "$output_dir/manifest.txt"
expect_grep $'nvcc-version\tmissing-tool' "$output_dir/query-status.tsv"
expect_grep $'bmc-sel\tmissing-tool' "$output_dir/query-status.tsv"
pass 'missing optional tools are explicit nonfatal gaps'

new_case fingerprint_mismatch
run_case 0 ROOT_HOST_INVENTORY=0 MOCK_FINGERPRINT=0000000000000000000000000000000000000000000000000000000000000000
expect_grep 'executable_fingerprint=mismatch' "$output_dir/manifest.txt"
expect_grep 'complete-with-gaps' "$output_dir/manifest.txt"
pass 'fingerprint mismatch reported without executing or rebuilding inspected ELF'

new_case timeout
run_case 0 ROOT_HOST_INVENTORY=0 MOCK_SLEEP_HELP=1 INVENTORY_QUERY_TIMEOUT_SECONDS=1
expect_grep $'nsys-profile-help\ttimeout\t124' "$output_dir/query-status.tsv"
expect_grep 'complete-with-gaps' "$output_dir/manifest.txt"
pass 'a genuinely sleeping optional query is bounded and archived as timeout'

new_case missing_executable
rm -- "$case_dir/repo/tests/cuda_sm75_token_row_arithmetic"
run_case 0 ROOT_HOST_INVENTORY=0
expect_grep 'executable_fingerprint=missing' "$output_dir/manifest.txt"
expect_grep $'elf-header\tmissing-file' "$output_dir/query-status.tsv"
expect_absent 'readelf ' "$case_dir/tool.log"
pass 'absent executable is explicit and never triggers a build or fallback workload'

new_case include_executable
run_case 0 ROOT_HOST_INVENTORY=0 INCLUDE_EXECUTABLE=1
expect_grep 'executable_copy=verified' "$output_dir/manifest.txt"
expect_grep 'copied_executable_sha256=5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414' "$output_dir/manifest.txt"
expect_grep '/artifacts/cuda_sm75_token_row_arithmetic' "$case_dir/archive-list.txt"
pass 'opt-in copies exact pinned bytes without modifying or executing the original'

new_case reject_copy_mismatch
run_case 2 ROOT_HOST_INVENTORY=0 INCLUDE_EXECUTABLE=1 MOCK_FINGERPRINT=bad-fingerprint
expect_grep 'executable_copy=rejected-fingerprint' "$output_dir/manifest.txt"
expect_grep 'partial-executable-copy-failed' "$output_dir/manifest.txt"
pass 'opt-in copy rejects a mismatched original fingerprint'

new_case reject_copy_size
run_case 2 ROOT_HOST_INVENTORY=0 INCLUDE_EXECUTABLE=1 MOCK_BINARY_SIZE=134217729
expect_grep 'executable_copy=rejected-size' "$output_dir/manifest.txt"
pass 'opt-in copy rejects a file beyond its fixed 128 MiB limit'

new_case reject_copy_verification
run_case 2 ROOT_HOST_INVENTORY=0 INCLUDE_EXECUTABLE=1 MOCK_COPY_FINGERPRINT=changed-during-copy
expect_grep 'executable_copy=verification-failed' "$output_dir/manifest.txt"
pass 'changed copy fails verification and temporary copied bytes are excluded'

new_case root_timeout
run_case 0 ROOT_HOST_INVENTORY=1 MOCK_SLEEP_BMC=1 INVENTORY_ROOT_TIMEOUT_SECONDS=1
expect_grep $'bmc-sel\ttimeout\t124' "$output_dir/query-status.tsv"
expect_grep 'complete-with-gaps' "$output_dir/manifest.txt"
pass 'root-side timeout bounds a privileged-query child independently of outer collector'

printf 'Passed %s read-only inventory scenarios.\n' "$count"
