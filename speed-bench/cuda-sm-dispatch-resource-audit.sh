#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
COMMIT=$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)
GIT_STATUS=$(git status --porcelain=v1 2>/dev/null || true)
GIT_DIRTY=$(test -n "$GIT_STATUS" && printf true || printf false)
OUT_DIR_WAS_SET=${OUT_DIR+x}
OUT_DIR=${OUT_DIR:-"$ROOT/sm-dispatch-resource-${COMMIT}-${STAMP}"}
NVCC_BIN=${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
CUOBJDUMP_BIN=${CUOBJDUMP:-${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump}
CXXFILT_BIN=${CXXFILT:-c++filt}
KEEP_OBJECTS=${KEEP_OBJECTS:-0}
KEEP_SASS=${KEEP_SASS:-0}
REPARSE_ONLY=${REPARSE_ONLY:-0}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

if [[ "$REPARSE_ONLY" == 1 && -z "$OUT_DIR_WAS_SET" ]]; then
    fail "REPARSE_ONLY=1 requires OUT_DIR to name an existing audit directory"
fi
if [[ "$REPARSE_ONLY" != 1 ]]; then
    command -v "$NVCC_BIN" >/dev/null 2>&1 || fail "nvcc not found: $NVCC_BIN"
    command -v "$CUOBJDUMP_BIN" >/dev/null 2>&1 || fail "cuobjdump not found: $CUOBJDUMP_BIN"
fi
command -v "$CXXFILT_BIN" >/dev/null 2>&1 || fail "C++ demangler not found: $CXXFILT_BIN"
command -v awk >/dev/null 2>&1 || fail "awk is required"

mkdir -p "$OUT_DIR"
ARCHIVE="${OUT_DIR}.tar.gz"
AUDIT_COMPLETE=0

archive_on_exit() {
    local rc=$?
    if [[ "$AUDIT_COMPLETE" == 1 ]]; then
        printf 'status=complete\n' > "$OUT_DIR/run-status.txt"
    else
        printf 'status=failed\nexit_code=%d\n' "$rc" > "$OUT_DIR/run-status.txt"
    fi
    tar -czf "$ARCHIVE" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")" || true
    printf 'Audit directory: %s\n' "$OUT_DIR"
    printf 'Archive to return: %s\n' "$ARCHIVE"
}
trap archive_on_exit EXIT

if [[ "$REPARSE_ONLY" != 1 ]]; then
    {
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'git_dirty=%s\n' "$GIT_DIRTY"
        printf 'ds4_cuda_blob=%s\n' "$(git hash-object ds4_cuda.cu 2>/dev/null || printf unknown)"
        printf 'ds4_c_blob=%s\n' "$(git hash-object ds4.c 2>/dev/null || printf unknown)"
        if [[ -n "$GIT_STATUS" ]]; then
            printf '%s\n' "$GIT_STATUS" | sed 's/^/git_status=/'
        fi
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'host=%s\n' "$(uname -a)"
        printf 'nvcc=%s\n' "$NVCC_BIN"
        "$NVCC_BIN" --version | sed 's/^/nvcc_version=/'
        printf 'cuobjdump=%s\n' "$CUOBJDUMP_BIN"
        "$CUOBJDUMP_BIN" --version | sed 's/^/cuobjdump_version=/'
        printf 'cxxfilt=%s\n' "$CXXFILT_BIN"
    } > "$OUT_DIR/manifest.txt"

    # These inventories are deliberately source-derived. They make additions
    # visible even when a kernel is absent from the hand-reviewed matrix.
    awk '/__global__/ {
             line=$0
             sub(/^.*void[[:space:]]+/, "", line)
             sub(/\(.*/, "", line)
             printf "%d\t%s\n", NR, line
         }' ds4_cuda.cu > "$OUT_DIR/kernel-declarations-and-definitions.tsv"

    awk '/<<</ { printf "%d\t%s\n", NR, $0 }' ds4_cuda.cu \
        > "$OUT_DIR/kernel-launch-sites.tsv"

    grep -nE '__CUDA_ARCH__|binaryVersion|cuda_sm75_mma_ok|cuda_q4_mma_ok|cudaFunc(Get|Set)Attribute|cudaDevAttr(MaxSharedMemory|ComputeCapability)' \
        ds4_cuda.cu > "$OUT_DIR/architecture-dispatch-gates.txt" || true

    grep -nE 'DS4_CUDA_[A-Za-z0-9_]+' ds4_cuda.cu ds4.c \
        > "$OUT_DIR/cuda-environment-gates.txt" || true
fi

compile_arch() {
    local arch=$1
    local obj="$OUT_DIR/ds4_cuda.${arch}.o"
    local log="$OUT_DIR/${arch}.ptxas.log"
    printf 'Compiling %s resource image...\n' "$arch"
    "$NVCC_BIN" \
        -O3 -g -lineinfo --use_fast_math \
        -arch="$arch" \
        -Xcompiler -march=native \
        -Xcompiler -pthread \
        -Xptxas=-v \
        -c -o "$obj" ds4_cuda.cu \
        > "$OUT_DIR/${arch}.nvcc.stdout.log" 2> "$log"

    "$CUOBJDUMP_BIN" --dump-resource-usage "$obj" \
        > "$OUT_DIR/${arch}.resource-usage.txt"
    "$CUOBJDUMP_BIN" --dump-sass "$obj" \
        > "$OUT_DIR/${arch}.sass.txt"
}

parse_resources() {
    local arch=$1
    local input="$OUT_DIR/${arch}.resource-usage.txt"
    local raw="$OUT_DIR/${arch}.resource-records.raw.tsv"
    local out="$OUT_DIR/${arch}.resources.tsv"

    awk -v arch="$arch" '
        /^[[:space:]]*Function[[:space:]]+/ {
            line=$0
            sub(/^[[:space:]]*Function[[:space:]]*:?[[:space:]]*/, "", line)
            sub(/:[[:space:]]*$/, "", line)
            fn=line
            next
        }
        /^[[:space:]]*REG:/ && fn != "" {
            printf "%s\t%s\t%s\n", arch, fn, $0
        }
    ' "$input" > "$raw"

    printf 'arch\tmangled\tdemangled\tresource_record\n' > "$out"
    while IFS=$'\t' read -r row_arch mangled record; do
        demangled=$(printf '%s\n' "$mangled" | "$CXXFILT_BIN")
        record=$(printf '%s' "$record" | sed 's/^[[:space:]]*//')
        printf '%s\t%s\t%s\t%s\n' \
            "$row_arch" "$mangled" "$demangled" "$record" >> "$out"
    done < "$raw"
    rm -f "$raw"
}

summarize_sass() {
    local arch=$1
    local input="$OUT_DIR/${arch}.sass.txt"
    local raw="$OUT_DIR/${arch}.sass-summary.raw.tsv"
    local out="$OUT_DIR/${arch}.sass-summary.tsv"

    awk -v arch="$arch" '
        function emit() {
            if (fn != "") {
                printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", \
                    arch, fn, inst, imma, hmma, idp4a, ldgsts, ldg, lds, bar
            }
        }
        /^[[:space:]]*Function[[:space:]]*:/ {
            emit()
            line=$0
            sub(/^[^:]*:[[:space:]]*/, "", line)
            fn=line
            inst=imma=hmma=idp4a=ldgsts=ldg=lds=bar=0
            next
        }
        /\/\*[[:xdigit:]]+\*\// {
            inst++
            if ($0 ~ /[[:space:]]IMMA[.[:space:]]/) imma++
            if ($0 ~ /[[:space:]]HMMA[.[:space:]]/) hmma++
            if ($0 ~ /IDP[.]4A|DP4A/) idp4a++
            if ($0 ~ /LDGSTS/) ldgsts++
            if ($0 ~ /[[:space:]]LDG[.[:space:]]/) ldg++
            if ($0 ~ /[[:space:]]LDS[.[:space:]]/) lds++
            if ($0 ~ /BAR[.]SYNC/) bar++
        }
        END { emit() }
    ' "$input" > "$raw"

    printf 'arch\tmangled\tdemangled\tinstructions\timma\thmma\tidp4a\tldgsts\tldg\tlds\tbar_sync\n' > "$out"
    while IFS=$'\t' read -r row_arch mangled inst imma hmma idp4a ldgsts ldg lds bar; do
        demangled=$(printf '%s\n' "$mangled" | "$CXXFILT_BIN")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$row_arch" "$mangled" "$demangled" "$inst" "$imma" "$hmma" \
            "$idp4a" "$ldgsts" "$ldg" "$lds" "$bar" >> "$out"
    done < "$raw"
    rm -f "$raw"
}

if [[ "$REPARSE_ONLY" != 1 ]]; then
    compile_arch sm_75
    compile_arch sm_80
else
    for required in \
        "$OUT_DIR/sm_75.resource-usage.txt" \
        "$OUT_DIR/sm_80.resource-usage.txt" \
        "$OUT_DIR/sm_75.sass-summary.tsv" \
        "$OUT_DIR/sm_80.sass-summary.tsv"; do
        [[ -f "$required" ]] || fail "reparse input not found: $required"
    done
fi
parse_resources sm_75
parse_resources sm_80
if [[ "$REPARSE_ONLY" != 1 ]]; then
    summarize_sass sm_75
    summarize_sass sm_80
fi

awk -F '\t' '
    NR == FNR {
        if (FNR > 1) {
            present[$2]=1
            demangled[$2]=$3
            sm75[$2]=$4
        }
        next
    }
    FNR > 1 {
        present[$2]=1
        demangled[$2]=$3
        sm80[$2]=$4
    }
    END {
        print "mangled\tdemangled\tsm_75_resource_record\tsm_80_resource_record"
        for (fn in present) {
            printf "%s\t%s\t%s\t%s\n", fn, demangled[fn], sm75[fn], sm80[fn]
        }
    }
' "$OUT_DIR/sm_75.resources.tsv" "$OUT_DIR/sm_80.resources.tsv" \
    | { IFS= read -r header; printf '%s\n' "$header"; sort; } \
    > "$OUT_DIR/sm75-vs-sm80-resources.tsv"

awk -F '\t' '
    NR == FNR {
        if (FNR > 1) {
            present[$2]=1
            demangled[$2]=$3
            sm75[$2]=$4 FS $5 FS $6 FS $7 FS $8 FS $9 FS $10 FS $11
        }
        next
    }
    FNR > 1 {
        present[$2]=1
        demangled[$2]=$3
        sm80[$2]=$4 FS $5 FS $6 FS $7 FS $8 FS $9 FS $10 FS $11
    }
    END {
        print "mangled\tdemangled\tsm_75_instructions\tsm_75_imma\tsm_75_hmma\tsm_75_idp4a\tsm_75_ldgsts\tsm_75_ldg\tsm_75_lds\tsm_75_bar_sync\tsm_80_instructions\tsm_80_imma\tsm_80_hmma\tsm_80_idp4a\tsm_80_ldgsts\tsm_80_ldg\tsm_80_lds\tsm_80_bar_sync"
        for (fn in present) {
            printf "%s\t%s\t%s\t%s\n", fn, demangled[fn], sm75[fn], sm80[fn]
        }
    }
' "$OUT_DIR/sm_75.sass-summary.tsv" "$OUT_DIR/sm_80.sass-summary.tsv" \
    | { IFS= read -r header; printf '%s\n' "$header"; sort; } \
    > "$OUT_DIR/sm75-vs-sm80-sass.tsv"

{
    printf 'metric\tsm_75\tsm_80\n'
    printf 'kernel_declarations_and_definitions\t%s\t%s\n' \
        "$(wc -l < "$OUT_DIR/kernel-declarations-and-definitions.tsv")" \
        "$(wc -l < "$OUT_DIR/kernel-declarations-and-definitions.tsv")"
    printf 'compiled_resource_records\t%s\t%s\n' \
        "$(( $(wc -l < "$OUT_DIR/sm_75.resources.tsv") - 1 ))" \
        "$(( $(wc -l < "$OUT_DIR/sm_80.resources.tsv") - 1 ))"
    printf 'compiled_sass_functions\t%s\t%s\n' \
        "$(( $(wc -l < "$OUT_DIR/sm_75.sass-summary.tsv") - 1 ))" \
        "$(( $(wc -l < "$OUT_DIR/sm_80.sass-summary.tsv") - 1 ))"
    printf 'object_imma_instructions\t%s\t%s\n' \
        "$(awk -F '\t' 'NR>1{s+=$5}END{print s+0}' "$OUT_DIR/sm_75.sass-summary.tsv")" \
        "$(awk -F '\t' 'NR>1{s+=$5}END{print s+0}' "$OUT_DIR/sm_80.sass-summary.tsv")"
    printf 'object_hmma_instructions\t%s\t%s\n' \
        "$(awk -F '\t' 'NR>1{s+=$6}END{print s+0}' "$OUT_DIR/sm_75.sass-summary.tsv")" \
        "$(awk -F '\t' 'NR>1{s+=$6}END{print s+0}' "$OUT_DIR/sm_80.sass-summary.tsv")"
    printf 'object_idp4a_instructions\t%s\t%s\n' \
        "$(awk -F '\t' 'NR>1{s+=$7}END{print s+0}' "$OUT_DIR/sm_75.sass-summary.tsv")" \
        "$(awk -F '\t' 'NR>1{s+=$7}END{print s+0}' "$OUT_DIR/sm_80.sass-summary.tsv")"
    printf 'object_ldgsts_instructions\t%s\t%s\n' \
        "$(awk -F '\t' 'NR>1{s+=$8}END{print s+0}' "$OUT_DIR/sm_75.sass-summary.tsv")" \
        "$(awk -F '\t' 'NR>1{s+=$8}END{print s+0}' "$OUT_DIR/sm_80.sass-summary.tsv")"
} > "$OUT_DIR/architecture-summary.tsv"

printf '%s\n' \
    'The object-wide instruction totals include every compiled template and symbols' \
    'that runtime architecture gates will never launch. They are compiler coverage' \
    'checks, not a workload instruction mix or a performance comparison. Use the' \
    'per-function paired table together with the production dispatch matrix.' \
    > "$OUT_DIR/architecture-summary-NOTICE.txt"

if [[ "$REPARSE_ONLY" != 1 && "$KEEP_SASS" != 1 ]]; then
    rm -f "$OUT_DIR/sm_75.sass.txt" "$OUT_DIR/sm_80.sass.txt"
fi
if [[ "$REPARSE_ONLY" != 1 && "$KEEP_OBJECTS" != 1 ]]; then
    rm -f "$OUT_DIR/ds4_cuda.sm_75.o" "$OUT_DIR/ds4_cuda.sm_80.o"
fi

AUDIT_COMPLETE=1
