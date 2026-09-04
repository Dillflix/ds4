#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the bounded SM75 attention-output Q8_0 warp-interleaved A/B.

The grouped A projection and K-sliced B projection are measured independently
at their production single-token shapes. The candidate is opt-in and stores
only the consumed B K-slice in the auxiliary interleaved cache.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=9
  TIMING_REPEATS=100
  RUN_SANITIZER=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q8_ATTN_OUTPUT_INTERLEAVED_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-9}
TIMING_REPEATS=${TIMING_REPEATS:-100}
RUN_SANITIZER=${RUN_SANITIZER:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q8_ATTN_OUTPUT_INTERLEAVED_DIR:-$repo_dir/sm75-q8-attention-output-interleaved-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
(( TIMING_ROUNDS % 2 == 1 )) || die "TIMING_ROUNDS must be odd"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in RUN_SANITIZER SKIP_BUILD CREATE_ARCHIVE; do
    [[ ${!value} == 0 || ${!value} == 1 ]] || die "$value must be 0 or 1"
done
for command in nvidia-smi make tar; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM for GPU $PROFILE_GPU"
(( free_mib >= 512 )) || die "GPU $PROFILE_GPU has only ${free_mib} MiB free; 512 MiB is required"

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
targets=(tests/cuda_long_context_smoke tests/cuda_sm75_decode_weight_profile)
clean=(env
    -u DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE
    -u DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE
    -u DS4_CUDA_Q8_WARP_INTERLEAVED_CACHE_MB
    -u DS4_Q8_INTERLEAVED_PROFILE_AB)

phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    if [[ -d $OUTPUT_DIR ]]; then
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$OUTPUT_DIR.tar.gz"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
                    "$(basename "$OUTPUT_DIR")"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu_physical=%s\ncuda_arch=%s\ncompute_capability=%s\n' \
        "$PROFILE_GPU" "$CUDA_ARCH" "$compute_cap"
    printf 'free_mib_at_preflight=%s\ntiming_rounds=%s\ntiming_repeats=%s\n' \
        "$free_mib" "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'candidate_default=off\nb_slice_cache=consumed-half-only\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'
    git status --short
} >"$OUTPUT_DIR/manifest.txt"

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    if ! make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH="$CUDA_ARCH" \
            >"$OUTPUT_DIR/build.log" 2>&1; then
        tail -n 180 "$OUTPUT_DIR/build.log" >&2 || true
        die "build failed"
    fi
else
    make -q "${targets[@]}" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found stale diagnostic targets"
fi

phase=exactness
if ! "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        ./tests/cuda_long_context_smoke >"$OUTPUT_DIR/smoke.log" 2>&1; then
    tail -n 220 "$OUTPUT_DIR/smoke.log" >&2 || true
    die "CUDA regression failed"
fi
grep -Fq 'SM75 warp-interleaved Q8 attention-output A+B exact' \
    "$OUTPUT_DIR/smoke.log" || die "attention-output exactness marker missing"

phase=bounded-timing
for scenario in q8-grouped-a-half q8-kslice-t256; do
    log="$OUTPUT_DIR/$scenario.log"
    if ! "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            DS4_Q8_INTERLEAVED_PROFILE_AB=1 \
            DS4_CUDA_Q8_WARP_INTERLEAVED_CACHE_MB=64 \
            TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
            ./tests/cuda_sm75_decode_weight_profile "$scenario" \
            >"$log" 2>&1; then
        tail -n 180 "$log" >&2 || true
        die "$scenario bounded A/B failed"
    fi
    grep -Fq 'harness_status=ok' "$log" || die "$scenario success marker missing"
    grep -Eq '^candidate_speedup=[0-9]' "$log" || die "$scenario timing result missing"
    if [[ $scenario == q8-grouped-a-half ]]; then
        counter=attn-a-calls
        require_b_refinements=0
    else
        counter=attn-b-calls
        require_b_refinements=1
    fi
    awk -v counter="$counter" -v require_b="$require_b_refinements" '
        /SM75 warp-interleaved Q8 summary/ {
            seen++; selected=direct_xq=k128=fills=fallbacks=-1
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]==counter) selected=a[2]+0
                if (a[1]=="attn-b-direct-xq-calls") direct_xq=a[2]+0
                if (a[1]=="attn-b-k128-calls") k128=a[2]+0
                if (a[1]=="fills") fills=a[2]+0
                if (a[1]=="fallbacks") fallbacks=a[2]+0
            }
            if (selected<1 || fills!=1 || fallbacks!=0) bad=1
            if (require_b && (direct_xq<1 || k128<1)) bad=1
        }
        END {exit !(seen==1 && !bad)}
    ' "$log" || die "$scenario silently missed the candidate dispatch"
done

if [[ $RUN_SANITIZER == 1 ]]; then
    phase=compute-sanitizer
    command -v compute-sanitizer >/dev/null 2>&1 || die "compute-sanitizer not found"
    for item in \
        'q8-grouped-a-half:DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE' \
        'q8-kslice-t256:DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE'; do
        scenario=${item%%:*}; selector=${item#*:}
        log="$OUTPUT_DIR/$scenario-compute-sanitizer.log"
        if ! "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
                "$selector=1" DS4_CUDA_Q8_WARP_INTERLEAVED_CACHE_MB=64 \
                compute-sanitizer --tool memcheck --error-exitcode 99 \
                ./tests/cuda_sm75_decode_weight_profile "$scenario" \
                >"$log" 2>&1; then
            tail -n 180 "$log" >&2 || true
            die "Compute Sanitizer failed for $scenario"
        fi
        grep -Fq 'ERROR SUMMARY: 0 errors' "$log" ||
            die "Compute Sanitizer zero-error summary missing for $scenario"
    done
fi

phase=summarization
{
    printf '# SM75 attention-output Q8 warp-interleaved bounded A/B\n\n'
    printf 'Exact engine A+B output: bit-identical.\n\n'
    printf '| Projection | Control ms | Candidate ms | Speedup |\n'
    printf '| --- | ---: | ---: | ---: |\n'
    for scenario in q8-grouped-a-half q8-kslice-t256; do
        log="$OUTPUT_DIR/$scenario.log"
        control=$(awk -F= '$1=="control_median_ms" {print $2}' "$log")
        candidate=$(awk -F= '$1=="candidate_median_ms" {print $2}' "$log")
        speedup=$(awk -F= '$1=="candidate_speedup" {print $2}' "$log")
        printf '| %s | %s | %s | %sx |\n' "$scenario" "$control" "$candidate" "$speedup"
    done
    b_log="$OUTPUT_DIR/q8-kslice-t256.log"
    b_baseline=$(awk -F= '$1=="baseline_interleaved_median_ms" {print $2}' \
        "$b_log")
    b_incremental=$(awk -F= '$1=="candidate_over_baseline_speedup" {print $2}' \
        "$b_log")
    printf '\nB original interleaved median: %s ms; direct-XQ/K128 incremental speedup: %sx.\n' \
        "$b_baseline" "$b_incremental"
    printf '\nCandidate selectors remain opt-in and independently rollbackable.\n'
} >"$OUTPUT_DIR/summary.md"
cat "$OUTPUT_DIR/summary.md"

phase=complete
printf 'SM75 attention-output Q8 warp-interleaved bounded A/B complete: %s\n' "$OUTPUT_DIR"
