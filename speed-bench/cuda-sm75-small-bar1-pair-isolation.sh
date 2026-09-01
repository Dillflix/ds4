#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Isolate the mixed/small-BAR1 NVLink pair without changing the other pair.
Each arm preserves partner-resident weights and partner projection work in the
current 22/21 four-GPU production path at PP32768/TG256.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  CUDA_DEVICE_ORDER=PCI_BUS_ID

Optional environment:
  PROMPT=...                         default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  SMALL_BAR1_PAIR=0                 logical home tier; physical 0<->1 here
  VARIANTS=attention-off,production  scheduling matrix is explicit and fixed
  VARIANTS=attention-host-bounce     host-stage pair-0 attention-owned copies
  VARIANTS=attention-q8-host-bounce  host-stage pair-0 attention and Q8 copies
  VARIANTS=attention-q8-async-completion  global pair-0 post-Q8 positive markers
  VARIANTS=attention-q8-pre-gather-fence  confirm pair-0 Q8 completion before result D2H
  VARIANTS=attention-q8-activation-fence  also fence before pair-0 activation H2D
  VARIANTS=attention-q8-global-compute-fence  fence every pair-0 partner Q8 compute
  VARIANTS=attention-q8-direct-gather-fence  compute-sync then direct host-bounce gather, without mapped markers
  VARIANTS=attention-q8-rows-serialized  same cut, plus partner/home attention compute serialization
  VARIANTS=attention-q8-row-compute-off  retain pair-0 cache mirrors, suppress only split row compute/gathers
  VARIANTS=attention-row-query-shadow    direct query handoff, then exact home recompute
  VARIANTS=attention-row-partner-shadow  direct query + partner attention, then exact home recompute
  VARIANTS=attention-row-gather-shadow   direct query + partner attention + gather, then exact home recompute
  VARIANTS=attention-row-gather-dst-shadow  same gather bytes/direction, destination-ordered submission
  VARIANTS=attention-row-gather-chunk16-shadow  same source gather, split into 16 MiB copy submissions
  VARIANTS=attention-row-gather-chunk16-paced-shadow  same chunks, destination acknowledgement between them
  VARIANTS=attention-row-gather-scratch-paced-shadow  same paced gather into dedicated unused home scratch
  VARIANTS=attention-row-gather-source-scratch-paced-shadow  stage partner output locally, then pace scratch-to-scratch
  VARIANTS=attention-row-gather-preinitialized-source-paced-shadow  pace pre-zeroed scratch without reading partner output
  VARIANTS=attention-row-gather-preinitialized-source-no-partner-paced-shadow  same static copy, omit partner attention
  VARIANTS=attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow  retain attention, redirect its output
  VARIANTS=attention-q8-phase-audit  same cut plus pair-0 Q8 phase checkpoints
  VARIANTS=attention-q8-targeted-phase-audit  phase-audit one exact Q8 binding only
  Q8_TARGETED_BINDING_LABEL=...   default: tensor:blk.14.attn_output_b.weight
  Q8_TARGETED_WEIGHT_OFFSET=...   default: 143571266304
  VARIANTS=attention-q8-l14-l15-phase-audit   cumulative layer-14/layer-15 audit
  VARIANTS=attention-q8-l12-phase-audit       first measured layer-12 audit only
  ATTN_PHASE_AUDIT_LAYER=17        one production row-split dispatch only
  ATTN_PHASE_AUDIT_POS=512
  ATTN_END_FENCE_LAYER=21          one end-only production completion fence
  ATTN_END_FENCE_POS=512
  ATTN_ROW_BOUNDARY_END_LAYER=17   combined row-boundary audit: prior row layer
  ATTN_ROW_BOUNDARY_ENTRY_LAYER=18 combined row-boundary audit: next row layer
  ATTN_ROW_BOUNDARY_POS=512
  PP_TOKENS=32768
  TG_TOKENS=256
  REPEATS=1
  REQUIRED_POWER_LIMITS_W=250,260,250,250  physical GPU0..3 native limits
  TELEMETRY_INTERVAL_MS=500
  POST_CASE_SETTLE_SECONDS=5
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  RESUME=0
  ONE_SHOT=0                       require one fresh, non-resumable arm
  ONE_SHOT_TIMEOUT_SECONDS=900     bound the one-shot case monitor
  SMALL_BAR1_ISOLATION_DIR=/absolute/output/directory

The transport/scheduling diagnostic arms run before the known full-production
reproducer. They preserve arithmetic work but can change its timing envelope.
The completed historical pre-gather bracket used the fixed order
VARIANTS=attention-q8-pre-gather-fence,attention-q8-async-completion; do not
rerun that comparison for the activation-fence follow-up.
The fence arm emits a compact fflush-only `pre-gather armed` breadcrumb per
confirmed call immediately before result gather:
  ds4: CUDA q8 partner pre-gather armed current_sequence=N marker_sequence=N
  marker_complement=X home_tier=H home_device=HD partner_tier=P partner_device=PD
After the synchronous helper returns success it emits the matching compact line:
  ds4: CUDA q8 partner pre-gather returned current_sequence=N
  result_gather_status=success home_tier=H home_device=HD partner_tier=P partner_device=PD
These are two fence-arm-only `fprintf`/`fflush` records per successful call;
neither adds an `fsync`. Full checkpoints stay sparse. Completed validation
requires alternating, contiguous armed/returned pairs. After watcher
termination, armed N without returned N is classified only as
`result-gather-return-not-observed`: SIGKILL can land after helper success but
before the returned line. Returned N proves that gather succeeded; the locus
of any later failure remains unresolved. The no-fence control rejects fence,
armed, and returned records as contamination.
Structured failures record D2H and H2D attempt/completion separately. After a
completed D2H, H2D not attempted isolates destination switch/setup; H2D
attempted but not completed identifies an entered-but-failed H2D API.
The global-compute-fence diagnostic does not target a layer or occurrence. It
arms immediately after every selected pair-0 partner Q8 submission and returns
only after that partner device synchronizes. Each record carries the binding
label and weight offset selected by the actual production call. An unmatched
arm or explicit synchronization failure identifies the observed boundary but
does not prove that the dynamically named binding caused the endpoint reset.
Set ONE_SHOT=1 with RESUME=0, exactly one variant and REPEATS=1. One-shot mode
requires CREATE_ARCHIVE=1, a fresh nonexistent output directory and archive
path, never prints resume guidance, and its EXIT trap still archives
interrupted device-loss evidence. The activation-fence variant requires this
mode. The case monitor is bounded by ONE_SHOT_TIMEOUT_SECONDS. Watch records
become visible only through an atomic marker-plus-ready handshake after
pidfd-bound PID signaling; watcher/telemetry death also becomes durable
evidence. An
uninterruptible CUDA PID receives bounded TERM/KILL cleanup instead of leaving
the shell blocked forever in wait(1).
The harness captures immutable process start times for the case and both
monitoring helpers. Every one-shot TERM/KILL is delivered through a pidfd bound
to the validated process, and the watcher never signals after a durable
child-exit notice.
The activation arm adds device-wide synchronization, marker validation and
host logging. Those perturb timing, and a surfaced API boundary is not proof
of the root cause.
For ordinary multi-arm runs only, if GPU loss interrupts the shell, reboot,
set RESUME=1 and reuse the printed SMALL_BAR1_ISOLATION_DIR. The incomplete arm
is retained without silently retrying it. It counts as a failed arm only when
a durable lost-device watch record or an unhealthy post-run GPU snapshot
corroborates device loss.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
readonly EXPECTED_SELECTED_IDENTITY="$repo_dir/speed-bench/sm75-small-bar1-expected-device-identity.csv"
readonly ROW_COMPUTE_OFF_MARKER='ds4: CUDA prefill attention row compute pair-scoped disable: logical-pairs=0; partner cache allocation and mirror traffic retained; disabled pairs use home attention/indexer fallback'
readonly PACED_CHUNK_MARKER='ds4: CUDA paced chunked default-stream peer copy scheduled: source_tier=2 destination_tier=0 bytes=33554432 chunk_bytes=16777216 submissions=2 readiness=one-destination-event-one-source-wait inter_chunk=one-destination-ack-event-one-source-wait completion=per-chunk-source-event-destination-wait'
readonly SCRATCH_GATHER_MARKER='ds4: CUDA prefill attention row scratch gather scheduled:'
readonly SOURCE_SCRATCH_GATHER_MARKER='ds4: CUDA prefill attention row source scratch gather scheduled:'
readonly PREINITIALIZED_SOURCE_GATHER_ARMED_MARKER='ds4: CUDA prefill attention row preinitialized source gather armed:'
readonly PREINITIALIZED_SOURCE_GATHER_MARKER='ds4: CUDA prefill attention row preinitialized source gather scheduled:'
readonly PREINITIALIZED_SOURCE_NO_PARTNER_GATHER_ARMED_MARKER='ds4: CUDA prefill attention row preinitialized source no-partner gather armed:'
readonly PREINITIALIZED_SOURCE_NO_PARTNER_GATHER_MARKER='ds4: CUDA prefill attention row preinitialized source no-partner gather scheduled:'
readonly PARTNER_OUTPUT_SCRATCH_ARMED_MARKER='ds4: CUDA prefill attention row partner output scratch armed:'
readonly PREINITIALIZED_SOURCE_PARTNER_OUTPUT_SCRATCH_GATHER_ARMED_MARKER='ds4: CUDA prefill attention row preinitialized source partner-output-scratch gather armed:'
readonly PREINITIALIZED_SOURCE_PARTNER_OUTPUT_SCRATCH_GATHER_MARKER='ds4: CUDA prefill attention row preinitialized source partner-output-scratch gather scheduled:'

row_shadow_phase() {
    case "$1" in
        attention-row-query-shadow) printf '%s\n' query-copy ;;
        attention-row-partner-shadow) printf '%s\n' partner-compute ;;
        attention-row-gather-shadow) printf '%s\n' result-gather ;;
        attention-row-gather-dst-shadow) printf '%s\n' result-gather-dst ;;
        attention-row-gather-chunk16-shadow) printf '%s\n' result-gather-chunk16 ;;
        attention-row-gather-chunk16-paced-shadow) printf '%s\n' result-gather-chunk16-paced ;;
        attention-row-gather-scratch-paced-shadow) printf '%s\n' result-gather-scratch-paced ;;
        attention-row-gather-source-scratch-paced-shadow) printf '%s\n' result-gather-source-scratch-paced ;;
        attention-row-gather-preinitialized-source-paced-shadow) printf '%s\n' result-gather-preinitialized-source-paced ;;
        attention-row-gather-preinitialized-source-no-partner-paced-shadow) printf '%s\n' result-gather-preinitialized-source-no-partner-paced ;;
        attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow) printf '%s\n' result-gather-preinitialized-source-partner-output-scratch-paced ;;
        *) return 1 ;;
    esac
}
MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
SMALL_BAR1_PAIR=${SMALL_BAR1_PAIR:-0}
Q8_TARGET_BINDING_LABEL=tensor:blk.14.attn_output_b.weight
Q8_TARGET_WEIGHT_OFFSET=143571266304
Q8_TARGETED_BINDING_LABEL=${Q8_TARGETED_BINDING_LABEL:-$Q8_TARGET_BINDING_LABEL}
Q8_TARGETED_WEIGHT_OFFSET=${Q8_TARGETED_WEIGHT_OFFSET:-$Q8_TARGET_WEIGHT_OFFSET}
Q8_TARGET_PASSED_LABEL=attn_output_b
Q8_TARGET_TOKENS=512
Q8_TARGET_IN_DIM=8192
Q8_TARGET_OUT_DIM=4096
Q8_TARGET_WEIGHT_BYTES=35651584
Q8_TARGET_TRANSFER_BYTES=8388608
Q8_TARGET_RESULT_BYTES=8388608
Q8_WINDOW_L15_BINDING_LABEL=tensor:blk.15.attn_output_b.weight
Q8_WINDOW_L15_WEIGHT_OFFSET=143723876608
Q8_WINDOW_TARGETS="${Q8_TARGET_BINDING_LABEL}@${Q8_TARGET_WEIGHT_OFFSET},${Q8_WINDOW_L15_BINDING_LABEL}@${Q8_WINDOW_L15_WEIGHT_OFFSET}"
Q8_L12_BINDING_LABEL=tensor:blk.12.attn_output_b.weight
Q8_L12_WEIGHT_OFFSET=143236281600
Q8_L12_TARGET="${Q8_L12_BINDING_LABEL}@${Q8_L12_WEIGHT_OFFSET}"
Q8_L12_SKIP_OCCURRENCES=1
Q8_L12_MAX_OCCURRENCES=1
Q8_L12_SELECTED_OCCURRENCE=2
Q8_L12_EXPECTED_SEQUENCES=1
VARIANTS=${VARIANTS:-attention-off,production}
ATTN_PHASE_AUDIT_LAYER=${ATTN_PHASE_AUDIT_LAYER:-17}
ATTN_PHASE_AUDIT_POS=${ATTN_PHASE_AUDIT_POS:-512}
ATTN_END_FENCE_LAYER=${ATTN_END_FENCE_LAYER:-21}
ATTN_END_FENCE_POS=${ATTN_END_FENCE_POS:-512}
ATTN_ROW_BOUNDARY_END_LAYER=${ATTN_ROW_BOUNDARY_END_LAYER:-17}
ATTN_ROW_BOUNDARY_ENTRY_LAYER=${ATTN_ROW_BOUNDARY_ENTRY_LAYER:-18}
ATTN_ROW_BOUNDARY_POS=${ATTN_ROW_BOUNDARY_POS:-512}
PP_TOKENS=${PP_TOKENS:-32768}
TG_TOKENS=${TG_TOKENS:-256}
Q8_TARGET_EXPECTED_SEQUENCES=$((PP_TOKENS / 512 + 1))
Q8_WINDOW_EXPECTED_TOTAL_SEQUENCES=$((Q8_TARGET_EXPECTED_SEQUENCES * 2))
REPEATS=${REPEATS:-1}
if [[ -n ${REQUIRED_POWER_LIMITS_W+x} ]]; then
    REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W}
elif [[ -n ${REQUIRED_POWER_LIMIT_W:-} ]]; then
    # Compatibility for historical 250 W matrices. New fault-audit arms use
    # the native physical-index profile below.
    REQUIRED_POWER_LIMITS_W="${REQUIRED_POWER_LIMIT_W},${REQUIRED_POWER_LIMIT_W},${REQUIRED_POWER_LIMIT_W},${REQUIRED_POWER_LIMIT_W}"
else
    REQUIRED_POWER_LIMITS_W=250,260,250,250
fi
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-500}
POST_CASE_SETTLE_SECONDS=${POST_CASE_SETTLE_SECONDS:-5}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
RESUME=${RESUME:-0}
ONE_SHOT=${ONE_SHOT:-0}
ONE_SHOT_TIMEOUT_SECONDS=${ONE_SHOT_TIMEOUT_SECONDS:-900}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${SMALL_BAR1_ISOLATION_DIR:-$repo_dir/sm75-small-bar1-pair-isolation-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged SM75 model"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this isolation requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "SMALL_BAR1_PAIR:$SMALL_BAR1_PAIR" "PP_TOKENS:$PP_TOKENS" \
            "TG_TOKENS:$TG_TOKENS" "REPEATS:$REPEATS" \
            "ATTN_PHASE_AUDIT_LAYER:$ATTN_PHASE_AUDIT_LAYER" \
            "ATTN_PHASE_AUDIT_POS:$ATTN_PHASE_AUDIT_POS" \
            "ATTN_END_FENCE_LAYER:$ATTN_END_FENCE_LAYER" \
            "ATTN_END_FENCE_POS:$ATTN_END_FENCE_POS" \
            "ATTN_ROW_BOUNDARY_END_LAYER:$ATTN_ROW_BOUNDARY_END_LAYER" \
            "ATTN_ROW_BOUNDARY_ENTRY_LAYER:$ATTN_ROW_BOUNDARY_ENTRY_LAYER" \
            "ATTN_ROW_BOUNDARY_POS:$ATTN_ROW_BOUNDARY_POS" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "POST_CASE_SETTLE_SECONDS:$POST_CASE_SETTLE_SECONDS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE" \
            "RESUME:$RESUME" "ONE_SHOT:$ONE_SHOT" \
            "ONE_SHOT_TIMEOUT_SECONDS:$ONE_SHOT_TIMEOUT_SECONDS"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
[[ $Q8_TARGETED_BINDING_LABEL =~ ^tensor:blk\.[0-9]+\.attn_output_b\.weight$ ]] ||
    die "Q8_TARGETED_BINDING_LABEL must name one tensor:blk.N.attn_output_b.weight binding"
[[ $Q8_TARGETED_WEIGHT_OFFSET =~ ^[0-9]+$ ]] ||
    die "Q8_TARGETED_WEIGHT_OFFSET must be a decimal byte offset"
(( SMALL_BAR1_PAIR == 0 && PP_TOKENS == 32768 && TG_TOKENS == 256 &&
   ATTN_PHASE_AUDIT_LAYER < STAGE_SPLIT &&
   ATTN_PHASE_AUDIT_POS < PP_TOKENS &&
   ATTN_PHASE_AUDIT_POS % 512 == 0 &&
   ATTN_END_FENCE_LAYER < STAGE_SPLIT &&
   ATTN_END_FENCE_POS < PP_TOKENS &&
   ATTN_END_FENCE_POS % 512 == 0 &&
   ATTN_ROW_BOUNDARY_END_LAYER == 17 &&
   ATTN_ROW_BOUNDARY_ENTRY_LAYER == 18 &&
   ATTN_ROW_BOUNDARY_POS == 512 &&
   REPEATS >= 1 &&
   TELEMETRY_INTERVAL_MS >= 100 &&
   POST_CASE_SETTLE_SECONDS <= 60 &&
   ONE_SHOT_TIMEOUT_SECONDS >= 60 && ONE_SHOT_TIMEOUT_SECONDS <= 3600 )) ||
    die "invalid fixed production isolation configuration"
case "$REQUIRED_POWER_LIMITS_W" in
    250,260,250,250|250,250,250,250) ;;
    *) die "require native 250,260,250,250 W or historical 250 W profile" ;;
esac
for flag in SKIP_BUILD CREATE_ARCHIVE RESUME ONE_SHOT; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"
[[ ${CUDA_DEVICE_ORDER:-} == PCI_BUS_ID ]] ||
    die "CUDA_DEVICE_ORDER=PCI_BUS_ID is required for strict CUDA/nvidia-smi identity"

IFS=, read -r -a variants <<<"$VARIANTS"
(( ${#variants[@]} >= 1 )) || die "VARIANTS selected no arms"
declare -A seen_variants=()
attention_copy_matrix_requested=0
for variant in "${variants[@]}"; do
    case "$variant" in
        attention-off|attention-host-bounce|attention-q8-host-bounce|attention-q8-async-completion|attention-q8-pre-gather-fence|attention-q8-activation-fence|attention-q8-global-compute-fence|attention-q8-direct-gather-fence|attention-q8-rows-serialized|attention-q8-row-compute-off|attention-row-query-shadow|attention-row-partner-shadow|attention-row-gather-shadow|attention-row-gather-dst-shadow|attention-row-gather-chunk16-shadow|attention-row-gather-chunk16-paced-shadow|attention-row-gather-scratch-paced-shadow|attention-row-gather-source-scratch-paced-shadow|attention-row-gather-preinitialized-source-paced-shadow|attention-row-gather-preinitialized-source-no-partner-paced-shadow|attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-q8-l12-phase-audit|attention-query-dst|attention-gather-dst|attention-both-dst|attention-phase-audit|attention-end-fence|attention-row-boundary-audit|partner-bounce|bounce-indexer-off|partner-serialized|indexer-off|production) ;;
        *) die "unknown variant: $variant" ;;
    esac
    [[ -z ${seen_variants[$variant]:-} ]] || die "duplicate variant: $variant"
    seen_variants[$variant]=1
    case "$variant" in
        attention-host-bounce|attention-q8-host-bounce|attention-q8-async-completion|attention-q8-pre-gather-fence|attention-q8-activation-fence|attention-q8-global-compute-fence|attention-q8-direct-gather-fence|attention-q8-rows-serialized|attention-q8-row-compute-off|attention-row-query-shadow|attention-row-partner-shadow|attention-row-gather-shadow|attention-row-gather-dst-shadow|attention-row-gather-chunk16-shadow|attention-row-gather-chunk16-paced-shadow|attention-row-gather-scratch-paced-shadow|attention-row-gather-source-scratch-paced-shadow|attention-row-gather-preinitialized-source-paced-shadow|attention-row-gather-preinitialized-source-no-partner-paced-shadow|attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-q8-l12-phase-audit|attention-query-dst|attention-gather-dst|attention-both-dst)
            attention_copy_matrix_requested=1
            ;;
    esac
done
if [[ $ONE_SHOT == 1 ]]; then
    [[ $RESUME == 0 ]] || die "ONE_SHOT=1 requires RESUME=0"
    [[ $CREATE_ARCHIVE == 1 ]] || die "ONE_SHOT=1 requires CREATE_ARCHIVE=1"
    (( ${#variants[@]} == 1 )) ||
        die "ONE_SHOT=1 requires exactly one variant"
    (( REPEATS == 1 )) || die "ONE_SHOT=1 requires REPEATS=1"
    python3 - <<'PY' >/dev/null 2>&1 ||
import os
import signal
assert callable(getattr(os, "pidfd_open", None))
assert callable(getattr(signal, "pidfd_send_signal", None))
PY
        die "ONE_SHOT=1 requires Python os.pidfd_open and signal.pidfd_send_signal"
fi
if [[ ( -n ${seen_variants[attention-q8-activation-fence]:-} ||
        -n ${seen_variants[attention-q8-global-compute-fence]:-} ||
        -n ${seen_variants[attention-q8-direct-gather-fence]:-} ||
        -n ${seen_variants[attention-q8-rows-serialized]:-} ||
        -n ${seen_variants[attention-q8-row-compute-off]:-} ||
        -n ${seen_variants[attention-row-query-shadow]:-} ||
        -n ${seen_variants[attention-row-partner-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-dst-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-chunk16-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-chunk16-paced-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-scratch-paced-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-source-scratch-paced-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-preinitialized-source-paced-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-preinitialized-source-no-partner-paced-shadow]:-} ||
        -n ${seen_variants[attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow]:-} ) &&
      $ONE_SHOT != 1 ]]; then
    die "destructive fence/row-shadow diagnostics require ONE_SHOT=1"
fi
if (( attention_copy_matrix_requested )) && [[ $SKIP_BUILD != 0 ]]; then
    die "attention copy diagnostic arms require SKIP_BUILD=0 so every reboot repeats the fixed CUDA/P2P preflight"
fi

for tool in awk basename cat cmp cp cut date dirname env find git grep kill make \
            mkdir mv nproc nvidia-smi python3 rm sha256sum sleep sort stat stdbuf sync \
            tail tar tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"

if [[ $RESUME == 0 ]]; then
    if [[ $CREATE_ARCHIVE == 1 && -e $OUTPUT_DIR.tar.gz ]]; then
        die "archive path already exists: $OUTPUT_DIR.tar.gz"
    fi
    mkdir -- "$OUTPUT_DIR" 2>/dev/null ||
        die "output path already exists or cannot be created: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"/{health,production,provenance,telemetry}
else
    [[ -d $OUTPUT_DIR ]] || die "resume directory not found: $OUTPUT_DIR"
    for subdir in health production provenance telemetry; do
        mkdir -p "$OUTPUT_DIR/$subdir"
    done
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
printf 'Diagnostic directory: %s\n' "$OUTPUT_DIR"
if [[ $ONE_SHOT == 0 ]]; then
    printf 'Resume with: export SMALL_BAR1_ISOLATION_DIR=%q; RESUME=1 ...\n' \
        "$OUTPUT_DIR"
else
    printf 'One-shot mode: no resume or control rerun; interrupted evidence will be archived.\n'
fi
if [[ $RESUME == 1 ]]; then
    [[ -s $OUTPUT_DIR/manifest.txt ]] ||
        die "resume manifest is missing or empty"
    grep -Fxq "git_commit=$(git rev-parse HEAD)" "$OUTPUT_DIR/manifest.txt" ||
        die "resume commit differs from the original isolation"
    grep -Fxq "model=$MODEL" "$OUTPUT_DIR/manifest.txt" ||
        die "resume model differs from the original isolation"
    grep -Fxq "variants=$VARIANTS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume variant order differs from the original isolation"
    grep -Fxq "small_bar1_pair=$SMALL_BAR1_PAIR" "$OUTPUT_DIR/manifest.txt" ||
        die "resume small-BAR1 pair differs from the original isolation"
    grep -Fxq "pp_tokens=$PP_TOKENS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume prefill length differs from the original isolation"
    grep -Fxq "tg_tokens=$TG_TOKENS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume decode length differs from the original isolation"
    grep -Fxq "repeats=$REPEATS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume repeat count differs from the original isolation"
    grep -Fxq "attention_q8_targeted_phase_audit_binding=$Q8_TARGETED_BINDING_LABEL" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume targeted Q8 binding differs from the original isolation"
    grep -Fxq "attention_q8_targeted_phase_audit_weight_offset=$Q8_TARGETED_WEIGHT_OFFSET" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume targeted Q8 weight offset differs from the original isolation"
    grep -Fxq "required_power_limits_w=$REQUIRED_POWER_LIMITS_W" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume power-limit profile differs from the original isolation"
    grep -Fxq "attention_phase_audit_layer=$ATTN_PHASE_AUDIT_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention phase-audit layer differs from the original isolation"
    grep -Fxq "attention_phase_audit_pos=$ATTN_PHASE_AUDIT_POS" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention phase-audit position differs from the original isolation"
    grep -Fxq "attention_end_fence_layer=$ATTN_END_FENCE_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention end-fence layer differs from the original isolation"
    grep -Fxq "attention_end_fence_pos=$ATTN_END_FENCE_POS" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention end-fence position differs from the original isolation"
    grep -Fxq "attention_row_boundary_end_layer=$ATTN_ROW_BOUNDARY_END_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention row-boundary end layer differs from the original isolation"
    grep -Fxq "attention_row_boundary_entry_layer=$ATTN_ROW_BOUNDARY_ENTRY_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention row-boundary entry layer differs from the original isolation"
    grep -Fxq "attention_row_boundary_pos=$ATTN_ROW_BOUNDARY_POS" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention row-boundary position differs from the original isolation"
fi

phase=initialization
telemetry_pid=
telemetry_identity=
telemetry_watch_pid=
telemetry_watch_identity=
active_case_pid=
active_case_identity=
pid_is_live() {
    local pid=$1 state=
    kill -0 "$pid" >/dev/null 2>&1 || return 1
    if [[ -r /proc/$pid/stat ]]; then
        state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
        [[ $state != Z ]] || return 1
    fi
}
process_start_time() {
    local pid=$1 stat rest
    local -a stat_fields
    [[ -r /proc/$pid/stat ]] || return 1
    stat=$(<"/proc/$pid/stat")
    rest=${stat##*) }
    read -r -a stat_fields <<<"$rest"
    # /proc/PID/stat field 22 (starttime), after removing fields 1-2.
    [[ ${stat_fields[19]:-} =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "${stat_fields[19]}"
}
pid_matches_identity() {
    local pid=$1 expected=$2 actual
    [[ -n $expected ]] || return 1
    pid_is_live "$pid" || return 1
    actual=$(process_start_time "$pid" 2>/dev/null) || return 1
    [[ $actual == "$expected" ]]
}
capture_process_identity() {
    local pid=$1 identity
    for _ in {1..50}; do
        if identity=$(process_start_time "$pid" 2>/dev/null) &&
                pid_matches_identity "$pid" "$identity"; then
            printf '%s\n' "$identity"
            return 0
        fi
        pid_is_live "$pid" || return 2
        sleep 0.01
    done
    return 1
}
signal_bound_process() {
    local pid=$1 identity=$2 signal_name=$3
    if [[ $ONE_SHOT != 1 ]]; then
        kill "-$signal_name" "$pid" >/dev/null 2>&1
        return
    fi
    if [[ -z $identity ]]; then
        printf 'error: refusing pidfd signal %s for PID %s; immutable identity was not captured and the process may still be live\n' \
            "$signal_name" "$pid" >&2
        return 1
    fi
    python3 - "$pid" "$identity" "$signal_name" <<'PY'
import os
import signal
import sys

pid = int(sys.argv[1])
expected = sys.argv[2]
sig = getattr(signal, "SIG" + sys.argv[3])
try:
    fd = os.pidfd_open(pid, 0)
except (ProcessLookupError, PermissionError):
    raise SystemExit(1)
try:
    with open(f"/proc/{pid}/stat", "r", encoding="ascii") as stream:
        rest = stream.read().rsplit(") ", 1)[1].split()
    if rest[0] == "Z" or rest[19] != expected:
        raise SystemExit(1)
    signal.pidfd_send_signal(fd, sig, None, 0)
finally:
    os.close(fd)
PY
}
stop_active_case() {
    if [[ -n ${active_case_pid:-} ]]; then
        local pid=$active_case_pid
        if [[ -z ${active_case_identity:-} ]]; then
            printf 'error: refusing bare-PID signal for case PID %s; immutable identity was not captured and the child may still be live\n' \
                "$pid" >&2
            return 1
        fi
        if ! pid_matches_identity "$pid" "$active_case_identity"; then
            active_case_pid=
            active_case_identity=
            return 0
        fi
        signal_bound_process "$pid" "$active_case_identity" TERM || true
        for _ in {1..20}; do
            if [[ -n ${active_case_identity:-} ]] &&
                    ! pid_matches_identity "$pid" "$active_case_identity"; then
                active_case_pid=
                active_case_identity=
                return 0
            fi
            if [[ -r /proc/$pid/stat ]] &&
                    [[ $(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) == Z ]]; then
                wait "$pid" >/dev/null 2>&1 || true
                active_case_pid=
                active_case_identity=
                return 0
            fi
            sleep 0.1
        done
        if pid_matches_identity "$pid" "$active_case_identity"; then
            signal_bound_process "$pid" "$active_case_identity" KILL || true
        fi
        for _ in {1..20}; do
            if [[ -n ${active_case_identity:-} ]] &&
                    ! pid_matches_identity "$pid" "$active_case_identity"; then
                active_case_pid=
                active_case_identity=
                return 0
            fi
            if [[ -r /proc/$pid/stat ]] &&
                    [[ $(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) == Z ]]; then
                wait "$pid" >/dev/null 2>&1 || true
                active_case_pid=
                active_case_identity=
                return 0
            fi
            sleep 0.1
        done
        return 1
    fi
    return 0
}
stop_telemetry() {
    if [[ -n ${telemetry_watch_pid:-} ]]; then
        local watch_pid=$telemetry_watch_pid
        local watch_identity=$telemetry_watch_identity
        telemetry_watch_pid=
        telemetry_watch_identity=
        signal_bound_process "$watch_pid" "$watch_identity" TERM || true
        if [[ $ONE_SHOT == 1 ]]; then
            for _ in {1..20}; do
                pid_is_live "$watch_pid" || break
                sleep 0.05
            done
            if pid_is_live "$watch_pid"; then
                signal_bound_process "$watch_pid" "$watch_identity" KILL || true
                for _ in {1..20}; do
                    pid_is_live "$watch_pid" || break
                    sleep 0.05
                done
            fi
            pid_is_live "$watch_pid" ||
                wait "$watch_pid" >/dev/null 2>&1 || true
        else
            wait "$watch_pid" >/dev/null 2>&1 || true
        fi
    fi
    for name in telemetry_pid; do
        local pid=${!name:-}
        local identity=$telemetry_identity
        [[ -n $pid ]] || continue
        printf -v "$name" '%s' ''
        telemetry_identity=
        signal_bound_process "$pid" "$identity" TERM || true
        for _ in {1..20}; do
            pid_is_live "$pid" || break
            sleep 0.05
        done
        if pid_is_live "$pid"; then
            signal_bound_process "$pid" "$identity" KILL || true
            for _ in {1..20}; do
                pid_is_live "$pid" || break
                sleep 0.05
            done
        fi
        pid_is_live "$pid" || wait "$pid" >/dev/null 2>&1 || true
    done
}
write_summary() {
    python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
        "$OUTPUT_DIR" >/dev/null 2>&1 || true
}
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    stop_telemetry
    stop_active_case || true
    write_summary
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    sync "$OUTPUT_DIR/run-status.txt" 2>/dev/null || sync
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if [[ -e $archive ]]; then
            status=1
            printf 'error: refusing to overwrite archive %s\n' "$archive" >&2
        elif tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv -n -- "$partial" "$archive" &&
                [[ ! -e $partial && -s $archive ]] &&
                (sync "$archive" 2>/dev/null || sync) &&
                (sync -f "$(dirname "$archive")" 2>/dev/null || sync); then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            rm -f -- "$partial"
            printf 'error: failed to create archive %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"
IFS=, read -r -a required_power_limits <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#required_power_limits[@]} == 4 )) ||
    die "REQUIRED_POWER_LIMITS_W must contain physical GPU0..3 limits"
for limit in "${required_power_limits[@]}"; do
    [[ $limit =~ ^[0-9]+$ ]] ||
        die "REQUIRED_POWER_LIMITS_W entries must be integer watts"
done

validate_power_limits() {
    local gpu limit required
    for gpu in "${gpu_ids[@]}"; do
        required=${required_power_limits[$gpu]:-}
        [[ -n $required ]] || return 1
        limit=$(timeout --kill-after=5s 20s nvidia-smi -i "$gpu" \
            --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null |
            awk 'NR==1 {print $1}') || return 1
        awk -v actual="$limit" -v required="$required" \
            'BEGIN {exit !(actual + 0.0 == required + 0.0)}' || return 1
    done
}

capture_and_validate_cuda_identity() {
    local suffix= stamp raw_cuda raw_nvidia raw_topology
    local cuda_inventory nvidia_inventory topology selected_inventory path
    if [[ $RESUME == 1 ]]; then
        stamp=$(date -u +%Y%m%dT%H%M%S.%NZ)
        suffix="-resume-${stamp}-$$"
    fi
    raw_cuda="$OUTPUT_DIR/provenance/cuda-device-inventory${suffix}.raw.csv"
    raw_nvidia="$OUTPUT_DIR/provenance/nvidia-device-inventory${suffix}.raw.csv"
    raw_topology="$OUTPUT_DIR/provenance/nvidia-topology${suffix}.txt"
    cuda_inventory="$OUTPUT_DIR/provenance/cuda-device-inventory${suffix}.csv"
    nvidia_inventory="$OUTPUT_DIR/provenance/nvidia-device-inventory${suffix}.csv"
    topology="$OUTPUT_DIR/provenance/nvidia-topology${suffix}.csv"
    selected_inventory="$OUTPUT_DIR/provenance/selected-device-identity${suffix}.csv"

    for path in "$raw_cuda" "$raw_nvidia" "$raw_topology" \
            "$cuda_inventory" "$nvidia_inventory" "$topology" \
            "$selected_inventory"; do
        [[ ! -e $path ]] || {
            printf 'refusing to overwrite CUDA identity snapshot %s\n' "$path" >&2
            return 1
        }
    done

    "${clean[@]}" ./tests/cuda_device_identity >"$raw_cuda" || {
        printf 'CUDA ordinal inventory failed\n' >&2
        return 1
    }
    {
        printf 'nvidia_index,pci_bus_id,uuid\n'
        timeout --kill-after=5s 20s nvidia-smi \
            --query-gpu=index,pci.bus_id,uuid \
            --format=csv,noheader,nounits
    } >"$raw_nvidia" || {
        printf 'nvidia-smi identity inventory failed\n' >&2
        return 1
    }
    timeout --kill-after=5s 20s nvidia-smi topo -m >"$raw_topology" || {
        printf 'nvidia-smi topology inventory failed\n' >&2
        return 1
    }
    python3 speed-bench/validate-cuda-device-identity.py capture \
        --cuda-inventory "$raw_cuda" \
        --nvidia-inventory "$raw_nvidia" \
        --topology "$raw_topology" \
        --selected "$GPU_DEVICES" \
        --expected-selected "$EXPECTED_SELECTED_IDENTITY" \
        --normalized-cuda-output "$cuda_inventory" \
        --normalized-nvidia-output "$nvidia_inventory" \
        --normalized-topology-output "$topology" \
        --output "$selected_inventory" || return 1

    if [[ $RESUME == 1 ]]; then
        cmp -s "$OUTPUT_DIR/provenance/cuda-device-inventory.csv" \
            "$cuda_inventory" || {
                printf 'resume CUDA ordinal identity differs from the original run\n' >&2
                return 1
            }
        cmp -s "$OUTPUT_DIR/provenance/nvidia-device-inventory.csv" \
            "$nvidia_inventory" || {
                printf 'resume nvidia-smi identity differs from the original run\n' >&2
                return 1
            }
        cmp -s "$OUTPUT_DIR/provenance/nvidia-topology.csv" \
            "$topology" || {
                printf 'resume normalized nvidia-smi topology differs from the original run\n' >&2
                return 1
            }
        cmp -s "$OUTPUT_DIR/provenance/selected-device-identity.csv" \
            "$selected_inventory" || {
                printf 'resume selected CUDA/NVLink mapping differs from the original run\n' >&2
                return 1
            }
    fi
}

validate_arm_cuda_identity() {
    local log=$1 evidence=$2 report=$3 status=0
    python3 speed-bench/validate-cuda-device-identity.py runtime \
        --runtime-log "$log" \
        --cuda-inventory "$OUTPUT_DIR/provenance/cuda-device-inventory.csv" \
        --selected-identity "$OUTPUT_DIR/provenance/selected-device-identity.csv" \
        --output "$evidence" >"$report" 2>&1 || status=$?
    if (( status == 0 )); then
        printf 'status=validated\n' >>"$report"
    else
        printf 'status=failed\nexit_status=%s\n' "$status" >>"$report"
    fi
    sync "$evidence" "$report" 2>/dev/null || sync
    return "$status"
}

record_arm_identity_failure() {
    local result=$1 variant=$2 repeat=$3 progress=$4 log=$5 report=$6
    local fields lp le lc lt
    write_result "$result" "$variant" identity-validation-failed 130 \
        "$progress" "$log"
    fields=$(last_progress_fields "$progress")
    IFS=$'\t' read -r lp le lc lt <<<"$fields"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
        identity-validation-failed 130 "$lp" "$le" \
        >>"$OUTPUT_DIR/run-journal.tsv"
    sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
    tail -n 80 "$report" >&2 || true
}

assert_no_compute_processes() {
    local listing
    listing=$(timeout --kill-after=5s 20s nvidia-smi \
        --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>&1) || {
        printf '%s\n' "$listing" >&2
        return 1
    }
    if [[ -n ${listing//[[:space:]]/} ]]; then
        printf 'unexpected GPU compute process(es) before arm:\n%s\n' \
            "$listing" >&2
        return 1
    fi
}

list_foreign_compute_processes() {
    local allowed_pid=$1 listing
    listing=$(timeout --kill-after=5s 20s nvidia-smi \
        --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>/dev/null) || return 1
    printf '%s\n' "$listing" | awk -F, -v allowed="$allowed_pid" '
        {
            pid=$1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
            if (pid ~ /^[0-9]+$/ && pid != allowed) print $0
        }
    '
}

gpu_health_snapshot() {
    local path=$1
    {
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        timeout --kill-after=5s 20s nvidia-smi -L
        timeout --kill-after=5s 20s nvidia-smi \
            --query-gpu=index,pci.bus_id,uuid,serial,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.free,utilization.gpu \
            --format=csv
        printf '\ntopology:\n'
        timeout --kill-after=5s 20s nvidia-smi topo -m
    } >"$path" 2>&1 || return 1
    gpu_health_snapshot_is_healthy "$path"
}

gpu_health_snapshot_is_healthy() {
    local path=$1
    [[ -s $path ]] || return 1
    [[ $(grep -c '^GPU [0-9]:' "$path") == 4 ]] || return 1
    ! grep -Eiq \
        'ERR!|GPU is lost|Unknown Error|Unable to determine|GPU Unavailable|Critical Xid' \
        "$path"
}

gpu_health_snapshot_is_unhealthy() {
    local path=$1
    [[ -s $path ]] || return 1
    grep -Eiq \
        'ERR!|GPU is lost|Unknown Error|Unable to determine|GPU Unavailable|Critical Xid' \
        "$path"
}

publish_watch_marker() {
    local marker=$1 status=$2 case_pid=$3 lost_devices=$4
    local foreign_processes=$5 monitor_detail=${6:-none}
    local marker_tmp ready ready_tmp marker_dir
    marker_tmp="${marker}.tmp.$BASHPID.$RANDOM"
    ready="${marker}.ready"
    ready_tmp="${ready}.tmp.$BASHPID.$RANDOM"
    marker_dir=$(dirname "$marker")
    printf 'timestamp_utc=%s\nstatus=%s\ncase_pid=%s\nrequired_power_limits_w=%s\nlost_devices=%s\nforeign_processes=%q\nmonitor_detail=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$case_pid" \
        "$REQUIRED_POWER_LIMITS_W" "$lost_devices" "$foreign_processes" \
        "$monitor_detail" >"$marker_tmp"
    sync "$marker_tmp" 2>/dev/null || sync
    if ! mv -n -- "$marker_tmp" "$marker" || [[ -e $marker_tmp ]]; then
        rm -f -- "$marker_tmp"
        [[ -s $marker ]] || return 1
    fi
    sync "$marker" 2>/dev/null || sync
    sync -f "$marker_dir" 2>/dev/null || sync
    printf 'marker=%s\nstate=critical-actions-complete\n' \
        "$(basename "$marker")" >"$ready_tmp"
    sync "$ready_tmp" 2>/dev/null || sync
    if ! mv -n -- "$ready_tmp" "$ready" || [[ -e $ready_tmp ]]; then
        rm -f -- "$ready_tmp"
        [[ -s $ready ]] || return 1
    fi
    sync "$ready" 2>/dev/null || sync
    sync -f "$marker_dir" 2>/dev/null || sync
}

publish_child_exit_notice() {
    local marker=$1 case_pid=$2
    local notice="${marker}.child-exit" ready="${marker}.child-exit.ready"
    local notice_tmp="${notice}.tmp.$BASHPID.$RANDOM"
    local ready_tmp="${ready}.tmp.$BASHPID.$RANDOM" marker_dir
    marker_dir=$(dirname "$marker")
    printf 'case_pid=%s\nstate=parent-observed-child-exit\n' \
        "$case_pid" >"$notice_tmp"
    sync "$notice_tmp" 2>/dev/null || sync
    mv -n -- "$notice_tmp" "$notice" || true
    rm -f -- "$notice_tmp"
    [[ -s $notice ]] || return 1
    sync "$notice" 2>/dev/null || sync
    sync -f "$marker_dir" 2>/dev/null || sync
    printf 'state=child-exit-notice-complete\n' >"$ready_tmp"
    sync "$ready_tmp" 2>/dev/null || sync
    mv -n -- "$ready_tmp" "$ready" || true
    rm -f -- "$ready_tmp"
    [[ -s $ready ]] || return 1
    sync "$ready" 2>/dev/null || sync
    sync -f "$marker_dir" 2>/dev/null || sync
}

watcher_stop_case_and_publish() {
    local marker=$1 status=$2 case_pid=$3 lost_devices=$4
    local foreign_processes=$5 case_identity=$6
    local monitor_detail=${7:-watcher-detected}
    local child_state=stopped
    signal_bound_process "$telemetry_pid" "$telemetry_identity" TERM || true
    if [[ -s ${marker}.child-exit && -s ${marker}.child-exit.ready ]]; then
        child_state=parent-observed-child-exit-no-signal
    elif pid_matches_identity "$case_pid" "$case_identity"; then
        signal_bound_process "$case_pid" "$case_identity" TERM || true
    else
        child_state=identity-not-live-or-mismatched-no-signal
    fi
    for _ in {1..20}; do
        pid_matches_identity "$case_pid" "$case_identity" || break
        sleep 0.1
    done
    if [[ ! -s ${marker}.child-exit &&
          ! -s ${marker}.child-exit.ready ]] &&
            pid_matches_identity "$case_pid" "$case_identity"; then
        signal_bound_process "$case_pid" "$case_identity" KILL || true
        for _ in {1..20}; do
            pid_matches_identity "$case_pid" "$case_identity" || break
            sleep 0.1
        done
    fi
    if pid_matches_identity "$case_pid" "$case_identity"; then
        child_state=still-present-after-bounded-kill
    fi
    publish_watch_marker "$marker" "$status" "$case_pid" "$lost_devices" \
        "$foreign_processes" "${monitor_detail};child_state=${child_state}"
}

start_telemetry() {
    local output=$1
    stdbuf -oL -eL nvidia-smi \
        --query-gpu=timestamp,index,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.free \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" >"$output" 2>&1 &
    telemetry_pid=$!
    telemetry_identity=$(capture_process_identity "$telemetry_pid") ||
        die "could not capture immutable identity for telemetry PID $telemetry_pid; helper may still be live and was not signaled"
}

start_telemetry_watch() {
    local output=$1 marker=$2 case_pid=$3 case_identity=$4
    local watcher_gate="${marker}.watcher-gate.$BASHPID.$RANDOM"
    (
        for _ in {1..100}; do
            [[ -e $watcher_gate ]] && break
            sleep 0.01
        done
        [[ -e $watcher_gate ]] || exit 125
        rm -f -- "$watcher_gate"
        while pid_is_live "$telemetry_pid"; do
            local watch_status= foreign_processes= lost_devices= recent_tail=
            recent_tail=$(tail -n 24 "$output" 2>/dev/null || true)
            if grep -Eiq 'GPU is lost|GPU requires reset|GPU Unavailable|Critical Xid|Unknown Error|ERR!|Unable to determine' \
                    <<<"$recent_tail"; then
                watch_status=lost-device-detected
                lost_devices=$(awk -F, '
                    function add(idx, bus, key) {
                        key=idx "@" bus
                        if (!seen[key]++) {
                            if (out != "") out=out ","
                            out=out key
                        }
                    }
                    {
                        lower=tolower($0)
                        if (lower !~ /gpu is lost|gpu requires reset|gpu unavailable|critical xid|unknown error|err!|unable to determine/) next
                        idx=$2; bus=$3
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", bus)
                        if (idx ~ /^[0-9]+$/) {
                            add(idx, bus)
                            next
                        }
                        if (index($0, "00000000:02:00.0")) add("0", "00000000:02:00.0")
                        if (index($0, "00000000:03:00.0")) add("1", "00000000:03:00.0")
                        if (index($0, "00000000:81:00.0")) add("2", "00000000:81:00.0")
                        if (index($0, "00000000:82:00.0")) add("3", "00000000:82:00.0")
                    }
                    END {print out}
                ' <<<"$recent_tail")
                [[ -n $lost_devices ]] || lost_devices=unknown
            elif printf '%s\n' "$recent_tail" |
                    awk -F, -v profile="$REQUIRED_POWER_LIMITS_W" '
                        BEGIN {split(profile, required, ",")}
                        {
                            index_field=$2
                            limit=$7
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", index_field)
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", limit)
                            sub(/[[:space:]]+W$/, "", limit)
                            if (index_field ~ /^[0-9]+$/ &&
                                limit ~ /^[0-9]+([.][0-9]+)?$/ &&
                                limit + 0.0 != required[index_field + 1] + 0.0) found=1
                        }
                        END {exit !found}
                    '; then
                watch_status=power-limit-drift
            elif foreign_processes=$(list_foreign_compute_processes "$case_pid") &&
                    [[ -n ${foreign_processes//[[:space:]]/} ]]; then
                watch_status=foreign-compute-process
            fi
            if [[ -n $watch_status ]]; then
                printf 'telemetry-watch: %s detected\n' "$watch_status" >>"$output"
                if [[ $ONE_SHOT == 1 ]]; then
                    watcher_stop_case_and_publish "$marker" "$watch_status" \
                        "$case_pid" "$lost_devices" "$foreign_processes" \
                        "$case_identity"
                else
                    printf 'timestamp_utc=%s\nstatus=%s\ncase_pid=%s\nrequired_power_limits_w=%s\nlost_devices=%s\nforeign_processes=%q\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$watch_status" \
                        "$case_pid" "$REQUIRED_POWER_LIMITS_W" "$lost_devices" \
                        "$foreign_processes" >"$marker"
                    sync "$marker" 2>/dev/null || sync
                    kill "$telemetry_pid" >/dev/null 2>&1 || true
                    kill -TERM "$case_pid" >/dev/null 2>&1 || true
                    for _ in {1..20}; do
                        kill -0 "$case_pid" >/dev/null 2>&1 || exit 0
                        sleep 0.1
                    done
                    kill -KILL "$case_pid" >/dev/null 2>&1 || true
                fi
                exit 0
            fi
            sleep 0.1
        done
        if [[ $ONE_SHOT == 1 ]]; then
            recent_tail=$(tail -n 24 "$output" 2>/dev/null || true)
            watch_status=telemetry-monitor-failed
            lost_devices=unknown
            if grep -Eiq 'GPU is lost|GPU requires reset|GPU Unavailable|Critical Xid|Unknown Error|ERR!|Unable to determine' \
                    <<<"$recent_tail"; then
                watch_status=lost-device-detected
            elif [[ -s ${marker}.child-exit &&
                    -s ${marker}.child-exit.ready ]] &&
                    grep -Fxq 'state=child-exit-notice-complete' \
                        "${marker}.child-exit.ready"; then
                # Parent-observed completion is not itself a monitor failure;
                # the parent will obtain the cached status with wait after
                # this watcher exits.  Never signal that now-dead bare PID.
                printf 'telemetry-watch: child exit notice consumed during final scan\n' \
                    >>"$output"
                exit 0
            fi
            printf 'telemetry-watch: %s detected during final scan\n' \
                "$watch_status" >>"$output"
            watcher_stop_case_and_publish "$marker" "$watch_status" \
                "$case_pid" "$lost_devices" '' "$case_identity" \
                telemetry-exited-final-scan
        fi
    ) &
    telemetry_watch_pid=$!
    telemetry_watch_identity=$(capture_process_identity "$telemetry_watch_pid") ||
        die "could not capture immutable identity for watcher PID $telemetry_watch_pid; helper may still be live and was not signaled"
    : >"$watcher_gate"
    sync "$watcher_gate" 2>/dev/null || sync
}

last_progress_fields() {
    local progress=$1
    if [[ -s $progress ]]; then
        awk -F, 'NR>1 {phase=$3; event=$4; current=$5; total=$6}
            END {printf "%s\t%s\t%s\t%s", phase, event, current, total}' \
            "$progress"
    else
        printf 'not-started\tno-progress-journal\t0\t0'
    fi
}

write_result() {
    local result=$1 variant=$2 status=$3 exit_status=$4 progress=$5 log=$6
    local fields last_phase last_event last_current last_total
    fields=$(last_progress_fields "$progress")
    IFS=$'\t' read -r last_phase last_event last_current last_total <<<"$fields"
    local q8_begin q8_complete row_begin row_complete
    q8_begin=$(grep -Fc 'q8 partner transfer audit event=begin' "$log" 2>/dev/null || true)
    q8_complete=$(grep -Fc 'q8 partner transfer audit event=complete' "$log" 2>/dev/null || true)
    row_begin=$(grep -Fc 'decode indexer row audit event=begin' "$log" 2>/dev/null || true)
    row_complete=$(grep -Fc 'decode indexer row audit event=complete' "$log" 2>/dev/null || true)
    {
        printf 'variant=%s\nstatus=%s\nexit_status=%s\n' \
            "$variant" "$status" "$exit_status"
        printf 'last_phase=%s\nlast_event=%s\nlast_current=%s\nlast_total=%s\n' \
            "$last_phase" "$last_event" "$last_current" "$last_total"
        printf 'q8_transfer_begin_checkpoints=%s\nq8_transfer_complete_checkpoints=%s\n' \
            "$q8_begin" "$q8_complete"
        printf 'indexer_row_begin_events=%s\nindexer_row_complete_events=%s\n' \
            "$row_begin" "$row_complete"
    } >"$result"
    sync "$result" 2>/dev/null || sync
}

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke tests/test_engine_mgpu_placement \
         tests/test_gpu_xdev tests/cuda_device_identity)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
    "${clean[@]}" ./tests/test_engine_mgpu_placement \
        >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/placement-tests.log" >&2
            die "CPU placement tests failed"
        }
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=device-identity
capture_and_validate_cuda_identity ||
    die "CUDA ordinal, nvidia-smi identity, or expected NVLink pair validation failed"

if [[ $SKIP_BUILD == 0 ]]; then
    assert_no_compute_processes ||
        die "foreign GPU compute process present before CUDA smoke tests"
    "${clean[@]}" ./tests/cuda_long_context_smoke \
        >"$OUTPUT_DIR/smoke.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/smoke.log" >&2
            die "CUDA long-context smoke failed"
        }
    assert_no_compute_processes ||
        die "foreign GPU compute process present before ordered-copy tests"
    "${clean[@]}" ./tests/test_gpu_xdev ordered-dst-copy \
        >"$OUTPUT_DIR/ordered-dst-copy-tests.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/ordered-dst-copy-tests.log" >&2
            die "ordered destination-stream peer-copy tests failed"
        }
fi

phase=manifest
gpu_health_snapshot "$OUTPUT_DIR/health/initial.log" ||
    die "initial four-GPU health check failed"
validate_power_limits ||
    die "selected GPUs must match physical-index power profile ${REQUIRED_POWER_LIMITS_W} W"
if [[ $RESUME == 0 || ! -s $OUTPUT_DIR/manifest.txt ]]; then
    {
        printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$(git branch --show-current)"
        printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
            "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
        printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
            "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
        printf 'cuda_device_order=%s\n' "$CUDA_DEVICE_ORDER"
        printf 'cuda_identity_validation=trusted-baseline-normalized-bus-uuid-join-and-nvml-topology\n'
        printf 'trusted_selected_identity_baseline=speed-bench/sm75-small-bar1-expected-device-identity.csv\n'
        printf 'trusted_selected_identity_baseline_sha256=%s\n' \
            "$(sha256sum "$EXPECTED_SELECTED_IDENTITY" | awk '{print $1}')"
        printf 'cuda_identity_raw_inventory=provenance/cuda-device-inventory.raw.csv\n'
        printf 'cuda_identity_inventory=provenance/cuda-device-inventory.csv\n'
        printf 'nvidia_identity_raw_inventory=provenance/nvidia-device-inventory.raw.csv\n'
        printf 'nvidia_identity_inventory=provenance/nvidia-device-inventory.csv\n'
        printf 'nvidia_topology_raw_inventory=provenance/nvidia-topology.txt\n'
        printf 'nvidia_topology_inventory=provenance/nvidia-topology.csv\n'
        printf 'selected_device_identity=provenance/selected-device-identity.csv\n'
        printf 'logical_pair_0=cuda_ordinal_0_cuda_ordinal_1\n'
        printf 'logical_pair_1=cuda_ordinal_3_cuda_ordinal_2\n'
        printf 'small_bar1_pair=%s\nvariants=%s\none_shot=%s\n' \
            "$SMALL_BAR1_PAIR" "$VARIANTS" "$ONE_SHOT"
        printf 'one_shot_timeout_seconds=%s\n' "$ONE_SHOT_TIMEOUT_SECONDS"
        printf 'attention_phase_audit_layer=%s\nattention_phase_audit_pos=%s\n' \
            "$ATTN_PHASE_AUDIT_LAYER" "$ATTN_PHASE_AUDIT_POS"
        printf 'attention_end_fence_layer=%s\nattention_end_fence_pos=%s\n' \
            "$ATTN_END_FENCE_LAYER" "$ATTN_END_FENCE_POS"
        printf 'attention_row_boundary_end_layer=%s\nattention_row_boundary_entry_layer=%s\n' \
            "$ATTN_ROW_BOUNDARY_END_LAYER" "$ATTN_ROW_BOUNDARY_ENTRY_LAYER"
        printf 'attention_row_boundary_pos=%s\n' "$ATTN_ROW_BOUNDARY_POS"
        printf 'pp_tokens=%s\ntg_tokens=%s\nrepeats=%s\n' \
            "$PP_TOKENS" "$TG_TOKENS" "$REPEATS"
        printf 'required_power_limits_w=%s\n' "$REQUIRED_POWER_LIMITS_W"
        printf 'partner_work_retained=yes\ndecode_indexer_pair_fallback=home\n'
        printf 'host_bounce_scope=q8_partner_activation_and_result_only\n'
        printf 'serialization_scope=q8_partner_projection_pair_only\n'
        printf 'attention_copy_scheduling_scope=pair0_query_and_gather_independent\n'
        printf 'attention_copy_scheduling_transport=ordered_direct_peer_no_fallback\n'
        printf 'attention_host_bounce_scope=pair0_attention_owned_copies_only\n'
        printf 'attention_q8_host_bounce_scope=pair0_attention_owned_and_q8_partner_copies\n'
        printf 'attention_q8_async_completion_scope=every_pair0_partner_q8_call_no_layer_selector\n'
        printf 'attention_q8_async_completion_marker=partner_default_stream_mapped_host_then_dedicated_event\n'
        printf 'attention_q8_async_completion_interpretation=positive_only_absence_is_inconclusive\n'
        printf 'attention_q8_pre_gather_fence_scope=every_pair0_partner_q8_call_immediately_before_result_d2h\n'
        printf 'attention_q8_pre_gather_fence_boundary=post_marker_event_synchronize_and_exact_marker_validation\n'
        printf 'attention_q8_pre_gather_fence_armed_record=ds4: CUDA q8 partner pre-gather armed current_sequence=N marker_sequence=N marker_complement=X home_tier=H home_device=HD partner_tier=P partner_device=PD\n'
        printf 'attention_q8_pre_gather_fence_armed_breadcrumb=every_confirmed_call_before_result_gather_fflush_only\n'
        printf 'attention_q8_pre_gather_fence_returned_record=ds4: CUDA q8 partner pre-gather returned current_sequence=N result_gather_status=success home_tier=H home_device=HD partner_tier=P partner_device=PD\n'
        printf 'attention_q8_pre_gather_fence_returned_breadcrumb=every_successful_result_gather_return_fflush_only\n'
        printf 'attention_q8_pre_gather_fence_success_checkpoints=first_and_every_64_confirmed_calls\n'
        printf 'attention_q8_pre_gather_fence_failure_records=durable_with_result_d2h_and_result_h2d_attempt_and_completion_state\n'
        printf 'attention_q8_pre_gather_fence_completed_validation=alternating_contiguous_armed_returned_pairs_required\n'
        printf 'attention_q8_pre_gather_fence_return_not_observed=armed_without_matching_returned_is_not_proof_of_failure\n'
        printf 'attention_q8_pre_gather_fence_durability=two_unique_fence_arm_fprintf_fflush_records_per_successful_call_no_per_call_fsync_full_checkpoints_sparse_failures_durable\n'
        printf 'attention_q8_pre_gather_fence_control=attention-q8-async-completion\n'
        printf 'attention_q8_pre_gather_fence_control_rejects=fence_armed_returned_records\n'
        printf 'attention_q8_pre_gather_fence_interpretation=failure_surface_boundary_not_root_cause\n'
        printf 'attention_q8_pre_gather_observed_result_gathers_returned=32\n'
        printf 'attention_q8_pre_gather_observed_next_call=33_activation_h2d_first_surfaced_error\n'
        printf 'attention_q8_activation_fence_scope=pair0_device_sync_and_exact_marker_before_activation_h2d_plus_existing_pre_gather_bracket\n'
        printf 'attention_q8_activation_fence_predecessor=attention-q8-pre-gather-fence\n'
        printf 'attention_q8_activation_fence_contract=fresh_one_shot_only_no_resume_no_control_rerun\n'
        printf 'attention_q8_activation_fence_perturbation=device_sync_marker_validation_and_host_logging\n'
        printf 'attention_q8_activation_fence_interpretation=failure_surface_boundary_not_root_cause\n'
        printf 'attention_q8_phase_audit_scope=pair0_q8_partner_phase_checkpoints_and_compute_sync\n'
        printf 'attention_q8_targeted_phase_audit_binding=%s\n' "$Q8_TARGETED_BINDING_LABEL"
        printf 'attention_q8_targeted_phase_audit_weight_offset=%s\n' "$Q8_TARGETED_WEIGHT_OFFSET"
        printf 'attention_q8_targeted_phase_audit_passed_label=%s\n' "$Q8_TARGET_PASSED_LABEL"
        printf 'attention_q8_targeted_phase_audit_shape=%sx%sx%s\n' \
            "$Q8_TARGET_TOKENS" "$Q8_TARGET_IN_DIM" "$Q8_TARGET_OUT_DIM"
        printf 'attention_q8_targeted_phase_audit_transfer_bytes=%s\n' \
            "$Q8_TARGET_TRANSFER_BYTES"
        printf 'attention_q8_targeted_phase_audit_result_bytes=%s\n' \
            "$Q8_TARGET_RESULT_BYTES"
        printf 'attention_q8_targeted_phase_audit_expected_sequences=%s\n' \
            "$Q8_TARGET_EXPECTED_SEQUENCES"
        printf 'attention_q8_l14_l15_phase_audit_targets=%s\n' "$Q8_WINDOW_TARGETS"
        printf 'attention_q8_l14_l15_phase_audit_expected_weight_bytes=%s\n' \
            "$Q8_TARGET_WEIGHT_BYTES"
        printf 'attention_q8_l14_l15_phase_audit_expected_per_binding=%s\n' \
            "$Q8_TARGET_EXPECTED_SEQUENCES"
        printf 'attention_q8_l14_l15_phase_audit_expected_total_sequences=%s\n' \
            "$Q8_WINDOW_EXPECTED_TOTAL_SEQUENCES"
        printf 'attention_q8_l14_l15_phase_audit_target_preflight=exact-partner-tuples-against-materialized-binding-inventory\n'
        printf 'attention_q8_l12_phase_audit_target=%s\n' "$Q8_L12_TARGET"
        printf 'attention_q8_l12_phase_audit_binding=%s\n' "$Q8_L12_BINDING_LABEL"
        printf 'attention_q8_l12_phase_audit_weight_offset=%s\n' "$Q8_L12_WEIGHT_OFFSET"
        printf 'attention_q8_l12_phase_audit_passed_label=%s\n' "$Q8_TARGET_PASSED_LABEL"
        printf 'attention_q8_l12_phase_audit_shape=%sx%sx%s\n' \
            "$Q8_TARGET_TOKENS" "$Q8_TARGET_IN_DIM" "$Q8_TARGET_OUT_DIM"
        printf 'attention_q8_l12_phase_audit_expected_weight_bytes=%s\n' \
            "$Q8_TARGET_WEIGHT_BYTES"
        printf 'attention_q8_l12_phase_audit_transfer_bytes=%s\n' \
            "$Q8_TARGET_TRANSFER_BYTES"
        printf 'attention_q8_l12_phase_audit_result_bytes=%s\n' \
            "$Q8_TARGET_RESULT_BYTES"
        printf 'attention_q8_l12_phase_audit_skip_occurrences=%s\n' \
            "$Q8_L12_SKIP_OCCURRENCES"
        printf 'attention_q8_l12_phase_audit_max_occurrences=%s\n' \
            "$Q8_L12_MAX_OCCURRENCES"
        printf 'attention_q8_l12_phase_audit_selected_occurrence=%s\n' \
            "$Q8_L12_SELECTED_OCCURRENCE"
        printf 'attention_q8_l12_phase_audit_expected_sequences=%s\n' \
            "$Q8_L12_EXPECTED_SEQUENCES"
        printf 'attention_q8_l12_phase_audit_target_preflight=one-exact-partner-tuple-against-materialized-binding-inventory\n'
        printf 'attention_copy_scheduling_preflight=build-smoke-ordered-copy-every-run\n'
        printf 'attention_row_shadow_gather_chunk_bytes=16777216\n'
        printf 'attention_row_shadow_gather_chunk_readiness=one-destination-event-then-one-source-wait\n'
        printf 'attention_row_shadow_gather_chunk_completion=one-source-event-then-one-destination-wait\n'
        printf 'attention_row_shadow_gather_paced_inter_chunk=one-destination-ack-event-then-one-source-wait\n'
        printf 'attention_row_shadow_gather_paced_completion=per-chunk-source-event-then-destination-wait\n'
        printf 'q8_transfer_audit=begin_complete_64-call-checkpoints\n'
        printf 'indexer_transfer_audit=every-dispatch-begin-complete\n'
        printf 'external_nvlink_counters=disabled-no-external-compute-workload\n'
    } >"$OUTPUT_DIR/manifest.txt"
    git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
    git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
    nvidia-smi -q >"$OUTPUT_DIR/provenance/nvidia-smi-q.txt" 2>&1 || true
    nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt" 2>&1 || true
    printf 'timestamp_utc\tvariant\trepeat\tstatus\texit_status\tlast_phase\tlast_event\n' \
        >"$OUTPUT_DIR/run-journal.tsv"
fi

validate_admission_retained() {
    local bindings=$1
    awk -F, '
        NR==1 {
            for (i=1;i<=NF;i++) h[$i]=i
            next
        }
        $(h["consumer_device"])==0 && $(h["resident_device"])==1 &&
        $(h["partner_offload"])==1 {found++}
        END {exit !(found>0)}
    ' "$bindings"
}

log_line_has() {
    local log=$1 first=$2 second=$3
    awk -v first="$first" -v second="$second" '
        index($0, first) && index($0, second) { found=1 }
        END { exit !found }
    ' "$log"
}

log_line_has_all() {
    local log=$1 line needle matched
    shift
    while IFS= read -r line; do
        matched=1
        for needle in "$@"; do
            if [[ $line != *"$needle"* ]]; then
                matched=0
                break
            fi
        done
        (( matched )) && return 0
    done < "$log"
    return 1
}

validate_full_production_load() {
    local csv=$1
    awk -F, '
        NR==1 {
            for (i=1;i<=NF;i++) if ($i=="prefill_tps") col=i
            next
        }
        NR==2 {ok=(col>0 && $col>=500.0)}
        END {exit !ok}
    ' "$csv"
}

validate_attention_copy_schedule() {
    local log=$1 home=$2 partner=$3 query_schedule=$4 gather_schedule=$5
    grep -Eq \
        "prefill attention row audit dispatch=split .*home=${home} partner=${partner} .*query_copy_stream=${query_schedule} gather_copy_stream=${gather_schedule}" \
        "$log"
}

validate_attention_copy_transport() {
    local log=$1 home=$2 partner=$3 query_transport=$4 gather_transport=$5
    grep -Eq \
        "prefill attention row audit dispatch=split .*home=${home} partner=${partner} .*query_copy_transport=${query_transport} gather_copy_transport=${gather_transport}" \
        "$log"
}

validate_success_path() {
    local variant=$1 log=$2 bindings=$3 csv=$4 require_load=${5:-1}
    validate_admission_retained "$bindings" || return 1
    grep -Fq 'CUDA q8 fp16 benefit plan materialized 344/344 candidates' "$log" ||
        return 1
    grep -Fq 'CUDA EP forced pipeline split 22/21' "$log" || return 1
    grep -Fq 'SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4 (production default)' \
        "$log" || return 1
    ! grep -Eiq 'illegal memory|GPU is lost|Unknown Error|CUDA .* failed' "$log" ||
        return 1
    grep -Fq 'q8 partner transfer audit event=begin home_tier=0' "$log" ||
        return 1
    grep -Fq 'q8 partner transfer audit event=begin home_tier=1' "$log" ||
        return 1
    case "$variant" in
        attention-off)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            grep -Fq 'prefill attention row split pair-scoped disable: logical-pairs=0' \
                "$log" || return 1
            ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-row-query-shadow|attention-row-partner-shadow|attention-row-gather-shadow|attention-row-gather-dst-shadow|attention-row-gather-chunk16-shadow|attention-row-gather-chunk16-paced-shadow|attention-row-gather-scratch-paced-shadow|attention-row-gather-source-scratch-paced-shadow|attention-row-gather-preinitialized-source-paced-shadow|attention-row-gather-preinitialized-source-no-partner-paced-shadow|attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow)
            local shadow_phase
            shadow_phase=$(row_shadow_phase "$variant") || return 1
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row transport override for logical pair 0' \
                "$log" || return 1
            grep -Fxq \
                "ds4: CUDA prefill attention row shadow audit enabled: logical-pairs=0 phase=$shadow_phase; production direct-P2P transport retained; accepted output recomputed on home" \
                "$log" || return 1
            grep -Eq \
                "prefill attention row shadow audit event=begin phase=$shadow_phase .*home=0 partner=2" \
                "$log" || return 1
            grep -Eq \
                "prefill attention row shadow audit event=complete phase=$shadow_phase .*home=0 partner=2" \
                "$log" || return 1
            if [[ $variant == attention-row-gather-chunk16-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA chunked default-stream peer copy scheduled: source_tier=2 destination_tier=0 bytes=33554432 chunk_bytes=16777216 submissions=2 readiness=one-destination-event-one-source-wait completion=one-source-event-one-destination-wait' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-chunk16 kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-chunk16 kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
            fi
            if [[ $variant == attention-row-gather-chunk16-paced-shadow ]]; then
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-chunk16-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-chunk16-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
            fi
            if [[ $variant == attention-row-gather-scratch-paced-shadow ]]; then
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$SCRATCH_GATHER_MARKER" \
                    'source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'scratch_allocation_bytes=' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
            fi
            if [[ $variant == attention-row-gather-source-scratch-paced-shadow ]]; then
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-source-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-source-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$SOURCE_SCRATCH_GATHER_MARKER" \
                    'production_source_tier=2 staged_source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_staging=partner-default-stream-d2d' \
                    'source_scratch_allocation_bytes=' \
                    'destination_scratch_allocation_bytes=' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
            fi
            if [[ $variant == attention-row-gather-preinitialized-source-paced-shadow ]]; then
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_GATHER_ARMED_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'production_source_read=no' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-preinitialized-source-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-preinitialized-source-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_GATHER_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'production_source_read=no' \
                    'source_scratch_allocation_bytes=' \
                    'destination_scratch_allocation_bytes=' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
            fi
            if [[ $variant == attention-row-gather-preinitialized-source-no-partner-paced-shadow ]]; then
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_NO_PARTNER_GATHER_ARMED_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'partner_attention=omitted' \
                    'production_source_read=no' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-preinitialized-source-no-partner-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-preinitialized-source-no-partner-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_NO_PARTNER_GATHER_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'partner_attention=omitted' \
                    'production_source_read=no' \
                    'source_scratch_allocation_bytes=' \
                    'destination_scratch_allocation_bytes=' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
            fi
            if [[ $variant == attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow ]]; then
                log_line_has_all "$log" "$PARTNER_OUTPUT_SCRATCH_ARMED_MARKER" \
                    'output_tier=2' \
                    'output_offset_bytes=134217728' \
                    'output_bytes=33554432' \
                    'partner_attention=retained' \
                    'production_output_write=no production_output_read=no' \
                    'output=dedicated-partner-allocation' \
                    'accepted_output=full-home-recompute' || return 1
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_PARTNER_OUTPUT_SCRATCH_GATHER_ARMED_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'partner_attention=retained' \
                    'production_source_read=no' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-preinitialized-source-partner-output-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=complete phase=result-gather-preinitialized-source-partner-output-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_PARTNER_OUTPUT_SCRATCH_GATHER_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'partner_attention=retained' \
                    'production_source_read=no' \
                    'source_scratch_allocation_bytes=268435456' \
                    'destination_scratch_allocation_bytes=268435456' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute' || return 1
            fi
            log_line_has "$log" 'q8 partner transfer audit event=begin home_tier=0' \
                'transport=peer' || return 1
            log_line_has "$log" 'q8 partner transfer audit event=begin home_tier=1' \
                'transport=peer' || return 1
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row compute pair-scoped disable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' \
                "$log" || return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' \
                "$log" || return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" || return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" || return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" || return 1
            ! grep -Fq 'CUDA q8 partner compute fence audit enabled:' "$log" || return 1
            ! grep -Fq 'CUDA q8 partner direct gather audit enabled:' "$log" || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-host-bounce)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'q8 partner transfer audit event=begin home_tier=0' \
                'transport=peer' || return 1
            log_line_has "$log" 'q8 partner transfer audit event=begin home_tier=1' \
                'transport=peer' || return 1
            grep -Fq 'prefill attention row transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            grep -Eq 'prefill attention host-bounce checkpoint event=complete .*pos=512 tokens=512 home=0 partner=2' \
                "$log" || return 1
            for cache_class in raw attn-comp index; do
                grep -Fq "prefill attention cache mirror transport=host-bounce home_tier=0 partner_tier=2 class=$cache_class event=complete" \
                    "$log" || return 1
            done
            ! grep -Fq 'prefill attention cache mirror transport=host-bounce home_tier=1' \
                "$log" || return 1
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            validate_attention_copy_schedule "$log" 0 2 source source || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            validate_attention_copy_transport "$log" 0 2 host-bounce host-bounce ||
                return 1
            validate_attention_copy_transport "$log" 1 3 peer peer || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split .*home=0 partner=2 .*query_copy_transport=peer|prefill attention row audit dispatch=split .*home=0 partner=2 .*gather_copy_transport=peer' \
                "$log" || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split .*home=1 partner=3 .*query_copy_transport=host-bounce|prefill attention row audit dispatch=split .*home=1 partner=3 .*gather_copy_transport=host-bounce' \
                "$log" || return 1
            grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=0 partner=2 .*topk_copy_transport=partner-local' \
                "$log" || return 1
            grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=1 partner=3 .*topk_copy_transport=partner-local' \
                "$log" || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=[01] partner=[23] .*topk_copy_transport=(peer|host-bounce)' \
                "$log" || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" ||
                return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-q8-host-bounce|attention-q8-async-completion|attention-q8-pre-gather-fence|attention-q8-activation-fence|attention-q8-global-compute-fence|attention-q8-direct-gather-fence|attention-q8-rows-serialized|attention-q8-row-compute-off|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-q8-l12-phase-audit)
            grep -Fq 'partner transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            grep -Fq 'prefill attention row transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1

            # Calls=1 proves that each transport was selected during the
            # untimed 512-token warmup.  The later fixed counts are reached
            # only after the measured 32K prefill has begun for this model and
            # placement, so a warmup-only survival cannot satisfy this arm.
            grep -Eq 'q8 partner transfer audit event=begin home_tier=0 partner_tier=2 calls=1 .*transport=host-bounce serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=0 partner_tier=2 calls=1 ' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=begin home_tier=0 partner_tier=2 calls=128 .*transport=host-bounce serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=0 partner_tier=2 calls=128 ' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=begin home_tier=1 partner_tier=3 calls=1 .*transport=peer serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=1 partner_tier=3 calls=1 ' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=begin home_tier=1 partner_tier=3 calls=64 .*transport=peer serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=1 partner_tier=3 calls=64 ' \
                "$log" || return 1
            ! grep -Eq 'q8 partner transfer audit event=begin home_tier=0 .*transport=peer' \
                "$log" || return 1
            ! grep -Eq 'q8 partner transfer audit event=begin home_tier=1 .*transport=host-bounce' \
                "$log" || return 1

            for cache_class in raw attn-comp index; do
                grep -Fq "prefill attention cache mirror transport=host-bounce home_tier=0 partner_tier=2 class=$cache_class event=complete" \
                    "$log" || return 1
            done
            ! grep -Fq 'prefill attention cache mirror transport=host-bounce home_tier=1' \
                "$log" || return 1
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            if [[ $variant == attention-q8-row-compute-off ]]; then
                grep -Fxq "$ROW_COMPUTE_OFF_MARKER" "$log" || return 1
                ! grep -Fq 'prefill attention host-bounce checkpoint event=complete ' \
                    "$log" || return 1
                ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' \
                    "$log" || return 1
                ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' \
                    "$log" || return 1
                ! grep -Eq 'prefill attention row audit dispatch=split .*home=0 partner=2' \
                    "$log" || return 1
                grep -Fq 'prefill attention query-row split enabled: tier 1 ' \
                    "$log" || return 1
                grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' \
                    "$log" || return 1
                validate_attention_copy_schedule "$log" 1 3 source source ||
                    return 1
                validate_attention_copy_transport "$log" 1 3 peer peer || return 1
                grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=1 partner=3 .*topk_copy_transport=partner-local' \
                    "$log" || return 1
            else
                grep -Eq 'prefill attention host-bounce checkpoint event=complete .*pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fq 'prefill attention query-row split enabled: tier 0 ' \
                    "$log" || return 1
                grep -Fq 'prefill attention query-row split enabled: tier 1 ' \
                    "$log" || return 1
                grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' \
                    "$log" || return 1
                grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' \
                    "$log" || return 1
                validate_attention_copy_schedule "$log" 0 2 source source || return 1
                validate_attention_copy_schedule "$log" 1 3 source source || return 1
                validate_attention_copy_transport "$log" 0 2 host-bounce host-bounce ||
                    return 1
                validate_attention_copy_transport "$log" 1 3 peer peer || return 1
                ! grep -Eq 'prefill attention row audit dispatch=split .*home=0 partner=2 .*query_copy_transport=peer|prefill attention row audit dispatch=split .*home=0 partner=2 .*gather_copy_transport=peer' \
                    "$log" || return 1
                ! grep -Eq 'prefill attention row audit dispatch=split .*home=1 partner=3 .*query_copy_transport=host-bounce|prefill attention row audit dispatch=split .*home=1 partner=3 .*gather_copy_transport=host-bounce' \
                    "$log" || return 1
                grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=0 partner=2 .*topk_copy_transport=partner-local' \
                    "$log" || return 1
                grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=1 partner=3 .*topk_copy_transport=partner-local' \
                    "$log" || return 1
            fi
            ! grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=[01] partner=[23] .*topk_copy_transport=(peer|host-bounce)' \
                "$log" || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" || return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" || return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" || return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-query-dst|attention-gather-dst|attention-both-dst)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            local query_schedule=source gather_schedule=source
            case "$variant" in
                attention-query-dst) query_schedule=destination ;;
                attention-gather-dst) gather_schedule=destination ;;
                attention-both-dst)
                    query_schedule=destination
                    gather_schedule=destination
                    ;;
            esac
            validate_attention_copy_schedule "$log" 0 2 \
                "$query_schedule" "$gather_schedule" || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" ||
                return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-phase-audit)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            for audit_phase in query-copy partner-attention home-attention result-gather; do
                log_line_has "$log" \
                    "prefill attention row phase audit event=complete phase=$audit_phase " \
                    "layer=$ATTN_PHASE_AUDIT_LAYER pos=$ATTN_PHASE_AUDIT_POS tokens=512 home=0 partner=2" ||
                    return 1
            done
            ! grep -Eq 'prefill attention row phase audit event=(submit-failed|device-switch-failed|sync-failed)' \
                "$log" || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-end-fence)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            for fence_target in partner home pair; do
                log_line_has "$log" \
                    "prefill attention row end fence event=complete target=$fence_target " \
                    "layer=$ATTN_END_FENCE_LAYER pos=$ATTN_END_FENCE_POS tokens=512 home=0 partner=2" ||
                    return 1
            done
            ! grep -Eq 'prefill attention row end fence event=(submit-failed|device-switch-failed|sync-failed|failed)' \
                "$log" || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-row-boundary-audit)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
                --validate-attention-row-boundary-log "$log" \
                "$ATTN_ROW_BOUNDARY_END_LAYER" \
                "$ATTN_ROW_BOUNDARY_ENTRY_LAYER" \
                "$ATTN_ROW_BOUNDARY_POS" || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        production)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            validate_attention_copy_schedule "$log" 0 2 source source || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" ||
                return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        partner-bounce)
            grep -Fq 'partner transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            ;;
        bounce-indexer-off)
            grep -Fq 'partner transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            grep -Fq 'decode indexer row split pair-scoped disable: logical-pairs=0' "$log" ||
                return 1
            if log_line_has "$log" 'decode indexer row audit event=begin' \
                    'home_tier=0 partner_tier=2'; then
                return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        partner-serialized)
            grep -Fq 'partner scheduling override for logical pair 0: projection-serialized' \
                "$log" || return 1
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            ;;
        indexer-off)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            grep -Fq 'decode indexer row split pair-scoped disable: logical-pairs=0' "$log" ||
                return 1
            if log_line_has "$log" 'decode indexer row audit event=begin' \
                    'home_tier=0 partner_tier=2'; then
                return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
    esac
    if [[ $variant == attention-q8-async-completion ||
          $variant == attention-q8-pre-gather-fence ||
          $variant == attention-q8-activation-fence ||
          $variant == attention-q8-global-compute-fence ]]; then
        grep -Fq \
            'CUDA q8 partner async completion audit enabled: logical_pairs=0 marker=partner-default-stream-mapped-host event=dedicated-post-marker interpretation=positive-only' \
            "$log" || return 1
        ! grep -Fq 'CUDA q8 partner async completion failure ' "$log" ||
            return 1
        ! grep -Eq 'CUDA q8 partner async completion (checkpoint|summary) home_tier=1 ' \
            "$log" || return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-async-completion-log "$log" || return 1
    fi
    if [[ $variant == attention-q8-pre-gather-fence ||
          $variant == attention-q8-activation-fence ||
          $variant == attention-q8-global-compute-fence ]]; then
        local fence_enable
        fence_enable='ds4: CUDA q8 partner pre-gather fence audit enabled: logical_pairs=0 boundary=post-marker-event-sync marker=exact-before-result-d2h'
        [[ $(grep -Fxc "$fence_enable" "$log" || true) == 1 ]] || return 1
        [[ $(grep -Fc 'CUDA q8 partner pre-gather fence audit enabled:' \
            "$log" || true) == 1 ]] || return 1
        ! grep -Fq 'CUDA q8 partner pre-gather fence failure ' "$log" ||
            return 1
        ! grep -Eq 'CUDA q8 partner pre-gather (fence (checkpoint|failure)|armed|returned) .*home_tier=1 ' \
            "$log" || return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-pre-gather-fence-log "$log" || return 1
    fi
    if [[ $variant == attention-q8-activation-fence ||
          $variant == attention-q8-global-compute-fence ]]; then
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-pre-activation-fence-log "$log" || return 1
    fi
    if [[ $variant == attention-q8-global-compute-fence ]]; then
        grep -Fxq \
            'ds4: CUDA q8 partner compute fence audit enabled: logical_pairs=0 boundary=post-submit-device-sync scope=every-selected-partner-call identity=dynamic' \
            "$log" || return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-compute-fence-log "$log" || return 1
    fi
    if [[ $variant == attention-q8-direct-gather-fence ||
          $variant == attention-q8-rows-serialized ||
          $variant == attention-q8-row-compute-off ]]; then
        grep -Fxq \
            'ds4: CUDA q8 partner compute fence audit enabled: logical_pairs=0 boundary=post-submit-device-sync scope=every-selected-partner-call identity=dynamic' \
            "$log" || return 1
        grep -Fxq \
            'ds4: CUDA q8 partner direct gather audit enabled: logical_pairs=0 boundary=compute-sync-to-synchronous-host-bounce mapped_host_marker=no event=no identity=dynamic' \
            "$log" || return 1
        ! grep -Fq 'CUDA q8 partner async completion audit enabled:' "$log" ||
            return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-compute-fence-log "$log" || return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-direct-gather-log "$log" || return 1
    fi
    if [[ $variant == attention-q8-rows-serialized ]]; then
        grep -Fxq \
            'ds4: CUDA prefill attention row compute serialization enabled: logical pair 0->2 order=partner-sync,home-sync; row ownership, kernels, caches, and transport retained' \
            "$log" || return 1
        ! grep -Fq \
            'CUDA prefill attention row compute serialization enabled: logical pair 1->3' \
            "$log" || return 1
        ! grep -Fq \
            'CUDA prefill attention serialized compute failed ' \
            "$log" || return 1
    fi
    if [[ $variant == attention-q8-async-completion ]]; then
        ! grep -Eq 'CUDA q8 partner pre-gather (fence|armed|returned) ' \
            "$log" || return 1
        ! grep -Eq 'CUDA q8 partner pre-activation (fence|armed|returned) ' \
            "$log" || return 1
    fi
    if [[ $variant == attention-q8-host-bounce ]]; then
        ! grep -Fq 'CUDA q8 partner async completion ' "$log" || return 1
        ! grep -Eq 'CUDA q8 partner pre-gather (fence|armed|returned) ' \
            "$log" || return 1
    fi
    if [[ $variant == attention-q8-l14-l15-phase-audit ]]; then
        grep -Fq \
            'CUDA q8 partner phase-audit target preflight validated 2 exact partner tuples against ' \
            "$log" || return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-l14-l15-log "$log" \
            "$Q8_TARGET_EXPECTED_SEQUENCES" || return 1
    fi
    if [[ $variant == attention-q8-l12-phase-audit ]]; then
        local l12_selection_count l12_skip_count
        local l12_skip_line l12_warmup_start_line l12_warmup_complete_line
        local l12_chunk_begin_line l12_selection_line
        local l12_skip_record l12_selection_record
        grep -Fq \
            'CUDA q8 partner phase-audit target preflight validated 1 exact partner tuples against ' \
            "$log" || return 1
        grep -Fq \
            "audited_pairs=${SMALL_BAR1_PAIR} expected_weight_bytes=${Q8_TARGET_WEIGHT_BYTES} expected_in=${Q8_TARGET_IN_DIM} expected_out=${Q8_TARGET_OUT_DIM} skip_occurrences=${Q8_L12_SKIP_OCCURRENCES} max_occurrences=${Q8_L12_MAX_OCCURRENCES}" \
            "$log" || return 1
        l12_skip_record="CUDA q8 partner phase audit skipped occurrence=${Q8_L12_SKIP_OCCURRENCES} binding_label=${Q8_L12_BINDING_LABEL} weight_offset=${Q8_L12_WEIGHT_OFFSET}"
        l12_selection_record="CUDA q8 partner phase audit selected occurrence=${Q8_L12_SELECTED_OCCURRENCE} sequence=1 binding_label=${Q8_L12_BINDING_LABEL} weight_offset=${Q8_L12_WEIGHT_OFFSET}"
        l12_skip_count=$(grep -Fc "$l12_skip_record" "$log" || true)
        l12_selection_count=$(grep -Fc \
            "$l12_selection_record" \
            "$log" || true)
        (( l12_skip_count == 1 )) || return 1
        (( l12_selection_count == 1 )) || return 1
        l12_warmup_start_line=$(grep -nFm1 \
            'starting untimed CUDA warm-up frontier' "$log" | cut -d: -f1 || true)
        l12_skip_line=$(grep -nFm1 "$l12_skip_record" "$log" | cut -d: -f1 || true)
        l12_warmup_complete_line=$(grep -nFm1 \
            'completed untimed CUDA warm-up frontier' "$log" | cut -d: -f1 || true)
        l12_chunk_begin_line=$(grep -nFm1 \
            'prefill fault breadcrumb event=chunk-begin' "$log" | cut -d: -f1 || true)
        l12_selection_line=$(grep -nFm1 "$l12_selection_record" "$log" | cut -d: -f1 || true)
        [[ -n $l12_warmup_start_line && -n $l12_skip_line &&
           -n $l12_warmup_complete_line && -n $l12_chunk_begin_line &&
           -n $l12_selection_line ]] || return 1
        (( l12_warmup_start_line < l12_skip_line &&
           l12_skip_line < l12_warmup_complete_line &&
           l12_warmup_complete_line < l12_chunk_begin_line &&
           l12_chunk_begin_line < l12_selection_line )) || return 1
    fi
    if [[ $variant == attention-q8-phase-audit ||
          $variant == attention-q8-targeted-phase-audit ||
          $variant == attention-q8-l12-phase-audit ]]; then
        # Require every observed pair-0 sequence to have an identical binding
        # identity and strict phase order. The original targeted arm must
        # contain all 65 calls established by the broad audit: one 512-token
        # warmup plus 64 measured 512-token microbatches. The layer-12 arm
        # instead skips the warmup occurrence and requires exactly the first
        # measured occurrence. Any pair-1 marker, failure, incomplete chain,
        # or missing selected target call invalidates the arm. The case
        # validation above independently proves pair 0 bounced and pair 1
        # retained direct transport.
        local required_binding= required_offset= required_passed=
        local required_tokens= required_in= required_out=
        local required_transfer= required_result= required_sequences=
        if [[ $variant == attention-q8-targeted-phase-audit ]]; then
            required_binding=$Q8_TARGETED_BINDING_LABEL
            required_offset=$Q8_TARGETED_WEIGHT_OFFSET
            required_passed=$Q8_TARGET_PASSED_LABEL
            required_tokens=$Q8_TARGET_TOKENS
            required_in=$Q8_TARGET_IN_DIM
            required_out=$Q8_TARGET_OUT_DIM
            required_transfer=$Q8_TARGET_TRANSFER_BYTES
            required_result=$Q8_TARGET_RESULT_BYTES
            required_sequences=$Q8_TARGET_EXPECTED_SEQUENCES
        elif [[ $variant == attention-q8-l12-phase-audit ]]; then
            required_binding=$Q8_L12_BINDING_LABEL
            required_offset=$Q8_L12_WEIGHT_OFFSET
            required_passed=$Q8_TARGET_PASSED_LABEL
            required_tokens=$Q8_TARGET_TOKENS
            required_in=$Q8_TARGET_IN_DIM
            required_out=$Q8_TARGET_OUT_DIM
            required_transfer=$Q8_TARGET_TRANSFER_BYTES
            required_result=$Q8_TARGET_RESULT_BYTES
            required_sequences=$Q8_L12_EXPECTED_SEQUENCES
        fi
        awk -v required_binding="$required_binding" \
            -v required_offset="$required_offset" \
            -v required_passed="$required_passed" \
            -v required_tokens="$required_tokens" \
            -v required_in="$required_in" \
            -v required_out="$required_out" \
            -v required_transfer="$required_transfer" \
            -v required_result="$required_result" \
            -v required_sequences="$required_sequences" '
            /ds4: CUDA q8 partner phase audit sequence=/ {
                sequence=event=stage=binding=passed=offset=home=partner=""
                tokens=in_dim=out_dim=transfer=result=""
                for (i=1; i<=NF; i++) {
                    split($i, field, "=")
                    if (field[1]=="sequence") sequence=field[2]
                    else if (field[1]=="event") event=field[2]
                    else if (field[1]=="stage") stage=field[2]
                    else if (field[1]=="binding_label") binding=field[2]
                    else if (field[1]=="passed_label") passed=field[2]
                    else if (field[1]=="weight_offset") offset=field[2]
                    else if (field[1]=="home_tier") home=field[2]
                    else if (field[1]=="partner_tier") partner=field[2]
                    else if (field[1]=="tokens") tokens=field[2]
                    else if (field[1]=="in") in_dim=field[2]
                    else if (field[1]=="out") out_dim=field[2]
                    else if (field[1]=="transfer_bytes") transfer=field[2]
                    else if (field[1]=="result_bytes") result=field[2]
                }
                if (home==1 || event ~ /-failed$/) bad=1
                if (home!=0) next
                if (partner!=2 || sequence !~ /^[0-9]+$/ ||
                    binding=="" || binding=="unavailable" ||
                    passed=="" || passed=="unavailable" ||
                    offset !~ /^[0-9]+$/) {
                    bad=1
                    next
                }
                if ((required_binding!="" && binding!=required_binding) ||
                    (required_offset!="" && offset!=required_offset) ||
                    (required_passed!="" && passed!=required_passed) ||
                    (required_tokens!="" && tokens!=required_tokens) ||
                    (required_in!="" && in_dim!=required_in) ||
                    (required_out!="" && out_dim!=required_out) ||
                    (required_transfer!="" && transfer!=required_transfer) ||
                    (required_result!="" && result!=required_result)) {
                    bad=1
                    next
                }
                key=sequence SUBSEP binding SUBSEP passed SUBSEP offset
                if (event=="begin" && stage=="activation-prepare") {
                    if (state[key]!=0) bad=1
                    else state[key]=1
                } else if (event=="activation-complete" &&
                           stage=="activation-copy") {
                    if (state[key]!=1) bad=1
                    else state[key]=2
                } else if (event=="pre-compute-sync-begin" &&
                           stage=="pre-compute-sync") {
                    if (state[key]!=2) bad=1
                    else state[key]=3
                } else if (event=="pre-compute-complete" &&
                           stage=="pre-compute-sync") {
                    if (state[key]!=3) bad=1
                    else state[key]=4
                } else if (event=="compute-submitted" && stage=="compute") {
                    if (state[key]!=4) bad=1
                    else state[key]=5
                } else if (event=="compute-complete" &&
                           stage=="compute-sync") {
                    if (state[key]!=5) bad=1
                    else state[key]=6
                } else if (event=="result-complete" &&
                           stage=="result-gather") {
                    if (state[key]!=6) bad=1
                    else state[key]=7
                }
            }
            END {
                observed=complete=0
                for (key in state) {
                    observed++
                    if (state[key]==7) complete++
                    else bad=1
                }
                if (required_sequences!="" &&
                    observed!=required_sequences) bad=1
                exit (bad || complete==0)
            }
        ' "$log" || return 1
    fi
}

validate_completed_path() {
    local variant=$1 log=$2 bindings=$3 csv=$4
    case "$variant" in
        attention-q8-host-bounce|attention-q8-async-completion|attention-q8-pre-gather-fence|attention-q8-activation-fence|attention-q8-global-compute-fence|attention-q8-direct-gather-fence|attention-q8-rows-serialized|attention-q8-row-compute-off|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-q8-l12-phase-audit)
            validate_success_path "$variant" "$log" "$bindings" "$csv" 0
            ;;
        *)
            validate_success_path "$variant" "$log" "$bindings" "$csv"
            ;;
    esac
}

failed_arm_activation_proven() {
    local variant=$1 log=$2
    case "$variant" in
        attention-q8-row-compute-off)
            grep -Fxq "$ROW_COMPUTE_OFF_MARKER" "$log"
            ;;
        attention-row-query-shadow|attention-row-partner-shadow|attention-row-gather-shadow|attention-row-gather-dst-shadow|attention-row-gather-chunk16-shadow|attention-row-gather-chunk16-paced-shadow|attention-row-gather-scratch-paced-shadow|attention-row-gather-source-scratch-paced-shadow|attention-row-gather-preinitialized-source-paced-shadow|attention-row-gather-preinitialized-source-no-partner-paced-shadow|attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow)
            local shadow_phase
            shadow_phase=$(row_shadow_phase "$variant") || return 1
            grep -Fxq \
                "ds4: CUDA prefill attention row shadow audit enabled: logical-pairs=0 phase=$shadow_phase; production direct-P2P transport retained; accepted output recomputed on home" \
                "$log" || return 1
            if [[ $variant == attention-row-gather-chunk16-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-chunk16 kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq \
                    'ds4: CUDA chunked default-stream peer copy scheduled: source_tier=2 destination_tier=0 bytes=33554432 chunk_bytes=16777216 submissions=2 readiness=one-destination-event-one-source-wait completion=one-source-event-one-destination-wait' \
                    "$log"
            elif [[ $variant == attention-row-gather-chunk16-paced-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-chunk16-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq "$PACED_CHUNK_MARKER" "$log"
            elif [[ $variant == attention-row-gather-scratch-paced-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                log_line_has_all "$log" "$SCRATCH_GATHER_MARKER" \
                    'source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'scratch_allocation_bytes=' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute'
            elif [[ $variant == attention-row-gather-source-scratch-paced-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-source-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                grep -Fxq "$PACED_CHUNK_MARKER" "$log" || return 1
                log_line_has_all "$log" "$SOURCE_SCRATCH_GATHER_MARKER" \
                    'production_source_tier=2 staged_source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_staging=partner-default-stream-d2d' \
                    'source_scratch_allocation_bytes=' \
                    'destination_scratch_allocation_bytes=' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute'
            elif [[ $variant == attention-row-gather-preinitialized-source-paced-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-preinitialized-source-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_GATHER_ARMED_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'production_source_read=no' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute'
            elif [[ $variant == attention-row-gather-preinitialized-source-no-partner-paced-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-preinitialized-source-no-partner-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$PREINITIALIZED_SOURCE_NO_PARTNER_GATHER_ARMED_MARKER" \
                    'production_source_tier=2 source_tier=2 destination_tier=0' \
                    'transfer_bytes=33554432' \
                    'source_initialization=one-time-partner-default-stream-zero-fill' \
                    'partner_attention=omitted' \
                    'production_source_read=no' \
                    'source=dedicated-partner-allocation' \
                    'destination=dedicated-home-allocation' \
                    'accepted_output=full-home-recompute'
            elif [[ $variant == attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow ]]; then
                grep -Fxq \
                    'ds4: CUDA prefill attention row shadow audit event=begin phase=result-gather-preinitialized-source-partner-output-scratch-paced kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2' \
                    "$log" || return 1
                log_line_has_all "$log" "$PARTNER_OUTPUT_SCRATCH_ARMED_MARKER" \
                    'output_tier=2' \
                    'output_offset_bytes=134217728' \
                    'output_bytes=33554432' \
                    'partner_attention=retained' \
                    'production_output_write=no production_output_read=no' \
                    'output=dedicated-partner-allocation' \
                    'accepted_output=full-home-recompute'
            else
                grep -Eq \
                    "prefill attention row shadow audit event=begin phase=$shadow_phase .*home=0 partner=2" \
                    "$log"
            fi
            ;;
        attention-q8-rows-serialized)
            grep -Fxq \
                'ds4: CUDA prefill attention row compute serialization enabled: logical pair 0->2 order=partner-sync,home-sync; row ownership, kernels, caches, and transport retained' \
                "$log"
            ;;
        *)
            return 0
            ;;
    esac
}

ctx_alloc=$((PP_TOKENS + TG_TOKENS + 1))
slot=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        tag="r${repeat}-s${slot}-${variant}"
        base="$OUTPUT_DIR/production/$tag"
        started="$base.started"
        result="$base.result"
        progress="$base-progress.csv"
        log="$base.log"
        csv="$base.csv"
        plan="$base-plan.csv"
        bindings="$base-bindings.csv"
        allocations="$base-allocations.csv"
        memory="$base-memory.csv"
        runtime_identity="$base-device-identity.csv"
        runtime_identity_report="$base-device-identity-validation.txt"
        telemetry="$OUTPUT_DIR/telemetry/$tag.csv"
        watch_marker="$OUTPUT_DIR/telemetry/$tag-watch-event.txt"
        pre_health="$OUTPUT_DIR/health/$tag-pre.log"
        post_health="$OUTPUT_DIR/health/$tag-post.log"

        if [[ -s $result ]]; then
            if ! validate_arm_cuda_identity "$log" "$runtime_identity" \
                    "$runtime_identity_report"; then
                record_arm_identity_failure "$result" "$variant" "$repeat" \
                    "$progress" "$log" "$runtime_identity_report"
                die "saved result for variant=$variant lacks exact runtime CUDA identity evidence"
            fi
            printf 'Reusing completed evidence for variant=%s repeat=%d...\n' \
                "$variant" "$repeat"
            continue
        fi
        if [[ $RESUME == 1 && -s $started ]]; then
            if ! validate_arm_cuda_identity "$log" "$runtime_identity" \
                    "$runtime_identity_report"; then
                record_arm_identity_failure "$result" "$variant" "$repeat" \
                    "$progress" "$log" "$runtime_identity_report"
                die "prior arm for variant=$variant lacks exact runtime CUDA identity evidence"
            fi
            if [[ $variant == attention-q8-host-bounce ]] &&
                    grep -Fq 'CUDA q8 partner async completion ' "$log"; then
                printf 'Retaining contaminated no-marker control as validation-failed: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
                write_result "$result" "$variant" validation-failed 129 \
                    "$progress" "$log"
                fields=$(last_progress_fields "$progress")
                IFS=$'\t' read -r lp le lc lt <<<"$fields"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                    recovered-validation-failed 129 "$lp" "$le" \
                    >>"$OUTPUT_DIR/run-journal.tsv"
                sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                continue
            fi
            if [[ $variant == attention-q8-async-completion ]] && {
                    grep -Eq 'CUDA q8 partner pre-gather (fence|armed|returned) ' \
                        "$log" ||
                    grep -Eq 'CUDA q8 partner pre-activation (fence|armed|returned) ' \
                        "$log"; }; then
                printf 'Retaining contaminated no-fence control as validation-failed: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
                write_result "$result" "$variant" validation-failed 129 \
                    "$progress" "$log"
                fields=$(last_progress_fields "$progress")
                IFS=$'\t' read -r lp le lc lt <<<"$fields"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                    recovered-validation-failed 129 "$lp" "$le" \
                    >>"$OUTPUT_DIR/run-journal.tsv"
                sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                continue
            fi
            if [[ -s $csv && $(wc -l <"$csv") == 2 && -s $bindings ]] &&
                    [[ $(wc -l <"$bindings") == 345 ]] &&
                    [[ ! -e $watch_marker ]] &&
                    gpu_health_snapshot_is_healthy "$post_health"; then
                if validate_completed_path \
                        "$variant" "$log" "$bindings" "$csv"; then
                    recovered_status=passed
                    recovered_exit=0
                    case "$variant" in
                        attention-q8-host-bounce|attention-q8-async-completion|attention-q8-pre-gather-fence|attention-q8-activation-fence|attention-q8-global-compute-fence|attention-q8-direct-gather-fence|attention-q8-rows-serialized|attention-q8-row-compute-off|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-q8-l12-phase-audit)
                            if ! validate_full_production_load "$csv"; then
                                recovered_status=inconclusive-underloaded
                                recovered_exit=128
                            fi
                            ;;
                    esac
                    printf 'Recovering completed prior arm: variant=%s repeat=%d status=%s...\n' \
                        "$variant" "$repeat" "$recovered_status"
                    write_result "$result" "$variant" "$recovered_status" \
                        "$recovered_exit" "$progress" "$log"
                    fields=$(last_progress_fields "$progress")
                    IFS=$'\t' read -r lp le lc lt <<<"$fields"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                        "recovered-$recovered_status" "$recovered_exit" "$lp" "$le" \
                        >>"$OUTPUT_DIR/run-journal.tsv"
                    sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                    continue
                fi
                printf 'Retaining completed prior arm as validation-failed: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
                write_result "$result" "$variant" validation-failed 127 \
                    "$progress" "$log"
                fields=$(last_progress_fields "$progress")
                IFS=$'\t' read -r lp le lc lt <<<"$fields"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                    recovered-validation-failed 127 "$lp" "$le" \
                    >>"$OUTPUT_DIR/run-journal.tsv"
                sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                continue
            fi
            retained_status=interrupted-prior-run
            retained_exit=125
            if { [[ -s $watch_marker ]] &&
                    grep -Fxq 'status=lost-device-detected' "$watch_marker"; } ||
                    gpu_health_snapshot_is_unhealthy "$post_health"; then
                retained_status=interrupted-prior-run-device-loss
                retained_exit=124
                if ! failed_arm_activation_proven "$variant" "$log"; then
                    retained_status=interrupted-prior-run-experiment-not-activated
                    retained_exit=123
                    printf 'Retaining interrupted prior arm as device loss before required experiment activation: variant=%s repeat=%d\n' \
                        "$variant" "$repeat"
                else
                    printf 'Retaining interrupted prior arm as corroborated device-loss failure: variant=%s repeat=%d\n' \
                        "$variant" "$repeat"
                fi
            else
                printf 'Retaining interrupted prior arm as incomplete evidence: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
            fi
            write_result "$result" "$variant" "$retained_status" "$retained_exit" \
                "$progress" "$log"
            fields=$(last_progress_fields "$progress")
            IFS=$'\t' read -r lp le lc lt <<<"$fields"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                "$retained_status" "$retained_exit" "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            continue
        fi

        phase="production-$variant"
        printf 'Pair isolation repeat=%d/%d variant=%s PP=%d TG=%d...\n' \
            "$repeat" "$REPEATS" "$variant" "$PP_TOKENS" "$TG_TOKENS"
        gpu_health_snapshot "$pre_health" ||
            die "GPU health failed before variant=$variant"
        validate_power_limits ||
            die "power limit changed before variant=$variant; require physical-index profile ${REQUIRED_POWER_LIMITS_W} W"
        assert_no_compute_processes ||
            die "foreign GPU compute process present before variant=$variant"
        printf 'timestamp_utc=%s\nvariant=%s\nrepeat=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" >"$started"
        sync "$started" 2>/dev/null || sync

        variant_env=()
        case "$variant" in
            attention-off)
                variant_env+=("DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-host-bounce)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-host-bounce)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-async-completion)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_ASYNC_COMPLETION_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-pre-gather-fence)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_ASYNC_COMPLETION_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PRE_GATHER_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-activation-fence)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_ASYNC_COMPLETION_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PRE_GATHER_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PRE_ACTIVATION_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-global-compute-fence)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_ASYNC_COMPLETION_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PRE_GATHER_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PRE_ACTIVATION_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_COMPUTE_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-direct-gather-fence)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_COMPUTE_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_DIRECT_GATHER_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-rows-serialized)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_SERIALIZE_COMPUTE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_COMPUTE_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_DIRECT_GATHER_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-row-compute-off)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_NO_TP_PREFILL_ATTN_ROW_COMPUTE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_COMPUTE_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_DIRECT_GATHER_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-row-query-shadow|attention-row-partner-shadow|attention-row-gather-shadow|attention-row-gather-dst-shadow|attention-row-gather-chunk16-shadow|attention-row-gather-chunk16-paced-shadow|attention-row-gather-scratch-paced-shadow|attention-row-gather-source-scratch-paced-shadow|attention-row-gather-preinitialized-source-paced-shadow|attention-row-gather-preinitialized-source-no-partner-paced-shadow|attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ROW_SHADOW_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ROW_SHADOW_PHASE=$(row_shadow_phase "$variant")")
                if [[ $variant == attention-row-gather-dst-shadow ]]; then
                    variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_GATHER_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                fi
                ;;
            attention-q8-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-targeted-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_BINDING_LABEL=$Q8_TARGETED_BINDING_LABEL")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_WEIGHT_OFFSET=$Q8_TARGETED_WEIGHT_OFFSET")
                ;;
            attention-q8-l14-l15-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_TARGETS=$Q8_WINDOW_TARGETS")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_WEIGHT_BYTES=$Q8_TARGET_WEIGHT_BYTES")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_IN_DIM=$Q8_TARGET_IN_DIM")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_OUT_DIM=$Q8_TARGET_OUT_DIM")
                ;;
            attention-q8-l12-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_TARGETS=$Q8_L12_TARGET")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_WEIGHT_BYTES=$Q8_TARGET_WEIGHT_BYTES")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_IN_DIM=$Q8_TARGET_IN_DIM")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_OUT_DIM=$Q8_TARGET_OUT_DIM")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_SKIP_OCCURRENCES=$Q8_L12_SKIP_OCCURRENCES")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_MAX_OCCURRENCES=$Q8_L12_MAX_OCCURRENCES")
                ;;
            attention-query-dst)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_QUERY_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-gather-dst)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_GATHER_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-both-dst)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_QUERY_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_GATHER_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_LAYER=$ATTN_PHASE_AUDIT_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_POS=$ATTN_PHASE_AUDIT_POS")
                ;;
            attention-end-fence)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_LAYER=$ATTN_END_FENCE_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_POS=$ATTN_END_FENCE_POS")
                ;;
            attention-row-boundary-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_LAYER=$ATTN_ROW_BOUNDARY_END_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_POS=$ATTN_ROW_BOUNDARY_POS")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ENTRY_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ENTRY_FENCE_LAYER=$ATTN_ROW_BOUNDARY_ENTRY_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ENTRY_FENCE_POS=$ATTN_ROW_BOUNDARY_POS")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_LAYER=$ATTN_ROW_BOUNDARY_ENTRY_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_POS=$ATTN_ROW_BOUNDARY_POS")
                ;;
            partner-bounce)
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            bounce-indexer-off)
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_NO_TP_DECODE_INDEXER_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            partner-serialized)
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_SERIALIZE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            indexer-off)
                variant_env+=("DS4_CUDA_NO_TP_DECODE_INDEXER_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            production) ;;
        esac

        start_telemetry "$telemetry"
        "${clean[@]}" "${variant_env[@]}" \
                "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
                DS4_CUDA_PREFILL_PIPELINE=1 \
                DS4_CUDA_PREFILL_PIPELINE_MB=512 \
                DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
                DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
                DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
                DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
                DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1 \
                DS4_CUDA_PREFILL_FAULT_BREADCRUMBS=1 \
                DS4_CUDA_Q8_F16_PARTNER_TRANSFER_AUDIT=1 \
                DS4_CUDA_TP_DECODE_INDEXER_ROWS_AUDIT=1 \
                DS4_BENCH_UNTIMED_WARMUP_TOKENS=512 \
                "DS4_BENCH_PROGRESS_JOURNAL=$progress" \
                "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$plan" \
                "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
                "DS4_CUDA_Q8_ALLOCATION_STATE_CSV=$allocations" \
                "DS4_CUDA_MEMORY_STATE_CSV=$memory" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$PP_TOKENS" --ctx-max "$PP_TOKENS" \
                    --ctx-alloc "$ctx_alloc" --step-incr "$PP_TOKENS" \
                    --prefill-chunk 2048 --gen-tokens "$TG_TOKENS" \
                    --csv "$csv" >"$log" 2>&1 &
        active_case_pid=$!
        case_pid=$active_case_pid
        capture_status=0
        case_identity=$(capture_process_identity "$active_case_pid") ||
            capture_status=$?
        if (( capture_status != 0 )); then
            if ! pid_is_live "$active_case_pid"; then
                if wait "$active_case_pid"; then
                    capture_child_status=0
                else
                    capture_child_status=$?
                fi
                active_case_pid=
                die "case exited before immutable identity capture completed (status=$capture_child_status)"
            fi
            die "could not capture immutable identity for case PID $case_pid after bounded launch handshake; child may still be live and was not signaled"
        fi
        active_case_identity=$case_identity
        start_telemetry_watch "$telemetry" "$watch_marker" \
            "$active_case_pid" "$case_identity"
        one_shot_child_stop_failed=0
        if [[ $ONE_SHOT == 1 ]]; then
            run_status=
            one_shot_deadline=$((SECONDS + ONE_SHOT_TIMEOUT_SECONDS))
            telemetry_dead_since=0
            child_exit_noticed=0
            telemetry_stop_requested=0
            while :; do
                if [[ -s $watch_marker && -s $watch_marker.ready ]] &&
                        grep -Eq '^status=.+$' "$watch_marker" &&
                        grep -Fxq 'state=critical-actions-complete' \
                            "$watch_marker.ready"; then
                    # A CUDA process can remain in uninterruptible kernel wait
                    # after a watcher event.  The ready handshake is published
                    # only after the watcher has completed its start-time-
                    # gated signaling and durably renamed the marker.
                    run_status=123
                    stop_active_case || one_shot_child_stop_failed=1
                    break
                fi
                if (( child_exit_noticed == 0 )) &&
                        ! pid_matches_identity "$active_case_pid" \
                            "$active_case_identity"; then
                    publish_child_exit_notice "$watch_marker" "$case_pid"
                    child_exit_noticed=1
                    # Publish exit before requesting the watcher's final scan.
                    # The watcher will not signal after consuming this notice,
                    # and independently revalidates start-time identity before
                    # every signal in case Bash has already reaped the job.
                    signal_bound_process "$telemetry_pid" \
                        "$telemetry_identity" TERM || true
                    telemetry_stop_requested=1
                    telemetry_dead_since=$SECONDS
                fi
                if ! pid_is_live "$telemetry_watch_pid"; then
                    # A concurrently detected watcher event has precedence.
                    # Re-enter the loop once after watcher exit so a fully
                    # published marker/ready pair is consumed before accepting
                    # the child's own exit status.
                    if [[ -s $watch_marker && -s $watch_marker.ready ]] &&
                            grep -Eq '^status=.+$' "$watch_marker" &&
                            grep -Fxq 'state=critical-actions-complete' \
                                "$watch_marker.ready"; then
                        continue
                    fi
                    if (( child_exit_noticed == 1 )); then
                        if wait "$active_case_pid"; then
                            run_status=0
                        else
                            run_status=$?
                        fi
                        active_case_pid=
                        active_case_identity=
                        # A requested final scan that published no event is not
                        # a telemetry-monitor failure. Preserve the child's
                        # zero or nonzero status for normal result handling.
                        break
                    else
                        stop_active_case || one_shot_child_stop_failed=1
                    fi
                    publish_watch_marker "$watch_marker" \
                        telemetry-monitor-failed "$case_pid" unknown '' \
                        watcher-exited-without-ready-handshake
                    run_status=123
                    break
                fi
                if ! pid_is_live "$telemetry_pid"; then
                    (( telemetry_dead_since > 0 )) || telemetry_dead_since=$SECONDS
                    # Once child exit has been noticed, its watcher owns
                    # the final telemetry scan and durable ready handshake.
                    # Do not race that handshake merely because we asked the
                    # telemetry producer to exit; wait for watcher ready,
                    # confirmed watcher exit, or the global one-shot bound.
                    if (( child_exit_noticed == 0 &&
                          SECONDS - telemetry_dead_since >= 5 )); then
                        stop_telemetry
                        if [[ -s $watch_marker && -s $watch_marker.ready ]] &&
                                grep -Eq '^status=.+$' "$watch_marker" &&
                                grep -Fxq 'state=critical-actions-complete' \
                                    "$watch_marker.ready"; then
                            continue
                        fi
                        stop_active_case || one_shot_child_stop_failed=1
                        publish_watch_marker "$watch_marker" \
                            telemetry-monitor-failed "$case_pid" unknown '' \
                            "telemetry-exited-watcher-handshake-timeout;requested=${telemetry_stop_requested}"
                        run_status=123
                        break
                    fi
                else
                    telemetry_dead_since=0
                fi
                if (( SECONDS >= one_shot_deadline )); then
                    # Ask telemetry to end, then stop/join the watcher before
                    # creating generic timeout evidence.  A device-loss marker
                    # completed by that final scan always wins.
                    signal_bound_process "$telemetry_pid" \
                        "$telemetry_identity" TERM || true
                    # Give the watcher a bounded opportunity to consume the
                    # telemetry tail and naturally publish specific evidence
                    # before helper cleanup can terminate it.
                    for _ in {1..50}; do
                        if [[ -s $watch_marker && -s $watch_marker.ready ]] &&
                                grep -Eq '^status=.+$' "$watch_marker" &&
                                grep -Fxq 'state=critical-actions-complete' \
                                    "$watch_marker.ready"; then
                            break
                        fi
                        pid_is_live "$telemetry_watch_pid" || break
                        sleep 0.1
                    done
                    stop_telemetry
                    if [[ -s $watch_marker && -s $watch_marker.ready ]] &&
                            grep -Eq '^status=.+$' "$watch_marker" &&
                            grep -Fxq 'state=critical-actions-complete' \
                                "$watch_marker.ready"; then
                        continue
                    fi
                    stop_active_case || one_shot_child_stop_failed=1
                    publish_watch_marker "$watch_marker" \
                        one-shot-monitor-timeout "$case_pid" unknown '' \
                        "timeout_seconds=${ONE_SHOT_TIMEOUT_SECONDS}"
                    run_status=124
                    break
                fi
                sleep 0.1
            done
        else
            if wait "$active_case_pid"; then run_status=0; else run_status=$?; fi
            active_case_pid=
            active_case_identity=
        fi
        stop_telemetry
        if (( run_status == 0 )) && [[ -e $watch_marker ]]; then
            run_status=123
        fi
        if (( run_status == 0 )) && ! validate_power_limits; then
            run_status=126
            if [[ ! -s $watch_marker ]]; then
                printf 'timestamp_utc=%s\nstatus=post-run-power-limit-drift\nrequired_power_limits_w=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    "$REQUIRED_POWER_LIMITS_W" >"$watch_marker"
                sync "$watch_marker" 2>/dev/null || sync
            fi
        fi
        post_health_status=0
        if [[ $ONE_SHOT == 1 && $one_shot_child_stop_failed == 1 ]]; then
            printf 'date_utc=%s\nstatus=skipped-child-still-present-after-bounded-kill\ncase_pid=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$case_pid" \
                >"$post_health"
            sync "$post_health" 2>/dev/null || sync
            post_health_status=124
        else
            gpu_health_snapshot "$post_health" || post_health_status=124
        fi
        if (( run_status == 0 && post_health_status != 0 )); then
            run_status=$post_health_status
        fi

        if ! validate_arm_cuda_identity "$log" "$runtime_identity" \
                "$runtime_identity_report"; then
            record_arm_identity_failure "$result" "$variant" "$repeat" \
                "$progress" "$log" "$runtime_identity_report"
            die "variant=$variant failed runtime CUDA identity validation"
        fi

        fields=$(last_progress_fields "$progress")
        IFS=$'\t' read -r lp le lc lt <<<"$fields"
        if [[ $variant == attention-q8-host-bounce ]] &&
                grep -Fq 'CUDA q8 partner async completion ' "$log"; then
            write_result "$result" "$variant" validation-failed 129 \
                "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                validation-failed 129 "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            tail -n 240 "$log" >&2 || true
            die "variant=$variant contained async-completion records in the no-marker control"
        fi
        if [[ $variant == attention-q8-async-completion ]] && {
                grep -Eq 'CUDA q8 partner pre-gather (fence|armed|returned) ' \
                    "$log" ||
                grep -Eq 'CUDA q8 partner pre-activation (fence|armed|returned) ' \
                    "$log"; }; then
            write_result "$result" "$variant" validation-failed 129 \
                "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                validation-failed 129 "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            tail -n 240 "$log" >&2 || true
            die "variant=$variant contained fence records in the no-fence control"
        fi
        if (( run_status != 0 )); then
            watch_status=
            if [[ -s $watch_marker ]]; then
                watch_status=$(awk -F= '$1=="status" {print $2; exit}' "$watch_marker")
            fi
            result_status=run-failed-unverified
            if ! failed_arm_activation_proven "$variant" "$log"; then
                result_status=experiment-not-activated
            elif [[ $watch_status == lost-device-detected ]] ||
                    gpu_health_snapshot_is_unhealthy "$post_health"; then
                result_status=failed-device-loss
            elif [[ -n $watch_status ]]; then
                result_status=environment-invalid
            fi
            write_result "$result" "$variant" "$result_status" "$run_status" \
                "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                "$result_status" "$run_status" "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            tail -n 240 "$log" >&2 || true
            die "variant=$variant ended with status=$result_status at phase=$lp event=$le"
        fi
        [[ -s $csv && $(wc -l <"$csv") == 2 ]] ||
            die "variant=$variant produced an invalid benchmark CSV"
        [[ -s $bindings ]] || die "variant=$variant omitted its binding inventory"
        if [[ $(wc -l <"$bindings") != 345 ]]; then
            die "variant=$variant omitted its 344-entry binding inventory"
        fi
        if ! validate_completed_path "$variant" "$log" "$bindings" "$csv"; then
            write_result "$result" "$variant" validation-failed 127 \
                "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                validation-failed 127 "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            die "variant=$variant failed production-path validation"
        fi
        case "$variant" in
            attention-q8-host-bounce|attention-q8-async-completion|attention-q8-pre-gather-fence|attention-q8-activation-fence|attention-q8-global-compute-fence|attention-q8-direct-gather-fence|attention-q8-rows-serialized|attention-q8-row-compute-off|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-q8-l12-phase-audit)
                if ! validate_full_production_load "$csv"; then
                    write_result "$result" "$variant" inconclusive-underloaded 128 \
                        "$progress" "$log"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                        inconclusive-underloaded 128 "$lp" "$le" \
                        >>"$OUTPUT_DIR/run-journal.tsv"
                    sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                    printf 'error: variant=%s completed below 500 prefill tok/s; evidence is inconclusive\n' \
                        "$variant" >&2
                    exit 128
                fi
                ;;
        esac
        write_result "$result" "$variant" passed 0 "$progress" "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
            passed 0 "$lp" "$le" >>"$OUTPUT_DIR/run-journal.tsv"
        sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
        sleep "$POST_CASE_SETTLE_SECONDS"
    done
done

write_summary
phase=complete
[[ -s $OUTPUT_DIR/summary.md ]] && cat "$OUTPUT_DIR/summary.md"
printf 'SM75 small-BAR1 pair isolation complete: %s\n' "$OUTPUT_DIR"
