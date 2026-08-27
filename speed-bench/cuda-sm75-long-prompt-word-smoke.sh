#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run one production-path long-prompt retrieval and early-decode quality smoke.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  CORPUS=...                 default: speed-bench/promessi_sposi.txt
  EXPECTED_WORD=LANTERN
  PAD_LINES=2200
  MIN_PROMPT_TOKENS=24000
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_ALLOC=262273
  GEN_TOKENS=8
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  WORD_SMOKE_DIR=/absolute/output/directory

The verification word appears before the long distractor. The final user
instruction requests exactly that one word. The run uses the ordinary
production configuration with no row-split enable override and fails unless
the qualified default dispatches both mixed and indexed split attention.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL=${MODEL:-}
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute GGUF"
CORPUS=${CORPUS:-$repo_dir/speed-bench/promessi_sposi.txt}
EXPECTED_WORD=${EXPECTED_WORD:-LANTERN}
PAD_LINES=${PAD_LINES:-2200}
MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS:-24000}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_ALLOC=${CTX_ALLOC:-262273}
GEN_TOKENS=${GEN_TOKENS:-8}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${WORD_SMOKE_DIR:-$repo_dir/sm75-long-prompt-word-smoke-$stamp}

[[ -f $CORPUS ]] || die "corpus not found: $CORPUS"
[[ $EXPECTED_WORD =~ ^[A-Za-z]+$ ]] ||
    die "EXPECTED_WORD must contain ASCII letters only"
for item in "PAD_LINES:$PAD_LINES" "MIN_PROMPT_TOKENS:$MIN_PROMPT_TOKENS" \
            "STAGE_SPLIT:$STAGE_SPLIT" "CTX_ALLOC:$CTX_ALLOC" \
            "GEN_TOKENS:$GEN_TOKENS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( PAD_LINES >= 1000 && MIN_PROMPT_TOKENS >= 16000 &&
   STAGE_SPLIT > 0 && STAGE_SPLIT < 43 && CTX_ALLOC > MIN_PROMPT_TOKENS &&
   GEN_TOKENS > 0 && SKIP_BUILD <= 1 && CREATE_ARCHIVE <= 1 )) ||
    die "invalid smoke-test bounds"
for tool in awk basename cat date dirname env git grep head make mkdir nproc \
            nvidia-smi sed sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/provenance"
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
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s\n' "$archive"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

phase=build
targets=(ds4 tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/placement-tests.log" >&2
    die "placement regression tests failed"
}

phase=prompt
prompt="$OUTPUT_DIR/prompt.txt"
{
    printf 'Remember this verification word: %s. You will be asked for it after a long distractor.\n\n' \
        "$EXPECTED_WORD"
    printf '%s\n' '--- BEGIN LONG DISTRACTOR ---'
    head -n "$PAD_LINES" "$CORPUS"
    printf '\n%s\n\n' '--- END LONG DISTRACTOR ---'
    printf 'What was the verification word stated before the distractor? Reply with exactly that one word and nothing else.\n'
} >"$prompt"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\n' \
        "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'corpus=%s\npad_lines=%s\nexpected_word=%s\n' \
        "$CORPUS" "$PAD_LINES" "$EXPECTED_WORD"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nctx_alloc=%s\ngen_tokens=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" \
        "$CTX_ALLOC" "$GEN_TOKENS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

phase=production-generation
"${clean[@]}" \
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
    DS4_CUDA_PREFILL_PIPELINE=1 \
    DS4_CUDA_PREFILL_PIPELINE_MB=512 \
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
    DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
    DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
    DS4_CUDA_SYNC_XDEV=1 \
    ./ds4 --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --ctx "$CTX_ALLOC" --prefill-chunk 2048 \
        --nothink --temp 0 --tokens "$GEN_TOKENS" --prompt-file "$prompt" \
        >"$OUTPUT_DIR/stdout.txt" 2>"$OUTPUT_DIR/stderr.log" || {
            tail -n 200 "$OUTPUT_DIR/stderr.log" >&2
            die "production generation failed"
        }

grep -Fq 't256-placement=balanced' "$OUTPUT_DIR/stderr.log" ||
    die "production generation missed balanced T256 placement"
[[ $(grep -Fc 'qualified=yes' "$OUTPUT_DIR/stderr.log") == 2 ]] ||
    die "production generation did not qualify both SM75 NVLink pairs"
[[ $(grep -Fc 'query-row split enabled:' "$OUTPUT_DIR/stderr.log") == 2 ]] ||
    die "qualified default did not enable both row-split stages"
grep -Fq 'dispatch=split kind=mixed' "$OUTPUT_DIR/stderr.log" ||
    die "production generation omitted mixed row-split dispatch"
grep -Fq 'dispatch=split kind=indexed' "$OUTPUT_DIR/stderr.log" ||
    die "production generation omitted indexed row-split dispatch"
grep -Eq 'dispatch=home reason=shape kind=(mixed|indexed).*tokens=[0-9]+' \
    "$OUTPUT_DIR/stderr.log" ||
    die "long prompt did not exercise shape-dispatched home-tail attention"
! grep -Fq 'required but unavailable' "$OUTPUT_DIR/stderr.log" ||
    die "an eligible row-split call was missing required runtime resources"

prompt_tokens=$(sed -n \
    's/.*processing \([0-9][0-9]*\) input tokens.*/\1/p' \
    "$OUTPUT_DIR/stderr.log" | tail -n 1)
[[ $prompt_tokens =~ ^[0-9]+$ && $prompt_tokens -ge $MIN_PROMPT_TOKENS ]] ||
    die "long prompt had only ${prompt_tokens:-unknown} tokens"

sed '/^ds4: GPU config: .*$/d' "$OUTPUT_DIR/stdout.txt" \
    >"$OUTPUT_DIR/answer.txt"
normalized=$(tr -d '\r' <"$OUTPUT_DIR/answer.txt" |
    tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z')
expected=$(printf '%s' "$EXPECTED_WORD" | tr '[:lower:]' '[:upper:]')
[[ $normalized == "$expected" ]] || {
    printf 'model answer:\n' >&2
    cat "$OUTPUT_DIR/answer.txt" >&2
    die "expected exactly one word: $EXPECTED_WORD"
}

printf 'result=PASS\nprompt_tokens=%s\nexpected=%s\nnormalized_answer=%s\n' \
    "$prompt_tokens" "$EXPECTED_WORD" "$normalized" \
    >"$OUTPUT_DIR/result.txt"
cat "$OUTPUT_DIR/result.txt"
phase=complete
printf 'SM75 long-prompt one-word quality smoke complete: %s\n' "$OUTPUT_DIR"
