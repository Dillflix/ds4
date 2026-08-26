## Benchmarking

### Tagged SM75-native full-Q4 production A/B

`cuda-sm75-native-q4-production.sh` is the acceptance runner for the packed
INT4 production layout. It losslessly repacks a stock full-Q4 model when the
native output does not already exist, requires the production API's standard
and native results to be exact for tile16 plus real 8/4 tails and decode, and
then requires bit-identical full-vocabulary logits from the two complete GGUFs.

The timed comparison is fixed to the measured 22/21 split and device order
`0,3,1,2`; process order alternates between standard and native. It also
captures an actual four-GPU Nsight Systems trace and bounded Nsight Compute
reports for native gate/up tile8 and down tile16/8/4. No model hash is taken.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export NATIVE_MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

REPEATS=2 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-native-q4-production.sh
```

Use `REPEATS=6` for replicated timing after the two-repeat pilot succeeds.

### Cost-planner default and rejected K-stage A/B

The cost-aware residual planner is now the tagged native-Q4 production default:
`1..4 -> tile4`, `5..8 -> tile8`, and `9..15 -> tile16`. Set
`DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=1` only for a diagnostic comparison with
the old minimum-active-tile planner.

The diagnostic `stream7` gate and `compact7` down candidates compile below
32 KiB declared shared memory, but both are rejected. Moving activation
payload and metadata reads into the repeated output-row loop raised LG
throttle and made both kernels more than three times slower in representative
Nsight Compute captures. They remain selectable only to reproduce the failure
audit; neither is a production candidate.

`cuda-sm75-native-q4-next-ab.sh` tests `baseline`, `down`, `gate`, and `both`
in alternating order at placement `0,3,1,2` and split 22/21. Before loading the
model it rejects either new kernel if compiled shared memory is at least 32 KiB,
registers exceed 128, or any stack/local storage is present. It then requires
exact production-API results, a real cost-planner audit, and bit-identical raw
logits for every selected variant. `legacy` remains an optional fifth variant.
`RUN_NSYS=1` captures matching `baseline` and `both` traces.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

REPEATS=2 \
RUN_NSYS=1 \
./speed-bench/cuda-sm75-native-q4-next-ab.sh
```

The script never modifies `NATIVE_MODEL`. Use `RUN_E2E_EXACT=0` only when
rerunning timing after the same binary/model combination has already passed
the raw-logit comparison.

### Gate/Up fixed-K16 and row-fused full64 A/B

`cuda-sm75-native-q4-gate-fused-ab.sh` isolates two Gate/Up changes while
leaving Down and the production cost planner untouched:

- `fixed` is the current fused tile8 path specialized for the model's fixed 16
  input blocks. It is the optimized control: same routes, staging, reduction
  order, and two low8/high8 launches, but compile-time K bounds and strides.
- `fused` retains all 16 activation payload rows in Turing's 65,536-byte
  opt-in shared-memory allocation. For each output-row group it computes Gate,
  lets Gate's scalar accumulator window die, holds only four reduced Gate
  values while computing Up, and immediately emits the fused SiLU product.
  It therefore reuses each Q4 weight fragment across low8/high8 without the
  rejected full64 design's global Gate scratch or phase barrier. Its tail8 and
  tail4 work uses the fixed-K16 control.

Before loading the model, the runner rejects local/stack traffic or more than
128 registers, requires exact Up auxiliary output and proves that enabling the
auxiliary stores does not perturb production API output, records
separate dynamic-K/fixed-K SASS summaries, times all three variants against the
captured early/late production histograms, and records bounded Nsight Compute
reports. That bounded Gate-only screen is the default (`RUN_FULL_MODEL=0`).
Model hashing and conversion are deliberately absent.

If only Nsight Compute failed after the bounded screen, set
`GATE_FUSED_DIR` to that existing result directory and rerun with
`RESUME_NCU_ONLY=1 SKIP_BUILD=1 RUN_FULL_MODEL=0 RUN_NSYS=0 RUN_NCU=1`.
The resume profiles all descendants but filters by the exact harness process
name, avoiding wrapper-process captures without repeating exactness or timing.

Only after the bounded evidence is viable, rerun with `RUN_FULL_MODEL=1`. That
mode additionally compares raw full-model logits, alternates
baseline/fixed/fused prefill runs, verifies placement and Q8-cache equivalence,
reports paired throughput ratios, and captures a short Nsight Systems trace for
each variant.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

REPEATS=2 \
CTX_START=2048 \
CTX_MAX=8192 \
RUN_FULL_MODEL=0 \
RUN_E2E_EXACT=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-native-q4-gate-fused-ab.sh
```

For the promotion pass, reuse the same exports with
`RUN_FULL_MODEL=1 RUN_NCU=0`; the already-collected kernel reports do not need
to be repeated.

### Compact N-split native-Q4 topology audit

`cuda-sm75-q4-nsplit.sh` is a harness-only go/no-go experiment for the compact
N-split CTA topology. It does not alter production dispatch or open a model.
For both gate/up and down it tests 4- and 8-warp CTAs where one CTA owns one
real cost-planner tile16 plus one output-row macro-tile. The CTA stages only
one 256-K activation slab at a time (4,672 bytes), keeps scalar K slots in
registers, and consumes each packed Q4 fragment for both low8 and high8 route
halves before advancing.

The early/late inputs use the exact captured production expert histograms;
8/4 residual work is excluded so the decision measures only the topology it
could replace. The runner requires bit-exact outputs, no PTXAS/SASS local
memory, packed m8n8k32 U4/U4 and S4/U4 instructions, and less than 8 KiB shared
memory. It then records balanced kernel timings and focused Nsight Compute
reports for baseline, nsplit4, and nsplit8. A duration candidate passes only
if it beats the matching shipping native-A/W kernel by at least 10% in both
early and late shapes; a timing miss is recorded rather than suppressing the
profiles needed to explain it.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
BENCH_ROUNDS=3 \
BENCH_LAUNCHES=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-q4-nsplit.sh
```

No `MODEL` variable is required. Return the printed
`sm75-q4-nsplit-<timestamp>.tar.gz` archive.

### Persistent 512-row tile16 Gate audit

`cuda-sm75-q4-persistent.sh` tests the successor to the rejected compact
N-split Gate topology without changing production dispatch. Both candidates
retain the shipping 512-row CTA, reuse every packed Q4 fragment across a real
16-route cost-planner tile, and keep only one matrix's scalar accumulator
window live. `persistent-seq16` computes Gate then Up sequentially;
`persistent-ws16` assigns eight warps to each matrix. Because the complete
16-route native-Q8 activation tile is 73.0 KiB (74,752 bytes) and cannot fit in Turing's
64 KiB shared-memory budget, both consume immutable activations directly
through the unified cache path.

The runner checks real early/late histograms for bit-exact output, rejects
local-memory traffic or more than 128 registers/thread, records balanced
timings against the shipping warp16 scalar kernel, and captures focused
early-layer Nsight Compute reports.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
BENCH_ROUNDS=3 \
BENCH_LAUNCHES=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-q4-persistent.sh
```

No model is opened. Return the printed
`sm75-q4-persistent-<timestamp>.tar.gz` archive.

### Exact production Q4 expert histogram

`cuda-q4-real-histogram.sh` captures the actual per-expert routed-pair counts
from one production full-Q4 prefill. The counts are appended by the existing
device metadata kernel to a preallocated device audit buffer. There is no new
kernel launch, timing event, synchronization, or host read inside the measured
prefill; each GPU buffer is copied once after the existing final
synchronization.

Gate/up and down use the same router assignment. One capture therefore derives
both the shipping gate/up tile8 plan and the down tile16 plan, as well as
minimum-active-tile gate 8/4 and down 16/8/4 tail plans. It also reports the
production grid-capacity entries that return early because the actual tile
count remains on-device. The runner refuses a model
unless all 43 routed layers report Q4 gate, Q4 up, and Q4 down. Model hashing is
omitted.

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,2,1,3 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
./speed-bench/cuda-q4-real-histogram.sh
```

The returnable archive contains the raw wide audit, an exact record-level tile
plan, a long per-expert table, a count-frequency table, the benchmark result,
and runtime provenance.

### Post-scalar full-Q4 kernel trace

`cuda-q4-post-scalar-trace.sh` re-establishes the current full-Q4 kernel-time
distribution after the production SM75 scalar-slot changes. It first runs one
audited process and requires all 43 routed layers to be Q4/Q4/Q4 plus exactly
172 scalar gate/down device-layer dispatch markers. A second clean process is
then traced only across its timed 2K prefill. The runner forces the 22/21 split,
prefill pipeline, Q8-to-F16 cache, and both Q4 scalar specializations instead
of relying on inherited settings or defaults.

The output includes the raw Nsight Systems report and CSV exports, grouped
kernel shares, and the exact two production Q4 rows with their specialization
and launch cardinality. It requires 344 gate/up and 344 down calls and rejects
a trace whose demangled kernel identity is not the scalar specialization. It
does not invoke Nsight Compute and does not hash the 153 GiB model.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,2,1,3 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
./speed-bench/cuda-q4-post-scalar-trace.sh
```

Return `q4-post-scalar-trace-<timestamp>.tar.gz`. This pass updates the Amdahl
weights used to rank persistent exact-grid work queues and native A/W kernels;
it does not itself change production dispatch.

### Full-Q4 placement and cuBLAS-cache A/B

`cuda-q4-layout-cache-ab.sh` tests the cache/pipeline imbalance exposed by the
post-scalar trace. It compares three fixed configurations:

- `baseline-22x21`: split 22/21 with devices `0,2,1,3`;
- `split-21x22`: split 21/22 with devices `0,2,1,3`;
- `swap-22x21`: split 22/21 with devices `0,3,1,2`.

Each process first runs an untimed 2K warm-up so every admitted Q8 weight has
already been expanded to F16 for the measured cuBLAS path. It snapshots the
per-device cache before and after the timed sweep and rejects any run whose
cache changes. Three rotated repeats place every variant in every process
position once. The default 2K/4K/8K sweep is a bounded placement decision;
increase `CTX_MAX` only after selecting the winner. No model hash or Nsight
capture is performed.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_VRAM=auto \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
./speed-bench/cuda-q4-layout-cache-ab.sh
```

Return `q4-layout-cache-ab-<timestamp>.tar.gz`. Its summaries include paired
throughput ratios and the exact cached F16 slice count and bytes on each stage
home GPU.

### Q8 F16 benefit-plan A/B

`cuda-q8-cache-benefit-ab.sh` holds the selected full-Q4 placement fixed at
22/21 and `0,3,1,2`, then compares the default benefit-per-expanded-byte plan
with the former first-use admission policy. The startup plan is deterministic
and gives the measured SM75 T32 `attn_q_b` and T256 `attn_output_b` paths first
claim on each device's live cache headroom. It registers exact per-device keys
at engine creation and defers allocation/dequantization until the untimed
warm-up, after graph/context buffers exist; measured frontiers therefore
exclude one-time materialization and cannot admit extra tensors by traversal
order. `DS4_CUDA_Q8_F16_FIRST_USE=1` is retained only as the A/B escape hatch.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=4 \
./speed-bench/cuda-q8-cache-benefit-ab.sh
```

Return `q8-cache-benefit-ab-<timestamp>.tar.gz`. The runner requires nonzero
T32/T256 planned coverage, freezes the cache before timing, alternates process
order, and reports paired planner/first-use throughput ratios per frontier.

### Q8 F16 NVLink-partner execution A/B

`cuda-q8-partner-offload-ab.sh` holds full-Q4 placement at the measured 22/21
split and logical device order `0,3,1,2`. It screens partner execution disabled,
T32-only, T256-only, shared-down-only, and the initial T32+T256 (`legacy`)
policy. Within each home-device bucket, at equal benefit per byte,
fixed/home-only candidates rank before partner-eligible candidates. The class
selector therefore controls both which home-cache misses may overflow and which
tied expensive class is preferentially kept local; this is an end-to-end
placement-policy A/B, not an identical-home-cache microbenchmark.

The runner requires all four logical home/partner routes to be validated
`DIRECT` and independently rejects any physical pair that `nvidia-smi topo -m`
does not report as `NV#`; PCIe peer access is not accepted as NVLink evidence.
Its `pair-topology.tsv` records the production homes-first mapping. Every
candidate run must execute only its requested class on a configured partner.
One-pair execution is valid when the other stage fits its ranked F16 set locally;
`class-evidence.csv` makes the partner devices observed in the bounded audit
sample explicit. It freezes cache state before timed frontiers, rotates process
order, and reports the absolute median candidate/local tokens per second
alongside every paired throughput ratio. Runtime summary fields are explicitly
labeled `process_total_*`; the bounded device-audit fields are labeled
`audit_*`, so a
class-pure sample may be smaller than the uncapped process total without
invalidating the run. Full-logit output records numerical drift and top-token
agreement for every class, while within-variant repeat determinism remains an
acceptance condition. The GPU regression deliberately fills a home device's
per-device F16 cap and proves bit-exact partner execution for the production
T32, T256, and shared-down shapes. By default it also captures one short,
measured-prefill Nsight Systems report for each isolated class so transfer,
cuBLAS, synchronization, and partner contention can be evaluated.
The runner sets `DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS` to the positive
`PREFILL_CHUNK`, so
custom chunk sizes reserve matching partner scratch before timing. Outside the
runner the engine defaults this admission bound to 2048 tokens and safely
declines larger microbatches instead of growing scratch during execution.

```bash
cd ~/ds4-iq2-q4
git remote set-branches --add origin agent/partner-q8-offload
git fetch origin
git switch --track origin/agent/partner-q8-offload
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
VARIANTS=local,t32,t256 \
RUN_NSYS=0 \
bash ./speed-bench/cuda-q8-partner-offload-ab.sh
```

The command above is throughput-only; set `RUN_NSYS=1` to capture timelines.
Set `RUN_GPU_TEST=0` only after all three exactness cases have passed for the
current binary.
`RUN_GPU_TEST=1` still executes the existing binary when `SKIP_BUILD=1`; the
latter skips compilation, not validation. `VARIANTS` must include `local`,
`t32`, and `t256` so every A/B archive proves both expensive projection
classes; it may additionally include the other named variants.
When `RUN_NSYS=1`, `NSYS_VARIANTS` accepts a comma-separated subset of
`VARIANTS`; it is ignored for a throughput-only `RUN_NSYS=0` run. Return
`q8-partner-offload-ab-<timestamp>.tar.gz`; its
`class-evidence.csv` records execution counts by class and its `nsys/`
directory contains the bounded timelines and exported summaries.

Repeated full-Q4 screens selected T256-only as the best performance candidate;
the latest measured 13.0--15.3% over local, versus 12.2--14.3% for T32-only. The
initial mixed legacy policy measured 12.3--14.4%. T32, T256, and legacy produced
byte-identical logits to one another, but the T256 trace moved only 3.12 GiB of
partner activation and result traffic versus T32's 12.70 GiB. Shared-down
gained only 2.9--3.4%,
changed the 2048-token top result in every repeat, and is rejected as a default.

The first production-default decision used
`cuda-q8-partner-production-validation.sh` rather than the exploratory variant
screen. The control disabled partner execution. The candidate deliberately
sets no `DS4_CUDA_Q8_F16_PARTNER_CLASSES` override, so it validates the measured
automatic candidate: T256 was admitted only for SM75 home/partner devices whose
startup peer-bandwidth measurements reach at least 18 GiB/s in both directions.
An explicit `t256` selector remains useful for expert diagnostics on other
targets, but such a run does not prove that automatic admission selected the
production default.

That production runner builds and runs the placement and multi-GPU exactness
tests, scores the same full-Q4 model with partner execution disabled and enabled
across the official 100-case Flash fixture using
`score_official --production-path`, and performs at least three paired prefill
repeats at the fixed 16K and 32K frontiers. It rejects missing production
path markers, non-T256 partner bindings, impure audit evidence, changed top-1
tokens, repeat nondeterminism, or quality/performance gates that do not pass.
The mandatory GPU test separately requires bit-exact local-versus-partner
projection output for all three supported classes. Full local/default logits
are not expected to be byte-identical: the control intentionally leaves an
overflowed T256 weight on native Q8 while the candidate admits its expanded
F16/cuBLAS path. That deliberate arithmetic change is covered by top-1,
teacher-forced NLL, first-token, and greedy-prefix gates instead.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REPEATS=3 \
./speed-bench/cuda-q8-partner-production-validation.sh
```

`MODEL` must be an existing absolute path; `PROMPT` defaults to the tracked
fixed prompt but is exported above to make the run self-contained. The runner
also accepts an absolute `QUALITY_MANIFEST` (default:
`gguf-tools/quality-testing/data/flash/manifest.tsv`), `QUALITY_CTX`,
`PREFILL_CHUNK`, `SKIP_BUILD`, `CREATE_ARCHIVE`, and `T256_VALIDATION_DIR`.
Production acceptance fixes `GPU_DEVICES=0,3,1,2`, `GPU_VRAM=auto`,
`STAGE_SPLIT=22`, `CTX_START=16384`, `CTX_MAX=32768`, `STEP_MUL=2`, and
`QUALITY_CTX=CTX_MAX+1`; the 100 cases remain short, but the scorer allocates
the same 32K session footprint so cache admission sees the tested production
memory pressure. This bounded pass does not validate 64K operation. The
three-class GPU exactness test is mandatory. The runner rejects
attempts to weaken the target or use fewer than three repeats. Return the generated
`t256-production-validation-<timestamp>.tar.gz`. Before accepting a result,
check that `run-status.txt` says `state=finished`, `summary.md` says
`Overall: **PASS**`, and `acceptance.json` has `"accepted": true`.

### T256 arithmetic-source isolation

`cuda-q8-partner-arithmetic-isolation.sh` explains a quality difference; it
does not select a production layer allowlist. It freezes the home cache and
the additive T256 tensor/layer set, then runs native Q8 plus five controlled
partner arithmetic arms. Adjacent comparisons isolate Tensor-GEMM arithmetic,
activation FP16 rounding, weight FP16 rounding, activation block-Q8
quantization, and finally the native INT32-dot/reduction structure. The scorer
records first-token IDs and signed target/greedy logit margins.

The default bounded pass contains the five cases whose first-token result
changed in the initial T256 isolation. Set `CASE_IDS` to a larger comma-
separated `case_NNN` set only after this causal pass succeeds.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
T256_LAYERS=15-21 \
CASE_IDS=case_017,case_025,case_030,case_048,case_056 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-partner-arithmetic-isolation.sh
```

Return `q8-partner-arithmetic-<timestamp>.tar.gz`. A valid archive reports
`Experiment integrity: PASS` and proves identical home bindings and identical
additive T256 layers across every arithmetic arm.

If a runner/check failure stops a pass after one or more complete arms, retain
the unpacked output directory and resume it without recomputing those arms:

```bash
Q8_ARITHMETIC_DIR="$PWD/q8-partner-arithmetic-<timestamp>" \
RESUME=1 \
SKIP_BUILD=1 \
./speed-bench/cuda-q8-partner-arithmetic-isolation.sh
```

Resume validates the complete experiment manifest, the prior runtime/scorer
commit, every reusable arm's row count and runtime dispatch, and refuses reuse
if the runtime or scorer changed.

### Tagged SM75 native-Q4 and automatic-T256 A/B

`cuda-sm75-native-q4-t256-ab.sh` is an evidence runner only; it does not
change the engine, either GGUF, the native-Q4 dispatch, or the automatic T256
policy. It requires separate existing absolute paths in `MODEL` for the
standard full-Q4 GGUF and `NATIVE_MODEL` for its tagged SM75-native repack.
The fixed 2x2 compares standard/native Q4 with partner execution disabled or
with the implicit automatic T256 policy. It uses the production `0,3,1,2`
device order and 22/21 split, sweeps 2K through 32K, and runs four
counterbalanced repeats so every arm occupies every run slot once.

The runner records file sizes and exact-output evidence but deliberately does
not hash either approximately 153-GiB model. It reuses the existing models and
does not create or modify `NATIVE_MODEL`.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
STEP_MUL=2 \
PREFILL_CHUNK=2048 \
REPEATS=4 \
./speed-bench/cuda-sm75-native-q4-t256-ab.sh
```

Return `sm75-native-q4-t256-ab-<timestamp>.tar.gz`. The archive contains the
four-arm throughput decomposition, exact-logit comparisons, T256-only
admission/binding evidence, frozen-cache checks, and per-run GPU telemetry.
A `PASS` from this runner accepts only that structural and performance
interaction. It does not replace or override the separate official T256
quality-isolation gate described below.

The first production-default result failed its predeclared per-case,
first-token, and greedy-prefix quality gates even though aggregate NLL improved.
It also established that the normal partner-priority tie-break changed the
home cache: six additional T32 and seven additional T256 projections switched
from native Q8 to F16/cuBLAS. Use the frozen-home quality-isolation runner
before attributing that result to either class. For candidate runs it removes
partner eligibility from the primary sort, proves that all home bindings remain
identical to the partner-disabled control, and then admits only previously
uncached T256 or T32 tensors on the partner. Quality failure is a measured
outcome rather than a runner failure; structural contamination still aborts.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
QUALITY_CTX=32769 \
REUSE_LOCAL_DIR="$PWD/t256-production-validation-20260826T035132Z" \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-partner-quality-isolation.sh
```

Return `q8-partner-quality-isolation-<timestamp>.tar.gz`. Its `summary.md`
reports the T256-only and T32-only quality gates and the exact additive partner
layers. This pass is quality-only; benchmark only a class/subset that survives
it. `REUSE_LOCAL_DIR` is optional; when set, the runner verifies the prior
model path and size, quality fixture, 32K allocation pressure, GPU order, and
22/21 split before reusing its local-control evidence. Omit it if that source
directory is no longer present.

The completed frozen-home class pass rejected T32. It worsened NLL by 0.998%,
lost 0.47 mean greedy-prefix tokens, and lost seven aggregate token-level top-1
matches. Full T256 was directionally better on the broader evidence: NLL
improved 1.339%, the paired-bootstrap upper bound was negative, 59/100 cases
improved, mean greedy prefix gained 0.17 tokens, and aggregate token-level top-1
matches rose by five. It nevertheless failed the predeclared first-token gate:
four cases lost their first token and one gained it. Do not weaken that gate or
enable the full class implicitly. Isolate the additive layers 15--21 instead.

The runner accepts a candidate subset through `VARIANTS=t256` and
`T256_LAYERS`; the latter uses comma-separated layers or inclusive ranges. It
records the request in the manifest and rejects an archive unless the observed
additive binding set exactly matches it. The first bounded split is 15--18
versus 19--21, each reusing the already validated local control:

```bash
COMMON_LOCAL="$PWD/t256-production-validation-20260826T035132Z"

GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 QUALITY_CTX=32769 \
VARIANTS=t256 T256_LAYERS=15-18 REUSE_LOCAL_DIR="$COMMON_LOCAL" \
SKIP_BUILD=0 ./speed-bench/cuda-q8-partner-quality-isolation.sh

GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 QUALITY_CTX=32769 \
VARIANTS=t256 T256_LAYERS=19-21 REUSE_LOCAL_DIR="$COMMON_LOCAL" \
SKIP_BUILD=1 ./speed-bench/cuda-q8-partner-quality-isolation.sh
```

If one half passes, retain it and subdivide only a failing half. A final
combined subset still requires its own 100-case run because floating-point
path effects are not guaranteed to add linearly.

The completed T32 fusion A/B can eliminate redundant local and T32 runs:

```bash
REUSE_T32_DIR="$PWD/q8-t32-fused-ab-20260805T052315Z" \
SKIP_BUILD=1 \
RUN_GPU_TEST=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
VARIANTS=local,t32,t256 \
RUN_NSYS=0 \
bash ./speed-bench/cuda-q8-partner-offload-ab.sh
```

### T32 FP16-output fusion and partner-transfer A/B

`cuda-q8-t32-fused-ab.sh` isolates the formerly stubbed CUDA T32 capability
from the broader cache-policy experiment. It measures four process-isolated
variants at the fixed full-Q4 22/21 placement: established FP32-output T32 on
local weights, FP16-output plus fused RMS/RoPE on local weights, established
FP32-output T32 with partner overflow, and FP16-output/fused T32 with partner
overflow. The last variant returns an FP16 intermediate over NVLink, halving
the T32 result transfer before postprocessing on the home GPU.

On the fixed full-Q4 22/21 A/B, T32 partner execution with the established
FP32 result improved prefill by 12.2--13.5% over local execution and preserved
the top token at all nine measured frontiers. FP16-output fusion added only
0.5--1.0% over that partner path, while changing the top token at the 4096
frontier in every repeat. It therefore remains an opt-in diagnostic and is not
a production default.

The GPU regression first proves that local and partner FP16 intermediates are
bit-exact, their final FP32 outputs are bit-exact, the backend-owned scratch
path used by CUDA graphs is exact, and the new half-input RMS/RoPE kernel is
bit-exact with the established combined FP32 kernel after the same FP16
rounding. End-to-end cross-policy logits are reported rather than required to
match: partner VRAM changes which Q8 tensors receive cached FP16 weights, and
the fused helper deliberately changes its projection-output boundary from
FP32 to FP16. The runner instead requires exact repeat determinism within each
variant. All four short Nsight Systems traces are enabled by default.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
bash ./speed-bench/cuda-q8-t32-fused-ab.sh
```

Return `q8-t32-fused-ab-<timestamp>.tar.gz`. Set `RUN_NSYS=0` only for a
throughput-only rerun after the execution-path validations have passed.
After a complete four-variant run, a code-only fused-path correction can reuse
the unaffected old variants and capture only the new variants:

```bash
REUSE_OLD_DIR="$PWD/q8-t32-fused-ab-<old-timestamp>" \
NSYS_VARIANTS=new_local,new_partner \
SKIP_BUILD=0 \
RUN_GPU_TEST=1 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
bash ./speed-bench/cuda-q8-t32-fused-ab.sh
```

### Production scalar-slot evidence

`cuda-sm75-production-scalar.sh` is the bounded, model-free acceptance driver
for scalar accumulator versions of the production SM75 Q4 gate/up, Q4 down,
and IQ2 gate/up kernels. It reuses `cuda_long_context_smoke` for nonzero
production correctness and `cuda_sm75_profile_harness` for the recorded layer-3
and layer-36 routing shapes; it never opens a GGUF.

The Q4 gate/up candidate deliberately consumes the existing standard GGUF
layout.  That isolates the scalar-accumulator effect from native-weight
packing, can be deployed without requantizing or transforming a model, and
provides the production baseline needed to decide whether an offline native
layout adds enough further benefit to justify its compatibility cost.

```bash
bash speed-bench/cuda-sm75-production-scalar.sh
```

The default run rebuilds both harnesses with verbose PTXAS diagnostics,
requires scalar kernels to have zero stack and spill bytes and no SASS
`LDL`/`STL`, preserves nonzero IMMA and the IQ2 fused slot updates, runs
bitwise base-versus-scalar CUDA smoke checks, runs memcheck
when available, and records two position-balanced samples for each early/late
Q4 gate, Q4 down,
IQ2 tile16, and IQ2 tile8 target. Each sample times ten production calls inside
the harness after an untimed correctness/audit warmup, excluding process setup.
IQ2 tile8 remains selected in both halves of its A/B so only the scalar
implementation changes. Set `TIMING_REPEATS` from 1 through 100 to change the
bounded calls per sample.

The default `TIMING_ROUNDS=2` is a quick, position-balanced pilot with two
samples per variant, not enough evidence for a production acceptance decision.
Increase it to a larger even value (for example `TIMING_ROUNDS=6`) for the
acceptance run. The summary reports median scalar/base speedup, percent change,
median absolute deviation, and coefficient of variation. Successful, failed,
and interrupted runs archive all evidence collected so far unless
`CREATE_ARCHIVE=0` is set.

`cuda-sm75-production-scalar-e2e.sh` is the next decision boundary after the
model-free evidence passes. It performs a balanced, fixed-frontier four-GPU
full-model A/B with only the validated scalar switches changed. For the
IQ2/IQ2/Q4 hybrid it jointly tests IQ2 gate/up plus Q4 down using the measured
25/18 pipeline split. For full Q4 it jointly tests Q4 gate/up plus Q4 down
using the memory-safe 22/21 split. It writes every raw CSV/log, a paired
frontier summary, GPU telemetry, model/binary/prompt provenance, and a
returnable archive. Two audited full sweeps run first and equally precondition
host page-cache/system state. They verify all 43 routed-layer quant recipes, every
expected layer/device dispatch, exact placement cardinality, all 344 dense-Q8
weight slices (including the exact eight tensor identities per layer), cache
decisions, and bit-exact raw FP32 full-vocabulary logits at every frontier. Any
mismatch stops before repeated timing. The timed AB/BA processes contain no
dispatch audit, Q8 lookup audit, or inter-frontier logit I/O. Instead, each
process first runs an untimed `CTX_START` prefill, recreates an empty session,
records the resulting Q8 cache ranges, and then starts measured frontiers. It
records those ranges again after the final synchronized frontier. The
runner requires exact pre/post, base/scalar, and validation/timed cache-state
equality. Thus every measured Q8 lookup starts from the same proven resident
set and no cache admission occurs during a timed sweep. The default two
repeats are explicitly a pilot; set `REPEATS=6` for a replicated measurement.
Production now selects the scalar specializations by default on SM75. The
runner still produces a controlled A/B: it explicitly sets all three scalar
switches to `0` for the base arm, then enables only the recipe-relevant paths
for the scalar arm. Set any of the following to `0` to roll back an individual
production path:

- `DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=0`
- `DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=0`
- `DS4_CUDA_MOE_IQ2_SCALAR_SM75=0`

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export RECIPE=hybrid
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,2,1,3 GPU_VRAM=auto REPEATS=2 \
  ./speed-bench/cuda-sm75-production-scalar-e2e.sh
```

Exact per-kernel NCU capture is opt-in:

```bash
RUN_NCU=1 NCU_USE_SUDO=1 \
  bash speed-bench/cuda-sm75-production-scalar.sh
```

### SM75 next-target A/B suite

`cuda-sm75-next-targets.sh` builds the CUDA regression binary, verifies every
new path against the retained baseline, and benchmarks one change at a time
before the combined configuration. New paths remain opt-in until the real
four-GPU suite demonstrates a win.

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export RECIPE=hybrid
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-sm75-next-targets.sh
```

Use `RECIPE=full-q4` with the stock Q4 model to test dense-Q8 token16 reuse
and compact Q4 gate/up tile16. Use `RECIPE=stock-q2` with the stock Q2 model
to test Q2_K-down IMMA. `PROMPT_MANIFEST` applies the same variants to a
tab-separated fixed prompt suite.

The individual production switches are:

- `DS4_CUDA_Q8_MMA_SM75_TOK16=1`
- `DS4_CUDA_MOE_Q4_GATE_TILE16_SM75=1`
- `DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75=1` (stage6 is the tile16 default)
- `DS4_CUDA_MOE_IQ2_STAGE6_SM75=1`
- `DS4_CUDA_MOE_IQ2_STAGE4_SM75=1`
- `DS4_CUDA_MOE_MIXED_TAIL_TILES=1`
- `DS4_CUDA_MOE_Q2_DOWN_MMA_SM75=1`

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### CUDA prefill audit

`cuda-prefill-audit.sh` captures both an ordinary 2K measurement and an Nsight
Systems trace of the same production pipeline. The trace starts after model
loading and workspace allocation, so the report contains the prefill itself
rather than the roughly 100 GiB model-residency setup. It also records the GPU
inventory and topology, NUMA layout, exact pipeline schedule and transfer-byte
accounting, GPU telemetry, kernel/API summaries, and the raw `.nsys-rep` and
SQLite trace.

The audit explicitly selects the established baseline: Q8-to-F16 cache and the
two-stage pipeline on, experimental attention head splitting off. It does not
enable the existing event-based MoE, attention-output, or layer-stage profilers;
those synchronize streams and alter the schedule being measured.

```bash
cd ~/ds4-iq2-q4
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-prefill-audit.sh
```

The default logical tier order is `0,2,1,3`. On a machine whose physical
NVLink pairs are `0<->1` and `2<->3`, this makes logical tier 2 the partner of
tier 0 and logical tier 3 the partner of tier 1 while the two pipeline homes
remain logical tiers 0 and 1. Override `GPU_DEVICES` only when the physical
topology differs.

### SM75 deep prefill audit

`cuda-prefill-deep-audit.sh` performs the next evidence pass in one command:

- repeated, uninstrumented 21/22 and 25/18 placement sweeps;
- one device-only tile audit recording actual owned pairs, tile count, active
  experts, padding, and owner identity for every layer and microbatch;
- four narrowly filtered Nsight Compute reports: IQ2 gate/up and Q4 down from
  one early and one late layer.

The tile audit extends the already-running tile-offset kernel. It adds no
per-layer launch, timing event, synchronization, or host read. Its buffers are
allocated before the measured prefill and copied once per GPU after the
existing prefill synchronization and timing window.

```bash
cd ~/ds4-iq2-q4
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-prefill-deep-audit.sh
```

The default suite measures context frontiers from 2K through 64K three times
for each placement. Set `PROMPT_MANIFEST` to a tab-separated `label<TAB>path`
file to run the same fixed frontiers over multiple prompt corpora. Set
`RUN_NCU=0` when only the placement and tile evidence is needed. The script
always packages partial results on exit and records the failing phase in
`run-status.txt`.

Use `DEEP_AUDIT_DIR` to select or resume a directory; the generic `AUDIT_DIR`
used by the earlier Nsight Systems script is intentionally ignored. To resume
only the four Nsight Compute reports after enabling performance-counter access:

```bash
export DEEP_AUDIT_DIR="$PWD/prefill-deep-audit-YYYYMMDDTHHMMSSZ"
SKIP_BUILD=1 SKIP_PLACEMENT=1 SKIP_TILE=1 \
  ./speed-bench/cuda-prefill-deep-audit.sh
```

If the system restricts counters to administrators, set `NCU_USE_SUDO=1` for
the resume run or change the NVIDIA driver profiling-permission policy.

### Full-Q4 SM75 evidence pass

`cuda-q4-prefill-evidence.sh` runs the fixed prompt suite on the untouched
22/21 production path, records every Q8-to-F16 cache decision, and captures an
Nsight Systems runtime trace. Full-model Nsight Compute is disabled: replaying
the 153.33 GiB four-GPU process either requires an unsafe kernel checkpoint or
relaunches a heavyweight application whose attachment boundary was not
reliable on the test system.

```bash
cd ~/ds4-iq2-q4
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-q4-prefill-evidence.sh
```

Replace the example `MODEL` value with the exact full-Q4 file on the machine.
The script rejects a missing model rather than searching for one implicitly.
Set `PROMPT_MANIFEST` to reuse the fixed multi-prompt suite. Partial results
are archived on failure. `RUN_NCU` must remain zero.

Use `cuda-sm75-kernel-profile.sh` for Nsight Compute. It links only the CUDA
backend, opens no model file, launches no helper processes, and has no DS4 CLI
instance lock. Its routed-expert scenarios allocate exactly 128 resident
experts at the production 4096->2048->4096 dimensions. They cover Q4 gate/up,
Q4 down, IQ2_XXS gate/up, and Q2_K down. The harness hard-caps predicted device
state at 3 GiB and confirms the synthetic weight mapping was copied into VRAM.
The early and late assignments reproduce the recorded pair count,
active-expert count, and tile16 count; their per-expert histogram is explicitly
synthetic because the prior audit did not retain that histogram.

The dense-Q8 scenarios cover every production reduction template and the
corresponding dominant shape: attention q_b T32 (1024x32768), shared-down T64
(2048x4096), attention q_a T128 (4096x1024), and attention-output-b T256
(8192x4096), all at the 512-token runtime microchunk. A small Q8 kernel-replay capture validates
counter permissions, attachment, kernel filtering, and report export before
the main captures begin. The focused metric set records duration, launch
resources, achieved occupancy, eligible warps, IMMA and integer-pipe activity,
DRAM/L1/L2 activity and hit rates, atomic activity, and scoreboard/MIO stalls.

```bash
cd ~/ds4-iq2-q4
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-kernel-profile.sh
```

Set `PROFILE_GPU` to choose the physical SM75 device. The script exposes only
that GPU to the harness, where it becomes logical device zero, and validates
every report against the harness process, device, kernel name, and positive
duration. `PROFILE_SET=remaining` captures Q2_K down and all four dense-Q8
templates. It deliberately repeats T64/T128 because changes to the common Q8
loader can change every template's memory instruction stream.
`NCU_SET=targeted` or `NCU_SET=full` are slower opt-ins.

### SM75 packed-INT4 Q4 experiment

`cuda-sm75-int4-mma-experiment.sh` is a bounded, model-free screening
experiment for the current Q4 nibble-to-INT8 path. Each warp computes a
16-token by 8-output tile over K=256, reusing each B fragment across two
8-token MMA tiles; eight independent warps run in each 256-thread CTA. It
compares six alternatives:

- the shipping-style two-`m8n8k16` INT8 baseline;
- direct `m8n8k32` INT4 with runtime packing from standard Q8/Q4 bytes;
- standard Q8 with a same-row, block-local `group32` Q4 layout;
- an MMA-native, size-neutral Q4 code-payload layout;
- size-neutral MMA-native Q8/Q4 value payloads (an idealized upper bound);
- an all-u4 correction formulation retained as a measured reference.

The preferred decomposition is `a_s8 = low_u4 + 16*high_s4`, so it uses one
`u4 x u4` and one `s4 x u4` MMA without introducing another zero-point term.
The harness proves only the scaled integer component
`sum_j scale_j * sum_k(a_k * q_k)`. It does not exercise Q4_K `d`, `dmin`,
packed scale/min decoding, the Q8_K scale and `bsums`, the minimum correction,
or the production floating-point accumulation order.

The measurements screen three narrower questions: hot-cache instruction
economics, streaming/layout economics, and whether a native packing is worth a
production-shaped prototype. The script runs both a 16-weight-case hot test and
a 16,384-weight-case streaming-size test (more than 17 MiB of uniquely touched
Q4 payload/scale data, versus the RTX 8000's 6 MiB L2) while keeping
activations in an independent 16-case hot set. Variant order is rotated and
reversed across nine rounds; the summary reports the median and range rather
than a single fixed-order sample.
Native-Q8 (`native-A`) results are an optimistic upper bound because the
benchmark starts after the 16-token transpose and nibble-plane transformation;
that recurring transformation cost is excluded.
The all-u4 alternative also needs one 16-bit weight sum per 32-value subgroup,
increasing a full Q4_K block from 144 to 160 bytes (+11.1%) unless those sums
are recomputed at runtime.

The experiment confirms emitted 8x8x32 IMMA instructions, runs Compute
Sanitizer when installed, benchmarks every layout, and captures focused Nsight
metrics. It never opens a GGUF or changes production dispatch. Before any
dispatch change, the winning idea still requires production-shaped gate/up and
down prototypes plus a full, bitwise Q4_K x Q8_K regression covering headers,
minimum correction, multiple blocks, and the production float reduction order.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-int4-mma-experiment.sh
```

Return the generated `sm75-int4-mma-<timestamp>.tar.gz`. These results can
reject unattractive layouts, but cannot by themselves justify production
dispatch.

### SM75 production-shaped Q4 down native-packing audit

`cuda-sm75-q4-down-native.sh` exercises the model-free, production-shaped
Q4_K down-projection follow-up. It compares the standard layout with native-W,
prepacked native-A/W consumption, and the combined activation-pack plus
native-A/W path. `native-aw-consumer` measures only the consumer after native-A
packing is complete; `native-aw-combined` charges the activation transform to
the timed result, while `pack-a` reports that transform separately. Ten rounds
balance every one of the five variants across every sample position. The early
layer-3 shape has 1,879 routed token/expert pairs,
99 active experts, 183 tile16 launches, and 1,049 padded slots. The late
layer-36 shape has 2,186 pairs, 76 active experts, 189 tile16 launches, and 838
padded slots. Both use the full 3,072-row production token/slot surface with
the owned pair IDs scattered through it. The harness validates activation
packing exactly, requires initialized owned outputs to be bit-exact against
its own standard-path reference, and verifies that every unowned output stays
poisoned before running the randomized, position-rotated benchmark rounds for
both recorded shapes. This is not a claim of bit-exact agreement with the
production kernel, which is built under a different compiler mode.

The script builds only an `sm_75` cubin, rejects any GPU other than compute
capability 7.5, runs Compute Sanitizer when available, and checks the emitted
standard/native kernels with `cuobjdump`. Focused Nsight Compute captures cover
one representative standard, native-W, and native-A/W consumer kernel, plus
the activation-pack transform that the combined result charges. Captures cover
launch resources, occupancy, IMMA utilization, memory behavior, and scheduler
stall metrics. Set `NCU_USE_SUDO=1` on systems with restricted performance
counters; the same `sudo -E` command is used for metric discovery and capture.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-down-native.sh
```

No `MODEL` variable is needed and no GGUF is opened. Return the generated
`sm75-q4-down-native-<timestamp>.tar.gz`; partial evidence is archived on
failure or interruption as well as on success. The result is an implementation
audit, not by itself evidence for changing production dispatch.

If the initial run completed its build, SASS, correctness, sanitizer, and both
benchmarks but failed in Nsight preflight or capture, resume only the Nsight
phase against the still-present output directory. For example:

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

Q4_DOWN_NATIVE_DIR="$PWD/sm75-q4-down-native-20260803T180034Z" \
RESUME_NCU=1 \
PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-down-native.sh
```

Resume mode rejects any run that did not fail in an Nsight phase and validates
the recorded source, Makefile, binary, SASS, correctness, sanitizer, benchmark,
and telemetry evidence before skipping it. It never overwrites the original
evidence or failure archive: new provenance, status, and Nsight files go under
`resume-<timestamp>/`, and the resulting archive is named
`sm75-q4-down-native-<original-timestamp>.resume-<timestamp>.tar.gz`.

### SM75 production-shaped Q4 gate/up native-packing audit

`cuda-sm75-q4-gate-up-native.sh` is the standalone, model-free gate/up
follow-up. It runs exact pack and output correctness, Compute Sanitizer when
available, balanced early/late benchmarks, GPU telemetry, SASS inspection,
and focused Nsight Compute captures in one command. The early layer-3 shape
has 1,879 routed pairs, 99 active experts, and 282 synthetic tile8 records;
the late layer-36 shape has 2,186 pairs, 76 active experts, and 331 tile8
records. Both use 512 tokens, six selected experts per token, a 4,096-wide
input, and a 2,048-wide gate/up result.

The fourteen benchmark variants retain all controls and add scalar-register
experiments for the stage7 and warp16 native-A/W consumers. The complete set is
`standard`, `standard-warp16`, `native-w`, `native-aw-consumer`,
`native-aw-warp16-consumer`, `native-aw-stage7-consumer`,
`native-aw-combined`, `native-aw-warp16-combined`,
`native-aw-stage7-combined`, `native-aw-stage7-scalar-consumer`,
`native-aw-warp16-scalar-consumer`, `native-aw-stage7-scalar-combined`,
`native-aw-warp16-scalar-combined`, and `pack-a`. Consumer variants exclude
the already-completed activation transform; combined variants charge
activation packing plus the matching consumer, and `pack-a` isolates that
transform. Native weight packing is an offline layout operation and is
excluded from timed consumer work. By default, 28 rounds rotate all fourteen
variants evenly through every sample position twice. Each sample requests at
least 20 launches and may automatically use more to reach a stable 100 ms
consumer timing window; both requested and effective launch counts are
recorded and validated.

For each scenario, `benchmark-*-scalar-comparison.csv` pairs scalar only with
its matching base stage7 or warp16 variant, separately for consumer and
combined timing. It records median time, delta, speedup, sample coefficient of
variation and median absolute deviation, plus the independently timed pack-A
share. This avoids cross-shape or cross-scenario comparisons.

Correctness requires exact activation and weight packing, bit-exact owned
outputs against the harness's standard path, and untouched poison in every
unowned output. The gate/up target deliberately keeps the shipping
`--use_fast_math` compilation mode because its epilogue executes millions of
`expf` operations; the bit-exact claim is therefore an intra-binary comparison
with the harness's own standard path, not an external production reference.
SASS validation requires the expected m8n8k16 or m8n8k32 packed-INT4 forms.
The evidence deliberately keeps three different concepts separate:
`sass-summary.csv` counts `LDL`/`STL` instructions,
`ptxas-resource-summary.csv` reports compiler stack-frame and spill bytes, and
`scalar-resource-comparison.csv` reports CUDA `localSizeBytes`, raw and
Turing-allocation-rounded registers, shared memory, blocks, warps, and
occupancy. `scalar-local-memory-comparison.csv` and
`scalar-acceptance-summary.csv` compare each scalar kernel directly with its
base. A scalar candidate passes only when its SASS local instructions, PTXAS
stack/spill bytes, CUDA local bytes, register ceiling, and exact occupancy gate
all pass. A missed experimental gate is recorded and warned about but does not
discard the timing and profiling evidence; it is not an eligibility or
production-dispatch claim.

Nsight produces eleven focused reports: standard, base stage7, scalar stage7,
base warp16, and scalar warp16 for both early and late, plus one early pack-A
capture. Optional discovered metrics include local-load/store sectors and
bytes so base-versus-scalar local traffic can be measured directly. The script
only observes clocks and uses Nsight's `--clock-control none`; it never changes
GPU clocks or power settings.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-gate-up-native.sh
```

No `MODEL` variable is needed and no GGUF is opened. Return
`sm75-q4-gate-up-native-<timestamp>.tar.gz`; the script also archives partial
evidence on failure or interruption. This is an implementation audit and does
not enable or alter production dispatch.

If a completed build/SASS/correctness/sanitizer/benchmark pass fails only in
the Nsight phase, resume it without replacing the original evidence:

```bash
Q4_GATE_UP_NATIVE_DIR="$PWD/sm75-q4-gate-up-native-YYYYMMDDTHHMMSSZ" \
RESUME_NCU=1 \
PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-gate-up-native.sh
```

Resume mode revalidates provenance, build freshness, separate SASS/PTXAS/CUDA
resource evidence and scalar acceptance classifications, correctness,
sanitizer, both balanced benchmarks and their pair tables, and telemetry.
New files go under `resume-<timestamp>/`; the resulting archive is named
`sm75-q4-gate-up-native-<original-timestamp>.resume-<timestamp>.tar.gz`.

### Comprehensive stock-Q2 and remaining-kernel pass

`cuda-sm75-comprehensive-audit.sh` combines the full stock-Q2 production pass
with the bounded missing-kernel captures. The production half runs the fixed
prompt suite, records routed tile occupancy and every Q8-to-F16 cache decision,
and generates an Nsight Systems kernel-time distribution. The NCU half opens no
GGUF and profiles early/late Q2_K down plus all four dense-Q8 templates. Partial
results are archived if either half fails.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL_Q2="$PWD/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-comprehensive-audit.sh
```

The command deliberately names and exports the actual stock-Q2 file; neither
script searches for a model or inherits the unrelated `MODEL` variable.
Set `PROMPT_MANIFEST` for the fixed multi-prompt suite. Set `PROFILE_SET=all`
to repeat all four routed-expert kernel families and all four native-Q8
templates under one build instead of capturing only the remaining gaps.

### SM80 to SM75 dispatch and resource audit

[`CUDA_SM80_TO_SM75_DISPATCH_MATRIX.md`](../CUDA_SM80_TO_SM75_DISPATCH_MATRIX.md)
is the source-reviewed architecture matrix for the normal DeepSeek V4 Flash
CUDA path. It covers dense projections, attention, indexer, all four routed
expert quant combinations, decode, cache-dependent dispatch, and four-GPU
orchestration. Unknown compiled resources are marked explicitly rather than
inferred from source.

Generate paired cubin resource and instruction reports without loading a model:

```bash
cd ~/ds4-iq2-q4
./speed-bench/cuda-sm-dispatch-resource-audit.sh
```

The script compiles `ds4_cuda.cu` for both `sm_75` and `sm_80`, records the
complete source kernel/launch/gate inventories, extracts per-function cubin
resource usage, and counts IMMA, HMMA, DP4A, load, and barrier instructions.
It packages the report as `sm-dispatch-resource-<commit>-<timestamp>.tar.gz`.
No `MODEL` variable is required because this audit does not execute inference.

To reparse and repackage an existing audit without recompiling its CUDA
objects, name its existing directory explicitly:

```bash
OUT_DIR="$PWD/sm-dispatch-resource-<commit>-<timestamp>" \
REPARSE_ONLY=1 \
./speed-bench/cuda-sm-dispatch-resource-audit.sh
```

Object-wide instruction totals are compiler-coverage checks only. Both objects
contain symbols excluded by runtime architecture gates; use the paired
per-function tables together with the production dispatch matrix.

### SM75 prefill critical-path and pair attribution

`cuda-sm75-critical-path-audit.sh` is the bounded follow-up to the T256
partner-policy screen. The production baseline defaults to the tagged
SM75-native full-Q4 model at
`/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf`;
an ordinary Q4 file is rejected unless `ALLOW_STANDARD_MODEL=1` is explicit.
It does not repeat a throughput A/B. It records two
2048-token Nsight Systems traces with the fixed 22/21 split: `0,3,1,2` and the
pair-preserving swap `3,0,2,1`. Thus each layer stage runs once on physical GPU
0 and once on physical GPU 3 while each home GPU retains its direct NVLink
partner. The same production-shaped Q4 and dense-Q8 harness work is also timed
on GPUs 0 and 3, with alternating order, to separate hardware/clock behavior
from layer-range behavior.

The opt-in `DS4_CUDA_CRITICAL_PATH_NVTX=1` annotations cover waves,
microbatches, stages, layers, inter-stage handoffs, output, and T256 partner
projections. They do not create CUDA events, synchronize, or add waits. The
SQLite summarizer attributes existing kernel and copy timestamps through their
CUDA correlation IDs and emits the complete operation table, per-microbatch
GPU envelopes, stage/device totals, the 2x2 physical-GPU versus stage factors,
and same-work GPU medians. Model hashing remains disabled.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

unset MODEL ALLOW_STANDARD_MODEL RESUME_DIR
export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

SKIP_BUILD=0 \
./speed-bench/cuda-sm75-critical-path-audit.sh
```

Return `sm75-critical-path-<timestamp>.tar.gz`. The archive is also produced
on interruption or failure and includes both `.nsys-rep` files, SQLite
exports, 100-ms GPU clock/power telemetry, harness logs, and derived CSVs.

If the historical standard-layout run from commit `ab54701` failed in
`same-work-harness` because a dense-Q8
scenario lacked `timed_per_call_ms`, reuse the completed Nsight Systems traces
instead of capturing them again:

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export ALLOW_STANDARD_MODEL=1
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export RESUME_DIR="$PWD/sm75-critical-path-20260805T071911Z"

SKIP_BUILD=0 \
./speed-bench/cuda-sm75-critical-path-audit.sh
```

Resume mode validates the model, layouts, split, failed phase, reports,
SQLite exports, and telemetry before rerunning only the bounded same-work
harness and summarizer. It preserves the failed run's provenance and writes a
separate `.resume-<timestamp>.tar.gz` archive.

### Four-GPU Q4 clock and power normalization

`cuda-sm75-q4-clock-audit.sh` diagnoses the physical-board effect without a
GGUF or multi-GPU placement. It times identical production-shaped early and
late Q4 work on GPUs 0, 1, 2, and 3 in alternating order, then repeats late Q4
with both NVLink pairs active concurrently. Its telemetry includes every
clock-event/throttle-reason field supported by the installed driver.

The default second phase is a reversible common-clock diagnostic at 1620 MHz.
It does not claim 1620 MHz is the desired production setting: the purpose is
to determine whether Q4 timing converges at equal clocks and whether GPU2
shares GPU3's behavior. Original persistence mode, power limits, and unlocked
clock state are restored on success, failure, or interruption.

`TARGET_SM_CLOCK_MODE=1` selects NVIDIA's closed-loop clock-locking mode, which
is documented as potentially improving performance per watt. This is relevant
when the default mode cannot sustain the requested clock because of software
power capping; it must still be measured rather than assumed beneficial.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

NORMALIZE=1 \
TARGET_SM_CLOCK=1620 \
RAISE_POWER_LIMIT=0 \
USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-clock-audit.sh
```

Return `sm75-q4-clock-<timestamp>.tar.gz`. Do not raise power limits until the
baseline report establishes the active clock-event reason and each board's
reported default and maximum limits.

### SM75 Q4 memory-clock power-headroom sweep

`cuda-sm75-memory-clock-sweep.sh` is the bounded follow-up when the clock audit
shows software power capping below the requested SM clock. It compares GPU1 as
the healthy passive-board control against GPUs2/3, discovers their common
supported memory clocks, and samples up to five clocks in the upper operating
range. Each point records actual SM/memory clocks, power, temperature, throttle
reasons, exactness, and production-shaped Q4 early/late timings.

The sweep does not change power limits. Runtime SM and memory locks are reset,
and original persistence settings restored, on success, failure, or interrupt.
Run it only when these GPUs have no other compute processes:

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

DEVICES=1,2,3 \
TARGET_SM_CLOCK=1620 \
TRIALS=3 \
REPEATS=100 \
USE_SUDO=1 \
./speed-bench/cuda-sm75-memory-clock-sweep.sh
```

Return `sm75-memory-clock-<timestamp>.tar.gz`. `decisions.csv` reports the
best measured clock per device and scenario, but a production clock should be
chosen only if both Q4 scenarios improve and the full native-Q4 benchmark then
confirms the result.

Some Turing boards expose only a full-performance memory state and a 405 MHz
idle state whose supported graphics ceiling is also low. In that case the
script records `outcome=not-applicable` and exits successfully without changing
clocks; forcing the idle memory state is not a useful power-headroom test.
