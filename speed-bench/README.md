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

### All-native Q8 versus complete production FP16-cache quality endpoint

`cuda-q8-fp16-full-quality.sh` scores only the two decision-relevant production
endpoints selected after the T256 placement screen:

- every Q8 projection uses native Q8 because the complete FP16 expansion cache
  is disabled; and
- the complete production FP16-cache policy runs every active T256 output-B
  projection with FP16/cuBLAS on its NVLink partner and retains whatever
  non-T256 FP16 projections the production admission planner selects locally.
  The T256 inventory is fixed at 43/43—one live partner binding and one live
  physical F16 allocation per layer—but the non-T256 inventory is deliberately
  dynamic rather than pinned to a stale count from an earlier model plan.

The runner validates execution, not just requested policy. Across the 100-case
suite the native endpoint must record exactly 4,300 native T256 calls. The
FP16 endpoint must record exactly 4,300 partner-FP16 calls, with no local-FP16
T256 call and no T256 native fallback. The scorer exports binding and physical
allocation state after all 100 cases. Every exported F16/F32 binding and
allocation must be used and live, dead expanded-weight bytes must be zero, and
no non-T256 binding may be partner-offloaded. The summary records the exact
dynamic non-T256 class and descriptor inventory before reporting
official-continuation NLL, first-token accuracy, greedy-prefix length, and
API-reference metrics with a paired bootstrap interval. It never hashes the
model and does not require a new quant.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-fp16-full-quality.sh
```

Return `q8-fp16-full-quality-<timestamp>.tar.gz`. A valid archive has
`Experiment integrity: PASS`; the quality gate is a measured result and may
independently pass or fail.

### T256 execution-placement A/B

Each of the 43 layers has one active T256 output-B GEMM and one logical cache
binding. The binding names its home consumer and the physical device holding
the single FP16 expansion; partner placement no longer creates an unused fixed
peer-consumer duplicate.

`cuda-q8-t256-placement-ab.sh` measures and verifies five policies:

- `native` (`native-T256-only`): zero T256 FP16 bindings and 43 native-Q8
  layer paths; every other Q8 cache class remains enabled so this isolates
  T256 placement rather than changing the arithmetic of all dense projections;
- `all-local`: 43 local plus zero partner bindings, with all 43 active GEMMs
  local. T256 is ranked ahead of tied T32 entries so the complete class fits;
- `balanced`: even layers execute locally and odd layers execute on their
  partner: 22 local plus 21 partner bindings. Each pipeline stage is balanced
  within its own NVLink pair (11/11 for layers 0-21 and 11/10 for layers
  22-42);
- `overflow`: exactly 43 live bindings, with zero to seven natural partner
  overflows drawn only from eligible layers 15-21. The runner records the
  observed local/partner count instead of assuming the earlier 36/7 split; and
- `all-partner`: zero local plus 43 partner bindings.

Every run performs an untimed warm-up, then exports binding, physical
allocation/liveness, and runtime-audit tables before disabling the success
counters for the measured sweep. The summarizer rejects a run whose active
layer paths do not match its policy, whose 43 T256 bindings do not map one-to-one
to 43 live physical weights, or whose forced partner placement silently misses.
It also records all other binding shapes and total resident/dead bytes so cache
entries displaced by `all-local` remain explicit in the legacy mode.

For the forthcoming mixed Q4/IQ2 model that has enough VRAM for every dense
FP16 expansion at a 256K allocation, set `REQUIRE_COMPLETE_DENSE_CACHE=1` and
`CTX_ALLOC=262273` (256K context + 128 generation tokens + 1). This strict
mode still measures only the requested 2K--32K
frontiers, but creates the production session at the 256K memory pressure
before the untimed warm-up materializes the cache. It explicitly disables the
experimental attention-head split and exports the planner audit as well as the
binding, physical-allocation, and post-warm-up CUDA-memory tables. Strict mode
defaults to a predeclared 512 MiB minimum free-memory floor on every device;
raise it with `MIN_FREE_MIB`, but do not lower it after seeing a result.

The strict summary requires the exact same 344-candidate production plan in
every arm:
43 each for Q-A, Q-B, KV, output-A, output-B, and shared-down, plus 86 shared
gate/up weights. Every required FP16 candidate must be admitted, used, and
backed one-to-one by a live allocation; displaced non-T256 entries, dead bytes,
aliases, or unreferenced allocations are fatal. The `native-T256-only` control
still exports all 344 plan rows, but its 43 output-B rows must be deliberately
`unadmitted`; its other 301 candidates must be admitted and live.
Strict mode omits the natural-overflow arm because complete coverage makes it
redundant; it compares native-T256-only, all-local, balanced, and all-partner.
Legacy mode retains the five-arm overflow reproduction.

The mixed-model generator sizes its reclaim against the worst-case all-local
home footprint (an additional 1.375 GiB on the 22-layer home and 1.3125 GiB on
the 21-layer home relative to all-partner). This is a placement-neutral capacity
envelope, not a claim that all-local is optimal. The runner still fails closed
unless every tested arm preserves complete coverage and the free-memory floor.

One-repeat screen:

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=262273 \
REQUIRE_COMPLETE_DENSE_CACHE=1 \
MIN_FREE_MIB=512 \
REPEATS=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-t256-placement-ab.sh
```

Omit `CTX_ALLOC` and `REQUIRE_COMPLETE_DENSE_CACHE` when reproducing the
legacy full-Q4 screen where displacement is part of the measured cache-policy
tradeoff.

For the strict acceptance result, rerun with `REPEATS=4`; the four rotations
put every strict policy in every thermal/run-order slot. Legacy five-arm
reproduction still requires `REPEATS=5`. Return
`q8-t256-placement-<timestamp>.tar.gz`.

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

### Exact native-Q8 T256 partner anchor

`cuda-q8-partner-native-exact.sh` is the prerequisite to any concurrent row
split. It transports the block-Q8 activations/scales over NVLink and invokes
the same SM75 exact-MMA projection used by the local path against the partner's
already-resident Q8 weights. It admits only T256 layers 15-21 and preserves the
frozen local cache plan. The runner refuses to benchmark unless the dedicated
two-GPU regression and all 100 production quality cases are byte-exact.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REPEATS=3 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-partner-native-exact.sh
```

The performance phase is deliberately limited to 16K and 32K. Every repeat
must retain the same seven additive T256 bindings, record only
`native_q8_partner_hit/exact_sm75_mma`, and produce byte-identical frontier
logits. Return `q8-partner-native-exact-<timestamp>.tar.gz`.

### Tagged SM75 native-Q4 with all-partner T256

`cuda-sm75-native-q4-t256-ab.sh` is a focused integration runner; it does not
change the engine, either GGUF, or the native-Q4 dispatch. It requires separate
existing absolute paths in `MODEL` for the
standard full-Q4 GGUF and `NATIVE_MODEL` for its tagged SM75-native repack.
Both arms force the measured all-partner T256 winner: 43 active partner
bindings backed by 43 physical T256 expansions, with all output-B GEMMs
executing on the NVLink partners. After the untimed warm-up, the runner also
requires every expanded-weight allocation to be live and compares the standard
and native arms by their exact live non-T256 descriptor multiset: label, source
offset/bytes, dimensions, consumer/resident device, storage kind, and arithmetic.
There is no historical minimum binding-count assumption. It uses the production
`0,3,1,2` device order and 22/21 split and sweeps 2K through 32K. One
standard-then-native pair is the default; increase `REPEATS` only when repeat
statistics are specifically needed.

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
REPEATS=1 \
./speed-bench/cuda-sm75-native-q4-t256-ab.sh
```

Return `sm75-native-q4-t256-ab-<timestamp>.tar.gz`. The archive contains the
focused throughput comparison, bit-exact standard/native full-logit checks,
exact all-partner T256 allocation/binding/runtime evidence, frozen-cache checks,
exact live non-T256 inventory equality, zero dead expanded-weight allocations,
and per-run GPU telemetry.
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

### SM75 routed-quant performance screen

`cuda-sm75-q3-q4-32.sh` is the model-free arithmetic and native-layout gate
for the format contracts in
[`SM75_Q3_Q4_32_DESIGN.md`](../SM75_Q3_Q4_32_DESIGN.md). It compares ten
M16xN8xK256 warp kernels with eight warps per 256-thread CTA:

- the existing affine Q4_K K32 arithmetic in a 1152-byte native tile, with
  independent scale and minimum accumulator chains;
- standard 110-byte Q3_K, losslessly repacked into an 880-byte native tile
  and executed exactly with K16 S8xU8 MMA plus `-4*bsum`;
- the experimental 104-byte Q3-32 format in an 832-byte native tile, using
  two K32 INT4 MMAs plus `-4*bsum`;
- the experimental symmetric 136-byte Q4-32 format in a 1088-byte native
  tile, using signed-B K32 INT4 MMA without an affine-minimum correction.
- Q2/Q3-50, Q2/Q3-75, Q3A4, Q3A6, Q3/Q4-50, and Q5-32, each in an exact
  size-neutral native tile matching the corresponding quality-screen law.

The harness carries opaque FP16 headers but deliberately measures only the
exact integer-accumulator component; it does not claim floating-output or
quantizer-byte exactness. It exhaustively validates Q3_K unpacking and
each Q3-32 bit plane, samples mixed Q3-32 states deterministically, checks
canonical/native byte round trips, and compares adversarial and seeded-random
MMA accumulators with separate scalar references. It reports both a
single-projection down-like shape and a dual-projection gate/up shape that
retains both accumulator chains and executes the SiLU-times-up epilogue.
Each has hot and L2-exceeding streamed-weight modes using the format's real
byte footprint. The runner records SASS/resource evidence, runs Compute
Sanitizer when available, and profiles both shapes for occupancy, tensor-pipe
activity, memory traffic, and stalls with a direct harness process.

IQ2_XXS is not forced into this linear-format harness. Instead, the runner
times the real early/late hybrid IQ2 gate/up + Q4 down production calls and
writes them to `iq2-production-control.csv`. Those absolute production times
are a separately labelled control, not a ratio against the bounded K256
tables.

This is not yet a quantizer, GGUF loader, routed-MoE dispatch, quality result,
or end-to-end speed claim. Standard Q3_K must later match the upstream
quantizer byte-for-byte. Q3-32 and Q4-32 require private versioned type/layout
tags and real-weight quality gates before a full model is generated.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
./speed-bench/cuda-sm75-q3-q4-32.sh
```

Return `sm75-q3-q4-32-<timestamp>.tar.gz`. The archive contains four timing
tables (`down-hot`, `down-streamed`, `gate-up-hot`, and
`gate-up-streamed`), exact layout/arithmetic gates for all ten linear formats,
the production IQ2 control, and per-shape Nsight reports. This pass ranks
formats only; it does not choose a deployment recipe.

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
partner-policy screen. It requires the tagged mixed-Q4/IQ2 model and its
production 256K allocation. It does not repeat a throughput A/B. It records
two 8192-token Nsight Systems traces with the fixed 22/21 split: `0,3,1,2` and the
pair-preserving swap `3,0,2,1`. Thus each layer stage runs once on physical GPU
0 and once on physical GPU 3 while each home GPU retains its direct NVLink
partner. The same production-shaped hybrid IQ2/Q4 routed work is also timed
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

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

PROFILE_TOKENS=8192 \
CTX_ALLOC=262273 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-critical-path-audit.sh
```

Return `sm75-critical-path-<timestamp>.tar.gz`. The archive is also produced
on interruption or failure and includes both `.nsys-rep` files, SQLite
exports, 100-ms GPU clock/power telemetry, harness logs, and derived CSVs.

If a run fails in `same-work-harness`, reuse the completed Nsight Systems
traces instead of loading the model again:

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export RESUME_DIR="$PWD/sm75-critical-path-20260805T071911Z"

SKIP_BUILD=0 \
./speed-bench/cuda-sm75-critical-path-audit.sh
```

Resume mode validates the model, layouts, split, failed phase, reports,
SQLite exports, and telemetry before rerunning only the bounded same-work
harness and summarizer. It preserves the failed run's provenance and writes a
separate `.resume-<timestamp>.tar.gz` archive.

### SM75 IQ2 tail-policy A/B

`cuda-sm75-iq2-next-ab.sh` preserves the accepted production-tail validation.
Its bounded harness verifies that one IQ2 MMA tile8 for every residual of
1--8 pairs is bit-exact with the retained tail4 rollback. A separate
alternating four-GPU A/B measures both policies; native-Q4 down retains its
existing 16/8/4 planner in both variants. Tail8 is the SM75 production default,
and `DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75=0` is the explicit rollback. The rejected
512-thread IQ2 CTA is no longer a production dispatch option.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
CTX_START=2048 \
CTX_MAX=8192 \
CTX_ALLOC=262273 \
REPEATS=3 \
PROFILE_GPU=0 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-iq2-next-ab.sh
```

### Mixed-Q4/IQ2 and balanced-T256 production profile

`cuda-sm75-native-q4-t256-profile.sh` profiles the accepted combined
configuration without repeating its throughput A/B. It opens the NFS-backed
model exactly once and captures one timed 8192-token frontier at the production
256K context allocation with Nsight Systems. No placement/class environment
override is set: the trace must prove that the engine selected its automatic
balanced policy, qualified both SM75 NVLink pairs, materialized all 344 dense
F16 weights, and produced exactly 22 local plus 21 odd-layer partner T256
bindings. It also requires the tagged native-Q4 dispatch and the exact mixed
Q4/IQ2/Q4 routed recipe.

Nsight Compute does not reopen the GGUF. Production-shaped harnesses capture
representative early/late native-Q4 gate/down and IQ2 tile16/tile8 calls, plus
512-token T32 and T256 dense-F16 cuBLAS calls. The reports include occupancy,
compute/memory throughput, cache traffic, and warp-stall sections. The
production trace separately attributes stage/device work, communication,
activation conversion, both NVLink copies, partner cuBLAS, and attention
row-split kernels/copies through semantic NVTX correlation ranges. Its summary
separates local FP16 GEMMs, indexer work, format/quantization/packing, norms and
hyperconnections, RoPE, compressor work, and any genuinely unknown kernels;
it also emits per-device/per-stage group totals. A 200 ms `nvidia-smi` stream records utilization,
power, clocks, temperature, and memory while a device-buffer tile audit records
real pair/tile/padding/ownership counts with one final host copy.
At `PROFILE_TOKENS=32768`, targeted Nsight Compute is enabled automatically
for production-kernel harnesses representing late-frontier indexed top-K and
mixed online attention calls; set `RUN_ATTENTION_NCU=0` only to omit them.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PROFILE_TOKENS=8192 \
CTX_ALLOC=262273 \
PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

For the requested 32K production trace and targeted attention pass, use the
same command with:

```bash
PROFILE_TOKENS=32768 \
CTX_ALLOC=262273 \
NCU_SET=attention \
RUN_ATTENTION_NCU=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

To rank the post-row-split production bottlenecks without another Nsight
Compute pass, capture one unsynchronized 32K Systems trace. The script clears
all inherited `DS4_*` variables, explicitly records `xdev_sync=disabled`, and
fails unless both qualified row splits are active:

```bash
PROFILE_TOKENS=32768 \
CTX_ALLOC=262273 \
RUN_NCU=0 \
RUN_ATTENTION_NCU=0 \
NCU_USE_SUDO=0 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

If the targeted 32K attention reports already exist and only the genuine
production trace is missing, avoid repeating them:

```bash
unset REUSE_NSYS_DIR REUSE_Q4_NCU_DIR
PROFILE_TOKENS=32768 \
CTX_ALLOC=262273 \
RUN_NCU=0 \
RUN_ATTENTION_NCU=0 \
NCU_SET=full \
NCU_USE_SUDO=0 \
SKIP_BUILD=1 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

Return `sm75-native-q4-t256-profile-<timestamp>.tar.gz`. Model hashing is
disabled. If the GGUF is cold in the Linux page cache, its single NFS
startup may still take roughly 25 minutes at 100--105 MiB/s; the later Nsight
Compute passes do not touch that file.

If a run completed its production trace but stopped during export or summary,
reuse that trace in a new result directory and proceed to Nsight Compute
without opening the GGUF payload again:

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export REUSE_NSYS_DIR="$PWD/sm75-native-q4-t256-profile-20260826T144211Z"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PROFILE_TOKENS=8192 \
CTX_ALLOC=262273 \
PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=1 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

Resume mode first requires the prior manifest to match the model path and
size, prompt, device order, VRAM policy, stage split, profile frontier,
allocation, chunk, pipeline size, and T256 policy. It also independently
requires exactly one benchmark row at the requested frontier. This prevents an
8K capture from being silently relabeled as a 32K trace. It then copies and
revalidates the existing report, benchmark, cache
snapshots, allocation/memory evidence, tile audit, and exact balanced binding
inventory. It reruns the corrected
summaries and the model-free Nsight Compute harnesses in a fresh output
directory; it does not mutate the original failed evidence.

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

### SM75 32K indexer audit

`cuda-sm75-indexer-audit.sh` isolates the ratio-4 indexer at the final
production 512-token chunk of a 32K prefill. It does not open a GGUF. Every
existing WMMA score tile (128/64/32/16) must reproduce the shipping WMMA128
scores bit-for-bit, including the causal `-inf` region. Both the shipping
8192-entry top-k and its existing chunked-tree alternative must return the
exact ordered top-512 set.

The timing pass reports score and top-k costs separately. The optional Nsight
Compute pass records launch resources, occupancy, FP16 Tensor Core activity,
memory hierarchy traffic, and barrier/scoreboard/MIO stalls. Optional metrics
are selected from the installed Nsight version instead of making a renamed
counter abort the whole audit.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
TIMING_REPEATS=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-audit.sh
```

Return `sm75-indexer-audit-<timestamp>.tar.gz`. This pass compares existing
exact paths before any persistent-F16 cache or SM75-specific accumulator
epilogue is admitted to production.

`cuda-sm75-indexer-wmma64-production-ab.sh` is the production transfer gate
for the best existing score tile found by that audit. It interleaves the
shipping WMMA128 path with WMMA64 at 2K, 4K, 8K, 16K, and 32K under the fixed
256K allocation. The script validates both kernels against WMMA128 in the
bounded harness, audits every production score dispatch, requires identical
dispatch counts, and compares every frontier-logit file byte-for-byte.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=262273 \
REPEATS=3 \
RUN_HARNESS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-wmma64-production-ab.sh
```

Return `sm75-indexer-wmma64-production-<timestamp>.tar.gz`. A harness-only
gain is not sufficient to change the production dispatch; promotion requires
the exact interleaved full-model result.

If the uninstrumented A/B does not reproduce the isolated kernel gain,
`cuda-sm75-indexer-production-trace.sh` captures one genuine, unsynchronized
32K production trace for each exact score tile. It retains the 256K allocation,
balanced T256 placement, mirrored-KV attention row split, and 512-token
pipeline microbatch. The trace comparison reports score time per stage/device,
position/stage score time, total score launches, pipeline GPU span, and trace
throughput. This separates two materially different outcomes: either WMMA64 is
no faster across the real production launch population, or its saved GPU work
is overlapped and absent from the pipeline critical path. The synchronizing
indexer stage profiler is not enabled.

The runner creates one detached Git worktree pinned to the caller's committed
HEAD, builds there once, and uses that same binary and post-processing source
for both captures. It records and compares the `ds4-bench` SHA-256. This is
intentional: a paired capture lasts long enough that another experiment can
otherwise switch or edit the shared checkout between the first and second
post-processing pass. `SKIP_BUILD=1` is therefore rejected by this runner.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_ALLOC=262273 \
PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-indexer-production-trace.sh
```

Return `sm75-indexer-production-trace-<timestamp>.tar.gz`. Both inner traces
export the 32K frontier logits and must match byte-for-byte. Each trace also
audits all indexer dispatches and rejects mixed score-kernel use. The general
production profiler accepts `INDEXER_SCORE_TILE=128|64`; 128 remains the
shipping default and 64 is diagnostic only. `score-position-stage.csv` gives
the paired result for every real context-position/stage cell.

The 2026-08-27 32K capture rejected WMMA64 as a production replacement:
WMMA128 used 9612.942 ms for 1260 score launches and WMMA64 used 9640.108 ms
(0.997x). WMMA64 won only 57 of 120 matched position/stage cells and was 0.995x
at the final 32256 position. The 1.003x trace-throughput difference was not
backed by reduced score work and is consistent with the preceding interleaved
uninstrumented A/B. Keep WMMA128; do not base subsequent indexer work on the
isolated WMMA64 event-timing result.

With WMMA128 retained, `cuda-sm75-indexer-f16-operands.sh` evaluates the next
mechanism independently: move the exact FP16 rounding already performed inside
every WMMA128 score tile into reusable operand tensors. The bounded harness
keeps the WMMA accumulation order and epilogue unchanged, requires bit-exact
scores and ordered top-512 selection, alternates paired timing order, and
includes one Q materialization in candidate end-to-end time. Full-history K
conversion is reported separately because a production sidecar would be
updated when each compressed row is written.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
TIMING_ROUNDS=5 \
TIMING_REPEATS=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-f16-operands.sh
```

Return `sm75-indexer-f16-operands-<timestamp>.tar.gz`. The candidate is still
diagnostic: no production storage or dispatch changes are made. At 256K, the
21 ratio-4 indexer layers would need 336 MiB of persistent FP16 K storage
(160 MiB in stage 0 and 176 MiB in stage 1); a live 512-token FP16 Q operand is
8 MiB on the executing tier. Promotion requires a subsequent full-model exact
frontier-logit and interleaved throughput A/B.

The standalone FP16-operand result showed that conversion/DRAM reduction by
itself is not material: its paired median was approximately 1.004x end to end.
`CANDIDATE=streaming64` therefore tests a separate SM75 mechanism. Each warp
loads its eight FP16 K fragments once and retains them across the ordered
64-head reduction; the kernel removes shared K and accumulator storage and
warp-broadcasts the 16 unique scalar weights. The runner records a failed
production-resource gate for local-memory traffic, more than 128 registers per
thread, or more than 16 KiB static shared memory—the envelope needed for at least
four 128-thread CTAs on Turing—but still collects exactness, timing, and Nsight
evidence so a compiler-resource failure is itself diagnosable.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

CANDIDATE=streaming64 \
PROFILE_GPU=0 \
TIMING_ROUNDS=5 \
TIMING_REPEATS=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-f16-operands.sh
```

Return `sm75-indexer-streaming64-<timestamp>.tar.gz`. The candidate remains
diagnostic and cannot affect the production graph.

The accepted systemic SM75 direction changes the persistent ratio-4 indexer
cache itself to F16 and makes streaming64 the default scorer. Compressor and
FP4-QAT arithmetic remain F32; each completed K row is rounded once when it is
committed. This matches the F16 operands that shipping WMMA prefill previously
created inside every score tile, eliminates the repeated K conversion, and
reduces the 21-layer 256K indexer cache from about 672 MiB to 336 MiB. Session
payloads remain F32 so checkpoint files are backend-independent. The diagnostic
controls are independent: `DS4_CUDA_NO_INDEXER_NATIVE_F16_CACHE=1` selects the
legacy F32 cache and WMMA128 scorer, while `DS4_CUDA_NO_INDEXER_STREAMING64=1`
keeps the native F16 cache but selects the F16-input WMMA128 scorer. Neither is
an automatic runtime fallback on a qualifying SM75 graph.

`cuda-sm75-indexer-native-controls.sh` isolates cache representation from score
kernel structure in one interleaved three-arm test:

1. legacy F32 cache + WMMA128;
2. native F16 cache + F16-input WMMA128;
3. native F16 cache + streaming WMMA64.

It preserves complete balanced dense-FP16 admission while suppressing partner
projection execution and attention row splitting, requires an unambiguous
dispatch audit for each arm, and byte-compares every frontier-logit artifact.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=16384 \
CTX_ALLOC=262273 \
REPEATS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-native-controls.sh
```

Return `sm75-indexer-native-controls-<timestamp>.tar.gz`.

After the cache/scorer controls are separated, use
`cuda-sm75-xdev-feature-isolation.sh` to reintroduce the two high-volume
cross-device mechanisms independently. It keeps the full 344/344 dense cache,
stage-aware 22/21 placement, automatic T32/T256/shared-down partner classes,
native F16 index cache, and streaming64 scorer fixed, then runs `neither`,
`partner-only`, `rows-only`, and `both` in a fixed safety order. Every arm
records per-GPU telemetry and a durable active-arm journal before launch;
successful arms must expose their requested runtime paths. Each new arm also
exports its dense-FP16 placement plan. Row splitting allocates partner-side
state before the stage-aware dense planner runs, so separately launched arms
can select different local/partner bindings even when all 344 weights remain
cached. The runner therefore reports NRMSE, maximum delta, top-1 agreement,
top-10 overlap, planned-partner counts, and exact placement-plan equality in
`production/logit-comparisons.csv`; it does not mislabel a cache-confounded
cross-process comparison as a row-split exactness test. The partner toggle is
also informational because native-Q8 fallback and FP16-expanded cuBLAS are
intentionally different arithmetic paths.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=32769 \
REPEATS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-xdev-feature-isolation.sh
```

Return `sm75-xdev-feature-isolation-<timestamp>.tar.gz`. If the host resets
before the archive is created, preserve the matching unarchived directory;
`active-arm.txt`, `run-journal.tsv`, and `telemetry/` identify the failing arm.
After a script or validator failure, a new invocation can reuse complete arms
without modifying the original evidence:

```bash
REUSE_XDEV_DIR="$PWD/sm75-xdev-feature-isolation-<prior-timestamp>" \
SKIP_BUILD=1 \
bash ./speed-bench/cuda-sm75-xdev-feature-isolation.sh
```

The model, prompt, topology order, stage split, context sweep, and fixed
indexer controls must match the prior manifest. Partially present arms are
rejected rather than silently rerun or mixed with new evidence.

`cuda-sm75-indexer-native-cache-production-ab.sh` is the indexer promotion
gate. It
tests aligned and 63-row-tail streaming tiles, compares the native one-token
F16-cache reader with the legacy F32 reader, runs the fixed 100-case production
quality suite, interleaves genuine 32K production runs under the exact 32K
frontier allocation, byte-compares their frontier logits, and finishes with a
separate 64K-allocation long-prompt one-word decode smoke. Both arms must prove
344/344 dense-F16 admission, balanced 43/43 T256 placement, and automatic
selection of both qualified attention row splits. Because both arms open the
same immutable GGUF, this gate does not re-audit or score the model's routed
quantization recipe; that is a separate model-production concern.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=32768 \
CTX_MAX=32768 \
CTX_ALLOC=32769 \
WORD_CTX_ALLOC=65537 \
REPEATS=3 \
RUN_QUALITY=1 \
RUN_WORD_SMOKE=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-native-cache-production-ab.sh
```

Return `sm75-indexer-native-cache-production-<timestamp>.tar.gz`. A mismatch
in any quality score, score bit, top-k ordering, frontier logit, or retrieval
answer fails the run before promotion evidence is accepted.

The accepted 32K result is 612.43 tokens/s for native F16/streaming64 versus
582.90 tokens/s for legacy F32/WMMA128, a 1.049768x paired-median speedup.
Every frontier logit was bit-identical and the two 100-case quality TSVs were
byte-identical. Native F16/streaming64 is therefore the production default on
qualifying SM75 graphs; the legacy arm remains only a diagnostic control.

If a prior run completed the legacy 100-case arm before a later validation
failure, extract its archive and pass the extracted `quality/` directory as
`REUSE_LEGACY_QUALITY_DIR`. The reused log must still prove its production
runtime, full dense cache, and T256 placement before the two TSVs are compared.

If both 100-case arms completed, pass that `quality/` directory as
`REUSE_QUALITY_DIR`. The gate copies and revalidates both TSV/log pairs, then
starts at the production A/B. Do not use `REUSE_QUALITY_DIR` together with
`REUSE_LEGACY_QUALITY_DIR`.

The 32K performance allocation is deliberate. A 256K allocation on the
current 4 x 48 GiB layout materializes only 271/344 dense-F16 candidates, so
it is a capacity/admission experiment rather than a valid full-cache
performance control.

If a completed score pass stops during one of the top-k Nsight captures, keep
the original directory and resume only the three top-k captures. Resume mode
requires the validated timing CSVs and all four score-kernel Nsight CSVs; it
will not silently reuse a partial score pass.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export INDEXER_AUDIT_DIR="$PWD/sm75-indexer-audit-<timestamp>"

PROFILE_GPU=0 \
RESUME=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=1 \
bash ./speed-bench/cuda-sm75-indexer-audit.sh
```

### SM75 32K attention query-row split experiment

`cuda-sm75-attention-rowsplit.sh` is a model-free, two-GPU experiment for the
measured 32K production attention shapes. It compares the shipping 512-row
kernel with concurrent 256/256 query-row execution across one NVLink pair. The
partner variant is measured both by directly peer-reading home Q/KV/top-k and
by using locally mirrored inputs. Both variants must reproduce non-zero
shipping-kernel output bit-for-bit before timings are reported.

This is intentionally not the older head split and not a split-KV reduction.
Every query row retains the same kernel and online-softmax operation order; the
harness only changes which GPU executes the row. No GGUF is opened.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
REPEATS=5 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-attention-rowsplit.sh
```

Return `sm75-attention-rowsplit-<timestamp>.tar.gz`. A production integration
is justified only if one of the exact candidates wins in both indexed and
mixed attention; the harness result does not include output-projection work or
the incremental cost of maintaining a partner KV mirror.

The harness also validates the production compressed-cache lifecycle.
`DS4_ROWSPLIT_CACHE_REPEATS` controls a separate sustained loop of
home-default-stream raw/F32-compressed KV production, ordered peer mirroring,
partner attention consumption, and result gathering (default: `REPEATS`,
maximum 4096). Use roughly 750 repetitions per physical pair to match the
number of per-layer 512-token cache updates through a 16K full-model prefill;
the final split output must remain bit-exact. This attention cache is distinct
from the SM75-native F16 indexer cache.

`cuda-sm75-attention-rowsplit-production-ab.sh` is the fail-closed production
promotion gate. It interleaves the unchanged home path with the mirrored-KV
query-row candidate at 2K, 4K, 8K, 16K, and 32K frontiers, validates the
bounded indexed and mixed harnesses on both configured NVLink pairs, requires
audited production dispatches of both target kernels, and compares the emitted
frontier logits byte-for-byte for every paired repeat. A requested
indexed/nonzero-mixed split that cannot dispatch aborts; it is never replaced
by peer-read or home-only execution. Initial static-mixed/raw-only work remains
explicitly outside this first production candidate and is identical in both
arms.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=262273 \
REPEATS=3 \
RUN_HARNESS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-attention-rowsplit-production-ab.sh
```

Return `sm75-attention-rowsplit-production-<timestamp>.tar.gz`.
After promotion, `cuda-sm75-long-prompt-word-smoke.sh` checks early user-visible
quality through the actual production CLI. It places a verification word before
a long public-domain distractor, requires a prompt of at least 24K tokens, and
greedily requests exactly that one word. No row-split enable override is set:
the log must prove that the qualified default dispatched both indexed and mixed
row-split attention on both stages without fallback. Cross-device synchronization
is not forced, so the smoke preserves normal production overlap.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_ALLOC=65537 \
EXPECTED_WORD=LANTERN \
PAD_LINES=2200 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-long-prompt-word-smoke.sh
```

Return `sm75-long-prompt-word-smoke-<timestamp>.tar.gz`.

# SM75 Q4-32/Q3A4 32K production profile

`cuda-sm75-q32-production-profile.sh` captures the next evidence pass for the
tagged Q4-32/Q3A4 model. It accepts only a genuine single 32768-token
production frontier with the fixed 22/21 stage split, stage-aware dense-F16 placement,
all 344 dense-F16 candidates resident, and the exact routed recipe used by
`DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf`: Q3A4 gate/up on layers
6,8,10,12,14,16,18,20,30,32,34,36,38,40,42, Q4-32 gate/up on the other 28
layers, and Q4-32 down on all 43 layers.

The GGUF is opened once under Nsight Systems. DS4 NVTX ranges are then reduced
to per-device stage, microbatch, layer, handoff, and partner-projection tables.
The trace uses the exact 32K frontier allocation and must prove complete
dense-F16 admission, stage-aware placement inside both logical 22/21 NVLink
pairs, fixed 50/50 attention and score/top-k indexer row splits on both pairs,
the native-F16 streaming64 indexer, and a saved 32K
frontier-logit artifact. The stage-aware planner uses only logical stage roles,
live per-device headroom, and projected dense work; it contains no physical GPU
ID or layer-parity policy. It is rejected for every placement other than the
fixed four-GPU 22/21 production layout.
Nsight Compute does not replay the full application: bounded exact harnesses
capture Q4-32 gate/up, Q4-32 down, Q3A4 gate/up, and both indexed and mixed
long-context attention shapes. This avoids five additional 139 GB model loads
and prevents application replay from perturbing the production trace.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PROFILE_TOKENS=32768 \
CTX_ALLOC=32769 \
PROFILE_GPU=0 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q32-production-profile.sh
```

Return `sm75-q32-production-profile-<timestamp>.tar.gz`. The primary results
are `profile-summary.md`, `stage-device-summary.csv`,
`stage-microbatch-device.csv`, `kernel-family-summary.csv`,
`kernel-family-total.csv`, `operation-family-summary.csv`, the genuine
`nsys/combined.nsys-rep`, and the five validated reports under `ncu/`.

Critical-path attribution uses an indexed NVTX interval lookup; a genuine 32K
trace that previously required more than an hour of single-core Python scanning
processes in seconds. If collection was interrupted after `combined.sqlite`
was written, reuse it without loading or tracing the model again:

```bash
PROFILE_DIR="$PWD/sm75-q32-production-profile-<timestamp>" \
RESUME_POSTPROCESS=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q32-production-profile.sh
```

Keep the same `MODEL`, `PROMPT`, `GPU_DEVICES`, and fixed production variables
from the original invocation. Resume validates the saved production artifacts,
reuses the SQLite export and Nsight Systems reports, regenerates attribution,
then continues with the bounded Nsight Compute captures.

# SM75 routed-quant real-weight quality sweep

After cuda-sm75-q3-q4-32.sh establishes bounded kernel correctness and
performance, run the real-weight/imatrix gate before adding a GGUF type or
production dispatch. Set HF_DIR to the DeepSeek V4 Flash snapshot and IMATRIX
to the routed-MoE calibration file, then run
speed-bench/cuda-sm75-routed-quant-quality-sweep.sh.

`QUALITY_PRESET=screen` (the default) samples layers 3,21,36, experts
0,127,255, all three routed parts, and 32 evenly spaced rows per tensor.
`QUALITY_PRESET=full` covers every routed layer 3 through 42 with the same
experts and rows. Override any dimension with QUALITY_LAYERS,
QUALITY_EXPERTS, QUALITY_PARTS, or QUALITY_ROWS.

The sweep reports gate (w1), up (w3), combined gate/up, and down (w2)
separately under identical expert-specific imatrix weights. It includes the
shipping IQ2_XXS gate/up control, Q4_K and Q3_K controls, SM75 Q3/Q4-32,
IQ3-32, affine Q3-32 at 3.375/3.50 bpw, fixed-quota Q3/Q4 at
3.53125/3.78125 bpw, affine Q4-32 at 4.375/4.4375 bpw, Q5-32 at 5.25 bpw,
and adaptive Q2/Q3 at 2.78125/3.03125 bpw. It emits per-role Pareto tables
and a role-aware recipe frontier that can choose gate/up and down formats
independently. IQ2 down and Q2_K remain intentionally outside this pass.
The tool does not hash or quantize a full model and does not enable a
production format.

# SM75 indexed-attention production dispatch

SM75 indexed attention uses eight heads in a 256-thread CTA by default.
`DS4_CUDA_INDEXED_HEADS8_SM75=0` is retained only as the exact-output and
performance control for the former sixteen-head, 512-thread dispatch. Per-head
row order and online-softmax arithmetic are unchanged.

Two fixed 22/21 full-production A/B passes with complete 344/344 dense-F16
admission and byte-identical frontier logits independently measured indexed8
gains of +1.07%/+1.33% and +1.57%/+1.31% at 8K/32K. The SM75 production
profiler therefore expects the indexed-attention kernel to use 256 threads.

The Q4-32-down stage8 and compact-tile16 experiments are abandoned. Although
their isolated harnesses improved, they regressed the full production path by
roughly 2% and 5.6-5.9%, respectively. Neither implementation, dispatch flag,
nor experimental runner is retained.

# SM75 production decode baseline

`cuda-sm75-decode-baseline.sh` establishes the canonical decode baseline for
the fixed SM75 production configuration. The full matrix is pp512/tg256,
pp512/tg512, pp2048/tg256, pp2048/tg512, pp4096/tg256, pp4096/tg512,
pp32768/tg256, and pp32768/tg512. Every case runs in a separate process with
`ctx_alloc=pp+tg+1`; short-prompt results therefore do not sacrifice dense-F16
cache coverage merely to reserve the 32K case's context buffers.

An untimed 512-token prefill materializes all 344 dense-F16 candidates before
each measured case. The runner rejects incomplete admission, a stage split
other than 22/21, device order other than 0,3,1,2, a missing direct peer route,
a prompt other than the fixed `promessi_sposi.txt` corpus, a non-SM75 routed
layout, or any required-path fallback.
Inherited `DS4_*` variables are removed, so experimental decode paths cannot
silently contaminate the production baseline. No model hash, Nsight capture,
GPU sampler, or logits dump is performed.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CASES=all \
REPEATS=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-baseline.sh
```

Later passes can select a stable subset without redefining the matrix, for
example `CASES=pp512-tg256,pp2048-tg512,pp32768-tg512`. To resume an interrupted
run, keep its original settings and set `RESUME=1` plus
`DECODE_BASELINE_DIR="$PWD/sm75-decode-baseline-<timestamp>"`. Valid completed
cases are reused; incomplete or invalid cases are rerun. Return the generated
`sm75-decode-baseline-<timestamp>.tar.gz`. The primary results are
`summary/summary.csv`, `summary/summary.md`, and `summary/samples.csv`.

# SM75 production decode evidence

`cuda-sm75-decode-evidence.sh` is the first attribution pass after the broad
decode baseline. It keeps the production model, device order, 22/21 split,
stage-aware fixed-22/21 dense-F16 placement, and complete 344/344 admission
fixed.

The throughput experiment isolates the implementation transition observed
between PP2048 and PP4096: PP4096 is run with the production 1024-compressed-row
indexing threshold and with a forced 4096-row dense threshold. Runs are paired
and order-reversed. These paths are not assumed to be semantically equivalent:
after the threshold, indexed attention changes the selected compressed-row
set. A separate semantic pass writes every full-vocabulary FP32 logit vector
for eight generated tokens and reports first divergence, top-1 agreement,
top-10 overlap, maximum absolute delta, and NRMSE. Its filesystem work is never
used as a throughput sample. Any proposal to change the production threshold
still requires a scored quality suite.

Nsight Systems captures begin after prefill and sixteen unprofiled decode
tokens. Only the next sixteen tokens are recorded. Opt-in NVTX ranges identify
the complete token, embedding, each layer and logical tier, attention and FFN
halves, and output head. They add no CUDA event, stream wait, or device
synchronization. The captured contexts are PP2048, PP4096, and PP32768, plus a
PP4096 forced-dense trace. Aggregate GPU work is reported separately from the
wall-clock GPU envelope because work on partner devices overlaps.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
AB_REPEATS=3 \
TRACE_CONTEXTS=2048,4096,32768 \
TRACE_SKIP=16 \
TRACE_TOKENS=16 \
RUN_AB=1 \
RUN_EXACT=1 \
RUN_NSYS=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-evidence.sh
```

To continue an interrupted run, keep every original setting and set
`RESUME=1` plus `DECODE_EVIDENCE_DIR` to its existing output directory.
Completed threshold samples are reused only after their CSV shape and every
production-path marker validate; missing or invalid samples are rerun.

Return `sm75-decode-evidence-<timestamp>.tar.gz`. The primary products are the
paired threshold result in `summary/summary.md`, semantic logit record under
`exact/`, the four bounded `.nsys-rep` files, and the operation, stage/device,
layer/device, kernel/device, and trace summaries under `summary/`.

# SM75 one-token decode pair experiments

`cuda-sm75-decode-pair-next.sh` tests the two high-value pair-parallel decode
mechanisms without changing the production default. The indexer candidate
partitions the production native-F16 compressed-row cache 50/50, gathers the
partner score half, and runs the unchanged deterministic top-k on the home
GPU. The attention candidate exercises DS4's existing, disabled one-token
32/32 head path with mirrored raw/compressed KV. Both must be bit-exact and
non-zero against the shipping one-GPU operation before timing begins.

The transfer-inclusive indexer timing includes the per-token index query and
64 head weights plus the score gather. The attention timing includes the
partner query heads, 512 selected indices, one raw KV update, one compressed
KV update, and the partner output gather. Persistent cache mirroring is
reported separately. `RUN_NCU=1` captures the index score half, unchanged
top-k, shipping indexed-attention kernel, and 32-head candidate with occupancy,
scheduler, warp-stall, compute-pipe, and memory-workload sections.

Set `RUN_PRODUCTION=1` to follow the mechanism checks with an order-reversed
fixed-production decode A/B of the existing `DS4_CUDA_TP_ATTN_HEADS` path.
Every production sample requires 22/21 placement, complete 344/344 dense-F16
admission, and the tagged SM75 Q32 model. Full-vocabulary logits from both
variants must be byte-identical.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
REPEATS=100 \
RUN_PRODUCTION=1 \
PRODUCTION_PP=32768 \
PRODUCTION_TG=256 \
PRODUCTION_REPEATS=3 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-pair-next.sh
```

Return `sm75-decode-pair-next-<timestamp>.tar.gz`. A fast first check can use
`RUN_PRODUCTION=0 RUN_NCU=0 REPEATS=25`; it still enforces exact operation
outputs, but does not claim an end-to-end result.

# SM75 production decode indexer-score A/B

`cuda-sm75-decode-indexer-production-ab.sh` advances only the successful part
of the one-token pair experiment. It partitions each native-F16 indexer score
vector 50/50 across the layer's NVLink pair, gathers the partner score half,
and runs the unchanged deterministic top-k on the home GPU. The candidate
reuses the index-cache mirror already required by production prefill row
splitting; it allocates no additional persistent cache. The rejected decode
attention head split is explicitly disabled in both variants.

Every sample requires the fixed 22/21 placement, stage-aware complete 344/344
dense-F16 admission, tagged SM75 Q32 tensors, direct pair routes, and the
production prefill indexer row split. Runs are paired and order-reversed.
Full-vocabulary FP32 logits for eight decode steps must be byte-identical before
the report is accepted.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CONTEXTS=2048,32768 \
TG_TOKENS=256 \
REPEATS=3 \
EXACT_CONTEXT=32768 \
EXACT_TOKENS=8 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-indexer-production-ab.sh
```

To resume, preserve the original settings and set `RESUME=1` plus
`DECODE_INDEXER_AB_DIR="$PWD/sm75-decode-indexer-production-<timestamp>"`.
Return the generated archive; the decision table is `summary/summary.md` and
the exact-output gate is `exact/verification.txt`.

# SM75 production Q32 decode CUDA Graph A/B

Before resuming a broad graph A/B after any lost-GPU event, use
`cuda-sm75-decode-crash-isolation.sh` to reproduce the graph-disabled 32K
control without six preceding benchmark processes. It runs TG16, TG64, and
TG256 in increasing order, with each case in a separate process. The production
row split, stage-aware 22/21 dense placement, partner execution, native-F16
indexer, and complete 344/344 dense cache remain enabled; owned routed-MoE CUDA
Graphs are explicitly disabled.

Every case persists `active-case.txt` and `run-journal.tsv` before launch,
captures per-GPU telemetry, exports the exact dense placement and binding
inventories, and requires healthy four-GPU snapshots before and after the
process. It never invokes `sudo`. If the host resets before the archive can be
created, return the matching unarchived directory.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PP_TOKENS=32768 \
TG_LEVELS=16,64,256 \
REPEATS=1 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-crash-isolation.sh
```

Only after this graph-disabled boundary is stable should the broader
`cuda-sm75-decode-graph-ab.sh` matrix be resumed.

`cuda-sm75-decode-graph-ab.sh` tests graph coverage for the routed kernels
that the production Q4-32/Q3A4 model actually executes. The candidate replaces
each owned expert half's six host submissions—Q8_K quantize, native activation
pack, fused Q4-32/Q3A4 gate/up, mid quantize, native activation pack, and
Q4-32 down—with one CUDA Graph launch. It deliberately leaves the cross-GPU
owned-slot combine outside the graph, preserving the established NVLink and
synchronization boundary.

Graph executables are cached per exact layer/tier binding after the untimed
warmup. Separate entries cover Q4-32 versus Q3A4, fused shared-Q8
prequantization versus ordinary Q8_K quantization, and six-slot home versus
fixed-three partner down output. The timed path therefore does not perform six
node-parameter updates per layer. If the bounded cache cannot represent a
binding, that call uses the unchanged launch path.

The CUDA smoke test compares the graph-owned home/partner result against the
independent full-expert production dispatcher for both gate formats and two
successive inputs. The production test then runs paired, order-reversed decode
A/B samples at PP512/2048/4096/32768 and requires byte-identical full-vocabulary
logits before reporting a speedup.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CONTEXTS=512,2048,4096,32768 \
TG_TOKENS=256 \
REPEATS=3 \
EXACT_CONTEXT=4096 \
EXACT_TOKENS=16 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-graph-ab.sh
```

To resume, preserve the original settings and set `RESUME=1` plus
`DECODE_GRAPH_AB_DIR="$PWD/sm75-decode-graph-ab-<timestamp>"`. Return the
generated archive. The decision table is `summary/summary.md`; exactness is
recorded in `exact/verification.txt`.
# SM75 decode packed-weight profiling

`cuda-sm75-decode-weight-profile.sh` profiles the actual one-token decode
dispatch for routed Q4-32/Q3A4 gate/up, the distinct packed-Q8 projection
shapes seen in the 32K production trace, and all observed ordered F16
compressor-pair widths.  It opens no GGUF and validates exact-zero outputs from
synthetic zero weights before collecting one precisely filtered kernel per
scenario.  The default `NCU_CACHE_CONTROL=all` represents cold per-layer
weight streams; use `none` only for an explicitly replay-warm L2 comparison.

```bash
PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-decode-weight-profile.sh
```

The resulting `summary.csv` and `summary.md` report achieved DRAM bandwidth,
DRAM peak utilization, L2 hit/throughput, global-load efficiency,
sectors/request, long-scoreboard and MIO stalls, and achieved occupancy.  A
subset can be captured with `PROFILE_SET=routed`, `q8`, or `f16`, or with an
explicit comma-separated `SCENARIOS` list.

### Bounded SM75 P2P direction audit

`cuda-sm75-p2p-direction-audit.sh` escalates a production-shaped partner-FP16
projection through explicit traffic levels without loading a full model.  The
first device in each pair is the logical home and the second is the execution
partner, so the large result travels from the second device back to the first.
Each level runs in a fresh process and leaves a durable `.started` marker if the
host disappears.  The default T32 shape transfers 2 MiB of activation and 64
MiB of result per call at 512 tokens.

```bash
PAIRS="1,0 0,1" \
SCENARIO=t32 \
TOKENS=512 \
REPEAT_LEVELS=1,64,256,512,1536 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-p2p-direction-audit.sh
```
