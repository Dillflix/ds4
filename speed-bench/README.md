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
128 registers, requires production API and auxiliary-output exactness, records
separate dynamic-K/fixed-K SASS summaries, times all three variants against the
captured early/late production histograms, and records bounded Nsight Compute
reports. That bounded Gate-only screen is the default (`RUN_FULL_MODEL=0`).
Model hashing and conversion are deliberately absent.

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
16-route native-Q8 activation tile is 74.75 KiB and cannot fit in Turing's
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
