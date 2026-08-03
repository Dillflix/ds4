## Benchmarking

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
duration. `PROFILE_SET=remaining` captures only the previously missing Q2_K
down and Q8 T32/T256 reports. `NCU_SET=targeted` or `NCU_SET=full` are slower
opt-ins.

### Comprehensive stock-Q2 and remaining-kernel pass

`cuda-sm75-comprehensive-audit.sh` combines the full stock-Q2 production pass
with the bounded missing-kernel captures. The production half runs the fixed
prompt suite, records routed tile occupancy and every Q8-to-F16 cache decision,
and generates an Nsight Systems kernel-time distribution. The NCU half opens no
GGUF and profiles early/late Q2_K down plus the Q8 T32/T256 shapes. Partial
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
