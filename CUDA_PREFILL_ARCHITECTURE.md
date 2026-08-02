# DeepSeek V4 Flash CUDA Prefill Architecture Audit

This document records the current four-GPU CUDA prefill path before another
optimization is attempted. Statements under **Established** follow directly
from model validation, placement, and dispatch code. Statements under
**Unmeasured** require a production-path trace; they are not optimization
claims.

## Established model shape

The Flash variant is fixed at 43 transformer layers, embedding width 4096,
four HC streams, 64 attention heads, head/value width 512, eight attention
output groups, Q rank 1024, output rank 1024, 256 routed experts, six selected
experts per token, and expert hidden width 2048.

Each layer's important matrix shapes are:

| Path | Matrix shape (input x output) | Stored type in the hybrid model |
|---|---:|---|
| Q A | 4096 x 1024 | Q8_0 |
| Q B | 1024 x 32768 | Q8_0 |
| KV | 4096 x 512 | Q8_0 |
| Attention output A | eight groups of 4096 x 1024 | Q8_0 |
| Attention output B | 8192 x 4096 | Q8_0 |
| Router | 4096 x 256 | F16 |
| Routed gate, per expert | 4096 x 2048 | IQ2_XXS |
| Routed up, per expert | 4096 x 2048 | IQ2_XXS |
| Routed down, per expert | 2048 x 4096 | Q4_K |
| Shared gate/up | 4096 x 2048 | Q8_0 |
| Shared down | 2048 x 4096 | Q8_0 |

The full attention-head output is 64 x 512 = 32768 floats per token. Output A
reduces each eight-head group (4096 floats) to rank 1024, producing 8 x 1024 =
8192 floats per token before output B. Consequently, half of that low-rank
matrix is 4096 floats per token--exactly the final output width. The failed
head-split path did not reduce transfer bytes by gathering low-rank halves
instead of projected-output halves.

## Established hybrid weight size

IQ2_XXS uses 66 bytes per 256 values and Q4_K uses 144 bytes per 256 values.
For one expert:

- Gate: `2048 * (4096 / 256) * 66 = 2,162,688` bytes.
- Up: `2048 * (4096 / 256) * 66 = 2,162,688` bytes.
- Down: `4096 * (2048 / 256) * 144 = 4,718,592` bytes.
- Total: 9,043,968 bytes = 8.625 MiB.

All 256 experts occupy 2.15625 GiB per layer and 92.71875 GiB across 43
layers. This is why expert ownership dominates residency and why each NVLink
partner owns half of every routed tensor.

## Established four-GPU ownership

With `--gpu-devices 0,2,1,3`, logical tiers and physical GPUs are:

| Logical tier | Physical GPU | Role |
|---:|---:|---|
| 0 | 0 | Pipeline stage 0 home, layers 0-20 and embedding |
| 1 | 2 | Pipeline stage 1 home, layers 21-42 and output head |
| 2 | 1 | Expert partner of logical tier 0 over NVLink |
| 3 | 3 | Expert partner of logical tier 1 over NVLink |

The placement creates two pipeline stages, not four. For every home layer:

- Experts 0-127 reside on and execute on the home GPU.
- Experts 128-255 reside on and execute on the NVLink partner.
- Non-routed layer weights are resident on both GPUs in the pair.
- In the established prefill path, attention, router, shared expert, and HC
  transforms execute on the home GPU. The partner participates in routed MoE.
- The vocabulary output is sharded across all four tiers.

## Established 512-row layer dataflow

The default 2048-token prefill chunk is divided into four 512-row
microbatches. Within each layer the home GPU computes attention, FFN HC/norm,
router logits, and top-six selection. Routed MoE then performs:

1. Home to partner: 512 x 4096 FP32 normalized activations (8 MiB), 512 x 6
   I32 selected IDs (12 KiB), and 512 x 6 FP32 weights (12 KiB).
2. The partner enqueues its owned-expert work; the host switches to the home
   tier and enqueues the home-owned work. These are separate device streams and
   are intended to overlap.
3. Partner to home: 512 x 4096 FP32 partial routed output (8 MiB).
4. The home sums both routed halves, executes the shared expert, and performs
   the final HC update.

The fixed expert-parallel payload is therefore 16.0234375 MiB per layer and
689.0078125 MiB across all 43 layers for one full microbatch. Four full
microbatches move 2.691436767578125 GiB of application payload inside the two
NVLink pairs. This excludes event/protocol traffic and does not by itself
measure exposed communication time.

At the pipeline boundary, the activation contains four HC streams, not one
4096-wide row. Each full microbatch copies
`512 * 4 * 4096 * 4 = 33,554,432` bytes (32 MiB) from physical GPU 0 to physical
GPU 2. A 2K chunk therefore carries 128 MiB across that non-NVLink, cross-NUMA
stage boundary.

## Established pipeline schedule

For four 512-row microbatches and two stages, prefill uses five waves:

| Wave | Stage 1 (physical GPU 2 + 3) | Stage 0 (physical GPU 0 + 1) |
|---:|---|---|
| 0 | - | microbatch 0 |
| 1 | microbatch 0 | microbatch 1 |
| 2 | microbatch 1 | microbatch 2 |
| 3 | microbatch 2 | microbatch 3 |
| 4 | microbatch 3 | - |

Work is enqueued on each device's default stream. Cross-device ordered copies
and CUDA events enforce expert-return and stage-boundary dependencies. The
pipeline synchronizes once after all five waves in the ordinary path.

The Q8-to-F16 expansion cache is enabled by default and bounded by the CUDA VRAM
reserve. The experimental attention-head split is disabled by default because
it measured about 20% slower across 2K-65K on the target system.

## Why previous internal profiles are not evidence

`DS4_METAL_LAYER_STAGE_PROFILE` ends commands at every named boundary. On CUDA,
ending commands synchronizes the active device, and enabling this profile also
prevents entry into the normal stage-major pipeline. Its measurements describe
a different schedule.

`DS4_CUDA_MOE_PROFILE` and `DS4_CUDA_ATTN_OUTPUT_PROFILE` record CUDA events and
synchronize the final event for every invocation. They are useful for isolated
kernel development but perturb cross-device and cross-stage overlap. They must
not be used to identify the production pipeline's critical path.

## Unmeasured questions required before optimization

The production-path Nsight trace must answer all of the following:

1. Actual duration of every kernel family on each physical GPU, including the
   SM75 IQ2 gate/up, Q4 down, dense Q8, exact attention, HC, router, and copies.
2. Per-wave stage duration and whether the 21/22-layer placement is balanced by
   time rather than bytes.
3. Home-versus-partner routed-MoE duration and the exposed wait before reduction.
4. Whether the 8 MiB expert transfers overlap expert computation, and whether
   the 32 MiB stage transfer overlaps useful work on other devices.
5. Kernel-launch/API gaps, default-stream serialization, event waits, and CPU
   scheduling gaps.
6. SM occupancy, tensor-core issue, memory bandwidth, and cache behavior for
   the kernels on the critical path. Nsight Systems identifies those kernels;
   only then should selected kernels be examined with Nsight Compute.
7. Whether the first measured prefill includes one-time Q8 expansion-cache work
   and whether subsequent in-process prefills have the same kernel mix.

No kernel rewrite, sharding change, or throughput estimate is justified until
these questions are answered against the exact target hardware and model.

Run `speed-bench/cuda-prefill-audit.sh` to collect the uninstrumented baseline
and capture-range-limited Nsight Systems evidence without changing the command
stream under measurement.
