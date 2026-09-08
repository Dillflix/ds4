# GPU1-only memory and ordering audit

Scope: the canonical FP16 `output-b-production103-no-row-owned` reproducer,
reviewed against source at `251acae`. No peer, native-stream, production enablement,
CUDA source change or rebuilt diagnostic is part of this audit update.

The uninstrumented 20260907T175457Z run failed at the final algorithm-103 half0
synchronization after the DEFAULT-512 call completed. Memcheck (183344Z),
initcheck (191251Z) and synccheck (192738Z) subsequently completed the full sequence
on executable SHA-256
`5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
These results do not identify the root cause or qualify uninstrumented execution.
The failing synchronization is an error-observation boundary, not identification
of the defective instruction. The subsequent racecheck run (195635Z) timed out
at 560/1024 calls and is not a completed clean result. A separate GPU2 Xid79 in
that run does not qualify GPU1. Do not repeat the run merely to complete a list.

## September 8 investigation update

The same executable failed again in `20260908T003609Z`, with DCGM disabled and
host-side capture enabled. GPU1 root-port Surprise Down/Fatal Error status became
set; GPU1 Xid79 preceded GPU0's GSP timeout by about ten seconds. The final B-only
half0 checkpoint still detects the application failure. This does not identify
the causal instruction or clear software. See the dated
[driver/hardware handoff](sm75-gpu1-driver-hardware-handoff-20260908.md),
[full prelude source audit](sm75-gpu1-prelude-software-audit.md), and
[runtime/software investigation](sm75-gpu1-runtime-software-investigation.md).

The prelude audit extends the original tail review below, checks fixed-shape
indexing and allocation accounting, and finds no demonstrated reachable
application bounds/alias/lifetime defect. Actual streams, handles, allocation
generations, compiled code and library internals remain unverified. The new
offline archive analyzer and CPU checks do not launch CUDA or qualify stability.

## Source-backed findings in the output-B tail

- `tests/cuda_sm75_token_row_arithmetic.c` allocates `low_full`, `out_full`, and
  `out_split` separately. The low-input half views occupy disjoint 8 MiB ranges
  of the 16 MiB `low_full`; the output half views occupy disjoint 4 MiB ranges
  of the 8 MiB `out_split`. They are views, not replacements for the parent
  allocations. Parents survive through the final comparisons and cleanup.
- The same fixture installs the three canonical FP16 weight ranges before the
  prelude. In `ds4_cuda.cu`, `cuda_q8_f16_ptr_impl` returns a resident range on a
  cache hit; this is not the native-Q8 borrowed-view relocation path fixed earlier.
- `cuda_matmul_q8_0_tensor_labeled_algo` converts the input into FP16 scratch,
  then passes that scratch to `cublasGemmEx`. Canonical output-B uses offset zero
  in `cuda_tmp_alloc_on`, which delegates to the single-device `g_cuda_tmp`
  arena. N=512 needs 8 MiB; N=256 needs 4 MiB. The allocator grows but does not
  shrink. Consequently the final full-to-half transition does not itself need
  to relocate scratch. Its activation kernel writes one element per in-range
  thread (`f32_to_f16_kernel`). The GEMM writes the separate output with beta=0.
- That conversion launch specifies no stream. `cuda_cublas_for_tier` returns
  the single GPU's handle created by `cublasCreate`; no `cublasSetStream` appears
  in this backend. The Makefile's default NVCC flags do not select per-thread
  default streams. Initialization also creates a device stream, but this B
  path does not submit its conversion or GEMM to that stream. This supports
  default-stream ordering for these calls; runtime verification is still needed.
  The documented cuBLAS default is the NULL stream when none is set.
  [cuBLAS stream documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream)
- Burn-in submits 1,024 half calls, synchronizing every 10 calls and at the end.
  After the structured input reset, DEFAULT-512, algorithm-103 half0 and half1
  are each followed by `ds4_gpu_synchronize`, which calls
  `cudaDeviceSynchronize`. Tensor writes/reads use `cudaMemcpy`, not their Async
  counterparts. A blanket new fence between those final calls would repeat an
  existing ordering boundary rather than test a demonstrated missing dependency.
- The reviewed backend uses ordinary `cudaMalloc`/`cudaFree`, not
  `cudaMallocAsync`/`cudaFreeAsync`. The single-device scratch allocator ignores
  its growth-time `cudaFree` result, an error-reporting weakness worth tracking.
  No growth is required by the final 512-to-256 transition, so that weakness is
  not established as its cause. No behavior is changed under this binary audit.

This is a bounded source audit of the tail, not proof that the entire prelude,
all global-memory accesses, cuBLAS internals, or driver/hardware are correct.
An earlier in-bounds overwrite can corrupt valid storage without being a
missing stream wait or an allocation-out-of-bounds error.

## Global-memory and cross-stream next steps

1. Preserve the exact executable, selected algorithm sequence, dimensions, data
   reset, working set and call counts. Compare against the GPU1-only failure,
   not a new peer workload or a reduced algorithm sweep.
2. Capture an API/kernel timeline for that same local sequence, including stream
   IDs, submitting host threads, copies, events/waits, allocations/frees and the
   cuBLAS launches. Verify that scratch conversion, GEMM consumption and next
   scratch overwrite are ordered on the actual runtime stream. Check preceding
   operations too, not just the synchronization where the fault surfaces.
   Tool availability and supported capture options must be checked before a
   separate bounded run; no trace has been collected in this update. Profiling
   also perturbs execution and a passing trace does not clear the failure.
3. Where the trace lacks pointer arguments, add separately gated diagnostics
   for allocation base/size/generation, tensor view offset/length, weight binding,
   `cublasGetStream`, and scratch producer/consumer identity. Verify ranges remain
   live and disjoint through their last consumers. Such executable changes must
   be explicitly distinguished from the current unchanged-binary comparison;
   logging is not a universal global-race detector. Extend the ownership review
   through prelude kernels for overlapping writes and incomplete initialization.
4. Investigate any observed missing dependency or lifetime/alias violation with
   a narrowly scoped check, not an indiscriminate synchronization change. Padded
   memcheck allocations can additionally expose writes crossing into neighboring
   allocations, but change allocation layout and cannot detect every valid-address
   race. Memcheck's `--track-stream-ordered-races all` concerns async-allocation
   lifetimes, not arbitrary global-memory data races; it is not automatically a
   new test of these ordinary allocation paths. Check runtime/library allocation
   behavior before treating that option as relevant additional coverage.
   [NVIDIA memory-check coverage](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#stream-ordered-race-detection)

Racecheck addresses shared-memory hazards; synccheck addresses supported kernel
synchronization primitives. Neither closes the global-memory investigation.
No peer GPU is needed to investigate multiple streams or allocation lifetime.
No clean diagnostic alone authorizes retrying the all-pairs production candidate.
