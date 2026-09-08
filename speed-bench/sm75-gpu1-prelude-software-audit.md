# GPU1-only canonical reproducer: prelude memory audit

Reviewed source: `039eef3e756412da0d2aec02ae0af835b2a09caf`.
Scope: `output-b-production103-no-row-owned`, including initialization, q_b,
RMS/RoPE, static mixed attention, inverse RoPE, output A/B, and the scratch
history carried into the failing B-only suffix. This extends
`sm75-gpu1-memory-ordering-audit.md`; it does not replace the unchanged failure
reproducer with a reduced workload.

Result: this bounded source review did not identify a reachable out-of-bounds,
uninitialized-input, overlapping-output, or escaped-temporary defect in these
fixed-shape application kernels. That is not a proof of runtime correctness,
and does not clear cuBLAS, compiled code, driver memory management, hardware,
or an earlier valid-address overwrite. No GPU workload, rebuild, production
change, or new synchronization was performed for this audit.

All `file:line` anchors below refer to the reviewed revision, not future edits.

## Exact scope and allocation accounting

The runner removes inherited `DS4_*` variables and `CUDA_LAUNCH_BLOCKING`, then
supplies its own diagnostic selectors (`speed-bench/cuda-sm75-token-row-arithmetic.sh:176`).
The fixture sets the copied-model, canonical FP16 cache, disabled FP32 cache,
fused q_b, and no-TF32 settings before initialization
(`tests/cuda_sm75_token_row_arithmetic.c:1363`). Native projections and peer
placement are separate branches, not requirements of this reproducer.

Constants are N=512, half N=256, input=1024, 64 heads of 512, eight groups of
4096, A rank=1024 per group, low=8192, output=4096, compressed rows=128,
raw window=128 and compression ratio=4 (`tests/cuda_sm75_token_row_arithmetic.c:14`).
Neither multiplication into these shapes nor the launch dimensions approach
the integer limits used by the reviewed kernels.

The 13 independently allocated device tensors are declared and populated at
`tests/cuda_sm75_token_row_arithmetic.c:1541`:

| Tensor allocation | Bytes per allocation | Count |
| --- | ---: | ---: |
| Input | 2,097,152 | 1 |
| FP32 query, full/split | 67,108,864 | 2 |
| FP16 query, full/split | 33,554,432 | 2 |
| FP32 attention heads, full/split | 67,108,864 | 2 |
| FP32 low-rank, full/split | 16,777,216 | 2 |
| FP32 output, full/split | 8,388,608 | 2 |
| Raw KV | 1,048,576 | 1 |
| Compressed KV | 262,144 | 1 |

Their sum is **389,283,840 bytes (371.25 MiB)**. The logged
`device_working_set_bytes` is this tensor sum, not the complete CUDA memory
footprint. It excludes the 106,955,008-byte canonical model copy (102 MiB plus
256 bytes of sinks), 201,326,592 bytes (192 MiB) of FP16 weight cache, and
up to 50,331,648 bytes (48 MiB) of application scratch. Those source-sized
allocations total 747,897,088 bytes before driver/cuBLAS/runtime allocations,
allocator overhead, or any temporary initialization allocations. BAR1 size is
not used as a capacity test in this arithmetic accounting.

Every half-view begins at half its parent's byte count and has exactly that
length (`tests/cuda_sm75_token_row_arithmetic.c:1570`). All offsets are divisible
by 256, so the views add no offset-alignment problem. Full-reference views
point into the full parent; split-output views point into the split parent.
The view constructor checks subtraction-based range containment and marks
views non-owning (`ds4_cuda.cu:6445`). Views are destroyed before their parents,
and no parent is freed until the fixture's final cleanup
(`tests/cuda_sm75_token_row_arithmetic.c:2415`).

Host comparison slabs are also deliberately shared, not overlapping live
snapshots accidentally. Each has 16,777,216 float slots. The actual-A snapshot
occupies [4,194,304, 8,388,608), combined output occupies
[8,388,608, 10,485,760), and shipping-B snapshot occupies
[10,485,760, 12,582,912). They fit below the slab end; subsequent low/output
readbacks use only the initial region (`tests/cuda_sm75_token_row_arithmetic.c:1324`).

## Weight initialization and lifetime

Each of q_b, A and B contains 33,554,432 dequantized weights, requiring exactly
35,651,584 Q8 bytes and 67,108,864 FP16 bytes. The model layout has adjacent,
non-overlapping q_b, sink, A and B ranges
(`tests/cuda_sm75_token_row_arithmetic.c:1212`). All model bytes and input/KV
elements are initialized before installation or tensor writes
(`tests/cuda_sm75_token_row_arithmetic.c:1332`). All three canonical FP16 ranges
are installed before any prelude call (`tests/cuda_sm75_token_row_arithmetic.c:1419`).

`dequant_q8_0_to_f16_kernel` maps `gid` to `(row, block, lane)`, guards
`gid < in_dim*out_dim`, reads two scale bytes plus one of 32 quantized bytes,
and writes exactly one output element (`ds4_cuda.cu:12361`). Here all input
dimensions are exact multiples of 32; the last source byte is inside its
34-byte block. The fixture's scales and integer values are finite and
deterministically initialized.

Each cache range owns a separate `cudaMalloc` allocation
(`ds4_cuda.cu:3244`), and successful metadata registration stores that pointer
(`ds4_cuda.cu:3302`). Metadata-vector growth does not relocate those independent
device allocations. A resident lookup validates model identity, byte size,
dimensions and device before returning it (`ds4_cuda.cu:3061`, `ds4_cuda.cu:3125`).
Consequently this successful canonical path is not the borrowed native-Q8
view-rebasing path fixed in earlier native-stream diagnostics. A cache-disable
or failed-fill path is a different execution path and must be identified in
runtime evidence rather than silently assumed equivalent.

## q_b and fused RMS/RoPE

The three external-q_half calls are full512, first256 and second256. Their
FP16 GEMM outputs fill the corresponding full or half FP16 tensor; RMS/RoPE
then writes separate FP32 query storage. The two additional half calls reuse
internal FP16 q_half scratch, not a pointer into either persistent query
tensor (`tests/cuda_sm75_token_row_arithmetic.c:1685`,
`tests/cuda_sm75_token_row_arithmetic.c:1709`).

The wrapper checks tensor sizes, dimensions, device consistency, weight range,
and head limits (`ds4_cuda.cu:42220`). Its q_b GEMM is
`m=32768, n=512/256, k=1024`, `lda=ldb=1024`, `ldc=32768`, with beta=0
(`ds4_cuda.cu:42391`). The output matrix exactly fits the supplied FP16 extent.
The activation conversion is one guarded write per FP16 input element
(`ds4_cuda.cu:9717`).

RMS/RoPE launches one 256-thread block per token/head. Its 256-element shared
reduction buffer is completely initialized, and every reduction step ends
with a block barrier. For head_dim512/n_rot64, columns [0,448) are written once
by the non-rotary loop; 32 threads write disjoint pairs covering [448,512).
Both branches read FP16 q_half and write FP32 output, so these writes cannot
overwrite their own input (`ds4_cuda.cu:13097`). The second-half `pos0=256`
changes position arithmetic while indexing its local tensor rows from zero.

## Static mixed attention and inverse RoPE

With the fixture's 512 tokens, 512 head dimension, no quality override and
no mask, the shipping wrapper selects the online window kernel
(`ds4_cuda.cu:26353`). It has no attention-score scratch allocation. The
range wrapper validates `q_row0+n_q <= n_tokens` and full KV backing sizes
(`ds4_cuda.cu:42535`). The full and half-range calls use identical q_full and
full raw/compact KV inputs (`tests/cuda_sm75_token_row_arithmetic.c:1803`).

For every global token t in [0,511], the raw reads span
`max(0,t+1-128)..t`, and compact reads span
`0..min(128,floor((t+1)/4))-1`. No raw index exceeds 511 and no compressed
index exceeds 127. A range call uses its local token index for q/output and
global token index only for causal cache bounds; there is no double-added
256-row pointer offset (`ds4_cuda.cu:16700`, `ds4_cuda.cu:16833`).

Each block has eight warps, one per head, and an 8 KiB shared KV tile of
512 float4 values. For each 1-4-row tile, cooperative threads initialize
exactly `nr*128` float4 slots, barrier, consume those initialized slots,
then barrier again before overwriting them. Branches around head work are
warp-uniform; both block barriers are outside that branch. The configured
64 heads make all eight warps valid in all eight head groups. The four
float4 writes per lane uniquely cover each 512-float head output. Sink
indices are limited to [0,63]. This establishes application indexing and
barrier structure for the fixed shape, not cuBLAS or hardware correctness.

Inverse RoPE first resets both parent head tensors from identical synchronized
host data (`tests/cuda_sm75_token_row_arithmetic.c:1845`). The in-place kernel
assigns one thread to each disjoint rotary pair. Each thread reads both old
values before writing them; no other thread uses that pair. Non-rotary
columns retain initialized input. The full launch has 1,048,576 pairs and a
half launch 524,288, so the wrapper's 32-bit pair count does not overflow here
(`ds4_cuda.cu:13183`, `ds4_cuda.cu:24743`). General oversized-shape admission is
not qualified by this result.

## Output A and the scratch history leading into B

Output A packs all `8*N*4096` head values from token-major FP32 to group-major
FP16 with a guarded, one-to-one permutation (`ds4_cuda.cu:13817`). Its
strided-batched GEMM uses eight batches, `m=1024,n=N,k=4096`, weight stride
4,194,304 FP16 values, activation stride `N*4096`, output stride `N*1024`,
and beta=0. The final batch's last value is the last allocated matrix value,
not beyond it (`ds4_cuda.cu:26685`). The unpack kernel is the inverse
group/token permutation into the separate persistent low tensor, again
one unique writer per value (`ds4_cuda.cu:13833`).

The single-device scratch accessor is a grow-only ordinary malloc/free arena
(`ds4_cuda.cu:1729`, `ds4_cuda.cu:1766`). For the successful path reviewed here:

| Operation | Requested scratch bytes | Layout inside scratch |
| --- | ---: | --- |
| External-q_half q_b, N512 | 1,048,576 | Input FP16 |
| External-q_half q_b, N256 | 524,288 | Input FP16 |
| Internal-q_half q_b, N256 | 17,301,504 | 16 MiB q_half, then 0.5 MiB input |
| Online attention / inverse RoPE | 0 | No arena use |
| Output A, N512 | 50,331,648 | 32 MiB packed heads, then 16 MiB packed low |
| Output A, N256 | 25,165,824 | 16 MiB packed heads, then 8 MiB packed low |
| Output B, N512 | 8,388,608 | Converted low input at offset zero |
| Output B, N256 | 4,194,304 | Converted low input at offset zero |

The ordinary q_b phase is synchronized before the internal-q_half growth;
the internal scratch phase is synchronized before attention and the later
output-A growth (`tests/cuda_sm75_token_row_arithmetic.c:1691`,
`tests/cuda_sm75_token_row_arithmetic.c:1713`). No scratch pointer is stored in
a persistent fixture tensor. The local stack q_half descriptor only exists
for its helper call (`ds4_cuda.cu:42285`).

After full512 A allocates 48 MiB, all subsequent A/B requests fit. B's
conversion reuses the old packed-head region, not persistent `low`, and
unpack's source starts at 32 MiB for full A or 16 MiB for half A. Even those
source regions are disjoint from the following 8/4 MiB B conversion region.
The low production/consumption still needs runtime stream ordering; disjoint
scratch alone does not establish it. All reviewed conversion, pack, unpack,
RMS, attention and RoPE launches omit an explicit stream. The backend creates
cuBLAS handles and has no `cublasSetStream` call in the reviewed source.
Runtime handle/stream identity remains an instrumentation target, not an
assumption that should replace measurement.

The final DEFAULT512-to-103-half256 B transition therefore neither needs a
larger allocation nor contains A pack/GEMM/unpack. A global-memory race in
that interval must be supported by actual stream/pointer evidence; adding
another fence between calls already separated by device synchronization
does not demonstrate a missing dependency.

## CPU-only index checks performed during this review

Using the literal index formulas above, a CPU enumeration checked every
pack and unpack element at N512 and N256 with a visited bitmap. All 31,457,280
mapped elements were in range and unique. A per-head write-count check
confirmed every one of the 512 RMS/RoPE output columns is written exactly
once. Enumerating all 512 causal rows checked 90,048 logical raw/compact
row references and every shared-tile initialization slot; all were in range
and every consumed tile slot had one producer.

These are independent arithmetic checks of the formulas, not execution of
compiled CUDA, floating-point validation, a memory sanitizer, or a test of
undocumented library accesses. They do not consume a GPU or alter the
reproducer binary. Tensor, model, cache and scratch byte counts in the tables
were independently recomputed in the same check.

Durable stdlib-only verification is in `tests/test_sm75_arithmetic_layout.py`:

```bash
python3 tests/test_sm75_arithmetic_layout.py
python3 tests/test_sm75_arithmetic_layout.py --exhaustive
```

The default suite checks complete token/group block permutations and their
affine within-block bounds/inverses, which avoids allocating a full tensor
bitmap. It also checks all parent/subview sizes, host snapshots, weight
source extents, q_b/A/B matrix extents, RMS reduction/column coverage, causal
rows, shared-tile initialization, inverse-RoPE pair bounds, and source-modeled
arena growth. `--exhaustive` additionally repeats the full 31,457,280-element
permutation enumeration with at most a 16 MiB bitmap. Both modes remain
CPU-only. The default run passed 14 tests with the one exhaustive test
explicitly skipped; the exhaustive run passed all 15 tests. These are fixed
reviewed formulas, not an automatic extraction or proof of future source
revisions; audit them again when the corresponding kernel formulas change.

## Open software avenues and discriminating evidence

1. Capture actual CUDA allocation address/size/generation, free events, tensor
   view containment, cache binding pointer/device, and cuBLAS stream/pointer
   mode. Preserve this reproducer's full prelude and suffix. Test whether
   observed addresses match the source-sized ownership model, including
   allocations made internally by cuBLAS. A separate instrumented binary or
   interposer must be identified as such; it must not replace the pinned
   unchanged executable unnoticed.
2. Capture API/kernel submission timeline, actual kernel names, submitting
   threads, streams, copy completion, and event dependencies. All application
   launches reviewed are ordinary default-stream launches; unknown library
   implementation details are not closed by searching source for explicit
   streams. No peer is required for this investigation.
3. Audit the emitted launch arguments against these exact dimensions and
   strides. Distinguish a wrong pointer/size from a correct argument followed
   by library or device failure. Binary/library hashes preserve provenance
   but do not provide argument traces or prove ABI correctness.
4. Keep earlier valid-address overwrite and library workspace lifetime open.
   Instrumented clean runs do not rule out timing-dependent corruption.
   Allocation guard/canary padding changes layout; full matrix checks cannot
   observe every dead or unused byte. Any new instrumentation must state the
   specific missing evidence it collects and its perturbation.
5. Retain the existing error-reporting weakness: the single-device arena
   ignores growth-time `cudaFree` status. That could obscure an earlier CUDA
   error, but its growth sites precede successful checkpoints and the final
   full-to-half B transition does not grow it. It is not an established
   explanation for this fault and was not changed during preservation.

This review narrows specific application-boundary hypotheses; it does not
claim every software avenue is exhausted or authorize further risky runs.
Driver/hardware investigation and bounded software evidence collection remain
parallel avenues. An identified PCIe link loss does not by itself prove that
software cannot initiate or expose the failure.
