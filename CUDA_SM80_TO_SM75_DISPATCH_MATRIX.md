# DeepSeek V4 Flash CUDA SM80 to SM75 dispatch and resource matrix

This is the architecture audit for the DeepSeek V4 Flash CUDA path at commit
`97201903f4e996763c80156f1458c3552ad38417`. It is the prerequisite for any
further kernel or quantization proposal.

This is a current-state inventory, not a list of newly discovered capabilities.
In particular, the dense-Q8, IQ2 gate/up, Q4-down, and exact-attention SM75
paths were added by the current SM75 initiative immediately before this audit.
Every architecture-specific row must therefore be read together with the
provenance ledger below. The relevant question is how well those new paths use
Turing, not whether they exist.

The scope is the normal resident four-GPU DeepSeek V4 Flash prefill and decode
path, including Q8 expansion caching, all four routed-expert quantization
combinations, output sharding, and inter-GPU movement. GLM 4.5/4.7, DSpark
support-model inference, SSD streaming, Metal, ROCm, and CPU kernels are not
reachable from that workload and are outside this matrix. Their CUDA kernels
remain visible in the machine-generated inventory so that the scope boundary
cannot hide a newly shared dependency.

## Evidence levels

| Mark | Meaning |
|---|---|
| `SOURCE` | Dispatch and source-visible storage are established by the cited code. |
| `COMPILED` | Registers, spills, local memory, emitted instructions, and exact static shared memory require the paired SM75/SM80 cubin report. |
| `MEASURED` | Established by the supplied Nsight capture on Quadro RTX 8000. |
| `LIBRARY` | cuBLAS selects an architecture-specific implementation outside `ds4_cuda.cu`; the application object cannot establish its resources. |
| `UNKNOWN` | Not yet captured. It is not evidence of either a problem or an optimization opportunity. |

The status classes are:

- `EQUIVALENT`: the same application kernel and launch policy is used. The
  generated machine code can still differ.
- `SM75_NATIVE`: there is an explicit Turing kernel or resource adaptation.
- `SM75_FALLBACK`: SM75 takes a generic, SIMT, or DP4A path while SM80 takes a
  different path.
- `LIBRARY_SELECTED`: both use cuBLAS but the selected kernels are opaque here.
- `SM75_IMPOSSIBLE`: the SM80 launch cannot fit or its required instruction is
  unavailable on SM75.
- `DIAGNOSTIC_ONLY`: not part of the release path unless explicitly enabled.

## Provenance ledger

| Origin | Change relevant to this matrix | Interpretation |
|---|---|---|
| `48beef8` (`CUDA support`) | Original CUDA kernels, including generic/DP4A quantized paths such as Q2 down | Inherited CUDA baseline |
| `36ee8c1` (`Add CUDA tensor parallelism and session batching`) | Paired TP/EP orchestration, SM80 dense-Q8 MMA, and the SM75-compatible Q4 tile8 MMA path | Inherited multi-GPU baseline |
| `118aca3` (`Fit exact attention tile on SM75`) | Added runtime shared-memory fitting and the 16x8 exact-score tile for Turing | **Added by this SM75 initiative** |
| `958ed67` (`Add SM75 IQ2 and dense Q8 MMA kernels`) | Added `m8n8k16` dense-Q8 and IQ2 gate/up kernels | **Added by this SM75 initiative** |
| `049d2da` (`Enable CUDA prefill Q8 cache by default`) | Made the expanded Q8-to-F16 cache the normal prefill policy | **Added by this SM75 initiative**; changes which matrix rows actually run |
| `7cac54e` (`Add SM75 tile16 hybrid MoE kernels`) | Added IQ2 tile16 gate/up, SM75 Q4 tile16 down, and the associated tile16 routed paths | **Added by this SM75 initiative** |
| `8f9ebcd` through `cfc0391` | Added, corrected, measured, and then disabled the regressed 32/32 attention-head split by default | Experimental branch work; not a performance capability |

`SM75_NATIVE` below describes the current dispatch, not its historical origin.
It must not be reported as a finding without also saying whether it is inherited
or was just implemented by this initiative.

## Architectural resource envelope

| Resource/capability | SM80 A100-class | SM75 TU102 / RTX 8000 | Consequence in this engine |
|---|---:|---:|---|
| Resident warps/SM | 64 | 32 | A 256-thread block is 12.5% of SM80 but 25% of SM75's warp capacity. |
| Resident blocks/SM | 32 | 16 | Usually secondary to shared memory, registers, or the 32-warp limit here. |
| 32-bit registers/SM | 65,536 | 65,536 | The same 64-register, 256-thread block consumes 16,384 registers on both, but represents twice the warp occupancy on SM75. |
| Max registers/thread | 255 | 255 | Equivalent hard limit. |
| Shared memory/SM | 164 KiB selectable on A100 | 64 KiB | SM80 designs using 65-75 KiB per block cannot be carried over. |
| Max shared memory/block | 163 KiB | 64 KiB opt-in; 48 KiB without opt-in | SM75 resource checks must precede launch. |
| Combined L1/shared | 192 KiB | 96 KiB | On SM75, crossing 32 KiB shared selects the 64 KiB shared / 32 KiB L1 carveout. |
| Shared-memory carveouts | 0, 8, 16, 32, 64, 100, 132, 164 KiB | 32 or 64 KiB | Reducing a Turing block below 32 KiB can both permit another resident block and restore 64 KiB L1. |
| Native integer MMA used by this source | `m16n8k32.s8` | `m8n8k16.s8` | The SM75 path does twice as many MMA instructions for the same 16x8x32 logical work before considering packing overhead. |
| Native sub-byte support | INT4 and binary, with larger Ampere forms | INT4 `m8n8k32`, binary `m8n8k128`; no native INT2 | IQ2 must be decoded to INT8 or evaluated with DP4A; Q4 currently expands nibbles to INT8 rather than issuing INT4 MMA. |
| Asynchronous global-to-shared copy | `cp.async` and hardware barriers | None | An Ampere multi-stage shared-memory pipeline cannot be mechanically backported. |
| Native warp reductions | Available | Not available | Turing uses shuffle/shared reduction sequences. |
| L2 capacity | 40 MiB on A100 | 6 MiB on full TU102 | Re-reading expert and attention weights is substantially less cacheable on TU102. |
| Device memory bandwidth | About 1,555 GB/s on A100 | About 672 GB/s on RTX 8000 | A fallback that increases bytes or rereads weights is disproportionately expensive on the target. |
| Interconnect | Up to 12 third-generation NVLinks on A100 | One second-generation NVLink peer per RTX 8000 in this host | Expert parallelism must remain inside pairs; the pipeline boundary remains cross-pair `SYS`. |

Turing has four static warp schedulers. With one eight-warp block resident, each
scheduler has only about two warps from that block to cover dependency, memory,
and MIO latency. This makes the `>32 KiB` rows below qualitatively different
from an SM80 block with the same nominal thread count.

## Core linear algebra and state transforms

`B` below is the number of 32-value Q8_0 blocks in one input row.

| Operation | Production dispatch condition | SM80 path | SM75 path | Launch/resources visible in source | Class/evidence |
|---|---|---|---|---|---|
| Q8_0 expanded-weight prefill GEMM | `n_tok > 1`, cache allowed, expansion fits | Q8_0 -> F16 once, FP32 -> F16 activations, `cublasGemmEx` | Same application dispatch | Expansion consumes `2 * in_dim * out_dim` bytes per cached matrix; cuBLAS resources opaque | `LIBRARY_SELECTED`; `SOURCE`, runtime cache coverage is workload-dependent |
| Q8_0 expanded-weight FP32 cache | Explicit F32 cache flags | Q8_0 -> F32, `cublasSgemm` | Same | Four bytes per weight; normally disabled | `DIAGNOSTIC_ONLY` |
| Native Q8_0 prefill GEMM | Cache unavailable; `n_tok >= 8`, `B <= 256` | `matmul_q8_0_mma_exact_kernel<T>` using `m16n8k32.s8`; 16 tokens x 64 rows/block | `matmul_q8_0_mma_sm75_exact_kernel<T>` using `m8n8k16.s8`; 8 tokens x 64 rows/block | 256 threads. Dynamic shared: SM80 `192*B` bytes; SM75 `160*B` bytes. At `in_dim=8192`, SM75 uses 40,960 bytes and is limited to one block/SM by shared memory. | `SM75_NATIVE`, added in `958ed67`; `SOURCE`, exact registers/spills `COMPILED` |
| Native Q8_0 short-batch fallback | MMA gate fails or batch below eight | Token-8/4/2, batch-warp, or exact kernels; DP4A selectable | Same dispatch order | 32-256 threads depending shape; resources `COMPILED` | `EQUIVALENT` dispatch, architecture codegen unknown |
| Native Q8_0 decode matvec | `n_tok == 1` | `matmul_q8_0_preq_warp8_kernel` | Same | 256 threads, DP4A unless disabled | `EQUIVALENT`; `SOURCE` |
| F16 batched matrix multiply | `n_tok > 1`, normal path | FP32 -> F16 activations + `cublasGemmEx` | Same | cuBLAS resources opaque | `LIBRARY_SELECTED` |
| F16 one-token matrix multiply | Default non-quality decode | `cublasGemmEx`; ordered/serial kernels under quality/debug gates | Same | Dispatch is shape and quality dependent, not architecture dependent | `LIBRARY_SELECTED` or `EQUIVALENT` fallback |
| F16 router prefill | Multiple rows | Four-row `cublasGemmStridedBatchedEx` groups plus one-row tail | Same | 4096 x 256 projection | `LIBRARY_SELECTED` |
| F16 router decode | One row | Ordered 32-thread chunk reduction by default; cuBLAS outside quality mode | Same | 32 threads/output in ordered path | `EQUIVALENT` dispatch |
| F32 matrix multiply | F32-stored weights | `cublasSgemm` when batched, custom fallback otherwise | Same | No architecture gate | `LIBRARY_SELECTED` / `EQUIVALENT` |
| Embedding lookup/HC replication | Token embedding | `embed_token(s)_hc_kernel` | Same | 256-thread elementwise grid | `EQUIVALENT` |
| RMS normalization | Width 4096, multiples of 2048, or generic | Fast4096, batch8, or generic kernel | Same | 256 threads/row; small shared reductions where used | `EQUIVALENT` |
| Fused Q/KV RMS + KV RoPE | Fusion enabled | `dsv4_qkv_rms_norm_rows_kv_rope_kernel` | Same | 256 threads, grid `(rows,2)` | `EQUIVALENT` |
| Head RMS/RoPE | Separate fallback or decode | Head/RoPE kernels | Same | 256-thread reductions/transforms | `EQUIVALENT` |
| FP8 QAT and KV store | Flash KV update | Software FP8 quantize/store kernels | Same | No SM80-only FP8 Tensor Core path is used | `EQUIVALENT`; functional, not hardware-FP8 acceleration |
| Compressor pool/update/ratio-4 state | Compressed KV maintenance | Compressor kernels plus F16 projections through the rows above | Same | Mostly elementwise/reduction kernels | `EQUIVALENT`; exact resources `COMPILED` |
| HC split/Sinkhorn/weighted sum | Layer pre/post transforms | Fused kernels when allowed, otherwise component kernels | Same | 256 threads; norm fusion has 1,024-byte reduction buffer | `EQUIVALENT` |
| HC expand/add/residual | Layer output assembly | `hc_expand_kernel` or Q8-down fused HC kernel | Same | Elementwise HC kernel; the fused Q8 decode path uses DP4A | `EQUIVALENT` dispatch |

Source anchors: Q8 architecture dispatch is in `ds4_cuda.cu:5414-5803` and
`ds4_cuda.cu:12561-12841`; F16 dispatch is in
`ds4_cuda.cu:13530-13852`; normalization and fused Q/KV normalization are in
`ds4_cuda.cu:14117-14287`; cache eligibility is in
`ds4_cuda.cu:1039-1067`.

## Attention, indexer, and output projection

| Operation | Production dispatch condition | SM80 path | SM75 path | Launch/resources visible in source | Class/evidence |
|---|---|---|---|---|---|
| Raw-window prefill attention | Non-quality prefill, `head_dim=512`, batch >=128 | `attention_static_mixed_heads8_online_kernel` | Same | 256 threads; 8,192-byte KV tile | `EQUIVALENT`; no architecture specialization |
| Raw prefill fallback | Window kernel disabled | cuBLAS score/value GEMMs if available; otherwise one block per token/head | Same | Custom fallback has about 1.5 KiB static shared | `LIBRARY_SELECTED` or `EQUIVALENT` |
| Static mixed prefill attention | No compressed mask, normal prefill | Same heads8 online kernel | Same | 256 threads; 8,192-byte KV tile | `EQUIVALENT` |
| Masked mixed prefill attention | Compressed mask required | cuBLAS score/value path, then custom softmax; custom mixed fallback if cuBLAS disabled | Same | Score scratch grows as `heads*tokens*keys` outside kernel | `LIBRARY_SELECTED` / `EQUIVALENT` |
| Indexed prefill attention | `top_k <= 512`, `head_dim=512` | `attention_indexed_mixed_heads8_online_kernel<8,16>` | Same | 512 threads; 16,384-byte KV tile plus about 1 KiB row metadata | `EQUIVALENT`; 512-thread block is 25% of SM80 versus 50% of SM75 warp capacity |
| Indexed two-pass fallback | Explicit two-pass flag | `attention_indexed_mixed_heads8_rb4_kernel` | Same | About 35.0 KiB static shared (`scores`, KV tile, and row metadata), hence one block/SM on SM75 | `DIAGNOSTIC_ONLY`; resource-risk row |
| Decode exact-score tiled dot | Exact score-split default, `head_dim=512` | 16 heads x 16 rows; 66,048-byte dynamic tile | 16 heads x 8 rows; 49,536-byte dynamic tile | 256 threads on SM75 (16*8). SM75 selection is based on `cudaDevAttrMaxSharedMemoryPerBlockOptin`. | `SM75_NATIVE`, added in `118aca3`; one block/SM and 32 KiB L1 on SM75 |
| Decode exact-score finalize | Exact score-split default | `attention_decode_score_split_finalize_kernel` | Same | About 34.0 KiB static shared from 8,192 scores, raw-row metadata, and reduction state | `EQUIVALENT` source, one block/SM on SM75 |
| Decode score variants | Explicit LDG/vec4/plain/dim2/graph flags | Corresponding score/finalize kernels | Same | Diagnostic dispatch, not architecture-selected | `DIAGNOSTIC_ONLY` |
| Decode online attention | Score count exceeds fixed buffer, or online flag | `attention_decode_mixed_heads8_online_kernel` | Same | About 9.0 KiB static shared | `EQUIVALENT` |
| Decode monolithic fallback | Score split declines | `attention_decode_mixed_kernel` | Same | More than 34 KiB static shared; one block/SM on SM75 | `EQUIVALENT`; Turing resource cliff |
| Experimental split-KV decode | Explicit/default fast-attention request | Split partial + combine kernels | Same | About 3.0 KiB static shared in partial kernel; scratch proportional to `heads*S*(head_dim+2)` | `DIAGNOSTIC_ONLY` until validated as release path |
| Indexer score, one token | 64 heads x 128 | Direct 128-thread kernel | Same | 528-byte shared tile/reduction | `EQUIVALENT` |
| Indexer score, prefill | Non-quality mode, 64 heads x 128 | WMMA128 FP16 kernel | Same WMMA source | 256 threads; exactly 45,056 bytes static shared. Turing is limited to one block/SM and the 64/32 shared/L1 carveout. | `EQUIVALENT`; Turing resource cliff, SASS and registers `COMPILED` |
| Indexer WMMA64/32/16 variants | Explicitly disable wider variants | Corresponding WMMA kernel | Same | Static shared falls with tile width | `DIAGNOSTIC_ONLY` tuning variants |
| Indexer top-1 | `top_k=1` | 1,024-thread reduction | Same | 8,192 bytes static shared; one block occupies all SM75 warps | `EQUIVALENT` |
| Indexer top-512/2048 | Shape-selected power-of-two or CUB sort | Same source | Same | 1,024 threads; 8-48 KiB static/dynamic shared depending `n_comp` | `EQUIVALENT` dispatch; exact CUB resource use `COMPILED` |
| Attention output A, cached | Non-quality prefill, cache fits | F16 packed heads + strided batched cuBLAS GEMM | Same | Eight 4096x1024 group projections; cache residency is part of model footprint | `LIBRARY_SELECTED` |
| Attention output A, uncached | Cache absent | Grouped SM80 Q8 MMA when batch >=8 | Grouped SM75 Q8 MMA when batch >=8 | Same Q8 formulas as core GEMM with `B=128`: SM80 24,576 B, SM75 20,480 B dynamic shared | `SM75_NATIVE`, supplied by the dense-Q8 work in `958ed67` |
| Attention output B | 8192 -> 4096 Q8 | Cached F16 cuBLAS if it fits; otherwise SM80 Q8 MMA | Cached F16 cuBLAS if it fits; otherwise SM75 Q8 MMA | Uncached SM75 uses 40,960 B dynamic shared and therefore one block/SM | `LIBRARY_SELECTED` or `SM75_NATIVE` from `958ed67` |
| Vocabulary output head | Four-way vocabulary shard | Q8/F16/cuBLAS dispatch inherited from general matmul, per shard | Same | Output gathering is separate from matmul | `LIBRARY_SELECTED` / `EQUIVALENT` orchestration |

Source anchors: attention prefill/decode dispatch is in
`ds4_cuda.cu:15362-16042`; exact-score tile selection is in
`ds4_cuda.cu:7354-7743` and `ds4_cuda.cu:8638-8840`; indexer dispatch is in
`ds4_cuda.cu:12049-12462`; attention output is in
`ds4_cuda.cu:16086-16305`.

## Routed experts: full `{IQ2,Q4}` gate/up x `{Q2,Q4}` down matrix

All four combinations share activation Q8_K quantization, expert sorting,
tile construction, and the Q8_K intermediate interface. Gate and up have one
type because they are fused and share a byte stride.

For the target shapes, `xq_blocks=16`, `midq_blocks=8`, 256 experts, six
selected experts, and prefill microbatches normally contain 512 tokens.

| Gate/up | Down | SM80 prefill dispatch | SM75 prefill dispatch | Source-visible block resource | SM75-path provenance | Class/evidence |
|---|---|---|---|---|---|---|
| IQ2_XXS | Q2_K | Generic IQ2 expert tile8 row-span + Q2 tile16 atomic down | IQ2 tile16 `m8n8k16.s8` + Q2 tile16 atomic down | SM75 IQ2: 39,748 B static shared, 256 threads. Q2 down: 37,376 B static shared, 256 threads. Both force one block/SM. | IQ2 tile8 `958ed67`; tile16 `7cac54e`; Q2 inherited from `48beef8` | Gate/up `SM75_NATIVE`; down `EQUIVALENT` DP4A |
| IQ2_XXS | Q4_K | Generic IQ2 expert tile8 row-span + SM80 Q4 tile16 `m16n8k32.s8` down | IQ2 tile16 `m8n8k16.s8` + SM75 Q4 tile16 `m8n8k16.s8` down | SM75 IQ2 39,748 B; SM75 Q4 down 37,444 B; both 256 threads and one block/SM | IQ2 tile8 `958ed67`; IQ2 tile16 and Q4 down `7cac54e` | Both `SM75_NATIVE`; `MEASURED` on hybrid |
| Q4_K | Q2_K | SM80 Q4 gate/up tile16 `m16n8k32.s8` + Q2 tile16 atomic down | Q4 gate/up tile8 `m8n8k16.s8` + Q2 tile16 atomic down | SM80 Q4 gate tile16: 74,948 B total shared. SM75 Q4 gate tile8: 37,476 B static. Q2 down: 37,376 B. | Q4 tile8 inherited from `36ee8c1`; Q2 inherited from `48beef8`; SM80 tile16 added in `7cac54e` | Gate/up `SM75_FALLBACK`; down `EQUIVALENT` DP4A |
| Q4_K | Q4_K | SM80 Q4 tile16 gate/up + SM80 Q4 tile16 down | SM75 Q4 tile8 gate/up + SM75 Q4 tile16 down | SM80: 74,948 B gate, 74,820 B down. SM75: 37,476 B gate, 37,444 B down. | Q4 tile8 inherited from `36ee8c1`; Q4 tile16 down added in `7cac54e` | Gate `SM75_FALLBACK`; down `SM75_NATIVE` |

### Routed-expert suboperations

| Operation | SM80 path | SM75 path | Resource/dispatch consequence | Evidence |
|---|---|---|---|---|
| Input F32 -> Q8_K | `q8_K_quantize_kernel` | Same | 256 threads per Q8_K block/row | `EQUIVALENT` |
| Count/prefix/scatter selected pairs | Count, prefix, scatter kernels | Same | Atomics/scans are not architecture-specialized | `EQUIVALENT` |
| Build tile8/tile16 descriptors | Expert tile builder | Same | Padding and active-expert behavior is data-dependent, not architecture-dependent | `EQUIVALENT`; tile counts `MEASURED` |
| IQ2 gate/up tile16 | Not selected: `cuda_sm75_mma_ok()` requires exactly 7.5 | Two `m8n8k16.s8` passes, one for gate and one for up | SM75-specific path is not an SM80 backport; SM80 currently falls to generic IQ2 DP4A | `SM75_NATIVE`, added in `958ed67`/`7cac54e`; 64 registers/thread and about 24.8% achieved occupancy `MEASURED` |
| Q4 gate/up tile16 | `m16n8k32.s8`; `binaryVersion >= 80` | Cannot launch: 74,752-byte activation tile alone exceeds SM75's 64 KiB block limit | SM75 dispatches tile8 even when the tile16 feature flag is true | `SM75_IMPOSSIBLE` for current layout |
| Q4 gate/up tile8 | Secondary SM80 path | Primary SM75 path, `m8n8k16.s8` | 37,476 B static shared, so one block/SM; Q4 gate/up remains unprofiled on target | `SM75_FALLBACK`; runtime cost `UNKNOWN` |
| IQ2 decode gate/up | Generic IQ2 decode LUT/DP4A | Same | No SM75 tensor-core decode specialization | `EQUIVALENT`; tuning status `UNKNOWN` |
| Q4 decode gate/up | Q4 warp32/optional graph variants | Same | DP4A-style decode kernels, no SM80-only launch | `EQUIVALENT`; tuning status `UNKNOWN` |
| Intermediate F32 -> Q8_K | Standard, owned, or sidecar quantizer | Same | Selected by ownership and decode flags | `EQUIVALENT` |
| Q2 down prefill | Tile16/row-span DP4A, usually atomic direct accumulation | Same | Exactly 37,376 B shared at tile16. Each thread also declares 16 pair IDs, 16 input pointers, and 16 accumulators; compiled register/spill cost is still `UNKNOWN`. No INT2 MMA exists on either architecture. | `EQUIVALENT`; architecture-independent algorithmic gap |
| Q4 down tile16 | `m16n8k32.s8`, dynamic activation tile | Turing-specific `m8n8k16.s8`, static half-width activation tile | SM75 path fits at 37,444 B but remains one block/SM | `SM75_NATIVE`, added in `7cac54e`; 64 registers/thread and about 24.9% achieved occupancy `MEASURED` |
| Direct decode down | Sum-6 or sum-3 Q4/Q2 kernels | Same | One-token fast path, separate from prefill tiles | `EQUIVALENT`; tuning status `UNKNOWN` |
| Owned output combine | Fixed-three slot/packed combine, then shared down/HC fusion | Same | Pair-local expert ownership, no architecture gate | `EQUIVALENT` |

The exact shared-memory calculations use `sizeof(cuda_block_q8_K)=292` bytes:

- IQ2 tile16 gate/up: `8*16*292 + 256*8 + 128 + 49*4 = 39,748` bytes.
- Q4 tile8 gate/up: `8*16*292 + 25*4 = 37,476` bytes.
- Q4 SM75 tile16 down: `16*8*292 + 17*4 = 37,444` bytes.
- Q2 tile16 down: `16*8*292 = 37,376` bytes. Its sizeable per-thread arrays
  affect registers/local memory rather than the source-visible shared total.
- SM80 Q4 tile16 gate/up: `16*16*292 + 49*4 = 74,948` bytes.
- SM80 Q4 tile16 down: `16*16*292 + 17*4 = 74,820` bytes.

Source anchors: quantized dot helpers start at `ds4_cuda.cu:16989`; SM75 and
SM80 MMA expert kernels are in `ds4_cuda.cu:20455-21578`; the complete routed
dispatch is in `ds4_cuda.cu:22251-23573`; the SM80 tile16 resource gate is in
`ds4_cuda.cu:21580-21624`.

## Shared expert, router, residual, and output assembly

| Operation | SM80 path | SM75 path | Resource/dispatch consequence | Class/evidence |
|---|---|---|---|---|
| Shared gate/up prefill | General Q8 pair/rows dispatch; expanded F16 cuBLAS when cached | Same | Two 4096x2048 matrices; cache costs 32 MiB per layer per device pair member | `LIBRARY_SELECTED` when cached |
| Shared gate/up decode | Fused exact Q8 pair then SwiGLU | Same | 256-thread DP4A kernel; no tensor-core decode path | `EQUIVALENT` |
| Shared down prefill | Expanded F16 cuBLAS when cached, then HC assembly | Same | 2048x4096 expansion costs 16 MiB per layer/device | `LIBRARY_SELECTED` |
| Shared down decode | Fused Q8 matvec + routed add + HC expansion | Same | 256-thread DP4A kernel | `EQUIVALENT` |
| Router top-six select | Parallel/warp-top-k kernel for batches | Same | Up to four warps of 256 probabilities in shared memory | `EQUIVALENT` |
| Residual add/SwiGLU/zero/fill | Elementwise kernels | Same | Bandwidth/SIMT path | `EQUIVALENT` |
| Output HC weights and expand | Output HC kernels | Same | Small elementwise/reduction path | `EQUIVALENT` |

## Four-GPU orchestration and memory residency

| Operation | SM80 design assumption | Actual SM75 target behavior | Class/evidence |
|---|---|---|---|
| Pipeline placement | Resource model chooses two layer stages | Same two-stage model, but the measured optimum is time-balanced 25/18 for the hybrid rather than byte-balanced 21/22 | Not an ISA dispatch; placement must be benchmarked per quant and clock state |
| Expert ownership | Half of routed experts on home, half on partner | Same, inside NVLink pairs 0<->1 and 2<->3 | `EQUIVALENT` orchestration |
| Home -> partner expert input | Peer copy of normalized rows, selected IDs, and weights | Direct peer access within each NVLink pair | Topology-specific, not SM version-specific |
| Partner -> home routed output | Peer copy of one partial output | Direct peer access within pair | Topology-specific |
| Stage boundary HC transfer | Pipeline activation copy | Physical GPU 0 -> 2 crosses `SYS`; no NVLink between pairs | Target constraint; 32 MiB per 512-row microbatch |
| Non-routed weights | Replicated in each pair | Same | Enables pair-local execution but consumes the headroom used by expanded Q8 caches |
| Routed weights | Half-resident by expert owner | Same | Dominant model residency term |
| Q8 -> F16 cache | Extra model-resident optimization | Enabled by default, bounded by live free VRAM and reserve | Coverage differs sharply between 100.92-GiB hybrid and 153.33-GiB full Q4 |
| Attention compressed KV | Resident on home stage device | Same | Small relative to weights; not duplicated for the disabled head-split experiment |
| Experimental head split | More device parallelism assumed beneficial | Disabled: measured about 20% slower and did not reduce transfer bytes | `DIAGNOSTIC_ONLY`, known regression |
| Output vocabulary sharding | Four shards | Same | Gather/reduction behavior is topology-sensitive, not ISA-gated |

## Source-visible SM75 resource cliffs

These are not performance rankings. They are the rows where Turing changes the
resource regime before registers are considered.

| Kernel family | SM75 shared memory | Threads | SM75 upper bound from this resource alone | Why it matters |
|---|---:|---:|---:|---|
| IQ2 tile16 gate/up | 39,748 B | 256 | 1 block, 8 warps, 25% | `MEASURED` at about 24.8% achieved occupancy; 53.6% of hybrid aggregate kernel time. |
| Q4 tile8 gate/up | 37,476 B | 256 | 1 block, 8 warps, 25% | Required fallback for full-Q4 gate/up; not yet profiled. |
| Q4 tile16 down | 37,444 B | 256 | 1 block, 8 warps, 25% | `MEASURED` at about 24.9% achieved occupancy. |
| Q2 tile16 down | 37,376 B | 256 | 1 block, 8 warps, 25% | Uses DP4A, not Tensor Cores; compiled register/spill pressure remains unknown. |
| Q8 native 8192-input projection | 40,960 B dynamic | 256 | 1 block, 8 warps, 25% | Applies to uncached attention output B and other 8192-wide Q8 projections. |
| Exact-score attention tile | 49,536 B dynamic + 64 B static | 256 | 1 block, 8 warps, 25% | SM75 already reduced the SM80 tile from 16 rows to eight. |
| Indexer WMMA128 | 45,056 B | 256 | 1 block, 8 warps, 25% | Important prefill path that had not been included in the earlier kernel-specific discussion. |
| Decode score/finalize full buffer | about 34 KiB | 256-512 | 1 block | Forces 32 KiB L1; decode must be analyzed separately from prefill. |
| Indexed two-pass heads8 | about 35 KiB | 256 | 1 block, 8 warps, 25% | Diagnostic fallback; online path is materially lighter. |
| Power-of-two top-k 4096/8192 | 32-48 KiB | 1,024 | 1 block, 32 warps, 100% | One block already fills the SM75 warp limit, so block occupancy is not the issue. |

## What static inspection cannot establish

The following columns are intentionally not guessed:

1. Registers/thread, spills, local memory, and exact compiled static shared
   memory for every template instantiation on both architectures.
2. Whether a nominal MMA/WMMA source path emits `IMMA`/`HMMA` in the exact
   cubin, and whether a fallback emits `IDP.4A`.
3. cuBLAS kernel names and resources for cached F16 projections.
4. Runtime branch coverage for a particular model, context, quality mode, and
   VRAM state.
5. Critical-path exposure after cross-device overlap.

Run the paired compiler audit to fill items 1 and 2 without executing the
model:

```bash
cd ~/ds4-iq2-q4
./speed-bench/cuda-sm-dispatch-resource-audit.sh
```

It compiles the same translation unit for `sm_75` and `sm_80`, records every
CUDA kernel declaration/definition and launch expression, exports cubin resource usage,
demangles every compiled function, counts architecture-specific SASS
instructions per function, and creates a small archive. Set `KEEP_SASS=1` or
`KEEP_OBJECTS=1` only when the raw artifacts are required.

The source-derived inventory is a drift detector, not proof of runtime
reachability. Runtime branch coverage still requires an unperturbed Nsight
Systems trace of the exact model and fixed prompt suite; selected critical
kernels then require Nsight Compute.

## Audit conclusions (not optimization proposals)

1. Dense-Q8 SM75 MMA, IQ2 gate/up SM75 MMA, Q4-down SM75 MMA, and the smaller
   exact-attention tile are **outputs of the current initiative**, not findings
   of this audit. Their existence says nothing about whether they are close to
   optimal. The measured MoE kernels still achieve only about 25% occupancy.
2. The inherited Q4 gate/up path already used Turing `m8n8k16` MMA. The
   remaining gap is specifically tile shape and staging: the newer SM80 tile16
   layout needs a 74,752-byte activation tile and cannot launch on SM75, which
   therefore remains on the inherited 37,476-byte tile8 kernel. Calling the
   entire Q4 gate/up operation an unoptimized DP4A fallback would be false.
3. Q2 down remains an inherited DP4A path with no native INT2 MMA alternative.
   Its tile16 kernel also declares large per-thread arrays whose compiled
   register and spill cost has not yet been captured.
4. Indexer WMMA128 and several attention/decode kernels cross Turing's 32-KiB
   shared-memory cliff even though their dispatch is architecture-equivalent.
   They are unresolved audit targets, not evidence that a replacement kernel
   will automatically be faster.
5. Expanded Q8-to-F16 cache coverage changes the active architecture matrix.
   A cached projection is a cuBLAS FP16 path; the same projection becomes the
   newly added custom SM75 INT8 kernel when the cache cannot fit. Quant size
   therefore changes both residency and executed kernels.
6. The next evidence pass must evaluate the work already added—not rediscover
   it: paired cubin resources/SASS, runtime branch coverage, and representative
   Nsight measurements for inherited and initiative-added paths. No further
   optimization or custom quant choice is justified solely from this static
   matrix.

## Architecture references

- NVIDIA, *Turing Tuning Guide*, sections 1.4.1 and 1.4.3: execution resources,
  independent thread scheduling, and the 96-KiB unified L1/shared-memory
  configurations.
- NVIDIA, *NVIDIA Turing GPU Architecture*, sections "Turing Streaming
  Multiprocessor" and "GDDR6 Memory Subsystem": scheduler, cache, Tensor Core,
  L2, and RTX 8000-class memory-system characteristics.
- NVIDIA, *Turing Tensor Core Architecture*, Hot Chips 31: Turing MMA operand
  forms and supported INT8/INT4/binary modes.
- NVIDIA, *Ampere Tuning Guide*, sections 1.4.1 and 1.4.2: A100 occupancy,
  shared-memory limits, asynchronous copies/barriers, L2, and carveouts.
