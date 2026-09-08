# GPU1 runtime and remaining software investigation

Reviewed 2026-09-08 against source commit
`039eef3e756412da0d2aec02ae0af835b2a09caf` and archive
`sm75-token-row-arithmetic-20260908T003609Z.tar.gz`. This document supplements
`sm75-gpu1-memory-ordering-audit.md`; it does not replace the frozen reproducer.
No GPU execution, driver installation, production change, or external submission
was performed for this review. Proposed execution controls below are **not a
request to run them now**. GPU loss requires host recovery and a deliberately
selected, separately approved experiment, not an automatic retry sequence.

Implementation follow-up: `sm75-runtime-contract-trace.md` describes the new
opt-in, same-ELF application-API instrumentation and its CPU-only qualification
gate. It measures allocation/view lifetimes, conversion arguments, handle and
context/stream state, and real call forwarding/completion at the known transition.
This is concrete instrumentation work, not a declaration that uncollected runtime
facts are resolved. The subsequent 03:42 UTC GPU1 failure capture measured these
contracts at the known transition; see
`sm75-gpu1-runtime-capture-findings-20260908.md`. No caller violation was detected,
but library-internal execution remains unmeasured. `sm75-nsys-device-timeline.md`
describes the next host-only profiler supervision gate; GPU timeline capture is
not yet enabled. The sections below retain the earlier incident's stack/history.

## Established runtime, not inferred from nvidia-smi

The archive's `failure-context/process-timeline.jsonl` records the actual mapped
libraries, and `failure-context/summary.json` records their SHA-256 values:

| Component | Actual mapped file | SHA-256 |
| --- | --- | --- |
| Executable | `tests/cuda_sm75_token_row_arithmetic` | `5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414` |
| CUDA runtime | `/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcudart.so.13.2.75` | `c7eaa64c99c20484b7401352aff99a8c5ab947abaab3a9bbb70142312da1fe18` |
| cuBLAS | `/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublas.so.13.4.0.1` | `5589b43e2aae4790ccb2abd22733eae0b6d4fe9503dcc4dbfe219fd857ddac41` |
| cuBLASLt | `/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublasLt.so.13.4.0.1` | `ac76bbb85d74313bba0d71029c71389616205f4018c8ff31d57c38f6e61100bd` |
| CUDA driver | `/usr/lib/x86_64-linux-gnu/libcuda.so.595.84` | `c147185c80a0270d635db626537fc9223b2ad1764f86cb12fd116d26e08b175d` |

Additional mapped driver components were `libnvidia-gpucomp.so.595.84`,
`libnvidia-ptxjitcompiler.so.595.84`, and `libnvidia-nvvm70.so.4`; their hashes are
also in the summary. Mapping a JIT library does not prove that this particular
kernel was JIT-compiled. Mapping cuBLASLt does not prove the application called
its API: the reviewed B wrapper calls `cublasGemmEx`.

The manifest records `SKIP_BUILD=1`. The run alone therefore proves executable
identity and loaded runtime identity, not original compilation flags. The later
`sm75-investigation-inventory.kzJxuv.tar.gz` now preserves the exact ELF. Offline
inspection finds an embedded CUDA 13.2.78 compiler banner and GCC 13.3/Broadwell
DWARF producer information, compatible SM75 cubins/PTX and .so.13 dependencies.
No RPATH/RUNPATH or PTDS/PTSZ imports were found. This is partial original-build
evidence, not reconstruction of every flag/source revision. Full findings and
limits are in `sm75-gpu1-inventory-binary-supplement-20260908.md`.

### SM75 support and release-note triage

CUDA 13.2's compiler documentation explicitly lists `sm_75` as supported and as
the default architecture. Turing is not among the Maxwell/Pascal/Volta targets
removed in CUDA 13. This is not an unsupported-SM75 explanation.
[CUDA 13.2 NVCC architecture documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
[CUDA 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html).

NVIDIA associates cuBLAS 13.4.0.1 with CUDA 13.2.1. Its documented 13.4.1 patch
fix concerns NVFP4 output scaling on compute capabilities 10.x/11.x, not this
SM75 FP16-input/FP32-output `GemmEx` case. That patch note is not evidence of a
fix for this failure. No matching public SM75 fix was identified in the reviewed
release/patch notes; absence of a public entry does not exclude a library bug.
[NVIDIA CUDA 13.2.1 component listing](https://docs.nvidia.com/deeplearning/frameworks/cuda-dl-release-notes/rel-26-04.html),
[cuBLAS patch notes](https://docs.nvidia.com/cuda/cublas-patch-release-notes/).

The CUDA 13.x compatibility table sets a driver-family minimum of 580, which
595.84 exceeds. This is only a broad compatibility check, not validation of this
driver build, firmware, board, kernel, and library combination.
[CUDA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## What the B call actually requests

`cuda_matmul_q8_0_tensor_labeled_algo` in `ds4_cuda.cu` supplies:

| Argument | Value for the final failing half |
| --- | --- |
| API | `cublasGemmEx` |
| Operations | A transposed, B not transposed |
| Dimensions | m=4096, n=256, k=8192 |
| Leading dimensions | lda=8192, ldb=8192, ldc=4096 |
| Storage | A/B FP16, C FP32 |
| Compute argument | `CUDA_R_32F` in the source's selected overload |
| Scalars | host locals alpha=1, beta=0 |
| Algorithm | `CUBLAS_GEMM_ALGO3_TENSOR_OP`, numeric 103 |

The predecessor is DEFAULT at n=512. The arithmetic fixture's 512-token local
microbatch splits into 256+256; an outer production prefill chunk of 2048 is not
the GEMM's n parameter. Changing this reproducer to n=2048 would change the
case, not repair its fidelity.

NVIDIA documents these explicit Tensor Core algorithm enums as deprecated but
still selectable; the documented lack of effect on Ampere-and-newer does not
apply to Turing. DEFAULT is selection policy, not a promise of the same kernel
across dimensions or library versions. Version changes need not preserve bits.
The default cuBLAS stream is NULL; shared-stream/handle workspace choices affect
reproducibility. These facts do not establish an unsafe stream or workspace in
this process.
[cuBLAS API, algorithms, streams and reproducibility](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html).

Source search finds no backend `cublasSetStream`, `cublasSetWorkspace`, or
`cublasSetPointerMode`. Initialization does call `cublasSetMathMode`, but discards
that call's status at the initialization site. The source-side facts favor the
existing default-stream ordering model; runtime handle state and pointer
ownership are still unobserved. This ignored return is an observability gap,
not a demonstrated cause of PCIe Surprise Down. Do not silently change math
mode, workspace, or algorithm while trying to measure the original failure.

## Evidence gaps that can be addressed without another GPU workload

1. **Original executable/build provenance: initial inspection completed.** The
   exact ELF is preserved. Dependencies, search tags, build ID, producer fields,
   embedded code targets, selected host GEMM adapter and SM75 conversion SASS
   have been inspected without execution. No defect was identified in that narrow
   review. Dynamic arguments/library code and full build flags remain open. Use
   `readelf` and `cuobjdump` as file-inspection tools; do not execute the
   reproducer, run `make`, or infer its compiler from today's `nvcc --version`.
   `cuobjdump --list-elf` and `--list-ptx` distinguish embedded code objects;
   they do not show which library kernel ran in the failed process.
   [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).
2. **Package integrity and change history.** Preserve installed-package
   versions/ownership and relevant apt/dpkg transaction history for the mapped
   CUDA/cuBLAS files, kernel module, and driver libraries. Ask NVIDIA to compare
   the exact file hashes/build IDs against its distributed builds. Do not
   overwrite the current CUDA installation to obtain this information.
3. **Effective runtime environment.** The archived command records its explicit
   settings, not every inherited library control. The runner clears DS4
   variables and `CUDA_LAUNCH_BLOCKING`; the collector rejects logging/injection
   variables. It does not record whether `CUBLAS_WORKSPACE_CONFIG`,
   `CUDA_MODULE_LOADING`, `CUDA_FORCE_PTX_JIT`, `CUDA_DISABLE_PTX_JIT`,
   `CUDA_DEVICE_MAX_CONNECTIONS`, `NVIDIA_TF32_OVERRIDE`, or `LD_LIBRARY_PATH`
   were set. Their historical values cannot be reconstructed from omission.
   Record an allowlist with explicit set/unset states going forward, not the
   whole environment, and do not silently unset a previously effective value.
   The mapped-library paths already resolve the actual loaded-file question.
4. **Review the exact prefix as well as the B tail.** Check arithmetic, sizes,
   aliasing, bounds, initialized regions, device identity and ownership from
   fixture setup through input replacement, half views, and final consumption.
   In-bounds corruption is not excluded by a passing allocation-bounds test.
   Keep the historical cases and their instrumentation status separate.

## Instrumentation design for a separately approved local run

### Available no-workload inventory collector

`collect-sm75-investigation-inventory.sh` implements the missing file/tool/package
inventory above without executing the diagnostic or calling `nvidia-smi`.
It creates a unique `sm75-investigation-inventory.XXXXXX` directory inside the
repository and a matching archive. Each external query has a timeout, output
log, exit status and timestamps. Missing optional tools are reported as gaps;
they do not trigger installation, a fallback workload or a hardware change.

The default is `ROOT_HOST_INVENTORY=0 INCLUDE_EXECUTABLE=0`. Explicit
`ROOT_HOST_INVENTORY=1` requests numeric DMI types 0/1/2/39 and read-only BMC SEL,
power-supply SDR, sensor and FRU queries. It validates sudo once in the foreground;
each privileged command has a root-owned inner timeout plus an outer bound.
Authentication failure retains and archives partial data and returns status 2.

Explicit `INCLUDE_EXECUTABLE=1` includes the original executable only after the
frozen hash matches and its size is at most 128 MiB. The copied bytes are hashed
again before inclusion; mismatch, oversize, or failed copy returns status 2.
It never rebuilds or executes the binary. Without that option, a mismatched or
missing executable is recorded but other read-only metadata still collects.
The current environment snapshot is not retrospective evidence of the previous
run's inherited values. Firmware/BMC data can include serial numbers; the
optional binary can include build/source paths. Review before external sharing.

### Future workload instrumentation remains separate

This section defines useful questions, not permission to retry the failed host.
The current capture helper intentionally rejects `CUBLAS_LOG*`, `CUPTI_*`,
`LD_PRELOAD`, and `CUDA_INJECTION64_PATH`. A profiled run requires an explicitly
labeled collection mode and tested process-tree cleanup, not removal of those
guards or disguising the trace as the uninstrumented control.

### API/stream timeline first

Nsight Systems can correlate CUDA APIs, kernels, copies, submitting threads,
contexts and streams, and optionally allocation lifetimes and cuBLAS calls.
The inventory now identifies installed version 2025.6.3.541-256337736014v0 and
preserves its help. It advertises `cuda-sw`, `cublas`, process-tree scope,
event-trace disablement, all-API/allocation tracking and flush controls. No trace
mode has yet been run or qualified for this failure. Candidate
features are CUDA plus cuBLAS tracing, `--cuda-memory-usage=true` and
`--cuda-trace-all-apis=true`. Disable extra device event-completion tracing when
testing ordering: NVIDIA warns that it can create false dependencies. Keep
tracing process-scoped, not system-wide. Do not request hardware trace features
documented for Blackwell on SM75. Crashes can lose unflushed trace records;
periodic flushing and detailed tracing also perturb execution.
[Nsight Systems CUDA trace and CLI documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace).

Required analysis: correlate the last DEFAULT and first 103 calls to their
actual kernels; verify conversion-to-GEMM-to-next-overwrite ordering; enumerate
allocations/frees before the fault; determine whether any additional stream
touches the same memory. A kernel name alone is not a proof of causality. Trace
duration/flush policy must cover the full unchanged prelude and burn-in within
a bounded collection; do not shorten the workload or promise complete device
records after bus loss.

### Pointer and handle state when the timeline is insufficient

For a separate diagnostic build or carefully reviewed interposer, record an
allocation-generation ledger, base/size, view offset/extent, device/context,
input/weight/output roles and handle state before relevant calls. Queries such
as `cudaPointerGetAttributes` and `cuMemGetAddressRange` supply placement/range
facts, but not application subview ownership or absence of valid-address races.
Record query errors and stop before submitting a call with an invalid contract.
[CUDA pointer attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html),
[CUDA allocation-range API](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-driver-api/group__CUDA__MEM.html).

CUPTI callbacks can capture CUDA API arguments and correlation data. They are
not permission to call arbitrary CUDA APIs recursively: NVIDIA warns that
runtime/driver calls within callbacks are unsupported unless explicitly allowed
and can hang. Collect callback arguments into a host-side ledger; perform any
approved extra queries outside those callbacks. Do not combine independent
profilers/subscribers without checking the installed CUPTI version's support.
[CUPTI callback restrictions](https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api).

cuBLAS's own logger can be enabled by environment without rebuilding the
application. Its actual output coverage needs inspection; it is not a global
race detector or a guaranteed allocation history. Treat it as a separately
instrumented experiment with the same executable and workload.
[cuBLAS logger configuration](https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html#cublasloggerconfigure).

## Decision-gated controls, not an automatic test matrix

Every execution row below preserves the frozen binary and all fixture data,
dimensions, calls, synchronization, GPU selection and power settings unless
the row explicitly declares a different artifact. Stop after one selected
experiment and analyze it; a pass is not permission to proceed down the list.

| Question | Single deliberately changed factor | Useful result and limitation |
| --- | --- | --- |
| Does actual runtime ordering match source? | One verified profiler/trace mode | Reveals streams and lifetimes; a passing trace may mask a timing-sensitive fault. |
| Is a specific library build implicated? | NVIDIA-approved same-ABI cuBLAS+Lt package pair, isolated location, same driver and binary | A changed outcome directs library investigation; it does not prove a repaired GPU. Capture resolved paths/hashes. Never replace only one half of an interdependent pair casually. |
| Is the driver build implicated? | NVIDIA-supported driver change, existing binary and user libraries retained where supported | Requires planned reboot and authority. Record kernel/module/GSP changes; do not simultaneously change BIOS, slot, power or toolkit. |
| Is workspace ownership/selection implicated? | One explicit workspace control, only after state/trace evidence warrants it | Changes footprint/selection and possibly timing; not a generic race fix. The present tail is already checkpointed. |
| Is embedded code/JIT selection implicated? | One supported loading/JIT control after offline fatbinary inspection | Failure to load a kernel is not the historical GPU-loss reproduction. Do not force absent PTX/cubin or flush caches indiscriminately. |
| Is a pointer contract violated? | New separately named checked diagnostic binary | Can stop on a contract violation before GPU submission; cannot be labeled the same binary control. |
| Is an earlier custom kernel corrupting valid memory? | One targeted checker/canary or snapshot at a justified boundary | New binary/layout/extra synchronization must be declared. Expand coverage only where a concrete untested region remains. |
| Is a compiler-generated kernel implicated? | Separately named rebuild from frozen source with exactly one compiler/flag variable | First prove code provenance and preserve the old artifact. This is lower priority than a concrete pointer or runtime finding. |

Changing cuBLAS major ABI (for example, replacing `.so.13` with `.so.12`) is not
an unchanged-binary library control. Do not use symlink tricks. A toolchain-major
comparison requires a separately built artifact and explicitly different
experiment. Numerical divergence across builds/libraries must be distinguished
from runtime failure; an exactness threshold must not conceal a hardware fault.

## Exclusions and stop conditions

- No peer GPU is needed for any local stream, pointer-lifetime, compiler or
  library question above. Pair-0 production re-enablement remains out of scope.
- Do not repeat the broad algorithm 103-106 sweep: the frozen failure already
  uses the production-selected 103, after DEFAULT at the known dimensions.
- Do not call a shortened component test a replacement for the complete known
  GPU1-only failure sequence.
- Do not add blanket synchronization to the final DEFAULT/103 boundary: the
  current reproducer already synchronizes each transition.
- Memcheck/initcheck/synccheck passes from earlier archives are evidence for
  their respective instrumented runs, not a global-memory-race clearance.
  Racecheck's incomplete run is not a passing result.
- Driver/hardware investigation and software audit remain parallel: a PCIe
  Surprise Down event does not by itself exonerate software, and a library call
  preceding the error does not identify the physical fault's initiator.
- No software avenue is declared exhausted. Unavailable runtime trace, remaining
  compiled-code/library inspection, package provenance, and vendor-private driver/cuBLAS
  investigation are explicitly pending; none justifies an unplanned reboot loop.
