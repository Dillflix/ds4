# Frozen GPU1 reproducer: runtime-contract instrumentation

This is an **opt-in instrumented experiment**, not a production fix or a claim
that the GPU-loss cause is resolved. The first qualification step is host-only:
compile/load the interposer and exercise forwarding against a CPU fake provider.
Do not run the GPU reproducer just to find out whether this tool loads.

## Questions this implementation measures

| Question | Recorded evidence | Interpretation limit |
| --- | --- | --- |
| Are GEMM views live and within their allocations? | Allocation/free history, allocation generations, pointer attributes, driver allocation base/size, calculated column-major operand extents | Allocation bounds do not prove application subview ownership or valid contents. |
| Is FP32-to-FP16 conversion writing the expected input? | Main-ELF kernel registration/launch mapping and safely copied host-side source/destination/count arguments for the recognized kernel | No GPU tensor contents are read; arbitrary kernel argument layouts are not guessed. |
| Which execution state owns a call? | Host PID/TID, device, current driver context, cuBLAS handle, stream, pointer mode and math mode | API success normally means submission, not GPU completion. |
| What happens at DEFAULT512 to production103-half256? | Operations/dimensions/strides/types/scalar bits, wrapper entry, marker before forwarding, real return, and existing device-synchronization return | Wrapper entry alone does not mean the real GEMM was invoked. |
| Is there an observable lifetime/order violation? | Generation-aware range checks and recognized conversion-to-GEMM relationships | Missing or cross-thread/context/stream ordering stays unknown, not automatically safe or racy. |
| Was the failure interval preserved? | Bounded JSONL, immutable prefix snapshot, source timestamps, unfinished calls/tail and existing journal/process/PCIe evidence | GPU loss or termination can leave missing records. |

The original executable remains SHA-256
`5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
The capture wrapper does not rebuild or patch it. It requires the existing
GPU1-only `output-b-production103-no-row-owned` workload: full512/half256
prelude, 1,024 production-103 burn-in calls in batches of ten, and original
7/10/3 timing settings. No peer, native-stream, row-owned selector, sanitizer,
algorithm sweep, reduced prelude, or added device synchronization is enabled.

Physical GPU1 is selected by its preflight-verified UUID. Only the directly
owned executable PID receives the preload; the collector, shell, journal and
inventory tools do not. Both frozen ELF and supplied trace-library SHA are
verified immediately before launching that one child.

## ABI and forwarding qualification

`sm75-runtime-contract-trace.cpp` uses NVIDIA's installed headers, not guessed
vendor structures. CUDA 13.2's `cudaPointerAttributes` is 88 bytes; substituting
an older 24-byte declaration would itself be unsafe. Compilation requires Linux
x86-64, CUDA runtime headers 13020 and the expected structure sizes.

The frozen ELF imports CUDA 13.2's compiler-side `__cudaLaunchKernel` route.
Interposing only public `cudaLaunchKernel` would miss its custom launches.
The header checker validates private host-launch/registration declarations
against the installed toolkit before compilation. The tracer follows registered
host functions to opaque launch handles and recognizes only
`_Z17f32_to_f16_kernelP6__halfPKfm` from the main executable. Unregistration
retires that mapping. Safe host reads collect its three scalar arguments,
never dereference device memory, and never infer values from partial reads.

Driver range/context queries use documented
`cudaGetDriverEntryPointByVersion`, version 13020 and legacy-stream semantics.
This avoids assuming that a locally loaded driver is accessible through
`dlsym(RTLD_NEXT)`, and avoids promoting or reloading it. Runtime/cuBLAS version
observations are saved. The constructor performs host file initialization only.
No CUDA/cuBLAS/driver library is linked into the interposer. Its logging lock
is released before calling the real API.

The host-only build gate checks exported interception symbols and rejects real
GPU-library dependencies before exercising a fake provider. The fake process
tests forwarding, output arguments, scalar pointer modes, recognized/unreadable
conversion arguments, context observations, lifecycle logging, and exclusive
log creation. These are CPU tests, not CUDA functionality tests.

## What instrumentation changes

It adds host logging and metadata queries before relevant calls. They do not
explicitly synchronize the device, but can perturb timing, trigger lazy runtime
or context initialization, or surface an earlier asynchronous error. Each added
query has its own entry/return record. Failure
of a required query stops further forwarding; it is **not** automatically a bad
pointer diagnosis. Missing optional driver-entrypoint coverage remains a gap.
A passing instrumented run does not establish that the original failure is gone.

Only host-mode FP32 alpha/beta bits are read; device-mode scalars are not
dereferenced. Logs contain addresses and execution metadata, not tensor data,
credentials or an environment dump. Treat the diagnostic archive as private
machine information when sharing it.

JSONL is limited to 64 MiB. Creation is exclusive, private and non-symlinked.
Logging/ABI failures stop with status 125 rather than silently forwarding with
missing evidence. `trace_end` means library unload, not GPU health. Following
termination attempts, the collector snapshots a bounded immutable prefix before
hashing/parsing, retaining the original log. An incomplete trace, wrong PID,
missing workload transition, or observed violation cannot silently qualify as a
clean capture. The analysis reports observations and gaps, never "race-free".

## First step: host-only build and fake-provider test

After the reviewed commit is present on the Linux host, this builds only the
tracer and fake-provider artifacts. It does not run the frozen ELF, load NVIDIA
libraries, alter services, query GPU health, or require sudo:

```bash
(
cd ~/ds4-iq2-q4 || exit
bash ./speed-bench/build-sm75-runtime-contract-trace.sh
)
```

Return the resulting `sm75-runtime-contract-build.*.tar.gz` archive, including
if qualification fails. Do not proceed to GPU execution on compilation alone.
Windows cross-compilation against matching official headers checks compilation
and linkage; it does not replace native Linux loading and fake-provider execution.

Development qualification before the first Linux check: 49 existing capture,
24 trace-capture, 85 analyzer, 6 header-gate and 82 shell-runner CPU-only cases
passed (246 total). The tracer, fake provider and fake executable cross-compiled
as Linux x86-64 ELF against official CUDA 13.2/cuBLAS 13.4.0.1 headers; required
tracer exports and absence of NVIDIA-library dependencies were inspected.
Neither those Linux artifacts nor the frozen GPU executable was executed here.

The runner's opt-in options are `RUNTIME_CONTRACT_TRACE_LIBRARY` and
`RUNTIME_CONTRACT_TRACE_SHA256`. Both require `CAPTURE_FAILURE_CONTEXT=1` and
the existing frozen-workload gates. No default changes. A GPU command should
be supplied only after host-only evidence is inspected and one instrumented
execution deliberately selected. No automated retries are added.

## Library-internal ordering is a separate investigation

Application-facing API interposition cannot promise interception of cuBLAS's
hidden driver calls, enumerate internal workspaces, show actual device-kernel
execution intervals, or prove absence of global-memory races. Nested library
calls are deliberately not presented as a complete internal trace.

If the observed API contract is consistent while the same failure persists,
the next useful evidence is a separately qualified process-scoped CUDA/cuBLAS
timeline correlated to these exact calls and saved failure clocks. Nsight is
not wrapped around this experiment implicitly: descendant-process ownership,
bounded shutdown, partial-report preservation and export need their own reviewed
integration. A same-ABI cuBLAS-plus-Lt comparison and a supported driver comparison
remain separate single-variable controls, not a batch of reboot tests or a
reason to introduce another GPU.
