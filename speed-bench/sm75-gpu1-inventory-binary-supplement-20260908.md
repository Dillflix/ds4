# GPU1 investigation: inventory and compiled-binary supplement

Updated 2026-09-08 after the 01:29 UTC host-only inventory and the owner's
01:47 UTC BMC follow-up. This supplements the preliminary incident handoff;
it is not a root-cause finding, hardware clearance, or request for another run.
No GPU workload, ELF execution, rebuild, driver change, reset, link retrain,
service change or external submission was performed for this review.

## Evidence identity

Inventory archive: `sm75-investigation-inventory.kzJxuv.tar.gz`, 8,943,118 bytes.
SHA-256: `c968dcaa24e14c3840185ab61b0c584a87cf2660f994599ae81cdcfb4317eb40`.
All 42 archive members were checked for ordinary file/directory types and safe,
unique paths before extraction. Collection ran 01:29:52 through 01:30:23 UTC,
31 seconds, `workload_executed=0`. It completed with one explicitly recorded gap:
`bmc-fru` returned 1 with `Unknown FRU header version 0xff`. Other queries
succeeded. A successful inventory exit does not mean all optional facts exist.

The copied Linux ELF is 14,619,104 bytes. Original, archived copy and local
inspection copy match the failure's pinned SHA-256:
`5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
It is included at `artifacts/cuda_sm75_token_row_arithmetic` inside the inventory
archive. The earlier 00:36 failure archive contains its hash, not its bytes.

## BMC evidence gap now explained

The following is transcribed from the owner's read-only command output, not
retroactively added to the 01:29 inventory archive:

| Item | Owner's 01:47 UTC follow-up |
| --- | --- |
| Host UTC | 2026-09-08 01:47:26 |
| BMC clock | 2026-09-08 01:47:26, same displayed second |
| BMC identity | ASUSTek, manufacturer 2623, product 3699 / 0x0e73 |
| Firmware / IPMI | Firmware 1.14, auxiliary 02/00/00/00; IPMI 2.0 |
| SEL entries / allocation units | 3,000 / 3,000 |
| Free bytes / free units | 0 / 0 |
| Percent used / overflow | 100% / true |
| Last addition | 2025-09-07 06:18:23 UTC |
| Last deletion | 2025-03-05 22:38:10 UTC |

This resolves the stale-log-versus-clock question: the current clock is correct
at displayed precision, and the event store is full with overflow flagged.
It has no recorded additions for a year. The September 2025 timestamps must
not be reinterpreted as September 2026 events. Nor can the lack of current SEL
power, thermal or PCIe entries be used to exclude those events. The checks do
not establish when the store first filled or its complete historical clock state.

Preserve the existing SEL and this status. Do not clear it, change its time,
reset the BMC, or update firmware as an incidental diagnostic step. If fresh BMC
event capture becomes part of a future approved experiment, first address log
capacity/retention explicitly after preserving the records. Clearing it would
not recover missing records. These findings do not require another GPU run.

The original SEL's many old fan/temperature entries are not contemporary
failure telemetry. The owner confirms that factory chassis fans are installed,
connected and running, and that the sensors are unreliable. Report zero RPM
readings as unreliable telemetry, not as proof of stopped fans or bad cooling.

PSU1 is intentionally unplugged. Its AC-lost/0 W readings are expected in that
configuration, not a newly discovered PSU failure. PSU2 reported approximately
232-240 W during the later idle inventory. The owner reports other workloads
can load all four GPUs successfully while this reproducer uses only GPU1.
That is meaningful counterevidence to a simple aggregate-wattage explanation;
it does not identify the cause or measure per-card transient power integrity.
There is no basis here to assign the failure to power or cooling.

DMI type 39 provides placeholder identity fields and unknown capacity; the FRU
query failed. PSU model/rated output, AC input voltage and exact per-card cable
routing remain unconfirmed. Do not substitute a possible chassis SKU rating.
These physical inventory gaps do not prevent continued local software analysis.

## Offline inspection of the actual failing binary

The pinned ELF was parsed/disassembled as data, never launched. The reproducible
`inspect-frozen-elf.py` and its JSON result accompany the revised package.
pyelftools 0.32 and Capstone 5.0.6 were installed only in the local analysis
directory. All 142 selected PLT relocation mappings were checked against their
GOT addresses; complete selected host functions were decoded before excerpting.

- ELF64 little-endian x86-64 PIE; GNU build ID
  `02077fa6432b057d5ade564032fb7d816ea55045`.
- Dependencies include `libcudart.so.13` and `libcublas.so.13`; no RPATH/RUNPATH.
  These agree at major-ABI level with the failure's actually mapped libraries.
- DWARF identifies GCC 13.3.0 Ubuntu and Broadwell host compilation at O3 with
  debug information. The C fixture reports C99, fast-math and
  `-fno-finite-math-only`; generated CUDA host units report C++17.
- The ELF itself embeds the CUDA 13.2.78 / compiler.37668154 banner, agreeing
  with the current tool inventory. This is stronger than today's NVCC version
  alone, but does not reconstruct every original flag or source revision.
- Two embedded SM75 cubins and SM75 PTX 9.2 are present. No wrong-architecture
  explanation is demonstrated. Compatible cubin availability does not establish
  whether runtime loading selected cubin or PTX.
- No `_ptds` / `_ptsz` imports or application imports of `cublasSetStream`,
  `cublasSetWorkspace` or `cublasSetPointerMode` were found. Optional peer APIs
  elsewhere in the binary do not establish peer use in the selected scope.

### Host GEMM call path and adapter

Addresses below are static PIE virtual offsets, not runtime pointer values.
The resident-FP16 path in the generic B wrapper is excerpted at
`0x101c70..0x101f48`; its call at `0x101e65` reaches the typed cuBLAS adapter at
`0x3b660`. It supplies alpha 1 and beta 0 and passes the selected algorithm
variable, rather than hardcoding DEFAULT at this common call site.

The adapter calls `cublasGetMathMode` at `0x3b6b1`, checks its result, and maps
the source's legacy FP32 datatype overload into the compute-type API. Depending
on the queried math mode it selects compute enum 0x44 or 0x45 (FP32 variants),
and passes FP16 A/B types. Its `cublasGemmEx` library call is at `0x3b71f`.
There is no evidence here of blindly passing the old datatype numeric value
as the new compute enum. Actual math mode and dynamic matrix/pointer arguments
remain runtime questions, not measurements supplied by disassembly.

The `ds4_gpu_synchronize` wrapper at `0x9a410` calls
`cudaDeviceSynchronize` at `0x9a418` and checks its returned status. This supports
the source-level checkpoint interpretation, not which kernel caused the error.
The generic wrapper also contains a separate native-dequant DEFAULT fallback;
that branch must not be mistaken for this resident-FP16/selected-103 path.

### Compiled conversion kernel

NVIDIA cuobjdump/nvdisasm 13.2.78 inspected the embedded
`_Z17f32_to_f16_kernelP6__halfPKfm` SM75 machine code. It forms a 64-bit linear
thread index, compares both halves against the count before memory access,
loads one FP32 value at source + 4*index, converts it and stores one FP16 at
destination + 2*index. Address arithmetic carries the high bits. This matches
the source's fixed-shape conversion bounds; no shared-memory or barrier use
is present in this kernel. The saved disassembly is included in the package.

This narrow compiled-code check did not expose a conversion indexing defect.
It cannot validate dynamic pointer/count values, allocation lifetime, a race at
valid addresses, the body of a cuBLAS kernel, or hardware behavior. A passing
static inspection is not clearance to run pair0 or introduce a peer GPU.

## Runtime inventory and remaining software evidence

Installed package versions agree with the previously mapped files:
CUDA runtime 13.2.75, cuBLAS 13.4.0.1, NVIDIA 595.84. Package ownership alone is
not vendor-file integrity verification or package transaction history.
The current allowlisted environment reports `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`LD_LIBRARY_PATH=/usr/local/cuda/lib64:`, and no listed JIT/workspace controls.
This later snapshot cannot reconstruct the failure process's inherited values.

Installed Nsight Systems is 2025.6.3.541-256337736014v0. Its captured help
advertises process-scoped `cuda-sw`/`cublas` tracing, all-API and allocation
tracking, CPU sampling disabled, event tracing explicitly disabled, and a
configurable flush interval. This removes guesswork about available CLI options;
no profiler session has been launched or qualified for fault-time recovery.

The next software work should preserve the known GPU1-only sequence and design
one separately labeled trace with these concrete questions:

1. Which kernels and streams implement the final DEFAULT512 and first 103/256
   calls, and when does conversion finish relative to consumers/overwrites?
2. Do runtime base/size/generation and view ranges agree with the offline model?
3. What handle math/stream/pointer state and effective GEMM arguments are used?
4. Are records flushed before device loss, and can owned-process cleanup and
   evidence retention be verified without adding an automatic rerun?

The existing collector rejects profiler/logger injection. A new labeled mode
needs review and CPU mock tests before any run recommendation. Do not remove
those guards, combine profilers casually, add a peer, shorten the reproducer,
or call an instrumented pass proof of uninstrumented stability. Library/driver
comparisons and vendor-private analysis remain independent open avenues.

## Offline tooling provenance and references

Official NVIDIA redistribution manifest:
[CUDA 13.2.1 redistributions](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.1.json).
Downloaded tool ZIP hashes were verified before local file inspection:

- cuobjdump 13.2.78 Windows, 6,190,179 bytes:
  `f41fd7ecb9de9db155a56aaf47d93ae407b73684ec434edd17c2637ddef9563e`.
- nvdisasm 13.2.78 Windows, 4,623,544 bytes:
  `6fa14f9f0a4c6ac85cc1ed07d412c05100f2e9a89860857ef0b04e0fc25945b1`.

Tools were confined to the Windows analysis directory; the Linux CUDA host was
not changed. Third-party tools are not redistributed in this evidence package.

- [NVIDIA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).
- [CUDA 13.2 cuBLAS API](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html#cublasgemmex).
- [Nsight Systems tracing and perturbation limits](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- [ipmitool read-only SEL information/time commands](https://github.com/ipmitool/ipmitool/blob/master/doc/ipmitool.1.in).

The handoff is inventory-updated but still has explicit physical and runtime
evidence gaps. No root cause is established and no software avenue is declared
exhausted. Missing old BMC events cannot now be recovered by re-querying them.
