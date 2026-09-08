# GPU1-only PCIe loss: driver and hardware investigation handoff

Prepared 2026-09-08 from the captured 00:36 UTC incident; updated with the
01:29 UTC inventory and 01:47 UTC BMC follow-up. This remains an evidence-qualified
investigation handoff, not a complete physical inventory, diagnosis of a defective
card, or authorization to run more tests. See the inventory/binary supplement.
No evidence has been submitted externally. Raw logs contain host identifiers,
serial numbers, process/library paths and older incident history; review before
posting publicly. Use a private vendor/support channel for the complete package.

## 1. Incident and requested assistance

Latest supplement: `sm75-nsys-loss-findings-20260908.md` records the 17:56 UTC
GPU1-only failure under Nsight, exact raw-archive hashes, subsequent failed
offline imports, and retained injection storage for vendor-assisted recovery.
No device-kernel attribution can be made from that incomplete capture.

New evidence: `sm75-gpu1-runtime-capture-findings-20260908.md` adds the 03:42 UTC
failure on driver 595.91.07/cuBLAS 13.4.1.3/cudart 13.2.86, with the same frozen
ELF. It records in-range live operands, identical failing/successful GEMM argument
sets and completed DEFAULT synchronization, followed by GPU1 Xid 79/root-port
Surprise Down. Device execution inside the conversion/103 interval remains
unresolved. This newer multi-component stack is not a one-variable comparison.
The incident-specific details below continue to describe the original 00:36 run.

A small, local-only CUDA/cuBLAS synthetic workload reproducibly loses physical
GPU1 on a four-Quadro-RTX-8000 host. The latest run has no application peer access,
native-Q8 streaming or row-owned pair execution. DCGM was disabled/inactive.
GPU1's root port acquires Surprise Down and Fatal Error status. GPU0 subsequently
reports a GSP RPC timeout; the driver requests a node reboot for all four GPUs.

Please investigate the initial GPU1 loss and subsequent cross-device driver/GSP
failure separately. We need to distinguish card/slot/root-port/power/firmware
failure from a software-triggered driver or cuBLAS fault. Please do not assume
that an application synchronization error identifies the offending instruction.

Specific questions for the GPU/driver vendor and system integrator:

1. Can the attached report establish why GPU1 became inaccessible, or identify
   reset/power/link/firmware events preceding Xid 79?
2. Is GPU0's subsequent GSP timeout expected containment/recovery behavior after
   this GPU1 loss? What does RPC function 76, control `0x20800a56`, data `0x5c`
   mean in this driver build?
3. Which supported driver/GSP configuration comparison is most diagnostic for
   this exact Turing board and host? Supply a version-specific procedure and
   rollback plan; do not change driver, cuBLAS and BIOS simultaneously.
4. Does the board/VBIOS/BAR1 difference below warrant a configuration review?
   A small BAR1 aperture is an observed difference, not evidence of exhaustion.
5. With firmware-first error handling and OS error reporting disabled, what BMC,
   firmware or platform logs should supplement the latched root-port status?
6. If physical isolation is warranted, specify an approved powered-off card/slot/
   power-cable comparison or field diagnostic. Keep the application reproducer
   frozen and identify cards by UUID, not their possibly changing ordinals.

## 2. Evidence identity and integrity

Primary raw archive:
`sm75-token-row-arithmetic-20260908T003609Z.tar.gz`

SHA-256:
`98e3129856e721284e663c9270b4e9a786daf6b4baf951aa9ac1e8fc00fc74ec`

Archive root: `sm75-token-row-arithmetic-20260908T003609Z/`.
All paths below are relative to that root unless explicitly stated otherwise.

Repository: `https://github.com/Dillflix/ds4`
Branch: `agent/sm75-row-owned-attention`
Captured checkout: `039eef3e756412da0d2aec02ae0af835b2a09caf`.

Frozen Linux executable: `tests/cuda_sm75_token_row_arithmetic`
SHA-256:
`5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
The Linux executable itself is NOT in the original failure archive. Its exact
14,619,104-byte copy is now preserved in the later inventory archive
`sm75-investigation-inventory.kzJxuv.tar.gz`, under
`sm75-investigation-inventory.kzJxuv/artifacts/cuda_sm75_token_row_arithmetic`.
Inventory archive SHA-256:
`c968dcaa24e14c3840185ab61b0c584a87cf2660f994599ae81cdcfb4317eb40`.
The original and copied ELF hashes match the invariant above. Do not rebuild
over it. Embedded compiler/DWARF evidence now supplies partial build provenance;
it does not reconstruct every flag or prove the original source revision.

| Evidence | Purpose |
| --- | --- |
| `manifest.txt`, `provenance/` | Requested scope, checkout, command and binary fingerprint |
| `diagnostic.log` | Application submission/completion checkpoints and comparisons |
| `failure-context/summary.json` | Actual command, clocks, first fault, library fingerprints, collector statuses |
| `failure-context/application-timeline.jsonl` | Host receipt timestamps for application output |
| `failure-context/kernel-live.jsonl` | Current-boot fault sequence with kernel-source and journal clocks |
| `failure-context/kernel-baseline.log` | Pre-workload current-boot kernel baseline |
| `failure-context/pre/`, `post/` | GPU, PCI, service and process snapshots |
| `failure-context/pre-sysfs.json`, `post-sysfs.json` | PCI resources, link state and software AER counters |
| `failure-context/process-timeline.jsonl` | Owned process threads and actual mapped CUDA libraries |
| `failure-context/nvidia-report/nvidia-bug-report.log.gz` | Completed safe-mode NVIDIA system/driver report |

The NVIDIA report also includes OLD log history. Use the boot ID and the
00:36 UTC interval to avoid confusing earlier Xid 31, GPU2 loss, or later-started
nvbandwidth with this incident.

## 3. Host, boards and software

- Host: `dillflix-ai-nvidia`, Ubuntu 24.04.4 LTS, x86_64.
- Kernel: `6.8.0-139-generic`.
- System: ASUS ESC4000 G3, PCIe Gen3 / DDR4 (owner-confirmed). Motherboard:
  ASUSTeK Z10PG-D16 Series; BIOS 3803, dated 2019-08-23 (captured).
- CPUs: two Intel Xeon E5-2630L v4, 10 cores/20 threads each.
- NVIDIA driver and GSP firmware: 595.84. The report identifies the NVIDIA UNIX
  kernel module; do not infer that it is the open kernel module from an unrelated
  `OpenRmEnableUnsupportedGpus` parameter.
- Current boot ID: `9c9247806852474889f012872d1656e6`.
- Kernel command line includes `intel_iommu=off iommu=off pcie_aspm=off
  pcie_port_pm=off nvme_core.default_ps_max_latency_us=0`.
  These command-line strings do not override the directly observed register state:
  GPU1's root port still reports ASPM L1 enabled.

| Physical GPU | Endpoint / upstream port | UUID suffix | VBIOS | Board part | BAR1 total | Power limit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `02:00.0` / `00:02.0` | `0c9d47503c62` | `90.02.4E.00.03` | `900-2G150-0150-030G` | 32768 MiB | 250 W |
| 1, failing | `03:00.0` / `00:03.0` | `59e548ac9227` | `90.02.4A.00.11` | `900-5G150-1700-000` | 256 MiB | 260 W |
| 2 | `81:00.0` / `80:02.0` | `1763abc7584f` | `90.02.4E.00.03` | `900-2G150-0150-030` | 32768 MiB | 250 W |
| 3 | `82:00.0` / `80:03.0` | `0dc3abaad694` | `90.02.4E.00.03` | `900-2G150-0150-030G` | 32768 MiB | 250 W |

All report 48 GiB Quadro RTX 8000, SM75. GPU1's complete identity is
`GPU-ba2c2d0b-6320-580f-208c-59e548ac9227`, board serial `1565020027030`;
lspci reports physical slot 3, NUMA node 0. Physical NVLink connections remain
installed; this CUDA workload does not enable/use application peer access.
That is not a claim that driver or physical interconnect activity is absent.

Captured idle clocks were graphics/SM 300 MHz and memory 405 MHz. GPU1 reports
maximum clocks of 2100/7001 MHz (graphics/memory), versus 1620/6501 MHz on the
other boards. These reported maxima and different board/VBIOS identities do not
establish overclocking or actual load-time clocks. Applications-clock queries
return a deprecation message, not a measured override value.

Pre-run temperatures were 36/37/43/42 C, all GPUs P8, no CUDA compute clients.
These idle samples do not rule out transient electrical/thermal problems.
The owner reports all four GPUs are installed directly in the official ASUS
ESC4000 G3 GPU cages: two cages, each containing two NVLink-paired GPUs. No
aftermarket riser is reported; internal cage/backplane interconnect details are
not independently mapped. VBIOS and clock configuration are unmodified to the
best of the owner's knowledge. The owner confirms PSU1 is intentionally unplugged,
and factory chassis fans are installed, connected and running; fan sensors are
reported unreliable. Do not label expected PSU1 AC-lost/0 W or reported zero fan
RPM as newly established faults. Other workloads reportedly load all four GPUs
successfully, countering a simple aggregate-power-shortage explanation without
establishing this failure's cause. PSU model/rating/input voltage, per-card cable
arrangement, BIOS settings/change history and inspection findings remain unconfirmed.

BMC firmware is 1.14 (auxiliary 02/00/00/00), ASUS product 0x0e73, IPMI 2.0.
At 2026-09-08 01:47:26 UTC the BMC and host clocks agree at displayed precision.
Its SEL has 3,000/3,000 entries, zero free bytes/units, 100% usage and overflow
flagged. Last addition is 2025-09-07 06:18:23 UTC. Thus absent contemporary BMC
events cannot exclude power/thermal/PCIe incidents. Do not redate old records,
clear the log or change firmware as an incidental diagnostic step. DMI PSU
records contain placeholders; FRU inventory failed with header version 0xff.

Actual mapped runtime files (not merely the CUDA version printed by nvidia-smi):

- `libcuda.so.595.84`
- `libcudart.so.13.2.75`
- `libcublas.so.13.4.0.1`, `libcublasLt.so.13.4.0.1`
- NVIDIA compiler/JIT libraries listed with SHA-256 in `summary.json`.

The report also records `EnableGpuFirmware=18`, `EnablePCIeGen3=0`,
`EnableResizableBar=0`, and `PreserveVideoMemoryAllocations=1`. These are raw
parameter values, not decoded causes or recommendations to change them.

## 4. Workload and exact observed checkpoint

Synthetic single-device test; no GGUF model or private prompt is needed.
The harness's `device_working_set_bytes=389283840` label counts only 371.25 MiB
of persistent activation tensors; it is NOT total GPU residency. It excludes
approximately 102 MiB Q8 model copy, 192 MiB canonical resident FP16 weight cache,
up to 48 MiB shared scratch and driver/cuBLAS allocations. These are source-based
components, not a measured peak. FP32 low dimension is 8192;
output dimension is 4096. Full batch is 512 rows, half batch 256. A production
outer prefill chunk of 2048 is not the GEMM N dimension in this reproducer.

Scope: `output-b-production103-no-row-owned`.
No peer, no native-stream, no exhaustive algorithm sweep, no sanitizer.
`POST_BURNIN_PAIR=ab` is supplied to the runner but does not activate a post-burn-in
A+B pair in this scope; the log explicitly says that pair is skipped.

Application sequence:

1. Query projection, RMS/RoPE, mixed attention, inverse RoPE and initial A+B
   full/half comparisons complete. Some separate DEFAULT-vs-shape numerical
   comparisons report their established differences; they are not runtime faults.
2. 1,024 B-only algorithm-103 half0 calls complete, synchronizing every 10 calls
   and at the end. Final readback is bit-exact against its reference.
3. The fixture replaces low-input values with its deterministic structured data.
   Addresses remain the same; data values differ from the burn-in's A output.
4. B-only DEFAULT full512 synchronizes successfully.
5. B-only algorithm103 half0-256 is submitted; device synchronization reports
   unspecified launch failure. Half1 is not reached.

The last half call performs F32-to-F16 activation conversion and
`cublasGemmEx(M=4096,N=256,K=8192)`, FP16 inputs/weights, FP32 output and compute,
beta zero, `CUBLAS_GEMM_ALGO3_TENSOR_OP` (103). There is no output A in this final
interval. Full-to-half scratch use decreases from 8 to 4 MiB without a shrink
allocation. It is an error-observation checkpoint, not an instruction-level trace.

## 5. Fault timeline and PCIe changes

Times below use kernel SOURCE monotonic seconds, not journal receipt timestamps.

| Source monotonic seconds | Event |
| --- | --- |
| 5549.281539 | GPU1 `03:00`: Xid 79, fallen off the bus |
| 5559.283138 | GPU0 `02:00`: Xid 175, 10-second GSP RPC response timeout |
| 5560.961856 through 5560.963252 | Xid 154 sets Node Reboot Required on all four GPUs |

Journal receipt of GPU1's Xid is approximately 00:36:14.171 UTC. The helper starts
the workload at monotonic 5547.162886 seconds. Application timeline timestamps
are output RECEIPT times: observed burn-in completion is 5549.354334 and observed
half0 submission is 5549.402513. Kernel source time is earlier. Output delivery,
scheduling and journal delays prevent a reliable sub-millisecond causal ordering
between CUDA operations and Xid detection. Preserve both clocks; do not rewrite
this as proof that the last GEMM caused Xid 79.

GPU1 upstream `0000:00:03.0` pre/post:

- `UESta SDES- -> SDES+` (Surprise Down); `DevSta FatalErr- -> FatalErr+`.
- First Error Pointer `00 -> 05`; slot-change Presence Detect/Link State bits set.
- Target Link Speed remains 8 GT/s. Current speed reads 2.5 -> 8 GT/s, x16;
  a later readable link state does not negate latched prior Surprise Down.

GPU1 endpoint `0000:03:00.0` pre/post:

- Memory decoding, bus mastering and MSI become disabled; BAR1/BAR3 show disabled.
- Unsupported Request and Advisory Non-Fatal Error status become set.
- AER header: `40000001 0000010f c3000180 f7f7f7f7`.

The other SIX endpoint/root-port lspci snapshots are byte-for-byte unchanged.
GPU0's later ERR state is not accompanied by an observed GPU0 PCIe register change.
Post-fault changes can include recovery activity; snapshots cannot establish
which actor changed each register or what initiated the drop.

Software AER counters remain zero. Root DevCtl and RootCmd error-reporting enables
are disabled. The boot log reports firmware-first handling and `_OSC` not
requesting OS control. Thus absent Linux AER messages/counters do not clear the
link. GPU2's root target is already 5 GT/s before the workload and remains so;
do not describe that as a consequence of this GPU1 event.

## 6. Confounders addressed and important history

- DCGM snap exporter and nv-hostengine: disabled/inactive before this run.
- No nvbandwidth or retrain process in the initial process snapshot.
- Retrain unit: Type oneshot, MainPID=0, successful active(exited); completed
  2026-09-07 23:07:36 UTC, long before the 00:36 test. It retrains ALL four GPUs
  despite its GPU2 name. Startup bandwidth validation passed; that does not prove
  later electrical stability. It was not rerun for this incident.
- No earlier serious GPU fault in this boot's pre-workload kernel baseline.
- GPU1-only failures existed before this capture wrapper. The collector is not
  necessary for the historical failure, though observation can perturb execution.
- Same-hash complete memcheck, initcheck and synccheck runs exist from September 7
  (183344Z, 191251Z, 192738Z). They do not establish uninstrumented correctness.
- Racecheck 195635Z timed out at 560/1024; it is NOT a completed clean result.
- An earlier separate GPU2 Xid79 occurred at September 7 20:17:24. Its cause remains
  unresolved. Do not conflate it with the latest GPU1 event.
- User reports nvbandwidth began only AFTER an earlier GPU1 failure. A later
  nvbandwidth process in historical logs is not evidence of a preceding cause.
- A prior native-Q8 borrowed-view relocation bug produced a separate local A+B
  fault and was fixed. The current canonical-FP16 reproducer does not traverse
  that native cache path; this does not retroactively invalidate the prior bug.

## 7. Capture quality and missing evidence

All recorded pre/post collector commands returned zero; the safe-mode NVIDIA
report completed and is present. The overall capture correctly says PARTIAL:
the helper could not confirm owned-child termination in its bounded stop window,
and its application-output reader did not finish in the bounded join window.
It does not report a successful CUDA exit. Importantly, `post/processes.log`
later shows the owned PID 4312 as `Zs` (exited/zombie), awaiting reaping. The
original summary retained its earlier `workload_returncode=null` and
`workload_terminated=false`. That metadata was not refreshed after the post-fault
collection interval. Pre-fault sampled threads were running/sleeping, not D-state.
The evidence therefore does NOT show a permanently uninterruptible process.
Later driver teardown warnings appear in the journal. The host collector's
late-reaping bookkeeping is a separate software issue, not the GPU-loss cause.

Not collected: GPU instruction/API/stream trace; runtime pointer/allocation
generations; detailed transient rail/power/temperature telemetry; definitive
firmware-first error record; verified PSU/cable/internal-cage routing; complete
original build provenance. The ELF has since been preserved and inspected, and
the full BMC log explains the lack of recent SEL evidence. Safe mode skips some dynamic
queries. A report file's existence does not guarantee recovery of every internal
GPU crash dump. These are explicit evidence gaps, not negative results.

## 8. Frozen reproducer recipe (reference only; do not auto-run)

This is the already executed, known-dangerous test. It can require node recovery.
Do not run it on a degraded host or as an automatic loop. A fresh run needs a
specific discriminating hypothesis, an agreed stop condition and user approval.

```bash
env -u CUDA_VISIBLE_DEVICES -u TOKEN_ROW_ARITHMETIC_DIR \
CUDA_DEVICE_ORDER=PCI_BUS_ID PROFILE_GPU=1 CUDA_ARCH=sm_75 \
DIAGNOSTIC_SCOPE=output-b-production103-no-row-owned POST_BURNIN_PAIR=ab \
OUTPUT_B_PRODUCTION103_CALLS=1024 OUTPUT_B_PRODUCTION103_BATCH=10 \
B_TIMING_ROUNDS=7 B_TIMING_REPEATS=10 B_TIMING_WARMUPS=3 \
CASE_TIMEOUT_SECONDS=600 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
SANITIZER_TOOL=memcheck SKIP_BUILD=1 CREATE_ARCHIVE=1 \
CAPTURE_FAILURE_CONTEXT=1 \
bash ./speed-bench/cuda-sm75-token-row-arithmetic.sh
```

The wrapper pins GPU1's UUID and binary hash, requires a healthy baseline,
disabled DCGM, completed retrain unit and cached sudo for HOST collection.
The CUDA workload is not run as root. Its limit is 600 seconds; post-failure
report collection has a separate 120-second limit. It does not ensure that a
driver-blocked task can be killed. Exact recipe provenance is in the raw archive.

## 9. Investigation decision matrix; no changes executed

| Avenue | Distinguishing evidence sought | Guardrails |
| --- | --- | --- |
| Driver/vendor report analysis | Initial failure vs secondary GSP propagation; relevant known defects | Analyze saved report first; no fresh workload needed |
| Platform/BMC inspection | Firmware-first PCIe/power records, riser/cabling/slot anomalies | Read-only inventory first; powered-off inspection only by owner/technician |
| Driver-only controlled comparison | Does exact local case depend on driver/GSP version? | Vendor-supported stack and rollback; keep binary, cuBLAS, inputs and hardware fixed |
| Library-only comparison | Does GEMM implementation selection change the failure? | Check supported driver/runtime combination; record mapped library hashes, not PATH alone |
| Runtime API/stream/lifetime trace | Actual ordering, algorithm, scratch producer/consumer and buffer lifetimes | Separately labeled instrumented run; no peer; a passing trace is not clearance |
| Card/slot/power-path comparison | Does fault follow UUID or physical path? | Requires physical-work approval; one change at a time; update ordinal mapping safely |
| Vendor field diagnostic | Independent hardware qualification | Vendor-directed; do not substitute another broad stress loop |

Do not simultaneously disable GSP, change PCIe speed, lower power, upgrade BIOS,
swap cards and replace cuBLAS: the result would not distinguish a cause. Do not
enable pair0 production attention or add a peer as an isolation shortcut. Do not
force OS PCIe ownership, issue live setpci retrains or reset individual linked
devices as part of this handoff. Raw BAR/register differences are not tuning advice.

## 10. References and parallel software investigation

- [NVIDIA Xid interpretation](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html):
  Xid79 means driver access over PCIe failed; hardware and driver causes are possible.
- [NVIDIA issue reporting and field diagnostics](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html):
  give the system vendor the incident, configuration, debug history, logs and report.
- [NVIDIA report collection](https://docs.nvidia.com/deploy/xid-errors/working-with-xid-errors.html):
  safe-mode report is a supported collection alternative; it has coverage limits.
- [Linux AER ownership/reporting](https://www.kernel.org/doc/html/latest/PCI/pcieaer-howto.html).

See the accompanying software audit/status documents for code-level bounds,
initialization, scratch ordering, CUDA/cuBLAS support and remaining runtime gaps.
No root cause is established and no production path is qualified by this handoff.

## 11. Completed no-workload inventory and remaining gaps

The owner ran `collect-sm75-investigation-inventory.sh` with host inventory and
ELF preservation enabled. It completed in 31 seconds without executing CUDA.
The original archive remains unchanged. Results are analyzed in
`sm75-gpu1-inventory-binary-supplement-20260908.md`:

- Host metadata is preserved; FRU is the sole failed inventory query. Physical
  PSU identity/rating/input voltage and cable routing are still not established.
- The subsequent read-only SEL capacity/clock query explains why current BMC
  events are unavailable. Repeating an old-log query cannot recover missing events.
- The exact ELF now supplies SM75 cubin/PTX, compiler-banner/DWARF, ABI dependency,
  selected host-call disassembly and conversion-kernel SASS evidence.
- Current runtime packages agree with mapped versions; installed Nsight Systems
  capabilities are known. Package transaction history/vendor hash verification
  and historical inherited environment values are not established.
- Runtime streams, pointers/lifetimes, actual GEMM arguments and library kernel
  selection remain unmeasured. No peer is needed for these software questions.

No repeat of that inventory or GPU run is requested here. The updated package
retains explicit gaps rather than treating query success as a complete hardware
history. A future trace needs a separately reviewed collector mode and stop
conditions; the current helper's profiler/injection guards remain in force.
Review host identifiers, serial numbers and executable build paths before any
external sharing; use a private vendor/support channel for the complete evidence.
