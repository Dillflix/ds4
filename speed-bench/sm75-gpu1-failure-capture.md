# GPU1 reproducer: host-side failure capture

This is an opt-in capture mode for the **existing GPU1-only failure case**, not
a new isolation workload. It does not introduce a peer, a different GEMM,
sanitizer, CUDA logger, GPU core-dump flag, kernel filter, or synchronization.
The executable must remain SHA256
`5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
No CUDA source, Makefile, or binary prerequisite changes accompany this mode.

The September 7 racecheck run timed out after 560/1024 burn-in calls; it did not
produce a completed zero-hazard result. Its separate GPU2 Xid 79 is not evidence
that GPU1 completed successfully. The next comparison is uninstrumented.

## What another failure can tell us

| Evidence | Question it addresses |
| --- | --- |
| Application log with host UTC/monotonic receipt timestamps | Which existing submission/completion checkpoint preceded the fault? Did progress stop before the kernel reported it? |
| All-GPU kernel journal, boot ID and journal timestamps | Which PCI device reported the **first** Xid/AER event, and in what order did subsequent faults occur? |
| Pre/post endpoint and upstream-port `lspci -vv`, sysfs AER counters, link state | Did link status, target speed, Surprise Down status, or error counters change during this window? |
| Pre/post `nvidia-smi -q` | BAR1, PCIe, driver and GPU state before the test versus after it; unavailable fields remain evidence gaps. |
| One-second snapshots of the owned process's threads, wait channels and mapped CUDA libraries | Was the application still runnable or waiting in the driver? Which actual CUDA/cuBLAS library files were mapped? |
| Bounded NVIDIA report after failure | Preserve driver/kernel/system evidence **before** another restart removes it. |

Application timestamps describe host receipt of already-flushed output, **not
GPU instruction execution time**. The last checkpoint or Xid does not by itself
identify the offending instruction. AER counters can be absent, reporting can be
disabled, and post-fault state may include driver recovery actions. Missing AER
messages do not prove a healthy link. This capture is not a global-memory race
detector or a CUDA API/stream trace; those require a separately labelled,
instrumented comparison and must not silently replace this workload.

## Preconditions and invariants

`CAPTURE_FAILURE_CONTEXT=1` requires physical GPU1, scope
`output-b-production103-no-row-owned`, 1024 calls, batch 10, unchanged timing
defaults, `SKIP_BUILD=1`, both sanitizer flags off, and archive creation on.
The helper fails closed on a different executable fingerprint.

Before launching it requires:

- Cached `sudo` authorization (`sudo -v` in the launching terminal).
- Python 3.11+ for separate process groups that retain the authenticated terminal
  session. The captured host is Ubuntu 24.04; its system Python normally meets
  this requirement. The helper checks before launching any GPU workload.
- Both DCGM snap services disabled/inactive and no `nv-hostengine`,
  `nvbandwidth`, or retraining process in the initial process snapshot.
- The existing retraining unit successfully finished: `active (exited)`,
  `Type=oneshot`, `MainPID=0`, successful exit. **Do not disable or restart this
  service for this comparison.** Its prior successful run ended at 18:14:21,
  before the 19:56:35 diagnostic and 20:17:24 GPU2 fault; overlap was not shown.
- All four expected GPUs accessible, physical GPU1's UUID matching the failing
  device, GPU1's established 260 W limit, no CUDA compute clients, and no earlier serious GPU/PCIe fault in the
  current boot. Do not run this on a host still affected by GPU loss.
- A readable kernel journal and cursor, so live capture includes events between
  the baseline and the workload. No GPU workload starts if these checks fail.

The helper records service configuration/history; it never changes services,
power limits, PCIe configuration, AER counters, or GPU state. There is no reset,
retrain, retry, peer workload, or concurrent bandwidth test. Pre/post NVML queries
are bounded; there is **no continuous all-GPU NVML polling** during execution.
This reduces monitoring traffic but does not make observation zero-overhead.
Process snapshots do not prove that no other process started between snapshots.

The initial capture commit mistakenly used `start_new_session=True` for sudo
collectors. That detached them from the terminal/session where `sudo -v` had
authenticated; the 23:24:03 run stopped at `pre/sudo` with `sudo: a password is
required`, before the CUDA executable launched. The fix uses `process_group=0`
and `start_new_session=False` for collectors: isolated group cleanup without
losing terminal-session authentication. It covers the live journal too, not
just the first check. No sudoers changes, password piping or root benchmark are
required. See [Python's subprocess process-group documentation](https://docs.python.org/3/library/subprocess.html)
and [sudo's timestamp/session documentation](https://github.com/sudo-project/sudo/blob/main/docs/sudoers.man.in).

While the owned workload runs, the helper renews the existing authorization with
bounded `sudo -n -v` once per minute, so a short sudo timestamp timeout does not
silently remove post-fault collection access. There is no password prompt or
background renewal after the workload ends. Renewal failure stops the workload
and marks collection partial. Preflight failures now include the collector's
stderr in the console as well as retaining it in the archive.

## Failure handling and artifacts

The helper launches the existing command once in its own process group. It
requests termination on a new Xid/serious PCIe or hardware fault **on any GPU**,
loss of live collection, a user interrupt, or the workload timeout. It signals
only its owned child group, with bounded TERM/KILL waits. No software timeout
can guarantee termination of an uninterruptible kernel/driver task; inability
to terminate is recorded, not treated as success.

After failure it runs one bounded
`nvidia-bug-report.sh --safe-mode --extra-system-data` in the capture directory.
This post-fault collection can query all GPUs. Its own output, status, timeouts
and partial report are retained. It never launches another CUDA benchmark.
See [NVIDIA's Xid collection guidance](https://docs.nvidia.com/deploy/xid-errors/working-with-xid-errors.html).

The **600-second limit is for the workload**, not the entire wrapper. Initial
inventory and final collection take additional time; the NVIDIA report has a
separate 120-second limit. A failed driver query is bounded as well. The live
journal has its own safety timeout. A hard power loss cannot guarantee that
buffered filesystem writes or final archive creation survive.

The existing archive now includes `failure-context/`:

- `summary.json`: workload return code/reason, separate collection completeness,
  first fault with physical PCI identity, boot ID, exact executable hash and
  observed CUDA library paths/hashes. `exited-zero` here is the process outcome;
  the existing runner still enforces arithmetic/completion validation afterward.
- `application-timeline.jsonl`, `process-timeline.jsonl`.
- `kernel-baseline.log`, `kernel-live.jsonl`, `kernel-live.stderr.log`.
- `pre/`, `post/`, pre/post sysfs JSON snapshots, and `nvidia-report/` if failed.

Every bounded query retains stdout/stderr and a result in `summary.json`.
Nonzero application exit, timeout, and incomplete collection cannot become a
passing result. Preflight rejection still gets an archive but no GPU workload.
The report contains host configuration and process/library names: review it
before sharing outside this diagnostic collaboration.

## Run after the capture commit is available

Use the existing runner with these settings, after checking out/pulling the
approved capture commit and recovering from any prior GPU fault. Do not rebuild
if the hash gate rejects the executable; return the rejection archive instead.

```bash
sudo -v &&
sudo nvidia-smi -i 1 -pm 1 &&
sudo nvidia-smi -i 1 -pl 260 &&
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

CPU-only verification: `python3 tests/test_gpu1_failure_capture.py` and
`bash tests/test_token_row_runner.sh`. These use fake host/driver collectors and
synthetic CPU children; they do not qualify GPU stability or a Linux driver's
response to termination. No production CUDA workload is run by either suite.
Windows tests assert the POSIX launch arguments but cannot validate a real
Linux terminal or sudo authentication policy. The initial mock-only tests did
not cover the terminal/session distinction; that coverage gap is now explicit.
