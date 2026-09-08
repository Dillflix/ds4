# Opt-in Nsight integration with the frozen GPU1 collector

Status: integrated native preflight passed at 17:44 UTC. The 17:56 UTC GPU1 run
reproduced loss but **did not retain an importable device report**; both offline
imports failed. The subsequent monitoring/shutdown revision is CPU-tested only.
CUDA/cuBLAS correlation and GPU-loss retention remain unqualified. This is not
a production fix or permission to run a peer/algorithm-sweep experiment.

The reviewed `sm75-nsys-retention-18q7z_z2.tar.gz` passed normal, abrupt and timeout
CPU fixtures. All 100 artifact hashes, actual target PID/starttime across exec,
0600 gate records and three exported SQLite databases were independently checked.
The new collector reuses its unchanged launch gate, pidfd ownership, prefix-store
and bounded export code. It does not require repeating those CPU fixtures.

## What changes, and what does not

`NSYS_CAPTURE=1` selects `capture-sm75-nsys.py` through the normal arithmetic
runner. The default is still zero. `NSYS_PREFLIGHT_ONLY=1` executes the integrated
host preflight and exits before profiler/workload launch; the outer runner still
archives the result. Both modes require the frozen GPU1 no-row-owned workload,
1,024 calls in batches of 10, 7/10/3 timing, no sanitizer, SKIP_BUILD=1 and archive
creation. Native-stream, peer execution, a shortened prelude and an algorithm
sweep are not enabled. No CUDA source, binary or production default is changed.

- Pin frozen ELF SHA `5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
- Require the exact reviewed qualification archive SHA
  `47cf0ba76e0d2a3f535af483106ea23f3ff38f5c6daee59a179299c833cd14c1` and unchanged
  qualified gate/retention/supervision sources. Reject other receipts.
- Pin all seven CUDA/driver library hashes recorded in the 03:42 failure, the
  qualified Nsight entry point and observed permanent profiler components.
  Require running NVIDIA module 595.91.07, reviewed CUDA aliases, historical
  `LD_LIBRARY_PATH=/usr/local/cuda/lib64:` and other recorded controls unset.
- Inventory additional static Nsight bundle libraries before launch. These are
  newly inventoried components, not GPU-qualified components. Actual target maps
  and on-disk mapped-path hashes are recorded; unexpected/mismatched driver/CUDA
  mappings stop the run. Post-exec observations cannot preclude every library
  initialization that happened before the first observation, and on-disk hashes
  are not hashes of the process's executable memory pages.
- Keep the existing sudo/session handling, DCGM/retrain checks, four-GPU health
  inventory, GPU1 UUID/260 W checks, live kernel source timestamps, PCI/AER state,
  bounded NVIDIA report and no competing workload checks. No settings are changed
  by the Python collector. An already faulted boot is rejected.
- Verify the launch gate's owned live PID, UID/starttime, nonce, argv and Python
  executable before releasing it. Verify the frozen ELF's same PID after exec.
  CUDA profiling may load injection libraries into the gate: record/check those
  maps rather than falsely declaring the profiler process to be the target.
- During actual ELF execution, queue mapped-file hashes to a bounded-job worker;
  record mapping observations separately from completed hashes. A hash mismatch
  or worker failure requests stop, and pending hashes at bounded join make the
  evidence incomplete. File metadata changes during hashing are rejected. This
  remains on-disk validation, not executable-memory-page attestation. Gate hashes
  and pinned-file checks still complete before release.
- Stop the verified target through pidfds first on a fault, capture failure,
  interrupt or deadline; then bound profiler finalization/owned-tree cleanup.
  Allow at most 30 seconds for profiler finalization ONLY after target exit is
  observed. A stuck/unverified target gets no such grace, and repeated cleanup
  never restarts the grace period. Record signals, target exit, profiler deadline,
  return status and remaining processes in `nsys/shutdown-timeline.jsonl`.
  A separate single-owner prefix worker copies during execution, finalization
  and postmortem, including a final copy after cleanup. Kernel/PCI
  postmortem is attempted even if profiler cleanup encounters an error.
- Never fill `workload_returncode` from the profiler return code. It remains null
  for this grandchild, with an explicit source label. Success requires verified
  target exec/exit, zero profiler status, the unchanged harness success marker,
  no recorded faults/issues, and completed export/mapping coverage. This still
  does not mark CUDA correlation or GPU-loss behavior qualified automatically.

The selected tracing is `cuda-sw,cublas-verbose`, process-tree scoped, all CUDA
APIs and exposed memory activity, with a 100 ms flush interval. CUDA event tracing,
CPU sampling/context switches, GPU metrics/video, system-wide tracing and replay
are not selected. Do not combine this with our LD_PRELOAD interposer or sanitizer.
**Instrumentation is a changed observation method**: it affects loading, timing,
memory, temporary-file traffic and potentially profiler-inherited process state.
There is no claim that Nsight inserts no internal work or leaves scheduling intact.
The installed version's archived help supports these flags. A 100 ms flush cannot
guarantee survival of the sub-millisecond failing tail or unfinished kernel records.

## Archived preflight command (not a request for another GPU run)

The command below remains available for host-only checking. Do not flip its
preflight flag and repeat the failed GPU run merely because CPU tests pass.
First resolve how actual CUDA records can be retained/imported; CPU fixture
exports do not establish that. See `sm75-nsys-loss-findings-20260908.md`.

After this change is pushed and pulled, run:

```bash
(
cd ~/ds4-iq2-q4 || exit
git switch agent/sm75-row-owned-attention || exit
git pull --ff-only || exit
sudo -v || exit

env -u CUDA_VISIBLE_DEVICES -u TOKEN_ROW_ARITHMETIC_DIR \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PROFILE_GPU=1 CUDA_ARCH=sm_75 \
DIAGNOSTIC_SCOPE=output-b-production103-no-row-owned \
POST_BURNIN_PAIR=ab \
OUTPUT_B_PRODUCTION103_CALLS=1024 OUTPUT_B_PRODUCTION103_BATCH=10 \
B_TIMING_ROUNDS=7 B_TIMING_REPEATS=10 B_TIMING_WARMUPS=3 \
CASE_TIMEOUT_SECONDS=600 \
SANITIZER_ONLY=0 RUN_SANITIZER=0 SANITIZER_TOOL=memcheck \
SKIP_BUILD=1 CREATE_ARCHIVE=1 CAPTURE_FAILURE_CONTEXT=1 \
NSYS_CAPTURE=1 NSYS_PREFLIGHT_ONLY=1 \
NSYS_QUALIFICATION_ARCHIVE="$PWD/sm75-nsys-retention-18q7z_z2.tar.gz" \
bash ./speed-bench/cuda-sm75-token-row-arithmetic.sh
)
```

Return the printed archive, including failure. This performs host-side queries,
file hashing and a journal-reader start/stop, but **does not launch the profiler
against a target, the CPU fixtures or the GPU executable**. No cold restart is
requested for this check. Do not rebuild a stale/mismatched frozen binary to
satisfy preflight; return the rejection evidence instead.

Any later GPU run requires a separately reviewed recovery/coverage plan.
Do not set `NSYS_PREFLIGHT_ONLY=0` based only on this document.
The real run retains the 600-second total supervision deadline, plus bounded
stop, postmortem and export. Host file hashing/archiving adds time; this is not a
hard end-to-end wall-clock guarantee on a failing OS/device stack.

## Evidence and interpretation

- `failure-context/summary.json`: pins, controls, admitted/actual target identity,
  actual mapped-path hashes, separate profiler status/target-exit observation,
  first kernel fault and export result.
- `mapping_hash_results` records asynchronous on-disk hash completion times;
  `actual_target_mapping_observations` records the earlier sampling times. Never
  substitute a queued observation for a completed hash or missing library map.
- Existing kernel, process, PCI/AER and NVIDIA-report artifacts remain present.
- `application-timeline.jsonl`: timestamped **mixed profiler/target stdout chunks**,
  not falsely labeled application API timestamps.
- `nsys/live-tmp` and `nsys/retained-prefixes`: private Nsight temp directory and
  nontransactional retained prefixes (16 MiB/file, 128 MiB prefix total, 64 paths).
  Live size checks are periodic, not an OS filesystem quota. Originals are not
  destroyed when a size limit is hit; collection fails with partial evidence.
- `nsys/report.nsys-rep`, `nsys/export`: report copies, bounded SQLite export,
  actual schema inventory. Export is deferred if owned exit is not established.
  Missing/incomplete export is an issue, never synthetic successful trace data.

The worker intervals are best-effort: blocked filesystem reads, Python scheduling
or OS failure can still delay capture. Worker joins are bounded at two seconds;
unfinished work is explicitly incomplete. These changes remove synchronous
library hashing and prefix copying from the supervision loop, not every possible
host-side delay, and do not prove the previous five-second grace caused data loss.

Next analysis must establish actual CUDA/cuBLAS/kernel correlation, context/stream
identity, launch geometry, relevant transition coverage and how much of the failing
tail survived. Absent records cannot prove the next kernel never ran. Neither an
API/device timeline nor in-range pointers proves absence of valid-address global
memory races or proprietary library workspace corruption. Keep the driver/hardware
handoff open; do not substitute a peer test for the known GPU1-only reproducer.
