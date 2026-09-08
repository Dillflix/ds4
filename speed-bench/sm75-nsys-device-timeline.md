# GPU1 device timeline: qualification before GPU execution

Status: **host-only supervision qualification implemented; CUDA capture is not
yet enabled or qualified.** The original runner, frozen ELF and production code
are unchanged. Do not wrap the reproducer in an ad-hoc `nsys profile` command.

The [03:42 UTC trace](sm75-gpu1-runtime-capture-findings-20260908.md) establishes
recorded caller contracts and DEFAULT completion, but cannot distinguish the
half-conversion's device execution from algorithm-103 execution.

## First qualify process ownership and shutdown without GPU work

The earlier inventory already preserved `nsys --version` and full profile help:
Nsight Systems 2025.6.3.541. No new installation is requested. Because the host
stack changed, the gate records the current CLI version, resolved entry-point
hash and help. That hash does not fingerprint every profiler bundle component.

The current failure collector directly owns the frozen ELF. Under Nsight the
direct child is a profiler; watching only that process can misattribute status,
miss loaded libraries, leave the actual target alive or lose unfinished output.
Establish the lifecycle with CPU fixtures before integrating the CUDA sequence.

`qualify-sm75-nsys-host.py`:

- Executes installed Nsight metadata queries and three Python CPU fixtures:
  normal exit, abrupt SIGKILL exit, and a TERM-ignoring timeout.
- Uses `--trace=none`, no CPU sampling/context switches, GPU metrics/video or
  GPU context-switch collection. No CUDA APIs, compiler, sanitizer, GPU test,
  power/clock changes, reboot, service changes or sudo commands are requested.
- Has **no option or code path to execute the CUDA reproducer**.
- Copies/hashes its source and CPU fixture into a new private directory.
- Validates readiness nonce, live ancestry, PID/starttime/UID, actual command
  line and executable identity. Rejects GPU libraries in the CPU fixture.
- Signals only observed owned processes through Linux pidfds. No process-name
  matching, system-wide cleanup or potentially reused numeric-PID signals.
  An unobserved/reparented target fails qualification; ownership is not guessed.
- Keeps profiler/target identity separate. On timeout, signals the verified CPU
  target first, allowing the profiler to finish, then bounds owned-tree cleanup.
  `--kill=none` disables Nsight's target-process-group termination policy.
- Caps console retention at 4 MiB and checks artifact limits of 256 MiB each/
  512 MiB total. Periodic checks are not an OS disk quota. Archives confined
  ordinary files; excluded files remain on disk and fail the qualification.
- Retains logs, process evidence, exit states and any reports actually emitted.
  With no traced APIs an absent activity report is possible; it is recorded,
  not replaced with a fabricated report or called GPU-retention qualification.

Metadata queries are bounded at 20 seconds each, fixtures at 60 seconds each,
plus bounded cleanup. Usually seconds to a few minutes; supervisor timeout is a
failed gate. No automatic retries/installations. The fixture has its own 90-second
limit. Proprietary profiler internals may perform incidental driver discovery;
the selected workload/tracing are CPU-only and the fixture calls no GPU API.

After the change is on the Linux host:

```bash
(
cd ~/ds4-iq2-q4 || exit
git switch agent/sm75-row-owned-attention || exit
git pull --ff-only || exit
python3 -I ./speed-bench/qualify-sm75-nsys-host.py
)
```

Return `sm75-nsys-host-*.tar.gz`, including failure. No cold restart is requested
for this CPU-only gate. Portable tests use:

```bash
python3 -m unittest discover -s tests -p 'test_sm75_nsys_host.py'
```

These tests exercise command policy, ownership, pidfds, timeout/error cleanup,
artifact limits and fixtures with CPU mocks. They do NOT establish native Linux
Nsight behavior, CUDA compatibility or survival of a GPU-loss event. The native
archive is the next evidence, not another full reproducer run.

## Requirements before enabling CUDA capture

1. Preserve GPU1-only UUID selection, frozen ELF, 1,024 calls, full prelude,
   7/10/3 settings and identified driver/CUDA/cuBLAS stack. Fingerprint actual
   profiler/injection components too. No peer, sweep, shortened workload or
   package changes combined with the tracing change.
2. Retain journal source clocks, PCI pre/post, actual target maps, deadline,
   fault stop and partial evidence. Separate profiler and application statuses.
   Bound target stop/report shutdown without losing the raw/intermediate files.
3. Qualify CUDA/cuBLAS API-to-device-kernel correlation, context/stream IDs,
   launch geometry, timestamps, synchronization and exposed allocation activity.
   Check whether the installed cuBLAS verbose trace includes dimensions and
   algorithm; do not assume an undocumented export schema.
4. Disable GPU metrics, system sampling, event-timing instrumentation and kernel
   replay. Do not combine our preload with Nsight without separately testing
   forwarding/injection interactions. Tracing changes timing and memory use;
   replacing instrumentation is not a zero-effect observer comparison.
5. Export retained report copies with a timeout, after target exit if possible.
   Device loss may prevent the final activity record from completing/flushing.
   A prefix-only report is incomplete, not proof that the next kernel never ran.
6. A timeline cannot prove freedom from valid-address global-memory races or
   correct proprietary workspace contents. Exposed allocations do not establish
   semantic workspace ownership. Keep vendor-side library/driver investigation
   open alongside the exact local reproducer.

## Vendor documentation checked

NVIDIA describes buffering/flush controls and warns that event tracing can add
overhead and false dependencies. A short flush period may retain more completed
activity but does not guarantee the sub-millisecond failing tail survives loss.
[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

CUPTI periodic flushing returns completed full buffers; forced flushing can
return incomplete records. Flush APIs have callback-context restrictions. Neither
mechanism makes missing/unfinished records evidence of successful execution.
No custom CUPTI injector is introduced here.
[CUPTI Activity API and buffering](https://docs.nvidia.com/cupti/main/main.html).
