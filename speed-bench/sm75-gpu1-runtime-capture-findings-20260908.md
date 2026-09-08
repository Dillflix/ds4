# GPU1 runtime capture: 03:42 UTC, 2026-09-08

This supplements the earlier driver/hardware handoff without rewriting its
00:36 incident history. No GPU test, reset or vendor submission was performed
during this offline analysis.

## Evidence identity

Archive: `sm75-token-row-arithmetic-20260908T034250Z.tar.gz`, SHA-256
`fa00124c7e2b8f7f033edba085cec6ebfed94657c68df228f81d6a8276867e7d`.
All 61 members were unique confined ordinary files/directories, validated before
extraction. Source checkout was `c9437cbf3962e18c33da60f3f303db6ba05fd1b9`.

The frozen executable remains SHA-256
`5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
The interposer matches the separately qualified native library SHA-256
`d0a1c52209ed40ef7c40c05ffebe146d88553ab5d46e842a4b318a8b28a96b47`.
Raw and immutable snapshot traces are identical, 17,469,092 bytes, SHA-256
`876d557497730a76889d086179d8e4a191bb64915b3885bac986f436cbe6392b`.
Independent replay of the original analyzer matched the archived analysis:
62,786 records, zero detected caller contract violations, both required
transition gates present. This is not total GPU coverage or race freedom.

## The loaded stack changed

| Component | Earlier 00:36 incident | This incident |
| --- | --- | --- |
| Driver/libcuda | 595.84 | 595.91.07 |
| cudart | 13.2.75 | 13.2.86 |
| cuBLAS and cuBLASLt | 13.4.0.1 | 13.4.1.3 |

These are mapped file identities, not nvidia-smi's CUDA compatibility banner.
The collector pinned the ELF and tracer, but only recorded the runtime packages.
This is NOT a one-variable tracer-only comparison. The failure sequence persists
on this stack; driver/library/instrumentation effects cannot be separately
attributed. Preserve this identified stack for the next tracing experiment.

## Newly measured boundary

All 1,024 burn-in calls completed, and their output comparison was bit-exact.
After structured input replacement:

1. Conversion `29195`, then DEFAULT/N=512 GEMM `29201`, returned success.
2. Existing device synchronization `29223` completed successfully at monotonic
   4639.694626736 seconds.
3. Half conversion `29226` read 8 MiB from `0x752c76a00000` and wrote 4 MiB to
   `0x752c46000000`, in range on GPU1's context/stream 0. It returned success.
4. Algorithm-103/N=256 GEMM `29232` forwarded at 4639.694918878 seconds and
   returned success at 4639.694932772 seconds.
5. Synchronization `29254` entered at 4639.694965891 seconds and returned 719
   at 4640.403401734 seconds. GPU1 Xid 79's kernel-source timestamp is
   4640.109914 seconds, inside that wait. Receipt clocks are later.

No output-A call or second half GEMM occurs in that interval. All added metadata
queries succeeded; this was not the interposer stopping before the real GEMM.

The failing GEMM's recorded arguments exactly match the last successful burn-in
GEMM `29161`, including pointers and scalar addresses. Contents are NOT identical:
the structured-input upload deliberately changes them. M=4096/N=256/K=8192,
transA=T/transB=N, lda=ldb=8192/ldc=4096, FP16 A/B, FP32 C, compute enum 68,
algorithm 103; host alpha=1/beta=0, pointer/math mode 0. Same handle, context,
device and stream. No intervening allocation or free.

| Operand | Address | Required | Live allocation |
| --- | --- | ---: | ---: |
| A weights | `0x752c60000000` | 64 MiB | 64 MiB |
| B scratch | `0x752c46000000` | 4 MiB | 48 MiB |
| C half output | `0x752c4b800000` | 4 MiB | 8 MiB |

All start at generation-1 allocation bases, are nonoverlapping and have matching
successful driver-range queries. The full call has a separate output allocation;
its successful synchronization precedes shared scratch reuse. This does not
support a stale/reallocated operand, wrong caller extent, changed handle state
or missing DEFAULT completion as the explanation for THIS captured tail.

There is no device fence between the half conversion and the 103 GEMM. Their
host returns establish submission, NOT individual device completion. Proprietary
kernel accesses, latent corruption and internal workspace behavior remain open.
Do not declare cuBLAS, driver or hardware cleared.

## Partial capture and reporting correction

There are no unmatched API calls. `trace_complete=false` reflects the missing
normal finalizer after abort, not loss of the required transition. The owned
process later exited -6; snapshot was after exit and matches the raw trace.
The earlier termination timeout is preserved separately.

The original analyzer retrospectively applied later failed-cleanup frees to
earlier conversions, yielding 1,037 lifetime-order warnings. The revised offline
analyzer records free entry/exit sequences and applies that warning only when a
failed free began before the conversion snapshot, or allocation is cross-thread.
Later cleanup failures remain reported. This reporting correction changes no
raw evidence, workload, instrumentation or failure classification.

DCGM was disabled/inactive. Startup retraining completed at 02:29:28 UTC,
over an hour earlier. Preflight found no competing CUDA clients or active
hostengine/nvbandwidth/retrain process. Root port `0000:00:03.0` acquired
FatalErr+ and SDES+; GPU1 became inaccessible. Pre/post link-speed snapshots
are not proof that a speed transition caused the loss.

The next question is which device operations executed in the conversion/103
interval, not another check of the same caller pointers. See
[device-timeline qualification](sm75-nsys-device-timeline.md). No peer GPU,
algorithm sweep, reduced prelude or new GPU command is provided by this report.
