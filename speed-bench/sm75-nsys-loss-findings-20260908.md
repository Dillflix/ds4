# GPU1 failure under Nsight: retained evidence and failed import

This supplement describes the 2026-09-08 17:56 UTC failure, not the older incident
timestamps in the main hardware handoff. No evidence has been sent externally.

## Evidence to preserve for private NVIDIA support

1. `sm75-token-row-arithmetic-20260908T175659Z.tar.gz`, SHA256
   `dfc1724b8763c9c91a20ad77974b33db135e16f7631f2bc8efe2252059fb0d2d`.
2. `sm75-nsys-recovery.uTxWNp.tar.gz`, SHA256
   `3f5867cc5a51ff2a3a075d5c7755adc695eec7820985fbc1db85ae2778a45474`.

The failure archive contains the safe-mode `nvidia-bug-report.log.gz`, kernel
baseline/live records, PCI pre/post snapshots, process samples, gate/exec identity,
application output, original qdstrm and retained Nsight injection storage.
Host identifiers, paths, serials and other history are present; use private support.

## Established results

- Frozen ELF SHA256 remains
  `5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414`.
  Actual ELF exec identity was observed as PID 6452 / starttime 4770697.
- Driver 595.91.07 / cudart 13.2.86 / cuBLAS 13.4.1.3, Nsight 2025.6.3.541.
  GPU1-only, no application peer access/native-stream/algorithm sweep, DCGM off.
- 1024 burn-in calls completed bit-exact. DEFAULT/full512 suffix and algorithm103
  half0/256 completed; synchronization after half1/256 submission failed.
  This is a reported failure interval, not identification of a faulty kernel.
- GPU1 Xid79 source monotonic 47712.553103; journal receipt 47713.924630;
  collector observation 47714.389439; stop request 47714.394030. Buffered stdout
  cannot be treated as CUDA execution timestamps. Source-to-observation delay
  was about 1.84 seconds despite about 4.6 ms observation-to-stop-request delay.
- GPU0 Xid175 followed about 10 seconds later, then Xid154/node-reboot-required
  for all four GPUs. These are not four independent GPU-bus-loss events.
- GPU1 root port 00:03.0 acquired FatalErr+, SDES+, CmpltTO+ versus preflight.
  These post-failure flags do not establish the initiating hardware/software cause.
- All recorded collector subprocess commands returned 0, including the bounded
  safe-mode NVIDIA report (with its listed omissions). Profiler return code was
  -9, not the target exit code. Target exit was eventually observed and final
  surviving-owned-process list was empty; earlier cleanup survival was recorded.

## What did not succeed

No completed nsys-rep or SQLite device trace survived. Actual-target sampling
missed libcuda; the completed actual-target cudart/cuBLAS/Lt hashes matched.
Preflight disk pins do not supply the missing runtime observation.

All 15 prefix lengths/hashes verify. `live-tmp/nsys-report-ba64.qdstrm` is 71,078 bytes,
SHA256 `b24560863e0c5739336cb91c61e34a852e6e4734f32e9907841c033769daee0e`.
`retained-prefixes/prefix-002.bin` is its 64,255-byte prefix, SHA256
`d6eaef6782e293395cfea76cdd5343b9a00080642adc8b1937f54d4ee3df93cd`.
The matching QdstrmImporter 2025.6.3.541 rejected both as incomplete or invalid,
exit 2, not a timeout. Both recovery input hashes match the failure archive.

`retained-prefixes/prefix-013.bin` preserves 2,101,248 bytes of
`injection_6452_1000_storage.dat`, SHA256
`7394ecbe9c641f07598b46f3ef476b8991a16214ecbadc03f2c05523a4cd7669`.
Please ask NVIDIA whether these proprietary records can be recovered/merged,
and what supported catastrophic-exit collection strategy can retain actual
conversion/GEMM device records on this exact stack. Do not rename storage as a
qdstrm or fabricate footer data. Raw API-name strings are not executed-call evidence.

## Remaining software investigation

Review target/global-memory ownership and cuBLAS ordering against the existing
runtime-contract evidence; request internal kernel/driver analysis of this exact
local reproducer. Do not substitute a peer test. Neither valid operand ranges nor
API timelines exclude valid-address races or proprietary workspace corruption.

Collector revision: offload runtime mapped-file hashing, run prefix retention
independently, record shutdown stages, and permit bounded profiler finalization
only after target death. CPU tests can verify supervision behavior, but cannot
qualify real CUDA trace coverage or prove these changes recover this failure.
