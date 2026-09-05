## Benchmarking

### Withdrawn runtime native-primary Q8 residency

`DS4_CUDA_Q8_WARP_INTERLEAVED_PRIMARY` is fail-closed. Repeated four-GPU
qualification lost PCI device `0000:03:00.0` during the first untimed batched
prefill, after F16-cache materialization and before deferred decode-native
materialization. The experiment changed more than storage format: it removed
T32/attention-A/attention-B tensors from the ordinary selective-cache ranges,
introduced dedicated source slabs, and intercepted strict cache lookup.

The bit-exact warp-interleaved kernels and their bounded/auxiliary-cache tests
remain available. A production VRAM reduction must use an explicitly tagged
native GGUF representation and the ordinary selective-cache ownership path; it
must not repack, replace, or free primary weight residency at runtime.

The implemented native-GGUF contract uses the private, size-neutral
`sm75_q8_warp32` type for all 43 T32/attention-A/attention-B tensor triplets.
T32 is home-resident once, A is split by output rows, and B is stored as two
contiguous K shards. The runtime registers execution views as borrowed aliases
of the ordinary selective cache; those aliases allocate and own zero bytes.
Eligible sharded prefill, single-token decode, and fused epilogues consume the
borrowed native-Q8 residency. A complete unsplit A+B prefill instead consumes
its startup-planned F16 execution binding, which is materialized directly from
the tagged bytes. It never means an F32 expansion or a late allocation after
the workload starts. Qualification covers both execution forms, not decode
alone. See `gguf-tools/README.md` for the lossless converter and its disk-space
boundary.

The F16 admission plan is part of that representation contract. Attention row
splitting is conditional on each chunk's geometry, so even an enabled pair can
fall back to a complete home-local A+B projection for a tail or another
ineligible shape. All 43 layers' attention-A and attention-B bindings
(`86/86`) are therefore mandatory and home-local. T32 already has one complete
native-Q8 residency on the layer's home GPU, and the fused F16 path excludes
single-token decode. Its prefill binding is consequently one required,
complete home-local F16 matrix per layer. The optional pair attention
query-row split happens only after `q_b` has assembled the complete query
tensor, so it does not require T32 row-half bindings or T32 partner execution.
This removes the failing plan's redundant 64 MiB F16 expansion per layer
(`2.6875 GiB` system-wide) and eliminates its new T32 activation/result traffic
without adding another persistent representation. The fixed 22/21 topology
therefore registers 344 candidates and requires 129 execution bindings; all
are materialized and synchronized during session startup. The engine rejects
the tagged model before prefill if any required binding is unavailable.
Eligible split chunks still consume the native A-row and B-K shards directly.
Decode and prefill therefore share one native-Q8 residency without making
either phase an afterthought or relying on a mid-run allocation fallback.

The legacy F32-expanded Q8 cache remains available only to older, explicitly
selected diagnostic SGEMM experiments. It is not a production fallback. A
tagged native-Q8 model rejects `DS4_CUDA_Q8_F32_*`,
`DS4_CUDA_ATTN_Q_B_F32_CACHE`, and non-F16
`DS4_CUDA_Q8_PARTNER_ARITHMETIC` settings rather than spending 4 bytes per
weight or allowing an A/B to qualify under different arithmetic. Production
uses the mandatory F16 execution bindings where appropriate and otherwise the
native-Q8 kernels.

`cuda-sm75-native-q8-gguf-production-ab.sh` is the engine-wide acceptance
test. Run it once for `mixed15` and once for `all43`, each with its canonical
model and the corresponding tagged native model. Every invocation uses the
stable 22/21 four-GPU policy (pair-0 attention rows disabled, indexer rows
enabled on both pairs), and executes both arms at PP512, PP4096, and PP32768.
It accepts the format only when prefill-frontier and per-token decode logits
are byte-identical, every prefill/decode throughput frontier retains at least
95% of canonical performance, peak aggregate VRAM falls by at least 3500 MiB,
and the log proves that all native execution views borrowed ordinary residency
with zero runtime repack/replacement allocation. The thresholds are explicit
environment variables, but lowering them does not constitute production
qualification.

```sh
unset CUDA_VISIBLE_DEVICES NATIVE_Q8_GGUF_PRODUCTION_AB_DIR
MODEL_LAYOUT=mixed15 \
CANONICAL_MODEL=/absolute/path/to/mixed15.gguf \
NATIVE_MODEL=/absolute/path/to/mixed15-sm75-native-q8.gguf \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-native-q8-gguf-production-ab.sh

unset NATIVE_Q8_GGUF_PRODUCTION_AB_DIR
MODEL_LAYOUT=all43 \
CANONICAL_MODEL=/absolute/path/to/all43.gguf \
NATIVE_MODEL=/absolute/path/to/all43-sm75-native-q8.gguf \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
SKIP_BUILD=1 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-native-q8-gguf-production-ab.sh
```

### SM75-native Q4-32 decode gate/up mappings

`cuda-sm75-decode-q4-gate-native.sh` compares the former Q4-32 one-row/
warp decode control with three measured mappings: two rows per warp,
eight-row native-tile signed DP4A, and eight-row native-tile packed
`m8n8k32` INT4 MMA. Tile32 packed-INT4 MMA is now the production default.
Set `DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=control` for the explicit rollback;
`hwarp16` and `tile32-dp4a` remain diagnostic selectors.

The exactness checkpoint uses the production 4096-element K dimension (16
K256 records), two deterministic nonzero inputs, nonuniform expert weights,
and interleaved owned/unowned slots.  It requires byte equality at both the
gate/up mid boundary and final Q4-32 down output, with nonzero reference and
candidate outputs.  Fresh PTXAS/SASS gates reject spills, local memory,
ATOM/RED, more than 128 allocated registers, or failure of the two-CTA/SM
register budget.  The DP4A candidate must contain `IDP.4A`; the packed-MMA
candidate must contain both `IMMA.8832.U4.S4` and `IMMA.8832.S4.S4`.  Timing
includes activation quantize/pack, fused gate/up, mid quantize/pack, and down.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q4-gate-native.sh
```

The Q4 decode evidence runners serialize access to the checkout because both
rebuild and execute the same objects and binaries. Interrupted runs write
`state=interrupted` with the signal-derived exit status; other incomplete
runs write `state=failed` even if Bash presents a transient zero status to the
EXIT trap. To resume, reuse the exact output directory explicitly:

```bash
RESUME=1 \
Q4_GATE_NATIVE_DIR="$PWD/sm75-decode-q4-gate-native-<timestamp>" \
./speed-bench/cuda-sm75-decode-q4-gate-native.sh
```

A successful Q4 audit build writes `build-complete.txt`, binding the Git
commit, complete tracked-worktree diff, CUDA flags, and both binary hashes.
The gate audit can reuse a down audit's build only by naming it explicitly;
it never selects a latest directory:

```bash
REUSE_BUILD_DIR="$PWD/sm75-decode-q4down-tile32-<timestamp>" \
SKIP_BUILD=1 \
./speed-bench/cuda-sm75-decode-q4-gate-native.sh
```

### SM75-native Q3A4 decode mapping

`cuda-sm75-decode-q3a4-native.sh` treats Q3A4 as its own decode problem. It
keeps Q4-32 unchanged and compares the shipping Q3A4 gate/up kernel against
the measured fused-u2 control, a two-row half-warp mapping, and a mapping in
which each warp follows one complete native eight-row Q3A4 tile. The latter
targets the measured idle half-warps and highly fragmented weight loads while
preserving fused gate/up traversal and the existing expert-parallel boundary.
The tile mapping is measured with both scalar nibble arithmetic and an exact
signed-DP4A implementation so layout and arithmetic effects remain separable.

The runner requires the real 16-K256-record byte-exact regression, fresh
PTXAS/SASS evidence, zero spills and local traffic, complete owned-call timing,
and focused Nsight Compute memory/scheduler captures for all five kernels.
This pass selected tile32 signed-DP4A. The subsequent real four-GPU production
A/B established byte-exact output and a 5.2--6.5% decode gain, so that mapping
is now the SM75 tagged-Q3A4 default; this harness remains the bounded kernel
audit rather than the production acceptance test.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q3a4-native.sh
```

### SM75 Q3A4 exact in-CTA K-split

`cuda-sm75-decode-q3a4-ksplit.sh` uses tile32 signed-DP4A K1 as the control
and tests Q3A4-only K2 and K4 CTA mappings. Two or four warps
cooperate on each native eight-row tile, write disjoint K256 leaves to the
same 4 KiB shared table, then the split-zero warp executes the unchanged
16-leaf `__fadd_rn` tree. This is an in-CTA experiment: it adds no global
partial buffer, reducer launch, atomic, or multi-GPU boundary.

The runner requires nonzero byte-exact gate/up intermediates and owned output,
memcheck of the 512-thread K4 case, fresh PTXAS/SASS identities, DP4A with no
atomics, zero stack/spill/local traffic, and at most 64 allocated registers for
K2/K4. Inclusive timing compares each candidate directly with K1 and includes
activation quantize/pack, gate/up, mid quantize/pack, and Q4-32 down. Focused
Nsight captures report duration, memory traffic, occupancy, and scheduler
stalls. K4 is now the production default; set
`DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT=1`, `=2`, or `=4` to select the split
explicitly.

The bounded harness materializes a zero-valued synthetic expert payload. Its
timings compare inclusive launch/resource structure, but do not establish
real-weight cache or DRAM behavior. The subsequent real-model end-to-end A/B
provided the production evidence used to promote K4.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q3a4-ksplit.sh
```

After the bounded K-split audit selects K4, use the real-model production A/B
to compare only tile32-DP4A K1 against tile32-DP4A K4. Both arms explicitly
retain mapping 3; Q4-32, the 22/21 placement, complete 344/344 dense-F16 cache,
indexer, attention, and cross-GPU topology must remain identical. The runner
exports and compares all 344 placement entries and canonical binding identities
between arms, rather than accepting matching aggregate counts. It also
uses PP512/4096/32768, TG256, alternating paired repeats, and 16 byte-exact
decode-logit comparisons per frontier. It is resumable and archives failures.
Resume accepts evidence only with healthy pre/post GPU snapshots; outputs from
a process that lost a GPU are intentionally rerun.
`Q3A4_LAYOUT=mixed15` validates the current 15-layer model and requires its
exact layer list. `Q3A4_LAYOUT=all43` is the target-model mode and requires
Q3A4 gate/up plus Q4-32 down on every routed layer. The selected layout, layer
count, and complete list are recorded, and owned-call expectations scale with
15 or 43 rather than accepting an inferred inventory.

On the mixed15 model, K4 improved steady decode by 1.109% at PP512, 1.364% at
PP4096, and 1.308% at PP32768 across three alternating paired repeats. The 16
decode logits at every frontier were byte-identical and the exported 344-entry
dense-F16 plans and bindings were identical between arms. This accepted K4 as
the SM75 Q3A4 production default. The all43 model will use the same K4 default
without another dispatch-code change; its end-to-end performance and exactness
remain to be measured separately with `Q3A4_LAYOUT=all43`.

```bash
export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
Q3A4_LAYOUT=mixed15 \
REPEATS=3 \
TG_TOKENS=256 \
EXACT_TOKENS=16 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
./speed-bench/cuda-sm75-decode-q3a4-k4-production-ab.sh
```

If a run reached Nsight Compute after all earlier gates passed, resume the
same evidence directory without rebuilding or repeating exactness, sanitizer,
or timing. Resume mode requires the original clean sanitizer result, the same
physical GPU and SM architecture, unchanged Q3A4 implementation sources, and
the same profile-harness machine code; it revalidates the reused artifacts
before capture.

```bash
RESUME=1 \
Q3A4_KSPLIT_DIR="$PWD/sm75-decode-q3a4-ksplit-YYYYMMDDTHHMMSSZ" \
RUN_SANITIZER=0 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=1 \
CREATE_ARCHIVE=1 \
./speed-bench/cuda-sm75-decode-q3a4-ksplit.sh
```

### SM75 Q3A4 K4 software-prefetch depth

`cuda-sm75-decode-q3a4-prefetch.sh` holds the production tile32-DP4A K4
mapping fixed and compares its ordinary weight stream with bounded prefetch
depths 1 and 2. It requires the nonzero exact regression, sanitizes depth 2,
and rejects any specialization above 64 allocated registers or with PTXAS
spills, stack, or SASS `LDL`/`STL`. Inclusive timing covers the complete
production owned call; focused Nsight captures report duration, DRAM bytes and
rate, long-scoreboard stalls, and eligible warps.

The profile harness still uses a zero-valued synthetic payload, so a bounded
winner is not production evidence. The runner archives success or failure and
can resume only the Nsight phase after revalidating the original exactness,
sanitizer, resource, timing, source, and binary evidence.

Prefetch depth 2 improved the complete bounded Q3A4 K4 owned call by 7.8%
(1.078x), passed exactness and sanitizer, used 64 registers with no stack,
spill, or SASS local traffic, and reduced the profiled kernel from 93.664 to
77.696 microseconds. The all-43-layer four-GPU production A/B then measured
gains of 1.787%, 1.387%, and 1.189% at PP512, PP4096, and PP32768,
respectively, with low paired variance. All 96 checked logits were
byte-identical, and dense-Q8 plans and Q3A4 dispatch counts matched between
arms. Prefetch depth 2 is therefore the eligible tile32-DP4A K4 production
default. Set `DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH=0` for the explicit
rollback.

Depth 1 remains architecture-harness-only, and there is no depth-4 kernel.
The current kernel has only eight inner groups; extending the pipeline would
increase live operand state beyond the already-qualified 64-register depth-2
kernel. A depth-4 implementation is not justified unless focused profiling
first shows substantial residual long-scoreboard latency that depth 2 does
not cover, after which it would require fresh exactness, sanitizer,
PTXAS/SASS resource, bounded timing, and four-GPU production acceptance.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q3a4-prefetch.sh
```

`cuda-sm75-decode-q3a4-prefetch2-production-ab.sh` preserves the real-model
acceptance comparison. It compares explicit rollback K4 (depth 0) with the
production-default depth 2
while fixing every other dispatch, the 22/21 stage split, complete 344/344
dense-F16 admission, and the four-GPU topology. The runner alternates paired
TG256 measurements at PP512/4096/32768 and requires 16 byte-identical decode
logits at each frontier. Dispatch counters must show K4 exclusively in both
arms and prefetch depth 0 versus 2 exclusively; dense-Q8 placement plans and
canonical binding identities must also match between arms.

The A/B also fixes the accepted stable prefill policy in every process:
pair 0 retains prefill indexer row splitting but not attention row splitting,
pair 1 retains both, attention-cache mirrors exist only for pair 1, and native
index-cache mirrors exist for both pairs. The runner rejects the old coupled
cache policy or any unexpected pair-0 attention dispatch. It also requires the
physical GPU 0/1/2/3 power limits to match `250,260,250,250` W by default and
requires the default fused T32 FP16-output path to dispatch. Thus a decode
result cannot silently regain the unstable prefill topology or lose an
accepted production optimization. Pair-1 attention and both indexer dispatches
are required in the eligible multi-frontier throughput workload; PP512-only
exact runs validate the policy masks without incorrectly requiring a split
below the production dispatch threshold.

`Q3A4_LAYOUT=mixed15` requires the current exact 15-layer allocation.
`Q3A4_LAYOUT=all43` requires Q3A4 gate/up and Q4-32 down on every routed
layer, so the same runner can validate the forthcoming all-Q3A4 gate/up model
without weakening its inventory checks. Exact checkpoints are isolated by
arm and frontier, but all use the fixed 33025-token PP32768+TG256 production
allocation and must export the same dense-Q8 plan. Resume rejects wrong-sized,
non-finite, or all-zero logits,
unhealthy GPU snapshots, wrong layer inventories, wrong dispatch counters,
invalid cache state, changed model stat identity, changed Q3A4 CUDA sources,
or a changed engine binary; failures are archived by default. Full hashing of
the 139 GiB model remains intentionally disabled.

```bash
export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
Q3A4_LAYOUT=mixed15 \
REPEATS=3 \
TG_TOKENS=256 \
EXACT_TOKENS=16 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
./speed-bench/cuda-sm75-decode-q3a4-prefetch2-production-ab.sh
```

Run the all-Q3A4 model as an independent invocation by changing `MODEL` to
`DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf` and setting
`Q3A4_LAYOUT=all43`. Separate output directories keep a reboot or failed arm
for one model from contaminating the other model's evidence.

### SM75 Q4-32 software-prefetch depth

`cuda-sm75-decode-q4-prefetch.sh` applies the same bounded software-pipeline
question to the current Q4-32 production kernels, but it keeps three ownership
and execution shapes separate:

- gate/up tile32 packed-INT4 MMA;
- down tile32 packed-INT4 MMA writing `owned_slots`;
- down tile32 packed-INT4 MMA writing `owned_packed`.

Each family compares prefetch depths 1 and 2 with that family's depth-0
production kernel. It does not compare against the superseded quarter-warp or
non-MMA controls, so any reported speedup is attributable only to instruction
scheduling/prefetch. The audit requires nonzero byte-exact regression for both
depths, independent dispatch counters, depth-2 memcheck for every family,
fresh PTXAS/SASS identities, native U4xS4 plus S4xS4 `m8n8k32` instructions,
zero stack/spill/local/atomic traffic, no DP4A fallback, at most 128 allocated
registers, and a two-CTA/SM register gate. Inclusive timing covers the complete
production-owned call. Focused Nsight Compute captures report duration, DRAM
traffic, long-scoreboard pressure, and eligible warps for all nine kernels.

The bounded audit rejected Q4 prefetch. Gate/up depth 1 and 2 measured
0.99967x and 0.99967x; down `owned_slots` measured 1.00021x and 0.99982x;
and down `owned_packed` measured 0.99741x and 0.99163x. With no material win
and a clear packed-path regression, all three Q4 production paths remain at
depth 0 and no Q4 candidate advances to a four-GPU production A/B. There is
also no Q4 depth-4 specialization; the existing kernels support only depths
0, 1, and 2.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
./speed-bench/cuda-sm75-decode-q4-prefetch.sh
```

### SM75 Q4-32 down tile32 packed-INT4 audit

`cuda-sm75-decode-q4down-tile32.sh` compares the former one-token
quarter-warp `owned_slots` and `owned_packed` Q4-32 down controls with the
production-default 128-thread tile32 kernels. A warp follows one native eight-row
tile and consumes the tagged packed Q4 weights directly through Turing
`m8n8k32` U4xS4 and S4xS4 MMA. All eight K256 float leaves are staged before
the unchanged 4/2/1 reduction. The packed exactness fixture owns slots 0+1 in
both three-slot groups, so it proves the real prefix-pair exact-add case rather
than only single packed operands.

The runner requires nonzero byte-exact outputs for both ownership modes,
poisoned independent control/candidate outputs, all packed ownership masks
`000` through `111`, the mask-`111` second prefix-pair cycle, and a signed-zero
boundary. It also requires memcheck, fresh PTXAS/SASS identities, both native
INT4 MMA opcode forms, zero
atomics, zero stack/spill/local traffic, at most 64 allocated registers, and
exactly 128 threads plus 1024 bytes of static shared memory for owned-slots
and 1152 bytes for owned-packed.  The packed path's extra 128 bytes explicitly
stage the first prefix-pair total instead of allowing it to spill across the
second fully unrolled MMA pass; this does not change the SM75 occupancy limit.
Timing includes the full production-owned call. Nsight Compute captures both
controls and candidates. Timing requires an odd round count and checks exact
control/candidate dispatch counters. The zero-weight timing harness measures
launch and resource shape; the independent long-context regression is the
arithmetic proof. Tile32 is now the SM75 tagged-layout production default;
set `DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=control` for the explicit
quarter-warp rollback.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q4down-tile32.sh
```

Set `RESUME=1` with the existing `Q4DOWN_TILE32_DIR` to reuse a verified
completed build after interruption. A missing or mismatched checkpoint forces
a fresh build; `SKIP_BUILD=1` is rejected in that case.

### SM75 fused low-register Q3A4/Q4-32 decode sweep

`cuda-sm75-decode-q32-fused-lowreg.sh` follows the rejected two-projection
split with a single-launch design. It preserves gate/up fusion and exact
accumulation while sweeping partial-unroll factors 1, 2, and 4. Q3A4 is timed
first; its measured ranking determines the Q4-32 execution order, but all
three Q4-32 variants are still tested independently.

The runner requires fresh PTXAS/SASS evidence, byte-exact non-zero regression,
and inclusive production-owned-call timing. Spills, SASS local traffic, or
more than 128 allocated registers reject an individual candidate without
hiding the other measurements. Nsight Compute profiles the control and the
best structurally eligible candidate for each format.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=7 \
TIMING_REPEATS=20 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q32-fused-lowreg.sh
```

### SM75 low-register Q4-32/Q3A4 decode gate/up split

`cuda-sm75-decode-q32-lowreg.sh` is a bounded, production-shaped experiment
for splitting the one-token Q4-32 and Q3A4 gate/up kernel into two independent
projections and an exact SiLU/multiply/weight combine. It tests the two formats
independently, rejects any byte mismatch, PTXAS stack/spill bytes, SASS
`LDL`/`STL`, or allocation above 128 registers/thread, and reports timing for
the complete owned-expert API call. That timing includes both additional
launches and the intermediate global-memory traffic.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=7 \
TIMING_REPEATS=20 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-decode-q32-lowreg.sh
```

### Tagged SM75-native full-Q4 production A/B

`cuda-sm75-native-q4-production.sh` is the acceptance runner for the packed
INT4 production layout. It losslessly repacks a stock full-Q4 model when the
native output does not already exist, requires the production API's standard
and native results to be exact for tile16 plus real 8/4 tails and decode, and
then requires bit-identical full-vocabulary logits from the two complete GGUFs.

The timed comparison is fixed to the measured 22/21 split and device order
`0,3,1,2`; process order alternates between standard and native. It also
captures an actual four-GPU Nsight Systems trace and bounded Nsight Compute
reports for native gate/up tile8 and down tile16/8/4. No model hash is taken.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export NATIVE_MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

REPEATS=2 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-native-q4-production.sh
```

Use `REPEATS=6` for replicated timing after the two-repeat pilot succeeds.

### Cost-planner default and rejected K-stage A/B

The cost-aware residual planner is now the tagged native-Q4 production default:
`1..4 -> tile4`, `5..8 -> tile8`, and `9..15 -> tile16`. Set
`DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES=1` only for a diagnostic comparison with
the old minimum-active-tile planner.

The diagnostic `stream7` gate and `compact7` down candidates compile below
32 KiB declared shared memory, but both are rejected. Moving activation
payload and metadata reads into the repeated output-row loop raised LG
throttle and made both kernels more than three times slower in representative
Nsight Compute captures. They remain selectable only to reproduce the failure
audit; neither is a production candidate.

`cuda-sm75-native-q4-next-ab.sh` tests `baseline`, `down`, `gate`, and `both`
in alternating order at placement `0,3,1,2` and split 22/21. Before loading the
model it rejects either new kernel if compiled shared memory is at least 32 KiB,
registers exceed 128, or any stack/local storage is present. It then requires
exact production-API results, a real cost-planner audit, and bit-identical raw
logits for every selected variant. `legacy` remains an optional fifth variant.
`RUN_NSYS=1` captures matching `baseline` and `both` traces.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

REPEATS=2 \
RUN_NSYS=1 \
./speed-bench/cuda-sm75-native-q4-next-ab.sh
```

The script never modifies `NATIVE_MODEL`. Use `RUN_E2E_EXACT=0` only when
rerunning timing after the same binary/model combination has already passed
the raw-logit comparison.

### Gate/Up fixed-K16 and row-fused full64 A/B

`cuda-sm75-native-q4-gate-fused-ab.sh` isolates two Gate/Up changes while
leaving Down and the production cost planner untouched:

- `fixed` is the current fused tile8 path specialized for the model's fixed 16
  input blocks. It is the optimized control: same routes, staging, reduction
  order, and two low8/high8 launches, but compile-time K bounds and strides.
- `fused` retains all 16 activation payload rows in Turing's 65,536-byte
  opt-in shared-memory allocation. For each output-row group it computes Gate,
  lets Gate's scalar accumulator window die, holds only four reduced Gate
  values while computing Up, and immediately emits the fused SiLU product.
  It therefore reuses each Q4 weight fragment across low8/high8 without the
  rejected full64 design's global Gate scratch or phase barrier. Its tail8 and
  tail4 work uses the fixed-K16 control.

Before loading the model, the runner rejects local/stack traffic or more than
128 registers, requires exact Up auxiliary output and proves that enabling the
auxiliary stores does not perturb production API output, records
separate dynamic-K/fixed-K SASS summaries, times all three variants against the
captured early/late production histograms, and records bounded Nsight Compute
reports. That bounded Gate-only screen is the default (`RUN_FULL_MODEL=0`).
Model hashing and conversion are deliberately absent.

If only Nsight Compute failed after the bounded screen, set
`GATE_FUSED_DIR` to that existing result directory and rerun with
`RESUME_NCU_ONLY=1 SKIP_BUILD=1 RUN_FULL_MODEL=0 RUN_NSYS=0 RUN_NCU=1`.
The resume profiles all descendants but filters by the exact harness process
name, avoiding wrapper-process captures without repeating exactness or timing.

Only after the bounded evidence is viable, rerun with `RUN_FULL_MODEL=1`. That
mode additionally compares raw full-model logits, alternates
baseline/fixed/fused prefill runs, verifies placement and Q8-cache equivalence,
reports paired throughput ratios, and captures a short Nsight Systems trace for
each variant.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

REPEATS=2 \
CTX_START=2048 \
CTX_MAX=8192 \
RUN_FULL_MODEL=0 \
RUN_E2E_EXACT=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-native-q4-gate-fused-ab.sh
```

For the promotion pass, reuse the same exports with
`RUN_FULL_MODEL=1 RUN_NCU=0`; the already-collected kernel reports do not need
to be repeated.

### Compact N-split native-Q4 topology audit

`cuda-sm75-q4-nsplit.sh` is a harness-only go/no-go experiment for the compact
N-split CTA topology. It does not alter production dispatch or open a model.
For both gate/up and down it tests 4- and 8-warp CTAs where one CTA owns one
real cost-planner tile16 plus one output-row macro-tile. The CTA stages only
one 256-K activation slab at a time (4,672 bytes), keeps scalar K slots in
registers, and consumes each packed Q4 fragment for both low8 and high8 route
halves before advancing.

The early/late inputs use the exact captured production expert histograms;
8/4 residual work is excluded so the decision measures only the topology it
could replace. The runner requires bit-exact outputs, no PTXAS/SASS local
memory, packed m8n8k32 U4/U4 and S4/U4 instructions, and less than 8 KiB shared
memory. It then records balanced kernel timings and focused Nsight Compute
reports for baseline, nsplit4, and nsplit8. A duration candidate passes only
if it beats the matching shipping native-A/W kernel by at least 10% in both
early and late shapes; a timing miss is recorded rather than suppressing the
profiles needed to explain it.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
BENCH_ROUNDS=3 \
BENCH_LAUNCHES=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-q4-nsplit.sh
```

No `MODEL` variable is required. Return the printed
`sm75-q4-nsplit-<timestamp>.tar.gz` archive.

### Persistent 512-row tile16 Gate audit

`cuda-sm75-q4-persistent.sh` tests the successor to the rejected compact
N-split Gate topology without changing production dispatch. Both candidates
retain the shipping 512-row CTA, reuse every packed Q4 fragment across a real
16-route cost-planner tile, and keep only one matrix's scalar accumulator
window live. `persistent-seq16` computes Gate then Up sequentially;
`persistent-ws16` assigns eight warps to each matrix. Because the complete
16-route native-Q8 activation tile is 73.0 KiB (74,752 bytes) and cannot fit in Turing's
64 KiB shared-memory budget, both consume immutable activations directly
through the unified cache path.

The runner checks real early/late histograms for bit-exact output, rejects
local-memory traffic or more than 128 registers/thread, records balanced
timings against the shipping warp16 scalar kernel, and captures focused
early-layer Nsight Compute reports.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
BENCH_ROUNDS=3 \
BENCH_LAUNCHES=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-q4-persistent.sh
```

No model is opened. Return the printed
`sm75-q4-persistent-<timestamp>.tar.gz` archive.

### Exact production Q4 expert histogram

`cuda-q4-real-histogram.sh` captures the actual per-expert routed-pair counts
from one production full-Q4 prefill. The counts are appended by the existing
device metadata kernel to a preallocated device audit buffer. There is no new
kernel launch, timing event, synchronization, or host read inside the measured
prefill; each GPU buffer is copied once after the existing final
synchronization.

Gate/up and down use the same router assignment. One capture therefore derives
both the shipping gate/up tile8 plan and the down tile16 plan, as well as
minimum-active-tile gate 8/4 and down 16/8/4 tail plans. It also reports the
production grid-capacity entries that return early because the actual tile
count remains on-device. The runner refuses a model
unless all 43 routed layers report Q4 gate, Q4 up, and Q4 down. Model hashing is
omitted.

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,2,1,3 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
./speed-bench/cuda-q4-real-histogram.sh
```

The returnable archive contains the raw wide audit, an exact record-level tile
plan, a long per-expert table, a count-frequency table, the benchmark result,
and runtime provenance.

### Post-scalar full-Q4 kernel trace

`cuda-q4-post-scalar-trace.sh` re-establishes the current full-Q4 kernel-time
distribution after the production SM75 scalar-slot changes. It first runs one
audited process and requires all 43 routed layers to be Q4/Q4/Q4 plus exactly
172 scalar gate/down device-layer dispatch markers. A second clean process is
then traced only across its timed 2K prefill. The runner forces the 22/21 split,
prefill pipeline, Q8-to-F16 cache, and both Q4 scalar specializations instead
of relying on inherited settings or defaults.

The output includes the raw Nsight Systems report and CSV exports, grouped
kernel shares, and the exact two production Q4 rows with their specialization
and launch cardinality. It requires 344 gate/up and 344 down calls and rejects
a trace whose demangled kernel identity is not the scalar specialization. It
does not invoke Nsight Compute and does not hash the 153 GiB model.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,2,1,3 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
./speed-bench/cuda-q4-post-scalar-trace.sh
```

Return `q4-post-scalar-trace-<timestamp>.tar.gz`. This pass updates the Amdahl
weights used to rank persistent exact-grid work queues and native A/W kernels;
it does not itself change production dispatch.

### Full-Q4 placement and cuBLAS-cache A/B

`cuda-q4-layout-cache-ab.sh` tests the cache/pipeline imbalance exposed by the
post-scalar trace. It compares three fixed configurations:

- `baseline-22x21`: split 22/21 with devices `0,2,1,3`;
- `split-21x22`: split 21/22 with devices `0,2,1,3`;
- `swap-22x21`: split 22/21 with devices `0,3,1,2`.

Each process first runs an untimed 2K warm-up so every admitted Q8 weight has
already been expanded to F16 for the measured cuBLAS path. It snapshots the
per-device cache before and after the timed sweep and rejects any run whose
cache changes. Three rotated repeats place every variant in every process
position once. The default 2K/4K/8K sweep is a bounded placement decision;
increase `CTX_MAX` only after selecting the winner. No model hash or Nsight
capture is performed.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_VRAM=auto \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
./speed-bench/cuda-q4-layout-cache-ab.sh
```

Return `q4-layout-cache-ab-<timestamp>.tar.gz`. Its summaries include paired
throughput ratios and the exact cached F16 slice count and bytes on each stage
home GPU.

### Q8 F16 benefit-plan A/B

`cuda-q8-cache-benefit-ab.sh` holds the selected full-Q4 placement fixed at
22/21 and `0,3,1,2`, then compares the default benefit-per-expanded-byte plan
with the former first-use admission policy. The startup plan is deterministic
and gives the measured SM75 T32 `attn_q_b` and T256 `attn_output_b` paths first
claim on each device's live cache headroom. It registers exact per-device keys
at engine creation and defers allocation/dequantization until the untimed
warm-up, after graph/context buffers exist; measured frontiers therefore
exclude one-time materialization and cannot admit extra tensors by traversal
order. `DS4_CUDA_Q8_F16_FIRST_USE=1` is retained only as the A/B escape hatch.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=4 \
./speed-bench/cuda-q8-cache-benefit-ab.sh
```

Return `q8-cache-benefit-ab-<timestamp>.tar.gz`. The runner requires nonzero
T32/T256 planned coverage, freezes the cache before timing, alternates process
order, and reports paired planner/first-use throughput ratios per frontier.

### Q8 F16 NVLink-partner execution A/B

`cuda-q8-partner-offload-ab.sh` holds full-Q4 placement at the measured 22/21
split and logical device order `0,3,1,2`. It screens partner execution disabled,
T32-only, T256-only, shared-down-only, and the initial T32+T256 (`legacy`)
policy. Within each home-device bucket, at equal benefit per byte,
fixed/home-only candidates rank before partner-eligible candidates. The class
selector therefore controls both which home-cache misses may overflow and which
tied expensive class is preferentially kept local; this is an end-to-end
placement-policy A/B, not an identical-home-cache microbenchmark.

The runner requires all four logical home/partner routes to be validated
`DIRECT` and independently rejects any physical pair that `nvidia-smi topo -m`
does not report as `NV#`; PCIe peer access is not accepted as NVLink evidence.
Its `pair-topology.tsv` records the production homes-first mapping. Every
candidate run must execute only its requested class on a configured partner.
One-pair execution is valid when the other stage fits its ranked F16 set locally;
`class-evidence.csv` makes the partner devices observed in the bounded audit
sample explicit. It freezes cache state before timed frontiers, rotates process
order, and reports the absolute median candidate/local tokens per second
alongside every paired throughput ratio. Runtime summary fields are explicitly
labeled `process_total_*`; the bounded device-audit fields are labeled
`audit_*`, so a
class-pure sample may be smaller than the uncapped process total without
invalidating the run. Full-logit output records numerical drift and top-token
agreement for every class, while within-variant repeat determinism remains an
acceptance condition. The GPU regression deliberately fills a home device's
per-device F16 cap and proves bit-exact partner execution for the production
T32, T256, and shared-down shapes. By default it also captures one short,
measured-prefill Nsight Systems report for each isolated class so transfer,
cuBLAS, synchronization, and partner contention can be evaluated.
The runner sets `DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS` to the positive
`PREFILL_CHUNK`, so
custom chunk sizes reserve matching partner scratch before timing. Outside the
runner the engine defaults this admission bound to 2048 tokens and safely
declines larger microbatches instead of growing scratch during execution.

```bash
cd ~/ds4-iq2-q4
git remote set-branches --add origin agent/partner-q8-offload
git fetch origin
git switch --track origin/agent/partner-q8-offload
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
VARIANTS=local,t32,t256 \
RUN_NSYS=0 \
bash ./speed-bench/cuda-q8-partner-offload-ab.sh
```

The command above is throughput-only; set `RUN_NSYS=1` to capture timelines.
Set `RUN_GPU_TEST=0` only after all three exactness cases have passed for the
current binary.
`RUN_GPU_TEST=1` still executes the existing binary when `SKIP_BUILD=1`; the
latter skips compilation, not validation. `VARIANTS` must include `local`,
`t32`, and `t256` so every A/B archive proves both expensive projection
classes; it may additionally include the other named variants.
When `RUN_NSYS=1`, `NSYS_VARIANTS` accepts a comma-separated subset of
`VARIANTS`; it is ignored for a throughput-only `RUN_NSYS=0` run. Return
`q8-partner-offload-ab-<timestamp>.tar.gz`; its
`class-evidence.csv` records execution counts by class and its `nsys/`
directory contains the bounded timelines and exported summaries.

Repeated full-Q4 screens selected T256-only as the best performance candidate;
the latest measured 13.0--15.3% over local, versus 12.2--14.3% for T32-only. The
initial mixed legacy policy measured 12.3--14.4%. T32, T256, and legacy produced
byte-identical logits to one another, but the T256 trace moved only 3.12 GiB of
partner activation and result traffic versus T32's 12.70 GiB. Shared-down
gained only 2.9--3.4%,
changed the 2048-token top result in every repeat, and is rejected as a default.

The first production-default decision used
`cuda-q8-partner-production-validation.sh` rather than the exploratory variant
screen. The control disabled partner execution. The candidate deliberately
sets no `DS4_CUDA_Q8_F16_PARTNER_CLASSES` override, so it validates the measured
automatic candidate: T256 was admitted only for SM75 home/partner devices whose
startup peer-bandwidth measurements reach at least 18 GiB/s in both directions.
An explicit `t256` selector remains useful for expert diagnostics on other
targets, but such a run does not prove that automatic admission selected the
production default.

That production runner builds and runs the placement and multi-GPU exactness
tests, scores the same full-Q4 model with partner execution disabled and enabled
across the official 100-case Flash fixture using
`score_official --production-path`, and performs at least three paired prefill
repeats at the fixed 16K and 32K frontiers. It rejects missing production
path markers, non-T256 partner bindings, impure audit evidence, changed top-1
tokens, repeat nondeterminism, or quality/performance gates that do not pass.
The mandatory GPU test separately requires bit-exact local-versus-partner
projection output for all three supported classes. Full local/default logits
are not expected to be byte-identical: the control intentionally leaves an
overflowed T256 weight on native Q8 while the candidate admits its expanded
F16/cuBLAS path. That deliberate arithmetic change is covered by top-1,
teacher-forced NLL, first-token, and greedy-prefix gates instead.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REPEATS=3 \
./speed-bench/cuda-q8-partner-production-validation.sh
```

`MODEL` must be an existing absolute path; `PROMPT` defaults to the tracked
fixed prompt but is exported above to make the run self-contained. The runner
also accepts an absolute `QUALITY_MANIFEST` (default:
`gguf-tools/quality-testing/data/flash/manifest.tsv`), `QUALITY_CTX`,
`PREFILL_CHUNK`, `SKIP_BUILD`, `CREATE_ARCHIVE`, and `T256_VALIDATION_DIR`.
Production acceptance fixes `GPU_DEVICES=0,3,1,2`, `GPU_VRAM=auto`,
`STAGE_SPLIT=22`, `CTX_START=16384`, `CTX_MAX=32768`, `STEP_MUL=2`, and
`QUALITY_CTX=CTX_MAX+1`; the 100 cases remain short, but the scorer allocates
the same 32K session footprint so cache admission sees the tested production
memory pressure. This bounded pass does not validate 64K operation. The
three-class GPU exactness test is mandatory. The runner rejects
attempts to weaken the target or use fewer than three repeats. Return the generated
`t256-production-validation-<timestamp>.tar.gz`. Before accepting a result,
check that `run-status.txt` says `state=finished`, `summary.md` says
`Overall: **PASS**`, and `acceptance.json` has `"accepted": true`.

### All-native Q8 versus complete production FP16-cache quality endpoint

`cuda-q8-fp16-full-quality.sh` scores only the two decision-relevant production
endpoints selected after the T256 placement screen:

- every Q8 projection uses native Q8 because the complete FP16 expansion cache
  is disabled; and
- the complete production FP16-cache policy runs every active T256 output-B
  projection with FP16/cuBLAS on its NVLink partner and retains whatever
  non-T256 FP16 projections the production admission planner selects locally.
  The T256 inventory is fixed at 43/43—one live partner binding and one live
  physical F16 allocation per layer—but the non-T256 inventory is deliberately
  dynamic rather than pinned to a stale count from an earlier model plan.

The runner validates execution, not just requested policy. Across the 100-case
suite the native endpoint must record exactly 4,300 native T256 calls. The
FP16 endpoint must record exactly 4,300 partner-FP16 calls, with no local-FP16
T256 call and no T256 native fallback. The scorer exports binding and physical
allocation state after all 100 cases. Every exported F16/F32 binding and
allocation must be used and live, dead expanded-weight bytes must be zero, and
no non-T256 binding may be partner-offloaded. The summary records the exact
dynamic non-T256 class and descriptor inventory before reporting
official-continuation NLL, first-token accuracy, greedy-prefix length, and
API-reference metrics with a paired bootstrap interval. It never hashes the
model and does not require a new quant.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-fp16-full-quality.sh
```

Return `q8-fp16-full-quality-<timestamp>.tar.gz`. A valid archive has
`Experiment integrity: PASS`; the quality gate is a measured result and may
independently pass or fail.

### T256 execution-placement A/B

Each of the 43 layers has one active T256 output-B GEMM and one logical cache
binding. The binding names its home consumer and the physical device holding
the single FP16 expansion; partner placement no longer creates an unused fixed
peer-consumer duplicate.

`cuda-q8-t256-placement-ab.sh` measures and verifies five policies:

- `native` (`native-T256-only`): zero T256 FP16 bindings and 43 native-Q8
  layer paths; every other Q8 cache class remains enabled so this isolates
  T256 placement rather than changing the arithmetic of all dense projections;
- `all-local`: 43 local plus zero partner bindings, with all 43 active GEMMs
  local. T256 is ranked ahead of tied T32 entries so the complete class fits;
- `balanced`: even layers execute locally and odd layers execute on their
  partner: 22 local plus 21 partner bindings. Each pipeline stage is balanced
  within its own NVLink pair (11/11 for layers 0-21 and 11/10 for layers
  22-42);
- `overflow`: exactly 43 live bindings, with zero to seven natural partner
  overflows drawn only from eligible layers 15-21. The runner records the
  observed local/partner count instead of assuming the earlier 36/7 split; and
- `all-partner`: zero local plus 43 partner bindings.

Every run performs an untimed warm-up, then exports binding, physical
allocation/liveness, and runtime-audit tables before disabling the success
counters for the measured sweep. The summarizer rejects a run whose active
layer paths do not match its policy, whose 43 T256 bindings do not map one-to-one
to 43 live physical weights, or whose forced partner placement silently misses.
It also records all other binding shapes and total resident/dead bytes so cache
entries displaced by `all-local` remain explicit in the legacy mode.

For the forthcoming mixed Q4/IQ2 model that has enough VRAM for every dense
FP16 expansion at a 256K allocation, set `REQUIRE_COMPLETE_DENSE_CACHE=1` and
`CTX_ALLOC=262273` (256K context + 128 generation tokens + 1). This strict
mode still measures only the requested 2K--32K
frontiers, but creates the production session at the 256K memory pressure
before the untimed warm-up materializes the cache. It explicitly disables the
experimental attention-head split and exports the planner audit as well as the
binding, physical-allocation, and post-warm-up CUDA-memory tables. Strict mode
defaults to a predeclared 512 MiB minimum free-memory floor on every device;
raise it with `MIN_FREE_MIB`, but do not lower it after seeing a result.

The strict summary requires the exact same 344-candidate production plan in
every arm:
43 each for Q-A, Q-B, KV, output-A, output-B, and shared-down, plus 86 shared
gate/up weights. Every required FP16 candidate must be admitted, used, and
backed one-to-one by a live allocation; displaced non-T256 entries, dead bytes,
aliases, or unreferenced allocations are fatal. The `native-T256-only` control
still exports all 344 plan rows, but its 43 output-B rows must be deliberately
`unadmitted`; its other 301 candidates must be admitted and live.
Strict mode omits the natural-overflow arm because complete coverage makes it
redundant; it compares native-T256-only, all-local, balanced, and all-partner.
Legacy mode retains the five-arm overflow reproduction.

The mixed-model generator sizes its reclaim against the worst-case all-local
home footprint (an additional 1.375 GiB on the 22-layer home and 1.3125 GiB on
the 21-layer home relative to all-partner). This is a placement-neutral capacity
envelope, not a claim that all-local is optimal. The runner still fails closed
unless every tested arm preserves complete coverage and the free-memory floor.

One-repeat screen:

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=262273 \
REQUIRE_COMPLETE_DENSE_CACHE=1 \
MIN_FREE_MIB=512 \
REPEATS=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-t256-placement-ab.sh
```

Omit `CTX_ALLOC` and `REQUIRE_COMPLETE_DENSE_CACHE` when reproducing the
legacy full-Q4 screen where displacement is part of the measured cache-policy
tradeoff.

For the strict acceptance result, rerun with `REPEATS=4`; the four rotations
put every strict policy in every thermal/run-order slot. Legacy five-arm
reproduction still requires `REPEATS=5`. Return
`q8-t256-placement-<timestamp>.tar.gz`.

### T256 arithmetic-source isolation

`cuda-q8-partner-arithmetic-isolation.sh` explains a quality difference; it
does not select a production layer allowlist. It freezes the home cache and
the additive T256 tensor/layer set, then runs native Q8 plus five controlled
partner arithmetic arms. Adjacent comparisons isolate Tensor-GEMM arithmetic,
activation FP16 rounding, weight FP16 rounding, activation block-Q8
quantization, and finally the native INT32-dot/reduction structure. The scorer
records first-token IDs and signed target/greedy logit margins.

The default bounded pass contains the five cases whose first-token result
changed in the initial T256 isolation. Set `CASE_IDS` to a larger comma-
separated `case_NNN` set only after this causal pass succeeds.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
T256_LAYERS=15-21 \
CASE_IDS=case_017,case_025,case_030,case_048,case_056 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-partner-arithmetic-isolation.sh
```

Return `q8-partner-arithmetic-<timestamp>.tar.gz`. A valid archive reports
`Experiment integrity: PASS` and proves identical home bindings and identical
additive T256 layers across every arithmetic arm.

If a runner/check failure stops a pass after one or more complete arms, retain
the unpacked output directory and resume it without recomputing those arms:

```bash
Q8_ARITHMETIC_DIR="$PWD/q8-partner-arithmetic-<timestamp>" \
RESUME=1 \
SKIP_BUILD=1 \
./speed-bench/cuda-q8-partner-arithmetic-isolation.sh
```

Resume validates the complete experiment manifest, the prior runtime/scorer
commit, every reusable arm's row count and runtime dispatch, and refuses reuse
if the runtime or scorer changed.

### Exact native-Q8 T256 partner anchor

`cuda-q8-partner-native-exact.sh` is the prerequisite to any concurrent row
split. It transports the block-Q8 activations/scales over NVLink and invokes
the same SM75 exact-MMA projection used by the local path against the partner's
already-resident Q8 weights. It admits only T256 layers 15-21 and preserves the
frozen local cache plan. The runner refuses to benchmark unless the dedicated
two-GPU regression and all 100 production quality cases are byte-exact.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REPEATS=3 \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-partner-native-exact.sh
```

The performance phase is deliberately limited to 16K and 32K. Every repeat
must retain the same seven additive T256 bindings, record only
`native_q8_partner_hit/exact_sm75_mma`, and produce byte-identical frontier
logits. Return `q8-partner-native-exact-<timestamp>.tar.gz`.

### Tagged SM75 native-Q4 with all-partner T256

`cuda-sm75-native-q4-t256-ab.sh` is a focused integration runner; it does not
change the engine, either GGUF, or the native-Q4 dispatch. It requires separate
existing absolute paths in `MODEL` for the
standard full-Q4 GGUF and `NATIVE_MODEL` for its tagged SM75-native repack.
Both arms force the measured all-partner T256 winner: 43 active partner
bindings backed by 43 physical T256 expansions, with all output-B GEMMs
executing on the NVLink partners. After the untimed warm-up, the runner also
requires every expanded-weight allocation to be live and compares the standard
and native arms by their exact live non-T256 descriptor multiset: label, source
offset/bytes, dimensions, consumer/resident device, storage kind, and arithmetic.
There is no historical minimum binding-count assumption. It uses the production
`0,3,1,2` device order and 22/21 split and sweeps 2K through 32K. One
standard-then-native pair is the default; increase `REPEATS` only when repeat
statistics are specifically needed.

The runner records file sizes and exact-output evidence but deliberately does
not hash either approximately 153-GiB model. It reuses the existing models and
does not create or modify `NATIVE_MODEL`.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export NATIVE_MODEL="/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-Q4KExperts-SM75-native.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
STEP_MUL=2 \
PREFILL_CHUNK=2048 \
REPEATS=1 \
./speed-bench/cuda-sm75-native-q4-t256-ab.sh
```

Return `sm75-native-q4-t256-ab-<timestamp>.tar.gz`. The archive contains the
focused throughput comparison, bit-exact standard/native full-logit checks,
exact all-partner T256 allocation/binding/runtime evidence, frozen-cache checks,
exact live non-T256 inventory equality, zero dead expanded-weight allocations,
and per-run GPU telemetry.
A `PASS` from this runner accepts only that structural and performance
interaction. It does not replace or override the separate official T256
quality-isolation gate described below.

The first production-default result failed its predeclared per-case,
first-token, and greedy-prefix quality gates even though aggregate NLL improved.
It also established that the normal partner-priority tie-break changed the
home cache: six additional T32 and seven additional T256 projections switched
from native Q8 to F16/cuBLAS. Use the frozen-home quality-isolation runner
before attributing that result to either class. For candidate runs it removes
partner eligibility from the primary sort, proves that all home bindings remain
identical to the partner-disabled control, and then admits only previously
uncached T256 or T32 tensors on the partner. Quality failure is a measured
outcome rather than a runner failure; structural contamination still aborts.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
QUALITY_CTX=32769 \
REUSE_LOCAL_DIR="$PWD/t256-production-validation-20260826T035132Z" \
SKIP_BUILD=0 \
./speed-bench/cuda-q8-partner-quality-isolation.sh
```

Return `q8-partner-quality-isolation-<timestamp>.tar.gz`. Its `summary.md`
reports the T256-only and T32-only quality gates and the exact additive partner
layers. This pass is quality-only; benchmark only a class/subset that survives
it. `REUSE_LOCAL_DIR` is optional; when set, the runner verifies the prior
model path and size, quality fixture, 32K allocation pressure, GPU order, and
22/21 split before reusing its local-control evidence. Omit it if that source
directory is no longer present.

The completed frozen-home class pass rejected T32. It worsened NLL by 0.998%,
lost 0.47 mean greedy-prefix tokens, and lost seven aggregate token-level top-1
matches. Full T256 was directionally better on the broader evidence: NLL
improved 1.339%, the paired-bootstrap upper bound was negative, 59/100 cases
improved, mean greedy prefix gained 0.17 tokens, and aggregate token-level top-1
matches rose by five. It nevertheless failed the predeclared first-token gate:
four cases lost their first token and one gained it. Do not weaken that gate or
enable the full class implicitly. Isolate the additive layers 15--21 instead.

The runner accepts a candidate subset through `VARIANTS=t256` and
`T256_LAYERS`; the latter uses comma-separated layers or inclusive ranges. It
records the request in the manifest and rejects an archive unless the observed
additive binding set exactly matches it. The first bounded split is 15--18
versus 19--21, each reusing the already validated local control:

```bash
COMMON_LOCAL="$PWD/t256-production-validation-20260826T035132Z"

GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 QUALITY_CTX=32769 \
VARIANTS=t256 T256_LAYERS=15-18 REUSE_LOCAL_DIR="$COMMON_LOCAL" \
SKIP_BUILD=0 ./speed-bench/cuda-q8-partner-quality-isolation.sh

GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 QUALITY_CTX=32769 \
VARIANTS=t256 T256_LAYERS=19-21 REUSE_LOCAL_DIR="$COMMON_LOCAL" \
SKIP_BUILD=1 ./speed-bench/cuda-q8-partner-quality-isolation.sh
```

If one half passes, retain it and subdivide only a failing half. A final
combined subset still requires its own 100-case run because floating-point
path effects are not guaranteed to add linearly.

The completed T32 fusion A/B can eliminate redundant local and T32 runs:

```bash
REUSE_T32_DIR="$PWD/q8-t32-fused-ab-20260805T052315Z" \
SKIP_BUILD=1 \
RUN_GPU_TEST=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
VARIANTS=local,t32,t256 \
RUN_NSYS=0 \
bash ./speed-bench/cuda-q8-partner-offload-ab.sh
```

### T32 FP16-output fusion and partner-transfer A/B

`cuda-q8-t32-fused-ab.sh` isolates the formerly stubbed CUDA T32 capability
from the broader cache-policy experiment. It measures four process-isolated
variants at the fixed full-Q4 22/21 placement: established FP32-output T32 on
local weights, FP16-output plus fused RMS/RoPE on local weights, established
FP32-output T32 with partner overflow, and FP16-output/fused T32 with partner
overflow. The last variant returns an FP16 intermediate over NVLink, halving
the T32 result transfer before postprocessing on the home GPU.

On the fixed full-Q4 22/21 A/B, T32 partner execution with the established
FP32 result improved prefill by 12.2--13.5% over local execution and preserved
the top token at all nine measured frontiers. FP16-output fusion added only
0.5--1.0% over that partner path, while changing the top token at the 4096
frontier in every repeat. It therefore remains an opt-in diagnostic and is not
a production default.

The GPU regression first proves that local and partner FP16 intermediates are
bit-exact, their final FP32 outputs are bit-exact, the backend-owned scratch
path used by CUDA graphs is exact, and the new half-input RMS/RoPE kernel is
bit-exact with the established combined FP32 kernel after the same FP16
rounding. End-to-end cross-policy logits are reported rather than required to
match: partner VRAM changes which Q8 tensors receive cached FP16 weights, and
the fused helper deliberately changes its projection-output boundary from
FP32 to FP16. The runner instead requires exact repeat determinism within each
variant. All four short Nsight Systems traces are enabled by default.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
bash ./speed-bench/cuda-q8-t32-fused-ab.sh
```

Return `q8-t32-fused-ab-<timestamp>.tar.gz`. Set `RUN_NSYS=0` only for a
throughput-only rerun after the execution-path validations have passed.
After a complete four-variant run, a code-only fused-path correction can reuse
the unaffected old variants and capture only the new variants:

```bash
REUSE_OLD_DIR="$PWD/q8-t32-fused-ab-<old-timestamp>" \
NSYS_VARIANTS=new_local,new_partner \
SKIP_BUILD=0 \
RUN_GPU_TEST=1 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=8192 \
REPEATS=3 \
bash ./speed-bench/cuda-q8-t32-fused-ab.sh
```

That result is historical isolation evidence, not a current-production
acceptance run. `cuda-sm75-t32-f16-production-ab.sh` is the replacement
decision boundary. It leaves the stage-aware 344/344 Q8 plan, automatic
T32/T256/shared-down placement, prefill pipeline, and native indexer defaults
unchanged in both arms. On the qualified machine it also keeps the known-stable
prefill attention topology fixed: query-row splitting is disabled only for
pair 0 and remains enabled for pair 1. Prefill indexer ownership is independent:
its row split is active on both pairs, with pair 0 gathering selected rows home
for unsplit attention. The independent decode indexer row split remains
unchanged.

The runner executes an alternating control/fused A/B at 512, 4096, and 32768
tokens for both the existing mixed15 model and the new all43-Q3A4 gate/up
model. Each process records the exact 344-entry plan and canonical binding
state, pre/post GPU identity and power limits, routed quantization for all 43
layers, FP16-result call counts, and full-vocabulary raw logits. The current
stage-aware plan places all 43 T32 projections on partners (22 for the 22-layer
stage and 21 for the 21-layer stage), so the candidate must report matching
partner-fused and FP16-result counts with zero local fused calls. It rejects a
plan change, missing production path, power-limit change, within-arm
nondeterminism, or any frontier top-1 change. A passing performance/numerical
screen requires a median win at 32K on both models and permits no measured
frontier below 0.995x. The engine now enables the fused path by default when
its production eligibility checks pass; `DS4_CUDA_T32_F16_FUSED=0` and
`DS4_CUDA_NO_T32_F16_FUSED=1` remain rollback controls. The first
stable-topology mixed15 pair measured +9.05%, +2.04%, and +0.76% at 512, 4096,
and 32768 tokens respectively, and reduced partner result traffic from 212.27
to 124.92 GiB. The dual-model repeated matrix remains the qualification
harness for tracking the mixed15 and all43 production shapes, not an opt-in
gate.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REPEATS=3 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-t32-f16-production-ab.sh
```

Return `sm75-t32-f16-production-ab-<timestamp>.tar.gz`. The runner is
deliberately one-shot; after a GPU loss, retain the partial archive and start a
new matrix after reboot rather than combining process histories with resume.

`cuda-sm75-t32-f16-rowsplit-probe.sh` is retained only as a fault-reproduction
harness. It fixes fused T32 output on, explicitly restores attention and
indexer row splitting on pair 0, and executes exactly one mixed15 or all43 32K
prefill process. The mixed15 experiment lost the GPU1 PCIe endpoint after 19 pair-0
attention splits even though partner result traffic was reduced. It therefore
rejects the hypothesis that T32 traffic reduction makes the all-pairs row-split
topology stable. Pair 1, the 344/344 placement, all-partner T32 ownership,
direct peer routes, power limits, and every other production selector remain
fixed. There is no control arm and no resume path because a GPU-loss result
requires a reboot and must not be combined with a later process history.

`cuda-sm75-prefill-indexer-pair0-probe.sh` removes the earlier diagnostic
confound between pair-0 prefill attention and pair-0 prefill indexer splitting.
Both one-shot arms keep pair-0 attention disabled, retain pair-1 attention and
indexer splitting, use the default fused T32 path, and run the same 32K
production-shaped prefill. `VARIANT=indexer-on` alone runs pair-0 indexer
score/top-k 50/50 and gathers the partner's integer top-k rows home before
unsplit attention; `VARIANT=control` keeps that pair's indexer work home. Run
the candidate first. If it leaves all GPUs healthy, run the control separately
with `REFERENCE_DIR` pointing at the candidate directory; the control then
requires every dumped frontier-logit file to be byte-identical. Separate
processes are mandatory because a GPU-loss arm requires a reboot.

The accepted mixed15 candidate/control pair measured 510.42 versus 498.81
tok/s (+2.33%); the all43-Q3A4 pair measured 479.59 versus 468.11 tok/s
(+2.45%). Both comparisons were byte-exact, both candidate runs completed 600
pair-0 indexer splits with pair-0 attention disabled, and all four GPUs remained
healthy. Pair-0 prefill indexer splitting is therefore a production default;
this runner remains the exactness and performance regression boundary for the
independent policy. The corresponding cache policy is also independent: pair 0
allocates and updates only the native-F16 indexer mirror, while pair 1 retains
both attention-cache mirrors and its indexer mirror.

```bash
VARIANT=indexer-on \
MODEL_LAYOUT=mixed15 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
bash ./speed-bench/cuda-sm75-prefill-indexer-pair0-probe.sh

VARIANT=control \
MODEL_LAYOUT=mixed15 \
REFERENCE_DIR="$PWD/sm75-prefill-indexer-pair0-indexer-on-<timestamp>" \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
bash ./speed-bench/cuda-sm75-prefill-indexer-pair0-probe.sh
```

Repeat the same pair with `MODEL_LAYOUT=all43` when requalifying a change. This
is a prefill-indexer qualification; it does not alter or retest the
independently promoted one-token decode-indexer split.

Set physical GPU power limits before launching. The default requirement is
250 W on passive GPUs 0, 2, and 3 and the workstation board's full 260 W on
GPU 1:

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 250
sudo nvidia-smi -i 1 -pl 260
sudo nvidia-smi -i 2 -pl 250
sudo nvidia-smi -i 3 -pl 250

unset CUDA_VISIBLE_DEVICES
unset T32_F16_ROWSPLIT_PROBE_DIR

MODEL_LAYOUT=mixed15 \
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-t32-f16-rowsplit-probe.sh
```

If the mixed15 probe completes with healthy post-run devices, repeat it only
after reboot with `MODEL_LAYOUT=all43` and `ALL43_MODEL` pointing to the
all-Q3A4 gate/up model. Direction-reversal experiments remain separate
follow-ups: combining them with this probe would confound transfer volume with
source/destination role and compute placement.

### Production scalar-slot evidence

`cuda-sm75-production-scalar.sh` is the bounded, model-free acceptance driver
for scalar accumulator versions of the production SM75 Q4 gate/up, Q4 down,
and IQ2 gate/up kernels. It reuses `cuda_long_context_smoke` for nonzero
production correctness and `cuda_sm75_profile_harness` for the recorded layer-3
and layer-36 routing shapes; it never opens a GGUF.

The Q4 gate/up candidate deliberately consumes the existing standard GGUF
layout.  That isolates the scalar-accumulator effect from native-weight
packing, can be deployed without requantizing or transforming a model, and
provides the production baseline needed to decide whether an offline native
layout adds enough further benefit to justify its compatibility cost.

```bash
bash speed-bench/cuda-sm75-production-scalar.sh
```

The default run rebuilds both harnesses with verbose PTXAS diagnostics,
requires scalar kernels to have zero stack and spill bytes and no SASS
`LDL`/`STL`, preserves nonzero IMMA and the IQ2 fused slot updates, runs
bitwise base-versus-scalar CUDA smoke checks, runs memcheck
when available, and records two position-balanced samples for each early/late
Q4 gate, Q4 down,
IQ2 tile16, and IQ2 tile8 target. Each sample times ten production calls inside
the harness after an untimed correctness/audit warmup, excluding process setup.
IQ2 tile8 remains selected in both halves of its A/B so only the scalar
implementation changes. Set `TIMING_REPEATS` from 1 through 100 to change the
bounded calls per sample.

The default `TIMING_ROUNDS=2` is a quick, position-balanced pilot with two
samples per variant, not enough evidence for a production acceptance decision.
Increase it to a larger even value (for example `TIMING_ROUNDS=6`) for the
acceptance run. The summary reports median scalar/base speedup, percent change,
median absolute deviation, and coefficient of variation. Successful, failed,
and interrupted runs archive all evidence collected so far unless
`CREATE_ARCHIVE=0` is set.

`cuda-sm75-production-scalar-e2e.sh` is the next decision boundary after the
model-free evidence passes. It performs a balanced, fixed-frontier four-GPU
full-model A/B with only the validated scalar switches changed. For the
IQ2/IQ2/Q4 hybrid it jointly tests IQ2 gate/up plus Q4 down using the measured
25/18 pipeline split. For full Q4 it jointly tests Q4 gate/up plus Q4 down
using the memory-safe 22/21 split. It writes every raw CSV/log, a paired
frontier summary, GPU telemetry, model/binary/prompt provenance, and a
returnable archive. Two audited full sweeps run first and equally precondition
host page-cache/system state. They verify all 43 routed-layer quant recipes, every
expected layer/device dispatch, exact placement cardinality, all 344 dense-Q8
weight slices (including the exact eight tensor identities per layer), cache
decisions, and bit-exact raw FP32 full-vocabulary logits at every frontier. Any
mismatch stops before repeated timing. The timed AB/BA processes contain no
dispatch audit, Q8 lookup audit, or inter-frontier logit I/O. Instead, each
process first runs an untimed `CTX_START` prefill, recreates an empty session,
records the resulting Q8 cache ranges, and then starts measured frontiers. It
records those ranges again after the final synchronized frontier. The
runner requires exact pre/post, base/scalar, and validation/timed cache-state
equality. Thus every measured Q8 lookup starts from the same proven resident
set and no cache admission occurs during a timed sweep. The default two
repeats are explicitly a pilot; set `REPEATS=6` for a replicated measurement.
Production now selects the scalar specializations by default on SM75. The
runner still produces a controlled A/B: it explicitly sets all three scalar
switches to `0` for the base arm, then enables only the recipe-relevant paths
for the scalar arm. Set any of the following to `0` to roll back an individual
production path:

- `DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75=0`
- `DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75=0`
- `DS4_CUDA_MOE_IQ2_SCALAR_SM75=0`

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export RECIPE=hybrid
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,2,1,3 GPU_VRAM=auto REPEATS=2 \
  ./speed-bench/cuda-sm75-production-scalar-e2e.sh
```

Exact per-kernel NCU capture is opt-in:

```bash
RUN_NCU=1 NCU_USE_SUDO=1 \
  bash speed-bench/cuda-sm75-production-scalar.sh
```

### SM75 next-target A/B suite

`cuda-sm75-next-targets.sh` builds the CUDA regression binary, verifies every
new path against the retained baseline, and benchmarks one change at a time
before the combined configuration. New paths remain opt-in until the real
four-GPU suite demonstrates a win.

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export RECIPE=hybrid
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-sm75-next-targets.sh
```

Use `RECIPE=full-q4` with the stock Q4 model to test dense-Q8 token16 reuse
and compact Q4 gate/up tile16. Use `RECIPE=stock-q2` with the stock Q2 model
to test Q2_K-down IMMA. `PROMPT_MANIFEST` applies the same variants to a
tab-separated fixed prompt suite.

The individual production switches are:

- `DS4_CUDA_Q8_MMA_SM75_TOK16=1`
- `DS4_CUDA_MOE_Q4_GATE_TILE16_SM75=1`
- `DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75=1` (stage6 is the tile16 default)
- `DS4_CUDA_MOE_IQ2_STAGE6_SM75=1`
- `DS4_CUDA_MOE_IQ2_STAGE4_SM75=1`
- `DS4_CUDA_MOE_MIXED_TAIL_TILES=1`
- `DS4_CUDA_MOE_Q2_DOWN_MMA_SM75=1`

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### CUDA prefill audit

`cuda-prefill-audit.sh` captures both an ordinary 2K measurement and an Nsight
Systems trace of the same production pipeline. The trace starts after model
loading and workspace allocation, so the report contains the prefill itself
rather than the roughly 100 GiB model-residency setup. It also records the GPU
inventory and topology, NUMA layout, exact pipeline schedule and transfer-byte
accounting, GPU telemetry, kernel/API summaries, and the raw `.nsys-rep` and
SQLite trace.

The audit explicitly selects the established baseline: Q8-to-F16 cache and the
two-stage pipeline on, experimental attention head splitting off. It does not
enable the existing event-based MoE, attention-output, or layer-stage profilers;
those synchronize streams and alter the schedule being measured.

```bash
cd ~/ds4-iq2-q4
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-prefill-audit.sh
```

The default logical tier order is `0,2,1,3`. On a machine whose physical
NVLink pairs are `0<->1` and `2<->3`, this makes logical tier 2 the partner of
tier 0 and logical tier 3 the partner of tier 1 while the two pipeline homes
remain logical tiers 0 and 1. Override `GPU_DEVICES` only when the physical
topology differs.

### SM75 deep prefill audit

`cuda-prefill-deep-audit.sh` performs the next evidence pass in one command:

- repeated, uninstrumented 21/22 and 25/18 placement sweeps;
- one device-only tile audit recording actual owned pairs, tile count, active
  experts, padding, and owner identity for every layer and microbatch;
- four narrowly filtered Nsight Compute reports: IQ2 gate/up and Q4 down from
  one early and one late layer.

The tile audit extends the already-running tile-offset kernel. It adds no
per-layer launch, timing event, synchronization, or host read. Its buffers are
allocated before the measured prefill and copied once per GPU after the
existing prefill synchronization and timing window.

```bash
cd ~/ds4-iq2-q4
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-prefill-deep-audit.sh
```

The default suite measures context frontiers from 2K through 64K three times
for each placement. Set `PROMPT_MANIFEST` to a tab-separated `label<TAB>path`
file to run the same fixed frontiers over multiple prompt corpora. Set
`RUN_NCU=0` when only the placement and tile evidence is needed. The script
always packages partial results on exit and records the failing phase in
`run-status.txt`.

Use `DEEP_AUDIT_DIR` to select or resume a directory; the generic `AUDIT_DIR`
used by the earlier Nsight Systems script is intentionally ignored. To resume
only the four Nsight Compute reports after enabling performance-counter access:

```bash
export DEEP_AUDIT_DIR="$PWD/prefill-deep-audit-YYYYMMDDTHHMMSSZ"
SKIP_BUILD=1 SKIP_PLACEMENT=1 SKIP_TILE=1 \
  ./speed-bench/cuda-prefill-deep-audit.sh
```

If the system restricts counters to administrators, set `NCU_USE_SUDO=1` for
the resume run or change the NVIDIA driver profiling-permission policy.

### Full-Q4 SM75 evidence pass

`cuda-q4-prefill-evidence.sh` runs the fixed prompt suite on the untouched
22/21 production path, records every Q8-to-F16 cache decision, and captures an
Nsight Systems runtime trace. Full-model Nsight Compute is disabled: replaying
the 153.33 GiB four-GPU process either requires an unsafe kernel checkpoint or
relaunches a heavyweight application whose attachment boundary was not
reliable on the test system.

```bash
cd ~/ds4-iq2-q4
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

./speed-bench/cuda-q4-prefill-evidence.sh
```

Replace the example `MODEL` value with the exact full-Q4 file on the machine.
The script rejects a missing model rather than searching for one implicitly.
Set `PROMPT_MANIFEST` to reuse the fixed multi-prompt suite. Partial results
are archived on failure. `RUN_NCU` must remain zero.

Use `cuda-sm75-kernel-profile.sh` for Nsight Compute. It links only the CUDA
backend, opens no model file, launches no helper processes, and has no DS4 CLI
instance lock. Its routed-expert scenarios allocate exactly 128 resident
experts at the production 4096->2048->4096 dimensions. They cover Q4 gate/up,
Q4 down, IQ2_XXS gate/up, and Q2_K down. The harness hard-caps predicted device
state at 3 GiB and confirms the synthetic weight mapping was copied into VRAM.
The early and late assignments reproduce the recorded pair count,
active-expert count, and tile16 count; their per-expert histogram is explicitly
synthetic because the prior audit did not retain that histogram.

The dense-Q8 scenarios cover every production reduction template and the
corresponding dominant shape: attention q_b T32 (1024x32768), shared-down T64
(2048x4096), attention q_a T128 (4096x1024), and attention-output-b T256
(8192x4096), all at the 512-token runtime microchunk. A small Q8 kernel-replay capture validates
counter permissions, attachment, kernel filtering, and report export before
the main captures begin. The focused metric set records duration, launch
resources, achieved occupancy, eligible warps, IMMA and integer-pipe activity,
DRAM/L1/L2 activity and hit rates, atomic activity, and scoreboard/MIO stalls.

```bash
cd ~/ds4-iq2-q4
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-kernel-profile.sh
```

Set `PROFILE_GPU` to choose the physical SM75 device. The script exposes only
that GPU to the harness, where it becomes logical device zero, and validates
every report against the harness process, device, kernel name, and positive
duration. `PROFILE_SET=remaining` captures Q2_K down and all four dense-Q8
templates. It deliberately repeats T64/T128 because changes to the common Q8
loader can change every template's memory instruction stream.
`NCU_SET=targeted` or `NCU_SET=full` are slower opt-ins.

### SM75 routed-quant performance screen

`cuda-sm75-q3-q4-32.sh` is the model-free arithmetic and native-layout gate
for the format contracts in
[`SM75_Q3_Q4_32_DESIGN.md`](../SM75_Q3_Q4_32_DESIGN.md). It compares ten
M16xN8xK256 warp kernels with eight warps per 256-thread CTA:

- the existing affine Q4_K K32 arithmetic in a 1152-byte native tile, with
  independent scale and minimum accumulator chains;
- standard 110-byte Q3_K, losslessly repacked into an 880-byte native tile
  and executed exactly with K16 S8xU8 MMA plus `-4*bsum`;
- the experimental 104-byte Q3-32 format in an 832-byte native tile, using
  two K32 INT4 MMAs plus `-4*bsum`;
- the experimental symmetric 136-byte Q4-32 format in a 1088-byte native
  tile, using signed-B K32 INT4 MMA without an affine-minimum correction.
- Q2/Q3-50, Q2/Q3-75, Q3A4, Q3A6, Q3/Q4-50, and Q5-32, each in an exact
  size-neutral native tile matching the corresponding quality-screen law.

The harness carries opaque FP16 headers but deliberately measures only the
exact integer-accumulator component; it does not claim floating-output or
quantizer-byte exactness. It exhaustively validates Q3_K unpacking and
each Q3-32 bit plane, samples mixed Q3-32 states deterministically, checks
canonical/native byte round trips, and compares adversarial and seeded-random
MMA accumulators with separate scalar references. It reports both a
single-projection down-like shape and a dual-projection gate/up shape that
retains both accumulator chains and executes the SiLU-times-up epilogue.
Each has hot and L2-exceeding streamed-weight modes using the format's real
byte footprint. The runner records SASS/resource evidence, runs Compute
Sanitizer when available, and profiles both shapes for occupancy, tensor-pipe
activity, memory traffic, and stalls with a direct harness process.

IQ2_XXS is not forced into this linear-format harness. Instead, the runner
times the real early/late hybrid IQ2 gate/up + Q4 down production calls and
writes them to `iq2-production-control.csv`. Those absolute production times
are a separately labelled control, not a ratio against the bounded K256
tables.

This is not yet a quantizer, GGUF loader, routed-MoE dispatch, quality result,
or end-to-end speed claim. Standard Q3_K must later match the upstream
quantizer byte-for-byte. Q3-32 and Q4-32 require private versioned type/layout
tags and real-weight quality gates before a full model is generated.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
./speed-bench/cuda-sm75-q3-q4-32.sh
```

Return `sm75-q3-q4-32-<timestamp>.tar.gz`. The archive contains four timing
tables (`down-hot`, `down-streamed`, `gate-up-hot`, and
`gate-up-streamed`), exact layout/arithmetic gates for all ten linear formats,
the production IQ2 control, and per-shape Nsight reports. This pass ranks
formats only; it does not choose a deployment recipe.

### SM75 packed-INT4 Q4 experiment

`cuda-sm75-int4-mma-experiment.sh` is a bounded, model-free screening
experiment for the current Q4 nibble-to-INT8 path. Each warp computes a
16-token by 8-output tile over K=256, reusing each B fragment across two
8-token MMA tiles; eight independent warps run in each 256-thread CTA. It
compares six alternatives:

- the shipping-style two-`m8n8k16` INT8 baseline;
- direct `m8n8k32` INT4 with runtime packing from standard Q8/Q4 bytes;
- standard Q8 with a same-row, block-local `group32` Q4 layout;
- an MMA-native, size-neutral Q4 code-payload layout;
- size-neutral MMA-native Q8/Q4 value payloads (an idealized upper bound);
- an all-u4 correction formulation retained as a measured reference.

The preferred decomposition is `a_s8 = low_u4 + 16*high_s4`, so it uses one
`u4 x u4` and one `s4 x u4` MMA without introducing another zero-point term.
The harness proves only the scaled integer component
`sum_j scale_j * sum_k(a_k * q_k)`. It does not exercise Q4_K `d`, `dmin`,
packed scale/min decoding, the Q8_K scale and `bsums`, the minimum correction,
or the production floating-point accumulation order.

The measurements screen three narrower questions: hot-cache instruction
economics, streaming/layout economics, and whether a native packing is worth a
production-shaped prototype. The script runs both a 16-weight-case hot test and
a 16,384-weight-case streaming-size test (more than 17 MiB of uniquely touched
Q4 payload/scale data, versus the RTX 8000's 6 MiB L2) while keeping
activations in an independent 16-case hot set. Variant order is rotated and
reversed across nine rounds; the summary reports the median and range rather
than a single fixed-order sample.
Native-Q8 (`native-A`) results are an optimistic upper bound because the
benchmark starts after the 16-token transpose and nibble-plane transformation;
that recurring transformation cost is excluded.
The all-u4 alternative also needs one 16-bit weight sum per 32-value subgroup,
increasing a full Q4_K block from 144 to 160 bytes (+11.1%) unless those sums
are recomputed at runtime.

The experiment confirms emitted 8x8x32 IMMA instructions, runs Compute
Sanitizer when installed, benchmarks every layout, and captures focused Nsight
metrics. It never opens a GGUF or changes production dispatch. Before any
dispatch change, the winning idea still requires production-shaped gate/up and
down prototypes plus a full, bitwise Q4_K x Q8_K regression covering headers,
minimum correction, multiple blocks, and the production float reduction order.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-int4-mma-experiment.sh
```

Return the generated `sm75-int4-mma-<timestamp>.tar.gz`. These results can
reject unattractive layouts, but cannot by themselves justify production
dispatch.

### SM75 production-shaped Q4 down native-packing audit

`cuda-sm75-q4-down-native.sh` exercises the model-free, production-shaped
Q4_K down-projection follow-up. It compares the standard layout with native-W,
prepacked native-A/W consumption, and the combined activation-pack plus
native-A/W path. `native-aw-consumer` measures only the consumer after native-A
packing is complete; `native-aw-combined` charges the activation transform to
the timed result, while `pack-a` reports that transform separately. The two
N-split candidates are retained as bounded comparisons. Fourteen rounds
balance every one of the seven variants across every sample position. The early
layer-3 shape has 1,879 routed token/expert pairs,
99 active experts, 183 tile16 launches, and 1,049 padded slots. The late
layer-36 shape has 2,186 pairs, 76 active experts, 189 tile16 launches, and 838
padded slots. Both use the full 3,072-row production token/slot surface with
the owned pair IDs scattered through it. The harness validates activation
packing exactly, requires initialized owned outputs to be bit-exact against
its own standard-path reference, and verifies that every unowned output stays
poisoned before running the randomized, position-rotated benchmark rounds for
both recorded shapes. This is not a claim of bit-exact agreement with the
production kernel, which is built under a different compiler mode.

The same benchmark logs contain a separate bounded producer comparison for a
direct-native Q8_K quantizer. It preserves the canonical signed
scale, all 16 `bsums`, and every low/high nibble-plane byte, but writes the
native 292-byte record directly. Its timing compares one direct launch against
the complete canonical quantize-plus-pack operation (two launches), including
the canonical record's 292-byte write and 292-byte reread. The bounded result
motivates an explicitly opt-in production candidate; it does not by itself
change the canonical production default.

The script builds only an `sm_75` cubin, rejects any GPU other than compute
capability 7.5, runs Compute Sanitizer when available, and checks the emitted
standard/native kernels with `cuobjdump`. Focused Nsight Compute captures cover
one representative standard, native-W, and native-A/W consumer kernel, plus
the activation-pack transform that the combined result charges. Captures cover
launch resources, occupancy, IMMA utilization, memory behavior, and scheduler
stall metrics. Set `NCU_USE_SUDO=1` on systems with restricted performance
counters; the same `sudo -E` command is used for metric discovery and capture.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-down-native.sh
```

No `MODEL` variable is needed and no GGUF is opened. Return the generated
`sm75-q4-down-native-<timestamp>.tar.gz`; partial evidence is archived on
failure or interruption as well as on success. The result is an implementation
audit, not by itself evidence for changing production dispatch.

If the initial run completed its build, SASS, correctness, sanitizer, and both
benchmarks but failed in Nsight preflight or capture, resume only the Nsight
phase against the still-present output directory. For example:

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

Q4_DOWN_NATIVE_DIR="$PWD/sm75-q4-down-native-20260803T180034Z" \
RESUME_NCU=1 \
PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-down-native.sh
```

Resume mode rejects any run that did not fail in an Nsight phase and validates
the recorded source, Makefile, binary, SASS, correctness, sanitizer, benchmark,
and telemetry evidence before skipping it. It never overwrites the original
evidence or failure archive: new provenance, status, and Nsight files go under
`resume-<timestamp>/`, and the resulting archive is named
`sm75-q4-down-native-<original-timestamp>.resume-<timestamp>.tar.gz`.

### SM75 direct native-Q8 production A/B

The direct native-Q8 producer is the production default for single-token
decode at both activation boundaries consumed by the tagged SM75 routed-MoE
kernels: the model input feeding Q4-32 or Q3A4 gate/up and the intermediate
activation feeding Q4-32 down. Decode's shared prequantization remains one
launch: the producer preserves the indexer's Q8_0 bytes while emitting the
Q8_K half in native form. Prefill retains the canonical Q8_K-plus-pack path.
`DS4_CUDA_MOE_DIRECT_NATIVE_Q8=1` explicitly selects direct production for
both phases; `DS4_CUDA_MOE_DIRECT_NATIVE_Q8=0` or
`DS4_CUDA_NO_MOE_DIRECT_NATIVE_Q8=1` is the full rollback.

`cuda-sm75-direct-native-q8-production-ab.sh` is the acceptance test. It runs
the mixed15 and all43 models under the same stable 22/21 four-GPU topology,
runs one canonical/direct pair by default, and measures both
prefill and 256-token decode at PP512, PP4096, and PP32768. Each process must
report exclusive canonical or direct dispatch at both boundaries for both
prefill and decode. The runner also requires identical dense-Q8 placement,
healthy fixed GPU identities and power limits, and byte-identical logits for
16 decode tokens at every frontier for each model. It is deliberately
one-shot: an interrupted or GPU-loss run is archived rather than resumed.
Increase `REPEATS` only when additional run-order/noise characterization is
specifically needed; it is not required for the production correctness gate.

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-direct-native-q8
git pull --ff-only

sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 250
sudo nvidia-smi -i 1 -pl 260
sudo nvidia-smi -i 2 -pl 250
sudo nvidia-smi -i 3 -pl 250

unset CUDA_VISIBLE_DEVICES
unset DIRECT_NATIVE_Q8_PRODUCTION_AB_DIR

MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
REPEATS=1 \
TG_TOKENS=256 \
EXACT_TOKENS=16 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-direct-native-q8-production-ab.sh
```

Return `sm75-direct-native-q8-production-ab-<timestamp>.tar.gz`. Promotion
requires a complete archive; the earlier one-GPU synthetic audit remains
supporting resource and microbenchmark evidence only.

The accepted `20260902T211111Z` archive passed byte-exact 16-token decode
logits at PP512, PP4096, and PP32768 for both mixed15 and all43, with exclusive
producer dispatch at both boundaries. Direct decode improved all six points by
0.052% to 0.370%. Prefill was mixed (-0.752% to +0.349%) and regressed four of
six points, so only decode was promoted.

### SM75 production-shaped Q4 gate/up native-packing audit

`cuda-sm75-q4-gate-up-native.sh` is the standalone, model-free gate/up
follow-up. It runs exact pack and output correctness, Compute Sanitizer when
available, balanced early/late benchmarks, GPU telemetry, SASS inspection,
and focused Nsight Compute captures in one command. The early layer-3 shape
has 1,879 routed pairs, 99 active experts, and 282 synthetic tile8 records;
the late layer-36 shape has 2,186 pairs, 76 active experts, and 331 tile8
records. Both use 512 tokens, six selected experts per token, a 4,096-wide
input, and a 2,048-wide gate/up result.

The fourteen benchmark variants retain all controls and add scalar-register
experiments for the stage7 and warp16 native-A/W consumers. The complete set is
`standard`, `standard-warp16`, `native-w`, `native-aw-consumer`,
`native-aw-warp16-consumer`, `native-aw-stage7-consumer`,
`native-aw-combined`, `native-aw-warp16-combined`,
`native-aw-stage7-combined`, `native-aw-stage7-scalar-consumer`,
`native-aw-warp16-scalar-consumer`, `native-aw-stage7-scalar-combined`,
`native-aw-warp16-scalar-combined`, and `pack-a`. Consumer variants exclude
the already-completed activation transform; combined variants charge
activation packing plus the matching consumer, and `pack-a` isolates that
transform. Native weight packing is an offline layout operation and is
excluded from timed consumer work. By default, 28 rounds rotate all fourteen
variants evenly through every sample position twice. Each sample requests at
least 20 launches and may automatically use more to reach a stable 100 ms
consumer timing window; both requested and effective launch counts are
recorded and validated.

For each scenario, `benchmark-*-scalar-comparison.csv` pairs scalar only with
its matching base stage7 or warp16 variant, separately for consumer and
combined timing. It records median time, delta, speedup, sample coefficient of
variation and median absolute deviation, plus the independently timed pack-A
share. This avoids cross-shape or cross-scenario comparisons.

Correctness requires exact activation and weight packing, bit-exact owned
outputs against the harness's standard path, and untouched poison in every
unowned output. The gate/up target deliberately keeps the shipping
`--use_fast_math` compilation mode because its epilogue executes millions of
`expf` operations; the bit-exact claim is therefore an intra-binary comparison
with the harness's own standard path, not an external production reference.
SASS validation requires the expected m8n8k16 or m8n8k32 packed-INT4 forms.
The evidence deliberately keeps three different concepts separate:
`sass-summary.csv` counts `LDL`/`STL` instructions,
`ptxas-resource-summary.csv` reports compiler stack-frame and spill bytes, and
`scalar-resource-comparison.csv` reports CUDA `localSizeBytes`, raw and
Turing-allocation-rounded registers, shared memory, blocks, warps, and
occupancy. `scalar-local-memory-comparison.csv` and
`scalar-acceptance-summary.csv` compare each scalar kernel directly with its
base. A scalar candidate passes only when its SASS local instructions, PTXAS
stack/spill bytes, CUDA local bytes, register ceiling, and exact occupancy gate
all pass. A missed experimental gate is recorded and warned about but does not
discard the timing and profiling evidence; it is not an eligibility or
production-dispatch claim.

Nsight produces eleven focused reports: standard, base stage7, scalar stage7,
base warp16, and scalar warp16 for both early and late, plus one early pack-A
capture. Optional discovered metrics include local-load/store sectors and
bytes so base-versus-scalar local traffic can be measured directly. The script
only observes clocks and uses Nsight's `--clock-control none`; it never changes
GPU clocks or power settings.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-gate-up-native.sh
```

No `MODEL` variable is needed and no GGUF is opened. Return
`sm75-q4-gate-up-native-<timestamp>.tar.gz`; the script also archives partial
evidence on failure or interruption. This is an implementation audit and does
not enable or alter production dispatch.

If a completed build/SASS/correctness/sanitizer/benchmark pass fails only in
the Nsight phase, resume it without replacing the original evidence:

```bash
Q4_GATE_UP_NATIVE_DIR="$PWD/sm75-q4-gate-up-native-YYYYMMDDTHHMMSSZ" \
RESUME_NCU=1 \
PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-gate-up-native.sh
```

Resume mode revalidates provenance, build freshness, separate SASS/PTXAS/CUDA
resource evidence and scalar acceptance classifications, correctness,
sanitizer, both balanced benchmarks and their pair tables, and telemetry.
New files go under `resume-<timestamp>/`; the resulting archive is named
`sm75-q4-gate-up-native-<original-timestamp>.resume-<timestamp>.tar.gz`.

### Comprehensive stock-Q2 and remaining-kernel pass

`cuda-sm75-comprehensive-audit.sh` combines the full stock-Q2 production pass
with the bounded missing-kernel captures. The production half runs the fixed
prompt suite, records routed tile occupancy and every Q8-to-F16 cache decision,
and generates an Nsight Systems kernel-time distribution. The NCU half opens no
GGUF and profiles early/late Q2_K down plus all four dense-Q8 templates. Partial
results are archived if either half fails.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL_Q2="$PWD/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

NCU_USE_SUDO=1 \
./speed-bench/cuda-sm75-comprehensive-audit.sh
```

The command deliberately names and exports the actual stock-Q2 file; neither
script searches for a model or inherits the unrelated `MODEL` variable.
Set `PROMPT_MANIFEST` for the fixed multi-prompt suite. Set `PROFILE_SET=all`
to repeat all four routed-expert kernel families and all four native-Q8
templates under one build instead of capturing only the remaining gaps.

### SM80 to SM75 dispatch and resource audit

[`CUDA_SM80_TO_SM75_DISPATCH_MATRIX.md`](../CUDA_SM80_TO_SM75_DISPATCH_MATRIX.md)
is the source-reviewed architecture matrix for the normal DeepSeek V4 Flash
CUDA path. It covers dense projections, attention, indexer, all four routed
expert quant combinations, decode, cache-dependent dispatch, and four-GPU
orchestration. Unknown compiled resources are marked explicitly rather than
inferred from source.

Generate paired cubin resource and instruction reports without loading a model:

```bash
cd ~/ds4-iq2-q4
./speed-bench/cuda-sm-dispatch-resource-audit.sh
```

The script compiles `ds4_cuda.cu` for both `sm_75` and `sm_80`, records the
complete source kernel/launch/gate inventories, extracts per-function cubin
resource usage, and counts IMMA, HMMA, DP4A, load, and barrier instructions.
It packages the report as `sm-dispatch-resource-<commit>-<timestamp>.tar.gz`.
No `MODEL` variable is required because this audit does not execute inference.

To reparse and repackage an existing audit without recompiling its CUDA
objects, name its existing directory explicitly:

```bash
OUT_DIR="$PWD/sm-dispatch-resource-<commit>-<timestamp>" \
REPARSE_ONLY=1 \
./speed-bench/cuda-sm-dispatch-resource-audit.sh
```

Object-wide instruction totals are compiler-coverage checks only. Both objects
contain symbols excluded by runtime architecture gates; use the paired
per-function tables together with the production dispatch matrix.

### SM75 prefill critical-path and pair attribution

`cuda-sm75-critical-path-audit.sh` is the bounded follow-up to the T256
partner-policy screen. It requires the tagged mixed-Q4/IQ2 model and its
production 256K allocation. It does not repeat a throughput A/B. It records
two 8192-token Nsight Systems traces with the fixed 22/21 split: `0,3,1,2` and the
pair-preserving swap `3,0,2,1`. Thus each layer stage runs once on physical GPU
0 and once on physical GPU 3 while each home GPU retains its direct NVLink
partner. The same production-shaped hybrid IQ2/Q4 routed work is also timed
on GPUs 0 and 3, with alternating order, to separate hardware/clock behavior
from layer-range behavior.

The opt-in `DS4_CUDA_CRITICAL_PATH_NVTX=1` annotations cover waves,
microbatches, stages, layers, inter-stage handoffs, output, and T256 partner
projections. They do not create CUDA events, synchronize, or add waits. The
SQLite summarizer attributes existing kernel and copy timestamps through their
CUDA correlation IDs and emits the complete operation table, per-microbatch
GPU envelopes, stage/device totals, the 2x2 physical-GPU versus stage factors,
and same-work GPU medians. Model hashing remains disabled.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

PROFILE_TOKENS=8192 \
CTX_ALLOC=262273 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-critical-path-audit.sh
```

Return `sm75-critical-path-<timestamp>.tar.gz`. The archive is also produced
on interruption or failure and includes both `.nsys-rep` files, SQLite
exports, 100-ms GPU clock/power telemetry, harness logs, and derived CSVs.

If a run fails in `same-work-harness`, reuse the completed Nsight Systems
traces instead of loading the model again:

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export RESUME_DIR="$PWD/sm75-critical-path-20260805T071911Z"

SKIP_BUILD=0 \
./speed-bench/cuda-sm75-critical-path-audit.sh
```

Resume mode validates the model, layouts, split, failed phase, reports,
SQLite exports, and telemetry before rerunning only the bounded same-work
harness and summarizer. It preserves the failed run's provenance and writes a
separate `.resume-<timestamp>.tar.gz` archive.

### SM75 IQ2 tail-policy A/B

`cuda-sm75-iq2-next-ab.sh` preserves the accepted production-tail validation.
Its bounded harness verifies that one IQ2 MMA tile8 for every residual of
1--8 pairs is bit-exact with the retained tail4 rollback. A separate
alternating four-GPU A/B measures both policies; native-Q4 down retains its
existing 16/8/4 planner in both variants. Tail8 is the SM75 production default,
and `DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75=0` is the explicit rollback. The rejected
512-thread IQ2 CTA is no longer a production dispatch option.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
CTX_START=2048 \
CTX_MAX=8192 \
CTX_ALLOC=262273 \
REPEATS=3 \
PROFILE_GPU=0 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-iq2-next-ab.sh
```

### Mixed-Q4/IQ2 and balanced-T256 production profile

`cuda-sm75-native-q4-t256-profile.sh` profiles the accepted combined
configuration without repeating its throughput A/B. It opens the NFS-backed
model exactly once and captures one timed 8192-token frontier at the production
256K context allocation with Nsight Systems. No placement/class environment
override is set: the trace must prove that the engine selected its automatic
balanced policy, qualified both SM75 NVLink pairs, materialized all 344 dense
F16 weights, and produced exactly 22 local plus 21 odd-layer partner T256
bindings. It also requires the tagged native-Q4 dispatch and the exact mixed
Q4/IQ2/Q4 routed recipe.

Nsight Compute does not reopen the GGUF. Production-shaped harnesses capture
representative early/late native-Q4 gate/down and IQ2 tile16/tile8 calls, plus
512-token T32 and T256 dense-F16 cuBLAS calls. The reports include occupancy,
compute/memory throughput, cache traffic, and warp-stall sections. The
production trace separately attributes stage/device work, communication,
activation conversion, both NVLink copies, partner cuBLAS, and attention
row-split kernels/copies through semantic NVTX correlation ranges. Its summary
separates local FP16 GEMMs, indexer work, format/quantization/packing, norms and
hyperconnections, RoPE, compressor work, and any genuinely unknown kernels;
it also emits per-device/per-stage group totals. A 200 ms `nvidia-smi` stream records utilization,
power, clocks, temperature, and memory while a device-buffer tile audit records
real pair/tile/padding/ownership counts with one final host copy.
At `PROFILE_TOKENS=32768`, targeted Nsight Compute is enabled automatically
for production-kernel harnesses representing late-frontier indexed top-K and
mixed online attention calls; set `RUN_ATTENTION_NCU=0` only to omit them.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PROFILE_TOKENS=8192 \
CTX_ALLOC=262273 \
PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

For the requested 32K production trace and targeted attention pass, use the
same command with:

```bash
PROFILE_TOKENS=32768 \
CTX_ALLOC=262273 \
NCU_SET=attention \
RUN_ATTENTION_NCU=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

To rank the post-row-split production bottlenecks without another Nsight
Compute pass, capture one unsynchronized 32K Systems trace. The script clears
all inherited `DS4_*` variables, explicitly records `xdev_sync=disabled`, and
fails unless both qualified row splits are active:

```bash
PROFILE_TOKENS=32768 \
CTX_ALLOC=262273 \
RUN_NCU=0 \
RUN_ATTENTION_NCU=0 \
NCU_USE_SUDO=0 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

If the targeted 32K attention reports already exist and only the genuine
production trace is missing, avoid repeating them:

```bash
unset REUSE_NSYS_DIR REUSE_Q4_NCU_DIR
PROFILE_TOKENS=32768 \
CTX_ALLOC=262273 \
RUN_NCU=0 \
RUN_ATTENTION_NCU=0 \
NCU_SET=full \
NCU_USE_SUDO=0 \
SKIP_BUILD=1 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

Return `sm75-native-q4-t256-profile-<timestamp>.tar.gz`. Model hashing is
disabled. If the GGUF is cold in the Linux page cache, its single NFS
startup may still take roughly 25 minutes at 100--105 MiB/s; the later Nsight
Compute passes do not touch that file.

If a run completed its production trace but stopped during export or summary,
reuse that trace in a new result directory and proceed to Nsight Compute
without opening the GGUF payload again:

```bash
export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export REUSE_NSYS_DIR="$PWD/sm75-native-q4-t256-profile-20260826T144211Z"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PROFILE_TOKENS=8192 \
CTX_ALLOC=262273 \
PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=1 \
./speed-bench/cuda-sm75-native-q4-t256-profile.sh
```

Resume mode first requires the prior manifest to match the model path and
size, prompt, device order, VRAM policy, stage split, profile frontier,
allocation, chunk, pipeline size, and T256 policy. It also independently
requires exactly one benchmark row at the requested frontier. This prevents an
8K capture from being silently relabeled as a 32K trace. It then copies and
revalidates the existing report, benchmark, cache
snapshots, allocation/memory evidence, tile audit, and exact balanced binding
inventory. It reruns the corrected
summaries and the model-free Nsight Compute harnesses in a fresh output
directory; it does not mutate the original failed evidence.

### Four-GPU Q4 clock and power normalization

`cuda-sm75-q4-clock-audit.sh` diagnoses the physical-board effect without a
GGUF or multi-GPU placement. It times identical production-shaped early and
late Q4 work on GPUs 0, 1, 2, and 3 in alternating order, then repeats late Q4
with both NVLink pairs active concurrently. Its telemetry includes every
clock-event/throttle-reason field supported by the installed driver.

The default second phase is a reversible common-clock diagnostic at 1620 MHz.
It does not claim 1620 MHz is the desired production setting: the purpose is
to determine whether Q4 timing converges at equal clocks and whether GPU2
shares GPU3's behavior. Original persistence mode, power limits, and unlocked
clock state are restored on success, failure, or interruption.

`TARGET_SM_CLOCK_MODE=1` selects NVIDIA's closed-loop clock-locking mode, which
is documented as potentially improving performance per watt. This is relevant
when the default mode cannot sustain the requested clock because of software
power capping; it must still be measured rather than assumed beneficial.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

NORMALIZE=1 \
TARGET_SM_CLOCK=1620 \
RAISE_POWER_LIMIT=0 \
USE_SUDO=1 \
./speed-bench/cuda-sm75-q4-clock-audit.sh
```

Return `sm75-q4-clock-<timestamp>.tar.gz`. Do not raise power limits until the
baseline report establishes the active clock-event reason and each board's
reported default and maximum limits.

### SM75 Q4 memory-clock power-headroom sweep

`cuda-sm75-memory-clock-sweep.sh` is the bounded follow-up when the clock audit
shows software power capping below the requested SM clock. It compares GPU1 as
the healthy passive-board control against GPUs2/3, discovers their common
supported memory clocks, and samples up to five clocks in the upper operating
range. Each point records actual SM/memory clocks, power, temperature, throttle
reasons, exactness, and production-shaped Q4 early/late timings.

The sweep does not change power limits. Runtime SM and memory locks are reset,
and original persistence settings restored, on success, failure, or interrupt.
Run it only when these GPUs have no other compute processes:

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

DEVICES=1,2,3 \
TARGET_SM_CLOCK=1620 \
TRIALS=3 \
REPEATS=100 \
USE_SUDO=1 \
./speed-bench/cuda-sm75-memory-clock-sweep.sh
```

Return `sm75-memory-clock-<timestamp>.tar.gz`. `decisions.csv` reports the
best measured clock per device and scenario, but a production clock should be
chosen only if both Q4 scenarios improve and the full native-Q4 benchmark then
confirms the result.

Some Turing boards expose only a full-performance memory state and a 405 MHz
idle state whose supported graphics ceiling is also low. In that case the
script records `outcome=not-applicable` and exits successfully without changing
clocks; forcing the idle memory state is not a useful power-headroom test.

### SM75 32K indexer audit

`cuda-sm75-indexer-audit.sh` isolates the ratio-4 indexer at the final
production 512-token chunk of a 32K prefill. It does not open a GGUF. Every
existing WMMA score tile (128/64/32/16) must reproduce the shipping WMMA128
scores bit-for-bit, including the causal `-inf` region. Both the shipping
8192-entry top-k and its existing chunked-tree alternative must return the
exact ordered top-512 set.

The timing pass reports score and top-k costs separately. The optional Nsight
Compute pass records launch resources, occupancy, FP16 Tensor Core activity,
memory hierarchy traffic, and barrier/scoreboard/MIO stalls. Optional metrics
are selected from the installed Nsight version instead of making a renamed
counter abort the whole audit.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
TIMING_REPEATS=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-audit.sh
```

Return `sm75-indexer-audit-<timestamp>.tar.gz`. This pass compares existing
exact paths before any persistent-F16 cache or SM75-specific accumulator
epilogue is admitted to production.

`cuda-sm75-indexer-wmma64-production-ab.sh` is the production transfer gate
for the best existing score tile found by that audit. It interleaves the
shipping WMMA128 path with WMMA64 at 2K, 4K, 8K, 16K, and 32K under the fixed
256K allocation. The script validates both kernels against WMMA128 in the
bounded harness, audits every production score dispatch, requires identical
dispatch counts, and compares every frontier-logit file byte-for-byte.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=262273 \
REPEATS=3 \
RUN_HARNESS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-wmma64-production-ab.sh
```

Return `sm75-indexer-wmma64-production-<timestamp>.tar.gz`. A harness-only
gain is not sufficient to change the production dispatch; promotion requires
the exact interleaved full-model result.

If the uninstrumented A/B does not reproduce the isolated kernel gain,
`cuda-sm75-indexer-production-trace.sh` captures one genuine, unsynchronized
32K production trace for each exact score tile. It retains the 256K allocation,
balanced T256 placement, mirrored-KV attention row split, and 512-token
pipeline microbatch. The trace comparison reports score time per stage/device,
position/stage score time, total score launches, pipeline GPU span, and trace
throughput. This separates two materially different outcomes: either WMMA64 is
no faster across the real production launch population, or its saved GPU work
is overlapped and absent from the pipeline critical path. The synchronizing
indexer stage profiler is not enabled.

The runner creates one detached Git worktree pinned to the caller's committed
HEAD, builds there once, and uses that same binary and post-processing source
for both captures. It records and compares the `ds4-bench` SHA-256. This is
intentional: a paired capture lasts long enough that another experiment can
otherwise switch or edit the shared checkout between the first and second
post-processing pass. `SKIP_BUILD=1` is therefore rejected by this runner.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_ALLOC=262273 \
PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-indexer-production-trace.sh
```

Return `sm75-indexer-production-trace-<timestamp>.tar.gz`. Both inner traces
export the 32K frontier logits and must match byte-for-byte. Each trace also
audits all indexer dispatches and rejects mixed score-kernel use. The general
production profiler accepts `INDEXER_SCORE_TILE=128|64`; 128 remains the
shipping default and 64 is diagnostic only. `score-position-stage.csv` gives
the paired result for every real context-position/stage cell.

The 2026-08-27 32K capture rejected WMMA64 as a production replacement:
WMMA128 used 9612.942 ms for 1260 score launches and WMMA64 used 9640.108 ms
(0.997x). WMMA64 won only 57 of 120 matched position/stage cells and was 0.995x
at the final 32256 position. The 1.003x trace-throughput difference was not
backed by reduced score work and is consistent with the preceding interleaved
uninstrumented A/B. Keep WMMA128; do not base subsequent indexer work on the
isolated WMMA64 event-timing result.

With WMMA128 retained, `cuda-sm75-indexer-f16-operands.sh` evaluates the next
mechanism independently: move the exact FP16 rounding already performed inside
every WMMA128 score tile into reusable operand tensors. The bounded harness
keeps the WMMA accumulation order and epilogue unchanged, requires bit-exact
scores and ordered top-512 selection, alternates paired timing order, and
includes one Q materialization in candidate end-to-end time. Full-history K
conversion is reported separately because a production sidecar would be
updated when each compressed row is written.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
TIMING_ROUNDS=5 \
TIMING_REPEATS=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-f16-operands.sh
```

Return `sm75-indexer-f16-operands-<timestamp>.tar.gz`. The candidate is still
diagnostic: no production storage or dispatch changes are made. At 256K, the
21 ratio-4 indexer layers would need 336 MiB of persistent FP16 K storage
(160 MiB in stage 0 and 176 MiB in stage 1); a live 512-token FP16 Q operand is
8 MiB on the executing tier. Promotion requires a subsequent full-model exact
frontier-logit and interleaved throughput A/B.

The standalone FP16-operand result showed that conversion/DRAM reduction by
itself is not material: its paired median was approximately 1.004x end to end.
`CANDIDATE=streaming64` therefore tests a separate SM75 mechanism. Each warp
loads its eight FP16 K fragments once and retains them across the ordered
64-head reduction; the kernel removes shared K and accumulator storage and
warp-broadcasts the 16 unique scalar weights. The runner records a failed
production-resource gate for local-memory traffic, more than 128 registers per
thread, or more than 16 KiB static shared memory—the envelope needed for at least
four 128-thread CTAs on Turing—but still collects exactness, timing, and Nsight
evidence so a compiler-resource failure is itself diagnosable.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

CANDIDATE=streaming64 \
PROFILE_GPU=0 \
TIMING_ROUNDS=5 \
TIMING_REPEATS=10 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-f16-operands.sh
```

Return `sm75-indexer-streaming64-<timestamp>.tar.gz`. The candidate remains
diagnostic and cannot affect the production graph.

The accepted systemic SM75 direction changes the persistent ratio-4 indexer
cache itself to F16 and makes streaming64 the default scorer. Compressor and
FP4-QAT arithmetic remain F32; each completed K row is rounded once when it is
committed. This matches the F16 operands that shipping WMMA prefill previously
created inside every score tile, eliminates the repeated K conversion, and
reduces the 21-layer 256K indexer cache from about 672 MiB to 336 MiB. Session
payloads remain F32 so checkpoint files are backend-independent. The diagnostic
controls are independent: `DS4_CUDA_NO_INDEXER_NATIVE_F16_CACHE=1` selects the
legacy F32 cache and WMMA128 scorer, while `DS4_CUDA_NO_INDEXER_STREAMING64=1`
keeps the native F16 cache but selects the F16-input WMMA128 scorer. Neither is
an automatic runtime fallback on a qualifying SM75 graph.

`cuda-sm75-indexer-native-controls.sh` isolates cache representation from score
kernel structure in one interleaved three-arm test:

1. legacy F32 cache + WMMA128;
2. native F16 cache + F16-input WMMA128;
3. native F16 cache + streaming WMMA64.

It preserves complete balanced dense-FP16 admission while suppressing partner
projection execution and attention row splitting, requires an unambiguous
dispatch audit for each arm, and byte-compares every frontier-logit artifact.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=16384 \
CTX_ALLOC=262273 \
REPEATS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-native-controls.sh
```

Return `sm75-indexer-native-controls-<timestamp>.tar.gz`.

After the cache/scorer controls are separated, use
`cuda-sm75-xdev-feature-isolation.sh` to reintroduce the two high-volume
cross-device mechanisms independently. It keeps the full 344/344 dense cache,
stage-aware 22/21 placement, automatic T32/T256/shared-down partner classes,
native F16 index cache, and streaming64 scorer fixed, then runs `neither`,
`partner-only`, `rows-only`, and `both` in a fixed safety order. Every arm
records per-GPU telemetry and a durable active-arm journal before launch;
successful arms must expose their requested runtime paths. Each new arm also
exports its dense-FP16 placement plan. Row splitting allocates partner-side
state before the stage-aware dense planner runs, so separately launched arms
can select different local/partner bindings even when all 344 weights remain
cached. The runner therefore reports NRMSE, maximum delta, top-1 agreement,
top-10 overlap, planned-partner counts, and exact placement-plan equality in
`production/logit-comparisons.csv`; it does not mislabel a cache-confounded
cross-process comparison as a row-split exactness test. The partner toggle is
also informational because native-Q8 fallback and FP16-expanded cuBLAS are
intentionally different arithmetic paths.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=32769 \
REPEATS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-xdev-feature-isolation.sh
```

Return `sm75-xdev-feature-isolation-<timestamp>.tar.gz`. If the host resets
before the archive is created, preserve the matching unarchived directory;
`active-arm.txt`, `run-journal.tsv`, and `telemetry/` identify the failing arm.
After a script or validator failure, a new invocation can reuse complete arms
without modifying the original evidence:

```bash
REUSE_XDEV_DIR="$PWD/sm75-xdev-feature-isolation-<prior-timestamp>" \
SKIP_BUILD=1 \
bash ./speed-bench/cuda-sm75-xdev-feature-isolation.sh
```

The model, prompt, topology order, stage split, context sweep, and fixed
indexer controls must match the prior manifest. Partially present arms are
rejected rather than silently rerun or mixed with new evidence.

`cuda-sm75-indexer-native-cache-production-ab.sh` is the indexer promotion
gate. It
tests aligned and 63-row-tail streaming tiles, compares the native one-token
F16-cache reader with the legacy F32 reader, runs the fixed 100-case production
quality suite, interleaves genuine 32K production runs under the exact 32K
frontier allocation, byte-compares their frontier logits, and finishes with a
separate 64K-allocation long-prompt one-word decode smoke. Both arms must prove
344/344 dense-F16 admission, balanced 43/43 T256 placement, and automatic
selection of both qualified attention row splits. Because both arms open the
same immutable GGUF, this gate does not re-audit or score the model's routed
quantization recipe; that is a separate model-production concern.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=32768 \
CTX_MAX=32768 \
CTX_ALLOC=32769 \
WORD_CTX_ALLOC=65537 \
REPEATS=3 \
RUN_QUALITY=1 \
RUN_WORD_SMOKE=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-indexer-native-cache-production-ab.sh
```

Return `sm75-indexer-native-cache-production-<timestamp>.tar.gz`. A mismatch
in any quality score, score bit, top-k ordering, frontier logit, or retrieval
answer fails the run before promotion evidence is accepted.

The accepted 32K result is 612.43 tokens/s for native F16/streaming64 versus
582.90 tokens/s for legacy F32/WMMA128, a 1.049768x paired-median speedup.
Every frontier logit was bit-identical and the two 100-case quality TSVs were
byte-identical. Native F16/streaming64 is therefore the production default on
qualifying SM75 graphs; the legacy arm remains only a diagnostic control.

If a prior run completed the legacy 100-case arm before a later validation
failure, extract its archive and pass the extracted `quality/` directory as
`REUSE_LEGACY_QUALITY_DIR`. The reused log must still prove its production
runtime, full dense cache, and T256 placement before the two TSVs are compared.

If both 100-case arms completed, pass that `quality/` directory as
`REUSE_QUALITY_DIR`. The gate copies and revalidates both TSV/log pairs, then
starts at the production A/B. Do not use `REUSE_QUALITY_DIR` together with
`REUSE_LEGACY_QUALITY_DIR`.

The 32K performance allocation is deliberate. A 256K allocation on the
current 4 x 48 GiB layout materializes only 271/344 dense-F16 candidates, so
it is a capacity/admission experiment rather than a valid full-cache
performance control.

If a completed score pass stops during one of the top-k Nsight captures, keep
the original directory and resume only the three top-k captures. Resume mode
requires the validated timing CSVs and all four score-kernel Nsight CSVs; it
will not silently reuse a partial score pass.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export INDEXER_AUDIT_DIR="$PWD/sm75-indexer-audit-<timestamp>"

PROFILE_GPU=0 \
RESUME=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=1 \
bash ./speed-bench/cuda-sm75-indexer-audit.sh
```

### SM75 32K attention query-row split experiment

`cuda-sm75-attention-rowsplit.sh` is a model-free, two-GPU experiment for the
measured 32K production attention shapes. It compares the shipping 512-row
kernel with concurrent 256/256 query-row execution across one NVLink pair. The
partner variant is measured both by directly peer-reading home Q/KV/top-k and
by using locally mirrored inputs. Both variants must reproduce non-zero
shipping-kernel output bit-for-bit before timings are reported.

This is intentionally not the older head split and not a split-KV reduction.
Every query row retains the same kernel and online-softmax operation order; the
harness only changes which GPU executes the row. No GGUF is opened.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

PROFILE_GPU=0 \
PROFILE_PARTNER_GPU=1 \
REPEATS=5 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-attention-rowsplit.sh
```

Return `sm75-attention-rowsplit-<timestamp>.tar.gz`. A production integration
is justified only if one of the exact candidates wins in both indexed and
mixed attention; the harness result does not include output-projection work or
the incremental cost of maintaining a partner KV mirror.

The harness also validates the production compressed-cache lifecycle.
`DS4_ROWSPLIT_CACHE_REPEATS` controls a separate sustained loop of
home-default-stream raw/F32-compressed KV production, ordered peer mirroring,
partner attention consumption, and result gathering (default: `REPEATS`,
maximum 4096). Use roughly 750 repetitions per physical pair to match the
number of per-layer 512-token cache updates through a 16K full-model prefill;
the final split output must remain bit-exact. This attention cache is distinct
from the SM75-native F16 indexer cache.

`cuda-sm75-attention-rowsplit-production-ab.sh` is the fail-closed production
promotion gate. It interleaves the unchanged home path with the mirrored-KV
query-row candidate at 2K, 4K, 8K, 16K, and 32K frontiers, validates the
bounded indexed and mixed harnesses on both configured NVLink pairs, requires
audited production dispatches of both target kernels, and compares the emitted
frontier logits byte-for-byte for every paired repeat. A requested
indexed/nonzero-mixed split that cannot dispatch aborts; it is never replaced
by peer-read or home-only execution. Initial static-mixed/raw-only work remains
explicitly outside this first production candidate and is identical in both
arms.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_START=2048 \
CTX_MAX=32768 \
CTX_ALLOC=262273 \
REPEATS=3 \
RUN_HARNESS=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-attention-rowsplit-production-ab.sh
```

Return `sm75-attention-rowsplit-production-<timestamp>.tar.gz`.
After promotion, `cuda-sm75-long-prompt-word-smoke.sh` checks early user-visible
quality through the actual production CLI. It places a verification word before
a long public-domain distractor, requires a prompt of at least 24K tokens, and
greedily requests exactly that one word. No row-split enable override is set:
the log must prove that the qualified default dispatched both indexed and mixed
row-split attention on both stages without fallback. Cross-device synchronization
is not forced, so the smoke preserves normal production overlap.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CTX_ALLOC=65537 \
EXPECTED_WORD=LANTERN \
PAD_LINES=2200 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-long-prompt-word-smoke.sh
```

Return `sm75-long-prompt-word-smoke-<timestamp>.tar.gz`.

# SM75 Q4-32/Q3A4 32K production profile

`cuda-sm75-q32-production-profile.sh` captures the next evidence pass for the
tagged Q4-32/Q3A4 model. It accepts only a genuine single 32768-token
production frontier with the fixed 22/21 stage split, stage-aware dense-F16 placement,
all 344 dense-F16 candidates resident, and the exact routed recipe used by
`DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf`: Q3A4 gate/up on layers
6,8,10,12,14,16,18,20,30,32,34,36,38,40,42, Q4-32 gate/up on the other 28
layers, and Q4-32 down on all 43 layers.

The GGUF is opened once under Nsight Systems. DS4 NVTX ranges are then reduced
to per-device stage, microbatch, layer, handoff, and partner-projection tables.
The trace uses the exact 32K frontier allocation and must prove complete
dense-F16 admission, stage-aware placement inside both logical 22/21 NVLink
pairs, fixed 50/50 attention and score/top-k indexer row splits on both pairs,
the native-F16 streaming64 indexer, and a saved 32K
frontier-logit artifact. The stage-aware planner uses only logical stage roles,
live per-device headroom, and projected dense work; it contains no physical GPU
ID or layer-parity policy. It is rejected for every placement other than the
fixed four-GPU 22/21 production layout.
Nsight Compute does not replay the full application: bounded exact harnesses
capture Q4-32 gate/up, Q4-32 down, Q3A4 gate/up, and both indexed and mixed
long-context attention shapes. This avoids five additional 139 GB model loads
and prevents application replay from perturbing the production trace.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PROFILE_TOKENS=32768 \
CTX_ALLOC=32769 \
PROFILE_GPU=0 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q32-production-profile.sh
```

Return `sm75-q32-production-profile-<timestamp>.tar.gz`. The primary results
are `profile-summary.md`, `stage-device-summary.csv`,
`stage-microbatch-device.csv`, `kernel-family-summary.csv`,
`kernel-family-total.csv`, `operation-family-summary.csv`, the genuine
`nsys/combined.nsys-rep`, and the five validated reports under `ncu/`.

Critical-path attribution uses an indexed NVTX interval lookup; a genuine 32K
trace that previously required more than an hour of single-core Python scanning
processes in seconds. If collection was interrupted after `combined.sqlite`
was written, reuse it without loading or tracing the model again:

```bash
PROFILE_DIR="$PWD/sm75-q32-production-profile-<timestamp>" \
RESUME_POSTPROCESS=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q32-production-profile.sh
```

Keep the same `MODEL`, `PROMPT`, `GPU_DEVICES`, and fixed production variables
from the original invocation. Resume validates the saved production artifacts,
reuses the SQLite export and Nsight Systems reports, regenerates attribution,
then continues with the bounded Nsight Compute captures.

# SM75 Phase 2 consolidated production profile

`cuda-sm75-phase2-consolidated-profile.sh` is the closeout evidence pass before
Phase 3 work begins. It profiles the current accepted engine—not the rejected
Phase 2 experiments—using both `mixed15` and `all43` models at the genuine 32K
frontier. For each model it captures one full-prefill Nsight Systems trace and
one bounded 16-token steady-decode trace after 16 untraced decode tokens. The
optional model-free Nsight Compute pass covers the active decode weight-kernel
families without reopening either 127+ GB GGUF.

The topology is intentionally fixed to the current stable production contract:
22/21 layers, pair-0 attention row splitting off, pair-0 indexer row splitting
on, and both pair-1 row splits on. The current stage-aware T256 placement is
only valid for 22/21, so a 21/22 capture would measure a different placement
policy rather than the shipping engine. The resulting per-stage imbalance is
the evidence used to decide whether implementing and qualifying a 21/22 policy
is the first Phase 3 pipeline task.

The branch consolidates the accepted Phase 2 defaults: direct native packed
Q8_K production for decode and fused compressor projection/recurrent-state
append. The other three Phase 2 candidates remain rejected by their bounded
evidence: grouped decode attention was exact but 6.27x/8.65x slower for group
2/8, compact compressed-KV storage reduced bytes by 64.1% but made its consumer
6.85x slower, and the exact two-score-pass online-softmax candidates were
4.27x/13.55x/38.97x slower for H1/H4/H8.

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-phase2-consolidated-profile
git pull --ff-only

sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -pl 250
sudo nvidia-smi -i 1 -pl 260
sudo nvidia-smi -i 2 -pl 250
sudo nvidia-smi -i 3 -pl 250

unset CUDA_VISIBLE_DEVICES
unset PHASE2_PROFILE_DIR

MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-phase2-consolidated-profile.sh
```

If a late model-free Nsight Compute capture fails after all four production
traces completed, retain the output directory and resume it without reopening
either GGUF. Resume revalidates the saved topology, GPU identity/power, engine
logs, benchmark rows, reports, and SQLite exports before reuse. Each saved NCU
report is also revalidated against the current kernel symbol and launch shape;
an invalid or incomplete report is recaptured while valid reports are reused.

```bash
export PHASE2_PROFILE_DIR="$PWD/sm75-phase2-consolidated-profile-<timestamp>"

RESUME=1 \
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
RUN_NCU=1 NCU_USE_SUDO=1 SKIP_BUILD=1 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-phase2-consolidated-profile.sh
```

Return `sm75-phase2-consolidated-profile-<timestamp>.tar.gz`. The primary
decision artifacts are `summary/summary.md`, `summary/throughput.csv`,
`summary/prefill-stage-balance.csv`, `summary/prefill-family-summary.csv`,
`summary/decode-family-summary.csv`, and `summary/phase3-target-evidence.csv`.
The archive also retains all four Nsight Systems reports, their SQLite exports,
telemetry, exact/default dispatch validation, and bounded Nsight Compute data.

# SM75 routed-quant real-weight quality sweep

After cuda-sm75-q3-q4-32.sh establishes bounded kernel correctness and
performance, run the real-weight/imatrix gate before adding a GGUF type or
production dispatch. Set HF_DIR to the DeepSeek V4 Flash snapshot and IMATRIX
to the routed-MoE calibration file, then run
speed-bench/cuda-sm75-routed-quant-quality-sweep.sh.

`QUALITY_PRESET=screen` (the default) samples layers 3,21,36, experts
0,127,255, all three routed parts, and 32 evenly spaced rows per tensor.
`QUALITY_PRESET=full` covers every routed layer 3 through 42 with the same
experts and rows. Override any dimension with QUALITY_LAYERS,
QUALITY_EXPERTS, QUALITY_PARTS, or QUALITY_ROWS.

The sweep reports gate (w1), up (w3), combined gate/up, and down (w2)
separately under identical expert-specific imatrix weights. It includes the
shipping IQ2_XXS gate/up control, Q4_K and Q3_K controls, SM75 Q3/Q4-32,
IQ3-32, affine Q3-32 at 3.375/3.50 bpw, fixed-quota Q3/Q4 at
3.53125/3.78125 bpw, affine Q4-32 at 4.375/4.4375 bpw, Q5-32 at 5.25 bpw,
and adaptive Q2/Q3 at 2.78125/3.03125 bpw. It emits per-role Pareto tables
and a role-aware recipe frontier that can choose gate/up and down formats
independently. IQ2 down and Q2_K remain intentionally outside this pass.
The tool does not hash or quantize a full model and does not enable a
production format.

# SM75 indexed-attention production dispatch

SM75 indexed attention uses eight heads in a 256-thread CTA by default.
`DS4_CUDA_INDEXED_HEADS8_SM75=0` is retained only as the exact-output and
performance control for the former sixteen-head, 512-thread dispatch. Per-head
row order and online-softmax arithmetic are unchanged.

Two fixed 22/21 full-production A/B passes with complete 344/344 dense-F16
admission and byte-identical frontier logits independently measured indexed8
gains of +1.07%/+1.33% and +1.57%/+1.31% at 8K/32K. The SM75 production
profiler therefore expects the indexed-attention kernel to use 256 threads.

The Q4-32-down stage8 and compact-tile16 experiments are abandoned. Although
their isolated harnesses improved, they regressed the full production path by
roughly 2% and 5.6-5.9%, respectively. Neither implementation, dispatch flag,
nor experimental runner is retained.

# SM75 production decode baseline

`cuda-sm75-decode-baseline.sh` establishes the canonical decode baseline for
the fixed SM75 production configuration. The full matrix is pp512/tg256,
pp512/tg512, pp2048/tg256, pp2048/tg512, pp4096/tg256, pp4096/tg512,
pp32768/tg256, and pp32768/tg512. Every case runs in a separate process with
`ctx_alloc=pp+tg+1`; short-prompt results therefore do not sacrifice dense-F16
cache coverage merely to reserve the 32K case's context buffers.

An untimed 512-token prefill materializes all 344 dense-F16 candidates before
each measured case. The runner rejects incomplete admission, a stage split
other than 22/21, device order other than 0,3,1,2, a missing direct peer route,
a prompt other than the fixed `promessi_sposi.txt` corpus, a non-SM75 routed
layout, or any required-path fallback.
Inherited `DS4_*` variables are removed, so experimental decode paths cannot
silently contaminate the production baseline. No model hash, Nsight capture,
GPU sampler, or logits dump is performed.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CASES=all \
REPEATS=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-baseline.sh
```

Later passes can select a stable subset without redefining the matrix, for
example `CASES=pp512-tg256,pp2048-tg512,pp32768-tg512`. To resume an interrupted
run, keep its original settings and set `RESUME=1` plus
`DECODE_BASELINE_DIR="$PWD/sm75-decode-baseline-<timestamp>"`. Valid completed
cases are reused; incomplete or invalid cases are rerun. Return the generated
`sm75-decode-baseline-<timestamp>.tar.gz`. The primary results are
`summary/summary.csv`, `summary/summary.md`, and `summary/samples.csv`.

# SM75 Q3A4 tile32-DP4A production decode A/B

`cuda-sm75-decode-q3a4-production-ab.sh` validates the production-default
Q3A4-only native decode mapping selected by the kernel and end-to-end audits.
SM75 tagged Q3A4 tensors now use the exact tile32 signed-DP4A kernel by default;
`DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING=1` is the explicit rollback to the prior
control.  The runner compares that control with an unmodified default process
while leaving Q4-32, expert ownership, shared-expert placement, the 22/21 stage
split, dense-F16 placement, and every cross-GPU boundary unchanged.

Each process evaluates PP512, PP4096, and PP32768 with TG256.  The geometric
frontier sequence avoids two redundant 139-GiB model loads per arm.  Variant
order reverses on alternating repeats, and the report uses the median of
within-repeat speedup ratios.  A separate pass emits sixteen full-vocabulary
FP32 logit vectors at every frontier and requires byte identity.

The runner also enables a host-side dispatch audit.  A control sample is valid
only when every Q3A4 owned call uses mapping zero; a candidate sample is valid
only when every Q3A4 owned call uses `tile32-dp4a`.  Q32 CUDA Graphs, the split
projection experiment, and the generic fused-low-register experiment are
explicitly disabled in both arms, so they cannot confound the result.  Every
sample still requires complete 344/344 dense-F16 admission and the fixed direct
NVLink routes.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REPEATS=3 \
TG_TOKENS=256 \
EXACT_TOKENS=16 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-q3a4-production-ab.sh
```

To resume, retain all original settings and set `RESUME=1` plus
`Q3A4_PRODUCTION_AB_DIR` to the existing output directory. Throughput results
are summarized before exact validation starts. A fresh exact arm evaluates all
three frontiers in one process; after an interrupted run, completed exact
frontiers are preserved and only missing frontiers are repaired. Every source
log must still prove the same dense-Q8 plan, exact mapping-call inventory, and
final expected filename inventory. Return the generated
`sm75-decode-q3a4-production-ab-<timestamp>.tar.gz`; the production decision
table is `summary/summary.md` and the exactness gate is
`exact/verification.txt`.

# SM75 production decode evidence

`cuda-sm75-decode-evidence.sh` is the first attribution pass after the broad
decode baseline. It keeps the production model, device order, 22/21 split,
stage-aware fixed-22/21 dense-F16 placement, and complete 344/344 admission
fixed.

The throughput experiment isolates the implementation transition observed
between PP2048 and PP4096: PP4096 is run with the production 1024-compressed-row
indexing threshold and with a forced 4096-row dense threshold. Runs are paired
and order-reversed. These paths are not assumed to be semantically equivalent:
after the threshold, indexed attention changes the selected compressed-row
set. A separate semantic pass writes every full-vocabulary FP32 logit vector
for eight generated tokens and reports first divergence, top-1 agreement,
top-10 overlap, maximum absolute delta, and NRMSE. Its filesystem work is never
used as a throughput sample. Any proposal to change the production threshold
still requires a scored quality suite.

Nsight Systems captures begin after prefill and sixteen unprofiled decode
tokens. Only the next sixteen tokens are recorded. Opt-in NVTX ranges identify
the complete token, embedding, each layer and logical tier, attention and FFN
halves, and output head. They add no CUDA event, stream wait, or device
synchronization. The captured contexts are PP2048, PP4096, and PP32768, plus a
PP4096 forced-dense trace. Aggregate GPU work is reported separately from the
wall-clock GPU envelope because work on partner devices overlaps.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
AB_REPEATS=3 \
TRACE_CONTEXTS=2048,4096,32768 \
TRACE_SKIP=16 \
TRACE_TOKENS=16 \
RUN_AB=1 \
RUN_EXACT=1 \
RUN_NSYS=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-evidence.sh
```

To continue an interrupted run, keep every original setting and set
`RESUME=1` plus `DECODE_EVIDENCE_DIR` to its existing output directory.
Completed threshold samples are reused only after their CSV shape and every
production-path marker validate; missing or invalid samples are rerun.

Return `sm75-decode-evidence-<timestamp>.tar.gz`. The primary products are the
paired threshold result in `summary/summary.md`, semantic logit record under
`exact/`, the four bounded `.nsys-rep` files, and the operation, stage/device,
layer/device, kernel/device, and trace summaries under `summary/`.

# SM75 production decode indexer-score A/B

The production SM75 path partitions each native-F16 indexer score vector 50/50
across the layer's NVLink pair, gathers the partner score half, and runs the
unchanged deterministic top-k on the home GPU. It reuses the index-cache mirror
already required by prefill row splitting and allocates no additional
persistent cache. `DS4_CUDA_NO_TP_DECODE_INDEXER_ROWS=1` is retained only as a
diagnostic control; the supported production path enables the split by default.

`cuda-sm75-decode-indexer-production-ab.sh` compares that default with the
diagnostic control and rejects any output drift.

Every sample requires the fixed 22/21 placement, stage-aware complete 344/344
dense-F16 admission, tagged SM75 Q32 tensors, direct pair routes, and the
production prefill indexer row split. Runs are paired and order-reversed.
Full-vocabulary FP32 logits for eight decode steps must be byte-identical before
the report is accepted.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CONTEXTS=2048,32768 \
TG_TOKENS=256 \
REPEATS=3 \
EXACT_CONTEXT=32768 \
EXACT_TOKENS=8 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-indexer-production-ab.sh
```

To resume, preserve the original settings and set `RESUME=1` plus
`DECODE_INDEXER_AB_DIR="$PWD/sm75-decode-indexer-production-<timestamp>"`.
Return the generated archive; the decision table is `summary/summary.md` and
the exact-output gate is `exact/verification.txt`.

# SM75 production Q32 decode CUDA Graph A/B

`cuda-sm75-decode-crash-isolation.sh` reproduces the graph-disabled 32K failure
with physical GPU0 in a home role. The established placement evidence is not
retested here: physical GPU0 succeeds as a partner in `1,3,0,2` and
`1,2,0,3`, while it fails as a home in both the original `0,3,1,2` placement
and the stage-swapped `2,0,3,1` placement. The default therefore runs only the
known TG16 reproducer. The production row split, stage-aware 22/21 dense
placement, partner execution, native-F16 indexer, and complete 344/344 dense
cache remain enabled; owned routed-MoE CUDA Graphs are explicitly disabled.

Every case persists `active-case.txt` and `run-journal.tsv` before launch,
captures per-GPU telemetry, exports the exact dense placement and binding
inventories, and requires healthy four-GPU snapshots before and after the
process. A separately fsynced progress journal records each completed warmup
and measured-prefill chunk. `DS4_CUDA_PREFILL_FAULT_BREADCRUMBS=1` also records
each queued stage and each begin/completion/failure of the existing cross-stage
host-bounce handoff. Those handoffs already synchronize the destination stream,
so a `handoff-complete` breadcrumb is completed-GPU evidence without adding a
new CUDA synchronization point. The final 160 breadcrumb records are extracted
to `fault-breadcrumbs.log`. The telemetry watcher stops `nvidia-smi -lms` when
it reports a lost device, preventing that poller from lingering after the CUDA
process exits. The runner never invokes `sudo`.
If the host resets before the archive can be created, return the matching
unarchived directory.

This runner performs the complete 32K prefill before it reaches decode. A
failure with no benchmark CSV row and no decode marker is therefore prefill or
device-stability evidence, not decode-graph evidence. Its purpose is to identify
the precise chunk, wave, microbatch, and handoff at which the home-role failure
becomes visible. It is not another board, slot, pair-direction, or generic-load
qualification pass.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
PP_TOKENS=32768 \
TG_LEVELS=16 \
REPEATS=1 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-crash-isolation.sh
```

`cuda-sm75-decode-graph-ab.sh` tests graph coverage for the routed kernels
that the production Q4-32/Q3A4 model actually executes. The candidate replaces
each owned expert half's six host submissions—Q8_K quantize, native activation
pack, fused Q4-32/Q3A4 gate/up, mid quantize, native activation pack, and
Q4-32 down—with one CUDA Graph launch. It deliberately leaves the cross-GPU
owned-slot combine outside the graph, preserving the established NVLink and
synchronization boundary.

Graph executables are cached per exact layer/tier binding after the untimed
warmup. Separate entries cover Q4-32 versus Q3A4, fused shared-Q8
prequantization versus ordinary Q8_K quantization, and six-slot home versus
fixed-three partner down output. The timed path therefore does not perform six
node-parameter updates per layer. If the bounded cache cannot represent a
binding, that call uses the unchanged launch path.

The CUDA smoke test compares the graph-owned home/partner result against the
independent full-expert production dispatcher for both gate formats and two
successive inputs. The production test then runs paired, order-reversed decode
A/B samples through PP32768 and requires byte-identical full-vocabulary logits
before reporting a speedup. The 32K path is enabled after the current
graph-disabled production configuration completed three PP32768/TG256 samples
with healthy pre/post GPU state.

```bash
cd ~/ds4-iq2-q4
git pull --ff-only

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
CONTEXTS=32768 \
TG_TOKENS=256 \
REPEATS=3 \
EXACT_CONTEXT=32768 \
EXACT_TOKENS=16 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-decode-graph-ab.sh
```

To resume, preserve the original settings and set `RESUME=1` plus
`DECODE_GRAPH_AB_DIR="$PWD/sm75-decode-graph-ab-<timestamp>"`. Return the
generated archive. The decision table is `summary/summary.md`; exactness is
recorded in `exact/verification.txt`.
# SM75 decode packed-weight profiling

`cuda-sm75-decode-weight-profile.sh` profiles the actual one-token decode
dispatch for routed Q4-32/Q3A4 gate/up, both Q4-32 down ownership modes, the
direct native-Q8 producer, the distinct packed-Q8 projection
shapes seen in the 32K production trace, and all observed fused F16
compressor-pair/state widths. It opens no GGUF and validates exact-zero outputs from
synthetic zero weights before collecting one precisely filtered kernel per
scenario.  The default `NCU_CACHE_CONTROL=all` represents cold per-layer
weight streams; use `none` only for an explicitly replay-warm L2 comparison.

```bash
PROFILE_GPU=0 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-decode-weight-profile.sh
```

The resulting `summary.csv` and `summary.md` report achieved DRAM bandwidth,
DRAM peak utilization, L2 hit/throughput, global-load efficiency,
sectors/request, long-scoreboard and MIO stalls, and achieved occupancy.  A
subset can be captured with `PROFILE_SET=routed`, `q8`, or `f16`, or with an
explicit comma-separated `SCENARIOS` list.

### Production Q4-32 decode factorial A/B

`cuda-sm75-decode-q4-production-ab.sh` measures the Q4-32 gate/up
tile32 packed-INT4 MMA and Q4-32 down tile32 paths independently and together.
It fixes the production 22/21 placement, mixed15 Q3A4 K4 path, dense-Q8 plan,
indexer, attention, and cross-GPU boundaries. Four-repeat blocks use a balanced
schedule. Every arm must reproduce 16 byte-exact decode logits at PP512,
PP4096, and PP32768 and its exact gate/down ownership-shape call inventory.

The first complete paired block, run with all four GPUs capped at 200 W,
measured gate/up MMA at +8.2% to +9.8%, down tile32 at +1.3% to +1.5%, and
both together at +9.7% to +11.7% across PP512/4096/32768. Their saved
milliseconds per token were almost exactly additive. Together with the prior
byte-exact nonzero regressions, sanitizer passes, zero-spill resource gates,
and bounded kernel wins, this promoted both paths. The production A/B still
forces both selectors in every arm so the `control`, `gate-mma`,
`down-tile32`, and `both` definitions remain stable after the defaults change.

The next Q4 work is deliberately format- and ownership-specific:

1. For gate/up, test exact in-CTA K2 and K4 cooperation on the production
   tile32 packed-INT4 MMA kernel. Preserve the 16 K256 leaves and current
   reduction order, reject spills/local traffic, and include the whole owned
   call in timing. The present kernel already allocates 96 registers and
   reaches about 75% of peak DRAM bandwidth, so software prefetch should be
   attempted only after the K-split/live-state result is known.
2. For down, test K2 and K4 independently for `owned_slots` and
   `owned_packed`; do not infer one ownership mode from the other. Both tile32
   kernels allocate 64 registers, while the profiles remain dominated by
   long-scoreboard stalls. Apply depth-1/2 prefetch only to a winning K-split.
3. Re-run the accepted defaults on an all-Q4 gate/up model under a stable
   power/clock condition. The current production result contains 28 Q4
   gate/up layers, so it establishes the dispatch but not the eventual
   all-Q4 aggregate gain.

The rejected separate gate/up projection launches, generic Q32 low-register
variants, and earlier Q4-down stage8/compact variants are not next steps.

```bash
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REPEATS=4 TG_TOKENS=256 EXACT_TOKENS=16 SKIP_BUILD=0 \
MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
bash ./speed-bench/cuda-sm75-decode-q4-production-ab.sh
```

Resume accepts only whole, healthy exact variants and rejects changed source,
binaries, model inode/size/mtime, placement plans, or partial GPU-loss output.

### Symmetric SM75 pair-stability matrix

`cuda-sm75-pair-stability-matrix.sh` compares both physical NVLink pairs with
the same model-independent workload. It runs the known-reference pair first,
then the NUMA0 pair, for four escalating cases: high VRAM residency, concurrent
Tensor-Core GEMM, bidirectional P2P, and combined compute plus P2P. The harness
uses exactly two visible GPUs, records physical PCI bus IDs, and fsyncs progress
about once per second. It performs no `nvidia-smi` polling while a load is
active and stops at the first failure.

This is a hardware/driver-path diagnostic, not a production placement policy:
both pairs receive identical work and no conclusion assumes heterogeneous GPU
performance.

```bash
cd ~/ds4-iq2-q4

PAIRS="2,3 0,1" \
SCENARIOS="residency compute p2p combined" \
DURATION_SECONDS=60 \
RESIDENT_MIB=43008 \
COPY_MIB=128 \
GEMM_N=4096 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-pair-stability-matrix.sh
```

The directory remains usable after a reboot. A `.started` or `.failed` marker
plus the last durable progress row identifies the exact scenario and pair that
were active when an endpoint disappeared.

### Small-BAR1 production-pair isolation

`cuda-sm75-small-bar1-pair-isolation.sh` keeps the production device order,
22/21 stage split, Q3A4 K4, Q4 tile32, and the healthy logical pair unchanged.
Every arm keeps the complete 344/344 dense cache, partner-resident FP16 weights,
partner cuBLAS projections, activation/result shapes, and arithmetic. It varies
only logical pair 0 (CUDA ordinals 0<->1 under `GPU_DEVICES=0,3,1,2`). The
completed localization control is:

Every invocation must export `CUDA_DEVICE_ORDER=PCI_BUS_ID` and leave
`CUDA_VISIBLE_DEVICES` unset. Before CUDA smoke tests or an arm, the harness
records every CUDA ordinal's PCI domain/bus/device/function and UUID, captures
the corresponding `nvidia-smi` inventory and topology matrix, and validates
the complete identity set. CUDA ordinals are joined to `nvidia-smi`/NVML
indices by the normalized `(PCI bus ID, UUID)` tuple; ordinal equality is never
assumed. The selected tier, pair, role, ordinal, bus ID, and UUID must exactly
match `sm75-small-bar1-expected-device-identity.csv`, and the derived NVML
indices must describe exactly the two NVLink-connected pairs in the full
topology matrix, with no cross-pair NVLink edge. Each arm then logs the complete
CUDA ordinal inventory plus every selected logical tier from inside DS4. The
harness preserves those lines as per-arm CSV evidence and validates them
against preflight before accepting either a pass or crash classification;
missing lines, failed CUDA queries, and identity changes fail closed. Resume
captures timestamped raw snapshots without overwriting prior evidence and
compares normalized full CUDA, `nvidia-smi`, topology, and selected mappings.
These records prove which boards and links an arm actually exercised; they do
not re-test BAR1 size and are not a fault mitigation.

All commands in this section therefore assume:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
```

- `attention-off`, which retains normal direct partner projections, all mirrored
  attention-cache updates, and pair-0 decode-indexer splitting but runs pair-0
  prefill attention on the home GPU. Pair-0 prefill indexer score/top-k also
  stays home because its split selected-row ownership exists only to feed split
  attention; pair 1 retains both production 50/50 prefill splits. This arm
  completed PP32768/TG256 at 250 W with healthy post-run GPUs, localizing the
  reproduced failure to the pair-0 prefill attention-row path or its
  interaction. It is a diagnostic control, not a proposed production fallback.

The next row-split-on diagnostic is:

- `attention-phase-audit`, which leaves the full production path enabled and
  adds device-completion checks only to one selected pair/layer/frontier
  dispatch (default pair 0, mixed layer 17, position 512). Durable events name
  `query-copy`, `partner-attention`, `home-attention`, and `result-gather`.
  Every production row-split operation and byte remains, and only that one
  known failure-coordinate dispatch is fenced. The fence deliberately removes
  its normal partner/home overlap and can drain other device work, so a passing
  arm implicates the overlap, instantaneous-load, or timing envelope rather
  than proving a specific asynchronous-ordering defect. It is still materially
  different from a low-load approximation: the surrounding 32K workload and
  direct-peer traffic remain enabled.

That audit reproduced the device loss, but all four layer-17/position-512
phase checkpoints completed. Layers 18 through 21 were then submitted and the
first CUDA error was observed by the following tier-0-to-tier-1 stage handoff.
The submission records do not prove those later asynchronous row-split calls
completed, and a poisoned multi-device CUDA context makes the handoff only the
observation point rather than the demonstrated cause.

The tighter row-split-on boundary is:

- `attention-end-fence`, which leaves query copy, partner attention, home
  attention, and result gather fully overlapped exactly as production through
  the final pair-0 attention layer (default layer 21, position 512). Only after
  the result gather has been submitted does it synchronize partner and then
  home once, with durable `begin`, per-device `complete`/failure, and final
  pair records. If partner synchronization fails first, the audit only attempts
  to restore the current device to home; it does not issue a causally secondary
  home synchronization into an already-poisoned context. A fence failure proves
  an error was pending before downstream
  layer-tail/stage-handoff work. A complete fence followed by a handoff failure
  moves the first observed boundary past row-split completion. A wholly clean
  run means that this single boundary removed a required overlap or
  instantaneous-load condition; it is diagnostic evidence, not a production
  mitigation.

The separate end-fence runs initially appeared to place a cross-run observation
bracket between layers 17 and 18: one layer-17 fence completed while later
layer-18 through layer-21 fences first observed a failed partner context. The
combined audit then reproduced the same endpoint loss at the very first
layer-17 partner synchronization, before any layer-18 entry or phase marker.
The failure remains reproducible, but the layer at which a newly inserted
synchronization observes it is not stable. These fences perturb the overlap and
instantaneous-load envelope, so further layer-number fence stepping is not used
to assign a causal kernel.

The completed combined diagnostic was:

- `attention-row-boundary-audit`, which keeps the full production workload
  and adds three ordered observations at the fixed mixed-attention
  layer-17/layer-18, position-512 boundary. It first
  applies the end-only pair fence after layer 17 attention, then synchronizes
  partner and home at the layer-18 row-launch boundary immediately before query
  copy, and finally enables the existing layer-18 `query-copy`,
  `partner-attention`, `home-attention`, and
  `result-gather` phase checkpoints. The entry fence emits durable
  `begin`, per-device `complete`/failure, and final pair markers under
  `prefill attention row entry fence`. This is not transformer-layer entry: the
  upstream layer-18 QKV/cache/indexer preparation has already run. If that entry
  fence fails, layer-18 query copy has not yet been submitted, so the remaining
  interval is layer-17 tail/MoE/Q8 through that upstream layer-18 preparation.
  If entry completes, the phase markers can distinguish the layer-18 row-split
  operations. In the captured combined run, the layer-17 partner end fence was
  already the first failing boundary and layer 18 never ran. That result only
  proves an asynchronous error was pending on the partner by the first forced
  synchronization; it does not identify layer 17 as the trigger.

The next workload-preserving diagnostic changes the CUDA context/default stream
that submits each peer copy and the surrounding event dependencies rather than
adding another host completion fence:

- `attention-query-dst` submits only the pair-0 query-row handoff in the
  destination device context on its default stream instead of in the source
  device context/default stream;
- `attention-gather-dst` makes the same scheduling change only for the pair-0
  partner-result gather;
- `attention-both-dst` moves both; and
- `production` retains source-context/default-stream submission for both copies.

The destination-scheduled primitive is an ordered, fail-closed direct-P2P handoff:
source readiness, destination copy, then a destination-complete event that the
source waits on before scratch reuse. It never silently falls back to host
bounce or source-stream copying. Each arm retains the same 32K/256-token
production workload, 50/50 attention rows, partner attention, direct NVLink/P2P
transport, transfer directions, and exact query/result byte counts. Destination
submission requires the opposite validated peer-access/mapping direction from
source submission. The factorial therefore changes a bundle: initiating CUDA
context/default stream, event ordering, and peer-access/mapping direction. CUDA
and the driver choose the physical transfer engine and low-level route; this
audit neither controls nor identifies a physical copy engine. The harness
requires at least 500 prefill tok/s and verified pair-0/pair-1 row-split copy
schedules before accepting either a completed candidate or the production
control, preventing a low-utilization or degraded control from being
misclassified as a safe pass.
With `SKIP_BUILD=0`, a focused two-GPU preflight also checks both transfer
directions for byte-exact destination data, immediate source-buffer reuse
ordering, and fail-closed behavior when host bounce is forced.

The copy-scheduling matrix subsequently reproduced device loss across the
source/destination submission arrangements. The next categorical transport arm
is `attention-host-bounce`. It retains pair-0 50/50 row ownership, partner
attention arithmetic, both indexer row splits, production Q8 partner transport,
and the full PP32768/TG256 workload, but stages every pair-0 attention-owned
copy through pinned host memory: query rows, mirrored raw/compressed/index cache
updates, gathered attention results, and any non-split top-k state. In this
production arm, top-k remains partner-local because prefill indexer row
splitting is retained; pair 1 remains direct peer. A completed arm is accepted
only when its per-dispatch
transport markers prove pair 0 used host bounce and pair 1 used direct peer,
including separate completed pair-0 raw, attention-compressed, and index-cache
mirror markers,
plus a durable completed measured-row checkpoint at `pos=512`, both schedules
remained source/source, all four GPUs stayed at 250 W and healthy, and prefill
remained at least 500 tok/s. A pass would implicate direct attention P2P/BAR1
traffic or its timing envelope without making host staging a production
mitigation. A device-loss failure rules out pair-0 attention-owned direct P2P as
necessary for an observed pair-0 loss only if the durable `pos=512` host-bounce
checkpoint proves the measured transport cut was reached and the watcher
identifies physical GPU 0 or 1, the pair under test. It does not rule out the
untouched pair-1 direct traffic as a systemic trigger. An earlier loss or a loss
confined to pair 1 is real but causally inconclusive for pair-0 attention
transport.

Run this as a fresh one-arm diagnostic, never by changing the variant list of
an existing resume directory:

```bash
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-host-bounce-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-host-bounce \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The pair-0 attention-only host-bounce arm still lost the tested pair after a
completed measured `pos=512` attention checkpoint. Direct pair-0 attention
P2P/BAR1 transport is therefore not necessary for that observed failure. Q8
partner activation/result transport on pair 0 was still direct in that arm,
while pair 1 also retained all direct traffic.

The next fresh one-arm diagnostic is `attention-q8-host-bounce`. It applies both
`DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=0` and
`DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=0`. Pair-0 attention query, cache,
and result copies and pair-0 Q8 partner activation/result copies are staged
through pinned host memory. It does **not** disable partner attention, partner
Q8 projection work, dense-cache admission, either 50/50 attention/indexer row
split, or any arithmetic. Pair-0 prefill and decode indexer row-split copies
remain direct peer by design; pair 1 remains direct peer for attention and Q8.
This is a diagnostic transport cut, not a proposed production mitigation.

A successful arm is accepted only if the log proves all of the following:

- pair-0 Q8 used host bounce in the untimed warmup (`calls=1`) and reached its
  measured-prefill checkpoint (`calls=128`), while pair 1 used direct peer in
  warmup and reached `calls=64`;
- pair-0 attention completed a measured host-bounce row at `pos=512`, while
  one-time route markers independently prove that raw, attention-compressed,
  and index-cache mirror copies selected host bounce; pair 1 remained direct
  peer;
- both attention and indexer row splits, all 344 dense-cache bindings, and
  partner execution remained present; and
- all four GPUs remained healthy at exactly 250 W and measured prefill was at
  least 500 tok/s.

A completed run below 500 prefill tok/s is recorded as
`inconclusive-underloaded`, never as a passing transport result. A pair-0 loss
after both measured checkpoints shows that pair-0 direct attention-owned and
Q8-partner payload transport is unnecessary for the observed failure, but does
not exclude the retained direct pair-0 indexer transfers, pair-1 direct traffic,
partner compute, aggregate load, or their interactions. A healthy >=500 tok/s
completion implicates one of the two cut pair-0 direct-peer payload routes or
its timing interaction, without identifying one individual copy family. A loss
before either measured checkpoint is real but inconclusive for this transport
cut.

Failure-only diagnostics record the requesting transfer class. When CUDA's
generic synchronous bounce failure line is immediately adjacent, the summary
also associates its exact phase (`d2h` or `h2d`) on a best-effort basis.
Attention records include query, result-gather, top-k, or cache class plus
layer/position, logical tiers, physical device IDs, and bytes; Q8 records
distinguish activation from result gather with the same endpoints and tensor
shape. These identify the first synchronous observation boundary, not
necessarily the earlier operation that poisoned the CUDA context.

Run it from a clean boot with a new directory; never resume an older variant
list under this name:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-host-bounce-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-host-bounce \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The follow-up one-arm diagnostic `attention-q8-phase-audit` retains the exact
same pair-0 attention/Q8 host-bounce transport cut and production workload,
then enables `DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=0`. The added audit is
pair-scoped: it records pair-0 Q8 partner phase checkpoints and explicitly
synchronizes the partner device after every audited Q8 compute submission,
before the result D2H copy. It preserves arithmetic, transferred bytes,
placement, admitted weights, and attention execution, but intentionally
perturbs Q8 timing and overlap to localize the failure phase. Pair 1 stays on
its production direct-peer paths. Pair-0 prefill and decode indexer row-split
transfers also remain direct peer; this arm does not host-bounce or disable
them. A successfully completed arm must contain one same-sequence pair-0 marker
chain from activation preparation through activation copy, a pre-compute
partner synchronization, compute submission, post-compute synchronization,
and result gather, with the same non-placeholder binding label, passed label,
and weight offset throughout. The pre-compute boundary separates earlier
queued partner work from the audited projection. Pair-1 phase markers or any
phase-failure marker fail validation.

Run it from a clean boot and a fresh output directory with all four GPUs fixed
at 250 W:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-phase-audit-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-phase-audit \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The production-load follow-up is the one-arm
`attention-q8-targeted-phase-audit`. It retains the same pair-0 attention and
Q8 host-bounce transport, the 50/50 attention row split, all partner compute,
and pair-1 direct-peer paths, but restricts Q8 phase checkpoints and the
explicit partner-device compute synchronization to the exact binding
reconstructed by joining the earlier failure context with the later broad
phase audit:

- binding label: `tensor:blk.14.attn_output_b.weight`;
- weight offset: `143571266304`;
- passed label: `attn_output_b`;
- shape at the observed failure: 512 tokens, 8192 inputs, 4096 outputs, and an
  8 MiB result gather.

Pair-0 prefill and decode indexer transfers remain direct peer. The target
filters reduce the audit from 4,290 pair-0 Q8 calls in the broad completed arm
to exactly 65 layer-14 attention-output calls (one warmup plus 64 measured
microbatches), preserving the failing workload far more closely while retaining
durable before/after markers at the relevant pre-compute, projection-compute,
and result-copy boundaries. Every one of those 65 chains must complete in strict
order; a missing or incomplete chain fails validation. A completed arm below
500 prefill tok/s is still classified as inconclusive-underloaded.

Run it from a clean boot and a fresh output directory with all four GPUs fixed
at 250 W:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-targeted-phase-audit-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-targeted-phase-audit \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The targeted layer-14 archive completed both of its strict seven-event chains
(warmup and the first measured microbatch). The immediately following
uninstrumented layer-15 `attn_output_b` result D2H then surfaced
`unspecified launch failure`, and both pair-0 recovery synchronizations failed.
That evidence rules out a synchronous failure inside either audited layer-14
chain and argues against layer 14 as a unique failing binding, while leaving a
delayed physical effect possible. It cannot tell whether layer-15 compute failed
asynchronously or its result-copy call was the first failing operation because
layer 15 had no pre/post-compute fences.

The next one-arm diagnostic is `attention-q8-l14-l15-phase-audit`. It preserves
the layer-14 checkpoints so the known clean boundary is not removed, and adds
the same checkpoints to the adjacent exact tuple
`tensor:blk.15.attn_output_b.weight@143723876608`. The selector consumes paired
`binding@offset` identities, not independent label and offset lists. A completed
arm must contain 65 complete chains for each binding (130 chains and 910 marker
lines total), with no layer-15 start outrunning the corresponding count of
layer-14 starts. At initialization the backend strictly validates the target
tuple list, audited logical-pair list, and exact expected Q8 shape. It checks
each label/absolute-offset pair and its home/partner direction against the
active model's Q8 plan. Materialization remains at the ordinary production
first-use boundary; before that first lookup can return, the backend requires
exactly one executable partner binding of the expected shape and physical pair
for each tuple. A stale or ambiguous hard-coded offset, malformed pair list, or
wrong binding therefore terminates the process instead of silently disabling
its markers, without moving cache planning ahead of session allocation.
Pair-0 attention/Q8 host-bounce, 50/50 row splitting, partner
compute, pair-1 direct transport, the 250 W power requirement, and the 500
prefill tok/s success floor remain unchanged. A passing arm means only that the
workload survived after perturbing both adjacent overlap windows; it is not a
production mitigation or root-cause proof.

Run it from a clean boot and a fresh output directory:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-l14-l15-phase-audit-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-l14-l15-phase-audit \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The resulting
`sm75-attention-q8-l14-l15-phase-audit-20260831T163552Z` archive does not show
the measured failure moving beyond layers 14 and 15. Both complete target
chains occurred inside the untimed 512-token warmup. After the warmup completed,
the first measured 2,048-token chunk began without reaching either measured
target marker; the pair-0 Q8 host-bounce result D2H then surfaced
`unspecified launch failure`, and the engine reported layer 12 attention encode
failure. Joining that failure with the materialized binding inventory strongly
correlates it with the exact tuple
`tensor:blk.12.attn_output_b.weight@143236281600` (512 tokens, 8192 inputs,
4096 outputs, 35,651,584 weight bytes, and 8 MiB activation/result transfers).
Warmup-only layer-14/layer-15 completions therefore must not be used as evidence
that the first measured failure occurred downstream of that window.

The code audit also compared this path with the earlier `b9b0c7e` GPU-loss fix.
That defect was an unconditional second-word load at the exact end of a packed
Q8 allocation. The present partner-resident F16/cuBLAS projection does not call
that loader, and its 2,048-token partner scratch reservation is at least 64 MiB;
the failing 512-token 8192-to-4096 call occupies only the first 16 MiB for its
activation and result. No analogous allocation-tail geometry or runtime scratch
size mismatch was found. The host-bounce D2H is synchronous, but it can still be
the first API to report a deferred partner compute/default-stream error or an
endpoint loss that already occurred. The surfaced D2H error therefore cannot be
assigned causality by itself. The mechanism-level diagnostic below observes
every pair-0 Q8 post-compute boundary
rather than selecting another presumed layer.

The subsequent `attention-q8-l12-phase-audit` result invalidated that targeting
strategy. Its measured workload failed before occurrence 2 of the L12 tuple was
selected. The first API-level Q8 failure instead surfaced while processing the
layer-0 `attn_output_b` partner projection. That does **not** prove layer 0 or
that projection caused the endpoint loss; CUDA can report an earlier
asynchronous failure at a later synchronous API. It does prove that selecting
another presumed layer from the final engine error cannot answer the fault
question. Do not continue walking the layer number.

The replacement mechanism diagnostic is `attention-q8-async-completion`. It has
no layer, binding, occurrence, or position selector. It retains the full
four-GPU 32K/TG256 production workload, 50/50 attention row split, partner
execution, pair-0 attention/Q8 host-bounce transport cut, pair-1 direct
transport, and exact 250 W requirement. Every admitted pair-0 partner-Q8 call
gets a monotonically increasing sequence. Immediately after that call's Q8
compute submission, on the same partner default stream, a one-thread kernel
writes the sequence and its complement to mapped host memory and a dedicated
event is recorded. No synchronization is added at that boundary. A
**successful** existing synchronous host-bounce result gather is the visibility
checkpoint.

The audit records `begun`, `submitted`, and `confirmed` separately. A completed
dedicated event is CUDA-synchronized evidence that the partner default stream
crossed that call's post-compute marker. If that event completes but the mapped
marker is invalid, the arm is classified as a marker-channel integrity failure
rather than a clean result. If the unhealthy device/context makes the event
unavailable but the mapped slot contains the exact current sequence and
complement, that is retained separately as a positive hardware breadcrumb—not
formal CUDA same-stream completion proof. An absent or stale marker in that
case is explicitly inconclusive; it must not be interpreted as proof that
compute failed. A clean completion is accepted only when all three counts
match, the final complement is valid, and that pair's partner context
synchronizes during teardown.

This arm deliberately adds one tiny kernel, one mapped-host write, one event,
and three host counter updates per pair-0 partner-Q8 call. After a successful
gather it also reads the sequence/complement pair, and call 1 plus every 64th
confirmed call emits an extra marker `fprintf`/`fflush`. Those are timing,
host-scheduling, and PCIe perturbations. The durable `fsync` at those sparse
checkpoints is issued by the existing Q8 transfer audit in both this arm and
its no-marker control; it is not unique to the marker arm. A completed arm
must still exceed the 500 prefill tok/s gate. A crashing arm has no valid
throughput result, so identical launch settings and telemetry establish the
intended load configuration but do not by themselves prove an achieved rate.

That comparison has now been run. The async-completion arm and its otherwise
identical no-marker host-bounce control both lost the pair-0 devices. In the
marker arm the final current call had `begun=53`, `submitted=53`, and
`confirmed=52`; its dedicated event query reported `cudaErrorLaunchFailure`
only after the synchronous result D2H had failed. In the no-marker control the
512-token warmup completed and the first measured-prefill pair-0
`attn_output_b` result D2H again surfaced the failure. This rules out the
post-compute marker as a necessary trigger, but it still does **not** identify
the result D2H as the cause: that synchronous copy can be the first API to
report an asynchronous partner-stream or endpoint failure.

The next mechanism-level bracket is `attention-q8-pre-gather-fence`. It uses
the same GPU compute and copy configuration as
`attention-q8-async-completion`, but immediately before attempting each
synchronous host-bounce result D2H it synchronizes the already recorded
post-marker event and validates the exact mapped-host sequence and complement.
It also adds the paired per-call host-log records described below. The fence
does not select a layer or binding. It records a sparse
success checkpoint on the first confirmed call and every 64th confirmed call;
every failure record is flushed and synchronized durably. In addition, every
call whose event and marker pass emits this compact fflush-only breadcrumb
immediately before result gathering:

`ds4: CUDA q8 partner pre-gather armed current_sequence=N marker_sequence=N marker_complement=X home_tier=H home_device=HD partner_tier=P partner_device=PD`

Immediately after the synchronous helper returns success, it emits the paired
compact line:

`ds4: CUDA q8 partner pre-gather returned current_sequence=N result_gather_status=success home_tier=H home_device=HD partner_tier=P partner_device=PD`

Neither breadcrumb is a full structured checkpoint. Together they add two
fence-arm-only `fprintf`/`fflush` records per successful call, but no per-call
`fsync`. Completed-run validation requires strictly alternating, contiguous
`armed N`/`returned N` pairs. The paired breadcrumbs exist because the
telemetry watcher can terminate the process after device loss before either a
structured CUDA failure record or teardown summary is written. In that case,
`armed N` without `returned N` is classified only as
`result-gather-return-not-observed`. It is not proof that the helper failed or
never returned: SIGKILL can land after helper success but before the returned
line is printed. Conversely, `returned N` proves that result gather succeeded
for that sequence; the locus of any later failure remains unresolved.

Structured failure records state separately whether the result D2H and H2D
APIs were attempted and whether each completed. This distinguishes failure
during destination-device switch/setup after a completed D2H from failure
reported by an H2D API that was actually entered. Together, the structured
records and paired breadcrumbs support exactly these outcomes:

- event synchronization reports failure with `result_d2h_attempted=no` and
  `result_d2h_completed=no`: the failure surfaced at the partner default-stream
  boundary before result transport was attempted;
- the event completes but the marker is invalid: the marker channel itself is
  inconsistent, and result D2H is neither attempted nor completed;
- the event and marker both pass, but result gathering fails with
  `result_d2h_attempted=no` and `result_d2h_completed=no`: partner compute was
  positively confirmed and the failure surfaced inside result-gather setup
  before the D2H API;
- the event and marker both pass, followed by `result_d2h_attempted=yes`,
  `result_d2h_completed=no`, and a result-gather failure: the D2H API was
  entered but did not complete successfully; H2D remains unattempted and
  incomplete;
- the event and marker both pass, followed by `result_d2h_attempted=yes`,
  `result_d2h_completed=yes`, `result_h2d_attempted=no`, and
  `result_h2d_completed=no`: D2H completed, then destination-device
  switch/setup failed before the H2D API was entered;
- the event and marker both pass, followed by `result_d2h_attempted=yes`,
  `result_d2h_completed=yes`, `result_h2d_attempted=yes`, and
  `result_h2d_completed=no`: D2H completed and the H2D API was entered but did
  not complete successfully;
- `armed N` lacks a matching `returned N`: result-gather return was not
  observed before termination, but helper failure versus post-success SIGKILL
  remains unresolved;
- `armed N` has a matching `returned N`: that complete result gather succeeded,
  so any failure first observed later belongs to an unresolved downstream
  interval; or
- the arm completes: the fence changed the failure behavior and is a timing/
  ordering perturbation, not a production fix.

That bracket has now produced the evidence needed from its first arm. The
pre-gather-fence run recorded 32 completed pair-0 result-gather returns. On call
33, the first reported error moved to the activation H2D for the next partner
projection. This proves that the previous result-gather return completed; it
does **not** prove that activation H2D caused the endpoint loss. The CUDA error
may still have originated in earlier asynchronous device work.

There is no reason to resume that directory or rerun its no-fence control. The
next diagnostic is one fresh `attention-q8-activation-fence` arm. It preserves
the just-tested pair-0 attention/Q8 host-bounce transport, asynchronous marker,
and pre-gather fence, and adds the activation boundary immediately before each
pair-0 activation H2D. `ONE_SHOT=1` requires `RESUME=0`, exactly one variant,
`REPEATS=1`, `CREATE_ARCHIVE=1`, and nonexistent output-directory and sibling
archive paths. The activation-fence variant refuses to run outside this mode.
It suppresses resume guidance. Watcher events are published by writing and
syncing a temporary marker, atomically renaming it, syncing the containing
directory, and then publishing a similarly durable `ready` handshake only
after pidfd-bound PID signaling has completed. The main monitor also bounds
watcher and telemetry failure and the entire case with
`ONE_SHOT_TIMEOUT_SECONDS`. If the case exits, the parent publishes a durable
child-exit notice. The watcher
captures immutable process start times for the case and both monitoring
helpers. Every one-shot TERM/KILL uses a pidfd bound to the validated process,
and the watcher never signals after observing a durable child-exit notice. If
the CUDA process remains uninterruptible after bounded TERM/KILL, post-health
probing is skipped and the already durable evidence is archived by the EXIT
trap. The harness never signals the shell or its process group.

This arm adds device-wide synchronization, exact marker validation, and host
logging. Each changes scheduling and timing. Whether the failure surfaces at
the synchronization boundary, during activation-copy setup, or in the entered
H2D API is evidence about where CUDA first reports the fault—not proof of the
underlying hardware or software root cause.

Run this single arm after a clean boot. `SKIP_BUILD=0` is mandatory so the CUDA
changes and the fixed CUDA/P2P preflight are both present:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset CUDA_VISIBLE_DEVICES
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-activation-fence-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-activation-fence \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

Do not resume or append a control arm after device loss. Return the archive
created from this fresh one-shot directory. The existing pre-gather archive is
the predecessor evidence; this is a serial boundary refinement, not a new
multi-stage A/B.

The global-compute-fence evidence supersedes the earlier layer-specific
interpretation. Every observed pair-0 partner-Q8 `cudaDeviceSynchronize()`
returned. On sequence 101, the following mapped-host marker exposed its new
sequence to the CPU but retained the complement from the immediately preceding
high-bit activation marker. The result D2H was never attempted. This is a
partial write/visibility failure in the diagnostic marker path itself (or an
error from concurrent work), not evidence against the named layer-11 Q8
binding. It also means another mapped-host marker cannot safely refine the
original production failure.

`attention-q8-direct-gather-fence` removes that instrumentation. It retains the
same production row split and pair-0 attention/Q8 host-bounce transports, then
performs only a partner-device compute synchronization followed immediately by
the ordinary synchronous result D2H/H2D helper. CPU-side records bracket the
copy and report whether D2H and H2D were attempted and completed. There is no
mapped-host allocation, marker kernel, system fence, or completion event in
this arm. Run it once from a clean boot; do not resume it after device loss:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset CUDA_VISIBLE_DEVICES
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-direct-gather-fence-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-direct-gather-fence \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The accepted outcomes are deliberately narrow: compute synchronization fails;
result D2H is entered but fails; D2H completes and H2D is not entered; H2D is
entered but fails; or the complete gather returns. A dynamic binding identity
is evidence for the final observed boundary only and is never reported as the
cause of the reset.

The direct-gather run proved that pair-0 partner Q8 compute had completed
before the synchronous result D2H surfaced the poisoned CUDA context. It did
not fence concurrent home-device work. The next single-variable cut is
`attention-q8-rows-serialized`: it retains the same row split, row ownership,
partner-local caches, attention kernels, host-bounce bytes, Q8 work, and 250 W
limits, but orders each pair-0 split dispatch as partner attention, partner
device synchronization, home attention, home device synchronization. Pair 1
remains unmodified. A failure at either synchronization names the execution
half on which CUDA observed the error. A full pass proves that overlap is
required for this reproducer; it does not by itself distinguish an electrical
concurrency limit from a driver scheduling defect.

Run it once from a clean boot with `ONE_SHOT=1`; never resume this arm:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset CUDA_VISIBLE_DEVICES
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-rows-serialized-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-rows-serialized \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The resulting `attention-q8-rows-serialized` device-loss artifact did not
contain the serialization marker: it failed during the layer-8 Q8 partner
path, before the first eligible split-attention launch. It therefore
reproduced device loss but did not test whether overlapping row halves were
required. Enabling row splitting also admits partner attention caches and
mirrors raw, compressed, and index-cache rows before a split kernel is
eligible. The next binary cut separates those prerequisite transfers from the
row kernels themselves.

`attention-q8-row-compute-off` leaves pair 0 selected for row splitting, keeps
its partner cache allocations and all three cache-mirror classes, and retains
the pair-0 Q8 host-bounce/direct-gather fences. It suppresses only pair-0 split
indexer/attention computation and result gathers; pair 0 uses the unchanged
home kernels while pair 1 remains fully row-split. The harness requires the
startup marker proving this cut was armed even when the process later loses a
GPU. Run one fresh, non-resumable arm:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset CUDA_VISIBLE_DEVICES
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-q8-row-compute-off-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-q8-row-compute-off \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

This arm did lose the pair, and its startup marker proved that pair-0 split
indexer/attention computation and gathers were disabled. The failure was first
observed at a pair-0 Q8 result D2H after the fenced partner computation had
completed. That does **not** isolate row-cache admission or mirror traffic: the
arm also changed pair-0 attention and Q8 transport to pinned-host bounce and
added Q8 compute/gather fences. The earlier healthy `attention-off` control
retained the same graph-wide partner-cache allocations and mirrors while using
ordinary direct Q8 transport. The defensible result is therefore narrower:
split attention computation is not required for a loss under that heavily
instrumented transport schedule, but the result cannot explain the original
production direct-P2P failure.

The replacement diagnostic uses production direct-P2P/default-stream
transport and no Q8 audit fences. Three one-shot shadow phases form a monotonic
cut of pair-0 attention-owned work:

- `attention-row-query-shadow` performs only the direct query-row handoff;
- `attention-row-partner-shadow` additionally performs partner attention; and
- `attention-row-gather-shadow` additionally gathers the partner rows.
- `attention-row-gather-dst-shadow` repeats the gather shadow with the same
  partner-to-home bytes and direct-P2P direction, but submits the gather from
  the destination/home CUDA context and default-stream ordering domain, which
  also reverses the required peer-access direction.
- `attention-row-gather-chunk16-shadow` returns to source scheduling but emits
  the same 32 MiB gather as two 16 MiB `cudaMemcpyPeerAsync` operations before
  the same single completion event and destination wait. It also retains the
  production destination-ready event/source-wait before the copy. Evidence is
  accepted only when the CUDA helper reports exactly 32 MiB, two 16 MiB
  submissions, source tier 2, destination tier 0, both readiness operations,
  and both completion operations.
- `attention-row-gather-chunk16-paced-shadow` preserves those two 16 MiB copy
  operations, but the destination acknowledges completion of chunk 1 and the
  source waits for that acknowledgement before submitting chunk 2. Its exact
  runtime marker is required for both successful and device-loss evidence.
- `attention-row-gather-scratch-paced-shadow` preserves the paced helper,
  partner-produced source, direction, total bytes, row ownership and accepted
  full-home recomputation, but receives the discarded result in a dedicated
  home allocation that no model kernel reads or reuses. The allocation exists
  only for this diagnostic and has graph-workspace lifetime.
- `attention-row-gather-source-scratch-paced-shadow` retains that dedicated
  home destination and first stages the completed production partner result
  into a dedicated partner allocation with an ordered D2D copy on the partner
  default stream. The same paced two-chunk helper then reads partner scratch
  and writes home scratch. This changes the P2P source allocation and adds an
  explicit source-ready dependency without changing attention compute, bytes,
  physical direction, pacing, or accepted full-home recomputation.
- `attention-row-gather-preinitialized-source-paced-shadow` pre-zeroes the
  same dedicated partner scratch on the partner default stream before the
  production partner attention launch, then performs the identical paced
  scratch-to-scratch P2P copy. The partner result allocation is never read.
  This preserves partner compute, post-compute source-stream ordering,
  direction, bytes, chunking, pacing, both dedicated endpoints, and accepted
  full-home recomputation while removing result contents and result-specific
  source readiness from the transfer.
- `attention-row-gather-preinitialized-source-no-partner-paced-shadow` keeps
  that pre-zeroed partner source, dedicated home destination, direct query
  handoff, and identical two-chunk paced partner-to-home copy, but omits only
  the selected pair's partner-attention launch. Full-home attention still
  produces the accepted output. This determines whether partner attention
  execution is required in the remaining trigger bundle.
- `attention-row-gather-preinitialized-source-partner-output-scratch-paced-shadow`
  retains the exact production partner-attention kernel, query and cache reads,
  and arithmetic, but redirects its discarded output from production
  `peer_heads` into the non-overlapping second half of the dedicated partner
  diagnostic allocation. The static P2P source remains in the first half, so
  its address, contents, direction, two 16 MiB chunks, pacing, and full-home
  accepted output are unchanged. This isolates the production attention-output
  destination/write from the rest of the retained partner-attention workload.
- `attention-row-partner-output-scratch-no-gather-shadow` retains the same
  direct query handoff, production partner-attention kernel/cache accesses,
  non-overlapping scratch output, diagnostic allocation initialization, and
  full-home accepted output, but submits no partner-to-home result transfer.
  Its partner synchronization proves completion of the retained attention
  kernel before home recomputation. This separates attention/cache execution
  from its interaction with the static reverse P2P schedule.
- `attention-row-gather-preinitialized-source-partner-output-scratch-pre-attention-paced-shadow`
  restores the identical stable-source 32 MiB paced reverse transfer but
  submits it before the retained partner-attention kernel instead of after it.
  Query handoff, allocations, source/destination addresses, chunking, pacing,
  attention/cache work, scratch output, and accepted full-home result are
  unchanged. This isolates transfer/attention ordering from combined work.
- `attention-row-gather-preinitialized-source-partner-output-scratch-pre-attention-paced-fenced-shadow`
  keeps that pre-attention ordering and inserts a partner-device synchronization
  followed by a home-device synchronization after the paced copy helper returns
  and before partner attention is submitted. Both required operations, their
  bytes, endpoints, kernels, caches, and accepted output remain unchanged. This
  tests whether their combination fails only while transfer events/work remain
  outstanding, or still fails after an explicit quiescent pair boundary. A
  surviving run below 400 prefill tok/s is classified as underloaded rather
  than accepted; the two preceding single-component controls sustained 459.89
  and 481.45 tok/s, so this rejects a gross scheduling/load collapse.

Pair-0 prefill indexer top-k stays on home in these arms so the accepted output
can be recomputed by the unchanged full-home attention kernel. Pair 1 and all
Q8 partner projections remain production paths. Each retained phase is device-
synchronized before home recomputation; this is an intentional diagnostic
completion boundary, not a performance candidate. Start with the smallest
cut, in a fresh one-shot directory:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 1 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-row-query-shadow-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-row-query-shadow \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

If query-only survives, the next fresh arm is `attention-row-partner-shadow`;
if that survives, the final fresh arm is `attention-row-gather-shadow`. Stop at
the first device loss. This orders the trigger boundary without returning to
layer-by-layer guesses. A failed run is accepted as an activated shadow arm
only after its exact `event=begin` marker is durable; a loss before that marker
is reported as `experiment-not-activated`, not attributed to the selected cut.

The first three arms established a direction-sensitive boundary: query-only
and partner-compute survived, while the source-scheduled partner-to-home gather
lost the pair even though its result was discarded before full-home
recomputation. The next arm is therefore `attention-row-gather-dst-shadow`, not
another layer fence. It changes only the gather's submitting CUDA context and
default-stream ordering plus the required peer-access direction; the physical
data direction and bytes stay fixed. If it survives, that software mechanism
is the next production-fix candidate. If it also loses the device, vary gather
granularity next while holding the partner-to-home direction and total bytes
fixed.

Both one-operation gather schedules lost the same physical pair. The
destination-ordered arm completed its first gather and then surfaced the
poisoned context at layer 15 in that same 512-token microbatch. Consequently,
submission context and peer-access direction are not sufficient explanations.
The next one-shot arm is `attention-row-gather-chunk16-shadow`. A pass isolates
the 32 MiB CUDA operation size; another loss means the next differential must
pace the two 16 MiB chunks with an acknowledgement between them rather than
merely segmenting one continuous transfer burst.

The back-to-back 16 MiB arm also lost the same physical pair after its exact
schedule marker and first shadow completion. Operation size alone is therefore
not sufficient. The next one-shot arm is
`attention-row-gather-chunk16-paced-shadow`; it adds a destination
acknowledgement between chunks while retaining source scheduling, direct P2P,
32 MiB total, row ownership, compute, destination address, and accepted home
recomputation. If it survives, continuous gather burst/overlap is isolated. If
it loses the device, the next differential is the destination allocation: send
the same paced result into dedicated home scratch that is never consumed.

The destination-acknowledged two-chunk arm also lost the same pair after its
exact schedule marker and first synchronized shadow completion. Individual
copy size, a continuous 32 MiB burst, and missing inter-chunk acknowledgement
are therefore not sufficient explanations. The next one-shot arm is
`attention-row-gather-scratch-paced-shadow`. It changes only the receiving
allocation/address and removes downstream reuse of the received bytes. A pass
isolates the production `batch_heads` destination view/allocation or its reuse;
another loss proves that destination is not required and advances the audit to
the production partner source allocation.

The dedicated-home-scratch arm also lost the pair after proving the exact
paced 32 MiB schedule. The production destination allocation/address and its
downstream reuse are therefore not required. The next fresh one-shot arm is
`attention-row-gather-source-scratch-paced-shadow`: production partner
attention still writes its normal result, but an ordered partner-local D2D
copy stages those rows into dedicated partner scratch before the paced P2P
copy reads them. A pass isolates the direct P2P source allocation/address or
its immediate readiness relationship. Another loss proves that neither P2P
endpoint allocation is required, while leaving the local read of the freshly
produced source as the next differential.

The source-and-destination-scratch arm also lost the pair after its exact
activation and completion marker. The cards remained at the requested native
250/260/250/250 W profile, and the production result was staged locally before
the same paced P2P copy. Neither P2P endpoint allocation/address nor downstream
consumption is required; however, that arm still read the freshly produced
partner result. The next one-shot arm is
`attention-row-gather-preinitialized-source-paced-shadow`. It keeps production
partner attention and its source-stream ordering but transfers a pre-zeroed
partner scratch allocation that never aliases or reads the attention output.
A loss reduces the necessary bundle to partner attention followed by generic
partner-to-home direct P2P under the production working set. A pass isolates
the fresh result read/staging boundary, which should then be tested without
result P2P.

The preinitialized-source arm also lost the pair after proving the exact
static scratch-to-scratch schedule. Its one-time source initialization,
durable armed/scheduled markers, and first complete shadow checkpoint all
preceded the later device loss. Production attention-result contents, the
result read, and every downstream use of that result are therefore excluded.
The next one-shot differential was
`attention-row-gather-preinitialized-source-no-partner-paced-shadow`: it keeps
the direct query handoff and identical static paced reverse transfer under the
full surrounding workload, but omits only pair 0's partner-attention launch.

That no-partner arm passed the complete 32K/256-token workload with all four
GPUs healthy: 481.45 prefill tok/s and 16.91 decode tok/s. Its exact first
mixed-row checkpoint proved the 32 MiB pre-zeroed source transfer, and the run
completed all 256 decode tokens. This establishes that partner attention is
required in the measured trigger bundle. It does not yet exclude the failed
arm's write to production `peer_heads`, even though that result was never read.
The next arm therefore retains the exact attention kernel and cache workload
while redirecting only that output write to non-overlapping partner scratch.

That scratch-output arm also lost the physical 02:00.0/03:00.0 pair. It proved
the exact output redirection, static 32 MiB paced transfer, and complete first
mixed-row checkpoint before CUDA later surfaced an unspecified launch failure
at layer 15. The watcher independently recorded both pair endpoints lost, with
no foreign GPU process. Using production `peer_heads` as the attention-kernel
destination is therefore not required. The next differential omits only the
static reverse transfer while preserving direct query handoff, partner
attention/cache work, scratch output, allocation shape, and full-home result.

The no-gather differential passed the entire PP32768/TG256 workload at the
native 250/260/250/250 W limits: 459.89 prefill tok/s, 16.90 decode tok/s,
frontier 256/256, and healthy post-run GPUs. Its exact markers prove the direct
query handoff, one-time partner scratch initialization, retained production
partner-attention/cache kernel, scratch output and full-home result; the paced
reverse-transfer markers are absent. Combined with the passing no-partner
static-transfer arm and the failed post-attention combined arm, each component
alone is insufficient. The next differential restores both and changes only
their order by placing the static reverse transfer before partner attention.

Historical source-scratch reproduction command:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
sudo nvidia-smi -i 1 -pl 260
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.default_limit,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES REQUIRED_POWER_LIMIT_W SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-row-gather-source-scratch-paced-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-row-gather-source-scratch-paced-shadow \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

Run the preinitialized-source differential after a fresh reboot:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
sudo nvidia-smi -i 1 -pl 260
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.default_limit,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES REQUIRED_POWER_LIMIT_W SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-row-gather-preinitialized-source-paced-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-row-gather-preinitialized-source-paced-shadow \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

Historical partner-output-scratch/no-gather command:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
sudo nvidia-smi -i 1 -pl 260
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.default_limit,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES REQUIRED_POWER_LIMIT_W SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-row-partner-output-scratch-no-gather-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-row-partner-output-scratch-no-gather-shadow \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

Run the explicit inter-operation completion-fence differential after a fresh
reboot. The unfenced transfer-before-attention arm has already failed and should
not be repeated:

```bash
cd ~/ds4-iq2-q4
git switch agent/sm75-attention-rowsplit-fault-audit
git pull --ff-only

sudo nvidia-smi -pm 1
for gpu in 0 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
sudo nvidia-smi -i 1 -pl 260
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.default_limit,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES REQUIRED_POWER_LIMIT_W SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-row-gather-before-partner-attention-fenced-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ONE_SHOT=1 \
ONE_SHOT_TIMEOUT_SECONDS=900 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-row-gather-preinitialized-source-partner-output-scratch-pre-attention-paced-fenced-shadow \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

The earlier workload-preserving transport and scheduling arms remain accepted
for reproducing existing evidence:

- `partner-bounce` stages only pair-0 Q8 partner activations and results through
  pinned host memory instead of direct peer copies;
- `bounce-indexer-off` combines that host-staged Q8 transport with pair-0
  decode-indexer scoring on the home GPU;
- `partner-serialized` retains direct peer copies but synchronizes pair-0 after
  every complete partner projection to remove cross-projection overlap;
- `indexer-off` retains ordinary direct partner execution but performs pair-0
  decode-indexer scoring on the home GPU; and
- `production` is the unmodified control.

The earlier Q8 `partner-bounce` arm failed during its 512-token warmup after one
complete 65 MiB pair-0 round trip. The serialized-direct arm completed that
warmup, then failed in the first measured 32K chunk immediately after pair-0 prefill
attention row splitting began. Decode-indexer-only controls therefore do not
isolate the observed prefill failure; `attention-off` controls the relevant
path.

The current fault-audit arm requires each card's native ceiling by physical
index: GPU0=250 W, GPU1 (the workstation card)=260 W, GPU2=250 W, and
GPU3=250 W. The telemetry watcher aborts the active case if any limit changes
or a device becomes unavailable. This matters because the serialized archive
recorded an external sequential rewrite from 250 W to 225 W during engine
startup; accepting that run as a 250 W arm would be invalid. The
transport/scheduling diagnostics run before the known control. They retain the
same arithmetic work but host staging and serialization necessarily change the
timing and instantaneous power envelope; a passing arm narrows the trigger but
is not described as a power-matched proof. Every case preserves a
flushed engine/prefill/decode journal, begin/complete byte checkpoints for Q8
partner transfers, per-dispatch byte records for decode-indexer row splitting,
and ordinary passive GPU telemetry. The harness intentionally does not poll
external NVLink counters during an arm. A separate home-built `nvbandwidth`
workload was observed running concurrently, so pre-arm and in-arm guards now
reject any GPU compute process other than the active `ds4-bench` PID. If a GPU
loss interrupts the shell,
reuse the same directory with `RESUME=1`; the interrupted arm is retained and
the next arm runs. It is classified as a device-loss failure only when the
watcher or an explicit unhealthy post-run snapshot corroborates that outcome.
If the benchmark completed but the runner stopped during post-run validation,
resume recovers that arm only when its existing CSV, binding inventory,
progress journal, log, and healthy post-run snapshot all validate and no watch
marker exists; otherwise it remains unverified instead of being rerun. Resume
advances the remaining never-started arms; a retained unverified arm must be
repeated in a fresh matrix rather than overwritten in place.

`RESUME=1` requires the exact original variant list and order. Use a new output
directory when selecting a different arm; do not point a new variant list at an
older directory.

```bash
cd ~/ds4-iq2-q4

sudo nvidia-smi -pm 1
for gpu in 0 2 3; do
  sudo nvidia-smi -i "$gpu" -pl 250
done
sudo nvidia-smi -i 1 -pl 260
nvidia-smi \
  --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
  --format=csv

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"

GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-off,production \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TELEMETRY_INTERVAL_MS=500 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

To audit the exact row-split failure coordinate without disabling row splitting:

```bash
ATTN_PHASE_AUDIT_LAYER=17 \
ATTN_PHASE_AUDIT_POS=512 \
VARIANTS=attention-phase-audit \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

To retain production overlap through the last pair-0 result gather and add
only the end-completion boundary before downstream work:

```bash
ATTN_END_FENCE_LAYER=21 \
ATTN_END_FENCE_POS=512 \
VARIANTS=attention-end-fence \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

To audit the adjacent layer-17/layer-18 boundary in one production-shaped arm:

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-row-boundary-audit-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
ATTN_ROW_BOUNDARY_END_LAYER=17 \
ATTN_ROW_BOUNDARY_ENTRY_LAYER=18 \
ATTN_ROW_BOUNDARY_POS=512 \
VARIANTS=attention-row-boundary-audit \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

Run the three scheduling candidates and the unmodified control as one fixed
matrix, in this order, under one output directory/manifest. Start from a clean
boot, then run the harness's fixed build, long-context smoke, and bidirectional
ordered-copy preflight before the first arm. If an arm loses a GPU, reboot,
restore and verify all four 250 W limits, then reuse the exact directory,
variant list, and order with `RESUME=1` and `SKIP_BUILD=0`. Repeating that
preflight after every reboot keeps the pre-arm CUDA/P2P history consistent; do
not use `SKIP_BUILD=1` for this matrix. The harness retains the interrupted arm
as evidence and advances to the next arm.
Do not create a new directory for each matrix arm: the per-directory summarizer
requires all four outcomes before it makes a factorial comparison. Production
runs last as the same-matrix positive control.

```bash
cd ~/ds4-iq2-q4

export MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf"
export PROMPT="$PWD/speed-bench/promessi_sposi.txt"
unset SMALL_BAR1_ISOLATION_DIR
export SMALL_BAR1_ISOLATION_DIR="$PWD/sm75-attention-copy-scheduling-matrix-$(date -u +%Y%m%dT%H%M%SZ)"

RESUME=0 \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
SMALL_BAR1_PAIR=0 \
VARIANTS=attention-query-dst,attention-gather-dst,attention-both-dst,production \
PP_TOKENS=32768 \
TG_TOKENS=256 \
REPEATS=1 \
REQUIRED_POWER_LIMIT_W=250 \
TELEMETRY_INTERVAL_MS=500 \
POST_CASE_SETTLE_SECONDS=5 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-small-bar1-pair-isolation.sh
```

After a reboot-triggering loss, do not rerun the `unset` or timestamp assignment.
Export the exact `SMALL_BAR1_ISOLATION_DIR` printed by the first run, then rerun
the environment block with the identical `VARIANTS` value/order, `RESUME=1`,
and `SKIP_BUILD=0`. A missing result or an unverified post-run health snapshot
is **no verified outcome**, not a failed arm and not a pass. Resume advances
arms that never started, but deliberately does not overwrite an incomplete
result record. If an arm remains incomplete after the matrix traversal, repeat
the full matrix in a fresh directory before factorial inference. If production
passes, the known fault was not reproduced in that matrix and candidate
differences are not causal evidence.

After the complete matrix identifies an apparent passing candidate while the
production control fails, confirm that candidate and the production control in
separate one-arm directories, each begun from a fresh boot. Those confirmation
runs test repeatability without losing the fixed-matrix comparison to boot,
temperature, and prior-arm history.

The earlier execution-off arm remains useful only as evidence that static
344/344 admission and low-volume row traffic can survive at reduced load. It is
not used for causal attribution because it removed thousands of partner
projections and cut 32K prefill throughput by roughly threefold. The new
factorial retains partner computation and separates Q8 transport, prefill
attention rows, prefill/decode-indexer rows, and overlap while telemetry
records the remaining timing/power differences.

### Bounded SM75 P2P direction audit

`cuda-sm75-p2p-direction-audit.sh` escalates a production-shaped partner-FP16
projection through explicit traffic levels without loading a full model.  The
first device in each pair is the logical home and the second is the execution
partner, so the large result travels from the second device back to the first.
Each level runs in a fresh process and leaves a durable `.started` marker if the
host disappears.  The default T32 shape transfers 2 MiB of activation and 64
MiB of result per call at 512 tokens.

```bash
PAIRS="1,0 0,1" \
SCENARIO=t32 \
TOKENS=512 \
REPEAT_LEVELS=1,64,256,512,1536 \
SKIP_BUILD=0 \
bash ./speed-bench/cuda-sm75-p2p-direction-audit.sh
```

# SM75 compressor projection/state-store fusion (synthetic qualification)

`cuda-sm75-compressor-state-fusion.sh` compares the diagnostic one-token
paired-F16 projection/state-store kernel with the shipping ordered paired-F16
projection followed by `compressor_store`.  It covers attention and indexer
widths, ratio-4 and ratio-128 intermediate/emit/wrap phases, F16 and F32 APE,
destination canaries, runtime occupancy/resources, per-symbol PTXAS/SASS, and
selected Compute Sanitizer cases.

```bash
PROFILE_GPU=0 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=25 \
RUN_SANITIZER=1 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compressor-state-fusion.sh
```

This remains useful as bounded kernel evidence. The fusion is now enabled by
default after exact four-GPU mixed15/all43 qualification. Set
`DS4_CUDA_DISABLE_COMPRESSOR_PAIR_STATE_STORE=1` for the retained reference
rollback path.

### SM75 compressor projection/state-store production A/B

`cuda-sm75-compressor-state-production-ab.sh` runs a one-repeat, one-shot
four-GPU **decode** A/B for the exact fused compressor pair projection and
recurrent-state append. It covers the mixed15 and all43 models at the 512, 4096,
and 32768 prompt frontiers, requires the stable 22/21 topology, proves decode
dispatch of all three production widths, and requires byte-identical logits for
every checked decode token before accepting the result. A prefill-only run
cannot exercise this one-token fusion. The script deliberately has no resume
mode because a GPU-loss run must be restarted from a clean host state.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compressor-state-production-ab.sh
```

The accepted production run was byte-exact for 16 decode tokens at PP512,
PP4096, and PP32768 on both models. All43 improved by 0.410% to 0.743%; mixed15
was effectively neutral (-0.312% to +0.058%) at the longer prompt frontiers.

### SM75 single-token T32 FP16-output diagnostic

`cuda-sm75-t32-decode-f16-ab.sh` is a bounded, one-GPU, production-shape
diagnostic for a single decode token. Here T32 means 32 native Q8 blocks per
1024-element input row; it does not mean a 32-token batch. The three arms
separate native-Q8/F32 projection, cached-F16/F32 projection, and the opt-in
cached-F16/FP16-output helper. Single-token admission is enabled only inside
this diagnostic; the production decode default remains unchanged.

```bash
PROFILE_GPU=0 \
CUDA_ARCH=sm_75 \
TIMING_ROUNDS=9 \
TIMING_REPEATS=100 \
WARMUPS=5 \
RUN_SANITIZER=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-t32-decode-f16-ab.sh
```

The native-Q8/F32 to cached-F16/F32 comparison measures the compute/cache
change. The cached-F16/F32 to cached-F16/FP16 comparison isolates output
storage only when the report says `storage_axis_isolated=1`; otherwise changing
the cuBLAS output type also changed numerical compute behavior, so that ratio is
a combined candidate result. This bounded test cannot promote the path: any
default change still requires exact four-GPU logits and production decode A/B
evidence.

### Size-neutral SM75 Q8_0 warp-interleaved prototype

`cuda-sm75-q8-warp-interleaved.sh` compares the shipping one-token prequantized
Q8_0 consumer with a standalone SM75 prototype that stores each 32-block group
as a contiguous half-scale plane plus eight interleaved `int32` word planes.
It applies the matching word-plane layout to the quantized activation. Both
representations retain exactly the canonical byte count, lane ownership, eight
DP4A operations per block, accumulation order, and warp reduction order.

```bash
PROFILE_GPU=0 \
CUDA_ARCH=sm_75 \
BENCH_ROUNDS=14 \
BENCH_LAUNCHES=100 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q8-warp-interleaved.sh
```

The report separates the steady consumer, per-token activation repack, and
one-time weight repack costs, under both warm and cache-scrubbed timing. The
candidate must remain bit-exact across the production T32 shape, a two-group
shape, and an unaligned partial-group/tail shape. This prototype does not alter
the production model format, cache, quantizer, or dispatch.

### SM75 warp-interleaved Q8_0 production decode A/B

`cuda-sm75-q8-warp-interleaved-production-ab.sh` validates the production
default for the full single-token T32 query projection (`1024 x 32768`). The
control sets `DS4_CUDA_Q8_WARP_INTERLEAVED_T32_DECODE=0` and retains canonical
Q8_0. The candidate uses the default selector and lazily creates one equal-size
interleaved copy per layer/device. Canonical storage remains available for
prefill, and the default per-device interleaved-cache cap is 1024 MiB. Set
`DS4_CUDA_Q8_WARP_INTERLEAVED_CACHE_MB` to override that cap; setting either
the cache cap or `DS4_CUDA_Q8_WARP_INTERLEAVED_T32_DECODE` to zero is an
explicit rollback.

The one-shot test covers mixed15 and all43 at PP512, PP4096, and PP32768. It
runs one 256-token throughput pass and a separate 16-token exact-logit pass per
arm, requires at least one complete 43-layer set of cache fills with zero
fallback, verifies GPU
identity and power limits before and after every process, and rejects any
non-byte-identical logit.

The `20260903T040314Z` archive is not candidate performance evidence. Its
opt-in gate incorrectly named the diagnostic half-width `1024 x 16384` shape,
while production decode executes the full `1024 x 32768` query projection.
Consequently the candidate recorded zero fills and zero calls; the small CSV
differences between those two runs are an unchanged-control noise sample.

The corrected `20260903T043348Z` four-GPU run exercised 33,024 interleaved
calls per throughput arm and 2,064 per exact-logit arm, with 43 cache fills and
zero fallbacks. All 16 decode logits at PP512, PP4096, and PP32768 were
byte-identical for mixed15 and all43. The interleaved path improved steady
decode by 5.97%, 5.03%, and 4.55% for mixed15 and by 5.79%, 5.04%, and 4.54%
for all43 at those frontiers, respectively. That evidence promotes the path to
the SM75 production default for this exact shape.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
INTERLEAVED_CACHE_MB=1024 TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q8-warp-interleaved-production-ab.sh
```

### SM75 attention-output Q8_0 warp-interleaved A/B

`cuda-sm75-q8-attention-output-interleaved.sh` extends the size-neutral
warp-interleaved Q8_0 representation to the two single-token attention-output
consumers: the grouped A projection and K-sliced B projection. Both paths are
SM75 production defaults behind independent
`DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE` and
`DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE` selectors. Set either selector to
zero for exact rollback. The B cache contains only the consumed 4096-wide
slice of the canonical 8192-wide matrix.

The first A follow-up used a four-block 128-wide fixture. The subsequent
`20260904T013039Z` production audit proved that fixture was not representative:
each one-token TP A call consumes four independent 4096-value groups and its
weight slice is 4096x4096 (128 Q8 blocks per output row). Direct-XQ selected,
but the rowtile4 gate correctly did not. That bounded result is therefore not
promotion evidence.

The corrected A candidate uses the same size-neutral 32-lane word planes and
fixed four-group K128 loop accepted for B. It selects the proper activation row
for each 1024-row rank slice, preserves the canonical per-lane accumulation
and final warp reduction, and quantizes the four source groups directly into
the interleaved representation. Set
`DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_K128=0` or
`DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DIRECT_XQ=0` for independent rollback.

The bounded diagnostic requires bit-identical A intermediates and B outputs.
The A acceptance test is deliberately two-arm: canonical versus the specialized
K128/direct-XQ candidate. Set
`PROJECTIONS=a` to time and sanitize A alone before any four-GPU production
test. B retains its useful three-arm comparison because both of its interleaved
implementations were accepted and the incremental result measures the K128
follow-up directly.

```bash
PROFILE_GPU=0 CUDA_ARCH=sm_75 \
PROJECTIONS=a \
TIMING_ROUNDS=9 TIMING_REPEATS=100 \
RUN_SANITIZER=1 SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q8-attention-output-interleaved.sh
```

The corrected `20260904T022051Z` A-only run exercised the real 4096x4096 TP
slice. Its engine regression was bit-identical, Compute Sanitizer passed, and
the K128/direct-XQ candidate reduced the owned call from 0.078653 ms to
0.039485 ms (1.992x).

The B follow-up removes the canonical activation scratch plus repack launch by
quantizing directly into the interleaved word planes. It also specializes the
production 4096-wide slice as four fixed 32-block groups, eliminating dynamic
group/tail control without changing DP4A, floating accumulation, or warp
reduction order. The bounded harness reports three arms: canonical control,
the original interleaved implementation, and the direct-XQ/fixed-K128
candidate. The two refinements can be rolled back independently with
`DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DIRECT_XQ=0` and
`DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_K128=0`.

The B-only production A/B leaves the accepted T32 interleaved default enabled
in both arms, explicitly keeps A disabled, and changes only the B selector.
Its 1536 MiB per-device cache floor accommodates the T32 and B representations
together. It requires direct-XQ and K128 dispatch on every candidate B call,
zero cache fallback, stable GPU identity, and byte-identical logits for both
models at all three frontiers.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
INTERLEAVED_CACHE_MB=1536 TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q8-attention-b-production-ab.sh
```

After bounded acceptance of both specialized projections, the combined A+B
production A/B changes both attention-output selectors together while leaving
the accepted T32 interleaved default enabled in both arms. Candidate validation
requires direct-XQ/K128 on every selected A call, direct-XQ/K128 on every
selected B call, at least 43 cache fills for each representation, zero cache
fallback, stable GPU identity, and byte-identical logits for mixed15 and all43
at 512, 4096, and 32768 tokens.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
INTERLEAVED_CACHE_MB=1536 TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q8-attention-ab-production-ab.sh
```

The `20260904T022454Z` four-GPU run accepted the combined A+B path for both
model layouts. All 96 control/candidate decode-logit files were byte-identical;
every selected A and B call used direct-XQ/K128; all candidate arms recorded
215 cache fills, zero fallbacks, and 4386 MiB total resident interleaved
weights; and every pre/post GPU identity and power snapshot matched. Decode
improved by 9.90%, 8.55%, and 7.84% for mixed15 and by 10.62%, 8.49%, and
7.65% for all43 at 512, 4096, and 32768 tokens. This promotes both attention
output paths to the SM75 single-token defaults and raises the default
per-device interleaved-cache ceiling from 1024 MiB to the validated 1536 MiB.
The A/B harness still forces both selectors off in its control arms.

### Withdrawn native-primary T32/A/B Q8 residency experiment

The remainder of this section is a historical record of the rejected
experiment. Do not run it: the harness and engine now reject the selector
before GPU allocation. Its successive workarounds never restored the exact
selective-cache allocation and lookup topology used by the stable auxiliary
configuration.

The interleaved T32, attention-A, and attention-B layouts are individually
size-neutral, but the first production candidates retain each layout beside
the replicated canonical Q8 tensor. Across 43 layers that arrangement keeps
8,772 MiB of canonical T32/A/B bytes plus 4,386 MiB of interleaved bytes on
the four GPUs. The auxiliary representation is therefore a validation
mechanism, not the desired storage policy.

The withdrawn experiment set `Q8_INTERLEAVED_PRODUCTION_TARGET=native-primary`
to compare that complete
auxiliary-cache arrangement with exact native-primary TP residency. T32 keeps
one full 34 MiB native matrix on its production home rank; A is row-sharded
and B is K-sharded into 17 MiB rank slices. The candidate therefore keeps
exactly 4,386 MiB system-wide and omits all 8,772 MiB of persistent canonical
GPU copies. Canonical bytes remain in the GGUF mmap and are staged only
transiently if the independently planned F16 cache needs to be materialized.
The largest conversion staging span is 34 MiB.

The gate holds the optimized T32/A/B dispatch and F16-cache residency constant,
requires 215 retained native shards plus 258 authorized canonical source
spans, rejects any lazy interleaved fill or fallback in the candidate, writes
per-device allocator state, and compares every one of 16
decode logits at 512, 4096, and 32768 tokens for mixed15 and all43. It reports
both total free-memory change and every relevant persistent category instead
of treating the 1,536 MiB auxiliary cap as the memory result.

Native-primary residency must also be execution-topology neutral. Omitting the
canonical copies exposes additional allocator headroom before the stage-aware
F16 planner runs; the first implementation consequently moved 128 of 129
movable projections to partners instead of the accepted auxiliary-cache
control's 117. More importantly, it eagerly allocated all native shards before
prefill, whereas the accepted auxiliary path did not materialize its native
decode cache until single-token decode. That eager-residency change remained
common to every native-primary GPU-loss run, including the partner-suppressed
diagnostics and the corrected 117/129 planner run.

Native-primary installation now preserves all 8,772 MiB of canonical T32/A/B
source residency through prefill and registers mmap-backed native decode
descriptors without allocating them. Consequently the F16 planner sees the
real control-equivalent budget; no synthetic residency shadow is involved.
At the first single-token decode request, the engine releases all canonical
primary-source slabs before allocating any native replacement, then lazily
packs and retains exactly 4,386 MiB of native shards. This makes the change a
bounded prefill-to-decode lifecycle swap instead of a startup residency
change. The A/B harness records all 344 plan rows and requires the candidate
plan to be byte-identical to its control before accepting the memory result.

Before the full two-model A/B, the native-primary mode runs a candidate-only
512-token mixed15 preflight. It requires the complete F16 plan, healthy GPU
identity before and after the run, positive prefill/decode results, and zero
selective-cache misses. It also requires the 8,772 MiB canonical-source
release marker, deferred native materialization markers, no residency-shadow
override, and the accepted 117/129 movable-partner count. This specifically
guards the batched-prefill contract:
attention A must select its resident F16 binding before any canonical-Q8
fallback, while attention B must defer all residency selection to the generic
F16/native-primary dispatcher. A failed preflight stops the harness before the
long production comparison begins.

The first native-primary preflight exposed two eager-lookup defects rather
than a valid storage comparison. The attention wrapper initially requested
canonical A before consulting its F16 binding. After that was corrected, the
generic Q8 dispatcher still requested canonical B before consulting resident
F32/F16 and partner bindings; the paired batched dispatcher had the same
ordering hazard before delegating to the generic path. The CUDA regression now
forbids both wrapper and generic attention-output canonical labels during a
batched F16 execution, so these storage-policy violations fail before the
four-GPU run. Canonical Q8 is resolved only after every resident batched path
has declined the operation.

For transfer isolation after a GPU-loss event, set
`NATIVE_PRIMARY_DISABLE_T32_PARTNER_PAIRS=0` together with
`NATIVE_PRIMARY_PREFLIGHT_ONLY=1`. The F16 planner, partner-resident weights,
scratch reservations, and native-primary layout remain unchanged, but T32
`attn_q_b` calls for logical pair 0 fall through to the full native-primary
matrix on the home GPU. Attention-output partner execution and all pair-1
execution remain enabled. The harness requires the scoped-suppression marker,
rejects any pair-0 T32 partner-dispatch marker, runs only the 512-token
preflight, archives it, and exits before the production A/B.

If that T32-only suppression still loses GPU 1, replace it with
`NATIVE_PRIMARY_DISABLE_ALL_PARTNER_PAIRS=0`. This preserves the same F16
admission, partner-resident weights, scratch reservations, native-primary
partner shards, pair-1 execution, and preflight workload, while making every
pair-0 Q8/F16 partner call fall through to its home-side path. Because the
production attention-B native representation is normally K-split, this
diagnostic expands its pair-0 home shard from 17 to 34 MiB per layer while
retaining the original 17 MiB partner shard. The resulting approximately
374 MiB of extra GPU0 residency is explicit and confined to the diagnostic;
without it there is no complete home-side B matrix and execution suppression
fails with a selective-cache miss before exercising the workload. Relative to
the T32-only arm, the additional removed execution is pair-0 attention-output
and other admitted pair-0 Q8/F16 partner work. The harness requires both the
pair-wide suppression and full-home-fallback markers and rejects every pair-0
partner-dispatch marker.

If that arm still loses GPU 1, use
`NATIVE_PRIMARY_DISABLE_PARTNER_ADMISSION_PAIRS=0`. This moves the boundary
earlier: pair 0 admits no partner F16 weights or partner scratch, and the
stage-aware planner cannot assign its movable candidates to GPU1. Pair 1 keeps
normal admission and execution. Pair-0 native-primary partner shards and the
peer-access topology remain installed, while attention B retains the explicit
full-home fallback above. A stable result therefore implicates pair-0 F16
admission/materialization or its added residency. Another GPU1 loss instead
moves the suspect boundary to native-primary/base selective residency or peer
mapping, rather than active Q8/F16 partner execution.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
Q8_INTERLEAVED_PRODUCTION_TARGET=native-primary \
INTERLEAVED_CACHE_MB=1536 TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-q8-warp-interleaved-production-ab.sh
```

This runtime form deliberately validates the final persistent layout before a
new GGUF tensor encoding is committed. A GGUF-native row-interleaved and
K-interleaved format family can then remove startup repacking without changing
placement, kernels, or retained VRAM. T32 uses a full row-interleaved matrix,
A uses row-sharded row-interleaved matrices, and B uses K-sharded
K-interleaved matrices; one ambiguous "interleaved Q8" type would not encode
those placement semantics safely.

### SM75 width-1024 compressor projection layout diagnostic

`cuda-sm75-compressor-projection-layout.sh` targets the expensive one-token
paired F16 attention-compressor projection (`4096 x 1024` twice) without
changing production dispatch. Its control reproduces the production ordered
32-thread kernel, including recurrent-state stores. A no-cache candidate keeps
the canonical model layout and cooperatively stages four padded 32x32 tiles in
shared memory using coalesced global loads. Two size-neutral candidates instead
transpose each row into lane-major order so a warp loads adjacent F16 values.
All candidates preserve each lane's original 128-element accumulation sequence
and the final lane-order reduction. The first lane-major candidate retains one
warp per output row; the second assigns the independent KV and score
accumulators to separate warps in the same CTA.

The one-time weight transpose is excluded. The 16 KiB activation transpose is
measured both outside and inside each candidate launch. The harness requires
bit-identical projection outputs and recurrent state, intact canaries, SM75
resource evidence, Compute Sanitizer, and validated Nsight captures for all
four kernels.

The `20260903T204338Z` three-arm run accepted both packed candidates as
bit-exact and sanitizer-clean. The control took 0.29959 ms; the one-warp
lane-major consumer took 0.03293 ms (9.10x), or 0.03580 ms including the
per-token activation transpose (8.36x). Two warps took 0.03285 ms, providing
no material benefit while increasing registers from 44 to 60. Nsight measured
global-load efficiency rising from 8.33% to 99.91%, sectors per request falling
from 31.83 to 2.66, and DRAM utilization rising from 13.10% to 73.64%. The
canonical shared-staging arm added here determines whether most of that gain
can be retained without a second weight representation; this remains bounded
diagnostic evidence rather than a production promotion.

The first canonical-staging run (`20260903T210550Z`) was bit-exact and
sanitizer-clean, improving 0.29952 ms to 0.13102 ms (2.29x), but it did not
fully test the no-cache design: only weights were staged while activation loads
remained in the original scattered lane pattern. Nsight consequently reported
22.22% load efficiency, 11.97 sectors per request, and a 9.41 MIO-throttle
stall ratio. The current arm stages the activation in
the same coalesced padded tile, eliminating that identified residual global-
load pattern without changing model storage or arithmetic order.

The corrected `20260903T211314Z` run was also bit-exact and sanitizer-clean.
Coalescing both canonical weights and activation reduced 0.27846 ms to
0.04277 ms (6.51x). Nsight measured 99.91% global-load efficiency, 2.66
sectors/request, and 360.68 GB/s versus 8.33%, 31.80, and 93.08 GB/s for the
control. The lane-major arm remained faster at 0.03523 ms including its
per-token activation transform (7.90x), but its remaining advantage projects
to only about 0.2 percentage points of aggregate decode GPU work while an
auxiliary copy of both width-1024 matrices would consume 336 MiB across the 21
ratio-4 layers. The no-cache canonical path is therefore the production
candidate; lane-major remains architecture evidence rather than a proposed
model-format change.

```bash
PROFILE_GPU=0 CUDA_ARCH=sm_75 \
BENCH_ROUNDS=9 BENCH_LAUNCHES=25 \
RUN_SANITIZER=1 RUN_NCU=1 NCU_USE_SUDO=1 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compressor-projection-layout.sh
```

### SM75 canonical-staged compressor projection production A/B

`cuda-sm75-compressor-projection-production-ab.sh` compares the existing
exact fused projection/state-store kernel with the opt-in canonical shared-
staging implementation for the production `4096 x 1024` ratio-4 compressor
pair. Both arms retain the already accepted state-store fusion. The candidate
reads the original row-major F16 model directly and reports zero auxiliary
model bytes; it does not create a repacked weight cache.

The accepted `20260903T213736Z` one-shot four-GPU test covered mixed15 and
all43 at PP512, PP4096, and PP32768. All 96 control/candidate logit pairs were
byte-identical; all 32 pre/post GPU snapshots matched; and each throughput arm
recorded 16,128 staged calls with zero auxiliary model bytes. Decode improved
by 9.16%, 8.50%, and 7.20% for mixed15 and by 10.33%, 9.05%, and 8.36% for
all43 at those frontiers. This promotes canonical staging to the SM75
width-1024 production default. Set
`DS4_CUDA_COMPRESSOR_PROJECTION_STAGED=0` or
`DS4_CUDA_NO_COMPRESSOR_PROJECTION_STAGED=1` for the exact retained rollback.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compressor-projection-production-ab.sh
```

### SM75 small-width canonical-staged compressor projection A/B

`cuda-sm75-compressor-projection-small-production-ab.sh` extends the same
exact canonical shared-staging implementation to the two compressor shapes
that were deliberately excluded from the width-1024 promotion: the ratio-128
attention projection (`4096 x 512` twice) and the ratio-4 indexer projection
(`4096 x 256` twice).  The control arm retains the accepted width-1024 staged
default in both variants, so this isolates only the incremental width-256/512
change.  The accepted `20260903T230735Z` run improved mixed15 decode by 7.13%,
6.07%, and 5.37% at PP512, PP4096, and PP32768; all43 improved by 8.18%,
7.05%, and 6.10%.  All 96 control/candidate logit pairs were byte-identical,
all 32 GPU-health snapshots matched, and both candidate throughput arms
recorded 16,128 width-256, 15,360 width-512, and 16,128 width-1024 staged
calls with zero auxiliary model bytes.  Small-width canonical staging is now
the SM75 production default.  Set
`DS4_CUDA_COMPRESSOR_PROJECTION_STAGED_SMALL=0` or
`DS4_CUDA_NO_COMPRESSOR_PROJECTION_STAGED_SMALL=1` for the small-width-only
rollback; the global staged rollback continues to override every width.  No
repacked model or persistent auxiliary cache is allocated.

```bash
MIXED_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf" \
ALL43_MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
TG_TOKENS=256 EXACT_TOKENS=16 \
SKIP_BUILD=0 CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compressor-projection-small-production-ab.sh
```

### Bounded exact compact-attention KV codec

`cuda-sm75-compact-attention-kv.sh` does not alter production dispatch. It
uses the shipping `ds4_cuda.o` quantizer, built with production flags, to
produce the reference 512-float compressed-attention row and packs it into a
736-byte exact representation: seven F32 power-of-two scales, 448 E4M3
sign/index bytes, and 64 untouched F32 RoPE values. The harness requires
bit-identical unpack and consumer output, rejects non-finite or
non-representable rows, checks guarded allocations under Compute Sanitizer,
covers scale-floor, alternate-scale, upper-exponent, and signed-zero
boundaries, records PTXAS/SASS resources, and times paired, alternating
production-shaped indexed-attention consumers. The F32 control and original
direct compact decoder remain present. Byte-exact candidates are timed
independently:

- selected-row materialization decodes the top-512 rows once into reusable
  F32 scratch and includes both that preprocessing and the F32 consumer;
- H16 lets one 512-thread block amortize each decoded row over sixteen heads;
- the warp-staged arm uses two dedicated loader warps and double-buffered
  shared memory while eight compute warps consume the prior stage;
- the exact F16-plus-exceptions representation stores 512 inline halves plus
  a fixed source-ordered capacity of 64 F32 exceptions, with a bitmap/prefix
  index for O(1) direct loads. Rows exceeding that capacity fail closed;
- the hybrid pipeline retains the persistent 736-byte cache, materializes
  reusable 256-row chunks as 448 exact F16 non-RoPE values plus 64 untouched
  F32 RoPE values, and overlaps the second materialization with attention using
  two buffers and independent CUDA streams. The retained exact H16 consumer is
  measured with default-priority streams and with high-priority attention plus
  low-priority materialization. This isolates whether the ready attention grid
  is delayed behind the second materializer grid; the prior H8 arm is not
  rerun. Any non-RoPE value that does not round-trip through F16 bit-exactly
  sets a failure status and rejects both scheduling arms.

Every accepted codec and attention result must be bit-identical to F32.
Selected-row scratch is reported separately as transient, reusable allocation;
packing is excluded from persistent-cache consumer timing, while selected-row
materialization is included because it recurs for every selection. The
selected-row arm is also decomposed into materialization-only and
already-materialized F32-consumer timings. Their sum of independent medians is
diagnostic; the existing combined launch sequence remains the authoritative
end-to-end result. The selected consumer is paired against the ordinary F32
consumer to expose any benefit from contiguous top-k ordering independently
of decode cost. The hybrid arm likewise reports materialization-only and
H16 attention-only component timing, but each paired scheduling pipeline is
the authoritative result. Its two buffers are reported as transient
storage; the F32 max/sum carry preserves the original online-softmax row order
across both chunks. With `RUN_NCU=1`, the harness captures exactly one
`compact_materialize_hybrid_chunk_kernel` launch and validates its 128-thread,
`256 selected rows * TOKENS`-CTA geometry. The report includes DRAM read/write traffic,
load/store sector efficiency, L2 hit rate, integer/conversion/memory
instruction counts when exposed by the installed Nsight version, occupancy,
waves, eligible warps, and long-scoreboard/MIO stalls. This separates
CTA-scheduling, compact-load, reconstruction-instruction, and output-bandwidth
limits without profiling an attention consumer by mistake. The
736-byte loader uses aligned code-word loads, half-warp scale broadcast, and
exact IEEE reconstruction with a boundary fallback. All results in this
section remain bounded diagnostic evidence. The standalone packer verifies
row status before accepting output. The production source contract is the
finite pre-quantization F32 producer row: the compact packer performs the one
shipping E4M3 rounding pass and retains its scale and codes directly.
Persistent rows retain fail-closed status metadata, and cold
unpack/checkpoint export propagates a rejected row as NaN instead of silently
accepting altered data. The reported SASS LDL/STL counts are explicitly
whole-binary diagnostics, not a per-kernel acceptance gate.

The 262144-row diagnostic selected H16 over H8: H16 reached `0.93723x` of its
paired F32 control, while H8 reached `0.92365x`. H16 attention-only was already
slightly faster than the paired F32 control (`1.04887 ms` versus `1.06011 ms`),
but the complete pipeline hid only about `0.013 ms` of `0.0969 ms`
materialization. The priority A/B therefore keeps the selected representation,
chunk size, H16 kernel, and arithmetic fixed and changes only CUDA scheduling.
The accepted `20260905T043103Z` 256K rerun was bit-exact and sanitizer-clean;
high-priority attention reached `0.98930x` of F32 versus `0.93886x` at default
priority. The standalone fixture's deliberately synthetic all-43-layer,
262144-rows-per-layer estimate falls from 23085449216 to 8296333312 bytes, a
64.0625% reduction. That is not the production allocation: DeepSeek-V4-Flash
uses 21 ratio-4 layers and 20 ratio-128 layers, so a 262145-token production
context holds only 65538 or 2050 compressed rows per layer, respectively.
The two reusable hybrid buffers remain about 18 MiB for 32 tokens.

```bash
PROFILE_GPU=0 \
ROWS=8192 \
TOKENS=32 \
TIMING_ROUNDS=7 \
TIMING_REPEATS=25 \
RUN_SANITIZER=1 \
RUN_NCU=1 \
NCU_USE_SUDO=1 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compact-attention-kv.sh
```

### SM75 exact compact-attention KV production A/B

`cuda-sm75-compact-kv-production-ab.sh` compares the ordinary persistent F32
compressed-attention cache with the exact 736-byte SM75 cache on the same
tagged-native all43 model. The candidate keeps persistent compact rows,
materializes only selected rows into two reusable 256-row hybrid buffers
(`448 x FP16 + 64 x FP32`), and consumes those buffers with high-priority H16
indexed attention. Large-token prefill consumes the exact compact rows
directly. Its producer bypasses the separate rounded-F32 staging quantizer;
compact packing owns the sole shipping E4M3 quantization pass. The engine
preserves F32 session/checkpoint payloads at the external boundary.

Single-token non-indexed decode reconstructs each compact row once into a
reusable 2 MiB per-home-device F32 stage, then uses the same exact two-pass
score and finalize kernels as the F32 cache. The stage is reserved during graph
setup rather than lazily in the first decode token. Score accumulation order,
softmax reduction, source-row order, and the value accumulation chain remain
identical. Direct compact loads and the older compact online-softmax kernel
remain fallbacks when materialization or exact score-split is unavailable.

The runner's untimed warm-up intentionally recreates the session while keeping
the engine-owned dense-Q8 residency. Session teardown therefore synchronizes
every participating CUDA device before releasing graph allocations; this is a
teardown-only lifetime boundary, not a production inference fence. Opt-in
phase, lifecycle, and compact snapshot traces remain enabled in this A/B so a
failure can be assigned to measured prefill, teardown, or session I/O.

The one-shot qualification allocates 256K context capacity while measuring
the PP512/4096/32768 frontiers. It requires byte-identical prefill and decode
logits, unchanged four-GPU health, no throughput frontier below `1.0x`, the
stable 22/21 pipeline and pair policy, and actual compact-hybrid dispatch. Its
default aggregate VRAM floor is 95% of the production allocation model: owner
caches for all 21 ratio-4 and 20 ratio-128 layers plus the pair-1 mirror for
the 11 ratio-4 and 10 ratio-128 late-stage layers. At 262145 tokens this is
approximately 2.7 GiB expected and a roughly 2.5 GiB telemetry floor, rather
than the invalid 12 GiB extrapolation from the synthetic standalone fixture.
F32 remains the default until this production gate passes;
`DS4_CUDA_ATTN_COMP_CACHE=sm75-compact` is the candidate selector and `f32` is
the immediate rollback.

`DIAGNOSTIC_PACK_AUDIT=1` runs a short PP512, one-token F32/compact exact A/B
and checks every packed compact row's embedded status before it can be
consumed. The embedded status now includes a bitwise pack/decode round-trip
check against the already-rounded F32 producer row, in addition to nonfinite
and encoding checks. Both the prefill and first-decode logits must be
byte-identical. On failure it reports either the first divergent output or the
first layer, destination row, and rejection bits; this is a
correctness-localization run, not promotion evidence.

`DIAGNOSTIC_PREFILL_ISOLATION=1` runs PP512 and PP4096 with three exact arms:
F32, compact with the selected-row hybrid consumer, and compact with that
consumer disabled. All compact arms enable the strengthened per-row round-trip
audit. The result records whether each PP4096 prefill output matches F32 and
whether the two compact consumers match each other, separating a persistent
codec failure from a hybrid-consumer failure. Divergence is a diagnostic
result rather than a runner failure; this is not promotion evidence.

The initial `20260905T231236Z` isolation attempt completed the F32 PP512 and
PP4096 frontiers, then the runner rejected its valid two-row CSV because the
shared validator still required the full PP512/4096/32768 inventory. No compact
arm ran, so that archive contains no compact-cache diagnosis. The validator
now keys its expected frontier inventory to the requested diagnostic maximum.

The follow-up `20260905T232957Z` attempt again completed both F32 frontiers,
then exposed a second runner-only validation error: the shared validator
required nonzero steady-decode throughput even for a one-token diagnostic.
One-token output correctly has zero steady tokens and zero steady throughput.
The validator now accepts that exact case while retaining positive first-token
latency/throughput checks; no compact arm ran in that archive either.

`DIAGNOSTIC_DECODE_PROFILE=1` generates two PP512 decode tokens and captures
exactly the second in Nsight Systems for both F32 and compact caches. The first
token primes common cross-device bounce state; excluding it prevents one-time
pinned-host allocator variance from being misclassified as compact-cache GPU
cost. The diagnostic emits a CUDA-kernel summary for each arm and requires the
compact materialized exact-score path. This is component-cost evidence, not
promotion evidence. It rejects any `cudaMalloc` or `cudaMallocHost` inside the
warmed capture and verifies that the two 2 MiB home-device stages were reserved
during graph setup. The harness gives Nsight Systems a private temporary
directory inside the result tree, so it does not require writable system
`/tmp/nvidia` state.

```bash
MODEL="$PWD/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down-SM75-Native-Q8.gguf" \
PROMPT="$PWD/speed-bench/promessi_sposi.txt" \
GPU_DEVICES=0,3,1,2 \
GPU_VRAM=auto \
STAGE_SPLIT=22 \
REQUIRED_POWER_LIMITS_W=250,260,250,250 \
CTX_ALLOC=262145 \
TG_TOKENS=256 \
EXACT_TOKENS=16 \
TELEMETRY_INTERVAL_MS=200 \
CASE_TIMEOUT_SECONDS=1800 \
MIN_COMPACT_VRAM_SAVING_MIB=12000 \
MIN_THROUGHPUT_RATIO=0.95 \
SKIP_BUILD=0 \
CREATE_ARCHIVE=1 \
bash ./speed-bench/cuda-sm75-compact-kv-production-ab.sh
```
