# SM75 Q3/Q4-32 experiment

This document fixes the arithmetic and byte-layout contract for the bounded
SM75 experiment.  These formats are **not** production GGUF types yet.  The
experiment must establish both quality and kernel value before the loader,
quantizer, or production dispatch accepts them.

## Invariants

- One super-block contains 256 weights.
- All multi-byte scalar fields are little-endian.  Every native eight-row tile
  starts at a 16-byte boundary; the byte offsets below are part of the tagged
  layout contract, not implementation suggestions.
- Activations remain DS4 Q8_K.  No additional activation approximation is
  permitted.
- Every integer kernel must reproduce its scalar Q8 dot exactly.  For the
  symmetric Q3/Q4 formats, floating output is
  `d_activation * d_weight * integer_sum` in the production reduction order.
  Affine Q4_K instead keeps `d*scale*dot` and `dmin*minimum*bsum` as distinct
  terms until their final subtraction.
- A native layout is an offline byte permutation only.  It may not expand a
  compact 3-bit payload to resident 4-bit weights.
- Custom Q3-32 and Q4-32 may not reuse the standard GGUF Q3_K or Q4_K type
  identifiers.  Their row sizes and quantization laws differ.

| Format | Canonical bytes / K256 | Native bytes / 8 rows | Local group | MMA per M8xN8xK256 | Exact correction |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4_K control | 144 | 1152 | K32 affine | 16 K32 INT4 | separate `minimum * bsum` accumulator |
| Standard Q3_K | 110 | 880 | K16 symmetric | 16 K16 INT8 | `-4 * bsum` |
| SM75 Q3-32 v1 | 104 | 832 | K32 symmetric | 16 K32 INT4 | `-4 * bsum` |
| SM75 Q4-32 v1 | 136 | 1088 | K32 symmetric | 16 K32 INT4 | none |

## Standard Q3_K, exact K16 execution

The canonical format remains the GGML Q3_K block:

```c
struct block_q3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    fp16 d;
}; // 110 bytes / 256 weights = 3.4375 bpw
```

There are sixteen K16 groups.  A packed six-bit scale code `c` decodes as
`s = c - 32`, and a weight decodes as

```text
w[k] = fp16(d) * s[k/16] * q[k],  q in [-4, 3].
```

The standard low-two-bit/high-mask encoding is preserved exactly.  The SM75
native tile groups eight output rows without changing its 880-byte total.  Its
plane-separated byte layout is:

```text
offset   0,  16 bytes: little-endian d[8]
offset  16,  96 bytes: packed six-bit scales[8][12]
offset 112, 512 bytes: low2[16][32], one byte/four values/lane
offset 624, 256 bytes: high[16][16], one nibble/four values/lane
```

Each lane reconstructs the standard biased code `u = q + 4` into one U8
register and issues one
`mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32`.  The exact signed dot is
`dot(a, u) - 4*sum(a)`; Q8_K already stores the required K16 sums.  Two K16
results are scaled independently and must not be merged before applying their
distinct scales.

Quantization and canonical bytes must match llama.cpp's reference and
importance-matrix paths bit-for-bit.  The native representation is a separate
lossless repack with a canonical/native decode-equality test.

## SM75 Q3-32 v1

Q3-32 changes the quantization group to K32 so its scale boundary matches
Turing `m8n8k32`:

```c
struct block_sm75_q3_32_v1 {
    fp16 d;
    uint8_t scales[6]; // eight biased signed-six-bit codes
    uint8_t qs[96];    // 256 signed three-bit values
}; // 104 bytes / 256 weights = 3.25 bpw
```

For group `j`, `s[j] = decode6(scales, j) - 32`.  The canonical payload is
two planes:

```text
low2(k) = (qs[k/4] >> (2*(k mod 4))) & 3
high1(k) = (qs[64 + k/8] >> (k mod 8)) & 1
u(k) = low2(k) | (high1(k) << 2)
```

The value equation is

```text
w[k] = fp16(d) * s[k/32] * (u(k) - 4),  u in [0, 7].
```

Eight scale codes use this canonical packing:

```text
bytes 0..3: low nibbles for scale j and j+4
byte 4:     high two bits for scales 0..3
byte 5:     high two bits for scales 4..7
```

The 832-byte native eight-row tile is size-neutral and plane-separated:

```text
offset   0,  16 bytes: little-endian d[8]
offset  16,  48 bytes: scales[8][6]
offset  64, 512 bytes: little-endian uint16_t low2[8][32]
offset 576, 256 bytes: uint8_t high[8][32]
```

One lane loads its sixteen low bits and eight high bits and dilates them into
eight unsigned nibbles in one register.  It uses the existing exact Q8
identity:

```text
a_s8 = low_u4(a) + 16 * high_s4(a)
```

Therefore each K32 dot uses one U4xU4 and one S4xU4 MMA, followed by the exact
`-4*sum(a)` correction.  The central measured risk is the 3-bit-to-U4 register
expansion, not Tensor Core throughput.

## SM75 Q4-32 v1

Q4-32 is symmetric and K32-aligned:

```c
struct block_sm75_q4_32_v1 {
    fp16 d;
    uint8_t scales[6]; // eight biased signed-six-bit codes
    uint8_t qs[128];   // 256 signed four-bit values
}; // 136 bytes / 256 weights = 4.25 bpw
```

Its canonical payload stores even `k` in the low nibble and odd `k` in the
high nibble of `qs[k/2]`; each nibble is four-bit two's-complement.  Its
equation is

```text
nibble(k) = (qs[k/2] >> (4 * (k & 1))) & 0xf
w[k]      = fp16(d) * s[k/32] * sign_extend_4(nibble(k)).
```

The native eight-row tile is exactly:

```text
offset   0,   16 bytes: little-endian d[8]
offset  16,   48 bytes: scales[8][6]
offset  64, 1024 bytes: little-endian uint32_t b[8][32]
```

`b` is in exact SM75 B-fragment order.  Weights are consumed directly by
U4xS4 and S4xS4 MMA.  Unlike Q4_K, there is no affine minimum and no Q8_K
block-sum correction.  This simplicity and the 5.6% smaller payload must be
weighed against the quality loss, if any, from symmetric rather than affine
quantization.

A biased `u = q + 8` encoding with an exact `-8*sum(a)` correction is a valid
fallback, but it is not the v1 choice: NVIDIA's SM75 MMA definitions support
both U4xS4 and S4xS4, so two's-complement codes remove that correction and its
dependency chain.  The build gate must confirm those exact SASS operand types
before the result is considered valid.

## Quantizer contract

For Q3-32/Q4-32, use the same deterministic weighted search structure as the
GGML K quantizers:

1. Quantize each K32 group with `make_qx_quants(32, 4, ...)` for Q3 or
   `make_qx_quants(32, 8, ...)` for Q4.  That helper's output is already the
   biased code `L`; Q3 stores `L` directly, while Q4 stores the four-bit
   two's-complement value `(L - 8) & 0xf`.
2. Compress the eight local scales with `make_qx_quants(8, 32, ...)`.  Those
   returned codes are likewise already biased and are stored directly in the
   packed six-bit field.  An encoder API that accepts signed values may add
   32 exactly once, but raw `make_qx_quants` output must never pass through
   that conversion.
3. Store `d` using round-to-nearest FP16, decode that stored value, then
   requantize and clamp the final codes.
4. Store Q3 as the biased unsigned code `q + 4`; this matches standard Q3_K
   and minimizes unpack work.  Store Q4 as four-bit two's-complement.

The CPU oracle and CUDA encoder must produce identical bytes.  CUDA uses one
thread per 256-weight block so the numerical search order stays canonical;
native lane-order packing is a separate second phase.

## GGUF identity and layout tags

The on-disk identity is deliberately unambiguous:

- Portable canonical Q3_K keeps the standard GGUF Q3_K type.  A losslessly
  repacked SM75 tensor uses the symbolic private type
  `DS4_GGML_TYPE_SM75_Q3K_NATIVE_V1` plus
  `ds4.routed_expert.q3.layout = sm75_m8n8k16_native_aw_v1` and layout version
  `1`; it must not retain type Q3_K because a generic reader may ignore unknown
  metadata and misinterpret the reordered payload.  The native type records
  standard-Q3_K quantization semantics, while untagged type Q3_K always retains
  its portable row-major interpretation.
- Q3-32 and Q4-32 use symbolic private types
  `DS4_GGML_TYPE_SM75_Q3_32_V1` and
  `DS4_GGML_TYPE_SM75_Q4_32_V1`; numeric GGML enum values are intentionally
  not allocated by this experiment.  Their
  native tags are `sm75_m8n8k32_q3_native_aw_v1` and
  `sm75_m8n8k32_q4s_native_aw_v1` respectively.
- DS4 rejects a private type/layout pair it does not recognize.  Generic GGUF
  readers will encounter an unknown private tensor type rather than silently
  reinterpreting native bytes as Q3_K or Q4_K.

Deferring the numeric IDs is intentional: there is no safe vendor-reserved
range in the upstream GGML enum.  The production patch must allocate the IDs
against the exact upstream revision that DS4 vendors and cover collision and
unknown-tag refusal in tests.

## Experiment gates

The first harness compares current native Q4_K, exact native-layout Q3_K,
Q3-32, and Q4-32 at identical M16xN8xK256 shapes.  The Q4_K control retains
separate integer accumulation chains for its scale and affine-minimum terms,
matching the register/dependency requirement imposed
by distinct FP16 `d` and `dmin` values in production.  This first harness
treats all FP16 headers as opaque and compares the two integer chains by their
difference only after both have completed; it does not claim floating Q4_K
bit equivalence.

Required evidence:

- edge/adversarial and seeded-random accumulator equality;
- canonical/native round-trip equality and exact byte-size assertions;
- SASS validation for K16 S8 and K32 S4 MMA, with no accidental DP4A path;
- hot-cache and L2-exceeding streamed-weight timing;
- register, shared-memory, occupancy, tensor-pipe, memory-traffic, and stall
  measurements under Nsight Compute;
- no sanitizer errors.

In this harness, `exact_status=ok` means exact integer-accumulator equality
against an independently written scalar decoder plus lossless native-layout
round trips.  A fixed raw standard-Q3_K full-block digest independently gates
the canonical hmask/low-plane mapping, but this still does not establish
quantizer byte parity, FP16 scaling and rounding parity, model logits, or
end-to-end quality.  Before production, standard Q3_K additionally requires
bit-for-bit upstream quantizer output tests; Q3-32/Q4-32 require their own
CPU/CUDA encoder parity and real-weight quality suite.

Q3 only advances to a routed-MoE production-shaped histogram harness if its
per-tile cost is at most 1.25x current native Q4.  No full GGUF is generated
until a small real-weight reconstruction/quality pass also succeeds.

## Primary references

- [llama.cpp `ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
  for standard Q3_K packing/dequantization and the biased output contract of
  `make_qx_quants`.
- [NVIDIA CUTLASS SM75 MMA definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h)
  for the legal K16 S8xU8 and K32 U4/S4 operand combinations.
