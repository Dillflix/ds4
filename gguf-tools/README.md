# DS4 GGUF Tools

This directory contains the offline tools used to build and evaluate DeepSeek
V4 Flash GGUF files for `ds4`.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer.  It implements the DS4 output formats we actually ship:
  `q8_0`, `q8_K`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with `ds4`.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Generate An Imatrix

First regenerate or inspect the calibration dataset:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Then collect activation statistics with the DS4 runtime:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with this DS4 quantizer.  Generic GGUF
tools need DS4-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `ds4 --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the DS4
prefill graph over the full calibration corpus and reads routed-MoE activation
statistics back from the GPU.  The latest published imatrix-generated GGUF files
are available in the antirez Hugging Face repository:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/tree/main
```

## Generate Q2 And Q4 GGUFs

The template GGUF supplies metadata, tokenizer, tensor order, and logical
shapes.  Tensor bytes are regenerated from the Hugging Face safetensors.  Full
generation is intentionally offline and heavy: expect roughly 80-90 GB outputs
for the 2-bit template family and roughly 150-170 GB for the 4-bit routed-expert
family, plus enough free disk for the temporary output.  Use `--dry-run` and
`--compare-tensor` before starting a full write, and use `--overwrite` only when
you really mean to replace an existing GGUF.

Q2 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

Q4 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

True Q8_K routed experts:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --threads 8
```

You can override tensor families:

```sh
--experts iq2_xxs
--routed-w2 q2_k
--attention-proj q8_0
--shared q8_0
--output q8_0
```

For Turing-only deployment, the quantizer can write the routed Q4 tensors in
DS4's size-neutral `sm75_m8n8k32_native_aw_v1` lane order. The GGUF retains
the ordinary `Q4_K` tensor type and adds explicit layout/version metadata, so
the runtime never guesses from a filename or silently reinterprets an
ordinary Q4 file:

```sh
gguf-tools/deepseek4-quantize-cuda \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template FULL-Q4-TEMPLATE.gguf \
  --out FULL-Q4-SM75-NATIVE.gguf \
  --experts q4_k --sm75-native-q4 \
  --quant-backend cuda --quant-gpu-devices 0,2,1,3
```

An existing all-Q4 routed GGUF can be transformed without requantization.
This mode copies every non-routed tensor byte-for-byte, repacks one routed
expert at a time, does not read Hugging Face weights, and does not hash the
model:

```sh
gguf-tools/deepseek4-quantize \
  --repack-sm75-native-q4 FULL-Q4-STANDARD.gguf \
  --out FULL-Q4-SM75-NATIVE.gguf
```

For a full Flash checkpoint, use the lossless four-GPU CUDA repacker rather
than the CPU byte permutation:

```sh
make -C gguf-tools deepseek4-quantize-cuda test-quants-cuda CUDA_ARCH=sm_75
gguf-tools/test-quants-cuda 0,1,2,3
gguf-tools/deepseek4-quantize-cuda \
  --repack-sm75-native-q4 FULL-Q4-STANDARD.gguf \
  --out FULL-Q4-SM75-NATIVE.gguf \
  --quant-backend cuda --quant-gpu-devices 0,1,2,3 --threads 8
```

The CUDA and CPU repackers are required to produce identical bytes. The
production A/B runner writes to a temporary file, verifies the planned final
size, and renames atomically; it removes a size-mismatched interrupted output
before retrying. No model hash is calculated.

Tagged files are deliberately accepted only by the SM75 CUDA path. Untagged
Q4_K files retain the existing CUDA/Metal/CPU/ROCm implementations.

The CUDA routed-MoE path supports the Cartesian product below. Gate and up
must use the same type because they are fused; down is selected independently.

| Role | Supported types |
| --- | --- |
| Gate/up | `iq2_xxs`, `q4_k`, `sm75_q4_32`, `sm75_q3a4` |
| Down | `q2_k`, `q4_k`, `sm75_q4_32` |

The last two rows are architecture-private production formats. They use GGUF
type IDs 42/43 plus the mandatory versioned
`sm75_m8n8k32_q4_32_q3a4_native_aw_v1` metadata tag. DS4 accepts them only as
aligned routed-expert tensors on the CUDA SM75 path. Q3A4 is gate/up-only;
Q4-32 supports gate/up and down. Ordinary Q4_K files and every other backend
retain their existing interpretation and dispatch.

Q3A4 is 108 bytes per K256 block, exactly 3.375 bits/weight. Its native M8
record is 864 bytes. It therefore reads 20.6% fewer weight bytes than Q4-32
(4.25 bpw) and 25% fewer than Q4_K (4.5 bpw), before cache-line and scheduling
effects. For one Flash routed tensor with 256 x 2048 x 4096 weights, those
figures are 864 MiB for Q3A4, 1088 MiB for Q4-32, and 1152 MiB for Q4_K.

Use the guarded production wrapper to make an explicit per-layer allocation:

```sh
Q3A4_LAYERS=3,7,12-18,42 \
QUANTIZE_ONLY=1 \
bash produce-sm75-q4-32-q3a4.sh HF_DIR OUTPUT.gguf
```

`Q3A4_LAYERS=none` makes all gate/up Q4-32; `all` makes all gate/up Q3A4.
Down is always Q4-32. The wrapper rejects the generic routed-type and
per-tensor override variables so a model from this workflow cannot silently
contain IQ2, Q2_K, Q3_K, or Q4_K expert tensors.

The two hybrid recipes can be generated with the existing family overrides:

```sh
# IQ2 gate/up with Q4 down
--experts iq2_xxs --routed-w2 q4_k

# Q4 gate/up with Q2 down
--experts q4_k --routed-w2 q2_k
```

These flags can be appended to either full quantizer command above. Use an
imatrix collected for the same model whenever possible.

## CUDA Quantization

Linux CUDA hosts can build the routed-expert encoder separately from the
portable CPU quantizer:

```sh
make -C gguf-tools deepseek4-quantize-cuda test-quants-cuda CUDA_ARCH=sm_75
gguf-tools/test-quants-cuda 0,2,1,3
```

The CUDA encoder implements imatrix-weighted `iq2_xxs`, `q4_k`, `q2_k`,
`sm75_q4_32`, and `sm75_q3a4`. Each host
worker decodes an expert from the source safetensors and submits its independent
256-value blocks to a persistent CUDA stream. Workers are distributed across
the selected devices, allowing source decode, transfer, and encoding to overlap.
Other tensor types and unweighted blocks retain the reference CPU path.

Select it on a full conversion with:

```sh
gguf-tools/deepseek4-quantize-cuda \
  ... \
  --quant-backend cuda \
  --quant-gpu-devices 0,2,1,3
```

`test-quants-cuda` byte-compares small CPU and CUDA encodes, including the
native Q4-32/Q3A4 records, before a long conversion. The one-step runners build and
runs this check automatically.

For the four-card RTX 8000 target, the repository includes a one-step producer
and benchmark runner for the `iq2_xxs` gate / `iq2_xxs` up / `q4_k` down
recipe:

```sh
bash produce-benchmark-iq2-iq2-q4.sh \
  /models/DeepSeek-V4-Flash-HF \
  /models/DeepSeek-V4-Flash-IQ2-IQ2-Q4.gguf
```

It automatically caches a pinned version of the published routed-MoE imatrix
and a metadata-only prefix of the published Q4 GGUF. It does not download that
GGUF's 165 GB tensor payload: the quantizer regenerates all tensors from the
supplied Hugging Face directory. Set `DS4_TEMPLATE_GGUF`, `DS4_IMATRIX`, or
`DS4_ASSET_CACHE` to override those assets. The original explicit four-path
interface remains available for fully offline runs.

For a cache-constrained Q4-first recipe, provide the already-downloaded full-Q4
GGUF as the calibration model:

```sh
bash produce-benchmark-q4-selective-q2-down.sh \
  /models/DeepSeek-V4-Flash-0731-HF \
  /models/DeepSeek-V4-Flash-Q4KExperts.gguf \
  /models/DeepSeek-V4-Flash-Q4GateUp-SelectiveQ2Down.gguf
```

This allocates the final requested context against the full-Q4 model, records
the actual per-device dense-Q8 FP16 cache deficit, and selects only enough
`ffn_down_exps` tensors for `q2_K` to cover that deficit plus
`CACHE_EXTRA_HEADROOM_MIB` (512 MiB per constrained stage by default). If a
stage does not contain enough routed-down tensors, the selector adds the
minimum number of matched `iq2_xxs` gate/up layer pairs, then avoids any Q2
down conversions made redundant by the final pair's extra reclaim. Gate and
up remain paired so the SM75 routed-IQ2 fast path remains available. The
generated model is then benchmarked with a second cache audit; the command
fails rather than claiming success if any eligible Q8 tensor still falls back
because of cache budget. `CACHE_ALL_Q8=1` is the default, so every dense Q8
weight consulted by an FP16-capable runtime path is eligible, not only DS4's
normal shape/label allow-list. `Q2_DOWN_LAYER_ORDER` controls the layer
preference order and should be replaced with the fixed quality suite's ranking
when that data is available. `IQ2_GATE_UP_LAYER_ORDER` controls the IQ2-pair
preference independently and defaults to `Q2_DOWN_LAYER_ORDER`.

The script builds the CUDA quantizer and an `sm_75` CUDA benchmark, verifies the
routed-format CPU/CUDA byte checks and the
routed-MoE matrix classification test, writes the GGUF through an atomic
temporary file, then benchmarks prefill and generation with
`--cuda-tensor-parallel`. The default device order is `0,2,1,3`, pairing
physical GPUs `(0,1)` and `(2,3)`. Set `GPU_DEVICES` if the CUDA numbering
differs; the script prints the detected topology before loading the model.
Results include CSV throughput data, an SVG chart when `python3` is available,
the quantization plan and logs, and a GPU/git metadata record beside the output
model.

For the four-RTX-8000 production target, the Q4/IQ2-only 256K workflow is:

```sh
bash produce-verify-q4-iq2-full-f16-256k.sh \
  /models/DeepSeek-V4-Flash-0731-HF \
  /mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf
```

No full-Q4 GGUF is an input. Tensor data is generated from the HF checkpoint;
the published 8 MiB metadata prefix and imatrix are fetched and cached
automatically. The runner uses an existing all-IQ2-gate/up + Q4-down model only
as a capacity probe when one is available. Otherwise it generates a temporary
probe from the same HF checkpoint and removes that file after successful final
verification unless `KEEP_GENERATED_CALIBRATION=1`.

The probe is loaded at the actual 262144-token allocation with
`GPU_DEVICES=0,3,1,2`, the 22/21 pipeline split, all dense-Q8 FP16 candidates,
and worst-case all-local T256 placement. After a production warm-up it
measures free VRAM and the active cache reserve on every GPU, then promotes as
many matched gate/up layer pairs to Q4_K as both members of each NVLink stage
can hold. Routed down stays Q4_K in every layer. This calibration direction is
intentional: full Q4 can fail the 256K placement check before a cache audit can
run, while the all-IQ2 starting point fits and gives a directly measurable Q4
promotion budget. Layers 0-2 are pinned to Q4 by the final recipe, so their
fixed 1872 MiB/device promotion cost is charged to stage 0 before any optional
Q4 promotions from layers 3-42. A second run with the final model must report
all 344 production dense-FP16 candidates resident; otherwise the workflow
fails and the unverified candidate is removed without replacing the requested
output path. The
512 MiB/device default safety margin is kept in addition to the runtime's
reported cache reserve. `IQ2_GATE_UP_LAYER_ORDER` controls which layers remain
IQ2; the default `3-42` is deterministic but is not presented as a
quality-sensitivity ranking. Tagged files store only their remaining Q4 routed
tensors in the SM75-native format; IQ2 tensors retain their standard layout and
dispatch.

All-local is a capacity envelope, not a presumed deployment winner: it places
an additional 1.375 GiB on the 22-layer home and 1.3125 GiB on the 21-layer
home relative to all-partner. The resulting model must subsequently pass the
strict all-local/balanced/all-partner screen before any placement is selected.

Useful checks before writing a full model:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template MODEL.gguf \
  --compare-tensor blk.0.attn_q_a.weight
```

`--compare-tensor` regenerates a single tensor and byte-compares it against the
template or `--compare-gguf`.  `--threads N` controls routed-expert workers.

## Convert A DSpark Support Checkpoint

The DSpark Flash checkpoint is published as Hugging Face safetensors and stores
the draft module under `mtp.0`, `mtp.1`, and `mtp.2`.  Before writing a support
GGUF, inspect the official index and verify that every DSpark tensor name is
understood by the converter:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash-DSpark \
  --dspark-manifest > /tmp/dspark-manifest.tsv
```

The manifest reads only `model.safetensors.index.json`; it does not require the
large shard files to be present.  The final summary should report three DSpark
stages and zero unknown DSpark tensors before attempting a full conversion.

To build the support GGUF used by `ds4 --mtp`, run the DSpark support mode.  This
mode writes standalone DSpark metadata plus the packed `mtp.*` tensor payloads;
it does not require a base-model GGUF template:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash-DSpark \
  --dspark-support \
  --out DeepSeek-V4-Flash-DSpark-support.gguf
```

`--dspark-support --dry-run` reads safetensors shard headers to derive exact
GGUF shapes and types, but it does not read tensor payloads.  The DSpark metadata
defaults match the published Flash DSpark config: block size 5, target layers
40,41,42, Markov rank 256, and noise token 128799.  Override them with
`--dspark-block-size`, `--dspark-target-layers`, `--dspark-markov-rank`, and
`--dspark-noise-token-id` if converting a different checkpoint.

Before a full write, regenerate one support tensor and record its checksum:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash-DSpark \
  --dspark-support \
  --compare-tensor mtp.0.main_proj.weight
```

This reads only the payloads needed for that tensor.  Add `--compare-gguf
DeepSeek-V4-Flash-DSpark-support.gguf` to byte-compare against an existing
support GGUF.

## When No Imatrix Is Given

`iq2_xxs` requires an importance vector.  If `--imatrix` is not provided and
the target type requires one, `deepseek4-quantize` computes a synthetic fallback
from the dequantized weight itself:

```text
importance[column] = sum(row[column]^2) over all rows
```

This is a weight-energy heuristic.  It is not as good as measuring real DS4
activations, but it gives the quantizer a stable column weighting and was good
enough for the first working 2-bit GGUFs.

## Quality Testing

See `quality-testing/README.md`.  The short version is:

```sh
python3 gguf-tools/quality-testing/collect_official.py
make -C gguf-tools quality-score
gguf-tools/quality-testing/score_official MODEL.gguf gguf-tools/quality-testing/data/manifest.tsv /tmp/model.tsv 4096
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```
