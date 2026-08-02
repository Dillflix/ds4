## Benchmarking

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
`RUN_NCU=0` when only the placement and tile evidence is needed.
