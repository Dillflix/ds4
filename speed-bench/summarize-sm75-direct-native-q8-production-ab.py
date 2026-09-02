#!/usr/bin/env python3
"""Summarize the dual-model direct-native Q8 production A/B."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path


LAYOUTS = ("mixed15", "all43")
VARIANTS = ("canonical", "direct")
CONTEXTS = (512, 4096, 32768)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_csv(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        values = {
            int(row["ctx_tokens"]):
                (float(row["prefill_tps"]), float(row["gen_steady_tps"]))
            for row in rows
        }
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid benchmark CSV {path}: {exc}")
    if tuple(sorted(values)) != CONTEXTS:
        fail(f"unexpected contexts in {path}: {sorted(values)}")
    if any(not math.isfinite(value) or value <= 0.0
           for pair in values.values() for value in pair):
        fail(f"non-positive or non-finite throughput in {path}")
    return values


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: summarize-sm75-direct-native-q8-production-ab.py RUNS.tsv OUTPUT_DIR")
    runs_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    with runs_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        fail("run manifest is empty")

    runs: dict[tuple[str, int, str], dict[int, tuple[float, float]]] = {}
    repeats: dict[str, set[int]] = {layout: set() for layout in LAYOUTS}
    for row in rows:
        try:
            layout = row["model_layout"]
            repeat = int(row["repeat"])
            variant = row["variant"]
            path = Path(row["csv"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid run manifest row: {exc}")
        if layout not in LAYOUTS or variant not in VARIANTS or repeat <= 0:
            fail(f"invalid run identity: {layout} repeat={repeat} variant={variant}")
        key = (layout, repeat, variant)
        if key in runs:
            fail(f"duplicate run: {key}")
        runs[key] = load_csv(path)
        repeats[layout].add(repeat)

    samples: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for layout in LAYOUTS:
        layout_repeats = sorted(repeats[layout])
        if not layout_repeats or layout_repeats != list(
                range(1, layout_repeats[-1] + 1)):
            fail(f"incomplete repeats for {layout}: {layout_repeats}")
        for repeat in layout_repeats:
            if any((layout, repeat, variant) not in runs
                   for variant in VARIANTS):
                fail(f"unpaired {layout} repeat {repeat}")
            for context in CONTEXTS:
                c_prefill, c_decode = runs[(layout, repeat, "canonical")][context]
                d_prefill, d_decode = runs[(layout, repeat, "direct")][context]
                samples.append({
                    "model_layout": layout,
                    "repeat": repeat,
                    "context": context,
                    "canonical_prefill_tps": c_prefill,
                    "direct_prefill_tps": d_prefill,
                    "prefill_speedup": d_prefill / c_prefill,
                    "canonical_decode_tps": c_decode,
                    "direct_decode_tps": d_decode,
                    "decode_speedup": d_decode / c_decode,
                })
        for context in CONTEXTS:
            group = [row for row in samples
                     if row["model_layout"] == layout and row["context"] == context]
            prefill_ratios = [float(row["prefill_speedup"]) for row in group]
            decode_ratios = [float(row["decode_speedup"]) for row in group]
            summary.append({
                "model_layout": layout,
                "context": context,
                "canonical_prefill_tps": statistics.median(
                    float(row["canonical_prefill_tps"]) for row in group),
                "direct_prefill_tps": statistics.median(
                    float(row["direct_prefill_tps"]) for row in group),
                "paired_prefill_speedup": statistics.median(prefill_ratios),
                "prefill_speedup_sd": (statistics.stdev(prefill_ratios)
                                       if len(prefill_ratios) > 1 else 0.0),
                "canonical_decode_tps": statistics.median(
                    float(row["canonical_decode_tps"]) for row in group),
                "direct_decode_tps": statistics.median(
                    float(row["direct_decode_tps"]) for row in group),
                "paired_decode_speedup": statistics.median(decode_ratios),
                "decode_speedup_sd": (statistics.stdev(decode_ratios)
                                      if len(decode_ratios) > 1 else 0.0),
                "samples": len(group),
            })

    def write_csv(name: str, records: list[dict[str, object]]) -> None:
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("paired-samples.csv", samples)
    write_csv("summary.csv", summary)
    lines = [
        "# SM75 direct native-Q8 production A/B",
        "",
        "The direct arm changes only the Q8_K producer at native routed-MoE input and intermediate boundaries; both arms use the same native consumers and stable four-GPU topology.",
        "",
        "| Model | Context | Canonical prefill tok/s | Direct prefill tok/s | Prefill speedup | Canonical decode tok/s | Direct decode tok/s | Decode speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model_layout']} | {row['context']} | "
            f"{float(row['canonical_prefill_tps']):.3f} | "
            f"{float(row['direct_prefill_tps']):.3f} | "
            f"{float(row['paired_prefill_speedup']):.6f}x | "
            f"{float(row['canonical_decode_tps']):.3f} | "
            f"{float(row['direct_decode_tps']):.3f} | "
            f"{float(row['paired_decode_speedup']):.6f}x |"
        )
    lines.extend([
        "",
        "Byte-exact decode-logit equality and exclusive dispatch are hard gates in the runner, not inferred from these timings.",
    ])
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
