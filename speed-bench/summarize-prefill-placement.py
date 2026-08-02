#!/usr/bin/env python3
"""Summarize repeated ds4-bench CSVs listed by the deep CUDA audit."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


if len(sys.argv) != 3:
    die("usage: summarize-prefill-placement.py RUNS.tsv SUMMARY.csv")

runs_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
values: dict[tuple[str, str, int], list[float]] = defaultdict(list)

with runs_path.open(newline="", encoding="utf-8") as stream:
    rows = csv.DictReader(stream, delimiter="\t")
    required = {"split", "prompt", "repeat", "csv"}
    if not rows.fieldnames or not required.issubset(rows.fieldnames):
        die(f"{runs_path} is missing columns {sorted(required)}")
    for run in rows:
        result = Path(run["csv"])
        if not result.is_file():
            die(f"benchmark CSV not found: {result}")
        with result.open(newline="", encoding="utf-8") as result_stream:
            for sample in csv.DictReader(result_stream):
                context = int(sample["ctx_tokens"])
                values[(run["prompt"], run["split"], context)].append(
                    float(sample["prefill_tps"])
                )

fields = [
    "prompt",
    "ctx_tokens",
    "split",
    "samples",
    "prefill_tps_median",
    "prefill_tps_min",
    "prefill_tps_max",
    "gain_vs_21_pct",
]
output_rows = []
for (prompt, split, context), samples in sorted(
    values.items(), key=lambda item: (item[0][0], item[0][2], int(item[0][1]))
):
    median = statistics.median(samples)
    baseline_samples = values.get((prompt, "21", context), [])
    baseline = statistics.median(baseline_samples) if baseline_samples else 0.0
    gain = 100.0 * (median / baseline - 1.0) if baseline else 0.0
    output_rows.append(
        {
            "prompt": prompt,
            "ctx_tokens": context,
            "split": split,
            "samples": len(samples),
            "prefill_tps_median": f"{median:.3f}",
            "prefill_tps_min": f"{min(samples):.3f}",
            "prefill_tps_max": f"{max(samples):.3f}",
            "gain_vs_21_pct": f"{gain:.3f}",
        }
    )

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output_rows)

print(",".join(fields))
for row in output_rows:
    print(",".join(str(row[field]) for field in fields))
