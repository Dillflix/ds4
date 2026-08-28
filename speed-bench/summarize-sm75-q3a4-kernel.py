#!/usr/bin/env python3
"""Summarize balanced production-shaped Q3A4 kernel A/B samples."""

import csv
import statistics
import sys


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


if len(sys.argv) != 4:
    fail("usage: summarize-sm75-q3a4-kernel.py SAMPLES OUTPUT EXPECTED")

source, output, expected_text = sys.argv[1:]
try:
    expected = int(expected_text)
except ValueError:
    fail(f"invalid expected sample count: {expected_text!r}")

samples: dict[str, list[float]] = {"baseline": [], "pair-fused": []}
with open(source, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    required = {"round", "slot", "variant", "timed_repeats",
                "timed_total_ms", "timed_per_call_ms"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        fail(f"incomplete sample columns: {reader.fieldnames}")
    for row in reader:
        variant = row["variant"]
        if variant not in samples:
            fail(f"unexpected variant: {variant!r}")
        try:
            value = float(row["timed_per_call_ms"])
        except ValueError:
            fail(f"invalid timing value: {row}")
        if value <= 0:
            fail(f"non-positive timing value: {row}")
        samples[variant].append(value)

for variant, values in samples.items():
    if len(values) != expected:
        fail(f"{variant} has {len(values)} samples; expected {expected}")


def mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


baseline = samples["baseline"]
candidate = samples["pair-fused"]
baseline_median = statistics.median(baseline)
candidate_median = statistics.median(candidate)
row = {
    "samples_per_variant": expected,
    "baseline_median_ms": f"{baseline_median:.6f}",
    "pair_fused_median_ms": f"{candidate_median:.6f}",
    "baseline_over_pair_fused_speedup_x":
        f"{baseline_median / candidate_median:.6f}",
    "pair_fused_change_pct":
        f"{(candidate_median / baseline_median - 1.0) * 100.0:.3f}",
    "baseline_mad_ms": f"{mad(baseline):.6f}",
    "pair_fused_mad_ms": f"{mad(candidate):.6f}",
    "baseline_min_ms": f"{min(baseline):.6f}",
    "baseline_max_ms": f"{max(baseline):.6f}",
    "pair_fused_min_ms": f"{min(candidate):.6f}",
    "pair_fused_max_ms": f"{max(candidate):.6f}",
}
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)

print(
    "Q3A4 pair-fused candidate: "
    f"{baseline_median:.3f} ms -> {candidate_median:.3f} ms "
    f"({baseline_median / candidate_median:.3f}x)"
)
