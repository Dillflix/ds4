#!/usr/bin/env python3
"""Summarize the controlled full-Q4 placement/cache benchmark."""

from __future__ import annotations

import csv
import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import NoReturn


VARIANTS = ("baseline-22x21", "split-21x22", "swap-22x21")


def die(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "repeat", "slot", "variant", "split", "gpu_devices", "csv", "log",
        "cache_before", "cache_after",
    }
    if not rows or not required.issubset(rows[0]):
        die(f"invalid run manifest: {path}")
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_benchmark(path: Path) -> list[dict[str, float | int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        die(f"empty benchmark: {path}")
    result = []
    for row in rows:
        try:
            ctx = int(row["ctx_tokens"])
            tokens = int(row["prefill_tokens"])
            tps = float(row["prefill_tps"])
            gen = int(row["gen_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            die(f"malformed benchmark row in {path}: {exc}")
        if ctx <= 0 or tokens <= 0 or tps <= 0.0 or gen != 0:
            die(f"invalid prefill-only benchmark row in {path}: {row}")
        result.append({"ctx": ctx, "tokens": tokens, "tps": tps})
    return result


def read_cache(path: Path) -> list[dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "physical_device", "weight_offset", "weight_bytes", "in_dim",
        "out_dim", "fp16_bytes",
    }
    if not rows or not expected.issubset(rows[0]):
        die(f"invalid Q8 cache state: {path}")
    parsed = []
    for row in rows:
        try:
            parsed.append({field: int(row[field]) for field in expected})
        except (KeyError, TypeError, ValueError) as exc:
            die(f"malformed Q8 cache state in {path}: {exc}")
    return parsed


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 4:
        die("usage: summarize-q4-layout-cache-ab.py RUNS.tsv OUTPUT_DIR REPEATS")
    manifest = Path(sys.argv[1])
    output = Path(sys.argv[2])
    try:
        repeats = int(sys.argv[3])
    except ValueError:
        die("REPEATS must be an integer")
    rows = read_tsv(manifest)
    if repeats <= 0 or len(rows) != repeats * len(VARIANTS):
        die("run-manifest cardinality does not match REPEATS")

    expected_repeats = set(range(1, repeats + 1))
    seen: set[tuple[int, str]] = set()
    series: dict[tuple[int, str], list[dict[str, float | int]]] = {}
    cache_by_variant: dict[str, list[dict[str, int]]] = {}
    metadata: dict[str, tuple[int, str]] = {}
    for row in rows:
        repeat = int(row["repeat"])
        slot = int(row["slot"])
        variant = row["variant"]
        if repeat not in expected_repeats or variant not in VARIANTS:
            die(f"unexpected run identity: repeat={repeat} variant={variant}")
        if not 1 <= slot <= len(VARIANTS):
            die(f"invalid sample slot: {slot}")
        key = (repeat, variant)
        if key in seen:
            die(f"duplicate run: repeat={repeat} variant={variant}")
        seen.add(key)
        before = Path(row["cache_before"])
        after = Path(row["cache_after"])
        if digest(before) != digest(after):
            die(f"Q8 cache changed during timing: repeat={repeat} variant={variant}")
        cache = read_cache(before)
        if variant in cache_by_variant:
            if cache != cache_by_variant[variant]:
                die(f"Q8 cache state changed across repeats for {variant}")
        else:
            cache_by_variant[variant] = cache
        metadata[variant] = (int(row["split"]), row["gpu_devices"])
        series[key] = read_benchmark(Path(row["csv"]))

    if seen != {(repeat, variant) for repeat in expected_repeats for variant in VARIANTS}:
        die("run manifest is incomplete")
    for repeat in expected_repeats:
        slots = {int(row["slot"]) for row in rows if int(row["repeat"]) == repeat}
        if slots != {1, 2, 3}:
            die(f"sample slots are incomplete for repeat {repeat}")

    baseline_contexts = [int(row["ctx"]) for row in series[(1, VARIANTS[0])]]
    for key, values in series.items():
        contexts = [int(row["ctx"]) for row in values]
        if contexts != baseline_contexts:
            die(f"context frontier mismatch for {key}")

    paired: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    speedups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for repeat in sorted(expected_repeats):
        baseline = {
            int(row["ctx"]): float(row["tps"])
            for row in series[(repeat, VARIANTS[0])]
        }
        for variant in VARIANTS:
            for row in series[(repeat, variant)]:
                ctx = int(row["ctx"])
                tps = float(row["tps"])
                ratio = tps / baseline[ctx]
                grouped[(variant, ctx)].append(tps)
                speedups[(variant, ctx)].append(ratio)
                paired.append({
                    "repeat": repeat,
                    "ctx_tokens": ctx,
                    "variant": variant,
                    "prefill_tps": f"{tps:.6f}",
                    "baseline_tps": f"{baseline[ctx]:.6f}",
                    "speedup_vs_baseline": f"{ratio:.9f}",
                })

    frontier_rows: list[dict[str, object]] = []
    for ctx in baseline_contexts:
        for variant in VARIANTS:
            values = grouped[(variant, ctx)]
            ratios = speedups[(variant, ctx)]
            split, devices = metadata[variant]
            frontier_rows.append({
                "ctx_tokens": ctx,
                "variant": variant,
                "split": f"{split}/{43 - split}",
                "gpu_devices": devices,
                "samples": len(values),
                "median_prefill_tps": f"{statistics.median(values):.6f}",
                "min_prefill_tps": f"{min(values):.6f}",
                "max_prefill_tps": f"{max(values):.6f}",
                "median_paired_speedup_vs_baseline": f"{statistics.median(ratios):.9f}",
                "median_paired_change_pct": f"{(statistics.median(ratios) - 1.0) * 100.0:.6f}",
            })

    cache_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        split, devices = metadata[variant]
        home_devices = [int(value) for value in devices.split(",")[:2]]
        by_device: dict[int, list[dict[str, int]]] = defaultdict(list)
        for row in cache_by_variant[variant]:
            by_device[row["physical_device"]].append(row)
        for stage, device in enumerate(home_devices):
            entries = by_device.get(device, [])
            cache_rows.append({
                "variant": variant,
                "split": f"{split}/{43 - split}",
                "gpu_devices": devices,
                "stage": stage,
                "home_physical_device": device,
                "cached_weight_slices": len(entries),
                "cached_fp16_bytes": sum(row["fp16_bytes"] for row in entries),
                "cached_fp16_gib": f"{sum(row['fp16_bytes'] for row in entries) / 2**30:.6f}",
            })
        unexpected = set(by_device) - set(home_devices)
        if unexpected:
            die(f"Q8 F16 cache unexpectedly resides on partner devices for {variant}: {unexpected}")

    write_csv(
        output / "paired-samples.csv",
        ["repeat", "ctx_tokens", "variant", "prefill_tps", "baseline_tps", "speedup_vs_baseline"],
        paired,
    )
    write_csv(
        output / "frontier-summary.csv",
        [
            "ctx_tokens", "variant", "split", "gpu_devices", "samples",
            "median_prefill_tps", "min_prefill_tps", "max_prefill_tps",
            "median_paired_speedup_vs_baseline", "median_paired_change_pct",
        ],
        frontier_rows,
    )
    write_csv(
        output / "cache-summary.csv",
        [
            "variant", "split", "gpu_devices", "stage", "home_physical_device",
            "cached_weight_slices", "cached_fp16_bytes", "cached_fp16_gib",
        ],
        cache_rows,
    )

    lines = [
        f"repeats={repeats}",
        f"frontiers={len(baseline_contexts)}",
        "measurement_grade=" + ("position-balanced" if repeats % 3 == 0 else "unbalanced"),
    ]
    for row in cache_rows:
        lines.append(
            f"cache variant={row['variant']} stage={row['stage']} "
            f"device={row['home_physical_device']} slices={row['cached_weight_slices']} "
            f"fp16_gib={row['cached_fp16_gib']}"
        )
    for row in frontier_rows:
        lines.append(
            f"performance ctx={row['ctx_tokens']} variant={row['variant']} "
            f"median_tps={row['median_prefill_tps']} "
            f"paired_change_pct={row['median_paired_change_pct']}"
        )
    text = "\n".join(lines) + "\n"
    (output / "overall-summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
