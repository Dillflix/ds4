#!/usr/bin/env python3
"""Summarize paired SM75 production prefill microbatch runs."""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics


VARIANTS = ("mb512", "mb1024")


def read_benchmark(path: pathlib.Path) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty benchmark CSV: {path}")
    result: dict[int, float] = {}
    for row in rows:
        ctx = int(row["ctx_tokens"])
        tps = float(row["prefill_tps"])
        if ctx in result or tps <= 0.0:
            raise ValueError(f"invalid or duplicate frontier {ctx} in {path}")
        result[ctx] = tps
    return result


def summarize(runs_path: pathlib.Path) -> list[dict[str, str]]:
    with runs_path.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    if not runs:
        raise ValueError("runs inventory is empty")

    samples: dict[tuple[str, int], list[float]] = {}
    paired: dict[tuple[int, str, int], float] = {}
    repeats = sorted({int(run["repeat"]) for run in runs})
    expected_repeats = list(range(1, max(repeats) + 1))
    if repeats != expected_repeats:
        raise ValueError(f"non-contiguous repeats: {repeats}")

    inventory: set[tuple[int, str]] = set()
    contexts: set[int] | None = None
    for run in runs:
        repeat = int(run["repeat"])
        variant = run["variant"]
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        key = (repeat, variant)
        if key in inventory:
            raise ValueError(f"duplicate run: repeat={repeat} variant={variant}")
        inventory.add(key)
        values = read_benchmark(pathlib.Path(run["csv"]))
        run_contexts = set(values)
        if contexts is None:
            contexts = run_contexts
        elif contexts != run_contexts:
            raise ValueError(
                f"frontier inventory differs for repeat={repeat} variant={variant}"
            )
        for ctx, tps in values.items():
            samples.setdefault((variant, ctx), []).append(tps)
            paired[(repeat, variant, ctx)] = tps

    expected = {(repeat, variant) for repeat in repeats for variant in VARIANTS}
    if inventory != expected:
        missing = sorted(expected - inventory)
        extra = sorted(inventory - expected)
        raise ValueError(f"incomplete run inventory: missing={missing} extra={extra}")
    assert contexts is not None

    output: list[dict[str, str]] = []
    for ctx in sorted(contexts):
        ratios = [
            paired[(repeat, "mb1024", ctx)] / paired[(repeat, "mb512", ctx)]
            for repeat in repeats
        ]
        speedup = statistics.median(ratios)
        output.append(
            {
                "ctx_tokens": str(ctx),
                "mb512_median_tps": f"{statistics.median(samples[('mb512', ctx)]):.3f}",
                "mb1024_median_tps": f"{statistics.median(samples[('mb1024', ctx)]):.3f}",
                "paired_median_speedup": f"{speedup:.6f}",
                "change_pct": f"{(speedup - 1.0) * 100.0:.3f}",
                "paired_speedup_sd": f"{statistics.pstdev(ratios):.6f}",
                "logits": "bit-exact",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_csv", type=pathlib.Path)
    parser.add_argument("summary_csv", type=pathlib.Path)
    args = parser.parse_args()

    rows = summarize(args.runs_csv)
    fields = list(rows[0])
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(",".join(fields))
    for row in rows:
        print(",".join(row[field] for field in fields))


if __name__ == "__main__":
    main()
