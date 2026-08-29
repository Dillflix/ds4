#!/usr/bin/env python3
"""Summarize the four-arm SM75 cross-device feature-isolation run."""

from __future__ import annotations

import array
import csv
import heapq
import math
import pathlib
import statistics
import sys


VARIANTS = ("neither", "partner-only", "rows-only", "both")
COMPARISONS = (
    ("neither", "partner-only", "partner-execution"),
    ("neither", "rows-only", "row-split-no-partner-execution"),
    ("partner-only", "both", "row-split-with-partner-execution"),
)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_floats(path: pathlib.Path) -> array.array[float]:
    size = path.stat().st_size
    values = array.array("f")
    if size == 0 or size % values.itemsize:
        fail(f"invalid float32 logit payload: {path}")
    with path.open("rb") as handle:
        values.fromfile(handle, size // values.itemsize)
    return values


def plan_signature(path_text: str) -> list[tuple[str, ...]] | None:
    path = pathlib.Path(path_text) if path_text else None
    if path is None or not path.is_file():
        return None
    fields = (
        "label",
        "consumer_device",
        "fallback_device",
        "target_device",
        "placement_locked",
        "resident_device",
        "status",
    )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(field not in rows[0] for field in fields):
        fail(f"invalid placement plan CSV: {path}")
    return sorted(tuple(row[field] for field in fields) for row in rows)


def expected_contexts(start: int, maximum: int) -> list[int]:
    result = []
    context = start
    while context <= maximum:
        result.append(context)
        context *= 2
    if not result or result[-1] != maximum:
        fail("CTX_MAX is not reachable from CTX_START by doubling")
    return result


def compare_logits(
    repeat: int,
    comparison: str,
    reference_name: str,
    candidate_name: str,
    reference: dict[str, str],
    candidate: dict[str, str],
) -> list[dict[str, object]]:
    reference_plan = plan_signature(reference["plan"])
    candidate_plan = plan_signature(candidate["plan"])
    plan_equal = (
        "unavailable"
        if reference_plan is None or candidate_plan is None
        else ("yes" if reference_plan == candidate_plan else "no")
    )
    reference_dir = pathlib.Path(reference["logits"])
    candidate_dir = pathlib.Path(candidate["logits"])
    reference_files = sorted(reference_dir.glob("*.logits.f32"))
    candidate_names = sorted(path.name for path in candidate_dir.glob("*.logits.f32"))
    if not reference_files or [path.name for path in reference_files] != candidate_names:
        fail(f"logit inventory mismatch for {comparison} repeat {repeat}")

    result = []
    for reference_path in reference_files:
        candidate_path = candidate_dir / reference_path.name
        a = load_floats(reference_path)
        b = load_floats(candidate_path)
        if len(a) != len(b):
            fail(f"logit length mismatch: {reference_path.name}")
        sum_sq = 0.0
        diff_sq = 0.0
        max_abs = 0.0
        exact = True
        for x, y in zip(a, b):
            x_float = float(x)
            delta = float(y) - x_float
            exact = exact and delta == 0.0
            sum_sq += x_float * x_float
            diff_sq += delta * delta
            max_abs = max(max_abs, abs(delta))
        top1_a = max(range(len(a)), key=a.__getitem__)
        top1_b = max(range(len(b)), key=b.__getitem__)
        top10_a = set(heapq.nlargest(10, range(len(a)), key=a.__getitem__))
        top10_b = set(heapq.nlargest(10, range(len(b)), key=b.__getitem__))
        try:
            context = int(reference_path.name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            fail(f"cannot parse frontier context: {reference_path.name}")
        result.append(
            {
                "repeat": repeat,
                "comparison": comparison,
                "reference": reference_name,
                "candidate": candidate_name,
                "ctx_tokens": context,
                "bit_exact": "yes" if exact else "no",
                "nrmse": f"{math.sqrt(diff_sq / sum_sq):.9g}" if sum_sq else "inf",
                "max_abs": f"{max_abs:.9g}",
                "top1_equal": "yes" if top1_a == top1_b else "no",
                "top10_overlap": len(top10_a & top10_b),
                "reference_planned_partner": reference["planned_partner"],
                "candidate_planned_partner": candidate["planned_partner"],
                "placement_plan_equal": plan_equal,
            }
        )
    return result


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        fail("usage: summarize-sm75-xdev-feature-isolation.py DIR START MAX")
    root = pathlib.Path(argv[1])
    expected = expected_contexts(int(argv[2]), int(argv[3]))
    runs_path = root / "production/runs.csv"
    with runs_path.open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    if not runs:
        fail("production run inventory is empty")

    values: dict[tuple[str, int], list[float]] = {}
    paired: dict[tuple[int, str, int], float] = {}
    indexed: dict[tuple[int, str], dict[str, str]] = {}
    for run in runs:
        repeat = int(run["repeat"])
        variant = run["variant"]
        if variant not in VARIANTS or (repeat, variant) in indexed:
            fail(f"invalid or duplicate run inventory entry: repeat={repeat} variant={variant}")
        indexed[(repeat, variant)] = run
        with pathlib.Path(run["csv"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        contexts = [int(row["ctx_tokens"]) for row in rows]
        if contexts != expected:
            fail(f"frontier mismatch for {variant}: {contexts}")
        for row in rows:
            context = int(row["ctx_tokens"])
            value = float(row["prefill_tps"])
            values.setdefault((variant, context), []).append(value)
            paired[(repeat, variant, context)] = value

    repeats = sorted({repeat for repeat, _ in indexed})
    for repeat in repeats:
        missing = [variant for variant in VARIANTS if (repeat, variant) not in indexed]
        if missing:
            fail(f"repeat {repeat} is missing variants: {','.join(missing)}")

    comparison_rows: list[dict[str, object]] = []
    for repeat in repeats:
        for reference_name, candidate_name, comparison in COMPARISONS:
            comparison_rows.extend(
                compare_logits(
                    repeat,
                    comparison,
                    reference_name,
                    candidate_name,
                    indexed[(repeat, reference_name)],
                    indexed[(repeat, candidate_name)],
                )
            )
    comparison_path = root / "production/logit-comparisons.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_path = root / "production/summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ctx_tokens",
                "neither_tps",
                "partner_only_tps",
                "rows_only_tps",
                "both_tps",
                "partner_vs_neither",
                "rows_vs_neither",
                "both_vs_neither",
                "interaction_ratio",
                "semantic_comparison",
            ]
        )
        for context in expected:
            medians = {
                variant: statistics.median(values[(variant, context)])
                for variant in VARIANTS
            }
            ratios = {
                variant: statistics.median(
                    paired[(repeat, variant, context)]
                    / paired[(repeat, "neither", context)]
                    for repeat in repeats
                )
                for variant in VARIANTS[1:]
            }
            interaction = statistics.median(
                paired[(repeat, "both", context)]
                * paired[(repeat, "neither", context)]
                / (
                    paired[(repeat, "partner-only", context)]
                    * paired[(repeat, "rows-only", context)]
                )
                for repeat in repeats
            )
            writer.writerow(
                [
                    context,
                    *(f"{medians[variant]:.3f}" for variant in VARIANTS),
                    f"{ratios['partner-only']:.6f}",
                    f"{ratios['rows-only']:.6f}",
                    f"{ratios['both']:.6f}",
                    f"{interaction:.6f}",
                    "see-logit-comparisons.csv",
                ]
            )


if __name__ == "__main__":
    main(sys.argv)
