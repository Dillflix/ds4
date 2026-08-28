#!/usr/bin/env python3
"""Validate and summarize the canonical SM75 production decode matrix."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


CANONICAL_CASES = (
    ("pp512-tg256", 512, 256),
    ("pp512-tg512", 512, 512),
    ("pp2048-tg256", 2048, 256),
    ("pp2048-tg512", 2048, 512),
    ("pp4096-tg256", 4096, 256),
    ("pp4096-tg512", 4096, 512),
    ("pp32768-tg256", 32768, 256),
    ("pp32768-tg512", 32768, 512),
)
CASE_SHAPES = {case: (pp, tg) for case, pp, tg in CANONICAL_CASES}


class ValidationError(RuntimeError):
    pass


def positive_float(row: dict[str, str], key: str, source: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValidationError(f"{source}: invalid {key}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{source}: {key} must be finite and positive")
    return value


def integer(row: dict[str, str], key: str, source: Path) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as exc:
        raise ValidationError(f"{source}: invalid {key}") from exc


def validate_case_csv(source: Path, case_id: str, pp: int, tg: int) -> dict[str, float | int | str]:
    if case_id not in CASE_SHAPES:
        raise ValidationError(f"unknown case: {case_id}")
    if CASE_SHAPES[case_id] != (pp, tg):
        raise ValidationError(f"{case_id}: declared shape {pp}/{tg} is not canonical")
    if not source.is_file() or source.stat().st_size == 0:
        raise ValidationError(f"missing or empty benchmark CSV: {source}")
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValidationError(f"{source}: expected exactly one benchmark row, found {len(rows)}")
    row = rows[0]
    expected_ints = {
        "ctx_tokens": pp,
        "prefill_tokens": pp,
        "gen_tokens": tg,
        "gen_steady_tokens": tg - 1,
    }
    for key, expected in expected_ints.items():
        actual = integer(row, key, source)
        if actual != expected:
            raise ValidationError(f"{source}: {key}={actual}, expected {expected}")
    result: dict[str, float | int | str] = {
        "case_id": case_id,
        "pp_tokens": pp,
        "tg_tokens": tg,
    }
    for key in ("prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps"):
        result[key] = positive_float(row, key, source)
    return result


def population_cv_pct(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return 100.0 * statistics.pstdev(values) / mean


def summarize(runs_path: Path, output_dir: Path) -> None:
    if not runs_path.is_file():
        raise ValidationError(f"runs inventory not found: {runs_path}")
    with runs_path.open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    if not runs:
        raise ValidationError("runs inventory is empty")

    samples: list[dict[str, float | int | str]] = []
    seen: set[tuple[int, str]] = set()
    cases_by_repeat: dict[int, set[str]] = {}
    slots_by_repeat: dict[int, set[int]] = {}
    for run in runs:
        try:
            repeat = int(run["repeat"])
            slot = int(run["slot"])
            case_id = run["case_id"]
            pp = int(run["pp_tokens"])
            tg = int(run["tg_tokens"])
            ctx_alloc = int(run["ctx_alloc"])
            source = Path(run["csv"])
        except (KeyError, ValueError) as exc:
            raise ValidationError("runs inventory contains an invalid row") from exc
        if repeat < 1 or slot < 1:
            raise ValidationError("repeat and slot must be positive")
        if ctx_alloc != pp + tg + 1:
            raise ValidationError(f"{case_id}: ctx_alloc={ctx_alloc}, expected {pp + tg + 1}")
        key = (repeat, case_id)
        if key in seen:
            raise ValidationError(f"duplicate run: repeat {repeat}, case {case_id}")
        seen.add(key)
        cases_by_repeat.setdefault(repeat, set()).add(case_id)
        if slot in slots_by_repeat.setdefault(repeat, set()):
            raise ValidationError(f"duplicate slot {slot} in repeat {repeat}")
        slots_by_repeat[repeat].add(slot)
        sample = validate_case_csv(source, case_id, pp, tg)
        sample.update({"repeat": repeat, "slot": slot, "ctx_alloc": ctx_alloc})
        samples.append(sample)

    first_cases = cases_by_repeat[min(cases_by_repeat)]
    for repeat, cases in cases_by_repeat.items():
        if cases != first_cases:
            raise ValidationError(f"repeat {repeat} does not contain the same case set")
        expected_slots = set(range(1, len(cases) + 1))
        if slots_by_repeat[repeat] != expected_slots:
            raise ValidationError(f"repeat {repeat} slots are not contiguous")

    canonical_order = {case: index for index, (case, _, _) in enumerate(CANONICAL_CASES)}
    samples.sort(key=lambda row: (canonical_order[str(row["case_id"])], int(row["repeat"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_fields = (
        "case_id", "pp_tokens", "tg_tokens", "repeat", "slot", "ctx_alloc",
        "prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps",
    )
    with (output_dir / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(samples)

    summary_rows: list[dict[str, float | int | str]] = []
    for case_id, pp, tg in CANONICAL_CASES:
        group = [row for row in samples if row["case_id"] == case_id]
        if not group:
            continue
        prefill = [float(row["prefill_tps"]) for row in group]
        total = [float(row["gen_tps"]) for row in group]
        first = [float(row["gen_first_ms"]) for row in group]
        steady = [float(row["gen_steady_tps"]) for row in group]
        steady_median = statistics.median(steady)
        summary_rows.append({
            "case_id": case_id,
            "pp_tokens": pp,
            "tg_tokens": tg,
            "median_prefill_tps": statistics.median(prefill),
            "median_gen_tps": statistics.median(total),
            "median_gen_first_ms": statistics.median(first),
            "median_gen_steady_tps": steady_median,
            "median_steady_ms_per_token": 1000.0 / steady_median,
            "gen_tps_cv_pct": population_cv_pct(total),
            "steady_tps_cv_pct": population_cv_pct(steady),
            "samples": len(group),
        })

    summary_fields = (
        "case_id", "pp_tokens", "tg_tokens", "median_prefill_tps",
        "median_gen_tps", "median_gen_first_ms", "median_gen_steady_tps",
        "median_steady_ms_per_token", "gen_tps_cv_pct", "steady_tps_cv_pct",
        "samples",
    )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })

    with (output_dir / "summary.md").open("w") as handle:
        handle.write("# SM75 production decode baseline\n\n")
        handle.write("| Case | PP | TG | Prefill tok/s | Decode tok/s | First token ms | Steady tok/s | Steady ms/token | Samples |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                f"| {row['case_id']} | {row['pp_tokens']} | {row['tg_tokens']} | "
                f"{row['median_prefill_tps']:.2f} | {row['median_gen_tps']:.2f} | "
                f"{row['median_gen_first_ms']:.3f} | {row['median_gen_steady_tps']:.2f} | "
                f"{row['median_steady_ms_per_token']:.3f} | {row['samples']} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", type=Path)
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--validate-case", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--pp", type=int)
    parser.add_argument("--tg", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.validate_case is not None:
            if args.case_id is None or args.pp is None or args.tg is None:
                raise ValidationError("--validate-case requires --case-id, --pp, and --tg")
            validate_case_csv(args.validate_case, args.case_id, args.pp, args.tg)
        else:
            if args.runs is None or args.output_dir is None:
                raise ValidationError("runs inventory and output directory are required")
            summarize(args.runs, args.output_dir)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
