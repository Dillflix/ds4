#!/usr/bin/env python3
"""Merge SM75 wide-CTA resources and summarize balanced timing samples."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


GATE_WIDTHS = (256, 384, 512)
DOWN_WIDTHS = (256, 384, 512, 640)
ROW_SPANS = (512, 1024, 2048)

GATE_KERNEL = "moe_gate_up_mid_q4K_tile8_mma_kernel"
DOWN_KERNEL = "moe_down_q4K_tile16_mma_sm75_kernel"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge runtime, PTXAS, and SASS evidence for the production SM75 "
            "wide-CTA specializations and optionally summarize balanced timing."
        )
    )
    parser.add_argument("--runtime-resources", required=True, type=Path)
    parser.add_argument("--ptxas-log", required=True, type=Path)
    parser.add_argument("--sass", required=True, type=Path)
    parser.add_argument("--resource-output", required=True, type=Path)
    parser.add_argument("--timing-samples", type=Path)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--position-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--cxxfilt", default="c++filt")
    args = parser.parse_args()
    timing_paths = (args.timing_samples, args.timing_output, args.position_output)
    if any(timing_paths) and not all(timing_paths):
        parser.error(
            "--timing-samples, --timing-output, and --position-output "
            "must be provided together"
        )
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args


def read_required(path: Path, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        fail(f"could not read {label} {path}: {exc}")
    if not text:
        fail(f"{label} is empty: {path}")
    return text


def integer(row: dict[str, str], field: str, context: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError):
        fail(f"invalid {field} in {context}: {row!r}")
    return value


def finite_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        fail(f"invalid {field} in {context}: {row!r}")
    if not math.isfinite(value):
        fail(f"non-finite {field} in {context}: {row!r}")
    return value


def demangle(symbols: list[str], cxxfilt: str) -> dict[str, str]:
    if not symbols:
        return {}
    try:
        completed = subprocess.run(
            [cxxfilt],
            input="\n".join(symbols) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"could not demangle CUDA symbols with {cxxfilt}: {exc}")
    lines = completed.stdout.splitlines()
    if len(lines) != len(symbols):
        fail(
            f"demangler returned {len(lines)} records for {len(symbols)} symbols"
        )
    return dict(zip(symbols, lines))


def parse_specialization(demangled: str) -> tuple[str, int, int] | None:
    for family, kernel in (("gate", GATE_KERNEL), ("down", DOWN_KERNEL)):
        marker = kernel + "<"
        begin = demangled.find(marker)
        if begin < 0:
            continue
        begin += len(marker)
        end = demangled.find(">", begin)
        if end < 0:
            return None
        arguments = demangled[begin:end]
        numbers = [int(value) for value in re.findall(r"\b(\d+)u?\b", arguments)]
        if "true" not in arguments or len(numbers) < 2:
            return None
        return family, numbers[0], numbers[-1]
    return None


def parse_ptxas(text: str) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.search(r"Function properties for\s+(\S+)", line)
        if match:
            current = match.group(1)
            records.setdefault(current, {})
            continue
        if current is None:
            continue
        match = re.search(
            r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
            r"(\d+) bytes spill loads",
            line,
        )
        if match:
            stack, stores, loads = map(int, match.groups())
            records[current].update(
                stack_frame_bytes=stack,
                spill_store_bytes=stores,
                spill_load_bytes=loads,
            )
            continue
        match = re.search(r"Used\s+(\d+) registers", line)
        if match:
            records[current]["raw_registers_per_thread"] = int(match.group(1))
    return records


def parse_sass(text: str) -> dict[str, dict[str, int]]:
    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for line in text.splitlines():
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match:
            current = match.group(1)
        if current is not None:
            sections[current].append(line)
    records: dict[str, dict[str, int]] = {}
    for symbol, lines in sections.items():
        body = "\n".join(lines)
        records[symbol] = {
            "ldl_instructions": len(re.findall(r"\bLDL(?:\.|\s)", body)),
            "stl_instructions": len(re.findall(r"\bSTL(?:\.|\s)", body)),
            "imma_instructions": len(re.findall(r"\bIMMA(?:\.|\s)", body)),
        }
    return records


def load_runtime_resources(path: Path) -> list[dict[str, str]]:
    required = {
        "family",
        "row_span",
        "cta_threads",
        "warps",
        "raw_registers_per_thread",
        "allocated_registers_per_thread",
        "allocated_registers_per_warp",
        "allocated_registers_per_block",
        "max_threads_per_block",
        "local_bytes_per_thread",
        "static_shared_bytes",
        "allocated_shared_bytes",
        "active_blocks_per_sm",
        "active_warps_per_sm",
        "occupancy_pct",
        "binary_version",
        "ptx_version",
    }
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                missing = sorted(required - set(reader.fieldnames or ()))
                fail(f"runtime resource CSV is missing columns: {missing}")
            rows = list(reader)
    except OSError as exc:
        fail(f"could not read runtime resource CSV {path}: {exc}")
    expected = {
        (family, span, width)
        for family, widths in (("gate", GATE_WIDTHS), ("down", DOWN_WIDTHS))
        for span in ROW_SPANS
        for width in widths
    }
    found: set[tuple[str, int, int]] = set()
    for row in rows:
        family = row.get("family", "")
        span = integer(row, "row_span", "runtime resources")
        width = integer(row, "cta_threads", "runtime resources")
        key = (family, span, width)
        if key in found:
            fail(f"duplicate runtime resource row: {key}")
        found.add(key)
    if found != expected:
        fail(
            "runtime resource matrix differs from the required matrix; "
            f"missing={sorted(expected - found)} unexpected={sorted(found - expected)}"
        )
    return rows


def merge_resources(
    runtime_rows: list[dict[str, str]],
    ptxas: dict[str, dict[str, int]],
    sass: dict[str, dict[str, int]],
    cxxfilt: str,
    resource_output: Path,
) -> list[dict[str, object]]:
    symbols = sorted(set(ptxas) | set(sass))
    demangled = demangle(symbols, cxxfilt)
    compiled: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for symbol in symbols:
        key = parse_specialization(demangled[symbol])
        if key is not None:
            compiled[key].append(symbol)

    merged: list[dict[str, object]] = []
    failures: list[str] = []
    for runtime in runtime_rows:
        family = runtime["family"]
        span = integer(runtime, "row_span", "runtime resources")
        width = integer(runtime, "cta_threads", "runtime resources")
        key = (family, span, width)
        candidates = [
            symbol
            for symbol in compiled.get(key, ())
            if symbol in ptxas and symbol in sass
        ]
        if len(candidates) != 1:
            failures.append(
                f"{key}: expected one PTXAS+SASS specialization, found "
                f"{len(candidates)} ({candidates})"
            )
            continue
        symbol = candidates[0]
        ptxas_row = ptxas[symbol]
        sass_row = sass[symbol]
        required_ptxas = {
            "raw_registers_per_thread",
            "stack_frame_bytes",
            "spill_store_bytes",
            "spill_load_bytes",
        }
        if not required_ptxas.issubset(ptxas_row):
            failures.append(f"{key}: incomplete PTXAS record for {symbol}")
            continue
        runtime_regs = integer(runtime, "raw_registers_per_thread", str(key))
        fatal_reasons: list[str] = []
        observations: list[str] = []
        if runtime_regs != ptxas_row["raw_registers_per_thread"]:
            fatal_reasons.append(
                f"register-mismatch(runtime={runtime_regs},ptxas="
                f"{ptxas_row['raw_registers_per_thread']})"
            )
        if ptxas_row["stack_frame_bytes"]:
            observations.append("ptxas-stack")
        if ptxas_row["spill_store_bytes"] or ptxas_row["spill_load_bytes"]:
            observations.append("ptxas-spill")
        if sass_row["ldl_instructions"] or sass_row["stl_instructions"]:
            observations.append("sass-local-memory")
        if sass_row["imma_instructions"] == 0:
            fatal_reasons.append("missing-imma")
        if integer(runtime, "local_bytes_per_thread", str(key)) != 0:
            observations.append("runtime-local-memory")
        if integer(runtime, "active_blocks_per_sm", str(key)) < 1:
            fatal_reasons.append("not-resident")
        if integer(runtime, "max_threads_per_block", str(key)) < width:
            fatal_reasons.append("width-exceeds-kernel-limit")
        if integer(runtime, "binary_version", str(key)) != 75:
            fatal_reasons.append("not-sm75-cubin")
        row: dict[str, object] = dict(runtime)
        row.update(
            kernel_symbol=symbol,
            kernel_demangled=demangled[symbol],
            ptxas_raw_registers_per_thread=ptxas_row[
                "raw_registers_per_thread"
            ],
            ptxas_stack_frame_bytes=ptxas_row["stack_frame_bytes"],
            ptxas_spill_store_bytes=ptxas_row["spill_store_bytes"],
            ptxas_spill_load_bytes=ptxas_row["spill_load_bytes"],
            sass_ldl_instructions=sass_row["ldl_instructions"],
            sass_stl_instructions=sass_row["stl_instructions"],
            sass_imma_instructions=sass_row["imma_instructions"],
            structural_status="pass" if not fatal_reasons else "fail",
            structural_reasons=";".join(fatal_reasons),
            resource_observations=";".join(observations),
        )
        merged.append(row)
        if fatal_reasons:
            failures.append(f"{key}: {', '.join(fatal_reasons)}")
    if len(merged) != len(runtime_rows):
        failures.append(
            f"merged {len(merged)} of {len(runtime_rows)} runtime resource rows"
        )
    if merged:
        fieldnames = list(merged[0])
        try:
            with resource_output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged)
        except OSError as exc:
            fail(f"could not write resource summary: {exc}")
    if failures:
        fail("resource structural gate failed:\n  " + "\n  ".join(failures))
    return merged


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_geomean_interval(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(math.exp(statistics.mean(math.log(value) for value in draw)))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_timing(
    source: Path,
    timing_output: Path,
    position_output: Path,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    required = {
        "family",
        "scenario",
        "cycle",
        "round",
        "sample_slot",
        "width",
        "timed_repeats",
        "timed_total_ms",
        "timed_per_call_ms",
        "status",
    }
    try:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                fail(
                    "timing sample CSV is missing columns: "
                    f"{sorted(required - set(reader.fieldnames or ()))}"
                )
            rows = list(reader)
    except OSError as exc:
        fail(f"could not read timing samples {source}: {exc}")
    if not rows:
        fail("timing sample CSV contains no samples")

    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    by_round: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    positions: dict[tuple[str, str, int, int], int] = defaultdict(int)
    cycles_seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        family = row.get("family", "")
        widths = GATE_WIDTHS if family == "gate" else DOWN_WIDTHS if family == "down" else ()
        if not widths:
            fail(f"invalid timing family: {family!r}")
        scenario = row.get("scenario", "")
        if scenario not in {"early", "late"}:
            fail(f"invalid timing scenario: {scenario!r}")
        width = integer(row, "width", "timing samples")
        if width not in widths:
            fail(f"unexpected width {width} for {family}")
        if row.get("status") != "ok":
            fail(f"non-ok timing sample: {row!r}")
        cycle = integer(row, "cycle", "timing samples")
        round_number = integer(row, "round", "timing samples")
        slot = integer(row, "sample_slot", "timing samples")
        if cycle < 1 or round_number < 1 or not 1 <= slot <= len(widths):
            fail(f"invalid timing cycle/round/slot: {row!r}")
        elapsed = finite_float(row, "timed_per_call_ms", "timing samples")
        total = finite_float(row, "timed_total_ms", "timing samples")
        repeats = integer(row, "timed_repeats", "timing samples")
        if elapsed <= 0.0 or total <= 0.0 or repeats < 1:
            fail(f"non-positive timing sample: {row!r}")
        if not math.isclose(total / repeats, elapsed, rel_tol=2e-4, abs_tol=1e-6):
            fail(f"inconsistent timing total/per-call values: {row!r}")
        key = (family, scenario, width)
        values[key].append(elapsed)
        round_key = (family, scenario, round_number)
        if width in by_round[round_key]:
            fail(f"duplicate width {width} in timing round {round_key}")
        by_round[round_key][width] = elapsed
        positions[(family, scenario, width, slot)] += 1
        cycles_seen[(family, scenario)].add(cycle)

    position_rows: list[dict[str, object]] = []
    for family, widths in (("gate", GATE_WIDTHS), ("down", DOWN_WIDTHS)):
        for scenario in ("early", "late"):
            cycles = cycles_seen[(family, scenario)]
            if not cycles:
                fail(f"missing timing samples for {family}/{scenario}")
            expected_cycles = set(range(1, max(cycles) + 1))
            if cycles != expected_cycles:
                fail(f"non-contiguous cycles for {family}/{scenario}: {cycles}")
            expected_per_position = len(cycles)
            for width in widths:
                for slot in range(1, len(widths) + 1):
                    count = positions[(family, scenario, width, slot)]
                    status = "pass" if count == expected_per_position else "fail"
                    position_rows.append(
                        {
                            "family": family,
                            "scenario": scenario,
                            "width": width,
                            "sample_slot": slot,
                            "expected_samples": expected_per_position,
                            "actual_samples": count,
                            "position_balance_status": status,
                        }
                    )
                    if status != "pass":
                        fail(
                            f"unbalanced timing positions for {family}/{scenario}/"
                            f"{width}/slot{slot}: {count} != {expected_per_position}"
                        )
            for (row_family, row_scenario, round_number), round_values in by_round.items():
                if row_family == family and row_scenario == scenario:
                    if set(round_values) != set(widths):
                        fail(
                            f"timing round {family}/{scenario}/{round_number} "
                            f"has widths {sorted(round_values)}, expected {list(widths)}"
                        )

    summary_rows: list[dict[str, object]] = []
    for family, widths in (("gate", GATE_WIDTHS), ("down", DOWN_WIDTHS)):
        for scenario in ("early", "late"):
            round_numbers = sorted(
                round_number
                for row_family, row_scenario, round_number in by_round
                if row_family == family and row_scenario == scenario
            )
            for width in widths:
                samples = values[(family, scenario, width)]
                paired = [
                    by_round[(family, scenario, round_number)][256]
                    / by_round[(family, scenario, round_number)][width]
                    for round_number in round_numbers
                ]
                median = statistics.median(samples)
                mean = statistics.mean(samples)
                stdev = statistics.pstdev(samples)
                geomean = math.exp(statistics.mean(math.log(value) for value in paired))
                ci_low, ci_high = bootstrap_geomean_interval(
                    paired,
                    bootstrap_samples,
                    seed=0x75C7A + (0 if family == "gate" else 1000)
                    + (0 if scenario == "early" else 100)
                    + width,
                )
                if width == 256:
                    classification = "baseline"
                elif ci_low > 1.0:
                    classification = "win"
                elif ci_high < 1.0:
                    classification = "regression"
                else:
                    classification = "inconclusive"
                summary_rows.append(
                    {
                        "family": family,
                        "scenario": scenario,
                        "width": width,
                        "samples": len(samples),
                        "median_ms": median,
                        "mean_ms": mean,
                        "mad_ms": median_absolute_deviation(samples),
                        "cv_pct": 100.0 * stdev / mean,
                        "min_ms": min(samples),
                        "max_ms": max(samples),
                        "paired_speedup_median_x": statistics.median(paired),
                        "paired_speedup_geomean_x": geomean,
                        "paired_speedup_ci95_low_x": ci_low,
                        "paired_speedup_ci95_high_x": ci_high,
                        "classification": classification,
                    }
                )

    for path, output_rows in (
        (timing_output, summary_rows),
        (position_output, position_rows),
    ):
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)
        except OSError as exc:
            fail(f"could not write timing summary {path}: {exc}")
    return summary_rows


if __name__ == "__main__":
    args = parse_args()
    runtime_rows = load_runtime_resources(args.runtime_resources)
    ptxas_records = parse_ptxas(read_required(args.ptxas_log, "PTXAS log"))
    sass_records = parse_sass(read_required(args.sass, "SASS dump"))
    merged_rows = merge_resources(
        runtime_rows,
        ptxas_records,
        sass_records,
        args.cxxfilt,
        args.resource_output,
    )
    print(f"resource_rows={len(merged_rows)}")
    print("resource_status=ok")
    if args.timing_samples:
        timing_rows = summarize_timing(
            args.timing_samples,
            args.timing_output,
            args.position_output,
            args.bootstrap_samples,
        )
        print(f"timing_summary_rows={len(timing_rows)}")
        print("timing_status=ok")
