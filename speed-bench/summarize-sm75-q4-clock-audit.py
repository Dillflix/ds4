#!/usr/bin/env python3
"""Summarize bounded per-device SM75 Q4 timing and nvidia-smi telemetry."""

from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def number(value: str) -> float:
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value or "")
    return float(match.group(0)) if match else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def timed_ms(path: Path) -> float:
    match = re.search(
        r"^timed_per_call_ms=([0-9]+(?:\.[0-9]+)?)$",
        path.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"{path} lacks timed_per_call_ms")
    return float(match.group(1))


def telemetry(path: Path, device: int) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        rows = [row for row in reader if int(number(row.get("index", "-1"))) == device]
        event_fields = [
            field for field in (reader.fieldnames or [])
            if "clocks_event_reasons." in field
            or "clocks_throttle_reasons." in field
        ]
    active = [row for row in rows if number(row.get("utilization.gpu [%]", "")) >= 20]
    if not active:
        active = [row for row in rows if number(row.get("utilization.gpu [%]", "")) > 0]
    clocks = [number(row.get("clocks.current.sm [MHz]", "")) for row in active]
    power = [number(row.get("power.draw [W]", "")) for row in active]
    temperature = [number(row.get("temperature.gpu", "")) for row in active]
    utilization = [number(row.get("utilization.gpu [%]", "")) for row in active]
    events: dict[str, str] = {}
    for field in event_fields:
        count = sum((row.get(field, "").strip().lower() == "active") for row in active)
        events[field] = f"{count}/{len(active)}" if active else "0/0"
    return {
        "active_samples": len(active),
        "util_p50": percentile(utilization, 0.50),
        "sm_clock_p10_mhz": percentile(clocks, 0.10),
        "sm_clock_p50_mhz": percentile(clocks, 0.50),
        "sm_clock_p90_mhz": percentile(clocks, 0.90),
        "power_p50_w": percentile(power, 0.50),
        "power_p90_w": percentile(power, 0.90),
        "temperature_max_c": max(temperature, default=0.0),
        "active_clock_events": ";".join(
            f"{key}={value}" for key, value in sorted(events.items())
            if not value.startswith("0/")
        ),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize-sm75-q4-clock-audit.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    run_map = output / "runs.tsv"
    with run_map.open(newline="", encoding="utf-8-sig") as handle:
        runs = list(csv.DictReader(handle, delimiter="\t"))
    samples: list[dict[str, object]] = []
    for run in runs:
        device = int(run["device"])
        row: dict[str, object] = {
            "mode": run["mode"],
            "scope": run["scope"],
            "trial": int(run["trial"]),
            "slot": int(run["slot"]),
            "scenario": run["scenario"],
            "device": device,
            "timed_per_call_ms": timed_ms(Path(run["log"])),
            "log": run["log"],
            "telemetry": run["telemetry"],
        }
        row.update(telemetry(Path(run["telemetry"]), device))
        samples.append(row)
    sample_fields = [
        "mode", "scope", "trial", "slot", "scenario", "device",
        "timed_per_call_ms", "active_samples", "util_p50",
        "sm_clock_p10_mhz", "sm_clock_p50_mhz", "sm_clock_p90_mhz",
        "power_p50_w", "power_p90_w", "temperature_max_c",
        "active_clock_events", "log", "telemetry",
    ]
    write_csv(output / "samples.csv", sample_fields, samples)

    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in samples:
        groups[(str(row["mode"]), str(row["scope"]),
                str(row["scenario"]), int(row["device"]))].append(row)
    summary: list[dict[str, object]] = []
    for (mode, scope, scenario, device), rows in sorted(groups.items()):
        summary.append({
            "mode": mode,
            "scope": scope,
            "scenario": scenario,
            "device": device,
            "trials": len(rows),
            "median_ms": statistics.median(float(row["timed_per_call_ms"]) for row in rows),
            "median_sm_clock_mhz": statistics.median(float(row["sm_clock_p50_mhz"]) for row in rows),
            "minimum_sm_clock_p10_mhz": min(float(row["sm_clock_p10_mhz"]) for row in rows),
            "median_power_p90_w": statistics.median(float(row["power_p90_w"]) for row in rows),
            "maximum_temperature_c": max(float(row["temperature_max_c"]) for row in rows),
            "clock_event_samples": ";".join(
                str(row["active_clock_events"]) for row in rows
                if row["active_clock_events"]
            ),
        })
    summary_fields = [
        "mode", "scope", "scenario", "device", "trials", "median_ms",
        "median_sm_clock_mhz", "minimum_sm_clock_p10_mhz",
        "median_power_p90_w", "maximum_temperature_c", "clock_event_samples",
    ]
    write_csv(output / "summary.csv", summary_fields, summary)

    lookup = {
        (str(row["mode"]), str(row["scope"]), str(row["scenario"]), int(row["device"])): row
        for row in summary
    }
    with (output / "analysis.txt").open("w", encoding="utf-8") as handle:
        handle.write("DS4 SM75 four-GPU Q4 clock/power audit\n\n")
        for row in summary:
            handle.write(
                f"{row['mode']} {row['scope']} {row['scenario']} GPU{row['device']}: "
                f"{float(row['median_ms']):.6f} ms, "
                f"clock-p50={float(row['median_sm_clock_mhz']):.0f} MHz, "
                f"power-p90={float(row['median_power_p90_w']):.1f} W, "
                f"temp-max={float(row['maximum_temperature_c']):.0f} C\n"
            )
        for mode in sorted({str(row["mode"]) for row in summary}):
            for scenario in sorted({str(row["scenario"]) for row in summary}):
                base = lookup.get((mode, "single", scenario, 0))
                if not base or not float(base["median_ms"]):
                    continue
                handle.write(f"\n{mode} single {scenario}, relative to GPU0:\n")
                for device in range(4):
                    row = lookup.get((mode, "single", scenario, device))
                    if row:
                        ratio = float(row["median_ms"]) / float(base["median_ms"])
                        handle.write(f"  GPU{device}: {ratio:.6f}x\n")
        if {"baseline", "normalized"}.issubset({str(row["mode"]) for row in summary}):
            handle.write("\nNormalized/baseline timing ratios (<1 is faster):\n")
            for key, row in sorted(lookup.items()):
                mode, scope, scenario, device = key
                if mode != "normalized":
                    continue
                base = lookup.get(("baseline", scope, scenario, device))
                if base and float(base["median_ms"]):
                    ratio = float(row["median_ms"]) / float(base["median_ms"])
                    handle.write(f"  {scope} {scenario} GPU{device}: {ratio:.6f}x\n")
    print((output / "analysis.txt").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
