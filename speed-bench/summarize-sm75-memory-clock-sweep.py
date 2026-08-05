#!/usr/bin/env python3
"""Summarize the reversible SM75 Q4 memory-clock sweep."""

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
        fields = reader.fieldnames or []
        rows = [row for row in reader if int(number(row.get("index", "-1"))) == device]
    active = [row for row in rows if number(row.get("utilization.gpu [%]", "")) >= 20]
    if not active:
        active = [row for row in rows if number(row.get("utilization.gpu [%]", "")) > 0]
    sm_clocks = [number(row.get("clocks.current.sm [MHz]", "")) for row in active]
    mem_clocks = [number(row.get("clocks.current.memory [MHz]", "")) for row in active]
    power = [number(row.get("power.draw [W]", "")) for row in active]
    temperature = [number(row.get("temperature.gpu", "")) for row in active]
    utilization = [number(row.get("utilization.gpu [%]", "")) for row in active]
    event_fields = sorted({
        field for field in fields
        if "clocks_event_reasons." in field or "clocks_throttle_reasons." in field
    })
    events: list[str] = []
    for field in event_fields:
        count = sum(row.get(field, "").strip().lower() == "active" for row in active)
        if count:
            events.append(f"{field}={count}/{len(active)}")
    return {
        "active_samples": len(active),
        "util_p50": percentile(utilization, 0.50),
        "sm_clock_p10_mhz": percentile(sm_clocks, 0.10),
        "sm_clock_p50_mhz": percentile(sm_clocks, 0.50),
        "sm_clock_p90_mhz": percentile(sm_clocks, 0.90),
        "memory_clock_p50_mhz": percentile(mem_clocks, 0.50),
        "power_p50_w": percentile(power, 0.50),
        "power_p90_w": percentile(power, 0.90),
        "temperature_max_c": max(temperature, default=0.0),
        "active_clock_events": ";".join(events),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize-sm75-memory-clock-sweep.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    with (output / "runs.tsv").open(newline="", encoding="utf-8-sig") as handle:
        runs = list(csv.DictReader(handle, delimiter="\t"))
    if not runs:
        raise RuntimeError("runs.tsv contains no runs")

    samples: list[dict[str, object]] = []
    for run in runs:
        device = int(run["device"])
        row: dict[str, object] = {
            "trial": int(run["trial"]),
            "slot": int(run["slot"]),
            "scenario": run["scenario"],
            "device": device,
            "requested_sm_clock_mhz": int(run["requested_sm_clock_mhz"]),
            "requested_memory_clock_mhz": int(run["requested_memory_clock_mhz"]),
            "timed_per_call_ms": timed_ms(Path(run["log"])),
            "log": run["log"],
            "telemetry": run["telemetry"],
        }
        row.update(telemetry(Path(run["telemetry"]), device))
        samples.append(row)
    sample_fields = [
        "trial", "slot", "scenario", "device", "requested_sm_clock_mhz",
        "requested_memory_clock_mhz", "timed_per_call_ms", "active_samples",
        "util_p50", "sm_clock_p10_mhz", "sm_clock_p50_mhz",
        "sm_clock_p90_mhz", "memory_clock_p50_mhz", "power_p50_w",
        "power_p90_w", "temperature_max_c", "active_clock_events", "log",
        "telemetry",
    ]
    write_csv(output / "samples.csv", sample_fields, samples)

    groups: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in samples:
        groups[(str(row["scenario"]), int(row["device"]),
                int(row["requested_memory_clock_mhz"]))].append(row)
    summary: list[dict[str, object]] = []
    for (scenario, device, memory_clock), rows in sorted(groups.items()):
        summary.append({
            "scenario": scenario,
            "device": device,
            "requested_memory_clock_mhz": memory_clock,
            "trials": len(rows),
            "median_ms": statistics.median(float(row["timed_per_call_ms"]) for row in rows),
            "median_sm_clock_mhz": statistics.median(float(row["sm_clock_p50_mhz"]) for row in rows),
            "median_memory_clock_mhz": statistics.median(float(row["memory_clock_p50_mhz"]) for row in rows),
            "median_power_p90_w": statistics.median(float(row["power_p90_w"]) for row in rows),
            "maximum_temperature_c": max(float(row["temperature_max_c"]) for row in rows),
            "clock_event_samples": ";".join(
                str(row["active_clock_events"]) for row in rows if row["active_clock_events"]
            ),
        })
    summary_fields = [
        "scenario", "device", "requested_memory_clock_mhz", "trials",
        "median_ms", "median_sm_clock_mhz", "median_memory_clock_mhz",
        "median_power_p90_w", "maximum_temperature_c", "clock_event_samples",
    ]
    write_csv(output / "summary.csv", summary_fields, summary)

    grouped_summary: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        grouped_summary[(str(row["scenario"]), int(row["device"]))].append(row)
    decisions: list[dict[str, object]] = []
    for (scenario, device), rows in sorted(grouped_summary.items()):
        maximum = max(rows, key=lambda row: int(row["requested_memory_clock_mhz"]))
        best = min(rows, key=lambda row: float(row["median_ms"]))
        max_ms = float(maximum["median_ms"])
        best_ms = float(best["median_ms"])
        decisions.append({
            "scenario": scenario,
            "device": device,
            "maximum_memory_clock_mhz": int(maximum["requested_memory_clock_mhz"]),
            "maximum_clock_ms": max_ms,
            "best_memory_clock_mhz": int(best["requested_memory_clock_mhz"]),
            "best_ms": best_ms,
            "best_sm_clock_mhz": float(best["median_sm_clock_mhz"]),
            "speedup_over_max": max_ms / best_ms if best_ms else 0.0,
        })
    decision_fields = [
        "scenario", "device", "maximum_memory_clock_mhz", "maximum_clock_ms",
        "best_memory_clock_mhz", "best_ms", "best_sm_clock_mhz",
        "speedup_over_max",
    ]
    write_csv(output / "decisions.csv", decision_fields, decisions)

    with (output / "analysis.txt").open("w", encoding="utf-8") as handle:
        handle.write("DS4 SM75 Q4 memory-clock/power-headroom sweep\n\n")
        for row in summary:
            handle.write(
                f"{row['scenario']} GPU{row['device']} requested-memory="
                f"{row['requested_memory_clock_mhz']} MHz: "
                f"{float(row['median_ms']):.6f} ms, actual-memory-p50="
                f"{float(row['median_memory_clock_mhz']):.0f} MHz, SM-p50="
                f"{float(row['median_sm_clock_mhz']):.0f} MHz, power-p90="
                f"{float(row['median_power_p90_w']):.1f} W\n"
            )
        handle.write("\nBest measured settings (selection is per scenario and device):\n")
        for row in decisions:
            handle.write(
                f"  {row['scenario']} GPU{row['device']}: memory="
                f"{row['best_memory_clock_mhz']} MHz, {float(row['best_ms']):.6f} ms, "
                f"SM={float(row['best_sm_clock_mhz']):.0f} MHz, "
                f"speedup-vs-max={float(row['speedup_over_max']):.6f}x\n"
            )
    print((output / "analysis.txt").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
