#!/usr/bin/env python3
"""Summarize one-row Nsight Compute captures of SM75 decode weight kernels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import NoReturn


SCENARIOS = [
    ("q4-32-gate-up", "routed Q4-32 gate/up", 25.5),
    ("q3a4-gate-up", "routed Q3A4 gate/up", 20.25),
    ("q8-single-t32", "dense Q8 single T32", 34.0),
    ("q8-pair-2048", "dense Q8 pair 2048", 34.0),
    ("q8-pair-1024", "dense Q8 pair 1024", 17.0),
    ("q8-kslice-t256", "dense Q8 K-slice T256", 17.0),
    ("q8-grouped-a-half", "dense Q8 grouped A half", 0.53125),
    ("q8-shared-mid", "dense Q8 shared gate/up", 34.0),
    ("f16-pair-256", "F16 compressor pair 256", 4.0),
    ("f16-pair-512", "F16 compressor pair 512", 8.0),
    ("f16-pair-1024", "F16 compressor pair 1024", 16.0),
]

CORE_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum.per_second",
    "dram__bytes.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def number(row: dict[str, str], key: str, required: bool = False) -> float:
    text = (row.get(key) or "").strip().replace(",", "")
    if not text:
        if required:
            fail(f"missing required Nsight metric {key}")
        return math.nan
    try:
        value = float(text)
    except ValueError:
        fail(f"invalid Nsight metric {key}={text!r}")
    if not math.isfinite(value):
        fail(f"non-finite Nsight metric {key}={text!r}")
    return value


def load_one(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = [metric for metric in CORE_METRICS if metric not in fields]
            if missing:
                fail(f"{path} lacks required metrics: {', '.join(missing)}")
            rows = [row for row in reader if (row.get("ID") or "").strip()]
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    if len(rows) != 1:
        fail(f"{path} contains {len(rows)} kernel rows; expected exactly one")
    return rows[0]


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def evidence_class(dram_pct: float, load_eff: float) -> str:
    if dram_pct >= 75.0:
        return "DRAM-saturated: reduce/distribute bytes"
    if load_eff < 75.0:
        return "load-efficiency limited: fix layout/coalescing"
    if dram_pct >= 50.0:
        return "high DRAM pressure: bytes and latency both matter"
    return "not DRAM-saturated: inspect latency/occupancy/instructions"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ncu_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scenarios", help="comma-separated scenario subset")
    args = parser.parse_args()

    selected = None
    if args.scenarios:
        selected = {item for item in args.scenarios.split(",") if item}
    records: list[dict[str, object]] = []
    for name, family, weight_mib in SCENARIOS:
        if selected is not None and name not in selected:
            continue
        path = args.ncu_dir / f"{name}.csv"
        if not path.is_file():
            fail(f"missing Nsight CSV for selected scenario {name}: {path}")
        row = load_one(path)
        duration_ms = number(row, "gpu__time_duration.sum", True)
        dram_bps = number(row, "dram__bytes.sum.per_second", True)
        dram_pct = number(
            row, "dram__bytes.avg.pct_of_peak_sustained_elapsed", True
        )
        l2_hit = number(row, "lts__t_sector_hit_rate.pct", True)
        load_eff = number(
            row,
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            True,
        )
        sectors_request = number(
            row,
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
            True,
        )
        long_scoreboard = number(
            row,
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            True,
        )
        mio = number(
            row,
            "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
            True,
        )
        occupancy = number(
            row, "sm__warps_active.avg.pct_of_peak_sustained_active", True
        )
        records.append(
            {
                "scenario": name,
                "family": family,
                "kernel": (row.get("Kernel Name") or "").strip(),
                "duration_ms": duration_ms,
                "expected_weight_stream_mib": weight_mib,
                "dram_gb_s": dram_bps / 1.0e9,
                "dram_peak_pct": dram_pct,
                "dram_read_mib": number(row, "dram__bytes_read.sum") / 1048576.0,
                "dram_write_mib": number(row, "dram__bytes_write.sum") / 1048576.0,
                "l2_hit_pct": l2_hit,
                "l2_peak_pct": number(
                    row, "lts__throughput.avg.pct_of_peak_sustained_elapsed"
                ),
                "global_load_efficiency_pct": load_eff,
                "global_load_sectors_per_request": sectors_request,
                "long_scoreboard_ratio": long_scoreboard,
                "mio_throttle_ratio": mio,
                "achieved_occupancy_pct": occupancy,
                "eligible_warps_per_cycle": number(
                    row, "smsp__warps_eligible.avg.per_cycle_active"
                ),
                "registers_per_thread": number(row, "launch__registers_per_thread"),
                "shared_mem_per_block": number(row, "launch__shared_mem_per_block"),
                "evidence_class": evidence_class(dram_pct, load_eff),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    fields = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(records)

    md_path = args.output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# SM75 one-token decode weight-kernel profile\n\n")
        handle.write(
            "Each row is one separately targeted shipping decode kernel. "
            "The default audit flushes caches between Nsight replay passes, "
            "representing first-use layer weights rather than replay-warm L2.\n\n"
        )
        handle.write(
            "| Scenario | Duration ms | DRAM GB/s | DRAM peak | L2 hit | "
            "Load efficiency | Sectors/request | Long scoreboard | MIO | "
            "Occupancy | Evidence |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | --- |\n"
        )
        for record in records:
            handle.write(
                f"| {record['scenario']} | {fmt(record['duration_ms'], 6)} | "
                f"{fmt(record['dram_gb_s'])} | "
                f"{fmt(record['dram_peak_pct'])}% | "
                f"{fmt(record['l2_hit_pct'])}% | "
                f"{fmt(record['global_load_efficiency_pct'])}% | "
                f"{fmt(record['global_load_sectors_per_request'])} | "
                f"{fmt(record['long_scoreboard_ratio'])} | "
                f"{fmt(record['mio_throttle_ratio'])} | "
                f"{fmt(record['achieved_occupancy_pct'])}% | "
                f"{record['evidence_class']} |\n"
            )
        handle.write(
            "\n`Sectors/request` is a coalescing proxy, not a load-efficiency "
            "percentage. The evidence classification uses only measured DRAM "
            "peak utilization and global-load efficiency; use the raw stall, "
            "occupancy, and L2 columns before choosing an implementation.\n"
        )

    print(md_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
