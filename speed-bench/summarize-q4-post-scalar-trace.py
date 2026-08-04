#!/usr/bin/env python3
"""Summarize an Nsight Systems CUDA kernel summary for full-Q4 prefill."""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NoReturn


def die(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def parse_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError:
        die(f"invalid {field}: {value!r}")


def classify(name: str) -> str:
    if "moe_gate_up_mid_q4K_tile8_mma_kernel" in name:
        return "q4_gate_up_tile8"
    if "moe_down_q4K_tile16_mma_sm75_kernel" in name:
        return "q4_down_tile16"
    if "matmul_q8_0_mma_sm75_exact_kernel" in name:
        for width in (32, 64, 128, 256):
            if re.search(rf"(?:unsigned int\))?{width}(?:,|>)", name):
                return f"dense_q8_t{width}"
        return "dense_q8_other"
    if "attention" in name.lower():
        return "attention"
    if "moe_" in name.lower():
        return "other_moe"
    return "other"


def scalar_specialization(name: str) -> str:
    if re.search(r"(?:\(bool\)1|\btrue\b|Lb1E)", name):
        return "true"
    if re.search(r"(?:\(bool\)0|\bfalse\b|Lb0E)", name):
        return "false"
    return "unknown"


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        raw = list(csv.reader(handle))
    header_index = next(
        (index for index, row in enumerate(raw) if row and row[0] == "Time (%)"),
        None,
    )
    if header_index is None:
        die(f"Nsight kernel-summary header not found in {path}")
    header = raw[header_index]
    required = {"Time (%)", "Total Time (ns)", "Instances", "Name"}
    if not required.issubset(header):
        die(f"Nsight kernel-summary columns are incomplete in {path}")
    parsed: list[dict[str, object]] = []
    for values in raw[header_index + 1 :]:
        if not values or not any(values):
            continue
        if len(values) != len(header):
            die(f"malformed Nsight row with {len(values)} columns; expected {len(header)}")
        row = dict(zip(header, values))
        name = row["Name"]
        parsed.append(
            {
                "reported_time_pct": float(row["Time (%)"]),
                "total_time_ns": parse_int(row["Total Time (ns)"], "Total Time (ns)"),
                "instances": parse_int(row["Instances"], "Instances"),
                "name": name,
                "group": classify(name),
                "scalar_specialization": scalar_specialization(name),
            }
        )
    if not parsed:
        die(f"no CUDA kernel rows found in {path}")
    return parsed


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 4:
        die("usage: summarize-q4-post-scalar-trace.py INPUT.csv GROUPS.csv TARGETS.csv")
    source, groups_path, targets_path = map(Path, sys.argv[1:])
    rows = read_rows(source)
    captured_ns = sum(int(row["total_time_ns"]) for row in rows)
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_time_ns": 0, "instances": 0, "kernel_rows": 0}
    )
    for row in rows:
        group = grouped[str(row["group"])]
        group["total_time_ns"] += int(row["total_time_ns"])
        group["instances"] += int(row["instances"])
        group["kernel_rows"] += 1

    group_rows: list[dict[str, object]] = []
    for group, values in grouped.items():
        group_rows.append(
            {
                "group": group,
                "total_time_ns": values["total_time_ns"],
                "time_pct_of_captured": f"{100.0 * values['total_time_ns'] / captured_ns:.6f}",
                "instances": values["instances"],
                "kernel_rows": values["kernel_rows"],
            }
        )
    group_rows.sort(key=lambda row: int(row["total_time_ns"]), reverse=True)
    write_csv(
        groups_path,
        ["group", "total_time_ns", "time_pct_of_captured", "instances", "kernel_rows"],
        group_rows,
    )

    targets = [
        row
        for row in rows
        if row["group"] in {"q4_gate_up_tile8", "q4_down_tile16"}
    ]
    targets.sort(key=lambda row: str(row["group"]))
    if {str(row["group"]) for row in targets} != {
        "q4_gate_up_tile8",
        "q4_down_tile16",
    }:
        die("both production Q4 kernel families were not present")
    write_csv(
        targets_path,
        [
            "group",
            "total_time_ns",
            "reported_time_pct",
            "instances",
            "scalar_specialization",
            "name",
        ],
        targets,
    )

    by_group = {row["group"]: row for row in group_rows}
    gate = by_group["q4_gate_up_tile8"]
    down = by_group["q4_down_tile16"]
    print(f"captured_kernel_time_ns={captured_ns}")
    print(
        "q4_gate_up="
        f"{gate['time_pct_of_captured']}% instances={gate['instances']} "
        f"kernel_rows={gate['kernel_rows']}"
    )
    print(
        "q4_down="
        f"{down['time_pct_of_captured']}% instances={down['instances']} "
        f"kernel_rows={down['kernel_rows']}"
    )
    print(
        "q4_combined="
        f"{100.0 * (int(gate['total_time_ns']) + int(down['total_time_ns'])) / captured_ns:.6f}%"
    )


if __name__ == "__main__":
    main()
