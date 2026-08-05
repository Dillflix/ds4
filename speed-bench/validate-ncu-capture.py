#!/usr/bin/env python3
"""Reject empty or mis-targeted Nsight Compute raw CSV exports."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from typing import NoReturn


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("kernel_regex")
    parser.add_argument("device")
    parser.add_argument("--process", default="ds4-bench")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--static-shared-kib", type=float)
    parser.add_argument("--dynamic-shared-kib", type=float)
    args = parser.parse_args()

    try:
        kernel_pattern = re.compile(args.kernel_regex)
    except re.error as exc:
        fail(f"invalid expected kernel regex {args.kernel_regex!r}: {exc}")

    try:
        handle = open(args.csv_path, newline="", encoding="utf-8-sig")
    except OSError as exc:
        fail(f"cannot read Nsight CSV {args.csv_path}: {exc}")

    with handle:
        reader = csv.DictReader(handle)
        required = {
            "ID",
            "Process Name",
            "Kernel Name",
            "Device",
            "gpu__time_duration.sum",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            fail("Nsight CSV is missing required columns: " + ", ".join(missing))
        rows = [row for row in reader if (row.get("ID") or "").strip()]

    if len(rows) != 1:
        fail(f"expected exactly one profiled kernel row, found {len(rows)}")

    row = rows[0]
    process_name = os.path.basename((row.get("Process Name") or "").strip())
    if process_name != args.process:
        fail(f"profiled process is {process_name!r}, expected {args.process!r}")

    actual_device = (row.get("Device") or "").strip()
    if actual_device != str(args.device):
        fail(f"profiled physical device is {actual_device!r}, expected {args.device!r}")

    kernel_name = (row.get("Kernel Name") or "").strip()
    if not kernel_pattern.search(kernel_name):
        fail(
            f"profiled kernel {kernel_name!r} does not match "
            f"{args.kernel_regex!r}"
        )

    duration_text = (row.get("gpu__time_duration.sum") or "").strip()
    try:
        duration_value = float(duration_text.replace(",", ""))
    except ValueError:
        fail(f"invalid gpu__time_duration.sum value: {duration_text!r}")
    if not math.isfinite(duration_value) or duration_value <= 0.0:
        fail(f"non-finite or non-positive gpu__time_duration.sum value: {duration_text!r}")

    if args.block_size is not None:
        block_text = (row.get("launch__block_size") or "").strip()
        try:
            block_size = int(block_text.replace(",", ""))
        except ValueError:
            fail(f"invalid launch__block_size value: {block_text!r}")
        if block_size != args.block_size:
            fail(
                f"profiled block size is {block_size}, expected "
                f"{args.block_size}"
            )

    if args.grid_size is not None:
        grid_text = (row.get("launch__grid_size") or "").strip()
        try:
            grid_size = int(grid_text.replace(",", ""))
        except ValueError:
            fail(f"invalid launch__grid_size value: {grid_text!r}")
        if grid_size != args.grid_size:
            fail(
                f"profiled grid size is {grid_size}, expected "
                f"{args.grid_size}"
            )

    def check_shared(column: str, expected: float, label: str) -> None:
        value_text = (row.get(column) or "").strip()
        try:
            value = float(value_text.replace(",", ""))
        except ValueError:
            fail(f"invalid {column} value: {value_text!r}")
        tolerance = max(0.001, abs(expected) * 1.0e-5)
        if not math.isfinite(value) or abs(value - expected) > tolerance:
            fail(
                f"profiled {label} shared memory is {value:g} KiB, "
                f"expected {expected:g} KiB"
            )

    if args.static_shared_kib is not None:
        check_shared(
            "launch__shared_mem_per_block_static",
            args.static_shared_kib,
            "static",
        )
    if args.dynamic_shared_kib is not None:
        check_shared(
            "launch__shared_mem_per_block_dynamic",
            args.dynamic_shared_kib,
            "dynamic",
        )

    print(
        "validated Nsight capture: "
        f"process={process_name} device={actual_device} "
        f"duration={duration_value:g} kernel={kernel_name}"
        + (f" block_size={args.block_size}" if args.block_size is not None else "")
        + (f" grid_size={args.grid_size}" if args.grid_size is not None else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
