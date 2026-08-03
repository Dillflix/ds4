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

    print(
        "validated Nsight capture: "
        f"process={process_name} device={actual_device} "
        f"duration={duration_value:g} kernel={kernel_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
