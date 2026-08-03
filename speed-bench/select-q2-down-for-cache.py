#!/usr/bin/env python3
"""Choose the minimum per-stage Q2_K down tensors needed by a Q8 cache audit."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


MIB = 1024 * 1024
# A full routed-down tensor is 1152 MiB in Q4_K and 672 MiB in Q2_K.
# DS4's paired expert ownership stores one half on each device, so one selected
# tensor reclaims exactly (1152 - 672) / 2 = 240 MiB on both GPUs in its pair.


def die(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_layout(path: Path) -> list[tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    pattern = re.compile(r"^\s*GPU(\d+): layers (\d+)-(\d+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            ranges[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))
    if 0 not in ranges or 1 not in ranges:
        die(f"could not find GPU0/GPU1 layer ranges in {path}")
    return [ranges[0], ranges[1]]


def cache_missing_by_device(path: Path) -> dict[int, int]:
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (
                    int(row["physical_device"]),
                    int(row["weight_offset"]),
                    int(row["weight_bytes"]),
                )
            except (KeyError, ValueError) as exc:
                die(f"invalid Q8 cache audit row in {path}: {exc}")
            grouped[key].append(row)

    missing: dict[int, int] = defaultdict(int)
    for (device, _offset, _weight_bytes), rows in grouped.items():
        if any(row.get("result", "").startswith("f16_") for row in rows):
            continue
        budget_rows = [
            row for row in rows
            if row.get("result") == "native_q8"
            and row.get("reason") == "budget_or_limit"
        ]
        if budget_rows:
            try:
                missing[device] += int(budget_rows[0]["fp16_bytes"])
            except (KeyError, ValueError) as exc:
                die(f"invalid fp16_bytes in {path}: {exc}")
    return dict(missing)


def layer_order(first: int, last: int, explicit: str | None) -> list[int]:
    if not explicit:
        return list(range(first, last + 1))
    result: list[int] = []
    seen: set[int] = set()
    for item in explicit.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_text, hi_text = item.split("-", 1)
            values = range(int(lo_text), int(hi_text) + 1)
        else:
            values = [int(item)]
        for value in values:
            if first <= value <= last and value not in seen:
                result.append(value)
                seen.add(value)
    if not result:
        die("Q2_DOWN_LAYER_ORDER did not contain any routed layers")
    return result


def verify(path: Path) -> None:
    missing = cache_missing_by_device(path)
    if missing:
        detail = ", ".join(
            f"device {device}: {value / MIB:.0f} MiB"
            for device, value in sorted(missing.items())
        )
        die(f"generated model still has budget-limited native Q8 weights ({detail})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--layout-log", type=Path)
    parser.add_argument("--gpu-devices", default="0,2,1,3")
    parser.add_argument("--extra-headroom-mib", type=int, default=512)
    parser.add_argument("--saving-per-device-mib", type=int, default=240)
    parser.add_argument("--routed-first", type=int, default=3)
    parser.add_argument("--routed-last", type=int, default=42)
    parser.add_argument("--layer-order")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify(args.audit)
        print("Q8 FP16 cache verification: complete residency, no budget misses")
        return 0
    if args.layout_log is None:
        die("--layout-log is required when selecting layers")

    devices = [int(value) for value in args.gpu_devices.split(",")]
    if len(devices) != 4 or len(set(devices)) != 4:
        die("--gpu-devices must contain four distinct device numbers")
    if args.extra_headroom_mib < 0 or args.saving_per_device_mib <= 0:
        die("headroom must be nonnegative and per-tensor saving must be positive")

    layout = parse_layout(args.layout_log)
    missing = cache_missing_by_device(args.audit)
    preferred = layer_order(args.routed_first, args.routed_last, args.layer_order)
    saving = args.saving_per_device_mib * MIB
    extra = args.extra_headroom_mib * MIB
    selected: list[int] = []

    for stage, (lo, hi) in enumerate(layout):
        physical_device = devices[stage]
        deficit = missing.get(physical_device, 0)
        if deficit == 0:
            print(
                f"stage {stage} device {physical_device}: cache already complete; "
                "no Q2 tensor required",
                file=sys.stderr,
            )
            continue
        required = deficit + extra
        count = math.ceil(required / saving)
        candidates = [layer for layer in preferred if lo <= layer <= hi]
        if count > len(candidates):
            die(
                f"stage {stage} needs {count} Q2 down tensors but only "
                f"{len(candidates)} routed layers are available in {lo}-{hi}"
            )
        chosen = candidates[:count]
        selected.extend(chosen)
        print(
            f"stage {stage} device {physical_device}: missing={deficit / MIB:.0f} MiB "
            f"extra={args.extra_headroom_mib} MiB, select={count}, "
            f"reclaim={count * args.saving_per_device_mib} MiB, "
            f"layers={','.join(map(str, chosen))}",
            file=sys.stderr,
        )

    if not selected:
        die("the baseline cache was complete; a selective-Q2 model is unnecessary")
    selected = sorted(set(selected))
    overrides = ",".join(
        f"blk.{layer}.ffn_down_exps.weight=q2_k" for layer in selected
    )
    print(f"{','.join(map(str, selected))}\t{overrides}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
