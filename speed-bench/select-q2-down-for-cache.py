#!/usr/bin/env python3
"""Choose a Q4-first routed recipe that satisfies a measured Q8 cache deficit."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


MIB = 1024 * 1024
# Every full routed tensor is 1152 MiB in Q4_K. Q2_K is 672 MiB and IQ2_XXS is
# 528 MiB. DS4's paired expert ownership stores one half on each device, so:
#   Q4_K down -> Q2_K down:                 240 MiB/device
#   Q4_K gate+up -> IQ2_XXS gate+up pair:   624 MiB/device
# Gate and up are selected as a matched pair because DS4's fast routed-IQ2
# kernels require both tensors in a layer to have the same quant type.


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


def select_stage_recipe(
    required_bytes: int,
    q2_candidates: list[int],
    iq2_candidates: list[int],
    q2_saving_bytes: int,
    iq2_pair_saving_bytes: int,
) -> tuple[list[int], list[int], int]:
    """Minimize IQ2 pairs, then minimize Q2-down tensors for a stage.

    The lexicographic objective preserves Q4 gate/up wherever a down-only plan
    is feasible. When it is not, it adds the smallest possible number of
    matched IQ2 gate/up pairs, then avoids unnecessary Q2-down conversions
    made redundant by the last pair's granularity.
    """
    if required_bytes <= 0:
        return [], [], 0
    if q2_saving_bytes <= 0 or iq2_pair_saving_bytes <= 0:
        die("quantization savings must be positive")

    max_q2_saving = len(q2_candidates) * q2_saving_bytes
    iq2_count = max(
        0,
        math.ceil((required_bytes - max_q2_saving) / iq2_pair_saving_bytes),
    )
    if iq2_count > len(iq2_candidates):
        max_reclaim = max_q2_saving + len(iq2_candidates) * iq2_pair_saving_bytes
        die(
            "cache target remains infeasible after every available Q2 down "
            "tensor and IQ2 gate/up pair "
            f"(need={required_bytes / MIB:.0f} MiB, "
            f"maximum={max_reclaim / MIB:.0f} MiB)"
        )

    remaining = max(0, required_bytes - iq2_count * iq2_pair_saving_bytes)
    q2_count = math.ceil(remaining / q2_saving_bytes)
    if q2_count > len(q2_candidates):
        die("internal selection error: Q2-down candidate capacity exceeded")

    q2_layers = q2_candidates[:q2_count]
    iq2_layers = iq2_candidates[:iq2_count]
    reclaim = (
        len(q2_layers) * q2_saving_bytes
        + len(iq2_layers) * iq2_pair_saving_bytes
    )
    return q2_layers, iq2_layers, reclaim


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
    parser.add_argument(
        "--saving-per-device-mib",
        "--q2-down-saving-per-device-mib",
        dest="q2_down_saving_per_device_mib",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--iq2-gate-up-pair-saving-per-device-mib", type=int, default=624
    )
    parser.add_argument("--routed-first", type=int, default=3)
    parser.add_argument("--routed-last", type=int, default=42)
    parser.add_argument("--layer-order")
    parser.add_argument("--iq2-layer-order")
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
    if (
        args.extra_headroom_mib < 0
        or args.q2_down_saving_per_device_mib <= 0
        or args.iq2_gate_up_pair_saving_per_device_mib <= 0
    ):
        die("headroom must be nonnegative and quantization savings must be positive")

    layout = parse_layout(args.layout_log)
    missing = cache_missing_by_device(args.audit)
    preferred = layer_order(args.routed_first, args.routed_last, args.layer_order)
    preferred_iq2 = layer_order(
        args.routed_first,
        args.routed_last,
        args.iq2_layer_order or args.layer_order,
    )
    q2_saving = args.q2_down_saving_per_device_mib * MIB
    iq2_pair_saving = args.iq2_gate_up_pair_saving_per_device_mib * MIB
    extra = args.extra_headroom_mib * MIB
    selected_q2: list[int] = []
    selected_iq2: list[int] = []

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
        q2_candidates = [layer for layer in preferred if lo <= layer <= hi]
        iq2_candidates = [layer for layer in preferred_iq2 if lo <= layer <= hi]
        chosen_q2, chosen_iq2, reclaim = select_stage_recipe(
            required,
            q2_candidates,
            iq2_candidates,
            q2_saving,
            iq2_pair_saving,
        )
        selected_q2.extend(chosen_q2)
        selected_iq2.extend(chosen_iq2)
        print(
            f"stage {stage} device {physical_device}: missing={deficit / MIB:.0f} MiB "
            f"extra={args.extra_headroom_mib} MiB, "
            f"q2_down={len(chosen_q2)} "
            f"({len(chosen_q2) * args.q2_down_saving_per_device_mib} MiB), "
            f"iq2_gate_up_pairs={len(chosen_iq2)} "
            f"({len(chosen_iq2) * args.iq2_gate_up_pair_saving_per_device_mib} MiB), "
            f"reclaim={reclaim / MIB:.0f} MiB, "
            f"q2_layers={','.join(map(str, chosen_q2)) or '-'}, "
            f"iq2_layers={','.join(map(str, chosen_iq2)) or '-'}",
            file=sys.stderr,
        )

    if not selected_q2 and not selected_iq2:
        die("the baseline cache was complete; a selective quant is unnecessary")
    selected_q2 = sorted(set(selected_q2))
    selected_iq2 = sorted(set(selected_iq2))
    override_list = [
        f"blk.{layer}.ffn_down_exps.weight=q2_k" for layer in selected_q2
    ]
    for layer in selected_iq2:
        override_list.extend(
            [
                f"blk.{layer}.ffn_gate_exps.weight=iq2_xxs",
                f"blk.{layer}.ffn_up_exps.weight=iq2_xxs",
            ]
        )
    q2_text = ",".join(map(str, selected_q2)) or "-"
    iq2_text = ",".join(map(str, selected_iq2)) or "-"
    print(f"{q2_text}\t{iq2_text}\t{','.join(override_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
