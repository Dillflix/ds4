#!/usr/bin/env python3
"""Maximize routed Q4 tensors while making the complete Q8-F16 plan resident."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


MIB = 1024 * 1024


def die(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_layout(path: Path) -> list[tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    pattern = re.compile(r"^\s*GPU(\d+): layers (\d+)-(\d+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            ranges[int(match.group(1))] = (
                int(match.group(2)),
                int(match.group(3)),
            )
    if 0 not in ranges or 1 not in ranges:
        die(f"could not find GPU0/GPU1 layer ranges in {path}")
    return [ranges[0], ranges[1]]


def parse_devices(text: str) -> list[int]:
    try:
        devices = [int(value) for value in text.split(",")]
    except ValueError as exc:
        die(f"invalid --gpu-devices: {exc}")
    if len(devices) != 4 or len(set(devices)) != 4:
        die("--gpu-devices must contain four distinct device numbers")
    return devices


def device_pairs(devices: list[int]) -> list[set[int]]:
    # CLI order is home0,home1,partner0,partner1.
    return [{devices[0], devices[2]}, {devices[1], devices[3]}]


def required_reclaim_by_stage(
    path: Path,
    pairs: list[set[int]],
    headroom_bytes_per_device: int,
) -> tuple[dict[int, int], dict[int, tuple[dict[int, int], list[int]]]]:
    """Return exact equal reclaim needed on both devices of each stage pair.

    Routed demotion frees the same bytes on both expert owners. Home-only Q8
    candidates must fit their consumer; candidates with a validated fallback
    may be assigned whole to either pair member. Subset-sum finds the optimal
    indivisible assignment instead of treating a large tensor as splittable.
    """
    fixed: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    flexible: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int, int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "consumer_device",
            "fallback_device",
            "weight_offset",
            "weight_bytes",
            "in_dim",
            "out_dim",
            "fp16_bytes",
            "status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            die(f"invalid Q8 plan audit header in {path}")
        for row in reader:
            if row["status"] != "unadmitted":
                continue
            try:
                device = int(row["consumer_device"])
                fallback = int(row["fallback_device"])
                key = (
                    device,
                    int(row["weight_offset"]),
                    int(row["weight_bytes"]),
                    int(row["in_dim"]),
                    int(row["out_dim"]),
                )
                fp16_bytes = int(row["fp16_bytes"])
            except (KeyError, ValueError) as exc:
                die(f"invalid Q8 plan audit row in {path}: {exc}")
            if key in seen:
                continue
            seen.add(key)
            stage = next((index for index, pair in enumerate(pairs)
                          if device in pair), None)
            if stage is None:
                die(f"plan candidate uses device {device} outside --gpu-devices")
            if fallback >= 0 and fallback != device:
                if fallback not in pairs[stage]:
                    die(
                        f"candidate fallback GPU{fallback} is outside stage "
                        f"pair {sorted(pairs[stage])}"
                    )
                flexible[stage].append(fp16_bytes)
            else:
                fixed[stage][device] += fp16_bytes

    required: dict[int, int] = {}
    detail: dict[int, tuple[dict[int, int], list[int]]] = {}
    for stage, pair in enumerate(pairs):
        devices = sorted(pair)
        fixed_by_device = {device: fixed[stage].get(device, 0)
                           for device in devices}
        movable = flexible.get(stage, [])
        if not any(fixed_by_device.values()) and not movable:
            continue
        subset_sums = {0}
        for size in movable:
            subset_sums |= {value + size for value in tuple(subset_sums)}
        movable_total = sum(movable)
        required[stage] = min(
            max(
                headroom_bytes_per_device + fixed_by_device[devices[0]] + left,
                headroom_bytes_per_device + fixed_by_device[devices[1]]
                + movable_total - left,
            )
            for left in subset_sums
        )
        detail[stage] = (fixed_by_device, movable)
    return required, detail


def verify_plan(path: Path) -> None:
    rows = 0
    unadmitted: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "status" not in reader.fieldnames:
            die(f"invalid Q8 plan audit header in {path}")
        for row in reader:
            rows += 1
            if row["status"] == "unadmitted":
                unadmitted.append(
                    f"{row.get('label', '?')}@GPU{row.get('consumer_device', '?')}"
                )
    if rows == 0:
        die(f"Q8 plan audit is empty: {path}")
    if unadmitted:
        sample = ", ".join(unadmitted[:8])
        suffix = "" if len(unadmitted) <= 8 else f", ... ({len(unadmitted)} total)"
        die(f"Q8 FP16 plan is not fully resident: {sample}{suffix}")


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
        die("layer preference did not contain any routed layers")
    return result


def select_stage_recipe(
    required_bytes: int,
    q2_candidates: list[int],
    iq2_candidates: list[int],
    q2_saving_bytes_per_device: int,
    iq2_saving_bytes_per_device: int,
    tie_break: str = "preserve-down",
) -> tuple[list[int], list[int], int]:
    """Solve the small integer problem exactly.

    A Q2-down conversion sacrifices one Q4 tensor. An IQ2 gate/up conversion
    sacrifices two. The primary objective therefore minimizes q2 + 2*iq2,
    which is exactly equivalent to maximizing the number of routed Q4 tensors.
    """
    if required_bytes <= 0:
        return [], [], 0
    if q2_saving_bytes_per_device <= 0 or iq2_saving_bytes_per_device <= 0:
        die("quantization savings must be positive")
    best: tuple[tuple[int, ...], int, int, int] | None = None
    for q2_count in range(len(q2_candidates) + 1):
        for iq2_count in range(len(iq2_candidates) + 1):
            reclaim = (
                q2_count * q2_saving_bytes_per_device
                + iq2_count * iq2_saving_bytes_per_device
            )
            if reclaim < required_bytes:
                continue
            non_q4 = q2_count + 2 * iq2_count
            overshoot = reclaim - required_bytes
            if tie_break == "preserve-down":
                objective = (non_q4, q2_count, overshoot, iq2_count)
            elif tie_break == "prefer-q2":
                objective = (non_q4, 2 * iq2_count, overshoot, q2_count)
            else:
                objective = (non_q4, overshoot, q2_count, iq2_count)
            candidate = (objective, q2_count, iq2_count, reclaim)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        maximum = (
            len(q2_candidates) * q2_saving_bytes_per_device
            + len(iq2_candidates) * iq2_saving_bytes_per_device
        )
        die(
            "cache target is infeasible after converting every available "
            f"routed tensor (need={required_bytes / MIB:.0f} MiB/device, "
            f"maximum={maximum / MIB:.0f} MiB/device)"
        )
    _objective, q2_count, iq2_count, reclaim = best
    return q2_candidates[:q2_count], iq2_candidates[:iq2_count], reclaim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-audit", type=Path, required=True)
    parser.add_argument("--layout-log", type=Path)
    parser.add_argument("--gpu-devices", default="0,3,1,2")
    parser.add_argument("--extra-headroom-mib-per-device", type=int, default=512)
    parser.add_argument("--q2-down-saving-mib-per-device", type=int, default=240)
    parser.add_argument("--iq2-gate-up-saving-mib-per-device", type=int, default=624)
    parser.add_argument("--routed-first", type=int, default=3)
    parser.add_argument("--routed-last", type=int, default=42)
    parser.add_argument("--q2-layer-order")
    parser.add_argument("--iq2-layer-order")
    parser.add_argument(
        "--tie-break",
        choices=("preserve-down", "prefer-q2", "least-overshoot"),
        default="preserve-down",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_plan(args.plan_audit)
        print("Q8 FP16 plan verification: every candidate is resident")
        return 0
    if args.layout_log is None:
        die("--layout-log is required when selecting layers")
    if args.extra_headroom_mib_per_device < 0:
        die("extra headroom must be nonnegative")

    devices = parse_devices(args.gpu_devices)
    pairs = device_pairs(devices)
    layout = parse_layout(args.layout_log)
    headroom = args.extra_headroom_mib_per_device * MIB
    required_by_stage, capacity_detail = required_reclaim_by_stage(
        args.plan_audit, pairs, headroom
    )
    preferred_q2 = layer_order(
        args.routed_first, args.routed_last, args.q2_layer_order
    )
    preferred_iq2 = layer_order(
        args.routed_first,
        args.routed_last,
        args.iq2_layer_order or args.q2_layer_order,
    )
    q2_saving = args.q2_down_saving_mib_per_device * MIB
    iq2_saving = args.iq2_gate_up_saving_mib_per_device * MIB
    selected_q2: list[int] = []
    selected_iq2: list[int] = []

    for stage, (lo, hi) in enumerate(layout):
        required = required_by_stage.get(stage, 0)
        if required == 0:
            print(
                f"stage {stage} pair {sorted(pairs[stage])}: cache already complete",
                file=sys.stderr,
            )
            continue
        fixed_by_device, movable = capacity_detail[stage]
        q2_candidates = [layer for layer in preferred_q2 if lo <= layer <= hi]
        iq2_candidates = [layer for layer in preferred_iq2 if lo <= layer <= hi]
        q2_layers, iq2_layers, reclaim = select_stage_recipe(
            required,
            q2_candidates,
            iq2_candidates,
            q2_saving,
            iq2_saving,
            args.tie_break,
        )
        selected_q2.extend(q2_layers)
        selected_iq2.extend(iq2_layers)
        print(
            f"stage {stage} pair {sorted(pairs[stage])}: "
            f"fixed_home="
            f"{','.join(f'GPU{device}:{value / MIB:.0f}' for device, value in sorted(fixed_by_device.items()))} MiB "
            f"partner_flexible={sum(movable) / MIB:.0f} MiB "
            f"headroom={args.extra_headroom_mib_per_device} MiB/device "
            f"required_reclaim={required / MIB:.0f} MiB/device; "
            f"non_q4={len(q2_layers) + 2 * len(iq2_layers)} "
            f"q2_down={len(q2_layers)} iq2_gate_up_pairs={len(iq2_layers)} "
            f"reclaim={reclaim / MIB:.0f} MiB; "
            f"q2_layers={','.join(map(str, q2_layers)) or '-'} "
            f"iq2_layers={','.join(map(str, iq2_layers)) or '-'}",
            file=sys.stderr,
        )

    if not selected_q2 and not selected_iq2:
        die("the baseline cache plan was complete; a selective quant is unnecessary")
    selected_q2 = sorted(set(selected_q2))
    selected_iq2 = sorted(set(selected_iq2))
    overrides = [
        f"blk.{layer}.ffn_down_exps.weight=q2_k" for layer in selected_q2
    ]
    for layer in selected_iq2:
        overrides.extend(
            (
                f"blk.{layer}.ffn_gate_exps.weight=iq2_xxs",
                f"blk.{layer}.ffn_up_exps.weight=iq2_xxs",
            )
        )
    print(
        f"{','.join(map(str, selected_q2)) or '-'}\t"
        f"{','.join(map(str, selected_iq2)) or '-'}\t"
        f"{','.join(overrides)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
