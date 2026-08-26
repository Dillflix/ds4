#!/usr/bin/env python3
"""Select the minimum IQ2 gate/up pairs needed for complete dense-F16 coverage."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


MIB = 1024 * 1024
N_LAYERS = 43
EXPECTED_PLAN_SUFFIXES = {
    "T32-q_b": ("attn_q_b.weight",),
    "T256-output_b": ("attn_output_b.weight",),
    "output_a": ("attn_output_a.weight",),
    "shared_down": ("ffn_down_shexp.weight",),
    "shared_gate_up": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    "other_attn": ("attn_q_a.weight", "attn_kv.weight"),
}
EXPECTED_PLAN_CLASS_COUNTS = {
    name: N_LAYERS * len(suffixes)
    for name, suffixes in EXPECTED_PLAN_SUFFIXES.items()
}
DEFAULT_EXPECTED_CANDIDATES = sum(EXPECTED_PLAN_CLASS_COUNTS.values())


def die(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_devices(text: str) -> list[int]:
    try:
        devices = [int(value) for value in text.split(",")]
    except ValueError as exc:
        die(f"invalid --gpu-devices: {exc}")
    if len(devices) != 4 or len(set(devices)) != 4:
        die("--gpu-devices must contain four distinct device numbers")
    return devices


def device_pairs(devices: list[int]) -> list[set[int]]:
    # CLI order is home-stage0, home-stage1, partner-stage0, partner-stage1.
    return [{devices[0], devices[2]}, {devices[1], devices[3]}]


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


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "label",
            "consumer_device",
            "fallback_device",
            "target_device",
            "placement_locked",
            "resident_device",
            "weight_offset",
            "weight_bytes",
            "in_dim",
            "out_dim",
            "fp16_bytes",
            "status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            die(f"invalid Q8 plan audit header in {path}; missing {missing}")
        rows = list(reader)
    if not rows:
        die(f"Q8 plan audit is empty: {path}")
    return rows


def read_device_memory(path: Path) -> dict[int, dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "physical_device",
            "free_bytes",
            "total_bytes",
            "q8_fp16_cached_bytes",
            "q8_fp16_reserve_bytes",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            die(f"invalid CUDA memory-state header in {path}; missing {missing}")
        rows = list(reader)
    result: dict[int, dict[str, int]] = {}
    for row in rows:
        try:
            device = int(row["physical_device"])
            values = {
                name: int(row[name])
                for name in (
                    "free_bytes",
                    "total_bytes",
                    "q8_fp16_cached_bytes",
                    "q8_fp16_reserve_bytes",
                )
            }
        except (KeyError, ValueError) as exc:
            die(f"invalid CUDA memory-state row in {path}: {exc}")
        if device in result:
            die(f"duplicate physical GPU{device} in {path}")
        if values["free_bytes"] < 0 or values["total_bytes"] <= 0:
            die(f"invalid memory capacity for physical GPU{device}")
        result[device] = values
    if not result:
        die(f"CUDA memory-state is empty: {path}")
    return result


def validate_plan_inventory(
    rows: list[dict[str, str]], expected_candidates: int
) -> None:
    """Require the exact production, non-head-split dense-Q8 directory.

    Production prefill head splitting is disabled by default, so output-A and
    q_b each contribute one full candidate per layer. Alternate half-slices or
    full-tensor fallbacks belong to the opt-in split path and must not be
    mistaken for additional coverage here.
    """

    if expected_candidates != DEFAULT_EXPECTED_CANDIDATES:
        die(
            f"expected-candidates must be {DEFAULT_EXPECTED_CANDIDATES} for "
            "the production non-head-split dense plan"
        )
    if len(rows) != DEFAULT_EXPECTED_CANDIDATES:
        die(
            f"Q8 plan has {len(rows)} candidates; expected "
            f"{DEFAULT_EXPECTED_CANDIDATES} for complete production coverage"
        )

    expected_labels: dict[str, str] = {}
    for class_name, suffixes in EXPECTED_PLAN_SUFFIXES.items():
        for layer in range(N_LAYERS):
            for suffix in suffixes:
                expected_labels[f"tensor:blk.{layer}.{suffix}"] = class_name

    labels = [row["label"] for row in rows]
    duplicates = sorted(
        label for label, count in Counter(labels).items() if count != 1
    )
    if duplicates:
        die(f"Q8 plan contains a duplicate candidate: {duplicates[0]}")

    actual = set(labels)
    unknown = sorted(actual - set(expected_labels))
    if unknown:
        die(f"Q8 plan contains an unknown/dead alternate candidate: {unknown[0]}")
    missing = sorted(set(expected_labels) - actual)
    if missing:
        die(f"Q8 plan is missing a production candidate: {missing[0]}")

    actual_counts = Counter(expected_labels[label] for label in labels)
    if actual_counts != Counter(EXPECTED_PLAN_CLASS_COUNTS):
        die(
            "Q8 plan class counts differ from production: "
            + ", ".join(
                f"{name}={actual_counts.get(name, 0)}/"
                f"{EXPECTED_PLAN_CLASS_COUNTS[name]}"
                for name in EXPECTED_PLAN_CLASS_COUNTS
            )
        )


def required_reclaim_by_stage(
    path: Path,
    pairs: list[set[int]],
    headroom_bytes_per_device: int,
) -> tuple[dict[int, int], dict[int, tuple[dict[int, int], list[int]]]]:
    """Return the equal routed-weight reclaim needed on both devices per pair.

    A routed gate/up demotion frees the same number of bytes on both expert
    owners. Locked candidates (notably all-partner T256) must fit their target.
    An ordinary home candidate with a validated partner fallback can use either
    pair member, so an exact subset sum finds its best indivisible assignment.
    """

    fixed: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    flexible: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int, int, int, int]] = set()
    for row in read_plan(path):
        if row["status"] != "unadmitted":
            continue
        try:
            consumer = int(row["consumer_device"])
            fallback = int(row["fallback_device"])
            target = int(row["target_device"])
            locked = int(row["placement_locked"]) != 0
            key = (
                consumer,
                int(row["weight_offset"]),
                int(row["weight_bytes"]),
                int(row["in_dim"]),
                int(row["out_dim"]),
            )
            fp16_bytes = int(row["fp16_bytes"])
        except (KeyError, ValueError) as exc:
            die(f"invalid Q8 plan audit row in {path}: {exc}")
        if fp16_bytes <= 0:
            die(f"plan candidate has invalid FP16 size: {row.get('label', '?')}")
        if key in seen:
            continue
        seen.add(key)
        stage = next(
            (index for index, pair in enumerate(pairs) if consumer in pair),
            None,
        )
        if stage is None:
            die(f"plan candidate uses GPU{consumer} outside --gpu-devices")
        pair = pairs[stage]
        if fallback >= 0 and fallback not in pair:
            die(f"fallback GPU{fallback} is outside stage pair {sorted(pair)}")
        if locked:
            if target not in pair:
                die(f"locked target GPU{target} is outside stage pair {sorted(pair)}")
            fixed[stage][target] += fp16_bytes
        elif fallback >= 0 and fallback != consumer:
            flexible[stage].append(fp16_bytes)
        else:
            fixed[stage][consumer] += fp16_bytes

    required: dict[int, int] = {}
    detail: dict[int, tuple[dict[int, int], list[int]]] = {}
    for stage, pair in enumerate(pairs):
        devices = sorted(pair)
        fixed_by_device = {
            device: fixed[stage].get(device, 0) for device in devices
        }
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
                headroom_bytes_per_device
                + fixed_by_device[devices[1]]
                + movable_total
                - left,
            )
            for left in subset_sums
        )
        detail[stage] = (fixed_by_device, movable)
    return required, detail


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
        die("IQ2 layer preference did not contain any routed layers")
    return result


def select_iq2_layers(
    required_bytes: int,
    candidates: list[int],
    saving_bytes_per_device: int,
) -> tuple[list[int], int]:
    if required_bytes <= 0:
        return [], 0
    if saving_bytes_per_device <= 0:
        die("IQ2 gate/up saving must be positive")
    count = math.ceil(required_bytes / saving_bytes_per_device)
    if count > len(candidates):
        maximum = len(candidates) * saving_bytes_per_device
        die(
            "complete dense FP16 coverage is infeasible with Q4/IQ2 alone "
            f"(need={required_bytes / MIB:.0f} MiB/device, "
            f"maximum={maximum / MIB:.0f} MiB/device)"
        )
    selected = candidates[:count]
    return selected, count * saving_bytes_per_device


def select_from_all_iq2_calibration(
    layout: list[tuple[int, int]],
    pairs: list[set[int]],
    memory: dict[int, dict[str, int]],
    preferred: list[int],
    selectable_first: int,
    selectable_last: int,
    saving_bytes_per_device: int,
    extra_headroom_bytes_per_device: int,
) -> list[int]:
    """Promote as many IQ2 gate/up pairs back to Q4 as measured VRAM permits.

    The calibration GGUF makes every routed gate/up pair IQ2.  The final
    recipe keeps layers outside the selectable range at its Q4 default, so
    those mandatory promotions must be charged before optional promotions.
    In the production recipe that is layers 0-2: omitting their fixed
    3*624-MiB/device cost overcommits stage 0 even when the calibration itself
    had the requested reserve and safety margin.
    """

    selected_iq2: list[int] = []
    for stage, (lo, hi) in enumerate(layout):
        stage_layers = [layer for layer in preferred if lo <= layer <= hi]
        expected_layers = [
            layer
            for layer in range(
                max(lo, selectable_first),
                min(hi, selectable_last) + 1,
            )
        ]
        if set(stage_layers) != set(expected_layers):
            missing = sorted(set(expected_layers) - set(stage_layers))
            die(
                "IQ2 layer preference must order every routed layer; "
                f"stage {stage} is missing {missing}"
            )
        mandatory_q4_layers = [
            layer
            for layer in range(max(lo, 0), min(hi, N_LAYERS - 1) + 1)
            if layer < selectable_first or layer > selectable_last
        ]
        mandatory_q4_bytes = (
            len(mandatory_q4_layers) * saving_bytes_per_device
        )
        available_by_device: dict[int, int] = {}
        for device in sorted(pairs[stage]):
            if device not in memory:
                die(f"CUDA memory-state is missing physical GPU{device}")
            row = memory[device]
            protected = (
                row["q8_fp16_reserve_bytes"]
                + extra_headroom_bytes_per_device
            )
            available_by_device[device] = max(0, row["free_bytes"] - protected)
        raw_pair_available = min(available_by_device.values())
        if mandatory_q4_bytes > raw_pair_available:
            die(
                f"stage {stage} cannot fit mandatory Q4 gate/up layers "
                f"{mandatory_q4_layers} while preserving the cache reserve "
                "and requested headroom "
                f"(need={mandatory_q4_bytes / MIB:.0f} MiB/device, "
                f"available={raw_pair_available / MIB:.0f} MiB/device)"
            )
        pair_available = raw_pair_available - mandatory_q4_bytes
        q4_promotions = min(
            len(stage_layers), pair_available // saving_bytes_per_device
        )
        iq2_count = len(stage_layers) - q4_promotions
        chosen = stage_layers[:iq2_count]
        selected_iq2.extend(chosen)
        free_text = ",".join(
            f"GPU{device}:{memory[device]['free_bytes'] / MIB:.0f}"
            for device in sorted(pairs[stage])
        )
        reserve_text = ",".join(
            f"GPU{device}:{memory[device]['q8_fp16_reserve_bytes'] / MIB:.0f}"
            for device in sorted(pairs[stage])
        )
        print(
            f"stage {stage} pair {sorted(pairs[stage])}: "
            f"all-IQ2 post-cache free={free_text} MiB "
            f"cache_reserve={reserve_text} MiB "
            f"extra_headroom={extra_headroom_bytes_per_device / MIB:.0f} "
            f"MiB/device; mandatory_Q4_layers="
            f"{','.join(map(str, mandatory_q4_layers)) or 'none'} "
            f"mandatory_Q4_cost={mandatory_q4_bytes / MIB:.0f} MiB/device; "
            f"optional_Q4_pairs={q4_promotions} "
            f"IQ2_pairs={iq2_count}; IQ2_layers={','.join(map(str, chosen))}",
            file=sys.stderr,
        )
    return selected_iq2


def ensure_plan_complete(
    rows: list[dict[str, str]], expected_candidates: int
) -> None:
    validate_plan_inventory(rows, expected_candidates)
    unadmitted = [row for row in rows if row["status"] == "unadmitted"]
    bad = [
        row for row in rows
        if row["status"] not in {"home", "partner", "unadmitted"}
    ]
    if bad:
        die(f"Q8 plan contains an unknown status: {bad[0]['status']}")
    if unadmitted:
        sample = ", ".join(
            f"{row['label']}@GPU{row['target_device']}" for row in unadmitted[:8]
        )
        suffix = "" if len(unadmitted) <= 8 else f", ... ({len(unadmitted)} total)"
        die(f"dense FP16 plan is incomplete: {sample}{suffix}")


def verify_plan(path: Path, expected_candidates: int) -> None:
    rows = read_plan(path)
    ensure_plan_complete(rows, expected_candidates)
    print(
        f"dense FP16 plan verification: {len(rows)}/{len(rows)} candidates resident"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-audit", type=Path, required=True)
    parser.add_argument("--layout-log", type=Path)
    parser.add_argument("--device-memory", type=Path)
    parser.add_argument("--gpu-devices", default="0,3,1,2")
    parser.add_argument("--extra-headroom-mib-per-device", type=int, default=512)
    parser.add_argument("--iq2-gate-up-saving-mib-per-device", type=int, default=624)
    parser.add_argument("--routed-first", type=int, default=3)
    parser.add_argument("--routed-last", type=int, default=42)
    parser.add_argument("--iq2-layer-order")
    parser.add_argument(
        "--expected-candidates", type=int, default=DEFAULT_EXPECTED_CANDIDATES
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_plan(args.plan_audit, args.expected_candidates)
        return 0
    if args.layout_log is None:
        die("--layout-log is required when selecting layers")
    if args.extra_headroom_mib_per_device < 0:
        die("extra headroom must be nonnegative")

    devices = parse_devices(args.gpu_devices)
    pairs = device_pairs(devices)
    layout = parse_layout(args.layout_log)
    baseline_rows = read_plan(args.plan_audit)
    validate_plan_inventory(baseline_rows, args.expected_candidates)
    headroom = args.extra_headroom_mib_per_device * MIB
    preferred = layer_order(
        args.routed_first, args.routed_last, args.iq2_layer_order
    )
    saving = args.iq2_gate_up_saving_mib_per_device * MIB
    selected: list[int] = []

    if args.device_memory is not None:
        ensure_plan_complete(baseline_rows, args.expected_candidates)
        selected = select_from_all_iq2_calibration(
            layout,
            pairs,
            read_device_memory(args.device_memory),
            preferred,
            args.routed_first,
            args.routed_last,
            saving,
            headroom,
        )
    else:
        required_by_stage, detail = required_reclaim_by_stage(
            args.plan_audit, pairs, headroom
        )
        for stage, (lo, hi) in enumerate(layout):
            required = required_by_stage.get(stage, 0)
            if required == 0:
                print(
                    f"stage {stage} pair {sorted(pairs[stage])}: "
                    "dense FP16 plan already complete",
                    file=sys.stderr,
                )
                continue
            candidates = [layer for layer in preferred if lo <= layer <= hi]
            chosen, reclaim = select_iq2_layers(required, candidates, saving)
            selected.extend(chosen)
            fixed, movable = detail[stage]
            fixed_text = ",".join(
                f"GPU{device}:{value / MIB:.0f}"
                for device, value in sorted(fixed.items())
            )
            print(
                f"stage {stage} pair {sorted(pairs[stage])}: "
                f"fixed_missing={fixed_text} MiB "
                f"flexible_missing={sum(movable) / MIB:.0f} MiB "
                f"headroom={args.extra_headroom_mib_per_device} MiB/device "
                f"required_reclaim={required / MIB:.0f} MiB/device; "
                f"iq2_gate_up_pairs={len(chosen)} "
                f"reclaim={reclaim / MIB:.0f} MiB; "
                f"layers={','.join(map(str, chosen))}",
                file=sys.stderr,
            )

    selected = sorted(set(selected))
    if not selected:
        die("full Q4 already has complete dense FP16 coverage at this allocation")
    overrides: list[str] = []
    for layer in selected:
        overrides.extend(
            (
                f"blk.{layer}.ffn_gate_exps.weight=iq2_xxs",
                f"blk.{layer}.ffn_up_exps.weight=iq2_xxs",
            )
        )
    print(f"{','.join(map(str, selected))}\t{','.join(overrides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
