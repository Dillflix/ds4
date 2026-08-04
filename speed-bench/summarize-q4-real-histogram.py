#!/usr/bin/env python3
"""Derive exact Q4 gate/up and down tile plans from deferred GPU counts."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


BASE_ID = (
    "logical_tier",
    "physical_device",
    "sequence",
    "layer",
    "token_offset",
    "n_tokens",
    "ownership",
    "owner_base",
    "owner_count",
)

GRID_FIELDS = (
    "gate_grid_tile_capacity",
    "gate_grid_inactive_tiles",
    "gate_grid_active_pct",
    "down_grid_tile_capacity",
    "down_grid_inactive_tiles",
    "down_grid_active_pct",
)


def die(message: str) -> "None":
    raise SystemExit(message)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def tile_plans(counts: list[int]) -> dict[str, int | float]:
    pairs = sum(counts)
    gate_t8 = sum(ceil_div(count, 8) for count in counts if count)
    gate_full8 = sum(count // 8 for count in counts)
    gate_tail8 = 0
    gate_tail4 = 0
    for count in counts:
        remainder = count % 8
        if 1 <= remainder <= 4:
            gate_tail4 += 1
        elif remainder:
            gate_tail8 += 1
    gate_adaptive_slots = (
        (gate_full8 + gate_tail8) * 8 + gate_tail4 * 4
    )
    down_t16 = sum(ceil_div(count, 16) for count in counts if count)
    down_full16 = sum(count // 16 for count in counts)
    down_tail8 = 0
    down_tail4 = 0
    for count in counts:
        remainder = count % 16
        if 1 <= remainder <= 4:
            down_tail4 += 1
        elif remainder <= 8 and remainder:
            down_tail8 += 1
        elif remainder <= 12 and remainder:
            down_tail8 += 1
            down_tail4 += 1
        elif remainder:
            down_tail8 += 2
    down_adaptive_slots = down_full16 * 16 + down_tail8 * 8 + down_tail4 * 4
    gate_slots = gate_t8 * 8
    down_slots = down_t16 * 16
    return {
        "pair_count": pairs,
        "active_experts": sum(count != 0 for count in counts),
        "gate_tile8_count": gate_t8,
        "gate_tile8_slots": gate_slots,
        "gate_tile8_padded_slots": gate_slots - pairs,
        "gate_tile8_fill_pct": 100.0 * pairs / gate_slots if gate_slots else 0.0,
        "gate_adaptive_full8_count": gate_full8,
        "gate_adaptive_tail8_count": gate_tail8,
        "gate_adaptive_tail4_count": gate_tail4,
        "gate_adaptive_slots": gate_adaptive_slots,
        "gate_adaptive_padded_slots": gate_adaptive_slots - pairs,
        "gate_adaptive_fill_pct": (
            100.0 * pairs / gate_adaptive_slots if gate_adaptive_slots else 0.0
        ),
        "down_tile16_count": down_t16,
        "down_tile16_slots": down_slots,
        "down_tile16_padded_slots": down_slots - pairs,
        "down_tile16_fill_pct": 100.0 * pairs / down_slots if down_slots else 0.0,
        "down_adaptive_full16_count": down_full16,
        "down_adaptive_tail8_count": down_tail8,
        "down_adaptive_tail4_count": down_tail4,
        "down_adaptive_slots": down_adaptive_slots,
        "down_adaptive_padded_slots": down_adaptive_slots - pairs,
        "down_adaptive_fill_pct": (
            100.0 * pairs / down_adaptive_slots if down_adaptive_slots else 0.0
        ),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 5:
        die(
            "usage: summarize-q4-real-histogram.py "
            "TILE.csv PLAN.csv EXPERTS.csv FREQUENCY.csv"
        )
    source, plan_path, expert_path, frequency_path = map(Path, sys.argv[1:])
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            die(f"empty tile audit: {source}")
        expert_fields = sorted(
            field for field in reader.fieldnames
            if field.startswith("expert_") and field.endswith("_pairs")
        )
        if not expert_fields:
            die(f"tile audit has no deferred expert counts: {source}")
        raw_rows = list(reader)
    if not raw_rows:
        die(f"tile audit has no data rows: {source}")

    plan_rows: list[dict[str, object]] = []
    expert_rows: list[dict[str, object]] = []
    frequency: Counter[tuple[int, str, int]] = Counter()
    for raw in raw_rows:
        captured = int(raw["captured_experts"])
        total = int(raw["n_total_expert"])
        if captured != total:
            die(
                f"truncated expert histogram at layer={raw['layer']} "
                f"tier={raw['logical_tier']}: captured={captured} total={total}"
            )
        if captured > len(expert_fields):
            die(f"captured_experts={captured} exceeds CSV capacity")
        counts = [int(raw[field]) for field in expert_fields[:captured]]
        plans = tile_plans(counts)
        if plans["pair_count"] != int(raw["pair_count"]):
            die(
                f"pair-count mismatch at layer={raw['layer']} "
                f"tier={raw['logical_tier']}: histogram={plans['pair_count']} "
                f"record={raw['pair_count']}"
            )
        if plans["active_experts"] != int(raw["active_experts"]):
            die(
                f"active-expert mismatch at layer={raw['layer']} "
                f"tier={raw['logical_tier']}"
            )
        selected_slots = int(raw["selected_slots"])
        gate_capacity = ceil_div(selected_slots, 8) + total
        down_capacity = ceil_div(selected_slots, 16) + total
        if gate_capacity < int(plans["gate_tile8_count"]) or (
            down_capacity < int(plans["down_tile16_count"])
        ):
            die(
                f"production grid capacity is smaller than active tiles at "
                f"layer={raw['layer']} tier={raw['logical_tier']}"
            )
        plans.update(
            {
                "gate_grid_tile_capacity": gate_capacity,
                "gate_grid_inactive_tiles": (
                    gate_capacity - int(plans["gate_tile8_count"])
                ),
                "gate_grid_active_pct": (
                    100.0 * int(plans["gate_tile8_count"]) / gate_capacity
                    if gate_capacity else 0.0
                ),
                "down_grid_tile_capacity": down_capacity,
                "down_grid_inactive_tiles": (
                    down_capacity - int(plans["down_tile16_count"])
                ),
                "down_grid_active_pct": (
                    100.0 * int(plans["down_tile16_count"]) / down_capacity
                    if down_capacity else 0.0
                ),
            }
        )
        identity = {field: raw[field] for field in BASE_ID}
        plan_rows.append({**identity, **plans})

        owner_base = int(raw["owner_base"])
        owner_count = int(raw["owner_count"])
        local_owner_numbering = total == owner_count
        for index, count in enumerate(counts):
            global_expert = owner_base + index if local_owner_numbering else index
            expert_rows.append(
                {
                    **identity,
                    "captured_index": index,
                    "global_expert": global_expert,
                    "pair_count": count,
                    "active": int(count != 0),
                    "gate_tile8_count": ceil_div(count, 8) if count else 0,
                    "down_tile16_count": ceil_div(count, 16) if count else 0,
                }
            )
            frequency[(int(raw["layer"]), raw["ownership"], count)] += 1

    plan_fields = list(BASE_ID) + list(tile_plans([]).keys()) + list(GRID_FIELDS)
    expert_fields_out = list(BASE_ID) + [
        "captured_index",
        "global_expert",
        "pair_count",
        "active",
        "gate_tile8_count",
        "down_tile16_count",
    ]
    frequency_rows = [
        {
            "layer": layer,
            "ownership": ownership,
            "expert_pair_count": pair_count,
            "expert_samples": samples,
        }
        for (layer, ownership, pair_count), samples in sorted(frequency.items())
    ]
    write_csv(plan_path, plan_fields, plan_rows)
    write_csv(expert_path, expert_fields_out, expert_rows)
    write_csv(
        frequency_path,
        ["layer", "ownership", "expert_pair_count", "expert_samples"],
        frequency_rows,
    )

    pairs = sum(int(row["pair_count"]) for row in plan_rows)
    gate_slots = sum(int(row["gate_tile8_slots"]) for row in plan_rows)
    down_slots = sum(int(row["down_tile16_slots"]) for row in plan_rows)
    mixed_slots = sum(int(row["down_adaptive_slots"]) for row in plan_rows)
    gate_mixed_slots = sum(
        int(row["gate_adaptive_slots"]) for row in plan_rows
    )
    gate_tiles = sum(int(row["gate_tile8_count"]) for row in plan_rows)
    gate_mixed_tiles = sum(
        int(row["gate_adaptive_full8_count"])
        + int(row["gate_adaptive_tail8_count"])
        + int(row["gate_adaptive_tail4_count"])
        for row in plan_rows
    )
    down_tiles = sum(int(row["down_tile16_count"]) for row in plan_rows)
    down_mixed_tiles = sum(
        int(row["down_adaptive_full16_count"])
        + int(row["down_adaptive_tail8_count"])
        + int(row["down_adaptive_tail4_count"])
        for row in plan_rows
    )
    gate_grid_capacity = sum(
        int(row["gate_grid_tile_capacity"]) for row in plan_rows
    )
    down_grid_capacity = sum(
        int(row["down_grid_tile_capacity"]) for row in plan_rows
    )
    print(f"records={len(plan_rows)}")
    print(f"expert_rows={len(expert_rows)}")
    print(f"pair_slots={pairs}")
    print(f"gate_tile8_fill_pct={100.0 * pairs / gate_slots:.6f}")
    print(f"gate_adaptive_8_4_fill_pct={100.0 * pairs / gate_mixed_slots:.6f}")
    print(f"gate_tile8_active_tiles={gate_tiles}")
    print(f"gate_adaptive_8_4_active_tiles={gate_mixed_tiles}")
    print(f"gate_grid_tile_capacity={gate_grid_capacity}")
    print(f"gate_grid_active_pct={100.0 * gate_tiles / gate_grid_capacity:.6f}")
    print(f"down_tile16_fill_pct={100.0 * pairs / down_slots:.6f}")
    print(f"down_adaptive_16_8_4_fill_pct={100.0 * pairs / mixed_slots:.6f}")
    print(f"down_tile16_active_tiles={down_tiles}")
    print(f"down_adaptive_16_8_4_active_tiles={down_mixed_tiles}")
    print(f"down_grid_tile_capacity={down_grid_capacity}")
    print(f"down_grid_active_pct={100.0 * down_tiles / down_grid_capacity:.6f}")


if __name__ == "__main__":
    main()
